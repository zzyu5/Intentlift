//===- ConvertIntentGatherToGPU.cpp - gather/scatter -> gpu.launch --------===//
//
// The C3 data-dependent-addressing pass. Lowers a func whose body is exactly ONE
// `intent.gather` (torch.gather / index_select / embedding row-gather) or ONE
// `intent.scatter` (torch.scatter / scatter_add_) over rank-2 data to a single
// block-cooperative `gpu.launch`: one block per output/scatter row, the column
// axis split over the block's T threads (K = cols/T columns each, unrolled).
//
// The NEW mechanism vs. the elementwise/reduce passes: the per-element access
// coordinate on `data` is a RUNTIME value read from the integer `index` memref —
// `idx = index_cast(memref.load index[...])` — used directly as a
// `memref.load`/`memref.store`/`memref.atomic_rmw` coordinate on `data`. This is
// a data-dependent gather/scatter address, not an affine function of the loop.
//
//   gather : destination-passing. new signature = (data, index) ++ [Ro,Co] out.
//            out[bx,col] = data[<coord with the index axis read from index>].
//   scatter: IN-PLACE on the (caller-cloned) data memref. new signature =
//            (data, index, src). data[<coord with axis=index[bx,col]>] (op)=
//            src[bx,col]; op = store (reduce none), legacy atomic_rmw addf
//            (implicit add), or a scoped LLVM/NVVM atomic for explicit add/CAS.
//
// It additionally recognizes a typed two-level chain
//   take(data,row_index,axis=0) -> along_axis(tmp,col_index,axis=1).
// The canonical state eagerly evaluates the complete first-gather row into
// workgroup memory before selecting its final element.  The Full-only composed
// state composes both runtime index maps and directly loads data[row,col].
//
// Correctness-only (the index stream is exact -> bit-faithful). Static shapes,
// cols % T == 0 (tail-mask is a follow-up). The two match() predicates for the
// existing passes reject any body with a gather/scatter op (they only accept
// map/reduce/select/iota), so this pass is disjoint and composes freely.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "MapEmit.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <string>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTGATHERTOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// One func's gather/scatter lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct GatherScatterLowering {
  func::FuncOp fn;
  unsigned blockThreads;
  bool requestComposeGatherChain;

  // populated by match()
  GatherOp gather;    // set for a gather func
  ScatterOp scatter;  // set for a scatter func
  GatherOp firstGather;
  GatherOp secondGather;
  AtomicScatterOp atomicScatter;
  tensor::ExpandShapeOp expandedColumnIndex;
  tensor::CollapseShapeOp collapsedResult;
  bool gatherChain = false;
  bool explicitAtomicAdd = false;
  int64_t axis = 0;
  // Iteration geometry: one block per row (Ro), T threads over Co columns, K
  // columns per thread. For gather these are the OUTPUT dims; for scatter the
  // SRC/index dims (that is what drives the writes).
  int64_t Ro = 0, Co = 0, T = 0, K = 0;

  GatherScatterLowering(func::FuncOp fn, unsigned bt, bool composeChain)
      : fn(fn), blockThreads(bt),
        requestComposeGatherChain(composeChain) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  LogicalResult setGeometry(int64_t rows, int64_t cols) {
    Ro = rows;
    Co = cols;
    T = blockThreads ? (int64_t)blockThreads : autoThreads(Co);
    if (T <= 0 || Co % T != 0)
      return failure(); // tail-mask (non-divisible) is a documented follow-up
    K = Co / T;
    return success();
  }

  static bool hasAtomicMetadata(Operation *op) {
    return isa<AtomicScatterOp>(op) ||
           (isa<ScatterOp>(op) &&
            (op->hasAttr("atomic") || op->hasAttr("ordering") ||
             op->hasAttr("memory_scope") || op->hasAttr("cas_retry")));
  }

  static std::string atomicCapabilityReason(Operation *op) {
    auto scope = op->getAttrOfType<StringAttr>("memory_scope");
    if (scope && (scope.getValue() == "work_item" ||
                  scope.getValue() == "subgroup"))
      return "memory_scope is not representable by the CUDA global-memory "
             "atomic implementation (work_item/subgroup are unsupported)";

    if (isa<AtomicScatterOp>(op)) {
      auto atomic = dyn_cast<AtomicScatterOp>(op);
      auto retry = op->getAttrOfType<StringAttr>("cas_retry");
      if (retry && retry.getValue() != "single_attempt")
        return "compare-exchange retry mode is unsupported; only "
               "cas_retry=single_attempt is implemented";
      auto dataTy = dyn_cast<RankedTensorType>(atomic.getData().getType());
      if (!dataTy || !dataTy.hasStaticShape() || dataTy.getRank() != 2)
        return "compare-exchange requires a static rank-2 data tensor";
      if (!dataTy.getElementType().isInteger(32) &&
          !dataTy.getElementType().isInteger(64))
        return "compare-exchange currently supports only i32 and i64 data";
      return "the compare-exchange graph is not admitted by the rank-2 "
             "specialized CUDA path";
    }

    auto scatter = dyn_cast<ScatterOp>(op);
    if (scatter && scatter->getAttrOfType<StringAttr>("atomic") &&
        scatter->getAttrOfType<StringAttr>("atomic").getValue() != "add")
      return "atomic='cas' must be materialized as the complete "
             "intent.atomic_scatter form";
    if (scatter) {
      auto dataTy = dyn_cast<RankedTensorType>(scatter.getData().getType());
      if (!dataTy || !dataTy.hasStaticShape() || dataTy.getRank() != 2)
        return "atomic add requires a static rank-2 data tensor";
      if (dataTy.getElementType() != Float32Type::get(op->getContext()))
        return "explicit atomic add currently supports f32 data only";
      return "the atomic scatter graph is not admitted by the rank-2 "
             "specialized CUDA path";
    }
    return "unknown atomic scatter operation";
  }

  static FailureOr<LLVM::AtomicOrdering>
  mapAtomicOrdering(StringRef ordering) {
    if (ordering == "relaxed")
      return LLVM::AtomicOrdering::monotonic;
    if (ordering == "acquire")
      return LLVM::AtomicOrdering::acquire;
    if (ordering == "release")
      return LLVM::AtomicOrdering::release;
    if (ordering == "acq_rel")
      return LLVM::AtomicOrdering::acq_rel;
    if (ordering == "seq_cst")
      return LLVM::AtomicOrdering::seq_cst;
    return failure();
  }

  static FailureOr<StringRef> mapAtomicScope(StringRef scope) {
    if (scope == "workgroup")
      return StringRef("cta");
    if (scope == "device")
      return StringRef("gpu");
    if (scope == "system")
      return StringRef("sys");
    return failure();
  }

  static LLVM::AtomicOrdering mapCmpXchgFailureOrdering(
      StringRef ordering) {
    if (ordering == "release")
      return LLVM::AtomicOrdering::monotonic;
    if (ordering == "acq_rel")
      return LLVM::AtomicOrdering::acquire;
    if (ordering == "acquire")
      return LLVM::AtomicOrdering::acquire;
    if (ordering == "seq_cst")
      return LLVM::AtomicOrdering::seq_cst;
    return LLVM::AtomicOrdering::monotonic;
  }

  static void copyIntentFunctionAttrs(func::FuncOp source,
                                      func::FuncOp destination) {
    for (NamedAttribute attr : source->getAttrs()) {
      if (attr.getName().getValue().starts_with("intent."))
        destination->setAttr(attr.getName(), attr.getValue());
    }
  }

  static void attachAtomicMetadata(Operation *op, StringRef atomic,
                                   StringRef ordering, StringRef scope,
                                   StringRef retry = StringRef()) {
    MLIRContext *ctx = op->getContext();
    op->setAttr("intent.atomic", StringAttr::get(ctx, atomic));
    op->setAttr("intent.ordering", StringAttr::get(ctx, ordering));
    op->setAttr("intent.memory_scope", StringAttr::get(ctx, scope));
    if (!retry.empty())
      op->setAttr("intent.cas_retry", StringAttr::get(ctx, retry));
  }

  static void emitAtomicFence(OpBuilder &b, Location loc, StringRef atomic,
                              StringRef ordering, StringRef scope,
                              StringRef retry, bool before) {
    LLVM::AtomicOrdering fenceOrdering;
    if (before) {
      if (ordering == "release" || ordering == "acq_rel")
        fenceOrdering = LLVM::AtomicOrdering::release;
      else if (ordering == "seq_cst")
        fenceOrdering = LLVM::AtomicOrdering::seq_cst;
      else
        return;
    } else {
      if (ordering == "acquire" || ordering == "acq_rel")
        fenceOrdering = LLVM::AtomicOrdering::acquire;
      else if (ordering == "seq_cst")
        fenceOrdering = LLVM::AtomicOrdering::seq_cst;
      else
        return;
    }
    // LLVM 20's NVPTX backend crashes while selecting custom sync scopes on
    // fence instructions ("Could not find scope ID").  The atomic itself
    // carries the target scope through a scoped NVVM intrinsic or LLVM atomic
    // syncscope.  A default fence is deliberately stronger for narrower
    // scopes, so it preserves the ordering contract without relying on an
    // unsupported fence encoding; the intent.* attrs retain the source scope.
    auto fence = b.create<LLVM::FenceOp>(loc, fenceOrdering);
    attachAtomicMetadata(fence, atomic, ordering, scope, retry);
  }

  static Value emitScopedAtomicAdd(OpBuilder &b, Location loc, Value address,
                                   Value value, StringRef scope) {
    if (scope != "cta" && scope != "sys")
      return Value();
    std::string intrinsic = "llvm.nvvm.atomic.add.gen.f.";
    intrinsic += scope.str();
    auto call = b.create<LLVM::CallIntrinsicOp>(
        loc, value.getType(), b.getStringAttr(intrinsic),
        ValueRange{address, value});
    return call.getResults();
  }

  static Value emitScopedAtomicCas(OpBuilder &b, Location loc, Value address,
                                   Value compare, Value replacement,
                                   StringRef scope) {
    if (scope != "cta" && scope != "sys")
      return Value();
    std::string intrinsic = "llvm.nvvm.atomic.cas.gen.i.";
    intrinsic += scope.str();
    auto call = b.create<LLVM::CallIntrinsicOp>(
        loc, compare.getType(), b.getStringAttr(intrinsic),
        ValueRange{address, compare, replacement});
    return call.getResults();
  }

  Value atomicAddress(OpBuilder &b, Location loc, Value dataMem, Value row,
                      Value column, int64_t columns, int64_t elementBytes) {
    MLIRContext *ctx = dataMem.getContext();
    Value base = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, dataMem);
    Value base64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), base);
    Value cColumns = b.create<arith::ConstantIndexOp>(loc, columns);
    Value linear = b.create<arith::MulIOp>(loc, row, cColumns);
    linear = b.create<arith::AddIOp>(loc, linear, column);
    Value cBytes = b.create<arith::ConstantIndexOp>(loc, elementBytes);
    Value byteOffset = b.create<arith::MulIOp>(loc, linear, cBytes);
    Value byteOffset64 =
        b.create<arith::IndexCastOp>(loc, b.getI64Type(), byteOffset);
    Value address = b.create<arith::AddIOp>(loc, base64, byteOffset64);
    return b.create<LLVM::IntToPtrOp>(
        loc, LLVM::LLVMPointerType::get(ctx), address);
  }

  LogicalResult matchGatherChain(Block &body, func::ReturnOp ret) {
    if (ret.getNumOperands() != 1)
      return failure();
    SmallVector<Operation *> operations;
    for (Operation &op : body.without_terminator())
      operations.push_back(&op);
    if (operations.size() != 4)
      return failure();

    auto first = dyn_cast<GatherOp>(operations[0]);
    auto expand = dyn_cast<tensor::ExpandShapeOp>(operations[1]);
    auto second = dyn_cast<GatherOp>(operations[2]);
    auto collapse = dyn_cast<tensor::CollapseShapeOp>(operations[3]);
    if (!first || !expand || !second || !collapse ||
        first.getMode() != "take" || first.getAxis() != 0 ||
        second.getMode() != "along_axis" || second.getAxis() != 1 ||
        second.getData() != first.getResult() ||
        second.getIndex() != expand.getResult() ||
        collapse.getSrc() != second.getResult() ||
        ret.getOperand(0) != collapse.getResult())
      return failure();

    auto dataTy = dyn_cast<RankedTensorType>(first.getData().getType());
    auto rowIndexTy = dyn_cast<RankedTensorType>(first.getIndex().getType());
    auto firstResultTy = dyn_cast<RankedTensorType>(first.getResult().getType());
    auto colIndexTy = dyn_cast<RankedTensorType>(expand.getSrc().getType());
    auto expandedIndexTy = dyn_cast<RankedTensorType>(expand.getResult().getType());
    auto secondResultTy = dyn_cast<RankedTensorType>(second.getResult().getType());
    auto resultTy = dyn_cast<RankedTensorType>(collapse.getResult().getType());
    if (!dataTy || !rowIndexTy || !firstResultTy || !colIndexTy ||
        !expandedIndexTy || !secondResultTy || !resultTy ||
        !dataTy.hasStaticShape() || !rowIndexTy.hasStaticShape() ||
        !firstResultTy.hasStaticShape() || !colIndexTy.hasStaticShape() ||
        !expandedIndexTy.hasStaticShape() || !secondResultTy.hasStaticShape() ||
        !resultTy.hasStaticShape() || dataTy.getRank() != 2 ||
        rowIndexTy.getRank() != 1 || colIndexTy.getRank() != 1 ||
        firstResultTy.getRank() != 2 || expandedIndexTy.getRank() != 2 ||
        secondResultTy.getRank() != 2 || resultTy.getRank() != 1)
      return failure();

    int64_t rows = rowIndexTy.getDimSize(0);
    int64_t columns = dataTy.getDimSize(1);
    if (rows <= 0 || columns <= 0 ||
        firstResultTy.getShape() != ArrayRef<int64_t>({rows, columns}) ||
        colIndexTy.getDimSize(0) != rows ||
        expandedIndexTy.getShape() != ArrayRef<int64_t>({rows, 1}) ||
        secondResultTy.getShape() != ArrayRef<int64_t>({rows, 1}) ||
        resultTy.getDimSize(0) != rows ||
        firstResultTy.getElementType() != dataTy.getElementType() ||
        secondResultTy.getElementType() != dataTy.getElementType() ||
        resultTy.getElementType() != dataTy.getElementType() ||
        !isa<IntegerType>(rowIndexTy.getElementType()) ||
        !isa<IntegerType>(colIndexTy.getElementType()))
      return failure();

    Ro = rows;
    Co = columns;
    T = blockThreads ? static_cast<int64_t>(blockThreads) : autoThreads(columns);
    if (T < 32 || T > 1024 || T % 32 != 0)
      return failure();
    K = (Co + T - 1) / T;
    firstGather = first;
    secondGather = second;
    expandedColumnIndex = expand;
    collapsedResult = collapse;
    gatherChain = true;
    return success();
  }

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret)
      return failure();
    if (succeeded(matchGatherChain(body, ret)))
      return success();
    // Exactly one non-terminator op, and it produces the returned value.
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    if (!only)
      return failure();

    if (auto cas = dyn_cast<AtomicScatterOp>(only)) {
      if (ret.getNumOperands() != 2 || ret.getOperand(0) != cas.getUpdated() ||
          ret.getOperand(1) != cas.getObserved())
        return failure();
      auto dataTy = dyn_cast<RankedTensorType>(cas.getData().getType());
      auto indexTy = dyn_cast<RankedTensorType>(cas.getIndex().getType());
      auto compareTy = dyn_cast<RankedTensorType>(cas.getCompare().getType());
      auto srcTy = dyn_cast<RankedTensorType>(cas.getSrc().getType());
      auto updatedTy = dyn_cast<RankedTensorType>(cas.getUpdated().getType());
      auto observedTy = dyn_cast<RankedTensorType>(cas.getObserved().getType());
      int64_t candidateAxis = cas.getAxis();
      auto ordering = cas->getAttrOfType<StringAttr>("ordering");
      auto scope = cas->getAttrOfType<StringAttr>("memory_scope");
      auto retry = cas->getAttrOfType<StringAttr>("cas_retry");
      if (!dataTy || !indexTy || !compareTy || !srcTy || !updatedTy ||
          !observedTy || !dataTy.hasStaticShape() ||
          !indexTy.hasStaticShape() || !compareTy.hasStaticShape() ||
          !srcTy.hasStaticShape() || !updatedTy.hasStaticShape() ||
          !observedTy.hasStaticShape() || dataTy.getRank() != 2 ||
          indexTy.getRank() != 2 || compareTy.getRank() != 2 ||
          srcTy.getRank() != 2 || updatedTy.getRank() != 2 ||
          observedTy.getRank() != 2 || (candidateAxis != 0 && candidateAxis != 1) ||
          cas.getMode() != "along_axis" || cas.getAtomic() != "cas" ||
          (retry && retry.getValue() != "single_attempt") ||
          !isa<IntegerType>(indexTy.getElementType()) ||
          (!dataTy.getElementType().isInteger(32) &&
           !dataTy.getElementType().isInteger(64)) ||
          compareTy.getShape() != srcTy.getShape() ||
          indexTy.getShape() != srcTy.getShape() ||
          updatedTy.getShape() != dataTy.getShape() ||
          observedTy.getShape() != srcTy.getShape() ||
          compareTy.getElementType() != dataTy.getElementType() ||
          srcTy.getElementType() != dataTy.getElementType() ||
          updatedTy.getElementType() != dataTy.getElementType() ||
          observedTy.getElementType() != dataTy.getElementType() ||
          !ordering || !scope || failed(mapAtomicOrdering(ordering.getValue())) ||
          failed(mapAtomicScope(scope.getValue())))
        return failure();
      int64_t preservedAxis = candidateAxis == 0 ? 1 : 0;
      if (dataTy.getDimSize(0) <= 0 || dataTy.getDimSize(1) <= 0 ||
          srcTy.getDimSize(0) <= 0 || srcTy.getDimSize(1) <= 0 ||
          srcTy.getDimSize(preservedAxis) > dataTy.getDimSize(preservedAxis))
        return failure();
      atomicScatter = cas;
      axis = candidateAxis;
      return setGeometry(srcTy.getDimSize(0), srcTy.getDimSize(1));
    }

    if (requestComposeGatherChain)
      return failure();

    if (ret.getNumOperands() != 1 || ret.getOperand(0) != only->getResult(0))
      return failure();

    if (auto g = dyn_cast<GatherOp>(only)) {
      auto dataTy = dyn_cast<RankedTensorType>(g.getData().getType());
      auto indexTy = dyn_cast<RankedTensorType>(g.getIndex().getType());
      auto resultTy = dyn_cast<RankedTensorType>(g.getResult().getType());
      int64_t candidateAxis = g.getAxis();
      StringRef mode = g.getMode();
      if (!dataTy || !indexTy || !resultTy ||
          !dataTy.hasStaticShape() || !indexTy.hasStaticShape() ||
          !resultTy.hasStaticShape() || dataTy.getRank() != 2 ||
          resultTy.getRank() != 2 ||
          (candidateAxis != 0 && candidateAxis != 1) ||
          (mode != "take" && mode != "along_axis") ||
          !isa<IntegerType>(indexTy.getElementType()) ||
          resultTy.getElementType() != dataTy.getElementType())
        return failure();

      if (mode == "take") {
        if (indexTy.getRank() != 1)
          return failure();
        SmallVector<int64_t> expectedShape(dataTy.getShape());
        expectedShape[candidateAxis] = indexTy.getDimSize(0);
        if (resultTy.getShape() != ArrayRef<int64_t>(expectedShape))
          return failure();
      } else {
        if (indexTy.getRank() != 2 ||
            resultTy.getShape() != indexTy.getShape())
          return failure();
        int64_t preservedAxis = candidateAxis == 0 ? 1 : 0;
        if (resultTy.getDimSize(preservedAxis) >
            dataTy.getDimSize(preservedAxis))
          return failure();
      }
      if (dataTy.getDimSize(0) <= 0 || dataTy.getDimSize(1) <= 0 ||
          resultTy.getDimSize(0) <= 0 || resultTy.getDimSize(1) <= 0)
        return failure();
      gather = g;
      axis = candidateAxis;
      return setGeometry(resultTy.getDimSize(0), resultTy.getDimSize(1));
    }
    if (auto s = dyn_cast<ScatterOp>(only)) {
      auto dataTy = dyn_cast<RankedTensorType>(s.getData().getType());
      auto indexTy = dyn_cast<RankedTensorType>(s.getIndex().getType());
      auto srcTy = dyn_cast<RankedTensorType>(s.getSrc().getType());
      auto resultTy = dyn_cast<RankedTensorType>(s.getResult().getType());
      int64_t candidateAxis = s.getAxis();
      if (!dataTy || !indexTy || !srcTy || !resultTy ||
          !dataTy.hasStaticShape() || !indexTy.hasStaticShape() ||
          !srcTy.hasStaticShape() || !resultTy.hasStaticShape() ||
          dataTy.getRank() != 2 || indexTy.getRank() != 2 ||
          srcTy.getRank() != 2 || resultTy.getRank() != 2 ||
          (candidateAxis != 0 && candidateAxis != 1) ||
          s.getMode() != "along_axis" ||
          (s.getReduce() != "none" && s.getReduce() != "add") ||
          (s.getReduce() == "add" &&
           !isa<FloatType>(dataTy.getElementType())) ||
          !isa<IntegerType>(indexTy.getElementType()) ||
          srcTy.getShape() != indexTy.getShape() ||
          resultTy.getShape() != dataTy.getShape() ||
          srcTy.getElementType() != dataTy.getElementType() ||
          resultTy.getElementType() != dataTy.getElementType())
        return failure();
      if (s->hasAttr("atomic")) {
        auto atomic = s->getAttrOfType<StringAttr>("atomic");
        auto ordering = s->getAttrOfType<StringAttr>("ordering");
        auto scope = s->getAttrOfType<StringAttr>("memory_scope");
        if (!atomic || !ordering || !scope || atomic.getValue() != "add" ||
            s.getReduce() != "add" ||
            dataTy.getElementType() != Float32Type::get(s->getContext()) ||
            failed(mapAtomicOrdering(ordering.getValue())) ||
            failed(mapAtomicScope(scope.getValue())))
          return failure();
        explicitAtomicAdd = true;
      }
      int64_t preservedAxis = candidateAxis == 0 ? 1 : 0;
      if (dataTy.getDimSize(0) <= 0 || dataTy.getDimSize(1) <= 0 ||
          srcTy.getDimSize(preservedAxis) >
              dataTy.getDimSize(preservedAxis) ||
          srcTy.getDimSize(0) <= 0 || srcTy.getDimSize(1) <= 0)
        return failure();
      scatter = s;
      axis = candidateAxis;
      return setGeometry(srcTy.getDimSize(0), srcTy.getDimSize(1));
    }
    return failure();
  }

  // ---- emission ---------------------------------------------------------- //
  // Load index[coord] and cast it to an `index`-typed value for use as a
  // data-memref coordinate. `coord` is 1-D (take: index is rank-1) or 2-D.
  Value loadIndex(OpBuilder &b, Location loc, Value idxMem, ValueRange coord) {
    Value raw = b.create<memref::LoadOp>(loc, idxMem, coord);
    if (raw.getType().isIndex())
      return raw;
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), raw);
  }

  void attachGatherChainState(func::FuncOp lowered, StringRef state) {
    MLIRContext *ctx = lowered.getContext();
    auto i64 = IntegerType::get(ctx, 64);
    bool composed = state.contains("composed");
    lowered->setAttr("intent.backend.lowering_path_kind",
                     StringAttr::get(ctx, composed ? "gather_composed"
                                                   : "gather"));
    lowered->setAttr("intent.backend.reduction_topology",
                     StringAttr::get(ctx, "none"));
    lowered->setAttr("intent.backend.approximation_mode",
                     StringAttr::get(ctx, "none"));
    lowered->setAttr(
        "intent.backend.transformation_trace",
        ArrayAttr::get(ctx, {StringAttr::get(ctx, state)}));
    lowered->setAttr("intent.backend.block_threads",
                     IntegerAttr::get(i64, T));
    lowered->setAttr("intent.backend.grid_blocks",
                     IntegerAttr::get(i64, composed ? (Ro + T - 1) / T : Ro));
    lowered->setAttr("intent.backend.vector_width", IntegerAttr::get(i64, 1));
    for (StringRef name : {"fma_count", "shuffle_count", "all_reduce_count",
                           "predicated_value_if_count",
                           "folded_constexpr_select_count"})
      lowered->setAttr(("intent.backend." + name).str(),
                       IntegerAttr::get(i64, 0));
    lowered->setAttr("intent.backend.barrier_count",
                     IntegerAttr::get(i64, composed ? 0 : 1));
    lowered->setAttr("intent.backend.gather_chain_depth",
                     IntegerAttr::get(i64, 2));
  }

  LogicalResult rewriteGatherChainMaterialized() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto dataTy = cast<RankedTensorType>(firstGather.getData().getType());
    auto rowIndexTy = cast<RankedTensorType>(firstGather.getIndex().getType());
    auto colIndexTy = cast<RankedTensorType>(expandedColumnIndex.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(collapsedResult.getResult().getType());
    Type elementType = dataTy.getElementType();
    unsigned bitWidth = elementType.getIntOrFloatBitWidth();
    if (bitWidth == 0 || bitWidth % 8 != 0)
      return failure();

    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(rowIndexTy),
                            memrefOf(colIndexTy), memrefOf(resultTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    copyIntentFunctionAttrs(fn, gpuFn);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value rowIndexMem = entry->getArgument(1);
    Value colIndexMem = entry->getArgument(2);
    Value outMem = entry->getArgument(3);

    Value cRows = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cThreads = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value sharedBytes = b.create<arith::ConstantOp>(
        loc, b.getI32IntegerAttr(static_cast<int32_t>(Co * bitWidth / 8)));
    auto launch = b.create<gpu::LaunchOp>(loc, cRows, c1, c1, cThreads,
                                         c1, c1, sharedBytes);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value cColumns = b.create<arith::ConstantIndexOp>(loc, Co);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto rowTy = MemRefType::get({Co}, elementType, nullptr, wgSpace);
    Value rowScratch =
        b.create<memref::ViewOp>(loc, rowTy, raw, c0, ValueRange{});

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0) {
        Value offset = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, offset);
      }
      Value active = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, col, cColumns);
      auto guarded = b.create<scf::IfOp>(loc, active, false);
      OpBuilder thenBuilder = guarded.getThenBodyBuilder();
      Value row = loadIndex(thenBuilder, loc, rowIndexMem, ValueRange{bx});
      Value value = thenBuilder.create<memref::LoadOp>(
          loc, dataMem, ValueRange{row, col});
      thenBuilder.create<memref::StoreOp>(loc, value, rowScratch,
                                          ValueRange{col});
    }
    b.create<gpu::BarrierOp>(loc);
    Value leader = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, tx, c0);
    auto leaderOnly = b.create<scf::IfOp>(loc, leader, false);
    OpBuilder leaderBuilder = leaderOnly.getThenBodyBuilder();
    Value selectedColumn =
        loadIndex(leaderBuilder, loc, colIndexMem, ValueRange{bx});
    Value value = leaderBuilder.create<memref::LoadOp>(
        loc, rowScratch, ValueRange{selectedColumn});
    leaderBuilder.create<memref::StoreOp>(loc, value, outMem, ValueRange{bx});

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    attachGatherChainState(gpuFn, "cuda.gather-chain.materialized-row.v1");
    fn.erase();
    return success();
  }

  LogicalResult rewriteGatherChainComposed() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto dataTy = cast<RankedTensorType>(firstGather.getData().getType());
    auto rowIndexTy = cast<RankedTensorType>(firstGather.getIndex().getType());
    auto colIndexTy = cast<RankedTensorType>(expandedColumnIndex.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(collapsedResult.getResult().getType());
    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(rowIndexTy),
                            memrefOf(colIndexTy), memrefOf(resultTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    copyIntentFunctionAttrs(fn, gpuFn);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value rowIndexMem = entry->getArgument(1);
    Value colIndexMem = entry->getArgument(2);
    Value outMem = entry->getArgument(3);

    int64_t blocks = (Ro + T - 1) / T;
    Value cBlocks = b.create<arith::ConstantIndexOp>(loc, blocks);
    Value cThreads = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch =
        b.create<gpu::LaunchOp>(loc, cBlocks, c1, c1, cThreads, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value linear = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, bx, cT), tx);
    Value cRows = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value active = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, linear, cRows);
    auto guarded = b.create<scf::IfOp>(loc, active, false);
    OpBuilder thenBuilder = guarded.getThenBodyBuilder();
    Value row =
        loadIndex(thenBuilder, loc, rowIndexMem, ValueRange{linear});
    Value column =
        loadIndex(thenBuilder, loc, colIndexMem, ValueRange{linear});
    Value value = thenBuilder.create<memref::LoadOp>(
        loc, dataMem, ValueRange{row, column});
    thenBuilder.create<memref::StoreOp>(loc, value, outMem,
                                        ValueRange{linear});

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    attachGatherChainState(gpuFn, "cuda.gather-chain.composed-index.v1");
    fn.erase();
    return success();
  }

  LogicalResult rewriteGather() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    auto dataTy = cast<RankedTensorType>(gather.getData().getType());
    auto idxTy = cast<RankedTensorType>(gather.getIndex().getType());
    auto outTy = cast<RankedTensorType>(gather.getResult().getType());
    bool alongAxis = gather.getMode() == "along_axis";

    // New signature: data, index (as memrefs) ++ one [Ro,Co] output memref.
    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(idxTy), memrefOf(outTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    copyIntentFunctionAttrs(fn, gpuFn);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value idxMem = entry->getArgument(1);
    Value outMem = entry->getArgument(2);

    Value cRo = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cRo, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, off);
      }
      // Data-dependent load coordinate: read the index, replace the `axis`
      // coordinate of the data access with it.
      Value idx;
      if (alongAxis) // index is [Ro,Co]: read at the output coordinate
        idx = loadIndex(b, loc, idxMem, ValueRange{bx, col});
      else           // take: index is rank-1, selecting along `axis`
        idx = loadIndex(b, loc, idxMem, ValueRange{axis == 0 ? bx : col});
      Value r0 = axis == 0 ? idx : bx;
      Value r1 = axis == 0 ? col : idx;
      Value v = b.create<memref::LoadOp>(loc, dataMem, ValueRange{r0, r1});
      b.create<memref::StoreOp>(loc, v, outMem, ValueRange{bx, col});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }

  LogicalResult rewriteScatter() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    auto dataTy = cast<RankedTensorType>(scatter.getData().getType());
    auto idxTy = cast<RankedTensorType>(scatter.getIndex().getType());
    auto srcTy = cast<RankedTensorType>(scatter.getSrc().getType());
    bool add = scatter.getReduce() == "add";
    bool explicitAdd = explicitAtomicAdd;
    LLVM::AtomicOrdering ordering = LLVM::AtomicOrdering::monotonic;
    StringRef syncScope;
    if (explicitAdd) {
      auto orderingAttr = scatter->getAttrOfType<StringAttr>("ordering");
      auto scopeAttr = scatter->getAttrOfType<StringAttr>("memory_scope");
      auto mappedOrdering = mapAtomicOrdering(orderingAttr.getValue());
      auto mappedScope = mapAtomicScope(scopeAttr.getValue());
      if (failed(mappedOrdering) || failed(mappedScope))
        return failure();
      ordering = *mappedOrdering;
      syncScope = *mappedScope;
    }

    // New signature: data (in/out, caller-cloned), index, src — all memrefs.
    // Scatter is IN-PLACE on `data`: no appended output arg.
    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(idxTy), memrefOf(srcTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    copyIntentFunctionAttrs(fn, gpuFn);
    if (explicitAdd)
      attachAtomicMetadata(gpuFn, "add",
                           scatter->getAttrOfType<StringAttr>("ordering")
                               .getValue(),
                           scatter->getAttrOfType<StringAttr>("memory_scope")
                               .getValue());
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value idxMem = entry->getArgument(1);
    Value srcMem = entry->getArgument(2);

    Value cRo = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cRo, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, off);
      }
      Value v = b.create<memref::LoadOp>(loc, srcMem, ValueRange{bx, col});
      Value idx = loadIndex(b, loc, idxMem, ValueRange{bx, col});
      // Data-dependent store coordinate on `data`: replace the `axis` coord.
      Value r0 = axis == 0 ? idx : bx;
      Value r1 = axis == 0 ? col : idx;
      if (explicitAdd) {
        Value address = atomicAddress(b, loc, dataMem, r0, r1,
                                      dataTy.getDimSize(1), 4);
        StringRef semanticOrdering =
            scatter->getAttrOfType<StringAttr>("ordering").getValue();
        StringRef semanticScope =
            scatter->getAttrOfType<StringAttr>("memory_scope").getValue();
        emitAtomicFence(b, loc, "add", semanticOrdering, semanticScope,
                        StringRef(), /*before=*/true);
        Value old = emitScopedAtomicAdd(b, loc, address, v, syncScope);
        if (old) {
          Operation *atomic = old.getDefiningOp();
          attachAtomicMetadata(atomic, "add", semanticOrdering, semanticScope);
        } else {
          auto atomic = b.create<LLVM::AtomicRMWOp>(
              loc, LLVM::AtomicBinOp::fadd, address, v, ordering, syncScope);
          attachAtomicMetadata(atomic, "add", semanticOrdering, semanticScope);
        }
        emitAtomicFence(b, loc, "add", semanticOrdering, semanticScope,
                        StringRef(), /*before=*/false);
      } else if (add)
        b.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addf, v,
                                      dataMem, ValueRange{r0, r1});
      else
        b.create<memref::StoreOp>(loc, v, dataMem, ValueRange{r0, r1});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }

  LogicalResult rewriteAtomicScatter() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    auto dataTy = cast<RankedTensorType>(atomicScatter.getData().getType());
    auto idxTy = cast<RankedTensorType>(atomicScatter.getIndex().getType());
    auto compareTy =
        cast<RankedTensorType>(atomicScatter.getCompare().getType());
    auto srcTy = cast<RankedTensorType>(atomicScatter.getSrc().getType());
    auto observedTy =
        cast<RankedTensorType>(atomicScatter.getObserved().getType());
    auto orderingAttr = atomicScatter->getAttrOfType<StringAttr>("ordering");
    auto scopeAttr = atomicScatter->getAttrOfType<StringAttr>("memory_scope");
    auto retryAttr = atomicScatter->getAttrOfType<StringAttr>("cas_retry");
    auto mappedOrdering = mapAtomicOrdering(orderingAttr.getValue());
    auto mappedScope = mapAtomicScope(scopeAttr.getValue());
    if (failed(mappedOrdering) || failed(mappedScope) ||
        (retryAttr && retryAttr.getValue() != "single_attempt"))
      return failure();
    LLVM::AtomicOrdering successOrdering = *mappedOrdering;
    LLVM::AtomicOrdering failureOrdering =
        mapCmpXchgFailureOrdering(orderingAttr.getValue());
    StringRef syncScope = *mappedScope;

    // The updated result aliases data in the admitted CUDA ABI. The observed
    // value is the one fresh destination passed after the four input memrefs.
    SmallVector<Type> inTys{memrefOf(dataTy), memrefOf(idxTy), memrefOf(compareTy),
                            memrefOf(srcTy), memrefOf(observedTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    copyIntentFunctionAttrs(fn, gpuFn);
    attachAtomicMetadata(gpuFn, "cas", orderingAttr.getValue(),
                         scopeAttr.getValue(),
                         retryAttr ? retryAttr.getValue() : "single_attempt");
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(0);
    Value idxMem = entry->getArgument(1);
    Value compareMem = entry->getArgument(2);
    Value srcMem = entry->getArgument(3);
    Value observedMem = entry->getArgument(4);

    Value cRo = b.create<arith::ConstantIndexOp>(loc, Ro);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cRo, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    int64_t elementBytes =
        dataTy.getElementType().getIntOrFloatBitWidth() / 8;

    for (int64_t k = 0; k < K; ++k) {
      Value col = tx;
      if (k != 0) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        col = b.create<arith::AddIOp>(loc, tx, off);
      }
      Value compare =
          b.create<memref::LoadOp>(loc, compareMem, ValueRange{bx, col});
      Value replacement =
          b.create<memref::LoadOp>(loc, srcMem, ValueRange{bx, col});
      Value idx = loadIndex(b, loc, idxMem, ValueRange{bx, col});
      Value r0 = axis == 0 ? idx : bx;
      Value r1 = axis == 0 ? col : idx;
      Value address = atomicAddress(b, loc, dataMem, r0, r1,
                                    dataTy.getDimSize(1), elementBytes);
      StringRef semanticOrdering = orderingAttr.getValue();
      StringRef semanticScope = scopeAttr.getValue();
      StringRef retry = retryAttr ? retryAttr.getValue() : "single_attempt";
      emitAtomicFence(b, loc, "cas", semanticOrdering, semanticScope, retry,
                      /*before=*/true);
      Value observed = emitScopedAtomicCas(b, loc, address, compare,
                                           replacement, syncScope);
      if (observed) {
        Operation *cmpxchg = observed.getDefiningOp();
        attachAtomicMetadata(cmpxchg, "cas", semanticOrdering, semanticScope,
                             retry);
      } else {
        auto cmpxchg = b.create<LLVM::AtomicCmpXchgOp>(
            loc, address, compare, replacement, successOrdering,
            failureOrdering, syncScope);
        attachAtomicMetadata(cmpxchg, "cas", semanticOrdering, semanticScope,
                             retry);
        observed = b.create<LLVM::ExtractValueOp>(loc, cmpxchg, 0);
      }
      emitAtomicFence(b, loc, "cas", semanticOrdering, semanticScope, retry,
                      /*before=*/false);
      b.create<memref::StoreOp>(loc, observed, observedMem,
                                ValueRange{bx, col});
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }

  LogicalResult rewrite() {
    if (gatherChain)
      return requestComposeGatherChain ? rewriteGatherChainComposed()
                                       : rewriteGatherChainMaterialized();
    if (gather)
      return rewriteGather();
    if (atomicScatter)
      return rewriteAtomicScatter();
    return rewriteScatter();
  }
};

struct ConvertIntentGatherToGPUPass
    : ::intent::impl::ConvertIntentGatherToGPUBase<
          ConvertIntentGatherToGPUPass> {
  using ConvertIntentGatherToGPUBase::ConvertIntentGatherToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      GatherScatterLowering probe(fn, blockThreads, composeGatherChain);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }

    // Selection probes are deliberately structural.  Once an explicit atomic
    // node is present, a probe miss must be a capability rejection rather than
    // permission to fall through to the legacy implicit-add/store lowering.
    for (func::FuncOp fn : mod.getOps<func::FuncOp>()) {
      bool admitted = llvm::is_contained(targets, fn);
      if (admitted)
        continue;
      fn.walk([&](Operation *op) {
        if (!GatherScatterLowering::hasAtomicMetadata(op))
          return;
        InFlightDiagnostic diag = op->emitError()
                                   << "CUDA capability rejection: ";
        diag << GatherScatterLowering::atomicCapabilityReason(op)
             << "; generic lowering refuses to erase atomic metadata";
        signalPassFailure();
      });
    }
    for (func::FuncOp fn : targets) {
      GatherScatterLowering low(fn, blockThreads, composeGatherChain);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("gather/scatter lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
