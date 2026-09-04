//===- ConvertIntentElementwiseToGPU.cpp - elementwise -> gpu.launch ------===//
//
// Lowers a PURE-elementwise intent func (a straight-line map_unary/map_binary
// graph with NO reduce) to ONE block-cooperative `gpu.launch`. Small rows use a
// read-once per-thread SSA register tile; large tensors switch to a bounded-size
// flat grid-stride loop so compiler IR does not grow with the element count.
// There is NO `gpu.all_reduce` — nothing is reduced.
//
// This is the sibling of `convert-intent-row-reduction-to-gpu`: that pass owns
// the funcs with >=1 reduce; this pass owns the pure-map funcs. The two match()
// predicates are disjoint (row-reduction returns failure when it finds no
// reduce; this pass returns failure the moment it finds one), so they compose in
// either order without overlap. It is general over the map DAG — there is NO
// per-kernel "recognize dyt/swiglu/geglu" branch; the three activation kernels
// (dyt = tanh(alpha*x)*gamma, swiglu = (a*sigmoid(a))*b, geglu = the tanh-approx
// GELU gate * b) flow through identically.
//
// Schedule per block (row `bx`), per thread `tx` owning columns col = tx + k*T:
//   load each Scalar arg once (uniform); hoist every fully-uniform map subtree;
//   load each Row input's K values ONCE into SSA (the register tile);
//   epilogue: for k=0..K, for each result: store <result evaluated at col> -> Y.
//
// Read-once register tile is the whole game: a dynamically-indexed alloca lands
// in local memory and caps bandwidth far below peak. Static shapes only
// (bare-ptr ABI). f16/bf16 storage is promoted to f32 within each semantic map
// op and rounded back at that op's typed result boundary. Emits
// math.exp/tanh/...; the `-lower-gpu-math-to-nvvm` peephole turns those into
// NVVM approx intrinsics downstream.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "MapEmit.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <string>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTELEMENTWISETOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// Paper extraction represents literal broadcasts as tensor splat constants.
/// They are uniform scalar values for this elementwise lowering, independent of
/// whether the source tensor type is rank-0 or a larger splat tensor.
static bool isSupportedSplatConstant(arith::ConstantOp cst) {
  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat())
    return false;
  auto shaped = dyn_cast<ShapedType>(dense.getType());
  return shaped && isa<FloatType, IntegerType>(shaped.getElementType());
}

static Value emitSplatScalarConstant(OpBuilder &b, Location loc,
                                     arith::ConstantOp cst) {
  auto dense = cast<DenseElementsAttr>(cst.getValue());
  Type elemTy = cast<ShapedType>(dense.getType()).getElementType();
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    return b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(floatTy, dense.getSplatValue<APFloat>()));
  auto intTy = cast<IntegerType>(elemTy);
  return b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(intTy, dense.getSplatValue<APInt>()));
}

static void attachElementwiseStateReceipt(func::FuncOp lowered, int64_t threads,
                                          int64_t gridBlocks,
                                          bool gridStrideFlat,
                                          int64_t gridStrideUnroll,
                                          int64_t logicalTileElements,
                                          int64_t vectorWidth,
                                          StringRef approximationMode,
                                          bool oneDimensionalResults,
                                          unsigned hostValueCaptureCount,
                                          unsigned shiftedExpCancellationCount = 0,
                                          bool pairedLastAxisRotation = false,
                                          bool pairedHeadAngleReuse = false,
                                          bool pairedDirectPlane = false,
                                          int64_t pairedHeadGroupSize = 1) {
  MLIRContext *context = lowered.getContext();
  SmallVector<Attribute> trace{
      StringAttr::get(context, "cuda.elementwise.row.vectorized.v1")};
  if (oneDimensionalResults)
    trace.push_back(
        StringAttr::get(context, "cuda.elementwise.rank1.output.v1"));
  if (gridStrideFlat)
    trace.push_back(
        StringAttr::get(context, "cuda.elementwise.flat_grid_stride.v1"));
  if (gridStrideFlat && gridStrideUnroll > 1)
    trace.push_back(StringAttr::get(
        context, "cuda.elementwise.flat_grid_stride.unroll.v1"));
  if (logicalTileElements > 0)
    trace.push_back(
        StringAttr::get(context, "cuda.elementwise.flat_logical_tile.v1"));
  if (approximationMode == "native_tanh")
    trace.push_back(
        StringAttr::get(context, "cuda.numeric.tanh_native_ptx.f32.v1"));
  else if (approximationMode == "stable_tanh")
    trace.push_back(
        StringAttr::get(context, "cuda.numeric.tanh_stable_ex2.f32.v1"));
  if (hostValueCaptureCount != 0)
    trace.push_back(
        StringAttr::get(context, "cuda.b.uniform_host_value_capture.v1"));
  if (shiftedExpCancellationCount != 0)
    trace.push_back(StringAttr::get(
        context, "cuda.algebra.exp_shift_product_cancellation.v1"));
  if (pairedLastAxisRotation)
    trace.push_back(StringAttr::get(
        context, "cuda.elementwise.paired_last_axis_rotation.v1"));
  if (pairedHeadAngleReuse)
    trace.push_back(StringAttr::get(
        context, "cuda.elementwise.paired_head_angle_reuse.v1"));
  if (pairedDirectPlane)
    trace.push_back(StringAttr::get(
        context, "cuda.elementwise.paired_direct_plane.v1"));
  auto i64 = IntegerType::get(context, 64);
  lowered->setAttr("intent.backend.lowering_path_kind",
                   StringAttr::get(context, "elementwise"));
  lowered->setAttr("intent.backend.reduction_topology",
                   StringAttr::get(context, "none"));
  lowered->setAttr("intent.backend.approximation_mode",
                   StringAttr::get(context, approximationMode));
  lowered->setAttr("intent.backend.transformation_trace",
                   ArrayAttr::get(context, trace));
  lowered->setAttr("intent.backend.fma_count", IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.shuffle_count", IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.barrier_count", IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.all_reduce_count", IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.block_threads",
                   IntegerAttr::get(i64, threads));
  lowered->setAttr("intent.backend.grid_blocks",
                   IntegerAttr::get(i64, gridBlocks));
  lowered->setAttr("intent.backend.grid_stride_loop_count",
                   IntegerAttr::get(i64, gridStrideFlat ? 1 : 0));
  lowered->setAttr("intent.backend.grid_stride_unroll",
                   IntegerAttr::get(i64, gridStrideUnroll));
  lowered->setAttr("intent.backend.logical_tile_elements",
                   IntegerAttr::get(i64, logicalTileElements));
  lowered->setAttr("intent.backend.vector_width",
                   IntegerAttr::get(i64, vectorWidth));
  lowered->setAttr("intent.backend.predicated_value_if_count",
                   IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.folded_constexpr_select_count",
                   IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.host_value_capture_count",
                   IntegerAttr::get(i64, hostValueCaptureCount));
  lowered->setAttr("intent.backend.epilogue_load_motion_count",
                   IntegerAttr::get(i64, 0));
  lowered->setAttr("intent.backend.shifted_exp_cancellation_count",
                   IntegerAttr::get(i64, shiftedExpCancellationCount));
  lowered->setAttr("intent.backend.paired_head_reuse_count",
                   IntegerAttr::get(i64, pairedHeadAngleReuse ? 1 : 0));
  lowered->setAttr("intent.backend.paired_head_group_size",
                   IntegerAttr::get(i64, pairedHeadAngleReuse
                                             ? pairedHeadGroupSize
                                             : 1));
  lowered->setAttr("intent.backend.paired_direct_plane_count",
                   IntegerAttr::get(i64, pairedDirectPlane ? 1 : 0));
}

struct ShiftedExpProduct {
  Operation *product;
  Value base;
};

/// Match the real-arithmetic identity exp(x - shift) * exp(shift) = exp(x).
/// The transform is deliberately not part of the hard-preserving path: target
/// rounding and overflow behavior may differ, so callers expose it only as an
/// observation-changing candidate guarded by the ordinary backend Q/D policy.
static Value matchShiftedExpProduct(MapUnaryOp shiftedExp,
                                    MapUnaryOp shiftExp) {
  if (!shiftedExp || !shiftExp || shiftedExp.getFn() != "exp" ||
      shiftExp.getFn() != "exp")
    return {};
  auto subtract = shiftedExp.getInput().getDefiningOp<MapBinaryOp>();
  if (!subtract || subtract.getFn() != "sub")
    return {};
  Value shift = subtract.getRhs();
  while (auto expand = shift.getDefiningOp<tensor::ExpandShapeOp>())
    shift = expand.getSrc();
  if (shift != shiftExp.getInput())
    return {};
  return subtract.getLhs();
}

static Value matchShiftedExpProduct(ContractionOp product) {
  auto lhsExp = product.getA().getDefiningOp<MapUnaryOp>();
  auto rhsExp = product.getB().getDefiningOp<MapUnaryOp>();
  Value shiftedBase = matchShiftedExpProduct(lhsExp, rhsExp);
  if (!shiftedBase)
    shiftedBase = matchShiftedExpProduct(rhsExp, lhsExp);
  if (!shiftedBase)
    return {};

  auto lhsType = dyn_cast<RankedTensorType>(product.getA().getType());
  auto rhsType = dyn_cast<RankedTensorType>(product.getB().getType());
  auto resultType = dyn_cast<RankedTensorType>(product.getResult().getType());
  if (!lhsType || !rhsType || !resultType || resultType.getRank() != 2 ||
      !((lhsType.getRank() == 2 && rhsType.getRank() == 1) ||
        (lhsType.getRank() == 1 && rhsType.getRank() == 2)) ||
      shiftedBase.getType() != product.getResult().getType())
    return {};
  auto pieces = product.getSubscripts().split("->");
  if (pieces.second.empty())
    return {};
  auto inputs = pieces.first.split(",");
  StringRef lhsLabels = inputs.first.trim();
  StringRef rhsLabels = inputs.second.trim();
  StringRef outputLabels = pieces.second.trim();
  if (!((lhsLabels.size() == 2 && rhsLabels.size() == 1 &&
         outputLabels == lhsLabels && rhsLabels[0] == lhsLabels[0]) ||
        (lhsLabels.size() == 1 && rhsLabels.size() == 2 &&
         outputLabels == rhsLabels && lhsLabels[0] == rhsLabels[0])))
    return {};
  return shiftedBase;
}

/// Rewrite only when cancellation removes every live reduction from the graph.
/// That makes the result an ordinary pure-elementwise Intent function consumed
/// by the existing MLIR emitter; no Python implementation or kernel-name recipe
/// is involved.
static unsigned rewriteShiftedExpProducts(func::FuncOp fn) {
  if (!fn.getBody().hasOneBlock())
    return 0;
  Block &body = fn.getBody().front();
  auto returned = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returned)
    return 0;

  SmallVector<ShiftedExpProduct> matches;
  DenseMap<Value, Value> replacementBases;
  for (Operation &operation : body.without_terminator()) {
    Operation *product = &operation;
    Value base;
    if (auto map = dyn_cast<MapBinaryOp>(&operation)) {
      if (map.getFn() != "mul")
        continue;
      auto lhsExp = map.getLhs().getDefiningOp<MapUnaryOp>();
      auto rhsExp = map.getRhs().getDefiningOp<MapUnaryOp>();
      base = matchShiftedExpProduct(lhsExp, rhsExp);
      if (!base)
        base = matchShiftedExpProduct(rhsExp, lhsExp);
    } else if (auto contraction = dyn_cast<ContractionOp>(&operation)) {
      base = matchShiftedExpProduct(contraction);
    } else {
      continue;
    }
    if (!base || base.getType() != operation.getResult(0).getType())
      continue;
    matches.push_back({product, base});
    replacementBases[operation.getResult(0)] = base;
  }
  if (matches.empty())
    return 0;

  // Check liveness with the proposed products treated as direct exp(base)
  // nodes. Any remaining reachable reduction means this candidate does not
  // actually open an elementwise implementation and must fail closed.
  SmallVector<Value> worklist(returned.getOperands().begin(),
                              returned.getOperands().end());
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    if (Value base = replacementBases.lookup(value)) {
      worklist.push_back(base);
      continue;
    }
    if (isa<BlockArgument>(value))
      continue;
    Operation *definition = value.getDefiningOp();
    if (!definition)
      return 0;
    if (isa<ReduceOp>(definition))
      return 0;
    worklist.append(definition->operand_begin(), definition->operand_end());
  }

  for (ShiftedExpProduct match : matches) {
    OpBuilder builder(match.product);
    auto directExp = builder.create<MapUnaryOp>(
        match.product->getLoc(), match.product->getResult(0).getType(), match.base,
        "exp");
    match.product->getResult(0).replaceAllUsesWith(directExp.getResult());
  }

  // Erase only the known-pure Intent/constant operations that became dead. The
  // newly created direct exp nodes stay live through the original return DAG.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> operations;
    for (Operation &operation : body.without_terminator())
      operations.push_back(&operation);
    for (Operation *operation : llvm::reverse(operations)) {
      if (!isa<ReduceOp, ContractionOp, MapUnaryOp, MapBinaryOp, SelectOp,
               IotaOp, CastOp, tensor::ExpandShapeOp,
               arith::ConstantOp>(operation) ||
          !llvm::all_of(operation->getResults(),
                        [](Value result) { return result.use_empty(); }))
        continue;
      operation->erase();
      changed = true;
    }
  }
  fn->setAttr("intent.cuda.shifted_exp_cancellation_count",
              IntegerAttr::get(IntegerType::get(fn.getContext(), 64),
                               matches.size()));
  return matches.size();
}

/// One func's pure-elementwise lowering. Reads the original (tensor) func body,
/// builds a new memref+gpu.launch func with the same name, erases the old.
struct ElementwiseLowering {
  func::FuncOp fn;
  unsigned blockThreads;
  bool requestFlattenContiguous;
  unsigned requestFlatVectorWidth;
  unsigned requestGridStrideUnroll;
  unsigned requestLogicalTileElements;
  unsigned requestRowVectorWidth;
  bool requestPromoteUniformScalars;
  bool requestRowMaskZeroFill;
  std::string requestTanhApproximation;

  // The 2-D logical layout of one register-tile access. A Row/result arg is
  // accessed at element (row*rowStride + col*colStride) of its bare pointer; the
  // strides come from an `intent.strides = [row, col]` arg-attr (default the
  // contiguous [N, 1]). A NON-default stride means the arg is a strided VIEW of a
  // larger buffer (rope's q_lo/q_hi half-views have rowStride = hd = 2N; llama4's
  // even/odd views have colStride = 2): the index-remap that makes rope a pure
  // elementwise op at matching logical column instead of a cross-column rotation.
  struct Strides {
    int64_t row, col;
    bool isDefault(int64_t N) const { return row == N && col == 1; }
  };

  // populated by match()
  int64_t M = 0, N = 0, T = 0, K = 0, gridBlocks = 0;
  bool oneDimensionalResults = false;
  bool flattenContiguous = false;
  bool gridStrideFlat = false;
  bool tailMasked = false;
  // Contiguous vectorization (N3/N4): prefer float4 when K is divisible by 4,
  // otherwise use float2 when K is divisible by 2. Every Row arg/result must
  // be contiguous and the body must have no iota/select scalar gather. This
  // lets a four-warp N=256 schedule retain Triton's coalesced v2 memory shape
  // while the normalized Intent DAG still exposes target-level fast sigmoid.
  int64_t vectorWidth = 1;
  int64_t numVectorTiles = 0;
  Type storageType;
  Type computeType;
  bool containsTanh = false;
  SmallVector<Role> argRoles;        // per func arg
  SmallVector<Type> argElementTypes; // per func arg storage/broadcast type
  SmallVector<int64_t> argRowDiv;    // per arg: 1, or M/d0 for a broadcast Row
  SmallVector<Strides> argStrides;   // per arg: [row,col] (Row only; else unused)
  SmallVector<Strides> resStrides;   // per result: [row,col]
  SmallVector<int64_t> inplaceArg;   // per result: aliased input arg idx, or -1
  SmallVector<Value> outputVals;     // the returned values (>=1, all [M,N])

  ElementwiseLowering(func::FuncOp fn, unsigned bt, bool flatten,
                      unsigned flatWidth, unsigned gridUnroll,
                      unsigned logicalTileElements,
                      unsigned rowVectorWidth,
                      bool promoteUniformScalars,
                      bool rowMaskZeroFill,
                      StringRef tanhApproximation)
      : fn(fn), blockThreads(bt), requestFlattenContiguous(flatten),
        requestFlatVectorWidth(flatWidth),
        requestGridStrideUnroll(gridUnroll),
        requestLogicalTileElements(logicalTileElements),
        requestRowVectorWidth(rowVectorWidth),
        requestPromoteUniformScalars(promoteUniformScalars),
        requestRowMaskZeroFill(rowMaskZeroFill),
        requestTanhApproximation(tanhApproximation.str()) {}

  /// Memref type for a Row/result view in the function's storage type.
  /// `memref<d0 x N x storage>` uses identity layout when contiguous, otherwise
  /// a strided
  /// `memref<d0 x N x storage, strided<[row, col], offset: 0>>`. Offset is 0: the
  /// bare pointer the harness passes is the strided torch view's data_ptr, which
  /// already folds in any base offset (q[:, hd/2:].data_ptr()).
  MemRefType rowMemref(MLIRContext *ctx, int64_t d0, Strides s) {
    if (s.isDefault(N))
      return MemRefType::get({d0, N}, storageType);
    return MemRefType::get({d0, N}, storageType,
                           StridedLayoutAttr::get(ctx, 0, {s.row, s.col}));
  }

  /// Vector-element memref for a contiguous Row/result view. Only used when
  /// vectorWidth > 1; the caller has already proved default contiguous strides.
  MemRefType rowVecMemref(MLIRContext *ctx, int64_t d0) {
    return MemRefType::get({d0, N / vectorWidth},
                           VectorType::get({vectorWidth}, storageType));
  }

  MemRefType flatVecMemref(MLIRContext *ctx) {
    return MemRefType::get({M * N / vectorWidth},
                           VectorType::get({vectorWidth}, storageType));
  }

  /// Destination-passing ABI for a result.  Public rank-1 tensors remain
  /// rank-1 memrefs; they are not silently reshaped to [1,N], because that
  /// would make the compiler ABI disagree with the accepted J shape.
  MemRefType resultMemref(MLIRContext *ctx, Strides s) {
    if (oneDimensionalResults)
      return MemRefType::get({N}, storageType);
    return rowMemref(ctx, M, s);
  }

  MemRefType resultVecMemref(MLIRContext *ctx) {
    if (oneDimensionalResults)
      return MemRefType::get({N / vectorWidth},
                             VectorType::get({vectorWidth}, storageType));
    return rowVecMemref(ctx, M);
  }

  /// A contraction with no label eliminated by the result is an elementwise
  /// multiply with typed broadcasting, not a reduction or matrix product.
  bool isPointwiseContraction(ContractionOp op) const {
    StringRef subscripts = op.getSubscripts();
    size_t comma = subscripts.find(',');
    size_t arrow = subscripts.find("->");
    if (comma == StringRef::npos || arrow == StringRef::npos || comma > arrow)
      return false;
    StringRef lhs = subscripts.take_front(comma);
    StringRef rhs = subscripts.slice(comma + 1, arrow);
    StringRef output = subscripts.drop_front(arrow + 2);
    auto lhsTy = dyn_cast<RankedTensorType>(op.getA().getType());
    auto rhsTy = dyn_cast<RankedTensorType>(op.getB().getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!lhsTy || !rhsTy || !outTy ||
        lhs.size() != static_cast<size_t>(lhsTy.getRank()) ||
        rhs.size() != static_cast<size_t>(rhsTy.getRank()) ||
        output.size() != static_cast<size_t>(outTy.getRank()))
      return false;
    auto uniqueLabels = [](StringRef labels) {
      for (char label : labels)
        if (labels.count(label) != 1)
          return false;
      return true;
    };
    if (!uniqueLabels(lhs) || !uniqueLabels(rhs) || !uniqueLabels(output))
      return false;
    for (char label : lhs)
      if (!output.contains(label))
        return false;
    for (char label : rhs)
      if (!output.contains(label))
        return false;
    for (char label : output)
      if (!lhs.contains(label) && !rhs.contains(label))
        return false;
    return true;
  }

  bool isRowMaskGraph(Block &body, func::ReturnOp ret) const {
    SmallVector<Operation *> operations;
    for (Operation &operation : body.without_terminator())
      operations.push_back(&operation);
    if (operations.size() != 2 || ret.getNumOperands() != 1)
      return false;
    auto castOp = dyn_cast<CastOp>(operations[0]);
    auto product = dyn_cast<ContractionOp>(operations[1]);
    if (!castOp || !product || ret.getOperand(0) != product.getResult() ||
        !isPointwiseContraction(product))
      return false;
    auto maskArg = dyn_cast<BlockArgument>(castOp.getInput());
    Value dataValue = product.getA() == castOp.getResult()
                          ? product.getB()
                          : product.getB() == castOp.getResult()
                                ? product.getA()
                                : Value();
    auto dataArg = dyn_cast_or_null<BlockArgument>(dataValue);
    auto maskTy = dyn_cast<RankedTensorType>(castOp.getInput().getType());
    auto castTy = dyn_cast<RankedTensorType>(castOp.getResult().getType());
    auto dataTy = dataValue ? dyn_cast<RankedTensorType>(dataValue.getType())
                            : RankedTensorType();
    auto resultTy = dyn_cast<RankedTensorType>(product.getResult().getType());
    if (!maskArg || !dataArg || !maskTy || !castTy || !dataTy || !resultTy ||
        maskArg.getArgNumber() == dataArg.getArgNumber() ||
        maskTy.getShape() != ArrayRef<int64_t>({M}) ||
        !maskTy.getElementType().isInteger(1) ||
        castTy.getShape() != ArrayRef<int64_t>({M}) ||
        !castTy.getElementType().isF32() ||
        dataTy.getShape() != ArrayRef<int64_t>({M, N}) ||
        resultTy.getShape() != ArrayRef<int64_t>({M, N}) ||
        dataTy.getElementType() != resultTy.getElementType() ||
        !dataTy.getElementType().isF32())
      return false;
    StringRef expected = product.getA() == dataValue ? "ab,a->ab" : "a,ab->ab";
    return product.getSubscripts() == expected;
  }

  // ---- recognition ------------------------------------------------------- //
  LogicalResult match() {
    if (requestTanhApproximation != "stable-ex2" &&
        requestTanhApproximation != "native-ptx")
      return failure();
    if (requestGridStrideUnroll != 1 && requestGridStrideUnroll != 2 &&
        requestGridStrideUnroll != 4 && requestGridStrideUnroll != 8)
      return failure();
    if (requestRowVectorWidth != 0 && requestRowVectorWidth != 1 &&
        requestRowVectorWidth != 2 && requestRowVectorWidth != 4)
      return failure();
    if (requestLogicalTileElements != 0 &&
        (requestLogicalTileElements < 32 || requestLogicalTileElements > 8192 ||
         (requestLogicalTileElements & (requestLogicalTileElements - 1)) != 0))
      return failure();
    auto fty = fn.getFunctionType();
    unsigned nRes = fty.getNumResults();
    if (nRes < 1)
      return failure();
    // Every result is either a static floating [N] vector or a full rank-2
    // floating [M,N] row output.  Rank-1 is represented as one logical row for
    // launch geometry only; its public destination-passing ABI remains [N].
    // f16 and bf16 are storage types; scalar computation is promoted to f32.
    // Anchor (rank,M,N,storage) from the first; all results must agree.
    int64_t anchorRank = -1, anchorM = -1, anchorN = -1;
    for (Type rt : fty.getResults()) {
      auto t = dyn_cast<RankedTensorType>(rt);
      if (!t || (t.getRank() != 1 && t.getRank() != 2) ||
          !t.hasStaticShape())
        return failure();
      Type elementType = t.getElementType();
      if (!isa<Float16Type, BFloat16Type, Float32Type>(elementType))
        return failure();
      if (anchorN < 0) {
        anchorRank = t.getRank();
        anchorM = anchorRank == 1 ? 1 : t.getDimSize(0);
        anchorN = t.getDimSize(anchorRank - 1);
        storageType = elementType;
      } else if (t.getRank() != anchorRank ||
                 (anchorRank == 2 && t.getDimSize(0) != anchorM) ||
                 t.getDimSize(anchorRank - 1) != anchorN)
        return failure();
      else if (elementType != storageType)
        return failure();
    }
    M = anchorM;
    N = anchorN;
    oneDimensionalResults = anchorRank == 1;
    computeType = Float32Type::get(fn.getContext());

    // Single basic block, terminated by a return of the output values.
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != nRes)
      return failure();
    outputVals.assign(ret.getOperands().begin(), ret.getOperands().end());

    // Per-result strides (default contiguous). `intent.result_strides` is a flat
    // [row0,col0, row1,col1, ...] of length 2*nRes — rope's outputs are strided
    // half-views of a fresh [M,hd] buffer (rowStride hd = 2N).
    resStrides.assign(nRes, Strides{N, 1});
    if (auto a = fn->getAttrOfType<DenseI64ArrayAttr>("intent.result_strides")) {
      // The registered relation currently defines row/column strides only for
      // rank-2 views.  Fail closed instead of inventing rank-1 semantics.
      if (oneDimensionalResults)
        return failure();
      if ((unsigned)a.size() != 2 * nRes)
        return failure();
      for (unsigned j = 0; j < nRes; ++j)
        resStrides[j] = {a[2 * j], a[2 * j + 1]};
    }

    // In-place aliasing: `intent.inplace = [a0, a1, ...]` (length nRes) maps each
    // result to the input arg whose buffer it overwrites (-1 = a fresh
    // destination memref). rope/llama4/qwen2vl rotate q/k IN PLACE: y_lo aliases
    // q_lo, y_hi aliases q_hi. The read-once register tile loads every input
    // before any store, so writing back through an input view is safe.
    inplaceArg.assign(nRes, -1);
    if (auto a = fn->getAttrOfType<DenseI64ArrayAttr>("intent.inplace")) {
      // Rank-2 Row aliases are loaded into the complete register tile before
      // any output store.  Rank-1 inputs are Col operands and are loaded on
      // demand, so accepting an alias could make one result overwrite a value
      // subsequently consumed by another result.  Keep this case conservative
      // until a rank-1 read-once tile is represented explicitly.
      if (oneDimensionalResults)
        return failure();
      if ((unsigned)a.size() != nRes)
        return failure();
      for (unsigned j = 0; j < nRes; ++j)
        inplaceArg[j] = a[j];
    }

    // Args: Row inputs use the result storage type. Col/Scalar broadcasts may
    // additionally be f32 (the common mixed-precision scale/bias contract).
    // A singleton rank-1 tensor is also a scalar broadcast: the public ABI
    // remains memref<1xT>, while the kernel reads element zero once.  Closed J
    // artifacts use this spelling for scalar-like learned parameters.
    // A 2-D arg whose first dim d0 < M but DIVIDES M is a "broadcast Row": one
    // row is shared by M/d0 consecutive blocks (rope's cos/sin: one per token,
    // shared by the token's n_head blocks). An `intent.strides` arg-attr makes
    // the Row a strided view of a larger buffer (half / even-odd partner views).
    for (unsigned i = 0; i < fty.getNumInputs(); ++i) {
      auto t = dyn_cast<RankedTensorType>(fty.getInput(i));
      if (!t || !t.hasStaticShape())
        return failure();
      ArrayRef<int64_t> s = t.getShape();
      Strides st{N, 1};
      if (auto a = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "intent.strides")) {
        if (a.size() != 2)
          return failure();
        st = {a[0], a[1]};
      }
      int64_t div = 1;
      Role role;
      bool floating =
          isa<Float16Type, BFloat16Type, Float32Type>(t.getElementType());
      if (body.getArgument(i).use_empty() &&
          isa<FloatType, IntegerType>(t.getElementType())) {
        // Constexpr control specialization can make a branch-only ABI input
        // dead (for example an optional label vector). Preserve the public
        // memref ABI, but do not force an unused tensor into a Row/Col/Scalar
        // execution role or synthesize a device load.
        role = Role::Unused;
      } else if (s.size() == 2 && s[1] == N && floating) {
        if (s[0] == M)
          div = 1;
        else if (s[0] > 0 && M % s[0] == 0)
          div = M / s[0];
        else
          return failure();
        role = Role::Row;
      } else if (s.size() == 1 && s[0] == N && floating) {
        role = Role::Col;
      } else if (s.size() == 1 && s[0] == M &&
                 isa<FloatType, IntegerType>(t.getElementType())) {
        // A rank-1 value indexed by the logical row is a typed row broadcast.
        // The integer form is what the row-mask J uses; floating [M] values are
        // also admitted when the surrounding pointwise graph makes that role
        // unambiguous.
        role = Role::PerRow;
      } else if ((s.empty() || (s.size() == 1 && s[0] == 1)) &&
                 isa<FloatType, IntegerType>(t.getElementType())) {
        role = Role::Scalar;
      } else {
        return failure();
      }
      if (role == Role::Row && t.getElementType() != storageType)
        return failure();
      argRoles.push_back(role);
      argElementTypes.push_back(t.getElementType());
      argRowDiv.push_back(div);
      argStrides.push_back(st);
    }

    // Body ops: tensor splat constants plus map_unary / map_binary / select /
    // iota. ANY reduce => not our pass (it belongs to
    // convert-intent-row-reduction-to-gpu). Paper extraction emits literal
    // broadcasts as rank-0 tensor constants, so rejecting arith.constant here
    // would force a semantically faithful primitive activation graph onto the
    // generic multi-kernel fallback.
    for (Operation &op : body.without_terminator()) {
      if (isa<ReduceOp>(&op))
        return failure();
      if (auto cst = dyn_cast<arith::ConstantOp>(&op)) {
        if (!isSupportedSplatConstant(cst))
          return failure();
        continue;
      }
      if (auto unary = dyn_cast<MapUnaryOp>(&op))
        containsTanh |= unary.getFn() == "tanh";
      if (auto contraction = dyn_cast<ContractionOp>(&op)) {
        if (!isPointwiseContraction(contraction))
          return failure();
        continue;
      }
      if (!isa<MapUnaryOp, MapBinaryOp, SelectOp, IotaOp, CastOp>(&op))
        return failure();
    }
    if (requestRowMaskZeroFill && !isRowMaskGraph(body, ret))
      return failure();
    if (requestTanhApproximation == "native-ptx" && !containsTanh)
      return failure();

    // A flattened schedule is admitted for fully contiguous Row values plus
    // ordinary Col/Scalar broadcasts.  The flat packet address is used for Row
    // loads/stores, while `flat_index % N` recovers the logical column for Col
    // operands.  Broadcast rows, strided views and indexed/select bodies remain
    // fail-closed. Each thread owns exactly one vec2/vec4/vec8 packet. Wider
    // packets are still lowered by the target pipeline, so this remains a
    // schedule coordinate rather than a Python-side code generator.
    if (requestFlattenContiguous) {
      if (requestGridStrideUnroll != 1)
        return failure();
      T = blockThreads ? (int64_t)blockThreads : 256;
      int64_t flatWidth = static_cast<int64_t>(requestFlatVectorWidth);
      bool eligible = T >= 32 && T % 32 == 0 &&
                      (flatWidth == 2 || flatWidth == 4 || flatWidth == 8);
      for (unsigned i = 0; i < argRoles.size(); ++i) {
        if (argRoles[i] == Role::Row)
          eligible &= argRowDiv[i] == 1 && argStrides[i].isDefault(N);
        else
          eligible &= argRoles[i] == Role::Col ||
                      argRoles[i] == Role::PerRow ||
                      argRoles[i] == Role::Scalar ||
                      argRoles[i] == Role::Unused;
      }
      for (Strides strides : resStrides)
        eligible &= strides.isDefault(N);
      for (Operation &op : body.without_terminator())
        eligible &= !isa<IotaOp, SelectOp>(&op);
      int64_t totalElements = M * N;
      int64_t defaultBlockElements = T * flatWidth;
      int64_t logicalBlockElements =
          requestLogicalTileElements == 0
              ? defaultBlockElements
              : static_cast<int64_t>(requestLogicalTileElements);
      eligible &= N % flatWidth == 0 &&
                  totalElements % defaultBlockElements == 0 &&
                  logicalBlockElements >= defaultBlockElements &&
                  logicalBlockElements % defaultBlockElements == 0;
      if (!eligible)
        return failure();
      flattenContiguous = true;
      vectorWidth = flatWidth;
      K = flatWidth;
      numVectorTiles = 1;
      gridBlocks =
          (totalElements + logicalBlockElements - 1) / logicalBlockElements;
      return success();
    }

    // Preserve an authored logical C tile as a logical block domain.  Thread
    // mapping remains an independent target coordinate.  The common
    // tile=4096,T=128 state therefore emits one 32-element register strip per
    // thread instead of pretending that the source tile means 4096 CUDA lanes.
    if (requestLogicalTileElements != 0) {
      if (requestGridStrideUnroll != 1)
        return failure();
      T = blockThreads ? static_cast<int64_t>(blockThreads) : 128;
      int64_t tile = static_cast<int64_t>(requestLogicalTileElements);
      if (T < 32 || T > 1024 || T % 32 != 0 || tile < T || tile % T != 0)
        return failure();
      K = tile / T;
      if (K <= 0 || K > 32)
        return failure();
      for (int64_t alias : inplaceArg)
        if (alias >= 0)
          return failure();
      gridStrideFlat = true;
      tailMasked = K > 1;
      vectorWidth = 1;
      numVectorTiles = 0;
      gridBlocks = (M * N + tile - 1) / tile;
      return success();
    }

    // Default geometry: T threads over the feature dim, K columns per thread.
    T = blockThreads ? (int64_t)blockThreads : autoThreads(N);
    if (T <= 0 || T > 1024)
      return failure();
    tailMasked = N % T != 0;
    K = (N + T - 1) / T;
    gridBlocks = M;

    // Keep the read-once register tile for the small rows it was designed for,
    // but never materialize O(N/T) copies of the map DAG in compiler IR.  A
    // flat grid-stride loop emits one scalar body and covers the complete M*N
    // domain with coalesced lanes.  In-place states remain on the historical
    // full-register path because that path's read-before-write proof is stronger;
    // reject an excessively large in-place tile instead of weakening it.
    constexpr int64_t maxUnrolledValuesPerThread = 32;
    if (K > maxUnrolledValuesPerThread) {
      for (int64_t alias : inplaceArg)
        if (alias >= 0)
          return failure();
      gridStrideFlat = true;
      K = requestGridStrideUnroll;
      tailMasked = K > 1;
      vectorWidth = 1;
      numVectorTiles = 0;
      gridBlocks = flatGridBlocks((M * N + K - 1) / K, T);
      return success();
    }

    // Vector gate: K a multiple of 4 (preferred) or 2, every Row arg + every
    // result view fully
    // CONTIGUOUS (default [N,1] strides — a strided half/even-odd view cannot be
    // a contiguous vec4 load, so rope/llama4/qwen2vl stay on the scalar path),
    // and no iota/select in the body (those index a per-column scalar, not a
    // vectorizable lane). Col/Scalar args stay scalar-loaded at col[k]/once —
    // the vector column layout keeps cols[k] pointing at the right scalar column.
    int64_t candidateVectorWidth = 1;
    if (!tailMasked && T >= 32) {
      if (requestRowVectorWidth != 0) {
        candidateVectorWidth = requestRowVectorWidth;
        if (candidateVectorWidth > 1 && K % candidateVectorWidth != 0)
          return failure();
      } else {
        candidateVectorWidth = K % 4 == 0 ? 4 : (K % 2 == 0 ? 2 : 1);
      }
    } else if (requestRowVectorWidth > 1) {
      return failure();
    }
    if (candidateVectorWidth > 1) {
      bool ok = true;
      for (unsigned i = 0; i < argRoles.size(); ++i)
        if (argRoles[i] == Role::Row && !argStrides[i].isDefault(N))
          ok = false;
      for (unsigned j = 0; j < resStrides.size(); ++j)
        if (!resStrides[j].isDefault(N))
          ok = false;
      for (Operation &op : body.without_terminator())
        if (isa<IotaOp, SelectOp>(&op))
          ok = false;
      if (ok) {
        vectorWidth = candidateVectorWidth;
        numVectorTiles = K / vectorWidth;
      } else if (requestRowVectorWidth > 1) {
        return failure();
      }
    }
    if (requestGridStrideUnroll != 1)
      return failure();
    if (requestRowVectorWidth == 1)
      vectorWidth = 1;
    return success();
  }

  // ---- emission ---------------------------------------------------------- //
  // The feature tile is held in REGISTERS, not a memref cache: K is static and
  // small, so we fully unroll over k and keep each Row input's K loaded values
  // as SSA (exactly the library's register tile).
  DenseMap<Value, Value> uniform;              // value -> hoisted per-row scalar
  DenseMap<Value, Value> argMemref;            // arg -> memref arg
  DenseMap<Value, SmallVector<Value>> rowVals; // Row arg -> K loaded SSA values
  DenseMap<Value, SmallVector<Value>> pointVals; // shared per-column DAG values
  SmallVector<Value> cols;                     // K column indices (tx + k*T)
  SmallVector<Value> logicalRows;              // row index for each emitted point

  Value toCompute(OpBuilder &b, Location loc, Value value) const {
    if (value.getType() == computeType || !isa<FloatType>(value.getType()))
      return value;
    if (isa<Float16Type, BFloat16Type>(value.getType()))
      return b.create<arith::ExtFOp>(loc, computeType, value);
    return value;
  }

  Value toStorage(OpBuilder &b, Location loc, Value value) const {
    if (value.getType() == storageType)
      return value;
    if (value.getType() == computeType &&
        isa<Float16Type, BFloat16Type>(storageType))
      return b.create<arith::TruncFOp>(loc, storageType, value);
    return value;
  }

  Type tensorElementType(Value value) const {
    return cast<RankedTensorType>(value.getType()).getElementType();
  }

  Value castFloatTo(OpBuilder &b, Location loc, Value value,
                    Type targetType) const {
    Type sourceType = value.getType();
    if (sourceType == targetType)
      return value;
    auto sourceFloat = dyn_cast<FloatType>(sourceType);
    auto targetFloat = dyn_cast<FloatType>(targetType);
    auto sourceInteger = dyn_cast<IntegerType>(sourceType);
    auto targetInteger = dyn_cast<IntegerType>(targetType);
    if (targetType.isInteger(1)) {
      if (sourceInteger) {
        Value integer = toSignlessInteger(b, loc, value);
        Value zero = b.create<arith::ConstantOp>(
            loc, b.getIntegerAttr(integer.getType(), 0));
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, integer,
                                      zero);
      }
      if (sourceFloat) {
        Value zero =
            b.create<arith::ConstantOp>(loc, b.getFloatAttr(sourceFloat, 0.0));
        return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, value,
                                      zero);
      }
    }
    if (sourceFloat && targetFloat)
      return sourceFloat.getWidth() < targetFloat.getWidth()
                 ? b.create<arith::ExtFOp>(loc, targetType, value).getResult()
                 : b.create<arith::TruncFOp>(loc, targetType, value).getResult();
    if (sourceFloat && targetInteger) {
      if (targetInteger.isUnsigned()) {
        Type signlessTarget = IntegerType::get(
            targetType.getContext(), targetInteger.getWidth());
        Value converted = b.create<arith::FPToUIOp>(loc, signlessTarget, value);
        return fromSignlessInteger(b, loc, converted, targetType);
      }
      return b.create<arith::FPToSIOp>(loc, targetType, value);
    }
    if (sourceInteger && targetFloat)
      return (sourceInteger.getWidth() == 1 || sourceInteger.isUnsigned())
                 ? b.create<arith::UIToFPOp>(
                       loc, targetType, toSignlessInteger(b, loc, value))
                       .getResult()
                 : b.create<arith::SIToFPOp>(loc, targetType, value).getResult();
    if (sourceInteger && targetInteger) {
      Value integer = toSignlessInteger(b, loc, value);
      Type signlessTarget = IntegerType::get(
          targetType.getContext(), targetInteger.getWidth());
      if (sourceInteger.getWidth() < targetInteger.getWidth()) {
        Value widened =
            (sourceInteger.getWidth() == 1 || sourceInteger.isUnsigned())
                ? b.create<arith::ExtUIOp>(loc, signlessTarget, integer)
                      .getResult()
                : b.create<arith::ExtSIOp>(loc, signlessTarget, integer)
                      .getResult();
        return fromSignlessInteger(b, loc, widened, targetType);
      }
      Value narrowed = b.create<arith::TruncIOp>(loc, signlessTarget, integer);
      return fromSignlessInteger(b, loc, narrowed, targetType);
    }
    return nullptr;
  }

  Value emitMappedUnary(OpBuilder &b, Location loc, MapUnaryOp op,
                        Value input) const {
    Type resultType = tensorElementType(op.getResult());
    if (isa<Float16Type, BFloat16Type>(input.getType()) ||
        isa<Float16Type, BFloat16Type>(resultType))
      input = toCompute(b, loc, input);
    Value result;
    if (op.getFn() == "tanh" &&
        requestTanhApproximation == "native-ptx") {
      // Keep the semantic math op intact until the outlined NVIDIA device
      // module.  emitScalarUnary's default tanh is already an ex2/reciprocal
      // expansion, so tagging its final arith.subf cannot possibly recover a
      // native tanh instruction later in the target pipeline.
      auto tanh = b.create<math::TanhOp>(loc, input);
      tanh->setAttr(
          "intent.backend.tanh_approximation",
          StringAttr::get(b.getContext(), "native-ptx"));
      result = tanh;
    } else {
      result = emitScalarUnary(b, loc, op.getFn(), input);
    }
    return result ? castFloatTo(b, loc, result, resultType) : Value();
  }

  Value emitMappedBinary(OpBuilder &b, Location loc, MapBinaryOp op,
                         Value lhs, Value rhs) const {
    if (isCmpFn(op.getFn())) {
      if (isa<IntegerType>(lhs.getType()))
        return emitScalarCmpI(b, loc, op.getFn(), lhs, rhs);
      if (lhs.getType() != computeType || rhs.getType() != computeType) {
        lhs = toCompute(b, loc, lhs);
        rhs = toCompute(b, loc, rhs);
      }
      return emitScalarCmp(b, loc, op.getFn(), lhs, rhs);
    }

    Type resultType = tensorElementType(op.getResult());
    if (isa<FloatType>(lhs.getType()) &&
        (lhs.getType() != computeType || rhs.getType() != computeType ||
         resultType != computeType)) {
      lhs = toCompute(b, loc, lhs);
      rhs = toCompute(b, loc, rhs);
    }
    Value result = emitScalarBinary(b, loc, op.getFn(), lhs, rhs);
    return result ? castFloatTo(b, loc, result, resultType) : Value();
  }

  /// Evaluate an original value for the thread's k-th column (fully unrolled).
  Value emitAt(OpBuilder &b, Location loc, Value v, unsigned k) {
    if (Value u = uniform.lookup(v))
      return u;
    auto cacheIt = pointVals.find(v);
    if (cacheIt != pointVals.end() && k < cacheIt->second.size() &&
        cacheIt->second[k])
      return cacheIt->second[k];
    auto remember = [&](Value result) -> Value {
      if (result) {
        auto &values = pointVals[v];
        if (values.size() < static_cast<size_t>(K))
          values.resize(static_cast<size_t>(K));
        values[k] = result;
      }
      return result;
    };
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      Role role = argRoles[barg.getArgNumber()];
      if (role == Role::Row)
        return rowVals[v][k];
      if (role == Role::Col)
        return b.create<memref::LoadOp>(loc, argMemref[v],
                                       ValueRange{cols[k]});
      if (role == Role::PerRow)
        return b.create<memref::LoadOp>(loc, argMemref[v],
                                       ValueRange{logicalRows[k]});
      if (role == Role::Unused)
        return nullptr;
      return b.create<memref::LoadOp>(loc, argMemref[v], ValueRange{});
    }
    Operation *def = v.getDefiningOp();
    if (auto cst = dyn_cast<arith::ConstantOp>(def))
      return remember(emitSplatScalarConstant(b, loc, cst));
    if (auto io = dyn_cast<IotaOp>(def)) {
      // iota at column col[k] is the column index itself, cast to the op's
      // integer width (no materialized vector).
      Type elt =
          cast<RankedTensorType>(io.getResult().getType()).getElementType();
      return remember(b.create<arith::IndexCastOp>(loc, elt, cols[k]));
    }
    if (auto castOp = dyn_cast<CastOp>(def)) {
      Type target = tensorElementType(castOp.getResult());
      return remember(castFloatTo(
          b, loc, emitAt(b, loc, castOp.getInput(), k), target));
    }
    if (auto mu = dyn_cast<MapUnaryOp>(def))
      return remember(
          emitMappedUnary(b, loc, mu, emitAt(b, loc, mu.getInput(), k)));
    if (auto mb = dyn_cast<MapBinaryOp>(def)) {
      Value lhs = emitAt(b, loc, mb.getLhs(), k);
      Value rhs = emitAt(b, loc, mb.getRhs(), k);
      return remember(emitMappedBinary(b, loc, mb, lhs, rhs));
    }
    if (auto contraction = dyn_cast<ContractionOp>(def)) {
      if (!isPointwiseContraction(contraction))
        return nullptr;
      Value lhs = emitAt(b, loc, contraction.getA(), k);
      Value rhs = emitAt(b, loc, contraction.getB(), k);
      Type resultType = tensorElementType(contraction.getResult());
      if (isa<FloatType>(lhs.getType()) &&
          (lhs.getType() != computeType || rhs.getType() != computeType ||
           resultType != computeType)) {
        lhs = toCompute(b, loc, lhs);
        rhs = toCompute(b, loc, rhs);
      }
      Value result = emitScalarBinary(b, loc, "mul", lhs, rhs);
      return remember(result ? castFloatTo(b, loc, result, resultType) : Value());
    }
    if (auto sel = dyn_cast<SelectOp>(def))
      return remember(b.create<arith::SelectOp>(
          loc, emitAt(b, loc, sel.getCond(), k),
          emitAt(b, loc, sel.getX(), k), emitAt(b, loc, sel.getY(), k)));
    return nullptr;
  }

  /// Is an original value uniform (per-row / rank-0)? Row/Col args are not.
  bool isUniform(Value v) {
    if (uniform.count(v))
      return true;
    if (auto barg = dyn_cast<BlockArgument>(v))
      return argRoles[barg.getArgNumber()] == Role::Scalar;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>())
      return isSupportedSplatConstant(cst);
    return false;
  }

  /// Hoist every map op whose operands are all uniform into `uniform` (once).
  void hoistUniforms(OpBuilder &b, Location loc) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op : fn.getBody().front().without_terminator()) {
        Value res = op.getNumResults() ? op.getResult(0) : Value();
        if (!res || uniform.count(res))
          continue;
        if (auto cst = dyn_cast<arith::ConstantOp>(&op)) {
          uniform[res] = emitSplatScalarConstant(b, loc, cst);
          changed = true;
        } else if (auto castOp = dyn_cast<CastOp>(&op)) {
          if (isUniform(castOp.getInput())) {
            uniform[res] = castFloatTo(
                b, loc, emitAt(b, loc, castOp.getInput(), 0),
                tensorElementType(castOp.getResult()));
            changed = true;
          }
        } else if (auto mu = dyn_cast<MapUnaryOp>(&op)) {
          if (isUniform(mu.getInput())) {
            uniform[res] = emitMappedUnary(
                b, loc, mu, emitAt(b, loc, mu.getInput(), 0));
            changed = true;
          }
        } else if (auto mb = dyn_cast<MapBinaryOp>(&op)) {
          // Comparison fns are never hoisted here: their condition depends on a
          // Row input (p > q), and comparison emission is handled separately.
          // A fully-uniform comparison/select (rare) is simply recomputed per column in
          // emitAt — still correct, just not hoisted.
          if (isCmpFn(mb.getFn()))
            continue;
          if (isUniform(mb.getLhs()) && isUniform(mb.getRhs())) {
            uniform[res] = emitMappedBinary(
                b, loc, mb, emitAt(b, loc, mb.getLhs(), 0),
                emitAt(b, loc, mb.getRhs(), 0));
            changed = true;
          }
        }
      }
    }
  }

  Type memrefOfArg(MLIRContext *ctx, unsigned i) {
    Role role = argRoles[i];
    if (role == Role::Row)
      return rowMemref(ctx, M / argRowDiv[i], argStrides[i]);
    if (role == Role::Col)
      return MemRefType::get({N}, argElementTypes[i]);
    if (role == Role::PerRow)
      return MemRefType::get({M}, argElementTypes[i]);
    auto original = cast<RankedTensorType>(fn.getFunctionType().getInput(i));
    if (role == Role::Unused)
      return MemRefType::get(original.getShape(), argElementTypes[i]);
    if (original.getRank() == 1)
      return MemRefType::get({1}, argElementTypes[i]);
    return MemRefType::get({}, argElementTypes[i]);
  }

  LogicalResult rewriteRowMaskZeroFill() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    Block &source = fn.getBody().front();
    auto castOp = dyn_cast<CastOp>(&*source.without_terminator().begin());
    auto product = dyn_cast<ContractionOp>(
        &*std::next(source.without_terminator().begin()));
    if (!castOp || !product)
      return failure();
    auto maskArg = cast<BlockArgument>(castOp.getInput());
    Value dataValue = product.getA() == castOp.getResult()
                          ? product.getB()
                          : product.getA();
    auto dataArg = cast<BlockArgument>(dataValue);

    SmallVector<Type> inputTypes;
    for (unsigned i = 0; i < argRoles.size(); ++i)
      inputTypes.push_back(flattenContiguous && argRoles[i] == Role::Row
                               ? Type(flatVecMemref(ctx))
                               : Type(memrefOfArg(ctx, i)));
    inputTypes.push_back(flattenContiguous
                             ? Type(flatVecMemref(ctx))
                             : Type(resultMemref(ctx, resStrides.front())));
    auto newType = FunctionType::get(ctx, inputTypes, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto lowered = b.create<func::FuncOp>(loc, name, newType);
    Block *entry = lowered.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value dataMem = entry->getArgument(dataArg.getArgNumber());
    Value maskMem = entry->getArgument(maskArg.getArgNumber());
    Value outMem = entry->getArguments().back();

    if (flattenContiguous) {
      Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridBlocks);
      Value cThreads = b.create<arith::ConstantIndexOp>(loc, T);
      Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
      auto launch =
          b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cThreads, c1, c1);
      b.setInsertionPointToStart(&launch.getBody().front());
      Value bx = launch.getBlockIds().x;
      Value tx = launch.getThreadIds().x;
      Value cBlockSpan = b.create<arith::ConstantIndexOp>(loc, T);
      Value first = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, bx, cBlockSpan), tx);
      int64_t packetCount = M * N / vectorWidth;
      Value upper = b.create<arith::ConstantIndexOp>(loc, packetCount);
      Value step = b.create<arith::ConstantIndexOp>(loc, gridBlocks * T);
      auto loop = b.create<scf::ForOp>(loc, first, upper, step);
      b.setInsertionPointToStart(loop.getBody());
      Value packet = loop.getInductionVar();
      Value cPacketsPerRow =
          b.create<arith::ConstantIndexOp>(loc, N / vectorWidth);
      Value row = b.create<arith::DivUIOp>(loc, packet, cPacketsPerRow);
      Value mask = b.create<memref::LoadOp>(loc, maskMem, ValueRange{row});
      auto branch = b.create<scf::IfOp>(loc, mask, /*withElseRegion=*/true);
      OpBuilder thenBuilder = branch.getThenBodyBuilder();
      Value input = thenBuilder.create<memref::LoadOp>(
          loc, dataMem, ValueRange{packet});
      thenBuilder.create<memref::StoreOp>(loc, input, outMem,
                                          ValueRange{packet});
      OpBuilder elseBuilder = branch.getElseBodyBuilder();
      Value zeroScalar = elseBuilder.create<arith::ConstantOp>(
          loc, elseBuilder.getF32FloatAttr(0.0));
      Value zero = elseBuilder.create<vector::SplatOp>(
          loc, VectorType::get({vectorWidth}, storageType), zeroScalar);
      elseBuilder.create<memref::StoreOp>(loc, zero, outMem,
                                          ValueRange{packet});
      b.setInsertionPointAfter(loop);
      b.create<gpu::TerminatorOp>(loc);
      b.setInsertionPointAfter(launch);
      b.create<func::ReturnOp>(loc);
      attachElementwiseStateReceipt(
          lowered, T, gridBlocks, /*gridStrideFlat=*/true,
          /*gridStrideUnroll=*/1,
          /*logicalTileElements=*/requestLogicalTileElements,
          /*vectorWidth=*/vectorWidth, "rowmask_zero_fill",
          /*oneDimensionalResults=*/false, /*hostValueCaptureCount=*/0);
      lowered->setAttr(
          "intent.backend.transformation_trace",
          ArrayAttr::get(
              ctx,
              {StringAttr::get(ctx, "cuda.elementwise.row.vectorized.v1"),
               StringAttr::get(ctx,
                               "cuda.elementwise.rowmask.zero-fill.v1"),
               StringAttr::get(ctx,
                               "cuda.elementwise.rowmask.flat-grid.v1")}));
      fn.erase();
      return success();
    }

    Value cM = b.create<arith::ConstantIndexOp>(loc, M);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cM, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value mask = b.create<memref::LoadOp>(loc, maskMem, ValueRange{bx});
    auto branch = b.create<scf::IfOp>(loc, mask, /*withElseRegion=*/true);

    auto emitRow = [&](OpBuilder &rowBuilder, bool copyInput) {
      Value zero = copyInput
                       ? Value()
                       : rowBuilder.create<arith::ConstantOp>(
                             loc, rowBuilder.getF32FloatAttr(0.0));
      Value cN = rowBuilder.create<arith::ConstantIndexOp>(loc, N);
      for (int64_t k = 0; k < K; ++k) {
        Value col = tx;
        if (k != 0) {
          Value offset =
              rowBuilder.create<arith::ConstantIndexOp>(loc, k * T);
          col = rowBuilder.create<arith::AddIOp>(loc, tx, offset);
        }
        auto emitPoint = [&](OpBuilder &pointBuilder) {
          Value value = copyInput
                            ? pointBuilder.create<memref::LoadOp>(
                                  loc, dataMem, ValueRange{bx, col})
                                  .getResult()
                            : zero;
          pointBuilder.create<memref::StoreOp>(loc, value, outMem,
                                               ValueRange{bx, col});
        };
        if (N % T == 0 || (k + 1) * T <= N) {
          emitPoint(rowBuilder);
          continue;
        }
        Value active = rowBuilder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, col, cN);
        auto guarded = rowBuilder.create<scf::IfOp>(
            loc, active, /*withElseRegion=*/false);
        OpBuilder pointBuilder = guarded.getThenBodyBuilder();
        emitPoint(pointBuilder);
      }
    };
    OpBuilder thenBuilder = branch.getThenBodyBuilder();
    emitRow(thenBuilder, /*copyInput=*/true);
    OpBuilder elseBuilder = branch.getElseBodyBuilder();
    emitRow(elseBuilder, /*copyInput=*/false);

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    attachElementwiseStateReceipt(
        lowered, T, M, /*gridStrideFlat=*/false, /*gridStrideUnroll=*/1,
        /*logicalTileElements=*/0, /*vectorWidth=*/1, "rowmask_zero_fill",
        /*oneDimensionalResults=*/false, /*hostValueCaptureCount=*/0);
    lowered->setAttr(
        "intent.backend.transformation_trace",
        ArrayAttr::get(
            ctx,
            {StringAttr::get(ctx, "cuda.elementwise.row.vectorized.v1"),
             StringAttr::get(ctx,
                             "cuda.elementwise.rowmask.zero-fill.v1")}));
    fn.erase();
    return success();
  }

  LogicalResult rewrite() {
    if (requestRowMaskZeroFill)
      return rewriteRowMaskZeroFill();
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    unsigned shiftedExpCancellationCount = 0;
    if (auto count = fn->getAttrOfType<IntegerAttr>(
            "intent.cuda.shifted_exp_cancellation_count"))
      shiftedExpCancellationCount = static_cast<unsigned>(count.getInt());

    // New signature: original args (as memrefs) ++ one shape-preserving output
    // memref per result (destination-passing).
    SmallVector<Type> inTys;
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      if (flattenContiguous && argRoles[i] == Role::Row)
        inTys.push_back(flatVecMemref(ctx));
      else if (vectorWidth > 1 && argRoles[i] == Role::Row)
        inTys.push_back(rowVecMemref(ctx, M / argRowDiv[i]));
      else
        inTys.push_back(memrefOfArg(ctx, i));
    }
    for (unsigned j = 0; j < outputVals.size(); ++j)
      if (inplaceArg[j] < 0) // fresh destination; in-place reuses an input arg
        inTys.push_back(
            flattenContiguous
                ? flatVecMemref(ctx)
                : (vectorWidth > 1 ? resultVecMemref(ctx)
                                   : resultMemref(ctx, resStrides[j])));
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src"); // free up the symbol for the new func

    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Block &srcBlock = fn.getBody().front();
    for (unsigned i = 0; i < argRoles.size(); ++i)
      argMemref[srcBlock.getArgument(i)] = entry->getArgument(i);
    SmallVector<Value> outMems(outputVals.size());
    unsigned freshIdx = argRoles.size();
    for (unsigned j = 0; j < outputVals.size(); ++j)
      outMems[j] = inplaceArg[j] >= 0
                       ? entry->getArgument(inplaceArg[j])
                       : entry->getArgument(freshIdx++);

    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridBlocks);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    // A read-only, non-alias B capability may admit host-by-value capture for
    // uniform invocation inputs.  This is a real search arm: the false arm
    // below retains the historical device-side load in every launched lane.
    unsigned hostValueCaptureCount = 0;
    if (requestPromoteUniformScalars) {
      for (unsigned i = 0; i < argRoles.size(); ++i) {
        if (argRoles[i] != Role::Scalar)
          continue;
        Value arg = srcBlock.getArgument(i);
        auto original = cast<RankedTensorType>(
            fn.getFunctionType().getInput(i));
        if (original.getRank() == 1) {
          Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
          uniform[arg] = b.create<memref::LoadOp>(
              loc, argMemref[arg], ValueRange{zero});
        } else {
          uniform[arg] =
              b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{});
        }
        ++hostValueCaptureCount;
      }
    }

    auto launch = b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    // Column indices + the read-once register tile: each Row input's K values
    // loaded ONCE into SSA. Two layouts:
    //  - scalar path: col[k] = tx + k*T (interleaved, coalesced scalar loads).
    //  - vector path: the K columns split into K/vectorWidth tiles; the register
    //    value at k = vt*W + j has scalar col vt*(T*W) + tx*W + j, i.e. vec index
    //    (tx + vt*T) in memref<d0x(N/W)xvector<Wxstorage>>. Each tile is one
    //    coalesced vector load across the warp; its lanes are extracted
    //    into the SAME register tile the scalar path feeds emitAt. Col/Scalar
    //    args still index cols[k]/once, so they need no retype.
    cols.clear();
    cols.resize(K);
    logicalRows.assign(K, bx);
    SmallVector<Value> columnActive(K);
    scf::ForOp gridStrideLoop;
    Type vectorTy = VectorType::get({vectorWidth}, storageType);
    SmallVector<Value> vecIdx; // vector tile index tx+vt*T (reused by store)
    if (gridStrideFlat) {
      // One iteration covers U coalesced lane waves:
      //   flat[k] = block*(T*U) + tx + k*T
      // and advances by grid*(T*U).  U is a real schedule coordinate.  The
      // first lane-wave is protected by the scf.for upper bound; later waves
      // use explicit tail predicates when the flattened domain is ragged.
      Value cBlockSpan =
          b.create<arith::ConstantIndexOp>(loc, T * K);
      Value first = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, bx, cBlockSpan), tx);
      Value upper = b.create<arith::ConstantIndexOp>(loc, M * N);
      Value step =
          b.create<arith::ConstantIndexOp>(loc, gridBlocks * T * K);
      gridStrideLoop = b.create<scf::ForOp>(loc, first, upper, step);
      b.setInsertionPointToStart(gridStrideLoop.getBody());
      Value flatBase = gridStrideLoop.getInductionVar();
      Value cN = b.create<arith::ConstantIndexOp>(loc, N);
      Value cTotal = b.create<arith::ConstantIndexOp>(loc, M * N);
      Value cZero = b.create<arith::ConstantIndexOp>(loc, 0);
      for (int64_t k = 0; k < K; ++k) {
        Value flatIndex = flatBase;
        if (k != 0) {
          Value offset = b.create<arith::ConstantIndexOp>(loc, k * T);
          flatIndex = b.create<arith::AddIOp>(loc, flatBase, offset);
        }
        if (tailMasked) {
          columnActive[k] = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, flatIndex, cTotal);
          flatIndex = b.create<arith::SelectOp>(
              loc, columnActive[k], flatIndex, cZero);
        }
        cols[k] = b.create<arith::RemUIOp>(loc, flatIndex, cN);
        logicalRows[k] =
            M == 1
                ? cZero
                : b.create<arith::DivUIOp>(loc, flatIndex, cN).getResult();
      }
    } else if (vectorWidth > 1) {
      vecIdx.resize(numVectorTiles);
      Value cWidth = b.create<arith::ConstantIndexOp>(loc, vectorWidth);
      Value cLogicalColumns =
          flattenContiguous
              ? b.create<arith::ConstantIndexOp>(loc, N).getResult()
              : Value();
      Value flatBlockBase;
      if (flattenContiguous) {
        Value flatBlockThreads =
            b.create<arith::ConstantIndexOp>(loc, T);
        flatBlockBase =
            b.create<arith::MulIOp>(loc, bx, flatBlockThreads);
      }
      for (int64_t vt = 0; vt < numVectorTiles; ++vt) {
        Value off = b.create<arith::ConstantIndexOp>(loc, vt * T);
        Value threadTile = b.create<arith::AddIOp>(loc, tx, off);
        vecIdx[vt] = flattenContiguous
                         ? b.create<arith::AddIOp>(loc, flatBlockBase,
                                                  threadTile)
                         : threadTile;
        Value base = b.create<arith::MulIOp>(loc, vecIdx[vt], cWidth);
        for (int64_t j = 0; j < vectorWidth; ++j) {
          Value cj = b.create<arith::ConstantIndexOp>(loc, j);
          Value absoluteColumn = b.create<arith::AddIOp>(loc, base, cj);
          if (flattenContiguous) {
            cols[vt * vectorWidth + j] = b.create<arith::RemUIOp>(
                loc, absoluteColumn, cLogicalColumns);
            logicalRows[vt * vectorWidth + j] = b.create<arith::DivUIOp>(
                loc, absoluteColumn, cLogicalColumns);
          } else {
            cols[vt * vectorWidth + j] = absoluteColumn;
          }
        }
      }
    } else {
      for (int64_t k = 0; k < K; ++k) {
        Value off = b.create<arith::ConstantIndexOp>(loc, k * T);
        cols[k] = b.create<arith::AddIOp>(loc, tx, off);
      }
    }
    if (tailMasked && !gridStrideFlat) {
      Value cN = b.create<arith::ConstantIndexOp>(loc, N);
      Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
      for (int64_t k = 0; k < K; ++k) {
        columnActive[k] = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, cols[k], cN);
        // Use an in-bounds address for the inactive tail lane. Its value is
        // observationally dead because the result store is guarded below.
        // This lowers to one predicated select + ordinary coalesced load,
        // avoiding a divergent scf.if around every input load.
        cols[k] = b.create<arith::SelectOp>(loc, columnActive[k], cols[k], c0);
      }
    }
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      Value arg = srcBlock.getArgument(i);
      if (argRoles[i] == Role::Row) {
        // Broadcast Row (div>1): one source row shared by `div` consecutive
        // blocks -> source row index = bx / div (rope cos: token = bx/n_head).
        SmallVector<Value> vals(K);
        if (vectorWidth > 1) {
          Value ridx = logicalRows[0];
          if (argRowDiv[i] != 1) {
            Value d = b.create<arith::ConstantIndexOp>(loc, argRowDiv[i]);
            ridx = b.create<arith::DivUIOp>(loc, ridx, d);
          }
          // One coalesced vector load per tile; the flattened schedule uses a
          // rank-1 packet memref, while the row schedule keeps [row, packet].
          for (int64_t vt = 0; vt < numVectorTiles; ++vt) {
            Value v = flattenContiguous
                          ? b.create<memref::LoadOp>(
                                loc, argMemref[arg], ValueRange{vecIdx[vt]})
                          : b.create<memref::LoadOp>(
                                loc, argMemref[arg],
                                ValueRange{ridx, vecIdx[vt]});
            for (int64_t j = 0; j < vectorWidth; ++j)
              vals[vt * vectorWidth + j] =
                  b.create<vector::ExtractOp>(loc, v,
                                              ArrayRef<int64_t>{j});
          }
        } else {
          for (int64_t k = 0; k < K; ++k) {
            Value ridx = logicalRows[k];
            if (argRowDiv[i] != 1) {
              Value d = b.create<arith::ConstantIndexOp>(loc, argRowDiv[i]);
              ridx = b.create<arith::DivUIOp>(loc, ridx, d);
            }
            vals[k] = b.create<memref::LoadOp>(
                loc, argMemref[arg], ValueRange{ridx, cols[k]});
          }
        }
        rowVals[arg] = vals;
      } else if (argRoles[i] == Role::Scalar) {
        if (uniform.count(arg))
          continue;
        auto original = cast<RankedTensorType>(
            fn.getFunctionType().getInput(i));
        if (original.getRank() == 1) {
          Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
          uniform[arg] = b.create<memref::LoadOp>(
              loc, argMemref[arg], ValueRange{zero});
        } else {
          uniform[arg] =
              b.create<memref::LoadOp>(loc, argMemref[arg], ValueRange{});
        }
      }
    }
    hoistUniforms(b, loc);

    // Epilogue: each result written from the register tile (read-once). No
    // reduction, no all_reduce, no thread-0 guard — every thread writes its own
    // columns for every output.
    for (unsigned j = 0; j < outputVals.size(); ++j) {
      // Tail-masked outputs live in distinct scf.if regions; values defined in
      // one result's region cannot dominate the next.  Divisible row/flat paths
      // stay in one block and retain cross-output DAG reuse.
      if (tailMasked)
        pointVals.clear();
      if (vectorWidth > 1) {
        // Group the contiguous lanes of each tile into one vector store.
        for (int64_t vt = 0; vt < numVectorTiles; ++vt) {
          SmallVector<Value> lanes(vectorWidth);
          for (int64_t lj = 0; lj < vectorWidth; ++lj)
            lanes[lj] = toStorage(
                b, loc,
                emitAt(b, loc, outputVals[j], vt * vectorWidth + lj));
          Value vec = b.create<vector::FromElementsOp>(loc, vectorTy, lanes);
          if (flattenContiguous)
            b.create<memref::StoreOp>(loc, vec, outMems[j],
                                      ValueRange{vecIdx[vt]});
          else if (oneDimensionalResults)
            b.create<memref::StoreOp>(loc, vec, outMems[j],
                                      ValueRange{vecIdx[vt]});
          else
            b.create<memref::StoreOp>(loc, vec, outMems[j],
                                      ValueRange{logicalRows[0], vecIdx[vt]});
        }
      } else {
        for (int64_t k = 0; k < K; ++k) {
          if (!tailMasked) {
            Value out =
                toStorage(b, loc, emitAt(b, loc, outputVals[j], k));
            if (oneDimensionalResults)
              b.create<memref::StoreOp>(loc, out, outMems[j],
                                        ValueRange{cols[k]});
            else
              b.create<memref::StoreOp>(loc, out, outMems[j],
                                        ValueRange{logicalRows[k], cols[k]});
            continue;
          }
          auto guarded = b.create<scf::IfOp>(
              loc, columnActive[k], /*withElseRegion=*/false);
          OpBuilder thenBuilder = guarded.getThenBodyBuilder();
          Value out = toStorage(
              thenBuilder, loc,
              emitAt(thenBuilder, loc, outputVals[j], k));
          if (oneDimensionalResults)
            thenBuilder.create<memref::StoreOp>(
                loc, out, outMems[j], ValueRange{cols[k]});
          else
            thenBuilder.create<memref::StoreOp>(
                loc, out, outMems[j], ValueRange{logicalRows[k], cols[k]});
        }
      }
    }

    if (gridStrideLoop)
      b.setInsertionPointAfter(gridStrideLoop);
    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    attachElementwiseStateReceipt(
        gpuFn, T, gridBlocks, gridStrideFlat,
        gridStrideFlat ? static_cast<int64_t>(requestGridStrideUnroll) : 1,
        static_cast<int64_t>(requestLogicalTileElements),
        vectorWidth,
        shiftedExpCancellationCount != 0
            ? "reassociated_exp"
            : containsTanh
                  ? (requestTanhApproximation == "native-ptx" ? "native_tanh"
                                                               : "stable_tanh")
                  : "none",
        oneDimensionalResults, hostValueCaptureCount,
        shiftedExpCancellationCount);
    fn.erase();
    return success();
  }
};

/// A contract-projected paired last-axis rotation.  Each GPU lane owns one
/// (low, high) pair and loads both source elements before either in-place store,
/// so the lowering preserves the J-owned read-before-write relation without
/// splitting a full-rank tensor into frontend-specific view arguments.
struct PairedRotationLowering {
  func::FuncOp fn;
  unsigned requestedThreads;
  bool requestFlattenContiguous;
  unsigned requestGridStrideUnroll;
  unsigned requestLogicalTileElements;
  unsigned requestRowVectorWidth;
  bool requestPairedHeadReuse;
  unsigned requestPairedHeadGroupSize;
  bool requestPairedDirectPlane;
  std::string requestTanhApproximation;

  enum class Layout { SharedPosition, ThreePlane, Interleaved };
  Layout layout = Layout::SharedPosition;
  unsigned qArg = 0, kArg = 0, cosineArg = 0, sineArg = 0;
  int64_t B = 0, S = 0, HQ = 0, HK = 0, D = 0, half = 0;
  int64_t qPairs = 0, kPairs = 0, totalPairs = 0, headReusePairs = 0;
  int64_t T = 0, gridBlocks = 0;
  SmallVector<int64_t> sections;
  Type storageType;
  Type computeType;

  PairedRotationLowering(func::FuncOp fn, unsigned threads, bool flatten,
                         unsigned gridStrideUnroll,
                         unsigned logicalTileElements,
                         unsigned rowVectorWidth,
                         bool pairedHeadReuse,
                         unsigned pairedHeadGroupSize,
                         bool pairedDirectPlane,
                         StringRef tanhApproximation)
      : fn(fn), requestedThreads(threads),
        requestFlattenContiguous(flatten),
        requestGridStrideUnroll(gridStrideUnroll),
        requestLogicalTileElements(logicalTileElements),
        requestRowVectorWidth(rowVectorWidth),
        requestPairedHeadReuse(pairedHeadReuse),
        requestPairedHeadGroupSize(pairedHeadGroupSize),
        requestPairedDirectPlane(pairedDirectPlane),
        requestTanhApproximation(tanhApproximation.str()) {}

  LogicalResult match() {
    auto version = fn->getAttrOfType<StringAttr>("intent.paired_rotation_version");
    auto layoutAttr = fn->getAttrOfType<StringAttr>("intent.paired_rotation");
    auto inputs =
        fn->getAttrOfType<DenseI64ArrayAttr>("intent.paired_rotation_inputs");
    auto inplace = fn->getAttrOfType<DenseI64ArrayAttr>("intent.inplace");
    if (!version || version.getValue() !=
                        "intentir.paired-last-axis-rotation.v1" ||
        !layoutAttr || !inputs || inputs.size() != 4 || !inplace ||
        inplace.size() != 2)
      return failure();
    if (requestFlattenContiguous || requestGridStrideUnroll != 1 ||
        requestLogicalTileElements != 0 ||
        (requestRowVectorWidth != 0 && requestRowVectorWidth != 1 &&
         requestRowVectorWidth != 2) ||
        requestTanhApproximation != "stable-ex2")
      return failure();
    // A zero option means "let the lowering choose its canonical scalar
    // packetization".  The paired path has only one explicitly supported
    // non-scalar packet (two adjacent rotary pairs), so normalize the default
    // before using it in the geometry below.
    if (requestRowVectorWidth == 0)
      requestRowVectorWidth = 1;
    if (requestPairedHeadReuse && requestPairedHeadGroupSize != 2 &&
        requestPairedHeadGroupSize != 4 && requestPairedHeadGroupSize != 8 &&
        requestPairedHeadGroupSize != 16)
      return failure();

    auto fty = fn.getFunctionType();
    if (fty.getNumResults() != 2)
      return failure();
    SmallVector<unsigned> mapped;
    for (int64_t raw : inputs.asArrayRef()) {
      if (raw < 0 || raw >= fty.getNumInputs())
        return failure();
      unsigned index = static_cast<unsigned>(raw);
      if (llvm::is_contained(mapped, index))
        return failure();
      mapped.push_back(index);
    }
    qArg = mapped[0];
    kArg = mapped[1];
    cosineArg = mapped[2];
    sineArg = mapped[3];
    if (inplace[0] != static_cast<int64_t>(qArg) ||
        inplace[1] != static_cast<int64_t>(kArg))
      return failure();

    auto qType = dyn_cast<RankedTensorType>(fty.getInput(qArg));
    auto kType = dyn_cast<RankedTensorType>(fty.getInput(kArg));
    auto cosineType = dyn_cast<RankedTensorType>(fty.getInput(cosineArg));
    auto sineType = dyn_cast<RankedTensorType>(fty.getInput(sineArg));
    if (!qType || !kType || !cosineType || !sineType ||
        !qType.hasStaticShape() || !kType.hasStaticShape() ||
        !cosineType.hasStaticShape() || !sineType.hasStaticShape() ||
        qType.getRank() != 4 || kType.getRank() != 4 ||
        qType.getElementType() != kType.getElementType() ||
        !isa<Float16Type, BFloat16Type, Float32Type>(qType.getElementType()) ||
        fty.getResult(0) != qType || fty.getResult(1) != kType)
      return failure();
    B = qType.getDimSize(0);
    S = qType.getDimSize(1);
    HQ = qType.getDimSize(2);
    D = qType.getDimSize(3);
    HK = kType.getDimSize(2);
    if (B <= 0 || S <= 0 || HQ <= 0 || HK <= 0 || D <= 0 || D % 2 != 0 ||
        kType.getDimSize(0) != B || kType.getDimSize(1) != S ||
        kType.getDimSize(3) != D)
      return failure();
    half = D / 2;

    if (layoutAttr.getValue() == "shared_position") {
      layout = Layout::SharedPosition;
      if (cosineType.getRank() != 3 ||
          cosineType.getShape() != ArrayRef<int64_t>({1, S, D}) ||
          sineType.getShape() != cosineType.getShape() ||
          fn->hasAttr("intent.paired_rotation_sections"))
        return failure();
    } else if (layoutAttr.getValue() == "three_plane") {
      layout = Layout::ThreePlane;
      if (cosineType.getRank() != 4 ||
          cosineType.getShape() != ArrayRef<int64_t>({3, B, S, D}) ||
          sineType.getShape() != cosineType.getShape())
        return failure();
      auto sectionAttr = fn->getAttrOfType<DenseI64ArrayAttr>(
          "intent.paired_rotation_sections");
      if (!sectionAttr || sectionAttr.size() != 3)
        return failure();
      sections.assign(sectionAttr.asArrayRef().begin(),
                      sectionAttr.asArrayRef().end());
      if (llvm::any_of(sections, [](int64_t value) { return value <= 0; }) ||
          sections[0] + sections[1] + sections[2] != half)
        return failure();
    } else if (layoutAttr.getValue() == "interleaved") {
      layout = Layout::Interleaved;
      if (cosineType.getRank() != 3 ||
          cosineType.getShape() != ArrayRef<int64_t>({S, half, 2}) ||
          cosineType.getElementType() != qType.getElementType() ||
          sineType.getRank() != 0 ||
          !isa<Float32Type>(sineType.getElementType()) ||
          fn->hasAttr("intent.paired_rotation_sections"))
        return failure();
    } else {
      return failure();
    }
    if (layout != Layout::Interleaved &&
        (qType.getElementType() != cosineType.getElementType() ||
         qType.getElementType() != sineType.getElementType()))
      return failure();
    if (requestRowVectorWidth == 2 &&
        (layout != Layout::SharedPosition || half % 2 != 0 ||
         !isa<Float16Type, BFloat16Type>(qType.getElementType())))
      return failure();
    if (requestPairedDirectPlane && layout != Layout::ThreePlane)
      return failure();

    // The attribute is produced only after fixed code proves the exact A/B
    // pattern.  Still reject side-effecting or structurally foreign bodies so a
    // hand-authored marker cannot turn an unrelated function into a rotation.
    if (!fn.getBody().hasOneBlock())
      return failure();
    bool sawGather = false, sawSub = false, sawAdd = false;
    for (Operation &operation : fn.getBody().front().without_terminator()) {
      sawGather |= isa<GatherOp>(operation);
      if (auto binary = dyn_cast<MapBinaryOp>(operation)) {
        sawSub |= binary.getFn() == "sub";
        sawAdd |= binary.getFn() == "add";
      }
      if (!isa<arith::ConstantOp, GatherOp, MapUnaryOp, MapBinaryOp, SelectOp,
               IotaOp, CastOp, ContractionOp, tensor::ExpandShapeOp>(operation))
        return failure();
    }
    auto returned =
        dyn_cast<func::ReturnOp>(fn.getBody().front().getTerminator());
    if (!returned || returned.getNumOperands() != 2 || !sawGather || !sawSub ||
        !sawAdd)
      return failure();

    T = requestedThreads ? static_cast<int64_t>(requestedThreads) : 256;
    if (T < 32 || T > 1024 || T % 32 != 0)
      return failure();
    int64_t pairGroups = half / requestRowVectorWidth;
    qPairs = B * S * HQ * pairGroups;
    kPairs = B * S * HK * pairGroups;
    totalPairs = qPairs + kPairs;
    // Full-only grouped head reuse owns one logical rotary pair and a bounded
    // head group per lane.  Four heads amortize the angle/index work without
    // serializing all HQ/HK memory operations behind one lane (the all-head
    // form exposes too little memory-level parallelism on large RoPE tensors).
    int64_t headGroups =
        (std::max(HQ, HK) + requestPairedHeadGroupSize - 1) /
        requestPairedHeadGroupSize;
    headReusePairs = B * S * headGroups * pairGroups;
    if (qPairs <= 0 || kPairs <= 0 || totalPairs < qPairs ||
        headReusePairs <= 0)
      return failure();
    int64_t launchPairs = requestPairedHeadReuse ? headReusePairs : totalPairs;
    gridBlocks = (launchPairs + T - 1) / T;
    storageType = qType.getElementType();
    computeType = Float32Type::get(fn.getContext());
    return success();
  }

  Value toCompute(OpBuilder &builder, Location location, Value value) const {
    if (value.getType() == computeType)
      return value;
    if (isa<Float16Type, BFloat16Type>(value.getType()))
      return builder.create<arith::ExtFOp>(location, computeType, value);
    return value;
  }

  Value toStorage(OpBuilder &builder, Location location, Value value) const {
    if (value.getType() == storageType)
      return value;
    if (value.getType() == computeType &&
        isa<Float16Type, BFloat16Type>(storageType))
      return builder.create<arith::TruncFOp>(location, storageType, value);
    return value;
  }

  Value typedMul(OpBuilder &builder, Location location, Value lhs,
                 Value rhs) const {
    Value product = builder.create<arith::MulFOp>(
        location, toCompute(builder, location, lhs),
        toCompute(builder, location, rhs));
    return toStorage(builder, location, product);
  }

  Value typedAddSub(OpBuilder &builder, Location location, Value lhs, Value rhs,
                    bool subtract) const {
    lhs = toCompute(builder, location, lhs);
    rhs = toCompute(builder, location, rhs);
    Value result = subtract
                       ? Value(builder.create<arith::SubFOp>(location, lhs, rhs))
                       : Value(builder.create<arith::AddFOp>(location, lhs, rhs));
    return toStorage(builder, location, result);
  }

  Value storageZero(OpBuilder &builder, Location location) const {
    return builder.create<arith::ConstantOp>(
        location, builder.getFloatAttr(cast<FloatType>(storageType), 0.0));
  }

  std::pair<Value, Value> pairCoordinates(OpBuilder &builder,
                                          Location location,
                                          Value pair) const {
    if (layout != Layout::Interleaved) {
      Value high = builder.create<arith::AddIOp>(
          location, pair,
          builder.create<arith::ConstantIndexOp>(location, half));
      return {pair, high};
    }
    Value even = builder.create<arith::MulIOp>(
        location, pair,
        builder.create<arith::ConstantIndexOp>(location, 2));
    Value odd = builder.create<arith::AddIOp>(
        location, even, builder.create<arith::ConstantIndexOp>(location, 1));
    return {even, odd};
  }

  std::pair<Value, Value>
  loadAngles(OpBuilder &builder, Location location, Value cosine, Value sine,
             Value batch, Value sequence, Value pair) const {
    if (layout == Layout::Interleaved) {
      Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value c = builder.create<memref::LoadOp>(
          location, cosine, ValueRange{sequence, pair, zero});
      Value s = builder.create<memref::LoadOp>(
          location, cosine, ValueRange{sequence, pair, one});
      Value sign =
          builder.create<memref::LoadOp>(location, sine, ValueRange{});
      s = builder.create<arith::MulFOp>(location,
                                        toCompute(builder, location, s), sign);
      return {c, s};
    }
    if (layout == Layout::SharedPosition) {
      Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
      Value c = builder.create<memref::LoadOp>(
          location, cosine, ValueRange{zero, sequence, pair});
      Value s = builder.create<memref::LoadOp>(
          location, sine, ValueRange{zero, sequence, pair});
      return {c, s};
    }

    Value c0 = builder.create<arith::ConstantIndexOp>(location, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(location, 1);
    Value c2 = builder.create<arith::ConstantIndexOp>(location, 2);
    Value first = builder.create<arith::ConstantIndexOp>(location, sections[0]);
    Value second = builder.create<arith::ConstantIndexOp>(
        location, sections[0] + sections[1]);
    Value inFirst = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ult, pair, first);
    Value inSecond = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ult, pair, second);
    Value zero = storageZero(builder, location);

    if (requestPairedDirectPlane) {
      Value plane = builder.create<arith::SelectOp>(
          location, inFirst, c0,
          builder.create<arith::SelectOp>(location, inSecond, c1, c2));
      Value notFirst = builder.create<arith::XOrIOp>(
          location, inFirst,
          builder.create<arith::ConstantIntOp>(location, 1, 1));
      Value inMiddle =
          builder.create<arith::AndIOp>(location, notFirst, inSecond);
      Value notSecond = builder.create<arith::XOrIOp>(
          location, inSecond,
          builder.create<arith::ConstantIntOp>(location, 1, 1));
      auto selectedPlaneSum = [&](Value source) -> Value {
        Value selected = builder.create<memref::LoadOp>(
            location, source, ValueRange{plane, batch, sequence, pair});
        Value v0 =
            builder.create<arith::SelectOp>(location, inFirst, selected, zero);
        Value v1 =
            builder.create<arith::SelectOp>(location, inMiddle, selected, zero);
        Value v2 = builder.create<arith::SelectOp>(location, notSecond,
                                                   selected, zero);
        return typedAddSub(
            builder, location,
            typedAddSub(builder, location, v0, v1, false), v2, false);
      };
      return {selectedPlaneSum(cosine), selectedPlaneSum(sine)};
    }

    auto maskedPlaneSum = [&](Value source) -> Value {
      Value v0 = builder.create<memref::LoadOp>(
          location, source, ValueRange{c0, batch, sequence, pair});
      Value v1 = builder.create<memref::LoadOp>(
          location, source, ValueRange{c1, batch, sequence, pair});
      Value v2 = builder.create<memref::LoadOp>(
          location, source, ValueRange{c2, batch, sequence, pair});
      Value notFirst = builder.create<arith::XOrIOp>(
          location, inFirst,
          builder.create<arith::ConstantIntOp>(location, 1, 1));
      Value inMiddle = builder.create<arith::AndIOp>(location, notFirst, inSecond);
      Value notSecond = builder.create<arith::XOrIOp>(
          location, inSecond,
          builder.create<arith::ConstantIntOp>(location, 1, 1));
      v0 = builder.create<arith::SelectOp>(location, inFirst, v0, zero);
      v1 = builder.create<arith::SelectOp>(location, inMiddle, v1, zero);
      v2 = builder.create<arith::SelectOp>(location, notSecond, v2, zero);
      return typedAddSub(builder, location,
                         typedAddSub(builder, location, v0, v1, false), v2,
                         false);
    };
    return {maskedPlaneSum(cosine), maskedPlaneSum(sine)};
  }

  void emitTensorRotation(OpBuilder &builder, Location location, Value data,
                          Value cosine, Value sine, Value local,
                          int64_t heads) const {
    Value cPairGroups = builder.create<arith::ConstantIndexOp>(
        location, half / requestRowVectorWidth);
    Value cHeads = builder.create<arith::ConstantIndexOp>(location, heads);
    Value cSequence = builder.create<arith::ConstantIndexOp>(location, S);
    Value pairGroup =
        builder.create<arith::RemUIOp>(location, local, cPairGroups);
    Value pair = pairGroup;
    if (requestRowVectorWidth == 2)
      pair = builder.create<arith::MulIOp>(
          location, pairGroup,
          builder.create<arith::ConstantIndexOp>(location, 2));
    Value outer = builder.create<arith::DivUIOp>(location, local, cPairGroups);
    Value head = builder.create<arith::RemUIOp>(location, outer, cHeads);
    outer = builder.create<arith::DivUIOp>(location, outer, cHeads);
    Value sequence =
        builder.create<arith::RemUIOp>(location, outer, cSequence);
    Value batch = builder.create<arith::DivUIOp>(location, outer, cSequence);
    if (requestRowVectorWidth == 2) {
      SmallVector<Value, 2> cosineValues;
      SmallVector<Value, 2> sineValues;
      for (int64_t lane = 0; lane < 2; ++lane) {
        Value lanePair = pair;
        if (lane != 0)
          lanePair = builder.create<arith::AddIOp>(
              location, pair,
              builder.create<arith::ConstantIndexOp>(location, lane));
        auto angles = loadAngles(builder, location, cosine, sine, batch,
                                 sequence, lanePair);
        cosineValues.push_back(angles.first);
        sineValues.push_back(angles.second);
      }
      emitRotationVector2(builder, location, data, batch, sequence, head, pair,
                          cosineValues, sineValues);
      return;
    }
    auto [low, high] = pairCoordinates(builder, location, pair);

    // Keep the canonical direct lowering's original operation order exactly:
    // data pair first, then angle loads.  Full-only reuse must not manufacture
    // a gain by perturbing the Schedule-Only spine.
    Value lowValue = builder.create<memref::LoadOp>(
        location, data, ValueRange{batch, sequence, head, low});
    Value highValue = builder.create<memref::LoadOp>(
        location, data, ValueRange{batch, sequence, head, high});
    auto [cosineValue, sineValue] =
        loadAngles(builder, location, cosine, sine, batch, sequence, pair);
    emitRotationValues(builder, location, data, batch, sequence, head, low,
                       high, lowValue, highValue, cosineValue, sineValue);
  }

  void emitRotationAt(OpBuilder &builder, Location location, Value data,
                      Value batch, Value sequence, Value head, Value pair,
                      Value cosineValue, Value sineValue) const {
    auto [low, high] = pairCoordinates(builder, location, pair);

    // Both source values are in SSA before either store: this is the essential
    // in-place safety property opened by the paired B store partition.
    Value lowValue = builder.create<memref::LoadOp>(
        location, data, ValueRange{batch, sequence, head, low});
    Value highValue = builder.create<memref::LoadOp>(
        location, data, ValueRange{batch, sequence, head, high});
    emitRotationValues(builder, location, data, batch, sequence, head, low,
                       high, lowValue, highValue, cosineValue, sineValue);
  }

  std::pair<Value, Value>
  computeRotationValues(OpBuilder &builder, Location location, Value lowValue,
                        Value highValue, Value cosineValue,
                        Value sineValue) const {
    if (layout == Layout::Interleaved) {
      Value lowF32 = toCompute(builder, location, lowValue);
      Value highF32 = toCompute(builder, location, highValue);
      Value cosineF32 = toCompute(builder, location, cosineValue);
      Value sineF32 = toCompute(builder, location, sineValue);
      Value lowProduct =
          builder.create<arith::MulFOp>(location, lowF32, cosineF32);
      Value highSin =
          builder.create<arith::MulFOp>(location, highF32, sineF32);
      Value rotatedLow = builder.create<arith::SubFOp>(
          location, lowProduct, highSin);

      // Preserve the accepted J's asymmetric evaluation contract: the odd*cos
      // product is rounded in storage precision before it is extended and added
      // to the f32 even*sin product.
      Value highProductStorage = builder.create<arith::MulFOp>(
          location, highValue, cosineValue);
      Value lowSin =
          builder.create<arith::MulFOp>(location, lowF32, sineF32);
      Value rotatedHigh = builder.create<arith::AddFOp>(
          location, toCompute(builder, location, highProductStorage), lowSin);
      return {toStorage(builder, location, rotatedLow),
              toStorage(builder, location, rotatedHigh)};
    }
    Value lowProduct =
        typedMul(builder, location, lowValue, cosineValue);
    Value highSin = typedMul(builder, location, highValue, sineValue);
    Value rotatedLow =
        typedAddSub(builder, location, lowProduct, highSin, true);
    Value highProduct =
        typedMul(builder, location, highValue, cosineValue);
    Value lowSin = typedMul(builder, location, lowValue, sineValue);
    Value rotatedHigh =
        typedAddSub(builder, location, highProduct, lowSin, false);
    return {rotatedLow, rotatedHigh};
  }

  void emitRotationVector2(OpBuilder &builder, Location location, Value data,
                           Value batch, Value sequence, Value head,
                           Value pairBase, ArrayRef<Value> cosineValues,
                           ArrayRef<Value> sineValues) const {
    auto vectorType = VectorType::get({2}, storageType);
    Value cHalf = builder.create<arith::ConstantIndexOp>(location, half);
    Value highBase =
        builder.create<arith::AddIOp>(location, pairBase, cHalf);
    SmallVector<Value, 4> lowIndices{batch, sequence, head, pairBase};
    SmallVector<Value, 4> highIndices{batch, sequence, head, highBase};
    Value lowPacket = builder.create<vector::LoadOp>(
        location, vectorType, data, lowIndices);
    Value highPacket = builder.create<vector::LoadOp>(
        location, vectorType, data, highIndices);
    SmallVector<Value, 2> rotatedLow;
    SmallVector<Value, 2> rotatedHigh;
    for (int64_t lane = 0; lane < 2; ++lane) {
      Value lowValue = builder.create<vector::ExtractOp>(
          location, lowPacket, ArrayRef<int64_t>{lane});
      Value highValue = builder.create<vector::ExtractOp>(
          location, highPacket, ArrayRef<int64_t>{lane});
      auto rotated = computeRotationValues(
          builder, location, lowValue, highValue, cosineValues[lane],
          sineValues[lane]);
      rotatedLow.push_back(rotated.first);
      rotatedHigh.push_back(rotated.second);
    }
    Value lowResult =
        builder.create<vector::FromElementsOp>(location, vectorType, rotatedLow);
    Value highResult = builder.create<vector::FromElementsOp>(
        location, vectorType, rotatedHigh);
    builder.create<vector::StoreOp>(location, lowResult, data, lowIndices);
    builder.create<vector::StoreOp>(location, highResult, data, highIndices);
  }

  void emitRotationValues(OpBuilder &builder, Location location, Value data,
                          Value batch, Value sequence, Value head, Value pair,
                          Value high, Value lowValue, Value highValue,
                          Value cosineValue, Value sineValue) const {
    auto rotated = computeRotationValues(builder, location, lowValue, highValue,
                                         cosineValue, sineValue);
    builder.create<memref::StoreOp>(
        location, rotated.first, data,
        ValueRange{batch, sequence, head, pair});
    builder.create<memref::StoreOp>(
        location, rotated.second, data,
        ValueRange{batch, sequence, head, high});
  }

  LogicalResult rewrite() {
    MLIRContext *context = fn.getContext();
    Location location = fn.getLoc();
    OpBuilder builder(context);
    auto sourceType = fn.getFunctionType();
    SmallVector<Type> argumentTypes;
    for (Type type : sourceType.getInputs()) {
      auto tensor = cast<RankedTensorType>(type);
      argumentTypes.push_back(
          MemRefType::get(tensor.getShape(), tensor.getElementType()));
    }
    auto loweredType = FunctionType::get(context, argumentTypes, {});
    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    builder.setInsertionPoint(fn);
    auto lowered = builder.create<func::FuncOp>(location, name, loweredType);
    Block *entry = lowered.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    Value grid = builder.create<arith::ConstantIndexOp>(location, gridBlocks);
    Value threads = builder.create<arith::ConstantIndexOp>(location, T);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    auto launch = builder.create<gpu::LaunchOp>(location, grid, one, one, threads,
                                                one, one);
    builder.setInsertionPointToStart(&launch.getBody().front());
    Value block = launch.getBlockIds().x;
    Value thread = launch.getThreadIds().x;
    Value cThreads = builder.create<arith::ConstantIndexOp>(location, T);
    Value flat = builder.create<arith::AddIOp>(
        location, builder.create<arith::MulIOp>(location, block, cThreads),
        thread);
    int64_t launchPairs = requestPairedHeadReuse ? headReusePairs : totalPairs;
    Value total = builder.create<arith::ConstantIndexOp>(location, launchPairs);
    Value active = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ult, flat, total);
    auto guarded = builder.create<scf::IfOp>(location, active,
                                             /*withElseRegion=*/false);
    OpBuilder bodyBuilder = guarded.getThenBodyBuilder();
    if (requestPairedHeadReuse) {
      int64_t pairGroups = half / requestRowVectorWidth;
      Value cPairGroups =
          bodyBuilder.create<arith::ConstantIndexOp>(location, pairGroups);
      int64_t headGroups =
          (std::max(HQ, HK) + requestPairedHeadGroupSize - 1) /
          requestPairedHeadGroupSize;
      Value cHeadGroups =
          bodyBuilder.create<arith::ConstantIndexOp>(location, headGroups);
      Value cSequence = bodyBuilder.create<arith::ConstantIndexOp>(location, S);
      Value pairGroup =
          bodyBuilder.create<arith::RemUIOp>(location, flat, cPairGroups);
      Value pair = pairGroup;
      if (requestRowVectorWidth == 2)
        pair = bodyBuilder.create<arith::MulIOp>(
            location, pairGroup,
            bodyBuilder.create<arith::ConstantIndexOp>(location, 2));
      Value outer =
          bodyBuilder.create<arith::DivUIOp>(location, flat, cPairGroups);
      Value headGroup =
          bodyBuilder.create<arith::RemUIOp>(location, outer, cHeadGroups);
      outer = bodyBuilder.create<arith::DivUIOp>(location, outer, cHeadGroups);
      Value sequence =
          bodyBuilder.create<arith::RemUIOp>(location, outer, cSequence);
      Value batch =
          bodyBuilder.create<arith::DivUIOp>(location, outer, cSequence);
      SmallVector<Value, 2> cosineValues;
      SmallVector<Value, 2> sineValues;
      for (unsigned lane = 0; lane < requestRowVectorWidth; ++lane) {
        Value lanePair = pair;
        if (lane != 0)
          lanePair = bodyBuilder.create<arith::AddIOp>(
              location, pair,
              bodyBuilder.create<arith::ConstantIndexOp>(location, lane));
        auto angles = loadAngles(
            bodyBuilder, location, entry->getArgument(cosineArg),
            entry->getArgument(sineArg), batch, sequence, lanePair);
        cosineValues.push_back(angles.first);
        sineValues.push_back(angles.second);
      }

      // Keep the angle in SSA while expanding the small, statically bounded head
      // group directly in the compiler graph.  A dynamic scf.for hid the fixed
      // trip count from NVVM and left loop/index overhead in every rotary pair;
      // the J shapes and the selected group size are both static here, so the
      // direct expansion preserves exactly the same accesses while exposing the
      // independent head operations to instruction scheduling.
      auto emitHeadGroup = [&](Value data, int64_t heads) {
        Value cGroupSize = bodyBuilder.create<arith::ConstantIndexOp>(
            location, requestPairedHeadGroupSize);
        Value lower =
            bodyBuilder.create<arith::MulIOp>(location, headGroup, cGroupSize);
        Value headUpper =
            bodyBuilder.create<arith::ConstantIndexOp>(location, heads);
        auto emitHead = [&](OpBuilder &targetBuilder, Value head) {
          if (requestRowVectorWidth == 2)
            emitRotationVector2(targetBuilder, location, data, batch, sequence,
                                head, pair, cosineValues, sineValues);
          else
            emitRotationAt(targetBuilder, location, data, batch, sequence, head,
                           pair, cosineValues.front(), sineValues.front());
        };
        auto emitStaticHeads = [&](OpBuilder &targetBuilder) {
          for (unsigned offset = 0; offset < requestPairedHeadGroupSize;
               ++offset) {
            Value head = lower;
            if (offset != 0)
              head = targetBuilder.create<arith::AddIOp>(
                  location, lower,
                  targetBuilder.create<arith::ConstantIndexOp>(location,
                                                                offset));
            emitHead(targetBuilder, head);
          }
        };

        if (heads % requestPairedHeadGroupSize == 0) {
          if (heads == std::max(HQ, HK)) {
            emitStaticHeads(bodyBuilder);
            return;
          }
          Value activeGroup = bodyBuilder.create<arith::CmpIOp>(
              location, arith::CmpIPredicate::ult, lower, headUpper);
          auto guardedGroup = bodyBuilder.create<scf::IfOp>(
              location, activeGroup, /*withElseRegion=*/false);
          OpBuilder groupBuilder = guardedGroup.getThenBodyBuilder();
          emitStaticHeads(groupBuilder);
          return;
        }

        for (unsigned offset = 0; offset < requestPairedHeadGroupSize;
             ++offset) {
          Value head = lower;
          if (offset != 0)
            head = bodyBuilder.create<arith::AddIOp>(
                location, lower,
                bodyBuilder.create<arith::ConstantIndexOp>(location, offset));
          Value activeHead = bodyBuilder.create<arith::CmpIOp>(
              location, arith::CmpIPredicate::ult, head, headUpper);
          auto guardedHead = bodyBuilder.create<scf::IfOp>(
              location, activeHead, /*withElseRegion=*/false);
          OpBuilder headBuilder = guardedHead.getThenBodyBuilder();
          emitHead(headBuilder, head);
        }
      };
      emitHeadGroup(entry->getArgument(qArg), HQ);
      emitHeadGroup(entry->getArgument(kArg), HK);
    } else {
      Value qLimit =
          bodyBuilder.create<arith::ConstantIndexOp>(location, qPairs);
      Value isQ = bodyBuilder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::ult, flat, qLimit);
      auto choose = bodyBuilder.create<scf::IfOp>(location, isQ,
                                                  /*withElseRegion=*/true);
      OpBuilder qBuilder = choose.getThenBodyBuilder();
      emitTensorRotation(qBuilder, location, entry->getArgument(qArg),
                         entry->getArgument(cosineArg),
                         entry->getArgument(sineArg), flat, HQ);
      OpBuilder kBuilder = choose.getElseBodyBuilder();
      Value local =
          kBuilder.create<arith::SubIOp>(location, flat, qLimit);
      emitTensorRotation(kBuilder, location, entry->getArgument(kArg),
                         entry->getArgument(cosineArg),
                         entry->getArgument(sineArg), local, HK);
    }

    builder.setInsertionPointAfter(guarded);
    builder.create<gpu::TerminatorOp>(location);
    builder.setInsertionPointAfter(launch);
    builder.create<func::ReturnOp>(location);
    attachElementwiseStateReceipt(
        lowered, T, gridBlocks, /*gridStrideFlat=*/false,
        /*gridStrideUnroll=*/1, /*logicalTileElements=*/0,
        /*vectorWidth=*/requestRowVectorWidth, "none",
        /*oneDimensionalResults=*/false,
        /*hostValueCaptureCount=*/0,
        /*shiftedExpCancellationCount=*/0,
        /*pairedLastAxisRotation=*/true,
        /*pairedHeadAngleReuse=*/requestPairedHeadReuse,
        /*pairedDirectPlane=*/requestPairedDirectPlane,
        /*pairedHeadGroupSize=*/requestPairedHeadGroupSize);
    fn.erase();
    return success();
  }
};

struct ConvertIntentElementwiseToGPUPass
    : ::intent::impl::ConvertIntentElementwiseToGPUBase<
          ConvertIntentElementwiseToGPUPass> {
  using ConvertIntentElementwiseToGPUBase::ConvertIntentElementwiseToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    SmallVector<func::FuncOp> pairedTargets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      if (cancelShiftedExp) {
        if (rewriteShiftedExpProducts(fn) == 0)
          continue;
        ElementwiseLowering transformed(
            fn, blockThreads, flattenContiguous, flatVectorWidth,
            gridStrideUnroll, logicalTileElements, rowVectorWidth,
            promoteUniformScalars, rowMaskZeroFill, tanhApproximation);
        if (failed(transformed.match())) {
          fn.emitError("shifted-exp cancellation did not produce a complete "
                       "elementwise lowering state");
          return signalPassFailure();
        }
        targets.push_back(fn);
        continue;
      }
      ElementwiseLowering probe(fn, blockThreads, flattenContiguous,
                                flatVectorWidth, gridStrideUnroll,
                                logicalTileElements, rowVectorWidth,
                                promoteUniformScalars, rowMaskZeroFill,
                                tanhApproximation);
      if (succeeded(probe.match()))
        targets.push_back(fn);
      else {
        PairedRotationLowering paired(
            fn, blockThreads, flattenContiguous, gridStrideUnroll,
            logicalTileElements, rowVectorWidth, pairedHeadReuse,
            pairedHeadGroupSize, pairedDirectPlane, tanhApproximation);
        if (succeeded(paired.match()))
          pairedTargets.push_back(fn);
      }
    }
    for (func::FuncOp fn : targets) {
      ElementwiseLowering low(fn, blockThreads, flattenContiguous,
                              flatVectorWidth, gridStrideUnroll,
                              logicalTileElements, rowVectorWidth,
                              promoteUniformScalars, rowMaskZeroFill,
                              tanhApproximation);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("elementwise lowering failed after match");
        return signalPassFailure();
      }
    }
    for (func::FuncOp fn : pairedTargets) {
      PairedRotationLowering low(
          fn, blockThreads, flattenContiguous, gridStrideUnroll,
          logicalTileElements, rowVectorWidth, pairedHeadReuse,
          pairedHeadGroupSize, pairedDirectPlane, tanhApproximation);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("paired last-axis rotation lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
