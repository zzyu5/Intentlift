//===- ConvertIntentFlashMMAToGPU.cpp - register-resident flash fwd -------===//
//
// M8c: the REGISTER-RESIDENT nvgpu.mma.sync sibling of M8a
// (ConvertIntentLoopContractionToGPU, which round-trips O/S/P/m/l through
// dynamic shared). Same recognizer (the FlashAttention forward as one
// intent.loop scanning the key axis: carries m[Sq,1], l[Sq,1], O[Sq,D]; body =
// S=Q@Kj^T -> online softmax -> O+=P@Vj). The rewrite keeps the online-softmax
// state RESIDENT IN REGISTERS so only Q/K/V touch shared.
//
// Lane layout (m16n8k16 f16->f32, the M7f-proven recipe, gpu.lane_id):
//   warp = tid/32, lane = tid%32, groupID = lane/4, tig = lane%4.
//   S = Q@Kj^T : A=Q (ldmatrix qSh[warp*16+lane%16, k+(lane/16)*8] -> vec<4x2xf16>),
//                B=K natural [BN,D] (ldmatrix kSh[n*8+lane%8, k+(lane/8)*8] ->
//                vec<2x2xf16>), acc d[i][j] -> row=groupID+i*8, col=8*tile+tig*2+j.
//   Each lane owns 2 query rows (groupID, groupID+8). A row's BN key columns are
//   spread over (tile,j) within the lane + the 4 lanes of its quad (tig). Per-row
//   rowmax/rowsum = quad gpu.shuffle xor (delta 1 then 2, width 32). NO shared.
//   P = exp(scale*S - m_new) cast f16, in registers. The P A-operand fragment for
//   PV is assembled DIRECTLY from the S accumulators -- the m16n8 acc layout IS
//   the m16n8k16 A-operand layout (two consecutive n8 S-tiles -> one k16 A-subtile):
//     aFrag[0]=P(tile=2s,i=0), aFrag[1]=P(2s,i=1),
//     aFrag[2]=P(2s+1,i=0), aFrag[3]=P(2s+1,i=1)   (each vec<2xf16> = [i2=0,i2=1]).
//   O += P@Vj : A=P (assembled), B=V transposed [D,BN] (ldmatrix vSh[n*8+lane%8,
//   k+(lane/8)*8]). O acc carried across the key loop as scf.for iter_args, rescaled
//   by alpha (per-row scalar, lane-local) each tile. Epilogue O/l -> global.
//
// M8d: the K/V key-tile staging is a cp.async multistage software-pipeline
// (num-stages option, default 2; port of the M7c/M8b schedule). N rotating
// dynamic-shared buffers hold K and V NATURAL [BN,D] (16-byte cp.async.cg,
// nvgpu.device_async_copy); the loads of tile ki+N-1 overlap the mma.sync+
// softmax of tile ki (device_async_wait{N-1}+barrier). The online-softmax
// carries (O/m/l) stay register-resident -- only the staging is pipelined;
// the compute (S=Q@Kj, softmax, O+=P@Vj) is byte-identical to M8c. The PV B
// operand still needs the TRANSPOSED [D,BN] V (ldmatrix x2 spans K=16=key
// along the contiguous dim, which natural V's d-contiguous layout cannot
// provide, and a transposed store cannot be cp.async-vectorized), so V is
// cp.async'd natural then transposed SHARED->SHARED per tile (cheap; no global
// latency). N=2 buys ~1.2x at >=512 keys; N>=3 plateaus. rel UNCHANGED (1.66e-4).
//
// Q/K/V f16, accumulators f32. Static shapes, BM=BN=64, D%16==0, Sq%64==0,
// Sk%64==0. The original rank-2 intent.loop path remains intact (including its
// optional LSE/causal controls). The live loop-free rank-4 path is the exact
// non-causal B=H=1 QK/max/exp/sum/LSE/TMP/PV graph and keeps its Out/LSE/TMP
// destination ABI. Bare-ptr ABI; lower_memref_module_to_kernels(scf_to_cf=True).
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "FlashMMAFragments.h"

#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTFLASHMMATOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

// Tile geometry: BM/BN = the block's query/key tile, 4 warps each owning a
// 16-row query strip. mma.sync m16n8k16. VPAD skews the transposed-V shared rows
// to avoid the 32-way bank conflict on the transposed write.
static constexpr int64_t MMA_M = 16, MMA_N = 8, MMA_K = 16;
static constexpr int64_t VPAD = 8;

static int64_t align16(int64_t x) { return (x + 15) / 16 * 16; }

struct FlashMMALowering {
  func::FuncOp fn;
  int numStages; // cp.async pipeline depth (clamped to the K-tile count)
  bool emitLse;  // also emit LSE = m + log(l) per query row (R-FLASH-3)
  bool causal;   // causal (lower-triangular) score masking (R-FLASH-3)
  bool certifiedFullAxisOnly;
  bool directVMatrixLoad;
  int64_t directVRowPadding;
  bool materializeTmpRoundtrip;
  bool pairedBMatrixLoads;
  bool vectorizedQStaging;
  bool streamPFragments;
  int64_t scoreFragmentColumns;
  bool hopperWarpgroup;
  bool hopperHybridPV;
  bool normalizedSoftmaxCarry;
  bool compactQueryWarpMapping;
  bool hopperRegisterQK;
  bool hopperRegisterPV;
  bool hopperAsyncKV;
  int64_t BM;
  int64_t BN;
  bool flatRank4 = false; // live B=H=1 flat graph, with Out/LSE/TMP results
  bool certifiedFullAxis = false;
  bool tmpInplace = false;

  int64_t B = 1, H = 1, Sq = 0, Sk = 0, D = 0;
  unsigned qArgIdx = 0, kArgIdx = 0, vArgIdx = 0, scaleArgIdx = 0,
           tmpArgIdx = 0;
  LoopOp loop;
  Type f16, f32;

  FlashMMALowering(func::FuncOp fn, int numStages, bool emitLse, bool causal,
                   bool certifiedFullAxisOnly, bool directVMatrixLoad,
                   int64_t directVRowPadding,
                   bool materializeTmpRoundtrip, bool pairedBMatrixLoads,
                   bool vectorizedQStaging, bool streamPFragments,
                   int64_t scoreFragmentColumns, bool hopperWarpgroup,
                   bool hopperHybridPV, bool normalizedSoftmaxCarry,
                   bool compactQueryWarpMapping, bool hopperRegisterQK,
                   bool hopperRegisterPV, bool hopperAsyncKV, int64_t blockM,
                   int64_t blockN)
      : fn(fn), numStages(numStages), emitLse(emitLse), causal(causal),
        certifiedFullAxisOnly(certifiedFullAxisOnly),
        directVMatrixLoad(directVMatrixLoad),
        directVRowPadding(directVRowPadding),
        materializeTmpRoundtrip(materializeTmpRoundtrip),
        pairedBMatrixLoads(pairedBMatrixLoads),
        vectorizedQStaging(vectorizedQStaging),
        streamPFragments(streamPFragments),
        scoreFragmentColumns(scoreFragmentColumns),
        hopperWarpgroup(hopperWarpgroup), hopperHybridPV(hopperHybridPV),
        normalizedSoftmaxCarry(normalizedSoftmaxCarry),
        compactQueryWarpMapping(compactQueryWarpMapping),
        hopperRegisterQK(hopperRegisterQK),
        hopperRegisterPV(hopperRegisterPV),
        hopperAsyncKV(hopperAsyncKV),
        BM(blockM), BN(blockN) {}

  // Match the live, loop-free P4 graph structurally.  This deliberately starts
  // from the three returned values and verifies the complete dataflow instead
  // of consulting the function name.  The first supported rank-4 case is the
  // static B=H=1 layout emitted for FlashAttention forward:
  //
  //   QK(bmhd,bnhd->nbhm) -> max(axis 0) -> scaled exp -> sum(axis 0)
  //   -> LSE/TMP, and P(f32->f16) V(nbhm,bnhd->dbhm) -> transpose -> Out f16.
  LogicalResult matchFlatRank4() {
    auto fty = fn.getFunctionType();
    if (causal || fty.getNumInputs() != 4 || fty.getNumResults() != 3 ||
        !fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != 3)
      return failure();

    auto outCast = ret.getOperand(0).getDefiningOp<CastOp>();
    auto lse = ret.getOperand(1).getDefiningOp<MapBinaryOp>();
    auto tmp = ret.getOperand(2).getDefiningOp<MapUnaryOp>();
    if (!outCast || !lse || !tmp || outCast.getToDtype() != "f16" ||
        lse.getFn() != "add" || tmp.getFn() != "exp")
      return failure();

    auto transpose = outCast.getInput().getDefiningOp<TransposeOp>();
    if (!transpose || transpose.getPerm().size() != 4 ||
        cast<IntegerAttr>(transpose.getPerm()[0]).getInt() != 1 ||
        cast<IntegerAttr>(transpose.getPerm()[1]).getInt() != 3 ||
        cast<IntegerAttr>(transpose.getPerm()[2]).getInt() != 2 ||
        cast<IntegerAttr>(transpose.getPerm()[3]).getInt() != 0)
      return failure();
    auto weighted = transpose.getInput().getDefiningOp<MapBinaryOp>();
    if (!weighted || weighted.getFn() != "mul" ||
        weighted.getRhs() != tmp.getResult())
      return failure();
    auto pv = weighted.getLhs().getDefiningOp<ContractionOp>();
    if (!pv || pv.getSubscripts() != "nbhm,bnhd->dbhm")
      return failure();
    auto pF16 = pv.getA().getDefiningOp<CastOp>();
    if (!pF16 || pF16.getToDtype() != "f16")
      return failure();

    auto mMinusLse = tmp.getInput().getDefiningOp<MapBinaryOp>();
    if (!mMinusLse || mMinusLse.getFn() != "sub" ||
        mMinusLse.getRhs() != lse.getResult())
      return failure();
    Value mValue = mMinusLse.getLhs();
    if (lse.getLhs() != mValue)
      return failure();
    auto logDenom = lse.getRhs().getDefiningOp<MapUnaryOp>();
    if (!logDenom || logDenom.getFn() != "log")
      return failure();
    auto denom = logDenom.getInput().getDefiningOp<ReduceOp>();
    if (!denom || denom.getFn() != "sum" || denom.getAxes().size() != 1 ||
        cast<IntegerAttr>(denom.getAxes()[0]).getInt() != 0)
      return failure();
    auto p = denom.getInput().getDefiningOp<MapUnaryOp>();
    if (!p || p.getFn() != "exp" || pF16.getInput() != p.getResult())
      return failure();
    auto shifted = p.getInput().getDefiningOp<MapBinaryOp>();
    if (!shifted || shifted.getFn() != "sub" ||
        shifted.getRhs() != mValue)
      return failure();
    auto scaledQK = shifted.getLhs().getDefiningOp<MapBinaryOp>();
    auto m = mValue.getDefiningOp<MapBinaryOp>();
    if (!scaledQK || !m || scaledQK.getFn() != "mul" ||
        m.getFn() != "mul" || scaledQK.getRhs() != m.getRhs())
      return failure();
    auto rowMax = m.getLhs().getDefiningOp<ReduceOp>();
    if (!rowMax || rowMax.getFn() != "max" ||
        rowMax.getAxes().size() != 1 ||
        cast<IntegerAttr>(rowMax.getAxes()[0]).getInt() != 0)
      return failure();
    auto qk = scaledQK.getLhs().getDefiningOp<ContractionOp>();
    if (!qk || qk.getSubscripts() != "bmhd,bnhd->nbhm" ||
        rowMax.getInput() != qk.getResult())
      return failure();

    auto qArg = dyn_cast<BlockArgument>(qk.getA());
    auto kArg = dyn_cast<BlockArgument>(qk.getB());
    auto vArg = dyn_cast<BlockArgument>(pv.getB());
    auto scaleArg = dyn_cast<BlockArgument>(scaledQK.getRhs());
    if (!qArg || !kArg || !vArg || !scaleArg ||
        qArg.getOwner() != &body || kArg.getOwner() != &body ||
        vArg.getOwner() != &body || scaleArg.getOwner() != &body ||
        qArg == kArg || qArg == vArg || qArg == scaleArg || kArg == vArg ||
        kArg == scaleArg || vArg == scaleArg)
      return failure();

    auto qTy = dyn_cast<RankedTensorType>(qArg.getType());
    auto kTy = dyn_cast<RankedTensorType>(kArg.getType());
    auto vTy = dyn_cast<RankedTensorType>(vArg.getType());
    auto scaleTy = dyn_cast<RankedTensorType>(scaleArg.getType());
    if (!qTy || !kTy || !vTy || !scaleTy || !qTy.hasStaticShape() ||
        !kTy.hasStaticShape() || !vTy.hasStaticShape() ||
        !scaleTy.hasStaticShape() || qTy.getRank() != 4 ||
        kTy.getRank() != 4 || vTy.getRank() != 4 ||
        scaleTy.getRank() != 0 || !isa<Float16Type>(qTy.getElementType()) ||
        kTy.getElementType() != qTy.getElementType() ||
        vTy.getElementType() != qTy.getElementType() ||
        !isa<Float32Type>(scaleTy.getElementType()))
      return failure();
    if (qTy.getDimSize(0) != 1 || qTy.getDimSize(2) != 1 ||
        kTy.getDimSize(0) != 1 || kTy.getDimSize(2) != 1 ||
        vTy.getDimSize(0) != 1 || vTy.getDimSize(2) != 1)
      return failure();

    Sq = qTy.getDimSize(1);
    D = qTy.getDimSize(3);
    Sk = kTy.getDimSize(1);
    if (kTy.getDimSize(3) != D || vTy.getDimSize(1) != Sk ||
        vTy.getDimSize(3) != D || Sq % BM != 0 || Sk % BN != 0 ||
        D % MMA_K != 0)
      return failure();
    f16 = qTy.getElementType();
    f32 = scaleTy.getElementType();

    auto isTensor = [](Type ty, ArrayRef<int64_t> shape,
                       Type elementType) -> bool {
      auto ranked = dyn_cast<RankedTensorType>(ty);
      return ranked && ranked.hasStaticShape() && ranked.getShape() == shape &&
             ranked.getElementType() == elementType;
    };
    SmallVector<int64_t> qShape{1, Sq, 1, D};
    SmallVector<int64_t> scoreShape{Sk, 1, 1, Sq};
    SmallVector<int64_t> statShape{1, 1, Sq};
    SmallVector<int64_t> weightedShape{D, 1, 1, Sq};
    if (!isTensor(qk.getResult().getType(), scoreShape, f32) ||
        !isTensor(rowMax.getResult().getType(), statShape, f32) ||
        !isTensor(m.getResult().getType(), statShape, f32) ||
        !isTensor(scaledQK.getResult().getType(), scoreShape, f32) ||
        !isTensor(shifted.getResult().getType(), scoreShape, f32) ||
        !isTensor(p.getResult().getType(), scoreShape, f32) ||
        !isTensor(denom.getResult().getType(), statShape, f32) ||
        !isTensor(logDenom.getResult().getType(), statShape, f32) ||
        !isTensor(lse.getResult().getType(), statShape, f32) ||
        !isTensor(mMinusLse.getResult().getType(), statShape, f32) ||
        !isTensor(tmp.getResult().getType(), statShape, f32) ||
        !isTensor(pF16.getResult().getType(), scoreShape, f16) ||
        !isTensor(pv.getResult().getType(), weightedShape, f32) ||
        !isTensor(weighted.getResult().getType(), weightedShape, f32) ||
        !isTensor(transpose.getResult().getType(), qShape, f32) ||
        !isTensor(outCast.getResult().getType(), qShape, f16) ||
        !isTensor(fty.getResult(0), qShape, f16) ||
        !isTensor(fty.getResult(1), statShape, f32) ||
        !isTensor(fty.getResult(2), statShape, f32))
      return failure();

    // With the exact dataflow above, requiring every non-terminator operation
    // to be one of these nodes prevents silently discarding an unconsumed mask,
    // bias, tail predicate, or side effect.
    SmallVector<Operation *> expected{
        qk.getOperation(),          rowMax.getOperation(),
        m.getOperation(),           scaledQK.getOperation(),
        shifted.getOperation(),     p.getOperation(),
        denom.getOperation(),       logDenom.getOperation(),
        lse.getOperation(),         mMinusLse.getOperation(),
        tmp.getOperation(),         pF16.getOperation(),
        pv.getOperation(),          weighted.getOperation(),
        transpose.getOperation(),   outCast.getOperation()};
    unsigned numOps = 0;
    for (Operation &op : body.without_terminator()) {
      ++numOps;
      if (std::find(expected.begin(), expected.end(), &op) == expected.end())
        return failure();
    }
    if (numOps != expected.size())
      return failure();

    qArgIdx = qArg.getArgNumber();
    kArgIdx = kArg.getArgNumber();
    vArgIdx = vArg.getArgNumber();
    scaleArgIdx = scaleArg.getArgNumber();
    flatRank4 = true;
    return success();
  }

  /// Match the structured rank-4 scan emitted from the accepted Flash-forward
  /// J.  This is intentionally stricter than recognizing an attention-shaped
  /// pair of contractions: the complete LSE/m/O recurrence, scratch update and
  /// three observable returns must all be the exact typed dataflow below.
  LogicalResult matchCertifiedStructuredRank4() {
    auto fty = fn.getFunctionType();
    if (causal || emitLse || fty.getNumInputs() != 6 ||
        fty.getNumResults() != 3 || !fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret || ret.getNumOperands() != 3)
      return failure();

    auto hasPermutation = [](TransposeOp op,
                             ArrayRef<int64_t> expected) -> bool {
      if (!op || op.getPerm().size() != expected.size())
        return false;
      for (auto [attribute, value] : llvm::zip(op.getPerm(), expected)) {
        auto integer = dyn_cast<IntegerAttr>(attribute);
        if (!integer || integer.getInt() != value)
          return false;
      }
      return true;
    };
    auto isReduction = [](ReduceOp op, StringRef fn,
                          int64_t axis) -> bool {
      return op && op.getFn() == fn && op.getAxes().size() == 1 &&
             cast<IntegerAttr>(op.getAxes()[0]).getInt() == axis;
    };
    auto isFloatSplat = [](Value value, bool negativeInfinity) -> bool {
      auto constant = value.getDefiningOp<arith::ConstantOp>();
      auto dense = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                            : DenseElementsAttr();
      if (!dense || !dense.isSplat() ||
          !isa<FloatType>(dense.getElementType()))
        return false;
      APFloat scalar = dense.getSplatValue<APFloat>();
      return negativeInfinity
                 ? scalar.isInfinity() && scalar.isNegative()
                 : scalar.isZero();
    };
    auto isIntegerZeroSplat = [](Value value) -> bool {
      auto constant = value.getDefiningOp<arith::ConstantOp>();
      auto dense = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                            : DenseElementsAttr();
      return dense && dense.isSplat() &&
             isa<IntegerType>(dense.getElementType()) &&
             dense.getSplatValue<APInt>().isZero();
    };
    auto isOverwrite = [](ScatterOp op) {
      return op && !op->hasAttr("atomic") && !op->hasAttr("ordering") &&
             !op->hasAttr("memory_scope") && !op->hasAttr("cas_retry") &&
             op.getAxis() == 2 &&
             op.getMode() == "along_axis" && op.getReduce() == "none";
    };

    LoopOp theLoop;
    for (Operation &operation : body.without_terminator()) {
      if (auto candidate = dyn_cast<LoopOp>(&operation)) {
        if (theLoop)
          return failure();
        theLoop = candidate;
      }
    }
    if (!theLoop || theLoop.getScanInputs().size() != 2 ||
        theLoop.getInits().size() != 3 || theLoop.getNumResults() != 3 ||
        theLoop.getScanAxes().size() != 2 || !theLoop.getTileHint() ||
        static_cast<int64_t>(*theLoop.getTileHint()) != BN ||
        cast<IntegerAttr>(theLoop.getScanAxes()[0]).getInt() != 1 ||
        cast<IntegerAttr>(theLoop.getScanAxes()[1]).getInt() != 1 ||
        !theLoop.getBody().hasOneBlock())
      return failure();
    loop = theLoop;

    auto outCast = ret.getOperand(0).getDefiningOp<CastOp>();
    auto lseScatter = ret.getOperand(1).getDefiningOp<ScatterOp>();
    auto tmpScatter = ret.getOperand(2).getDefiningOp<ScatterOp>();
    if (!outCast || outCast.getToDtype() != "f16" ||
        !isOverwrite(lseScatter) || !isOverwrite(tmpScatter) ||
        lseScatter.getIndex() != tmpScatter.getIndex())
      return failure();

    auto normalize = outCast.getInput().getDefiningOp<ContractionOp>();
    auto inverse = tmpScatter.getSrc().getDefiningOp<MapUnaryOp>();
    auto mMinusLse = inverse ? inverse.getInput().getDefiningOp<MapBinaryOp>()
                             : MapBinaryOp();
    auto inverseTranspose = normalize
                                ? normalize.getB().getDefiningOp<TransposeOp>()
                                : TransposeOp();
    if (!normalize || normalize.getSubscripts() != "abcd,abc->abcd" ||
        normalize.getA() != loop.getResult(2) || !inverse ||
        inverse.getFn() != "exp" || !mMinusLse ||
        mMinusLse.getFn() != "sub" ||
        mMinusLse.getLhs() != loop.getResult(1) ||
        mMinusLse.getRhs() != loop.getResult(0) ||
        !hasPermutation(inverseTranspose, {0, 2, 1}) ||
        inverseTranspose.getInput() != inverse.getResult() ||
        lseScatter.getSrc() != loop.getResult(0))
      return failure();

    auto tmpIdentity = tmpScatter.getData().getDefiningOp<MapUnaryOp>();
    auto tmpArg = tmpIdentity
                      ? dyn_cast<BlockArgument>(tmpIdentity.getInput())
                      : BlockArgument();
    if (!tmpIdentity || tmpIdentity.getFn() != "identity" || !tmpArg ||
        tmpArg.getOwner() != &body)
      return failure();
    auto lseZero = lseScatter.getData().getDefiningOp<arith::ConstantOp>();
    if (!lseZero || !isFloatSplat(lseZero.getResult(), false))
      return failure();

    auto indexAdd = tmpScatter.getIndex().getDefiningOp<MapBinaryOp>();
    if (!indexAdd || indexAdd.getFn() != "add")
      return failure();
    IotaOp iota = indexAdd.getLhs().getDefiningOp<IotaOp>();
    Value indexZero = indexAdd.getRhs();
    if (!iota) {
      iota = indexAdd.getRhs().getDefiningOp<IotaOp>();
      indexZero = indexAdd.getLhs();
    }
    if (!iota || !isIntegerZeroSplat(indexZero))
      return failure();

    if (!isFloatSplat(loop.getInits()[0], true) ||
        !isFloatSplat(loop.getInits()[1], true) ||
        !isFloatSplat(loop.getInits()[2], false))
      return failure();

    Block &lbody = loop.getBody().front();
    if (lbody.getNumArguments() != 5)
      return failure();
    auto yield = dyn_cast<YieldOp>(lbody.getTerminator());
    if (!yield || yield.getNumOperands() != 3)
      return failure();
    Value kTile = lbody.getArgument(0);
    Value vTile = lbody.getArgument(1);
    Value lseOld = lbody.getArgument(2);
    Value mOld = lbody.getArgument(3);
    Value oOld = lbody.getArgument(4);

    auto lseNew = yield.getOperand(0).getDefiningOp<MapBinaryOp>();
    Value mNewValue = yield.getOperand(1);
    auto mNew = mNewValue.getDefiningOp<MapBinaryOp>();
    auto oNew = yield.getOperand(2).getDefiningOp<MapBinaryOp>();
    if (!lseNew || lseNew.getFn() != "add" || lseNew.getLhs() != mNewValue ||
        !mNew || mNew.getFn() != "max" || !oNew || oNew.getFn() != "add")
      return failure();

    auto logDenom = lseNew.getRhs().getDefiningOp<MapUnaryOp>();
    auto denom = logDenom ? logDenom.getInput().getDefiningOp<MapBinaryOp>()
                          : MapBinaryOp();
    auto oldNorm = denom ? denom.getLhs().getDefiningOp<MapUnaryOp>()
                         : MapUnaryOp();
    auto lseMinusM = oldNorm
                         ? oldNorm.getInput().getDefiningOp<MapBinaryOp>()
                         : MapBinaryOp();
    auto rowSum = denom ? denom.getRhs().getDefiningOp<ReduceOp>() : ReduceOp();
    if (!logDenom || logDenom.getFn() != "log" || !denom ||
        denom.getFn() != "add" || !oldNorm || oldNorm.getFn() != "exp" ||
        !lseMinusM || lseMinusM.getFn() != "sub" ||
        lseMinusM.getLhs() != lseOld || lseMinusM.getRhs() != mNewValue ||
        !isReduction(rowSum, "sum", 3))
      return failure();

    auto p = rowSum.getInput().getDefiningOp<MapUnaryOp>();
    auto shifted = p ? p.getInput().getDefiningOp<MapBinaryOp>() : MapBinaryOp();
    auto mExpand = shifted
                       ? shifted.getRhs().getDefiningOp<tensor::ExpandShapeOp>()
                       : tensor::ExpandShapeOp();
    auto scaledQK = shifted
                        ? shifted.getLhs().getDefiningOp<MapBinaryOp>()
                        : MapBinaryOp();
    auto qk = scaledQK
                  ? scaledQK.getLhs().getDefiningOp<ContractionOp>()
                  : ContractionOp();
    auto scaleArg = scaledQK
                        ? dyn_cast<BlockArgument>(scaledQK.getRhs())
                        : BlockArgument();
    if (!p || p.getFn() != "exp" || !shifted || shifted.getFn() != "sub" ||
        !mExpand || mExpand.getSrc() != mNewValue || !scaledQK ||
        scaledQK.getFn() != "mul" || !qk ||
        qk.getSubscripts() != "bqhd,bkhd->bhqk" || qk.getB() != kTile ||
        !scaleArg || scaleArg.getOwner() != &body)
      return failure();

    auto scaledRowMax = mNew.getLhs().getDefiningOp<MapBinaryOp>();
    if (!scaledRowMax || scaledRowMax.getFn() != "mul" ||
        scaledRowMax.getRhs() != scaleArg || mNew.getRhs() != lseOld)
      return failure();
    auto rowMax = scaledRowMax.getLhs().getDefiningOp<ReduceOp>();
    if (!isReduction(rowMax, "max", 3) || rowMax.getInput() != qk.getResult())
      return failure();

    auto oldOScaled = oNew.getLhs().getDefiningOp<ContractionOp>();
    auto pv = oNew.getRhs().getDefiningOp<ContractionOp>();
    if (!oldOScaled || oldOScaled.getSubscripts() != "abcd,abc->abcd" ||
        oldOScaled.getA() != oOld || !pv ||
        pv.getSubscripts() != "bhqk,bkhd->bqhd" || pv.getB() != vTile)
      return failure();
    auto alphaTranspose = oldOScaled.getB().getDefiningOp<TransposeOp>();
    auto alpha = alphaTranspose
                     ? alphaTranspose.getInput().getDefiningOp<MapUnaryOp>()
                     : MapUnaryOp();
    auto mMinusNew = alpha
                         ? alpha.getInput().getDefiningOp<MapBinaryOp>()
                         : MapBinaryOp();
    auto pCast = pv.getA().getDefiningOp<CastOp>();
    if (!hasPermutation(alphaTranspose, {0, 2, 1}) || !alpha ||
        alpha.getFn() != "exp" || !mMinusNew || mMinusNew.getFn() != "sub" ||
        mMinusNew.getLhs() != mOld || mMinusNew.getRhs() != mNewValue ||
        !pCast || pCast.getToDtype() != "f16" ||
        pCast.getInput() != p.getResult())
      return failure();

    ScatterOp scratchScatter;
    for (Operation &operation : lbody.without_terminator()) {
      if (auto candidate = dyn_cast<ScatterOp>(&operation)) {
        if (candidate->hasAttr("atomic") || candidate->hasAttr("ordering") ||
            candidate->hasAttr("memory_scope") ||
            candidate->hasAttr("cas_retry"))
          return failure();
        if (scratchScatter)
          return failure();
        scratchScatter = candidate;
      }
    }
    auto scratchIdentity = scratchScatter
                               ? scratchScatter.getData().getDefiningOp<MapUnaryOp>()
                               : MapUnaryOp();
    if (!isOverwrite(scratchScatter) ||
        scratchScatter.getIndex() != tmpScatter.getIndex() ||
        scratchScatter.getSrc() != alpha.getResult() ||
        !scratchScatter.getResult().use_empty() || !scratchIdentity ||
        scratchIdentity.getFn() != "identity" ||
        scratchIdentity.getInput() != tmpArg)
      return failure();

    auto qArg = dyn_cast<BlockArgument>(qk.getA());
    auto kArg = dyn_cast<BlockArgument>(loop.getScanInputs()[0]);
    auto vArg = dyn_cast<BlockArgument>(loop.getScanInputs()[1]);
    if (!qArg || !kArg || !vArg || qArg.getOwner() != &body ||
        kArg.getOwner() != &body || vArg.getOwner() != &body)
      return failure();
    auto qTy = dyn_cast<RankedTensorType>(qArg.getType());
    auto kTy = dyn_cast<RankedTensorType>(kArg.getType());
    auto vTy = dyn_cast<RankedTensorType>(vArg.getType());
    auto scaleTy = dyn_cast<RankedTensorType>(scaleArg.getType());
    auto tmpTy = dyn_cast<RankedTensorType>(tmpArg.getType());
    if (!qTy || !kTy || !vTy || !scaleTy || !tmpTy ||
        !qTy.hasStaticShape() || !kTy.hasStaticShape() ||
        !vTy.hasStaticShape() || !tmpTy.hasStaticShape() ||
        qTy.getRank() != 4 || kTy.getRank() != 4 || vTy.getRank() != 4 ||
        scaleTy.getRank() != 0 || tmpTy.getRank() != 3 ||
        !isa<Float16Type>(qTy.getElementType()) ||
        kTy.getElementType() != qTy.getElementType() ||
        vTy.getElementType() != qTy.getElementType() ||
        !isa<Float32Type>(scaleTy.getElementType()) ||
        !isa<Float32Type>(tmpTy.getElementType()))
      return failure();

    B = qTy.getDimSize(0);
    Sq = qTy.getDimSize(1);
    H = qTy.getDimSize(2);
    D = qTy.getDimSize(3);
    Sk = kTy.getDimSize(1);
    if (B <= 0 || H <= 0 || Sq <= 0 || Sk <= 0 ||
        kTy.getShape() != ArrayRef<int64_t>({B, Sk, H, D}) ||
        vTy.getShape() != ArrayRef<int64_t>({B, Sk, H, D}) ||
        tmpTy.getShape() != ArrayRef<int64_t>({B, H, Sq}) ||
        Sq % BM != 0 || Sk % BN != 0 || D % MMA_K != 0)
      return failure();
    f16 = qTy.getElementType();
    f32 = scaleTy.getElementType();

    auto isTensor = [](Type type, ArrayRef<int64_t> shape,
                       Type elementType) -> bool {
      auto tensor = dyn_cast<RankedTensorType>(type);
      return tensor && tensor.hasStaticShape() && tensor.getShape() == shape &&
             tensor.getElementType() == elementType;
    };
    SmallVector<int64_t> qShape{B, Sq, H, D};
    SmallVector<int64_t> scoreShape{B, H, Sq, BN};
    SmallVector<int64_t> statShape{B, H, Sq};
    if (!isTensor(lbody.getArgument(0).getType(), {B, BN, H, D}, f16) ||
        !isTensor(lbody.getArgument(1).getType(), {B, BN, H, D}, f16) ||
        !isTensor(qk.getResult().getType(), scoreShape, f32) ||
        !isTensor(p.getResult().getType(), scoreShape, f32) ||
        !isTensor(pCast.getResult().getType(), scoreShape, f16) ||
        !isTensor(mNew.getResult().getType(), statShape, f32) ||
        !isTensor(lseNew.getResult().getType(), statShape, f32) ||
        !isTensor(oNew.getResult().getType(), qShape, f32) ||
        !isTensor(normalize.getResult().getType(), qShape, f32) ||
        !isTensor(outCast.getResult().getType(), qShape, f16) ||
        !isTensor(fty.getResult(0), qShape, f16) ||
        !isTensor(fty.getResult(1), statShape, f32) ||
        !isTensor(fty.getResult(2), statShape, f32))
      return failure();

    qArgIdx = qArg.getArgNumber();
    kArgIdx = kArg.getArgNumber();
    vArgIdx = vArg.getArgNumber();
    scaleArgIdx = scaleArg.getArgNumber();
    tmpArgIdx = tmpArg.getArgNumber();
    auto inplace = fn->getAttrOfType<DenseI64ArrayAttr>("intent.inplace");
    if (!inplace || inplace.size() != 3 || inplace.asArrayRef()[0] != -1 ||
        inplace.asArrayRef()[1] != -1 ||
        inplace.asArrayRef()[2] != static_cast<int64_t>(tmpArgIdx))
      return failure();

    SmallVector<Operation *> expectedTop{
        loop.getInits()[0].getDefiningOp(), loop.getInits()[1].getDefiningOp(),
        loop.getInits()[2].getDefiningOp(), iota.getOperation(),
        indexZero.getDefiningOp(), indexAdd.getOperation(), loop.getOperation(),
        tmpIdentity.getOperation(), mMinusLse.getOperation(),
        inverse.getOperation(), tmpScatter.getOperation(), lseZero.getOperation(),
        lseScatter.getOperation(), inverseTranspose.getOperation(),
        normalize.getOperation(), outCast.getOperation()};
    unsigned topCount = 0;
    for (Operation &operation : body.without_terminator()) {
      ++topCount;
      if (std::find(expectedTop.begin(), expectedTop.end(), &operation) ==
          expectedTop.end())
        return failure();
    }
    if (topCount != expectedTop.size())
      return failure();

    SmallVector<Operation *> expectedLoop{
        scratchIdentity.getOperation(), qk.getOperation(),
        scaledQK.getOperation(), rowMax.getOperation(),
        scaledRowMax.getOperation(), mNew.getOperation(),
        mExpand.getOperation(), shifted.getOperation(), p.getOperation(),
        rowSum.getOperation(), mMinusNew.getOperation(), alpha.getOperation(),
        alphaTranspose.getOperation(), oldOScaled.getOperation(),
        pCast.getOperation(), pv.getOperation(), oNew.getOperation(),
        lseMinusM.getOperation(), oldNorm.getOperation(), denom.getOperation(),
        logDenom.getOperation(), lseNew.getOperation(),
        scratchScatter.getOperation()};
    unsigned loopCount = 0;
    for (Operation &operation : lbody.without_terminator()) {
      ++loopCount;
      if (std::find(expectedLoop.begin(), expectedLoop.end(), &operation) ==
          expectedLoop.end())
        return failure();
    }
    if (loopCount != expectedLoop.size())
      return failure();

    flatRank4 = true;
    certifiedFullAxis = true;
    tmpInplace = true;
    return success();
  }

  LogicalResult match() {
    if (BM <= 0 || BM % MMA_M != 0 || BM > 128 ||
        BN <= 0 || BN % MMA_K != 0)
      return failure();
    if (scoreFragmentColumns < 0 ||
        (scoreFragmentColumns != 0 &&
         (scoreFragmentColumns >= BN || scoreFragmentColumns % MMA_K != 0 ||
          BN % scoreFragmentColumns != 0 || !streamPFragments)))
      return failure();
    if (hopperWarpgroup &&
        (!certifiedFullAxisOnly || (BM != 64 && BM != 128) || BN != 128 ||
         numStages != 1 ||
         directVMatrixLoad || directVRowPadding != 0 ||
         materializeTmpRoundtrip || pairedBMatrixLoads ||
         vectorizedQStaging || streamPFragments ||
         scoreFragmentColumns != 0))
      return failure();
    if (hopperHybridPV && (!hopperWarpgroup || BM != 64))
      return failure();
    if (normalizedSoftmaxCarry &&
        (!certifiedFullAxisOnly || materializeTmpRoundtrip))
      return failure();
    if (compactQueryWarpMapping &&
        (!certifiedFullAxisOnly || hopperWarpgroup || BM != 128 ||
         materializeTmpRoundtrip))
      return failure();
    if (hopperRegisterQK &&
        (!hopperHybridPV || !hopperWarpgroup || BM != 64 || BN != 128))
      return failure();
    if (hopperRegisterPV && !hopperRegisterQK)
      return failure();
    if (hopperAsyncKV && !hopperWarpgroup)
      return failure();
    if (pairedBMatrixLoads && !directVMatrixLoad)
      return failure();
    if ((directVRowPadding != 0 && directVRowPadding != VPAD) ||
        (directVRowPadding != 0 && !directVMatrixLoad))
      return failure();
    if (certifiedFullAxisOnly)
      return matchCertifiedStructuredRank4();
    if (succeeded(matchFlatRank4()))
      return success();
    return matchLoopRank2();
  }

  // ---- recognition (a focused subset of the M8a matcher: rank-2, 1 result) -//
  LogicalResult matchLoopRank2() {
    auto fty = fn.getFunctionType();
    if (fty.getNumResults() != 1 || !fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();

    LoopOp theLoop;
    for (Operation &op : body.without_terminator()) {
      if (auto lp = dyn_cast<LoopOp>(&op)) {
        if (theLoop)
          return failure();
        theLoop = lp;
      } else if (!isa<arith::ConstantOp, MapBinaryOp, MapUnaryOp>(&op)) {
        return failure();
      }
    }
    if (!theLoop)
      return failure();
    loop = theLoop;

    if (loop.getScanInputs().size() != 2 || loop.getInits().size() != 3)
      return failure();
    auto kArg = dyn_cast<BlockArgument>(loop.getScanInputs()[0]);
    auto vArg = dyn_cast<BlockArgument>(loop.getScanInputs()[1]);
    if (!kArg || !vArg || kArg.getOwner() != &body || vArg.getOwner() != &body)
      return failure();
    kArgIdx = kArg.getArgNumber();
    vArgIdx = vArg.getArgNumber();

    Block &lbody = loop.getBody().front();
    ContractionOp qk, pv;
    int nReduce = 0;
    bool sawMax = false, sawSum = false;
    for (Operation &op : lbody.without_terminator()) {
      if (auto c = dyn_cast<ContractionOp>(&op)) {
        StringRef sub = c.getSubscripts();
        if (sub == "qd,kd->qk")
          qk = c;
        else if (sub == "qk,kd->qd")
          pv = c;
        else
          return failure();
      } else if (auto r = dyn_cast<ReduceOp>(&op)) {
        ++nReduce;
        if (r.getFn() == "max")
          sawMax = true;
        else if (r.getFn() == "sum")
          sawSum = true;
      } else if (!isa<MapBinaryOp, MapUnaryOp, SelectOp, arith::ConstantOp,
                      tensor::SplatOp>(&op)) {
        return failure();
      }
    }
    if (!qk || !pv || nReduce != 2 || !sawMax || !sawSum)
      return failure();

    auto qArg = dyn_cast<BlockArgument>(qk.getA());
    if (!qArg || qArg.getOwner() != &body)
      return failure();
    qArgIdx = qArg.getArgNumber();
    if (qk.getB() != lbody.getArgument(0) || pv.getB() != lbody.getArgument(1))
      return failure();

    auto qTy = dyn_cast<RankedTensorType>(qk.getA().getType());
    auto kTy = dyn_cast<RankedTensorType>(kArg.getType());
    auto vTy = dyn_cast<RankedTensorType>(vArg.getType());
    if (!qTy || !kTy || !vTy)
      return failure();
    if (!qTy.hasStaticShape() || !kTy.hasStaticShape() || !vTy.hasStaticShape())
      return failure();
    if (qTy.getRank() != 2 || kTy.getRank() != 2 || vTy.getRank() != 2)
      return failure();
    if (!isa<Float16Type>(qTy.getElementType()) ||
        !isa<Float16Type>(kTy.getElementType()) ||
        !isa<Float16Type>(vTy.getElementType()))
      return failure();
    f16 = qTy.getElementType();

    Sq = qTy.getDimSize(0);
    D = qTy.getDimSize(1);
    Sk = kTy.getDimSize(0);
    if (kTy.getDimSize(1) != D || vTy.getDimSize(0) != Sk ||
        vTy.getDimSize(1) != D)
      return failure();
    if (Sq % BM != 0 || Sk % BN != 0 || D % MMA_K != 0)
      return failure();

    auto oTy = dyn_cast<RankedTensorType>(fty.getResult(0));
    if (!oTy || !oTy.hasStaticShape() || oTy.getRank() != 2 ||
        !isa<Float32Type>(oTy.getElementType()))
      return failure();
    if (oTy.getDimSize(0) != Sq || oTy.getDimSize(1) != D)
      return failure();
    f32 = oTy.getElementType();

    bool haveScale = false;
    for (unsigned i = 0; i < fty.getNumInputs(); ++i) {
      if (isa<Float32Type>(fty.getInput(i))) {
        scaleArgIdx = i;
        haveScale = true;
      } else if (!isa<RankedTensorType>(fty.getInput(i))) {
        return failure();
      }
    }
    if (!haveScale)
      return failure();
    return success();
  }

  // Hopper structure: use the stock NVGPU warpgroup abstraction for both
  // contractions while retaining the authored scalar online-softmax update.
  // The first contraction is emitted as K*Q^T so natural K rows can be staged
  // cooperatively; its [key,query] result is consumed transposed by the scalar
  // softmax.  P*V then produces the ordinary [query,feature] update.
  LogicalResult rewriteWarpgroup() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    auto oldFty = fn.getFunctionType();
    SmallVector<Type> inTys;
    for (Type at : oldFty.getInputs()) {
      if (auto t = dyn_cast<RankedTensorType>(at))
        inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
      else
        inTys.push_back(at);
    }
    for (auto [index, resultTy] : llvm::enumerate(oldFty.getResults())) {
      if (tmpInplace && index == 2)
        continue;
      auto t = cast<RankedTensorType>(resultTy);
      inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
    }
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Value qMem = entry->getArgument(qArgIdx);
    Value kMem = entry->getArgument(kArgIdx);
    Value vMem = entry->getArgument(vArgIdx);
    Value scaleInput = entry->getArgument(scaleArgIdx);
    unsigned numInputs = oldFty.getNumInputs();
    Value oMem = entry->getArgument(numInputs);
    Value lseMem = entry->getArgument(numInputs + 1);
    Value tmpMem = entry->getArgument(tmpArgIdx);

    const int64_t WG_Q = BM;
    constexpr int64_t WG_HALF = 64;
    constexpr int64_t WG_N = 128;
    constexpr int64_t WG_THREADS = 128;
    const int64_t queryHalves = WG_Q / WG_HALF;
    const int64_t rowsPerThread = 2 * queryHalves;
    const int64_t warpRowSpan = rowsPerThread * 8;
    constexpr int64_t SCORE_STRIDE = WG_HALF + 1;
    int64_t queryTiles = Sq / WG_Q;
    int64_t keyTiles = Sk / WG_N;
    int64_t gridM = B * H * queryTiles;

    // Shared layout: Q^T, K, V, score^T, P, and PV delta.  Reserve one
    // additional 1024-byte swizzle period so the WGMMA operands can be aligned
    // at run time even when CUDA's dynamic-shared base is only 16-byte aligned.
    int64_t oQt = 0;
    int64_t oK = align16(oQt + D * WG_Q * 2);
    int64_t oV = align16(oK + WG_N * D * 2);
    int64_t vStride = hopperHybridPV && !hopperRegisterPV ? D + VPAD : D;
    int64_t oScore = align16(oV + WG_N * vStride * 2);
    int64_t scoreBlockElements = WG_HALF * SCORE_STRIDE;
    int64_t oP = ((oScore + queryHalves * 2 * scoreBlockElements * 4 + 1023) /
                  1024) *
                 1024;
    int64_t oDelta = align16(oP + WG_Q * WG_N * 2);
    int64_t shPayload = align16(oDelta + WG_Q * D * 4);
    int64_t shTotal = shPayload + 1024;

    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridM);
    Value cBlock = b.create<arith::ConstantIndexOp>(loc, WG_THREADS);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value shBytes =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(shTotal));
    auto launch =
        b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlock, c1, c1, shBytes);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);
    Value c4 = b.create<arith::ConstantIndexOp>(loc, 4);
    Value c8 = b.create<arith::ConstantIndexOp>(loc, 8);
    Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value c64 = b.create<arith::ConstantIndexOp>(loc, 64);
    Value c128 = b.create<arith::ConstantIndexOp>(loc, 128);
    Value c1024 = b.create<arith::ConstantIndexOp>(loc, 1024);
    Value cQueryRows = b.create<arith::ConstantIndexOp>(loc, WG_Q);
    Value cQueryTiles = b.create<arith::ConstantIndexOp>(loc, queryTiles);
    Value cHeads = b.create<arith::ConstantIndexOp>(loc, H);
    Value scale =
        b.create<memref::LoadOp>(loc, scaleInput, ValueRange{}).getResult();
    Value ninf = b.create<arith::ConstantOp>(
        loc, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value zeroF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
    Value oneF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1.0));

    Value queryTile = b.create<arith::RemUIOp>(loc, bx, cQueryTiles);
    Value batchHead = b.create<arith::DivUIOp>(loc, bx, cQueryTiles);
    Value head = b.create<arith::RemUIOp>(loc, batchHead, cHeads);
    Value batch = b.create<arith::DivUIOp>(loc, batchHead, cHeads);
    Value rowBlk = b.create<arith::MulIOp>(loc, queryTile, cQueryRows);
    Value warp = b.create<arith::DivUIOp>(loc, tx, c32);
    Value lane = b.create<arith::RemUIOp>(loc, tx, c32);
    Value laneDiv4 = b.create<arith::DivUIOp>(loc, lane, c4);
    Value laneMod4 = b.create<arith::RemUIOp>(loc, lane, c4);
    Value warpRowBase = b.create<arith::MulIOp>(
        loc, warp, b.create<arith::ConstantIndexOp>(loc, warpRowSpan));
    Value stCol = b.create<arith::MulIOp>(loc, laneMod4, c2);

    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    Value sharedBase =
        b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, raw);
    Value sharedRemainder =
        b.create<arith::RemUIOp>(loc, sharedBase, c1024);
    Value bytesToAlignment =
        b.create<arith::SubIOp>(loc, c1024, sharedRemainder);
    Value sharedPad =
        b.create<arith::RemUIOp>(loc, bytesToAlignment, c1024);
    auto view = [&](int64_t offset, ArrayRef<int64_t> shape,
                    Type elementType) -> Value {
      auto ty = MemRefType::get(shape, elementType, nullptr, wgSpace);
      Value off = b.create<arith::AddIOp>(
          loc, sharedPad, b.create<arith::ConstantIndexOp>(loc, offset));
      return b.create<memref::ViewOp>(loc, ty, raw, off, ValueRange{});
    };
    Value kSh = view(oK, {WG_N, D}, f16);
    Value vSh = view(oV, {WG_N, vStride}, f16);
    Value scoreSh =
        view(oScore, {queryHalves, 2, WG_HALF, SCORE_STRIDE}, f32);
    Value pSh = view(oP, {queryHalves, 2, WG_HALF, WG_HALF}, f16);
    Value deltaSh = view(oDelta, {queryHalves, WG_HALF, D}, f32);

    // Keep every stock-MLIR WGMMA accumulator at M64/N64/K64.  MLIR 20's
    // larger fragmented stores/wide descriptors are not reliable on SM90, so
    // BM128 is represented by adjacent, independently verified 64x64 blocks.
    SmallVector<Value> qtBlocks, kBlocks, vBlocks, scoreBlocks, pBlocks,
        deltaBlocks;
    for (int64_t qh = 0; qh < queryHalves; ++qh) {
      qtBlocks.push_back(
          view(oQt + qh * D * WG_HALF * 2, {D, WG_HALF}, f16));
      deltaBlocks.push_back(view(oDelta + qh * WG_HALF * D * 4,
                                 {WG_HALF, D}, f32));
      for (int64_t kh = 0; kh < 2; ++kh) {
        int64_t block = qh * 2 + kh;
        Value physical = view(oScore + block * scoreBlockElements * 4,
                              {WG_HALF, SCORE_STRIDE}, f32);
        scoreBlocks.push_back(b.create<memref::SubViewOp>(
            loc, physical, ArrayRef<int64_t>{0, 0},
            ArrayRef<int64_t>{WG_HALF, WG_HALF},
            ArrayRef<int64_t>{1, 1}));
        pBlocks.push_back(view(oP + block * WG_HALF * WG_HALF * 2,
                               {WG_HALF, WG_HALF}, f16));
      }
    }
    for (int64_t kh = 0; kh < 2; ++kh) {
      kBlocks.push_back(
          view(oK + kh * WG_HALF * D * 2, {WG_HALF, D}, f16));
      if (!hopperHybridPV)
        vBlocks.push_back(
            view(oV + kh * WG_HALF * D * 2, {WG_HALF, D}, f16));
    }

    auto globalIndices = [&](Value row, Value col) {
      return SmallVector<Value>{batch, row, head, col};
    };
    auto statIndices = [&](Value row) {
      return SmallVector<Value>{batch, head, row};
    };
    auto swizzledColumn = [&](OpBuilder &nb, Value row, Value col) -> Value {
      // 128-byte TMA swizzle.  Every operand base is now 1024-byte aligned and
      // every relative operand offset is a multiple of that repeat period.
      Value rowLow = nb.create<arith::RemUIOp>(loc, row, c8);
      Value xorOffset = nb.create<arith::MulIOp>(loc, rowLow, c8);
      return nb.create<arith::XOrIOp>(loc, col, xorOffset);
    };
    auto syncGenericStoresForWgmma = [&](OpBuilder &nb, Location nl) {
      auto barrier = nb.create<gpu::BarrierOp>(nl);
      barrier->setAttr("intent.backend.wgmma_proxy_fence", nb.getUnitAttr());
    };

    // Q is reused across the entire key scan.  Each query half is independently
    // staged as a 64x64 Q^T operand, allowing BM128 without a wide descriptor.
    for (int64_t qh = 0; qh < queryHalves; ++qh) {
      for (int64_t it = 0; it < (WG_HALF * D) / WG_THREADS; ++it) {
        Value base = b.create<arith::ConstantIndexOp>(loc, it * WG_THREADS);
        Value linear = b.create<arith::AddIOp>(loc, tx, base);
        Value qLocal = b.create<arith::DivUIOp>(loc, linear, c64);
        Value dim = b.create<arith::RemUIOp>(loc, linear, c64);
        Value qOffset = b.create<arith::ConstantIndexOp>(loc, qh * WG_HALF);
        Value qRow = b.create<arith::AddIOp>(loc, qOffset, qLocal);
        Value globalRow = b.create<arith::AddIOp>(loc, rowBlk, qRow);
        Value q = b.create<memref::LoadOp>(
            loc, qMem, globalIndices(globalRow, dim));
        if (hopperRegisterQK) {
          Value physicalDim = swizzledColumn(b, qLocal, dim);
          b.create<memref::StoreOp>(loc, q, qtBlocks[qh],
                                    ValueRange{qLocal, physicalDim});
        } else {
          Value physicalCol = swizzledColumn(b, dim, qLocal);
          b.create<memref::StoreOp>(loc, q, qtBlocks[qh],
                                    ValueRange{dim, physicalCol});
        }
      }
    }
    syncGenericStoresForWgmma(b, loc);

    // Warpgroup descriptors use the stock NVGPU lowering. It derives all bits
    // from the shared memref and tensor-map *type*, so a device-local dummy value
    // avoids adding host tensor-map ABI arguments to the portable launch.
    auto makeDescriptor = [&](Value tensor) -> Value {
      auto tensorTy = cast<MemRefType>(tensor.getType());
      // A 128-byte swizzle describes one 64xf16 sector. Wider logical rows are
      // represented by multiple sectors, exactly as in MLIR's SM90 integration
      // test (the warpgroup descriptor retains the complete logical shape).
      MemRefType tensorMapMemrefTy = tensorTy;
      if (tensorTy.getShape().back() > 64) {
        SmallVector<int64_t> mapShape(tensorTy.getShape());
        mapShape.back() = 64;
        tensorMapMemrefTy = MemRefType::get(
            mapShape, tensorTy.getElementType(), nullptr, wgSpace);
      }
      auto tensorMapTy = nvgpu::TensorMapDescriptorType::get(
          ctx, tensorMapMemrefTy,
          nvgpu::TensorMapSwizzleKind::SWIZZLE_128B,
          nvgpu::TensorMapL2PromoKind::L2PROMO_NONE,
          nvgpu::TensorMapOOBKind::OOB_ZERO,
          nvgpu::TensorMapInterleaveKind::INTERLEAVE_NONE);
      Value zero = b.create<arith::ConstantIntOp>(loc, 0, 64);
      Value fakeMap =
          b.create<UnrealizedConversionCastOp>(loc, TypeRange{tensorMapTy},
                                               ValueRange{zero})
              .getResult(0);
      auto descriptorTy =
          nvgpu::WarpgroupMatrixDescriptorType::get(ctx, tensorTy);
      return b.create<nvgpu::WarpgroupGenerateDescriptorOp>(
          loc, descriptorTy, tensor, fakeMap);
    };
    SmallVector<Value> qtDescs, kDescs, vDescs, pDescs;
    for (Value qt : qtBlocks)
      qtDescs.push_back(makeDescriptor(qt));
    for (Value k : kBlocks)
      kDescs.push_back(makeDescriptor(k));
    if (!hopperHybridPV) {
      for (Value v : vBlocks)
        vDescs.push_back(makeDescriptor(v));
      for (Value p : pBlocks)
        pDescs.push_back(makeDescriptor(p));
    }

    SmallVector<Value> qRegisterDescs, kRegisterDescs, pRegisterDescs,
        vRegisterDescs;
    LLVM::LLVMStructType registerWgmmaAccTy;
    if (hopperRegisterQK) {
      // Encode the SM90 128-byte-swizzled matrix descriptor directly in MLIR.
      // The fields match MLIR's stock nvgpu->nvvm conversion for one 64xf16
      // sector: start-address/16 in bits [0,13], 512 in [16,29], 64 in
      // [32,45], and swizzle mode 1 in [62,63].
      auto i64 = b.getI64Type();
      auto cI64 = [&](int64_t value) -> Value {
        return b.create<arith::ConstantIntOp>(loc, value, i64);
      };
      auto makeRegisterDescriptor = [&](int64_t offset) -> Value {
        Value address = b.create<arith::AddIOp>(loc, sharedBase, sharedPad);
        address = b.create<arith::AddIOp>(
            loc, address, b.create<arith::ConstantIndexOp>(loc, offset));
        Value address64 = b.create<arith::IndexCastOp>(loc, i64, address);
        Value base16 = b.create<arith::ShRUIOp>(loc, address64, cI64(4));
        Value base = b.create<arith::AndIOp>(loc, base16, cI64(0x3fff));
        constexpr uint64_t descriptorFields =
            (uint64_t{1} << 62) | (uint64_t{64} << 32) |
            (uint64_t{512} << 16);
        return b.create<arith::OrIOp>(
            loc, base, cI64(static_cast<int64_t>(descriptorFields)));
      };
      for (int64_t qh = 0; qh < queryHalves; ++qh)
        qRegisterDescs.push_back(
            makeRegisterDescriptor(oQt + qh * D * WG_HALF * 2));
      for (int64_t kh = 0; kh < 2; ++kh)
        kRegisterDescs.push_back(
            makeRegisterDescriptor(oK + kh * WG_HALF * D * 2));
      if (hopperRegisterPV) {
        for (int64_t kh = 0; kh < 2; ++kh) {
          pRegisterDescs.push_back(makeRegisterDescriptor(
              oP + kh * WG_HALF * WG_HALF * 2));
          vRegisterDescs.push_back(makeRegisterDescriptor(
              oV + kh * WG_HALF * D * 2));
        }
      }
      SmallVector<Type> fields(32, f32);
      registerWgmmaAccTy = LLVM::LLVMStructType::getLiteral(ctx, fields);
    }

    auto scoreAccTy = nvgpu::WarpgroupAccumulatorType::get(
        ctx, VectorType::get({WG_HALF, WG_HALF}, f32));
    auto deltaAccTy = nvgpu::WarpgroupAccumulatorType::get(
        ctx, VectorType::get({WG_HALF, D}, f32));
    UnitAttr transposeB = b.getUnitAttr();
    auto pFragTy = VectorType::get({4, 2}, f16);
    auto bFragTy = VectorType::get({2, 2}, f16);
    auto mmaShape = b.getI64ArrayAttr({MMA_M, MMA_N, MMA_K});
    Value zeroPFrag = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(pFragTy,
                                    b.getF16FloatAttr(0.0).getValue()));
    Value laneMod16 = b.create<arith::RemUIOp>(loc, lane, c16);

    auto quadReduce = [&](OpBuilder &nb, Value value, bool isMax) -> Value {
      for (int delta : {1, 2}) {
        Value shuffled =
            nb.create<gpu::ShuffleOp>(loc, value, delta, 32,
                                      gpu::ShuffleMode::XOR)
                .getShuffleResult();
        value = isMax
                    ? nb.create<arith::MaxNumFOp>(loc, value, shuffled)
                          .getResult()
                    : nb.create<arith::AddFOp>(loc, value, shuffled)
                          .getResult();
      }
      return value;
    };

    auto accTy = VectorType::get({rowsPerThread, 2}, f32);
    Value zeroAcc = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(accTy, b.getF32FloatAttr(0.0).getValue()));
    auto makeZeroRegisterAccumulator = [&](OpBuilder &ib,
                                           Location il) -> Value {
      Value acc = ib.create<LLVM::UndefOp>(il, registerWgmmaAccTy);
      for (int64_t r = 0; r < 32; ++r)
        acc = ib.create<LLVM::InsertValueOp>(
            il, registerWgmmaAccTy, acc, zeroF, ArrayRef<int64_t>{r});
      return acc;
    };
    int64_t nColO = D / MMA_N;
    int64_t oCarryCount = hopperRegisterPV ? 1 : nColO;
    SmallVector<Value> iterInit;
    if (hopperRegisterPV)
      iterInit.push_back(makeZeroRegisterAccumulator(b, loc));
    else
      iterInit.append(nColO, zeroAcc);
    for (int64_t i = 0; i < rowsPerThread; ++i)
      iterInit.push_back(ninf);
    for (int64_t i = 0; i < rowsPerThread; ++i)
      iterInit.push_back(normalizedSoftmaxCarry ? zeroF : ninf);
    if (normalizedSoftmaxCarry)
      for (int64_t i = 0; i < rowsPerThread; ++i)
        iterInit.push_back(ninf);

    Value cKeyTiles = b.create<arith::ConstantIndexOp>(loc, keyTiles);
    auto asyncTokenTy = nvgpu::DeviceAsyncTokenType::get(ctx);
    IntegerAttr asyncPacketElements = b.getIndexAttr(8);
    UnitAttr asyncBypassL1 = b.getUnitAttr();
    auto loopOp = b.create<scf::ForOp>(
        loc, c0, cKeyTiles, c1, iterInit,
        [&](OpBuilder &nb, Location nl, Value tile, ValueRange args) {
          SmallVector<Value> oAcc;
          Value oRegisterAcc;
          if (hopperRegisterPV)
            oRegisterAcc = args[0];
          else
            oAcc.append(args.begin(), args.begin() + nColO);
          SmallVector<Value> mOld, lOld;
          for (int64_t i = 0; i < rowsPerThread; ++i) {
            mOld.push_back(args[oCarryCount + i]);
            lOld.push_back(args[oCarryCount + rowsPerThread + i]);
          }
          SmallVector<Value> tmpPivotOld;
          if (normalizedSoftmaxCarry)
            for (int64_t i = 0; i < rowsPerThread; ++i)
              tmpPivotOld.push_back(
                  args[oCarryCount + 2 * rowsPerThread + i]);
          Value keyBase = nb.create<arith::MulIOp>(nl, tile, c128);

          // Natural K/V rows, vectorized in the contiguous feature dimension.
          auto vec8Ty = VectorType::get({8}, f16);
          SmallVector<Value> asyncTokens;
          for (int64_t it = 0;
               it < (WG_N * (D / 8)) / WG_THREADS; ++it) {
            Value base =
                nb.create<arith::ConstantIndexOp>(nl, it * WG_THREADS);
            Value linear = nb.create<arith::AddIOp>(nl, tx, base);
            Value keyRow = nb.create<arith::DivUIOp>(nl, linear, c8);
            Value colGroup = nb.create<arith::RemUIOp>(nl, linear, c8);
            Value col = nb.create<arith::MulIOp>(nl, colGroup, c8);
            Value globalKey = nb.create<arith::AddIOp>(nl, keyBase, keyRow);
            Value physicalCol = swizzledColumn(nb, keyRow, col);
            SmallVector<Value> vDestination{
                keyRow,
                hopperHybridPV && !hopperRegisterPV ? col : physicalCol};
            if (hopperAsyncKV) {
              SmallVector<Value> sourceIndices =
                  globalIndices(globalKey, col);
              asyncTokens.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
                  nl, asyncTokenTy, kSh, ValueRange{keyRow, physicalCol}, kMem,
                  sourceIndices, asyncPacketElements, /*srcElements=*/Value(),
                  asyncBypassL1));
              asyncTokens.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
                  nl, asyncTokenTy, vSh, ValueRange(vDestination), vMem,
                  sourceIndices,
                  asyncPacketElements, /*srcElements=*/Value(), asyncBypassL1));
            } else {
              Value kPacket = nb.create<vector::LoadOp>(
                  nl, vec8Ty, kMem, globalIndices(globalKey, col));
              Value vPacket = nb.create<vector::LoadOp>(
                  nl, vec8Ty, vMem, globalIndices(globalKey, col));
              nb.create<vector::StoreOp>(nl, kPacket, kSh,
                                         ValueRange{keyRow, physicalCol});
              nb.create<vector::StoreOp>(nl, vPacket, vSh,
                                         ValueRange(vDestination));
            }
          }
          if (hopperAsyncKV) {
            Value group = nb.create<nvgpu::DeviceAsyncCreateGroupOp>(
                nl, asyncTokenTy, asyncTokens);
            nb.create<nvgpu::DeviceAsyncWaitOp>(
                nl, group, nb.getI32IntegerAttr(0));
          }
          syncGenericStoresForWgmma(nb, nl);

          SmallVector<Value> registerScores;
          if (hopperRegisterQK) {
            nb.create<NVVM::WgmmaFenceAlignedOp>(nl);
            SmallVector<Value> scoreRegisters;
            for (int64_t kh = 0; kh < 2; ++kh) {
              Value acc = makeZeroRegisterAccumulator(nb, nl);
              for (int64_t ks = 0; ks < D / MMA_K; ++ks) {
                Value descriptorOffset = nb.create<arith::ConstantIntOp>(
                    nl, 2 * ks, nb.getI64Type());
                Value qDesc = nb.create<arith::AddIOp>(
                    nl, qRegisterDescs[0], descriptorOffset);
                Value kDesc = nb.create<arith::AddIOp>(
                    nl, kRegisterDescs[kh], descriptorOffset);
                acc = nb.create<NVVM::WgmmaMmaAsyncOp>(
                    nl, registerWgmmaAccTy, acc, qDesc, kDesc,
                    NVVM::MMAShapeAttr::get(ctx, 64, 64, 16),
                    NVVM::WGMMATypes::f16, NVVM::WGMMATypes::f16,
                    NVVM::WGMMATypes::f32, NVVM::WGMMAScaleOut::one,
                    NVVM::WGMMAScaleIn::one, NVVM::WGMMAScaleIn::one,
                    NVVM::MMALayout::row, NVVM::MMALayout::col,
                    NVVM::MMAIntOverflowAttr());
              }
              scoreRegisters.push_back(acc);
            }
            nb.create<NVVM::WgmmaGroupSyncAlignedOp>(nl);
            nb.create<NVVM::WgmmaWaitGroupSyncOp>(
                nl, nb.getI64IntegerAttr(0));

            // PTX m64n64 accumulator ownership: each lane owns two rows and
            // two adjacent columns in every n8 tile.  Rebuild the existing
            // vector<2x2xf32> score fragment interface entirely in SSA.
            for (int64_t kh = 0; kh < 2; ++kh)
              for (int64_t nt = 0; nt < 8; ++nt) {
                Value fragment = zeroAcc;
                for (int i = 0; i < 2; ++i)
                  for (int i2 = 0; i2 < 2; ++i2) {
                    int64_t reg = nt * 4 + i * 2 + i2;
                    Value score = nb.create<LLVM::ExtractValueOp>(
                        nl, f32, scoreRegisters[kh], ArrayRef<int64_t>{reg});
                    fragment = nb.create<vector::InsertOp>(
                        nl, score, fragment,
                        ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
                  }
                registerScores.push_back(fragment);
              }
          } else {
            for (int64_t qh = 0; qh < queryHalves; ++qh)
              for (int64_t kh = 0; kh < 2; ++kh) {
                int64_t block = qh * 2 + kh;
                Value scoreZero =
                    nb.create<nvgpu::WarpgroupMmaInitAccumulatorOp>(nl,
                                                                    scoreAccTy);
                Value scoreAcc = nb.create<nvgpu::WarpgroupMmaOp>(
                    nl, scoreAccTy, kDescs[kh], qtDescs[qh], /*waitGroup=*/0,
                    /*transposeA=*/UnitAttr(), transposeB, scoreZero);
                nb.create<nvgpu::WarpgroupMmaStoreOp>(nl, scoreAcc,
                                                      scoreBlocks[block]);
              }
            nb.create<gpu::BarrierOp>(nl);
          }

          Value rowLocalBase = nb.create<arith::AddIOp>(
              nl, warpRowBase, laneDiv4);
          SmallVector<Value> queryRows, queryHalfValues, queryLocalValues;
          for (int64_t i = 0; i < rowsPerThread; ++i) {
            Value queryRow = nb.create<arith::AddIOp>(
                nl, rowLocalBase,
                nb.create<arith::ConstantIndexOp>(nl, i * 8));
            queryRows.push_back(queryRow);
            queryHalfValues.push_back(
                nb.create<arith::DivUIOp>(nl, queryRow, c64));
            queryLocalValues.push_back(
                nb.create<arith::RemUIOp>(nl, queryRow, c64));
          }

          // Keep the fixed key-group traversal as a compact device loop.  The
          // previous C++ unrolling materialized hundreds of simultaneous SSA
          // temporaries at BM128 and forced ptxas to spill.  Interleaving the
          // independent query rows preserves each row's reduction order while
          // keeping only one score pair per row live.
          Value cKeyGroups =
              nb.create<arith::ConstantIndexOp>(nl, WG_N / 8);
          SmallVector<Value> maxValues(rowsPerThread, ninf);
          if (WG_Q == WG_HALF) {
            // BM64 has only two rows per lane; static traversal removes the
            // dynamic div/rem/address arithmetic without creating the BM128
            // live-range cliff that motivated the compact loop below.
            for (int64_t group = 0; group < WG_N / 8; ++group) {
              Value keyHalf =
                  nb.create<arith::ConstantIndexOp>(nl, group / 8);
              Value keyBase =
                  nb.create<arith::ConstantIndexOp>(nl, (group % 8) * 8);
              for (int64_t i = 0; i < rowsPerThread; ++i)
                for (int i2 = 0; i2 < 2; ++i2) {
                  Value keyLocal = nb.create<arith::AddIOp>(
                      nl, keyBase,
                      nb.create<arith::AddIOp>(
                          nl, stCol,
                          nb.create<arith::ConstantIndexOp>(nl, i2)));
                  Value score;
                  if (hopperRegisterQK) {
                    score = nb.create<vector::ExtractOp>(
                        nl, registerScores[group],
                        ArrayRef<int64_t>{i, i2});
                  } else {
                    score = nb.create<memref::LoadOp>(
                        nl, scoreSh,
                        ValueRange{queryHalfValues[i], keyHalf, keyLocal,
                                   queryLocalValues[i]});
                  }
                  Value scaled = nb.create<arith::MulFOp>(nl, score, scale);
                  maxValues[i] =
                      nb.create<arith::MaxNumFOp>(nl, maxValues[i], scaled);
                }
            }
          } else {
            auto maxLoop = nb.create<scf::ForOp>(
                nl, c0, cKeyGroups, c1, maxValues,
                [&](OpBuilder &lb, Location ll, Value group, ValueRange values) {
                  Value keyHalf = lb.create<arith::DivUIOp>(ll, group, c8);
                  Value localGroup = lb.create<arith::RemUIOp>(ll, group, c8);
                  Value keyBase = lb.create<arith::MulIOp>(ll, localGroup, c8);
                  SmallVector<Value> next(values.begin(), values.end());
                  for (int64_t i = 0; i < rowsPerThread; ++i)
                    for (int i2 = 0; i2 < 2; ++i2) {
                      Value keyLocal = lb.create<arith::AddIOp>(
                          ll, keyBase,
                          lb.create<arith::AddIOp>(
                              ll, stCol,
                              lb.create<arith::ConstantIndexOp>(ll, i2)));
                      Value score = lb.create<memref::LoadOp>(
                          ll, scoreSh,
                          ValueRange{queryHalfValues[i], keyHalf, keyLocal,
                                     queryLocalValues[i]});
                      Value scaled =
                          lb.create<arith::MulFOp>(ll, score, scale);
                      next[i] =
                          lb.create<arith::MaxNumFOp>(ll, next[i], scaled);
                    }
                  lb.create<scf::YieldOp>(ll, next);
                });
            maxValues.assign(maxLoop.getResults().begin(),
                             maxLoop.getResults().end());
          }

          SmallVector<Value> rowMaxValues(rowsPerThread), mNew(rowsPerThread),
              alpha(rowsPerThread);
          for (int64_t i = 0; i < rowsPerThread; ++i) {
            rowMaxValues[i] =
                quadReduce(nb, maxValues[i], /*isMax=*/true);
            Value maxCarry =
                normalizedSoftmaxCarry ? mOld[i] : lOld[i];
            mNew[i] = nb.create<arith::MaxNumFOp>(nl, maxCarry,
                                                  rowMaxValues[i]);
            Value alphaExp =
                nb.create<arith::SubFOp>(nl, mOld[i], mNew[i]);
            alpha[i] = nb.create<math::ExpOp>(nl, alphaExp);
          }

          SmallVector<Value> laneSum(rowsPerThread, zeroF);
          SmallVector<Value> hybridPFragments(WG_N / MMA_K, zeroPFrag);
          if (hopperRegisterPV) {
            for (int64_t t = 0; t < WG_N / 8; ++t) {
              int64_t keyHalf = t / (WG_HALF / 8);
              int64_t localGroup = t % (WG_HALF / 8);
              for (int64_t i = 0; i < rowsPerThread; ++i)
                for (int i2 = 0; i2 < 2; ++i2) {
                  Value keyLocal = nb.create<arith::AddIOp>(
                      nl,
                      nb.create<arith::ConstantIndexOp>(
                          nl, localGroup * 8 + i2),
                      stCol);
                  Value score = nb.create<vector::ExtractOp>(
                      nl, registerScores[t], ArrayRef<int64_t>{i, i2});
                  Value scaled = nb.create<arith::MulFOp>(nl, score, scale);
                  Value delta =
                      nb.create<arith::SubFOp>(nl, scaled, mNew[i]);
                  Value probability = nb.create<math::ExpOp>(nl, delta);
                  laneSum[i] =
                      nb.create<arith::AddFOp>(nl, laneSum[i], probability);
                  Value p16 =
                      nb.create<arith::TruncFOp>(nl, f16, probability);
                  Value physicalKey =
                      swizzledColumn(nb, queryLocalValues[i], keyLocal);
                  nb.create<memref::StoreOp>(
                      nl, p16, pSh,
                      ValueRange{
                          queryHalfValues[i],
                          nb.create<arith::ConstantIndexOp>(nl, keyHalf),
                          queryLocalValues[i], physicalKey});
                }
            }
            syncGenericStoresForWgmma(nb, nl);
          } else if (hopperHybridPV) {
            for (int64_t t = 0; t < WG_N / 8; ++t) {
              int64_t keyHalf = t / (WG_HALF / 8);
              int64_t localGroup = t % (WG_HALF / 8);
              for (int64_t i = 0; i < rowsPerThread; ++i)
                for (int i2 = 0; i2 < 2; ++i2) {
                  Value keyLocal = nb.create<arith::AddIOp>(
                      nl,
                      nb.create<arith::ConstantIndexOp>(
                          nl, localGroup * 8 + i2),
                      stCol);
                  Value score;
                  if (hopperRegisterQK) {
                    score = nb.create<vector::ExtractOp>(
                        nl, registerScores[t],
                        ArrayRef<int64_t>{i, i2});
                  } else {
                    score = nb.create<memref::LoadOp>(
                        nl, scoreSh,
                        ValueRange{
                            queryHalfValues[i],
                            nb.create<arith::ConstantIndexOp>(nl, keyHalf),
                            keyLocal, queryLocalValues[i]});
                  }
                  Value scaled = nb.create<arith::MulFOp>(nl, score, scale);
                  Value delta =
                      nb.create<arith::SubFOp>(nl, scaled, mNew[i]);
                  Value probability = nb.create<math::ExpOp>(nl, delta);
                  laneSum[i] =
                      nb.create<arith::AddFOp>(nl, laneSum[i], probability);
                  Value p16 =
                      nb.create<arith::TruncFOp>(nl, f16, probability);
                  int64_t sub = t / 2;
                  int64_t fragmentRow = (t % 2) * 2 + i;
                  hybridPFragments[sub] = nb.create<vector::InsertOp>(
                      nl, p16, hybridPFragments[sub],
                      ArrayRef<int64_t>{fragmentRow, i2});
                }
            }
          } else {
            if (WG_Q == WG_HALF) {
              for (int64_t group = 0; group < WG_N / 8; ++group) {
                Value keyHalf =
                    nb.create<arith::ConstantIndexOp>(nl, group / 8);
                Value keyBase =
                    nb.create<arith::ConstantIndexOp>(nl, (group % 8) * 8);
                for (int64_t i = 0; i < rowsPerThread; ++i)
                  for (int i2 = 0; i2 < 2; ++i2) {
                    Value keyLocal = nb.create<arith::AddIOp>(
                        nl, keyBase,
                        nb.create<arith::AddIOp>(
                            nl, stCol,
                            nb.create<arith::ConstantIndexOp>(nl, i2)));
                    Value score = nb.create<memref::LoadOp>(
                        nl, scoreSh,
                        ValueRange{queryHalfValues[i], keyHalf, keyLocal,
                                   queryLocalValues[i]});
                    Value scaled = nb.create<arith::MulFOp>(nl, score, scale);
                    Value delta =
                        nb.create<arith::SubFOp>(nl, scaled, mNew[i]);
                    Value probability = nb.create<math::ExpOp>(nl, delta);
                    laneSum[i] =
                        nb.create<arith::AddFOp>(nl, laneSum[i], probability);
                    Value p16 =
                        nb.create<arith::TruncFOp>(nl, f16, probability);
                    Value physicalKey =
                        swizzledColumn(nb, queryLocalValues[i], keyLocal);
                    nb.create<memref::StoreOp>(
                        nl, p16, pSh,
                        ValueRange{queryHalfValues[i], keyHalf,
                                   queryLocalValues[i], physicalKey});
                  }
              }
            } else {
              auto probabilityLoop = nb.create<scf::ForOp>(
                  nl, c0, cKeyGroups, c1, laneSum,
                  [&](OpBuilder &lb, Location ll, Value group,
                      ValueRange values) {
                    Value keyHalf = lb.create<arith::DivUIOp>(ll, group, c8);
                    Value localGroup = lb.create<arith::RemUIOp>(ll, group, c8);
                    Value keyBase =
                        lb.create<arith::MulIOp>(ll, localGroup, c8);
                    SmallVector<Value> next(values.begin(), values.end());
                    for (int64_t i = 0; i < rowsPerThread; ++i)
                      for (int i2 = 0; i2 < 2; ++i2) {
                        Value keyLocal = lb.create<arith::AddIOp>(
                            ll, keyBase,
                            lb.create<arith::AddIOp>(
                                ll, stCol,
                                lb.create<arith::ConstantIndexOp>(ll, i2)));
                        Value score = lb.create<memref::LoadOp>(
                            ll, scoreSh,
                            ValueRange{queryHalfValues[i], keyHalf, keyLocal,
                                       queryLocalValues[i]});
                        Value scaled =
                            lb.create<arith::MulFOp>(ll, score, scale);
                        Value delta =
                            lb.create<arith::SubFOp>(ll, scaled, mNew[i]);
                        Value probability = lb.create<math::ExpOp>(ll, delta);
                        next[i] =
                            lb.create<arith::AddFOp>(ll, next[i], probability);
                        Value p16 =
                            lb.create<arith::TruncFOp>(ll, f16, probability);
                        Value physicalKey =
                            swizzledColumn(lb, queryLocalValues[i], keyLocal);
                        lb.create<memref::StoreOp>(
                            ll, p16, pSh,
                            ValueRange{queryHalfValues[i], keyHalf,
                                       queryLocalValues[i], physicalKey});
                      }
                    lb.create<scf::YieldOp>(ll, next);
                  });
              laneSum.assign(probabilityLoop.getResults().begin(),
                             probabilityLoop.getResults().end());
            }
            syncGenericStoresForWgmma(nb, nl);
          }

          if (hopperRegisterPV) {
            for (int64_t nt = 0; nt < nColO; ++nt)
              for (int64_t i = 0; i < rowsPerThread; ++i)
                for (int i2 = 0; i2 < 2; ++i2) {
                  int64_t reg = nt * 4 + i * 2 + i2;
                  Value element = nb.create<LLVM::ExtractValueOp>(
                      nl, f32, oRegisterAcc, ArrayRef<int64_t>{reg});
                  Value scaled =
                      nb.create<arith::MulFOp>(nl, element, alpha[i]);
                  oRegisterAcc = nb.create<LLVM::InsertValueOp>(
                      nl, registerWgmmaAccTy, oRegisterAcc, scaled,
                      ArrayRef<int64_t>{reg});
                }
          } else {
            for (int64_t jo = 0; jo < nColO; ++jo) {
              Value acc = oAcc[jo];
              for (int64_t i = 0; i < rowsPerThread; ++i)
                for (int i2 = 0; i2 < 2; ++i2) {
                  Value element = nb.create<vector::ExtractOp>(
                      nl, acc,
                      ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
                  Value scaled =
                      nb.create<arith::MulFOp>(nl, element, alpha[i]);
                  acc = nb.create<vector::InsertOp>(
                      nl, scaled, acc,
                      ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
                }
              oAcc[jo] = acc;
            }
          }

          if (hopperRegisterPV) {
            nb.create<NVVM::WgmmaFenceAlignedOp>(nl);
            for (int64_t kh = 0; kh < 2; ++kh)
              for (int64_t ks = 0; ks < WG_HALF / MMA_K; ++ks) {
                Value pOffset = nb.create<arith::ConstantIntOp>(
                    nl, 2 * ks, nb.getI64Type());
                Value vOffset = nb.create<arith::ConstantIntOp>(
                    nl, 128 * ks, nb.getI64Type());
                Value pDesc = nb.create<arith::AddIOp>(
                    nl, pRegisterDescs[kh], pOffset);
                Value vDesc = nb.create<arith::AddIOp>(
                    nl, vRegisterDescs[kh], vOffset);
                oRegisterAcc = nb.create<NVVM::WgmmaMmaAsyncOp>(
                    nl, registerWgmmaAccTy, oRegisterAcc, pDesc, vDesc,
                    NVVM::MMAShapeAttr::get(ctx, 64, 64, 16),
                    NVVM::WGMMATypes::f16, NVVM::WGMMATypes::f16,
                    NVVM::WGMMATypes::f32, NVVM::WGMMAScaleOut::one,
                    NVVM::WGMMAScaleIn::one, NVVM::WGMMAScaleIn::one,
                    NVVM::MMALayout::row, NVVM::MMALayout::row,
                    NVVM::MMAIntOverflowAttr());
              }
            nb.create<NVVM::WgmmaGroupSyncAlignedOp>(nl);
            nb.create<NVVM::WgmmaWaitGroupSyncOp>(
                nl, nb.getI64IntegerAttr(0));
          } else if (hopperHybridPV) {
            for (int64_t s = 0; s < WG_N / MMA_K; ++s) {
              Value kBase =
                  nb.create<arith::ConstantIndexOp>(nl, s * MMA_K);
              Value sourceRow =
                  nb.create<arith::AddIOp>(nl, kBase, laneMod16);
              for (int64_t jo = 0; jo < nColO; ++jo) {
                Value col =
                    nb.create<arith::ConstantIndexOp>(nl, jo * MMA_N);
                Value vFrag = nb.create<nvgpu::LdMatrixOp>(
                    nl, bFragTy, vSh, ValueRange{sourceRow, col},
                    /*transpose=*/true, /*numTiles=*/2);
                oAcc[jo] = nb.create<nvgpu::MmaSyncOp>(
                    nl, hybridPFragments[s], vFrag, oAcc[jo], mmaShape);
              }
            }
          } else {
            for (int64_t qh = 0; qh < queryHalves; ++qh) {
              Value deltaZero = nb.create<
                  nvgpu::WarpgroupMmaInitAccumulatorOp>(nl, deltaAccTy);
              Value deltaLoAcc = nb.create<nvgpu::WarpgroupMmaOp>(
                  nl, deltaAccTy, pDescs[qh * 2], vDescs[0],
                  /*waitGroup=*/0, /*transposeA=*/UnitAttr(), transposeB,
                  deltaZero);
              Value deltaAcc = nb.create<nvgpu::WarpgroupMmaOp>(
                  nl, deltaAccTy, pDescs[qh * 2 + 1], vDescs[1],
                  /*waitGroup=*/0, /*transposeA=*/UnitAttr(), transposeB,
                  deltaLoAcc);
              nb.create<nvgpu::WarpgroupMmaStoreOp>(nl, deltaAcc,
                                                    deltaBlocks[qh]);
            }
            nb.create<gpu::BarrierOp>(nl);

            for (int64_t jo = 0; jo < nColO; ++jo) {
              Value colBase = nb.create<arith::AddIOp>(
                  nl, nb.create<arith::ConstantIndexOp>(nl, jo * 8), stCol);
              Value acc = oAcc[jo];
              for (int64_t i = 0; i < rowsPerThread; ++i) {
                Value row = nb.create<arith::AddIOp>(
                    nl, rowLocalBase,
                    nb.create<arith::ConstantIndexOp>(nl, i * 8));
                Value queryHalf = nb.create<arith::DivUIOp>(nl, row, c64);
                Value queryLocal = nb.create<arith::RemUIOp>(nl, row, c64);
                for (int i2 = 0; i2 < 2; ++i2) {
                  Value col = nb.create<arith::AddIOp>(
                      nl, colBase,
                      nb.create<arith::ConstantIndexOp>(nl, i2));
                  Value update = nb.create<memref::LoadOp>(
                      nl, deltaSh, ValueRange{queryHalf, queryLocal, col});
                  Value old = nb.create<vector::ExtractOp>(
                      nl, acc,
                      ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
                  Value sum = nb.create<arith::AddFOp>(nl, old, update);
                  acc = nb.create<vector::InsertOp>(
                      nl, sum, acc,
                      ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
                }
              }
              oAcc[jo] = acc;
            }
          }

          SmallVector<Value> lNew(rowsPerThread);
          for (int64_t i = 0; i < rowsPerThread; ++i) {
            Value rowSum = quadReduce(nb, laneSum[i], /*isMax=*/false);
            if (normalizedSoftmaxCarry) {
              Value oldMass =
                  nb.create<arith::MulFOp>(nl, lOld[i], alpha[i]);
              lNew[i] = nb.create<arith::AddFOp>(nl, oldMass, rowSum);
            } else {
              Value oldDelta =
                  nb.create<arith::SubFOp>(nl, lOld[i], mNew[i]);
              Value oldMass = nb.create<math::ExpOp>(nl, oldDelta);
              Value denominator =
                  nb.create<arith::AddFOp>(nl, oldMass, rowSum);
              Value logDenominator = nb.create<math::LogOp>(nl, denominator);
              lNew[i] =
                  nb.create<arith::AddFOp>(nl, mNew[i], logDenominator);
            }
          }
          SmallVector<Value> tmpPivotNew;
          if (normalizedSoftmaxCarry) {
            Value lastTile =
                nb.create<arith::ConstantIndexOp>(nl, keyTiles - 1);
            Value isLast = nb.create<arith::CmpIOp>(
                nl, arith::CmpIPredicate::eq, tile, lastTile);
            for (int64_t i = 0; i < rowsPerThread; ++i) {
              auto choosePivot = nb.create<scf::IfOp>(
                  nl, TypeRange{f32}, isLast, /*withElseRegion=*/true);
              {
                OpBuilder::InsertionGuard guard(nb);
                nb.setInsertionPointToStart(
                    &choosePivot.getThenRegion().front());
                Value logMass = nb.create<math::LogOp>(nl, lOld[i]);
                Value prefixLse =
                    nb.create<arith::AddFOp>(nl, mOld[i], logMass);
                Value pivot = nb.create<arith::MaxNumFOp>(
                    nl, prefixLse, rowMaxValues[i]);
                nb.create<scf::YieldOp>(nl, pivot);
              }
              {
                OpBuilder::InsertionGuard guard(nb);
                nb.setInsertionPointToStart(
                    &choosePivot.getElseRegion().front());
                nb.create<scf::YieldOp>(nl, tmpPivotOld[i]);
              }
              tmpPivotNew.push_back(choosePivot.getResult(0));
            }
          }
          nb.create<gpu::BarrierOp>(nl);

          SmallVector<Value> results;
          if (hopperRegisterPV)
            results.push_back(oRegisterAcc);
          else
            results.append(oAcc.begin(), oAcc.end());
          results.append(mNew.begin(), mNew.end());
          results.append(lNew.begin(), lNew.end());
          if (normalizedSoftmaxCarry)
            results.append(tmpPivotNew.begin(), tmpPivotNew.end());
          nb.create<scf::YieldOp>(nl, results);
        });

    SmallVector<Value> carries(loopOp.getResults().begin(),
                               loopOp.getResults().end());
    SmallVector<Value> mFin, lFin, tmpPivotFin, finalScale(rowsPerThread),
        finalLse(rowsPerThread), tmpScale(rowsPerThread);
    for (int64_t i = 0; i < rowsPerThread; ++i) {
      mFin.push_back(carries[oCarryCount + i]);
      lFin.push_back(carries[oCarryCount + rowsPerThread + i]);
      if (normalizedSoftmaxCarry) {
        tmpPivotFin.push_back(
            carries[oCarryCount + 2 * rowsPerThread + i]);
        finalScale[i] = b.create<arith::DivFOp>(loc, oneF, lFin[i]);
        Value logMass = b.create<math::LogOp>(loc, lFin[i]);
        finalLse[i] = b.create<arith::AddFOp>(loc, mFin[i], logMass);
        Value tmpExponent =
            b.create<arith::SubFOp>(loc, tmpPivotFin[i], finalLse[i]);
        tmpScale[i] = b.create<math::ExpOp>(loc, tmpExponent);
      } else {
        Value exponent = b.create<arith::SubFOp>(loc, mFin[i], lFin[i]);
        finalScale[i] = b.create<math::ExpOp>(loc, exponent);
        finalLse[i] = lFin[i];
        tmpScale[i] = finalScale[i];
      }
    }
    Value rowLocalBase = b.create<arith::AddIOp>(loc, warpRowBase, laneDiv4);
    Value isLead = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           laneMod4, c0);
    b.create<scf::IfOp>(loc, isLead, [&](OpBuilder &nb, Location nl) {
      for (int64_t i = 0; i < rowsPerThread; ++i) {
        Value rowLocal = nb.create<arith::AddIOp>(
            nl, rowLocalBase,
            nb.create<arith::ConstantIndexOp>(nl, i * 8));
        Value row = nb.create<arith::AddIOp>(nl, rowBlk, rowLocal);
        SmallVector<Value> indices = statIndices(row);
        nb.create<memref::StoreOp>(nl, finalLse[i], lseMem, indices);
        nb.create<memref::StoreOp>(nl, tmpScale[i], tmpMem, indices);
      }
      nb.create<scf::YieldOp>(nl);
    });

    for (int64_t jo = 0; jo < nColO; ++jo) {
      Value colBase = b.create<arith::AddIOp>(
          loc, b.create<arith::ConstantIndexOp>(loc, jo * 8), stCol);
      for (int64_t i = 0; i < rowsPerThread; ++i) {
        Value rowLocal = b.create<arith::AddIOp>(
            loc, rowLocalBase,
            b.create<arith::ConstantIndexOp>(loc, i * 8));
        Value row = b.create<arith::AddIOp>(loc, rowBlk, rowLocal);
        for (int i2 = 0; i2 < 2; ++i2) {
          Value col = b.create<arith::AddIOp>(
              loc, colBase, b.create<arith::ConstantIndexOp>(loc, i2));
          Value element;
          if (hopperRegisterPV) {
            int64_t reg = jo * 4 + i * 2 + i2;
            element = b.create<LLVM::ExtractValueOp>(
                loc, f32, carries[0], ArrayRef<int64_t>{reg});
          } else {
            element = b.create<vector::ExtractOp>(
                loc, carries[jo],
                ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
          }
          Value normalized =
              b.create<arith::MulFOp>(loc, element, finalScale[i]);
          Value out = b.create<arith::TruncFOp>(loc, f16, normalized);
          b.create<memref::StoreOp>(loc, out, oMem,
                                    globalIndices(row, col));
        }
      }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }

  // ---- emission ---------------------------------------------------------- //
  LogicalResult rewrite() {
    if (hopperWarpgroup)
      return rewriteWarpgroup();
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    // Destination-passing ABI.  The legacy loop form appends O (and optionally
    // LSE); the live flat form appends all three original results verbatim:
    // Out[1,Sq,1,D] f16, LSE[1,1,Sq] f32, TMP[1,1,Sq] f32.
    auto oldFty = fn.getFunctionType();
    SmallVector<Type> inTys;
    for (Type at : oldFty.getInputs()) {
      if (auto t = dyn_cast<RankedTensorType>(at))
        inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
      else
        inTys.push_back(at);
    }
    if (flatRank4) {
      for (auto [index, resultTy] : llvm::enumerate(oldFty.getResults())) {
        if (tmpInplace && index == 2)
          continue;
        auto t = cast<RankedTensorType>(resultTy);
        inTys.push_back(MemRefType::get(t.getShape(), t.getElementType()));
      }
    } else {
      inTys.push_back(MemRefType::get({Sq, D}, f32)); // O
      if (emitLse)
        inTys.push_back(MemRefType::get({Sq, 1}, f32)); // LSE
    }
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");

    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);

    Value qMem = entry->getArgument(qArgIdx);
    Value kMem = entry->getArgument(kArgIdx);
    Value vMem = entry->getArgument(vArgIdx);
    Value scaleInput = entry->getArgument(scaleArgIdx);
    unsigned numInputs = oldFty.getNumInputs();
    Value oMem = entry->getArgument(numInputs);
    bool writeLse = flatRank4 || emitLse;
    Value lseMem = writeLse ? entry->getArgument(numInputs + 1) : Value();
    Value tmpMem = flatRank4
                       ? (tmpInplace ? entry->getArgument(tmpArgIdx)
                                     : entry->getArgument(numInputs + 2))
                       : Value();

    int64_t numWarps = compactQueryWarpMapping ? 4 : BM / MMA_M;
    int64_t queryGroupsPerWarp =
        BM / (numWarps * MMA_M); // compact BM128 maps two M16 groups per warp
    int64_t kBlock = numWarps * 32;
    int64_t numKT = Sk / BN;
    int64_t nColQK = BN / MMA_N; // S n8 column tiles per warp (even)
    int64_t nColPV = D / MMA_N;  // O n8 column tiles per warp
    int64_t kSubQK = D / MMA_K;  // k16 subtiles for QK^T (contraction = D)
    int64_t queryTiles = Sq / BM;
    int64_t gridM = B * H * queryTiles;

    // cp.async multistage pipeline depth: N rotating dynamic-shared K/V buffers
    // let the global loads of tiles ki+1..ki+N-1 overlap the mma.sync+softmax
    // compute of tile ki. Clamped to the K-tile count; 48KB smem cap (D=64,
    // BN=64: per stage K+V = 16KB, so N<=2 fits alongside Q). The online-softmax
    // carries (O/m/l) stay RESIDENT IN REGISTERS (scf.for iter_args) -- only the
    // K/V staging is pipelined.
    int64_t nStages = std::min<int64_t>((int64_t)numStages, numKT);
    if (nStages < 1)
      nStages = 1;

    // Dynamic-shared layout (cp.async hides the GLOBAL latency of BOTH K and V):
    //   Q[BM,D]                staged once.
    //   K[nStages,BN,D]        NATURAL rotating cp.async buffers (read directly
    //                          for S=Q@Kj^T -- the B operand is [d,key] obtained
    //                          from natural K via non-transpose ldmatrix).
    //   V[nStages,BN,D]        NATURAL rotating cp.async buffers.
    //   Vt[D,BN+VPAD]          TRANSPOSED single buffer. The PV B operand
    //                          ([D,BN], read non-transpose) REQUIRES the
    //                          transposed [d,key] storage -- ldmatrix x2 spans
    //                          K=16=key along the contiguous dim, which natural V
    //                          (d-contiguous, 8 wide) cannot provide, and a
    //                          transposed store cannot be cp.async-vectorized.
    //                          So V is cp.async'd NATURAL then transposed
    //                          SHARED->SHARED per tile (cheap; no global latency).
    int64_t oQ = 0;
    int64_t oK = align16(oQ + BM * D * 2);
    int64_t vRowStride = D + directVRowPadding;
    int64_t oV = align16(oK + nStages * BN * D * 2);
    int64_t oVt = align16(oV + nStages * BN * vRowStride * 2);
    int64_t shTotal = directVMatrixLoad
                          ? align16(oV + nStages * BN * vRowStride * 2)
                          : align16(oVt + D * (BN + VPAD) * 2);

    Value cGrid = b.create<arith::ConstantIndexOp>(loc, gridM);
    Value cBlk = b.create<arith::ConstantIndexOp>(loc, kBlock);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value shBytes =
        b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr((int32_t)shTotal));

    auto launch =
        b.create<gpu::LaunchOp>(loc, cGrid, c1, c1, cBlk, c1, c1, shBytes);
    b.setInsertionPointToStart(&launch.getBody().front());

    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    // The live P4 scale is tensor<f32> -> memref<f32>.  Load it on the device
    // with the legal rank-0 index list instead of attempting a host dereference.
    Value scale = flatRank4
                      ? b.create<memref::LoadOp>(loc, scaleInput, ValueRange{})
                            .getResult()
                      : scaleInput;
    Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);
    Value c4 = b.create<arith::ConstantIndexOp>(loc, 4);
    Value c8 = b.create<arith::ConstantIndexOp>(loc, 8);
    Value c16 = b.create<arith::ConstantIndexOp>(loc, 16);
    Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
    Value cBM = b.create<arith::ConstantIndexOp>(loc, BM);
    Value cBN = b.create<arith::ConstantIndexOp>(loc, BN);
    Value cQueryTiles =
        b.create<arith::ConstantIndexOp>(loc, queryTiles);
    Value cHeads = b.create<arith::ConstantIndexOp>(loc, H);
    Value ninf = b.create<arith::ConstantOp>(
        loc, b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value zeroF = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
    // Both the flat certified artifact and the legacy structured form use the
    // normalized-mass epilogue when the Source-specific log-mass recurrence is
    // not selected.  Keep the divisor available for both paths.
    Value oneF =
        b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1.0)).getResult();

    Value queryTile = certifiedFullAxis
                          ? b.create<arith::RemUIOp>(loc, bx, cQueryTiles)
                                .getResult()
                          : bx;
    Value batchHead = certifiedFullAxis
                          ? b.create<arith::DivUIOp>(loc, bx, cQueryTiles)
                                .getResult()
                          : c0;
    Value head = certifiedFullAxis
                     ? b.create<arith::RemUIOp>(loc, batchHead, cHeads)
                           .getResult()
                     : c0;
    Value batch = certifiedFullAxis
                      ? b.create<arith::DivUIOp>(loc, batchHead, cHeads)
                            .getResult()
                      : c0;
    Value rowBlk =
        b.create<arith::MulIOp>(loc, queryTile, cBM); // query base row
    Value warp = b.create<arith::DivUIOp>(loc, tx, c32);
    Value lane = b.create<arith::RemUIOp>(loc, tx, c32);
    Value warpRow16 = b.create<arith::MulIOp>(loc, warp, c16); // strip base
    // ldmatrix lane offsets.
    Value laneMod16 = b.create<arith::RemUIOp>(loc, lane, c16);
    Value laneDiv16 = b.create<arith::DivUIOp>(loc, lane, c16);
    Value aColLane = b.create<arith::MulIOp>(loc, laneDiv16, c8); // (lane/16)*8
    Value aRow = b.create<arith::AddIOp>(loc, warpRow16, laneMod16);
    Value laneMod8 = b.create<arith::RemUIOp>(loc, lane, c8);
    Value laneDiv8 = b.create<arith::DivUIOp>(loc, lane, c8);
    Value bColLane = b.create<arith::MulIOp>(loc, laneDiv8, c8); // (lane/8)*8
    Value laneDiv4 = b.create<arith::DivUIOp>(loc, lane, c4);    // acc row group
    Value laneMod4 = b.create<arith::RemUIOp>(loc, lane, c4);
    Value stCol = b.create<arith::MulIOp>(loc, laneMod4, c2); // (lane%4)*2

    // Shared views.
    auto rawTy = MemRefType::get({ShapedType::kDynamic}, b.getI8Type(), nullptr,
                                 wgSpace);
    Value raw = b.create<gpu::DynamicSharedMemoryOp>(loc, rawTy);
    auto viewF16 = [&](int64_t off, ArrayRef<int64_t> shp) -> Value {
      auto ty = MemRefType::get(shp, f16, nullptr, wgSpace);
      Value o = b.create<arith::ConstantIndexOp>(loc, off);
      return b.create<memref::ViewOp>(loc, ty, raw, o, ValueRange{});
    };
    Value qSh = viewF16(oQ, {BM, D});
    Value kSh = viewF16(oK, {nStages, BN, D});  // rotating cp.async K buffers
    Value vSh = viewF16(oV, {nStages, BN, vRowStride});
    Value vtSh = directVMatrixLoad
                     ? Value()
                     : viewF16(oVt, {D, BN + VPAD});

    auto globalMatrixIndices = [&](Value row, Value col) {
      if (certifiedFullAxis)
        return SmallVector<Value>{batch, row, head, col};
      if (flatRank4)
        return SmallVector<Value>{c0, row, c0, col};
      return SmallVector<Value>{row, col};
    };
    auto statIndices = [&](Value row) {
      if (certifiedFullAxis)
        return SmallVector<Value>{batch, head, row};
      if (flatRank4)
        return SmallVector<Value>{c0, c0, row};
      return SmallVector<Value>{row, c0};
    };

    // Cooperative copy helper: run body(i,r,c) over each linear element of [R,C].
    auto forEachElem = [&](OpBuilder &nb, int64_t R, int64_t C,
                           function_ref<void(OpBuilder &, Value, Value)> fn2) {
      Value cC = nb.create<arith::ConstantIndexOp>(loc, C);
      int64_t iters = (R * C) / kBlock;
      for (int64_t it = 0; it < iters; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(loc, it * kBlock);
        Value i = nb.create<arith::AddIOp>(loc, tx, base);
        Value r = nb.create<arith::DivUIOp>(loc, i, cC);
        Value c = nb.create<arith::RemUIOp>(loc, i, cC);
        fn2(nb, r, c);
      }
    };

    // Stage Q[rowBlk:+BM,:] -> qSh once.  Full can reuse the wide, validated
    // global-to-shared transaction shape from the attention backward path;
    // DirectLower retains the scalar canonical staging so this remains an
    // independently searchable instruction-layout state.
    if (vectorizedQStaging) {
      auto vec8F16Ty = VectorType::get({8}, f16);
      Value qColsPerRow = b.create<arith::ConstantIndexOp>(loc, D / 8);
      int64_t vectorsPerThread = (BM * (D / 8)) / kBlock;
      for (int64_t it = 0; it < vectorsPerThread; ++it) {
        Value base = b.create<arith::ConstantIndexOp>(loc, it * kBlock);
        Value vectorIndex = b.create<arith::AddIOp>(loc, tx, base);
        Value row =
            b.create<arith::DivUIOp>(loc, vectorIndex, qColsPerRow);
        Value columnGroup =
            b.create<arith::RemUIOp>(loc, vectorIndex, qColsPerRow);
        Value column = b.create<arith::MulIOp>(loc, columnGroup, c8);
        Value globalRow = b.create<arith::AddIOp>(loc, rowBlk, row);
        Value packet = b.create<vector::LoadOp>(
            loc, vec8F16Ty, qMem, globalMatrixIndices(globalRow, column));
        b.create<vector::StoreOp>(loc, packet, qSh,
                                  ValueRange{row, column});
      }
    } else {
      forEachElem(b, BM, D, [&](OpBuilder &nb, Value r, Value c) {
        Value gr = nb.create<arith::AddIOp>(loc, rowBlk, r);
        SmallVector<Value> qIndices = globalMatrixIndices(gr, c);
        Value v = nb.create<memref::LoadOp>(loc, qMem, qIndices);
        nb.create<memref::StoreOp>(loc, v, qSh, ValueRange{r, c});
      });
    }
    b.create<gpu::BarrierOp>(loc);

    auto aFragTy = VectorType::get({4, 2}, f16);
    auto bFragTy = VectorType::get({2, 2}, f16);
    auto bPairTy = VectorType::get({4, 2}, f16);
    auto accTy = VectorType::get({2, 2}, f32);
    auto mmaShape = b.getI64ArrayAttr({MMA_M, MMA_N, MMA_K});
    Value zeroAcc = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(accTy, b.getF32FloatAttr(0.0).getValue()));
    Value zeroPFrag = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(aFragTy, b.getF16FloatAttr(0.0).getValue()));
    Value zeroBFrag = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(bFragTy, b.getF16FloatAttr(0.0).getValue()));

    // Quad reduction over the 4 lanes {4q..4q+3} (differ in low 2 bits).
    auto quadReduce = [&](OpBuilder &nb, Value v, bool isMax) -> Value {
      for (int delta : {1, 2}) {
        Value sh = nb.create<gpu::ShuffleOp>(loc, v, delta, 32,
                                             gpu::ShuffleMode::XOR)
                       .getShuffleResult();
        v = isMax ? nb.create<arith::MaxNumFOp>(loc, v, sh).getResult()
                  : nb.create<arith::AddFOp>(loc, v, sh).getResult();
      }
      return v;
    };

    // ---- cp.async machinery (port of the M7c/M8b multistage K-loop pipeline) //
    // Each key tile K[BN,D] and V[BN,D] is streamed global->shared by
    // nvgpu.device_async_copy (-> cp.async.cg, 16-byte/8-f16 vectors, L1
    // bypassed), grouped + waited by device_async_create_group / device_async_
    // wait. N rotating buffers let the loads of tiles ki+1..ki+N-1 overlap the
    // mma.sync+softmax compute of tile ki. The online-softmax carries (O/m/l)
    // stay RESIDENT IN REGISTERS (scf.for iter_args / threaded SSA) -- only the
    // K/V staging is pipelined.
    auto tokTy = nvgpu::DeviceAsyncTokenType::get(ctx);
    IntegerAttr dstElems = b.getIndexAttr(8); // 8 f16 = 16 bytes per copy
    UnitAttr bypassL1 = b.getUnitAttr();      // cp.async.cg (skip L1)
    int64_t colsPerRow = D / 8;               // 8-wide vectors along D
    int64_t vPerThread = (BN * D) / 8 / kBlock;
    Value cCPR = b.create<arith::ConstantIndexOp>(loc, colsPerRow);
    Value cNStages = b.create<arith::ConstantIndexOp>(loc, nStages);
    Value cNm1 = b.create<arith::ConstantIndexOp>(loc, nStages - 1);

    // Issue the K+V async copies for the key tile at row `keyBase` into rotating
    // buffer `stageVal`; return the group token. Both K and V are NATURAL [BN,D]
    // (contiguous in global rows and D-wide shared rows) so the 16-byte cp.async
    // vectorizes cleanly.
    auto loadKV = [&](OpBuilder &nb, Location nl, Value keyBase,
                      Value stageVal) -> Value {
      SmallVector<Value> toks;
      for (int64_t it = 0; it < vPerThread; ++it) {
        Value base = nb.create<arith::ConstantIndexOp>(nl, it * kBlock);
        Value vt = nb.create<arith::AddIOp>(nl, tx, base);
        Value r = nb.create<arith::DivUIOp>(nl, vt, cCPR);
        Value colg = nb.create<arith::RemUIOp>(nl, vt, cCPR);
        Value cc = nb.create<arith::MulIOp>(nl, colg, c8);
        Value gr = nb.create<arith::AddIOp>(nl, keyBase, r);
        SmallVector<Value> srcIndices = globalMatrixIndices(gr, cc);
        toks.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
            nl, tokTy, kSh, ValueRange{stageVal, r, cc}, kMem,
            srcIndices, dstElems, /*srcElements=*/Value(), bypassL1));
        toks.push_back(nb.create<nvgpu::DeviceAsyncCopyOp>(
            nl, tokTy, vSh, ValueRange{stageVal, r, cc}, vMem,
            srcIndices, dstElems, /*srcElements=*/Value(), bypassL1));
      }
      return nb.create<nvgpu::DeviceAsyncCreateGroupOp>(nl, tokTy, toks);
    };

    auto pIdx = [&](int64_t t, int i, int i2) { return (t * 2 + i) * 2 + i2; };

    // Compute one key tile from rotating buffers `stageR`: transpose V (natural
    // -> vtSh, shared->shared), then S=Q@Kj^T -> register online softmax ->
    // O+=P@Vj, all via mma.sync. Carries in/out = [oAcc(nColPV), m0,m1,l0,l1].
    auto computeTile = [&](OpBuilder &nb, Location nl, Value stageR,
                           Value keyBase,
                           ArrayRef<Value> carries) -> SmallVector<Value> {
      if (!directVMatrixLoad) {
        // Canonical path: materialize V^T once per key tile, then let every
        // compact query group consume the same transposed staging buffer.
        forEachElem(nb, BN, D, [&](OpBuilder &cb, Value r, Value c) {
          Value vv =
              cb.create<memref::LoadOp>(nl, vSh, ValueRange{stageR, r, c});
          cb.create<memref::StoreOp>(nl, vv, vtSh, ValueRange{c, r});
        });
        nb.create<gpu::BarrierOp>(nl);
      }

      SmallVector<Value> allResults;
      int64_t carryWidth = nColPV + 4 + (normalizedSoftmaxCarry ? 2 : 0);
      for (int64_t queryGroup = 0; queryGroup < queryGroupsPerWarp;
           ++queryGroup) {
      int64_t carryBase = queryGroup * carryWidth;
      SmallVector<Value> oAcc(carries.begin() + carryBase,
                              carries.begin() + carryBase + nColPV);
      Value mOld[2] = {carries[carryBase + nColPV],
                       carries[carryBase + nColPV + 1]};
      Value lOld[2] = {carries[carryBase + nColPV + 2],
                       carries[carryBase + nColPV + 3]};
      Value tmpPivotOld[2], tmpPivotNew[2];
      if (normalizedSoftmaxCarry) {
        tmpPivotOld[0] = carries[carryBase + nColPV + 4];
        tmpPivotOld[1] = carries[carryBase + nColPV + 5];
      }
      Value queryRowOffset = nb.create<arith::ConstantIndexOp>(
          nl, queryGroup * numWarps * MMA_M);
      Value groupWarpRow = queryGroup == 0
                               ? warpRow16
                               : nb.create<arith::AddIOp>(nl, warpRow16,
                                                         queryRowOffset)
                                     .getResult();
      Value aRowGroup = queryGroup == 0
                            ? aRow
                            : nb.create<arith::AddIOp>(nl, aRow, queryRowOffset)
                                  .getResult();
      // Causal masking (R-FLASH-3): a lane's S accumulator element [i][i2] of n8
      // tile t is query row (rowBlk + warp*16 + lane/4 + i*8), key col
      // (keyBase + 8*t + (lane%4)*2 + i2). We keep only key <= query. Precompute
      // the lane-local global query-row base (rowBlk + warp*16 + lane/4).
      Value qGlobBase;
      if (causal)
        qGlobBase = nb.create<arith::AddIOp>(
            nl, nb.create<arith::AddIOp>(nl, rowBlk, groupWarpRow), laneDiv4);
      // ss <- -inf when the element is above the causal diagonal.
      auto maskCausal = [&](OpBuilder &mb, Value ss, int i, int64_t t,
                            int i2) -> Value {
        if (!causal)
          return ss;
        Value qG = mb.create<arith::AddIOp>(
            nl, qGlobBase, mb.create<arith::ConstantIndexOp>(nl, i * 8));
        Value kG = mb.create<arith::AddIOp>(
            nl, keyBase,
            mb.create<arith::AddIOp>(
                nl, stCol, mb.create<arith::ConstantIndexOp>(nl, 8 * t + i2)));
        Value keep =
            mb.create<arith::CmpIOp>(nl, arith::CmpIPredicate::ule, kG, qG);
        return mb.create<arith::SelectOp>(nl, keep, ss, ninf);
      };

      // Form a contiguous fragment of S=Q@Kj^T.  The default path requests the
      // complete BN columns once.  Full may request smaller fragments twice:
      // first to derive the same BN-wide row maximum, then to recompute and
      // immediately consume P in PV.  Fragmenting the implementation does not
      // split the authored online-softmax update.
      auto computeScoreFragment =
          [&](int64_t firstTile,
              int64_t tileCount) -> SmallVector<Value> {
        SmallVector<Value> scores(tileCount, zeroAcc);
        for (int64_t ks = 0; ks < kSubQK; ++ks) {
          Value ksc = nb.create<arith::ConstantIndexOp>(nl, ks * MMA_K);
          Value aKcol = nb.create<arith::AddIOp>(nl, ksc, aColLane);
          Value aFrag = nb.create<nvgpu::LdMatrixOp>(
              nl, aFragTy, qSh, ValueRange{aRowGroup, aKcol},
              /*transpose=*/false, /*numTiles=*/4);
          Value bKcol = nb.create<arith::AddIOp>(nl, ksc, bColLane);
          if (pairedBMatrixLoads) {
            for (int64_t j = 0; j < tileCount; j += 2) {
              int64_t globalTile = firstTile + j;
              Value nb8 = nb.create<arith::ConstantIndexOp>(
                  nl, globalTile * MMA_N);
              Value pair = cuda_detail::loadMmaBPair(
                  nb, nl, kSh, nb8, ksc, laneDiv8, laneMod8, c2, c8,
                  bPairTy, ValueRange{stageR});
              for (int64_t half = 0; half < 2; ++half)
                scores[j + half] = nb.create<nvgpu::MmaSyncOp>(
                    nl, aFrag,
                    cuda_detail::extractMmaBPairHalf(
                        nb, nl, pair, zeroBFrag, half),
                    scores[j + half], mmaShape);
            }
          } else {
            for (int64_t j = 0; j < tileCount; ++j) {
              int64_t globalTile = firstTile + j;
              Value nb8 = nb.create<arith::ConstantIndexOp>(
                  nl, globalTile * MMA_N);
              Value bRow = nb.create<arith::AddIOp>(nl, nb8, laneMod8);
              Value bFrag = nb.create<nvgpu::LdMatrixOp>(
                  nl, bFragTy, kSh, ValueRange{stageR, bRow, bKcol},
                  /*transpose=*/false, /*numTiles=*/2);
              scores[j] = nb.create<nvgpu::MmaSyncOp>(
                  nl, aFrag, bFrag, scores[j], mmaShape);
            }
          }
        }
        return scores;
      };

      // Form one P fragment and consume it immediately in PV.  This helper is
      // shared by the canonical whole-P state and the Full streaming state; the
      // only difference is whether all P scalars remain live before the first
      // PV MMA.
      auto accumulatePV = [&](int64_t s, Value pFrag) {
        Value ksc = nb.create<arith::ConstantIndexOp>(nl, s * MMA_K);
        Value bKcol = nb.create<arith::AddIOp>(nl, ksc, bColLane);
        if (pairedBMatrixLoads) {
          for (int64_t jo = 0; jo < nColPV; jo += 2) {
            Value nb8 = nb.create<arith::ConstantIndexOp>(nl, jo * MMA_N);
            Value pair = cuda_detail::loadMmaBTransposedPair(
                nb, nl, vSh, nb8, ksc, laneMod16, laneDiv16, c8,
                bPairTy, ValueRange{stageR});
            for (int64_t half = 0; half < 2; ++half)
              oAcc[jo + half] = nb.create<nvgpu::MmaSyncOp>(
                  nl, pFrag,
                  cuda_detail::extractMmaBPairHalf(nb, nl, pair, zeroBFrag,
                                                   half),
                  oAcc[jo + half], mmaShape);
          }
        } else {
          for (int64_t jo = 0; jo < nColPV; ++jo) {
            Value nb8 = nb.create<arith::ConstantIndexOp>(nl, jo * MMA_N);
            Value bFrag;
            if (directVMatrixLoad) {
              Value sourceRow = nb.create<arith::AddIOp>(nl, ksc, laneMod16);
              bFrag = nb.create<nvgpu::LdMatrixOp>(
                  nl, bFragTy, vSh, ValueRange{stageR, sourceRow, nb8},
                  /*transpose=*/true, /*numTiles=*/2);
            } else {
              Value bRow = nb.create<arith::AddIOp>(nl, nb8, laneMod8);
              bFrag = nb.create<nvgpu::LdMatrixOp>(
                  nl, bFragTy, vtSh, ValueRange{bRow, bKcol},
                  /*transpose=*/false, /*numTiles=*/2);
            }
            oAcc[jo] = nb.create<nvgpu::MmaSyncOp>(
                nl, pFrag, bFrag, oAcc[jo], mmaShape);
          }
        }
      };

      // (2) Derive the BN-wide row maximum.  With score fragmentation, only
      // lane-local scalar maxima survive between fragments; the quad reduction
      // remains after the same t-major leaf sequence as the canonical path.
      int64_t scoreTilesPerFragment =
          scoreFragmentColumns == 0 ? nColQK
                                    : scoreFragmentColumns / MMA_N;
      SmallVector<Value> wholeScores;
      Value laneMax[2] = {ninf, ninf};
      for (int64_t firstTile = 0; firstTile < nColQK;
           firstTile += scoreTilesPerFragment) {
        SmallVector<Value> scores =
            computeScoreFragment(firstTile, scoreTilesPerFragment);
        for (int i = 0; i < 2; ++i)
          for (int64_t t = 0; t < scoreTilesPerFragment; ++t)
            for (int i2 = 0; i2 < 2; ++i2) {
              Value score = nb.create<vector::ExtractOp>(
                  nl, scores[t],
                  ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
              Value scaled = nb.create<arith::MulFOp>(nl, score, scale);
              scaled = maskCausal(nb, scaled, i, firstTile + t, i2);
              laneMax[i] =
                  nb.create<arith::MaxNumFOp>(nl, laneMax[i], scaled);
            }
        if (scoreFragmentColumns == 0)
          wholeScores = std::move(scores);
      }

      Value rowMax[2], mNew[2], alpha[2], lNew[2];
      for (int i = 0; i < 2; ++i) {
        rowMax[i] = quadReduce(nb, laneMax[i], /*isMax=*/true);
        // The certified Source uses its previous log-sum-exp, not the previous
        // max, as the second operand of this maximum.  Keep that authored
        // recurrence distinct from the historical normalized loop.
        Value maxCarry = certifiedFullAxis && !normalizedSoftmaxCarry
                             ? lOld[i]
                             : mOld[i];
        mNew[i] = nb.create<arith::MaxNumFOp>(nl, maxCarry, rowMax[i]);
        Value md = nb.create<arith::SubFOp>(nl, mOld[i], mNew[i]);
        alpha[i] = nb.create<math::ExpOp>(nl, md);
        if (materializeTmpRoundtrip) {
          Value ic = nb.create<arith::ConstantIndexOp>(nl, i * 8);
          Value row = nb.create<arith::AddIOp>(
              nl,
              nb.create<arith::AddIOp>(nl, rowBlk, groupWarpRow),
              nb.create<arith::AddIOp>(nl, laneDiv4, ic));
          SmallVector<Value> indices = statIndices(row);
          nb.create<memref::StoreOp>(nl, alpha[i], tmpMem, indices);
          alpha[i] = nb.create<memref::LoadOp>(nl, tmpMem, indices);
        }
      }

      SmallVector<Value> Pf;
      if (!streamPFragments) {
        Pf.resize(nColQK * 2 * 2);
      }
      Value streamedLSum[2] = {zeroF, zeroF};
      if (!streamPFragments) {
        for (int i = 0; i < 2; ++i)
          for (int64_t t = 0; t < nColQK; ++t)
            for (int i2 = 0; i2 < 2; ++i2) {
              Value score = nb.create<vector::ExtractOp>(
                  nl, wholeScores[t],
                  ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
              Value scaled = nb.create<arith::MulFOp>(nl, score, scale);
              scaled = maskCausal(nb, scaled, i, t, i2);
              Value delta = nb.create<arith::SubFOp>(nl, scaled, mNew[i]);
              Value probability = nb.create<math::ExpOp>(nl, delta);
              streamedLSum[i] = nb.create<arith::AddFOp>(
                  nl, streamedLSum[i], probability);
              Pf[pIdx(t, i, i2)] =
                  nb.create<arith::TruncFOp>(nl, f16, probability);
            }
      }

      // (3) rescale O by alpha (per-row, lane-local: O row = groupID+i*8).
      for (int64_t jo = 0; jo < nColPV; ++jo) {
        Value v = oAcc[jo];
        for (int i = 0; i < 2; ++i)
          for (int i2 = 0; i2 < 2; ++i2) {
            Value e = nb.create<vector::ExtractOp>(
                nl, v, ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
            Value se = nb.create<arith::MulFOp>(nl, e, alpha[i]);
            v = nb.create<vector::InsertOp>(
                nl, se, v, ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
          }
        oAcc[jo] = v;
      }

      // (4) O += P@Vj via mma.sync.  Fragmented scores are recomputed here only
      // after the complete authored row maximum is known, and each fragment is
      // consumed before the next becomes live.  PV and row-sum leaves retain the
      // canonical global key order.
      for (int64_t firstTile = 0; firstTile < nColQK;
           firstTile += scoreTilesPerFragment) {
        SmallVector<Value> scores = scoreFragmentColumns == 0
                                        ? wholeScores
                                        : computeScoreFragment(
                                              firstTile,
                                              scoreTilesPerFragment);
        int64_t fragmentKSub = scoreTilesPerFragment / 2;
        for (int64_t localSub = 0; localSub < fragmentKSub; ++localSub) {
          int64_t globalSub = firstTile / 2 + localSub;
          int64_t localT0 = 2 * localSub, localT1 = 2 * localSub + 1;
          int64_t globalT0 = firstTile + localT0;
          int64_t globalT1 = firstTile + localT1;
          int64_t localMap[4][2] = {
              {localT0, 0}, {localT0, 1}, {localT1, 0}, {localT1, 1}};
          int64_t globalTiles[4] = {
              globalT0, globalT0, globalT1, globalT1};
          Value pFrag = zeroPFrag;
          for (int r = 0; r < 4; ++r)
            for (int i2 = 0; i2 < 2; ++i2) {
              int i = localMap[r][1];
              Value probability;
              if (streamPFragments) {
                Value score = nb.create<vector::ExtractOp>(
                    nl, scores[localMap[r][0]],
                    ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
                Value scaled = nb.create<arith::MulFOp>(nl, score, scale);
                scaled = maskCausal(nb, scaled, i, globalTiles[r], i2);
                Value delta =
                    nb.create<arith::SubFOp>(nl, scaled, mNew[i]);
                Value p32 = nb.create<math::ExpOp>(nl, delta);
                streamedLSum[i] =
                    nb.create<arith::AddFOp>(nl, streamedLSum[i], p32);
                probability = nb.create<arith::TruncFOp>(nl, f16, p32);
              } else {
                probability = Pf[pIdx(globalTiles[r], i, i2)];
              }
              pFrag = nb.create<vector::InsertOp>(
                  nl, probability, pFrag,
                  ArrayRef<int64_t>{(int64_t)r, (int64_t)i2});
            }
          accumulatePV(globalSub, pFrag);
        }
      }

      // The streaming path delays only the row-summary finalization; it keeps
      // the same t-major leaf order and the same quad reduction as DirectLower.
      for (int i = 0; i < 2; ++i) {
        Value rsum = quadReduce(nb, streamedLSum[i], /*isMax=*/false);
        if (certifiedFullAxis && !normalizedSoftmaxCarry) {
          Value lseDelta =
              nb.create<arith::SubFOp>(nl, lOld[i], mNew[i]);
          Value oldMass = nb.create<math::ExpOp>(nl, lseDelta);
          Value denominator = nb.create<arith::AddFOp>(nl, oldMass, rsum);
          Value logDenominator = nb.create<math::LogOp>(nl, denominator);
          lNew[i] =
              nb.create<arith::AddFOp>(nl, mNew[i], logDenominator);
        } else {
          Value ls = nb.create<arith::MulFOp>(nl, lOld[i], alpha[i]);
          lNew[i] = nb.create<arith::AddFOp>(nl, ls, rsum);
        }
      }
      if (normalizedSoftmaxCarry) {
        // TMP exposes the Source realization's final normalization pivot, even
        // though Out/LSE do not.  Reconstruct that pivot only for the last key
        // tile: lse(prefix) = mOld + log(lOld), then Source chooses
        // max(lse(prefix), rowMax(last)).  The uniform branch avoids paying the
        // logarithm on every tile while preserving the observable scratch value.
        Value lastKeyBase =
            nb.create<arith::ConstantIndexOp>(nl, (numKT - 1) * BN);
        Value isLast = nb.create<arith::CmpIOp>(
            nl, arith::CmpIPredicate::eq, keyBase, lastKeyBase);
        for (int i = 0; i < 2; ++i) {
          auto choosePivot =
              nb.create<scf::IfOp>(nl, TypeRange{f32}, isLast,
                                   /*withElseRegion=*/true);
          {
            OpBuilder::InsertionGuard guard(nb);
            nb.setInsertionPointToStart(&choosePivot.getThenRegion().front());
            Value logMass = nb.create<math::LogOp>(nl, lOld[i]);
            Value prefixLse =
                nb.create<arith::AddFOp>(nl, mOld[i], logMass);
            Value pivot =
                nb.create<arith::MaxNumFOp>(nl, prefixLse, rowMax[i]);
            nb.create<scf::YieldOp>(nl, pivot);
          }
          {
            OpBuilder::InsertionGuard guard(nb);
            nb.setInsertionPointToStart(&choosePivot.getElseRegion().front());
            nb.create<scf::YieldOp>(nl, tmpPivotOld[i]);
          }
          tmpPivotNew[i] = choosePivot.getResult(0);
        }
      }
      SmallVector<Value> results(oAcc.begin(), oAcc.end());
      results.push_back(mNew[0]);
      results.push_back(mNew[1]);
      results.push_back(lNew[0]);
      results.push_back(lNew[1]);
      if (normalizedSoftmaxCarry) {
        results.push_back(tmpPivotNew[0]);
        results.push_back(tmpPivotNew[1]);
      }
      allResults.append(results.begin(), results.end());
      }
      nb.create<gpu::BarrierOp>(nl); // all K/V reads done before stage reuse
      return allResults;
    };

    // One independent register carry packet per query group.
    SmallVector<Value> iterInit;
    for (int64_t queryGroup = 0; queryGroup < queryGroupsPerWarp;
         ++queryGroup) {
      for (int64_t j = 0; j < nColPV; ++j)
        iterInit.push_back(zeroAcc);
      iterInit.push_back(ninf); // m for row groupID    (i=0)
      iterInit.push_back(ninf); // m for row groupID+8  (i=1)
      iterInit.push_back(certifiedFullAxis && !normalizedSoftmaxCarry ? ninf
                                                                      : zeroF);
      iterInit.push_back(certifiedFullAxis && !normalizedSoftmaxCarry ? ninf
                                                                      : zeroF);
      if (normalizedSoftmaxCarry) {
        iterInit.push_back(ninf);
        iterInit.push_back(ninf);
      }
    }

    // PROLOGUE: prefetch key tiles 0..N-2 (none when N==1). Keep the last
    // group's token live (the numGroups-driven epilogue wait dependency).
    Value prologTok;
    for (int s = 0; s < nStages - 1; ++s) {
      Value keyBase = b.create<arith::ConstantIndexOp>(loc, s * BN);
      Value stage = b.create<arith::ConstantIndexOp>(loc, s);
      prologTok = loadKV(b, loc, keyBase, stage);
    }

    // STEADY STATE: ki = 0..numKT-N. Prefetch tile ki+N-1 into buffer
    // (ki+N-1)%N, wait for tile ki's group (keep N-1 groups in flight), barrier,
    // compute tile ki from buffer ki%N. Carries threaded via scf.for iter_args.
    int64_t steadyTrips = numKT - nStages + 1; // >= 1 (nStages <= numKT)
    Value cSteady = b.create<arith::ConstantIndexOp>(loc, steadyTrips);
    IntegerAttr waitNm1 = b.getI32IntegerAttr(nStages - 1);
    auto forOp = b.create<scf::ForOp>(
        loc, c0, cSteady, c1, iterInit,
        [&](OpBuilder &nb, Location nl, Value ki, ValueRange args) {
          Value nextTile = nb.create<arith::AddIOp>(nl, ki, cNm1);
          Value keyNext = nb.create<arith::MulIOp>(nl, nextTile, cBN);
          Value stageW = nb.create<arith::RemUIOp>(nl, nextTile, cNStages);
          Value grp = loadKV(nb, nl, keyNext, stageW);
          nb.create<nvgpu::DeviceAsyncWaitOp>(nl, grp, waitNm1);
          nb.create<gpu::BarrierOp>(nl); // async tile ki visible to all warps
          Value stageR = nb.create<arith::RemUIOp>(nl, ki, cNStages);
          Value keyBaseR = nb.create<arith::MulIOp>(nl, ki, cBN);
          SmallVector<Value> cur(args.begin(), args.end());
          SmallVector<Value> res = computeTile(nb, nl, stageR, keyBaseR, cur);
          nb.create<scf::YieldOp>(nl, res);
        });

    // EPILOGUE of the pipeline: the last N-1 already-prefetched tiles, draining
    // one in-flight group per tile via a decreasing static numGroups. Carries
    // threaded as plain SSA.
    SmallVector<Value> carries(forOp.getResults().begin(),
                               forOp.getResults().end());
    for (int j = 0; j < nStages - 1; ++j) {
      int64_t tile = numKT - nStages + 1 + j;
      Value stageR = b.create<arith::ConstantIndexOp>(loc, tile % nStages);
      Value keyBaseE = b.create<arith::ConstantIndexOp>(loc, tile * BN);
      b.create<nvgpu::DeviceAsyncWaitOp>(loc, prologTok,
                                         b.getI32IntegerAttr(nStages - 2 - j));
      b.create<gpu::BarrierOp>(loc);
      carries = computeTile(b, loc, stageR, keyBaseE, carries);
    }

    // Epilogue: O_global[row,col] = oAcc / l, straight from each compact
    // query group's independent register carry packet.
    int64_t carryWidth = nColPV + 4 + (normalizedSoftmaxCarry ? 2 : 0);
    for (int64_t queryGroup = 0; queryGroup < queryGroupsPerWarp;
         ++queryGroup) {
    int64_t carryBase = queryGroup * carryWidth;
    SmallVector<Value> finalAcc(carries.begin() + carryBase,
                                carries.begin() + carryBase + nColPV);
    Value mFin[2] = {carries[carryBase + nColPV],
                     carries[carryBase + nColPV + 1]};
    Value lFin[2] = {carries[carryBase + nColPV + 2],
                     carries[carryBase + nColPV + 3]};
    Value tmpPivotFin[2];
    if (normalizedSoftmaxCarry) {
      tmpPivotFin[0] = carries[carryBase + nColPV + 4];
      tmpPivotFin[1] = carries[carryBase + nColPV + 5];
    }
    Value finalScale[2], finalLse[2], tmpScale[2];
    for (int i = 0; i < 2; ++i) {
      if (certifiedFullAxis && !normalizedSoftmaxCarry) {
        Value exponent = b.create<arith::SubFOp>(loc, mFin[i], lFin[i]);
        finalScale[i] = b.create<math::ExpOp>(loc, exponent);
        finalLse[i] = lFin[i];
        tmpScale[i] = finalScale[i];
      } else {
        finalScale[i] = b.create<arith::DivFOp>(loc, oneF, lFin[i]);
        Value logMass = b.create<math::LogOp>(loc, lFin[i]);
        finalLse[i] = b.create<arith::AddFOp>(loc, mFin[i], logMass);
        tmpScale[i] = finalScale[i];
        if (normalizedSoftmaxCarry) {
          Value exponent =
              b.create<arith::SubFOp>(loc, tmpPivotFin[i], finalLse[i]);
          tmpScale[i] = b.create<math::ExpOp>(loc, exponent);
        }
      }
    }
    Value queryRowOffset = b.create<arith::ConstantIndexOp>(
        loc, queryGroup * numWarps * MMA_M);
    Value groupWarpRow = queryGroup == 0
                             ? warpRow16
                             : b.create<arith::AddIOp>(loc, warpRow16,
                                                      queryRowOffset)
                                   .getResult();
    Value warpRowGlobal = b.create<arith::AddIOp>(loc, rowBlk, groupWarpRow);

    // LSE = m + log(l) per query row (R-FLASH-3). All 4 lanes of a row's quad
    // hold the same m/l (quad-reduced), so only tig==0 (lane%4==0) writes each of
    // its 2 owned rows (groupID=lane/4, and groupID+8). Global row layout matches
    // the O epilogue (warpRowGlobal + laneDiv4 + i*8).
    if (writeLse) {
      Value isLead = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             laneMod4, c0);
      b.create<scf::IfOp>(loc, isLead, [&](OpBuilder &nb, Location nl) {
        for (int i = 0; i < 2; ++i) {
          Value ic = nb.create<arith::ConstantIndexOp>(nl, i * 8);
          Value row = nb.create<arith::AddIOp>(
              nl, warpRowGlobal, nb.create<arith::AddIOp>(nl, laneDiv4, ic));
          SmallVector<Value> indices = statIndices(row);
          nb.create<memref::StoreOp>(nl, finalLse[i], lseMem, indices);
          if (flatRank4) {
            nb.create<memref::StoreOp>(nl, tmpScale[i], tmpMem, indices);
          }
        }
        nb.create<scf::YieldOp>(nl);
      });
    }
    if (materializeTmpRoundtrip) {
      b.create<gpu::BarrierOp>(loc);
      for (int i = 0; i < 2; ++i) {
        Value ic = b.create<arith::ConstantIndexOp>(loc, i * 8);
        Value row = b.create<arith::AddIOp>(
            loc, warpRowGlobal, b.create<arith::AddIOp>(loc, laneDiv4, ic));
        finalScale[i] =
            b.create<memref::LoadOp>(loc, tmpMem, statIndices(row));
      }
    }
    for (int64_t jo = 0; jo < nColPV; ++jo) {
      Value nb8 = b.create<arith::ConstantIndexOp>(loc, jo * MMA_N);
      for (int i = 0; i < 2; ++i) {
        Value ic = b.create<arith::ConstantIndexOp>(loc, i * 8);
        Value row = b.create<arith::AddIOp>(
            loc, warpRowGlobal, b.create<arith::AddIOp>(loc, laneDiv4, ic));
        for (int i2 = 0; i2 < 2; ++i2) {
          Value jc = b.create<arith::ConstantIndexOp>(loc, i2);
          Value col = b.create<arith::AddIOp>(
              loc, nb8, b.create<arith::AddIOp>(loc, stCol, jc));
          Value e = b.create<vector::ExtractOp>(
              loc, finalAcc[jo], ArrayRef<int64_t>{(int64_t)i, (int64_t)i2});
          Value out = b.create<arith::MulFOp>(loc, e, finalScale[i]);
          if (flatRank4)
            out = b.create<arith::TruncFOp>(loc, f16, out);
          SmallVector<Value> indices = globalMatrixIndices(row, col);
          b.create<memref::StoreOp>(loc, out, oMem, indices);
        }
      }
    }
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);

    fn.erase();
    return success();
  }
};

struct ConvertIntentFlashMMAToGPUPass
    : ::intent::impl::ConvertIntentFlashMMAToGPUBase<
          ConvertIntentFlashMMAToGPUPass> {
  using ConvertIntentFlashMMAToGPUBase::ConvertIntentFlashMMAToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      FlashMMALowering probe(fn, (int)numStages, emitLse, causal,
                             certifiedFullAxisOnly, directVMatrixLoad,
                             directVRowPadding,
                             materializeTmpRoundtrip, pairedBMatrixLoads,
                             vectorizedQStaging, streamPFragments,
                             scoreFragmentColumns, hopperWarpgroup,
                             hopperHybridPV, normalizedSoftmaxCarry,
                             compactQueryWarpMapping, hopperRegisterQK,
                             hopperRegisterPV, hopperAsyncKV, blockM, blockN);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      FlashMMALowering low(fn, (int)numStages, emitLse, causal,
                           certifiedFullAxisOnly, directVMatrixLoad,
                           directVRowPadding,
                           materializeTmpRoundtrip, pairedBMatrixLoads,
                           vectorizedQStaging, streamPFragments,
                           scoreFragmentColumns, hopperWarpgroup,
                           hopperHybridPV, normalizedSoftmaxCarry,
                           compactQueryWarpMapping, hopperRegisterQK,
                           hopperRegisterPV, hopperAsyncKV, blockM, blockN);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("flash-mma lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
