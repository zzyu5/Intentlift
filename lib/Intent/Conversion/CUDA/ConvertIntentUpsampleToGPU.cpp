//===- ConvertIntentUpsampleToGPU.cpp - upsample -> gpu.launch ------------===//
//
// The C7 interpolation pass. Lowers a func whose body is exactly ONE
// `intent.upsample` over a rank-4 `[N,C,H,W]` f32 input to a single
// block-cooperative `gpu.launch`: grid = (N*C, OH, 1) (one block per output
// channel-plane row), block = (T,1,1) threads over the OW output columns
// (K = OW/T columns each, unrolled). Two modes:
//
// * "nearest" — pure DATA MOVEMENT (C4-layout-like): each output element's
//   SOURCE coord is an affine `floor` of the output coord
//   (ih = min(floor(oh*IH/OH), IH-1), matching flag_gems' truncation), followed
//   by a `memref.load` at (n,c,ih,iw). Bit-EXACT vs torch + flag_gems.
//
// * "bicubic" (align_corners=False, antialias=False) — a 4x4 cubic-convolution
//   neighborhood (a=-0.75, the torch/PIL kernel). The source position
//   r=(o+0.5)*IN/OUT-0.5 yields a floor base + fraction t; the four per-axis
//   weights (cubic_convolution1/2) scale the 16 clamped-neighbor loads, summed.
//   floor is computed in pure arith (no libdevice __nv_floorf). On the enlarging
//   path antialias is a no-op, so this matches upsample_bicubic2d_aa; rel<1e-3 vs
//   F.interpolate(mode="bicubic"). The variable-support aa DOWNSAMPLE is a TODO.
//
// Destination-passing: new signature = (input memref) ++ [out]. Static shapes,
// OW % T == 0.
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTUPSAMPLETOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

struct UpsampleLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  UpsampleOp up;
  RankedTensorType inTy, outTy;
  int64_t N = 0, C = 0, H = 0, W = 0, OH = 0, OW = 0;
  int64_t scaleH = 0, scaleW = 0;
  int64_t T = 0, G = 0, total = 0;
  bool bicubic = false;

  UpsampleLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret)
      return failure();
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    up = dyn_cast_or_null<UpsampleOp>(only);
    if (!up || (up.getMode() != "nearest" && up.getMode() != "bicubic"))
      return failure();
    bicubic = up.getMode() == "bicubic";
    inTy = cast<RankedTensorType>(up.getInput().getType());
    outTy = cast<RankedTensorType>(up.getResult().getType());
    auto scale = up.getScale();
    if (scale.size() != 2)
      return failure();
    auto scaleHAttr = dyn_cast<IntegerAttr>(scale[0]);
    auto scaleWAttr = dyn_cast<IntegerAttr>(scale[1]);
    if (!scaleHAttr || !scaleWAttr || scaleHAttr.getInt() < 1 ||
        scaleWAttr.getInt() < 1)
      return failure();
    scaleH = scaleHAttr.getInt();
    scaleW = scaleWAttr.getInt();
    if (inTy.getRank() != 4 || outTy.getRank() != 4 || !inTy.hasStaticShape() ||
        !outTy.hasStaticShape() || !inTy.getElementType().isF32())
      return failure();
    N = inTy.getDimSize(0);
    C = inTy.getDimSize(1);
    H = inTy.getDimSize(2);
    W = inTy.getDimSize(3);
    OH = outTy.getDimSize(2);
    OW = outTy.getDimSize(3);
    if (H > std::numeric_limits<int64_t>::max() / scaleH ||
        W > std::numeric_limits<int64_t>::max() / scaleW ||
        OH != H * scaleH || OW != W * scaleW)
      return failure();
    // Flat grid-stride over the whole output (see MapEmit.h): full-width blocks
    // + tail-mask, replacing the old one-block-per-(N*C,OH)-row / T-over-OW map.
    T = flatThreads(blockThreads);
    total = N * C * OH * OW;
    G = flatGridBlocks(total, T);
    return success();
  }

  // src = min( (i32)(o * (IN/OUT)), IN-1 ) as an index.
  Value srcCoord(OpBuilder &b, Location loc, Value oIdx, int64_t IN,
                 int64_t OUT) {
    Type f32 = b.getF32Type();
    Type i32 = b.getI32Type();
    Value oF = b.create<arith::UIToFPOp>(
        loc, f32, b.create<arith::IndexCastOp>(loc, i32, oIdx));
    Value rscale = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(f32, (double)IN / (double)OUT));
    Value sF = b.create<arith::MulFOp>(loc, oF, rscale);
    Value sI = b.create<arith::FPToSIOp>(loc, i32, sF);  // trunc toward 0 (>=0)
    Value cap = b.create<arith::ConstantIntOp>(loc, IN - 1, 32);
    sI = b.create<arith::MinSIOp>(loc, sI, cap);
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), sI);
  }

  //===------------------------------------------------------------------===//
  // Bicubic (align_corners=False, antialias=False) helpers. The source position
  // ``r = (o + 0.5)*IN/OUT - 0.5`` yields an integer base ``i = floor(r)`` and a
  // fractional ``t = r - i``; the four cubic-convolution weights (a = -0.75, the
  // torch/PIL kernel) are applied to the four taps ``i-1..i+2`` (edges clamped).
  // For UPSAMPLING (OUT >= IN) antialias is a no-op, so this matches
  // upsample_bicubic2d_aa exactly on the enlarging path.
  //===------------------------------------------------------------------===//
  Value fcst(OpBuilder &b, Location loc, double v) {
    return b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(v));
  }
  // cubic_convolution1: ((A+2)s - (A+3)) s^2 + 1, for |s| in [0,1].
  Value cubic1(OpBuilder &b, Location loc, Value s) {
    const double A = -0.75;
    Value t0 = b.create<arith::MulFOp>(loc, fcst(b, loc, A + 2.0), s);
    t0 = b.create<arith::SubFOp>(loc, t0, fcst(b, loc, A + 3.0));
    Value s2 = b.create<arith::MulFOp>(loc, s, s);
    t0 = b.create<arith::MulFOp>(loc, t0, s2);
    return b.create<arith::AddFOp>(loc, t0, fcst(b, loc, 1.0));
  }
  // cubic_convolution2: ((A s - 5A) s + 8A) s - 4A, for |s| in [1,2].
  Value cubic2(OpBuilder &b, Location loc, Value s) {
    const double A = -0.75;
    Value t0 = b.create<arith::MulFOp>(loc, fcst(b, loc, A), s);
    t0 = b.create<arith::SubFOp>(loc, t0, fcst(b, loc, 5.0 * A));
    t0 = b.create<arith::MulFOp>(loc, t0, s);
    t0 = b.create<arith::AddFOp>(loc, t0, fcst(b, loc, 8.0 * A));
    t0 = b.create<arith::MulFOp>(loc, t0, s);
    return b.create<arith::SubFOp>(loc, t0, fcst(b, loc, 4.0 * A));
  }
  // Base tap index i = floor(r) and fraction t = r - i for an output coord `o`.
  void bicubicPos(OpBuilder &b, Location loc, Value oIdx, int64_t IN,
                  int64_t OUT, Value &iBase, Value &t) {
    Type f32 = b.getF32Type();
    Type i32 = b.getI32Type();
    Value oF = b.create<arith::UIToFPOp>(
        loc, f32, b.create<arith::IndexCastOp>(loc, i32, oIdx));
    Value scale = fcst(b, loc, (double)IN / (double)OUT);
    // r = (o + 0.5)*scale - 0.5
    Value r = b.create<arith::AddFOp>(loc, oF, fcst(b, loc, 0.5));
    r = b.create<arith::MulFOp>(loc, r, scale);
    r = b.create<arith::SubFOp>(loc, r, fcst(b, loc, 0.5));
    // floor(r) in pure arith (no libdevice __nv_floorf): trunc toward zero, then
    // step back one when the truncation overshot a negative r.
    Value ti = b.create<arith::FPToSIOp>(loc, i32, r);
    Value tf = b.create<arith::SIToFPOp>(loc, b.getF32Type(), ti);
    Value over = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, tf, r);
    Value tiMinus =
        b.create<arith::SubIOp>(loc, ti, b.create<arith::ConstantIntOp>(loc, 1, 32));
    Value flMinus = b.create<arith::SubFOp>(loc, tf, fcst(b, loc, 1.0));
    iBase = b.create<arith::SelectOp>(loc, over, tiMinus, ti);
    Value fl = b.create<arith::SelectOp>(loc, over, flMinus, tf);
    t = b.create<arith::SubFOp>(loc, r, fl);
  }
  // Four cubic weights for fraction t at taps i-1, i, i+1, i+2.
  void bicubicWeights(OpBuilder &b, Location loc, Value t, Value w[4]) {
    Value one = fcst(b, loc, 1.0), two = fcst(b, loc, 2.0);
    w[0] = cubic2(b, loc, b.create<arith::AddFOp>(loc, t, one));      // s = t+1
    w[1] = cubic1(b, loc, t);                                         // s = t
    w[2] = cubic1(b, loc, b.create<arith::SubFOp>(loc, one, t));      // s = 1-t
    w[3] = cubic2(b, loc, b.create<arith::SubFOp>(loc, two, t));      // s = 2-t
  }
  // clamp(base + off, 0, IN-1) as an index.
  Value clampIdx(OpBuilder &b, Location loc, Value baseI32, int off, int64_t IN) {
    Value v = b.create<arith::AddIOp>(
        loc, baseI32, b.create<arith::ConstantIntOp>(loc, off, 32));
    v = b.create<arith::MaxSIOp>(loc, v, b.create<arith::ConstantIntOp>(loc, 0, 32));
    v = b.create<arith::MinSIOp>(loc, v,
                                 b.create<arith::ConstantIntOp>(loc, IN - 1, 32));
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);

    SmallVector<Type> inTys{memrefOf(inTy), memrefOf(outTy)};
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    Value inMem = entry->getArgument(0);
    Value outMem = entry->getArgument(1);

    Value cG = b.create<arith::ConstantIndexOp>(loc, G);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cG, c1, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;

    // Flat grid-stride over every output cell; unravel into (n, c, oh, ow).
    Value iv;
    auto gsFor = emitGridStride(b, loc, bx, tx, T, G, total, iv);
    SmallVector<Value> co;
    unravelIndex(b, loc, iv, {N, C, OH, OW}, co);
    Value nIdx = co[0], cIdx = co[1], by = co[2], ow = co[3];

    if (!bicubic) {
      Value ihIdx = srcCoord(b, loc, by, H, OH);
      Value iwIdx = srcCoord(b, loc, ow, W, OW);
      Value v = b.create<memref::LoadOp>(loc, inMem,
                                         ValueRange{nIdx, cIdx, ihIdx, iwIdx});
      b.create<memref::StoreOp>(loc, v, outMem, ValueRange{nIdx, cIdx, by, ow});
    } else {
      // Bicubic: sum_{p,q} wy[p]*wx[q]*in[n,c, clamp(yBase-1+p), clamp(xBase-1+q)].
      Value yBase, ty, wy[4];
      bicubicPos(b, loc, by, H, OH, yBase, ty);
      bicubicWeights(b, loc, ty, wy);
      Value xBase, tx2, wx[4];
      bicubicPos(b, loc, ow, W, OW, xBase, tx2);
      bicubicWeights(b, loc, tx2, wx);
      Value acc = fcst(b, loc, 0.0);
      for (int p = 0; p < 4; ++p) {
        Value yy = clampIdx(b, loc, yBase, p - 1, H);
        Value rowAcc = fcst(b, loc, 0.0);
        for (int q = 0; q < 4; ++q) {
          Value xx = clampIdx(b, loc, xBase, q - 1, W);
          Value v = b.create<memref::LoadOp>(loc, inMem,
                                             ValueRange{nIdx, cIdx, yy, xx});
          Value wv = b.create<arith::MulFOp>(loc, v, wx[q]);
          rowAcc = b.create<arith::AddFOp>(loc, rowAcc, wv);
        }
        acc = b.create<arith::AddFOp>(
            loc, acc, b.create<arith::MulFOp>(loc, rowAcc, wy[p]));
      }
      b.create<memref::StoreOp>(loc, acc, outMem,
                                ValueRange{nIdx, cIdx, by, ow});
    }

    b.setInsertionPointAfter(gsFor);
    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentUpsampleToGPUPass
    : ::intent::impl::ConvertIntentUpsampleToGPUBase<
          ConvertIntentUpsampleToGPUPass> {
  using ConvertIntentUpsampleToGPUBase::ConvertIntentUpsampleToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      UpsampleLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      UpsampleLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("upsample lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
