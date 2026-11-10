//===- LowerGpuMathToNVVM.cpp - math transcendentals -> NVVM intrinsics ---===//
//
// Backend codegen peephole for the NVIDIA target, run on the outlined device
// module just before `--gpu-lower-to-nvvm-pipeline`.
//
// The stock GpuToNVVM lowering turns `math.exp` into a `__nv_expf` libdevice
// call. On system MLIR 20.1.8 that path is unusable for us: without libdevice
// on the search path ptxas fails with "Unresolved extern function '__nv_expf'";
// with libdevice linked, the integrated serializer internalizes + DCEs the
// kernel entry and emits a cubin with no `.text` section (driver load rc 200).
//
// We sidestep libdevice entirely by computing exp with the native hardware
// instruction: exp(x) = 2^(x * log2(e)), where the 2^ is the PTX
// `ex2.approx.ftz.f32` op, emitted here as the `llvm.nvvm.ex2.approx.ftz.f`
// LLVM intrinsic. Verified on an RTX 5090 D (sm_120a): max rel err ~1.7e-7 vs
// torch.exp, well inside the softmax tolerance flash attention needs.
//
//===----------------------------------------------------------------------===//

#include "Target/CUDA/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace intent {
#define GEN_PASS_DEF_LOWERGPUMATHTONVVM
#include "Target/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// log2(e) as an f32 — the change-of-base constant for exp(x) = 2^(x*log2e).
static constexpr double kLog2E = 1.4426950408889634;
/// ln(2) as an f32 — the change-of-base constant for log(x) = log2(x)*ln2.
static constexpr double kLn2 = 0.6931471805599453;

static bool isHalfStorageType(Type type) {
  return isa<Float16Type, BFloat16Type>(type);
}

/// Promote one scalar f16/bf16 math operation to the backend's f32 compute
/// domain, then round back to its declared storage type.  Greedy rewriting
/// subsequently applies the existing libdevice-free f32 pattern to the newly
/// created math op.
template <typename MathOp>
struct PromoteHalfUnaryMath : OpRewritePattern<MathOp> {
  using OpRewritePattern<MathOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {
    Type storageType = op.getOperand().getType();
    if (!isHalfStorageType(storageType))
      return rewriter.notifyMatchFailure(op, "operand is not f16/bf16");
    Type f32 = rewriter.getF32Type();
    Value widened =
        rewriter.create<arith::ExtFOp>(op.getLoc(), f32, op.getOperand());
    Value computed = rewriter.create<MathOp>(op.getLoc(), widened);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(
        op, storageType, computed);
    return success();
  }
};

template <typename MathOp>
struct PromoteHalfBinaryMath : OpRewritePattern<MathOp> {
  using OpRewritePattern<MathOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {
    Type storageType = op.getLhs().getType();
    if (!isHalfStorageType(storageType) || op.getRhs().getType() != storageType)
      return rewriter.notifyMatchFailure(
          op, "operands are not matching f16/bf16 scalars");
    Type f32 = rewriter.getF32Type();
    Value lhs = rewriter.create<arith::ExtFOp>(op.getLoc(), f32, op.getLhs());
    Value rhs = rewriter.create<arith::ExtFOp>(op.getLoc(), f32, op.getRhs());
    Value computed = rewriter.create<MathOp>(op.getLoc(), lhs, rhs);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(
        op, storageType, computed);
    return success();
  }
};

struct PromoteHalfFma : OpRewritePattern<math::FmaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const override {
    Type storageType = op.getA().getType();
    if (!isHalfStorageType(storageType) || op.getB().getType() != storageType ||
        op.getC().getType() != storageType)
      return rewriter.notifyMatchFailure(
          op, "operands are not matching f16/bf16 scalars");
    Type f32 = rewriter.getF32Type();
    Value a = rewriter.create<arith::ExtFOp>(op.getLoc(), f32, op.getA());
    Value b = rewriter.create<arith::ExtFOp>(op.getLoc(), f32, op.getB());
    Value c = rewriter.create<arith::ExtFOp>(op.getLoc(), f32, op.getC());
    Value computed = rewriter.create<math::FmaOp>(op.getLoc(), a, b, c);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(
        op, storageType, computed);
    return success();
  }
};

/// `math.log(%x) - %y` -> `fma(lg2.approx(%x), ln2, -%y)`.
///
/// The generic log rewrite would materialize `lg2 * ln2` followed by a
/// subtraction.  When the log has exactly this single consumer, one native FMA
/// preserves the same approximate-log policy with one fewer dependent FP32
/// instruction.  The match is purely SSA-structural and applies equally to any
/// row loss/divergence graph.
struct LogSubToLg2Fma : OpRewritePattern<arith::SubFOp> {
  explicit LogSubToLg2Fma(MLIRContext *ctx)
      : OpRewritePattern<arith::SubFOp>(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(arith::SubFOp op,
                                PatternRewriter &rewriter) const override {
    auto log = op.getLhs().getDefiningOp<math::LogOp>();
    auto f32 = dyn_cast<Float32Type>(op.getType());
    if (!log || !log->hasOneUse() || !f32 || op.getRhs().getType() != f32)
      return rewriter.notifyMatchFailure(
          op, "requires a single-use scalar f32 log minuend");

    Location loc = op.getLoc();
    auto lg2 = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.nvvm.lg2.approx.f"),
        ValueRange{log.getOperand()});
    Value ln2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(kLn2));
    Value negativeRhs = rewriter.create<arith::NegFOp>(loc, op.getRhs());
    auto fused = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.fma"),
        ValueRange{lg2.getResult(0), ln2, negativeRhs});
    rewriter.replaceOp(op, fused.getResults());
    rewriter.eraseOp(log);
    return success();
  }
};

/// `math.exp %x : f32`  ->  `ex2.approx.ftz.f32 (%x * log2e)`.
struct ExpToEx2Approx : OpRewritePattern<math::ExpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const override {
    // ex2.approx.ftz.f is a scalar f32 instruction; only handle that form.
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");

    Location loc = op.getLoc();
    Value log2e = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(kLog2E));
    Value scaled =
        rewriter.create<arith::MulFOp>(loc, op.getOperand(), log2e);
    auto approx = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.nvvm.ex2.approx.ftz.f"),
        ValueRange{scaled});
    rewriter.replaceOp(op, approx.getResults());
    return success();
  }
};

static bool isF32One(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto attr = dyn_cast<FloatAttr>(constant.getValue());
  return attr && attr.getType().isF32() && attr.getValueAsDouble() == 1.0;
}

/// Explicit precision-search state: `lhs / rhs` ->
/// `lhs * rcp.approx.ftz(rhs)`.  Only the source-neutral row lowering can attach
/// the marker after candidate enumeration; ordinary arith.divf operations and
/// the exact ablation state remain untouched.
struct MarkedDivToNVVMApprox : OpRewritePattern<arith::DivFOp> {
  explicit MarkedDivToNVVMApprox(MLIRContext *ctx)
      : OpRewritePattern<arith::DivFOp>(ctx, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getType());
    if (!f32 || !op->hasAttr("intent.cuda.approximate_div"))
      return rewriter.notifyMatchFailure(
          op, "requires a marked scalar f32 precision-search division");
    auto reciprocal = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32,
        rewriter.getStringAttr("llvm.nvvm.rcp.approx.ftz.f"),
        ValueRange{op.getRhs()});
    rewriter.replaceOpWithNewOp<arith::MulFOp>(
        op, op.getLhs(), reciprocal.getResult(0));
    return success();
  }
};

/// `1 / (1 + ex2.approx(x))` -> `rcp.approx.ftz(1 + ex2.approx(x))`.
///
/// This is the normalized scalar shape of sigmoid after ExpToEx2Approx. The
/// matcher is deliberately structural and narrow: it does not inspect function
/// names, and it cannot rewrite unrelated reciprocal scales whose denominator
/// is not the native approximate exponential plus one.
struct SigmoidReciprocalToNVVMApprox : OpRewritePattern<arith::DivFOp> {
  explicit SigmoidReciprocalToNVVMApprox(MLIRContext *ctx)
      : OpRewritePattern<arith::DivFOp>(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getType());
    auto add = op.getRhs().getDefiningOp<arith::AddFOp>();
    if (!f32 || !isF32One(op.getLhs()) || !add)
      return rewriter.notifyMatchFailure(
          op, "requires scalar f32 one divided by an addition");
    Value exponential;
    if (isF32One(add.getLhs()))
      exponential = add.getRhs();
    else if (isF32One(add.getRhs()))
      exponential = add.getLhs();
    else
      return rewriter.notifyMatchFailure(op, "denominator has no f32 one");
    auto intrinsic = exponential.getDefiningOp<LLVM::CallIntrinsicOp>();
    if (!intrinsic ||
        intrinsic.getIntrin() != "llvm.nvvm.ex2.approx.ftz.f")
      return rewriter.notifyMatchFailure(
          op, "denominator is not one plus native approximate exp2");
    auto reciprocal = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32,
        rewriter.getStringAttr("llvm.nvvm.rcp.approx.ftz.f"),
        ValueRange{op.getRhs()});
    rewriter.replaceOp(op, reciprocal.getResults());
    return success();
  }
};

/// `math.log %x : f32`  ->  `lg2.approx.f32(%x) * ln2`. The stock GpuToNVVM
/// lowering routes `math.log` to a `__nv_logf` libdevice extern (verified on
/// this path: ptxas dead end / DCE'd kernel, same failure mode as exp/sqrt).
/// `lg2.approx.f32` is the native PTX hardware op, emitted here as the
/// `llvm.nvvm.lg2.approx.f` intrinsic; log(x) = log2(x)*ln2 is the change of
/// base. kl_div/jsd's loss is a per-element log on log-probabilities, so the
/// approx rel err (~2^-23-class mantissa) is far inside the 1e-3 PK tolerance.
struct LogToLg2Approx : OpRewritePattern<math::LogOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    Location loc = op.getLoc();
    auto lg2 = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.nvvm.lg2.approx.f"),
        ValueRange{op.getOperand()});
    Value ln2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(kLn2));
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, lg2.getResult(0), ln2);
    return success();
  }
};

/// `math.log2 %x : f32` is already in the native PTX logarithm base.
struct Log2ToLg2Approx : OpRewritePattern<math::Log2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Log2Op op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    auto lg2 = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32, rewriter.getStringAttr("llvm.nvvm.lg2.approx.f"),
        ValueRange{op.getOperand()});
    rewriter.replaceOp(op, lg2.getResults());
    return success();
  }
};

/// `math.log %x : f64` -> a self-contained range-reduced f64 polynomial.
///
/// Standalone ptxas cannot resolve the stock `__nv_log` libdevice call.  Unlike
/// f32, PTX has no native f64 lg2 instruction, so use IEEE-754 exponent/mantissa
/// decomposition and the atanh series
///
///   log(m) = 2 * (z + z^3/3 + ... + z^17/17), z = (m-1)/(m+1),
///
/// after reducing m to [1/sqrt(2), sqrt(2)].  On that interval |z|<=0.1716 and
/// nine terms are comfortably below double precision needs for the paper's f32
/// state promoted to an f64 LSE destination.  The implementation also preserves
/// zero, negative, infinity and NaN behavior without any external symbol.
struct LogF64Polynomial : OpRewritePattern<math::LogOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const override {
    auto f64 = dyn_cast<Float64Type>(op.getOperand().getType());
    if (!f64)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f64");

    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    auto cF64 = [&](double value) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF64FloatAttr(value));
    };
    auto cI64 = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, value, i64);
    };

    Value x = op.getOperand();
    Value zero = cF64(0.0);
    Value one = cF64(1.0);
    Value two = cF64(2.0);
    Value isZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, x, zero);
    Value isNegative = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, x, zero);

    // Normalize positive subnormals by 2^54, then compensate the exponent.
    Value originalBits = rewriter.create<arith::BitcastOp>(loc, i64, x);
    Value shift52 = cI64(52);
    Value exponentMask = cI64(0x7ff);
    Value originalExponent = rewriter.create<arith::AndIOp>(
        loc,
        rewriter.create<arith::ShRUIOp>(loc, originalBits, shift52),
        exponentMask);
    Value exponentZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, originalExponent, cI64(0));
    Value positive = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, x, zero);
    Value isSubnormal = rewriter.create<arith::AndIOp>(loc, exponentZero,
                                                       positive);
    Value scaledSubnormal = rewriter.create<arith::MulFOp>(
        loc, x, cF64(18014398509481984.0)); // 2^54
    Value normalized = rewriter.create<arith::SelectOp>(
        loc, isSubnormal, scaledSubnormal, x);

    Value bits = rewriter.create<arith::BitcastOp>(loc, i64, normalized);
    Value exponentBits = rewriter.create<arith::AndIOp>(
        loc, rewriter.create<arith::ShRUIOp>(loc, bits, shift52),
        exponentMask);
    Value isNonFinite = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, exponentBits, cI64(0x7ff));
    Value exponent = rewriter.create<arith::SubIOp>(loc, exponentBits,
                                                    cI64(1023));
    Value exponentSubnormal = rewriter.create<arith::SubIOp>(
        loc, exponent, cI64(54));
    exponent = rewriter.create<arith::SelectOp>(
        loc, isSubnormal, exponentSubnormal, exponent);

    constexpr uint64_t kMantissaMask = (uint64_t{1} << 52) - 1;
    constexpr uint64_t kOneExponent = uint64_t{1023} << 52;
    Value mantissaBits = rewriter.create<arith::OrIOp>(
        loc,
        rewriter.create<arith::AndIOp>(
            loc, bits, cI64(static_cast<int64_t>(kMantissaMask))),
        cI64(static_cast<int64_t>(kOneExponent)));
    Value mantissa = rewriter.create<arith::BitcastOp>(loc, f64, mantissaBits);
    Value aboveSqrt2 = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, mantissa,
        cF64(1.4142135623730950488));
    Value reducedMantissa = rewriter.create<arith::SelectOp>(
        loc, aboveSqrt2,
        rewriter.create<arith::MulFOp>(loc, mantissa, cF64(0.5)), mantissa);
    exponent = rewriter.create<arith::SelectOp>(
        loc, aboveSqrt2,
        rewriter.create<arith::AddIOp>(loc, exponent, cI64(1)), exponent);

    Value numerator = rewriter.create<arith::SubFOp>(loc, reducedMantissa, one);
    Value denominator = rewriter.create<arith::AddFOp>(loc, reducedMantissa, one);
    Value z = rewriter.create<arith::DivFOp>(loc, numerator, denominator);
    Value z2 = rewriter.create<arith::MulFOp>(loc, z, z);
    Value power = z;
    Value series = z;
    for (int denominatorValue = 3; denominatorValue <= 17;
         denominatorValue += 2) {
      power = rewriter.create<arith::MulFOp>(loc, power, z2);
      Value term = rewriter.create<arith::DivFOp>(
          loc, power, cF64(static_cast<double>(denominatorValue)));
      series = rewriter.create<arith::AddFOp>(loc, series, term);
    }
    Value logMantissa = rewriter.create<arith::MulFOp>(loc, two, series);
    Value exponentF64 =
        rewriter.create<arith::SIToFPOp>(loc, f64, exponent);
    Value exponentTerm = rewriter.create<arith::MulFOp>(
        loc, exponentF64, cF64(kLn2));
    Value approximate = rewriter.create<arith::AddFOp>(
        loc, exponentTerm, logMantissa);

    Value positiveOrSpecial = rewriter.create<arith::SelectOp>(
        loc, isNonFinite, x, approximate);
    Value negativeInfinity = cF64(-std::numeric_limits<double>::infinity());
    Value nan = cF64(std::numeric_limits<double>::quiet_NaN());
    Value zeroHandled = rewriter.create<arith::SelectOp>(
        loc, isZero, negativeInfinity, positiveOrSpecial);
    Value result = rewriter.create<arith::SelectOp>(
        loc, isNegative, nan, zeroHandled);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// `math.OP %x : f32` -> `llvm.call_intrinsic "<intr>"(%x)` for the NVVM
/// approximate transcendentals. Same libdevice-dead-end fix as exp: the stock
/// GpuToNVVM lowering routes `math.sqrt`/`math.rsqrt` to `__nv_sqrtf`/
/// `__nv_rsqrtf` externs, which ptxas cannot resolve (or the integrated
/// serializer DCEs the kernel). The `*.approx.f` PTX ops compute it in hardware;
/// measured rel vs fp32-ref is ~1.3e-7 (rms_norm's rstd is one value/row, error
/// negligible) — the same class of fix, applied to the row-reduction normalizer.
template <typename MathOp>
struct UnaryToNVVMApprox : OpRewritePattern<MathOp> {
  UnaryToNVVMApprox(MLIRContext *ctx, StringRef intrinsic)
      : OpRewritePattern<MathOp>(ctx), intrinsic(intrinsic) {}

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    auto approx = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32, rewriter.getStringAttr(intrinsic),
        ValueRange{op.getOperand()});
    rewriter.replaceOp(op, approx.getResults());
    return success();
  }

  StringRef intrinsic;
};

/// Expand the less common scalar math operations into the already supported
/// native-ex2/log2/atan paths.  This keeps the CUDA elementwise lowering
/// independent of unresolved libdevice symbols while preserving the declared
/// f32 result type.
struct ExpM1ToExpSubOne : OpRewritePattern<math::ExpM1Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpM1Op op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    Value exponential = rewriter.create<math::ExpOp>(op.getLoc(), op.getOperand());
    Value one = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(1.0));
    rewriter.replaceOpWithNewOp<arith::SubFOp>(op, exponential, one);
    return success();
  }
};

struct Log1pToLogOnePlus : OpRewritePattern<math::Log1pOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Log1pOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    Value one = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(1.0));
    Value sum = rewriter.create<arith::AddFOp>(op.getLoc(), one, op.getOperand());
    rewriter.replaceOpWithNewOp<math::LogOp>(op, sum);
    return success();
  }
};

struct Log10ToLogScale : OpRewritePattern<math::Log10Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Log10Op op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    Value log = rewriter.create<math::LogOp>(op.getLoc(), op.getOperand());
    Value inverseLn10 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(0.4342944819032518));
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, log, inverseLn10);
    return success();
  }
};

static Value emitAsinApprox(PatternRewriter &rewriter, Location loc, Value x) {
  Value one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF32FloatAttr(1.0));
  Value square = rewriter.create<arith::MulFOp>(loc, x, x);
  Value underRoot = rewriter.create<arith::SubFOp>(loc, one, square);
  Value root = rewriter.create<math::SqrtOp>(loc, underRoot);
  Value ratio = rewriter.create<arith::DivFOp>(loc, x, root);
  return rewriter.create<math::AtanOp>(loc, ratio);
}

struct AsinToAtanApprox : OpRewritePattern<math::AsinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::AsinOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    rewriter.replaceOp(op, emitAsinApprox(rewriter, op.getLoc(), op.getOperand()));
    return success();
  }
};

struct AcosToAtanApprox : OpRewritePattern<math::AcosOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::AcosOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    Value asin = emitAsinApprox(rewriter, op.getLoc(), op.getOperand());
    Value halfPi = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(1.5707963267948966));
    rewriter.replaceOpWithNewOp<arith::SubFOp>(op, halfPi, asin);
    return success();
  }
};

struct CbrtToExpLog : OpRewritePattern<math::CbrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::CbrtOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    Location loc = op.getLoc();
    Value magnitude = rewriter.create<math::AbsFOp>(loc, op.getOperand());
    Value logarithm = rewriter.create<math::LogOp>(loc, magnitude);
    Value oneThird = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(1.0 / 3.0));
    Value scaled = rewriter.create<arith::MulFOp>(loc, logarithm, oneThird);
    Value root = rewriter.create<math::ExpOp>(loc, scaled);
    rewriter.replaceOpWithNewOp<math::CopySignOp>(op, root, op.getOperand());
    return success();
  }
};

template <typename MathOp>
struct UnaryToLLVMIntrinsic : OpRewritePattern<MathOp> {
  UnaryToLLVMIntrinsic(MLIRContext *ctx, StringRef intrinsic)
      : OpRewritePattern<MathOp>(ctx), intrinsic(intrinsic) {}

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    auto lowered = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32, rewriter.getStringAttr(intrinsic),
        ValueRange{op.getOperand()});
    rewriter.replaceOp(op, lowered.getResults());
    return success();
  }

  StringRef intrinsic;
};

struct CopySignToLLVMIntrinsic : OpRewritePattern<math::CopySignOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::CopySignOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getLhs().getType());
    if (!f32 || op.getRhs().getType() != f32)
      return rewriter.notifyMatchFailure(op, "operands are not scalar f32");
    auto lowered = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32, rewriter.getStringAttr("llvm.copysign"),
        ValueRange{op.getLhs(), op.getRhs()});
    rewriter.replaceOp(op, lowered.getResults());
    return success();
  }
};

/// `math.exp2 %x : f32` -> `ex2.approx.ftz.f32(%x)`. exp2 IS the native 2^x
/// hardware op with NO change-of-base prescale (unlike exp, which multiplies by
/// log2(e) first). The stock GpuToNVVM path routes math.exp2 to a `__nv_exp2f`
/// libdevice extern (ptxas dead end, same failure mode as exp/log/sqrt).
struct Exp2ToEx2Approx : OpRewritePattern<math::Exp2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Exp2Op op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    auto approx = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32,
        rewriter.getStringAttr("llvm.nvvm.ex2.approx.ftz.f"),
        ValueRange{op.getOperand()});
    rewriter.replaceOp(op, approx.getResults());
    return success();
  }
};

/// `math.tanh %x : f32` -> a stable native-ex2/reciprocal approximation.
///
/// The stock GPU lowering emits an unresolved `__nv_tanhf` libdevice call.
/// Use
///
///   tanh(x) = sign(x) * (1 - exp(-2 * abs(x))) /
///                         (1 + exp(-2 * abs(x)))
///
/// so the exponential argument is always non-positive.  This avoids overflow
/// for large negative inputs while mapping entirely to arithmetic plus the
/// native `ex2.approx.ftz.f32` and `rcp.approx.ftz.f32` instructions already
/// used by the exp/sigmoid fast paths.
struct TanhToEx2Approx : OpRewritePattern<math::TanhOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::TanhOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");

    Location loc = op.getLoc();
    if (auto mode = op->getAttrOfType<StringAttr>(
            "intent.backend.tanh_approximation");
        mode && mode.getValue() == "native-ptx") {
      // PTX ISA provides a single-instruction f32 approximation on current
      // CUDA targets, but LLVM/NVVM does not expose a stable intrinsic for it.
      // Keep it as target inline assembly rather than replacing the kernel in
      // Python.  This state is explicitly marked by the upstream C++ lowering
      // and is admitted only through Q/D-certified numeric search.
      auto native = rewriter.create<LLVM::InlineAsmOp>(
          loc, f32, ValueRange{op.getOperand()},
          "tanh.approx.f32 $0, $1;", "=f,f",
          /*has_side_effects=*/false, /*is_align_stack=*/false,
          LLVM::AsmDialectAttr(), ArrayAttr());
      rewriter.replaceOp(op, native.getResults());
      return success();
    }
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(0.0));
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(1.0));
    Value negative = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, op.getOperand(), zero);
    Value negated = rewriter.create<arith::NegFOp>(loc, op.getOperand());
    Value magnitude = rewriter.create<arith::SelectOp>(
        loc, negative, negated, op.getOperand());
    Value scale = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(-2.0 * kLog2E));
    Value exponent = rewriter.create<arith::MulFOp>(loc, magnitude, scale);
    auto exponential = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.nvvm.ex2.approx.ftz.f"),
        ValueRange{exponent});
    Value numerator = rewriter.create<arith::SubFOp>(
        loc, one, exponential.getResult(0));
    Value denominator = rewriter.create<arith::AddFOp>(
        loc, one, exponential.getResult(0));
    auto reciprocal = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.nvvm.rcp.approx.ftz.f"),
        ValueRange{denominator});
    Value magnitudeResult = rewriter.create<arith::MulFOp>(
        loc, numerator, reciprocal.getResult(0));
    Value negativeResult =
        rewriter.create<arith::NegFOp>(loc, magnitudeResult);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, negative, negativeResult, magnitudeResult);
    return success();
  }
};

/// `math.absf %x : f32` -> LLVM fabs.  The stock NVIDIA conversion currently
/// materializes `__nv_fabsf`, even though PTX has a native absolute-value
/// modifier and LLVM already models it without libdevice.
struct AbsFToLLVMIntrinsic : OpRewritePattern<math::AbsFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::AbsFOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    auto absolute = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32, rewriter.getStringAttr("llvm.fabs"),
        ValueRange{op.getOperand()});
    rewriter.replaceOp(op, absolute.getResults());
    return success();
  }
};

/// Return the IEEE-754 sign bit, including for negative zero.  Floating-point
/// ordered comparisons deliberately treat `-0.0 < 0.0` as false, so atan2 and
/// other sign-sensitive functions need to inspect the bit representation.
static Value emitF32SignBit(PatternRewriter &rewriter, Location loc, Value x) {
  Type i32 = rewriter.getI32Type();
  Value bits = rewriter.create<arith::BitcastOp>(loc, i32, x);
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  return rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, bits, zero);
}

/// Emit atan(x) without libdevice.  Two exact range-reduction identities keep
/// the Taylor argument in [-tan(pi/8), tan(pi/8)], where the degree-17 odd
/// series has an error below 3e-9 before f32 rounding.
static Value emitAtanApprox(PatternRewriter &rewriter, Location loc, Value x) {
  auto c = [&](double value) -> Value {
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(value));
  };
  Value zero = c(0.0);
  Value one = c(1.0);
  Value negative = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, x, zero);
  Value negated = rewriter.create<arith::NegFOp>(loc, x);
  Value magnitude =
      rewriter.create<arith::SelectOp>(loc, negative, negated, x);
  Value reciprocal = rewriter.create<arith::DivFOp>(loc, one, magnitude);
  Value aboveOne = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, magnitude, one);
  Value z = rewriter.create<arith::SelectOp>(
      loc, aboveOne, reciprocal, magnitude);

  Value tanPiOver8 = c(0.4142135623730950488);
  Value useQuarterTurn = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, z, tanPiOver8);
  Value shiftedNumerator = rewriter.create<arith::SubFOp>(loc, z, one);
  Value shiftedDenominator = rewriter.create<arith::AddFOp>(loc, z, one);
  Value shifted = rewriter.create<arith::DivFOp>(
      loc, shiftedNumerator, shiftedDenominator);
  Value u = rewriter.create<arith::SelectOp>(
      loc, useQuarterTurn, shifted, z);
  Value u2 = rewriter.create<arith::MulFOp>(loc, u, u);

  constexpr double coefficients[] = {
      1.0 / 17.0, -1.0 / 15.0, 1.0 / 13.0, -1.0 / 11.0,
      1.0 / 9.0,  -1.0 / 7.0,  1.0 / 5.0,  -1.0 / 3.0,
      1.0,
  };
  Value polynomial = c(coefficients[0]);
  for (double coefficient : llvm::ArrayRef(coefficients).drop_front()) {
    polynomial = rewriter.create<arith::AddFOp>(
        loc, rewriter.create<arith::MulFOp>(loc, polynomial, u2),
        c(coefficient));
  }
  Value angle = rewriter.create<arith::MulFOp>(loc, u, polynomial);
  Value quarterTurn = rewriter.create<arith::SelectOp>(
      loc, useQuarterTurn, c(0.7853981633974483096), zero);
  angle = rewriter.create<arith::AddFOp>(loc, quarterTurn, angle);
  Value complement = rewriter.create<arith::SubFOp>(
      loc, c(1.5707963267948966192), angle);
  angle = rewriter.create<arith::SelectOp>(
      loc, aboveOne, complement, angle);
  Value negativeAngle = rewriter.create<arith::NegFOp>(loc, angle);
  Value signedAngle = rewriter.create<arith::SelectOp>(
      loc, negative, negativeAngle, angle);
  Value isZero = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OEQ, x, zero);
  return rewriter.create<arith::SelectOp>(loc, isZero, x, signedAngle);
}

struct AtanToPolynomial : OpRewritePattern<math::AtanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::AtanOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<Float32Type>(op.getOperand().getType()))
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    rewriter.replaceOp(op,
                       emitAtanApprox(rewriter, op.getLoc(), op.getOperand()));
    return success();
  }
};

/// f32 atan2 built from the self-contained atan approximation plus explicit
/// IEEE-754 quadrant, signed-zero, infinity, and NaN handling.
struct Atan2ToPolynomial : OpRewritePattern<math::Atan2Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Atan2Op op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getLhs().getType());
    if (!f32 || op.getRhs().getType() != f32)
      return rewriter.notifyMatchFailure(op, "operands are not scalar f32");
    Location loc = op.getLoc();
    auto c = [&](double value) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(value));
    };
    Value y = op.getLhs();
    Value x = op.getRhs();
    Value zero = c(0.0);
    Value ySign = emitF32SignBit(rewriter, loc, y);
    Value xSign = emitF32SignBit(rewriter, loc, x);
    Value ratio = rewriter.create<arith::DivFOp>(loc, y, x);
    Value angle = emitAtanApprox(rewriter, loc, ratio);
    Value pi = c(3.1415926535897932385);
    Value signedPi = rewriter.create<arith::SelectOp>(
        loc, ySign, rewriter.create<arith::NegFOp>(loc, pi), pi);
    Value quadrantAdjusted = rewriter.create<arith::AddFOp>(loc, angle, signedPi);
    angle = rewriter.create<arith::SelectOp>(
        loc, xSign, quadrantAdjusted, angle);

    Value xZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, x, zero);
    Value yZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, y, zero);
    Value halfPi = c(1.5707963267948966192);
    Value vertical = rewriter.create<arith::SelectOp>(
        loc, ySign, rewriter.create<arith::NegFOp>(loc, halfPi), halfPi);
    Value zeroPair = rewriter.create<arith::SelectOp>(
        loc, xSign, signedPi, y);
    Value xZeroResult = rewriter.create<arith::SelectOp>(
        loc, yZero, zeroPair, vertical);

    Value positiveInfinity = c(std::numeric_limits<float>::infinity());
    Value negativeInfinity = c(-std::numeric_limits<float>::infinity());
    auto isInfinity = [&](Value value) -> Value {
      Value positive = rewriter.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OEQ, value, positiveInfinity);
      Value negative = rewriter.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OEQ, value, negativeInfinity);
      return rewriter.create<arith::OrIOp>(loc, positive, negative);
    };
    Value bothInfinite = rewriter.create<arith::AndIOp>(
        loc, isInfinity(x), isInfinity(y));
    Value diagonalMagnitude = rewriter.create<arith::SelectOp>(
        loc, xSign, c(2.3561944901923449288), c(0.7853981633974483096));
    Value diagonal = rewriter.create<arith::SelectOp>(
        loc, ySign, rewriter.create<arith::NegFOp>(loc, diagonalMagnitude),
        diagonalMagnitude);
    Value result = rewriter.create<arith::SelectOp>(
        loc, bothInfinite, diagonal, angle);
    result = rewriter.create<arith::SelectOp>(
        loc, xZero, xZeroResult, result);

    Value unordered = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNO, x, y);
    Value nan = c(std::numeric_limits<float>::quiet_NaN());
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, unordered, nan, result);
    return success();
  }
};

/// Abramowitz-Stegun 7.1.26 erf approximation.  It is self-contained, stable
/// for the complete finite f32 range, and has max absolute error ~1.5e-7 before
/// the native ex2 approximation used by the existing backend.
struct ErfToPolynomial : OpRewritePattern<math::ErfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ErfOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getOperand().getType());
    if (!f32)
      return rewriter.notifyMatchFailure(op, "operand is not scalar f32");
    Location loc = op.getLoc();
    auto c = [&](double value) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(value));
    };
    Value x = op.getOperand();
    Value zero = c(0.0);
    Value one = c(1.0);
    Value negative = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, x, zero);
    Value magnitude = rewriter.create<arith::SelectOp>(
        loc, negative, rewriter.create<arith::NegFOp>(loc, x), x);
    Value denominator = rewriter.create<arith::AddFOp>(
        loc, one,
        rewriter.create<arith::MulFOp>(loc, c(0.3275911), magnitude));
    Value t = rewriter.create<arith::DivFOp>(loc, one, denominator);

    Value polynomial = c(1.061405429);
    for (double coefficient : {-1.453152027, 1.421413741, -0.284496736,
                               0.254829592}) {
      polynomial = rewriter.create<arith::AddFOp>(
          loc, rewriter.create<arith::MulFOp>(loc, polynomial, t),
          c(coefficient));
    }
    polynomial = rewriter.create<arith::MulFOp>(loc, polynomial, t);
    Value square = rewriter.create<arith::MulFOp>(loc, magnitude, magnitude);
    Value exponent = rewriter.create<arith::MulFOp>(
        loc, square, c(-kLog2E));
    auto exponential = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32, rewriter.getStringAttr("llvm.nvvm.ex2.approx.ftz.f"),
        ValueRange{exponent});
    Value tail = rewriter.create<arith::MulFOp>(
        loc, polynomial, exponential.getResult(0));
    Value result = rewriter.create<arith::SubFOp>(loc, one, tail);
    Value negativeResult = rewriter.create<arith::NegFOp>(loc, result);
    result = rewriter.create<arith::SelectOp>(
        loc, negative, negativeResult, result);
    Value isZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, x, zero);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isZero, x, result);
    return success();
  }
};

/// A self-contained f32 log2 for pow.  IEEE-754 exponent/mantissa reduction
/// keeps the atanh series on [1/sqrt(2), sqrt(2)], where seven odd terms are
/// below f32 precision.  Positive subnormals are normalized by 2^24 and the
/// extracted exponent is compensated accordingly.
static Value emitLog2F32Polynomial(PatternRewriter &rewriter, Location loc,
                                   Value x) {
  auto f32 = cast<Float32Type>(x.getType());
  Type i32 = rewriter.getI32Type();
  auto cF32 = [&](double value) -> Value {
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(value));
  };
  auto cI32 = [&](int32_t value) -> Value {
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(value));
  };

  Value bits = rewriter.create<arith::BitcastOp>(loc, i32, x);
  Value exponentField = rewriter.create<arith::AndIOp>(
      loc, rewriter.create<arith::ShRUIOp>(loc, bits, cI32(23)),
      cI32(0xff));
  Value exponentFieldZero = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, exponentField, cI32(0));
  Value valueNonzero = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, bits, cI32(0));
  Value subnormal = rewriter.create<arith::AndIOp>(
      loc, exponentFieldZero, valueNonzero);
  Value scaledSubnormal = rewriter.create<arith::MulFOp>(
      loc, x, cF32(16777216.0)); // 2^24
  Value normalized = rewriter.create<arith::SelectOp>(
      loc, subnormal, scaledSubnormal, x);

  Value normalizedBits =
      rewriter.create<arith::BitcastOp>(loc, i32, normalized);
  Value normalizedExponentField = rewriter.create<arith::AndIOp>(
      loc,
      rewriter.create<arith::ShRUIOp>(loc, normalizedBits, cI32(23)),
      cI32(0xff));
  Value exponent = rewriter.create<arith::SubIOp>(
      loc, normalizedExponentField, cI32(127));
  Value subnormalExponent = rewriter.create<arith::SubIOp>(
      loc, exponent, cI32(24));
  exponent = rewriter.create<arith::SelectOp>(
      loc, subnormal, subnormalExponent, exponent);

  Value mantissaBits = rewriter.create<arith::OrIOp>(
      loc,
      rewriter.create<arith::AndIOp>(loc, normalizedBits, cI32(0x007fffff)),
      cI32(0x3f800000));
  Value mantissa =
      rewriter.create<arith::BitcastOp>(loc, f32, mantissaBits);
  Value aboveSqrt2 = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, mantissa,
      cF32(1.4142135623730950488));
  Value reducedMantissa = rewriter.create<arith::SelectOp>(
      loc, aboveSqrt2,
      rewriter.create<arith::MulFOp>(loc, mantissa, cF32(0.5)), mantissa);
  exponent = rewriter.create<arith::SelectOp>(
      loc, aboveSqrt2,
      rewriter.create<arith::AddIOp>(loc, exponent, cI32(1)), exponent);

  Value one = cF32(1.0);
  Value numerator = rewriter.create<arith::SubFOp>(
      loc, reducedMantissa, one);
  Value denominator = rewriter.create<arith::AddFOp>(
      loc, reducedMantissa, one);
  Value z = rewriter.create<arith::DivFOp>(loc, numerator, denominator);
  Value z2 = rewriter.create<arith::MulFOp>(loc, z, z);
  auto fma = [&](Value lhs, Value rhs, Value addend) -> Value {
    return rewriter
        .create<LLVM::CallIntrinsicOp>(
            loc, f32, rewriter.getStringAttr("llvm.fma"),
            ValueRange{lhs, rhs, addend})
        .getResult(0);
  };
  Value polynomial = cF32(1.0 / 13.0);
  for (int denominatorValue = 11; denominatorValue >= 1;
       denominatorValue -= 2) {
    polynomial = fma(
        polynomial, z2,
        cF32(1.0 / static_cast<double>(denominatorValue)));
  }
  Value series = rewriter.create<arith::MulFOp>(loc, z, polynomial);
  Value exponentF32 =
      rewriter.create<arith::SIToFPOp>(loc, f32, exponent);
  return fma(series, cF32(2.0 * kLog2E), exponentF32);
}

/// `math.powf %base, %exponent : f32` -> polynomial log2/native ex2 plus an
/// explicit C99-compatible special-value decision tree.
///
/// The stock NVIDIA conversion emits an unresolved `__nv_powf`.  For the
/// regular magnitude use
///
///   |base|^exponent = 2^(exponent * log2(|base|)),
///
/// with a range-reduced f32 log2 and the native ex2 PTX instruction.  Float
/// exponents are classified as integer/odd directly from their IEEE-754
/// representation, avoiding an out-of-range float-to-int conversion.  This
/// preserves negative-base integer powers and the required signed
/// zero/infinity results while rejecting negative finite bases raised to a
/// non-integer exponent.
struct PowFToLg2Ex2Approx : OpRewritePattern<math::PowFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::PowFOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getLhs().getType());
    if (!f32 || op.getRhs().getType() != f32)
      return rewriter.notifyMatchFailure(op, "operands are not scalar f32");

    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    auto cF32 = [&](double value) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(value));
    };
    auto cI32 = [&](int32_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(value));
    };
    Value trueValue = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    auto logicalNot = [&](Value value) -> Value {
      return rewriter.create<arith::XOrIOp>(loc, value, trueValue);
    };

    Value base = op.getLhs();
    Value exponent = op.getRhs();
    Value zero = cF32(0.0);
    Value one = cF32(1.0);
    Value infinity = cF32(std::numeric_limits<float>::infinity());
    Value nan = cF32(std::numeric_limits<float>::quiet_NaN());

    Value baseBits = rewriter.create<arith::BitcastOp>(loc, i32, base);
    Value exponentBits =
        rewriter.create<arith::BitcastOp>(loc, i32, exponent);
    Value zeroI32 = cI32(0);
    Value absoluteMask = cI32(0x7fffffff);
    Value infinityBits = cI32(0x7f800000);
    Value mantissaMask = cI32(0x007fffff);
    Value baseAbsoluteBits = rewriter.create<arith::AndIOp>(
        loc, baseBits, absoluteMask);
    Value exponentAbsoluteBits = rewriter.create<arith::AndIOp>(
        loc, exponentBits, absoluteMask);
    Value absoluteBase =
        rewriter.create<arith::BitcastOp>(loc, f32, baseAbsoluteBits);
    Value baseSign = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, baseBits, zeroI32);
    Value baseZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, baseAbsoluteBits, zeroI32);
    Value baseInfinity = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, baseAbsoluteBits, infinityBits);
    Value baseNaN = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ugt, baseAbsoluteBits, infinityBits);
    Value baseFinite = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, baseAbsoluteBits, infinityBits);

    Value exponentZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, exponentAbsoluteBits, zeroI32);
    Value exponentInfinity = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, exponentAbsoluteBits, infinityBits);
    Value exponentNaN = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ugt, exponentAbsoluteBits, infinityBits);
    Value exponentPositive = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, exponent, zero);
    Value exponentNegative = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, exponent, zero);

    // A finite f32 is integral when all fractional significand bits below its
    // unbiased exponent are zero.  Clamp the dynamic shift into [0, 23] so the
    // integer shift itself is defined even when a later select discards it.
    Value exponentField = rewriter.create<arith::AndIOp>(
        loc,
        rewriter.create<arith::ShRUIOp>(loc, exponentAbsoluteBits, cI32(23)),
        cI32(0xff));
    Value unbiasedExponent = rewriter.create<arith::SubIOp>(
        loc, exponentField, cI32(127));
    Value exponentAtLeastZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, unbiasedExponent, zeroI32);
    Value exponentAtMost23 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, unbiasedExponent, cI32(23));
    Value exponentAbove23 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, unbiasedExponent, cI32(23));
    Value integerRange = rewriter.create<arith::AndIOp>(
        loc, exponentAtLeastZero, exponentAtMost23);
    Value rawShift = rewriter.create<arith::SubIOp>(
        loc, cI32(23), unbiasedExponent);
    Value shiftBelowZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, rawShift, zeroI32);
    Value nonnegativeShift = rewriter.create<arith::SelectOp>(
        loc, shiftBelowZero, zeroI32, rawShift);
    Value shiftAbove23 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, nonnegativeShift, cI32(23));
    Value safeShift = rewriter.create<arith::SelectOp>(
        loc, shiftAbove23, cI32(23), nonnegativeShift);
    Value integerBit = rewriter.create<arith::ShLIOp>(
        loc, cI32(1), safeShift);
    Value fractionalMask = rewriter.create<arith::SubIOp>(
        loc, integerBit, cI32(1));
    Value mantissa = rewriter.create<arith::AndIOp>(
        loc, exponentAbsoluteBits, mantissaMask);
    Value noFraction = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<arith::AndIOp>(loc, mantissa, fractionalMask),
        zeroI32);
    Value finiteExponent = logicalNot(exponentInfinity);
    finiteExponent = rewriter.create<arith::AndIOp>(
        loc, finiteExponent, logicalNot(exponentNaN));
    Value integralMagnitude = rewriter.create<arith::OrIOp>(
        loc, exponentAbove23,
        rewriter.create<arith::AndIOp>(loc, integerRange, noFraction));
    Value exponentIsInteger = rewriter.create<arith::OrIOp>(
        loc, exponentZero,
        rewriter.create<arith::AndIOp>(loc, finiteExponent,
                                       integralMagnitude));

    Value significand = rewriter.create<arith::OrIOp>(
        loc, mantissa, cI32(1 << 23));
    Value oddBitSet = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne,
        rewriter.create<arith::AndIOp>(loc, significand, integerBit),
        zeroI32);
    Value exponentIsOdd = rewriter.create<arith::AndIOp>(
        loc, exponentIsInteger,
        rewriter.create<arith::AndIOp>(loc, integerRange, oddBitSet));
    Value negativeResult = rewriter.create<arith::AndIOp>(
        loc, baseSign, exponentIsOdd);
    auto applyResultSign = [&](Value magnitude) -> Value {
      Value negativeMagnitude =
          rewriter.create<arith::NegFOp>(loc, magnitude);
      return rewriter.create<arith::SelectOp>(
          loc, negativeResult, negativeMagnitude, magnitude);
    };

    Value log2Base =
        emitLog2F32Polynomial(rewriter, loc, absoluteBase);
    Value scaled = rewriter.create<arith::MulFOp>(
        loc, exponent, log2Base);
    Value needsSubnormalResult = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, scaled, cF32(-126.0));
    Value exponentAdjustment = rewriter.create<arith::SelectOp>(
        loc, needsSubnormalResult, cF32(24.0), zero);
    Value adjustedScaled = rewriter.create<arith::AddFOp>(
        loc, scaled, exponentAdjustment);
    auto ex2 = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, f32,
        rewriter.getStringAttr("llvm.nvvm.ex2.approx.ftz.f"),
        ValueRange{adjustedScaled});
    Value magnitudeScale = rewriter.create<arith::SelectOp>(
        loc, needsSubnormalResult, cF32(5.9604644775390625e-8), one);
    Value magnitude = rewriter.create<arith::MulFOp>(
        loc, ex2.getResult(0), magnitudeScale);
    Value result = applyResultSign(magnitude);

    Value negativeFiniteNonzeroBase = rewriter.create<arith::AndIOp>(
        loc, baseSign,
        rewriter.create<arith::AndIOp>(loc, baseFinite,
                                       logicalNot(baseZero)));
    Value invalidNegativePower = rewriter.create<arith::AndIOp>(
        loc, negativeFiniteNonzeroBase, logicalNot(exponentIsInteger));
    result = rewriter.create<arith::SelectOp>(
        loc, invalidNegativePower, nan, result);

    Value absoluteBaseGreaterOne = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, absoluteBase, one);
    Value absoluteBaseLessOne = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, absoluteBase, one);
    Value growingInfiniteExponent = rewriter.create<arith::SelectOp>(
        loc, exponentPositive, infinity, zero);
    Value decayingInfiniteExponent = rewriter.create<arith::SelectOp>(
        loc, exponentPositive, zero, infinity);
    Value infiniteExponentResult = rewriter.create<arith::SelectOp>(
        loc, absoluteBaseGreaterOne, growingInfiniteExponent,
        rewriter.create<arith::SelectOp>(loc, absoluteBaseLessOne,
                                         decayingInfiniteExponent, one));
    result = rewriter.create<arith::SelectOp>(
        loc, exponentInfinity, infiniteExponentResult, result);

    Value infiniteBaseMagnitude = rewriter.create<arith::SelectOp>(
        loc, exponentNegative, zero, infinity);
    Value infiniteBaseResult = applyResultSign(infiniteBaseMagnitude);
    result = rewriter.create<arith::SelectOp>(
        loc, baseInfinity, infiniteBaseResult, result);

    Value zeroBaseMagnitude = rewriter.create<arith::SelectOp>(
        loc, exponentNegative, infinity, zero);
    Value zeroBaseResult = applyResultSign(zeroBaseMagnitude);
    result = rewriter.create<arith::SelectOp>(
        loc, baseZero, zeroBaseResult, result);

    Value anyNaN = rewriter.create<arith::OrIOp>(loc, baseNaN, exponentNaN);
    result = rewriter.create<arith::SelectOp>(loc, anyNaN, nan, result);
    Value positiveOne = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, baseBits, cI32(0x3f800000));
    result = rewriter.create<arith::SelectOp>(loc, positiveOne, one, result);
    result = rewriter.create<arith::SelectOp>(
        loc, exponentZero, one, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// `math.fma %a, %b, %c : f32` -> LLVM's target-independent fused multiply-add
/// intrinsic.  Leaving the math op to the stock GPU pipeline produces a
/// `__nv_fmaf` libdevice extern, which has the same unresolved-symbol failure
/// as the transcendental calls above.  `llvm.fma` lowers directly to PTX
/// `fma.rn.f32` and preserves the single-rounding semantics requested by the
/// row-reduction accumulator.
struct FmaToLLVMIntrinsic : OpRewritePattern<math::FmaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const override {
    auto f32 = dyn_cast<Float32Type>(op.getA().getType());
    if (!f32 || op.getB().getType() != f32 || op.getC().getType() != f32)
      return rewriter.notifyMatchFailure(op, "operands are not scalar f32");
    auto fused = rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), f32, rewriter.getStringAttr("llvm.fma"),
        ValueRange{op.getA(), op.getB(), op.getC()});
    rewriter.replaceOp(op, fused.getResults());
    return success();
  }
};

struct LowerGpuMathToNVVM
    : ::intent::impl::LowerGpuMathToNVVMBase<LowerGpuMathToNVVM> {
  using LowerGpuMathToNVVMBase::LowerGpuMathToNVVMBase;

  void runOnOperation() override {
    // WGMMA reads shared memory through the async proxy.  A CTA barrier orders
    // the producer threads but does not cross the generic/async proxy boundary.
    // The structural Flash lowering marks exactly the producer barriers that
    // need PTX `fence.proxy.async.shared::cta`; materialize that target fence
    // here, after GPU outlining and immediately before the NVVM pipeline.
    SmallVector<Operation *> proxyFenceBarriers;
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "gpu.barrier" &&
          op->hasAttr("intent.backend.wgmma_proxy_fence"))
        proxyFenceBarriers.push_back(op);
    });
    for (Operation *barrier : proxyFenceBarriers) {
      OpBuilder builder(barrier);
      builder.setInsertionPointAfter(barrier);
      builder.create<LLVM::InlineAsmOp>(
          barrier->getLoc(), TypeRange{}, ValueRange{},
          "fence.proxy.async.shared::cta;", "",
          /*has_side_effects=*/true, /*is_align_stack=*/false,
          LLVM::AsmDialectAttr(), ArrayAttr());
      barrier->removeAttr("intent.backend.wgmma_proxy_fence");
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<PromoteHalfUnaryMath<math::ExpOp>,
                 PromoteHalfUnaryMath<math::LogOp>,
                 PromoteHalfUnaryMath<math::Log2Op>,
                 PromoteHalfUnaryMath<math::AbsFOp>,
                 PromoteHalfUnaryMath<math::SqrtOp>,
                 PromoteHalfUnaryMath<math::RsqrtOp>,
                 PromoteHalfUnaryMath<math::TanhOp>,
                 PromoteHalfUnaryMath<math::CosOp>,
                 PromoteHalfUnaryMath<math::SinOp>,
                 PromoteHalfUnaryMath<math::Exp2Op>,
                 PromoteHalfUnaryMath<math::AtanOp>,
                 PromoteHalfUnaryMath<math::ErfOp>,
                 PromoteHalfUnaryMath<math::AcosOp>,
                 PromoteHalfUnaryMath<math::AsinOp>,
                 PromoteHalfUnaryMath<math::CbrtOp>,
                 PromoteHalfUnaryMath<math::CeilOp>,
                 PromoteHalfUnaryMath<math::FloorOp>,
                 PromoteHalfUnaryMath<math::RoundOp>,
                 PromoteHalfUnaryMath<math::RoundEvenOp>,
                 PromoteHalfUnaryMath<math::TruncOp>,
                 PromoteHalfUnaryMath<math::Log10Op>,
                 PromoteHalfUnaryMath<math::Log1pOp>,
                 PromoteHalfUnaryMath<math::ExpM1Op>,
                 PromoteHalfBinaryMath<math::PowFOp>,
                 PromoteHalfBinaryMath<math::CopySignOp>,
                 PromoteHalfBinaryMath<math::Atan2Op>, PromoteHalfFma>(
        &getContext());
    patterns.add<ExpM1ToExpSubOne>(&getContext());
    patterns.add<Log1pToLogOnePlus>(&getContext());
    patterns.add<Log10ToLogScale>(&getContext());
    patterns.add<AsinToAtanApprox>(&getContext());
    patterns.add<AcosToAtanApprox>(&getContext());
    patterns.add<CbrtToExpLog>(&getContext());
    patterns.add<CopySignToLLVMIntrinsic>(&getContext());
    patterns.add<LogSubToLg2Fma>(&getContext());
    patterns.add<ExpToEx2Approx>(&getContext());
    patterns.add<MarkedDivToNVVMApprox>(&getContext());
    patterns.add<SigmoidReciprocalToNVVMApprox>(&getContext());
    patterns.add<LogToLg2Approx>(&getContext());
    patterns.add<Log2ToLg2Approx>(&getContext());
    patterns.add<LogF64Polynomial>(&getContext());
    patterns.add<Exp2ToEx2Approx>(&getContext());
    patterns.add<TanhToEx2Approx>(&getContext());
    patterns.add<AbsFToLLVMIntrinsic>(&getContext());
    patterns.add<AtanToPolynomial>(&getContext());
    patterns.add<Atan2ToPolynomial>(&getContext());
    patterns.add<ErfToPolynomial>(&getContext());
    patterns.add<PowFToLg2Ex2Approx>(&getContext());
    patterns.add<FmaToLLVMIntrinsic>(&getContext());
    patterns.add<UnaryToNVVMApprox<math::SqrtOp>>(&getContext(),
                                                  "llvm.nvvm.sqrt.approx.f");
    patterns.add<UnaryToNVVMApprox<math::RsqrtOp>>(&getContext(),
                                                   "llvm.nvvm.rsqrt.approx.f");
    // sin/cos: the `sin.approx.f`/`cos.approx.f` PTX hardware ops (math.sin /
    // math.cos would otherwise become `__nv_sinf`/`__nv_cosf` libdevice externs).
    patterns.add<UnaryToNVVMApprox<math::SinOp>>(&getContext(),
                                                 "llvm.nvvm.sin.approx.f");
    patterns.add<UnaryToNVVMApprox<math::CosOp>>(&getContext(),
                                                 "llvm.nvvm.cos.approx.f");
    patterns.add<UnaryToLLVMIntrinsic<math::CeilOp>>(&getContext(),
                                                     "llvm.ceil");
    patterns.add<UnaryToLLVMIntrinsic<math::FloorOp>>(&getContext(),
                                                      "llvm.floor");
    patterns.add<UnaryToLLVMIntrinsic<math::RoundOp>>(&getContext(),
                                                      "llvm.round");
    patterns.add<UnaryToLLVMIntrinsic<math::RoundEvenOp>>(&getContext(),
                                                          "llvm.roundeven");
    patterns.add<UnaryToLLVMIntrinsic<math::TruncOp>>(&getContext(),
                                                      "llvm.trunc");
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
