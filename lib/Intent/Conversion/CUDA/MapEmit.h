//===- MapEmit.h - shared scalar emission for intent map lowerings --------===//
//
// Pure, stateless helpers shared by the block-cooperative intent->gpu lowerings
// (`convert-intent-row-reduction-to-gpu` and `convert-intent-elementwise-to-gpu`).
// Only the genuinely stateless pieces live here: the per-`fn` scalar arith/math
// emitters, the per-arg `Role` taxonomy (inferred from static shape vs (M,N)),
// and the launch-geometry default. The stateful read-once register-tile
// machinery (emitAt recursion / hoistUniforms / the per-arg SSA caches) stays
// self-contained in each pass — it is ~60 lines, intertwined with each pass's
// DenseMaps, and duplicating it keeps the M1 pass byte-for-byte unchanged in
// behavior (no shared base struct = no destabilization risk).
//
//===----------------------------------------------------------------------===//

#ifndef INTENT_CONVERSION_MAPEMIT_H
#define INTENT_CONVERSION_MAPEMIT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <limits>

namespace intent {

inline bool isUnsignedInteger(mlir::Type type) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  return integer && integer.isUnsigned();
}

inline mlir::Value toSignlessInteger(mlir::OpBuilder &b, mlir::Location loc,
                                     mlir::Value value) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  if (!integer || integer.isSignless())
    return value;
  mlir::Type signless = mlir::IntegerType::get(
      value.getContext(), integer.getWidth());
  return b.create<mlir::UnrealizedConversionCastOp>(
             loc, mlir::TypeRange{signless}, mlir::ValueRange{value})
      .getResult(0);
}

inline mlir::Value fromSignlessInteger(mlir::OpBuilder &b, mlir::Location loc,
                                       mlir::Value value, mlir::Type target) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(target);
  if (!integer || integer.isSignless() || value.getType() == target)
    return value;
  return b.create<mlir::UnrealizedConversionCastOp>(
             loc, mlir::TypeRange{target}, mlir::ValueRange{value})
      .getResult(0);
}

inline mlir::Value truthy(mlir::OpBuilder &b, mlir::Location loc,
                          mlir::Value value) {
  using namespace mlir;
  if (auto integer = dyn_cast<IntegerType>(value.getType())) {
    Value signless = toSignlessInteger(b, loc, value);
    Value zero = b.create<arith::ConstantOp>(
        loc, b.getIntegerAttr(signless.getType(), 0));
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, signless,
                                   zero);
  }
  if (auto floating = dyn_cast<FloatType>(value.getType())) {
    Value zero = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(floating, 0.0));
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, value, zero);
  }
  return nullptr;
}

/// Scalar arith/math op for one `map_unary` fn. nullptr = unsupported here.
inline mlir::Value emitScalarUnary(mlir::OpBuilder &b, mlir::Location loc,
                                   llvm::StringRef fn, mlir::Value x) {
  using namespace mlir;
  if (fn == "identity")
    return x;
  if (fn == "exp")
    return b.create<math::ExpOp>(loc, x);
  if (fn == "log")
    return b.create<math::LogOp>(loc, x);
  if (fn == "log2")
    return b.create<math::Log2Op>(loc, x);
  if (fn == "sqrt")
    return b.create<math::SqrtOp>(loc, x);
  if (fn == "rsqrt")
    return b.create<math::RsqrtOp>(loc, x);
  if (fn == "abs") {
    if (auto integer = dyn_cast<IntegerType>(x.getType())) {
      if (integer.isUnsigned())
        return x;
      return b.create<math::AbsIOp>(loc, toSignlessInteger(b, loc, x));
    }
    // |x| = max(x, -x). math.absf lowers to a `__nv_fabsf` libdevice extern
    // (ptxas dead end, same failure mode as exp/tanh/sqrt); the arith form is
    // resolved entirely on-device.
    Value negx = b.create<arith::NegFOp>(loc, x);
    return b.create<arith::MaxNumFOp>(loc, x, negx);
  }
  if (fn == "neg") {
    if (isa<IntegerType>(x.getType())) {
      Value integer = toSignlessInteger(b, loc, x);
      Value zero = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(integer.getType(), 0));
      Value result = b.create<arith::SubIOp>(loc, zero, integer);
      return fromSignlessInteger(b, loc, result, x.getType());
    }
    return b.create<arith::NegFOp>(loc, x);
  }
  if (fn == "tanh") {
    // tanh(x) = 1 - 2/(exp(2x)+1). Routed through math.exp (-> ex2.approx.ftz
    // hardware instruction by the `-lower-gpu-math-to-nvvm` peephole) instead of
    // math.tanh, whose stock GpuToNVVM lowering emits an unresolved `__nv_tanhf`
    // libdevice extern (ptxas dead end, same failure mode as exp/sqrt). This
    // form is overflow-safe: large +x -> exp->inf -> 1-0 = 1; large -x -> exp->0
    // -> 1-2 = -1 (the (exp-1)/(exp+1) form would give inf/inf = NaN).
    Value one =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    Value two =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 2.0));
    Value e = b.create<math::ExpOp>(loc, b.create<arith::MulFOp>(loc, x, two));
    Value den = b.create<arith::AddFOp>(loc, e, one);
    return b.create<arith::SubFOp>(loc, one,
                                   b.create<arith::DivFOp>(loc, two, den));
  }
  if (fn == "recip") {
    Value one =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    return b.create<arith::DivFOp>(loc, one, x);
  }
  if (fn == "sigmoid") {
    Value one =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    Value e = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, x));
    return b.create<arith::DivFOp>(loc, one,
                                   b.create<arith::AddFOp>(loc, one, e));
  }
  // cos/sin route through math.cos/math.sin, which the `-lower-gpu-math-to-nvvm`
  // peephole turns into the `cos.approx.f`/`sin.approx.f` PTX hardware ops
  // (the stock GpuToNVVM path emits `__nv_cosf`/`__nv_sinf` libdevice externs =
  // ptxas dead end, the same failure mode as exp/tanh/sqrt).
  if (fn == "cos")
    return b.create<math::CosOp>(loc, x);
  if (fn == "sin")
    return b.create<math::SinOp>(loc, x);
  if (fn == "tan") {
    // tan(x) = sin(x)/cos(x): stay off `__nv_tanf` by composing the two approx
    // hardware ops the peephole already provides.
    Value s = b.create<math::SinOp>(loc, x);
    Value c = b.create<math::CosOp>(loc, x);
    return b.create<arith::DivFOp>(loc, s, c);
  }
  if (fn == "exp2")
    // 2^x directly on the `ex2.approx.ftz.f` hardware op (no log2(e) prescale,
    // unlike exp) via the math.exp2 peephole.
    return b.create<math::Exp2Op>(loc, x);
  if (fn == "log2")
    return b.create<math::Log2Op>(loc, x);
  if (fn == "atan")
    return b.create<math::AtanOp>(loc, x);
  if (fn == "erf")
    return b.create<math::ErfOp>(loc, x);
  if (fn == "erfc") {
    Value one = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(x.getType(), 1.0));
    return b.create<arith::SubFOp>(loc, one, b.create<math::ErfOp>(loc, x));
  }
  if (fn == "acos")
    return b.create<math::AcosOp>(loc, x);
  if (fn == "asin")
    return b.create<math::AsinOp>(loc, x);
  if (fn == "cbrt")
    return b.create<math::CbrtOp>(loc, x);
  if (fn == "ceil")
    return b.create<math::CeilOp>(loc, x);
  if (fn == "floor")
    return b.create<math::FloorOp>(loc, x);
  if (fn == "round")
    return b.create<math::RoundOp>(loc, x);
  if (fn == "roundeven")
    return b.create<math::RoundEvenOp>(loc, x);
  if (fn == "trunc")
    return b.create<math::TruncOp>(loc, x);
  if (fn == "sign") {
    if (auto integer = dyn_cast<IntegerType>(x.getType())) {
      Value value = toSignlessInteger(b, loc, x);
      Type valueType = value.getType();
      Value zero = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(valueType, 0));
      Value positive = b.create<arith::CmpIOp>(
          loc, integer.isUnsigned() ? arith::CmpIPredicate::ugt
                                     : arith::CmpIPredicate::sgt,
          value, zero);
      Value one = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(valueType, 1));
      return fromSignlessInteger(
          b, loc, b.create<arith::SelectOp>(loc, positive, one, zero),
          x.getType());
    }
    Value zero;
    Value negative;
    Value positive;
    if (isa<FloatType>(x.getType())) {
      zero = b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 0.0));
      negative = b.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OLT, x, zero);
      positive = b.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OGT, x, zero);
    } else {
      zero = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(x.getType(), 0));
      negative = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, x, zero);
      positive = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, x, zero);
    }
    Value minusOne = isa<FloatType>(x.getType())
                         ? b.create<arith::ConstantOp>(
                               loc, b.getFloatAttr(x.getType(), -1.0))
                         : b.create<arith::ConstantOp>(
                               loc, b.getIntegerAttr(x.getType(), -1));
    Value plusOne = isa<FloatType>(x.getType())
                        ? b.create<arith::ConstantOp>(
                              loc, b.getFloatAttr(x.getType(), 1.0))
                        : b.create<arith::ConstantOp>(
                              loc, b.getIntegerAttr(x.getType(), 1));
    Value signedValue = b.create<arith::SelectOp>(loc, negative, minusOne, x);
    return b.create<arith::SelectOp>(loc, positive, plusOne, signedValue);
  }
  if (fn == "log10")
    return b.create<math::Log10Op>(loc, x);
  if (fn == "log1p")
    return b.create<math::Log1pOp>(loc, x);
  if (fn == "expm1")
    return b.create<math::ExpM1Op>(loc, x);
  if (fn == "isnan")
    // NaN is the only value unordered with itself. -> i1 (feeds intent.select,
    // e.g. nan_to_num = select(isnan(x), 0, x)). No libdevice.
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO, x, x);
  if (fn == "isinf" || fn == "isfinite") {
    Value inf = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(x.getType(),
                            std::numeric_limits<double>::infinity()));
    Value ax = b.create<math::AbsFOp>(loc, x);
    return b.create<arith::CmpFOp>(
        loc, fn == "isinf" ? arith::CmpFPredicate::OEQ
                             : arith::CmpFPredicate::ONE,
        ax, inf);
  }
  if (fn == "logical_not") {
    // i1 complement: x XOR true.
    Value t = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
    return b.create<arith::XOrIOp>(loc, x, t);
  }
  return nullptr;
}

/// Scalar arith op for one `map_binary` fn. nullptr = unsupported here.
inline mlir::Value emitScalarBinary(mlir::OpBuilder &b, mlir::Location loc,
                                    llvm::StringRef fn, mlir::Value a,
                                    mlir::Value c) {
  using namespace mlir;
  if (auto integer = dyn_cast<IntegerType>(a.getType())) {
    bool unsignedArithmetic = integer.isUnsigned();
    Value lhs = toSignlessInteger(b, loc, a);
    Value rhs = toSignlessInteger(b, loc, c);
    Value result;
    if (fn == "add")
      result = b.create<arith::AddIOp>(loc, lhs, rhs);
    else if (fn == "sub")
      result = b.create<arith::SubIOp>(loc, lhs, rhs);
    else if (fn == "mul")
      result = b.create<arith::MulIOp>(loc, lhs, rhs);
    else if (fn == "div")
      result = unsignedArithmetic
                   ? Value(b.create<arith::DivUIOp>(loc, lhs, rhs))
                   : Value(b.create<arith::DivSIOp>(loc, lhs, rhs));
    else if (fn == "floordiv")
      result = unsignedArithmetic
                   ? Value(b.create<arith::DivUIOp>(loc, lhs, rhs))
                   : Value(b.create<arith::FloorDivSIOp>(loc, lhs, rhs));
    else if (fn == "remainder") {
      if (unsignedArithmetic) {
        result = b.create<arith::RemUIOp>(loc, lhs, rhs);
      } else {
        Value remainder = b.create<arith::RemSIOp>(loc, lhs, rhs);
        Value zero = b.create<arith::ConstantOp>(
            loc, b.getIntegerAttr(lhs.getType(), 0));
        Value lhsNegative = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, lhs, zero);
        Value rhsNegative = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, rhs, zero);
        Value differentSigns = b.create<arith::XOrIOp>(
            loc, lhsNegative, rhsNegative);
        Value nonzero = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, remainder, zero);
        Value adjust = b.create<arith::AndIOp>(
            loc, differentSigns, nonzero);
        Value adjusted = b.create<arith::AddIOp>(loc, remainder, rhs);
        result = b.create<arith::SelectOp>(loc, adjust, adjusted, remainder);
      }
    } else if (fn == "left_shift")
      result = b.create<arith::ShLIOp>(loc, lhs, rhs);
    else if (fn == "right_shift")
      result = unsignedArithmetic
                   ? Value(b.create<arith::ShRUIOp>(loc, lhs, rhs))
                   : Value(b.create<arith::ShRSIOp>(loc, lhs, rhs));
    else if (fn == "max")
      result = unsignedArithmetic
                   ? Value(b.create<arith::MaxUIOp>(loc, lhs, rhs))
                   : Value(b.create<arith::MaxSIOp>(loc, lhs, rhs));
    else if (fn == "min")
      result = unsignedArithmetic
                   ? Value(b.create<arith::MinUIOp>(loc, lhs, rhs))
                   : Value(b.create<arith::MinSIOp>(loc, lhs, rhs));
    else if (fn == "logical_and" || fn == "and")
      return b.create<arith::AndIOp>(loc, truthy(b, loc, a), truthy(b, loc, c));
    else if (fn == "logical_or" || fn == "or")
      return b.create<arith::OrIOp>(loc, truthy(b, loc, a), truthy(b, loc, c));
    else if (fn == "logical_xor" || fn == "xor")
      return b.create<arith::XOrIOp>(loc, truthy(b, loc, a), truthy(b, loc, c));
    else
      return nullptr;
    return fromSignlessInteger(b, loc, result, a.getType());
  }
  if (fn == "add")
    return b.create<arith::AddFOp>(loc, a, c);
  if (fn == "sub")
    return b.create<arith::SubFOp>(loc, a, c);
  if (fn == "mul")
    return b.create<arith::MulFOp>(loc, a, c);
  if (fn == "div")
    return b.create<arith::DivFOp>(loc, a, c);
  if (fn == "floordiv") {
    Value quotient = b.create<arith::DivFOp>(loc, a, c);
    return b.create<math::FloorOp>(loc, quotient);
  }
  if (fn == "remainder") {
    Value quotient = b.create<arith::DivFOp>(loc, a, c);
    Value floored = b.create<math::FloorOp>(loc, quotient);
    return b.create<arith::SubFOp>(
        loc, a, b.create<arith::MulFOp>(loc, floored, c));
  }
  if (fn == "max")
    return b.create<arith::MaxNumFOp>(loc, a, c);
  if (fn == "min")
    return b.create<arith::MinNumFOp>(loc, a, c);
  if (fn == "pow")
    return b.create<math::PowFOp>(loc, a, c);
  if (fn == "atan2")
    return b.create<math::Atan2Op>(loc, a, c);
  if (fn == "copysign")
    return b.create<math::CopySignOp>(loc, a, c);
  if (fn == "logical_and" || fn == "and")
    return b.create<arith::AndIOp>(loc, truthy(b, loc, a), truthy(b, loc, c));
  if (fn == "logical_or" || fn == "or")
    return b.create<arith::OrIOp>(loc, truthy(b, loc, a), truthy(b, loc, c));
  if (fn == "logical_xor" || fn == "xor")
    return b.create<arith::XOrIOp>(loc, truthy(b, loc, a), truthy(b, loc, c));
  if (fn == "isclose") {
    Value rtol = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(a.getType(), 1e-5));
    Value atol = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(a.getType(), 1e-8));
    Value diff = b.create<arith::SubFOp>(loc, a, c);
    Value absoluteDiff = b.create<math::AbsFOp>(loc, diff);
    Value absoluteRight = b.create<math::AbsFOp>(loc, c);
    Value allowed = b.create<arith::AddFOp>(
        loc, atol, b.create<arith::MulFOp>(loc, rtol, absoluteRight));
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE,
                                   absoluteDiff, allowed);
  }
  return nullptr;
}

/// Is `fn` a `map_binary` comparison (gt/ge/lt/le/eq/ne -> i1)?
inline bool isCmpFn(llvm::StringRef fn) {
  return fn == "gt" || fn == "ge" || fn == "lt" || fn == "le" ||
         fn == "eq" || fn == "ne";
}

inline mlir::Value emitScalarCmpI(mlir::OpBuilder &b, mlir::Location loc,
                                  llvm::StringRef fn, mlir::Value a,
                                  mlir::Value c);

/// Scalar arith.cmpf for one `map_binary` comparison fn (-> i1). nullptr if not
/// a comparison fn. Ordered predicates (NaN-false), matching numpy/triton.
inline mlir::Value emitScalarCmp(mlir::OpBuilder &b, mlir::Location loc,
                                 llvm::StringRef fn, mlir::Value a,
                                 mlir::Value c) {
  using namespace mlir;
  if (isa<IntegerType>(a.getType()))
    return emitScalarCmpI(b, loc, fn, a, c);
  arith::CmpFPredicate p;
  if (fn == "gt")
    p = arith::CmpFPredicate::OGT;
  else if (fn == "ge")
    p = arith::CmpFPredicate::OGE;
  else if (fn == "lt")
    p = arith::CmpFPredicate::OLT;
  else if (fn == "le")
    p = arith::CmpFPredicate::OLE;
  else if (fn == "eq")
    p = arith::CmpFPredicate::OEQ;
  else if (fn == "ne")
    p = arith::CmpFPredicate::UNE;
  else
    return nullptr;
  return b.create<arith::CmpFOp>(loc, p, a, c);
}

/// Scalar arith.cmpi for one `map_binary` comparison fn over INTEGER operands
/// (-> i1). Used by cross_entropy's `iota == target` one-hot condition (column
/// index vs the per-row target class), where the operands are integers and the
/// float `arith.cmpf` form does not apply. Signedness follows the J operand
/// type, with explicit unsigned predicates for ui8/ui16/ui32.
inline mlir::Value emitScalarCmpI(mlir::OpBuilder &b, mlir::Location loc,
                                  llvm::StringRef fn, mlir::Value a,
                                  mlir::Value c) {
  using namespace mlir;
  bool unsignedComparison = isUnsignedInteger(a.getType());
  Value lhs = toSignlessInteger(b, loc, a);
  Value rhs = toSignlessInteger(b, loc, c);
  arith::CmpIPredicate p;
  if (fn == "gt")
    p = unsignedComparison ? arith::CmpIPredicate::ugt
                           : arith::CmpIPredicate::sgt;
  else if (fn == "ge")
    p = unsignedComparison ? arith::CmpIPredicate::uge
                           : arith::CmpIPredicate::sge;
  else if (fn == "lt")
    p = unsignedComparison ? arith::CmpIPredicate::ult
                           : arith::CmpIPredicate::slt;
  else if (fn == "le")
    p = unsignedComparison ? arith::CmpIPredicate::ule
                           : arith::CmpIPredicate::sle;
  else if (fn == "eq")
    p = arith::CmpIPredicate::eq;
  else if (fn == "ne")
    p = arith::CmpIPredicate::ne;
  else
    return nullptr;
  return b.create<arith::CmpIOp>(loc, p, lhs, rhs);
}

/// Per-arg semantic role, inferred purely from static shape vs (M,N):
///   Row[M,N]  — a full feature row, register-tiled read-once per thread;
///   Col[N]    — a per-column vector (gamma/W), loaded at col[k];
///   Scalar[]  — a rank-0 uniform (alpha/eps/constant), loaded once per block;
///   RowInt[M] — a per-row INTEGER (cross_entropy's target class Y), loaded
///               once per block (uniform across the block's threads) as Y[bx].
///   ColSeg[C] — a "broadcast-segment" column vector (group_norm's W/B): a 1-D
///               f32 arg whose value at feature column `col` of a block in group
///               `g` is loaded at the channel index given by an affine
///               `intent.col_index_map = affine_map<(group, col) -> channel>`
///               arg-attr (e.g. col floordiv SPATIAL + group*CPG). A structured
///               data-independent gather (the index is affine in (group,col)),
///               general over any affine-broadcast Col — NOT a per-kernel branch.
///   ParamVector[P] — a small parameter vector addressed only through a
///                    structurally validated scalar/per-column intent.gather.
enum class Role {
  Row,
  Col,
  Scalar,
  RowInt,
  PerRow,
  ColSeg,
  ParamVector,
  Unused,
};

/// Default block size over the feature dim: largest power-of-two <= min(N,256)
/// dividing-friendly. The PK harness passes the library kernel's geometry.
inline int64_t autoThreads(int64_t N) {
  int64_t t = 1;
  while (t * 2 <= N && t * 2 <= 256)
    t *= 2;
  return t;
}

//===----------------------------------------------------------------------===//
// M1 shared high-performance emit primitives (R-CODEGEN-1/2, R3 §3, R5 P1/P2).
//
// The "flat grid-stride" occupancy fix + tail-mask: instead of one block per
// output row (sub-warp blocks that leave most SMs idle on NCHW pool/interp),
// map the WHOLE flattened output element space onto a full-width block-stride
// grid so every SM stays busy and any N (no divisibility) lowers. The grid is
// sized to a few waves of a numSM-SM device; the `scf.for i = gtid..total step
// grid*T` loop tail-masks the ragged remainder for free (threads past `total`
// just run zero iterations).
//===----------------------------------------------------------------------===//

/// Grid blocks for a flat grid-stride kernel over `total` output elements with
/// `T` threads/block: enough to fill a ~numSM-SM device several waves deep, but
/// never more than needed. numSM defaults to the RTX 5090D (GB202, sm_120a).
inline int64_t flatGridBlocks(int64_t total, int64_t T, int64_t numSM = 170) {
  if (T <= 0)
    return 1;
  int64_t need = (total + T - 1) / T;
  int64_t threadsPerSM = 2048, waves = 4;
  int64_t cap = (waves * numSM * threadsPerSM + T - 1) / T;
  int64_t g = need < cap ? need : cap;
  return g < 1 ? 1 : g;
}

/// Effective block size for a flat kernel: honor an explicit `blockThreads`
/// hint (the PK harness / autotuner), else a full 256-thread block (8 warps).
inline int64_t flatThreads(unsigned blockThreads) {
  return blockThreads ? (int64_t)blockThreads : 256;
}

/// Emit `gtid = bx*T + tx; scf.for iv = gtid to total step (G*T) { ... }` and
/// leave `b` positioned at the start of the loop body. `iv` receives the flat
/// output element index. The loop bound (static `total`) is the tail-mask: no
/// `total % (G*T) == 0` requirement. Returns the loop op.
inline mlir::scf::ForOp emitGridStride(mlir::OpBuilder &b, mlir::Location loc,
                                       mlir::Value bx, mlir::Value tx, int64_t T,
                                       int64_t G, int64_t total,
                                       mlir::Value &iv) {
  using namespace mlir;
  Value cT = b.create<arith::ConstantIndexOp>(loc, T);
  Value gtid =
      b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bx, cT), tx);
  Value ub = b.create<arith::ConstantIndexOp>(loc, total);
  Value step = b.create<arith::ConstantIndexOp>(loc, G * T);
  auto forOp = b.create<scf::ForOp>(loc, gtid, ub, step);
  b.setInsertionPointToStart(forOp.getBody());
  iv = forOp.getInductionVar();
  return forOp;
}

/// Unravel a flat row-major index `flat` into per-dimension index coords for a
/// tensor of static shape `dims` (coords[i] in [0,dims[i])). coords is resized.
inline void unravelIndex(mlir::OpBuilder &b, mlir::Location loc,
                         mlir::Value flat, llvm::ArrayRef<int64_t> dims,
                         llvm::SmallVectorImpl<mlir::Value> &coords) {
  using namespace mlir;
  coords.assign(dims.size(), nullptr);
  Value rem = flat;
  for (int i = (int)dims.size() - 1; i >= 0; --i) {
    Value d = b.create<arith::ConstantIndexOp>(loc, dims[i]);
    coords[i] = b.create<arith::RemUIOp>(loc, rem, d);
    if (i > 0)
      rem = b.create<arith::DivUIOp>(loc, rem, d);
  }
}

} // namespace intent

#endif // INTENT_CONVERSION_MAPEMIT_H
