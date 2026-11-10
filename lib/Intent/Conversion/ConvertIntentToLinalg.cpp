//===- ConvertIntentToLinalg.cpp - intent -> linalg/tensor/arith ----------===//
//
// Target-neutral first leg of the back-end spine. Lowers Layer-A atomic ops to
// standard MLIR so independent CPU, CUDA, and future target pipelines can
// continue from the same representation.
// The general elementwise path supports right-aligned/scalar/size-1 broadcast
// through affine input maps.  Frontends may first materialize explicit semantic
// rank insertion (for example Boundary.broadcasts) as tensor.expand_shape.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTTOLINALG
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

/// MLIR arith integer operations consume signless integers.  J's unsigned
/// tensor types remain unsigned at the Intent/ABI boundary; these no-op
/// conversion casts expose the same bit pattern to arith and restore the
/// declared element type at the linalg yield boundary.
static bool isUnsignedInteger(Type type) {
  auto integer = dyn_cast<IntegerType>(type);
  return integer && integer.isUnsigned();
}

static Value toSignlessInteger(OpBuilder &b, Location loc, Value value) {
  auto integer = dyn_cast<IntegerType>(value.getType());
  if (!integer || integer.isSignless())
    return value;
  Type signless = IntegerType::get(value.getContext(), integer.getWidth());
  return b.create<UnrealizedConversionCastOp>(loc, TypeRange{signless},
                                              ValueRange{value})
      .getResult(0);
}

static Value fromSignlessInteger(OpBuilder &b, Location loc, Value value,
                                 Type target) {
  auto integer = dyn_cast<IntegerType>(target);
  if (!integer || integer.isSignless() || value.getType() == target)
    return value;
  return b.create<UnrealizedConversionCastOp>(loc, TypeRange{target},
                                              ValueRange{value})
      .getResult(0);
}

/// Coerce a scalar to an i1 truthiness (x != 0), used by the logical fns.
/// i1 passes through; floats use unordered-not-equal-0; integers use ne-0.
static Value truthy(OpBuilder &b, Location loc, Value v) {
  Type t = v.getType();
  if (t.isInteger(1))
    return v;
  if (isa<FloatType>(t)) {
    Value z = b.create<arith::ConstantOp>(loc, b.getFloatAttr(t, 0.0));
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, v, z);
  }
  Value integer = toSignlessInteger(b, loc, v);
  Type integerType = integer.getType();
  Value z = b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(integerType, 0));
  return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, integer, z);
}

/// Scalar comparison over float OR integer/i1 operands -> an i1. Float uses the
/// ordered predicates (ne = unordered-not-equal) so NaN handling matches numpy
/// (np.equal(nan,nan)=False, np.not_equal(nan,nan)=True); integer/i1 use signed.
static Value emitScalarCmp(OpBuilder &b, Location loc, StringRef fn, Value a,
                           Value c) {
  if (isa<FloatType>(a.getType())) {
    arith::CmpFPredicate p;
    if (fn == "gt")      p = arith::CmpFPredicate::OGT;
    else if (fn == "ge") p = arith::CmpFPredicate::OGE;
    else if (fn == "lt") p = arith::CmpFPredicate::OLT;
    else if (fn == "le") p = arith::CmpFPredicate::OLE;
    else if (fn == "eq") p = arith::CmpFPredicate::OEQ;
    else                 p = arith::CmpFPredicate::UNE; // ne
    return b.create<arith::CmpFOp>(loc, p, a, c);
  }
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
  else
    p = arith::CmpIPredicate::ne;
  return b.create<arith::CmpIOp>(loc, p, lhs, rhs);
}

/// True for the map_binary fns that yield an i1 result: the comparisons plus the
/// logical family and isclose (C2c). Their operands may be float/int/i1.
static bool isI1BinaryFn(StringRef fn) {
  return fn == "gt" || fn == "ge" || fn == "lt" || fn == "le" || fn == "eq" ||
         fn == "ne" || fn == "logical_and" || fn == "logical_or" ||
         fn == "logical_xor" || fn == "and" || fn == "or" || fn == "xor" ||
         fn == "isclose";
}

/// True iff `emitScalarBinary` can lower `fn` for the operand element type.
static bool hasScalarBinary(StringRef fn, Type elementType) {
  if (isa<IntegerType>(elementType))
    return fn == "add" || fn == "sub" || fn == "mul" || fn == "div" ||
           fn == "floordiv" || fn == "remainder" || fn == "max" ||
           fn == "min" || fn == "left_shift" || fn == "right_shift" ||
           fn == "gt" || fn == "ge" || fn == "lt" || fn == "le" ||
           fn == "eq" || fn == "ne" || fn == "logical_and" ||
           fn == "logical_or" || fn == "logical_xor" || fn == "and" ||
           fn == "or" || fn == "xor";
  return fn == "add" || fn == "sub" || fn == "mul" || fn == "div" ||
         fn == "floordiv" || fn == "remainder" || fn == "max" ||
         fn == "min" || fn == "pow" || fn == "atan2" || fn == "copysign" ||
         isI1BinaryFn(fn);
}

/// Emit the scalar arith op for one `map_binary` fn over two scalar operands.
/// Returns nullptr for fns that have no scalar form here yet.
static Value emitScalarBinary(OpBuilder &b, Location loc, StringRef fn, Value a,
                              Value c) {
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
        Type signedType = lhs.getType();
        Value zero = b.create<arith::ConstantOp>(
            loc, b.getIntegerAttr(signedType, 0));
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
    } else if (fn == "max")
      result = unsignedArithmetic
                   ? Value(b.create<arith::MaxUIOp>(loc, lhs, rhs))
                   : Value(b.create<arith::MaxSIOp>(loc, lhs, rhs));
    else if (fn == "min")
      result = unsignedArithmetic
                   ? Value(b.create<arith::MinUIOp>(loc, lhs, rhs))
                   : Value(b.create<arith::MinSIOp>(loc, lhs, rhs));
    else if (fn == "left_shift")
      result = b.create<arith::ShLIOp>(loc, lhs, rhs);
    else if (fn == "right_shift")
      result = unsignedArithmetic
                   ? Value(b.create<arith::ShRUIOp>(loc, lhs, rhs))
                   : Value(b.create<arith::ShRSIOp>(loc, lhs, rhs));
    else if (fn == "gt" || fn == "ge" || fn == "lt" || fn == "le" ||
             fn == "eq" || fn == "ne")
      return emitScalarCmp(b, loc, fn, a, c);
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
    return b.create<arith::MaximumFOp>(loc, a, c);
  if (fn == "min")
    return b.create<arith::MinimumFOp>(loc, a, c);
  if (fn == "pow")
    return b.create<math::PowFOp>(loc, a, c);
  // atan2(y=a, x=c): accurate math.atan2 on the CPU/linalg path (the GPU pass
  // uses the libdevice-safe minimax atan2 in MapEmit.h). angle=atan2(imag,real).
  if (fn == "atan2")
    return b.create<math::Atan2Op>(loc, a, c);
  if (fn == "copysign")
    return b.create<math::CopySignOp>(loc, a, c);
  if (fn == "gt" || fn == "ge" || fn == "lt" || fn == "le" || fn == "eq" ||
      fn == "ne")
    return emitScalarCmp(b, loc, fn, a, c);
  // Logical family: truthy-cast each operand to i1, then the bitwise i1 op.
  if (fn == "logical_and" || fn == "and")
    return b.create<arith::AndIOp>(loc, truthy(b, loc, a), truthy(b, loc, c));
  if (fn == "logical_or" || fn == "or")
    return b.create<arith::OrIOp>(loc, truthy(b, loc, a), truthy(b, loc, c));
  if (fn == "logical_xor" || fn == "xor")
    return b.create<arith::XOrIOp>(loc, truthy(b, loc, a), truthy(b, loc, c));
  if (fn == "isclose") {
    // |a-c| <= atol + rtol*|c|  (torch defaults rtol=1e-5, atol=1e-8).
    Type t = a.getType();
    Value rtol = b.create<arith::ConstantOp>(loc, b.getFloatAttr(t, 1e-5));
    Value atol = b.create<arith::ConstantOp>(loc, b.getFloatAttr(t, 1e-8));
    Value diff = b.create<arith::SubFOp>(loc, a, c);
    Value ad = b.create<math::AbsFOp>(loc, diff);
    Value ac = b.create<math::AbsFOp>(loc, c);
    Value allowed = b.create<arith::AddFOp>(
        loc, atol, b.create<arith::MulFOp>(loc, rtol, ac));
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE, ad, allowed);
  }
  return nullptr;
}

/// True for map_unary fns that yield an i1 result: the finiteness predicates and
/// the i1 complement (C2c), mirroring the map_binary comparison carve-out.
static bool isToI1UnaryFn(StringRef fn) {
  return fn == "isnan" || fn == "isinf" || fn == "isfinite" ||
         fn == "logical_not";
}

/// Emit the scalar op for one `map_unary` fn over a scalar operand.
/// Returns nullptr for fns that have no scalar form here yet.
static Value emitScalarUnary(OpBuilder &b, Location loc, StringRef fn, Value x) {
  if (fn == "identity")
    return x;
  if (fn == "exp")
    return b.create<math::ExpOp>(loc, x);
  if (fn == "log")
    return b.create<math::LogOp>(loc, x);
  if (fn == "log2")
    return b.create<math::Log2Op>(loc, x);
  if (fn == "abs") {
    // Unsigned absolute value is already non-negative.
    if (isUnsignedInteger(x.getType()))
      return x;
    if (isa<IntegerType>(x.getType())) {
      return b.create<math::AbsIOp>(loc, x);
    }
    return b.create<math::AbsFOp>(loc, x);
  }
  if (fn == "sqrt")
    return b.create<math::SqrtOp>(loc, x);
  if (fn == "rsqrt")
    return b.create<math::RsqrtOp>(loc, x);
  if (fn == "neg") {
    if (isUnsignedInteger(x.getType())) {
      Value integer = toSignlessInteger(b, loc, x);
      Value zero = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(integer.getType(), 0));
      Value result = b.create<arith::SubIOp>(loc, zero, integer);
      return fromSignlessInteger(b, loc, result, x.getType());
    }
    if (isa<IntegerType>(x.getType())) {
      Value zero = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(x.getType(), 0));
      return b.create<arith::SubIOp>(loc, zero, x);
    }
    return b.create<arith::NegFOp>(loc, x);
  }
  if (fn == "recip") {
    Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    return b.create<arith::DivFOp>(loc, one, x);
  }
  if (fn == "tanh")
    return b.create<math::TanhOp>(loc, x);
  // The common path emits target-neutral math.* operations. Approximation and
  // target-intrinsic selection belong to the selected target pipeline.
  if (fn == "cos")
    return b.create<math::CosOp>(loc, x);
  if (fn == "sin")
    return b.create<math::SinOp>(loc, x);
  if (fn == "tan")
    return b.create<arith::DivFOp>(loc, b.create<math::SinOp>(loc, x),
                                   b.create<math::CosOp>(loc, x));
  if (fn == "exp2")
    return b.create<math::Exp2Op>(loc, x);
  // atan: accurate math.atan on the CPU/linalg path (the GPU pass uses the
  // libdevice-safe minimax atan in MapEmit.h).
  if (fn == "atan")
    return b.create<math::AtanOp>(loc, x);
  if (fn == "erf")
    return b.create<math::ErfOp>(loc, x);
  if (fn == "erfc") {
    Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
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
    if (isUnsignedInteger(x.getType())) {
      Value integer = toSignlessInteger(b, loc, x);
      Type integerType = integer.getType();
      Value zero = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(integerType, 0));
      Value positive = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, integer, zero);
      Value one = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(integerType, 1));
      Value result = b.create<arith::SelectOp>(loc, positive, one, zero);
      return fromSignlessInteger(b, loc, result, x.getType());
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
  if (fn == "sigmoid") {
    // 1 / (1 + exp(-x))
    Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(x.getType(), 1.0));
    Value negx = b.create<arith::NegFOp>(loc, x);
    Value e = b.create<math::ExpOp>(loc, negx);
    Value den = b.create<arith::AddFOp>(loc, one, e);
    return b.create<arith::DivFOp>(loc, one, den);
  }
  // i1-result predicate fns (C2c). isnan: NaN is the only value unordered with
  // itself. isinf: |x| == +inf. isfinite: |x| ordered-not-equal +inf (ONE is
  // false for NaN, so it already excludes NaN). logical_not: the i1 complement.
  if (fn == "isnan")
    return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO, x, x);
  if (fn == "isinf" || fn == "isfinite") {
    Value inf = b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(x.getType(),
                            std::numeric_limits<double>::infinity()));
    Value ax = b.create<math::AbsFOp>(loc, x);
    return b.create<arith::CmpFOp>(loc,
                                   fn == "isinf" ? arith::CmpFPredicate::OEQ
                                                 : arith::CmpFPredicate::ONE,
                                   ax, inf);
  }
  if (fn == "logical_not") {
    Value t = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
    return b.create<arith::XOrIOp>(loc, x, t);
  }
  return nullptr;
}

/// True iff `emitScalarUnary` can lower `fn` (the guard for MapUnaryLowering).
static bool hasScalarUnary(StringRef fn) {
  return fn == "identity" || fn == "exp" || fn == "log" || fn == "log2" || fn == "abs" ||
         fn == "sqrt" ||
         fn == "rsqrt" || fn == "neg" || fn == "recip" || fn == "tanh" ||
         fn == "cos" || fn == "sin" || fn == "tan" || fn == "exp2" ||
         fn == "atan" || fn == "erf" || fn == "erfc" || fn == "sigmoid" ||
         fn == "acos" || fn == "asin" || fn == "cbrt" || fn == "ceil" ||
         fn == "floor" || fn == "round" || fn == "roundeven" || fn == "trunc" ||
         fn == "sign" || fn == "log10" || fn == "log1p" || fn == "expm1" ||
         isToI1UnaryFn(fn);
}

/// Create a destination tensor while preserving runtime extents from one of
/// the operation's ranked tensor operands.  Dynamic Ragged member results use
/// this helper before entering an ordinary linalg.generic consumer; an empty
/// tensor with omitted dynamic-size operands is not a complete lowering.
static Value createTensorEmptyLike(OpBuilder &builder, Location loc,
                                   RankedTensorType resultType,
                                   ValueRange shapeSources) {
  if (resultType.hasStaticShape())
    return builder.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                           resultType.getElementType());

  SmallVector<Value> dynamicSizes;
  for (auto [axis, dimension] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(dimension))
      continue;
    Value size;
    for (Value source : shapeSources) {
      auto sourceType = dyn_cast<RankedTensorType>(source.getType());
      if (!sourceType || sourceType.getRank() != resultType.getRank())
        continue;
      size = builder.create<tensor::DimOp>(loc, source, axis);
      break;
    }
    if (!size)
      return Value();
    dynamicSizes.push_back(size);
  }
  return builder.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                         resultType.getElementType(),
                                         dynamicSizes);
}

struct MapUnaryLowering : OpRewritePattern<MapUnaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapUnaryOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    StringRef fn = op.getFn();
    bool toI1 = isToI1UnaryFn(fn);
    if (toI1) {
      if (!resTy.getElementType().isInteger(1))
        return rewriter.notifyMatchFailure(op, "i1 predicate needs i1 result");
    } else if (!isa<FloatType>(resTy.getElementType()) &&
               !(fn == "identity" || fn == "neg" || fn == "abs" ||
                 fn == "sign")) {
      return rewriter.notifyMatchFailure(op, "non-float unary not implemented");
    }
    if (!hasScalarUnary(fn))
      return rewriter.notifyMatchFailure(op, "fn has no scalar lowering");

    Location loc = op.getLoc();
    Value init = createTensorEmptyLike(rewriter, loc, resTy,
                                       ValueRange{op.getInput()});
    if (!init)
      return rewriter.notifyMatchFailure(
          op, "dynamic unary result has no operand-provided extent");
    unsigned rank = resTy.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getInput()}, ValueRange{init},
        maps, iters, [&](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, emitScalarUnary(b, l, fn, args[0]));
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower `intent.iota` to the backend-neutral rank-1 linalg realization used by
/// production lowering. The scalar value is the current iteration
/// index, cast to the declared integer width when necessary; no source-specific
/// tile or program-id structure is materialized.
struct IotaLowering : OpRewritePattern<IotaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IotaOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    Type elemTy = resTy.getElementType();
    if (resTy.getRank() != 1 || !resTy.hasStaticShape() ||
        !elemTy.isIntOrIndex())
      return rewriter.notifyMatchFailure(
          op, "expected a static rank-1 integer/index result");

    Location loc = op.getLoc();
    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(), elemTy);
    SmallVector<AffineMap> maps{rewriter.getMultiDimIdentityMap(1)};
    SmallVector<utils::IteratorType> iters(1, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{}, ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange) {
          Value index = b.create<linalg::IndexOp>(l, 0);
          Value value;
          if (elemTy.isIndex()) {
            value = index;
          } else if (auto integer = dyn_cast<IntegerType>(elemTy);
                     integer && integer.isUnsigned()) {
            Type signless = IntegerType::get(l.getContext(), integer.getWidth());
            Value cast = b.create<arith::IndexCastOp>(l, signless, index);
            value = fromSignlessInteger(b, l, cast, elemTy);
          } else {
            value = b.create<arith::IndexCastOp>(l, elemTy, index).getResult();
          }
          b.create<linalg::YieldOp>(l, value);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Convert one scalar integer tensor value to an MLIR index.  Ragged offsets
/// and the outer selector are allowed to use different integer widths and
/// signedness, while tensor.extract_slice requires index-typed dynamic bounds.
static Value raggedToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  auto integer = dyn_cast<IntegerType>(value.getType());
  if (!integer)
    return Value();
  return builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(), toSignlessInteger(builder, loc, value));
}

/// Convert an MLIR index back to the declared scalar tensor element type.
static Value raggedFromIndex(OpBuilder &builder, Location loc, Value value,
                             Type target) {
  if (target.isIndex())
    return value;
  auto integer = dyn_cast<IntegerType>(target);
  if (!integer)
    return Value();
  Type signless = IntegerType::get(builder.getContext(), integer.getWidth());
  Value cast = builder.create<arith::IndexCastOp>(loc, signless, value);
  return fromSignlessInteger(builder, loc, cast, target);
}

/// Lower the tuple-shaped descriptor producer after all of its semantic users
/// have been rewritten.  A live tuple cannot cross the target ABI: the only
/// supported representation is the two carrier tensors consumed directly by
/// RaggedOuterLowering/RaggedMemberLowering below.
struct RaggedCleanupLowering : OpRewritePattern<RaggedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RaggedOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResult().use_empty())
      return rewriter.notifyMatchFailure(
          op, "ragged descriptor still has an unlowered consumer");
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower ragged_outer to offsets.size - 1 as a rank-0 tensor of the authored
/// offsets element type.  The descriptor is intentionally required to come
/// from RaggedOp: this keeps tuple values from becoming an implicit ABI and
/// makes a future external descriptor ABI an explicit extension point.
struct RaggedOuterLowering : OpRewritePattern<RaggedOuterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RaggedOuterOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getRagged().getDefiningOp<RaggedOp>();
    if (!producer)
      return rewriter.notifyMatchFailure(
          op, "ragged_outer requires a local ragged descriptor producer");
    auto offsetsTy = dyn_cast<RankedTensorType>(producer.getOffsets().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!offsetsTy || !resultTy || !isRaggedCarrierTensorType(offsetsTy) ||
        resultTy.getRank() != 0 ||
        !isRaggedCarrierElementType(resultTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "invalid ragged outer tensor types");

    Location loc = op.getLoc();
    Value count;
    if (offsetsTy.isDynamicDim(0))
      count = rewriter.create<tensor::DimOp>(loc, producer.getOffsets(), 0);
    else
      count = rewriter.create<arith::ConstantIndexOp>(loc,
                                                      offsetsTy.getDimSize(0));
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value outer = rewriter.create<arith::SubIOp>(loc, count, one);
    Value scalar = raggedFromIndex(rewriter, loc, outer,
                                   resultTy.getElementType());
    if (!scalar)
      return rewriter.notifyMatchFailure(
          op, "ragged outer result is not an integer/index type");
    Value result = rewriter.create<tensor::FromElementsOp>(
        loc, resultTy, ValueRange{scalar});
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower ragged_member to a dynamic one-dimensional tensor.extract_slice:
/// indices[offsets[i] : offsets[i + 1]].  Both offsets and the selector retain
/// their authored integer ABI types; only the slice bounds are converted to
/// MLIR index values.
struct RaggedMemberLowering : OpRewritePattern<RaggedMemberOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RaggedMemberOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getRagged().getDefiningOp<RaggedOp>();
    if (!producer)
      return rewriter.notifyMatchFailure(
          op, "ragged_member requires a local ragged descriptor producer");
    auto offsetsTy = dyn_cast<RankedTensorType>(producer.getOffsets().getType());
    auto indicesTy = dyn_cast<RankedTensorType>(producer.getIndices().getType());
    auto indexTy = dyn_cast<RankedTensorType>(op.getIndex().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!offsetsTy || !indicesTy || !indexTy || !resultTy ||
        !isRaggedCarrierTensorType(offsetsTy) ||
        !isRaggedCarrierTensorType(indicesTy) ||
        indexTy.getRank() != 0 ||
        !isRaggedCarrierElementType(indexTy.getElementType()) ||
        resultTy.getRank() != 1 ||
        !isRaggedCarrierElementType(resultTy.getElementType()) ||
        !resultTy.isDynamicDim(0))
      return rewriter.notifyMatchFailure(op, "invalid ragged member tensor types");

    Location loc = op.getLoc();
    Value selector = rewriter.create<tensor::ExtractOp>(
        loc, op.getIndex(), ValueRange{});
    selector = raggedToIndex(rewriter, loc, selector);
    if (!selector)
      return rewriter.notifyMatchFailure(
          op, "ragged member selector is not an integer/index type");

    Value nextSelector = rewriter.create<arith::AddIOp>(
        loc, selector, rewriter.create<arith::ConstantIndexOp>(loc, 1));
    Value startRaw = rewriter.create<tensor::ExtractOp>(
        loc, producer.getOffsets(), ValueRange{selector});
    Value endRaw = rewriter.create<tensor::ExtractOp>(
        loc, producer.getOffsets(), ValueRange{nextSelector});
    Value start = raggedToIndex(rewriter, loc, startRaw);
    Value end = raggedToIndex(rewriter, loc, endRaw);
    if (!start || !end)
      return rewriter.notifyMatchFailure(
          op, "ragged offsets are not integer/index values");
    Value length = rewriter.create<arith::SubIOp>(loc, end, start);

    SmallVector<OpFoldResult> offsets{start};
    SmallVector<OpFoldResult> sizes{length};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
    Value result = rewriter.create<tensor::ExtractSliceOp>(
        loc, resultTy, producer.getIndices(), offsets, sizes, strides);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Materialize the static shape operand required by tensor.reshape.  The
/// Layer-A JSON form may use symbolic extents, but this MLIR front end is
/// instantiated at concrete dimensions, so the target-neutral lowering keeps
/// the result type and shape attribute in lockstep.
struct ReshapeLowering : OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto inputTy = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputTy || !resultTy || !inputTy.hasStaticShape() ||
        !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "reshape lowering currently requires static ranked tensors");
    if (static_cast<int64_t>(op.getShape().size()) != resultTy.getRank())
      return rewriter.notifyMatchFailure(op, "shape rank differs from result rank");

    SmallVector<int64_t> extents;
    extents.reserve(op.getShape().size());
    for (Attribute attr : op.getShape())
      extents.push_back(cast<IntegerAttr>(attr).getInt());

    auto shapeTy = RankedTensorType::get(
        {static_cast<int64_t>(extents.size())}, rewriter.getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeTy, extents);
    Value shape = rewriter.create<arith::ConstantOp>(op.getLoc(), shapeAttr);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, resultTy, op.getInput(),
                                                  shape);
    return success();
  }
};

/// Eliminate a full-cover identity scatter before lowering its index graph.
///
/// For an along-axis overwrite whose index is the right-aligned iota plus a
/// zero splat, every source element is written exactly once to its own logical
/// coordinate.  The functional result is therefore exactly `src`, independent
/// of the old `data` value.  This canonical form occurs in certified graphs
/// that expose an explicit padded-output update even when the active valuation
/// has no padding.  Keep the proof deliberately narrow: last axis, identical
/// static shapes, overwrite-only, and the literal zero+iota index construction.
struct IdentityScatterLowering : OpRewritePattern<ScatterOp> {
  explicit IdentityScatterLowering(MLIRContext *context)
      : OpRewritePattern<ScatterOp>(context, PatternBenefit(2)) {}

  LogicalResult matchAndRewrite(ScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("atomic") || op->hasAttr("ordering") ||
        op->hasAttr("memory_scope") || op->hasAttr("cas_retry"))
      return rewriter.notifyMatchFailure(
          op, "identity scatter lowering does not consume atomic metadata");
    if (op.getMode() != "along_axis" || op.getReduce() != "none")
      return failure();
    auto dataTy = dyn_cast<RankedTensorType>(op.getData().getType());
    auto indexTy = dyn_cast<RankedTensorType>(op.getIndex().getType());
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!dataTy || !indexTy || !srcTy || !resultTy ||
        !dataTy.hasStaticShape() || dataTy.getShape() != indexTy.getShape() ||
        dataTy.getShape() != srcTy.getShape() ||
        dataTy.getShape() != resultTy.getShape() || dataTy.getRank() == 0 ||
        static_cast<int64_t>(op.getAxis()) != dataTy.getRank() - 1)
      return failure();

    auto add = op.getIndex().getDefiningOp<MapBinaryOp>();
    if (!add || add.getFn() != "add")
      return failure();
    IotaOp iota = add.getLhs().getDefiningOp<IotaOp>();
    Value zeroValue = add.getRhs();
    if (!iota) {
      iota = add.getRhs().getDefiningOp<IotaOp>();
      zeroValue = add.getLhs();
    }
    auto iotaTy = iota ? dyn_cast<RankedTensorType>(iota.getResult().getType())
                       : RankedTensorType();
    auto zero = zeroValue.getDefiningOp<arith::ConstantOp>();
    auto dense = zero ? dyn_cast<DenseElementsAttr>(zero.getValue())
                      : DenseElementsAttr();
    if (!iota || !iotaTy || iotaTy.getRank() != 1 ||
        iotaTy.getDimSize(0) != dataTy.getDimSize(dataTy.getRank() - 1) ||
        !dense || !dense.isSplat() ||
        !isa<IntegerType>(dense.getElementType()) ||
        !dense.getSplatValue<APInt>().isZero())
      return failure();

    rewriter.replaceOp(op, op.getSrc());
    return success();
  }
};

/// Build the linalg input indexing map for a (possibly broadcasting) operand
/// against a result of rank `R`. Operand dims align right (numpy rules); a dim
/// whose operand extent is 1 but result extent is >1 broadcasts (constant-0
/// index). Returns the operand's affine map over the R-dim iteration domain.
static AffineMap broadcastMap(MLIRContext *ctx, ArrayRef<int64_t> opShape,
                              ArrayRef<int64_t> resShape) {
  unsigned R = resShape.size(), r = opShape.size();
  SmallVector<AffineExpr> exprs;
  for (unsigned i = 0; i < r; ++i) {
    unsigned d = R - r + i; // right-aligned result dim
    if (opShape[i] == 1 && resShape[d] != 1)
      exprs.push_back(getAffineConstantExpr(0, ctx));
    else
      exprs.push_back(getAffineDimExpr(d, ctx));
  }
  return AffineMap::get(R, 0, exprs, ctx);
}

struct MapBinaryLowering : OpRewritePattern<MapBinaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapBinaryOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    auto lhsTy = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsTy = cast<RankedTensorType>(op.getRhs().getType());

    StringRef fnName = op.getFn();
    bool toI1 = isI1BinaryFn(fnName);
    // The i1-result fns (comparisons / logical / isclose) accept the supported
    // operand kinds and yield an i1; other fns preserve float or integer type.
    if (toI1) {
      (void)lhsTy;
      (void)rhsTy;
    } else if (!isa<FloatType, IntegerType>(resTy.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "elementwise result type is not float or integer");
    }
    if (!hasScalarBinary(fnName, lhsTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "fn has no scalar lowering");

    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    unsigned rank = resTy.getRank();
    // Result is the broadcast shape; each input maps in (identity when equal,
    // constant-0 on broadcast dims). Output is identity.
    SmallVector<AffineMap> maps = {
        broadcastMap(ctx, lhsTy.getShape(), resTy.getShape()),
        broadcastMap(ctx, rhsTy.getShape(), resTy.getShape()),
        rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

    Value init = createTensorEmptyLike(
        rewriter, loc, resTy, ValueRange{op.getLhs(), op.getRhs()});
    if (!init)
      return rewriter.notifyMatchFailure(
          op, "dynamic binary result has no operand-provided extent");
    StringRef fn = op.getFn();
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, emitScalarBinary(b, l, fn, args[0], args[1]));
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower `intent.select` (numpy where) to a linalg.generic over `arith.select`.
/// cond is i1; cond/x/y broadcast numpy-style into the result shape (so a rank-0
/// scalar branch or an [M,N] i1 condition over rank-0 x/y both work). C2c: select
/// now lowers on the general linalg path (was GPU-pass-only), so
/// map+comparison+select
/// composites (masked_fill / threshold / nan_to_num / logical_*) reach zero
/// residual.
struct SelectLowering : OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    auto condTy = cast<RankedTensorType>(op.getCond().getType());
    auto xTy = cast<RankedTensorType>(op.getX().getType());
    auto yTy = cast<RankedTensorType>(op.getY().getType());
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    unsigned rank = resTy.getRank();
    SmallVector<AffineMap> maps = {
        broadcastMap(ctx, condTy.getShape(), resTy.getShape()),
        broadcastMap(ctx, xTy.getShape(), resTy.getShape()),
        broadcastMap(ctx, yTy.getShape(), resTy.getShape()),
        rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    Value init = createTensorEmptyLike(
        rewriter, loc, resTy,
        ValueRange{op.getCond(), op.getX(), op.getY()});
    if (!init)
      return rewriter.notifyMatchFailure(
          op, "dynamic select result has no operand-provided extent");
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getCond(), op.getX(), op.getY()},
        ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value sel = b.create<arith::SelectOp>(l, args[0], args[1], args[2]);
          b.create<linalg::YieldOp>(l, sel);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower `intent.gather` to a parallel linalg.generic whose scalar body reads
/// the runtime index tensor and then performs a data-dependent tensor.extract.
/// This is the correctness-oriented CPU/linalg path; the specialized GPU path
/// remains `convert-intent-gather-to-gpu`.
struct GatherLowering : OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto dataTy = cast<RankedTensorType>(op.getData().getType());
    auto indexTy = cast<RankedTensorType>(op.getIndex().getType());
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    if (!resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic gather result unsupported");

    int64_t axis = op.getAxis();
    int64_t dataRank = dataTy.getRank();
    if (axis < 0)
      axis += dataRank;
    if (axis < 0 || axis >= dataRank)
      return rewriter.notifyMatchFailure(op, "gather axis out of range");

    bool alongAxis = op.getMode() == "along_axis";
    if (!alongAxis && op.getMode() != "take")
      return rewriter.notifyMatchFailure(op, "unknown gather mode");

    Type indexElementTy = indexTy.getElementType();
    if (!isa<IntegerType, IndexType>(indexElementTy))
      return rewriter.notifyMatchFailure(op,
                                         "gather index must be integer/index");

    unsigned resultRank = resultTy.getRank();
    unsigned indexRank = indexTy.getRank();
    if (alongAxis) {
      if (indexRank != static_cast<unsigned>(dataRank) ||
          resultRank != static_cast<unsigned>(dataRank))
        return rewriter.notifyMatchFailure(
            op, "along_axis gather requires equal data/index/result ranks");
    } else if (resultRank !=
               static_cast<unsigned>(dataRank - 1) + indexRank) {
      return rewriter.notifyMatchFailure(
          op, "take gather result rank does not match data/index ranks");
    }

    Location loc = op.getLoc();
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultTy.getElementType());
    SmallVector<AffineMap> maps{
        rewriter.getMultiDimIdentityMap(resultRank)};
    SmallVector<utils::IteratorType> iters(
        resultRank, utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultTy}, ValueRange{}, ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange) {
          SmallVector<Value> resultCoords;
          for (unsigned d = 0; d < resultRank; ++d)
            resultCoords.push_back(b.create<linalg::IndexOp>(l, d));

          SmallVector<Value> indexCoords;
          if (alongAxis) {
            indexCoords.append(resultCoords.begin(), resultCoords.end());
          } else {
            for (unsigned d = 0; d < indexRank; ++d)
              indexCoords.push_back(resultCoords[axis + d]);
          }
          Value rawIndex =
              b.create<tensor::ExtractOp>(l, op.getIndex(), indexCoords);
          Value gathered = rawIndex;
          if (!isa<IndexType>(rawIndex.getType())) {
            Value integer = toSignlessInteger(b, l, rawIndex);
            gathered = b.create<arith::IndexCastOp>(
                l, b.getIndexType(), integer);
          }

          SmallVector<Value> dataCoords;
          if (alongAxis) {
            dataCoords.append(resultCoords.begin(), resultCoords.end());
            dataCoords[axis] = gathered;
          } else {
            for (unsigned d = 0; d < dataTy.getRank(); ++d) {
              if (d < static_cast<unsigned>(axis))
                dataCoords.push_back(resultCoords[d]);
              else if (d == static_cast<unsigned>(axis))
                dataCoords.push_back(gathered);
              else
                dataCoords.push_back(resultCoords[d - 1 + indexRank]);
            }
          }
          Value value =
              b.create<tensor::ExtractOp>(l, op.getData(), dataCoords);
          b.create<linalg::YieldOp>(l, value);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Scalar combiner for a `reduce` fn. Numeric reductions preserve the operand
/// element kind: float uses the floating-point arith ops, while signless
/// integers use signed max/min and integer add/mul. Boolean all/any remain the
/// dedicated i1 reductions.
static Value emitReduceCombine(OpBuilder &b, Location loc, StringRef fn, Value a,
                               Value c,
                               bool allowFloatReassociation = false) {
  auto integer = dyn_cast<IntegerType>(a.getType());
  bool isInteger = integer != nullptr;
  bool unsignedInteger = integer && integer.isUnsigned();
  Value lhs = isInteger ? toSignlessInteger(b, loc, a) : a;
  Value rhs = isInteger ? toSignlessInteger(b, loc, c) : c;
  arith::FastMathFlags floatFlags =
      allowFloatReassociation ? arith::FastMathFlags::reassoc
                              : arith::FastMathFlags::none;
  if (fn == "sum" || fn == "mean") {
    if (isInteger)
      return fromSignlessInteger(b, loc,
                                 b.create<arith::AddIOp>(loc, lhs, rhs),
                                 a.getType());
    return b.create<arith::AddFOp>(loc, a, c, floatFlags);
  }
  if (fn == "max") {
    if (isInteger)
      return fromSignlessInteger(
          b, loc,
          unsignedInteger ? Value(b.create<arith::MaxUIOp>(loc, lhs, rhs))
                          : Value(b.create<arith::MaxSIOp>(loc, lhs, rhs)),
          a.getType());
    return b.create<arith::MaxNumFOp>(loc, a, c);
  }
  if (fn == "min") {
    if (isInteger)
      return fromSignlessInteger(
          b, loc,
          unsignedInteger ? Value(b.create<arith::MinUIOp>(loc, lhs, rhs))
                          : Value(b.create<arith::MinSIOp>(loc, lhs, rhs)),
          a.getType());
    return b.create<arith::MinNumFOp>(loc, a, c);
  }
  if (fn == "prod") {
    if (isInteger)
      return fromSignlessInteger(b, loc,
                                 b.create<arith::MulIOp>(loc, lhs, rhs),
                                 a.getType());
    return b.create<arith::MulFOp>(loc, a, c, floatFlags);
  }
  if (fn == "all")
    return b.create<arith::AndIOp>(loc, a, c);
  if (fn == "any")
    return b.create<arith::OrIOp>(loc, a, c);
  return nullptr;
}

/// True for the boolean reductions (i1 input/output).
static bool isBoolReduce(StringRef fn) { return fn == "all" || fn == "any"; }

static bool isNumericReduce(StringRef fn) {
  return fn == "sum" || fn == "mean" || fn == "max" || fn == "min" ||
         fn == "prod";
}

/// Build the identity constant for a reduce fn at the given element type (float
///, integer, or i1). Integer max/min follow the signed interpretation used by
/// the scalar combiner.
static Value reduceIdentityConst(OpBuilder &b, Location loc, StringRef fn,
                                 Type elemTy) {
  if (elemTy.isInteger(1))
    return b.create<arith::ConstantOp>(loc, b.getBoolAttr(fn == "all"));

  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned width = intTy.getWidth();
    llvm::APInt ident(width, 0);
    if (fn == "prod")
      ident = llvm::APInt(width, 1);
    else if (fn == "max")
      ident = isUnsignedInteger(elemTy)
                  ? llvm::APInt::getMinValue(width)
                  : llvm::APInt::getSignedMinValue(width);
    else if (fn == "min")
      ident = isUnsignedInteger(elemTy)
                  ? llvm::APInt::getMaxValue(width)
                  : llvm::APInt::getSignedMaxValue(width);
    return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(elemTy, ident));
  }

  double ident = 0.0;
  if (fn == "prod")
    ident = 1.0;
  else if (fn == "max")
    ident = -std::numeric_limits<double>::infinity();
  else if (fn == "min")
    ident = std::numeric_limits<double>::infinity();
  return b.create<arith::ConstantOp>(loc, b.getFloatAttr(elemTy, ident));
}

/// Lower `intent.reduce` to a linalg.generic reduction. Reduced axes become
/// reduction iterators; the output drops them. If the result keeps the reduced
/// axes as size-1 (keepdims, result rank == input rank), a tensor.expand_shape
/// re-inserts them so downstream broadcasts (e.g. S - rowmax) line up.
struct ReduceLowering : OpRewritePattern<ReduceOp> {
  ReduceLowering(MLIRContext *context, bool reassociateFloatReductions)
      : OpRewritePattern<ReduceOp>(context),
        reassociateFloatReductions(reassociateFloatReductions) {}

  bool reassociateFloatReductions;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto inTy = cast<RankedTensorType>(op.getInput().getType());
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    Type elemTy = inTy.getElementType();
    StringRef fn = op.getFn();
    bool allowFloatReassociation =
        reassociateFloatReductions &&
        op.getAssoc() == "float_nondeterministic";
    // Float supports numeric reductions; signless integer supports the
    // non-mean numeric reductions; i1 is reserved for all/any.
    bool supported =
        (isa<FloatType>(elemTy) && isNumericReduce(fn)) ||
        (isa<IntegerType>(elemTy) && !elemTy.isInteger(1) &&
         isNumericReduce(fn) && fn != "mean") ||
        (elemTy.isInteger(1) && isBoolReduce(fn));
    if (!supported)
      return rewriter.notifyMatchFailure(op, "unsupported reduce element type");

    // Ragged member produces a runtime-length rank-one tensor.  Keep this
    // target-neutral path in the common lowering instead of leaving a high-
    // level reduce for one backend to rediscover.  The dynamic form is the
    // exact one-dimensional reduction used by the J contract: it reduces the
    // complete member vector to a scalar result.  Static multi-dimensional
    // reductions continue through the linalg.generic path below.
    if (!inTy.hasStaticShape()) {
      int64_t dynamicAxis =
          op.getAxes().size() == 1
              ? cast<IntegerAttr>(op.getAxes()[0]).getInt()
              : std::numeric_limits<int64_t>::min();
      if (inTy.getRank() != 1 || resTy.getRank() != 0 ||
          (dynamicAxis != 0 && dynamicAxis != -1) ||
          resTy.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(
            op, "dynamic reduce currently requires a rank-one full reduction");

      Location loc = op.getLoc();
      Value lower = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value upper = rewriter.create<tensor::DimOp>(loc, op.getInput(), 0);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      StringRef combineFn = fn == "mean" ? "sum" : fn;
      Value identity = reduceIdentityConst(rewriter, loc, combineFn, elemTy);
      auto reduction = rewriter.create<scf::ForOp>(
          loc, lower, upper, step, ValueRange{identity},
          [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
            Value element = b.create<tensor::ExtractOp>(
                l, op.getInput(), ValueRange{iv});
            Value combined = emitReduceCombine(
                b, l, combineFn, iterArgs.front(), element,
                allowFloatReassociation);
            b.create<scf::YieldOp>(l, combined);
          });
      Value scalar = reduction.getResult(0);
      if (fn == "mean") {
        auto floatTy = dyn_cast<FloatType>(elemTy);
        if (!floatTy)
          return rewriter.notifyMatchFailure(
              op, "dynamic mean requires a floating-point member tensor");
        Value count = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI64Type(), upper);
        Value countFloat = rewriter.create<arith::SIToFPOp>(
            loc, floatTy, count);
        scalar = rewriter.create<arith::DivFOp>(loc, scalar, countFloat);
      }
      Value result = rewriter.create<tensor::FromElementsOp>(
          loc, resTy, ValueRange{scalar});
      rewriter.replaceOp(op, result);
      return success();
    }

    unsigned R = inTy.getRank();
    SmallVector<bool> isReduced(R, false);
    for (Attribute a : op.getAxes()) {
      int64_t ax = cast<IntegerAttr>(a).getInt();
      if (ax < 0)
        ax += R;
      if (ax < 0 || ax >= (int64_t)R)
        return rewriter.notifyMatchFailure(op, "reduce axis out of range");
      isReduced[ax] = true;
    }

    // reducedTy = input with reduced axes removed.
    SmallVector<int64_t> reducedShape;
    for (unsigned d = 0; d < R; ++d)
      if (!isReduced[d])
        reducedShape.push_back(inTy.getDimSize(d));
    auto reducedTy = RankedTensorType::get(reducedShape, elemTy);
    unsigned r = reducedShape.size();

    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    // Full reduction (every axis reduced -> no surviving dims) to an all-size-1
    // keepdims tensor [1,..,1] (result rank == input rank).
    //
    // C2b cubin fix: a naive all-reduction linalg.generic (all iterators
    // "reduction", no parallel dim) is SYNTACTICALLY valid and reaches zero
    // residual, but does NOT lower to a runnable cubin — with no parallel
    // iterator, `-convert-parallel-loops-to-gpu` cannot outline it, so the whole
    // reduction stays a HOST scf.for nest (only the identity-fill is mapped to
    // GPU). The launched cubin then returns only the fill identity (-inf / 0),
    // silently wrong. Verified end-to-end (amax -> -inf) before this fix.
    //
    // Fix: expand the input with a leading UNIT parallel dim
    // ([d0,..] -> [1, d0, ..]) and reduce the trailing R axes, keeping dim0
    // parallel. Now `-convert-parallel-loops-to-gpu` maps the size-1 parallel
    // loop to a 1-block grid and outlines the inner reduction scf.for INTO the
    // GPU kernel. Collapse the [1,1,..,1] (rank R+1) result back to the keepdims
    // [1,..,1] (rank R) declared shape, or collapse all unit dims to tensor<1xT>,
    // extract its only element, and rebuild a rank-0 tensor when the ABI declares
    // a scalar result. Verified end-to-end (amax/vector_norm/mse_loss and SCRC
    // rank-0 sum/mean-final).
    if (r == 0) {
      bool scalarResult = resTy.getRank() == 0;
      if (!scalarResult && resTy.getRank() != (int64_t)R)
        return rewriter.notifyMatchFailure(
            op, "full-reduction result must be rank-0 or keepdims rank");
      for (unsigned d = 0; d < R; ++d)
        if (inTy.isDynamicDim(d))
          return rewriter.notifyMatchFailure(op, "full reduce over dynamic extent");

      unsigned RE = R + 1;  // expanded rank: [1, d0, .., d_{R-1}]

      // expand reassociation: source dim0 -> result {0,1}; source dim i>0 ->
      // result {i+1}.
      SmallVector<ReassociationIndices> expandRe;
      expandRe.push_back({0, 1});
      for (unsigned d = 1; d < R; ++d)
        expandRe.push_back({(int64_t)d + 1});
      SmallVector<int64_t> expShape;
      expShape.push_back(1);
      for (unsigned d = 0; d < R; ++d)
        expShape.push_back(inTy.getDimSize(d));
      auto expTy = RankedTensorType::get(expShape, elemTy);
      Value xin = rewriter.create<tensor::ExpandShapeOp>(loc, expTy,
                                                         op.getInput(), expandRe);

      // reduce: iterators [parallel, reduction*R]; out [1,1,..,1] (RE ones),
      // out map = (d0, 0, .., 0) so the leading parallel dim survives.
      SmallVector<int64_t> redShape(RE, 1);
      auto redTy = RankedTensorType::get(redShape, elemTy);
      SmallVector<utils::IteratorType> iters(RE, utils::IteratorType::reduction);
      iters[0] = utils::IteratorType::parallel;
      SmallVector<AffineExpr> outExprs;
      outExprs.push_back(getAffineDimExpr(0, ctx));
      for (unsigned d = 1; d < RE; ++d)
        outExprs.push_back(getAffineConstantExpr(0, ctx));
      SmallVector<AffineMap> maps = {rewriter.getMultiDimIdentityMap(RE),
                                     AffineMap::get(RE, 0, outExprs, ctx)};

      Value identVal = reduceIdentityConst(rewriter, loc, fn, elemTy);
      Value empty = rewriter.create<tensor::EmptyOp>(loc, redShape, elemTy);
      Value init = rewriter.create<linalg::FillOp>(loc, ValueRange{identVal},
                                                   ValueRange{empty}).getResult(0);
      Value reduced =
          rewriter
              .create<linalg::GenericOp>(
                  loc, TypeRange{redTy}, ValueRange{xin}, ValueRange{init},
                  maps, iters,
                  [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(
                        l, emitReduceCombine(b, l, fn, args[1], args[0],
                                             allowFloatReassociation));
                  })
              .getResult(0);

      // mean = sum / (product of all extents), an elementwise scale over the
      // all-parallel [1,..,1] result.
      if (fn == "mean") {
        int64_t count = 1;
        for (unsigned d = 0; d < R; ++d)
          count *= inTy.getDimSize(d);
        Value inv = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(elemTy, 1.0 / (double)count));
        Value empty2 = rewriter.create<tensor::EmptyOp>(loc, redShape, elemTy);
        SmallVector<AffineMap> smaps(2, rewriter.getMultiDimIdentityMap(RE));
        SmallVector<utils::IteratorType> siters(RE,
                                                utils::IteratorType::parallel);
        reduced = rewriter
                      .create<linalg::GenericOp>(
                          loc, TypeRange{redTy}, ValueRange{reduced},
                          ValueRange{empty2}, smaps, siters,
                          [&](OpBuilder &b, Location l, ValueRange args) {
                            Value scaled =
                                b.create<arith::MulFOp>(l, args[0], inv);
                            b.create<linalg::YieldOp>(l, scaled);
                          })
                      .getResult(0);
      }

      if (scalarResult) {
        SmallVector<ReassociationIndices> collapseAll(1);
        for (unsigned d = 0; d < RE; ++d)
          collapseAll[0].push_back(d);
        auto oneTy = RankedTensorType::get({1}, elemTy);
        Value one = rewriter.create<tensor::CollapseShapeOp>(
            loc, oneTy, reduced, collapseAll);
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value scalar =
            rewriter.create<tensor::ExtractOp>(loc, one, ValueRange{zero});
        Value out = rewriter.create<tensor::FromElementsOp>(
            loc, resTy, ValueRange{scalar});
        rewriter.replaceOp(op, out);
        return success();
      }

      // collapse [1,1,..,1] (rank R+1) -> keepdims [1,..,1] (rank R): merge the
      // leading two unit dims, keep the rest. (For R==1 this collapses [1,1]
      // -> [1] via the single group {0,1}.)
      SmallVector<ReassociationIndices> collapseRe;
      collapseRe.push_back({0, 1});
      for (unsigned d = 1; d < R; ++d)
        collapseRe.push_back({(int64_t)d + 1});
      Value out = rewriter.create<tensor::CollapseShapeOp>(loc, resTy, reduced,
                                                           collapseRe);
      rewriter.replaceOp(op, out);
      return success();
    }

    // Put every surviving parallel iterator before every reduction iterator,
    // independent of the source-axis order. This is the common structured form:
    // target pipelines retain an outer parallel iteration space and an inner
    // reduction instead of having to recover parallelism from a serial outer
    // loop.
    //
    // Indexing maps retain the original operand/output dimension order: loop
    // dim -> input dim is a permutation; output dims remain the non-reduced
    // input dims in their original order.
    SmallVector<unsigned> loopPosForInput(R);
    unsigned nextParallel = 0;
    unsigned nextReduction = r;
    SmallVector<AffineExpr> outExprs;
    for (unsigned d = 0; d < R; ++d) {
      if (isReduced[d])
        loopPosForInput[d] = nextReduction++;
      else {
        loopPosForInput[d] = nextParallel++;
        outExprs.push_back(getAffineDimExpr(loopPosForInput[d], ctx));
      }
    }
    SmallVector<utils::IteratorType> iters(
        R, utils::IteratorType::reduction);
    for (unsigned d = 0; d < r; ++d)
      iters[d] = utils::IteratorType::parallel;
    SmallVector<AffineExpr> inputExprs;
    inputExprs.reserve(R);
    for (unsigned d = 0; d < R; ++d)
      inputExprs.push_back(getAffineDimExpr(loopPosForInput[d], ctx));
    SmallVector<AffineMap> maps = {AffineMap::get(R, 0, inputExprs, ctx),
                                   AffineMap::get(R, 0, outExprs, ctx)};

    Value identVal = reduceIdentityConst(rewriter, loc, fn, elemTy);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, reducedShape, elemTy);
    Value init = rewriter.create<linalg::FillOp>(loc, ValueRange{identVal},
                                                 ValueRange{empty}).getResult(0);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{reducedTy}, ValueRange{op.getInput()}, ValueRange{init},
        maps, iters, [&](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(
              l, emitReduceCombine(b, l, fn, args[1], args[0],
                                   allowFloatReassociation));
        });
    Value reduced = generic.getResult(0);

    // mean = sum / count, where count = product of the reduced extents (static).
    if (fn == "mean") {
      int64_t count = 1;
      for (unsigned d = 0; d < R; ++d) {
        if (!isReduced[d])
          continue;
        if (inTy.isDynamicDim(d))
          return rewriter.notifyMatchFailure(op, "mean over dynamic extent");
        count *= inTy.getDimSize(d);
      }
      Value inv = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elemTy, 1.0 / (double)count));
      Value empty2 = rewriter.create<tensor::EmptyOp>(loc, reducedShape, elemTy);
      SmallVector<AffineMap> smaps(2, rewriter.getMultiDimIdentityMap(r));
      SmallVector<utils::IteratorType> siters(r, utils::IteratorType::parallel);
      reduced = rewriter
                    .create<linalg::GenericOp>(
                        loc, TypeRange{reducedTy}, ValueRange{reduced},
                        ValueRange{empty2}, smaps, siters,
                        [&](OpBuilder &b, Location l, ValueRange args) {
                          Value scaled =
                              b.create<arith::MulFOp>(l, args[0], inv);
                          b.create<linalg::YieldOp>(l, scaled);
                        })
                    .getResult(0);
    }

    // keepdims: result rank == input rank, reduced axes are size-1. Expand back.
    if (resTy.getRank() == (int64_t)R) {
      SmallVector<ReassociationIndices> reassoc(r);
      int src = -1;
      for (unsigned d = 0; d < R; ++d) {
        if (!isReduced[d])
          ++src;
        reassoc[src < 0 ? 0 : src].push_back(d);
      }
      Value expanded = rewriter.create<tensor::ExpandShapeOp>(
          loc, resTy, reduced, reassoc);
      rewriter.replaceOp(op, expanded);
      return success();
    }
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

/// Lower a static rank-2 last-axis inclusive scan to nested scf.for loops with
/// one scalar accumulator per row.  This is the conservative generic baseline
/// used when no target scan path is selected; target-specific row lowering
/// still owns the cooperative/vector scan states.
struct ScanLowering : OpRewritePattern<ScanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScanOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        inputType != resultType || inputType.getRank() != 2 ||
        op.getAxis() != 1)
      return rewriter.notifyMatchFailure(
          op, "generic scan requires one static rank-2 last axis");
    StringRef fn = op.getFn();
    Type elementType = inputType.getElementType();
    bool supported =
        (isa<FloatType>(elementType) && isNumericReduce(fn) && fn != "mean") ||
        (isa<IntegerType>(elementType) && !elementType.isInteger(1) &&
         isNumericReduce(fn) && fn != "mean") ||
        (elementType.isInteger(1) && isBoolReduce(fn));
    if (!supported)
      return rewriter.notifyMatchFailure(op, "unsupported scan combiner/type");

    Location location = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(location, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(location, 1);
    Value rows = rewriter.create<arith::ConstantIndexOp>(
        location, inputType.getDimSize(0));
    Value columns = rewriter.create<arith::ConstantIndexOp>(
        location, inputType.getDimSize(1));
    Value empty = rewriter.create<tensor::EmptyOp>(
        location, resultType.getShape(), elementType);
    auto outer = rewriter.create<scf::ForOp>(
        location, zero, rows, one, ValueRange{empty},
        [&](OpBuilder &outerBuilder, Location outerLocation, Value row,
            ValueRange outerArgs) {
          Value identity = reduceIdentityConst(
              outerBuilder, outerLocation, fn, elementType);
          auto inner = outerBuilder.create<scf::ForOp>(
              outerLocation, zero, columns, one,
              ValueRange{identity, outerArgs[0]},
              [&](OpBuilder &innerBuilder, Location innerLocation, Value column,
                  ValueRange innerArgs) {
                Value current = innerBuilder.create<tensor::ExtractOp>(
                    innerLocation, op.getInput(), ValueRange{row, column});
                Value combined = emitReduceCombine(
                    innerBuilder, innerLocation, fn, innerArgs[0], current);
                Value first = innerBuilder.create<arith::CmpIOp>(
                    innerLocation, arith::CmpIPredicate::eq, column, zero);
                Value inclusive = innerBuilder.create<arith::SelectOp>(
                    innerLocation, first, current, combined);
                Value updated = innerBuilder.create<tensor::InsertOp>(
                    innerLocation, inclusive, innerArgs[1],
                    ValueRange{row, column});
                innerBuilder.create<scf::YieldOp>(
                    innerLocation, ValueRange{inclusive, updated});
              });
          outerBuilder.create<scf::YieldOp>(outerLocation,
                                             inner.getResult(1));
        });
    rewriter.replaceOp(op, outer.getResult(0));
    return success();
  }
};

/// Lower an einsum-style contraction to a linalg.generic over the union of
/// index labels: output labels are parallel iterators, contracted labels
/// (present in inputs but not the output) are reduction iterators. Body is a
/// fused multiply-accumulate.  If no label is contracted, this is a pure
/// broadcast/outer multiply: it writes the product directly instead of adding
/// it to a +0 initializer, which is required to preserve IEEE signed zero.
struct ContractionLowering : OpRewritePattern<ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ContractionOp op,
                                PatternRewriter &rewriter) const override {
    StringRef sub = op.getSubscripts();
    size_t arrow = sub.find("->");
    size_t comma = sub.find(',');
    if (arrow == StringRef::npos || comma == StringRef::npos)
      return rewriter.notifyMatchFailure(op, "malformed subscripts");
    StringRef aL = sub.substr(0, comma);
    StringRef bL = sub.substr(comma + 1, arrow - comma - 1);
    StringRef oL = sub.substr(arrow + 2);

    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    Type elemTy = resTy.getElementType();
    if (!isa<FloatType, IntegerType>(elemTy))
      return rewriter.notifyMatchFailure(op,
                                         "non-numeric contraction not implemented");
    auto aElemTy = cast<RankedTensorType>(op.getA().getType()).getElementType();
    auto bElemTy = cast<RankedTensorType>(op.getB().getType()).getElementType();
    if (!isa<FloatType, IntegerType>(aElemTy) ||
        !isa<FloatType, IntegerType>(bElemTy))
      return rewriter.notifyMatchFailure(
          op, "non-numeric contraction operands not implemented");
    bool isFloat = isa<FloatType>(elemTy);
    if (!isFloat &&
        (isa<FloatType>(aElemTy) || isa<FloatType>(bElemTy)))
      return rewriter.notifyMatchFailure(
          op, "floating operands into integer contraction not implemented");
    if (!isFloat && (aElemTy != elemTy || bElemTy != elemTy))
      return rewriter.notifyMatchFailure(
          op, "mixed-width integer contraction not implemented");

    // Iteration domain = output labels (parallel) ++ contracted labels
    // (reduction). Contracted = appears in an input but not the output.
    llvm::DenseMap<char, unsigned> dim;
    SmallVector<utils::IteratorType> iters;
    for (char c : oL)
      if (!dim.count(c)) {
        dim[c] = dim.size();
        iters.push_back(utils::IteratorType::parallel);
      }
    for (StringRef in : {aL, bL})
      for (char c : in)
        if (!dim.count(c)) {
          dim[c] = dim.size();
          iters.push_back(utils::IteratorType::reduction);
        }
    bool hasReduction = llvm::any_of(iters, [](utils::IteratorType iterator) {
      return iterator == utils::IteratorType::reduction;
    });

    MLIRContext *ctx = op.getContext();
    auto mapFor = [&](StringRef labels) {
      SmallVector<AffineExpr> exprs;
      for (char c : labels)
        exprs.push_back(getAffineDimExpr(dim[c], ctx));
      return AffineMap::get(dim.size(), 0, exprs, ctx);
    };
    SmallVector<AffineMap> maps = {mapFor(aL), mapFor(bL), mapFor(oL)};

    Location loc = op.getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, resTy.getShape(), elemTy);
    Value init = empty;
    if (hasReduction) {
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(elemTy));
      init = rewriter
                 .create<linalg::FillOp>(loc, ValueRange{zero},
                                         ValueRange{empty})
                 .getResult(0);
    }

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getA(), op.getB()},
        ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          auto castToAccumulator = [&](Value value) -> Value {
            if (!isFloat)
              return toSignlessInteger(b, l, value);
            auto dstTy = cast<FloatType>(elemTy);
            if (auto integer = dyn_cast<IntegerType>(value.getType())) {
              // Mixed numeric no-reduction contractions are the canonical
              // outer-mask spelling emitted for one-hot coefficient grids.
              // i1 and explicit unsigned integers use the UI conversion; the
              // result type owns the explicit numeric promotion.
              if (integer.getWidth() == 1 || integer.isUnsigned())
                return b.create<arith::UIToFPOp>(
                    l, dstTy, toSignlessInteger(b, l, value));
              return b.create<arith::SIToFPOp>(l, dstTy, value);
            }
            auto srcTy = cast<FloatType>(value.getType());
            if (srcTy == dstTy)
              return value;
            if (srcTy.getWidth() < dstTy.getWidth())
              return b.create<arith::ExtFOp>(l, dstTy, value);
            return b.create<arith::TruncFOp>(l, dstTy, value);
          };
          Value lhs = castToAccumulator(args[0]);
          Value rhs = castToAccumulator(args[1]);
          Value prod = isFloat
                           ? Value(b.create<arith::MulFOp>(l, lhs, rhs))
                           : Value(b.create<arith::MulIOp>(l, lhs, rhs));
          if (!hasReduction) {
            b.create<linalg::YieldOp>(
                l, isFloat ? prod
                           : fromSignlessInteger(b, l, prod, elemTy));
            return;
          }
          Value acc = isFloat
                          ? Value(b.create<arith::AddFOp>(l, args[2], prod))
                          : Value(b.create<arith::AddIOp>(
                                l, toSignlessInteger(b, l, args[2]), prod));
          b.create<linalg::YieldOp>(
              l, isFloat ? acc : fromSignlessInteger(b, l, acc, elemTy));
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower `intent.loop` to an `scf.for` over the trip count with the carries as
/// tensor `iter_args`. Each trip slices the scanned inputs (`tensor.extract_slice`
/// along `scan_axes`, `tile_hint` rows) and the body's intent ops are cloned in
/// (they lower to linalg via the patterns above). This is the natural tensor-
/// level lowering; the fused CUDA schedule is selected later by the production
/// lowering pipeline and implemented by LowerFlashAttentionToLoops.cpp.
struct LoopLowering : OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned nScan = op.getScanInputs().size();
    bool dynamicTrip = static_cast<bool>(op.getUpperBound());
    int64_t tile = 1;
    int64_t trip = 0;
    if (!dynamicTrip) {
      if (nScan == 0)
        return rewriter.notifyMatchFailure(op, "loop with no scan inputs");
      tile =
          op.getTileHint() ? static_cast<int64_t>(*op.getTileHint()) : 1;
      auto scan0Ty = cast<RankedTensorType>(op.getScanInputs()[0].getType());
      int64_t axis0 = cast<IntegerAttr>(op.getScanAxes()[0]).getInt();
      int64_t extent = scan0Ty.getDimSize(axis0);
      if (ShapedType::isDynamic(extent))
        return rewriter.notifyMatchFailure(op, "dynamic scan extent");
      if (tile <= 0 || extent % tile != 0)
        return rewriter.notifyMatchFailure(
            op, "scan extent not a multiple of tile_hint (tail tiles unsupported)");
      trip = extent / tile;
    } else if (nScan != 0) {
      return rewriter.notifyMatchFailure(
          op, "dynamic upper_bound form cannot lower scanned inputs");
    }

    Block &body = op.getBody().front();
    Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value ub;
    if (dynamicTrip) {
      auto upperTy = cast<RankedTensorType>(op.getUpperBound().getType());
      Value upperScalar = rewriter.create<tensor::ExtractOp>(
          loc, op.getUpperBound(), ValueRange{});
      ub = raggedToIndex(rewriter, loc, upperScalar);
      if (!ub)
        return rewriter.notifyMatchFailure(
            op, "dynamic upper_bound is not an integer/index tensor");
      (void)upperTy;
    } else {
      ub = rewriter.create<arith::ConstantIndexOp>(loc, trip);
    }
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value tileV = rewriter.create<arith::ConstantIndexOp>(loc, tile);
    SmallVector<Value> inits(op.getInits().begin(), op.getInits().end());

    auto forOp = rewriter.create<scf::ForOp>(
        loc, lb, ub, step, inits,
        [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
          IRMapping map;
          if (!dynamicTrip) {
            Value off = b.create<arith::MulIOp>(l, iv, tileV);
            for (unsigned i = 0; i < nScan; ++i) {
              Value src = op.getScanInputs()[i];
              auto srcTy = cast<RankedTensorType>(src.getType());
              int64_t ax = cast<IntegerAttr>(op.getScanAxes()[i]).getInt();
              unsigned rank = srcTy.getRank();
              SmallVector<OpFoldResult> offs(rank, b.getIndexAttr(0));
              SmallVector<OpFoldResult> sizes;
              for (unsigned d = 0; d < rank; ++d)
                sizes.push_back(b.getIndexAttr(srcTy.getDimSize(d)));
              SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
              offs[ax] = off;
              sizes[ax] = b.getIndexAttr(tile);
              auto sliceTy =
                  cast<RankedTensorType>(body.getArgument(i).getType());
              Value slice = b.create<tensor::ExtractSliceOp>(
                  l, sliceTy, src, offs, sizes, strides);
              map.map(body.getArgument(i), slice);
            }
          } else {
            // The dynamic J ABI exposes the SCF induction value as a rank-0
            // tensor using the authored upper-bound element type.  Keeping
            // this conversion explicit prevents an MLIR index from leaking
            // into the Layer-A value graph and keeps CPU/CUDA/RISC-V emitters
            // on the same contract.
            auto inductionTy = cast<RankedTensorType>(
                body.getArgument(nScan).getType());
            Value induction = raggedFromIndex(
                b, l, iv, inductionTy.getElementType());
            if (!induction)
              return;
            Value inductionTensor = b.create<tensor::FromElementsOp>(
                l, inductionTy, ValueRange{induction});
            map.map(body.getArgument(nScan), inductionTensor);
          }
          unsigned carryArgBase = nScan + (dynamicTrip ? 1 : 0);
          for (unsigned j = 0; j < iterArgs.size(); ++j)
            map.map(body.getArgument(carryArgBase + j), iterArgs[j]);
          for (Operation &o : body.without_terminator())
            b.clone(o, map);
          auto yield = cast<YieldOp>(body.getTerminator());
          SmallVector<Value> yvals;
          for (Value v : yield.getResults())
            yvals.push_back(map.lookupOrDefault(v));
          b.create<scf::YieldOp>(l, yvals);
        });
    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }
};

/// Emit the scalar arith conversion for a (src -> dst) element-type pair.
/// Identity when equal; the natural arith op otherwise.
static Value emitScalarCast(OpBuilder &b, Location loc, Type dst, Value v) {
  Type src = v.getType();
  if (src == dst)
    return v;
  // Cast to i1 is a truthiness test (numpy .astype(bool) = x != 0), NOT a
  // low-bit fptosi/trunci — so `logical_not(x.bool())` and friends are correct.
  if (dst.isInteger(1))
    return truthy(b, loc, v);
  auto sf = dyn_cast<FloatType>(src);
  auto df = dyn_cast<FloatType>(dst);
  auto si = dyn_cast<IntegerType>(src);
  auto di = dyn_cast<IntegerType>(dst);
  if (sf && df)
    return sf.getWidth() < df.getWidth()
               ? b.create<arith::ExtFOp>(loc, dst, v).getResult()
               : b.create<arith::TruncFOp>(loc, dst, v).getResult();
  if (sf && di) {
    if (di.isUnsigned()) {
      Type signlessTarget = IntegerType::get(dst.getContext(), di.getWidth());
      Value converted = b.create<arith::FPToUIOp>(loc, signlessTarget, v);
      return fromSignlessInteger(b, loc, converted, dst);
    }
    return b.create<arith::FPToSIOp>(loc, dst, v);
  }
  if (si && df)
    return (si.isInteger(1) || si.isUnsigned())
               ? b.create<arith::UIToFPOp>(loc, dst,
                                           toSignlessInteger(b, loc, v))
                     .getResult()
               : b.create<arith::SIToFPOp>(loc, dst, v).getResult();
  if (si && di) {
    Value source = toSignlessInteger(b, loc, v);
    if (si.getWidth() < di.getWidth()) {
      Type signlessTarget = IntegerType::get(
          dst.getContext(), di.getWidth());
      Value widened =
          (si.isInteger(1) || si.isUnsigned())
              ? b.create<arith::ExtUIOp>(loc, signlessTarget, source).getResult()
              : b.create<arith::ExtSIOp>(loc, signlessTarget, source).getResult();
      return fromSignlessInteger(b, loc, widened, dst);
    }
    if (si.getWidth() == di.getWidth() &&
        (si.isUnsigned() != di.isUnsigned()))
      return fromSignlessInteger(b, loc, source, dst);
    Type signlessTarget = IntegerType::get(dst.getContext(), di.getWidth());
    Value narrowed = b.create<arith::TruncIOp>(loc, signlessTarget, source);
    return fromSignlessInteger(b, loc, narrowed, dst);
  }
  return nullptr;
}

struct CastLowering : OpRewritePattern<CastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    auto inTy = cast<RankedTensorType>(op.getInput().getType());
    // Same element type -> identity alias (the f32-only corpus path).
    if (inTy.getElementType() == resTy.getElementType()) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    Location loc = op.getLoc();
    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                  resTy.getElementType());
    unsigned rank = resTy.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    Type dstElt = resTy.getElementType();
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getInput()}, ValueRange{init}, maps,
        iters, [&](OpBuilder &b, Location l, ValueRange args) {
          Value c = emitScalarCast(b, l, dstElt, args[0]);
          b.create<linalg::YieldOp>(l, c);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower a static dimension permutation to a parallel linalg.generic.  The
/// iteration space is the result shape; the input indexing map applies the
/// inverse permutation so result dimension i reads input dimension perm[i].
struct TransposeLowering : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    if (!resTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic transpose is unsupported");
    unsigned rank = resTy.getRank();
    SmallVector<AffineExpr> inputExprs(rank);
    for (unsigned outDim = 0; outDim < rank; ++outDim) {
      int64_t inputDim = cast<IntegerAttr>(op.getPerm()[outDim]).getInt();
      inputExprs[inputDim] = getAffineDimExpr(outDim, op.getContext());
    }
    SmallVector<AffineMap> maps = {
        AffineMap::get(rank, 0, inputExprs, op.getContext()),
        rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<utils::IteratorType> iters(rank,
                                           utils::IteratorType::parallel);
    Location loc = op.getLoc();
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resTy.getShape(), resTy.getElementType());
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{op.getInput()}, ValueRange{init},
        maps, iters, [&](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, args[0]);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower a static variadic concat to slice-writing linalg.generic operations.
/// Keeping each slice as real linalg work avoids bufferization turning
/// tensor.insert_slice into external memrefCopy calls that a compiled target
/// pipeline may not support inside its executable artifact.
struct ConcatLowering : OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resTy || !resTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic concat is unsupported");
    int64_t axis = op.getAxis();
    if (axis < 0 || axis >= resTy.getRank())
      return rewriter.notifyMatchFailure(op, "concat axis is out of range");

    Location loc = op.getLoc();
    Value result = rewriter.create<tensor::EmptyOp>(
        loc, resTy.getShape(), resTy.getElementType());
    int64_t offset = 0;
    for (Value input : op.getInputs()) {
      auto inputTy = dyn_cast<RankedTensorType>(input.getType());
      if (!inputTy || !inputTy.hasStaticShape() ||
          inputTy.getRank() != resTy.getRank() ||
          inputTy.getElementType() != resTy.getElementType())
        return rewriter.notifyMatchFailure(
            op, "concat inputs must have one static common rank/type");

      unsigned rank = resTy.getRank();
      SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
      for (int64_t dim : inputTy.getShape())
        sizes.push_back(rewriter.getIndexAttr(dim));
      offsets[axis] = rewriter.getIndexAttr(offset);
      Value destinationSlice = rewriter.create<tensor::ExtractSliceOp>(
          loc, inputTy, result, offsets, sizes, strides);
      SmallVector<AffineMap> maps(2,
                                  rewriter.getMultiDimIdentityMap(rank));
      SmallVector<utils::IteratorType> iterators(
          rank, utils::IteratorType::parallel);
      auto generic = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{inputTy}, ValueRange{input},
          ValueRange{destinationSlice}, maps,
          iterators, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            b.create<linalg::YieldOp>(nestedLoc, args[0]);
          });
      result = rewriter.create<tensor::InsertSliceOp>(
          loc, generic.getResult(0), result, offsets, sizes, strides);
      offset += inputTy.getDimSize(axis);
    }
    if (offset != resTy.getDimSize(axis))
      return rewriter.notifyMatchFailure(
          op, "concat input extents do not fill the result axis");
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower constant padding to tensor.pad.  The downstream one-shot
/// bufferization/SCF pipeline already carries tensor.pad to executable loops.
struct PadLowering : OpRewritePattern<PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter &rewriter) const override {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<OpFoldResult> low, high;
    for (Attribute attr : op.getLo()) {
      int64_t value = cast<IntegerAttr>(attr).getInt();
      if (value < 0)
        return rewriter.notifyMatchFailure(op, "negative low padding");
      low.push_back(rewriter.getIndexAttr(value));
    }
    for (Attribute attr : op.getHi()) {
      int64_t value = cast<IntegerAttr>(attr).getInt();
      if (value < 0)
        return rewriter.notifyMatchFailure(op, "negative high padding");
      high.push_back(rewriter.getIndexAttr(value));
    }
    Type elemTy = resTy.getElementType();
    TypedAttr valueAttr;
    if (isa<FloatType>(elemTy))
      valueAttr = rewriter.getFloatAttr(
          elemTy, op.getValue().convertToDouble());
    else if (isa<IntegerType>(elemTy))
      valueAttr = rewriter.getIntegerAttr(
          elemTy, static_cast<int64_t>(op.getValue().convertToDouble()));
    else
      return rewriter.notifyMatchFailure(op, "unsupported pad element type");
    Value padValue =
        rewriter.create<arith::ConstantOp>(op.getLoc(), valueAttr);
    Value padded = rewriter.create<tensor::PadOp>(
        op.getLoc(), resTy, op.getInput(), low, high, padValue,
        /*nofold=*/false);
    rewriter.replaceOp(op, padded);
    return success();
  }
};

struct ConvertIntentToLinalgPass
    : ::intent::impl::ConvertIntentToLinalgBase<ConvertIntentToLinalgPass> {
  using ConvertIntentToLinalgBase::ConvertIntentToLinalgBase;

  void runOnOperation() override {
    bool rejectedAtomic = false;
    getOperation().walk([&](Operation *operation) {
      bool explicitAtomic = false;
      if (auto scatter = dyn_cast<ScatterOp>(operation))
        explicitAtomic = scatter->hasAttr("atomic") ||
                         scatter->hasAttr("ordering") ||
                         scatter->hasAttr("memory_scope") ||
                         scatter->hasAttr("cas_retry");
      else if (isa<AtomicScatterOp>(operation))
        explicitAtomic = true;
      if (!explicitAtomic)
        return;
      operation->emitError()
          << "generic Intent-to-Linalg lowering rejects explicit atomic "
             "scatter because it has no target-neutral atomic operation; "
             "atomic metadata would otherwise be lost";
      rejectedAtomic = true;
    });
    if (rejectedAtomic) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<IotaLowering, RaggedCleanupLowering,
                 RaggedOuterLowering,
                 RaggedMemberLowering, ReshapeLowering,
                 IdentityScatterLowering,
                 MapUnaryLowering,
                 MapBinaryLowering, SelectLowering, GatherLowering,
                 ScanLowering, ContractionLowering,
                 LoopLowering, CastLowering, TransposeLowering, ConcatLowering,
                 PadLowering>(
        &getContext());
    patterns.add<ReduceLowering>(&getContext(), reassociateFloatReductions);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool liveUnsupported = false;
    getOperation().walk([&](Operation *operation) {
      if (isa<RaggedOp, RaggedOuterOp, RaggedMemberOp>(operation)) {
        operation->emitError()
            << "ragged operation was not fully lowered to carrier tensors";
        liveUnsupported = true;
      }
    });
    if (liveUnsupported)
      signalPassFailure();
  }
};

} // namespace
