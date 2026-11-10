#include "Intent/IntentOps.h"
#include "Intent/IntentDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <limits>

using namespace mlir;
using namespace intent;

#define GET_OP_CLASSES
#include "Intent/IntentOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

namespace {

bool isRankOneIntegerTensor(Type type) {
  return isRaggedCarrierTensorType(type);
}

LogicalResult verifyRaggedDescriptor(Operation *op, Type type,
                                     TupleType &descriptor) {
  auto tuple = dyn_cast<TupleType>(type);
  if (!tuple)
    return op->emitOpError("ragged descriptor must be a builtin tuple type");
  if (tuple.size() != 2)
    return op->emitOpError(
        "ragged descriptor must contain exactly offsets and indices fields");
  if (!isRankOneIntegerTensor(tuple.getType(0)))
    return op->emitOpError(
        "ragged descriptor offsets field must use a rank-1 closed carrier tensor");
  if (!isRankOneIntegerTensor(tuple.getType(1)))
    return op->emitOpError(
        "ragged descriptor indices field must use a rank-1 closed carrier tensor");
  descriptor = tuple;
  return success();
}

} // namespace

LogicalResult YieldOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  if (isa<LoopOp>(parent))
    return success();
  return emitOpError(
      "yield must be nested in intent.loop");
}

LogicalResult RaggedOp::verify() {
  auto offsetsTy = dyn_cast<RankedTensorType>(getOffsets().getType());
  auto indicesTy = dyn_cast<RankedTensorType>(getIndices().getType());
  if (!offsetsTy || !isRankOneIntegerTensor(offsetsTy))
    return emitOpError("offsets must be a rank-1 closed carrier tensor");
  if (!indicesTy || !isRankOneIntegerTensor(indicesTy))
    return emitOpError("indices must be a rank-1 closed carrier tensor");

  TupleType descriptor;
  if (failed(verifyRaggedDescriptor(getOperation(), getResult().getType(),
                                    descriptor)))
    return failure();
  if (descriptor.getType(0) != offsetsTy)
    return emitOpError(
        "descriptor offsets field type must equal offsets operand");
  if (descriptor.getType(1) != indicesTy)
    return emitOpError(
        "descriptor indices field type must equal indices operand");
  return success();
}

LogicalResult RaggedOuterOp::verify() {
  TupleType descriptor;
  if (failed(verifyRaggedDescriptor(getOperation(), getRagged().getType(),
                                    descriptor)))
    return failure();

  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultTy || resultTy.getRank() != 0)
    return emitOpError("result must be a rank-0 integer tensor");
  if (!isRaggedCarrierElementType(resultTy.getElementType()))
    return emitOpError("result must use a closed Ragged carrier dtype");

  auto offsetsTy = dyn_cast<RankedTensorType>(descriptor.getType(0));
  if (!offsetsTy)
    return emitOpError("descriptor offsets field must be ranked");
  if (resultTy.getElementType() != offsetsTy.getElementType())
    return emitOpError(
        "result element type must equal the descriptor offsets element type");
  return success();
}

LogicalResult RaggedMemberOp::verify() {
  TupleType descriptor;
  if (failed(verifyRaggedDescriptor(getOperation(), getRagged().getType(),
                                    descriptor)))
    return failure();

  auto indexTy = dyn_cast<RankedTensorType>(getIndex().getType());
  if (!indexTy || indexTy.getRank() != 0 ||
      !isRaggedCarrierElementType(indexTy.getElementType()))
    return emitOpError("index must be a rank-0 closed Ragged carrier tensor");

  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultTy || resultTy.getRank() != 1 || !resultTy.isDynamicDim(0))
    return emitOpError(
        "result must be a dynamic-length rank-1 integer tensor");
  if (!isRaggedCarrierElementType(resultTy.getElementType()))
    return emitOpError("result must use a closed Ragged carrier dtype");

  auto indicesTy = dyn_cast<RankedTensorType>(descriptor.getType(1));
  if (!indicesTy)
    return emitOpError("descriptor indices field must be ranked");
  if (resultTy.getElementType() != indicesTy.getElementType())
    return emitOpError(
        "result element type must equal the descriptor indices element type");
  return success();
}

LogicalResult MapBinaryOp::verify() {
  static constexpr llvm::StringRef kArithmeticFns[] = {
      "add", "sub", "mul", "div", "max", "min"};
  static constexpr llvm::StringRef kComparisonFns[] = {
      "gt", "ge", "lt", "le", "eq", "ne"};
  static constexpr llvm::StringRef kLogicalFns[] = {
      "logical_and", "logical_or", "logical_xor", "and", "or", "xor"};
  static constexpr llvm::StringRef kMatchingNumericFns[] = {
      "floordiv", "remainder"};
  static constexpr llvm::StringRef kShiftFns[] = {"left_shift", "right_shift"};

  auto lhsTy = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsTy = dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhsTy || !rhsTy || !resultTy)
    return emitOpError("requires ranked tensor operands and result");

  Type lhsElement = lhsTy.getElementType();
  Type rhsElement = rhsTy.getElementType();
  Type resultElement = resultTy.getElementType();
  auto isNumeric = [](Type type) {
    return isa<FloatType, IntegerType>(type);
  };
  auto in = [](llvm::StringRef value,
               llvm::ArrayRef<llvm::StringRef> values) {
    return llvm::is_contained(values, value);
  };

  if (in(getFn(), kComparisonFns) || in(getFn(), kLogicalFns) ||
      getFn() == "isclose") {
    if (!resultElement.isInteger(1))
      return emitOpError("predicate fn '")
             << getFn() << "' must produce an i1-typed result tensor";
    if (getFn() == "isclose") {
      if (!isa<FloatType>(lhsElement) || !isa<FloatType>(rhsElement))
        return emitOpError("isclose requires floating-point operands");
    } else if (!isNumeric(lhsElement) || !isNumeric(rhsElement)) {
      return emitOpError("predicate fn '")
             << getFn() << "' requires numeric operands";
    }
    if (lhsElement != rhsElement)
      return emitOpError("predicate fn '")
             << getFn() << "' requires matching operand element types";
    return success();
  }

  if (in(getFn(), kShiftFns)) {
    if (!isa<IntegerType>(lhsElement) || !isa<IntegerType>(rhsElement) ||
        !isa<IntegerType>(resultElement))
      return emitOpError("shift fn '")
             << getFn() << "' requires integer operands and result";
    if (lhsElement != rhsElement || resultElement != lhsElement)
      return emitOpError("shift fn '")
             << getFn() << "' requires matching operand/result integer types";
    return success();
  }

  if (in(getFn(), kMatchingNumericFns)) {
    if (!isNumeric(lhsElement) || lhsElement != rhsElement ||
        resultElement != lhsElement)
      return emitOpError("fn '")
             << getFn() << "' requires matching numeric operand/result types";
    return success();
  }

  if (getFn() == "pow" || getFn() == "atan2" || getFn() == "copysign") {
    if (!isa<FloatType>(lhsElement) || !isa<FloatType>(rhsElement) ||
        !isa<FloatType>(resultElement))
      return emitOpError("fn '")
             << getFn() << "' requires floating-point operands and result";
    if (lhsElement != rhsElement || resultElement != lhsElement)
      return emitOpError("fn '")
             << getFn() << "' requires matching operand/result types";
    return success();
  }

  if (in(getFn(), kArithmeticFns)) {
    if (!isNumeric(lhsElement) || lhsElement != rhsElement ||
        resultElement != lhsElement)
      return emitOpError("fn '")
             << getFn() << "' requires matching numeric operand/result types";
    return success();
  }
  return emitOpError("unknown fn '") << getFn() << "'";
}

LogicalResult SelectOp::verify() {
  auto cTy = cast<RankedTensorType>(getCond().getType());
  if (!cTy.getElementType().isInteger(1))
    return emitOpError("cond must be an i1-typed tensor");
  auto xTy = cast<RankedTensorType>(getX().getType());
  auto yTy = cast<RankedTensorType>(getY().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  // Element types only (shapes broadcast numpy-style, e.g. an [M,N] i1 condition
  // selecting between two rank-0 scalars).
  if (xTy.getElementType() != yTy.getElementType())
    return emitOpError("x and y must share one element type");
  if (rTy.getElementType() != xTy.getElementType())
    return emitOpError("result element type must equal x/y element type");
  return success();
}

LogicalResult IotaOp::verify() {
  auto rTy = cast<RankedTensorType>(getResult().getType());
  if (rTy.getRank() != 1 || !rTy.hasStaticShape())
    return emitOpError("result must be a rank-1 static-shape tensor");
  if (!rTy.getElementType().isIntOrIndex())
    return emitOpError("result element type must be integer/index");
  return success();
}

LogicalResult MapUnaryOp::verify() {
  // Element-type-preserving fns: result type == input type.
  static constexpr llvm::StringRef kFns[] = {
      "identity", "exp", "log", "log2", "neg", "abs", "sqrt", "rsqrt", "recip",
      "tanh", "sigmoid", "cos", "sin", "tan",  "exp2", "erf", "atan",
      "acos", "asin", "cbrt", "ceil", "floor", "round", "roundeven", "trunc",
      "sign", "log10", "log1p", "expm1", "erfc",
      // logical_not is i1->i1, still element-type-preserving.
      "logical_not"};
  // Predicate fns produce an i1-typed result tensor (the boolean condition for
  // intent.select), mirroring the map_binary comparison fns. C2c adds the rest
  // of the finiteness family (isinf/isfinite) alongside isnan.
  static constexpr llvm::StringRef kToI1Fns[] = {"isnan", "isinf", "isfinite"};
  auto rTy = cast<RankedTensorType>(getResult().getType());
  for (llvm::StringRef f : kToI1Fns)
    if (getFn() == f) {
      if (!rTy.getElementType().isInteger(1))
        return emitOpError("predicate fn '")
               << getFn() << "' must produce an i1-typed result tensor";
      auto iTy = cast<RankedTensorType>(getInput().getType());
      if (!isa<FloatType>(iTy.getElementType()))
        return emitOpError("predicate fn '")
               << getFn() << "' requires a floating-point input tensor";
      if (iTy.getShape() != rTy.getShape())
        return emitOpError("result shape must equal input shape");
      return success();
    }
  bool ok = false;
  for (llvm::StringRef f : kFns)
    if (getFn() == f)
      ok = true;
  if (!ok)
    return emitOpError("unknown fn '") << getFn() << "'";
  // Shape-preserving: result type matches the input type.
  if (getInput().getType() != getResult().getType())
    return emitOpError("result type must equal input type");
  auto iTy = cast<RankedTensorType>(getInput().getType());
  Type element = iTy.getElementType();
  if (getFn() == "logical_not") {
    if (!element.isInteger(1))
      return emitOpError("logical_not requires an i1 input/result tensor");
  } else if (getFn() == "identity") {
    return success();
  } else if (getFn() == "sign" || getFn() == "neg" || getFn() == "abs") {
    if (!isa<FloatType, IntegerType>(element))
      return emitOpError("unary fn '")
             << getFn() << "' requires a numeric input/result tensor";
  } else if (!isa<FloatType>(element)) {
    return emitOpError("unary fn '")
           << getFn() << "' requires a floating-point input/result tensor";
  }
  return success();
}

LogicalResult CastOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  // Shape-preserving: only the element type may differ.
  if (iTy.getShape() != rTy.getShape())
    return emitOpError("cast is shape-preserving: result shape ")
           << rTy.getShape() << " != input shape " << iTy.getShape();
  return success();
}

LogicalResult ScanOp::verify() {
  static constexpr llvm::StringRef kFns[] = {"sum", "max", "min"};
  bool ok = false;
  for (llvm::StringRef f : kFns)
    if (getFn() == f)
      ok = true;
  if (!ok)
    return emitOpError("unknown scan fn '") << getFn() << "'";
  // Shape-preserving: result type matches the input type.
  if (getInput().getType() != getResult().getType())
    return emitOpError("result type must equal input type");
  auto inTy = cast<RankedTensorType>(getInput().getType());
  // The generated accessor for the I64 attribute is unsigned.  A negative
  // textual attribute is therefore represented outside the valid rank range,
  // so the upper-bound check rejects it as well without a tautological < 0
  // comparison.
  const uint64_t rank = static_cast<uint64_t>(inTy.getRank());
  if (getAxis() >= rank)
    return emitOpError("axis ")
           << getAxis() << " out of range for rank-" << rank
           << " input";
  return success();
}

LogicalResult ArgReduceOp::verify() {
  if (getFn() != "argmax" && getFn() != "argmin")
    return emitOpError("unknown arg_reduce fn '")
           << getFn() << "' (expected argmax/argmin)";
  auto inTy = cast<RankedTensorType>(getInput().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  if (!rTy.getElementType().isIntOrIndex())
    return emitOpError("arg_reduce result must be an integer-typed tensor (an "
                       "index)");
  int64_t axis = getAxis();
  if (axis < 0 || axis >= inTy.getRank())
    return emitOpError("axis ")
           << axis << " out of range for rank-" << inTy.getRank() << " input";
  // Result is the input with `axis` collapsed: either dropped (rank-1) or kept
  // as size 1 (keepdims). Both layouts are accepted.
  bool keepdims = rTy.getRank() == inTy.getRank();
  bool collapsed = rTy.getRank() == inTy.getRank() - 1;
  if (!keepdims && !collapsed)
    return emitOpError("result rank must be input rank (keepdims) or input "
                       "rank-1 (collapsed)");
  return success();
}

LogicalResult SortOp::verify() {
  auto inTy = cast<RankedTensorType>(getInput().getType());
  auto vTy = cast<RankedTensorType>(getValues().getType());
  auto iTy = cast<RankedTensorType>(getIndices().getType());
  int64_t rank = inTy.getRank();
  int64_t axis = getAxis();
  if (axis < 0 || axis >= rank)
    return emitOpError("axis ")
           << axis << " out of range for rank-" << rank << " input";
  if (axis != rank - 1)
    return emitOpError("sort axis must be the last axis (got ")
           << axis << " for rank " << rank << ")";
  if (vTy.getElementType() != inTy.getElementType())
    return emitOpError("values element type must equal input element type");
  if (!iTy.getElementType().isInteger(32))
    return emitOpError("indices must be an i32 tensor (the argsort permutation)");
  if (vTy.getRank() != rank || iTy.getRank() != rank)
    return emitOpError("values/indices must have the input rank ") << rank;
  // Values and indices share a shape; every non-axis extent matches the input;
  // the axis extent W may be <= N (W < N = topk).
  for (int64_t d = 0; d < rank; ++d) {
    int64_t vd = vTy.getDimSize(d), id = iTy.getDimSize(d), nd = inTy.getDimSize(d);
    if (!ShapedType::isDynamic(vd) && !ShapedType::isDynamic(id) && vd != id)
      return emitOpError("values/indices extent mismatch at dim ") << d;
    if (d == axis) {
      if (!ShapedType::isDynamic(vd) && !ShapedType::isDynamic(nd) && vd > nd)
        return emitOpError("sort output extent ")
               << vd << " exceeds input extent " << nd << " at axis " << d;
    } else if (!ShapedType::isDynamic(vd) && !ShapedType::isDynamic(nd) &&
               vd != nd) {
      return emitOpError("non-axis extent ")
             << vd << " != input extent " << nd << " at dim " << d;
    }
  }
  return success();
}

LogicalResult GatherOp::verify() {
  auto dTy = cast<RankedTensorType>(getData().getType());
  auto iTy = cast<RankedTensorType>(getIndex().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  if (!iTy.getElementType().isIntOrIndex())
    return emitOpError("index must be an integer-typed tensor");
  if (dTy.getElementType() != rTy.getElementType())
    return emitOpError("result element type must equal data element type");
  uint64_t axis = getAxis();
  if (axis >= static_cast<uint64_t>(dTy.getRank()))
    return emitOpError("axis ")
           << axis << " out of range for rank-" << dTy.getRank() << " data";
  llvm::StringRef mode = getMode();
  if (mode == "along_axis") {
    if (iTy.getRank() != dTy.getRank())
      return emitOpError("along_axis gather: index rank ")
             << iTy.getRank() << " must equal data rank " << dTy.getRank();
    if (rTy.getRank() != iTy.getRank())
      return emitOpError("along_axis gather: result must be shaped like index");
    // Static extents must match index exactly (dynamic dims are wildcards).
    for (int64_t d = 0; d < rTy.getRank(); ++d) {
      int64_t rd = rTy.getDimSize(d), id = iTy.getDimSize(d);
      if (!ShapedType::isDynamic(rd) && !ShapedType::isDynamic(id) && rd != id)
        return emitOpError("along_axis gather: result extent ")
               << rd << " != index extent " << id << " at dim " << d;
    }
    return success();
  }
  if (mode == "take") {
    int64_t want = dTy.getRank() - 1 + iTy.getRank();
    if (rTy.getRank() != want)
      return emitOpError("take gather: result rank ")
             << rTy.getRank() << " != data.rank-1+index.rank (" << want << ")";
    return success();
  }
  return emitOpError("unknown gather mode '") << mode << "'";
}

LogicalResult ScatterOp::verify() {
  auto dTy = cast<RankedTensorType>(getData().getType());
  auto iTy = cast<RankedTensorType>(getIndex().getType());
  auto sTy = cast<RankedTensorType>(getSrc().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  if (!iTy.getElementType().isIntOrIndex())
    return emitOpError("index must be an integer-typed tensor");
  if (dTy.getElementType() != rTy.getElementType())
    return emitOpError("result element type must equal data element type");
  if (getMode() != "along_axis")
    return emitOpError("unknown scatter mode '") << getMode() << "'";
  if (getReduce() != "none" && getReduce() != "add")
    return emitOpError("reduce must be 'none' or 'add'");
  int64_t axis = getAxis();
  if (axis < 0 || axis >= dTy.getRank())
    return emitOpError("axis ")
           << axis << " out of range for rank-" << dTy.getRank() << " data";
  if (iTy.getRank() != dTy.getRank() || sTy.getRank() != dTy.getRank())
    return emitOpError("along_axis scatter: index/src rank must equal data rank");
  if (rTy.getShape() != dTy.getShape())
    return emitOpError("scatter result shape must equal data shape");

  // The three atomic fields are additive metadata.  An old scatter has none of
  // them and follows the historical verifier/lowering path unchanged.  Once
  // any field is present, require the complete ordering/scope contract so no
  // consumer can accidentally erase part of the synchronization semantics.
  Operation *operation = getOperation();
  bool hasAtomic = operation->hasAttr("atomic");
  bool hasOrdering = operation->hasAttr("ordering");
  bool hasMemoryScope = operation->hasAttr("memory_scope");
  auto atomicAttr = operation->getAttrOfType<StringAttr>("atomic");
  auto orderingAttr = operation->getAttrOfType<StringAttr>("ordering");
  auto memoryScopeAttr = operation->getAttrOfType<StringAttr>("memory_scope");
  if ((hasAtomic && !atomicAttr) || (hasOrdering && !orderingAttr) ||
      (hasMemoryScope && !memoryScopeAttr))
    return emitOpError(
        "atomic, ordering, and memory_scope must be string attributes");
  if (hasAtomic || hasOrdering || hasMemoryScope) {
    if (!hasAtomic || !hasOrdering || !hasMemoryScope)
      return emitOpError(
          "atomic scatter requires atomic, ordering, and memory_scope");
    StringRef atomic = atomicAttr.getValue();
    if (atomic != "add" && atomic != "cas")
      return emitOpError("unknown atomic kind '") << atomic << "'";
    StringRef ordering = orderingAttr.getValue();
    if (ordering != "relaxed" && ordering != "acquire" &&
        ordering != "release" && ordering != "acq_rel" &&
        ordering != "seq_cst")
      return emitOpError("unknown atomic ordering '") << ordering << "'";
    StringRef memoryScope = memoryScopeAttr.getValue();
    if (memoryScope != "work_item" && memoryScope != "subgroup" &&
        memoryScope != "workgroup" && memoryScope != "device" &&
        memoryScope != "system")
      return emitOpError("unknown atomic memory_scope '") << memoryScope << "'";
    if (atomic == "add" && getReduce() != "add")
      return emitOpError("atomic='add' requires reduce='add'");
    if (atomic == "cas")
      return emitOpError(
          "atomic='cas' requires the complete intent.atomic_scatter form");
  }
  return success();
}

LogicalResult AtomicScatterOp::verify() {
  auto dTy = cast<RankedTensorType>(getData().getType());
  auto iTy = cast<RankedTensorType>(getIndex().getType());
  auto cTy = cast<RankedTensorType>(getCompare().getType());
  auto sTy = cast<RankedTensorType>(getSrc().getType());
  auto uTy = cast<RankedTensorType>(getUpdated().getType());
  auto oTy = cast<RankedTensorType>(getObserved().getType());

  if (getAtomic() != "cas")
    return emitOpError("atomic_scatter currently requires atomic='cas'");
  if (getMode() != "along_axis")
    return emitOpError("atomic_scatter currently requires mode='along_axis'");
  if (getAxis() >= static_cast<uint64_t>(dTy.getRank()))
    return emitOpError("axis is outside the data rank");
  if (!isa<IntegerType>(iTy.getElementType()) &&
      !iTy.getElementType().isIndex())
    return emitOpError("index must have an integer or index element type");
  if (iTy.getShape() != cTy.getShape() || iTy.getShape() != sTy.getShape())
    return emitOpError("index, compare, and src must share one shape");
  if (iTy.getRank() != dTy.getRank())
    return emitOpError("index rank must equal data rank");
  if (uTy.getShape() != dTy.getShape() || oTy.getShape() != sTy.getShape())
    return emitOpError("updated/observed shapes do not match data/index shapes");
  if (cTy.getElementType() != dTy.getElementType() ||
      sTy.getElementType() != dTy.getElementType() ||
      uTy.getElementType() != dTy.getElementType() ||
      oTy.getElementType() != dTy.getElementType())
    return emitOpError(
        "compare, src, updated, and observed must match the data element type");

  StringRef ordering = getOrdering();
  if (ordering != "relaxed" && ordering != "acquire" &&
      ordering != "release" && ordering != "acq_rel" &&
      ordering != "seq_cst")
    return emitOpError("unknown atomic ordering '") << ordering << "'";
  StringRef memoryScope = getMemoryScope();
  if (memoryScope != "work_item" && memoryScope != "subgroup" &&
      memoryScope != "workgroup" && memoryScope != "device" &&
      memoryScope != "system")
    return emitOpError("unknown atomic memory_scope '") << memoryScope << "'";
  if (auto retry = getOperation()->getAttrOfType<StringAttr>("cas_retry")) {
    if (retry.getValue() != "single_attempt" &&
        retry.getValue() != "until_success")
      return emitOpError("unknown cas_retry mode '") << retry.getValue() << "'";
  }
  return success();
}

LogicalResult NonzeroOp::verify() {
  auto cTy = cast<RankedTensorType>(getCond().getType());
  auto rTy = cast<RankedTensorType>(getIndices().getType());
  // Result is rank-2 [nnz, ndim] integer: one row per nonzero element, each row
  // the ndim-vector coordinate (ndim = cond.rank). The leading `nnz` extent is
  // data-dependent (typically dynamic).
  if (rTy.getRank() != 2)
    return emitOpError("nonzero result must be a rank-2 [nnz, ndim] tensor");
  if (!rTy.getElementType().isIntOrIndex())
    return emitOpError("nonzero result must be an integer-typed (index) tensor");
  int64_t ndim = rTy.getDimSize(1);
  if (!ShapedType::isDynamic(ndim) && ndim != cTy.getRank())
    return emitOpError("nonzero result inner extent ")
           << ndim << " must equal cond rank " << cTy.getRank();
  return success();
}

LogicalResult MaskedSelectOp::verify() {
  auto dTy = cast<RankedTensorType>(getData().getType());
  auto mTy = cast<RankedTensorType>(getMask().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  // mask shares data's shape; result is a rank-1 [nnz] compaction (data elt).
  if (mTy.getShape() != dTy.getShape())
    return emitOpError("mask shape must equal data shape");
  if (rTy.getRank() != 1)
    return emitOpError("masked_select result must be a rank-1 [nnz] tensor");
  if (rTy.getElementType() != dTy.getElementType())
    return emitOpError("result element type must equal data element type");
  return success();
}

LogicalResult MaskedScatterOp::verify() {
  auto dTy = cast<RankedTensorType>(getData().getType());
  auto mTy = cast<RankedTensorType>(getMask().getType());
  auto sTy = cast<RankedTensorType>(getSrc().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  // data-shaped result; mask shares data's shape; src is a rank-1 fill vector.
  if (mTy.getShape() != dTy.getShape())
    return emitOpError("mask shape must equal data shape");
  if (sTy.getRank() != 1)
    return emitOpError("masked_scatter src must be a rank-1 vector");
  if (sTy.getElementType() != dTy.getElementType())
    return emitOpError("src element type must equal data element type");
  if (rTy.getShape() != dTy.getShape())
    return emitOpError("masked_scatter result shape must equal data shape");
  if (rTy.getElementType() != dTy.getElementType())
    return emitOpError("result element type must equal data element type");
  return success();
}

LogicalResult ConcatOp::verify() {
  auto rTy = cast<RankedTensorType>(getResult().getType());
  int64_t rank = rTy.getRank();
  int64_t axis = getAxis();
  if (axis < 0 || axis >= rank)
    return emitOpError("axis ")
           << axis << " out of range for rank-" << rank << " result";
  if (getInputs().empty())
    return emitOpError("concat requires at least one input");
  int64_t axisSum = 0;
  bool axisStatic = true;
  for (auto in : getInputs()) {
    auto iTy = cast<RankedTensorType>(in.getType());
    if (iTy.getRank() != rank)
      return emitOpError("all inputs must have the result rank ") << rank;
    if (iTy.getElementType() != rTy.getElementType())
      return emitOpError("all inputs must share the result element type");
    for (int64_t d = 0; d < rank; ++d) {
      int64_t id = iTy.getDimSize(d);
      if (d == axis) {
        if (ShapedType::isDynamic(id))
          axisStatic = false;
        else
          axisSum += id;
      } else {
        int64_t rd = rTy.getDimSize(d);
        if (!ShapedType::isDynamic(id) && !ShapedType::isDynamic(rd) && id != rd)
          return emitOpError("input extent ")
                 << id << " != result extent " << rd << " at non-concat dim "
                 << d;
      }
    }
  }
  int64_t ra = rTy.getDimSize(axis);
  if (axisStatic && !ShapedType::isDynamic(ra) && axisSum != ra)
    return emitOpError("sum of input axis extents ")
           << axisSum << " != result axis extent " << ra;
  return success();
}

LogicalResult ReshapeOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  if (iTy.getElementType() != rTy.getElementType())
    return emitOpError("reshape is element-type-preserving");
  if (static_cast<int64_t>(getShape().size()) != rTy.getRank())
    return emitOpError("shape attribute rank must equal result rank");

  SmallVector<int64_t> shape;
  shape.reserve(getShape().size());
  for (Attribute attr : getShape()) {
    int64_t extent = cast<IntegerAttr>(attr).getInt();
    if (extent < 0)
      return emitOpError("shape entries must be non-negative");
    shape.push_back(extent);
  }
  for (int64_t dim = 0; dim < rTy.getRank(); ++dim) {
    int64_t declared = rTy.getDimSize(dim);
    if (!ShapedType::isDynamic(declared) && declared != shape[dim])
      return emitOpError("result extent ")
             << declared << " != shape attribute extent " << shape[dim]
             << " at dim " << dim;
  }

  // A static source and a static shape must describe the same linear element
  // sequence.  Keep the multiplication checked so a malformed i64 attribute
  // cannot wrap into a false equality.
  if (iTy.hasStaticShape()) {
    auto elementCount = [](ArrayRef<int64_t> extents,
                           int64_t &count) -> bool {
      count = 1;
      for (int64_t extent : extents) {
        if (extent < 0 ||
            (extent != 0 &&
             count > std::numeric_limits<int64_t>::max() / extent))
          return false;
        count *= extent;
      }
      return true;
    };
    int64_t inputCount = 0;
    int64_t resultCount = 0;
    if (!elementCount(iTy.getShape(), inputCount) ||
        !elementCount(shape, resultCount))
      return emitOpError("static element count overflows i64");
    if (inputCount != resultCount)
      return emitOpError("reshape changes element count from ")
             << inputCount << " to " << resultCount;
  }
  return success();
}

LogicalResult TransposeOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  int64_t rank = iTy.getRank();
  if (rTy.getRank() != rank)
    return emitOpError("result rank must equal input rank");
  if (iTy.getElementType() != rTy.getElementType())
    return emitOpError("transpose is element-type-preserving");
  if ((int64_t)getPerm().size() != rank)
    return emitOpError("perm must have one entry per dim (rank ") << rank << ")";
  SmallVector<bool> seen(rank, false);
  for (int64_t outDim = 0; outDim < rank; ++outDim) {
    int64_t inDim = cast<IntegerAttr>(getPerm()[outDim]).getInt();
    if (inDim < 0 || inDim >= rank)
      return emitOpError("perm axis ")
             << inDim << " out of range for rank-" << rank << " input";
    if (seen[inDim])
      return emitOpError("perm contains duplicate axis ") << inDim;
    seen[inDim] = true;
    int64_t id = iTy.getDimSize(inDim), rd = rTy.getDimSize(outDim);
    if (!ShapedType::isDynamic(id) && !ShapedType::isDynamic(rd) && id != rd)
      return emitOpError("result extent ")
             << rd << " != permuted input extent " << id
             << " at result dim " << outDim;
  }
  return success();
}

LogicalResult PadOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  int64_t rank = iTy.getRank();
  if (rTy.getRank() != rank)
    return emitOpError("result rank must equal input rank");
  if (iTy.getElementType() != rTy.getElementType())
    return emitOpError("pad is element-type-preserving");
  if ((int64_t)getLo().size() != rank || (int64_t)getHi().size() != rank)
    return emitOpError("lo/hi must have one entry per dim (rank ") << rank << ")";
  for (int64_t d = 0; d < rank; ++d) {
    int64_t id = iTy.getDimSize(d), rd = rTy.getDimSize(d);
    int64_t lo = cast<IntegerAttr>(getLo()[d]).getInt();
    int64_t hi = cast<IntegerAttr>(getHi()[d]).getInt();
    if (lo < 0 || hi < 0)
      return emitOpError("lo/hi entries must be non-negative at dim ") << d;
    if (ShapedType::isDynamic(id) || ShapedType::isDynamic(rd))
      continue;
    if (id + lo + hi != rd)
      return emitOpError("result extent ")
             << rd << " != input+lo+hi (" << (id + lo + hi) << ") at dim " << d;
  }
  return success();
}

LogicalResult TileOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  int64_t rank = iTy.getRank();
  if (rTy.getRank() != rank)
    return emitOpError("result rank must equal input rank");
  if (iTy.getElementType() != rTy.getElementType())
    return emitOpError("tile is element-type-preserving");
  if ((int64_t)getReps().size() != rank)
    return emitOpError("reps must have one entry per dim (rank ") << rank << ")";
  for (int64_t d = 0; d < rank; ++d) {
    int64_t id = iTy.getDimSize(d), rd = rTy.getDimSize(d);
    if (ShapedType::isDynamic(id) || ShapedType::isDynamic(rd))
      continue;
    int64_t rep = cast<IntegerAttr>(getReps()[d]).getInt();
    if (id * rep != rd)
      return emitOpError("result extent ")
             << rd << " != input*reps (" << (id * rep) << ") at dim " << d;
  }
  return success();
}

LogicalResult FlipOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  if (getInput().getType() != getResult().getType())
    return emitOpError("flip is shape- and element-type-preserving");
  for (Attribute a : getAxes()) {
    int64_t ax = cast<IntegerAttr>(a).getInt();
    if (ax < 0 || ax >= iTy.getRank())
      return emitOpError("flip axis ")
             << ax << " out of range for rank-" << iTy.getRank() << " input";
  }
  return success();
}

LogicalResult TriuOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  if (iTy.getRank() != 2)
    return emitOpError("triu operates on a rank-2 tensor");
  if (getInput().getType() != getResult().getType())
    return emitOpError("triu is shape- and element-type-preserving");
  return success();
}

LogicalResult RandOp::verify() {
  auto rTy = cast<RankedTensorType>(getResult().getType());
  if (!rTy.getElementType().isF32())
    return emitOpError("rand produces a uniform[0,1) f32 tensor; result element "
                       "type must be f32");
  return success();
}

LogicalResult PoolOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  auto oTy = cast<RankedTensorType>(getOutput().getType());
  if (iTy.getRank() != 4 || oTy.getRank() != 4)
    return emitOpError("pool operates on rank-4 NCHW tensors");
  if (iTy.getElementType() != oTy.getElementType())
    return emitOpError("pool output element type must equal input");
  if (getFn() != "max" && getFn() != "avg")
    return emitOpError("pool fn must be 'max' or 'avg'");
  if (getIndices()) {
    if (getFn() != "max")
      return emitOpError("pool indices result is only valid for fn = 'max'");
    auto xTy = cast<RankedTensorType>(getIndices().getType());
    if (xTy.getShape() != oTy.getShape())
      return emitOpError("pool indices must share the output shape");
    if (!xTy.getElementType().isInteger(64) && !xTy.getElementType().isInteger(32))
      return emitOpError("pool indices must be an integer (i64/i32) tensor");
  }
  return success();
}

LogicalResult UpsampleOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  if (iTy.getRank() != 4 || rTy.getRank() != 4)
    return emitOpError("upsample operates on rank-4 NCHW tensors");
  if (iTy.getElementType() != rTy.getElementType())
    return emitOpError("upsample is element-type-preserving");
  if (getMode() != "nearest" && getMode() != "bicubic")
    return emitOpError("upsample mode must be 'nearest' or 'bicubic'");
  if (getScale().size() != 2)
    return emitOpError("upsample scale must contain exactly [sh, sw]");
  int64_t scale[2];
  for (unsigned index = 0; index < 2; ++index) {
    auto attr = dyn_cast<IntegerAttr>(getScale()[index]);
    if (!attr || attr.getInt() < 1)
      return emitOpError("upsample scale entries must be positive i64 integers");
    scale[index] = attr.getInt();
  }
  for (unsigned index = 0; index < 2; ++index) {
    int64_t inputExtent = iTy.getDimSize(index);
    int64_t resultExtent = rTy.getDimSize(index);
    if (!ShapedType::isDynamic(inputExtent) &&
        !ShapedType::isDynamic(resultExtent) && inputExtent != resultExtent)
      return emitOpError("upsample preserves the N and C extents");
  }
  if (!iTy.hasStaticShape() || !rTy.hasStaticShape())
    return success();
  for (unsigned index = 0; index < 2; ++index) {
    int64_t inputExtent = iTy.getDimSize(2 + index);
    int64_t expectedExtent =
        inputExtent > std::numeric_limits<int64_t>::max() / scale[index]
            ? -1
            : inputExtent * scale[index];
    if (expectedExtent < 0 || rTy.getDimSize(2 + index) != expectedExtent)
      return emitOpError("upsample result spatial extent does not equal input "
                         "extent multiplied by scale");
  }
  return success();
}

LogicalResult PolarOp::verify() {
  auto aTy = cast<RankedTensorType>(getAbs().getType());
  auto gTy = cast<RankedTensorType>(getAngle().getType());
  if (aTy != gTy)
    return emitOpError("polar abs and angle must have the same type");
  for (Value r : {getReal(), getImag()}) {
    auto rTy = cast<RankedTensorType>(r.getType());
    if (rTy.getShape() != aTy.getShape() ||
        rTy.getElementType() != aTy.getElementType())
      return emitOpError("polar real/imag must match the abs shape+element type");
  }
  return success();
}

LogicalResult ConvOp::verify() {
  auto iTy = cast<RankedTensorType>(getInput().getType());
  auto wTy = cast<RankedTensorType>(getWeight().getType());
  auto oTy = cast<RankedTensorType>(getOutput().getType());
  int64_t rank = iTy.getRank();
  if (rank < 3 || rank > 5)
    return emitOpError("conv input must be rank 3/4/5 ([N,Cin,*spatial])");
  int64_t R = rank - 2;
  if (wTy.getRank() != rank || oTy.getRank() != rank)
    return emitOpError("conv weight/output rank must match the input rank");
  if (iTy.getElementType() != wTy.getElementType() ||
      iTy.getElementType() != oTy.getElementType())
    return emitOpError("conv input/weight/output element types must match");
  int64_t groups = getGroups();
  if (groups < 1)
    return emitOpError("conv groups must be >= 1");
  if ((int64_t)getStride().size() != R || (int64_t)getPadding().size() != R ||
      (int64_t)getDilation().size() != R)
    return emitOpError("conv stride/padding/dilation must have length = "
                       "spatial rank");
  if (getBias()) {
    auto bTy = cast<RankedTensorType>(getBias().getType());
    if (bTy.getRank() != 1)
      return emitOpError("conv bias must be a rank-1 [Cout] tensor");
    if (bTy.getElementType() != iTy.getElementType())
      return emitOpError("conv bias element type must match the input");
  }
  if (!iTy.hasStaticShape() || !wTy.hasStaticShape() || !oTy.hasStaticShape())
    return success();
  int64_t Cin = iTy.getDimSize(1);
  int64_t Cout = wTy.getDimSize(0);
  if (Cin % groups != 0)
    return emitOpError("conv Cin must be divisible by groups");
  if (Cout % groups != 0)
    return emitOpError("conv Cout must be divisible by groups");
  if (wTy.getDimSize(1) != Cin / groups)
    return emitOpError("conv weight dim 1 must equal Cin/groups");
  if (oTy.getDimSize(0) != iTy.getDimSize(0) || oTy.getDimSize(1) != Cout)
    return emitOpError("conv output must be [N, Cout, *out_spatial]");
  if (getBias()) {
    auto bTy = cast<RankedTensorType>(getBias().getType());
    if (bTy.getDimSize(0) != Cout)
      return emitOpError("conv bias length must equal Cout");
  }
  auto asI64 = [](Attribute a) {
    return cast<IntegerAttr>(a).getInt();
  };
  for (int64_t j = 0; j < R; ++j) {
    int64_t H = iTy.getDimSize(2 + j);
    int64_t k = wTy.getDimSize(2 + j);
    int64_t s = asI64(getStride()[j]);
    int64_t p = asI64(getPadding()[j]);
    int64_t d = asI64(getDilation()[j]);
    if (s < 1 || d < 1)
      return emitOpError("conv stride/dilation must be >= 1");
    int64_t O = (H + 2 * p - d * (k - 1) - 1) / s + 1;
    if (oTy.getDimSize(2 + j) != O)
      return emitOpError("conv output spatial extent mismatch on axis ") << j;
  }
  return success();
}

LogicalResult ReduceOp::verify() {
  static constexpr llvm::StringRef kFns[] = {"sum", "mean", "max", "min",
                                             "prod", "any", "all"};
  bool ok = false;
  for (llvm::StringRef f : kFns)
    if (getFn() == f)
      ok = true;
  if (!ok)
    return emitOpError("unknown reduce fn '") << getFn() << "'";
  if (getAssoc() != "associative" && getAssoc() != "float_nondeterministic")
    return emitOpError("assoc must be 'associative' or 'float_nondeterministic'");
  return success();
}

LogicalResult ContractionOp::verify() {
  // Parse einsum-style "lhs,rhs->out" (exactly two inputs).
  llvm::StringRef sub = getSubscripts();
  size_t arrow = sub.find("->");
  if (arrow == llvm::StringRef::npos)
    return emitOpError("subscripts must contain '->'");
  llvm::StringRef ins = sub.substr(0, arrow);
  llvm::StringRef outLabels = sub.substr(arrow + 2);
  size_t comma = ins.find(',');
  if (comma == llvm::StringRef::npos || ins.substr(comma + 1).contains(','))
    return emitOpError("contraction expects exactly two inputs (one ',')");
  llvm::StringRef aLabels = ins.substr(0, comma);
  llvm::StringRef bLabels = ins.substr(comma + 1);

  auto aTy = cast<RankedTensorType>(getA().getType());
  auto bTy = cast<RankedTensorType>(getB().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());

  if (aLabels.size() != static_cast<size_t>(aTy.getRank()))
    return emitOpError("lhs labels '")
           << aLabels << "' (" << aLabels.size() << ") != operand rank "
           << aTy.getRank();
  if (bLabels.size() != static_cast<size_t>(bTy.getRank()))
    return emitOpError("rhs labels '")
           << bLabels << "' (" << bLabels.size() << ") != operand rank "
           << bTy.getRank();
  if (outLabels.size() != static_cast<size_t>(rTy.getRank()))
    return emitOpError("result labels '")
           << outLabels << "' (" << outLabels.size() << ") != result rank "
           << rTy.getRank();

  // Each index label must have one consistent static extent everywhere it
  // appears (dynamic dims are wildcards).
  llvm::DenseMap<char, int64_t> dims;
  auto bind = [&](llvm::StringRef labels, RankedTensorType ty) -> LogicalResult {
    for (size_t i = 0; i < labels.size(); ++i) {
      int64_t d = ty.getDimSize(i);
      if (ShapedType::isDynamic(d))
        continue;
      char c = labels[i];
      auto it = dims.find(c);
      if (it == dims.end())
        dims[c] = d;
      else if (it->second != d)
        return emitOpError("index '")
               << llvm::Twine(c) << "' has conflicting extents " << it->second
               << " and " << d;
    }
    return success();
  };
  if (failed(bind(aLabels, aTy)) || failed(bind(bLabels, bTy)))
    return failure();

  for (size_t i = 0; i < outLabels.size(); ++i) {
    char c = outLabels[i];
    auto it = dims.find(c);
    if (it == dims.end())
      return emitOpError("result index '")
             << llvm::Twine(c) << "' does not appear in any input";
    int64_t d = rTy.getDimSize(i);
    if (!ShapedType::isDynamic(d) && it->second != d)
      return emitOpError("result index '")
             << llvm::Twine(c) << "' extent " << d << " != input extent "
             << it->second;
  }
  return success();
}

LogicalResult LoopOp::verify() {
  unsigned nScan = getScanInputs().size();
  unsigned nCarry = getInits().size();
  Value upperBound = getUpperBound();
  bool dynamicTrip = static_cast<bool>(upperBound);

  auto isLoopIntegerElement = [](Type type) {
    if (type.isIndex())
      return true;
    auto integer = dyn_cast<IntegerType>(type);
    // A boolean is an integer type in MLIR, but it is not a meaningful loop
    // bound or induction carrier.  Keep this identical for the bound and the
    // loop-local induction value so the J/MLIR ABI cannot drift by width.
    return integer && !integer.isInteger(1);
  };

  if (dynamicTrip) {
    auto upperTy = dyn_cast<RankedTensorType>(upperBound.getType());
    if (!upperTy || upperTy.getRank() != 0 ||
        !isLoopIntegerElement(upperTy.getElementType()))
      return emitOpError(
          "dynamic upper_bound must be a rank-0 non-boolean integer tensor");
    if (nScan != 0)
      return emitOpError(
          "dynamic upper_bound form cannot have scan inputs; use the static "
          "semantic-axis form for scanned loops");
  } else if (nScan == 0) {
    return emitOpError(
        "static semantic-axis loop requires at least one scan input");
  }

  if (nCarry != getResults().size())
    return emitOpError("number of inits (")
           << nCarry << ") must equal number of results (" << getResults().size()
           << ") — carry types are loop-invariant";
  if (getScanAxes().size() != nScan)
    return emitOpError("scan_axes (")
           << getScanAxes().size() << ") must have one entry per scan input ("
           << nScan << ")";
  for (auto [i, in] : llvm::enumerate(getScanInputs())) {
    int64_t ax = cast<IntegerAttr>(getScanAxes()[i]).getInt();
    int64_t rank = cast<RankedTensorType>(in.getType()).getRank();
    if (ax < 0 || ax >= rank)
      return emitOpError("scan_axes[")
             << i << "] = " << ax << " out of range for rank-" << rank
             << " scan input";
  }

  Block &body = getBody().front();
  unsigned inductionArgs = dynamicTrip ? 1 : 0;
  if (body.getNumArguments() != nScan + inductionArgs + nCarry)
    return emitOpError("body must have ")
           << (nScan + inductionArgs + nCarry)
           << " block args (scan slices + optional induction + carries), got "
           << body.getNumArguments();

  if (dynamicTrip) {
    auto upperTy = cast<RankedTensorType>(upperBound.getType());
    auto inductionTy =
        dyn_cast<RankedTensorType>(body.getArgument(nScan).getType());
    if (!inductionTy || inductionTy.getRank() != 0 ||
        !isLoopIntegerElement(inductionTy.getElementType()))
      return emitOpError(
          "dynamic induction block argument must be a rank-0 non-boolean "
          "integer tensor");
    if (inductionTy.getElementType() != upperTy.getElementType())
      return emitOpError(
          "dynamic induction block argument element type must equal "
          "upper_bound element type");
  }

  unsigned carryArgBase = nScan + inductionArgs;
  // Carry block args (after scan slices and the optional induction) and
  // results must match inits.
  for (auto [i, init] : llvm::enumerate(getInits())) {
    if (isa<TupleType>(init.getType()))
      return emitOpError(
          "tuple/Ragged descriptors are region-local and cannot cross a "
          "loop carry or result ABI");
    Type carryArgTy = body.getArgument(carryArgBase + i).getType();
    if (isa<TupleType>(carryArgTy) || isa<TupleType>(getResult(i).getType()))
      return emitOpError(
          "tuple/Ragged descriptors are region-local and cannot cross a "
          "loop carry or result ABI");
    if (carryArgTy != init.getType())
      return emitOpError("carry block arg ")
             << i << " type " << carryArgTy << " != init type " << init.getType();
    if (getResult(i).getType() != init.getType())
      return emitOpError("result ")
             << i << " type " << getResult(i).getType() << " != init type "
             << init.getType();
  }
  // The implicit terminator is a YieldOp (trait-enforced); check its arity/types.
  auto yield = cast<YieldOp>(body.getTerminator());
  if (yield.getResults().size() != nCarry)
    return emitOpError("yield yields ")
           << yield.getResults().size() << " values but loop has " << nCarry
           << " carries";
  for (auto [i, y] : llvm::enumerate(yield.getResults()))
    if (y.getType() != getResult(i).getType())
      return emitOpError("yield operand ")
             << i << " type " << y.getType() << " != result type "
             << getResult(i).getType();
  return success();
}
