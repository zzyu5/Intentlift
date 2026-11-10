#ifndef INTENT_INTENTOPS_H
#define INTENT_INTENTOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Intent/IntentOps.h.inc"

namespace intent {

// Ragged carrier types are part of the cross-backend ABI.  Keep this helper in
// the public Intent op header so the verifier and every lowering consumer use
// the same closed set: signless i8/i16/i32/i64, unsigned ui8/ui16/ui32, and
// logical index (materialized as i64).  i1/bool, ui64, explicit signed integer
// types, and other widths are not Ragged carriers.
inline bool isRaggedCarrierElementType(mlir::Type type) {
  if (type.isIndex())
    return true;
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!integer || integer.isSigned())
    return false;
  switch (integer.getWidth()) {
  case 8:
  case 16:
  case 32:
    return true;
  case 64:
    return !integer.isUnsigned();
  default:
    return false;
  }
}

inline bool isRaggedCarrierTensorType(mlir::Type type) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return tensor && tensor.getRank() == 1 &&
         isRaggedCarrierElementType(tensor.getElementType());
}

} // namespace intent

#endif // INTENT_INTENTOPS_H
