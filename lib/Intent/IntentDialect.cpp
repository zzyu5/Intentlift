#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace intent;

#define GET_TYPEDEF_CLASSES
#include "Intent/IntentTypes.cpp.inc"

#include "Intent/IntentDialect.cpp.inc"

void IntentDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Intent/IntentTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/IntentOps.cpp.inc"
      >();
}
