set(LLVM_TARGET_DEFINITIONS IntentOps.td)
intent_mlir_tablegen(IntentOps.h.inc -gen-op-decls)
intent_mlir_tablegen(IntentOps.cpp.inc -gen-op-defs)
intent_mlir_tablegen(IntentDialect.h.inc -gen-dialect-decls -dialect=intent)
intent_mlir_tablegen(IntentDialect.cpp.inc -gen-dialect-defs -dialect=intent)
add_public_tablegen_target(IntentIncGen)

add_subdirectory(Conversion)
