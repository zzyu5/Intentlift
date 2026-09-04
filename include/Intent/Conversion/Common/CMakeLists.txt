set(LLVM_TARGET_DEFINITIONS Passes.td)
intent_mlir_tablegen(Passes.h.inc -gen-pass-decls -name IntentConversionCommon)
add_public_tablegen_target(IntentConversionCommonPassIncGen)
