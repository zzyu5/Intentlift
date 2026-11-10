"""RISC-V Vector production backend.

The package export surface is lazy so target-neutral registry and manifest
readers do not import the LLVM/native compiler path unless requested.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "RiscvCompiledObject": ("intentir.backend.riscv.compile", "RiscvCompiledObject"),
    "RiscvJLowering": ("intentir.backend.riscv.compile", "RiscvJLowering"),
    "RiscvJLoweringError": ("intentir.backend.riscv.compile", "RiscvJLoweringError"),
    "RiscvLoweringCandidate": ("intentir.backend.riscv.compile", "RiscvLoweringCandidate"),
    "RiscvSchedule": ("intentir.backend.riscv.compile", "RiscvSchedule"),
    "RiscvTransformationRecord": ("intentir.backend.riscv.compile", "RiscvTransformationRecord"),
    "RiscvTransformationState": ("intentir.backend.riscv.compile", "RiscvTransformationState"),
    "compile_riscv_plugin_object": ("intentir.backend.riscv.compile", "compile_riscv_plugin_object"),
    "compile_riscv_object": ("intentir.backend.riscv.compile", "compile_riscv_object"),
    "compile_riscv_target_plugin": ("intentir.backend.riscv.compile", "compile_riscv_target_plugin"),
    "lower_j_to_riscv": ("intentir.backend.riscv.compile", "lower_j_to_riscv"),
    "RiscvContractError": ("intentir.backend.riscv.contract", "RiscvContractError"),
    "RiscvContractView": ("intentir.backend.riscv.contract", "RiscvContractView"),
    "derive_riscv_contract_view": ("intentir.backend.riscv.contract", "derive_riscv_contract_view"),
    "riscv_allowed_authorizations": ("intentir.backend.riscv.contract", "riscv_allowed_authorizations"),
    "validate_riscv_target_contract": ("intentir.backend.riscv.contract", "validate_riscv_target_contract"),
    "RiscvSearchPlan": ("intentir.backend.riscv.search", "RiscvSearchPlan"),
    "build_riscv_target_plugin": ("intentir.backend.riscv.search", "build_riscv_target_plugin"),
    "build_riscv_target_artifact": ("intentir.backend.riscv.search", "build_riscv_target_artifact"),
    "plan_riscv_search": ("intentir.backend.riscv.search", "plan_riscv_search"),
    "resolve_riscv_target_plugin": ("intentir.backend.riscv.search", "resolve_riscv_target_plugin"),
    "riscv_candidate_digest": ("intentir.backend.riscv.search", "riscv_candidate_digest"),
    "riscv_candidate_key": ("intentir.backend.riscv.search", "riscv_candidate_key"),
    "riscv_layer_c_interface": ("intentir.backend.riscv.search", "riscv_layer_c_interface"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "RiscvCompiledObject",
    "RiscvJLowering",
    "RiscvJLoweringError",
    "RiscvContractError",
    "RiscvContractView",
    "RiscvLoweringCandidate",
    "RiscvSchedule",
    "RiscvSearchPlan",
    "RiscvTransformationRecord",
    "RiscvTransformationState",
    "build_riscv_target_plugin",
    "build_riscv_target_artifact",
    "compile_riscv_plugin_object",
    "compile_riscv_object",
    "compile_riscv_target_plugin",
    "lower_j_to_riscv",
    "derive_riscv_contract_view",
    "plan_riscv_search",
    "resolve_riscv_target_plugin",
    "riscv_allowed_authorizations",
    "riscv_candidate_digest",
    "riscv_candidate_key",
    "riscv_layer_c_interface",
    "validate_riscv_target_contract",
]
