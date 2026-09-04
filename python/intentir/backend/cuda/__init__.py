"""NVIDIA CUDA compilation, selection, and execution backend.

Capability-registry readers must not import the CUDA compiler/runtime stack at
package import time.  Keep the public API stable while loading heavy modules
only for an operation that actually needs them.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "CudaJExecutionError": ("intentir.backend.cuda.program", "CudaJExecutionError"),
    "CudaJLowering": ("intentir.backend.cuda.program", "CudaJLowering"),
    "CudaJLoweringError": ("intentir.backend.cuda.program", "CudaJLoweringError"),
    "analyze_j_for_cuda": ("intentir.backend.cuda.program", "analyze_j_for_cuda"),
    "cuda_target": ("intentir.backend.cuda.program", "cuda_target"),
    "execute_j_cuda": ("intentir.backend.cuda.program", "execute_j_cuda"),
    "lower_j_to_cuda": ("intentir.backend.cuda.program", "lower_j_to_cuda"),
    "CUDA_LAYER_C_BACKEND_KIND": (
        "intentir.backend.cuda.search",
        "CUDA_LAYER_C_BACKEND_KIND",
    ),
    "build_cuda_target_plugin": (
        "intentir.backend.cuda.search",
        "build_cuda_target_plugin",
    ),
    "build_cuda_target_artifact": (
        "intentir.backend.cuda.search",
        "build_cuda_target_artifact",
    ),
    "cuda_allowed_authorizations": (
        "intentir.backend.cuda.search",
        "cuda_allowed_authorizations",
    ),
    "cuda_candidate_digest": (
        "intentir.backend.cuda.search",
        "cuda_candidate_digest",
    ),
    "cuda_candidate_key": ("intentir.backend.cuda.search", "cuda_candidate_key"),
    "cuda_layer_c_interface": (
        "intentir.backend.cuda.search",
        "cuda_layer_c_interface",
    ),
    "resolve_cuda_target_plugin": (
        "intentir.backend.cuda.search",
        "resolve_cuda_target_plugin",
    ),
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
    "CudaJExecutionError",
    "CudaJLowering",
    "CudaJLoweringError",
    "CUDA_LAYER_C_BACKEND_KIND",
    "analyze_j_for_cuda",
    "build_cuda_target_plugin",
    "build_cuda_target_artifact",
    "cuda_allowed_authorizations",
    "cuda_candidate_digest",
    "cuda_candidate_key",
    "cuda_layer_c_interface",
    "cuda_target",
    "execute_j_cuda",
    "lower_j_to_cuda",
    "resolve_cuda_target_plugin",
]
