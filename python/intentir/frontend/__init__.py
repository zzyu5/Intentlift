"""Front-end: lift IntentIR JSON into the `intent` MLIR dialect.

The module is an import-light package boundary.  Contract and manifest tools
can inspect the repository without importing the NumPy-backed materializer;
the emitters are loaded only when a caller requests them.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "emit_j_module": ("intentir.frontend.canonical_j", "emit_j_module"),
    "emit_body_function": ("intentir.frontend.emit_mlir", "emit_body_function"),
    "emit_loop_module": ("intentir.frontend.emit_mlir", "emit_loop_module"),
    "emit_module": ("intentir.frontend.emit_mlir", "emit_module"),
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
    "emit_body_function",
    "emit_j_module",
    "emit_loop_module",
    "emit_module",
]
