"""SCRC verification entry point.

The package boundary is intentionally import-light.  Manifest, schema, and
backend-neutral contract consumers must be usable without importing Gate2's
numeric/runtime stack (and therefore without requiring NumPy).  The public
names below retain their historical import surface, but the implementation
modules are loaded only when a caller actually requests one of those names.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "gate1": ("intentir.verify.gate1", None),
    "gate2": ("intentir.verify.gate2", None),
    "BoundGate1Adapter": ("intentir.verify.adapters", "BoundGate1Adapter"),
    "TritonGate2ExecutionAdapter": (
        "intentir.verify.adapters",
        "TritonGate2ExecutionAdapter",
    ),
    "TypeCheckDiagnostic": ("intentir.verify.typecheck", "TypeCheckDiagnostic"),
    "TypeCheckErrorCode": ("intentir.verify.typecheck", "TypeCheckErrorCode"),
    "TypeCheckReport": ("intentir.verify.typecheck", "TypeCheckReport"),
    "TypeCheckStatus": ("intentir.verify.typecheck", "TypeCheckStatus"),
    "type_check": ("intentir.verify.typecheck", "type_check"),
    "CandidateLiftError": ("intentir.verify.scrc_runner", "CandidateLiftError"),
    "FeedbackKind": ("intentir.verify.scrc_runner", "FeedbackKind"),
    "HarnessBindingInstance": (
        "intentir.verify.scrc_runner",
        "HarnessBindingInstance",
    ),
    "LiftCallResult": ("intentir.verify.scrc_runner", "LiftCallResult"),
    "LiftFeedback": ("intentir.verify.scrc_runner", "LiftFeedback"),
    "MainStatus": ("intentir.verify.scrc_runner", "MainStatus"),
    "SCRCGateRunner": ("intentir.verify.scrc_runner", "SCRCGateRunner"),
    "SCRCInfrastructureError": (
        "intentir.verify.scrc_runner",
        "SCRCInfrastructureError",
    ),
    "SCRCRunDependencies": (
        "intentir.verify.scrc_runner",
        "SCRCRunDependencies",
    ),
    "SCRCRunReport": ("intentir.verify.scrc_runner", "SCRCRunReport"),
    "SCRCRunRequest": ("intentir.verify.scrc_runner", "SCRCRunRequest"),
    "SourceProgram": ("intentir.verify.scrc_runner", "SourceProgram"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value

__all__ = [
    "TypeCheckDiagnostic",
    "TypeCheckErrorCode",
    "TypeCheckReport",
    "TypeCheckStatus",
    "BoundGate1Adapter",
    "CandidateLiftError",
    "FeedbackKind",
    "HarnessBindingInstance",
    "LiftCallResult",
    "LiftFeedback",
    "MainStatus",
    "SCRCGateRunner",
    "SCRCInfrastructureError",
    "SCRCRunDependencies",
    "SCRCRunReport",
    "SCRCRunRequest",
    "SourceProgram",
    "TritonGate2ExecutionAdapter",
    "gate1",
    "gate2",
    "type_check",
]
