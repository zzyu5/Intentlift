"""Target-neutral IntentIR Layer-A operation model.

Canonical J=A+B is the single ``SemanticArchive`` object used by Lift, Gate2
and target backends; it is intentionally not wrapped or re-serialized here.
"""

from .model import (
    EffectIfArm,
    IntentFunction,
    IntentOp,
    IntentReturn,
    LoopScanInput,
    LoopStateBinding,
    LoopTrip,
    LoopValueType,
    Tensor,
    ValueType,
    intent_op_producers,
    ValueIfArm,
    intent_op_regions,
    validate_ragged_region_boundaries,
    validate_loop_wiring,
    validate_intent_function_wiring,
    walk_intent_op_paths,
    walk_intent_ops,
    walk_intent_op_scopes,
)

__all__ = [
    "EffectIfArm", "IntentFunction", "IntentOp", "IntentReturn", "LoopScanInput",
    "LoopStateBinding", "LoopTrip", "LoopValueType", "Tensor",
    "ValueIfArm", "ValueType", "intent_op_producers", "intent_op_regions",
    "validate_ragged_region_boundaries",
    "validate_loop_wiring", "validate_intent_function_wiring",
    "walk_intent_op_paths", "walk_intent_ops", "walk_intent_op_scopes",
]
