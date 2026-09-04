"""Shared fail-closed consumers for target-independent logical updates."""

from __future__ import annotations

from collections.abc import Sequence

from intentir.ir.model import IntentOp, walk_intent_op_paths
from intentir.verify.gate1.model import (
    DuplicatePolicy,
    LogicalUpdateSpec,
    SemanticArchive,
    UpdateCombiner,
    UpdateOrder,
)


def canonical_node_table(j: SemanticArchive) -> dict[str, IntentOp]:
    """Resolve canonical node ids to their typed Layer-A operations."""

    by_path = dict(walk_intent_op_paths(j.intent.body))
    return {
        node_ref: by_path[path]
        for node_ref, path in j.node_ids.items()
        if path in by_path
    }


def consume_exact_internal_overwrite_update(
    j: SemanticArchive,
    item: LogicalUpdateSpec,
    *,
    nodes: dict[str, IntentOp] | None = None,
) -> str | None:
    """Consume the functional internal-scatter contract shared by all targets.

    This is the structured-attention scratch/state form: the authored update and
    Layer-A scatter agree exactly, destinations are unique, and input order is
    preserved.  It needs no target atomic or racing-write permission.
    """

    boundary_values = {
        tensor.name for tensor in [*j.intent.inputs, *j.intent.outputs]
    }
    if all(
        name in boundary_values
        for name in (item.base, item.indices, item.values)
    ):
        return None
    operation = (nodes or canonical_node_table(j)).get(item.node_ref)
    if (
        operation is None
        or operation.op != "scatter"
        or operation.operands != [item.base, item.indices, item.values]
        or operation.attrs.get("axis") != item.axis
        or operation.attrs.get("mode", "along_axis") != "along_axis"
        or operation.attrs.get("reduce", "none") != "none"
        or item.combiner is not UpdateCombiner.OVERWRITE
        or item.duplicate_policy is not DuplicatePolicy.FORBID
        or item.order is not UpdateOrder.INPUT_ORDER
    ):
        return None
    return (
        "update:"
        f"{item.node_ref}:{item.combiner.value}:"
        f"{item.duplicate_policy.value}:{item.order.value}"
    )


__all__ = [
    "canonical_node_table",
    "consume_exact_internal_overwrite_update",
]
