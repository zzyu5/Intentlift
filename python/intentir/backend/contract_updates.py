"""Shared fail-closed consumers for target-independent logical updates."""

from __future__ import annotations

from collections.abc import Sequence

from intentir.ir.canonical_j import j_elaboration
from intentir.ir.model import IntentOp
from intentir.verify.gate1.model import (
    DuplicatePolicy,
    LogicalUpdateSpec,
    SemanticArchive,
    UpdateCombiner,
    UpdateOrder,
)


def canonical_node_table(j: SemanticArchive) -> dict[str, IntentOp]:
    """Resolve canonical node ids to their typed Layer-A operations."""

    nodes = j_elaboration(j).typed_operation_by_ref(j.node_ids)
    if nodes is None:
        raise ValueError(
            "logical-update consumption requires canonical operation records"
        )
    return nodes


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

    # Output identity/order is owned by the canonical construction record.
    # Keep this update-admission projection from recreating it from the raw
    # output declaration wire.
    boundary_values = {
        *(tensor.name for tensor in j.intent.inputs),
        *(binding.output for binding in j_elaboration(j).output_bindings),
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
        or operation.attrs["mode"] != "along_axis"
        or operation.attrs["reduce"] != "none"
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
