"""Typed Layer-B projection used by deterministic production materialization.

The accepted J already owns membership and semantic closure.  Constructing M0
must therefore preserve the complete contract without asking a backend-shaped
matcher whether J is a supported source recipe.  This module projects every
authored domain/access expression through :mod:`intentir.ir.contract_ir` and
extracts only the two Layer-B facts directly needed by the current logical MLIR
emitter: explicit broadcast axes and proven gather/scatter bounds.

No Source, kernel name, C value, target, or runtime observation enters this
projection.  Target backends remain responsible for consuming or rejecting the
remaining preserved relations before applying transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from intentir.ir.canonical_j import (
    ContractExpr,
    direct_unit_dimension_refs,
    elaborate_j,
    j_elaboration,
)
from intentir.ir.contract_ir import CanonicalContractIR, project_contract_ir
from intentir.ir.canonical_construction import (
    unique_exact_rank_insertion_projection,
)
from intentir.ir.intent_op_contract import (
    _broadcast_projection,
    is_closed_boolean_dtype,
)
from intentir.ir.model import IntentOp
from intentir.verify.gate1.model import AliasKind, SemanticArchive


CONTRACT_MATERIALIZATION_PROTOCOL = (
    "intentir.typed-contract-materialization-projection.v1"
)


class ContractMaterializationError(ValueError):
    """A typed contract fact required by M0 cannot be projected uniquely."""


def _canonical_nodes(j: SemanticArchive) -> dict[str, IntentOp]:
    elaboration = j_elaboration(j)
    result = elaboration.construction_records.typed_operation_by_ref(j.node_ids)
    if result is None:
        raise ContractMaterializationError(
            "canonical operation references cannot be materialized"
        )
    return result


@dataclass(frozen=True)
class ContractAccessGuardProjection:
    access_ref: str
    value_ref: str
    node_ref: str
    consumer_result: str
    axes: tuple[int, ...]
    polarity: str
    access_kind: str
    tensor: str
    operand_index: int | None = None
    fill_value: Any = None
    store_profile: str | None = None

    def to_json(self) -> dict[str, Any]:
        result = {
            "proof": "authored_access_guard_relation",
            "access_ref": self.access_ref,
            "value_ref": self.value_ref,
            "node_ref": self.node_ref,
            "consumer_result": self.consumer_result,
            "axes": list(self.axes),
            "polarity": self.polarity,
            "access_kind": self.access_kind,
            "tensor": self.tensor,
        }
        if self.access_kind == "load":
            result.update(
                {
                    "operand_index": self.operand_index,
                    "fill_value": self.fill_value,
                    "materialization": "eager_logical_read",
                }
            )
        elif self.store_profile is not None:
            result["materialization"] = self.store_profile
        return result


@dataclass(frozen=True)
class ContractMaterializationProjection:
    """Complete typed contract plus the facts consumed by logical MLIR emission."""

    contract_ir: CanonicalContractIR
    broadcast_axes: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ]
    proven_index_bounds: frozenset[str]
    consumed: tuple[str, ...]
    predicated_access_count: int
    access_guard_relations: tuple[ContractAccessGuardProjection, ...]
    unsupported_lowering: tuple[str, ...]
    erased_fill_count: int = 0
    runtime_predicate_proofs: tuple[Any, ...] = ()

    @property
    def profile(self) -> str:
        return "typed_contract_ir_logical_graph"

    def emitter_broadcast_axes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            result: dict(operands) for result, operands in self.broadcast_axes
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CONTRACT_MATERIALIZATION_PROTOCOL,
            "profile": self.profile,
            "contract_ir": self.contract_ir.to_json(),
            "broadcast_axes": [
                [
                    result,
                    [[operand, list(axes)] for operand, axes in operands],
                ]
                for result, operands in self.broadcast_axes
            ],
            "proven_index_bounds": sorted(self.proven_index_bounds),
            "consumed": list(self.consumed),
            "predicated_access_count": self.predicated_access_count,
            "erased_fill_count": self.erased_fill_count,
            "runtime_predicate_proofs": [],
            "access_guard_relations": [
                relation.to_json() for relation in self.access_guard_relations
            ],
            "unsupported_lowering": list(self.unsupported_lowering),
        }


def _reference(expression: ContractExpr) -> str | None:
    if expression.op == "ref" and isinstance(expression.value, str):
        return expression.value
    return None


def _logical_access_coordinates(
    j: SemanticArchive,
    index: Any,
    target_shape: Sequence[int | str],
) -> tuple[str | int, ...] | None:
    """Recognize currently attached tensor ABI layouts from the typed AST.

    General ContractExpr lowering is available independently, but the current
    tensor-valued M0 ABI can attach only identity coordinates or a J-proved
    rank-2 row-major linearization.  Other formulas stay in ContractIR and are
    rejected by the target ContractView before codegen.
    """

    direct = index.direct_coordinates
    if direct is not None:
        return direct if len(direct) == len(target_shape) else None
    if len(target_shape) != 2 or len(index.components) != 1:
        return None
    expression = index.components[0]
    if expression.op != "add" or len(expression.args) != 2:
        return None
    trailing_extent = target_shape[1]
    value_dimensions = {
        relation.value_ref: relation.dimension_ref
        for relation in j.value_dimension_relations
    }

    def is_stride(term: ContractExpr) -> bool:
        if (
            term.op == "int"
            and isinstance(term.value, int)
            and not isinstance(term.value, bool)
        ):
            return type(trailing_extent) is int and term.value == trailing_extent
        name = _reference(term)
        return (
            name is not None
            and isinstance(trailing_extent, str)
            and value_dimensions.get(name) == trailing_extent
        )

    candidates: set[tuple[str, str]] = set()
    for product, column_term in (
        (expression.args[0], expression.args[1]),
        (expression.args[1], expression.args[0]),
    ):
        column = _reference(column_term)
        if column is None or product.op != "mul" or len(product.args) != 2:
            continue
        for row_term, stride_term in (
            (product.args[0], product.args[1]),
            (product.args[1], product.args[0]),
        ):
            row = _reference(row_term)
            if row is not None and row != column and is_stride(stride_term):
                candidates.add((row, column))
    return next(iter(candidates)) if len(candidates) == 1 else None


def project_contract_materialization(
    j: SemanticArchive,
    *,
    consume_loop_storage: bool = False,
) -> ContractMaterializationProjection:
    """Project accepted J without using a capability matcher as membership gate."""

    if not isinstance(j, SemanticArchive):
        raise TypeError("contract materialization requires canonical J")
    canonical = elaborate_j(j)
    contract_ir = project_contract_ir(canonical)
    nodes = _canonical_nodes(canonical)
    elaboration = j_elaboration(canonical)

    broadcasts: dict[str, dict[str, tuple[int, ...]]] = {}
    unsupported: set[str] = set()
    consumed: set[str] = {"contract_ir:typed"}
    consumed.update(
        f"domain:{domain.domain_id}:typed" for domain in contract_ir.domains
    )
    consumed.update(
        f"access:{access.access_id}:typed" for access in contract_ir.accesses
    )
    consumed.update(
        f"value_dimension:{relation.value_ref}:{relation.dimension_ref}:eq"
        for relation in canonical.value_dimension_relations
    )

    tensors = {
        tensor.name: tensor
        for tensor in [*canonical.intent.inputs, *canonical.intent.outputs]
    }
    value_types = elaboration.value_type_by_name
    for access in canonical.boundary.accesses:
        structured = elaboration.access_by_id[access.access_id]
        tensor = tensors[access.tensor]
        if _logical_access_coordinates(canonical, structured.index, tensor.shape) is None:
            unsupported.add(
                f"access {access.access_id} physical index is not a direct ordered "
                "axis vector or J-proved row-major linearization"
            )

    effects_by_access: dict[str, list[Any]] = {}
    for effect in canonical.boundary.effects:
        if effect.access_ref is not None:
            effects_by_access.setdefault(effect.access_ref, []).append(effect)
    expected_effect = {"load": "read", "store": "write", "update": "update"}
    for access in canonical.boundary.accesses:
        matches = effects_by_access.get(access.access_id, [])
        if len(matches) != 1:
            unsupported.add(
                f"access {access.access_id} requires exactly one matching effect"
            )
            continue
        effect = matches[0]
        if (
            effect.kind.value != expected_effect[access.kind.value]
            or effect.target != access.tensor
            or effect.node_ref != access.node_ref
        ):
            unsupported.add(
                f"access {access.access_id} disagrees with its effect"
            )

    parameters = {parameter.name: parameter for parameter in canonical.signature.parameters}
    declared_alias_kinds = {
        frozenset((alias.left, alias.right)): alias.kind
        for alias in canonical.boundary.aliases
    }
    for parameter in canonical.signature.parameters:
        if parameter.alias_of is None:
            continue
        aliased = parameters.get(parameter.alias_of)
        logical_pair = (
            frozenset((parameter.logical_ref, aliased.logical_ref))
            if aliased is not None
            and parameter.logical_ref is not None
            and aliased.logical_ref is not None
            else None
        )
        if (
            logical_pair is None
            or len(logical_pair) != 2
            or declared_alias_kinds.get(logical_pair) is not AliasKind.MUST_ALIAS
        ):
            unsupported.add(
                "typed contract projection does not implement physical alias ABI "
                f"{parameter.name}->{parameter.alias_of}; expected one "
                "distinct logical pair with Boundary kind='must_alias'"
            )

    def register_broadcast(
        *, result: str, operand: str, axes: tuple[int, ...], context: str
    ) -> None:
        target = broadcasts.setdefault(result, {})
        previous = target.get(operand)
        if previous is not None and previous != axes:
            raise ContractMaterializationError(
                f"conflicting broadcast relations for {context}"
            )
        target[operand] = axes

    for relation in canonical.boundary.broadcasts:
        operation = nodes.get(relation.node_ref)
        if operation is None:
            raise ContractMaterializationError(
                f"broadcast references unknown node {relation.node_ref!r}"
            )
        if not isinstance(operation.result, str):
            raise ContractMaterializationError(
                "broadcast relation requires one named operation result"
            )
        if relation.operand not in operation.operands:
            raise ContractMaterializationError(
                f"broadcast operand {relation.operand!r} is not a direct operand "
                f"of {relation.node_ref!r}"
            )
        result_type = value_types.get(operation.result)
        result_shape = (
            list(result_type.shape)
            if result_type is not None and result_type.is_tensor
            else None
        )
        if not isinstance(result_shape, list):
            raise ContractMaterializationError(
                "broadcast relation requires one explicitly shaped result"
            )
        axes: list[int] = []
        for axis in relation.axes:
            if isinstance(axis, bool) or not isinstance(axis, int):
                raise ContractMaterializationError(
                    "production MLIR broadcast axes must be canonical integers"
                )
            if axis < 0 or axis >= len(result_shape):
                raise ContractMaterializationError(
                    f"broadcast axis {axis} is outside result rank {len(result_shape)}"
                )
            axes.append(axis)
        if len(axes) != len(set(axes)):
            raise ContractMaterializationError("broadcast axes must be unique")
        source_type = value_types.get(relation.operand)
        result_type = value_types.get(operation.result)
        if (
            source_type is None
            or result_type is None
            or not source_type.is_tensor
            or not result_type.is_tensor
        ):
            raise ContractMaterializationError(
                "broadcast relation requires closed tensor operand and result types"
            )
        try:
            projected_shape, inserted_axes = _broadcast_projection(
                list(source_type.shape),
                axes,
                list(result_type.shape),
                unit_dimensions=direct_unit_dimension_refs(
                    canonical.dimension_constraints
                ),
            )
        except (NotImplementedError, ValueError) as exc:
            raise ContractMaterializationError(
                f"broadcast relation cannot project its typed operand: {exc}"
            ) from exc
        if projected_shape is None or len(projected_shape) != len(result_type.shape):
            raise ContractMaterializationError(
                "broadcast relation has no closed target-rank projection"
            )
        if len(source_type.shape) != len(
            [
                index
                for index in range(len(result_type.shape))
                if index not in set(inserted_axes)
            ]
        ):
            raise ContractMaterializationError(
                "broadcast relation does not preserve the operand rank"
            )
        projected = tuple(axes)
        register_broadcast(
            result=operation.result,
            operand=relation.operand,
            axes=projected,
            context=f"broadcast {relation.node_ref!r}/{relation.operand!r}",
        )
        consumed.add(
            "broadcast:"
            f"{relation.node_ref}:{relation.operand}:"
            f"{','.join(str(axis) for axis in projected)}"
        )

    # A guarded store is either the established inout/select merge or a closed
    # output-only realization guard.  The latter keeps the physical i1 out of
    # the logical output producer: M0 receives the complete logical output,
    # while the receipt preserves that a guarded physical realization was
    # authorized and consumed.  Guarded loads may be implemented as eager
    # logical reads because M0 receives the complete logical tensor; the
    # authored fill/guard remains in ContractIR and no out-of-domain physical
    # address is generated by this tensor ABI.
    accesses = {item.access_id: item for item in canonical.boundary.accesses}
    input_names = {item.name for item in canonical.intent.inputs}
    output_names = {item.name for item in canonical.intent.outputs}
    output_parameter_counts: dict[str, int] = {}
    for parameter in canonical.signature.parameters:
        if parameter.mode.value == "output" and parameter.logical_ref is not None:
            output_parameter_counts[parameter.logical_ref] = (
                output_parameter_counts.get(parameter.logical_ref, 0) + 1
            )
    consumer_extensions_by_access: dict[str, list[Any]] = {}
    for extension in canonical.boundary.consumer_extensions:
        consumer_extensions_by_access.setdefault(
            extension.access_ref, []
        ).append(extension)
    canonical_output_bindings = {
        item.output: item for item in j_elaboration(canonical).output_bindings
    }
    guard_projections: list[ContractAccessGuardProjection] = []
    for relation in canonical.boundary.access_guard_relations:
        access = accesses[relation.access_ref]
        bindings: list[tuple[str, Any, int | None]] = []
        extensions = consumer_extensions_by_access.get(
            relation.access_ref, []
        )
        if access.kind.value == "store":
            if access.node_ref is None:
                continue
            operation = nodes.get(access.node_ref)
            if operation is None or not isinstance(operation.result, str):
                continue
            bindings.append((access.node_ref, operation, None))
        elif access.kind.value == "load":
            if access.fill_value is None:
                continue
            if extensions:
                for extension in extensions:
                    operation = nodes.get(extension.consumer_ref)
                    operand_index = extension.operand_index
                    if (
                        operation is None
                        or not isinstance(operation.result, str)
                        or isinstance(operand_index, bool)
                        or not isinstance(operand_index, int)
                        or operand_index < 0
                        or operand_index >= len(operation.operands)
                    ):
                        continue
                    bindings.append(
                        (extension.consumer_ref, operation, operand_index)
                    )
            elif access.node_ref is not None:
                operation = nodes.get(access.node_ref)
                if operation is None or not isinstance(operation.result, str):
                    continue
                positions = [
                    index
                    for index, operand in enumerate(operation.operands)
                    if operand == access.tensor
                ]
                if len(positions) != 1:
                    continue
                bindings.append((access.node_ref, operation, positions[0]))
            else:
                continue
        else:
            continue

        axes = tuple(relation.axes)
        projected_for_relation = False
        for node_ref, operation, operand_index in bindings:
            if access.kind.value == "store":
                output_binding = canonical_output_bindings.get(access.tensor)
                is_output_only = (
                    access.tensor in output_names
                    and access.tensor not in input_names
                    and output_parameter_counts.get(access.tensor) == 1
                )
                if is_output_only:
                    parsed_access = elaboration.access_by_id.get(access.access_id)
                    parsed_predicate = (
                        parsed_access.predicate if parsed_access is not None else None
                    )
                    if not (
                        access.predicate == "true"
                        and parsed_predicate is not None
                        and parsed_predicate.op == "bool"
                        and parsed_predicate.value is True
                        and not parsed_predicate.args
                    ):
                        raise ContractMaterializationError(
                            "output-only guarded store requires the exact literal "
                            "logical predicate true"
                        )
                    if (
                        output_binding is None
                        or output_binding.value != operation.result
                        or output_binding.node_ref != node_ref
                    ):
                        raise ContractMaterializationError(
                            "output-only guarded store owner must equal the exact "
                            "canonical output binding"
                        )
                    if relation.value_ref in operation.operands:
                        raise ContractMaterializationError(
                            "output-only physical guard must not be a logical "
                            "output-producer operand"
                        )
                    if any(
                        item.operand == relation.value_ref
                        and item.node_ref == node_ref
                        for item in canonical.boundary.broadcasts
                    ):
                        raise ContractMaterializationError(
                            "output-only access guard is the unique physical-guard "
                            "axes authority"
                        )
                else:
                    if (
                        operation.op != "select"
                        or len(operation.operands) != 3
                        or operation.operands[0] != relation.value_ref
                    ):
                        continue
                    active_index = (
                        1 if relation.polarity.value == "positive" else 2
                    )
                    inactive_index = 2 if active_index == 1 else 1
                    if (
                        operation.operands[active_index] == access.tensor
                        or operation.operands[inactive_index] != access.tensor
                    ):
                        continue
                    # The ABI/output relationship is owned by the elaborated
                    # output record.  Boundary output_mappings are sparse
                    # store/inout anchors and must not become a second output
                    # authority for materialization.
                    if (
                        output_binding is None
                        or output_binding.value != operation.result
                        or output_binding.node_ref != node_ref
                    ):
                        continue
            result_type = value_types.get(operation.result)
            result_shape = (
                list(result_type.shape)
                if result_type is not None and result_type.is_tensor
                else None
            )
            if not isinstance(result_shape, list) or any(
                axis < 0 or axis >= len(result_shape) for axis in axes
            ):
                continue
            if access.kind.value == "store" and is_output_only:
                guard_type = value_types.get(relation.value_ref)
                target_type = value_types.get(access.tensor)
                if (
                    guard_type is None
                    or target_type is None
                    or not guard_type.is_tensor
                    or not is_closed_boolean_dtype(guard_type.dtype)
                    or result_type != target_type
                ):
                    raise ContractMaterializationError(
                        "output-only guarded store requires exact i1 guard and "
                        "output result types"
                    )
                projected_axes = unique_exact_rank_insertion_projection(
                    guard_type.shape,
                    target_type.shape,
                )
                if projected_axes != axes:
                    raise ContractMaterializationError(
                        "output-only guarded store guard cannot project to its "
                        "output through the canonical exact rank-insertion rule"
                    )
            if relation.value_ref in operation.operands:
                register_broadcast(
                    result=operation.result,
                    operand=relation.value_ref,
                    axes=axes,
                    context=f"access guard {relation.access_ref!r}",
                )
            guard_projections.append(
                ContractAccessGuardProjection(
                    relation.access_ref,
                    relation.value_ref,
                    node_ref,
                    operation.result,
                    axes,
                    relation.polarity.value,
                    access.kind.value,
                    access.tensor,
                    operand_index,
                    access.fill_value,
                    (
                        "output_only_realization_guard"
                        if access.kind.value == "store" and is_output_only
                        else None
                    ),
                )
            )
            projected_for_relation = True
        if projected_for_relation:
            consumed.add(
                "access_guard:"
                f"{relation.access_ref}:{relation.value_ref}:"
                f"{','.join(str(axis) for axis in axes)}:"
                f"{relation.polarity.value}"
            )

    proven_bounds = frozenset(
        relation.node_ref for relation in canonical.boundary.index_bounds
    )
    consumed.update(f"index_bounds:{node_ref}" for node_ref in proven_bounds)
    if consume_loop_storage:
        consumed.update(
            f"storage:{storage.storage_id}" for storage in canonical.storage
        )

    serialized_broadcasts = tuple(
        (result, tuple(sorted(operands.items())))
        for result, operands in sorted(broadcasts.items())
    )
    return ContractMaterializationProjection(
        contract_ir=contract_ir,
        broadcast_axes=serialized_broadcasts,
        proven_index_bounds=proven_bounds,
        consumed=tuple(sorted(consumed)),
        predicated_access_count=sum(
            access.predicate is not None for access in contract_ir.accesses
        ),
        access_guard_relations=tuple(guard_projections),
        unsupported_lowering=tuple(sorted(unsupported)),
    )


__all__ = [
    "CONTRACT_MATERIALIZATION_PROTOCOL",
    "ContractAccessGuardProjection",
    "ContractMaterializationError",
    "ContractMaterializationProjection",
    "project_contract_materialization",
]
