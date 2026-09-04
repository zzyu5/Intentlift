"""Registered J-only materialization of dense predicated loads.

The ordinary functional-boundary projection erases only predicates that are
already the complete logical tensor domain.  Some accepted J artifacts carry
runtime-valid extents or an access-local boolean guard instead.  This module
turns those exact Layer-B load predicates into ordinary Layer-A ``iota`` /
comparison / ``select`` operations before Intent MLIR is emitted.

The transformation is deliberately narrow and fail-closed:

* only flat, direct dense loads are accepted;
* predicates are conjunctions of positive scalar boolean guards, zero lower
  bounds, and ``axis < scalar-or-constant`` upper bounds;
* an omitted fill is accepted only when the guarded producer has one direct,
  output-reaching ``select`` use that makes the value unobservable outside the
  guard;
* stores are never widened or rewritten here.

No Source, kernel name, Gate evidence, C, target, or runtime value enters this
analysis.  The returned function is ephemeral compiler state, not another J.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.frontend.emit_mlir import BroadcastAxes
from intentir.ir.canonical_j import (
    ContractExpr,
    elaborate_j,
    j_elaboration,
)
from intentir.ir.contract_materialization import (
    ContractMaterializationProjection,
    project_contract_materialization,
)
from intentir.ir.model import IntentFunction, IntentOp, walk_intent_op_paths
from intentir.verify.gate1.model import SemanticArchive


DENSE_PREDICATE_MATERIALIZATION_PROTOCOL = (
    "intentir.j-dense-predicate-materialization.v1"
)
BOUNDARY_MATERIALIZATION_PROTOCOL = "intentir.j-boundary-materialization.v1"


class DensePredicateMaterializationError(ValueError):
    pass


@dataclass(frozen=True)
class DensePredicateProjection:
    guards: frozenset[str]
    lower_axes: frozenset[str]
    upper_bounds: tuple[tuple[str, int | str], ...]


@dataclass(frozen=True)
class PredicateMaterializationRecord:
    access_id: str
    node_ref: str
    tensor: str
    authored_fill: bool
    guard_dominated_fill: bool
    guards: tuple[str, ...]
    upper_bounds: tuple[tuple[str, int | str], ...]
    erased_full_domain_terms: tuple[tuple[str, int | str], ...]
    generated_values: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": DENSE_PREDICATE_MATERIALIZATION_PROTOCOL,
            "access_id": self.access_id,
            "node_ref": self.node_ref,
            "tensor": self.tensor,
            "authored_fill": self.authored_fill,
            "guard_dominated_fill": self.guard_dominated_fill,
            "guards": list(self.guards),
            "upper_bounds": [list(item) for item in self.upper_bounds],
            "erased_full_domain_terms": [
                list(item) for item in self.erased_full_domain_terms
            ],
            "generated_values": list(self.generated_values),
        }


@dataclass(frozen=True)
class BoundaryMaterializationProjection:
    """Projection of one rewritten boundary view."""

    logical_projection: ContractMaterializationProjection
    predicate_materializations: tuple[PredicateMaterializationRecord, ...]
    synthetic_broadcast_axes: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ] = ()
    materialized_access_guards: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.predicate_materializations:
            raise DensePredicateMaterializationError(
                "boundary materialization requires at least one predicate record"
            )
        if self.logical_projection.runtime_predicate_proofs:
            raise DensePredicateMaterializationError(
                "production boundary materialization cannot use runtime proofs"
            )

    @property
    def profile(self) -> str:
        return "functional_dense_with_materialized_predicates"

    @property
    def runtime_predicate_proofs(self) -> tuple[Any, ...]:
        return ()

    @property
    def unsupported_lowering(self) -> tuple[str, ...]:
        return self.logical_projection.unsupported_lowering

    @property
    def proven_index_bounds(self) -> frozenset[str]:
        return self.logical_projection.proven_index_bounds

    @property
    def consumed(self) -> tuple[str, ...]:
        extra = (
            f"predicate_materialization:{item.access_id}"
            for item in self.predicate_materializations
        )
        return tuple(
            sorted(
                {
                    *self.logical_projection.consumed,
                    *extra,
                    *self.materialized_access_guards,
                }
            )
        )

    @property
    def predicated_access_count(self) -> int:
        return self.logical_projection.predicated_access_count

    @property
    def erased_fill_count(self) -> int:
        return self.logical_projection.erased_fill_count

    def emitter_broadcast_axes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        result = self.logical_projection.emitter_broadcast_axes()
        for output, operands in self.synthetic_broadcast_axes:
            target = result.setdefault(output, {})
            for operand, axes in operands:
                previous = target.get(operand)
                if previous is not None and previous != axes:
                    raise DensePredicateMaterializationError(
                        "synthetic broadcast conflicts with J boundary projection"
                    )
                target[operand] = axes
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": BOUNDARY_MATERIALIZATION_PROTOCOL,
            "profile": self.profile,
            "logical_projection": self.logical_projection.to_json(),
            "predicate_materializations": [
                item.to_json() for item in self.predicate_materializations
            ],
            "synthetic_broadcast_axes": [
                [
                    output,
                    [[operand, list(axes)] for operand, axes in operands],
                ]
                for output, operands in self.synthetic_broadcast_axes
            ],
            "materialized_access_guards": list(self.materialized_access_guards),
            "consumed": list(self.consumed),
        }


@dataclass(frozen=True)
class DensePredicateFunctionMaterialization:
    function: IntentFunction
    projection: BoundaryMaterializationProjection


def _walk(
    body: Sequence[IntentOp],
    prefix: str = "body",
) -> tuple[tuple[str, IntentOp], ...]:
    return tuple(walk_intent_op_paths(body, prefix))


def _results(operation: IntentOp) -> tuple[str, ...]:
    if isinstance(operation.result, str):
        return (operation.result,)
    if isinstance(operation.result, list):
        return tuple(item for item in operation.result if isinstance(item, str))
    return ()


def _flatten_and(expression: ContractExpr) -> tuple[ContractExpr, ...]:
    if expression.op != "and":
        return (expression,)
    return tuple(
        nested
        for argument in expression.args
        for nested in _flatten_and(argument)
    )


def _positive_guards(
    expression: ContractExpr,
    logical_guards: frozenset[str],
) -> tuple[str, ...]:
    result: list[str] = []
    for term in _flatten_and(expression):
        if term.op == "ref" and term.value in logical_guards:
            assert isinstance(term.value, str)
            result.append(term.value)
            continue
        if (
            term.op == "not"
            and len(term.args) == 1
            and term.args[0].op == "ref"
            and term.args[0].value in logical_guards
        ):
            raise NotImplementedError(
                "dense predicate materialization does not implement negated guards"
            )
    return tuple(sorted(set(result)))


def _reference(expression: ContractExpr) -> str | None:
    if expression.op == "ref" and isinstance(expression.value, str):
        return expression.value
    return None


def _bound_value(expression: ContractExpr) -> int | str | None:
    if expression.op == "int" and isinstance(expression.value, int):
        return expression.value
    return _reference(expression)


def _project_typed_predicate(
    expression: ContractExpr,
    logical_guards: frozenset[str],
) -> DensePredicateProjection:
    """Extract the registered dense-mask subset from the canonical typed AST."""

    lower: set[str] = set()
    guards: set[str] = set()
    upper: dict[str, int | str] = {}
    for term in _flatten_and(expression):
        guard = _reference(term)
        if guard is None and term.op == "not" and len(term.args) == 1:
            guard = _reference(term.args[0])
        if guard in logical_guards:
            guards.add(guard)
            continue
        left, right = term.args if len(term.args) == 2 else (None, None)
        lower_axis: str | None = None
        if term.op == "le" and left is not None and right is not None:
            if left.op == "int" and left.value == 0:
                lower_axis = _reference(right)
        elif term.op == "ge" and left is not None and right is not None:
            if right.op == "int" and right.value == 0:
                lower_axis = _reference(left)
        if lower_axis is not None:
            lower.add(lower_axis)
            continue
        upper_axis: str | None = None
        extent: int | str | None = None
        if term.op == "lt" and left is not None and right is not None:
            upper_axis = _reference(left)
            extent = _bound_value(right)
        if upper_axis is None or extent is None:
            raise NotImplementedError(
                "dense predicate materialization supports only logical-domain "
                "bounds and explicit scalar guards"
            )
        previous = upper.get(upper_axis)
        if previous is not None and previous != extent:
            raise DensePredicateMaterializationError(
                f"predicate axis {upper_axis!r} has conflicting extents"
            )
        upper[upper_axis] = extent
    if not upper and not guards:
        raise NotImplementedError(
            "dense predicate materialization requires an upper bound or J guard"
        )
    return DensePredicateProjection(
        frozenset(guards),
        frozenset(lower),
        tuple(sorted(upper.items())),
    )


def _guard_dominates_output(
    j: SemanticArchive,
    operation: IntentOp,
    guard: str,
) -> bool:
    """Prove the narrow positive-guard select profile used by dense loads."""

    produced = _results(operation)
    if len(produced) != 1:
        return False
    value = produced[0]
    operations = [item for _path, item in _walk(j.intent.body)]
    uses = [
        (candidate, index)
        for candidate in operations
        for index, operand in enumerate(candidate.operands)
        if operand == value
    ]
    if len(uses) != 1:
        return False
    select, operand_index = uses[0]
    if (
        select.op != "select"
        or len(select.operands) != 3
        or select.operands[0] != guard
        or operand_index != 1
        or not isinstance(select.result, str)
    ):
        return False
    output_values = {
        item.value for item in j_elaboration(j).output_bindings
    }
    return select.result in output_values


def _shape_tuple(shape: Sequence[int | str]) -> tuple[int | str, ...]:
    return tuple(shape)


def _missing_axes(rank: int, retained_axis: int) -> tuple[int, ...]:
    return tuple(axis for axis in range(rank) if axis != retained_axis)


def _serialize_broadcasts(
    value: Mapping[str, Mapping[str, tuple[int, ...]]],
) -> tuple[tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...]:
    return tuple(
        (
            output,
            tuple(sorted(operands.items())),
        )
        for output, operands in sorted(value.items())
    )


def materialize_dense_load_predicates(
    j: SemanticArchive,
    function: IntentFunction,
) -> DensePredicateFunctionMaterialization | None:
    """Materialize unsupported dense load predicates into a function copy.

    ``None`` means that every predicate belongs to the ordinary functional
    projection profile and no graph rewrite is needed.
    """

    j = elaborate_j(j)
    if any(operation.op == "loop" for _path, operation in _walk(j.intent.body)):
        return None

    scalar_inputs = {
        item.name: item
        for item in j.intent.inputs
        if not item.shape
    }
    logical_guards = frozenset(
        name for name, tensor in scalar_inputs.items() if tensor.dtype == "bool"
    )
    dimensions = {
        extent
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for extent in tensor.shape
        if isinstance(extent, str) and extent != "?"
    }
    value_dimensions = {
        item.value_ref: item.dimension_ref
        for item in j.value_dimension_relations
    }
    parsed_accesses = j_elaboration(j).access_by_id
    tensors = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }

    candidates: list[tuple[Any, DensePredicateProjection, tuple[str, ...]]] = []
    for access in j.boundary.accesses:
        if access.predicate is None:
            continue
        structured_predicate = parsed_accesses[access.access_id].predicate
        if structured_predicate is None:  # pragma: no cover - canonical J owns this
            raise DensePredicateMaterializationError(
                f"access {access.access_id!r} lost its canonical predicate"
            )
        projected = _project_typed_predicate(
            structured_predicate, logical_guards
        )
        substituted = DensePredicateProjection(
            projected.guards,
            projected.lower_axes,
            tuple(
                (
                    axis,
                    value_dimensions.get(bound, bound)
                    if isinstance(bound, str)
                    else bound,
                )
                for axis, bound in projected.upper_bounds
            ),
        )
        cross_axis = any(
            isinstance(bound, str) and bound not in dimensions
            for _axis, bound in substituted.upper_bounds
        )
        if not substituted.guards and not cross_axis:
            continue
        if access.kind.value != "load":
            raise NotImplementedError(
                "dense predicate materialization never rewrites predicated stores"
            )
        guards = _positive_guards(structured_predicate, logical_guards)
        if set(guards) != set(substituted.guards):
            raise NotImplementedError(
                "dense predicate materialization requires positive scalar guards"
            )
        candidates.append((access, substituted, guards))

    if not candidates:
        return None

    transformed = copy.deepcopy(function)
    transformed_nodes: dict[str, IntentOp] = {}
    path_nodes = dict(_walk(transformed.body))
    for node_id, path in j.node_ids.items():
        if path in path_nodes:
            transformed_nodes[node_id] = path_nodes[path]
    for path, operation in path_nodes.items():
        transformed_nodes[path] = operation
        if operation.node_id is not None:
            transformed_nodes[operation.node_id] = operation

    prefix: list[IntentOp] = []
    broadcasts: dict[str, dict[str, tuple[int, ...]]] = {}
    records: list[PredicateMaterializationRecord] = []
    predicate_value_cache: dict[
        tuple[str, tuple[int | str, ...]],
        tuple[str, tuple[int | str, ...], tuple[str, ...]],
    ] = {}
    const_cache: dict[tuple[str, str], str] = {}
    counter = 0

    def fresh(kind: str) -> str:
        nonlocal counter
        value = f"__intentir_{kind}_{counter}"
        counter += 1
        return value

    def constant(dtype: str, value: Any) -> str:
        key = (dtype, repr(value))
        existing = const_cache.get(key)
        if existing is not None:
            return existing
        name = fresh("predicate_const")
        prefix.append(
            IntentOp(
                op="const",
                result=name,
                attrs={"value": value, "dtype": dtype},
                type={"dtype": dtype, "shape": []},
            )
        )
        const_cache[key] = name
        return name

    for access, predicate, guards in candidates:
        if access.node_ref is None:
            raise NotImplementedError(
                "dense predicate materialization requires a node-bound load"
            )
        consumer = transformed_nodes.get(access.node_ref)
        original_consumer = next(
            (
                operation
                for path, operation in _walk(j.intent.body)
                if path == j.node_ids.get(access.node_ref)
                or operation.node_id == access.node_ref
            ),
            None,
        )
        if consumer is None or original_consumer is None:
            raise DensePredicateMaterializationError(
                f"load {access.access_id!r} references an unknown node"
            )
        operand_positions = [
            index
            for index, operand in enumerate(consumer.operands)
            if operand == access.tensor
        ]
        if not operand_positions:
            raise DensePredicateMaterializationError(
                f"load {access.access_id!r} is not a direct consumer operand"
            )
        tensor = tensors.get(access.tensor)
        if tensor is None or not tensor.shape:
            raise NotImplementedError(
                "dense predicate materialization requires a ranked tensor load"
            )
        coordinates = parsed_accesses[access.access_id].index.direct_coordinates
        if coordinates is None or len(coordinates) != len(tensor.shape):
            raise NotImplementedError(
                "dense predicate materialization requires direct full-rank indices"
            )
        coordinate_axis = {
            coordinate: axis
            for axis, coordinate in enumerate(coordinates)
            if isinstance(coordinate, str)
        }
        if not predicate.lower_axes.issubset(set(coordinate_axis)):
            raise DensePredicateMaterializationError(
                "predicate lower bound is outside the dense load coordinates"
            )

        full_terms: list[tuple[str, int | str]] = []
        residual_upper: list[tuple[str, int | str]] = []
        for axis_ref, bound in predicate.upper_bounds:
            position = coordinate_axis.get(axis_ref)
            if position is None:
                raise DensePredicateMaterializationError(
                    "predicate upper bound is outside the dense load coordinates"
                )
            if bound == tensor.shape[position]:
                full_terms.append((axis_ref, bound))
            else:
                residual_upper.append((axis_ref, bound))

        fill = access.fill_value
        guard_dominated = False
        if fill is None:
            if (
                len(guards) != 1
                or not _guard_dominates_output(j, original_consumer, guards[0])
            ):
                raise NotImplementedError(
                    "predicate load without fill is not closed by one output select"
                )
            fill = 0
            guard_dominated = True

        predicate_key = (access.predicate, _shape_tuple(tensor.shape))
        cached = predicate_value_cache.get(predicate_key)
        generated: list[str] = []
        if cached is None:
            components: list[tuple[str, tuple[int | str, ...], int | None]] = []
            for guard in guards:
                components.append((guard, (), None))
            for axis_ref, bound in residual_upper:
                position = coordinate_axis[axis_ref]
                bound_dtype: str
                if isinstance(bound, str):
                    scalar = scalar_inputs.get(bound)
                    if scalar is None or scalar.dtype not in {"i32", "i64"}:
                        raise NotImplementedError(
                            "predicate upper bound requires an integer scalar input"
                        )
                    bound_value = bound
                    bound_dtype = scalar.dtype
                elif isinstance(bound, int) and not isinstance(bound, bool):
                    bound_dtype = "i64"
                    bound_value = constant(bound_dtype, bound)
                else:
                    raise NotImplementedError(
                        "predicate upper bound must be an integer or scalar ref"
                    )
                index_value = fresh("predicate_iota")
                extent = tensor.shape[position]
                prefix.append(
                    IntentOp(
                        op="iota",
                        result=index_value,
                        attrs={"n": extent, "dtype": bound_dtype},
                        type={"dtype": bound_dtype, "shape": [extent]},
                    )
                )
                compare_value = fresh("predicate_lt")
                prefix.append(
                    IntentOp(
                        op="map_binary",
                        result=compare_value,
                        operands=[index_value, bound_value],
                        attrs={"fn": "lt"},
                        type={"dtype": "bool", "shape": [extent]},
                    )
                )
                generated.extend((index_value, compare_value))
                components.append((compare_value, (extent,), position))

            if not components:
                raise DensePredicateMaterializationError(
                    "predicate candidate has no residual mask component"
                )
            active_positions = tuple(
                sorted(
                    {
                        position
                        for _value, _shape, position in components
                        if position is not None
                    }
                )
            )
            compact_shape = tuple(
                tensor.shape[position] for position in active_positions
            )
            compact_axis = {
                position: index
                for index, position in enumerate(active_positions)
            }
            if len(components) == 1:
                mask_value, mask_shape, mask_axis = components[0]
                mask_axes = (
                    _missing_axes(len(tensor.shape), mask_axis)
                    if mask_axis is not None and len(tensor.shape) > 1
                    else ()
                )
            else:
                left_value, left_shape, left_axis = components[0]
                for right_value, right_shape, right_axis in components[1:]:
                    combined = fresh("predicate_and")
                    prefix.append(
                        IntentOp(
                            op="map_binary",
                            result=combined,
                            operands=[left_value, right_value],
                            attrs={"fn": "logical_and"},
                            type={"dtype": "bool", "shape": list(compact_shape)},
                        )
                    )
                    operand_axes: dict[str, tuple[int, ...]] = {}
                    if left_axis is not None and len(left_shape) == 1:
                        operand_axes[left_value] = _missing_axes(
                            len(compact_shape), compact_axis[left_axis]
                        )
                    if right_axis is not None and len(right_shape) == 1:
                        operand_axes[right_value] = _missing_axes(
                            len(compact_shape), compact_axis[right_axis]
                        )
                    if operand_axes:
                        broadcasts[combined] = operand_axes
                    generated.append(combined)
                    left_value = combined
                    left_shape = compact_shape
                    left_axis = None
                mask_value, mask_shape, mask_axis = (
                    left_value,
                    left_shape,
                    left_axis,
                )
                mask_axes = tuple(
                    axis
                    for axis in range(len(tensor.shape))
                    if axis not in active_positions
                ) if compact_shape else ()
            predicate_value_cache[predicate_key] = (
                mask_value,
                mask_shape,
                mask_axes,
            )
        else:
            mask_value, mask_shape, mask_axes = cached

        fill_value = constant(tensor.dtype, fill)
        selected = fresh("predicate_select")
        prefix.append(
            IntentOp(
                op="select",
                result=selected,
                operands=[mask_value, access.tensor, fill_value],
                type={"dtype": tensor.dtype, "shape": list(tensor.shape)},
            )
        )
        if mask_axes:
            broadcasts[selected] = {mask_value: mask_axes}
        generated.append(selected)
        for position in operand_positions:
            consumer.operands[position] = selected

        records.append(
            PredicateMaterializationRecord(
                access_id=access.access_id,
                node_ref=access.node_ref,
                tensor=access.tensor,
                authored_fill=access.fill_value is not None,
                guard_dominated_fill=guard_dominated,
                guards=guards,
                upper_bounds=tuple(residual_upper),
                erased_full_domain_terms=tuple(full_terms),
                generated_values=tuple(generated),
            )
        )
    transformed.body = [*prefix, *transformed.body]

    logical_projection = project_contract_materialization(j)
    materialized_accesses = {item.access_id for item in records}
    materialized_guards = tuple(
        sorted(
            "access_guard:"
            f"{relation.access_ref}:{relation.value_ref}:"
            f"{','.join(str(axis) for axis in relation.axes)}:"
            f"{relation.polarity.value}"
            for relation in j.boundary.access_guard_relations
            if relation.access_ref in materialized_accesses
        )
    )
    projection = BoundaryMaterializationProjection(
        logical_projection=logical_projection,
        predicate_materializations=tuple(records),
        synthetic_broadcast_axes=_serialize_broadcasts(broadcasts),
        materialized_access_guards=materialized_guards,
    )
    return DensePredicateFunctionMaterialization(transformed, projection)


__all__ = [
    "BOUNDARY_MATERIALIZATION_PROTOCOL",
    "DENSE_PREDICATE_MATERIALIZATION_PROTOCOL",
    "BoundaryMaterializationProjection",
    "DensePredicateProjection",
    "DensePredicateFunctionMaterialization",
    "DensePredicateMaterializationError",
    "PredicateMaterializationRecord",
    "materialize_dense_load_predicates",
]
