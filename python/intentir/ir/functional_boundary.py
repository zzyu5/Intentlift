"""Backend-neutral projection of J's complete functional memory boundary.

This is a deterministic compiler analysis over the one accepted J.  It is not
serialized as an artifact and has no identity outside the analysis invocation.
Both RefEval and production lowering use it to prove that explicit Layer-B
domain/access/effect relations can be represented by the complete logical
tensors of ``IntentFunction`` without silently dropping semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import re
from typing import Any, Mapping, Sequence

from intentir.ir.canonical_j import (
    CanonicalValueType,
    ContractExpr,
    ContractSort,
    direct_dimension_lower_bounds,
    direct_unit_dimension_refs,
    elaborate_j,
    j_elaboration,
    parse_contract_expr,
    parse_index_expression,
)
from intentir.ir.contract_semantics import (
    ContractSemanticsError,
    evaluate_contract_expr,
)
from intentir.ir.model import (
    IntentOp,
    intent_op_regions,
    walk_intent_op_paths,
    walk_intent_ops,
)
from intentir.ir.dimension_relation import (
    resolve_dimension_relations,
    validate_dimension_relation_instantiation,
    validate_dimension_relations,
)
from intentir.ir.dim_expr import dimension_expression_references
from intentir.ir.intent_op_contract import (
    is_closed_floating_dtype,
    is_closed_integer_dtype,
)
from intentir.ir.value_dimension import (
    validate_value_dimension_instantiation,
    validate_value_dimension_relations,
)


FUNCTIONAL_BOUNDARY_PROTOCOL = "intentir.j-functional-boundary.v22"
RUNTIME_PREDICATE_PROOF_PROTOCOL = (
    "intentir.j-functional-runtime-predicate-tautology.v1"
)


@dataclass(frozen=True)
class PredicateProjection:
    guards: frozenset[str]
    lower_axes: frozenset[str]
    upper_bounds: tuple[tuple[str, int | str], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "guards": sorted(self.guards),
            "lower_axes": sorted(self.lower_axes),
            "upper_bounds": [list(item) for item in self.upper_bounds],
        }


@dataclass(frozen=True)
class RuntimePredicateProofReceipt:
    """Source-free receipt for one invocation-local tautology proof.

    The receipt deliberately contains no runtime value or value digest.  The
    enclosing Gate2 case already binds the execution to its hidden input
    digest; this record says only which J-owned predicate was proved over
    which J-owned closed axes for that invocation.
    """

    access_id: str
    predicate_digest: str
    logical_input_refs: tuple[str, ...]
    domain_axes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": RUNTIME_PREDICATE_PROOF_PROTOCOL,
            "proof": "closed_domain_tautology",
            "access_id": self.access_id,
            "predicate_digest": self.predicate_digest,
            "logical_input_refs": list(self.logical_input_refs),
            "domain_axes": list(self.domain_axes),
        }


@dataclass(frozen=True)
class ConsumerOutsideProjection:
    """One J-authored outside-domain relation consumed by RefEval/backend."""

    consumer_ref: str
    operand_index: int
    access_ref: str
    tile_axis: int
    dtype: str
    value: Any
    identity_law: str

    def to_json(self) -> dict[str, Any]:
        return {
            "consumer_ref": self.consumer_ref,
            "operand_index": self.operand_index,
            "access_ref": self.access_ref,
            "tile_axis": self.tile_axis,
            "outside": {"dtype": self.dtype, "value": self.value},
            "identity_law": self.identity_law,
        }


@dataclass(frozen=True)
class ConstantGatherProjection:
    """One J-only proof that a constant boundary load is one typed gather."""

    access_id: str
    node_ref: str
    tensor: str
    axis: int
    index_ref: str
    coordinate: int

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "constant_gather",
            "access_id": self.access_id,
            "node_ref": self.node_ref,
            "tensor": self.tensor,
            "axis": self.axis,
            "index_ref": self.index_ref,
            "coordinate": self.coordinate,
        }


@dataclass(frozen=True)
class ConditionalEffectStoreProjection:
    """One full logical output write selected by an authored ``effect_if``.

    Layer A owns the branch and the producer value; Layer B owns the matching
    access/effect destination.  This ephemeral receipt merely joins those two
    already-authored facts after the ordinary full-store checks have succeeded.
    It neither creates a select nor supplies a value for an untaken arm.
    """

    access_id: str
    node_ref: str
    output: str
    value_ref: str

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "authored_effect_if_full_store",
            "access_id": self.access_id,
            "node_ref": self.node_ref,
            "output": self.output,
            "value_ref": self.value_ref,
        }


@dataclass(frozen=True)
class IndexBoundedGatherProjection:
    """One dynamic J gather whose data access is closed by ``index_bounds``."""

    access_id: str
    node_ref: str
    tensor: str
    axis: int
    index_ref: str
    consumer_axes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "index_bounded_gather",
            "access_id": self.access_id,
            "node_ref": self.node_ref,
            "tensor": self.tensor,
            "axis": self.axis,
            "index_ref": self.index_ref,
            "consumer_axes": list(self.consumer_axes),
        }


@dataclass(frozen=True)
class ConcatStoreSliceProjection:
    """One unit-width physical store derived from concat operand order."""

    access_id: str
    operand_ref: str
    start: int
    width: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "operand_ref": self.operand_ref,
            "start": self.start,
            "width": self.width,
        }


@dataclass(frozen=True)
class ConcatStorePartitionProjection:
    """One exhaustive, disjoint unit-width concat output partition."""

    output: str
    producer_ref: str
    axis: int
    slices: tuple[ConcatStoreSliceProjection, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "unit_width_concat_store_partition",
            "output": self.output,
            "producer_ref": self.producer_ref,
            "axis": self.axis,
            "slices": [item.to_json() for item in self.slices],
        }


@dataclass(frozen=True)
class DirectStoreSliceProjection:
    """One unit-width point store into a complete producer result."""

    access_id: str
    start: int
    width: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "start": self.start,
            "width": self.width,
        }


@dataclass(frozen=True)
class DirectStorePartitionProjection:
    """One exhaustive point-store partition of the same complete J value."""

    output: str
    value_ref: str
    producer_ref: str
    axis: int
    slices: tuple[DirectStoreSliceProjection, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "unit_width_direct_store_partition",
            "output": self.output,
            "value_ref": self.value_ref,
            "producer_ref": self.producer_ref,
            "axis": self.axis,
            "slices": [item.to_json() for item in self.slices],
        }


@dataclass(frozen=True)
class AccessGuardProjection:
    """One consumed J-authored access-local guard relation."""

    access_ref: str
    value_ref: str
    node_ref: str
    consumer_result: str
    axes: tuple[int, ...]
    polarity: str
    access_kind: str = "store"
    tensor: str | None = None
    operand_index: int | None = None
    fill_value: Any = None

    def to_json(self) -> dict[str, Any]:
        result = {
            "proof": "authored_access_guard_relation",
            "access_ref": self.access_ref,
            "value_ref": self.value_ref,
            "node_ref": self.node_ref,
            "consumer_result": self.consumer_result,
            "axes": list(self.axes),
            "polarity": self.polarity,
        }
        if self.access_kind == "load":
            result.update(
                {
                    "access_kind": "load",
                    "tensor": self.tensor,
                    "operand_index": self.operand_index,
                    "fill_value": self.fill_value,
                    "materialization": "access_local_pre_execution_select",
                }
            )
        return result


@dataclass(frozen=True)
class MixedRadixDigitProjection:
    """One tensor coordinate decoded from a complete flattened domain axis."""

    tensor_axis: int
    extent: int | str
    stride_factors: tuple[int | str, ...]
    expression: str

    def to_json(self) -> dict[str, Any]:
        return {
            "tensor_axis": self.tensor_axis,
            "extent": self.extent,
            "stride_factors": list(self.stride_factors),
            "expression": self.expression,
        }


@dataclass(frozen=True)
class MixedRadixAccessProjection:
    """J-only proof that one flat physical axis bijects logical dimensions."""

    access_id: str
    flat_axis: str
    domain_factors: tuple[int | str, ...]
    digits: tuple[MixedRadixDigitProjection, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "complete_mixed_radix_bijection",
            "access_id": self.access_id,
            "flat_axis": self.flat_axis,
            "domain_factors": list(self.domain_factors),
            "digits": [item.to_json() for item in self.digits],
        }


@dataclass(frozen=True)
class FunctionalBoundaryProjection:
    """One ephemeral proof that Layer B projects to logical tensor execution."""

    profile: str
    domains: tuple[tuple[str, int | str], ...]
    access_axes: tuple[tuple[str, tuple[str, ...]], ...]
    access_coordinates: tuple[
        tuple[str, tuple[str | int, ...]], ...
    ]
    predicates: tuple[tuple[str, PredicateProjection | None], ...]
    broadcast_axes: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ]
    proven_index_bounds: frozenset[str]
    consumed: tuple[str, ...]
    boundary_digest: str
    predicated_access_count: int
    erased_fill_count: int
    runtime_predicate_proofs: tuple[RuntimePredicateProofReceipt, ...] = ()
    consumer_extensions: tuple[ConsumerOutsideProjection, ...] = ()
    value_dimension_relations: tuple[tuple[str, str], ...] = ()
    dimension_relations: tuple[
        tuple[str, str, str, str | None, int | None], ...
    ] = ()
    constant_gathers: tuple[ConstantGatherProjection, ...] = ()
    concat_store_partitions: tuple[ConcatStorePartitionProjection, ...] = ()
    direct_store_partitions: tuple[DirectStorePartitionProjection, ...] = ()
    access_guard_relations: tuple[AccessGuardProjection, ...] = ()
    mixed_radix_accesses: tuple[MixedRadixAccessProjection, ...] = ()
    context_domains: tuple[tuple[str, int | str], ...] = ()
    conditional_effect_stores: tuple[ConditionalEffectStoreProjection, ...] = ()
    index_bounded_gathers: tuple[IndexBoundedGatherProjection, ...] = ()

    @property
    def access_axes_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.access_axes)

    @property
    def access_coordinates_map(self) -> dict[str, tuple[str | int, ...]]:
        return dict(self.access_coordinates)

    @property
    def predicate_map(self) -> dict[str, PredicateProjection | None]:
        return dict(self.predicates)

    def emitter_broadcast_axes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            result: dict(operands) for result, operands in self.broadcast_axes
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": FUNCTIONAL_BOUNDARY_PROTOCOL,
            "profile": self.profile,
            "domains": [list(item) for item in self.domains],
            "access_axes": [
                [access_id, list(axes)] for access_id, axes in self.access_axes
            ],
            "access_coordinates": [
                [access_id, list(coordinates)]
                for access_id, coordinates in self.access_coordinates
            ],
            "predicates": [
                [access_id, predicate.to_json() if predicate is not None else None]
                for access_id, predicate in self.predicates
            ],
            "broadcast_axes": [
                [
                    result,
                    [[operand, list(axes)] for operand, axes in operands],
                ]
                for result, operands in self.broadcast_axes
            ],
            "proven_index_bounds": sorted(self.proven_index_bounds),
            "consumed": list(self.consumed),
            "boundary_digest": self.boundary_digest,
            "predicated_access_count": self.predicated_access_count,
            "erased_fill_count": self.erased_fill_count,
            "runtime_predicate_proofs": [
                receipt.to_json() for receipt in self.runtime_predicate_proofs
            ],
            "consumer_extensions": [
                relation.to_json() for relation in self.consumer_extensions
            ],
            "value_dimension_relations": [
                {
                    "value_ref": value_ref,
                    "dimension_ref": dimension_ref,
                    "relation": "eq",
                }
                for value_ref, dimension_ref in self.value_dimension_relations
            ],
            "dimension_relations": [
                {
                    "result": result,
                    "expr": {
                        "op": op,
                        "arg": arg,
                        **({"rhs": rhs} if rhs is not None else {}),
                        **({"value": value} if value is not None else {}),
                    },
                }
                for result, op, arg, rhs, value in self.dimension_relations
            ],
            "constant_gathers": [
                proof.to_json() for proof in self.constant_gathers
            ],
            "concat_store_partitions": [
                proof.to_json() for proof in self.concat_store_partitions
            ],
            "direct_store_partitions": [
                proof.to_json() for proof in self.direct_store_partitions
            ],
            "access_guard_relations": [
                proof.to_json() for proof in self.access_guard_relations
            ],
            "mixed_radix_accesses": [
                proof.to_json() for proof in self.mixed_radix_accesses
            ],
            "context_domains": [
                {"axis": axis, "extent": extent}
                for axis, extent in self.context_domains
            ],
            "conditional_effect_stores": [
                proof.to_json() for proof in self.conditional_effect_stores
            ],
            "index_bounded_gathers": [
                proof.to_json() for proof in self.index_bounded_gathers
            ],
        }


@dataclass(frozen=True)
class FunctionalBoundaryClosureIssue:
    """One J-only inconsistency in the projectable logical-domain profile.

    This is deliberately narrower than ``FunctionalBoundaryProjection``.  It
    records only contradictions that can be decided from J's own typed tensor
    shapes, direct access vectors, predicates, and explicit domains.  An empty
    issue list does not claim that every valid J feature is implemented by the
    current RefEval/backend projection.
    """

    path: str
    detail: str
    ref: str | None = None


@dataclass(frozen=True, order=True)
class ConsumerExtensionRequirement:
    """One proven physical-padding edge that needs J-owned outside semantics."""

    consumer_ref: str
    operand_index: int
    access_ref: str


def parse_direct_axes(expression: str) -> tuple[str, ...]:
    """Return the unique closed direct-axis interpretation of a wire form."""

    axes = parse_index_expression(
        expression,
        context="J boundary direct index",
    ).direct_axes
    if axes is None:
        raise NotImplementedError(
            "functional dense access requires a direct ordered axis vector"
        )
    if len(set(axes)) != len(axes):
        raise ValueError("functional dense access axes must be unique")
    return axes


def project_predicate_text(
    expression: str,
    *,
    logical_guards: frozenset[str] = frozenset(),
) -> PredicateProjection:
    """Parse one predicate through the same typed projection used by consumers."""

    return _predicate_projection(
        parse_contract_expr(
            expression,
            expected=ContractSort.BOOLEAN,
            context="J boundary access predicate",
        ),
        logical_guards=logical_guards,
    )


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _result_names(op: IntentOp) -> tuple[str, ...]:
    if isinstance(op.result, str):
        return (op.result,)
    if isinstance(op.result, list):
        return tuple(item for item in op.result if isinstance(item, str))
    return ()


def _child_regions(op: IntentOp) -> tuple[tuple[str, Sequence[IntentOp]], ...]:
    """Return every authored child region with its canonical path suffix."""
    return intent_op_regions(op)


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    return tuple(walk_intent_ops(body))


def _effect_if_arm_node_refs(j: Any) -> frozenset[str]:
    """Return nodes lexically controlled by at least one authored effect arm."""

    return frozenset(_effect_if_arm_memberships(j))


def _effect_if_arm_memberships(
    j: Any,
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Index node refs by enclosing authored effect-if and arm polarity."""

    result: dict[str, tuple[tuple[str, str], ...]] = {}
    refs_by_path: dict[str, str] = {}
    for node_ref, path in j.node_ids.items():
        previous = refs_by_path.setdefault(path, node_ref)
        if previous != node_ref:
            raise ValueError("effect-if membership has duplicate authored node paths")

    def visit(
        sequence: Sequence[IntentOp],
        prefix: str,
        memberships: tuple[tuple[str, str], ...],
    ) -> None:
        for index, operation in enumerate(sequence):
            path = f"{prefix}[{index}]"
            authored_ref = refs_by_path.get(path)
            if (
                operation.node_id is not None
                and authored_ref is not None
                and operation.node_id != authored_ref
            ):
                raise ValueError("inline and authored effect-if node refs disagree")
            node_ref = operation.node_id or authored_ref
            if memberships and node_ref is not None:
                previous = result.setdefault(node_ref, memberships)
                if previous != memberships:
                    raise ValueError("effect-if node has ambiguous lexical memberships")
            visit(operation.body, f"{path}.body", memberships)
            if operation.op in {"effect_if", "value_if"}:
                assert operation.positive is not None
                assert operation.negative is not None
                control_ref = node_ref
                positive_memberships = memberships
                negative_memberships = memberships
                if operation.op == "effect_if" and control_ref is not None:
                    positive_memberships += ((control_ref, "positive"),)
                    negative_memberships += ((control_ref, "negative"),)
                visit(
                    operation.positive.body,
                    f"{path}.positive.body",
                    positive_memberships,
                )
                visit(
                    operation.negative.body,
                    f"{path}.negative.body",
                    negative_memberships,
                )

    visit(j.intent.body, "body", ())
    return result


def _is_direct_unit_extent(
    extent: int | str,
    direct_unit_dimensions: frozenset[str],
) -> bool:
    return extent == 1 or (
        isinstance(extent, str) and extent in direct_unit_dimensions
    )


def _store_result_reaches_output_through_enclosing_carry(
    j: Any,
    *,
    operation: IntentOp,
    producer_value: str,
    observable_value: str,
) -> bool:
    """Recognize one explicit body-result -> enclosing-loop-result edge.

    Boundary stores are anchored at their actual Layer-A access event.  For a
    store inside a structured loop, that event can produce ``carry.next_value``
    while the observable function output is ``carry.result_value``.  This is
    not an inferred merge: both ends and their enclosing loop are authored in
    J.  Refuse missing, cross-scope, or ambiguous edges.
    """

    matches: list[tuple[IntentOp, Any]] = []

    def visit(body: Sequence[IntentOp], enclosing: tuple[IntentOp, ...]) -> None:
        for candidate in body:
            if candidate is operation:
                for loop in enclosing:
                    for state in loop.carry:
                        if (
                            state.next_value == producer_value
                            and state.result_value == observable_value
                        ):
                            matches.append((loop, state))
            nested = (
                (*enclosing, candidate)
                if candidate.op == "loop"
                else enclosing
            )
            for _suffix, region_body in _child_regions(candidate):
                visit(region_body, nested)

    visit(j.intent.body, ())
    if len(matches) != 1:
        return False
    value_types = j_elaboration(j).value_type_by_name
    return (
        value_types.get(producer_value) is not None
        and value_types.get(producer_value) == value_types.get(observable_value)
    )


def _node_table(j: Any) -> dict[str, IntentOp]:
    by_path = dict(walk_intent_op_paths(j.intent.body))
    result = dict(by_path)
    for node_id, path in j.node_ids.items():
        if path in by_path:
            result[node_id] = by_path[path]
    return result


def _explicit_inout_poststate_reload(
    j: Any,
    access_id: str,
) -> tuple[str, str] | None:
    """Return the unique authored store/value consumed by one physical reload.

    A logical inout may carry several compiler-visible store/reload stages even
    though Layer A directly connects each already-authored producer SSA value to
    its consumer.  This helper recognizes only one closed pair at a time: an
    inout handle, one unique matching producer store, identical coordinates and
    predicate, store-before-load effects in one scope, and no intervening write
    to the same handle.  A node-bound reload must retain direct producer/consumer
    dataflow.  A node-less reload is accepted only as an immediately adjacent
    identity roundtrip whose stored full-tensor result is already authored by J;
    it introduces no new semantic value.  The helper never invents a state edge
    or consults Harness, C, pi, Source, or Replay.
    """

    j = elaborate_j(j)
    accesses = {item.access_id: item for item in j.boundary.accesses}
    access = accesses.get(access_id)
    if (
        access is None
        or access.kind.value != "load"
    ):
        return None
    tensor = access.tensor
    if (
        tensor not in {item.name for item in j.intent.inputs}
        or tensor not in {item.name for item in j.intent.outputs}
    ):
        return None
    bindings = [
        item
        for item in j.signature.parameters
        if item.mode.value == "inout"
        and item.logical_ref == tensor
        and item.alias_of is None
    ]
    if len(bindings) != 1:
        return None

    returns = [item for item in j.intent.returns if item.output == tensor]
    mappings = [
        item for item in j.boundary.output_mappings if item.output == tensor
    ]
    if len(returns) != 1 or len(mappings) > 1:
        return None
    mapping = mappings[0] if mappings else None
    final_poststate = returns[0].value
    if (
        returns
        and mapping is not None
        and mapping.value != final_poststate
    ):
        return None

    nodes = _node_table(j)
    consumer = (
        nodes.get(access.node_ref) if access.node_ref is not None else None
    )
    if access.node_ref is not None and consumer is None:
        return None
    dependencies = (
        _operation_dependencies(consumer) if consumer is not None else ()
    )

    effects_by_access = {
        item.access_ref: item
        for item in j.boundary.effects
        if item.access_ref is not None
    }
    if len(effects_by_access) != len(
        [item for item in j.boundary.effects if item.access_ref is not None]
    ):
        return None
    load_effect = effects_by_access.get(access.access_id)
    if (
        load_effect is None
        or load_effect.kind.value != "read"
        or load_effect.target != tensor
        or load_effect.node_ref != access.node_ref
    ):
        return None
    effect_order = [item.effect_id for item in j.boundary.effects]
    load_index = effect_order.index(load_effect.effect_id)
    candidates: list[tuple[str, str]] = []
    stores = [
        item
        for item in j.boundary.accesses
        if item.kind.value == "store"
        and item.tensor == tensor
        and item.node_ref is not None
        and item.index_expression == access.index_expression
        and item.predicate == access.predicate
    ]
    for store in stores:
        producer = nodes.get(store.node_ref)
        store_effect = effects_by_access.get(store.access_id)
        if (
            producer is None
            or store_effect is None
            or store_effect.kind.value != "write"
            or store_effect.target != tensor
            or store_effect.node_ref != store.node_ref
        ):
            continue
        produced_dependencies = (
            [
                dependency
                for dependency in dependencies
                if dependency in _result_names(producer)
            ]
            if consumer is not None
            else list(_result_names(producer))
        )
        if len(produced_dependencies) != 1:
            continue
        store_index = effect_order.index(store_effect.effect_id)
        if store_index >= load_index:
            continue
        if any(
            other.access_id != store.access_id
            and other.kind.value == "store"
            and other.tensor == tensor
            and (other_effect := effects_by_access.get(other.access_id)) is not None
            and store_index < effect_order.index(other_effect.effect_id) < load_index
            for other in j.boundary.accesses
        ):
            continue
        owning_scopes = [
            scope
            for scope in j.scopes
            if store_effect.effect_id in scope.effects
            or load_effect.effect_id in scope.effects
        ]
        if owning_scopes:
            if len(owning_scopes) != 1:
                continue
            scope_effects = owning_scopes[0].effects
            if (
                store_effect.effect_id not in scope_effects
                or load_effect.effect_id not in scope_effects
                or scope_effects.index(store_effect.effect_id)
                >= scope_effects.index(load_effect.effect_id)
            ):
                continue
            if consumer is None and (
                scope_effects.index(load_effect.effect_id)
                != scope_effects.index(store_effect.effect_id) + 1
            ):
                continue
        candidates.append((store.access_id, produced_dependencies[0]))
    if len(candidates) != 1:
        return None
    return candidates[0]


def explicit_inout_poststate_reload_value(
    j: Any,
    access_id: str,
) -> str | None:
    """Return the authored SSA value consumed by one physical inout reload."""

    pair = _explicit_inout_poststate_reload(j, access_id)
    return None if pair is None else pair[1]


_PREDICATE_DOMINATED_POINTWISE_OPS = frozenset(
    {"map_unary", "map_binary", "select", "cast", "value_if"}
)
_PREDICATE_DOMINATED_SPECIAL_OPS = frozenset(
    {"contraction", "reduce", "pool", "scan"}
)


def _predicate_dominated_special_provenance(
    operation: IntentOp,
    reachable_operands: Sequence[str],
    results: Sequence[str],
    provenance: Mapping[str, tuple[int | None, ...]],
    value_types: Mapping[str, CanonicalValueType],
) -> dict[str, tuple[int | None, ...]] | None:
    """Project only special operations whose value-domain map is identity-like.

    A missing predicate-load fill is observable through a reduction, window,
    or indexed operation unless the operation has a closed identity proof.  Do
    not approximate that proof from the result rank alone: each case below
    checks the authored operation attributes and typed extents directly.
    """

    if len(results) != 1:
        # Multi-result pool/sort-like operations include metadata whose values
        # may depend on the missing load even when the primary tensor looks
        # pointwise.  They remain outside this proof.
        return None
    result = results[0]
    result_type = value_types.get(result)
    if result_type is None or not result_type.is_tensor:
        return None

    if operation.op == "scan":
        if (
            len(operation.operands) != 1
            or len(reachable_operands) != 1
            or reachable_operands[0] not in provenance
        ):
            return None
        source = reachable_operands[0]
        source_type = value_types.get(source)
        axis = operation.attrs.get("axis")
        if (
            source_type is None
            or not source_type.is_tensor
            or type(axis) is not int
        ):
            return None
        axis = axis if axis >= 0 else axis + len(source_type.shape)
        if not 0 <= axis < len(source_type.shape):
            return None
        if type(source_type.shape[axis]) is not int or source_type.shape[axis] != 1:
            return None
        if result_type != source_type:
            return None
        return {result: provenance[source]}

    if operation.op == "reduce":
        if (
            len(operation.operands) != 1
            or len(reachable_operands) != 1
            or reachable_operands[0] not in provenance
        ):
            return None
        source = reachable_operands[0]
        source_type = value_types.get(source)
        axes = operation.attrs.get("axes")
        if (
            source_type is None
            or not source_type.is_tensor
            or not isinstance(axes, list)
            or any(type(axis) is not int for axis in axes)
        ):
            return None
        normalized = [
            axis if axis >= 0 else axis + len(source_type.shape)
            for axis in axes
        ]
        if (
            len(set(normalized)) != len(normalized)
            or any(axis < 0 or axis >= len(source_type.shape) for axis in normalized)
            or any(
                type(source_type.shape[axis]) is not int
                or source_type.shape[axis] != 1
                for axis in normalized
            )
        ):
            return None
        keepdims_shape = tuple(
            1 if axis in set(normalized) else extent
            for axis, extent in enumerate(source_type.shape)
        )
        collapsed_shape = tuple(
            extent
            for axis, extent in enumerate(source_type.shape)
            if axis not in set(normalized)
        )
        keepdims = operation.attrs.get("keepdims")
        if keepdims is None:
            if result_type.shape == keepdims_shape:
                keepdims = True
            elif result_type.shape == collapsed_shape:
                keepdims = False
            else:
                return None
        elif type(keepdims) is not bool:
            return None
        expected_shape = keepdims_shape if keepdims else collapsed_shape
        if result_type.shape != expected_shape:
            return None
        source_provenance = provenance[source]
        if len(source_provenance) != len(source_type.shape):
            return None
        projected = (
            tuple(source_provenance)
            if keepdims
            else tuple(
                source_provenance[axis]
                for axis in range(len(source_type.shape))
                if axis not in set(normalized)
            )
        )
        return {result: projected}

    if operation.op == "pool":
        if (
            len(operation.operands) != 1
            or len(reachable_operands) != 1
            or reachable_operands[0] not in provenance
            or any(
                type(operation.attrs.get(name)) is not int
                for name in ("kH", "kW", "sH", "sW", "padH", "padW")
            )
            or operation.attrs.get("kH") != 1
            or operation.attrs.get("kW") != 1
            or operation.attrs.get("sH") != 1
            or operation.attrs.get("sW") != 1
            or operation.attrs.get("padH") != 0
            or operation.attrs.get("padW") != 0
        ):
            return None
        source = reachable_operands[0]
        source_type = value_types.get(source)
        if (
            source_type is None
            or not source_type.is_tensor
            or len(source_type.shape) != 4
            or result_type != source_type
        ):
            return None
        return {result: provenance[source]}

    if operation.op == "contraction":
        if len(operation.operands) != 2:
            return None
        subscripts = operation.attrs.get("subscripts")
        if not isinstance(subscripts, str):
            return None
        compact = "".join(subscripts.split())
        pieces = compact.split("->")
        if len(pieces) != 2:
            return None
        input_terms = pieces[0].split(",")
        output_labels = pieces[1]
        if (
            len(input_terms) != 2
            or not output_labels
            or len(set(output_labels)) != len(output_labels)
            or any(len(set(term)) != len(term) for term in input_terms)
            or set(output_labels) != set("".join(input_terms))
        ):
            # Every input label must survive in the output.  This excludes
            # reduction, diagonal, and implicit-index semantics.
            return None
        operand_types = [value_types.get(operand) for operand in operation.operands]
        if any(item is None or not item.is_tensor for item in operand_types):
            return None
        label_extents: dict[str, Any] = {}
        for term, operand_type in zip(input_terms, operand_types, strict=True):
            assert operand_type is not None
            if len(term) != len(operand_type.shape):
                return None
            for label, extent in zip(term, operand_type.shape, strict=True):
                previous = label_extents.setdefault(label, extent)
                if previous != extent:
                    return None
        expected_shape = tuple(label_extents[label] for label in output_labels)
        if result_type.shape != expected_shape:
            return None
        expanded: list[tuple[int | None, ...]] = []
        for operand, term in zip(operation.operands, input_terms, strict=True):
            if operand not in provenance:
                continue
            operand_provenance = provenance[operand]
            if len(operand_provenance) != len(term):
                return None
            projected: list[int | None] = [None] * len(output_labels)
            for source_axis, label in enumerate(term):
                projected[output_labels.index(label)] = operand_provenance[source_axis]
            expanded.append(tuple(projected))
        if not expanded or len(set(expanded)) != 1:
            return None
        return {result: expanded[0]}

    return None


def _predicate_dominated_operation_is_supported(operation: IntentOp) -> bool:
    return operation.op in (
        _PREDICATE_DOMINATED_POINTWISE_OPS
        | _PREDICATE_DOMINATED_SPECIAL_OPS
        | {"transpose"}
    )


def _predicate_dominated_load_start(
    j: Any,
    *,
    access: Any,
    nodes: Mapping[str, IntentOp],
) -> tuple[str, IntentOp, str] | None:
    """Resolve the unique first J consumer of one predicate-bearing load.

    A node-bound access already names that consumer.  A node-less access is the
    canonical whole-tensor form and is usable only when it is the sole load of
    its logical input and that input has exactly one direct canonical consumer.
    This is a J-only reference resolution: it neither invents a consumer edge
    nor consults Source, C, pi, Replay, or Harness.
    """

    if access.node_ref is not None:
        start = nodes.get(access.node_ref)
        if start is None:
            return None
        dependencies = _operation_dependencies(start)
        if access.tensor in dependencies:
            return access.node_ref, start, access.tensor
        poststate = explicit_inout_poststate_reload_value(j, access.access_id)
        if poststate is not None and poststate in dependencies:
            return access.node_ref, start, poststate
        return None

    matching_loads = [
        item
        for item in j.boundary.accesses
        if item.kind.value == "load" and item.tensor == access.tensor
    ]
    if (
        len(matching_loads) != 1
        or matching_loads[0].access_id != access.access_id
    ):
        return None
    consumers = [
        (node_ref, nodes[node_ref])
        for node_ref in j.node_ids
        if node_ref in nodes
        and access.tensor in _operation_dependencies(nodes[node_ref])
    ]
    if len(consumers) != 1:
        return None
    node_ref, start = consumers[0]
    return node_ref, start, access.tensor


def _masked_load_axis_provenance(
    j: Any,
    *,
    access: Any,
    nodes: Mapping[str, IntentOp],
) -> tuple[set[str], dict[str, tuple[int | None, ...]]] | None:
    """Track masked axes through pointwise broadcast and exact transposes.

    Each result-axis entry names the originating masked-load tensor axis.
    ``None`` marks an axis introduced by a closed pointwise broadcast; its
    coordinate remains governed by the ordinary functional boundary proof.
    """

    value_types = j_elaboration(j).value_type_by_name
    tensor_type = value_types.get(access.tensor)
    resolved_start = _predicate_dominated_load_start(
        j,
        access=access,
        nodes=nodes,
    )
    if tensor_type is None or resolved_start is None:
        return None
    start_ref, start, source_ref = resolved_start
    try:
        broadcast_axes = _broadcast_axes(j)
    except (NotImplementedError, ValueError):
        return None
    source_type = value_types.get(source_ref)
    if source_type != tensor_type:
        return None
    start_results = _result_names(start)
    if not start_results:
        return None
    provenance: dict[str, tuple[int | None, ...]] = {}
    for result in start_results:
        result_type = value_types.get(result)
        if result_type is None:
            return None
        declared_axes = broadcast_axes.get(result, {}).get(source_ref)
        positions = (
            _axis_broadcast_positions(
                source_type.shape,
                result_type.shape,
                declared_axes,
            )
            if declared_axes is not None
            else _numpy_broadcast_positions(
                source_type.shape,
                result_type.shape,
            )
        )
        if positions is None:
            return None
        projected: list[int | None] = [None] * len(result_type.shape)
        for source_axis, target_axis in enumerate(positions):
            projected[target_axis] = source_axis
        provenance[result] = tuple(projected)

    canonical_refs = tuple(j.node_ids)
    if any(node_ref not in nodes for node_ref in canonical_refs):
        return None
    reachable_refs = {start_ref}
    changed = True
    while changed:
        changed = False
        for node_ref in canonical_refs:
            if node_ref in reachable_refs:
                continue
            operation = nodes[node_ref]
            reachable_operands = [
                operand
                for operand in _operation_dependencies(operation)
                if operand in provenance
            ]
            if not reachable_operands:
                continue
            results = _result_names(operation)
            if not results:
                return None

            if operation.op in _PREDICATE_DOMINATED_POINTWISE_OPS:
                for result in results:
                    result_type = value_types.get(result)
                    if result_type is None:
                        return None
                    expanded: list[tuple[int | None, ...]] = []
                    for operand in reachable_operands:
                        operand_type = value_types.get(operand)
                        if operand_type is None:
                            return None
                        declared_axes = broadcast_axes.get(result, {}).get(
                            operand
                        )
                        positions = (
                            _axis_broadcast_positions(
                                operand_type.shape,
                                result_type.shape,
                                declared_axes,
                            )
                            if declared_axes is not None
                            else _numpy_broadcast_positions(
                                operand_type.shape,
                                result_type.shape,
                            )
                        )
                        source_provenance = provenance[operand]
                        if (
                            positions is None
                            or len(source_provenance) != len(operand_type.shape)
                        ):
                            return None
                        projected: list[int | None] = [None] * len(
                            result_type.shape
                        )
                        for source_axis, target_axis in enumerate(positions):
                            projected[target_axis] = source_provenance[source_axis]
                        expanded.append(tuple(projected))
                    if len(set(expanded)) != 1:
                        return None
                    provenance[result] = expanded[0]
            elif operation.op == "transpose":
                if len(operation.operands) != 1 or len(reachable_operands) != 1:
                    return None
                operand = reachable_operands[0]
                operand_type = value_types.get(operand)
                permutation = operation.attrs.get("perm")
                if (
                    operand_type is None
                    or not isinstance(permutation, (list, tuple))
                    or any(type(item) is not int for item in permutation)
                    or sorted(permutation) != list(range(len(operand_type.shape)))
                    or len(results) != 1
                ):
                    return None
                result_type = value_types.get(results[0])
                expected_shape = tuple(
                    operand_type.shape[item] for item in permutation
                )
                if result_type is None or result_type.shape != expected_shape:
                    return None
                input_permutation = provenance[operand]
                provenance[results[0]] = tuple(
                    input_permutation[item] for item in permutation
                )
            elif operation.op in _PREDICATE_DOMINATED_SPECIAL_OPS:
                special = _predicate_dominated_special_provenance(
                    operation,
                    reachable_operands,
                    results,
                    provenance,
                    value_types,
                )
                if special is None:
                    return None
                provenance.update(special)
            else:
                return None
            reachable_refs.add(node_ref)
            changed = True
    return reachable_refs, provenance


def predicate_dominated_pointwise_load(
    j: Any,
    access_id: str,
) -> bool:
    """Prove that one missing masked-load fill is unobservable in J.

    A physical load may omit ``other`` when every value depending on its
    masked lanes stays inside a pointwise chain and every observable output of
    that chain is stored under the exact same predicate.  An exact tensor-axis
    transpose is also permitted when its store coordinates are the authored
    permutation of the load coordinates.  The logical J domain contains only
    predicate-true elements, so the physical outside-domain value cannot
    affect an observable result.  In addition to the pointwise chain, the
    proof admits only the structurally closed special cases implemented by
    ``_predicate_dominated_special_provenance``: a contraction with no
    reduction/repeated labels, a reduction over statically unit axes, a 1x1
    stride-one unpadded pool, and a scan over a statically unit axis.  Loops,
    padding, ordinary gather/scatter/sort/arg-reduce/conv, unmatched effects,
    and differently predicated stores remain unsupported.

    The proof is deliberately J-only.  It neither derives an artifact field
    nor consults Harness, C, pi, Replay, or Source.
    """

    j = elaborate_j(j)
    accesses = {item.access_id: item for item in j.boundary.accesses}
    if len(accesses) != len(j.boundary.accesses):
        return False
    access = accesses.get(access_id)
    if (
        access is None
        or access.kind.value != "load"
        or access.predicate is None
        or access.fill_value is not None
    ):
        return False

    nodes = _node_table(j)
    resolved_start = _predicate_dominated_load_start(
        j,
        access=access,
        nodes=nodes,
    )
    if (
        resolved_start is None
        or not _predicate_dominated_operation_is_supported(resolved_start[1])
    ):
        return False
    provenance_result = _masked_load_axis_provenance(
        j,
        access=access,
        nodes=nodes,
    )
    if provenance_result is None:
        return False
    reachable_refs, provenance = provenance_result
    reachable_values = set(provenance)
    parsed_accesses = j_elaboration(j).access_by_id
    parsed_load = parsed_accesses.get(access.access_id)
    if parsed_load is None:
        return False

    # Canonical elaboration owns output binding, including the identity form
    # where an operation directly produces an output-named value and the wire
    # mapping is deliberately omitted.  Mask observability must not depend on
    # whether Lift redundantly serialized that same destination fact.
    mappings = [
        item
        for item in j_elaboration(j).output_bindings
        if item.value in reachable_values
    ]
    if not mappings:
        return False

    matched_store_ids: set[str] = set()
    for mapping in mappings:
        producer_refs = [
            node_ref
            for node_ref in reachable_refs
            if mapping.value in _result_names(nodes[node_ref])
        ]
        if len(producer_refs) != 1:
            return False
        producer_ref = producer_refs[0]
        if mapping.node_ref is not None and mapping.node_ref != producer_ref:
            return False
        stores = [
            item
            for item in j.boundary.accesses
            if item.kind.value == "store" and item.tensor == mapping.output
        ]
        if (
            len(stores) != 1
            or stores[0].node_ref != producer_ref
            or stores[0].predicate != access.predicate
        ):
            return False
        store = stores[0]
        parsed_store = parsed_accesses.get(store.access_id)
        permutation = provenance.get(mapping.value)
        if parsed_store is None or permutation is None:
            return False
        if len(parsed_store.index.components) != len(permutation):
            return False
        originating_axes = [item for item in permutation if item is not None]
        if sorted(originating_axes) != list(
            range(len(parsed_load.index.components))
        ):
            return False
        if any(
            parsed_store.index.components[target_axis]
            != parsed_load.index.components[source_axis]
            for target_axis, source_axis in enumerate(permutation)
            if source_axis is not None
        ):
            return False
        matched_store_ids.add(store.access_id)

    for item in j.boundary.accesses:
        if item.node_ref not in reachable_refs or item.kind.value == "load":
            continue
        if item.kind.value != "store" or item.access_id not in matched_store_ids:
            return False
    return True


def static_index_bounds_status(j: Any, relation: Any) -> bool | None:
    """Decide a closed internal-constant index precondition from J alone.

    ``True`` and ``False`` are proofs for an integer ``const`` index against a
    statically sized data axis.  ``None`` means the relation still needs a
    runtime checker.  No Source, C, pi, Harness value, or target fact enters
    this decision.
    """

    j = elaborate_j(j)
    value_types = j_elaboration(j).value_type_by_name
    data_type = value_types.get(relation.data)
    index_type = value_types.get(relation.indices)
    if (
        data_type is None
        or index_type is None
        or relation.axis >= len(data_type.shape)
        or not is_closed_integer_dtype(
            index_type.dtype,
            include_boolean=False,
        )
    ):
        return None
    extent = data_type.shape[relation.axis]
    producer = next(
        (
            operation
            for operation in _node_table(j).values()
            if relation.indices in _result_names(operation)
        ),
        None,
    )
    if producer is None or producer.op != "const":
        return None
    value = producer.attrs.get("value")
    if type(value) is not int:
        return None
    if value < 0:
        return False
    if type(extent) is int:
        return value < extent
    if isinstance(extent, str):
        lower = direct_dimension_lower_bounds(j.dimension_constraints).get(extent)
        if lower is not None and lower > value:
            return True
    return None


def _canonical_nodes(j: Any) -> dict[str, IntentOp]:
    nodes = _node_table(j)
    return {node_ref: nodes[node_ref] for node_ref in j.node_ids if node_ref in nodes}


def _constant_int_producer(j: Any, value_ref: str) -> tuple[str, int] | None:
    matches: list[tuple[str, int]] = []
    for node_ref, operation in _canonical_nodes(j).items():
        if value_ref not in _result_names(operation) or operation.op != "const":
            continue
        value = operation.attrs.get("value")
        if type(value) is int:
            matches.append((node_ref, value))
    return matches[0] if len(matches) == 1 else None


def _project_constant_gather_load(
    j: Any,
    *,
    access: Any,
    coordinates: tuple[str | int, ...],
    tensor: Any,
    operation: IntentOp | None,
    direct_unit_dimensions: frozenset[str],
    runtime_logical_dimensions: Mapping[str, int] | None = None,
) -> ConstantGatherProjection | None:
    """Prove one non-singleton constant coordinate through exact J dataflow."""

    non_singleton_constants = [
        (axis, coordinate)
        for axis, (coordinate, extent) in enumerate(
            zip(coordinates, tensor.shape, strict=True)
        )
        if isinstance(coordinate, int)
        and not (
            coordinate == 0
            and _is_direct_unit_extent(extent, direct_unit_dimensions)
        )
    ]
    if not non_singleton_constants:
        return None
    if len(non_singleton_constants) != 1:
        raise NotImplementedError(
            "functional constant load implements one selected tensor axis"
        )
    axis, coordinate = non_singleton_constants[0]
    if access.node_ref is None or operation is None:
        raise ValueError(
            "functional constant load requires one node-bound gather"
        )
    if (
        operation.op != "gather"
        or len(operation.operands) != 2
        or operation.operands[0] != access.tensor
        or operation.attrs.get("axis") != axis
        or operation.attrs.get("mode", "along_axis") != "take"
    ):
        raise ValueError(
            "functional constant load must match a take gather on the same axis"
        )
    index_ref = operation.operands[1]
    producer = _constant_int_producer(j, index_ref)
    if producer is None or producer[1] != coordinate:
        raise ValueError(
            "functional constant load coordinate differs from its J gather index"
        )
    matches = [
        relation
        for relation in j.boundary.index_bounds
        if (
            relation.node_ref == access.node_ref
            and relation.data == access.tensor
            and relation.indices == index_ref
            and relation.axis == axis
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "functional constant load requires one exact index-bounds relation"
        )
    if static_index_bounds_status(j, matches[0]) is not True:
        declared_extent = tensor.shape[axis]
        runtime_extent = (
            runtime_logical_dimensions.get(declared_extent)
            if runtime_logical_dimensions is not None
            and isinstance(declared_extent, str)
            else None
        )
        if runtime_extent is None or coordinate < 0 or coordinate >= runtime_extent:
            raise ValueError(
                "functional constant load lacks a closed in-bounds proof"
            )

    value_types = j_elaboration(j).value_type_by_name
    data_type = value_types.get(access.tensor)
    index_type = value_types.get(index_ref)
    result_names = _result_names(operation)
    result_type = value_types.get(result_names[0]) if len(result_names) == 1 else None
    if data_type is None or index_type is None or result_type is None:
        raise ValueError("functional constant gather lacks a closed J type")
    expected_shape = (
        *data_type.shape[:axis],
        *index_type.shape,
        *data_type.shape[axis + 1 :],
    )
    if (
        index_type.shape
        or result_type.dtype != data_type.dtype
        or result_type.shape != expected_shape
    ):
        raise ValueError(
            "functional constant gather result type differs from scalar take"
        )
    return ConstantGatherProjection(
        access.access_id,
        access.node_ref,
        access.tensor,
        axis,
        index_ref,
        coordinate,
    )


def _project_concat_store_partition(
    j: Any,
    *,
    output: str,
    stores: Sequence[Any],
    coordinates_by_access: Mapping[str, tuple[str | int, ...]],
    predicates: Mapping[str, PredicateProjection | None],
    observable_value: str,
) -> ConcatStorePartitionProjection:
    """Prove multiple point stores are one complete unit-width concat output."""

    producer_refs = {access.node_ref for access in stores}
    if len(producer_refs) != 1 or None in producer_refs:
        raise NotImplementedError(
            "functional boundary requires exactly one store per logical output"
        )
    producer_ref = next(iter(producer_refs))
    assert producer_ref is not None
    operation = _canonical_nodes(j).get(producer_ref)
    if operation is None or operation.op != "concat":
        raise NotImplementedError(
            "functional boundary requires exactly one store per logical output"
        )
    if any(predicates[access.access_id] is not None for access in stores):
        raise NotImplementedError(
            "functional concat store partition requires unpredicated stores"
        )
    if observable_value not in _result_names(operation):
        raise ValueError(
            "functional concat store partition does not produce its J output"
        )

    value_types = j_elaboration(j).value_type_by_name
    result_type = value_types.get(observable_value)
    operand_types = [value_types.get(operand) for operand in operation.operands]
    axis = operation.attrs.get("axis")
    if (
        result_type is None
        or not isinstance(axis, int)
        or isinstance(axis, bool)
        or axis < 0
        or axis >= len(result_type.shape)
        or not operand_types
        or any(item is None for item in operand_types)
    ):
        raise ValueError("functional concat store partition lacks a closed type")
    typed_operands = [item for item in operand_types if item is not None]
    for operand_type in typed_operands:
        if (
            operand_type.dtype != result_type.dtype
            or len(operand_type.shape) != len(result_type.shape)
            or any(
                operand_type.shape[dimension] != result_type.shape[dimension]
                for dimension in range(len(result_type.shape))
                if dimension != axis
            )
        ):
            raise ValueError(
                "functional concat store partition disagrees with concat types"
            )
    widths = [item.shape[axis] for item in typed_operands]
    if any(type(width) is not int or width != 1 for width in widths):
        raise NotImplementedError(
            "functional concat store partition implements unit-width operands"
        )
    result_extent = result_type.shape[axis]
    if type(result_extent) is not int or result_extent != len(typed_operands):
        raise ValueError(
            "functional concat store partition does not cover the output extent"
        )
    if len(stores) != len(typed_operands):
        raise ValueError(
            "functional concat store partition has a missing or extra store"
        )

    non_axis_coordinates = {
        tuple(
            coordinate
            for dimension, coordinate in enumerate(
                coordinates_by_access[access.access_id]
            )
            if dimension != axis
        )
        for access in stores
    }
    if len(non_axis_coordinates) != 1:
        raise ValueError(
            "functional concat stores disagree on non-concat coordinates"
        )
    by_start: dict[int, Any] = {}
    for access in stores:
        coordinates = coordinates_by_access[access.access_id]
        coordinate = coordinates[axis]
        if not isinstance(coordinate, int) or coordinate in by_start:
            raise ValueError(
                "functional concat store partition has duplicate/nonconstant slices"
            )
        by_start[coordinate] = access
    expected_starts = set(range(len(typed_operands)))
    if set(by_start) != expected_starts:
        raise ValueError(
            "functional concat store partition has a gap or out-of-range slice"
        )
    slices = tuple(
        ConcatStoreSliceProjection(
            by_start[start].access_id,
            operation.operands[start],
            start,
        )
        for start in range(len(typed_operands))
    )
    return ConcatStorePartitionProjection(output, producer_ref, axis, slices)


def _project_direct_store_partition(
    j: Any,
    *,
    output: str,
    stores: Sequence[Any],
    coordinates_by_access: Mapping[str, tuple[str | int, ...]],
    predicates: Mapping[str, PredicateProjection | None],
    observable_value: str,
    direct_unit_dimensions: frozenset[str],
) -> DirectStorePartitionProjection:
    """Prove point stores exhaust one static axis of one complete J value.

    This relation is deliberately independent of the producer opcode.  The
    stores must all name the same producer and its complete observable result;
    no Source, C, pi, declaration order, or target lowering is consulted.
    """

    producer_refs = {access.node_ref for access in stores}
    if len(producer_refs) != 1 or None in producer_refs:
        raise NotImplementedError(
            "functional direct store partition requires one common producer"
        )
    producer_ref = next(iter(producer_refs))
    assert producer_ref is not None
    operation = _canonical_nodes(j).get(producer_ref)
    if operation is None:
        raise NotImplementedError(
            "functional direct store partition requires a J producer"
        )
    if any(predicates[access.access_id] is not None for access in stores):
        raise NotImplementedError(
            "functional direct store partition requires unpredicated stores"
        )
    if observable_value not in _result_names(operation):
        raise ValueError(
            "functional direct store partition does not store its observable J value"
        )

    value_types = j_elaboration(j).value_type_by_name
    output_type = value_types.get(output)
    result_type = value_types.get(observable_value)
    if output_type is None or result_type is None:
        raise ValueError("functional direct store partition lacks a closed type")
    if (
        result_type.dtype != output_type.dtype
        or result_type.shape != output_type.shape
    ):
        raise ValueError(
            "functional direct store partition producer type differs from output"
        )

    coordinates = [
        coordinates_by_access[access.access_id] for access in stores
    ]
    rank = len(output_type.shape)
    if any(len(item) != rank for item in coordinates):
        raise ValueError(
            "functional direct store partition rank differs from output"
        )
    varying_axes = [
        axis
        for axis in range(rank)
        if len({item[axis] for item in coordinates}) > 1
    ]
    if not varying_axes:
        raise ValueError(
            "functional boundary requires exactly one store per logical output"
        )
    if len(varying_axes) != 1:
        raise NotImplementedError(
            "functional direct stores require exactly one partition axis"
        )
    axis = varying_axes[0]
    starts = [item[axis] for item in coordinates]
    if any(not isinstance(start, int) for start in starts):
        raise NotImplementedError(
            "functional direct store partition requires constant coordinates"
        )
    extent = output_type.shape[axis]
    if type(extent) is not int:
        raise NotImplementedError(
            "functional direct store partition requires a static output extent"
        )

    for dimension in range(rank):
        if dimension == axis:
            continue
        coordinate = coordinates[0][dimension]
        if any(item[dimension] != coordinate for item in coordinates[1:]):
            raise ValueError(
                "functional direct stores disagree on non-partition coordinates"
            )
        dimension_extent = output_type.shape[dimension]
        if isinstance(coordinate, int) and not (
            coordinate == 0
            and _is_direct_unit_extent(
                dimension_extent, direct_unit_dimensions
            )
        ):
            raise ValueError(
                "functional direct stores do not cover a non-partition extent"
            )

    if len(stores) != extent:
        raise ValueError(
            "functional direct store partition has a missing or extra store"
        )
    by_start: dict[int, Any] = {}
    for access, start in zip(stores, starts, strict=True):
        assert isinstance(start, int)
        if start in by_start:
            raise ValueError(
                "functional direct store partition has duplicate slices"
            )
        by_start[start] = access
    if set(by_start) != set(range(extent)):
        raise ValueError(
            "functional direct store partition has a gap or out-of-range slice"
        )
    slices = tuple(
        DirectStoreSliceProjection(by_start[start].access_id, start)
        for start in range(extent)
    )
    return DirectStorePartitionProjection(
        output,
        observable_value,
        producer_ref,
        axis,
        slices,
    )


def _project_store_partition(
    j: Any,
    *,
    output: str,
    stores: Sequence[Any],
    coordinates_by_access: Mapping[str, tuple[str | int, ...]],
    predicates: Mapping[str, PredicateProjection | None],
    observable_value: str,
    direct_unit_dimensions: frozenset[str],
) -> ConcatStorePartitionProjection | DirectStorePartitionProjection:
    """Dispatch one multi-store output to its narrow J-only proof profile."""

    producer_refs = {access.node_ref for access in stores}
    producer_ref = (
        next(iter(producer_refs))
        if len(producer_refs) == 1 and None not in producer_refs
        else None
    )
    operation = (
        _canonical_nodes(j).get(producer_ref)
        if isinstance(producer_ref, str)
        else None
    )
    if operation is not None and operation.op == "concat":
        return _project_concat_store_partition(
            j,
            output=output,
            stores=stores,
            coordinates_by_access=coordinates_by_access,
            predicates=predicates,
            observable_value=observable_value,
        )
    return _project_direct_store_partition(
        j,
        output=output,
        stores=stores,
        coordinates_by_access=coordinates_by_access,
        predicates=predicates,
        observable_value=observable_value,
        direct_unit_dimensions=direct_unit_dimensions,
    )


def _is_total_conditional_store_pair(
    *,
    stores: Sequence[Any],
    coordinates_by_access: Mapping[str, tuple[str | int, ...]],
    predicates: Mapping[str, PredicateProjection | None],
    output_shape: Sequence[int | str],
    conditional_accesses: frozenset[str],
    arm_memberships: Mapping[str, tuple[tuple[str, str], ...]],
    direct_unit_dimensions: frozenset[str],
) -> bool:
    """Recognize two full writes selected by sibling arms of one effect_if."""

    if len(stores) != 2 or any(
        access.access_id not in conditional_accesses
        or predicates[access.access_id] is not None
        or access.node_ref is None
        for access in stores
    ):
        return False
    coordinates = [
        coordinates_by_access[access.access_id] for access in stores
    ]
    if coordinates[0] != coordinates[1] or len(coordinates[0]) != len(
        output_shape
    ):
        return False
    for coordinate, extent in zip(
        coordinates[0], output_shape, strict=True
    ):
        if isinstance(coordinate, int) and not (
            coordinate == 0
            and _is_direct_unit_extent(extent, direct_unit_dimensions)
        ):
            return False

    memberships = [
        dict(arm_memberships.get(access.node_ref, ())) for access in stores
    ]
    opposite_controls = {
        control_ref
        for control_ref, polarity in memberships[0].items()
        if memberships[1].get(control_ref)
        == ("negative" if polarity == "positive" else "positive")
    }
    return len(opposite_controls) == 1


def _flatten_conjunction(expression: ContractExpr) -> tuple[ContractExpr, ...]:
    if expression.op != "and":
        return (expression,)
    result: list[ContractExpr] = []
    for argument in expression.args:
        result.extend(_flatten_conjunction(argument))
    return tuple(result)


def _reference(expression: ContractExpr) -> str | None:
    if expression.op == "ref" and isinstance(expression.value, str):
        return expression.value
    return None


def _bound_value(expression: ContractExpr) -> int | str | None:
    if expression.op == "int" and isinstance(expression.value, int):
        return expression.value
    reference = _reference(expression)
    if reference is not None:
        return reference
    factors = _positive_product_factors(expression)
    if factors is not None and len(factors) > 1:
        return expression.to_text()
    if (
        expression.op == "floordiv"
        and len(expression.args) == 2
        and _reference(expression.args[0]) is not None
        and expression.args[1].op == "int"
        and isinstance(expression.args[1].value, int)
        and not isinstance(expression.args[1].value, bool)
        and expression.args[1].value > 0
    ):
        # Preserve the exact typed J expression.  This does not prove that the
        # sub-domain equals a tensor extent; it merely lets a ContractIR-aware
        # consumer distinguish an authored affine slice from unknown syntax.
        return expression.to_text()
    return None


def _positive_product_factors(
    expression: ContractExpr,
) -> tuple[int | str, ...] | None:
    """Return authored positive integer/ref factors without algebraic guessing."""

    if expression.op == "mul" and len(expression.args) == 2:
        left = _positive_product_factors(expression.args[0])
        right = _positive_product_factors(expression.args[1])
        if left is None or right is None:
            return None
        return (*left, *right)
    if (
        expression.op == "int"
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
        and expression.value > 0
    ):
        return (expression.value,)
    reference = _reference(expression)
    return (reference,) if reference is not None else None


def _domain_product_factors(j: Any) -> dict[str, tuple[int | str, ...]]:
    """Index only explicit product upper bounds already authored in J domains."""

    result: dict[str, tuple[int | str, ...]] = {}
    for _domain_id, expression in j_elaboration(j).domains:
        for term in _flatten_conjunction(expression):
            if term.op != "lt" or len(term.args) != 2:
                continue
            axis = _reference(term.args[0])
            factors = _positive_product_factors(term.args[1])
            if axis is None or factors is None or len(factors) < 2:
                continue
            previous = result.get(axis)
            if previous is not None and previous != factors:
                raise ValueError(
                    f"functional domain axis {axis!r} has conflicting products"
                )
            result[axis] = factors
    return result


def _predicate_projection(
    expression: ContractExpr,
    *,
    logical_guards: frozenset[str] = frozenset(),
) -> PredicateProjection:
    lower: set[str] = set()
    guards: set[str] = set()
    upper: dict[str, int | str] = {}
    for term in _flatten_conjunction(expression):
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
                "functional boundary supports only logical-domain predicates"
            )
        previous = upper.get(upper_axis)
        if previous is not None and previous != extent:
            raise ValueError(
                f"functional predicate axis {upper_axis!r} has conflicting extents"
            )
        upper[upper_axis] = extent
    if not upper and not guards:
        raise NotImplementedError(
            "functional boundary predicate requires an upper bound or J guard"
        )
    return PredicateProjection(
        frozenset(guards),
        frozenset(lower),
        tuple(sorted(upper.items())),
    )


def _value_dimension_pairs(j: Any) -> tuple[tuple[str, str], ...]:
    """Return the validated J-owned symbolic value/dimension equalities."""

    validate_value_dimension_relations(j)
    return tuple(
        (relation.value_ref, relation.dimension_ref)
        for relation in j.value_dimension_relations
    )


def _dimension_relation_tuples(
    j: Any,
) -> tuple[tuple[str, str, str, str | None, int | None], ...]:
    """Return exact, validated J-owned dimension definitions."""

    validate_dimension_relations(j)
    return tuple(
        (
            relation.result,
            relation.expr.op,
            relation.expr.arg,
            relation.expr.rhs,
            relation.expr.value,
        )
        for relation in j.dimension_relations
    )


def _functional_access_coordinates(
    j: Any,
    index: Any,
    target_shape: Sequence[Any],
) -> tuple[str | int, ...] | None:
    """Project direct or J-proved rank-2 row-major access coordinates.

    The affine case is deliberately narrow: one component must be exactly
    ``row * stride + column`` (up to commutative operand order), and the stride
    must be equated by J's value-dimension relation to the target's trailing
    extent.  General affine expressions remain unsupported.
    """

    direct = index.direct_coordinates
    if direct is not None:
        return direct
    scalar_value_refs = {
        parameter.logical_ref
        for parameter in j.signature.parameters
        if parameter.kind.value == "scalar" and parameter.logical_ref is not None
    }
    # A rank-1 physical view may be addressed by one logical domain coordinate
    # scaled by a runtime scalar stride.  The stride is already a typed J value
    # used by the ContractReference evaluator; the result still has the sole
    # logical axis, and concrete bounds are checked when that access executes.
    # This is intentionally narrower than a general affine projection: no
    # offset, sum, second domain axis, or non-scalar factor is accepted here.
    if len(target_shape) == 1 and len(index.components) == 1:
        expression = index.components[0]
        if expression.op == "mul" and len(expression.args) == 2:
            for axis_term, stride_term in (
                (expression.args[0], expression.args[1]),
                (expression.args[1], expression.args[0]),
            ):
                axis = (
                    axis_term.value
                    if axis_term.op == "ref" and isinstance(axis_term.value, str)
                    else None
                )
                scalar_stride = (
                    stride_term.op == "int"
                    or (
                        stride_term.op == "ref"
                        and isinstance(stride_term.value, str)
                        and stride_term.value in scalar_value_refs
                    )
                )
                if axis is not None and scalar_stride:
                    return (axis,)
    if len(target_shape) != 2 or len(index.components) != 1:
        return None
    expression = index.components[0]
    if expression.op != "add" or len(expression.args) != 2:
        return None

    # A captured rank-2 strided tensor may spell its physical address as the
    # ordered affine form ``axis0 * stride0 + axis1 * stride1``.  The expression
    # itself authors the tensor-axis order; both stride carriers must be typed
    # scalar J inputs, and no offset, coefficient folding, repeated axis, or
    # extra term is accepted.  ContractReference then evaluates the same access
    # through the tensor's logical coordinates while the runtime adapter retains
    # the captured physical strides.
    def ordered_axis_stride(term: ContractExpr) -> tuple[str, str] | None:
        if term.op != "mul" or len(term.args) != 2:
            return None
        for axis_term, stride_term in (
            (term.args[0], term.args[1]),
            (term.args[1], term.args[0]),
        ):
            axis = (
                axis_term.value
                if axis_term.op == "ref"
                and isinstance(axis_term.value, str)
                and re.fullmatch(r"k[0-9]+", axis_term.value)
                else None
            )
            stride = (
                stride_term.value
                if stride_term.op == "ref"
                and isinstance(stride_term.value, str)
                and stride_term.value in scalar_value_refs
                else None
            )
            if axis is not None and stride is not None:
                return axis, stride
        return None

    ordered_terms = tuple(ordered_axis_stride(term) for term in expression.args)
    if all(term is not None for term in ordered_terms):
        axes = tuple(term[0] for term in ordered_terms if term is not None)
        strides = tuple(term[1] for term in ordered_terms if term is not None)
        if len(set(axes)) == 2 and len(set(strides)) == 2:
            return axes

    trailing_extent = target_shape[1]
    value_dimensions = dict(_value_dimension_pairs(j))

    def ref_name(term: ContractExpr) -> str | None:
        return (
            term.value
            if term.op == "ref" and isinstance(term.value, str)
            else None
        )

    def is_stride(term: ContractExpr) -> bool:
        if (
            term.op == "int"
            and isinstance(term.value, int)
            and not isinstance(term.value, bool)
        ):
            return type(trailing_extent) is int and term.value == trailing_extent
        name = ref_name(term)
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
        column = ref_name(column_term)
        if column is None or product.op != "mul" or len(product.args) != 2:
            continue
        for row_term, stride_term in (
            (product.args[0], product.args[1]),
            (product.args[1], product.args[0]),
        ):
            row = ref_name(row_term)
            if row is None or row == column or not is_stride(stride_term):
                continue
            candidates.add((row, column))
    if len(candidates) != 1:
        return None
    return next(iter(candidates))


def _logical_index_axis_refs(
    index: Any,
    domains: Mapping[str, int | str],
) -> tuple[str, ...]:
    """Return J-domain axes used by one authored index, in expression order.

    An index-bounded gather has two independent pieces of logical context: the
    axes carried by the index tensor and the outer axes that select the source
    tensor slice.  The former can be recovered from the gather result type, but
    dropping the latter turns a predicate such as ``k2 < x2`` into a spurious
    non-access-axis error.  This projection reads only references already
    authored in J and filters out dimension/value refs by the closed domain map.
    """

    result: list[str] = []

    def visit(expression: ContractExpr) -> None:
        if expression.op == "ref":
            assert isinstance(expression.value, str)
            if expression.value in domains and expression.value not in result:
                result.append(expression.value)
        for argument in expression.args:
            visit(argument)

    for component in index.components:
        visit(component)
    return tuple(result)


def _flat_stride_factors(
    expression: ContractExpr,
    flat_axis: str,
) -> tuple[int | str, ...] | None:
    reference = _reference(expression)
    if reference == flat_axis:
        return ()
    if expression.op != "floordiv" or len(expression.args) != 2:
        return None
    if _reference(expression.args[0]) != flat_axis:
        return None
    return _positive_product_factors(expression.args[1])


def _mixed_radix_digit(
    expression: ContractExpr,
    *,
    flat_axis: str,
    declared_extent: int | str,
    domain_factors: tuple[int | str, ...],
) -> tuple[int, tuple[int | str, ...]] | None:
    """Match one exact mixed-radix digit and its authored suffix stride."""

    if expression.op == "mod" and len(expression.args) == 2:
        modulus = _positive_product_factors(expression.args[1])
        if modulus != (declared_extent,):
            return None
        stride = _flat_stride_factors(expression.args[0], flat_axis)
        if stride is None:
            return None
    else:
        stride = _flat_stride_factors(expression, flat_axis)
        if stride is None or not stride:
            return None

    matches = [
        index
        for index, factor in enumerate(domain_factors)
        if factor == declared_extent
        and tuple(domain_factors[index + 1 :]) == stride
    ]
    if len(matches) != 1:
        return None
    position = matches[0]
    if expression.op != "mod" and position != 0:
        return None
    return position, stride


def _mixed_radix_access_projection(
    access_id: str,
    index: Any,
    target_shape: Sequence[Any],
    domain_products: Mapping[str, tuple[int | str, ...]],
) -> tuple[
    tuple[str | int, ...],
    tuple[str, ...],
    MixedRadixAccessProjection,
] | None:
    """Prove one complete product-domain unflatten from J expressions alone."""

    components = index.components
    if len(components) != len(target_shape):
        return None
    candidates: list[
        tuple[
            tuple[str | int, ...],
            tuple[str, ...],
            MixedRadixAccessProjection,
        ]
    ] = []
    for flat_axis, factors in domain_products.items():
        coordinates: list[str | int] = []
        physical_axes: list[str] = []
        digits: list[MixedRadixDigitProjection] = []
        positions: set[int] = set()
        valid = True
        for tensor_axis, (component, extent) in enumerate(
            zip(components, target_shape, strict=True)
        ):
            if component.op == "int" and isinstance(component.value, int):
                coordinates.append(component.value)
                continue
            direct = _reference(component)
            if direct is not None and direct != flat_axis:
                coordinates.append(direct)
                if direct not in physical_axes:
                    physical_axes.append(direct)
                continue
            matched = _mixed_radix_digit(
                component,
                flat_axis=flat_axis,
                declared_extent=extent,
                domain_factors=factors,
            )
            if matched is None or matched[0] in positions:
                valid = False
                break
            position, stride = matched
            positions.add(position)
            coordinates.append(component.to_text())
            if flat_axis not in physical_axes:
                physical_axes.append(flat_axis)
            digits.append(
                MixedRadixDigitProjection(
                    tensor_axis,
                    extent,
                    stride,
                    component.to_text(),
                )
            )
        if not valid or positions != set(range(len(factors))):
            continue
        candidates.append(
            (
                tuple(coordinates),
                tuple(physical_axes),
                MixedRadixAccessProjection(
                    access_id,
                    flat_axis,
                    factors,
                    tuple(sorted(digits, key=lambda item: item.tensor_axis)),
                ),
            )
        )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _substitute_value_dimension_bounds(
    predicate: PredicateProjection,
    value_dimensions: Mapping[str, str],
) -> PredicateProjection:
    """Canonicalize only exact upper-bound refs authorized by J equality."""

    if not value_dimensions:
        return predicate
    return PredicateProjection(
        predicate.guards,
        predicate.lower_axes,
        tuple(
            (
                axis,
                value_dimensions.get(extent, extent)
                if isinstance(extent, str)
                else extent,
            )
            for axis, extent in predicate.upper_bounds
        ),
    )


def _resolve_runtime_dimension_bound(
    extent: int | str,
    logical_dimensions: Mapping[str, int] | None,
) -> int | str:
    """Evaluate one exact J integer bound only from this J invocation's x*."""

    if isinstance(extent, int) or logical_dimensions is None:
        return extent
    if extent == "?":
        return extent
    if extent in logical_dimensions:
        return logical_dimensions[extent]
    try:
        expression = parse_contract_expr(
            extent,
            expected=ContractSort.INTEGER,
            context="J functional runtime dimension bound",
        )
        if not expression.references.issubset(logical_dimensions):
            return extent
        value = evaluate_contract_expr(
            expression,
            logical_dimensions,
            expected_sort=ContractSort.INTEGER,
            context="J functional runtime dimension bound",
        )
    except (ContractSemanticsError, TypeError, ValueError):
        return extent
    return value if type(value) is int and value >= 0 else extent


def _resolve_runtime_predicate_bounds(
    predicate: PredicateProjection,
    logical_dimensions: Mapping[str, int] | None,
) -> PredicateProjection:
    if logical_dimensions is None:
        return predicate
    return PredicateProjection(
        predicate.guards,
        predicate.lower_axes,
        tuple(
            (
                axis,
                _resolve_runtime_dimension_bound(extent, logical_dimensions),
            )
            for axis, extent in predicate.upper_bounds
        ),
    )


def _j_predicate_projection(
    j: Any,
    expression: ContractExpr,
    *,
    logical_guards: frozenset[str] = frozenset(),
) -> PredicateProjection:
    """Project a predicate under J's dependent value/dimension domain."""

    return _substitute_value_dimension_bounds(
        _predicate_projection(expression, logical_guards=logical_guards),
        dict(_value_dimension_pairs(j)),
    )


def _logical_scalar_inputs(j: Any) -> frozenset[str]:
    inputs = {item.name: item for item in j.intent.inputs}
    return frozenset(
        parameter.logical_ref
        for parameter in j.signature.parameters
        if parameter.kind.value in {"scalar", "constexpr"}
        and parameter.logical_ref in inputs
        and not inputs[parameter.logical_ref].shape
    )


def _logical_boolean_guards(j: Any) -> frozenset[str]:
    inputs = {item.name: item for item in j.intent.inputs}
    return frozenset(
        name
        for name in _logical_scalar_inputs(j)
        if inputs[name].dtype in {"bool", "i1"}
    )


_MISSING_RUNTIME_VALUE = object()


def _rank_zero_runtime_value(value: Any) -> Any:
    """Return a Python scalar without coercing arrays or numeric kinds."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if tuple(shape) != ():
                return _MISSING_RUNTIME_VALUE
        except TypeError:
            return _MISSING_RUNTIME_VALUE
        item = getattr(value, "item", None)
        if not callable(item):
            return _MISSING_RUNTIME_VALUE
        try:
            value = item()
        except (TypeError, ValueError):
            return _MISSING_RUNTIME_VALUE
    elif isinstance(value, (list, tuple, dict, set)):
        return _MISSING_RUNTIME_VALUE
    return value


def _runtime_logical_scalar(
    name: str,
    *,
    expected_dtype: str,
    logical_inputs: Mapping[str, Any] | None,
) -> int | bool | object:
    if logical_inputs is None or name not in logical_inputs:
        return _MISSING_RUNTIME_VALUE
    value = _rank_zero_runtime_value(logical_inputs[name])
    if expected_dtype in {"bool", "i1"}:
        return value if type(value) is bool else _MISSING_RUNTIME_VALUE
    if expected_dtype == "index" or is_closed_integer_dtype(expected_dtype):
        if isinstance(value, Integral) and not isinstance(value, bool):
            return int(value)
    return _MISSING_RUNTIME_VALUE


def _runtime_predicate_tautology(
    expression: ContractExpr,
    *,
    access_id: str,
    access_axes: tuple[str, ...],
    domains: Mapping[str, int | str],
    logical_input_types: Mapping[str, str],
    logical_inputs: Mapping[str, Any] | None,
    logical_dimensions: Mapping[str, int] | None,
) -> tuple[PredicateProjection, RuntimePredicateProofReceipt] | None:
    """Prove a narrow predicate true at every point of one closed access domain.

    This is an invocation-local RefEval capability proof, never a J legality
    rule.  It handles only conjunctions of exact logical boolean guards,
    zero-lower bounds, and strict integer upper bounds.  A runtime bound is
    sufficient exactly when it is at least the concrete closed-domain extent.
    """

    lower_axes: set[str] = set()
    effective_upper: dict[str, int | str] = {}
    logical_refs: set[str] = set()
    saw_upper = False
    access_axis_set = set(access_axes)

    for term in _flatten_conjunction(expression):
        guard = _reference(term)
        negated = False
        if guard is None and term.op == "not" and len(term.args) == 1:
            guard = _reference(term.args[0])
            negated = guard is not None
        if guard is not None and logical_input_types.get(guard) in {"bool", "i1"}:
            value = _runtime_logical_scalar(
                guard,
                expected_dtype=logical_input_types[guard],
                logical_inputs=logical_inputs,
            )
            if value is _MISSING_RUNTIME_VALUE or bool(value) == negated:
                return None
            logical_refs.add(guard)
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
            if lower_axis not in access_axis_set or lower_axis not in domains:
                return None
            lower_axes.add(lower_axis)
            continue

        if term.op != "lt" or left is None or right is None:
            return None
        axis = _reference(left)
        if axis is None or axis not in access_axis_set:
            return None
        domain_extent = domains.get(axis)
        if domain_extent is None:
            return None
        concrete_domain_extent = (
            logical_dimensions.get(domain_extent)
            if isinstance(domain_extent, str)
            and logical_dimensions is not None
            else domain_extent
        )

        bound: int | str | None = _bound_value(right)
        if bound is None:
            return None
        if isinstance(bound, int):
            if (
                isinstance(concrete_domain_extent, int)
                and bound < concrete_domain_extent
            ):
                return None
            if not isinstance(concrete_domain_extent, int):
                return None
        elif bound in logical_input_types:
            dtype = logical_input_types[bound]
            if dtype in {"bool", "i1"} or not (
                dtype == "index" or is_closed_integer_dtype(dtype)
            ):
                return None
            runtime_bound = _runtime_logical_scalar(
                bound,
                expected_dtype=dtype,
                logical_inputs=logical_inputs,
            )
            if (
                runtime_bound is _MISSING_RUNTIME_VALUE
                or not isinstance(concrete_domain_extent, int)
                or runtime_bound < concrete_domain_extent
            ):
                return None
            logical_refs.add(bound)
        elif bound != domain_extent:
            return None

        effective_upper[axis] = domain_extent
        saw_upper = True

    if not saw_upper:
        return None
    canonical = PredicateProjection(
        frozenset(),
        frozenset(lower_axes),
        tuple(sorted(effective_upper.items())),
    )
    receipt = RuntimePredicateProofReceipt(
        access_id=access_id,
        predicate_digest=_digest({"predicate": expression.to_text()}),
        logical_input_refs=tuple(sorted(logical_refs)),
        domain_axes=tuple(sorted(effective_upper)),
    )
    return canonical, receipt


def _boundary_referenced_axes(j: Any) -> frozenset[str]:
    """Return logical axes that participate in an observable boundary use."""

    referenced: set[str] = set()
    for access in j_elaboration(j).accesses:
        for component in access.index.components:
            referenced.update(component.references)
        if access.predicate is not None:
            referenced.update(access.predicate.references)
    return frozenset(referenced)


def _domain_defined_axes(j: Any, expression: ContractExpr) -> frozenset[str]:
    """Separate domain coordinates from extents and logical scalar inputs."""

    external_refs = {
        dimension
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dimension in tensor.shape
        if isinstance(dimension, str) and dimension != "?"
    }
    external_refs.update(item.name for item in j.intent.inputs)
    return expression.references - external_refs


def _full_domains(j: Any) -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    logical_guards = _logical_boolean_guards(j)
    boundary_axes = _boundary_referenced_axes(j)
    for domain_id, expression in j_elaboration(j).domains:
        try:
            contract = _j_predicate_projection(
                j, expression, logical_guards=logical_guards
            )
        except NotImplementedError as exc:
            domain_axes = _domain_defined_axes(j, expression)
            if domain_axes and domain_axes.isdisjoint(boundary_axes):
                continue
            raise NotImplementedError(
                "functional dense domain requires complete logical axis bounds; "
                f"unsupported domain {domain_id!r}"
            ) from exc
        if contract.guards and not contract.upper_bounds:
            continue
        if {axis for axis, _ in contract.upper_bounds} - set(contract.lower_axes):
            raise NotImplementedError(
                f"functional dense domain {domain_id!r} lacks a lower zero bound"
            )
        for axis, extent in contract.upper_bounds:
            previous = result.get(axis)
            if previous is not None and previous != extent:
                raise ValueError(
                    f"functional domain axis {axis!r} has conflicting extents"
                )
            result[axis] = extent
    return result


def access_predicates_are_logical_domain_total(j: Any) -> bool:
    """Prove the narrow case where every access predicate is domain-total.

    This uses only authored J domains and predicates.  It does not infer an
    access mapping or erase a consumer-local guard; indexed J evaluators use it
    only when gather/scatter already carry the complete typed index semantics.
    """

    try:
        domains = _full_domains(j)
        logical_guards = _logical_boolean_guards(j)
        for access in j_elaboration(j).accesses:
            if access.predicate is None:
                continue
            projection = _j_predicate_projection(
                j,
                access.predicate,
                logical_guards=logical_guards,
            )
            if projection.guards:
                return False
            if any(
                domains.get(axis) != extent
                for axis, extent in projection.upper_bounds
            ):
                return False
            if not set(projection.lower_axes).issubset(domains):
                return False
    except (NotImplementedError, ValueError):
        return False
    return True


def functional_boundary_closure_issues(
    j: Any,
) -> tuple[FunctionalBoundaryClosureIssue, ...]:
    """Check the shared, source-free logical-domain closure subset.

    The checker intentionally recognizes only the same zero-based rectangular
    logical-domain profile that the functional projection can erase into
    complete logical tensors.  Unsupported affine/logical predicates remain a
    consumer capability question; they are not made semantically illegal by
    TypeCheck.  Once an access predicate is in the recognized profile, however,
    its direct index axis and extent must agree with ``boundary.domains`` and
    the typed tensor shape.  Neither Source, C, pi, nor runtime values are
    consulted to fill a missing relation.
    """

    j = elaborate_j(j)
    direct_unit_dimensions = direct_unit_dimension_refs(
        j.dimension_constraints
    )
    boundary = j.boundary
    if not boundary.accesses:
        return ()

    # A non-rectangular or otherwise unsupported domain may still be a valid J
    # contract.  Keep checking independent typed coordinate/dataflow facts,
    # while skipping only the rectangular domain/predicate legality subset.
    rectangular_domains = True
    try:
        domains = _full_domains(j)
    except NotImplementedError:
        domains = {}
        rectangular_domains = False
    except ValueError as exc:
        return (
            FunctionalBoundaryClosureIssue(
                "j.boundary.domains",
                str(exc),
            ),
        )

    dimensions = {
        reference
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dimension in tensor.shape
        for reference in dimension_expression_references(dimension)
        if reference != "?"
    }
    tensors = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }
    logical_guards = _logical_boolean_guards(j)
    elaboration = j_elaboration(j)
    parsed_accesses = elaboration.access_by_id
    value_types = elaboration.value_type_by_name
    nodes = _canonical_nodes(j)
    access_positions = {
        access.access_id: index for index, access in enumerate(boundary.accesses)
    }
    direct_coordinates: dict[str, tuple[str | int, ...]] = {}
    issues: list[FunctionalBoundaryClosureIssue] = []

    null_load_views: dict[str, dict[tuple[str, str | None, str], list[str]]] = {}
    for access in boundary.accesses:
        if access.kind.value != "load" or access.node_ref is not None:
            continue
        parsed_access = parsed_accesses.get(access.access_id)
        if parsed_access is None:
            continue
        view = (
            parsed_access.index.to_text(),
            (
                parsed_access.predicate.to_text()
                if parsed_access.predicate is not None
                else None
            ),
            json.dumps(
                access.fill_value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
        null_load_views.setdefault(access.tensor, {}).setdefault(view, []).append(
            access.access_id
        )
    for tensor, views in sorted(null_load_views.items()):
        if len(views) <= 1:
            continue
        access_ids = sorted(
            access_id for ids in views.values() for access_id in ids
        )
        issues.append(
            FunctionalBoundaryClosureIssue(
                "j.boundary.accesses",
                "one logical input has multiple distinct node_ref=null access "
                "views; consumer-specific index/predicate views must identify "
                "their Layer-A consumer",
                f"{tensor}:{','.join(access_ids)}",
            )
        )

    for index, access in enumerate(boundary.accesses):
        parsed_access = parsed_accesses.get(access.access_id)
        tensor = tensors.get(access.tensor)
        if parsed_access is None or tensor is None:
            # Canonical elaboration/ordinary reference checking owns these
            # errors.  Do not duplicate them under the closure rule.
            continue
        operation = nodes.get(access.node_ref) if access.node_ref else None
        matching_index_bounds = [
            item
            for item in boundary.index_bounds
            if operation is not None
            and operation.op == "gather"
            and access.kind.value == "load"
            and item.node_ref == access.node_ref
            and item.data == access.tensor
            and len(operation.operands) > 1
            and operation.operands[0] == access.tensor
            and item.indices == operation.operands[1]
        ]
        bounded_gather_axis = (
            matching_index_bounds[0].axis
            if len(matching_index_bounds) == 1
            else None
        )
        coordinates = parsed_access.index.direct_coordinates
        if coordinates is None:
            # General affine/diagonal accesses are outside this projection
            # profile, not automatically invalid J.
            continue
        axes = tuple(
            coordinate for coordinate in coordinates
            if isinstance(coordinate, str)
        )
        if len(set(axes)) != len(axes):
            # A diagonal/ref-reusing access remains a general capability case.
            continue
        direct_coordinates[access.access_id] = coordinates

        index_path = f"j.boundary.accesses[{index}].index_expression"
        if len(coordinates) != len(tensor.shape):
            issues.append(
                FunctionalBoundaryClosureIssue(
                    index_path,
                    "functional direct access rank differs from its J tensor",
                    access.access_id,
                )
            )
            continue
        has_static_coordinate_error = False
        for tensor_axis, (coordinate, declared_extent) in enumerate(
            zip(coordinates, tensor.shape, strict=True)
        ):
            if isinstance(coordinate, int):
                if (
                    type(declared_extent) is int
                    and not 0 <= coordinate < declared_extent
                ):
                    issues.append(
                        FunctionalBoundaryClosureIssue(
                            index_path,
                            "functional constant access coordinate is out of bounds",
                            access.access_id,
                        )
                    )
                    has_static_coordinate_error = True
                # A constant against a symbolic extent remains a consumer
                # capability question; TypeCheck does not invent a lower bound.
                continue
            axis = coordinate
            domain_extent = domains.get(axis) if rectangular_domains else None
            if tensor_axis == bounded_gather_axis:
                # The consumer domain describes the typed gather result, not the
                # full source-tensor extent on the gathered axis.  The exact
                # data/index/axis ownership above and the ordinary index-bound
                # checks retain the safety obligation.
                continue
            if domain_extent is not None and domain_extent != declared_extent:
                issues.append(
                    FunctionalBoundaryClosureIssue(
                        index_path,
                        f"functional axis {axis!r} extent {domain_extent!r} "
                        f"differs from J tensor extent {declared_extent!r}",
                        access.access_id,
                    )
                )

        if not has_static_coordinate_error and access.kind.value == "load":
            non_singleton_constants = [
                (axis, coordinate)
                for axis, (coordinate, extent) in enumerate(
                    zip(coordinates, tensor.shape, strict=True)
                )
                if isinstance(coordinate, int)
                and type(extent) is int
                and not (coordinate == 0 and extent == 1)
            ]
            # Only contradictions inside the one-axis scalar-take profile are
            # J legality.  Other constant selections remain valid-but-unsupported.
            if len(non_singleton_constants) == 1:
                coordinate_axis, coordinate = non_singleton_constants[0]
                if (
                    operation is not None
                    and operation.op == "gather"
                    and len(operation.operands) == 2
                    and operation.operands[0] == access.tensor
                    and operation.attrs.get("mode", "along_axis") == "take"
                ):
                    gather_axis = operation.attrs.get("axis")
                    if gather_axis != coordinate_axis:
                        issues.append(
                            FunctionalBoundaryClosureIssue(
                                index_path,
                                "functional constant load axis differs from its "
                                "J gather axis",
                                access.access_id,
                            )
                        )
                    index_ref = operation.operands[1]
                    producer = _constant_int_producer(j, index_ref)
                    if producer is not None:
                        if producer[1] != coordinate:
                            issues.append(
                                FunctionalBoundaryClosureIssue(
                                    index_path,
                                    "functional constant load coordinate differs "
                                    "from its J gather index",
                                    access.access_id,
                                )
                            )
                        matching_bounds = [
                            relation
                            for relation in boundary.index_bounds
                            if (
                                relation.node_ref == access.node_ref
                                and relation.data == access.tensor
                                and relation.indices == index_ref
                                and relation.axis == gather_axis
                            )
                        ]
                        if len(matching_bounds) != 1:
                            issues.append(
                                FunctionalBoundaryClosureIssue(
                                    "j.boundary.index_bounds",
                                    "functional constant gather requires one "
                                    "matching index-bounds relation",
                                    access.node_ref,
                                )
                            )
                        data_type = value_types.get(access.tensor)
                        index_type = value_types.get(index_ref)
                        result_names = _result_names(operation)
                        result_type = (
                            value_types.get(result_names[0])
                            if len(result_names) == 1 else None
                        )
                        if (
                            isinstance(gather_axis, int)
                            and data_type is not None
                            and index_type is not None
                            and result_type is not None
                            and not index_type.shape
                            and 0 <= gather_axis < len(data_type.shape)
                        ):
                            expected_shape = (
                                *data_type.shape[:gather_axis],
                                *data_type.shape[gather_axis + 1 :],
                            )
                            if (
                                result_type.dtype != data_type.dtype
                                or result_type.shape != expected_shape
                            ):
                                issues.append(
                                    FunctionalBoundaryClosureIssue(
                                        f"j.node_ids.{access.node_ref}",
                                        "functional constant gather result type "
                                        "differs from scalar take",
                                        access.node_ref,
                                    )
                                )

        parsed_predicate = parsed_access.predicate
        if parsed_predicate is None or not rectangular_domains:
            continue
        try:
            predicate = _j_predicate_projection(
                j,
                parsed_predicate,
                logical_guards=logical_guards,
            )
        except NotImplementedError:
            # Logical predicates outside the rectangular mask-erasure profile
            # remain valid J and must not be blanket-rejected by TypeCheck.
            continue
        except ValueError as exc:
            issues.append(
                FunctionalBoundaryClosureIssue(
                    f"j.boundary.accesses[{index}].predicate",
                    str(exc),
                    access.access_id,
                )
            )
            continue

        # ``i < j`` is a logical cross-axis predicate, not a rectangular
        # physical padding bound.  Only integer constants and declared J shape
        # dimensions identify the shared functional-domain profile.
        if any(
            not isinstance(extent, int) and extent not in dimensions
            for _axis, extent in predicate.upper_bounds
        ):
            continue
        predicate_path = f"j.boundary.accesses[{index}].predicate"
        for axis, extent in predicate.upper_bounds:
            if axis not in axes:
                issues.append(
                    FunctionalBoundaryClosureIssue(
                        predicate_path,
                        "functional predicate upper bound is not an access axis",
                        access.access_id,
                    )
                )
                continue
            if domains.get(axis) != extent:
                issues.append(
                    FunctionalBoundaryClosureIssue(
                        predicate_path,
                        "functional predicate differs from the full logical "
                        "domain declared by boundary.domains",
                        access.access_id,
                    )
                )

    returns = {item.output: item.value for item in j.intent.returns}
    stores_by_output: dict[str, list[Any]] = {}
    for access in boundary.accesses:
        if access.kind.value == "store":
            stores_by_output.setdefault(access.tensor, []).append(access)
    for output, stores in stores_by_output.items():
        if len(stores) < 2:
            continue
        producer_refs = {access.node_ref for access in stores}
        if len(producer_refs) != 1 or None in producer_refs:
            continue
        producer_ref = next(iter(producer_refs))
        assert producer_ref is not None
        operation = nodes.get(producer_ref)
        output_type = value_types.get(output)
        if operation is None or output_type is None:
            continue
        if (
            any(access.predicate is not None for access in stores)
            or any(access.access_id not in direct_coordinates for access in stores)
        ):
            continue
        observable_value = returns[output]
        try:
            _project_store_partition(
                j,
                output=output,
                stores=stores,
                coordinates_by_access=direct_coordinates,
                predicates={access.access_id: None for access in stores},
                observable_value=observable_value,
                direct_unit_dimensions=direct_unit_dimensions,
            )
        except NotImplementedError:
            # General/wide/multi-axis/symbolic/predicated store profiles remain
            # valid J and are a consumer capability question.
            continue
        except ValueError as exc:
            first_index = access_positions[stores[0].access_id]
            issues.append(
                FunctionalBoundaryClosureIssue(
                    f"j.boundary.accesses[{first_index}]",
                    str(exc),
                    output,
                )
            )

    return tuple(issues)


def _broadcast_axes(j: Any) -> dict[str, dict[str, tuple[int, ...]]]:
    nodes = _node_table(j)
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for projection in j.boundary.broadcasts:
        op = nodes.get(projection.node_ref)
        if op is None or not isinstance(op.result, str):
            raise NotImplementedError(
                "functional broadcast requires one result consumer"
            )
        if projection.operand not in op.operands:
            raise ValueError("functional broadcast names a non-consumed operand")
        if any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in projection.axes
        ):
            raise NotImplementedError(
                "functional projection does not implement symbolic broadcast axes"
            )
        axes = tuple(int(axis) for axis in projection.axes)
        previous = result.setdefault(op.result, {}).get(projection.operand)
        if previous is not None and previous != axes:
            raise ValueError("conflicting functional broadcast relations")
        result[op.result][projection.operand] = axes
    return result


def _numpy_broadcast_positions(
    source: Sequence[int | str],
    target: Sequence[int | str],
) -> tuple[int, ...] | None:
    if len(source) > len(target):
        return None
    offset = len(target) - len(source)
    if not all(
        source_extent == 1 or source_extent == target[offset + index]
        for index, source_extent in enumerate(source)
    ):
        return None
    return tuple(range(offset, len(target)))


def _shape_can_numpy_broadcast(
    source: Sequence[int | str],
    target: Sequence[int | str],
) -> bool:
    return _numpy_broadcast_positions(source, target) is not None


def _axis_broadcast_positions(
    source: Sequence[int | str],
    target: Sequence[int | str],
    axes: Sequence[int],
) -> tuple[int, ...] | None:
    """Mirror the emitter's closed rank-insertion broadcast proof."""

    target_rank = len(target)
    normalized: list[int] = []
    for raw_axis in axes:
        axis = raw_axis if raw_axis >= 0 else raw_axis + target_rank
        if axis < 0 or axis >= target_rank:
            return None
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        return None
    if not source:
        return ()
    axis_set = set(normalized)
    memo: dict[tuple[int, int], tuple[int, ...] | None] = {}

    def align(source_index: int, target_index: int) -> tuple[int, ...] | None:
        key = (source_index, target_index)
        if key in memo:
            return memo[key]
        if target_index == target_rank:
            result = () if source_index == len(source) else None
            memo[key] = result
            return result
        if target_index in axis_set:
            if (
                source_index < len(source)
                and source[source_index] == 1
            ):
                suffix = align(source_index + 1, target_index + 1)
                if suffix is not None:
                    result = (target_index, *suffix)
                    memo[key] = result
                    return result
            suffix = align(source_index, target_index + 1)
            if suffix is not None:
                memo[key] = suffix
                return suffix
        elif (
            source_index < len(source)
            and source[source_index] == target[target_index]
        ):
            suffix = align(source_index + 1, target_index + 1)
            if suffix is not None:
                result = (target_index, *suffix)
                memo[key] = result
                return result
        memo[key] = None
        return None

    return align(0, 0)


def _shape_can_broadcast_on_axes(
    source: Sequence[int | str],
    target: Sequence[int | str],
    axes: Sequence[int],
) -> bool:
    return _axis_broadcast_positions(source, target, axes) is not None


def _validate_broadcast_closure(
    j: Any,
    broadcasts: Mapping[str, Mapping[str, tuple[int, ...]]],
) -> None:
    """Prove every pointwise operand reaches its declared result shape."""

    value_types = j_elaboration(j).value_type_by_name
    for loop in (op for op in _walk_ops(j.intent.body) if op.op == "loop"):
        for scan in loop.scan:
            source = value_types.get(scan.source)
            if source is not None:
                value_types[scan.name] = source
        for state in loop.carry:
            if state.type is None:
                continue
            state_type = CanonicalValueType.from_value_type(
                state.type.as_value_type()
            )
            value_types[state.body_argument] = state_type
            value_types[state.next_value] = state_type
            value_types[state.result_value] = state_type
        for child in loop.body:
            if not isinstance(child.result, str):
                continue
            if isinstance(child.type, dict):
                if child.type.get("kind") in {"record", "ragged"}:
                    value_types[child.result] = CanonicalValueType.from_json(
                        child.type,
                        context=f"IntentOp {child.op}.type",
                    )
                    continue
                dtype = child.type.get("dtype")
                shape = child.type.get("shape")
                if isinstance(dtype, str) and isinstance(shape, list):
                    value_types[child.result] = CanonicalValueType(dtype, tuple(shape))
                    continue
            if child.op in {"map_unary", "cast", "scan"} and child.operands:
                source = value_types.get(child.operands[0])
                if source is not None:
                    value_types[child.result] = source
            elif child.op == "reduce" and child.operands:
                source = value_types.get(child.operands[0])
                axes = child.attrs.get("axes")
                if source is not None and isinstance(axes, list):
                    normalized = {
                        axis if axis >= 0 else len(source.shape) + axis
                        for axis in axes
                        if isinstance(axis, int) and not isinstance(axis, bool)
                    }
                    value_types[child.result] = CanonicalValueType(
                        source.dtype,
                        tuple(
                            dimension
                            for index, dimension in enumerate(source.shape)
                            if index not in normalized
                        ),
                    )
    operations = _walk_ops(j.intent.body)
    consumers = {
        op.result: op
        for op in operations
        if isinstance(op.result, str)
    }
    for result, operands in broadcasts.items():
        consumer = consumers.get(result)
        if consumer is None or consumer.op not in {"map_binary", "select"}:
            raise NotImplementedError(
                "functional broadcast relation requires a pointwise consumer"
            )
        if any(operand not in consumer.operands for operand in operands):
            raise ValueError(
                "functional broadcast relation names a non-consumed operand"
            )

    for op in operations:
        if op.op not in {"map_binary", "select"} or not isinstance(op.result, str):
            continue
        result_type = value_types.get(op.result)
        if result_type is None:
            raise ValueError("functional pointwise result has no closed type")
        target_shape = result_type.shape
        declared = broadcasts.get(op.result, {})
        for operand in op.operands:
            operand_type = value_types.get(operand)
            if operand_type is None:
                raise ValueError("functional pointwise operand has no closed type")
            axes = declared.get(operand)
            if axes is None:
                closed = _shape_can_numpy_broadcast(
                    operand_type.shape,
                    target_shape,
                )
            else:
                closed = _shape_can_broadcast_on_axes(
                    operand_type.shape,
                    target_shape,
                    axes,
                )
            if not closed:
                raise NotImplementedError(
                    "functional broadcast relation is incomplete for a "
                    "pointwise operand"
                )


def _project_access_guard_relations(
    j: Any,
    broadcasts: dict[str, dict[str, tuple[int, ...]]],
) -> tuple[AccessGuardProjection, ...]:
    """Consume every v5 access guard without creating value semantics.

    A guarded store is executable only when its Layer-A producer is already an
    explicit ``select``.  A guarded load is projected to exactly one direct
    operand edge of its authored consumer; RefEval uses that receipt to apply
    the authored guard, polarity and fill before evaluating the consumer.
    Ambiguous repeated operands and guarded updates remain unsupported.
    """

    if not j.boundary.access_guard_relations:
        return ()
    accesses = {item.access_id: item for item in j.boundary.accesses}
    nodes = _node_table(j)
    value_types = j_elaboration(j).value_type_by_name
    outputs = {item.name for item in j.intent.outputs}
    returns = {item.output: item.value for item in j.intent.returns}
    mappings = {item.output: item.value for item in j.boundary.output_mappings}
    inout_by_output: dict[str, int] = {}
    for parameter in j.signature.parameters:
        if parameter.mode.value != "inout" or parameter.logical_ref is None:
            continue
        inout_by_output[parameter.logical_ref] = (
            inout_by_output.get(parameter.logical_ref, 0) + 1
        )

    receipts: list[AccessGuardProjection] = []
    seen_accesses: set[str] = set()
    for relation in j.boundary.access_guard_relations:
        access = accesses.get(relation.access_ref)
        if access is None:
            raise ValueError("functional access guard references an unknown access")
        if relation.access_ref in seen_accesses:
            raise ValueError(
                "functional access guard requires exactly one relation per access"
            )
        seen_accesses.add(relation.access_ref)
        guard_type = value_types.get(relation.value_ref)
        target_type = value_types.get(access.tensor)
        if guard_type is None or target_type is None:
            raise ValueError("functional access guard lacks a closed Layer-A type")
        if (
            not guard_type.is_tensor
            or guard_type.dtype is None
            or guard_type.dtype.casefold() != "i1"
        ):
            raise ValueError("functional access guard value must have dtype i1")
        axes = tuple(relation.axes)
        if axes != tuple(sorted(set(axes))):
            raise ValueError(
                "functional access guard axes must be strictly increasing and unique"
            )
        if not _shape_can_broadcast_on_axes(
            guard_type.shape,
            target_type.shape,
            axes,
        ):
            raise ValueError(
                "functional access guard axes cannot project the guard to its target"
            )

        if access.kind.value == "load":
            if access.fill_value is None:
                raise ValueError("functional guarded load requires an explicit fill")
            if access.node_ref is None:
                raise NotImplementedError(
                    "functional guarded load requires one explicit consumer node"
                )
            operation = nodes.get(access.node_ref)
            if operation is None or not isinstance(operation.result, str):
                raise NotImplementedError(
                    "functional guarded load requires one single-result consumer"
                )
            operand_indices = [
                index
                for index, operand in enumerate(operation.operands)
                if operand == access.tensor
            ]
            if len(operand_indices) != 1:
                raise NotImplementedError(
                    "functional guarded load requires one unambiguous direct "
                    "consumer operand"
                )
            operation_type = value_types.get(operation.result)
            if (
                relation.value_ref in operation.operands
                and operation_type == target_type
            ):
                prior = broadcasts.setdefault(operation.result, {}).get(
                    relation.value_ref
                )
                if prior is not None and prior != axes:
                    raise ValueError(
                        "functional guarded load conflicts with another "
                        "broadcast mapping"
                    )
                broadcasts[operation.result][relation.value_ref] = axes
            receipts.append(
                AccessGuardProjection(
                    relation.access_ref,
                    relation.value_ref,
                    access.node_ref,
                    operation.result,
                    axes,
                    relation.polarity.value,
                    access_kind="load",
                    tensor=access.tensor,
                    operand_index=operand_indices[0],
                    fill_value=access.fill_value,
                )
            )
            continue
        if access.kind.value == "update":
            raise NotImplementedError(
                "functional access guard does not implement guarded update access"
            )
        if access.kind.value != "store":  # pragma: no cover - closed enum
            raise ValueError("functional access guard has an unknown access kind")
        if access.tensor not in outputs or inout_by_output.get(access.tensor) != 1:
            raise ValueError(
                "functional guarded store requires exactly one explicit inout output"
            )
        if access.node_ref is None:
            raise NotImplementedError(
                "functional guarded store requires an explicit Layer-A select"
            )
        operation = nodes.get(access.node_ref)
        if (
            operation is None
            or operation.op != "select"
            or not isinstance(operation.result, str)
            or len(operation.operands) != 3
            or operation.operands[0] != relation.value_ref
        ):
            raise NotImplementedError(
                "functional guarded store requires an explicit Layer-A select "
                "whose condition is relation.value_ref"
            )
        observable_value = returns[access.tensor]
        if (
            observable_value != operation.result
            and not _store_result_reaches_output_through_enclosing_carry(
                j,
                operation=operation,
                producer_value=operation.result,
                observable_value=observable_value,
            )
        ):
                raise ValueError(
                    "functional guarded store select does not reach its "
                    "observable output through one explicit carry edge"
                )

        result_type = value_types.get(operation.result)
        if result_type is None:
            raise ValueError("functional access guard lacks a closed Layer-A type")
        if result_type != target_type:
            raise ValueError(
                "functional guarded store select type differs from its output"
            )
        if not _shape_can_broadcast_on_axes(
            guard_type.shape,
            target_type.shape,
            axes,
        ):
            raise ValueError(
                "functional access guard axes cannot project the guard to its output"
            )

        prior = broadcasts.setdefault(operation.result, {}).get(
            relation.value_ref
        )
        if prior is not None and prior != axes:
            raise ValueError(
                "functional access guard conflicts with another broadcast mapping"
            )
        broadcasts[operation.result][relation.value_ref] = axes
        receipts.append(
            AccessGuardProjection(
                relation.access_ref,
                relation.value_ref,
                access.node_ref,
                operation.result,
                axes,
                relation.polarity.value,
            )
        )

    if len(receipts) != len(j.boundary.access_guard_relations):
        raise ValueError("functional access guard left an authored relation unconsumed")
    return tuple(receipts)


def _reduction_padding_analysis(
    j: Any,
    *,
    predicate_overrides: Mapping[str, PredicateProjection | None] | None = None,
) -> tuple[
    tuple[FunctionalBoundaryClosureIssue, ...],
    tuple[FunctionalBoundaryClosureIssue, ...],
    tuple[ConsumerExtensionRequirement, ...],
    frozenset[str],
]:
    j = elaborate_j(j)
    elaboration = j_elaboration(j)
    value_types = elaboration.value_type_by_name
    broadcasts = _broadcast_axes(j)
    logical_guards = _logical_boolean_guards(j)
    try:
        full_domains = _full_domains(j)
    except (NotImplementedError, ValueError):
        # The TypeCheck-facing analysis is intentionally only the closed
        # zero-based rectangular profile.  General logical predicates remain
        # valid J and are handled as a consumer capability boundary.
        full_domains = {}
    tensors = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }

    padding_positions: dict[str, dict[str, frozenset[int]]] = {
        item.name: {} for item in j.intent.inputs
    }
    operations = _node_table(j)
    edge_padding: dict[
        tuple[str, str], dict[str, frozenset[int]]
    ] = {}
    explicit_load_edges: set[tuple[str, str]] = set()
    for access in j.boundary.accesses:
        if access.kind.value != "load" or access.node_ref is None:
            continue
        operation = operations.get(access.node_ref)
        operand_refs = (
            {
                operand
                for operand in operation.operands
                if operand == access.tensor
            }
            if operation is not None
            else set()
        )
        if not operand_refs and operation is not None and len(
            operation.operands
        ) == 1:
            operand_refs = {operation.operands[0]}
        if len(operand_refs) == 1:
            explicit_load_edges.add(
                (access.node_ref, next(iter(operand_refs)))
            )
    seed_issues: list[FunctionalBoundaryClosureIssue] = []
    for access in j.boundary.accesses:
        if access.kind.value != "load":
            continue
        parsed = elaboration.access_by_id.get(access.access_id)
        if parsed is None or parsed.index.direct_axes is None:
            continue
        axes = parsed.index.direct_axes
        if parsed.predicate is None:
            continue
        tensor = tensors.get(access.tensor)
        if tensor is None or len(axes) != len(tensor.shape):
            continue
        if predicate_overrides is not None:
            predicate = predicate_overrides.get(access.access_id)
            if predicate is None:
                continue
        else:
            try:
                predicate = _j_predicate_projection(
                    j,
                    parsed.predicate,
                    logical_guards=logical_guards,
                )
            except (NotImplementedError, ValueError):
                continue
        if predicate.guards or not predicate.upper_bounds:
            continue
        # Seed physical-padding provenance only when J itself proves a full
        # rectangular logical domain and the predicate bound agrees with the
        # typed tensor extent.  A guard or cross-axis/logical predicate is not
        # silently reclassified as padding by TypeCheck.
        if any(
            axis not in axes
            or full_domains.get(axis) != extent
            or tensor.shape[axes.index(axis)] != extent
            for axis, extent in predicate.upper_bounds
        ):
            continue
        masked = {
            axes.index(axis)
            for axis, _extent in predicate.upper_bounds
            if axis in axes
        }
        if masked:
            target = padding_positions.setdefault(access.tensor, {})
            if access.node_ref is not None:
                operation = operations.get(access.node_ref)
                operand_refs = (
                    {
                        operand
                        for operand in operation.operands
                        if operand == access.tensor
                    }
                    if operation is not None
                    else set()
                )
                if not operand_refs and operation is not None and len(
                    operation.operands
                ) == 1:
                    operand_refs = {operation.operands[0]}
                if len(operand_refs) != 1:
                    if not predicate_dominated_pointwise_load(
                        j, access.access_id
                    ):
                        seed_issues.append(
                            FunctionalBoundaryClosureIssue(
                                "j.boundary.accesses",
                                "functional projection cannot bind one predicated "
                                "load to a unique J operand edge",
                                access.access_id,
                            )
                        )
                        continue
                else:
                    target = edge_padding.setdefault(
                        (access.node_ref, next(iter(operand_refs))),
                        {},
                    )
            by_access = target
            by_access[access.access_id] = frozenset(
                set(by_access.get(access.access_id, frozenset())) | masked
            )

    path_by_marker: dict[int, str] = {}

    def record_paths(body: Sequence[IntentOp], prefix: str = "body") -> None:
        for index, op in enumerate(body):
            path = f"{prefix}[{index}]"
            path_by_marker[id(op)] = path
            for suffix, region_body in _child_regions(op):
                record_paths(region_body, f"{path}.{suffix}")

    record_paths(j.intent.body)
    ref_by_path = {path: node_ref for node_ref, path in j.node_ids.items()}
    mean_issues: list[FunctionalBoundaryClosureIssue] = []
    consumer_issues: list[FunctionalBoundaryClosureIssue] = list(seed_issues)
    requirements: set[ConsumerExtensionRequirement] = set()
    affected_consumers: set[str] = set()

    def merge_projected(
        target: dict[str, set[int]],
        source: Mapping[str, frozenset[int]],
        positions: tuple[int, ...] | None = None,
    ) -> None:
        for access_ref, axes in source.items():
            projected = (
                axes
                if positions is None
                else frozenset(
                    positions[index]
                    for index in axes
                    if index < len(positions)
                )
            )
            target.setdefault(access_ref, set()).update(projected)

    def freeze(source: Mapping[str, set[int]]) -> dict[str, frozenset[int]]:
        return {
            access_ref: frozenset(axes)
            for access_ref, axes in source.items()
            if axes
        }

    def padding_for_operand(
        node_ref: str | None,
        operand: str,
    ) -> dict[str, frozenset[int]]:
        merged: dict[str, set[int]] = {}
        edge_key = (node_ref, operand) if node_ref is not None else None
        if edge_key not in explicit_load_edges:
            merge_projected(merged, padding_positions.get(operand, {}))
        else:
            # One explicit node-bound physical load owns this consumer edge.
            # A separate node-less whole-input load may feed other consumers,
            # but its padding provenance must not leak through this reload.
            merge_projected(merged, edge_padding.get(edge_key, {}))
        return freeze(merged)

    def padding_events(
        body: Sequence[IntentOp],
    ) -> tuple[tuple[str, IntentOp], ...]:
        events: list[tuple[str, IntentOp]] = []
        for operation in body:
            if operation.op == "loop":
                events.append(("enter", operation))
                events.extend(padding_events(operation.body))
                events.append(("exit", operation))
            elif operation.op in {"effect_if", "value_if"}:
                assert operation.positive is not None
                assert operation.negative is not None
                events.append(("enter", operation))
                events.extend(padding_events(operation.positive.body))
                events.extend(padding_events(operation.negative.body))
                events.append(("exit", operation))
            else:
                events.append(("ordinary", operation))
        return tuple(events)

    for phase, op in padding_events(j.intent.body):
        path = path_by_marker.get(id(op), "body")
        node_ref = ref_by_path.get(path)
        if phase == "enter":
            if op.op == "loop":
                for scan in op.scan:
                    padding_positions[scan.name] = padding_positions.get(
                        scan.source, {}
                    ).copy()
                for state in op.carry:
                    padding_positions[state.body_argument] = (
                        padding_positions.get(state.init_value, {}).copy()
                        if isinstance(state.init_value, str)
                        else {}
                    )
            continue
        if phase == "exit":
            if op.op == "loop":
                for state in op.carry:
                    padding_positions[state.result_value] = (
                        padding_positions.get(state.next_value, {}).copy()
                    )
            elif op.op == "value_if":
                assert isinstance(op.result, str)
                assert op.positive is not None and op.negative is not None
                merged: dict[str, set[int]] = {}
                for yielded in (
                    op.positive.yield_value,
                    op.negative.yield_value,
                ):
                    merge_projected(
                        merged,
                        padding_positions.get(yielded, {}),
                    )
                padding_positions[op.result] = freeze(merged)
            # effect_if has no SSA result; its branch-local padding facts do
            # not escape the authored regions.  Access/effect consumption is
            # checked independently through their exact nested node refs.
            continue
        operand_padding = {
            operand: padding_for_operand(node_ref, operand)
            for operand in op.operands
        }
        padded_operands = tuple(
            operand for operand in op.operands if operand_padding[operand]
        )
        if not isinstance(op.result, str):
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.op",
                        "functional projection cannot erase masked padding "
                        f"provenance through unmodeled op {op.op!r}",
                        node_ref,
                    )
                )
            continue
        result_type = value_types.get(op.result)
        if result_type is None:
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.type",
                        "functional projection cannot propagate masked padding "
                        "provenance without a closed result type",
                        node_ref,
                    )
                )
            continue
        if op.op in {"map_unary", "cast", "pad"}:
            if len(op.operands) != 1:
                if padded_operands:
                    consumer_issues.append(
                        FunctionalBoundaryClosureIssue(
                            f"j.intent.intent_function.{path}.operands",
                            "functional projection cannot propagate masked "
                            f"padding provenance through malformed {op.op!r}",
                            node_ref,
                        )
                    )
                continue
            padding_positions[op.result] = operand_padding[
                op.operands[0]
            ].copy()
            continue
        if op.op in {"map_binary", "select"}:
            projected: dict[str, set[int]] = {}
            declared = broadcasts.get(op.result, {})
            for operand in op.operands:
                operand_type = value_types.get(operand)
                if operand_type is None:
                    continue
                axes = declared.get(operand)
                positions = (
                    _numpy_broadcast_positions(
                        operand_type.shape,
                        result_type.shape,
                    )
                    if axes is None
                    else _axis_broadcast_positions(
                        operand_type.shape,
                        result_type.shape,
                        axes,
                    )
                )
                if positions is None:
                    if operand_padding[operand]:
                        consumer_issues.append(
                            FunctionalBoundaryClosureIssue(
                                f"j.intent.intent_function.{path}.op",
                                "functional projection cannot propagate masked "
                                "padding provenance through an incomplete "
                                "broadcast relation",
                                node_ref,
                            )
                        )
                    continue
                merge_projected(
                    projected,
                    operand_padding[operand],
                    positions,
                )
            padding_positions[op.result] = freeze(projected)
            continue
        if op.op == "transpose":
            if len(op.operands) != 1:
                if padded_operands:
                    consumer_issues.append(
                        FunctionalBoundaryClosureIssue(
                            f"j.intent.intent_function.{path}.operands",
                            "functional projection cannot propagate masked "
                            "padding provenance through malformed transpose",
                            node_ref,
                        )
                    )
                continue
            raw_perm = op.attrs.get("perm")
            if not isinstance(raw_perm, list) or any(
                isinstance(axis, bool) or not isinstance(axis, int)
                for axis in raw_perm
            ):
                if padded_operands:
                    consumer_issues.append(
                        FunctionalBoundaryClosureIssue(
                            f"j.intent.intent_function.{path}.attrs.perm",
                            "functional projection cannot propagate masked "
                            "padding provenance through an unclosed transpose",
                            node_ref,
                        )
                    )
                continue
            perm = tuple(raw_perm)
            source_padding = operand_padding[op.operands[0]]
            if sorted(perm) == list(range(len(perm))):
                projected: dict[str, set[int]] = {}
                for access_ref, axes in source_padding.items():
                    projected[access_ref] = {
                        perm.index(axis) for axis in axes
                    }
                padding_positions[op.result] = freeze(projected)
            elif any(source_padding.values()):
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.attrs.perm",
                        "functional projection cannot propagate masked padding "
                        "provenance through an unclosed transpose",
                        node_ref,
                    )
                )
            continue
        if op.op == "gather":
            matching_bounds = [
                relation
                for relation in j.boundary.index_bounds
                if (
                    node_ref is not None
                    and relation.node_ref == node_ref
                    and len(op.operands) == 2
                    and relation.data == op.operands[0]
                    and relation.indices == op.operands[1]
                    and relation.axis == op.attrs.get("axis")
                )
            ]
            data_type = (
                value_types.get(op.operands[0]) if len(op.operands) == 2 else None
            )
            index_type = (
                value_types.get(op.operands[1]) if len(op.operands) == 2 else None
            )
            gather_axis = op.attrs.get("axis")
            if (
                len(matching_bounds) != 1
                or data_type is None
                or index_type is None
                or isinstance(gather_axis, bool)
                or not isinstance(gather_axis, int)
                or gather_axis < 0
                or gather_axis >= len(data_type.shape)
                or result_type.shape != index_type.shape
            ):
                if padded_operands:
                    consumer_issues.append(
                        FunctionalBoundaryClosureIssue(
                            f"j.intent.intent_function.{path}.op",
                            "functional projection cannot propagate masked padding "
                            "through an unclosed bounded gather",
                            node_ref,
                        )
                    )
                continue

            projected: dict[str, set[int]] = {}
            data_padding = operand_padding[op.operands[0]]
            for access_ref, positions in data_padding.items():
                # A padding axis other than the gathered axis remains the same
                # logical result axis.  Padding on the gathered axis depends on
                # the actual index values and is deliberately left unsupported.
                if gather_axis in positions or any(
                    axis >= len(result_type.shape)
                    or data_type.shape[axis] != result_type.shape[axis]
                    for axis in positions
                ):
                    consumer_issues.append(
                        FunctionalBoundaryClosureIssue(
                            f"j.intent.intent_function.{path}.op",
                            "functional bounded gather cannot project padding on "
                            "its gathered or shape-changing axis",
                            node_ref,
                        )
                    )
                    continue
                projected.setdefault(access_ref, set()).update(positions)
            merge_projected(projected, operand_padding[op.operands[1]])
            padding_positions[op.result] = freeze(projected)
            continue
        if op.op not in {"reduce", "scan"}:
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.op",
                        "functional projection cannot erase masked padding "
                        f"provenance through unmodeled op {op.op!r}",
                        node_ref,
                    )
                )
            continue
        if len(op.operands) != 1:
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.operands",
                        "functional projection cannot propagate masked padding "
                        "provenance through malformed reduce",
                        node_ref,
                    )
                )
            continue

        operand_type = value_types.get(op.operands[0])
        raw_axes: Any = (
            op.attrs.get("axes")
            if op.op == "reduce"
            else [op.attrs.get("axis")]
        )
        if operand_type is None or not isinstance(raw_axes, list):
            if padded_operands:
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.attrs.axes",
                        "functional projection cannot propagate masked padding "
                            f"provenance through an unclosed {op.op}",
                        node_ref,
                    )
                )
            continue
        rank = len(operand_type.shape)
        reduced: set[int] = set()
        for raw_axis in raw_axes:
            if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
                reduced.clear()
                break
            axis = raw_axis if raw_axis >= 0 else raw_axis + rank
            if axis < 0 or axis >= rank:
                reduced.clear()
                break
            reduced.add(axis)
        source_padding = operand_padding[op.operands[0]]
        if raw_axes and not reduced:
            if any(source_padding.values()):
                consumer_issues.append(
                    FunctionalBoundaryClosureIssue(
                        f"j.intent.intent_function.{path}.attrs.axes",
                        "functional projection cannot propagate masked padding "
                        "provenance through an unclosed reduction",
                        node_ref,
                    )
                )
            continue
        consumed_accesses = {
            access_ref
            for access_ref, positions in source_padding.items()
            if reduced.intersection(positions)
        }
        if consumed_accesses and node_ref is not None:
            affected_consumers.add(node_ref)
        if op.attrs.get("fn") == "mean" and consumed_accesses:
            mean_issues.append(
                FunctionalBoundaryClosureIssue(
                    f"j.intent.intent_function.{path}.attrs.fn",
                    "padded mean cannot prove that the physical denominator "
                    "uses the logical valid count; ReductionBoundary has no "
                    "count/denominator contract",
                    node_ref,
                )
            )

        candidates, ambiguous = _consumer_extension_candidates(
            j, operand=op.operands[0]
        )
        if ambiguous:
            consumer_issues.append(
                FunctionalBoundaryClosureIssue(
                    "j.boundary.consumer_extensions",
                    "node-less same-tensor predicate accesses are ambiguous; "
                    "bind access.node_ref before declaring consumer provenance",
                    ambiguous[0],
                )
            )
        legacy = any(
            contract.node_ref == node_ref
            and contract.masked_contribution is not None
            for contract in j.boundary.reductions
        )
        if (
            consumed_accesses
            and node_ref is not None
            and op.attrs.get("fn") in _CONSUMER_EXTENSION_FNS
            and not legacy
        ):
            requirements.update(
                ConsumerExtensionRequirement(node_ref, 0, access_ref)
                for access_ref in consumed_accesses.intersection(candidates)
            )

        remaining: dict[str, frozenset[int]] = {
            access_ref: frozenset(
                axis for axis in positions if axis not in reduced
            )
            for access_ref, positions in source_padding.items()
        }
        if op.op == "scan":
            # An identity-padded suffix of an inclusive scan contains the
            # final valid prefix, not the identity.  Keep the provenance until
            # an observable mask kills it; a later consumer must not inherit
            # the scan input's outside-domain law.
            padding_positions[op.result] = source_padding.copy()
        elif len(result_type.shape) == rank:
            padding_positions[op.result] = {
                access_ref: axes
                for access_ref, axes in remaining.items()
                if axes
            }
        else:
            padding_positions[op.result] = {
                access_ref: frozenset(
                    axis - sum(item < axis for item in reduced)
                    for axis in axes
                )
                for access_ref, axes in remaining.items()
                if axes
            }

    return (
        tuple(mean_issues),
        tuple(consumer_issues),
        tuple(sorted(requirements)),
        frozenset(affected_consumers),
    )


def functional_reduction_padding_issues(
    j: Any,
) -> tuple[FunctionalBoundaryClosureIssue, ...]:
    """Return the versioned TypeCheck's narrow padded-mean omissions.

    ``typecheck.v14`` consumes this J-only necessary condition.  It reports only
    cases where modeled direct-access and pointwise/broadcast provenance proves
    that a masked axis is eliminated by ``reduce(mean)`` without a logical
    count contract.  Unsupported graph operators deliberately stop this
    TypeCheck propagation; the functional consumer separately rejects masked
    provenance at that unsupported operation instead of silently erasing it.
    """

    mean_issues, _consumer_issues, _requirements, _affected = (
        _reduction_padding_analysis(j)
    )
    return mean_issues


def functional_consumer_extension_requirements(
    j: Any,
) -> tuple[ConsumerExtensionRequirement, ...]:
    """Return only statically proven physical-padding consumer edges."""

    _mean, _consumer, requirements, _affected = _reduction_padding_analysis(j)
    return requirements


def _is_reduction_neutral(fn: str, value: Any) -> bool:
    if fn in {"sum", "mean"}:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == 0.0
        ) or value == "ZERO"
    if fn == "prod":
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == 1.0
        ) or value == "ONE"
    if fn == "max":
        return isinstance(value, str) and value in {"NEG_INF", "-inf"}
    if fn == "min":
        return isinstance(value, str) and value in {"POS_INF", "+inf", "inf"}
    if fn == "any":
        return value is False or value == "false"
    if fn == "all":
        return value is True or value == "true"
    return False


_CONSUMER_EXTENSION_OPS = frozenset(
    {
        "map_unary",
        "map_binary",
        "select",
        "cast",
        "transpose",
        "pad",
        "gather",
    }
)
_CONSUMER_EXTENSION_FNS = frozenset(
    {"sum", "prod", "max", "min", "any", "all"}
)


def _same_closed_scalar(left: Any, right: Any) -> bool:
    aliases = {
        "ZERO": "zero",
        "NEG_INF": "neg_inf",
        "-inf": "neg_inf",
        "POS_INF": "pos_inf",
        "+inf": "pos_inf",
        "inf": "pos_inf",
    }
    if isinstance(left, str) or isinstance(right, str):
        return aliases.get(str(left), str(left)) == aliases.get(
            str(right), str(right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and float(left) == float(right)
    )


def _floating_dtype(dtype: str) -> bool:
    return is_closed_floating_dtype(dtype)


def _is_consumer_identity(fn: str, dtype: str, value: Any) -> bool:
    if fn == "sum":
        return value == "ZERO" or (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == 0.0
        )
    if fn == "prod":
        return value == "ONE" or (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == 1.0
        )
    if fn == "any":
        return value is False or value == "false"
    if fn == "all":
        return value is True or value == "true"
    if not _floating_dtype(dtype):
        return False
    if fn == "max":
        return isinstance(value, str) and value in {"NEG_INF", "-inf"}
    if fn == "min":
        return isinstance(value, str) and value in {"POS_INF", "+inf", "inf"}
    return False


def _operation_dependencies(op: IntentOp) -> tuple[str, ...]:
    dependencies: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value not in dependencies:
            dependencies.append(value)

    for operand in op.operands:
        add(operand)
    if op.op == "loop":
        if op.trip is not None and op.trip.is_dynamic:
            add(op.trip.upper_bound)
        for scan in op.scan:
            add(scan.source)
        for carry in op.carry:
            add(carry.init_value)
            add(carry.next_value)
        for value in op.yields:
            add(value)
    if op.op in {"effect_if", "value_if"}:
        add(op.predicate)
    if op.op == "value_if":
        assert op.positive is not None and op.negative is not None
        add(op.positive.yield_value)
        add(op.negative.yield_value)
    return tuple(dependencies)


def direct_logical_operand_sources(j: Any, op: IntentOp) -> tuple[str, ...]:
    """Resolve one op's direct operands across authored region binders.

    A structured-loop ``scan.name`` is a lexical alias for the complete
    logical tensor named by ``scan.from``.  It is not a new load and it does
    not erase the scanned axis.  Boundary consumers therefore need this small
    J-only binder resolution when deciding whether a load feeds an operation
    directly.  Carry values are intentionally not aliases: after an update
    they no longer denote the original logical input.
    """

    j = elaborate_j(j)
    resolved_by_operation: dict[int, tuple[str, ...]] = {}

    def resolve(value: str, aliases: Mapping[str, str]) -> str:
        current = value
        seen: set[str] = set()
        while current in aliases and current not in seen:
            seen.add(current)
            current = aliases[current]
        return current

    def visit(body: Sequence[IntentOp], aliases: Mapping[str, str]) -> None:
        for operation in body:
            dependencies: list[str] = []
            for dependency in _operation_dependencies(operation):
                source = resolve(dependency, aliases)
                if source not in dependencies:
                    dependencies.append(source)
            resolved_by_operation[id(operation)] = tuple(dependencies)

            nested_aliases = dict(aliases)
            if operation.op == "loop":
                for scan in operation.scan:
                    nested_aliases[scan.name] = resolve(scan.source, aliases)
            for _suffix, region_body in _child_regions(operation):
                visit(region_body, nested_aliases)

    visit(j.intent.body, {})
    return resolved_by_operation.get(id(op), _operation_dependencies(op))


def _consumer_dependency_path(
    j: Any,
    *,
    operand: str,
    access_node_ref: str | None,
    access_tensor: str,
) -> tuple[bool, tuple[IntentOp, ...]]:
    """Find the exact J-only dependency path from an access to an operand."""

    operations = _walk_ops(j.intent.body)
    path_to_ref = {path: ref for ref, path in j.node_ids.items()}
    path_by_marker: dict[int, str] = {}

    def record_paths(body: Sequence[IntentOp], prefix: str = "body") -> None:
        for index, op in enumerate(body):
            path = f"{prefix}[{index}]"
            path_by_marker[id(op)] = path
            for suffix, region_body in _child_regions(op):
                record_paths(region_body, f"{path}.{suffix}")

    record_paths(j.intent.body)
    producers: dict[str, tuple[str | None, IntentOp]] = {}
    for op in operations:
        node_ref = path_to_ref.get(path_by_marker[id(op)])
        for result in _result_names(op):
            producers[result] = (node_ref, op)

    memo: dict[str, tuple[bool, tuple[IntentOp, ...]]] = {}
    active: set[str] = set()

    def path_score(path: tuple[IntentOp, ...]) -> tuple[int, int, tuple[str, ...]]:
        return (
            sum(op.op not in _CONSUMER_EXTENSION_OPS for op in path),
            len(path),
            tuple(op.op for op in path),
        )

    def visit(value: str) -> tuple[bool, tuple[IntentOp, ...]]:
        if access_node_ref is None and value == access_tensor:
            return True, ()
        if value in memo:
            return memo[value]
        if value in active:
            return False, ()
        active.add(value)
        producer = producers.get(value)
        if producer is None:
            result = (False, ())
        else:
            node_ref, operation = producer
            if access_node_ref is not None and node_ref == access_node_ref:
                result = (True, (operation,))
            else:
                candidates: list[tuple[IntentOp, ...]] = []
                for dependency in _operation_dependencies(operation):
                    child_found, child_path = visit(dependency)
                    if child_found:
                        candidates.append((*child_path, operation))
                # One access can reach a pointwise operand through both a
                # direct padded branch and an already-consumed statistic
                # branch.  A provenance witness is one path, not the union of
                # every semantic dependency path.  Prefer the closed path; if
                # every path crosses an unsupported op, retain one such path so
                # validation still fails closed (notably for scan -> reduce).
                result = (
                    (True, min(candidates, key=path_score))
                    if candidates
                    else (False, ())
                )
        active.remove(value)
        memo[value] = result
        return result

    return visit(operand)


def _consumer_extension_candidates(
    j: Any,
    *,
    operand: str,
) -> tuple[dict[str, tuple[Any, tuple[IntentOp, ...]]], tuple[str, ...]]:
    """Resolve predicated-load provenance without a tensor-name fallback."""

    by_tensor: dict[str, list[tuple[Any, tuple[IntentOp, ...]]]] = {}
    for access in j.boundary.accesses:
        if access.kind.value != "load" or access.predicate is None:
            continue
        found, path = _consumer_dependency_path(
            j,
            operand=operand,
            access_node_ref=access.node_ref,
            access_tensor=access.tensor,
        )
        if found:
            by_tensor.setdefault(access.tensor, []).append((access, path))

    selected: dict[str, tuple[Any, tuple[IntentOp, ...]]] = {}
    ambiguous: list[str] = []
    for tensor, candidates in by_tensor.items():
        node_bound = [item for item in candidates if item[0].node_ref is not None]
        if node_bound:
            for access, path in node_bound:
                selected[access.access_id] = (access, path)
            continue
        if len(candidates) != 1:
            ambiguous.append(tensor)
            continue
        access, path = candidates[0]
        selected[access.access_id] = (access, path)
    return selected, tuple(sorted(ambiguous))


def _validate_consumer_path(
    j: Any,
    path_operations: tuple[IntentOp, ...],
    *,
    outside_value: Any,
    outside_dtype: str,
) -> None:
    for operation in path_operations:
        if operation.op not in _CONSUMER_EXTENSION_OPS:
            raise NotImplementedError(
                "consumer-domain provenance crosses unsupported J op "
                f"{operation.op!r}"
            )
        if operation.op == "gather":
            matching_bounds = [
                relation
                for relation in j.boundary.index_bounds
                if (
                    len(operation.operands) == 2
                    and relation.data == operation.operands[0]
                    and relation.indices == operation.operands[1]
                    and relation.axis == operation.attrs.get("axis")
                    and _node_table(j).get(relation.node_ref) is operation
                )
            ]
            if len(matching_bounds) != 1:
                raise ValueError(
                    "gather provenance requires one exact typed index-bounds relation"
                )
            continue
        if operation.op != "pad":
            continue
        if (
            set(operation.attrs) != {"lo", "hi", "value"}
            or len(operation.operands) != 1
            or not isinstance(operation.result, str)
            or operation.type is None
        ):
            raise ValueError(
                "pad provenance requires closed lo/hi/value and one typed operand"
            )
        lo = operation.attrs.get("lo")
        hi = operation.attrs.get("hi")
        result_shape = operation.type.get("shape")
        if (
            not isinstance(lo, list)
            or not isinstance(hi, list)
            or not isinstance(result_shape, list)
            or len(lo) != len(result_shape)
            or len(hi) != len(result_shape)
            or any(
                isinstance(item, bool) or not isinstance(item, (int, str))
                for item in [*lo, *hi]
            )
        ):
            raise ValueError(
                "pad provenance requires closed affine lo/hi for every axis"
            )
        if str(operation.type.get("dtype", "")).casefold() != outside_dtype.casefold():
            raise ValueError("pad result dtype differs from consumer outside dtype")
        if not _same_closed_scalar(operation.attrs.get("value"), outside_value):
            raise ValueError(
                "pad value differs from the declared consumer outside value"
            )


def resolve_consumer_local_tile_axis(
    j: Any,
    consumer_ref: str,
) -> int:
    """Resolve Layer B's physical local axis, never Layer A's tensor axis."""

    j = elaborate_j(j)
    nodes = _node_table(j)
    operation = nodes.get(consumer_ref)
    if operation is None or operation.op not in {"reduce", "scan"}:
        raise ValueError("consumer local axis requires a reduce or scan node")
    if operation.op == "scan":
        if (
            set(operation.attrs) != {"fn", "axis"}
            or len(operation.operands) != 1
            or not isinstance(operation.result, str)
        ):
            raise ValueError(
                "scan outside-domain erasure requires the closed forward "
                "inclusive one-operand contract"
            )
        semantic_axis = operation.attrs.get("axis")
        value_types = j_elaboration(j).value_type_by_name
        operand_type = value_types.get(operation.operands[0])
        result_type = value_types.get(operation.result)
        if (
            isinstance(semantic_axis, bool)
            or not isinstance(semantic_axis, int)
            or semantic_axis < 0
            or operand_type is None
            or semantic_axis >= len(operand_type.shape)
            or result_type != operand_type
        ):
            raise ValueError(
                "scan outside-domain erasure requires an in-range semantic "
                "axis and a shape/dtype-preserving result"
            )

    contracts = [
        contract
        for contract in j.boundary.reductions
        if nodes.get(contract.node_ref) is operation
    ]
    if len(contracts) != 1:
        raise ValueError(
            "reduce/scan consumer requires one local-axis algebra boundary "
            "contract"
        )
    local_axis = contracts[0].tile_axis
    if isinstance(local_axis, bool) or not isinstance(local_axis, int):
        raise ValueError("consumer local tile axis must be a closed integer")
    return local_axis


def _project_consumer_extensions(
    j: Any,
    requirements: Sequence[ConsumerExtensionRequirement],
) -> tuple[ConsumerOutsideProjection, ...]:
    """Consume v4's unique per-consumer outside-domain authority.

    Legacy v1-v3 reductions that carry ``masked_contribution`` retain their
    exact historical behavior.  A v4 relation is never synthesized from a
    fill value, a reducer kind, Source, C, or pi.
    """

    value_types = j_elaboration(j).value_type_by_name
    accesses = {item.access_id: item for item in j.boundary.accesses}
    nodes = _node_table(j)
    canonical_nodes = {
        ref: nodes[ref]
        for ref in j.node_ids
        if ref in nodes
    }
    legacy_masked_reductions = {
        item.node_ref
        for item in j.boundary.reductions
        if item.masked_contribution is not None
    }
    declared = {
        (item.consumer_ref, item.operand_index, item.access_ref): item
        for item in j.boundary.consumer_extensions
    }
    required = {
        (item.consumer_ref, item.operand_index, item.access_ref)
        for item in requirements
    }

    projected: list[ConsumerOutsideProjection] = []
    for extension in j.boundary.consumer_extensions:
        operation = canonical_nodes.get(extension.consumer_ref)
        fn = operation.attrs.get("fn") if operation is not None else None
        if (
            operation is None
            or operation.op not in {"reduce", "scan"}
            or fn not in _CONSUMER_EXTENSION_FNS
        ):
            raise ValueError(
                "consumer extension requires a supported reduce or scan with "
                "a closed two-sided identity"
            )
        if extension.operand_index >= len(operation.operands):
            raise ValueError(
                "consumer operand ordinal is outside the closed operand list"
            )
        local_axis = resolve_consumer_local_tile_axis(
            j, extension.consumer_ref
        )
        if extension.tile_axis != local_axis:
            raise ValueError(
                "consumer extension tile axis differs from the consumer's "
                "closed local axis"
            )
        operand = operation.operands[extension.operand_index]
        operand_type = value_types.get(operand)
        if (
            operand_type is None
            or not operand_type.is_tensor
            or operand_type.dtype is None
            or extension.outside.dtype.casefold() != operand_type.dtype.casefold()
        ):
            raise ValueError(
                "outside dtype must equal the selected consumer operand dtype"
            )
        if not _is_consumer_identity(
            str(fn), extension.outside.dtype, extension.outside.value
        ):
            raise ValueError(
                "outside value is not the declared consumer's two-sided identity"
            )
        access = accesses.get(extension.access_ref)
        if access is None or access.kind.value != "load" or access.predicate is None:
            raise ValueError(
                "consumer extension requires one predicate-bearing load access"
            )
        candidates, ambiguous = _consumer_extension_candidates(j, operand=operand)
        selected = candidates.get(extension.access_ref)
        if selected is None or access.tensor in ambiguous:
            raise ValueError(
                "access provenance does not unambiguously reach this consumer operand"
            )
        _validate_consumer_path(
            j,
            selected[1],
            outside_value=extension.outside.value,
            outside_dtype=extension.outside.dtype,
        )
        if extension.consumer_ref in legacy_masked_reductions:
            raise ValueError(
                "masked_contribution and consumer extension are duplicate authorities"
            )
        projected.append(
            ConsumerOutsideProjection(
                extension.consumer_ref,
                extension.operand_index,
                extension.access_ref,
                extension.tile_axis,
                extension.outside.dtype,
                extension.outside.value,
                extension.identity_law,
            )
        )

    missing = sorted(required - set(declared))
    if missing:
        consumer_ref, operand_index, access_ref = missing[0]
        raise NotImplementedError(
            "predicate-bearing reduce/scan operand requires an explicit "
            "outside-domain identity relation: "
            f"{consumer_ref}:{operand_index}:{access_ref}"
        )
    return tuple(
        sorted(
            projected,
            key=lambda item: (
                item.consumer_ref, item.operand_index, item.access_ref
            ),
        )
    )


def _validate_reduction_padding_closure(
    j: Any,
    *,
    has_physical_padding: bool,
    padding_issues: Sequence[FunctionalBoundaryClosureIssue],
    consumer_issues: Sequence[FunctionalBoundaryClosureIssue],
    affected_consumers: frozenset[str],
) -> None:
    """Consume reduction B facts without constraining the Layer-A graph shape.

    A predicated physical realization may disappear from logical-tensor
    RefEval only through an explicit J-owned per-consumer relation.  The old
    reduction-wide masked contribution remains accepted only for legacy J.
    """

    if has_physical_padding:
        if consumer_issues:
            raise NotImplementedError(consumer_issues[0].detail)
        if padding_issues:
            raise NotImplementedError(padding_issues[0].detail)

    nodes = _node_table(j)
    reductions = tuple(
        op for op in _walk_ops(j.intent.body) if op.op in {"reduce", "scan"}
    )
    contract_by_reduction: dict[int, Any] = {}
    for contract in j.boundary.reductions:
        reduction = nodes.get(contract.node_ref)
        if reduction is None or reduction.op not in {"reduce", "scan"}:
            raise ValueError(
                "Layer-B algebra boundary does not target reduce or scan"
            )
        marker = id(reduction)
        if marker in contract_by_reduction:
            raise ValueError(
                "functional reduction has duplicate boundary contracts"
            )
        contract_by_reduction[marker] = contract

    affected_markers = {
        id(nodes[consumer_ref])
        for consumer_ref in affected_consumers
        if consumer_ref in nodes
    }
    if has_physical_padding and not affected_markers.issubset(
        contract_by_reduction
    ):
        raise NotImplementedError(
            "functional padded reduce/scan requires one local-axis algebra "
            "contract for every affected consumer"
        )

    for reduction in reductions:
        contract = contract_by_reduction.get(id(reduction))
        if contract is None:
            continue
        fn = reduction.attrs.get("fn")
        if not isinstance(fn, str) or fn not in {
            "sum", "mean", "prod", "max", "min", "any", "all"
        }:
            raise NotImplementedError(
                "functional projection has no closed reduction neutral element"
            )
        if contract.identity is not None and not _is_reduction_neutral(
            fn, contract.identity
        ):
            raise NotImplementedError(
                f"functional projection requires neutral {fn!r} identity"
            )
        if has_physical_padding:
            if contract.masked_contribution is not None:
                if not _is_reduction_neutral(fn, contract.masked_contribution):
                    raise NotImplementedError(
                        f"functional projection requires neutral {fn!r} "
                        "masked_contribution"
                    )
        elif (
            contract.masked_contribution is not None
            and not _is_reduction_neutral(fn, contract.masked_contribution)
        ):
            raise NotImplementedError(
                f"functional projection requires neutral {fn!r} "
                "masked_contribution"
            )


def _structured_loop_boundary_refs(
    j: Any,
) -> tuple[frozenset[str], frozenset[str]]:
    """Validate the narrow semantic storage consumed by structured RefEval."""

    loops = tuple(op for op in _walk_ops(j.intent.body) if op.op == "loop")
    mutable = tuple(
        item for item in j.storage if item.mutability.value == "mutable"
    )
    if not loops:
        if mutable:
            raise NotImplementedError(
                "functional boundary does not implement general mutable storage"
            )
        return frozenset(), frozenset()

    state_contracts: dict[str, tuple[IntentOp, Any]] = {}
    for loop in loops:
        if loop.scope_id is None:
            raise ValueError("structured loop requires one semantic loop scope")
        for state in loop.carry:
            state_contracts[state.state_id] = (loop, state)

    consumed_storage: set[str] = set()
    consumed_scopes: set[str] = set()
    for state_id, (loop, state) in state_contracts.items():
        required_members = {
            state.state_id,
            state.body_argument,
            state.next_value,
            state.result_value,
        }
        if isinstance(state.init_value, str) and state.init_value not in {
            "NEG_INF", "POS_INF", "ZERO", "ONE", "zeros", "ones",
            "false", "true",
        }:
            try:
                float(state.init_value)
            except ValueError:
                required_members.add(state.init_value)
        owners = [
            item
            for item in j.storage
            if state_id in item.members
            and required_members.issubset(item.members)
        ]
        if len(owners) != 1:
            raise ValueError(
                "structured loop state requires one complete semantic storage owner"
            )
        owner = owners[0]
        if (
            owner.mutability.value != "mutable"
            or owner.scope_id != loop.scope_id
            or not owner.crosses_iteration
            or not owner.escapes
        ):
            raise ValueError(
                "structured loop carry storage must be mutable, loop-scoped, "
                "cross-iteration, and escaping as the loop result"
            )
        consumed_storage.add(owner.storage_id)
        consumed_scopes.add(loop.scope_id)

    unexplained_mutable = {
        item.storage_id for item in mutable
    } - consumed_storage
    if unexplained_mutable:
        raise NotImplementedError(
            "functional boundary does not implement mutable storage outside "
            "typed structured-loop carries"
        )
    return frozenset(consumed_scopes), frozenset(consumed_storage)


def _project_functional_boundary(
    j: Any,
    *,
    runtime_logical_inputs: Mapping[str, Any] | None,
    runtime_logical_dimensions: Mapping[str, int] | None,
) -> FunctionalBoundaryProjection:
    """Prove that complete Layer B can use J's logical tensor execution."""

    j = elaborate_j(j)
    closure_issues = functional_boundary_closure_issues(j)
    if closure_issues:
        # RefEval/backend projection and TypeCheck share this exact J-only
        # necessary condition.  The detailed path is retained by TypeCheck;
        # consumers preserve their established exception surface.
        raise ValueError(closure_issues[0].detail)
    boundary = j.boundary
    access_by_id = {item.access_id: item for item in boundary.accesses}
    if any(parameter.alias_of is not None for parameter in j.signature.parameters):
        raise NotImplementedError(
            "functional boundary does not implement physical alias ABI"
        )
    if boundary.aliases:
        raise NotImplementedError("functional boundary does not implement aliases")
    broadcast_axes = _broadcast_axes(j)
    access_guard_relations = _project_access_guard_relations(
        j,
        broadcast_axes,
    )
    projected_guarded_store_accesses = {
        item.access_ref
        for item in access_guard_relations
        if item.access_kind == "store"
    }
    projected_guarded_load_accesses = {
        item.access_ref
        for item in access_guard_relations
        if item.access_kind == "load"
    }
    structured_scopes, structured_storage = _structured_loop_boundary_refs(j)

    domains = _full_domains(j) if boundary.accesses else {}
    if runtime_logical_dimensions is not None:
        domains = {
            axis: _resolve_runtime_dimension_bound(
                extent, runtime_logical_dimensions
            )
            for axis, extent in domains.items()
        }
    domain_products = _domain_product_factors(j) if boundary.accesses else {}
    if not boundary.accesses and (boundary.domains or boundary.effects):
        raise ValueError(
            "functional boundary domains/effects require explicit accesses"
        )
    has_predicate_or_fill = any(
        access.predicate is not None or access.fill_value is not None
        for access in boundary.accesses
    )
    if has_predicate_or_fill and boundary.updates:
        raise NotImplementedError(
            "functional mask projection does not implement update padding"
        )

    nodes = _node_table(j)
    effect_if_arm_nodes = _effect_if_arm_node_refs(j)
    effect_if_arm_memberships = _effect_if_arm_memberships(j)
    conditional_effect_stores: list[ConditionalEffectStoreProjection] = []
    direct_unit_dimensions = direct_unit_dimension_refs(
        j.dimension_constraints
    )
    direct_dimension_lowers = direct_dimension_lower_bounds(
        j.dimension_constraints
    )
    tensors = {
        item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
    }
    input_names = {item.name for item in j.intent.inputs}
    output_names = {item.name for item in j.intent.outputs}
    returns = {item.output: item.value for item in j.intent.returns}
    graph_operands = {
        operand
        for operation in _node_table(j).values()
        for operand in _operation_dependencies(operation)
    }
    loads_by_tensor: dict[str, list[Any]] = {}
    for access in boundary.accesses:
        if access.kind.value == "load":
            loads_by_tensor.setdefault(access.tensor, []).append(access)
    mappings = {item.output: item for item in boundary.output_mappings}
    for output, mapping in mappings.items():
        if output in returns and mapping.value != returns[output]:
            raise ValueError("functional output mapping differs from J return")

    effects_by_access: dict[str, list[Any]] = {}
    for effect in boundary.effects:
        if effect.access_ref is None:
            raise NotImplementedError(
                "functional boundary requires access-bound effects"
            )
        effects_by_access.setdefault(effect.access_ref, []).append(effect)
    paired_store_values: dict[str, str] = {}
    for access in boundary.accesses:
        if access.kind.value != "load":
            continue
        pair = _explicit_inout_poststate_reload(j, access.access_id)
        if pair is None:
            continue
        store_access_id, poststate = pair
        prior = paired_store_values.setdefault(store_access_id, poststate)
        if prior != poststate:
            raise ValueError("functional inout store has conflicting reload values")
    access_ids = [item.access_id for item in boundary.accesses]
    if len(access_ids) != len(set(access_ids)):
        raise ValueError("functional boundary has duplicate access ids")
    effect_ids = [item.effect_id for item in boundary.effects]
    if len(effect_ids) != len(set(effect_ids)):
        raise ValueError("functional boundary has duplicate effect ids")

    consumed_domain_axes: set[str] = set()
    inferred_extents: dict[str, int | str] = dict(domains)
    # ``access_axes`` is a public compatibility receipt: retain legacy
    # singleton ``"0"`` sentinels when the old direct-axis profile exists.
    # All semantic predicate/domain reasoning uses only real reference axes.
    access_axes_by_access: dict[str, tuple[str, ...]] = {}
    axis_refs_by_access: dict[str, tuple[str, ...]] = {}
    coordinates_by_access: dict[str, tuple[str | int, ...]] = {}
    predicates: dict[str, PredicateProjection | None] = {}
    loaded: set[str] = set()
    inout_poststate_reloads: set[tuple[str, str, str]] = set()
    predicate_dominated_loads: set[tuple[str, str]] = set()
    stores_by_output: dict[str, list[Any]] = {
        name: [] for name in output_names
    }
    whole_load_tensors: set[str] = set()
    constant_gathers: list[ConstantGatherProjection] = []
    index_bounded_gathers: list[IndexBoundedGatherProjection] = []
    mixed_radix_accesses: list[MixedRadixAccessProjection] = []
    constant_gather_edges: dict[str, dict[tuple[str, int], str]] = {}
    logical_scalar_inputs = _logical_scalar_inputs(j)
    access_guard_input_refs = {
        relation.value_ref
        for relation in boundary.access_guard_relations
        if relation.value_ref in input_names
        and relation.value_ref not in logical_scalar_inputs
    }
    logical_guards = _logical_boolean_guards(j)
    logical_input_types = {
        item.name: item.dtype
        for item in j.intent.inputs
        if item.name in logical_scalar_inputs
    }
    runtime_predicate_proofs: list[RuntimePredicateProofReceipt] = []
    declared_dimensions = {
        dimension
        for tensor in tensors.values()
        for dimension in tensor.shape
        if isinstance(dimension, str) and dimension != "?"
    }
    elaboration = j_elaboration(j)
    parsed_accesses = elaboration.access_by_id
    value_types = elaboration.value_type_by_name
    predicates_are_domain_total = access_predicates_are_logical_domain_total(j)
    expected_effect = {"load": "read", "store": "write"}
    for access in boundary.accesses:
        if access.kind.value == "update":
            raise NotImplementedError(
                "functional boundary does not implement update access"
            )
        op = nodes.get(access.node_ref) if access.node_ref is not None else None
        if access.node_ref is not None and op is None:
            raise ValueError(
                f"functional access {access.access_id!r} has unknown J node"
            )
        tensor = tensors.get(access.tensor)
        if tensor is None:
            raise ValueError(
                f"functional access {access.access_id!r} has unknown tensor"
            )
        coordinates = _functional_access_coordinates(
            j,
            parsed_accesses[access.access_id].index,
            tensor.shape,
        )
        mixed_projection: MixedRadixAccessProjection | None = None
        bounded_gather_projection: IndexBoundedGatherProjection | None = None
        matching_bounds = [
            item
            for item in boundary.index_bounds
            if op is not None
            and op.op == "gather"
            and access.kind.value == "load"
            and item.node_ref == access.node_ref
            and item.data == access.tensor
            and bool(op.operands)
            and op.operands[0] == access.tensor
            and len(op.operands) > 1
            and item.indices == op.operands[1]
        ]
        if len(matching_bounds) == 1:
            bound = matching_bounds[0]
            index_type = value_types.get(bound.indices)
            if index_type is None:
                raise ValueError(
                    "functional bounded gather index type is unavailable"
                )
            consumer_axes: list[str] = []
            for extent in index_type.shape:
                # A literal singleton dimension is only the broadcast carrier of
                # the gather index (for example the trailing ``1`` in a diagonal
                # index shaped ``[x0, 1]``).  It consumes no independent logical
                # domain axis and therefore must not participate in the otherwise
                # unique symbolic-extent matching below.
                if extent == 1:
                    continue
                resolved_extent = _resolve_runtime_dimension_bound(
                    extent, runtime_logical_dimensions
                )
                candidates = [
                    axis
                    for axis, domain_extent in domains.items()
                    if domain_extent == resolved_extent and axis not in consumer_axes
                ]
                if len(candidates) != 1:
                    raise NotImplementedError(
                        "functional bounded gather cannot uniquely bind its "
                        "index shape to logical domain axes"
                    )
                consumer_axes.append(candidates[0])
            authored_axes = _logical_index_axis_refs(
                parsed_accesses[access.access_id].index,
                domains,
            )
            axes = tuple(dict.fromkeys((*authored_axes, *consumer_axes)))
            if coordinates is None:
                coordinates = tuple(bound.indices for _ in tensor.shape)
            bounded_gather_projection = IndexBoundedGatherProjection(
                access.access_id,
                access.node_ref,
                access.tensor,
                bound.axis,
                bound.indices,
                tuple(consumer_axes),
            )
            index_bounded_gathers.append(bounded_gather_projection)
        elif coordinates is None:
            mixed = _mixed_radix_access_projection(
                access.access_id,
                parsed_accesses[access.access_id].index,
                tensor.shape,
                domain_products,
            )
            if mixed is None:
                raise NotImplementedError(
                    "functional dense access requires a direct ordered axis "
                    "vector, a complete product-domain mixed-radix bijection, "
                    "or one index-bounded J gather"
                )
            coordinates, axes, mixed_projection = mixed
            mixed_radix_accesses.append(mixed_projection)
        else:
            axes = tuple(
                coordinate for coordinate in coordinates
                if isinstance(coordinate, str)
            )
        if len(set(axes)) != len(axes):
            raise ValueError("functional dense access axes must be unique")
        if len(coordinates) != len(tensor.shape):
            raise ValueError(
                f"functional access {access.access_id!r} rank differs from tensor"
            )
        legacy_axes = parsed_accesses[access.access_id].index.direct_axes
        access_axes_by_access[access.access_id] = (
            legacy_axes if legacy_axes is not None else axes
        )
        axis_refs_by_access[access.access_id] = axes
        coordinates_by_access[access.access_id] = coordinates
        mixed_tensor_axes = {
            item.tensor_axis
            for item in (
                mixed_projection.digits if mixed_projection is not None else ()
            )
        }
        if mixed_projection is not None:
            consumed_domain_axes.add(mixed_projection.flat_axis)
        if bounded_gather_projection is not None:
            consumed_domain_axes.update(axes)
        for dimension, (coordinate, declared_extent) in enumerate(
            zip(coordinates, tensor.shape, strict=True)
        ):
            if bounded_gather_projection is not None:
                continue
            if dimension in mixed_tensor_axes:
                continue
            runtime_declared_extent = _resolve_runtime_dimension_bound(
                declared_extent, runtime_logical_dimensions
            )
            if isinstance(coordinate, int):
                effective_extent = (
                    1
                    if _is_direct_unit_extent(
                        declared_extent, direct_unit_dimensions
                    )
                    else runtime_declared_extent
                )
                if type(effective_extent) is not int:
                    if not (
                        isinstance(effective_extent, str)
                        and coordinate >= 0
                        and direct_dimension_lowers.get(effective_extent, 0)
                        > coordinate
                    ):
                        raise NotImplementedError(
                            "functional constant access requires a static tensor "
                            "extent or a direct J lower-bound proof"
                        )
                    continue
                if coordinate < 0 or coordinate >= effective_extent:
                    raise ValueError(
                        "functional constant access coordinate is out of bounds"
                    )
                continue
            axis = coordinate
            actual_extent = inferred_extents.get(axis, runtime_declared_extent)
            if actual_extent != runtime_declared_extent:
                raise ValueError(
                    f"functional axis {axis!r} extent {actual_extent!r} differs "
                    f"from J tensor extent {runtime_declared_extent!r}"
                )
            inferred_extents.setdefault(axis, runtime_declared_extent)
            if axis in domains:
                consumed_domain_axes.add(axis)

        constant_gather = (
            _project_constant_gather_load(
                j,
                access=access,
                coordinates=coordinates,
                tensor=tensor,
                operation=op,
                direct_unit_dimensions=direct_unit_dimensions,
                runtime_logical_dimensions=runtime_logical_dimensions,
            )
            if access.kind.value == "load"
            else None
        )

        parsed = parsed_accesses[access.access_id].predicate
        predicate = (
            _j_predicate_projection(
                j, parsed, logical_guards=logical_guards
            )
            if parsed is not None else None
        )
        if predicate is not None:
            predicate = _resolve_runtime_predicate_bounds(
                predicate, runtime_logical_dimensions
            )
        runtime_predicate_proved = False
        if predicate is not None:
            runtime_refs = {
                *predicate.guards,
                *(
                    extent
                    for _axis, extent in predicate.upper_bounds
                    if isinstance(extent, str)
                    and extent in logical_input_types
                ),
            }
            if runtime_refs:
                runtime_proof = _runtime_predicate_tautology(
                    parsed,
                    access_id=access.access_id,
                    access_axes=axes,
                    domains=domains,
                    logical_input_types=logical_input_types,
                    logical_inputs=runtime_logical_inputs,
                    logical_dimensions=runtime_logical_dimensions,
                )
                if runtime_proof is not None:
                    predicate, receipt = runtime_proof
                    predicate = _substitute_value_dimension_bounds(
                        predicate, dict(_value_dimension_pairs(j))
                    )
                    runtime_predicate_proofs.append(receipt)
                    runtime_predicate_proved = True
            if predicate.guards:
                raise NotImplementedError(
                    "functional mask projection does not implement "
                    "guard-bearing access predicates"
                )
            if any(
                not isinstance(extent, int)
                and extent not in declared_dimensions
                for _axis, extent in predicate.upper_bounds
            ):
                raise NotImplementedError(
                    "functional mask projection does not implement "
                    "cross-axis logical access predicates"
                )
            if not predicate.lower_axes.issubset(set(axes)):
                raise ValueError("functional predicate lower bound is not an axis")
            for axis, extent in predicate.upper_bounds:
                if axis not in axes:
                    raise ValueError(
                        "functional predicate upper bound is not an access axis"
                    )
                if domains.get(axis) != extent:
                    raise NotImplementedError(
                        "functional full-tensor mask erasure does not implement "
                        "an authored consumer-local sub-domain predicate"
                    )
            if predicate.upper_bounds:
                missing_domains = sorted(
                    axis for axis in axes if axis != "0" and axis not in domains
                )
                if missing_domains:
                    raise NotImplementedError(
                        "functional mask projection requires a complete logical "
                        "domain for every predicated access axis"
                    )
            if (
                predicate.upper_bounds
                and access.kind.value == "load"
                and access.fill_value is None
                and not runtime_predicate_proved
                and not predicates_are_domain_total
            ):
                if (
                    bounded_gather_projection is None
                    and not predicate_dominated_pointwise_load(
                        j, access.access_id
                    )
                ):
                    raise NotImplementedError(
                        "functional mask projection requires an explicit "
                        "masked-load fill"
                    )
                predicate_dominated_loads.add(
                    (access.access_id, access.predicate)
                )
        elif (
            access.fill_value is not None
            and access.access_id not in projected_guarded_load_accesses
        ):
            raise ValueError("functional unpredicated access must not declare fill")
        predicates[access.access_id] = predicate
        if access.kind.value != "load" and access.fill_value is not None:
            raise ValueError("functional store/update must not declare fill")

        matching_effects = effects_by_access.get(access.access_id, [])
        if len(matching_effects) != 1:
            raise ValueError(
                f"functional access {access.access_id!r} requires one effect"
            )
        effect = matching_effects[0]
        if (
            effect.kind.value != expected_effect[access.kind.value]
            or effect.target != access.tensor
            or effect.node_ref != access.node_ref
        ):
            raise ValueError(
                f"functional access {access.access_id!r} disagrees with its effect"
            )

        if access.kind.value == "load":
            poststate: str | None = None
            if access.tensor not in input_names:
                raise ValueError("functional load is not a J input")
            if op is None:
                poststate = explicit_inout_poststate_reload_value(
                    j, access.access_id
                )
            elif access.tensor not in direct_logical_operand_sources(j, op):
                poststate = explicit_inout_poststate_reload_value(
                    j, access.access_id
                )
                if poststate is None:
                    raise ValueError("functional load is not a direct J operand")
            if poststate is not None:
                inout_poststate_reloads.add(
                    (access.access_id, access.tensor, poststate)
                )
            if op is None:
                has_authored_consumers = any(
                    relation.access_ref == access.access_id
                    for relation in boundary.consumer_extensions
                )
                if (
                    len(loads_by_tensor[access.tensor]) != 1
                    and poststate is None
                    and not has_authored_consumers
                ):
                    raise ValueError(
                        "functional node-less input load must be the sole load "
                        "for its J tensor unless consumer_extensions bind its "
                        "padded reduction consumers"
                    )
                if (
                    access.tensor not in graph_operands
                    and (
                        j.intent.body
                        or access.tensor not in set(returns.values())
                    )
                ):
                    raise ValueError(
                        "functional node-less input load is absent from J "
                        "value dataflow"
                    )
            if constant_gather is not None:
                if predicate is not None:
                    raise NotImplementedError(
                        "functional constant gather load requires no access predicate"
                    )
                # The recognized gather profile has exactly one data operand,
                # ordinal 0.  Keep the edge key explicit so future variadic-op
                # profiles cannot accidentally collapse operand multiplicity.
                edge = (constant_gather.node_ref, 0)
                by_edge = constant_gather_edges.setdefault(access.tensor, {})
                if edge in by_edge:
                    raise ValueError(
                        "functional constant gather has duplicate boundary loads"
                    )
                by_edge[edge] = access.access_id
                constant_gathers.append(constant_gather)
            elif poststate is None:
                whole_load_tensors.add(access.tensor)
            # A validated post-state reload still covers the logical inout
            # tensor at the boundary; it is excluded only from the whole-input
            # load projection, not from input-access coverage.
            loaded.add(access.tensor)
            continue

        if access.tensor not in output_names:
            raise ValueError("functional store does not target a J output")
        final_observable_value = returns[access.tensor]
        mapping = mappings.get(access.tensor)
        dead_exact_overwrite = False
        paired_value = paired_store_values.get(access.access_id)
        observable_value = (
            paired_value if paired_value is not None else final_observable_value
        )
        mapping_matches_store = (
            mapping is None
            or mapping.node_ref is None
            or nodes.get(mapping.node_ref) is op
        )
        is_observable_store = paired_value is None or (
            paired_value == final_observable_value and mapping_matches_store
        )
        if mapping is not None:
            if (
                mapping.node_ref is not None
                and nodes.get(mapping.node_ref) is not op
                and paired_value is None
                and access.access_id not in projected_guarded_store_accesses
            ):
                raise ValueError("functional store disagrees with output mapping")
        if op is not None:
            if (
                observable_value not in _result_names(op)
                and access.access_id not in projected_guarded_store_accesses
            ):
                update = next(
                    (
                        item
                        for item in boundary.updates
                        if item.node_ref == access.node_ref
                    ),
                    None,
                )
                op_results = _result_names(op)
                dead_exact_overwrite = bool(
                    update is not None
                    and op.op == "scatter"
                    and op.operands
                    == [update.base, update.indices, update.values]
                    and op.attrs.get("axis") == update.axis
                    and op.attrs.get("mode", "along_axis") == "along_axis"
                    and op.attrs.get("reduce", "none") == "none"
                    and update.combiner.value == "overwrite"
                    and update.duplicate_policy.value == "forbid"
                    and update.order.value == "input_order"
                    and all(
                        result not in graph_operands
                        and result not in returns.values()
                        for result in op_results
                    )
                )
                carry_results = [
                    result
                    for result in op_results
                    if _store_result_reaches_output_through_enclosing_carry(
                        j,
                        operation=op,
                        producer_value=result,
                        observable_value=observable_value,
                    )
                ]
                if dead_exact_overwrite:
                    pass
                elif len(carry_results) == 1:
                    pass
                elif access.node_ref in effect_if_arm_nodes:
                    produced = _result_names(op)
                    if len(produced) != 1:
                        raise ValueError(
                            "functional conditional store requires one unique "
                            "branch-local producer result"
                        )
                    if (
                        op.type is None
                        or op.type.get("dtype") != tensor.dtype
                        or op.type.get("shape") != tensor.shape
                    ):
                        raise ValueError(
                            "functional conditional store producer type differs "
                            "from output"
                        )
                    conditional_effect_stores.append(
                        ConditionalEffectStoreProjection(
                            access_id=access.access_id,
                            node_ref=access.node_ref,
                            output=access.tensor,
                            value_ref=produced[0],
                        )
                    )
                else:
                    raise ValueError(
                        f"functional store {access.access_id!r} is not produced "
                        "by its J node or one unique enclosing loop carry"
                    )
            if has_predicate_or_fill and (
                op.type is None
                or op.type.get("dtype") != tensor.dtype
                or op.type.get("shape") != tensor.shape
            ):
                raise ValueError("functional store producer type differs from output")
        elif (
            j.intent.body or mapping is None or observable_value not in input_names
        ):
            raise NotImplementedError(
                "node-less store is supported only for direct identity J"
            )
        if is_observable_store and not dead_exact_overwrite:
            stores_by_output[access.tensor].append(access)

    if set(effects_by_access) != set(access_ids):
        raise ValueError("functional boundary has effect for unknown access")
    canonical_output_input_refs = {
        item.value
        for item in j_elaboration(j).output_bindings
        if item.value in input_names
    }
    # An ABI may retain tensor parameters that are semantically unused by J.
    # Requiring physical accesses for those values would make RefEval depend on
    # dead Source interface entries.  Conversely, every tensor that is live in
    # Layer A, directly returned, or named by a J access-guard relation remains
    # mandatory here.
    live_tensor_inputs = (
        input_names
        - set(logical_scalar_inputs)
    ).intersection(
        graph_operands
        | canonical_output_input_refs
        | access_guard_input_refs
    )
    # ``loaded`` also legitimately contains authored rank-0 logical scalars and
    # inout post-state reloads.  Those accesses were validated individually above
    # and must not make coverage of ordinary live tensor inputs fail.  Coverage is
    # therefore one-way: every live tensor input needs a load/guard, while extra
    # already-validated authored loads remain legal.
    if boundary.accesses and not live_tensor_inputs.issubset(
        loaded | access_guard_input_refs
    ):
        raise NotImplementedError(
            "functional boundary must cover every live logical tensor input"
        )
    canonical_nodes = _canonical_nodes(j)
    for tensor, matched_edges in constant_gather_edges.items():
        if tensor in whole_load_tensors:
            continue
        direct_uses = {
            (node_ref, operand_index)
            for node_ref, operation in canonical_nodes.items()
            for operand_index, operand in enumerate(operation.operands)
            if operand == tensor
        }
        if direct_uses != set(matched_edges):
            raise ValueError(
                "functional constant loads do not cover the tensor's direct J uses"
            )

    concat_store_partitions: list[ConcatStorePartitionProjection] = []
    direct_store_partitions: list[DirectStorePartitionProjection] = []
    for output, stores in stores_by_output.items():
        if not boundary.accesses:
            continue
        if len(stores) == 1:
            coordinates = coordinates_by_access[stores[0].access_id]
            tensor = tensors[output]
            if any(
                isinstance(coordinate, int)
                and not (
                    coordinate == 0
                    and _is_direct_unit_extent(
                        extent, direct_unit_dimensions
                    )
                )
                for coordinate, extent in zip(
                    coordinates, tensor.shape, strict=True
                )
            ):
                raise NotImplementedError(
                    "functional boundary requires exactly one full store per "
                    "logical output"
                )
            continue
        if not stores:
            raise NotImplementedError(
                "functional boundary requires exactly one store per logical output"
            )
        if _is_total_conditional_store_pair(
            stores=stores,
            coordinates_by_access=coordinates_by_access,
            predicates=predicates,
            output_shape=tensors[output].shape,
            conditional_accesses=frozenset(
                item.access_id for item in conditional_effect_stores
            ),
            arm_memberships=effect_if_arm_memberships,
            direct_unit_dimensions=direct_unit_dimensions,
        ):
            continue
        observable_value = returns[output]
        partition = _project_store_partition(
            j,
            output=output,
            stores=stores,
            coordinates_by_access=coordinates_by_access,
            predicates=predicates,
            observable_value=observable_value,
            direct_unit_dimensions=direct_unit_dimensions,
        )
        if isinstance(partition, ConcatStorePartitionProjection):
            concat_store_partitions.append(partition)
        else:
            direct_store_partitions.append(partition)
    context_domains = tuple(
        sorted(
            (axis, domains[axis])
            for axis in set(domains) - consumed_domain_axes
        )
    )
    concrete_predicates = [item for item in predicates.values() if item is not None]
    # Predicates are access-local Layer-B facts.  The checks above have already
    # proved that each projected upper bound equals the complete logical domain
    # of that access axis.  A physical padding predicate on one tensor therefore
    # cannot be globalized to every other access that happens to reuse the same
    # logical axis: those accesses may have different physical tiling/capacity
    # while denoting the same dense J domain.  Reduction/scan outside-lane
    # semantics remain checked per access and per consumer below.

    _validate_broadcast_closure(j, broadcast_axes)
    (
        padding_issues,
        consumer_issues,
        extension_requirements,
        affected_consumers,
    ) = _reduction_padding_analysis(j, predicate_overrides=predicates)
    _validate_reduction_padding_closure(
        j,
        has_physical_padding=has_predicate_or_fill,
        padding_issues=padding_issues,
        consumer_issues=consumer_issues,
        affected_consumers=affected_consumers,
    )
    consumer_extensions = _project_consumer_extensions(
        j, extension_requirements
    )

    # Canonical elaboration owns the one complete output-binding rule:
    # IntentFunction.returns is authoritative for every output.  Boundary
    # output_mappings are sparse store/inout anchors and may only repeat a
    # return binding when their value agrees with it.  There is no output-name
    # or identity fallback at this boundary.
    canonical_output_values = {
        item.output: item.value
        for item in j_elaboration(j).output_bindings
    }
    if set(canonical_output_values) != output_names:
        raise ValueError("functional output destination relation is incomplete")
    for item in boundary.index_bounds:
        if nodes.get(item.node_ref) is None:
            raise ValueError("functional index bounds reference unknown node")
    for item in boundary.updates:
        if nodes.get(item.node_ref) is None:
            raise ValueError("functional update references unknown node")
    outputs = {item.name for item in j.intent.outputs}
    if any(item.output not in outputs for item in boundary.result_relations):
        raise ValueError("functional result relation references unknown output")

    proven_bounds = frozenset(
        j.node_ids.get(item.node_ref, item.node_ref)
        for item in boundary.index_bounds
    )
    if runtime_predicate_proofs:
        profile = (
            (
                "logical_domain_runtime_reduction_padding_v2"
                if consumer_extensions
                else "logical_domain_runtime_reduction_padding_v1"
            )
            if boundary.reductions
            else "logical_domain_runtime_mask_erasure_v1"
        )
    else:
        profile = (
            (
                "logical_domain_reduction_padding_v3"
                if consumer_extensions
                else "logical_domain_reduction_padding_v2"
            )
            if concrete_predicates and boundary.reductions
            else (
                "logical_domain_mask_erasure_v1"
                if concrete_predicates
                else (
                    "dense_functional_v1"
                    if boundary.accesses
                    else "pure_value_v1"
                )
            )
        )
    if structured_storage:
        profile = (
            "structured_dense_functional_v1"
            if boundary.accesses
            else "structured_value_v1"
        )
    if context_domains:
        profile = f"context_{profile}"
    value_dimension_pairs = _value_dimension_pairs(j)
    dimension_relation_tuples = _dimension_relation_tuples(j)
    consumed = {
        *(f"domain:{name}" for name in boundary.domains),
        *(f"access:{item.access_id}" for item in boundary.accesses),
        *(f"effect:{item.effect_id}" for item in boundary.effects),
        *(f"output:{item.output}" for item in boundary.output_mappings),
        *(f"reduction:{item.node_ref}" for item in boundary.reductions),
        *(f"broadcast:{item.node_ref}:{item.operand}" for item in boundary.broadcasts),
        *(f"index_bounds:{item.node_ref}" for item in boundary.index_bounds),
        *(f"update:{item.node_ref}" for item in boundary.updates),
        *(f"result_relation:{item.output}" for item in boundary.result_relations),
        *(
            "consumer_extension:"
            f"{item.consumer_ref}:{item.operand_index}:{item.access_ref}"
            for item in boundary.consumer_extensions
        ),
        *(
            "access_guard:"
            f"{item.access_ref}:{item.value_ref}:"
            f"{','.join(str(axis) for axis in item.axes)}:{item.polarity}"
            for item in access_guard_relations
        ),
        *(
            f"value_dimension:{value_ref}:{dimension_ref}:eq"
            for value_ref, dimension_ref in value_dimension_pairs
        ),
        *(
            f"dimension_relation:{result}:{op}:{arg}:{rhs or ''}:"
            f"{value if value is not None else ''}"
            for result, op, arg, rhs, value in dimension_relation_tuples
        ),
        *(f"scope:{item}" for item in structured_scopes),
        *(f"storage:{item}" for item in structured_storage),
        *(
            f"inout_poststate_reload:{access_id}:{tensor}:{poststate}"
            for access_id, tensor, poststate in inout_poststate_reloads
        ),
        *(
            f"predicate_dominated_load:{access_id}:{predicate}"
            for access_id, predicate in predicate_dominated_loads
        ),
        *(
            f"context_domain:{axis}:{extent}"
            for axis, extent in context_domains
        ),
    }
    return FunctionalBoundaryProjection(
        profile=profile,
        domains=tuple(sorted(domains.items())),
        access_axes=tuple(sorted(access_axes_by_access.items())),
        access_coordinates=tuple(sorted(coordinates_by_access.items())),
        predicates=tuple(sorted(predicates.items())),
        broadcast_axes=tuple(
            sorted(
                (
                    result,
                    tuple(sorted(operands.items())),
                )
                for result, operands in broadcast_axes.items()
            )
        ),
        proven_index_bounds=proven_bounds,
        consumed=tuple(sorted(consumed)),
        boundary_digest=_digest(
            {
                "boundary": boundary.to_json(),
                "value_dimension_relations": [
                    {
                        "value_ref": value_ref,
                        "dimension_ref": dimension_ref,
                        "relation": "eq",
                    }
                    for value_ref, dimension_ref in value_dimension_pairs
                ],
                "dimension_relations": [
                    {
                        "result": result,
                        "expr": {
                            "op": op,
                            "arg": arg,
                            **({"rhs": rhs} if rhs is not None else {}),
                            **({"value": value} if value is not None else {}),
                        },
                    }
                    for result, op, arg, rhs, value in dimension_relation_tuples
                ],
            }
        ),
        predicated_access_count=sum(
            item.predicate is not None for item in boundary.accesses
        ),
        erased_fill_count=sum(
            item.fill_value is not None
            and item.access_id not in projected_guarded_load_accesses
            for item in boundary.accesses
        ),
        runtime_predicate_proofs=tuple(runtime_predicate_proofs),
        consumer_extensions=consumer_extensions,
        value_dimension_relations=value_dimension_pairs,
        dimension_relations=dimension_relation_tuples,
        constant_gathers=tuple(constant_gathers),
        concat_store_partitions=tuple(concat_store_partitions),
        direct_store_partitions=tuple(direct_store_partitions),
        access_guard_relations=access_guard_relations,
        mixed_radix_accesses=tuple(mixed_radix_accesses),
        context_domains=context_domains,
        conditional_effect_stores=tuple(conditional_effect_stores),
        index_bounded_gathers=tuple(index_bounded_gathers),
    )


def project_functional_boundary(j: Any) -> FunctionalBoundaryProjection:
    """Project J without invocation values, as production consumers must."""

    return _project_functional_boundary(
        j,
        runtime_logical_inputs=None,
        runtime_logical_dimensions=None,
    )


def complete_conditional_effect_outputs(j: Any) -> frozenset[str]:
    """Return outputs exhaustively defined by one authored ``effect_if``.

    This deliberately recognizes only the closed single-level profile: exactly one
    full-store receipt in each sibling polarity of the same effect-if, with no other
    conditional-store receipt for that output.  Producer/type/full-store validity is
    inherited from the shared functional projection.  Nested or partial coverage is
    not guessed.
    """

    canonical = elaborate_j(j)
    projection = project_functional_boundary(canonical)
    memberships = _effect_if_arm_memberships(canonical)
    by_output: dict[
        str,
        list[tuple[str, str, ConditionalEffectStoreProjection]],
    ] = {}
    for relation in projection.conditional_effect_stores:
        membership = memberships.get(relation.node_ref, ())
        if len(membership) != 1:
            continue
        control_ref, polarity = membership[0]
        by_output.setdefault(relation.output, []).append(
            (control_ref, polarity, relation)
        )

    result: set[str] = set()
    for output, rows in by_output.items():
        controls = {control_ref for control_ref, _polarity, _relation in rows}
        if len(controls) != 1 or len(rows) != 2:
            continue
        polarities = [polarity for _control, polarity, _relation in rows]
        if sorted(polarities) == ["negative", "positive"]:
            result.add(output)
    return frozenset(result)


def project_functional_boundary_for_refeval(
    j: Any,
    logical_input_values: Mapping[str, Any],
    logical_dimensions: Mapping[str, int],
) -> FunctionalBoundaryProjection:
    """Project J for one RefEval invocation using only J-owned input values."""

    j = elaborate_j(j)
    expected = {item.name for item in j.intent.inputs}
    if set(logical_input_values) != expected:
        raise ValueError(
            "RefEval runtime predicate proof requires exactly the J logical inputs"
        )
    expected_dimensions = {
        dimension
        for item in [*j.intent.inputs, *j.intent.outputs]
        for dimension in item.shape
        if isinstance(dimension, str) and dimension != "?"
    }
    expected_dimensions.update(
        parameter.logical_ref
        for parameter in j.signature.parameters
        if isinstance(parameter.logical_ref, str)
        and re.fullmatch(r"x[0-9]+", parameter.logical_ref)
    )
    resolved_dimensions = resolve_dimension_relations(j, logical_dimensions)
    if set(resolved_dimensions) != expected_dimensions:
        raise ValueError(
            "RefEval runtime predicate proof requires exactly the J logical dimensions"
        )
    validate_value_dimension_instantiation(
        j,
        resolved_dimensions,
        logical_input_values,
    )
    validate_dimension_relation_instantiation(j, resolved_dimensions)
    return _project_functional_boundary(
        j,
        runtime_logical_inputs=dict(logical_input_values),
        runtime_logical_dimensions=resolved_dimensions,
    )


__all__ = [
    "FUNCTIONAL_BOUNDARY_PROTOCOL",
    "RUNTIME_PREDICATE_PROOF_PROTOCOL",
    "AccessGuardProjection",
    "ConsumerExtensionRequirement",
    "ConstantGatherProjection",
    "ConcatStorePartitionProjection",
    "ConcatStoreSliceProjection",
    "ConditionalEffectStoreProjection",
    "IndexBoundedGatherProjection",
    "DirectStorePartitionProjection",
    "DirectStoreSliceProjection",
    "FunctionalBoundaryClosureIssue",
    "FunctionalBoundaryProjection",
    "complete_conditional_effect_outputs",
    "ConsumerOutsideProjection",
    "PredicateProjection",
    "RuntimePredicateProofReceipt",
    "functional_boundary_closure_issues",
    "functional_consumer_extension_requirements",
    "functional_reduction_padding_issues",
    "access_predicates_are_logical_domain_total",
    "direct_logical_operand_sources",
    "explicit_inout_poststate_reload_value",
    "predicate_dominated_pointwise_load",
    "parse_direct_axes",
    "project_predicate_text",
    "resolve_consumer_local_tile_axis",
    "static_index_bounds_status",
    "project_functional_boundary",
    "project_functional_boundary_for_refeval",
]
