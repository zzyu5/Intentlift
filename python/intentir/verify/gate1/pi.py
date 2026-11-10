"""Typed canonical normalization witness for the current J workflow.

Fresh Lift candidates author direct J/C-bound normalization relations only.
The canonical Source/Harness construction record owns replay topology and
execution realization; pi contains neither a graph nor a target recipe.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from intentir.ir.canonical_construction import (
    CANONICAL_PI_REQUIRED_FIELDS,
    CANONICAL_PI_WIRE_FIELDS,
    canonical_archive_records,
)
from intentir.ir.canonical_j import j_elaboration
from intentir.ir.dimension_relation import declared_dimension_handles
from intentir.ir.intent_op_contract import is_scalar_literal_or_value_reference


class RealizationWitnessError(ValueError):
    """Raised when pi contains malformed or semantic-bearing data."""


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

NORMALIZATION_WITNESS_SCHEMA = "intentir.pi.normalization-witness.v1"
DEFAULT_ZERO_LOWER_BOUND_CLASS = "default_zero_lower_bound"
# Class-level Gate1 quotient for a closed producer/identity mapping.
CLOSED_CONSUMER_IDENTITY_CLASS = "closed_consumer_identity"
# A logical rowwise ``keepdims`` result can be a scalar within one target
# program when its retained row is directly owned by that program.  The class
# preserves the J rank-2 relation; it only selects the carrier shape used by a
# source-blind backend implementation.
PER_PROGRAM_SCALAR_KEEPDIMS_CLASS = "per_program_scalar_keepdims"
NORMALIZATION_CLASS_SCOPES: dict[str, str] = {
    DEFAULT_ZERO_LOWER_BOUND_CLASS: "all_predicated_accesses",
    CLOSED_CONSUMER_IDENTITY_CLASS: "closed_consumer_extensions",
    PER_PROGRAM_SCALAR_KEEPDIMS_CLASS: "all_rowwise_keepdims_reductions",
}

PROGRAM_BINDING_RELATIONS: tuple[str, ...] = (
    "direct",
    "product_linearized",
    "guarded_union",
)

ITERATION_BINDING_RELATIONS: tuple[str, ...] = (
    "tiled_domain",
    "tiled_recurrence",
)

CONTROL_BINDING_RELATIONS: tuple[str, ...] = (
    "terminal_domain_guard",
    "terminal_effect_guard",
)

MATERIALIZATION_BINDING_RELATIONS: tuple[str, ...] = (
    "scratch_roundtrip",
    "recompute",
)

ACCESS_REPRESENTATION_BINDING_RELATIONS: tuple[str, ...] = (
    "address_recomputed",
    "boolean_storage_byte",
    "guard_lower_bound_elided",
    "pointer_advanced",
)


def _boolean_terms(expression: Any) -> tuple[Any, ...]:
    """Flatten only explicit J conjunctions for a bounded domain proof."""

    if getattr(expression, "op", None) != "and":
        return (expression,)
    terms: list[Any] = []
    for argument in getattr(expression, "args", ()):
        terms.extend(_boolean_terms(argument))
    return tuple(terms)


def zero_lower_bound_coordinates_from_j(
    j: Any,
    access_ref: str,
) -> frozenset[str]:
    """Return access coordinates covered by an explicit dense zero-based J domain.

    This is a J-only certificate.  It deliberately does not inspect Source,
    TTIR, Replay, C, or concrete runtime values.  A coordinate is eligible only
    when the same canonical J domain contains both an exact zero lower bound and
    a finite upper-bound comparison, and the coordinate is used by this access's
    index expression.  Therefore an access predicate may carry only the upper
    mask while the domain still proves its lower bound.
    """

    try:
        elaboration = j_elaboration(j)
        access = elaboration.access_by_id.get(access_ref)
        if access is None or access.predicate is None:
            return frozenset()
        index_refs = {
            reference
            for component in access.index.components
            for reference in component.references
        }
        declared_values = set(elaboration.value_type_by_name)
        declared_dimensions = set(declared_dimension_handles(j))
        domain_refs: set[str] = set()
        for _domain_ref, expression in elaboration.domains:
            domain_refs.update(expression.references)
        domain_coordinates = domain_refs - declared_values - declared_dimensions
        eligible = index_refs & domain_coordinates
        result: set[str] = set()
        for _domain_ref, expression in elaboration.domains:
            lower: set[str] = set()
            upper: set[str] = set()
            for term in _boolean_terms(expression):
                if len(getattr(term, "args", ())) != 2:
                    continue
                left, right = term.args
                if (
                    term.op == "ge"
                    and getattr(left, "op", None) == "ref"
                    and getattr(right, "op", None) == "int"
                    and right.value == 0
                ):
                    lower.add(left.value)
                elif (
                    term.op == "le"
                    and getattr(left, "op", None) == "int"
                    and left.value == 0
                    and getattr(right, "op", None) == "ref"
                ):
                    lower.add(right.value)
                elif (
                    term.op in {"lt", "le"}
                    and getattr(left, "op", None) == "ref"
                ):
                    upper.add(left.value)
                elif (
                    term.op in {"gt", "ge"}
                    and getattr(right, "op", None) == "ref"
                ):
                    upper.add(right.value)
            result.update(eligible & lower & upper)
        return frozenset(result)
    except Exception:
        # This helper is an authorization certificate, never a repair path.
        return frozenset()


def default_zero_lower_bound_accesses(
    j: Any,
    pi: Any,
) -> tuple[str, ...]:
    """Return the fully authorized access class for zero-bound normalization.

    The class is active when π declares one closed class row with the fixed
    ``all_predicated_accesses`` scope.  The older per-access
    ``guard_lower_bound_elided`` rows remain accepted as a compatibility form,
    but they are not required when the class row is present.  In both forms,
    J independently proves a dense zero-based coordinate for every predicated
    access.  A partial or ambiguous class returns empty and remains fail-closed.
    """

    try:
        accesses = tuple(getattr(j.boundary, "accesses", ()))
        predicated = {
            access.access_id
            for access in accesses
            if access.predicate is not None
        }
        raw_classes = getattr(pi, "normalization_classes", None)
        if raw_classes is None and isinstance(pi, Mapping):
            raw_classes = pi.get("normalization_classes")
        if isinstance(raw_classes, list):
            for item in raw_classes:
                if not isinstance(item, Mapping):
                    continue
                if item.get("name") != DEFAULT_ZERO_LOWER_BOUND_CLASS:
                    continue
                if (
                    item.get("scope")
                    != NORMALIZATION_CLASS_SCOPES[DEFAULT_ZERO_LOWER_BOUND_CLASS]
                ):
                    return ()
                if not predicated:
                    return ()
                if any(
                    not zero_lower_bound_coordinates_from_j(j, access_ref)
                    for access_ref in predicated
                ):
                    return ()
                return tuple(sorted(predicated))
        bound = {
            str(item["access_ref"])
            for item in getattr(pi, "access_representation_bindings", ())
            if item.get("relation") == "guard_lower_bound_elided"
        }
        if not predicated or bound != predicated:
            return ()
        if any(
            not zero_lower_bound_coordinates_from_j(j, access_ref)
            for access_ref in bound
        ):
            return ()
        return tuple(sorted(bound))
    except Exception:
        return ()


def closed_consumer_identity_extensions(
    j: Any,
    pi: Any,
) -> tuple[tuple[str, str, str], ...]:
    """Return the J consumer edges covered by a closed identity class.

    This is deliberately a J/π-only certificate.  It authorizes only a fixed
    value-preserving quotient for a compiler-materialized closed consumer
    identity (for example a ``select/where`` at the exp boundary).  It does
    not authorize changing a reduction, predicate, memory effect, output, or
    arbitrary dataflow edge.

    The supported producer mappings are intentionally small:

    * an unchanged outside scalar (possibly through a cast/identity map); and
    * ``-inf -> 0`` through ``sub`` by a max result followed by ``exp``.

    The latter is the closed softmax mapping ``(-inf - finite_or_-inf)`` then
    ``exp``.  Any other chain remains hard and must be fixed by Replay/Lift.
    """

    def declared(value: Any, name: str) -> bool:
        rows = getattr(value, "normalization_classes", None)
        if rows is None and isinstance(value, Mapping):
            rows = value.get("normalization_classes")
        if not isinstance(rows, list):
            return False
        return any(
            isinstance(row, Mapping)
            and row.get("name") == name
            and row.get("scope") == NORMALIZATION_CLASS_SCOPES[name]
            for row in rows
        )

    if not declared(pi, CLOSED_CONSUMER_IDENTITY_CLASS):
        return ()

    try:
        boundary = j.boundary
        accesses = {item.access_id: item for item in boundary.accesses}
        operation_records = j_elaboration(j).operation_records
        by_result = {
            result: operation
            for operation in operation_records
            for result in operation.get("results", ())
        }
        by_id = {
            operation["id"]: operation
            for operation in operation_records
        }

        def path_from_tensor(
            value: Any,
            tensor: str,
            active: frozenset[str] = frozenset(),
        ) -> tuple[Mapping[str, Any], ...] | None:
            if value == tensor:
                return ()
            if not isinstance(value, str) or value in active:
                return None
            operation = by_result.get(value)
            if operation is None or not isinstance(
                operation.get("operands"), (list, tuple)
            ):
                return None
            for operand in operation["operands"]:
                path = path_from_tensor(operand, tensor, active | {value})
                if path is not None:
                    return (*path, operation)
            return None

        def value_comes_from_max(
            value: Any,
            active: frozenset[str] = frozenset(),
        ) -> bool:
            """Recognize only max plus shape-preserving J provenance."""

            if not isinstance(value, str) or value in active:
                return False
            operation = by_result.get(value)
            if operation is None:
                return False
            if (
                operation.get("op") == "reduce"
                and operation.get("attrs", {}).get("fn") == "max"
            ):
                return True
            if operation.get("op") == "reshape":
                operands = operation.get("operands")
                return (
                    isinstance(operands, (list, tuple))
                    and len(operands) == 1
                    and value_comes_from_max(
                        operands[0], active | {value}
                    )
                )
            if operation.get("op") == "map_unary" and operation.get(
                "attrs", {}
            ).get("fn") in {"cast", "identity"}:
                operands = operation.get("operands")
                return (
                    isinstance(operands, (list, tuple))
                    and len(operands) == 1
                    and value_comes_from_max(
                        operands[0], active | {value}
                    )
                )
            return False

        covered: list[tuple[str, str, str]] = []
        for extension in boundary.consumer_extensions:
            access = accesses.get(extension.access_ref)
            consumer = by_id.get(extension.consumer_ref)
            if access is None or consumer is None:
                return ()
            operands = consumer.get("operands")
            if not isinstance(operands, (list, tuple)) or extension.operand_index >= len(operands):
                return ()
            path = path_from_tensor(
                operands[extension.operand_index],
                access.tensor,
            )
            if path is None:
                return ()

            fill = access.fill_value
            outside = extension.outside.value
            valid = False
            if fill == outside:
                valid = all(
                    (
                        item.get("op") == "reshape"
                        or (
                            item.get("op") == "map_unary"
                            and item.get("attrs", {}).get("fn")
                            in {"cast", "identity"}
                        )
                    )
                    for item in path
                )
            elif fill == "-inf" and outside == 0:
                unary = [
                    item.get("attrs", {}).get("fn")
                    for item in path
                    if item.get("op") == "map_unary"
                ]
                binary = [
                    item
                    for item in path
                    if item.get("op") == "map_binary"
                ]
                sub_ops = [
                    item
                    for item in binary
                    if item.get("attrs", {}).get("fn") == "sub"
                ]
                has_max_operand = any(
                    value_comes_from_max(operand)
                    for item in sub_ops
                    for operand in item.get("operands", [])
                )
                valid = (
                    unary.count("exp") == 1
                    and all(
                        fn in {"exp", "cast", "identity"} for fn in unary
                    )
                    and len(sub_ops) == 1
                    and len(binary) == 1
                    and all(
                        item.get("op") in {"reshape", "map_unary", "map_binary", "reduce"}
                        and (
                            item.get("op") != "reduce"
                            or item.get("attrs", {}).get("fn") == "max"
                        )
                        for item in path
                    )
                    and has_max_operand
                )
            if not valid:
                return ()
            covered.append(
                (extension.consumer_ref, extension.access_ref, extension.outside.dtype)
            )
        return tuple(covered)
    except Exception:
        return ()


def per_program_scalar_keepdims_reductions(
    j: Any,
    pi: Any,
) -> tuple[str, ...]:
    """Return J reduce nodes safely carried as one scalar per target program.

    This is a class certificate, not a target recipe.  It applies only when
    every authored ``keepdims`` reduction is the closed rank-2 rowwise form
    ``[row, reduction] -> [row, 1]`` and the retained row dimension has one
    direct pi program-domain binding.  J still owns the kept singleton axis,
    following broadcast, and any reshape.  The class merely says that a target
    program responsible for one row may hold that logical ``[row,1]`` value as
    a scalar; a later vector consumer must perform the ordinary target
    broadcast.  Any mixed, tiled, multi-axis, or ambiguous form stays hard.
    """

    def declared(value: Any) -> bool:
        rows = getattr(value, "normalization_classes", None)
        if rows is None and isinstance(value, Mapping):
            rows = value.get("normalization_classes")
        return isinstance(rows, list) and any(
            isinstance(row, Mapping)
            and row.get("name") == PER_PROGRAM_SCALAR_KEEPDIMS_CLASS
            and row.get("scope")
            == NORMALIZATION_CLASS_SCOPES[PER_PROGRAM_SCALAR_KEEPDIMS_CLASS]
            for row in rows
        )

    if not declared(pi):
        return ()
    try:
        elaboration = j_elaboration(j)
        value_shapes = {
            name: tuple(value_type.shape)
            for name, value_type in elaboration.value_type_by_name.items()
        }
        # The certificate must see the same complete lexical operation table
        # as canonical J, Replay, and Gate1.  Reading only the root body would
        # omit reductions nested in loop/conditional regions.
        reductions = [
            record
            for record in elaboration.operation_records
            if record.get("op") == "reduce"
            and isinstance(record.get("attrs"), Mapping)
            and record["attrs"].get("keepdims") is True
            and len(record.get("results", ())) == 1
        ]
        if not reductions:
            return ()

        projection = canonical_program_coordinate_projection(pi)
        raw_pi = pi.to_json() if callable(getattr(pi, "to_json", None)) else pi
        if not isinstance(raw_pi, Mapping):
            return ()
        direct_domains = {
            str(row.get("domain_ref"))
            for row in raw_pi.get("program_axes", [])
            if isinstance(row, Mapping) and isinstance(row.get("domain_ref"), str)
        }
        for row in raw_pi.get("program_bindings", []):
            if not isinstance(row, Mapping) or row.get("relation") != "direct":
                continue
            logical_axes = row.get("logical_axes")
            if not isinstance(logical_axes, list) or len(logical_axes) != 1:
                continue
            logical_axis = logical_axes[0]
            if isinstance(logical_axis, Mapping) and isinstance(
                logical_axis.get("domain_ref"), str
            ):
                direct_domains.add(str(logical_axis["domain_ref"]))
        if not direct_domains or not projection.has_mapping:
            return ()

        domains_by_dimension: dict[str, set[str]] = {}
        for domain_ref, expression in elaboration.domains:
            for reference in expression.references:
                domains_by_dimension.setdefault(reference, set()).add(domain_ref)

        certified: list[str] = []
        for operation in reductions:
            node_ref = operation.get("id")
            operands = operation.get("operands")
            result = operation.get("results", ())[0]
            axes = operation.get("attrs", {}).get("axes")
            if (
                not isinstance(node_ref, str)
                or not isinstance(operands, (list, tuple))
                or len(operands) != 1
                or not isinstance(operands[0], str)
                or not isinstance(result, str)
                or axes != [1]
            ):
                return ()
            input_shape = value_shapes.get(operands[0])
            result_shape = value_shapes.get(result)
            if result_shape is None:
                return ()
            if (
                input_shape is None
                or len(input_shape) != 2
                or len(result_shape) != 2
                or result_shape != (input_shape[0], 1)
                or not isinstance(input_shape[0], str)
            ):
                return ()
            domains = domains_by_dimension.get(input_shape[0], set()) & direct_domains
            if len(domains) != 1:
                return ()
            certified.append(node_ref)
        return tuple(certified)
    except Exception:
        # π certificates are authorization gates.  Any malformed J/pi surface
        # is deliberately unproved rather than repaired or guessed here.
        return ()



@dataclass(frozen=True)
class ProgramCoordinateRow:
    """One canonical physical-axis to J-domain projection row.

    ``program_axes`` is the compact direct form and therefore carries no
    logical tensor-axis selector; its ``axis_ref`` entries are represented as
    ``None`` here.  ``program_bindings`` carries the authored logical axis
    selectors and preserves them verbatim.  Keeping both forms in this one
    backend-neutral row type prevents consumers from silently treating the
    relation form as if it were the direct form.
    """

    family: str
    physical_axis: int
    relation: str
    logical_axes: tuple[tuple[str, int | str | None], ...]


@dataclass(frozen=True)
class ProgramCoordinateProjection:
    """Canonical semantic view of either pi program-mapping family."""

    family: str | None
    rows: tuple[ProgramCoordinateRow, ...]
    domain_to_axis: tuple[tuple[str, int], ...]
    physical_axes: tuple[int, ...]

    @property
    def has_mapping(self) -> bool:
        return bool(self.rows)

    def axis_for_domain(self, domain_ref: str) -> int | None:
        for candidate, axis in self.domain_to_axis:
            if candidate == domain_ref:
                return axis
        return None


class ProgramCoordinateProjectionError(ValueError):
    """Raised when a raw pi program mapping cannot be projected safely."""


def canonical_program_coordinate_projection(
    value: Any,
) -> ProgramCoordinateProjection:
    """Project either pi program-mapping family into one canonical view.

    ``value`` may be a raw pi mapping or a typed ``RealizationWitness``.  The
    helper intentionally performs no inference: it only reads the explicitly
    authored ``program_axes`` or ``program_bindings`` rows, rejects a malformed
    or mixed mapping, and returns the same domain-to-physical-axis relation to
    every downstream consumer.
    """

    if not isinstance(value, Mapping):
        to_json = getattr(value, "to_json", None)
        if not callable(to_json):
            raise ProgramCoordinateProjectionError(
                "pi program mapping must be a mapping or typed witness"
            )
        value = to_json()
    if not isinstance(value, Mapping):
        raise ProgramCoordinateProjectionError(
            "typed pi witness did not produce a mapping"
        )

    raw_axes = value.get("program_axes", [])
    raw_bindings = value.get("program_bindings", [])
    if raw_axes is None:
        raw_axes = []
    if raw_bindings is None:
        raw_bindings = []
    if not isinstance(raw_axes, list) or not isinstance(raw_bindings, list):
        raise ProgramCoordinateProjectionError(
            "pi.program_axes and pi.program_bindings must be arrays"
        )
    if raw_axes and raw_bindings:
        raise ProgramCoordinateProjectionError(
            "pi.program_axes and pi.program_bindings are mutually exclusive"
        )

    rows: list[ProgramCoordinateRow] = []
    domain_to_axis: dict[str, int] = {}
    claimed_logical_axes: set[tuple[str, int | str | None]] = set()
    claimed_physical_axes: set[int] = set()

    def add_domain(domain_ref: Any, physical_axis: Any, *, path: str) -> str:
        if (
            not isinstance(domain_ref, str)
            or re.fullmatch(r"d[0-9]+", domain_ref) is None
        ):
            raise ProgramCoordinateProjectionError(
                f"{path}.domain_ref must be a J domain d* handle"
            )
        if (
            isinstance(physical_axis, bool)
            or not isinstance(physical_axis, int)
            or not 0 <= physical_axis <= 2
        ):
            raise ProgramCoordinateProjectionError(
                f"{path}.axis must be an integer in [0, 2]"
            )
        previous = domain_to_axis.get(domain_ref)
        if previous is not None and previous != physical_axis:
            raise ProgramCoordinateProjectionError(
                f"{path}.domain_ref maps to multiple physical launch axes"
            )
        domain_to_axis[domain_ref] = physical_axis
        return domain_ref

    if raw_axes:
        family = "program_axes"
        for index, raw in enumerate(raw_axes):
            path = f"pi.program_axes[{index}]"
            if not isinstance(raw, Mapping):
                raise ProgramCoordinateProjectionError(f"{path} must be an object")
            domain_ref = add_domain(raw.get("domain_ref"), raw.get("axis"), path=path)
            physical_axis = int(raw["axis"])
            claimed_physical_axes.add(physical_axis)
            logical_key = (domain_ref, None)
            if logical_key in claimed_logical_axes:
                raise ProgramCoordinateProjectionError(
                    f"{path}.domain_ref is declared more than once"
                )
            claimed_logical_axes.add(logical_key)
            rows.append(
                ProgramCoordinateRow(
                    family=family,
                    physical_axis=physical_axis,
                    relation="direct",
                    logical_axes=(logical_key,),
                )
            )
    else:
        family = "program_bindings" if raw_bindings else None
        for index, raw in enumerate(raw_bindings):
            path = f"pi.program_bindings[{index}]"
            if not isinstance(raw, Mapping):
                raise ProgramCoordinateProjectionError(f"{path} must be an object")
            physical_axis = raw.get("axis")
            if (
                isinstance(physical_axis, bool)
                or not isinstance(physical_axis, int)
                or not 0 <= physical_axis <= 2
            ):
                raise ProgramCoordinateProjectionError(
                    f"{path}.axis must be an integer in [0, 2]"
                )
            if physical_axis in claimed_physical_axes:
                raise ProgramCoordinateProjectionError(
                    f"{path}.axis is declared more than once"
                )
            claimed_physical_axes.add(physical_axis)
            relation = raw.get("relation")
            if relation not in PROGRAM_BINDING_RELATIONS:
                raise ProgramCoordinateProjectionError(
                    f"{path}.relation is not a closed program binding relation"
                )
            logical_values = raw.get("logical_axes")
            if not isinstance(logical_values, list) or not logical_values:
                raise ProgramCoordinateProjectionError(
                    f"{path}.logical_axes must be a non-empty array"
                )
            if relation == "direct" and len(logical_values) != 1:
                raise ProgramCoordinateProjectionError(
                    f"{path}.logical_axes must contain exactly one axis for direct"
                )
            if relation != "direct" and len(logical_values) < 2:
                raise ProgramCoordinateProjectionError(
                    f"{path}.logical_axes must contain at least two axes for {relation}"
                )
            logical_axes: list[tuple[str, int | str]] = []
            for logical_index, logical in enumerate(logical_values):
                logical_path = f"{path}.logical_axes[{logical_index}]"
                if not isinstance(logical, Mapping):
                    raise ProgramCoordinateProjectionError(
                        f"{logical_path} must be an object"
                    )
                domain_ref = add_domain(
                    logical.get("domain_ref"),
                    physical_axis,
                    path=logical_path,
                )
                axis_ref = logical.get("axis_ref")
                if isinstance(axis_ref, bool):
                    raise ProgramCoordinateProjectionError(
                        f"{logical_path}.axis_ref must be a non-negative axis"
                    )
                if isinstance(axis_ref, int):
                    if axis_ref < 0:
                        raise ProgramCoordinateProjectionError(
                            f"{logical_path}.axis_ref must be non-negative"
                        )
                elif not (
                    isinstance(axis_ref, str)
                    and re.fullmatch(r"x[0-9]+", axis_ref) is not None
                ):
                    raise ProgramCoordinateProjectionError(
                        f"{logical_path}.axis_ref must be an integer or x*"
                    )
                logical_key = (domain_ref, axis_ref)
                if logical_key in claimed_logical_axes:
                    raise ProgramCoordinateProjectionError(
                        f"{logical_path} claims a logical J axis more than once"
                    )
                claimed_logical_axes.add(logical_key)
                logical_axes.append(logical_key)
            rows.append(
                ProgramCoordinateRow(
                    family=family or "program_bindings",
                    physical_axis=physical_axis,
                    relation=relation,
                    logical_axes=tuple(logical_axes),
                )
            )

    rows.sort(
        key=lambda row: (
            row.physical_axis,
            row.family,
            row.relation,
            tuple((domain, str(axis)) for domain, axis in row.logical_axes),
        )
    )
    return ProgramCoordinateProjection(
        family=family,
        rows=tuple(rows),
        domain_to_axis=tuple(sorted(domain_to_axis.items())),
        physical_axes=tuple(sorted(claimed_physical_axes)),
    )


def _record(
    value: Any,
    *,
    path: str,
    required: frozenset[str],
    allowed: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RealizationWitnessError(f"{path} must be an object")
    allowed = required if allowed is None else allowed
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise RealizationWitnessError(f"{path} has forbidden fields: {unknown}")
    if missing:
        raise RealizationWitnessError(f"{path} is missing fields: {missing}")
    return value


def _wire_object(value: Any, path: str) -> dict[str, Any]:
    """Read one JSON object without importing the Gate1 model layer.

    The public normalization validator is intentionally owned by this module.
    It therefore accepts either the persisted mapping or a typed object with a
    ``to_json`` method, but it must not depend on ``model.py`` to discover the
    declaration namespaces it is validating.
    """

    if not isinstance(value, Mapping):
        to_json = getattr(value, "to_json", None)
        if not callable(to_json):
            raise RealizationWitnessError(f"{path} must be an object")
        try:
            value = to_json()
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise RealizationWitnessError(
                f"{path} could not be converted to a JSON object: {exc}"
            ) from exc
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RealizationWitnessError(f"{path} must be an object")
    return dict(value)


def _wire_array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise RealizationWitnessError(f"{path} must be an array")
    return value


def _wire_handle(
    value: Any,
    path: str,
    pattern: str,
    description: str,
) -> str:
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise RealizationWitnessError(f"{path} must be {description}")
    return value


def _wire_handle_set(
    values: Any,
    *,
    path: str,
    pattern: str,
    fields: tuple[str, ...] = ("name", "id"),
) -> set[str]:
    result: set[str] = set()
    for index, value in enumerate(_wire_array(values, path)):
        item = _wire_object(value, f"{path}[{index}]")
        handle = next(
            (item.get(field) for field in fields if item.get(field) is not None),
            None,
        )
        if not isinstance(handle, str) or re.fullmatch(pattern, handle) is None:
            raise RealizationWitnessError(
                f"{path}[{index}] must declare a handle matching {pattern}"
            )
        if handle in result:
            raise RealizationWitnessError(f"{path} contains duplicate handle {handle!r}")
        result.add(handle)
    return result


def _wire_operation_results(value: Any, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (
            _wire_handle(value, path, r"v[0-9]+", "an SSA v* result handle"),
        )
    if isinstance(value, list):
        return tuple(
            _wire_handle(
                item,
                f"{path}[{index}]",
                r"v[0-9]+",
                "an SSA v* result handle",
            )
            for index, item in enumerate(value)
        )
    raise RealizationWitnessError(f"{path} must be a string, string array, or null")


def _wire_declared_j_names(j: Any) -> dict[str, set[str] | dict[str, Any]]:
    """Collect only J-owned namespaces needed by normalization bindings."""

    data = _wire_object(j, "j")
    boundary = _wire_object(data.get("boundary"), "j.boundary")
    domains_object = _wire_object(boundary.get("domains"), "j.boundary.domains")
    domains = set(domains_object)
    if any(re.fullmatch(r"d[0-9]+", item) is None for item in domains):
        raise RealizationWitnessError(
            "j.boundary.domains must contain only declaration-order d* handles"
        )

    access_refs: set[str] = set()
    for index, raw in enumerate(
        _wire_array(boundary.get("accesses"), "j.boundary.accesses")
    ):
        item = _wire_object(raw, f"j.boundary.accesses[{index}]")
        access_ref = item.get("access_id")
        if not isinstance(access_ref, str) or re.fullmatch(r"a[0-9]+", access_ref) is None:
            raise RealizationWitnessError(
                f"j.boundary.accesses[{index}].access_id must be an a* handle"
            )
        if access_ref in access_refs:
            raise RealizationWitnessError("J access handles must be unique")
        access_refs.add(access_ref)

    effect_refs: set[str] = set()
    for index, raw in enumerate(
        _wire_array(boundary.get("effects"), "j.boundary.effects")
    ):
        item = _wire_object(raw, f"j.boundary.effects[{index}]")
        effect_ref = item.get("effect_id")
        if not isinstance(effect_ref, str) or re.fullmatch(r"e[0-9]+", effect_ref) is None:
            raise RealizationWitnessError(
                f"j.boundary.effects[{index}].effect_id must be an e* handle"
            )
        if effect_ref in effect_refs:
            raise RealizationWitnessError("J effect handles must be unique")
        effect_refs.add(effect_ref)

    scopes = _wire_handle_set(
        data.get("scopes", []),
        path="j.scopes",
        pattern=r"q[0-9]+",
        fields=("scope_id",),
    )
    storage = _wire_handle_set(
        data.get("storage", []),
        path="j.storage",
        pattern=r"(?:b|s)[0-9]+",
        fields=("storage_id",),
    )
    authorizations = _wire_handle_set(
        boundary.get("realization_authorizations", []),
        path="j.boundary.realization_authorizations",
        pattern=r"u[0-9]+",
        fields=("auth_id",),
    )

    node_ids = _wire_object(data.get("node_ids"), "j.node_ids")
    nodes: set[str] = set()
    for node_ref, node_path in node_ids.items():
        if re.fullmatch(r"n[0-9]+", node_ref) is None or not isinstance(node_path, str):
            raise RealizationWitnessError(
                "j.node_ids must map n* handles to operation paths"
            )
        if node_ref in nodes:
            raise RealizationWitnessError("J node handles must be unique")
        nodes.add(node_ref)

    intent = _wire_object(data.get("intent"), "j.intent")
    intent_function = _wire_object(
        intent.get("intent_function"), "j.intent.intent_function"
    )
    value_refs: set[str] = set()
    tensor_refs: set[str] = set()
    states: set[str] = set()
    dimensions: set[str] = set()
    for field_name in ("inputs", "outputs"):
        for index, raw in enumerate(
            _wire_array(intent_function.get(field_name), f"j.intent.intent_function.{field_name}")
        ):
            item = _wire_object(
                raw, f"j.intent.intent_function.{field_name}[{index}]"
            )
            name = item.get("name")
            shape = item.get("shape")
            name = _wire_handle(
                name,
                f"j.intent.intent_function.{field_name}[{index}].name",
                r"t[0-9]+",
                "a logical tensor t* handle",
            )
            if not isinstance(shape, list):
                raise RealizationWitnessError(
                    f"j.intent.intent_function.{field_name}[{index}].shape must be an array"
                )
            tensor_refs.add(name)
            value_refs.add(name)
            for extent_index, extent in enumerate(shape):
                if isinstance(extent, str) and re.fullmatch(r"x[0-9]+", extent):
                    dimensions.add(extent)
                elif isinstance(extent, str) and re.fullmatch(r"p[0-9]+", extent):
                    raise RealizationWitnessError(
                        f"j.intent.intent_function.{field_name}[{index}].shape"
                        f"[{extent_index}] cannot use physical parameter {extent!r}"
                    )

    construction_records = canonical_archive_records(data)
    if construction_records is None:
        raise RealizationWitnessError(
            "J cannot produce a complete canonical construction bundle"
        )
    operation_records = construction_records.operation_records
    # Pi ownership validation consumes the same canonical operation stream as
    # Gate1, Replay, and backend search.  Re-walking raw nested body/arm arrays
    # here would create a second path and node interpretation.
    operation_rows = tuple(
        (str(record["path"]), dict(record["wire"]))
        for record in operation_records
        if isinstance(record.get("path"), str)
        and isinstance(record.get("wire"), Mapping)
    )
    if len(operation_rows) != len(operation_records):
        raise RealizationWitnessError(
            "canonical operation records must contain complete path and wire rows"
        )
    operations_by_path = {path: operation for path, operation in operation_rows}
    operations_by_node: dict[str, dict[str, Any]] = {}
    for node_ref, node_path in node_ids.items():
        operation = operations_by_path.get(node_path)
        if operation is None:
            raise RealizationWitnessError(
                f"j.node_ids.{node_ref} does not resolve to a J operation"
            )
        operations_by_node[node_ref] = operation
        value_refs.update(
            _wire_operation_results(
                operation.get("result"),
                f"j.node_ids.{node_ref}.result",
            )
        )
        carries = operation.get("carry")
        if carries is None:
            continue
        for carry_index, raw_carry in enumerate(_wire_array(carries, f"j.node_ids.{node_ref}.carry")):
            carry = _wire_object(
                raw_carry,
                f"j.node_ids.{node_ref}.carry[{carry_index}]",
            )
            carry_path = f"j.node_ids.{node_ref}.carry[{carry_index}]"
            required_fields = {
                "state_id",
                "init_value",
                "body_argument",
                "next_value",
                "result_value",
                "type",
            }
            missing_fields = sorted(required_fields - set(carry))
            if missing_fields:
                raise RealizationWitnessError(
                    f"{carry_path} is missing complete loop-carry fields: "
                    + ", ".join(missing_fields)
                )
            state_id = _wire_handle(
                carry["state_id"],
                f"{carry_path}.state_id",
                r"s[0-9]+",
                "an s* loop-state handle",
            )
            if state_id in states:
                raise RealizationWitnessError(
                    f"{carry_path}.state_id duplicates loop state {state_id!r}"
                )
            states.add(state_id)
            if not isinstance(carry["type"], dict):
                raise RealizationWitnessError(f"{carry_path}.type must be an object")
            init_value = carry["init_value"]
            if not is_scalar_literal_or_value_reference(init_value):
                raise RealizationWitnessError(
                    f"{carry_path}.init_value must be a scalar literal or t*/v* "
                    "value reference; physical p* parameters are not J values"
                )
            if isinstance(init_value, str) and re.fullmatch(
                r"(?:t|v)[0-9]+", init_value
            ):
                value_refs.add(init_value)
            for field_name in ("body_argument", "next_value", "result_value"):
                ref = _wire_handle(
                    carry[field_name],
                    f"{carry_path}.{field_name}",
                    r"v[0-9]+",
                    "an SSA v* handle",
                )
                value_refs.add(ref)

    return {
        "domains": domains,
        "nodes": nodes,
        "operations_by_node": operations_by_node,
        "accesses": access_refs,
        "effects": effect_refs,
        "tensors": tensor_refs,
        "values": value_refs,
        "states": states,
        "dimensions": dimensions,
        "scopes": scopes,
        "storage": storage,
        "authorizations": authorizations,
    }


def _wire_axis_ref(value: Any, path: str) -> int | str:
    if isinstance(value, bool):
        raise RealizationWitnessError(f"{path} must be a non-negative integer or x*")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"x[0-9]+", value):
        return value
    raise RealizationWitnessError(f"{path} must be a non-negative integer or x*")


def _wire_unique_relation_rows(
    rows: list[Any],
    *,
    path: str,
    field: str,
    relations: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for index, raw in enumerate(rows):
        item_path = f"{path}[{index}]"
        item = _record(
            raw,
            path=item_path,
            required=frozenset({"relation", field}),
        )
        relation = _name(item["relation"], f"{item_path}.relation")
        if relation not in relations:
            raise RealizationWitnessError(
                f"{item_path}.relation is not a closed relation"
            )
        ref = _opaque_anchor_ref(item[field], f"{item_path}.{field}")
        key = (relation, ref)
        if key in seen:
            raise RealizationWitnessError(f"{path} rows must be unique")
        seen.add(key)
        result.append(key)
    return tuple(result)


def validate_normalization_bindings_against_j(
    j: Any,
    c: Any,
    pi: Any,
) -> None:
    """Validate the public J/C ownership boundary of active normalization π.

    This API deliberately has no dependency on :mod:`gate1.model` or
    :mod:`verify.typecheck`.  It accepts either JSON mappings or typed objects,
    verifies that every authored relation names a real J/C declaration, and
    raises :class:`RealizationWitnessError` when the relation cannot be proved
    from those declarations.  The deeper canonical axis/predicate checks remain
    in TypeCheck; this function prevents malformed or cross-owner references
    from reaching that private checker.
    """

    j_data = _wire_object(j, "j")
    c_data = _wire_object(c, "c")
    pi_data = _wire_object(pi, "pi")
    schema = pi_data.get("schema")
    if schema is None:
        return
    if schema != NORMALIZATION_WITNESS_SCHEMA:
        raise RealizationWitnessError(f"unsupported pi subwire schema {schema!r}")

    declared = _wire_declared_j_names(j_data)
    domains = declared["domains"]
    nodes = declared["nodes"]
    operations_by_node = declared["operations_by_node"]
    accesses = declared["accesses"]
    effects = declared["effects"]
    tensors = declared["tensors"]
    values = declared["values"]
    scopes = declared["scopes"]
    storage = declared["storage"]
    authorizations = declared["authorizations"]

    signature = _wire_object(j_data.get("signature"), "j.signature")
    parameter_rows = _wire_array(
        signature.get("parameters"), "j.signature.parameters"
    )
    parameters: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(parameter_rows):
        item = _record(
            raw,
            path=f"j.signature.parameters[{index}]",
            required=frozenset({"name", "kind", "mode"}),
            allowed=frozenset(
                {"name", "kind", "mode", "dtype", "rank", "alias_of", "logical_ref"}
            ),
        )
        name = _name(item["name"], f"j.signature.parameters[{index}].name")
        if name in parameters:
            raise RealizationWitnessError(
                "j.signature parameter names must be unique"
            )
        parameters[name] = item

    raw_slots = _wire_array(c_data.get("slots"), "c.slots")
    slots: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_slots):
        path = f"c.slots[{index}]"
        item = _record(
            raw,
            path=path,
            required=frozenset({"name", "kind", "role"}),
            allowed=frozenset(
                {
                    "name",
                    "kind",
                    "role",
                    "value_type",
                    "required",
                }
            ),
        )
        name = _name(item["name"], f"{path}.name")
        if re.fullmatch(r"c[0-9]+", name) is None:
            raise RealizationWitnessError(f"{path}.name must be a C slot c*")
        if name in slots:
            raise RealizationWitnessError("C slot names must be unique")
        kind = _name(item["kind"], f"{path}.kind")
        if kind not in {"constexpr", "compiler"}:
            raise RealizationWitnessError(f"{path}.kind is not a closed C slot kind")
        role = _name(item["role"], f"{path}.role")
        slots[name] = {
            "kind": kind,
            "role": role,
        }

    def rows(name: str) -> list[Any]:
        raw = pi_data.get(name, [])
        return _wire_array(raw, f"pi.{name}")

    normalization_classes = rows("normalization_classes")
    seen_normalization_classes: set[str] = set()
    for index, raw in enumerate(normalization_classes):
        path = f"pi.normalization_classes[{index}]"
        item = _record(
            raw,
            path=path,
            required=frozenset({"name", "scope"}),
            allowed=frozenset({"name", "scope"}),
        )
        name = _name(item["name"], f"{path}.name")
        scope = _name(item["scope"], f"{path}.scope")
        expected_scope = NORMALIZATION_CLASS_SCOPES.get(name)
        if expected_scope is None:
            raise RealizationWitnessError(
                f"{path}.name is not a closed normalization class"
            )
        if scope != expected_scope:
            raise RealizationWitnessError(
                f"{path}.scope must be {expected_scope!r} for {name!r}"
            )
        if name in seen_normalization_classes:
            raise RealizationWitnessError(
                "pi.normalization_classes must not repeat a class"
            )
        seen_normalization_classes.add(name)

    program_axes = rows("program_axes")
    bound_domains: set[str] = set()
    for index, raw in enumerate(program_axes):
        path = f"pi.program_axes[{index}]"
        item = _record(raw, path=path, required=frozenset({"domain_ref", "axis"}))
        domain_ref = _name(item["domain_ref"], f"{path}.domain_ref")
        if domain_ref not in domains:
            raise RealizationWitnessError(
                f"{path}.domain_ref is not a declared J boundary domain"
            )
        axis = item["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis <= 2:
            raise RealizationWitnessError(f"{path}.axis must be an integer in [0, 2]")
        if domain_ref in bound_domains:
            raise RealizationWitnessError("pi program-axis domain bindings must be unique")
        bound_domains.add(domain_ref)

    program_bindings = rows("program_bindings")
    claimed_program_axes: set[tuple[str, int | str]] = set()
    claimed_physical_axes: set[int] = set()
    physical_axis_by_domain: dict[str, int] = {}
    for index, raw in enumerate(program_bindings):
        path = f"pi.program_bindings[{index}]"
        item = _record(
            raw,
            path=path,
            required=frozenset({"axis", "relation", "logical_axes"}),
        )
        physical_axis = item["axis"]
        if (
            isinstance(physical_axis, bool)
            or not isinstance(physical_axis, int)
            or not 0 <= physical_axis <= 2
        ):
            raise RealizationWitnessError(f"{path}.axis must be an integer in [0, 2]")
        if physical_axis in claimed_physical_axes:
            raise RealizationWitnessError(
                "pi program bindings must bind each physical launch axis once"
            )
        claimed_physical_axes.add(physical_axis)
        relation = _name(item["relation"], f"{path}.relation")
        if relation not in PROGRAM_BINDING_RELATIONS:
            raise RealizationWitnessError(f"{path}.relation is not a closed program relation")
        axes = _wire_array(item["logical_axes"], f"{path}.logical_axes")
        if not axes:
            raise RealizationWitnessError(f"{path}.logical_axes must not be empty")
        if relation == "direct" and len(axes) != 1:
            raise RealizationWitnessError(f"{path}.direct requires one logical axis")
        if relation != "direct" and len(axes) < 2:
            raise RealizationWitnessError(
                f"{path}.{relation} requires at least two logical axes"
            )
        local: set[tuple[str, int | str]] = set()
        for axis_index, raw_axis in enumerate(axes):
            axis_path = f"{path}.logical_axes[{axis_index}]"
            axis = _record(
                raw_axis,
                path=axis_path,
                required=frozenset({"domain_ref", "axis_ref"}),
            )
            domain_ref = _name(axis["domain_ref"], f"{axis_path}.domain_ref")
            if domain_ref not in domains:
                raise RealizationWitnessError(
                    f"{axis_path}.domain_ref is not a declared J domain"
                )
            axis_ref = _wire_axis_ref(axis["axis_ref"], f"{axis_path}.axis_ref")
            key = (domain_ref, axis_ref)
            if key in local or key in claimed_program_axes:
                raise RealizationWitnessError(
                    "pi program bindings must not claim one logical J axis twice"
                )
            previous_physical_axis = physical_axis_by_domain.setdefault(
                domain_ref, physical_axis
            )
            if previous_physical_axis != physical_axis:
                raise RealizationWitnessError(
                    "pi program bindings must map one J domain to one physical "
                    "launch axis"
                )
            local.add(key)
            claimed_program_axes.add(key)
    if program_axes and program_bindings:
        raise RealizationWitnessError(
            "pi.program_axes and pi.program_bindings are alternative complete "
            "program-coordinate mappings and are mutually exclusive"
        )

    iteration_bindings = rows("iteration_bindings")
    claimed_iteration_axes: set[tuple[str, int | str]] = set()
    claimed_iteration_tile_slots: set[str] = set()
    for index, raw in enumerate(iteration_bindings):
        path = f"pi.iteration_bindings[{index}]"
        item = _record(
            raw,
            path=path,
            required=frozenset({"relation", "logical_axes", "tile_slot", "mode_slot"}),
        )
        relation = _name(item["relation"], f"{path}.relation")
        if relation not in ITERATION_BINDING_RELATIONS:
            raise RealizationWitnessError(f"{path}.relation is not a closed iteration relation")
        axes = _wire_array(item["logical_axes"], f"{path}.logical_axes")
        if not axes:
            raise RealizationWitnessError(f"{path}.logical_axes must not be empty")
        valid_axis_keys: set[tuple[str, int | str]] = set()
        for axis_index, raw_axis in enumerate(axes):
            axis_path = f"{path}.logical_axes[{axis_index}]"
            axis = _record(
                raw_axis,
                path=axis_path,
                required=frozenset({"subject_ref", "axis_ref"}),
            )
            subject_ref = _name(axis["subject_ref"], f"{axis_path}.subject_ref")
            if subject_ref not in domains and subject_ref not in nodes:
                raise RealizationWitnessError(
                    f"{axis_path}.subject_ref is not a declared J domain or node"
                )
            if relation == "tiled_domain" and subject_ref not in domains:
                raise RealizationWitnessError(
                    f"{axis_path}.subject_ref for tiled_domain must be a J domain"
                )
            if relation == "tiled_recurrence" and (
                subject_ref not in nodes
                or operations_by_node[subject_ref].get("op") != "loop"
            ):
                raise RealizationWitnessError(
                    f"{axis_path}.subject_ref for tiled_recurrence must be a J loop"
                )
            axis_ref = _wire_axis_ref(axis["axis_ref"], f"{axis_path}.axis_ref")
            key = (subject_ref, axis_ref)
            if key in valid_axis_keys or key in claimed_iteration_axes:
                raise RealizationWitnessError(
                    "pi iteration bindings must not claim one logical J axis twice"
                )
            if key in claimed_program_axes:
                raise RealizationWitnessError(
                    "pi iteration bindings must not reinterpret a logical J axis "
                    "already claimed by a program binding"
                )
            valid_axis_keys.add(key)
            claimed_iteration_axes.add(key)

        tile_slot_ref = _name(item["tile_slot"], f"{path}.tile_slot")
        if tile_slot_ref not in slots:
            raise RealizationWitnessError(
                f"{path}.tile_slot is not declared by C"
            )
        tile_slot = slots[tile_slot_ref]
        if tile_slot["role"] not in {"tile_extent", "reduction_tile_extent"}:
            raise RealizationWitnessError(
                f"{path}.tile_slot must have role tile_extent or reduction_tile_extent"
            )
        if tile_slot["kind"] != "constexpr":
            raise RealizationWitnessError(
                f"{path}.tile_slot must reference a constexpr C slot"
            )
        if tile_slot_ref in claimed_iteration_tile_slots:
            raise RealizationWitnessError(
                "pi iteration bindings must not reuse one C tile slot"
            )
        claimed_iteration_tile_slots.add(tile_slot_ref)

        mode_slot_ref = item["mode_slot"]
        if mode_slot_ref is not None:
            mode_slot_ref = _name(mode_slot_ref, f"{path}.mode_slot")
            if mode_slot_ref not in slots:
                raise RealizationWitnessError(
                    f"{path}.mode_slot is not declared by C"
                )
            _mode_slot = slots[mode_slot_ref]

    for slot_name, slot in slots.items():
        if slot["role"] not in {"tile_extent", "reduction_tile_extent"}:
            continue
        if slot["kind"] != "constexpr":
            raise RealizationWitnessError(
                f"c slot {slot_name} with a tile role must be constexpr"
            )
        if slot_name not in claimed_iteration_tile_slots:
            raise RealizationWitnessError(
                f"c slot {slot_name} with a tile role must be consumed by "
                "pi.iteration_bindings"
            )

    _wire_unique_relation_rows(
        rows("control_bindings"),
        path="pi.control_bindings",
        field="subject_ref",
        relations=CONTROL_BINDING_RELATIONS,
    )
    for index, raw in enumerate(rows("control_bindings")):
        path = f"pi.control_bindings[{index}]"
        item = _record(raw, path=path, required=frozenset({"relation", "subject_ref"}))
        relation = item["relation"]
        subject_ref = item["subject_ref"]
        allowed = domains if relation == "terminal_domain_guard" else nodes | effects
        if subject_ref not in allowed:
            raise RealizationWitnessError(
                f"{path}.subject_ref is not a declared J anchor for {relation}"
            )

    _wire_unique_relation_rows(
        rows("materialization_bindings"),
        path="pi.materialization_bindings",
        field="value_ref",
        relations=MATERIALIZATION_BINDING_RELATIONS,
    )
    for index, raw in enumerate(rows("materialization_bindings")):
        path = f"pi.materialization_bindings[{index}]"
        item = _record(raw, path=path, required=frozenset({"relation", "value_ref"}))
        value_ref = item["value_ref"]
        if value_ref not in tensors | values | nodes:
            raise RealizationWitnessError(
                f"{path}.value_ref is not a J value-producing anchor"
            )

    # One access can legitimately have several independently observed address
    # carriers (for example one stride per logical tensor axis).  Uniqueness is
    # therefore the complete authored occurrence, not just (relation,
    # access_ref).  The latter incorrectly rejected valid FlashAttention
    # bindings before Replay could consume them.
    seen_access_bindings: set[tuple[str, str, str | None, int | None]] = set()
    for index, raw in enumerate(rows("access_representation_bindings")):
        path = f"pi.access_representation_bindings[{index}]"
        item = _record(
            raw,
            path=path,
            required=frozenset({"relation", "access_ref"}),
            allowed=frozenset(
                {"relation", "access_ref", "carrier_ref", "axis_ref"}
            ),
        )
        relation = _name(item["relation"], f"{path}.relation")
        if relation not in ACCESS_REPRESENTATION_BINDING_RELATIONS:
            raise RealizationWitnessError(
                f"{path}.relation is not a closed access representation relation"
            )
        access_ref = _name(item["access_ref"], f"{path}.access_ref")
        if access_ref not in accesses:
            raise RealizationWitnessError(
                f"{path}.access_ref is not a declared J boundary access"
            )
        address_relation = relation in {"address_recomputed", "pointer_advanced"}
        carrier_ref = item.get("carrier_ref")
        axis_ref = item.get("axis_ref")
        if address_relation:
            if carrier_ref is None or axis_ref is None:
                raise RealizationWitnessError(
                    f"{path} address relation requires carrier_ref and axis_ref"
                )
            carrier_ref = _wire_handle(
                carrier_ref,
                f"{path}.carrier_ref",
                r"p[0-9]+",
                "a p* ABI carrier",
            )
            parameter = parameters.get(carrier_ref)
            if parameter is None:
                raise RealizationWitnessError(
                    f"{path}.carrier_ref must name a declared J.signature parameter"
                )
            if (
                parameter.get("kind") != "scalar"
                or parameter.get("mode", "value") != "value"
                or parameter.get("logical_ref") is not None
            ):
                raise RealizationWitnessError(
                    f"{path}.carrier_ref must be a null-logical-ref scalar ABI carrier"
                )
            if (
                isinstance(axis_ref, bool)
                or not isinstance(axis_ref, int)
                or axis_ref < 0
            ):
                raise RealizationWitnessError(
                    f"{path}.axis_ref must be a non-negative tensor-axis integer"
                )
        elif carrier_ref is not None or axis_ref is not None:
            raise RealizationWitnessError(
                f"{path} carrier_ref/axis_ref are only valid for address relations"
            )
        key = (relation, access_ref, carrier_ref, axis_ref)
        if key in seen_access_bindings:
            raise RealizationWitnessError(
                "pi.access_representation_bindings rows must be unique"
            )
        seen_access_bindings.add(key)


def _name(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise RealizationWitnessError(f"{path} must be a non-empty string")
    return value


def _opaque_anchor_ref(value: Any, path: str) -> str:
    result = _name(value, path)
    if re.fullmatch(r"(?:t|n|a|d|e|q|b|s|v|p|c|x)[0-9]+", result) is None:
        raise RealizationWitnessError(
            f"{path} must be an anonymous declaration-order handle"
        )
    return result






def render_realization_witness_wire_grammar() -> str:
    """Return the single model-visible grammar for canonical pi."""

    return (
        "Pi is a canonical normalization witness, not a realization graph. "
        "Its only permitted fields are schema, normalization_classes, "
        "program_axes, program_bindings, iteration_bindings, control_bindings, "
        "materialization_bindings, and access_representation_bindings. "
        "program_axes and program_bindings are alternative complete mappings "
        "from declared J boundary domains to launch axes; neither carries "
        "expressions, predicates, grid extents, Source spelling, or frontend API. "
        "iteration_bindings is the sole semantic connection from a physical C "
        "constexpr tile slot c* to J logical axes. C slots carry physical "
        "metadata only and never subject_ref or axis_ref. "
        "access address relations may reference only p* ABI carriers and an "
        "access-local non-negative tensor axis. p* is an ABI carrier, never a "
        "logical tensor. All other rows cite already-declared J handles only. "
        "normalization_classes select closed J-proven classes; they never add "
        "operations, dataflow, effects, scopes, loops, predicates, or topology. "
        "Replay derives realization from the canonical J/C construction record, "
        "not from an authored pi graph."
    )






@dataclass
class RealizationWitness:
    """Typed canonical normalization witness.

    Lift supplies only the source-neutral normalization choices.  The
    Source/Harness construction record, not a second materialized pi graph,
    owns replay topology and execution realization.
    """

    wire_schema: str = NORMALIZATION_WITNESS_SCHEMA
    normalization_classes: list[dict[str, JsonValue]] = field(
        default_factory=list
    )
    program_axes: list[dict[str, JsonValue]] = field(default_factory=list)
    program_bindings: list[dict[str, JsonValue]] = field(default_factory=list)
    iteration_bindings: list[dict[str, JsonValue]] = field(default_factory=list)
    control_bindings: list[dict[str, JsonValue]] = field(default_factory=list)
    materialization_bindings: list[dict[str, JsonValue]] = field(default_factory=list)
    access_representation_bindings: list[dict[str, JsonValue]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.wire_schema != NORMALIZATION_WITNESS_SCHEMA:
            raise RealizationWitnessError(
                "pi must use the canonical normalization witness wire"
            )
        self._check_normalization_bindings()
        return
    def _check_normalization_bindings(self) -> None:
        """Validate only the closed active subwire, without deriving relations."""

        checked_normalization_classes: list[dict[str, JsonValue]] = []
        seen_normalization_classes: set[str] = set()
        for index, raw in enumerate(self.normalization_classes):
            path = f"pi.normalization_classes[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"name", "scope"}),
                allowed=frozenset({"name", "scope"}),
            )
            name = _name(item["name"], f"{path}.name")
            scope = _name(item["scope"], f"{path}.scope")
            expected_scope = NORMALIZATION_CLASS_SCOPES.get(name)
            if expected_scope is None:
                raise RealizationWitnessError(
                    f"{path}.name is not a closed normalization class"
                )
            if scope != expected_scope:
                raise RealizationWitnessError(
                    f"{path}.scope must be {expected_scope!r} for {name!r}"
                )
            if name in seen_normalization_classes:
                raise RealizationWitnessError(
                    "pi.normalization_classes must not repeat a class"
                )
            seen_normalization_classes.add(name)
            checked_normalization_classes.append({"name": name, "scope": scope})

        checked_program_axes: list[dict[str, JsonValue]] = []
        bound_domains: set[str] = set()
        for index, raw in enumerate(self.program_axes):
            path = f"pi.program_axes[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"domain_ref", "axis"}),
            )
            domain_ref = _name(item["domain_ref"], f"{path}.domain_ref")
            if re.fullmatch(r"d[0-9]+", domain_ref) is None:
                raise RealizationWitnessError(
                    f"{path}.domain_ref must be a J boundary domain handle d*"
                )
            axis = item["axis"]
            if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis <= 2:
                raise RealizationWitnessError(
                    f"{path}.axis must be an integer launch axis in [0, 2]"
                )
            if domain_ref in bound_domains:
                raise RealizationWitnessError(
                    "pi program-axis domain bindings must be unique"
                )
            bound_domains.add(domain_ref)
            checked_program_axes.append({"domain_ref": domain_ref, "axis": axis})

        def axis_ref(value: Any, path: str) -> int | str:
            if isinstance(value, bool):
                raise RealizationWitnessError(
                    f"{path} must be a non-negative integer or J dimension x*"
                )
            if isinstance(value, int) and value >= 0:
                return value
            if isinstance(value, str) and re.fullmatch(r"x[0-9]+", value):
                return value
            raise RealizationWitnessError(
                f"{path} must be a non-negative integer or J dimension x*"
            )

        def logical_axes(
            value: Any,
            *,
            path: str,
            subject_field: str,
            subject_pattern: str,
        ) -> list[dict[str, JsonValue]]:
            if not isinstance(value, list) or not value:
                raise RealizationWitnessError(f"{path} must be a non-empty array")
            result: list[dict[str, JsonValue]] = []
            seen: set[tuple[str, int | str]] = set()
            for ordinal, raw_axis in enumerate(value):
                axis_path = f"{path}[{ordinal}]"
                item = _record(
                    raw_axis,
                    path=axis_path,
                    required=frozenset({subject_field, "axis_ref"}),
                )
                subject = _name(item[subject_field], f"{axis_path}.{subject_field}")
                if re.fullmatch(subject_pattern, subject) is None:
                    raise RealizationWitnessError(
                        f"{axis_path}.{subject_field} has an unsupported J handle"
                    )
                logical_axis = axis_ref(item["axis_ref"], f"{axis_path}.axis_ref")
                key = (subject, logical_axis)
                if key in seen:
                    raise RealizationWitnessError(
                        f"{path} must not repeat a logical J axis"
                    )
                seen.add(key)
                result.append({subject_field: subject, "axis_ref": logical_axis})
            return result

        checked_program_bindings: list[dict[str, JsonValue]] = []
        used_program_axes: set[tuple[str, int | str]] = set()
        used_physical_axes: set[int] = set()
        physical_axis_by_domain: dict[str, int] = {}
        for index, raw in enumerate(self.program_bindings):
            path = f"pi.program_bindings[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"axis", "relation", "logical_axes"}),
            )
            physical_axis = item["axis"]
            if (
                isinstance(physical_axis, bool)
                or not isinstance(physical_axis, int)
                or not 0 <= physical_axis <= 2
            ):
                raise RealizationWitnessError(
                    f"{path}.axis must be an integer launch axis in [0, 2]"
                )
            if physical_axis in used_physical_axes:
                raise RealizationWitnessError(
                    "pi program bindings must bind each physical launch axis once"
                )
            used_physical_axes.add(physical_axis)
            relation = _name(item["relation"], f"{path}.relation")
            if relation not in PROGRAM_BINDING_RELATIONS:
                raise RealizationWitnessError(
                    f"{path}.relation is not a closed program binding relation"
                )
            axes = logical_axes(
                item["logical_axes"],
                path=f"{path}.logical_axes",
                subject_field="domain_ref",
                subject_pattern=r"d[0-9]+",
            )
            if relation == "direct" and len(axes) != 1:
                raise RealizationWitnessError(
                    f"{path}.logical_axes must contain exactly one axis for direct"
                )
            if relation != "direct" and len(axes) < 2:
                raise RealizationWitnessError(
                    f"{path}.logical_axes must contain at least two axes for {relation}"
                )
            keys = {
                (str(axis["domain_ref"]), axis["axis_ref"])
                for axis in axes
            }
            if used_program_axes & keys:
                raise RealizationWitnessError(
                    "pi program bindings must not claim one logical J axis twice"
                )
            for domain_ref, _axis_ref in keys:
                previous_physical_axis = physical_axis_by_domain.setdefault(
                    domain_ref, physical_axis
                )
                if previous_physical_axis != physical_axis:
                    raise RealizationWitnessError(
                        "pi program bindings must map one J domain to one physical "
                        "launch axis"
                    )
            used_program_axes.update(keys)
            checked_program_bindings.append(
                {
                    "axis": physical_axis,
                    "relation": relation,
                    "logical_axes": axes,
                }
            )
        if checked_program_axes and checked_program_bindings:
            raise RealizationWitnessError(
                "pi.program_axes and pi.program_bindings are alternative complete "
                "program-coordinate mappings and are mutually exclusive"
            )

        checked_iteration_bindings: list[dict[str, JsonValue]] = []
        used_iteration_axes: set[tuple[str, int | str]] = set()
        used_iteration_tile_slots: set[str] = set()
        for index, raw in enumerate(self.iteration_bindings):
            path = f"pi.iteration_bindings[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset(
                    {"relation", "logical_axes", "tile_slot", "mode_slot"}
                ),
            )
            relation = _name(item["relation"], f"{path}.relation")
            if relation not in ITERATION_BINDING_RELATIONS:
                raise RealizationWitnessError(
                    f"{path}.relation is not a closed iteration relation"
                )
            axes = logical_axes(
                item["logical_axes"],
                path=f"{path}.logical_axes",
                subject_field="subject_ref",
                subject_pattern=r"(?:d|n)[0-9]+",
            )
            keys = {
                (str(axis["subject_ref"]), axis["axis_ref"])
                for axis in axes
            }
            if used_iteration_axes & keys:
                raise RealizationWitnessError(
                    "pi iteration bindings must not claim one logical J axis twice"
                )
            if used_program_axes & keys:
                raise RealizationWitnessError(
                    "pi iteration bindings must not reinterpret a logical J axis "
                    "already claimed by a program binding"
                )
            used_iteration_axes.update(keys)
            tile_slot = _name(item["tile_slot"], f"{path}.tile_slot")
            if re.fullmatch(r"c[0-9]+", tile_slot) is None:
                raise RealizationWitnessError(f"{path}.tile_slot must be a C slot c*")
            if tile_slot in used_iteration_tile_slots:
                raise RealizationWitnessError(
                    "pi iteration bindings must not reuse one C tile slot"
                )
            used_iteration_tile_slots.add(tile_slot)
            mode_slot = item["mode_slot"]
            if mode_slot is not None:
                mode_slot = _name(mode_slot, f"{path}.mode_slot")
                if re.fullmatch(r"c[0-9]+", mode_slot) is None:
                    raise RealizationWitnessError(
                        f"{path}.mode_slot must be null or a C slot c*"
                    )
            checked_iteration_bindings.append(
                {
                    "relation": relation,
                    "logical_axes": axes,
                    "tile_slot": tile_slot,
                    "mode_slot": mode_slot,
                }
            )

        def closed_anchor_rows(
            rows: list[dict[str, JsonValue]],
            *,
            field: str,
            relations: tuple[str, ...],
            path_name: str,
        ) -> list[dict[str, JsonValue]]:
            result: list[dict[str, JsonValue]] = []
            seen: set[tuple[str, str]] = set()
            for index, raw in enumerate(rows):
                path = f"pi.{path_name}[{index}]"
                item = _record(
                    raw,
                    path=path,
                    required=frozenset({"relation", field}),
                )
                relation = _name(item["relation"], f"{path}.relation")
                if relation not in relations:
                    raise RealizationWitnessError(
                        f"{path}.relation is not supported by this closed relation family"
                    )
                anchor = _opaque_anchor_ref(item[field], f"{path}.{field}")
                key = (relation, anchor)
                if key in seen:
                    raise RealizationWitnessError(f"{path_name} rows must be unique")
                seen.add(key)
                result.append({"relation": relation, field: anchor})
            return result

        checked_control_bindings = closed_anchor_rows(
            self.control_bindings,
            field="subject_ref",
            relations=CONTROL_BINDING_RELATIONS,
            path_name="control_bindings",
        )
        checked_materialization_bindings = closed_anchor_rows(
            self.materialization_bindings,
            field="value_ref",
            relations=MATERIALIZATION_BINDING_RELATIONS,
            path_name="materialization_bindings",
        )
        checked_access_bindings: list[dict[str, JsonValue]] = []
        # Several source stride carriers may own different axes of the same
        # J access.  Keep exact occurrence uniqueness while allowing those
        # distinct (access, carrier, axis) rows.
        seen_access_bindings: set[tuple[str, str, str | None, int | None]] = set()
        for index, raw in enumerate(self.access_representation_bindings):
            path = f"pi.access_representation_bindings[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"relation", "access_ref"}),
                allowed=frozenset(
                    {"relation", "access_ref", "carrier_ref", "axis_ref"}
                ),
            )
            relation = _name(item["relation"], f"{path}.relation")
            if relation not in ACCESS_REPRESENTATION_BINDING_RELATIONS:
                raise RealizationWitnessError(
                    f"{path}.relation is not a closed access representation relation"
                )
            access_ref = _opaque_anchor_ref(
                item["access_ref"], f"{path}.access_ref"
            )
            checked: dict[str, JsonValue] = {
                "relation": relation,
                "access_ref": access_ref,
            }
            address_relation = relation in {
                "address_recomputed",
                "pointer_advanced",
            }
            carrier_ref = item.get("carrier_ref")
            axis_ref = item.get("axis_ref")
            if address_relation:
                if carrier_ref is None or axis_ref is None:
                    raise RealizationWitnessError(
                        f"{path} address relation requires carrier_ref and axis_ref"
                    )
                carrier = _opaque_anchor_ref(
                    carrier_ref, f"{path}.carrier_ref"
                )
                if re.fullmatch(r"p[0-9]+", carrier) is None:
                    raise RealizationWitnessError(
                        f"{path}.carrier_ref must be a p* ABI carrier"
                    )
                if (
                    isinstance(axis_ref, bool)
                    or not isinstance(axis_ref, int)
                    or axis_ref < 0
                ):
                    raise RealizationWitnessError(
                        f"{path}.axis_ref must be a non-negative tensor-axis integer"
                    )
                checked["carrier_ref"] = carrier
                checked["axis_ref"] = axis_ref
            elif carrier_ref is not None or axis_ref is not None:
                raise RealizationWitnessError(
                    f"{path} carrier_ref/axis_ref are only valid for address relations"
                )
            key = (
                relation,
                access_ref,
                checked.get("carrier_ref"),
                checked.get("axis_ref"),
            )
            if key in seen_access_bindings:
                raise RealizationWitnessError(
                    "pi.access_representation_bindings rows must be unique"
                )
            seen_access_bindings.add(key)
            checked_access_bindings.append(checked)

        self.normalization_classes = checked_normalization_classes
        self.program_axes = checked_program_axes
        self.program_bindings = checked_program_bindings
        self.iteration_bindings = checked_iteration_bindings
        self.control_bindings = checked_control_bindings
        self.materialization_bindings = checked_materialization_bindings
        self.access_representation_bindings = checked_access_bindings
    @classmethod
    def from_json(cls, value: Any) -> "RealizationWitness":
        if not isinstance(value, dict) or "schema" not in value:
            raise RealizationWitnessError(
                "pi must use the canonical normalization witness wire; "
                "legacy materialized pi graphs are not accepted"
            )
        if isinstance(value, dict) and "schema" in value:
            data = _record(
                value,
                path="pi",
                required=frozenset(CANONICAL_PI_REQUIRED_FIELDS),
                allowed=frozenset(CANONICAL_PI_WIRE_FIELDS),
            )
            schema = _name(data["schema"], "pi.schema")
            if schema != NORMALIZATION_WITNESS_SCHEMA:
                raise RealizationWitnessError(
                    f"unsupported pi subwire schema {schema!r}"
                )
            if not isinstance(data["program_axes"], list):
                raise RealizationWitnessError("pi.program_axes must be an array")
            for key in (
                "normalization_classes",
                "program_bindings",
                "iteration_bindings",
                "control_bindings",
                "materialization_bindings",
                "access_representation_bindings",
            ):
                if key in data and not isinstance(data[key], list):
                    raise RealizationWitnessError(f"pi.{key} must be an array")
            return cls(
                wire_schema=schema,
                normalization_classes=copy.deepcopy(
                    data.get("normalization_classes", [])
                ),
                program_axes=copy.deepcopy(data["program_axes"]),
                program_bindings=copy.deepcopy(data["program_bindings"]),
                iteration_bindings=copy.deepcopy(data.get("iteration_bindings", [])),
                control_bindings=copy.deepcopy(data.get("control_bindings", [])),
                materialization_bindings=copy.deepcopy(
                    data.get("materialization_bindings", [])
                ),
                access_representation_bindings=copy.deepcopy(
                    data.get("access_representation_bindings", [])
                ),
            )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": NORMALIZATION_WITNESS_SCHEMA,
            "program_axes": copy.deepcopy(self.program_axes),
            "program_bindings": copy.deepcopy(self.program_bindings),
        }
        for name in (
            "normalization_classes",
            "iteration_bindings",
            "control_bindings",
            "materialization_bindings",
            "access_representation_bindings",
        ):
            rows = getattr(self, name)
            if rows:
                result[name] = copy.deepcopy(rows)
        return result


__all__ = [
    "NORMALIZATION_WITNESS_SCHEMA",
    "DEFAULT_ZERO_LOWER_BOUND_CLASS",
    "CLOSED_CONSUMER_IDENTITY_CLASS",
    "PER_PROGRAM_SCALAR_KEEPDIMS_CLASS",
    "NORMALIZATION_CLASS_SCOPES",
    "ProgramCoordinateProjection",
    "ProgramCoordinateProjectionError",
    "ProgramCoordinateRow",
    "PROGRAM_BINDING_RELATIONS",
    "ITERATION_BINDING_RELATIONS",
    "CONTROL_BINDING_RELATIONS",
    "MATERIALIZATION_BINDING_RELATIONS",
    "ACCESS_REPRESENTATION_BINDING_RELATIONS",
    "RealizationWitness",
    "RealizationWitnessError",
    "canonical_program_coordinate_projection",
    "default_zero_lower_bound_accesses",
    "closed_consumer_identity_extensions",
    "per_program_scalar_keepdims_reductions",
    "validate_normalization_bindings_against_j",
    "zero_lower_bound_coordinates_from_j",
    "render_realization_witness_wire_grammar",
]
