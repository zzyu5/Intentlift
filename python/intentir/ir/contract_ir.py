"""Total, target-neutral projection of canonical J's authored contract.

``CanonicalContractIR`` is an ephemeral compiler IR derived one-to-one from J.
It preserves every Layer-B relation while replacing domain, predicate, and
index surface strings with typed ContractExpr programs.  It performs no kernel
shape recognition and does not decide whether a particular RefEval or target
backend can execute the contract.

This distinction is deliberate: constructing M0 must not use a backend
capability matcher as a J-membership oracle.  Concrete consumers lower this IR
and may report an explicit unsupported capability without changing J.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .canonical_j import (
    ContractSort,
    elaborate_j,
    j_elaboration,
    parse_contract_expr,
)
from .contract_semantics import ContractSemanticsError, TypedContractProgram
from .dimension_relation import (
    DimensionRelationError,
    declared_dimension_handles,
    validate_dimension_relations,
)
from .value_dimension import (
    ValueDimensionRelationError,
    validate_value_dimension_relations,
)
from .intent_op_contract import (
    INTEGER_DTYPES,
    canonicalize_dtype,
)


CONTRACT_IR_PROTOCOL = "intentir.canonical-contract-ir.v2"


class ContractIRProjectionError(ValueError):
    """Canonical J cannot be projected without dropping or inventing a relation."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _dtype_sort(dtype: str) -> ContractSort | None:
    try:
        normalized = canonicalize_dtype(dtype, context="ContractIR dtype")
    except (TypeError, ValueError):
        return None
    if normalized == "i1":
        return ContractSort.BOOLEAN
    if normalized in INTEGER_DTYPES - {"i1"}:
        return ContractSort.INTEGER
    return None


@dataclass(frozen=True)
class ContractDomainIR:
    domain_id: str
    predicate: TypedContractProgram

    def to_json(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "predicate": self.predicate.expression.to_json(),
            "reference_sorts": {
                name: sort.value for name, sort in self.predicate.reference_sorts
            },
        }


@dataclass(frozen=True)
class ContractAccessIR:
    access_id: str
    kind: str
    tensor: str
    node_ref: str | None
    index: tuple[TypedContractProgram, ...]
    predicate: TypedContractProgram | None
    fill_json: str

    @property
    def fill_value(self) -> Any:
        return json.loads(self.fill_json)

    def to_json(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "kind": self.kind,
            "tensor": self.tensor,
            "node_ref": self.node_ref,
            "index": [program.expression.to_json() for program in self.index],
            "index_reference_sorts": [
                {name: sort.value for name, sort in program.reference_sorts}
                for program in self.index
            ],
            "predicate": (
                self.predicate.expression.to_json()
                if self.predicate is not None else None
            ),
            "predicate_reference_sorts": (
                {
                    name: sort.value
                    for name, sort in self.predicate.reference_sorts
                }
                if self.predicate is not None else None
            ),
            "fill_value": self.fill_value,
        }


@dataclass(frozen=True)
class CanonicalContractIR:
    """Immutable, complete Layer-B/compiler-interface view of one canonical J."""

    domains: tuple[ContractDomainIR, ...]
    accesses: tuple[ContractAccessIR, ...]
    reference_sorts: tuple[tuple[str, ContractSort], ...]
    remaining_contract_json: str

    @property
    def remaining_contract(self) -> dict[str, Any]:
        value = json.loads(self.remaining_contract_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor owns it
            raise AssertionError("canonical contract payload is not an object")
        return value

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.to_json()).encode("utf-8")
        ).hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CONTRACT_IR_PROTOCOL,
            "domains": [domain.to_json() for domain in self.domains],
            "accesses": [access.to_json() for access in self.accesses],
            "reference_sorts": {
                name: sort.value for name, sort in self.reference_sorts
            },
            "contract": self.remaining_contract,
        }

    @property
    def typed_dimension_constraints(self) -> tuple[TypedContractProgram, ...]:
        """Return J dimension preconditions as typed ContractExpr programs.

        ``remaining_contract`` remains the canonical serialized payload so the
        contract digest stays compatible with existing artifacts.  Consumers do
        not need to reopen that JSON and re-infer the meaning of a dimension
        constraint: this view compiles the exact preserved expression against
        the reference sorts already established by the projection.
        """

        raw_constraints = self.remaining_contract.get("dimension_constraints", [])
        if not isinstance(raw_constraints, list):  # pragma: no cover - projection owns it
            raise AssertionError("canonical dimension constraints are not an array")
        known_sorts = dict(self.reference_sorts)
        programs: list[TypedContractProgram] = []
        for index, raw in enumerate(raw_constraints):
            if not isinstance(raw, str):  # pragma: no cover - SemanticArchive owns it
                raise AssertionError("canonical dimension constraint is not text")
            expression = parse_contract_expr(
                raw,
                expected=ContractSort.BOOLEAN,
                context=f"J.dimension_constraints[{index}]",
            )
            programs.append(
                TypedContractProgram.compile(
                    expression,
                    known_sorts,
                    expected_sort=ContractSort.BOOLEAN,
                    context=f"J.dimension_constraints[{index}]",
                )
            )
        return tuple(programs)

    @property
    def value_dimension_relations(self) -> tuple[dict[str, Any], ...]:
        """Return the preserved, model-validated J value/shape relations."""

        values = self.remaining_contract.get("value_dimension_relations", [])
        if not isinstance(values, list):  # pragma: no cover - projection owns it
            raise AssertionError("canonical value dimension relations are not an array")
        return tuple(dict(item) for item in values)

    @property
    def dimension_relations(self) -> tuple[dict[str, Any], ...]:
        """Return the preserved, model-validated derived dimension definitions."""

        values = self.remaining_contract.get("dimension_relations", [])
        if not isinstance(values, list):  # pragma: no cover - projection owns it
            raise AssertionError("canonical dimension relations are not an array")
        return tuple(dict(item) for item in values)


def _logical_sort_hints(j: Any) -> dict[str, ContractSort]:
    elaboration = j_elaboration(j)
    hints: dict[str, ContractSort] = {}
    for name, value_type in elaboration.value_types:
        if value_type.shape or not value_type.is_tensor or value_type.dtype is None:
            continue
        sort = _dtype_sort(value_type.dtype)
        if sort is not None:
            hints[name] = sort
    for tensor in [*j.intent.inputs, *j.intent.outputs]:
        for extent in tensor.shape:
            if isinstance(extent, str) and extent != "?":
                previous = hints.get(extent)
                if previous not in {None, ContractSort.INTEGER}:
                    raise ContractIRProjectionError(
                        f"symbolic dimension {extent!r} conflicts with a boolean value"
                    )
                hints[extent] = ContractSort.INTEGER
    for relation in j.dimension_relations:
        for extent in (
            relation.result,
            relation.expr.arg,
            *((relation.expr.rhs,) if relation.expr.rhs is not None else ()),
        ):
            previous = hints.get(extent)
            if previous not in {None, ContractSort.INTEGER}:
                raise ContractIRProjectionError(
                    f"dimension relation reference {extent!r} conflicts with a boolean value"
                )
            hints[extent] = ContractSort.INTEGER
    return hints


def _merge_sorts(
    target: dict[str, ContractSort],
    source: Mapping[str, ContractSort],
    *,
    context: str,
) -> None:
    for name, sort in source.items():
        previous = target.get(name)
        if previous is not None and previous is not sort:
            raise ContractIRProjectionError(
                f"{context} gives reference {name!r} conflicting sorts "
                f"{previous.value!r} and {sort.value!r}"
            )
        target[name] = sort


def project_contract_ir(j: Any) -> CanonicalContractIR:
    """Project all of J's contract without testing a target capability profile."""

    canonical = elaborate_j(j)
    try:
        validate_dimension_relations(canonical)
    except DimensionRelationError as exc:
        raise ContractIRProjectionError(str(exc)) from exc
    try:
        validate_value_dimension_relations(canonical)
    except ValueDimensionRelationError as exc:
        raise ContractIRProjectionError(str(exc)) from exc
    elaboration = j_elaboration(canonical)
    hints = _logical_sort_hints(canonical)
    global_sorts = dict(hints)
    declared_dimensions = set(declared_dimension_handles(canonical))
    dimension_constraint_programs: list[TypedContractProgram] = []
    try:
        for index, raw_constraint in enumerate(canonical.dimension_constraints):
            expression = parse_contract_expr(
                raw_constraint,
                expected=ContractSort.BOOLEAN,
                context=f"J.dimension_constraints[{index}]",
            )
            unknown = sorted(set(expression.references) - declared_dimensions)
            if unknown:
                raise ContractIRProjectionError(
                    f"J.dimension_constraints[{index}] references non-dimension "
                    + ", ".join(unknown)
                )
            program = TypedContractProgram.infer(
                expression,
                expected_sort=ContractSort.BOOLEAN,
                known_sorts=global_sorts,
                context=f"J.dimension_constraints[{index}]",
            )
            _merge_sorts(
                global_sorts,
                program.reference_sort_map,
                context=f"J.dimension_constraints[{index}]",
            )
            dimension_constraint_programs.append(program)
    except ContractSemanticsError as exc:
        raise ContractIRProjectionError(str(exc)) from exc
    domains: list[ContractDomainIR] = []
    accesses: list[ContractAccessIR] = []
    try:
        for domain_id, expression in elaboration.domains:
            program = TypedContractProgram.infer(
                expression,
                expected_sort=ContractSort.BOOLEAN,
                known_sorts=global_sorts,
                context=f"J.boundary.domains.{domain_id}",
            )
            _merge_sorts(
                global_sorts,
                program.reference_sort_map,
                context=f"J.boundary.domains.{domain_id}",
            )
            domains.append(ContractDomainIR(domain_id, program))

        structured_accesses = elaboration.access_by_id
        for authored in canonical.boundary.accesses:
            structured = structured_accesses[authored.access_id]
            index_programs: list[TypedContractProgram] = []
            for position, component in enumerate(structured.index.components):
                program = TypedContractProgram.infer(
                    component,
                    expected_sort=ContractSort.INTEGER,
                    known_sorts=global_sorts,
                    context=(
                        f"J.boundary.accesses.{authored.access_id}"
                        f".index[{position}]"
                    ),
                )
                _merge_sorts(
                    global_sorts,
                    program.reference_sort_map,
                    context=f"J.boundary.accesses.{authored.access_id}",
                )
                index_programs.append(program)
            predicate = None
            if structured.predicate is not None:
                predicate = TypedContractProgram.infer(
                    structured.predicate,
                    expected_sort=ContractSort.BOOLEAN,
                    known_sorts=global_sorts,
                    context=f"J.boundary.accesses.{authored.access_id}.predicate",
                )
                _merge_sorts(
                    global_sorts,
                    predicate.reference_sort_map,
                    context=f"J.boundary.accesses.{authored.access_id}.predicate",
                )
            accesses.append(
                ContractAccessIR(
                    access_id=authored.access_id,
                    kind=authored.kind.value,
                    tensor=authored.tensor,
                    node_ref=authored.node_ref,
                    index=tuple(index_programs),
                    predicate=predicate,
                    fill_json=_canonical_json(authored.fill_value),
                )
            )
    except ContractSemanticsError as exc:
        raise ContractIRProjectionError(str(exc)) from exc

    boundary = canonical.boundary.to_json()
    boundary.pop("domains", None)
    boundary.pop("accesses", None)
    remaining = {
        "node_ids": dict(canonical.node_ids),
        "signature": canonical.signature.to_json(),
        "boundary_relations": boundary,
        "dimension_constraints": list(canonical.dimension_constraints),
        "value_dimension_relations": [
            item.to_json() for item in canonical.value_dimension_relations
        ],
        "dimension_relations": [
            item.to_json() for item in canonical.dimension_relations
        ],
        "scopes": [item.to_json() for item in canonical.scopes],
        "storage": [item.to_json() for item in canonical.storage],
    }
    return CanonicalContractIR(
        tuple(domains),
        tuple(accesses),
        tuple(sorted(global_sorts.items())),
        _canonical_json(remaining),
    )


__all__ = [
    "CONTRACT_IR_PROTOCOL",
    "CanonicalContractIR",
    "ContractAccessIR",
    "ContractDomainIR",
    "ContractIRProjectionError",
    "project_contract_ir",
]
