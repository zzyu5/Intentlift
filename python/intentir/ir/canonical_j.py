"""Deterministically decode and elaborate an LLM response into the one J.

The Lift model is allowed to choose semantic structure.  It is not required to
guess punctuation preferred by a particular evaluator or backend.  This module
is the single closed-world boundary that parses the bounded ContractExpr
language, resolves index vectors, materializes uniquely inferable value types,
and removes mechanically redundant output bindings.

The accepted :class:`SemanticArchive` is J.  ``JElaboration`` below is only its
private parsed-expression/type cache, analogous to a compiler symbol table; it
does not contain, serialize, or identify another J.
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from .canonical_construction import (
    CanonicalConstructionRecords,
    canonical_boundary_records,
    canonical_operation_records,
)
from .model import (
    IntentFunction,
    IntentOp,
    ValueType,
    validate_loop_wiring,
    validate_ragged_region_boundaries,
    walk_intent_op_scopes,
)
from .intent_op_contract import (
    IntentOpContractError,
    RAGGED_OPS,
    _broadcast_projection,
    is_non_boolean_integer_dtype,
    is_ragged_carrier_dtype,
    validate_const_literal,
)

if TYPE_CHECKING:  # pragma: no cover - imports are only for static checkers
    from intentir.verify.gate1.model import Gate1Artifact, SemanticArchive


class CanonicalJError(ValueError):
    """J wire syntax has no unique, closed ContractExpr elaboration."""


class ContractSort(str, Enum):
    INTEGER = "integer"
    BOOLEAN = "boolean"
    UNKNOWN_REF = "unknown_ref"


_BINARY_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "floordiv": "//",
    "mod": "%",
    "left_shift": "<<",
    "right_shift": ">>",
}
_COMPARE_SYMBOLS = {
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
}


@dataclass(frozen=True)
class ContractExpr:
    """One node in the closed integer/boolean ContractExpr v1 AST."""

    op: str
    args: tuple["ContractExpr", ...] = ()
    value: str | int | bool | None = None
    sort: ContractSort = ContractSort.UNKNOWN_REF

    @property
    def references(self) -> frozenset[str]:
        result: set[str] = set()
        if self.op == "ref":
            assert isinstance(self.value, str)
            result.add(self.value)
        for argument in self.args:
            result.update(argument.references)
        return frozenset(result)

    def to_text(self) -> str:
        if self.op == "ref":
            assert isinstance(self.value, str)
            return self.value
        if self.op == "int":
            assert isinstance(self.value, int) and not isinstance(self.value, bool)
            return str(self.value)
        if self.op == "bool":
            assert isinstance(self.value, bool)
            return "true" if self.value else "false"
        if self.op == "not":
            return f"not ({self.args[0].to_text()})"
        if self.op == "neg":
            return f"-({self.args[0].to_text()})"
        if self.op == "pos":
            return f"+({self.args[0].to_text()})"
        if self.op in _BINARY_SYMBOLS:
            left, right = self.args
            return (
                f"({left.to_text()} {_BINARY_SYMBOLS[self.op]} "
                f"{right.to_text()})"
            )
        if self.op in _COMPARE_SYMBOLS:
            left, right = self.args
            return (
                f"{left.to_text()} {_COMPARE_SYMBOLS[self.op]} "
                f"{right.to_text()}"
            )
        if self.op in {"and", "or"}:
            return f" {self.op} ".join(
                f"({argument.to_text()})" for argument in self.args
            )
        if self.op == "call":
            assert isinstance(self.value, str)
            return (
                f"{self.value}("
                + ", ".join(argument.to_text() for argument in self.args)
                + ")"
            )
        raise AssertionError(f"unknown ContractExpr opcode {self.op!r}")

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op, "sort": self.sort.value}
        if self.value is not None:
            result["value"] = self.value
        if self.args:
            result["args"] = [argument.to_json() for argument in self.args]
        return result


def _integer_like(expression: ContractExpr) -> bool:
    return expression.sort in {ContractSort.INTEGER, ContractSort.UNKNOWN_REF}


def _boolean_like(expression: ContractExpr) -> bool:
    return expression.sort in {ContractSort.BOOLEAN, ContractSort.UNKNOWN_REF}


def _contract_expr(node: ast.AST, *, context: str) -> ContractExpr:
    if isinstance(node, ast.Name):
        if node.id in {"true", "false", "True", "False"}:
            return ContractExpr(
                "bool",
                value=node.id.lower() == "true",
                sort=ContractSort.BOOLEAN,
            )
        return ContractExpr(
            "ref", value=node.id, sort=ContractSort.UNKNOWN_REF
        )
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return ContractExpr(
                "bool", value=node.value, sort=ContractSort.BOOLEAN
            )
        if isinstance(node.value, int):
            return ContractExpr(
                "int", value=node.value, sort=ContractSort.INTEGER
            )
        raise CanonicalJError(
            f"{context} permits only integer and boolean literals"
        )
    if isinstance(node, ast.UnaryOp):
        argument = _contract_expr(node.operand, context=context)
        if isinstance(node.op, ast.Not):
            if not _boolean_like(argument):
                raise CanonicalJError(f"{context} applies not to an integer")
            return ContractExpr("not", (argument,), sort=ContractSort.BOOLEAN)
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            if not _integer_like(argument):
                raise CanonicalJError(f"{context} applies a sign to a boolean")
            return ContractExpr(
                "neg" if isinstance(node.op, ast.USub) else "pos",
                (argument,),
                sort=ContractSort.INTEGER,
            )
        raise CanonicalJError(f"{context} contains an unsupported unary operator")
    if isinstance(node, ast.BinOp):
        opcode = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.FloorDiv: "floordiv",
            ast.Mod: "mod",
            ast.LShift: "left_shift",
            ast.RShift: "right_shift",
        }.get(type(node.op))
        if opcode is None:
            raise CanonicalJError(
                f"{context} contains an unsupported arithmetic operator"
            )
        left = _contract_expr(node.left, context=context)
        right = _contract_expr(node.right, context=context)
        if not _integer_like(left) or not _integer_like(right):
            raise CanonicalJError(f"{context} uses boolean arithmetic")
        return ContractExpr(
            opcode, (left, right), sort=ContractSort.INTEGER
        )
    if isinstance(node, ast.BoolOp):
        opcode = "and" if isinstance(node.op, ast.And) else (
            "or" if isinstance(node.op, ast.Or) else None
        )
        if opcode is None:
            raise CanonicalJError(f"{context} contains an unsupported boolean op")
        arguments = tuple(
            _contract_expr(value, context=context) for value in node.values
        )
        if not all(_boolean_like(argument) for argument in arguments):
            raise CanonicalJError(f"{context} uses an integer as a predicate")
        return ContractExpr(opcode, arguments, sort=ContractSort.BOOLEAN)
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        terms: list[ContractExpr] = []
        for left_node, operator, right_node in zip(
            operands[:-1], node.ops, operands[1:], strict=True
        ):
            opcode = {
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Gt: "gt",
                ast.GtE: "ge",
                ast.Eq: "eq",
                ast.NotEq: "ne",
            }.get(type(operator))
            if opcode is None:
                raise CanonicalJError(
                    f"{context} contains an unsupported comparison"
                )
            left = _contract_expr(left_node, context=context)
            right = _contract_expr(right_node, context=context)
            if not _integer_like(left) or not _integer_like(right):
                raise CanonicalJError(
                    f"{context} compares a boolean as an integer"
                )
            terms.append(
                ContractExpr(opcode, (left, right), sort=ContractSort.BOOLEAN)
            )
        if len(terms) == 1:
            return terms[0]
        return ContractExpr("and", tuple(terms), sort=ContractSort.BOOLEAN)
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in {"ceil_div", "cdiv", "min", "max"}
            or node.keywords
            or len(node.args) != 2
        ):
            raise CanonicalJError(
                f"{context} permits only binary ceil_div/cdiv/min/max calls"
            )
        arguments = tuple(
            _contract_expr(argument, context=context) for argument in node.args
        )
        if not all(_integer_like(argument) for argument in arguments):
            raise CanonicalJError(f"{context} calls an integer function on bool")
        return ContractExpr(
            "call",
            arguments,
            value="ceil_div" if node.func.id == "cdiv" else node.func.id,
            sort=ContractSort.INTEGER,
        )
    raise CanonicalJError(
        f"{context} contains forbidden syntax {type(node).__name__}"
    )


def _parse_python_expression(expression: str, *, context: str) -> ast.AST:
    if not isinstance(expression, str) or not expression.strip():
        raise CanonicalJError(f"{context} must be a non-empty expression")
    try:
        return ast.parse(expression.strip(), mode="eval").body
    except SyntaxError as exc:
        raise CanonicalJError(f"{context} has invalid ContractExpr syntax") from exc


def parse_contract_expr(
    expression: str,
    *,
    expected: ContractSort,
    context: str = "ContractExpr",
) -> ContractExpr:
    parsed = _contract_expr(
        _parse_python_expression(expression, context=context), context=context
    )
    if expected is ContractSort.INTEGER and not _integer_like(parsed):
        raise CanonicalJError(f"{context} must have integer/index sort")
    if expected is ContractSort.BOOLEAN and not _boolean_like(parsed):
        raise CanonicalJError(f"{context} must have boolean sort")
    return parsed


@dataclass(frozen=True)
class CanonicalIndex:
    """A structured ordered index vector, never an unparsed wire string."""

    components: tuple[ContractExpr, ...]

    @property
    def direct_coordinates(self) -> tuple[str | int, ...] | None:
        """Return the closed ref/integer coordinate vector, if direct.

        Integer coordinates remain integers instead of being conflated with
        axis references.  Consumers can therefore distinguish the literal
        coordinate ``0`` from an axis handle without reparsing wire text.
        General affine expressions deliberately remain outside this profile.
        """

        coordinates: list[str | int] = []
        for component in self.components:
            if component.op == "ref":
                assert isinstance(component.value, str)
                coordinates.append(component.value)
            elif (
                component.op == "int"
                and isinstance(component.value, int)
                and not isinstance(component.value, bool)
            ):
                coordinates.append(component.value)
            else:
                return None
        return tuple(coordinates)

    @property
    def direct_axes(self) -> tuple[str, ...] | None:
        axes: list[str] = []
        for component in self.components:
            if component.op == "ref":
                assert isinstance(component.value, str)
                axes.append(component.value)
            elif component.op == "int" and component.value == 0:
                axes.append("0")
            else:
                return None
        return tuple(axes)

    def to_text(self) -> str:
        return "[" + ", ".join(item.to_text() for item in self.components) + "]"

    def to_json(self) -> list[dict[str, Any]]:
        return [component.to_json() for component in self.components]


def parse_index_expression(
    expression: str, *, context: str = "index_expression"
) -> CanonicalIndex:
    node = _parse_python_expression(expression, context=context)
    if isinstance(node, (ast.List, ast.Tuple)):
        elements = node.elts
    else:
        elements = [node]
    components = tuple(
        _contract_expr(element, context=context) for element in elements
    )
    if not all(_integer_like(component) for component in components):
        raise CanonicalJError(f"{context} contains a boolean index component")
    return CanonicalIndex(components)


@dataclass(frozen=True)
class CanonicalValueType:
    dtype: str | None
    shape: tuple[Any, ...]
    kind: str = "tensor"
    fields: tuple[tuple[str, "CanonicalValueType"], ...] = ()

    def __post_init__(self) -> None:
        # CanonicalValueType is also a public typed-construction boundary.
        # Do not let it grow a weaker validation language than ValueType:
        # callers can construct record/ragged values directly without going
        # through from_json().  Convert nested canonical fields to the model's
        # ValueType representation, validate once through the shared model,
        # then normalize the validated representation back to canonical
        # fields.  This keeps kind/dtype/shape/field and Ragged-carrier rules
        # identical for wire and typed callers.
        def as_value_type(value: "CanonicalValueType") -> ValueType:
            raw_fields: Any = value.fields
            if isinstance(raw_fields, (list, tuple)):
                converted_fields: list[Any] = []
                for item in raw_fields:
                    if (
                        isinstance(item, (list, tuple))
                        and len(item) == 2
                        and isinstance(item[1], CanonicalValueType)
                    ):
                        converted_fields.append(
                            (item[0], as_value_type(item[1]))
                        )
                    else:
                        # Preserve malformed entries for ValueType to reject
                        # with its normal closed-field diagnostics.
                        converted_fields.append(item)
                raw_fields = tuple(converted_fields)
            return ValueType(
                value.kind,
                dtype=value.dtype,
                shape=value.shape,
                fields=raw_fields,
            )

        validated = as_value_type(self)
        object.__setattr__(self, "dtype", validated.dtype)
        object.__setattr__(self, "shape", tuple(validated.shape))
        object.__setattr__(
            self,
            "fields",
            tuple(
                (name, CanonicalValueType.from_value_type(field_type))
                for name, field_type in validated.fields
            ),
        )

    @classmethod
    def from_json(
        cls, value: dict[str, Any], *, context: str = "CanonicalValueType"
    ) -> "CanonicalValueType":
        parsed = ValueType.from_json(value, context=context)
        return cls(
            dtype=parsed.dtype,
            shape=parsed.shape,
            kind=parsed.kind,
            fields=tuple(
                (name, cls.from_value_type(field_type))
                for name, field_type in parsed.fields
            ),
        )

    @classmethod
    def from_value_type(cls, value: ValueType) -> "CanonicalValueType":
        return cls(
            dtype=value.dtype,
            shape=value.shape,
            kind=value.kind,
            fields=tuple(
                (name, cls.from_value_type(field_type))
                for name, field_type in value.fields
            ),
        )

    @property
    def is_tensor(self) -> bool:
        return self.kind == "tensor"

    @property
    def is_record(self) -> bool:
        return self.kind == "record"

    @property
    def is_ragged(self) -> bool:
        return self.kind == "ragged"

    def field(self, name: str) -> "CanonicalValueType":
        for field_name, field_type in self.fields:
            if field_name == name:
                return field_type
        raise KeyError(name)

    def to_json(self) -> dict[str, Any]:
        if self.kind in {"record", "ragged"}:
            return {
                "kind": self.kind,
                "fields": [
                    {"name": name, "type": field_type.to_json()}
                    for name, field_type in self.fields
                ],
            }
        return {"dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class CanonicalBoundaryAccess:
    access_id: str
    index: CanonicalIndex
    predicate: ContractExpr | None


@dataclass(frozen=True)
class CanonicalOutputBinding:
    output: str
    output_type: CanonicalValueType
    value: str
    node_ref: str | None
    abi_index: int
    abi_parameter_index: int
    return_index: int
    mapping_index: int | None
    abi_parameter_name: str
    abi_parameter_mode: str
    mapping_value: str | None
    authored_node_ref: str | None
    producer_record: Mapping[str, Any] | None

    @property
    def store_node_ref(self) -> str | None:
        """Return the canonical physical store anchor for this output.

        ``node_ref`` is the SSA producer of ``value``.  An explicit sparse
        output mapping may additionally name the store/scatter that writes
        the ABI output; that anchor is kept in ``authored_node_ref``.  A
        consumer that validates a physical output access must use this
        projection instead of comparing the producer directly with the store
        operation.
        """

        return self.authored_node_ref or self.node_ref


@dataclass(frozen=True)
class JElaboration:
    """Private structured facts cached on the one accepted J object."""

    domains: tuple[tuple[str, ContractExpr], ...]
    accesses: tuple[CanonicalBoundaryAccess, ...]
    value_types: tuple[tuple[str, CanonicalValueType], ...]
    output_bindings: tuple[CanonicalOutputBinding, ...]
    operation_records: tuple[Mapping[str, Any], ...]
    scope_records: tuple[Mapping[str, Any], ...]
    _snapshot: str
    construction_records: CanonicalConstructionRecords

    @property
    def access_by_id(self) -> dict[str, CanonicalBoundaryAccess]:
        return {access.access_id: access for access in self.accesses}

    @property
    def domain_by_id(self) -> dict[str, ContractExpr]:
        return dict(self.domains)

    @property
    def value_type_by_name(self) -> dict[str, CanonicalValueType]:
        return dict(self.value_types)

    @property
    def output_by_name(self) -> dict[str, CanonicalOutputBinding]:
        """Index the canonical output records without changing their order."""

        return {binding.output: binding for binding in self.output_bindings}

    @property
    def output_names(self) -> tuple[str, ...]:
        """Logical output names in the canonical ABI sequence."""

        return self.construction_records.output_names

    @property
    def scope_by_id(self) -> dict[str, Mapping[str, Any]]:
        """Index the lexical scope records from the same construction pass."""

        return {
            str(scope["scope_id"]): scope
            for scope in self.scope_records
            if isinstance(scope, Mapping) and isinstance(scope.get("scope_id"), str)
        }

    @property
    def boundary_record(self) -> Mapping[str, Any]:
        """The boundary projection built together with all other records."""

        record = self.construction_records.boundary_record
        if record is None:
            raise ValueError("canonical boundary construction record is unavailable")
        return record

    def boundary_rows(self, name: str) -> tuple[Mapping[str, Any], ...]:
        """Return a construction-owned boundary collection."""

        rows = self.construction_records.boundary_rows(name)
        if rows is None:
            raise ValueError(
                f"canonical boundary construction collection {name!r} is unavailable"
            )
        return rows

    @property
    def boundary_domains(self) -> Mapping[str, Any]:
        """Return the construction-owned logical domain map."""

        domains = self.construction_records.boundary_domains
        if domains is None:
            raise ValueError("canonical boundary domain record is unavailable")
        return domains

    @property
    def typed_operation_rows(self) -> tuple[tuple[str, Any], ...]:
        """Materialize the one typed operation tree from canonical records."""

        rows = self.construction_records.typed_operation_rows
        if rows is None:
            raise ValueError("canonical operation records have no typed operation view")
        return rows

    @property
    def operation_by_path(self) -> dict[str, Mapping[str, Any]]:
        """Canonical operation records indexed by lexical path."""

        return self.construction_records.operation_by_path

    @property
    def operation_by_id(self) -> dict[str, Mapping[str, Any]]:
        """Canonical operation records indexed by opaque node id."""

        return self.construction_records.operation_by_id

    def operation_record_by_ref(
        self,
        node_ids: Mapping[str, str] | None = None,
    ) -> dict[str, Mapping[str, Any]]:
        """Raw operation records indexed by canonical path and node id."""

        result = self.construction_records.operation_record_by_ref(node_ids)
        if result is None:
            raise ValueError("canonical operation references are not closed")
        return result

    def authored_node_by_path(
        self,
        node_ids: Mapping[str, str],
    ) -> dict[str, str]:
        """Return the construction-validated authored path-to-node map."""

        result = self.construction_records.authored_node_by_path(node_ids)
        if result is None:
            raise ValueError("canonical authored node references are not closed")
        return result

    @property
    def typed_operation_by_path(self) -> dict[str, Any]:
        """Typed operations indexed by the construction-owned lexical path."""

        result = self.construction_records.typed_operation_by_path
        if result is None:
            raise ValueError("canonical operation records have no typed operation view")
        return result

    @property
    def typed_operation_by_id(self) -> dict[str, Any]:
        """Typed operations indexed by the construction-owned node id."""

        result = self.construction_records.typed_operation_by_id
        if result is None:
            raise ValueError("canonical operation records have no typed operation view")
        return result

    def typed_operation_by_ref(
        self,
        node_ids: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return the shared path/id operation view used by boundary consumers."""

        result = self.construction_records.typed_operation_by_ref(node_ids)
        if result is None:
            raise ValueError("canonical operation references are not closed")
        return result

    @property
    def output_values(self) -> dict[str, str]:
        """Logical output to returned SSA value, in canonical ABI ownership."""

        return self.construction_records.output_values

    def structured_boundary_json(self) -> dict[str, Any]:
        """Diagnostic typed view; it is not an additional persisted artifact."""

        return {
            "boundary": copy.deepcopy(dict(self.boundary_record)),
            "domains": {
                name: expression.to_json() for name, expression in self.domains
            },
            "accesses": [
                {
                    "access_id": access.access_id,
                    "axes": access.index.to_json(),
                    "predicate": (
                        access.predicate.to_json()
                        if access.predicate is not None else None
                    ),
                }
                for access in self.accesses
            ],
            "output_bindings": [
                {
                    "output": binding.output,
                    "output_type": binding.output_type.to_json(),
                    "value": binding.value,
                    "node_ref": binding.node_ref,
                    "abi_index": binding.abi_index,
                    "abi_parameter_index": binding.abi_parameter_index,
                    "abi_parameter_name": binding.abi_parameter_name,
                    "abi_parameter_mode": binding.abi_parameter_mode,
                    "return_index": binding.return_index,
                    "mapping_index": binding.mapping_index,
                    "mapping_value": binding.mapping_value,
                    "authored_node_ref": binding.authored_node_ref,
                    "producer_record": copy.deepcopy(
                        dict(binding.producer_record)
                    )
                    if binding.producer_record is not None
                    else None,
                }
                for binding in self.output_bindings
            ],
            "operations": [
                {
                    "path": record["path"],
                    "id": record["id"],
                    "op": record["op"],
                    "parent_id": record["parent_id"],
                    "lexical_role": record["lexical_role"],
                    "depth": record["depth"],
                    "region_scope_id": record["region_scope_id"],
                    "is_final": record["is_final"],
                    "operands": list(record["operands"]),
                    "results": list(record["results"]),
                    "type": copy.deepcopy(record["type"]),
                    "attrs": copy.deepcopy(record["attrs"]),
                    "operand_types": copy.deepcopy(
                        record.get("operand_types", [])
                    ),
                    "result_types": copy.deepcopy(
                        record.get("result_types", [])
                    ),
                }
                for record in self.operation_records
            ],
            "scopes": [copy.deepcopy(dict(scope)) for scope in self.scope_records],
            "dimension_relations": [
                copy.deepcopy(dict(row))
                for row in self.construction_records.dimension_records
            ],
            "value_dimension_relations": [
                copy.deepcopy(dict(row))
                for row in self.construction_records.value_dimension_records
            ],
        }


def _result_names(op: IntentOp) -> tuple[str, ...]:
    if isinstance(op.result, str):
        return (op.result,)
    if isinstance(op.result, list):
        return tuple(item for item in op.result if isinstance(item, str))
    return ()


def _broadcast_shape(shapes: Sequence[tuple[Any, ...]]) -> tuple[Any, ...] | None:
    if not shapes:
        return None
    rank = max(len(shape) for shape in shapes)
    result: list[Any] = []
    for offset in range(1, rank + 1):
        dimensions = [
            shape[-offset] if offset <= len(shape) else 1 for shape in shapes
        ]
        nonunit = [dimension for dimension in dimensions if dimension != 1]
        if not nonunit:
            result.append(1)
            continue
        first = nonunit[0]
        if any(dimension != first for dimension in nonunit[1:]):
            return None
        result.append(first)
    return tuple(reversed(result))


def _project_explicit_broadcast_shape(
    shape: tuple[Any, ...],
    axes: Sequence[int],
    target_shape: tuple[Any, ...] | None,
    unit_dimensions: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    """Project one operand using only broadcast axes explicitly present in J."""

    try:
        projected, _inserted = _broadcast_projection(
            list(shape),
            axes,
            list(target_shape) if target_shape is not None else None,
            unit_dimensions=unit_dimensions,
        )
    except (NotImplementedError, ValueError) as exc:
        raise CanonicalJError(str(exc)) from exc
    if projected is None:  # pragma: no cover - concrete shape is supplied above
        raise CanonicalJError("explicit broadcast has no closed shape projection")
    return tuple(projected)


_DYNAMIC_EXTENT = "?"
def _is_integer_tensor(value: CanonicalValueType | None, *, rank: int | None = None) -> bool:
    return bool(
        value is not None
        and value.is_tensor
        and value.dtype is not None
        and is_non_boolean_integer_dtype(value.dtype)
        and (rank is None or len(value.shape) == rank)
    )


def _is_ragged_carrier_tensor(
    value: CanonicalValueType | None, *, rank: int | None = None
) -> bool:
    return bool(
        value is not None
        and value.is_tensor
        and is_ragged_carrier_dtype(value.dtype)
        and (rank is None or len(value.shape) == rank)
    )


def _infer_ragged_type(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
) -> CanonicalValueType | None:
    if op.op == "ragged" and len(op.operands) == 2:
        offsets = known.get(op.operands[0])
        indices = known.get(op.operands[1])
        if _is_ragged_carrier_tensor(offsets, rank=1) and _is_ragged_carrier_tensor(
            indices, rank=1
        ):
            return CanonicalValueType(
                None,
                (),
                kind="ragged",
                fields=(
                    ("offsets", offsets),
                    ("indices", indices),
                ),
            )
    if op.op == "ragged_outer" and len(op.operands) == 1:
        ragged = known.get(op.operands[0])
        if ragged is not None and ragged.is_ragged:
            offsets = ragged.field("offsets")
            if offsets.dtype is not None:
                return CanonicalValueType(offsets.dtype, ())
    if op.op == "ragged_member" and len(op.operands) == 2:
        ragged = known.get(op.operands[0])
        index = known.get(op.operands[1])
        if ragged is not None and ragged.is_ragged and _is_ragged_carrier_tensor(
            index, rank=0
        ):
            indices = ragged.field("indices")
            if indices.dtype is not None:
                return CanonicalValueType(indices.dtype, (_DYNAMIC_EXTENT,))
    return None


def _validate_ragged_operation(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
    declared: CanonicalValueType | None,
) -> None:
    if op.op not in RAGGED_OPS:
        return
    if op.attrs:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} {op.op} does not accept attrs"
        )
    names = _result_names(op)
    if len(names) != 1:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} {op.op} requires one result"
        )
    if declared is None:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} {op.op} requires an explicit result type"
        )
    if op.op == "ragged":
        if len(op.operands) != 2:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} ragged requires offsets and indices"
            )
        offsets = known.get(op.operands[0])
        indices = known.get(op.operands[1])
        if offsets is not None and not _is_ragged_carrier_tensor(offsets, rank=1):
            raise CanonicalJError(
                "ragged offsets must be a rank-1 tensor with a closed carrier dtype"
            )
        if indices is not None and not _is_ragged_carrier_tensor(indices, rank=1):
            raise CanonicalJError(
                "ragged indices must be a rank-1 tensor with a closed carrier dtype"
            )
        if declared is not None:
            if not declared.is_ragged:
                raise CanonicalJError("ragged result must have kind='ragged'")
            if tuple(name for name, _ in declared.fields) != ("offsets", "indices"):
                raise CanonicalJError("ragged result fields must be offsets, indices")
            if offsets is not None and declared.field("offsets") != offsets:
                raise CanonicalJError("ragged offsets field type differs from operand")
            if indices is not None and declared.field("indices") != indices:
                raise CanonicalJError("ragged indices field type differs from operand")
        return
    if len(op.operands) != (1 if op.op == "ragged_outer" else 2):
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} {op.op} has invalid operand count"
        )
    ragged = known.get(op.operands[0])
    if ragged is not None and not ragged.is_ragged:
        raise CanonicalJError(f"{op.op} requires a ragged descriptor operand")
    if op.op == "ragged_member":
        index = known.get(op.operands[1])
        if index is not None and not _is_ragged_carrier_tensor(index, rank=0):
            raise CanonicalJError(
                "ragged_member outer index must be a rank-0 closed carrier"
            )
    if declared is not None:
        if not declared.is_tensor or not _is_ragged_carrier_tensor(declared):
            raise CanonicalJError(
                f"{op.op} result must use a closed Ragged carrier dtype"
            )
        expected_rank = 0 if op.op == "ragged_outer" else 1
        if len(declared.shape) != expected_rank:
            raise CanonicalJError(
                f"{op.op} result must have rank {expected_rank}"
            )
        if op.op == "ragged_member" and declared.shape != (_DYNAMIC_EXTENT,):
            raise CanonicalJError(
                "ragged_member result shape must be the dynamic rank-1 shape ['?']"
            )
        expected_field = "offsets" if op.op == "ragged_outer" else "indices"
        if ragged is not None and declared.dtype != ragged.field(expected_field).dtype:
            raise CanonicalJError(
                f"{op.op} result dtype must equal descriptor {expected_field} dtype"
            )


def _infer_shape(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
    broadcast_axes: Mapping[str, Mapping[str, tuple[int, ...]]] | None = None,
    unit_dimensions: frozenset[str] = frozenset(),
) -> tuple[Any, ...] | None:
    if op.op == "ragged_outer":
        return ()
    if op.op == "ragged_member":
        return (_DYNAMIC_EXTENT,)
    if op.op == "ragged":
        return None
    if op.op == "record_get" and op.operands:
        record = known.get(op.operands[0])
        field = op.attrs.get("field")
        if record is not None and record.is_record and isinstance(field, str):
            return record.field(field).shape
    if op.op == "value_if":
        result = known.get(op.result) if isinstance(op.result, str) else None
        arm_shapes = [
            known.get(arm.yield_value).shape
            for arm in (op.positive, op.negative)
            if arm is not None
            and known.get(arm.yield_value) is not None
            and known.get(arm.yield_value).is_tensor
        ]
        if any(shape != arm_shapes[0] for shape in arm_shapes[1:]):
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op} value_if arm yields have "
                "different tensor shapes"
            )
        if result is not None and result.is_tensor:
            if arm_shapes and result.shape != arm_shapes[0]:
                raise CanonicalJError(
                    f"IntentOp {op.node_id or op.op} result shape differs from "
                    "its arm yield shape"
                )
            return result.shape
        return arm_shapes[0] if arm_shapes else None
    operand_shapes = [known[name].shape for name in op.operands if name in known]
    if op.op in {"map_unary", "cast", "scan", "flip"} and operand_shapes:
        return operand_shapes[0]
    if op.op == "scatter" and operand_shapes:
        return operand_shapes[0]
    if op.op == "nonzero" and operand_shapes:
        return (_DYNAMIC_EXTENT, len(operand_shapes[0]))
    if op.op == "masked_select" and operand_shapes:
        return (_DYNAMIC_EXTENT,)
    if op.op == "masked_scatter" and operand_shapes:
        return operand_shapes[0]
    if op.op == "reshape" and operand_shapes:
        shape = op.attrs.get("shape")
        if not isinstance(shape, list):
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} reshape requires attrs.shape"
            )
        if any(
            not (
                isinstance(dimension, str)
                or isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension >= 0
            )
            for dimension in shape
        ):
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} reshape shape must contain "
                "non-negative integers or symbolic dimensions"
            )
        return tuple(shape)
    if op.op in {"map_binary", "select"} and len(operand_shapes) == len(op.operands):
        explicit = (
            broadcast_axes.get(op.result, {})
            if broadcast_axes is not None and isinstance(op.result, str)
            else {}
        )
        target = known.get(op.result) if isinstance(op.result, str) else None
        target_shape = target.shape if target is not None else None
        if target_shape is None and explicit:
            target_shape = next(
                (
                    shape
                    for operand, shape in zip(
                        op.operands, operand_shapes, strict=True
                    )
                    if operand not in explicit
                    and len(shape) == max(len(item) for item in operand_shapes)
                ),
                None,
            )
        projected = tuple(
            _project_explicit_broadcast_shape(
                shape,
                explicit.get(operand, ()),
                target_shape,
                unit_dimensions,
            )
            for operand, shape in zip(op.operands, operand_shapes, strict=True)
        )
        inferred = _broadcast_shape(projected)
        if inferred is None and op.op in {"map_binary", "select"}:
            shape_details = ", ".join(
                f"{operand}={list(shape)}"
                for operand, shape in zip(
                    op.operands, operand_shapes, strict=True
                )
            )
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} operands have incompatible "
                "NumPy broadcast shapes; a non-trailing broadcast must be "
                "declared explicitly in J.boundary.broadcasts; "
                f"operands: {shape_details}. Use a source-established "
                "keepdims shape or one exact broadcast row; do not change "
                "the result shape alone"
            )
        return inferred
    if op.op == "transpose" and operand_shapes:
        permutation = op.attrs.get("perm")
        if (
            isinstance(permutation, list)
            and all(isinstance(axis, int) and not isinstance(axis, bool) for axis in permutation)
            and sorted(permutation) == list(range(len(operand_shapes[0])))
        ):
            return tuple(operand_shapes[0][axis] for axis in permutation)
    if op.op == "concat" and len(operand_shapes) == len(op.operands):
        if not operand_shapes:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} concat requires an operand"
            )
        axis = op.attrs.get("axis")
        rank = len(operand_shapes[0])
        if (
            not isinstance(axis, int)
            or isinstance(axis, bool)
            or axis < 0
            or axis >= rank
        ):
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} concat axis {axis!r} is "
                f"out of range for rank-{rank} operands"
            )
        if any(len(shape) != rank for shape in operand_shapes[1:]):
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} concat operands must have "
                "one common rank"
            )
        first = operand_shapes[0]
        for dimension in range(rank):
            if dimension == axis:
                continue
            expected = first[dimension]
            if any(shape[dimension] != expected for shape in operand_shapes[1:]):
                raise CanonicalJError(
                    f"IntentOp {op.node_id or op.op!r} concat operands disagree "
                    f"at non-concat dimension {dimension}"
                )
        axis_extents = [shape[axis] for shape in operand_shapes]
        if all(isinstance(extent, int) and not isinstance(extent, bool)
               for extent in axis_extents):
            result = list(first)
            result[axis] = sum(axis_extents)
            return tuple(result)
        # Symbolic sums are not guessed by the elaborator.  The checks above
        # still reject rank/axis/non-concat contradictions; an explicit closed
        # result type remains authoritative for the unresolved concat extent.
        return None
    if op.op == "pool" and len(operand_shapes) == len(op.operands):
        source = operand_shapes[0]
        if len(source) != 4:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op} pool requires an NCHW rank-4 operand"
            )
        values: dict[str, int] = {}
        for name in ("kH", "kW", "sH", "sW", "padH", "padW"):
            value = op.attrs.get(name)
            minimum = 0 if name.startswith("pad") else 1
            if type(value) is not int or value < minimum:
                raise CanonicalJError(
                    f"IntentOp {op.node_id or op.op} pool attr {name} must be "
                    f"an integer >= {minimum}"
                )
            values[name] = value
        if type(source[2]) is not int or type(source[3]) is not int:
            return None
        height = (
            source[2] + 2 * values["padH"] - values["kH"]
        ) // values["sH"] + 1
        width = (
            source[3] + 2 * values["padW"] - values["kW"]
        ) // values["sW"] + 1
        if height < 0 or width < 0:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op} pool window produces a negative extent"
            )
        return (source[0], source[1], height, width)
    if op.op == "upsample" and len(operand_shapes) == len(op.operands):
        source = operand_shapes[0]
        scale = op.attrs.get("scale")
        if len(source) != 4:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op} upsample requires an NCHW rank-4 operand"
            )
        if (
            not isinstance(scale, list)
            or len(scale) != 2
            or any(type(value) is not int or value < 1 for value in scale)
        ):
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op} upsample scale must be two positive integers"
            )
        if type(source[2]) is not int or type(source[3]) is not int:
            return None
        return (source[0], source[1], source[2] * scale[0], source[3] * scale[1])
    if op.op == "polar" and len(operand_shapes) == len(op.operands):
        if operand_shapes[0] != operand_shapes[1]:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op} polar inputs must have one identical shape"
            )
        return operand_shapes[0]
    if op.op == "conv" and len(operand_shapes) == len(op.operands):
        input_shape, weight_shape = operand_shapes[:2]
        spatial_rank = len(input_shape) - 2
        if spatial_rank not in {1, 2, 3} or len(weight_shape) != len(input_shape):
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op} conv input and weight must have "
                "matching rank 3, 4, or 5 shapes"
            )
        sequences: dict[str, list[int]] = {}
        for name, minimum in (("stride", 1), ("padding", 0), ("dilation", 1)):
            values = op.attrs.get(name)
            if (
                not isinstance(values, list)
                or len(values) != spatial_rank
                or any(type(value) is not int or value < minimum for value in values)
            ):
                raise CanonicalJError(
                    f"IntentOp {op.node_id or op.op} conv {name} must have "
                    f"{spatial_rank} integers >= {minimum}"
                )
            sequences[name] = values
        groups = op.attrs.get("groups")
        if type(groups) is not int or groups < 1:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op} conv groups must be a positive integer"
            )
        spatial: list[int] = []
        for index in range(spatial_rank):
            input_extent = input_shape[2 + index]
            kernel_extent = weight_shape[2 + index]
            if type(input_extent) is not int or type(kernel_extent) is not int:
                return None
            extent = (
                input_extent
                + 2 * sequences["padding"][index]
                - sequences["dilation"][index] * (kernel_extent - 1)
                - 1
            ) // sequences["stride"][index] + 1
            if extent < 0:
                raise CanonicalJError(
                    f"IntentOp {op.node_id or op.op} conv produces a negative spatial extent"
                )
            spatial.append(extent)
        return (input_shape[0], weight_shape[0], *spatial)
    if op.op == "reduce" and operand_shapes:
        axes_value = op.attrs.get("axes", [op.attrs.get("axis")])
        if isinstance(axes_value, list) and all(
            isinstance(axis, int) and not isinstance(axis, bool)
            for axis in axes_value
        ):
            source = operand_shapes[0]
            axes = {axis if axis >= 0 else len(source) + axis for axis in axes_value}
            if any(axis < 0 or axis >= len(source) for axis in axes):
                return None
            keepdims_shape = tuple(
                1 if index in axes else dim
                for index, dim in enumerate(source)
            )
            collapsed_shape = tuple(
                dim for index, dim in enumerate(source) if index not in axes
            )
            keepdims = op.attrs.get("keepdims")
            if keepdims is not None and type(keepdims) is not bool:
                raise CanonicalJError(
                    f"IntentOp {op.node_id or op.op} reduce keepdims must be a boolean"
                )
            if keepdims is True:
                selected = keepdims_shape
            elif keepdims is False:
                selected = collapsed_shape
            else:
                # The result declaration/output binding is the only valid
                # disambiguator for an intermediate reduction.  This mirrors
                # emit_mlir: a non-full reduction otherwise keeps dimensions
                # for broadcast consumers, while a full reduction defaults to
                # a genuine rank-0 value.
                declared = (
                    known.get(op.result)
                    if isinstance(op.result, str)
                    else None
                )
                declared_shape = (
                    declared.shape
                    if declared is not None and declared.is_tensor
                    else None
                )
                if declared_shape is not None:
                    if declared_shape not in (keepdims_shape, collapsed_shape):
                        raise CanonicalJError(
                            f"IntentOp {op.node_id or op.op} reduce result shape "
                            f"{declared_shape!r} is neither keepdims "
                            f"{keepdims_shape!r} nor collapsed {collapsed_shape!r}"
                        )
                    selected = tuple(declared_shape)
                elif axes == set(range(len(source))):
                    selected = collapsed_shape
                else:
                    selected = keepdims_shape
            return selected
    if op.op == "arg_reduce" and operand_shapes:
        axes_value = [op.attrs.get("axis")]
        if isinstance(axes_value[0], int) and not isinstance(axes_value[0], bool):
            source = operand_shapes[0]
            axis = axes_value[0] if axes_value[0] >= 0 else len(source) + axes_value[0]
            if 0 <= axis < len(source):
                return tuple(dim for index, dim in enumerate(source) if index != axis)
    if op.op == "sort" and operand_shapes:
        shape = list(operand_shapes[0])
        width = op.attrs.get("k")
        if width is not None and shape:
            shape[-1] = width
        return tuple(shape)
    if op.op == "const":
        result_type = op.type if isinstance(op.type, dict) else None
        try:
            literal = validate_const_literal(
                op.attrs,
                result_type=result_type,
                context=f"IntentOp {op.node_id or op.op}",
            )
        except IntentOpContractError as exc:
            raise CanonicalJError(str(exc)) from exc
        return tuple(literal.shape)
    if op.op == "rand":
        shape = op.attrs.get("shape", [])
        if isinstance(shape, list):
            return tuple(shape)
    if op.op == "iota" and "n" in op.attrs:
        return (op.attrs["n"],)
    return None


def _infer_dtypes(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
) -> dict[str, str]:
    if op.op in RAGGED_OPS:
        inferred = _infer_ragged_type(op, known)
        if inferred is not None and inferred.is_tensor and inferred.dtype is not None:
            names = _result_names(op)
            return {names[0]: inferred.dtype} if len(names) == 1 else {}
        return {}
    if op.op == "record_get" and isinstance(op.result, str) and op.operands:
        record = known.get(op.operands[0])
        field = op.attrs.get("field")
        if record is not None and record.is_record and isinstance(field, str):
            try:
                field_type = record.field(field)
            except KeyError:
                return {}
            if field_type.is_tensor and field_type.dtype is not None:
                return {op.result: field_type.dtype}
        return {}
    if op.op == "cast":
        # Keep malformed cast attributes at the J contract boundary. Without
        # this guard the shared MLIR dtype helper receives ``None`` and emits
        # an implementation-level error before CandidateCheck/TypeCheck can
        # explain the authored contract violation.
        attrs = op.attrs
        if set(attrs) != {"to_dtype"}:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} cast requires exactly one "
                "attrs.to_dtype"
            )
        target = attrs["to_dtype"]
        from .intent_op_contract import canonicalize_dtype

        try:
            canonicalize_dtype(
                target,
                context=f"IntentOp {op.node_id or op.op}.attrs.to_dtype",
            )
        except (TypeError, ValueError) as exc:
            raise CanonicalJError(str(exc)) from exc
    # Keep one source of truth with the MLIR emitter's closed operation table.
    from intentir.frontend.emit_mlir import _infer_op_dtypes

    dtypes = {
        name: value.dtype
        for name, value in known.items()
        if value.is_tensor and value.dtype is not None
    }
    return _infer_op_dtypes(op, dtypes)


def _merge_type(
    known: dict[str, CanonicalValueType],
    name: str,
    value: CanonicalValueType,
    *,
    context: str,
) -> bool:
    previous = known.get(name)
    if previous is not None and previous != value:
        raise CanonicalJError(
            f"{context} infers conflicting types for {name!r}: "
            f"{previous.to_json()} versus {value.to_json()}"
        )
    if previous is None:
        known[name] = value
        return True
    return False


def _validate_record_operation(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
    declared: CanonicalValueType | None,
) -> None:
    """Validate named record construction/extraction without flattening it."""

    if op.op == "record":
        if not isinstance(op.result, str) or declared is None or not declared.is_record:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} requires one record result type"
            )
        fields = op.attrs.get("fields")
        names = [name for name, _field_type in declared.fields]
        if not isinstance(fields, list) or fields != names or len(op.operands) != len(names):
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} fields must match its "
                "record type and operand order"
            )
        for name, operand in zip(names, op.operands, strict=True):
            operand_type = known.get(operand)
            if operand_type is None:
                continue
            if operand_type != declared.field(name):
                raise CanonicalJError(
                    f"IntentOp {op.node_id or op.op!r} field {name!r} type "
                    f"does not match operand {operand!r}"
                )
        return
    if op.op == "record_get":
        if not isinstance(op.result, str) or len(op.operands) != 1:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} record_get requires one "
                "record operand and one result"
            )
        field = op.attrs.get("field")
        record = known.get(op.operands[0])
        if not isinstance(field, str) or record is None or not record.is_record:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} record_get requires a "
                "known record operand and field"
            )
        try:
            field_type = record.field(field)
        except KeyError as exc:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} references unknown record "
                f"field {field!r}"
            ) from exc
        if declared is not None and declared != field_type:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} result type differs from "
                f"record field {field!r}"
            )


def _validate_gather_operation(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
    declared: CanonicalValueType | None,
) -> None:
    """Validate the shared J-level gather shape contract.

    ``along_axis`` has two source-level spellings: an equal-rank index, or a
    compressed index with the selected axis omitted.  The latter is legalized
    to equal rank before the Intent MLIR op is emitted.  Keeping this rule in
    canonical J makes TypeCheck, the CPU reference, and the MLIR projection
    consume one semantic contract instead of allowing only one of them to
    accept the compressed form.
    """

    if op.op != "gather":
        return
    if len(op.operands) != 2:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} gather requires data and index"
        )
    mode = op.attrs.get("mode")
    if mode not in {"along_axis", "take"}:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} gather mode must be "
            "'along_axis' or 'take'"
        )
    axis = op.attrs.get("axis")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} gather axis must be an integer"
        )
    data = known.get(op.operands[0])
    index = known.get(op.operands[1])
    if data is None or index is None:
        return
    if not data.is_tensor or not index.is_tensor:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} gather requires tensor data "
            "and index values"
        )
    if not is_non_boolean_integer_dtype(index.dtype):
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} gather index must use a "
            "closed non-boolean integer dtype"
        )
    rank = len(data.shape)
    normalized_axis = axis + rank if axis < 0 else axis
    if normalized_axis < 0 or normalized_axis >= rank:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} gather axis {axis} is outside "
            f"data rank {rank}"
        )

    if mode == "along_axis":
        equal_rank = len(index.shape) == rank
        compressed_rank = len(index.shape) == rank - 1 and len(index.shape) > 0
        if not (equal_rank or compressed_rank):
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} along_axis gather index "
                "rank must equal data rank or data rank minus one"
            )
        expected_index_shape = (
            data.shape
            if equal_rank
            else data.shape[:normalized_axis] + data.shape[normalized_axis + 1 :]
        )
        if index.shape != expected_index_shape:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} along_axis gather index "
                f"shape {index.shape!r} must match non-axis data shape "
                f"{expected_index_shape!r}; if Source supplies a selected-axis "
                "vector shared across an outer domain, author an explicit J "
                "broadcast to the data rank before gather"
            )
        expected_shape = index.shape
    else:
        expected_shape = (
            data.shape[:normalized_axis]
            + index.shape
            + data.shape[normalized_axis + 1 :]
        )

    if declared is not None:
        if not declared.is_tensor:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} gather result must be a tensor"
            )
        if declared.dtype != data.dtype:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} gather result dtype must "
                "equal data dtype"
            )
        if declared.shape != expected_shape:
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} gather result shape "
                f"{declared.shape!r} must be {expected_shape!r}"
            )


def _validate_reshape_operation(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
    declared: CanonicalValueType | None,
) -> None:
    if op.op != "reshape":
        return
    if len(op.operands) != 1:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} reshape requires one operand"
        )
    shape = op.attrs.get("shape")
    if not isinstance(shape, list) or any(
        not (
            isinstance(item, str)
            or isinstance(item, int)
            and not isinstance(item, bool)
            and item >= 0
        )
        for item in shape
    ):
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} reshape requires a closed "
            "shape list"
        )
    if declared is not None and tuple(shape) != declared.shape:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} attrs.shape differs from its "
            "declared result shape"
        )
    source = known.get(op.operands[0])
    if source is not None and declared is not None:
        if source.is_tensor != declared.is_tensor:
            raise CanonicalJError("reshape cannot change tensor/record kind")
        if source.is_tensor and source.dtype != declared.dtype:
            raise CanonicalJError("reshape must preserve the element dtype")
        if source.is_tensor and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in [*source.shape, *declared.shape]
        ):
            source_elements = 1
            result_elements = 1
            for item in source.shape:
                source_elements *= item
            for item in declared.shape:
                result_elements *= item
            if source_elements != result_elements:
                raise CanonicalJError(
                    f"IntentOp {op.node_id or op.op!r} changes the number of "
                    "elements"
                )


def _validate_const_operation(
    op: IntentOp,
    known: Mapping[str, CanonicalValueType],
    declared: CanonicalValueType | None,
) -> None:
    if op.op != "const":
        return
    names = _result_names(op)
    if len(op.operands) != 0:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} const requires zero operands"
        )
    if len(names) != 1:
        raise CanonicalJError(
            f"IntentOp {op.node_id or op.op!r} const requires one result"
        )
    result_type = declared or known.get(names[0])
    result_json = result_type.to_json() if result_type is not None else None
    try:
        validate_const_literal(
            op.attrs,
            result_type=result_json,
            context=f"IntentOp {op.node_id or op.op}",
        )
    except IntentOpContractError as exc:
        raise CanonicalJError(str(exc)) from exc


def _elaborate_function_types(
    fn: IntentFunction,
    broadcast_axes: Mapping[str, Mapping[str, tuple[int, ...]]] | None = None,
    unit_dimensions: frozenset[str] = frozenset(),
) -> dict[str, CanonicalValueType]:
    try:
        validate_ragged_region_boundaries(fn.body)
    except ValueError as exc:
        raise CanonicalJError(str(exc)) from exc
    known: dict[str, CanonicalValueType] = {
        tensor.name: CanonicalValueType(tensor.dtype, tuple(tensor.shape))
        for tensor in [*fn.inputs, *fn.outputs]
    }
    outputs = {
        tensor.name: CanonicalValueType(tensor.dtype, tuple(tensor.shape))
        for tensor in fn.outputs
    }
    for binding in fn.returns:
        output_type = outputs.get(binding.output)
        if output_type is None:
            raise CanonicalJError(
                f"IntentFunction return names unknown output {binding.output!r}"
            )
        _merge_type(
            known,
            binding.value,
            output_type,
            context="IntentFunction return",
        )

    scoped_operations = tuple(
        walk_intent_op_scopes(
            fn.body,
            visible=(tensor.name for tensor in fn.inputs),
        )
    )
    operations = tuple(item[1] for item in scoped_operations)
    visible_names = {
        id(operation): frozenset(bindings)
        for _path, operation, bindings in scoped_operations
    }
    for _ in range(len(operations) + 3):
        changed = False
        for op in operations:
            operation_known = {
                name: known[name]
                for name in visible_names.get(id(op), ())
                if name in known
            }
            if op.op == "loop":
                try:
                    validate_loop_wiring(op, context="canonical J loop")
                except ValueError as exc:
                    raise CanonicalJError(str(exc)) from exc
                # Structured-loop binders are ordinary typed J values.  A scan
                # name denotes the complete logical source tensor (physical
                # tiling belongs to C/pi), while every carry role has exactly
                # the authored carry type.  Seed these facts before checking
                # the loop body so an explicit body type cannot hide a shape
                # contradiction at the region boundary.
                for scan in op.scan:
                    source_type = operation_known.get(scan.source)
                    if source_type is None:
                        continue
                    if scan.axis < 0 or scan.axis >= len(source_type.shape):
                        raise CanonicalJError(
                            f"IntentOp {op.node_id or op.op!r} scan axis "
                            f"{scan.axis} is outside source rank "
                            f"{len(source_type.shape)}"
                        )
                    changed |= _merge_type(
                        known,
                        scan.name,
                        source_type,
                        context=f"IntentOp {op.node_id or op.op} scan",
                    )
                if op.trip is not None and op.trip.is_dynamic:
                    upper_type = operation_known.get(op.trip.upper_bound or "")
                    if upper_type is not None:
                        if not _is_integer_tensor(upper_type, rank=0):
                            raise CanonicalJError(
                                f"IntentOp {op.node_id or op.op} dynamic trip upper_bound "
                                "must be a rank-0 integer tensor"
                            )
                        changed |= _merge_type(
                            known,
                            op.trip.index or "",
                            CanonicalValueType(upper_type.dtype, ()),
                            context=f"IntentOp {op.node_id or op.op} trip.index",
                        )
                for state in op.carry:
                    if state.type is None:
                        continue
                    carry_type = CanonicalValueType.from_value_type(
                        state.type.as_value_type()
                    )
                    for role, name in (
                        ("body_argument", state.body_argument),
                        ("next_value", state.next_value),
                        ("result_value", state.result_value),
                    ):
                        changed |= _merge_type(
                            known,
                            name,
                            carry_type,
                            context=(
                                f"IntentOp {op.node_id or op.op} carry.{role}"
                            ),
                        )
                    if (
                        isinstance(state.init_value, str)
                        and state.init_value in operation_known
                        and operation_known[state.init_value] != carry_type
                    ):
                        raise CanonicalJError(
                            f"IntentOp {op.node_id or op.op!r} carry initializer "
                            f"{state.init_value!r} has type "
                            f"{operation_known[state.init_value].to_json()} but the carry "
                            f"declares {carry_type.to_json()}"
                        )
                # A loop carry is the only Layer-A consumer that can constrain
                # an otherwise untyped literal before its producer is visited.
                # Reuse the emitter's closed dtype table so canonical J and
                # MLIR emission normalize the same loop dataflow.
                from intentir.frontend.emit_mlir import (
                    _align_loop_body_literal_dtypes,
                )

                _align_loop_body_literal_dtypes(
                    op,
                    {
                        name: value.dtype
                        for name, value in known.items()
                        if value.is_tensor and value.dtype is not None
                    },
                )
            names = _result_names(op)
            declared: CanonicalValueType | None = None
            if op.type is not None:
                if not isinstance(op.type, dict):
                    raise CanonicalJError(
                        f"IntentOp {op.node_id or op.op!r} has a partial result type"
                    )
                if op.type.get("kind") in {"record", "ragged"}:
                    try:
                        declared = CanonicalValueType.from_json(
                            op.type,
                            context=f"IntentOp {op.node_id or op.op}.type",
                        )
                    except (TypeError, ValueError) as exc:
                        raise CanonicalJError(str(exc)) from exc
                elif (
                    not isinstance(op.type.get("dtype"), str)
                    or not isinstance(op.type.get("shape"), list)
                ):
                    # Preserve the historical tensor-type validation and error
                    # path for every pre-record operation.
                    raise CanonicalJError(
                        f"IntentOp {op.node_id or op.op!r} has a partial result type"
                    )
                else:
                    try:
                        declared = CanonicalValueType.from_json(
                            op.type,
                            context=f"IntentOp {op.node_id or op.op}.type",
                        )
                    except (TypeError, ValueError) as exc:
                        raise CanonicalJError(str(exc)) from exc
                if len(names) != 1 and declared.is_record:
                    raise CanonicalJError(
                        f"IntentOp {op.node_id or op.op!r} record type requires "
                        "one result binding"
                    )
                if len(names) == 1:
                    changed |= _merge_type(
                        known,
                        names[0],
                        declared,
                        context=f"IntentOp {op.node_id or op.op}",
                    )
                    operation_known[names[0]] = known[names[0]]

            _validate_const_operation(op, operation_known, declared)
            if op.op in {"record", "record_get"}:
                _validate_record_operation(op, operation_known, declared)
            _validate_gather_operation(op, operation_known, declared)
            _validate_ragged_operation(op, operation_known, declared)
            _validate_reshape_operation(op, operation_known, declared)

            inferred_value_type = _infer_ragged_type(op, operation_known)
            if inferred_value_type is not None and len(names) == 1:
                changed |= _merge_type(
                    known,
                    names[0],
                    inferred_value_type,
                    context=f"IntentOp {op.node_id or op.op}",
                )

            inferred_dtypes = _infer_dtypes(op, operation_known)
            inferred_shape = _infer_shape(
                op,
                operation_known,
                broadcast_axes,
                unit_dimensions,
            )
            for name in names:
                previous = operation_known.get(name)
                dtype = inferred_dtypes.get(name) or (
                    previous.dtype if previous is not None else None
                )
                shape = inferred_shape if inferred_shape is not None else (
                    previous.shape if previous is not None else None
                )
                if dtype is None or shape is None:
                    continue
                changed |= _merge_type(
                    known,
                    name,
                    CanonicalValueType(dtype, tuple(shape)),
                    context=f"IntentOp {op.node_id or op.op}",
                )
            if declared is not None and len(names) == 1:
                actual = known.get(names[0])
                if actual is not None and actual != declared:
                    raise CanonicalJError(
                        f"IntentOp {op.node_id or op.op!r} explicit type conflicts "
                        "with its closed operation signature"
                    )
        if not changed:
            break

    for op in operations:
        names = _result_names(op)
        if op.type is None and len(names) == 1 and names[0] in known:
            op.type = known[names[0]].to_json()
    return known


def _node_results(j: Any) -> dict[str, tuple[str, ...]]:
    records = canonical_operation_records(j.intent.to_json())
    if records is None:
        return {}
    by_path = {
        record["path"]: tuple(record["results"])
        for record in records
    }
    result = dict(by_path)
    for node_id, path in j.node_ids.items():
        record = next(
            (item for item in records if item.get("path") == path),
            None,
        )
        if record is None or record.get("id") != node_id:
            raise CanonicalJError(
                f"J node id disagrees with canonical operation record at {path!r}"
            )
        result[node_id] = by_path[path]
    return result


def _declared_broadcast_axes(
    j: Any,
    *,
    operation_records: Sequence[Mapping[str, Any]] | None = None,
    boundary_record: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Resolve only broadcast facts explicitly serialized in canonical J.

    An access-guard relation is the unique access-local authority for its
    select condition.  It is consumed directly here rather than copied into a
    second ``boundary.broadcasts`` declaration.
    """

    # Before the accepted construction exists, the initial operation table is
    # necessarily built from the strict J wire.  Once elaboration has closed,
    # callers must pass that same construction-owned table instead of making a
    # second walk over ``intent.body``.
    records = (
        tuple(operation_records)
        if operation_records is not None
        else canonical_operation_records(j.intent.to_json())
    )
    if records is None:
        raise CanonicalJError(
            "J broadcast projection has no complete operation construction record"
        )
    by_path = {
        record["path"]: record["wire"]
        for record in records
        if isinstance(record.get("wire"), Mapping)
    }
    nodes = dict(by_path)
    for node_id, path in j.node_ids.items():
        operation = by_path.get(path)
        record = next(
            (item for item in records if item.get("path") == path),
            None,
        )
        if operation is None or record is None or record.get("id") != node_id:
            raise CanonicalJError(
                f"J node id disagrees with canonical operation record at {path!r}"
            )
        nodes[node_id] = operation

    if boundary_record is not None:
        boundary = boundary_record
    else:
        boundary_records = canonical_boundary_records(j.boundary.to_json())
        boundary = boundary_records[0] if boundary_records is not None else None
    if boundary is None:
        raise CanonicalJError(
            "J broadcast projection has no complete boundary construction record"
        )

    result: dict[str, dict[str, tuple[int, ...]]] = {}
    guard_projection_keys: set[tuple[str, str]] = set()
    accesses = {
        access.get("access_id"): access
        for access in boundary.get("accesses", [])
        if isinstance(access, Mapping)
    }
    for relation in boundary.get("access_guard_relations", []):
        if not isinstance(relation, Mapping):
            raise CanonicalJError(
                "J boundary guard relation is not a construction record"
            )
        access = accesses.get(relation.get("access_ref"))
        operation = (
            nodes.get(access.get("node_ref"))
            if isinstance(access, Mapping)
            and isinstance(access.get("node_ref"), str)
            else None
        )
        if (
            operation is None
            or operation.get("op") != "select"
            or not isinstance(operation.get("result"), str)
            or not isinstance(operation.get("operands"), list)
            or len(operation["operands"]) != 3
            or operation["operands"][0] != relation.get("value_ref")
        ):
            continue
        axes = tuple(relation.get("axes", ()))
        result_ref = operation["result"]
        value_ref = relation.get("value_ref")
        if not isinstance(value_ref, str):
            continue
        key = (result_ref, value_ref)
        guard_projection_keys.add(key)
        result.setdefault(result_ref, {}).setdefault(
            value_ref, axes
        )

    for projection in boundary.get("broadcasts", []):
        if not isinstance(projection, Mapping):
            raise CanonicalJError(
                "J boundary broadcast is not a construction record"
            )
        operation = nodes.get(projection.get("node_ref"))
        result_ref = operation.get("result") if isinstance(operation, Mapping) else None
        if operation is None or not isinstance(result_ref, str):
            raise CanonicalJError(
                "J boundary broadcast references an unknown single-result node"
            )
        if operation.get("op") not in {"map_binary", "select"}:
            raise CanonicalJError(
                "J boundary broadcast must target map_binary or select"
            )
        operands = operation.get("operands")
        operand = projection.get("operand")
        if not isinstance(operands, list) or operand not in operands:
            raise CanonicalJError(
                "J boundary broadcast operand is not consumed by its node"
            )
        axes_value = projection.get("axes")
        if not isinstance(axes_value, list) or any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in axes_value
        ):
            raise CanonicalJError(
                "canonical value typing requires integer broadcast axes"
            )
        axes = tuple(int(axis) for axis in axes_value)
        if (result_ref, operand) in guard_projection_keys:
            # TypeCheck rejects this double authority.  Keep the relation as
            # the sole typing input so the diagnostic is not pre-empted by a
            # conflicting legacy broadcast spelling.
            continue
        previous = result.setdefault(result_ref, {}).get(operand)
        if previous is not None and previous != axes:
            raise CanonicalJError(
                "J boundary contains conflicting broadcast projections"
            )
        result[result_ref][operand] = axes
    return result


def _remove_redundant_identity_mappings(j: Any, payload: dict[str, Any]) -> None:
    boundary = payload["boundary"]
    mappings = boundary.get("output_mappings", [])
    if not mappings:
        return
    outputs = {tensor.name for tensor in j.intent.outputs}
    # An inout identity mapping is not redundant metadata: it is the authored
    # statement that the post-state of this logical tensor is the observable
    # output.  Active TypeCheck deliberately requires that statement rather
    # than deriving it from the ABI, so canonicalization must preserve it.
    inout_outputs = {
        parameter.logical_ref
        for parameter in j.signature.parameters
        if getattr(parameter.mode, "value", parameter.mode) == "inout"
        and parameter.logical_ref is not None
    }
    node_results = _node_results(j)
    stores = [
        access for access in j.boundary.accesses
        if access.kind.value == "store"
    ]
    retained: list[dict[str, Any]] = []
    for mapping in mappings:
        output = mapping["output"]
        value = mapping["value"]
        node_ref = mapping.get("node_ref")
        matching_stores = [
            access for access in stores
            if access.tensor == output and access.node_ref == node_ref
        ]
        proven_identity = (
            output == value
            and output in outputs
            and output not in inout_outputs
            and (
                node_ref is None
                or (
                    output in node_results.get(node_ref, ())
                    and len(matching_stores) == 1
                )
            )
        )
        if not proven_identity:
            retained.append(mapping)
    boundary["output_mappings"] = retained


def direct_unit_dimension_refs(
    constraints: Sequence[str],
) -> frozenset[str]:
    """Return dimensions that J directly and unambiguously equates to one.

    This is deliberately not an affine solver. It consumes only authored
    ``x* == 1`` or ``1 == x*`` ContractExpr atoms. Conflicting direct integer
    equalities do not authorize singleton broadcast.
    """

    direct_values: dict[str, set[int]] = {}
    for index, constraint in enumerate(constraints):
        expression = parse_contract_expr(
            constraint,
            expected=ContractSort.BOOLEAN,
            context=f"J.dimension_constraints[{index}]",
        )
        if expression.op != "eq" or len(expression.args) != 2:
            continue
        left, right = expression.args
        reference: str | None = None
        literal: int | None = None
        if (
            left.op == "ref"
            and isinstance(left.value, str)
            and right.op == "int"
            and isinstance(right.value, int)
            and not isinstance(right.value, bool)
        ):
            reference = left.value
            literal = right.value
        elif (
            right.op == "ref"
            and isinstance(right.value, str)
            and left.op == "int"
            and isinstance(left.value, int)
            and not isinstance(left.value, bool)
        ):
            reference = right.value
            literal = left.value
        if reference is not None and literal is not None:
            direct_values.setdefault(reference, set()).add(literal)
    return frozenset(
        reference
        for reference, values in direct_values.items()
        if values == {1}
    )


def direct_dimension_lower_bounds(
    constraints: Sequence[str],
) -> dict[str, int]:
    """Return only direct, contradiction-free integer lower bounds.

    This is deliberately smaller than affine or SMT reasoning.  It consumes
    authored conjunction atoms of the forms ``k <= x``, ``k < x``, ``x >= k``,
    ``x > k`` and direct equality with an integer literal.  Bounds hidden
    behind arithmetic, disjunction or another dimension are ignored.  A
    directly contradictory interval authorizes nothing.
    """

    lower: dict[str, int] = {}
    upper: dict[str, int] = {}

    def update_lower(reference: str, value: int) -> None:
        lower[reference] = max(lower.get(reference, value), value)

    def update_upper(reference: str, value: int) -> None:
        upper[reference] = min(upper.get(reference, value), value)

    def consume(expression: ContractExpr) -> None:
        if expression.op == "and":
            for argument in expression.args:
                consume(argument)
            return
        if expression.op not in {"lt", "le", "gt", "ge", "eq"}:
            return
        if len(expression.args) != 2:
            return
        left, right = expression.args
        reference: str | None = None
        literal: int | None = None
        reference_on_left = False
        if (
            left.op == "ref"
            and isinstance(left.value, str)
            and right.op == "int"
            and isinstance(right.value, int)
            and not isinstance(right.value, bool)
        ):
            reference = left.value
            literal = right.value
            reference_on_left = True
        elif (
            right.op == "ref"
            and isinstance(right.value, str)
            and left.op == "int"
            and isinstance(left.value, int)
            and not isinstance(left.value, bool)
        ):
            reference = right.value
            literal = left.value
        if reference is None or literal is None:
            return

        if expression.op == "eq":
            update_lower(reference, literal)
            update_upper(reference, literal)
        elif reference_on_left:
            if expression.op == "ge":
                update_lower(reference, literal)
            elif expression.op == "gt":
                update_lower(reference, literal + 1)
            elif expression.op == "le":
                update_upper(reference, literal)
            elif expression.op == "lt":
                update_upper(reference, literal - 1)
        else:
            if expression.op == "le":
                update_lower(reference, literal)
            elif expression.op == "lt":
                update_lower(reference, literal + 1)
            elif expression.op == "ge":
                update_upper(reference, literal)
            elif expression.op == "gt":
                update_upper(reference, literal - 1)

    for index, constraint in enumerate(constraints):
        consume(
            parse_contract_expr(
                constraint,
                expected=ContractSort.BOOLEAN,
                context=f"J.dimension_constraints[{index}]",
            )
        )

    return {
        reference: value
        for reference, value in lower.items()
        if reference not in upper or value <= upper[reference]
    }


def _snapshot(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _build_output_bindings(
    output_records: Sequence[Mapping[str, Any]],
) -> tuple[CanonicalOutputBinding, ...]:
    return tuple(
        CanonicalOutputBinding(
            record["output"],
            CanonicalValueType.from_json(
                dict(record["output_type"]),
                context=f"canonical output {record['output']}",
            ),
            record["return_value"],
            record["node_ref"],
            record["abi_index"],
            record["abi_parameter_index"],
            record["return_index"],
            record["mapping_index"],
            record["abi_parameter_name"],
            record["abi_parameter_mode"],
            record["mapping_value"],
            record["authored_node_ref"],
            copy.deepcopy(dict(record["producer_record"]))
            if record["producer_record"] is not None
            else None,
        )
        for record in output_records
    )


def _elaborate_j(j: "SemanticArchive") -> tuple["SemanticArchive", JElaboration]:
    """Construct the accepted J and its private parsed cache."""

    # Local import prevents the J wire dataclasses from depending on their
    # elaborated view while they are being defined.
    from intentir.verify.gate1.model import SemanticArchive

    if not isinstance(j, SemanticArchive):
        raise TypeError("elaborate_j requires SemanticArchive J")
    source_payload = j.to_json()
    source_snapshot = _snapshot(source_payload)
    cached = getattr(j, "_intentir_j_elaboration", None)
    if isinstance(cached, JElaboration) and cached._snapshot == source_snapshot:
        return j, cached
    source_cached = getattr(j, "_intentir_elaborated_source", None)
    if (
        isinstance(source_cached, tuple)
        and len(source_cached) == 3
        and source_cached[0] == source_snapshot
        and isinstance(source_cached[1], SemanticArchive)
        and isinstance(source_cached[2], JElaboration)
    ):
        return source_cached[1], source_cached[2]

    payload = copy.deepcopy(source_payload)
    # ``j`` has already crossed the strict Gate1 decoder.  Reusing the accepted
    # object avoids reopening a permissive IntentFunction parser in the middle
    # of canonicalization and keeps returns/type ownership on that one object.
    function = copy.deepcopy(j.intent)
    _elaborate_function_types(
        function,
        _declared_broadcast_axes(j),
        direct_unit_dimension_refs(j.dimension_constraints),
    )
    payload["intent"] = function.to_json()

    boundary = payload["boundary"]
    for domain_id, expression in list(boundary.get("domains", {}).items()):
        parsed = parse_contract_expr(
            expression,
            expected=ContractSort.BOOLEAN,
            context=f"J.boundary.domains.{domain_id}",
        )
        boundary["domains"][domain_id] = parsed.to_text()
    for index, access in enumerate(boundary.get("accesses", [])):
        parsed_index = parse_index_expression(
            access["index_expression"],
            context=f"J.boundary.accesses[{index}].index_expression",
        )
        access["index_expression"] = parsed_index.to_text()
        predicate = access.get("predicate")
        if predicate is not None:
            parsed_predicate = parse_contract_expr(
                predicate,
                expected=ContractSort.BOOLEAN,
                context=f"J.boundary.accesses[{index}].predicate",
            )
            access["predicate"] = parsed_predicate.to_text()

    # Inspect identity against the original node/access objects, then serialize
    # the one canonical omission form.
    _remove_redundant_identity_mappings(j, payload)
    canonical_archive = (
        j if payload == source_payload else SemanticArchive.from_json(payload)
    )
    canonical_payload = canonical_archive.to_json()
    canonical_snapshot = _snapshot(canonical_payload)

    # Build the structural bundle before any parsed boundary view is exposed.
    # Domains and accesses below are projections of this same boundary record,
    # not a second walk over ``SemanticArchive.boundary``.
    operation_records = canonical_operation_records(
        canonical_archive.intent.to_json()
    )
    construction_records = CanonicalConstructionRecords.build(
        canonical_archive.intent.to_json(),
        canonical_archive.boundary.to_json(),
        [scope.to_json() for scope in canonical_archive.scopes],
        canonical_archive.node_ids,
        operation_records=operation_records,
        signature=canonical_archive.signature.to_json(),
        dimension_relations=[
            item.to_json() for item in canonical_archive.dimension_relations
        ],
        value_dimension_relations=[
            item.to_json()
            for item in canonical_archive.value_dimension_relations
        ],
    )
    if construction_records is None or construction_records.boundary_record is None:
        raise CanonicalJError(
            "canonical J has no complete construction record"
        )
    boundary_record = construction_records.boundary_record

    domains = tuple(
        (
            name,
            parse_contract_expr(
                expression,
                expected=ContractSort.BOOLEAN,
                context=f"J.boundary.domains.{name}",
            ),
        )
        for name, expression in boundary_record["domains"].items()
    )
    accesses = tuple(
        CanonicalBoundaryAccess(
            access["access_id"],
            parse_index_expression(
                access["index_expression"],
                context=f"J.boundary.accesses.{access['access_id']}.index_expression",
            ),
            (
                parse_contract_expr(
                    access["predicate"],
                    expected=ContractSort.BOOLEAN,
                    context=f"J.boundary.accesses.{access['access_id']}.predicate",
                )
                if access.get("predicate") is not None else None
            ),
        )
        for access in boundary_record["accesses"]
    )
    typed_function = copy.deepcopy(canonical_archive.intent)
    value_types = tuple(
        sorted(
            _elaborate_function_types(
                typed_function,
                _declared_broadcast_axes(
                    canonical_archive,
                    operation_records=construction_records.operation_records,
                    boundary_record=boundary_record,
                ),
                direct_unit_dimension_refs(
                    canonical_archive.dimension_constraints
                ),
            ).items()
        )
    )
    elaborated_by_name = dict(value_types)
    construction_records = construction_records.with_elaborated_value_types(
        {
            name: value.to_json()
            for name, value in elaborated_by_name.items()
        }
    )
    if construction_records is None:
        raise CanonicalJError(
            "canonical construction value types disagree with elaborated J"
        )
    for operation in construction_records.operation_records:
        for result in operation.get("results", ()):
            if result not in elaborated_by_name:
                raise CanonicalJError(
                    f"canonical operation result {result!r} has no elaborated value type"
                )
    for name, declared in construction_records.value_type_records.items():
        if not isinstance(name, str) or not isinstance(declared, Mapping):
            raise CanonicalJError(
                "canonical construction contains an invalid value-type record"
            )
        try:
            declared_type = CanonicalValueType.from_json(
                dict(declared),
                context=f"canonical construction value type {name}",
            )
        except (TypeError, ValueError) as exc:
            raise CanonicalJError(str(exc)) from exc
        elaborated_type = elaborated_by_name.get(name)
        if elaborated_type is None or elaborated_type != declared_type:
            raise CanonicalJError(
                f"canonical construction value type for {name!r} disagrees "
                "with elaborated J type"
            )
    # An operation may omit its local ``type`` when canonical elaboration can
    # derive it from the closed operation contract.  The output record still
    # has to close that inferred SSA value against the declared output ABI;
    # otherwise output_mapping/return could expose a value whose inferred
    # dtype or shape differs from the output tensor while all explicit type
    # rows above appear consistent.
    for output in construction_records.output_records:
        value = output.get("return_value")
        expected = output.get("output_type")
        actual = elaborated_by_name.get(value)
        if not isinstance(expected, Mapping) or actual is None:
            raise CanonicalJError(
                "canonical output record has no complete elaborated value type"
            )
        try:
            expected_type = CanonicalValueType.from_json(
                dict(expected),
                context=f"canonical output {output.get('output')}",
            )
        except (TypeError, ValueError) as exc:
            raise CanonicalJError(str(exc)) from exc
        if actual != expected_type:
            raise CanonicalJError(
                f"canonical output value {value!r} type disagrees with output "
                f"{output.get('output')!r} ABI type"
            )
    record = JElaboration(
        domains,
        accesses,
        value_types,
        _build_output_bindings(
            construction_records.output_records,
        ),
        construction_records.operation_records,
        construction_records.scope_records,
        canonical_snapshot,
        construction_records,
    )
    setattr(canonical_archive, "_intentir_j_elaboration", record)
    if canonical_archive is not j:
        setattr(
            j,
            "_intentir_elaborated_source",
            (source_snapshot, canonical_archive, record),
        )
    return canonical_archive, record


def elaborate_j(j: "SemanticArchive") -> "SemanticArchive":
    """Decode/normalize a response-form archive into the one accepted J."""

    return _elaborate_j(j)[0]


def j_elaboration(j: "SemanticArchive") -> JElaboration:
    """Return structured facts derived from J; never a second J object."""

    accepted, record = _elaborate_j(j)
    if accepted is not j:
        raise ValueError(
            "j_elaboration requires the accepted J returned by elaborate_j"
        )
    return record


def canonical_top_level_operations(
    j: "SemanticArchive",
) -> tuple[IntentOp, ...]:
    """Return the top-level execution sequence from the shared typed table.

    This is the common execution view for target-neutral consumers.  It keeps
    the lexical operation order and nested-region object identity supplied by
    ``CanonicalConstructionRecords``; callers must not rebuild the sequence
    from ``SemanticArchive.intent.body`` independently.
    """

    rows = j_elaboration(j).typed_operation_rows
    return tuple(
        operation
        for path, operation in rows
        if path.startswith("body[") and "." not in path
    )


def canonical_output_bindings(
    j: "SemanticArchive",
) -> tuple[CanonicalOutputBinding, ...]:
    """Return the accepted J output sequence owned by one output record.

    Consumers must not reconstruct output order from ``intent.outputs``,
    ``returns`` or ``boundary.output_mappings`` independently.  Canonical
    construction has already proved that these rows agree with the physical
    output/inout ABI subsequence; this accessor exposes that one sequence.
    """

    # ``CanonicalConstructionRecords.output_records`` is already emitted in
    # physical ABI order.  Preserve that sequence verbatim; sorting here
    # would create a second output-order authority and could hide a malformed
    # construction bundle.
    return tuple(j_elaboration(j).output_bindings)


def canonical_output_names(
    j: "SemanticArchive",
) -> tuple[str, ...]:
    """Return logical output names in canonical physical ABI order."""

    return tuple(item.output for item in canonical_output_bindings(j))


def canonical_broadcast_axes(
    j: "SemanticArchive",
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return only J-authored broadcast projections in canonical value form."""

    canonical = elaborate_j(j)
    elaboration = j_elaboration(canonical)
    return {
        result: dict(operands)
        for result, operands in _declared_broadcast_axes(
            canonical,
            operation_records=elaboration.construction_records.operation_records,
            boundary_record=elaboration.boundary_record,
        ).items()
    }


def canonicalize_semantic_archive(j: "SemanticArchive") -> "SemanticArchive":
    return elaborate_j(j)


def canonicalize_gate1_artifact(artifact: "Gate1Artifact") -> "Gate1Artifact":
    from intentir.verify.gate1.model import Gate1Artifact

    if not isinstance(artifact, Gate1Artifact):
        raise TypeError("canonicalize_gate1_artifact requires Gate1Artifact")
    archive = elaborate_j(artifact.j)
    if archive is artifact.j:
        return artifact
    return Gate1Artifact(
        archive,
        artifact.c,
        artifact.pi,
        wire_schema=artifact.wire_schema,
    )


__all__ = [
    "CanonicalBoundaryAccess",
    "CanonicalIndex",
    "CanonicalJError",
    "CanonicalOutputBinding",
    "CanonicalValueType",
    "ContractExpr",
    "ContractSort",
    "JElaboration",
    "canonical_broadcast_axes",
    "canonical_top_level_operations",
    "canonical_output_bindings",
    "canonical_output_names",
    "canonicalize_gate1_artifact",
    "canonicalize_semantic_archive",
    "direct_dimension_lower_bounds",
    "direct_unit_dimension_refs",
    "elaborate_j",
    "j_elaboration",
    "parse_contract_expr",
    "parse_index_expression",
]
