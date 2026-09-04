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
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

from .model import (
    IntentFunction,
    IntentOp,
    ValueType,
    validate_loop_wiring,
    validate_ragged_region_boundaries,
    walk_intent_op_paths,
    walk_intent_op_scopes,
    walk_intent_ops,
)
from .intent_op_contract import (
    IntentOpContractError,
    RAGGED_OPS,
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
    value: str
    node_ref: str | None


@dataclass(frozen=True)
class JElaboration:
    """Private structured facts cached on the one accepted J object."""

    domains: tuple[tuple[str, ContractExpr], ...]
    accesses: tuple[CanonicalBoundaryAccess, ...]
    value_types: tuple[tuple[str, CanonicalValueType], ...]
    output_bindings: tuple[CanonicalOutputBinding, ...]
    _snapshot: str

    @property
    def access_by_id(self) -> dict[str, CanonicalBoundaryAccess]:
        return {access.access_id: access for access in self.accesses}

    @property
    def domain_by_id(self) -> dict[str, ContractExpr]:
        return dict(self.domains)

    @property
    def value_type_by_name(self) -> dict[str, CanonicalValueType]:
        return dict(self.value_types)

    def structured_boundary_json(self) -> dict[str, Any]:
        """Diagnostic typed view; it is not an additional persisted artifact."""

        return {
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
                    "value": binding.value,
                    "node_ref": binding.node_ref,
                }
                for binding in self.output_bindings
            ],
        }


def _result_names(op: IntentOp) -> tuple[str, ...]:
    if isinstance(op.result, str):
        return (op.result,)
    if isinstance(op.result, list):
        return tuple(item for item in op.result if isinstance(item, str))
    return ()


def _walk_ops(body: Sequence[IntentOp]) -> Iterable[IntentOp]:
    yield from walk_intent_ops(body)


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

    from intentir.frontend.emit_mlir import _broadcast_projection

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
            raise CanonicalJError(
                f"IntentOp {op.node_id or op.op!r} operands have incompatible "
                "NumPy broadcast shapes; a non-trailing broadcast must be "
                "declared explicitly in J.boundary.broadcasts"
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
    if op.op in {"reduce", "arg_reduce"} and operand_shapes:
        axes_value = op.attrs.get("axes", [op.attrs.get("axis")])
        if isinstance(axes_value, list) and all(
            isinstance(axis, int) and not isinstance(axis, bool)
            for axis in axes_value
        ):
            source = operand_shapes[0]
            axes = {axis if axis >= 0 else len(source) + axis for axis in axes_value}
            if any(axis < 0 or axis >= len(source) for axis in axes):
                return None
            if op.attrs.get("keepdims") is True:
                return tuple(1 if index in axes else dim for index, dim in enumerate(source))
            return tuple(dim for index, dim in enumerate(source) if index not in axes)
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
                f"{expected_index_shape!r}"
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
    by_path = {
        path: _result_names(operation)
        for path, operation in walk_intent_op_paths(j.intent.body)
    }
    result = dict(by_path)
    for node_id, path in j.node_ids.items():
        if path in by_path:
            result[node_id] = by_path[path]
    return result


def _declared_broadcast_axes(
    j: Any,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Resolve only broadcast facts explicitly serialized in canonical J.

    An access-guard relation is the unique access-local authority for its
    select condition.  It is consumed directly here rather than copied into a
    second ``boundary.broadcasts`` declaration.
    """

    by_path = dict(walk_intent_op_paths(j.intent.body))
    nodes = dict(by_path)
    for node_id, path in j.node_ids.items():
        operation = by_path.get(path)
        if operation is not None:
            nodes[node_id] = operation

    result: dict[str, dict[str, tuple[int, ...]]] = {}
    guard_projection_keys: set[tuple[str, str]] = set()
    accesses = {
        access.access_id: access for access in j.boundary.accesses
    }
    for relation in j.boundary.access_guard_relations:
        access = accesses.get(relation.access_ref)
        operation = (
            nodes.get(access.node_ref)
            if access is not None and access.node_ref is not None
            else None
        )
        if (
            operation is None
            or operation.op != "select"
            or not isinstance(operation.result, str)
            or len(operation.operands) != 3
            or operation.operands[0] != relation.value_ref
        ):
            continue
        axes = tuple(relation.axes)
        key = (operation.result, relation.value_ref)
        guard_projection_keys.add(key)
        result.setdefault(operation.result, {}).setdefault(
            relation.value_ref, axes
        )

    for projection in j.boundary.broadcasts:
        operation = nodes.get(projection.node_ref)
        if operation is None or not isinstance(operation.result, str):
            raise CanonicalJError(
                "J boundary broadcast references an unknown single-result node"
            )
        if operation.op not in {"map_binary", "select"}:
            raise CanonicalJError(
                "J boundary broadcast must target map_binary or select"
            )
        if projection.operand not in operation.operands:
            raise CanonicalJError(
                "J boundary broadcast operand is not consumed by its node"
            )
        if any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in projection.axes
        ):
            raise CanonicalJError(
                "canonical value typing requires integer broadcast axes"
            )
        axes = tuple(int(axis) for axis in projection.axes)
        if (operation.result, projection.operand) in guard_projection_keys:
            # TypeCheck rejects this double authority.  Keep the relation as
            # the sole typing input so the diagnostic is not pre-empted by a
            # conflicting legacy broadcast spelling.
            continue
        previous = result.setdefault(operation.result, {}).get(
            projection.operand
        )
        if previous is not None and previous != axes:
            raise CanonicalJError(
                "J boundary contains conflicting broadcast projections"
            )
        result[operation.result][projection.operand] = axes
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


def _build_output_bindings(j: Any) -> tuple[CanonicalOutputBinding, ...]:
    returns = {item.output: item.value for item in j.intent.returns}
    mappings = {item.output: item for item in j.boundary.output_mappings}
    node_results = _node_results(j)
    result_nodes: dict[str, list[str]] = {}
    for node_ref, results in node_results.items():
        if not node_ref.startswith("n"):
            continue
        for result in results:
            result_nodes.setdefault(result, []).append(node_ref)
    bindings: list[CanonicalOutputBinding] = []
    for output in j.intent.outputs:
        mapping = mappings.get(output.name)
        try:
            returned = returns[output.name]
        except KeyError as exc:
            raise CanonicalJError(
                f"output {output.name!r} has no complete IntentFunction return"
            ) from exc
        if mapping is not None and mapping.value != returned:
            raise CanonicalJError(
                f"output {output.name!r} has conflicting return and boundary mapping"
            )
        node_ref = mapping.node_ref if mapping is not None else None
        if node_ref is None:
            candidates = sorted(set(result_nodes.get(returned, [])))
            if len(candidates) == 1:
                node_ref = candidates[0]
        bindings.append(CanonicalOutputBinding(output.name, returned, node_ref))
    return tuple(bindings)


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

    domains = tuple(
        (
            name,
            parse_contract_expr(
                expression,
                expected=ContractSort.BOOLEAN,
                context=f"J.boundary.domains.{name}",
            ),
        )
        for name, expression in canonical_archive.boundary.domains.items()
    )
    accesses = tuple(
        CanonicalBoundaryAccess(
            access.access_id,
            parse_index_expression(
                access.index_expression,
                context=f"J.boundary.accesses.{access.access_id}.index_expression",
            ),
            (
                parse_contract_expr(
                    access.predicate,
                    expected=ContractSort.BOOLEAN,
                    context=f"J.boundary.accesses.{access.access_id}.predicate",
                )
                if access.predicate is not None else None
            ),
        )
        for access in canonical_archive.boundary.accesses
    )
    typed_function = copy.deepcopy(canonical_archive.intent)
    value_types = tuple(
        sorted(
            _elaborate_function_types(
                typed_function,
                _declared_broadcast_axes(canonical_archive),
                direct_unit_dimension_refs(
                    canonical_archive.dimension_constraints
                ),
            ).items()
        )
    )
    record = JElaboration(
        domains,
        accesses,
        value_types,
        _build_output_bindings(canonical_archive),
        canonical_snapshot,
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


def canonical_broadcast_axes(
    j: "SemanticArchive",
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return only J-authored broadcast projections in canonical value form."""

    canonical = elaborate_j(j)
    return {
        result: dict(operands)
        for result, operands in _declared_broadcast_axes(canonical).items()
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
    "canonicalize_gate1_artifact",
    "canonicalize_semantic_archive",
    "direct_dimension_lower_bounds",
    "direct_unit_dimension_refs",
    "elaborate_j",
    "j_elaboration",
    "parse_contract_expr",
    "parse_index_expression",
]
