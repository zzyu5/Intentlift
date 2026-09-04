"""Restricted J-only NumPy denotation for structured Layer-A value semantics.

This is a Gate2 reference executor, not the production CPU backend and not a
fallback for arbitrary IntentOps.  It accepts only a closed operation table,
executes each structured loop over its complete logical trip axis using bounded
J-type-derived scan slices, and validates every declared input, carry, result,
and output type.  Its API deliberately has no C or pi parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from intentir.backend.cpu.abi import (
    CPUBackendUnsupportedError,
    numpy_dtype,
    validate_array,
)
from intentir.backend.cpu.pipeline import CPUExecutionResult
from intentir.ir.intent_op_contract import (
    ATOMIC_CAS_RETRIES,
    ATOMIC_KINDS,
    ATOMIC_MEMORY_SCOPES,
    ATOMIC_ORDERINGS,
    map_binary_result_dtype,
    map_unary_result_dtype,
    RAGGED_OPS,
    validate_const_literal,
)
from intentir.ir.canonical_j import (
    CanonicalValueType,
    elaborate_j,
    j_elaboration,
)
from intentir.ir.dimension_relation import resolve_dimension_relations
from intentir.ir.value_dimension import (
    RaggedRuntimeError,
    ragged_member_length,
    ragged_outer_count,
    validate_ragged_runtime_invocation,
    validate_ragged_runtime_carrier,
    validate_value_dimension_instantiation,
)
from intentir.ir.model import (
    IntentOp,
    LoopStateBinding,
    validate_loop_wiring,
    validate_ragged_region_boundaries,
    walk_intent_op_paths,
    walk_intent_ops,
)

if TYPE_CHECKING:
    from intentir.verify.gate1.model import SemanticArchive


STRUCTURED_CPU_REFERENCE_PROTOCOL = "intentir.j-structured-cpu-reference.v10"

# This is an executor resource bound, not a semantic tile and not a C value.
# A loop slice is chosen solely from exact J result types so no individual
# trip-axis-bearing temporary exceeds this many bytes.
_MAX_REFERENCE_TEMPORARY_BYTES = 128 * 1024 * 1024


class StructuredReferenceUnsupportedError(CPUBackendUnsupportedError):
    """The typed J lies outside this deliberately narrow denotation."""


@dataclass(frozen=True)
class _RaggedValue:
    """Closed runtime value for one offsets/indices descriptor."""

    offsets: np.ndarray
    indices: np.ndarray


def _validate_ragged_arrays(
    offsets: Any,
    indices: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    offsets_array = np.asarray(offsets)
    indices_array = np.asarray(indices)
    try:
        validate_ragged_runtime_carrier(
            offsets_array,
            indices_array,
            context=context,
        )
    except RaggedRuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return offsets_array, indices_array


_ORDINARY_OPS = frozenset(
    {
        "cast",
        "concat",
        "const",
        "contraction",
        "gather",
        "iota",
        "map_binary",
        "map_unary",
        "pad",
        "reduce",
        "reshape",
        "scan",
        "scatter",
        "select",
        "tile",
        "transpose",
    }
) | RAGGED_OPS

_EINSUM_TERM = re.compile(r"^[A-Za-z]*$")

def _isclose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Closed J ``isclose`` predicate with the lowering constants."""

    compute_dtype = np.result_type(np.asarray(left).dtype, np.asarray(right).dtype)
    left_value = np.asarray(left, dtype=compute_dtype)
    right_value = np.asarray(right, dtype=compute_dtype)
    return np.less_equal(
        np.abs(left_value - right_value),
        np.asarray(1e-8, dtype=compute_dtype)
        + np.asarray(1e-5, dtype=compute_dtype) * np.abs(right_value),
    )


_BINARY = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "floordiv": np.floor_divide,
    "remainder": np.remainder,
    "left_shift": np.left_shift,
    "right_shift": np.right_shift,
    "max": np.maximum,
    "min": np.minimum,
    "pow": np.power,
    "atan2": np.arctan2,
    "copysign": np.copysign,
    "isclose": _isclose,
    "gt": np.greater,
    "ge": np.greater_equal,
    "lt": np.less,
    "le": np.less_equal,
    "eq": np.equal,
    "ne": np.not_equal,
    "and": np.logical_and,
    "or": np.logical_or,
    "xor": np.logical_xor,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_xor": np.logical_xor,
}


def _integer_divide(
    left: np.ndarray,
    right: np.ndarray,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Implement MLIR ``arith.divsi/divui`` without a float round-trip."""

    lhs, rhs = np.broadcast_arrays(
        np.asarray(left, dtype=dtype),
        np.asarray(right, dtype=dtype),
    )
    if bool(np.any(rhs == 0)):
        raise ValueError("structured integer division by zero")
    if np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        if bool(np.any((lhs == info.min) & (rhs == -1))):
            raise ValueError("structured signed integer division overflows")
        quotient = np.floor_divide(lhs, rhs)
        remainder = np.remainder(lhs, rhs)
        adjust = ((lhs < 0) != (rhs < 0)) & (remainder != 0)
        return np.asarray(quotient + adjust.astype(dtype), dtype=dtype)
    return np.asarray(np.floor_divide(lhs, rhs), dtype=dtype)


def _integer_floor_divide(
    left: np.ndarray,
    right: np.ndarray,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Implement mathematical floor division for signed integer tensors."""

    lhs, rhs = np.broadcast_arrays(
        np.asarray(left, dtype=dtype),
        np.asarray(right, dtype=dtype),
    )
    if bool(np.any(rhs == 0)):
        raise ValueError("structured floor division by zero")
    if np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        if bool(np.any((lhs == info.min) & (rhs == -1))):
            raise ValueError("structured signed floor division overflows")
    return np.asarray(np.floor_divide(lhs, rhs), dtype=dtype)


def _integer_remainder(
    left: np.ndarray,
    right: np.ndarray,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Implement the divisor-sign remainder paired with floor division."""

    lhs, rhs = np.broadcast_arrays(
        np.asarray(left, dtype=dtype),
        np.asarray(right, dtype=dtype),
    )
    if bool(np.any(rhs == 0)):
        raise ValueError("structured remainder by zero")
    return np.asarray(np.remainder(lhs, rhs), dtype=dtype)


def _erf(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    return np.asarray(
        np.vectorize(math.erf, otypes=[float])(array), dtype=array.dtype
    )


def _erfc(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    return np.asarray(
        np.vectorize(math.erfc, otypes=[float])(array), dtype=array.dtype
    )


def _round_away_from_zero(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    magnitude = np.floor(np.abs(array) + np.asarray(0.5, dtype=array.dtype))
    return np.copysign(magnitude, array).astype(array.dtype, copy=False)


_UNARY = {
    "identity": lambda value: value,
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "sqrt": np.sqrt,
    "rsqrt": lambda value: np.reciprocal(np.sqrt(value)),
    "neg": np.negative,
    "recip": np.reciprocal,
    "abs": np.abs,
    "tanh": np.tanh,
    "sigmoid": lambda value: np.reciprocal(
        np.asarray(1, dtype=value.dtype) + np.exp(-value)
    ),
    "cos": np.cos,
    "sin": np.sin,
    "tan": np.tan,
    "exp2": np.exp2,
    "atan": np.arctan,
    "acos": np.arccos,
    "asin": np.arcsin,
    "cbrt": np.cbrt,
    "ceil": np.ceil,
    "floor": np.floor,
    "round": _round_away_from_zero,
    "roundeven": np.rint,
    "trunc": np.trunc,
    "sign": np.sign,
    "log10": np.log10,
    "log1p": np.log1p,
    "expm1": np.expm1,
    "erf": _erf,
    "erfc": _erfc,
    "isnan": np.isnan,
    "isinf": np.isinf,
    "isfinite": np.isfinite,
    "logical_not": np.logical_not,
}

_CARRY_LITERALS: dict[str, Any] = {
    "NEG_INF": -np.inf,
    "POS_INF": np.inf,
    "ZERO": 0,
    "zeros": 0,
    "ONE": 1,
    "ones": 1,
    "false": False,
    "true": True,
}


def _runtime_const_literal(value: Any) -> Any:
    """Map the closed J special-float spellings to NumPy scalar values."""

    if value == "-inf":
        return -np.inf
    if value == "inf":
        return np.inf
    if value == "nan":
        return np.nan
    return value


def _walk_ops(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    return tuple(walk_intent_ops(body))


def _walk_op_paths(
    body: Sequence[IntentOp],
    prefix: str = "body",
) -> tuple[tuple[str, IntentOp], ...]:
    """Walk operations using the canonical paths named by ``J.node_ids``."""

    return tuple(walk_intent_op_paths(body, prefix))


def _result_names(operation: IntentOp) -> tuple[str, ...]:
    if isinstance(operation.result, str):
        return (operation.result,)
    if isinstance(operation.result, list):
        return tuple(
            item for item in operation.result if isinstance(item, str)
        )
    return ()


def requires_structured_reference(j: SemanticArchive) -> bool:
    """Return whether J needs the J-only typed reference denotation.

    Structured control/tile/scan and every contraction select one closed
    evaluator for the complete J graph.  In particular, a dense contraction is
    not sent through the scalar native loop executor merely because that
    lowering is technically legal; the structured path preserves the same J
    denotation while using bounded NumPy/BLAS implementations for model extents.
    """

    accepted = elaborate_j(j)
    operations = _walk_ops(accepted.intent.body)
    return any(
        op.op in {
            "contraction",
            "loop",
            "scan",
            "tile",
            "effect_if",
            "value_if",
            "gather",
            "scatter",
        } | RAGGED_OPS
        for op in operations
    )


def _resolved_shape(shape: Sequence[Any], dims: Mapping[str, int]) -> tuple[int, ...]:
    result: list[int] = []
    for dimension in shape:
        if isinstance(dimension, bool):
            raise ValueError("structured reference shape contains a boolean extent")
        if isinstance(dimension, int):
            extent = dimension
        elif isinstance(dimension, str) and dimension in dims:
            extent = dims[dimension]
        else:
            raise StructuredReferenceUnsupportedError(
                f"structured reference cannot resolve shape extent {dimension!r}"
            )
        if extent < 0:
            raise ValueError("structured reference shape contains a negative extent")
        result.append(int(extent))
    return tuple(result)


def _resolved_declared_shape(
    shape: Sequence[Any],
    dims: Mapping[str, int],
    actual_shape: Sequence[int] | None,
) -> tuple[int, ...] | None:
    """Resolve a declared shape, using ``?`` only against an actual value."""

    if "?" not in shape:
        return _resolved_shape(shape, dims)
    if actual_shape is None or len(actual_shape) != len(shape):
        return None
    resolved: list[int] = []
    for index, dimension in enumerate(shape):
        if dimension == "?":
            extent = actual_shape[index]
            if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
                raise ValueError("structured dynamic shape contains an invalid extent")
            resolved.append(extent)
            continue
        resolved.append(_resolved_shape((dimension,), dims)[0])
    return tuple(resolved)


class _Evaluator:
    def __init__(
        self,
        j: SemanticArchive,
        dims: Mapping[str, int],
        broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]],
        access_guard_relations: Sequence[Any] = (),
        conditional_effect_stores: Sequence[Any] = (),
    ) -> None:
        self.j = j
        self.dims = dict(dims)
        self.broadcast_axes = {
            result: {
                operand: tuple(int(axis) for axis in axes)
                for operand, axes in operands.items()
            }
            for result, operands in broadcast_axes.items()
        }
        self.value_types = j_elaboration(j).value_type_by_name
        self.access_guard_loads: dict[tuple[str, int], Any] = {}
        for relation in access_guard_relations:
            if getattr(relation, "access_kind", None) != "load":
                continue
            node_ref = getattr(relation, "node_ref", None)
            operand_index = getattr(relation, "operand_index", None)
            tensor = getattr(relation, "tensor", None)
            if (
                not isinstance(node_ref, str)
                or isinstance(operand_index, bool)
                or not isinstance(operand_index, int)
                or operand_index < 0
                or not isinstance(tensor, str)
            ):
                raise ValueError(
                    "structured guarded-load receipt is not mechanically closed"
                )
            key = (node_ref, operand_index)
            if key in self.access_guard_loads:
                raise ValueError(
                    "structured guarded-load receipts duplicate one operand edge"
                )
            self.access_guard_loads[key] = relation
        refs_by_path: dict[str, str] = {}
        for node_ref, path in j.node_ids.items():
            previous = refs_by_path.setdefault(path, node_ref)
            if previous != node_ref:
                raise ValueError("structured reference has duplicate node paths")
        nodes: dict[str, IntentOp] = {}
        self.node_refs_by_operation: dict[int, str] = {}
        for path, operation in _walk_op_paths(j.intent.body):
            authored_ref = refs_by_path.get(path)
            if (
                operation.node_id is not None
                and authored_ref is not None
                and operation.node_id != authored_ref
            ):
                raise ValueError("structured inline and authored node refs disagree")
            node_ref = operation.node_id or authored_ref
            if node_ref is None:
                continue
            previous = nodes.setdefault(node_ref, operation)
            if previous is not operation:
                raise ValueError("structured reference has duplicate node refs")
            self.node_refs_by_operation[id(operation)] = node_ref
        self.logical_updates: dict[str, Any] = {}
        for update in j.boundary.updates:
            operation = nodes.get(update.node_ref)
            if operation is None or operation.op != "scatter":
                raise ValueError(
                    "structured logical update does not name one scatter node"
                )
            if update.node_ref in self.logical_updates:
                raise ValueError(
                    "structured logical updates duplicate one scatter node"
                )
            self.logical_updates[update.node_ref] = update
        output_names = {tensor.name for tensor in j.intent.outputs}
        self.conditional_effect_stores: dict[str, list[Any]] = {}
        seen_accesses: set[str] = set()
        for relation in conditional_effect_stores:
            access_id = getattr(relation, "access_id", None)
            node_ref = getattr(relation, "node_ref", None)
            output = getattr(relation, "output", None)
            value_ref = getattr(relation, "value_ref", None)
            operation = nodes.get(node_ref)
            if (
                not isinstance(access_id, str)
                or not isinstance(node_ref, str)
                or not isinstance(output, str)
                or not isinstance(value_ref, str)
                or operation is None
                or value_ref not in _result_names(operation)
                or output not in output_names
            ):
                raise ValueError(
                    "structured conditional-store receipt is not mechanically closed"
                )
            if access_id in seen_accesses:
                raise ValueError(
                    "structured conditional-store receipts duplicate one access"
                )
            seen_accesses.add(access_id)
            self.conditional_effect_stores.setdefault(node_ref, []).append(relation)
        self.effect_output_state: dict[str, np.ndarray] = {}
        self.executed_conditional_effect_store_count = 0
        self.operation_count = 0
        self.scan_count = 0
        self.scatter_count = 0
        self.loop_count = 0
        self.effect_if_count = 0
        self.value_if_count = 0
        self.logical_loop_steps = 0
        self.zero_trip_loops = 0
        self.max_scan_tile_elements = 0

    def _record_conditional_effect_stores(
        self,
        operation: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> None:
        node_ref = self.node_refs_by_operation.get(id(operation), operation.node_id)
        if node_ref is None:
            return
        for relation in self.conditional_effect_stores.get(node_ref, ()):
            try:
                value = env[relation.value_ref]
            except KeyError as exc:
                raise ValueError(
                    "structured conditional store producer is unavailable"
                ) from exc
            declared = self._declared_type(relation.output)
            self.effect_output_state[relation.output] = np.array(
                self._validate_value(
                    value,
                    name=relation.output,
                    declared=declared,
                ),
                copy=True,
            )
            self.executed_conditional_effect_store_count += 1

    def _declared_type(
        self,
        name: str,
        op: IntentOp | None = None,
        *,
        actual_shape: Sequence[int] | None = None,
    ) -> tuple[str, tuple[int, ...]] | None:
        known = self.value_types.get(name)
        if known is not None:
            if not known.is_tensor or known.dtype is None:
                return None
            shape = _resolved_declared_shape(known.shape, self.dims, actual_shape)
            if shape is None:
                return None
            return known.dtype, shape
        if op is not None and isinstance(op.type, dict):
            if op.type.get("kind") in {"record", "ragged"}:
                return None
            dtype = op.type.get("dtype")
            shape = op.type.get("shape")
            if isinstance(dtype, str) and isinstance(shape, list):
                resolved = _resolved_declared_shape(shape, self.dims, actual_shape)
                if resolved is None:
                    return None
                return dtype, resolved
        return None

    def _declared_value_type(
        self,
        name: str,
        op: IntentOp | None = None,
    ) -> CanonicalValueType | None:
        known = self.value_types.get(name)
        if known is not None:
            return known
        if op is not None and isinstance(op.type, dict):
            return CanonicalValueType.from_json(
                op.type,
                context=f"structured IntentOp {op.op} type",
            )
        return None

    def _validate_value(
        self,
        value: Any,
        *,
        name: str,
        declared: tuple[str, tuple[int, ...]] | CanonicalValueType | None,
    ) -> Any:
        if isinstance(value, _RaggedValue):
            if declared is None:
                return value
            if not isinstance(declared, CanonicalValueType) or not declared.is_ragged:
                raise ValueError(f"structured value {name!r} has a ragged value")
            offsets, indices = _validate_ragged_arrays(
                value.offsets,
                value.indices,
                context=f"structured value {name!r}",
            )
            for field_name, field_value in (
                ("offsets", offsets),
                ("indices", indices),
            ):
                field_type = declared.field(field_name)
                if not field_type.is_tensor or field_type.dtype is None:
                    raise ValueError(
                        f"structured ragged field {field_name!r} is not a tensor"
                    )
                field_shape = _resolved_shape(field_type.shape, self.dims)
                if field_value.shape != field_shape:
                    raise ValueError(
                        f"structured ragged field {field_name!r} shape mismatch"
                    )
                if field_value.dtype != numpy_dtype(field_type.dtype):
                    raise ValueError(
                        f"structured ragged field {field_name!r} dtype mismatch"
                    )
            return _RaggedValue(offsets, indices)
        if isinstance(declared, CanonicalValueType):
            if not declared.is_tensor or declared.dtype is None:
                raise ValueError(f"structured value {name!r} is not a tensor")
            actual_shape = np.asarray(value).shape
            resolved_shape = _resolved_declared_shape(
                declared.shape,
                self.dims,
                actual_shape,
            )
            if resolved_shape is None:
                raise ValueError(f"structured value {name!r} shape is unresolved")
            declared = (declared.dtype, resolved_shape)
        array = np.asarray(value)
        if declared is None:
            return array
        dtype, shape = declared
        expected_dtype = numpy_dtype(dtype)
        if array.shape != shape:
            raise ValueError(
                f"structured value {name!r} shape mismatch: expected {shape}, "
                f"got {array.shape}"
            )
        if array.dtype != expected_dtype:
            raise ValueError(
                f"structured value {name!r} dtype mismatch: expected "
                f"{expected_dtype}, got {array.dtype}"
            )
        return array

    def _result_name(self, op: IntentOp) -> str:
        if not isinstance(op.result, str) or not op.result:
            raise StructuredReferenceUnsupportedError(
                f"structured reference requires one result for op {op.op!r}"
            )
        return op.result

    def _operand(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
        index: int,
    ) -> Any:
        try:
            name = op.operands[index]
            value = env[name]
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"structured op {op.op!r} has an unavailable operand"
            ) from exc
        node_ref = self.node_refs_by_operation.get(id(op), op.node_id)
        materialization = self.access_guard_loads.get((node_ref, index))
        if isinstance(value, _RaggedValue) and materialization is not None:
            raise StructuredReferenceUnsupportedError(
                "ragged descriptors cannot participate in guarded-load "
                "materialization"
            )
        if materialization is not None:
            if name != materialization.tensor:
                raise ValueError(
                    "structured guarded-load receipt names a different operand"
                )
            try:
                condition = np.asarray(env[materialization.value_ref])
            except KeyError as exc:
                raise ValueError(
                    "structured guarded-load guard does not dominate its consumer"
                ) from exc
            if condition.dtype != np.dtype(np.bool_):
                raise ValueError("structured guarded-load guard must have dtype i1")
            for axis in sorted(materialization.axes):
                if axis < 0 or axis > condition.ndim:
                    raise ValueError(
                        "structured guarded-load broadcast axis is outside target rank"
                    )
                condition = np.expand_dims(condition, axis=axis)
            if materialization.polarity == "positive":
                active = condition
            elif materialization.polarity == "negative":
                active = np.logical_not(condition)
            else:
                raise ValueError("structured guarded-load polarity is unknown")
            try:
                fill = np.asarray(
                    materialization.fill_value,
                    dtype=value.dtype,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise StructuredReferenceUnsupportedError(
                    "structured guarded-load fill has no typed NumPy denotation"
                ) from exc
            value = np.asarray(np.where(active, value, fill))
            if value.shape != env[name].shape or value.dtype != env[name].dtype:
                raise ValueError(
                    "structured guarded-load materialization changed operand type"
                )
        result_names = _result_names(op)
        if not result_names:
            raise StructuredReferenceUnsupportedError(
                f"structured op {op.op!r} requires at least one result"
            )
        result = result_names[0]
        axes = self.broadcast_axes.get(result, {}).get(name, ())
        if isinstance(value, _RaggedValue):
            if materialization is not None or axes:
                raise StructuredReferenceUnsupportedError(
                    "ragged descriptors cannot participate in tensor broadcast "
                    "or guarded-load materialization"
                )
            return value
        projected = value
        for axis in sorted(axes):
            if axis < 0 or axis > projected.ndim:
                raise ValueError("structured broadcast axis is outside result rank")
            projected = np.expand_dims(projected, axis=axis)
        return projected

    def _reduce(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) - {"fn", "axes", "assoc", "keepdims"}:
            raise StructuredReferenceUnsupportedError(
                "structured reduce has unsupported control attributes"
            )
        value = self._operand(op, env, 0)
        raw_axes = op.attrs.get("axes")
        if not isinstance(raw_axes, list) or not raw_axes:
            raise StructuredReferenceUnsupportedError(
                "structured reduce requires explicit non-empty axes"
            )
        axes = tuple(int(axis) for axis in raw_axes)
        if any(axis < 0 or axis >= value.ndim for axis in axes):
            raise ValueError("structured reduce axis is outside operand rank")
        expected = self._declared_type(self._result_name(op), op)
        keepdims = bool(op.attrs.get("keepdims", False))
        if expected is not None:
            target_rank = len(expected[1])
            if target_rank == value.ndim:
                keepdims = True
            elif target_rank != value.ndim - len(set(axes)):
                raise ValueError("structured reduce result rank is not closed")
        fn = op.attrs.get("fn")
        if fn == "sum":
            return np.sum(value, axis=axes, keepdims=keepdims, dtype=value.dtype)
        if fn == "prod":
            return np.prod(value, axis=axes, keepdims=keepdims, dtype=value.dtype)
        if fn == "max":
            return np.max(value, axis=axes, keepdims=keepdims)
        if fn == "min":
            return np.min(value, axis=axes, keepdims=keepdims)
        if fn == "mean":
            if value.dtype.kind != "f":
                raise StructuredReferenceUnsupportedError(
                    "structured integer mean has no closed reference policy"
                )
            return np.mean(value, axis=axes, keepdims=keepdims, dtype=value.dtype)
        if fn == "any":
            return np.any(value, axis=axes, keepdims=keepdims)
        if fn == "all":
            return np.all(value, axis=axes, keepdims=keepdims)
        raise StructuredReferenceUnsupportedError(
            f"structured reduce has unsupported fn {fn!r}"
        )

    def _scan(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"fn", "axis"} or len(op.operands) != 1:
            raise StructuredReferenceUnsupportedError(
                "structured scan requires exactly forward inclusive fn/axis control"
            )
        value = self._operand(op, env, 0)
        axis = op.attrs.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise StructuredReferenceUnsupportedError(
                "structured scan axis must be a closed integer"
            )
        if axis < 0 or axis >= value.ndim:
            raise ValueError("structured scan axis is outside operand rank")
        self.scan_count += 1
        fn = op.attrs.get("fn")
        if fn == "sum":
            return np.cumsum(value, axis=axis, dtype=value.dtype)
        if fn == "max":
            return np.maximum.accumulate(value, axis=axis)
        if fn == "min":
            return np.minimum.accumulate(value, axis=axis)
        raise StructuredReferenceUnsupportedError(
            f"structured scan has unsupported fn {fn!r}"
        )

    def _concat(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"axis"} or not op.operands:
            raise StructuredReferenceUnsupportedError(
                "structured concat requires one axis and at least one operand"
            )
        axis = op.attrs.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise StructuredReferenceUnsupportedError(
                "structured concat axis must be a closed integer"
            )
        values = [self._operand(op, env, index) for index in range(len(op.operands))]
        rank = values[0].ndim
        if axis < 0 or axis >= rank:
            raise ValueError("structured concat axis is outside operand rank")
        if any(value.ndim != rank for value in values[1:]):
            raise ValueError("structured concat operands must have one common rank")
        if any(value.dtype != values[0].dtype for value in values[1:]):
            raise ValueError("structured concat operands must have one common dtype")
        for dimension in range(rank):
            if dimension == axis:
                continue
            if any(
                value.shape[dimension] != values[0].shape[dimension]
                for value in values[1:]
            ):
                raise ValueError(
                    "structured concat operands disagree at a non-concat dimension"
                )
        expected = self._declared_type(self._result_name(op), op)
        if expected is None:
            raise StructuredReferenceUnsupportedError(
                "structured concat requires one closed result type"
            )
        expected_dtype, expected_shape = expected
        inferred_shape = list(values[0].shape)
        inferred_shape[axis] = sum(value.shape[axis] for value in values)
        if expected_shape != tuple(inferred_shape):
            raise ValueError("structured concat result shape is not the axis sum")
        if numpy_dtype(expected_dtype) != values[0].dtype:
            raise ValueError("structured concat result dtype differs from its inputs")
        return np.concatenate(values, axis=axis)

    def _gather(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"axis", "mode"} or len(op.operands) != 2:
            raise StructuredReferenceUnsupportedError(
                "structured gather requires exactly axis/mode and two operands"
            )
        data = self._operand(op, env, 0)
        index = self._operand(op, env, 1)
        axis = op.attrs.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise StructuredReferenceUnsupportedError(
                "structured gather axis must be a closed integer"
            )
        if axis < 0 or axis >= data.ndim:
            raise ValueError("structured gather axis is outside data rank")
        if index.dtype.kind not in {"i", "u"}:
            raise ValueError("structured gather index must have an integer dtype")
        if index.size and (
            bool(np.any(index < 0))
            or bool(np.any(index >= data.shape[axis]))
        ):
            raise ValueError("structured gather index is outside the data extent")
        declared_value = self._declared_value_type(self._result_name(op), op)
        if (
            declared_value is None
            or not declared_value.is_tensor
            or declared_value.dtype is None
        ):
            raise StructuredReferenceUnsupportedError(
                "structured gather requires one closed result type"
            )
        if numpy_dtype(declared_value.dtype) != data.dtype:
            raise ValueError("structured gather result dtype differs from data")

        def expected_shape(actual_shape: Sequence[int]) -> tuple[int, ...]:
            expected = self._declared_type(
                self._result_name(op),
                op,
                actual_shape=actual_shape,
            )
            if expected is None:
                raise StructuredReferenceUnsupportedError(
                    "structured gather result shape is unresolved"
                )
            return expected[1]

        mode = op.attrs.get("mode")
        if mode == "along_axis":
            if index.ndim == data.ndim - 1:
                if index.ndim == 0:
                    raise StructuredReferenceUnsupportedError(
                        "structured compressed along_axis gather does not yet "
                        "support a rank-0 index/result"
                    )
                non_axis_shape = (*data.shape[:axis], *data.shape[axis + 1 :])
                if index.shape != non_axis_shape:
                    raise ValueError(
                        "structured compressed along_axis gather index must "
                        "match every non-axis data extent"
                    )
                if expected_shape(index.shape) != index.shape:
                    raise ValueError(
                        "structured compressed along_axis gather result must be "
                        "shaped like index"
                    )
                expanded = np.expand_dims(index, axis=axis)
                gathered = np.take_along_axis(data, expanded, axis=axis)
                return np.squeeze(gathered, axis=axis)
            if index.ndim != data.ndim:
                raise ValueError(
                    "structured along_axis gather index rank must equal data rank "
                    "or data rank minus one"
                )
            if any(
                index.shape[dimension] != data.shape[dimension]
                for dimension in range(data.ndim)
                if dimension != axis
            ):
                raise ValueError(
                    "structured along_axis gather index must match every "
                    "non-axis data extent"
                )
            inferred_shape = index.shape
            if expected_shape(inferred_shape) != inferred_shape:
                raise ValueError(
                    "structured along_axis gather result must be shaped like index"
                )
            return np.take_along_axis(data, index, axis=axis)
        if mode == "take":
            inferred_shape = (
                *data.shape[:axis],
                *index.shape,
                *data.shape[axis + 1 :],
            )
            if expected_shape(inferred_shape) != inferred_shape:
                raise ValueError(
                    "structured take gather result shape differs from data/index"
                )
            return np.take(data, index, axis=axis)
        raise StructuredReferenceUnsupportedError(
            f"structured gather has unsupported mode {mode!r}"
        )

    def _scatter(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        base_attrs = {"axis", "mode", "reduce"}
        atomic_attrs = {
            "atomic", "ordering", "memory_scope", "cas_retry"
        }
        if (
            not base_attrs.issubset(op.attrs)
            or set(op.attrs) - base_attrs - atomic_attrs
            or len(op.operands) not in {3, 4}
        ):
            raise StructuredReferenceUnsupportedError(
                "structured scatter requires closed attrs and three or four operands"
            )
        atomic = op.attrs.get("atomic")
        is_cas = atomic == "cas"
        if is_cas:
            if (
                len(op.operands) != 4
                or not isinstance(op.result, list)
                or len(op.result) != 2
            ):
                raise StructuredReferenceUnsupportedError(
                    "structured atomic='cas' requires four operands and two results"
                )
            if op.attrs.get("reduce") != "none":
                raise ValueError("structured atomic='cas' requires reduce='none'")
            retry = op.attrs.get("cas_retry", "single_attempt")
            if retry not in ATOMIC_CAS_RETRIES:
                raise StructuredReferenceUnsupportedError(
                    "structured atomic='cas' requires a closed cas_retry"
                )
            if retry != "single_attempt":
                raise StructuredReferenceUnsupportedError(
                    "structured reference implements only single-attempt compare-exchange"
                )
        elif len(op.operands) != 3:
            raise StructuredReferenceUnsupportedError(
                "structured non-CAS scatter requires three operands"
            )
        if atomic is not None:
            if atomic not in ATOMIC_KINDS:
                raise StructuredReferenceUnsupportedError(
                    f"structured scatter has unsupported atomic kind {atomic!r}"
                )
            if op.attrs.get("ordering") not in ATOMIC_ORDERINGS:
                raise StructuredReferenceUnsupportedError(
                    "structured atomic scatter requires a closed ordering"
                )
            if op.attrs.get("memory_scope") not in ATOMIC_MEMORY_SCOPES:
                raise StructuredReferenceUnsupportedError(
                    "structured atomic scatter requires a closed memory_scope"
                )
            if atomic == "add" and "cas_retry" in op.attrs:
                raise StructuredReferenceUnsupportedError(
                    "cas_retry is only valid for atomic='cas'"
                )
            if atomic == "add" and op.attrs.get("reduce") != "add":
                raise ValueError("structured atomic='add' requires reduce='add'")
        elif (
            "ordering" in op.attrs
            or "memory_scope" in op.attrs
            or "cas_retry" in op.attrs
        ):
            raise StructuredReferenceUnsupportedError(
                "ordering, memory_scope, and cas_retry require an explicit atomic scatter"
            )
        node_ref = self.node_refs_by_operation.get(id(op), op.node_id)
        update = self.logical_updates.get(node_ref)
        if update is None:
            raise StructuredReferenceUnsupportedError(
                "structured scatter requires one typed logical update"
            )
        data = self._operand(op, env, 0)
        index = self._operand(op, env, 1)
        compare = self._operand(op, env, 2) if is_cas else None
        source = self._operand(op, env, 3 if is_cas else 2)
        axis = op.attrs.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise StructuredReferenceUnsupportedError(
                "structured scatter axis must be a closed integer"
            )
        if not 0 <= axis < data.ndim:
            raise ValueError("structured scatter axis is outside data rank")
        if op.attrs.get("mode") != "along_axis":
            raise StructuredReferenceUnsupportedError(
                "structured scatter implements only canonical along_axis mode"
            )
        if index.dtype.kind not in {"i", "u"}:
            raise ValueError("structured scatter index must have an integer dtype")
        if (
            index.shape != source.shape
            or index.ndim != data.ndim
            or (is_cas and compare is not None and compare.shape != index.shape)
        ):
            raise ValueError(
                "structured scatter index/source/compare must have one common data rank"
            )
        if any(
            index.shape[dimension] > data.shape[dimension]
            for dimension in range(data.ndim)
            if dimension != axis
        ):
            raise ValueError(
                "structured scatter index exceeds one non-axis data extent"
            )
        if index.size and (
            bool(np.any(index < 0))
            or bool(np.any(index >= data.shape[axis]))
        ):
            raise ValueError("structured scatter index is outside the data extent")

        expected_operands = [update.base, update.indices]
        if is_cas:
            expected_operands.extend([update.compare, update.values])
        else:
            expected_operands.append(update.values)
        if expected_operands != list(op.operands) or update.axis != axis:
            raise ValueError(
                "structured scatter operands disagree with its logical update"
            )
        if is_cas:
            result_names = _result_names(op)
            if (
                update.compare != op.operands[2]
                or update.observed != result_names[1]
            ):
                raise ValueError(
                    "structured compare-exchange bindings disagree with its logical update"
                )
        reduce = op.attrs.get("reduce")
        expected_combiner = "overwrite" if reduce == "none" else reduce
        if update.combiner.value != expected_combiner:
            raise ValueError(
                "structured scatter reduce mode disagrees with its logical update"
            )
        if update.duplicate_policy.value == "forbid" and index.shape[axis] > 1:
            ordered = np.sort(index, axis=axis)
            if bool(np.any(np.diff(ordered, axis=axis) == 0)):
                raise ValueError(
                    "structured scatter violates duplicate_policy='forbid'"
                )
        if update.duplicate_policy.value == "one_of":
            raise StructuredReferenceUnsupportedError(
                "structured scatter cannot choose one member of an authored "
                "one_of update result set"
            )

        result = np.array(data, copy=True)
        if is_cas:
            observed = np.empty(index.shape, dtype=result.dtype)
            target_keys: list[tuple[int, ...]] = []
            for position in np.ndindex(index.shape):
                target = list(position)
                target[axis] = int(index[position])
                target_keys.append(tuple(target))
            if update.duplicate_policy.value == "forbid" and len(
                set(target_keys)
            ) != len(target_keys):
                raise ValueError(
                    "structured atomic='cas' violates duplicate_policy='forbid'"
                )
            for position, target in zip(
                np.ndindex(index.shape), target_keys, strict=True
            ):
                old = np.asarray(result[target]).copy()
                observed[position] = old
                if bool(np.equal(old, compare[position])):
                    result[target] = source[position]
            self.scatter_count += 1
            return result, observed
        if reduce == "none":
            np.put_along_axis(result, index, source, axis=axis)
        elif reduce == "add":
            coordinates = np.indices(index.shape, sparse=False)
            coordinates[axis] = index
            np.add.at(result, tuple(coordinates), source)
        else:
            raise StructuredReferenceUnsupportedError(
                f"structured scatter has unsupported reduce mode {reduce!r}"
            )
        self.scatter_count += 1
        return result

    def _ragged_operand(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> _RaggedValue:
        value = self._operand(op, env, 0)
        if not isinstance(value, _RaggedValue):
            raise ValueError(f"structured {op.op} requires a ragged descriptor")
        return value

    def _ragged(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> _RaggedValue:
        if op.attrs or len(op.operands) != 2:
            raise StructuredReferenceUnsupportedError(
                "structured ragged requires two operands and no attrs"
            )
        offsets = self._operand(op, env, 0)
        indices = self._operand(op, env, 1)
        offsets, indices = _validate_ragged_arrays(
            offsets,
            indices,
            context="structured ragged",
        )
        declared = self._declared_value_type(self._result_name(op), op)
        if declared is not None:
            if not declared.is_ragged:
                raise ValueError("structured ragged result is not a ragged type")
            for field_name, field_value in (
                ("offsets", offsets),
                ("indices", indices),
            ):
                field_type = declared.field(field_name)
                if not field_type.is_tensor or field_type.dtype is None:
                    raise ValueError(
                        f"structured ragged field {field_name!r} is not a tensor"
                    )
                if field_value.shape != _resolved_shape(field_type.shape, self.dims):
                    raise ValueError(
                        f"structured ragged field {field_name!r} shape differs "
                        "from its operand"
                    )
                if field_value.dtype != numpy_dtype(field_type.dtype):
                    raise ValueError(
                        f"structured ragged field {field_name!r} dtype differs "
                        "from its operand"
                    )
        return _RaggedValue(offsets, indices)

    def _ragged_outer(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if op.attrs or len(op.operands) != 1:
            raise StructuredReferenceUnsupportedError(
                "structured ragged_outer requires one operand and no attrs"
            )
        descriptor = self._ragged_operand(op, env)
        try:
            outer_count = ragged_outer_count(
                descriptor.offsets,
                descriptor.indices,
                context="structured ragged_outer",
            )
        except RaggedRuntimeError as exc:
            raise ValueError(str(exc)) from exc
        result = np.asarray(outer_count, dtype=descriptor.offsets.dtype)
        declared = self._declared_type(
            self._result_name(op),
            op,
            actual_shape=(),
        )
        if declared is None:
            raise StructuredReferenceUnsupportedError(
                "structured ragged_outer requires one closed scalar result type"
            )
        if declared[1] != () or numpy_dtype(declared[0]) != result.dtype:
            raise ValueError("structured ragged_outer result type is not closed")
        return result

    def _ragged_member(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if op.attrs or len(op.operands) != 2:
            raise StructuredReferenceUnsupportedError(
                "structured ragged_member requires descriptor/index operands "
                "and no attrs"
            )
        descriptor = self._ragged_operand(op, env)
        outer_index = self._operand(op, env, 1)
        if outer_index.ndim != 0 or outer_index.dtype.kind not in {"i", "u"}:
            raise ValueError(
                "structured ragged_member outer index must be a rank-0 integer"
            )
        try:
            length = ragged_member_length(
                descriptor.offsets,
                descriptor.indices,
                outer_index,
                context="structured ragged_member",
            )
        except RaggedRuntimeError as exc:
            raise ValueError(str(exc)) from exc
        group = int(outer_index.item())
        start = int(descriptor.offsets[group])
        stop = start + length
        result = np.array(descriptor.indices[start:stop], copy=True)
        declared = self._declared_type(
            self._result_name(op),
            op,
            actual_shape=result.shape,
        )
        if declared is None:
            declared_value = self._declared_value_type(self._result_name(op), op)
            if (
                declared_value is None
                or not declared_value.is_tensor
                or declared_value.dtype is None
                or declared_value.shape != ("?",)
            ):
                raise StructuredReferenceUnsupportedError(
                    "structured ragged_member requires a dynamic rank-1 result "
                    "type"
                )
            declared = (declared_value.dtype, result.shape)
        if declared[1] != result.shape:
            raise ValueError("structured ragged_member result shape is not closed")
        if numpy_dtype(declared[0]) != result.dtype:
            raise ValueError("structured ragged_member result dtype differs from indices")
        return result

    def _reshape(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"shape"} or len(op.operands) != 1:
            raise StructuredReferenceUnsupportedError(
                "structured reshape requires exactly shape and one operand"
            )
        shape = op.attrs.get("shape")
        if not isinstance(shape, list):
            raise StructuredReferenceUnsupportedError(
                "structured reshape shape must be a list"
            )
        resolved_shape = _resolved_shape(shape, self.dims)
        value = self._operand(op, env, 0)
        declared = self._declared_type(self._result_name(op), op)
        if declared is None:
            raise StructuredReferenceUnsupportedError(
                "structured reshape requires one closed result type"
            )
        dtype, expected_shape = declared
        if expected_shape != resolved_shape:
            raise ValueError(
                "structured reshape result shape differs from attrs.shape"
            )
        if value.dtype != numpy_dtype(dtype):
            raise ValueError("structured reshape must preserve the input dtype")
        if value.size != math.prod(resolved_shape):
            raise ValueError("structured reshape changes the number of elements")
        return np.reshape(value, resolved_shape)

    def _tile(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"reps"} or len(op.operands) != 1:
            raise StructuredReferenceUnsupportedError(
                "structured tile requires exactly reps and one operand"
            )
        value = self._operand(op, env, 0)
        reps = op.attrs.get("reps")
        if not isinstance(reps, list) or any(
            isinstance(rep, bool) or not isinstance(rep, int) for rep in reps
        ):
            raise StructuredReferenceUnsupportedError(
                "structured tile reps must be closed integers"
            )
        if any(rep < 0 for rep in reps):
            raise ValueError("structured tile reps must be non-negative")
        if len(reps) != value.ndim:
            raise ValueError(
                "structured tile requires one repetition for every input dimension"
            )
        expected = self._declared_type(self._result_name(op), op)
        if expected is None:
            raise StructuredReferenceUnsupportedError(
                "structured tile requires one closed result type"
            )
        expected_dtype, expected_shape = expected
        inferred_shape = tuple(
            extent * rep for extent, rep in zip(value.shape, reps, strict=True)
        )
        if len(expected_shape) != value.ndim or expected_shape != inferred_shape:
            raise ValueError(
                "structured tile result shape must equal input shape times reps"
            )
        if numpy_dtype(expected_dtype) != value.dtype:
            raise ValueError("structured tile result dtype differs from its input")
        return np.tile(value, tuple(reps))

    def _iota(self, op: IntentOp) -> np.ndarray:
        if op.operands or set(op.attrs) not in ({"n"}, {"n", "dtype"}):
            raise StructuredReferenceUnsupportedError(
                "structured iota requires no operands and one closed extent"
            )
        expected = self._declared_type(self._result_name(op), op)
        if expected is None:
            raise StructuredReferenceUnsupportedError(
                "structured iota requires one closed result type"
            )
        expected_dtype, expected_shape = expected
        raw_extent = op.attrs.get("n")
        if isinstance(raw_extent, bool):
            raise StructuredReferenceUnsupportedError(
                "structured iota extent must be a closed integer or dimension"
            )
        if isinstance(raw_extent, int):
            extent = raw_extent
        elif isinstance(raw_extent, str) and raw_extent in self.dims:
            extent = self.dims[raw_extent]
        else:
            raise StructuredReferenceUnsupportedError(
                "structured iota extent must be a closed integer or dimension"
            )
        if extent < 0:
            raise ValueError("structured iota extent must be non-negative")
        if expected_shape != (extent,):
            raise ValueError(
                "structured iota result must be rank one with its declared extent"
            )
        attribute_dtype = op.attrs.get("dtype", expected_dtype)
        if not isinstance(attribute_dtype, str):
            raise StructuredReferenceUnsupportedError(
                "structured iota dtype must be a closed string"
            )
        expected_numpy_dtype = numpy_dtype(expected_dtype)
        if expected_numpy_dtype.kind not in {"i", "u"}:
            raise StructuredReferenceUnsupportedError(
                "structured iota result dtype must be a non-boolean integer"
            )
        if numpy_dtype(attribute_dtype) != expected_numpy_dtype:
            raise ValueError(
                "structured iota result dtype differs from its declared dtype"
            )
        return np.arange(extent, dtype=expected_numpy_dtype)

    def _contraction(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if set(op.attrs) != {"subscripts"} or len(op.operands) != 2:
            raise StructuredReferenceUnsupportedError(
                "structured contraction requires exactly two operands and one "
                "subscripts relation"
            )
        result = self._result_name(op)
        expected = self._declared_type(result, op)
        if expected is None:
            raise StructuredReferenceUnsupportedError(
                "structured contraction requires one closed declared result type"
            )
        expected_dtype, expected_shape = expected
        subscripts = op.attrs.get("subscripts")
        if not isinstance(subscripts, str):
            raise StructuredReferenceUnsupportedError(
                "structured contraction subscripts must be a closed string"
            )
        compact = "".join(subscripts.split())
        parts = compact.split("->")
        input_terms = parts[0].split(",") if len(parts) == 2 else []
        if (
            len(parts) != 2
            or len(input_terms) != 2
            or not all(_EINSUM_TERM.fullmatch(term) for term in input_terms)
            or _EINSUM_TERM.fullmatch(parts[-1]) is None
        ):
            raise StructuredReferenceUnsupportedError(
                "structured contraction requires the closed binary einsum grammar"
            )
        lhs_labels, rhs_labels = input_terms
        output_labels = parts[1]
        if len(set(output_labels)) != len(output_labels):
            raise ValueError("structured contraction output labels must be unique")
        if not set(output_labels).issubset(set(lhs_labels) | set(rhs_labels)):
            raise ValueError(
                "structured contraction output labels must occur in an input"
            )

        operands = [self._operand(op, env, index) for index in range(2)]
        label_extents: dict[str, int] = {}
        for index, (labels, value) in enumerate(
            zip((lhs_labels, rhs_labels), operands, strict=True)
        ):
            if len(labels) != value.ndim:
                raise ValueError(
                    f"structured contraction operand {index} labels differ from "
                    "its declared rank"
                )
            for label, extent in zip(labels, value.shape, strict=True):
                previous = label_extents.get(label)
                if previous is not None and previous != extent:
                    raise ValueError(
                        f"structured contraction label {label!r} has "
                        "inconsistent extents"
                    )
                label_extents[label] = extent
        inferred_shape = tuple(label_extents[label] for label in output_labels)
        if inferred_shape != expected_shape:
            raise ValueError(
                "structured contraction declared result shape differs from "
                "its subscripts"
            )

        result_dtype = numpy_dtype(expected_dtype)
        operand_kinds = {value.dtype.kind for value in operands}
        if result_dtype.kind == "f":
            if not operand_kinds.issubset({"b", "i", "u", "f"}):
                raise StructuredReferenceUnsupportedError(
                    "structured floating contraction requires numeric operands"
                )
        elif result_dtype.kind in {"i", "u"}:
            if not operand_kinds.issubset({"b", "i", "u"}):
                raise StructuredReferenceUnsupportedError(
                    "structured integer contraction cannot consume floating operands"
                )
        else:
            raise StructuredReferenceUnsupportedError(
                "structured contraction result dtype must be numeric"
            )
        cast_operands = [
            value.astype(result_dtype, copy=False) for value in operands
        ]
        try:
            contracted = (
                set(lhs_labels).intersection(rhs_labels).difference(output_labels)
            )
            lhs_free = [label for label in lhs_labels if label not in contracted]
            rhs_free = [label for label in rhs_labels if label not in contracted]
            batch_labels = [
                label
                for label in output_labels
                if label in lhs_labels and label in rhs_labels
            ]
            lhs_matrix_labels = [
                label
                for label in lhs_labels
                if label not in contracted and label not in batch_labels
            ]
            rhs_matrix_labels = [
                label
                for label in rhs_labels
                if label not in contracted and label not in batch_labels
            ]
            batched_matrix_product = (
                len(contracted) == 1
                and len(lhs_matrix_labels) == 1
                and len(rhs_matrix_labels) == 1
                and len(set(lhs_labels)) == len(lhs_labels)
                and len(set(rhs_labels)) == len(rhs_labels)
                and len(set(output_labels)) == len(output_labels)
                and set(output_labels)
                == set((*batch_labels, *lhs_matrix_labels, *rhs_matrix_labels))
            )
            if batched_matrix_product:
                contraction_label = next(iter(contracted))
                lhs_order = [
                    *batch_labels,
                    lhs_matrix_labels[0],
                    contraction_label,
                ]
                rhs_order = [
                    *batch_labels,
                    contraction_label,
                    rhs_matrix_labels[0],
                ]
                lhs_matrix = np.transpose(
                    cast_operands[0],
                    axes=tuple(lhs_labels.index(label) for label in lhs_order),
                )
                rhs_matrix = np.transpose(
                    cast_operands[1],
                    axes=tuple(rhs_labels.index(label) for label in rhs_order),
                )
                value = np.matmul(lhs_matrix, rhs_matrix)
                result_order = [
                    *batch_labels,
                    lhs_matrix_labels[0],
                    rhs_matrix_labels[0],
                ]
                if result_order != list(output_labels):
                    value = np.transpose(
                        value,
                        axes=tuple(
                            result_order.index(label) for label in output_labels
                        ),
                    )
            else:
                value = np.einsum(
                    compact,
                    cast_operands[0],
                    cast_operands[1],
                    optimize=False,
                )
        except (TypeError, ValueError) as exc:
            raise StructuredReferenceUnsupportedError(
                "structured contraction dtype is unsupported by the closed "
                "NumPy denotation"
            ) from exc
        value = np.asarray(value)
        if value.dtype != result_dtype:
            raise StructuredReferenceUnsupportedError(
                "structured contraction did not preserve its declared result dtype"
            )
        return value

    def _control_predicate(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> bool:
        if not isinstance(op.predicate, str) or not op.predicate:
            raise StructuredReferenceUnsupportedError(
                f"structured {op.op} requires one predicate value"
            )
        try:
            predicate = np.asarray(env[op.predicate])
        except KeyError as exc:
            raise ValueError(
                f"structured {op.op} predicate does not dominate its region"
            ) from exc
        if predicate.shape != () or predicate.dtype != np.dtype(np.bool_):
            raise ValueError(
                f"structured {op.op} predicate must be one rank-0 i1 value"
            )
        return bool(predicate.item())

    def _ordinary(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if op.op not in _ORDINARY_OPS:
            raise StructuredReferenceUnsupportedError(
                f"structured reference has no denotation for IntentOp {op.op!r}"
            )
        if op.op == "ragged":
            return self._ragged(op, env)
        if op.op == "ragged_outer":
            return self._ragged_outer(op, env)
        if op.op == "ragged_member":
            return self._ragged_member(op, env)
        if op.op == "const":
            if set(op.attrs) - {"value", "dtype", "shape"}:
                raise StructuredReferenceUnsupportedError(
                    "structured const has unsupported attributes"
                )
            result = self._result_name(op)
            declared = self._declared_type(result, op)
            if declared is None:
                raise StructuredReferenceUnsupportedError(
                    "structured const requires one closed result type"
                )
            dtype, shape = declared
            try:
                literal = validate_const_literal(
                    op.attrs,
                    result_type={"dtype": dtype, "shape": list(shape)},
                    context="structured const",
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            runtime_dtype = numpy_dtype(dtype)
            if literal.splat:
                return np.full(
                    shape,
                    _runtime_const_literal(literal.values[0]),
                    dtype=runtime_dtype,
                )
            values = [
                _runtime_const_literal(value) for value in literal.values
            ]
            return np.asarray(values, dtype=runtime_dtype).reshape(shape)
        if op.op == "concat":
            return self._concat(op, env)
        if op.op == "gather":
            return self._gather(op, env)
        if op.op == "scatter":
            return self._scatter(op, env)
        if op.op == "iota":
            return self._iota(op)
        if op.op == "map_binary":
            if set(op.attrs) != {"fn"} or len(op.operands) != 2:
                raise StructuredReferenceUnsupportedError(
                    "structured map_binary requires exactly fn and two operands"
                )
            fn = op.attrs.get("fn")
            implementation = _BINARY.get(str(fn))
            if implementation is None:
                raise StructuredReferenceUnsupportedError(
                    f"structured map_binary has unsupported fn {fn!r}"
                )
            lhs_type = self.value_types.get(op.operands[0])
            rhs_type = self.value_types.get(op.operands[1])
            result_name = self._result_name(op)
            result_type = self.value_types.get(result_name)
            if (
                lhs_type is None
                or rhs_type is None
                or result_type is None
                or not lhs_type.is_tensor
                or not rhs_type.is_tensor
                or not result_type.is_tensor
                or lhs_type.dtype is None
                or rhs_type.dtype is None
                or result_type.dtype is None
                or map_binary_result_dtype(
                    fn, lhs_type.dtype, rhs_type.dtype
                )
                != result_type.dtype
            ):
                raise StructuredReferenceUnsupportedError(
                    f"structured map_binary fn {fn!r} violates the shared "
                    "closed dtype contract"
                )
            left = self._operand(op, env, 0)
            right = self._operand(op, env, 1)
            declared = self._declared_type(self._result_name(op), op)
            if fn in {"div", "floordiv", "remainder"} and declared is not None:
                result_dtype = np.dtype(numpy_dtype(declared[0]))
                if np.issubdtype(result_dtype, np.integer):
                    if fn == "div":
                        return _integer_divide(left, right, result_dtype)
                    if fn == "floordiv":
                        return _integer_floor_divide(left, right, result_dtype)
                    return _integer_remainder(left, right, result_dtype)
            arithmetic = {
                "add", "sub", "mul", "div", "floordiv", "remainder",
                "max", "min", "pow", "atan2", "copysign",
            }
            logical = {
                "and", "or", "xor", "logical_and", "logical_or", "logical_xor"
            }
            integer = {"left_shift", "right_shift"}
            if fn in arithmetic and declared is not None:
                # NumPy's scalar/0-D weak-promotion rules can retain the array's
                # lower precision (for example f16-array * f32-scalar).  J has
                # already closed the result dtype, so execute the pointwise op
                # in that dtype rather than discovering a second type rule here.
                compute_dtype = np.dtype(numpy_dtype(declared[0]))
                left_value = np.asarray(left, dtype=compute_dtype)
                right_value = np.asarray(right, dtype=compute_dtype)
            elif fn in logical:
                left_value = np.asarray(left, dtype=np.bool_)
                right_value = np.asarray(right, dtype=np.bool_)
            elif fn in integer and declared is not None:
                compute_dtype = np.dtype(numpy_dtype(declared[0]))
                if not np.issubdtype(compute_dtype, np.integer):
                    raise StructuredReferenceUnsupportedError(
                        f"structured {fn} requires an integer result dtype"
                    )
                left_value = np.asarray(left, dtype=compute_dtype)
                right_value = np.asarray(right, dtype=compute_dtype)
            else:
                promoted = np.result_type(np.asarray(left).dtype, np.asarray(right).dtype)
                left_value = np.asarray(left, dtype=promoted)
                right_value = np.asarray(right, dtype=promoted)
            result = np.asarray(implementation(left_value, right_value))
            if declared is not None:
                result = np.asarray(result, dtype=numpy_dtype(declared[0]))
            return result
        if op.op == "map_unary":
            if set(op.attrs) != {"fn"} or len(op.operands) != 1:
                raise StructuredReferenceUnsupportedError(
                    "structured map_unary requires exactly fn and one operand"
                )
            fn = op.attrs.get("fn")
            implementation = _UNARY.get(str(fn))
            if implementation is None:
                raise StructuredReferenceUnsupportedError(
                    f"structured map_unary has unsupported fn {fn!r}"
                )
            operand_type = self.value_types.get(op.operands[0])
            result_name = self._result_name(op)
            result_type = self.value_types.get(result_name)
            if (
                operand_type is None
                or result_type is None
                or not operand_type.is_tensor
                or not result_type.is_tensor
                or operand_type.dtype is None
                or result_type.dtype is None
                or map_unary_result_dtype(fn, operand_type.dtype)
                != result_type.dtype
            ):
                raise StructuredReferenceUnsupportedError(
                    f"structured map_unary fn {fn!r} violates the shared "
                    "closed dtype contract"
                )
            result = np.asarray(implementation(self._operand(op, env, 0)))
            declared = self._declared_type(self._result_name(op), op)
            if declared is not None:
                result = np.asarray(result, dtype=numpy_dtype(declared[0]))
            return result
        if op.op == "reduce":
            return self._reduce(op, env)
        if op.op == "scan":
            return self._scan(op, env)
        if op.op == "scatter":
            return self._scatter(op, env)
        if op.op == "contraction":
            return self._contraction(op, env)
        if op.op == "select":
            if op.attrs or len(op.operands) != 3:
                raise StructuredReferenceUnsupportedError(
                    "structured select requires three operands and no control attrs"
                )
            condition = self._operand(op, env, 0)
            if condition.dtype != np.dtype(np.bool_):
                raise ValueError("structured select condition must have i1 dtype")
            return np.asarray(
                np.where(
                    condition,
                    self._operand(op, env, 1),
                    self._operand(op, env, 2),
                )
            )
        if op.op == "cast":
            if len(op.operands) != 1 or set(op.attrs) not in (
                {"to_dtype"},
                {"dtype"},
            ):
                raise StructuredReferenceUnsupportedError(
                    "structured cast requires one canonical dtype target"
                )
            target = op.attrs.get("to_dtype", op.attrs.get("dtype"))
            if not isinstance(target, str):
                raise StructuredReferenceUnsupportedError(
                    "structured cast target must be a dtype string"
                )
            return self._operand(op, env, 0).astype(numpy_dtype(target))
        if op.op == "transpose":
            if len(op.operands) != 1 or set(op.attrs) != {"perm"}:
                raise StructuredReferenceUnsupportedError(
                    "structured transpose requires one closed permutation"
                )
            perm = op.attrs.get("perm")
            if not isinstance(perm, list) or any(
                isinstance(axis, bool) or not isinstance(axis, int)
                for axis in perm
            ):
                raise StructuredReferenceUnsupportedError(
                    "structured transpose permutation must be integers"
                )
            return np.transpose(self._operand(op, env, 0), axes=tuple(perm))
        if op.op == "pad":
            if len(op.operands) != 1 or set(op.attrs) != {"lo", "hi", "value"}:
                raise StructuredReferenceUnsupportedError(
                    "structured pad requires closed lo/hi/value attributes"
                )
            lo = op.attrs.get("lo")
            hi = op.attrs.get("hi")
            if (
                not isinstance(lo, list)
                or not isinstance(hi, list)
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in [*lo, *hi]
                )
            ):
                raise StructuredReferenceUnsupportedError(
                    "structured reference currently requires literal pad extents"
                )
            return np.pad(
                self._operand(op, env, 0),
                tuple(zip(lo, hi, strict=True)),
                mode="constant",
                constant_values=op.attrs["value"],
            )
        if op.op == "tile":
            return self._tile(op, env)
        if op.op == "reshape":
            return self._reshape(op, env)
        raise AssertionError(op.op)  # pragma: no cover - closed above

    def _carry_type(self, state: LoopStateBinding) -> tuple[str, tuple[int, ...]]:
        if state.type is None or state.type.kind != "tensor" or state.type.dtype is None:
            raise StructuredReferenceUnsupportedError(
                "structured loop carry requires one tensor value type; named "
                "record carries need a record-aware reference ABI"
            )
        return state.type.dtype, _resolved_shape(state.type.shape, self.dims)

    def _carry_init(
        self,
        state: LoopStateBinding,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        declared = self._carry_type(state)
        dtype, shape = declared
        init = state.init_value
        if isinstance(init, str) and init in env:
            value = np.array(env[init], copy=True)
        else:
            literal = _CARRY_LITERALS.get(str(init), init)
            if isinstance(literal, str):
                try:
                    literal = float(literal)
                except ValueError as exc:
                    raise StructuredReferenceUnsupportedError(
                        f"structured carry has unknown initializer {init!r}"
                    ) from exc
            try:
                value = np.full(shape, literal, dtype=numpy_dtype(dtype))
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"structured carry initializer {init!r} violates {dtype}"
                ) from exc
        return self._validate_value(
            value,
            name=state.body_argument,
            declared=declared,
        )

    def _loop_scan_tile(self, op: IntentOp, trip_axis: str, extent: int) -> int:
        """Choose a deterministic J-only scan tile under the memory bound."""

        if extent <= 1:
            return extent
        if any(
            state.type is not None and trip_axis in state.type.shape
            for state in op.carry
        ):
            # A trip-shaped carry would require a separate, explicitly proved
            # partition/merge denotation.  Keep the existing full-axis path.
            return extent

        tile = extent
        candidate_names = {
            scan.name for scan in op.scan
        }
        candidate_names.update(
            name
            for child in _walk_ops(op.body)
            for name in _result_names(child)
        )
        for name in candidate_names:
            value_type = self.value_types.get(name)
            if value_type is None or trip_axis not in value_type.shape:
                continue
            if tuple(value_type.shape).count(trip_axis) != 1:
                return extent
            elements_per_trip = 1
            for dimension in value_type.shape:
                if dimension == trip_axis:
                    continue
                if isinstance(dimension, bool):
                    raise ValueError(
                        "structured loop type contains a boolean extent"
                    )
                if isinstance(dimension, int):
                    dimension_extent = dimension
                elif isinstance(dimension, str) and dimension in self.dims:
                    dimension_extent = self.dims[dimension]
                else:
                    raise StructuredReferenceUnsupportedError(
                        "structured loop cannot resolve a scan-tiled result type"
                    )
                elements_per_trip *= int(dimension_extent)
            bytes_per_trip = max(
                1,
                elements_per_trip
                * np.dtype(numpy_dtype(value_type.dtype)).itemsize,
            )
            tile = min(
                tile,
                max(1, _MAX_REFERENCE_TEMPORARY_BYTES // bytes_per_trip),
            )
        return max(1, tile)

    def _loop(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        validate_loop_wiring(op, context="structured loop")
        if op.trip is None or not op.carry:
            raise StructuredReferenceUnsupportedError(
                "structured loop requires one trip and typed carries"
            )
        if any(child.op == "loop" for child in _walk_ops(op.body)):
            raise StructuredReferenceUnsupportedError(
                "structured reference does not implement nested loops"
            )
        # The shared wiring validator has already established the exact result
        # and yield order.  Keep the explicit arity assertion here because this
        # executor consumes the list directly when it returns the final carry
        # values.
        if not isinstance(op.result, list) or len(op.result) != len(op.carry):
            raise ValueError("structured loop results differ from carry order")
        for state in op.carry:
            declared = self._carry_type(state)
            for role in (
                state.body_argument,
                state.next_value,
                state.result_value,
            ):
                known = self.value_types.get(role)
                if known is None:
                    continue
                if (
                    not known.is_tensor
                    or known.dtype is None
                    or known.dtype != declared[0]
                    or _resolved_shape(known.shape, self.dims) != declared[1]
                ):
                    raise ValueError(
                        f"structured loop carry role {role!r} type differs "
                        "from its authored carry type"
                    )
        carries = {
            state.body_argument: self._carry_init(state, env)
            for state in op.carry
        }
        self.loop_count += 1
        if op.trip.is_dynamic:
            if op.scan:
                raise StructuredReferenceUnsupportedError(
                    "structured dynamic value trips do not support scan inputs"
                )
            upper_name = op.trip.upper_bound
            index_name = op.trip.index
            if not isinstance(upper_name, str) or not isinstance(index_name, str):
                raise StructuredReferenceUnsupportedError(
                    "structured dynamic loop requires upper_bound and index"
                )
            try:
                upper_value = env[upper_name]
            except KeyError as exc:
                raise ValueError(
                    f"structured dynamic loop upper bound {upper_name!r} is unavailable"
                ) from exc
            if (
                isinstance(upper_value, _RaggedValue)
                or upper_value.ndim != 0
                or upper_value.dtype.kind not in {"i", "u"}
            ):
                raise ValueError(
                    "structured dynamic loop upper bound must be a rank-0 integer"
                )
            extent = int(upper_value.item())
            if extent < 0:
                raise ValueError("structured dynamic loop upper bound must be non-negative")
            if extent == 0:
                if len(carries) != len(op.carry):
                    raise StructuredReferenceUnsupportedError(
                        "zero-trip structured loop cannot return a null-initialized carry"
                    )
                self.zero_trip_loops += 1
                final_carries = carries
            else:
                final_carries = dict(carries)
                for iteration in range(extent):
                    local = dict(env)
                    local[index_name] = np.asarray(
                        iteration,
                        dtype=upper_value.dtype,
                    )
                    local.update(final_carries)
                    local = self.sequence(op.body, local, allow_loop=False)
                    next_carries: dict[str, np.ndarray] = {}
                    for state in op.carry:
                        if state.next_value not in local:
                            raise ValueError(
                                "structured loop body did not define "
                                f"{state.next_value!r}"
                            )
                        next_carries[state.body_argument] = self._validate_value(
                            local[state.next_value],
                            name=state.next_value,
                            declared=self._carry_type(state),
                        )
                    final_carries = next_carries
                    self.logical_loop_steps += 1
            return {
                state.result_value: self._validate_value(
                    final_carries[state.body_argument],
                    name=state.result_value,
                    declared=self._carry_type(state),
                )
                for state in op.carry
            }

        trip_axis = op.trip.axis
        if trip_axis not in self.dims:
            raise StructuredReferenceUnsupportedError(
                f"structured loop trip extent {trip_axis!r} is unresolved"
            )
        extent = self.dims[trip_axis]
        if extent < 0:
            raise ValueError("structured loop trip extent must be non-negative")

        scan_sources: list[tuple[Any, np.ndarray]] = []
        for scan in op.scan:
            try:
                source = env[scan.source]
            except KeyError as exc:
                raise ValueError(
                    f"structured loop scan source {scan.source!r} is unavailable"
                ) from exc
            if scan.axis < 0 or scan.axis >= source.ndim:
                raise ValueError("structured loop scan axis is outside source rank")
            if source.shape[scan.axis] != extent:
                raise ValueError(
                    "structured loop scan extent differs from semantic trip axis"
                )
            scan_sources.append((scan, source))

        if extent == 0:
            if len(carries) != len(op.carry):
                raise StructuredReferenceUnsupportedError(
                    "zero-trip structured loop cannot return a null-initialized carry"
                )
            self.zero_trip_loops += 1
            final_carries = carries
        else:
            tile = self._loop_scan_tile(op, trip_axis, extent)
            self.max_scan_tile_elements = max(
                self.max_scan_tile_elements, tile
            )
            original_extent = self.dims[trip_axis]
            final_carries = dict(carries)
            for start in range(0, extent, tile):
                stop = min(extent, start + tile)
                self.dims[trip_axis] = stop - start
                try:
                    local = dict(env)
                    for scan, source in scan_sources:
                        slices = [slice(None)] * source.ndim
                        slices[scan.axis] = slice(start, stop)
                        local[scan.name] = np.asarray(source[tuple(slices)])
                    local.update(final_carries)
                    local = self.sequence(op.body, local, allow_loop=False)
                    next_carries: dict[str, np.ndarray] = {}
                    for state in op.carry:
                        if state.next_value not in local:
                            raise ValueError(
                                "structured loop body did not define "
                                f"{state.next_value!r}"
                            )
                        next_carries[state.body_argument] = self._validate_value(
                            local[state.next_value],
                            name=state.next_value,
                            declared=self._carry_type(state),
                        )
                    final_carries = next_carries
                    self.logical_loop_steps += 1
                finally:
                    self.dims[trip_axis] = original_extent

        return {
            state.result_value: self._validate_value(
                final_carries[state.body_argument],
                name=state.result_value,
                declared=self._carry_type(state),
            )
            for state in op.carry
        }

    def sequence(
        self,
        body: Sequence[IntentOp],
        env: Mapping[str, np.ndarray],
        *,
        allow_loop: bool,
    ) -> dict[str, np.ndarray]:
        result = dict(env)
        for op in body:
            self.operation_count += 1
            if op.op in {"effect_if", "value_if"}:
                if op.positive is None or op.negative is None:
                    raise StructuredReferenceUnsupportedError(
                        f"structured {op.op} requires two explicit arms"
                    )
                arm = (
                    op.positive
                    if self._control_predicate(op, result)
                    else op.negative
                )
                branch = self.sequence(
                    arm.body,
                    result,
                    allow_loop=allow_loop,
                )
                if op.op == "effect_if":
                    self.effect_if_count += 1
                    # effect_if exports no SSA value.  Branch-local values are
                    # deliberately discarded; observable state must already be
                    # represented by authored J outputs/carries.
                    continue
                self.value_if_count += 1
                name = self._result_name(op)
                try:
                    yielded = branch[arm.yield_value]
                except KeyError as exc:
                    raise ValueError(
                        "structured value_if selected arm did not produce its "
                        "authored yield"
                    ) from exc
                result[name] = self._validate_value(
                    yielded,
                    name=name,
                    declared=self._declared_value_type(name, op),
                )
                self._record_conditional_effect_stores(op, result)
                continue
            if op.op == "loop":
                if not allow_loop:
                    raise StructuredReferenceUnsupportedError(
                        "structured reference does not implement nested loops"
                    )
                result.update(self._loop(op, result))
                self._record_conditional_effect_stores(op, result)
                continue
            value = self._ordinary(op, result)
            names = _result_names(op)
            if len(names) == 1:
                name = names[0]
                result[name] = self._validate_value(
                    value,
                    name=name,
                    declared=self._declared_value_type(name, op),
                )
            elif len(names) > 1:
                if not isinstance(value, tuple) or len(value) != len(names):
                    raise StructuredReferenceUnsupportedError(
                        f"structured {op.op} returned the wrong number of results"
                    )
                for name, item in zip(names, value, strict=True):
                    result[name] = self._validate_value(
                        item,
                        name=name,
                        declared=self._declared_value_type(name, op),
                    )
            else:
                raise StructuredReferenceUnsupportedError(
                    f"structured {op.op} requires at least one result"
                )
            self._record_conditional_effect_stores(op, result)
        return result


def execute_structured_reference(
    j: SemanticArchive,
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    access_guard_relations: Sequence[Any] = (),
    conditional_effect_stores: Sequence[Any] = (),
) -> CPUExecutionResult:
    """Execute the restricted typed denotation of one canonical J.

    The signature intentionally excludes C and pi.  In particular,
    ``pi.loop_carriers`` cannot affect initialization, recurrence, or results.
    """

    j = elaborate_j(j)
    try:
        validate_ragged_region_boundaries(j.intent.body)
    except ValueError as exc:
        raise StructuredReferenceUnsupportedError(str(exc)) from exc
    guarded_loads = tuple(
        relation
        for relation in access_guard_relations
        if getattr(relation, "access_kind", None) == "load"
    )
    if not requires_structured_reference(j) and not guarded_loads:
        raise StructuredReferenceUnsupportedError(
            "structured reference requires a structured op or a contraction "
            "outside the scalar CPU capability"
        )
    unsupported = sorted(
        {op.op for op in _walk_ops(j.intent.body)}
        - (_ORDINARY_OPS | {"loop", "effect_if", "value_if"})
    )
    if unsupported:
        raise StructuredReferenceUnsupportedError(
            "structured reference graph contains IntentOp(s) outside its "
            "closed operation table: "
            + ", ".join(unsupported)
        )
    if len(inputs) != len(j.intent.inputs):
        raise ValueError("structured reference input arity differs from J")
    checked_dims = {
        name: int(extent)
        for name, extent in dims.items()
        if isinstance(name, str)
        and isinstance(extent, int)
        and not isinstance(extent, bool)
        and extent >= 0
    }
    if checked_dims != dict(dims):
        raise ValueError(
            "structured reference dims must be non-negative integer bindings"
        )
    checked_dims = resolve_dimension_relations(j, checked_dims)
    checked_inputs = {
        tensor.name: validate_array(
            np.asarray(value),
            shape=_resolved_shape(tensor.shape, checked_dims),
            dtype=tensor.dtype,
            context=f"structured input {tensor.name!r}",
        )
        for tensor, value in zip(j.intent.inputs, inputs, strict=True)
    }
    validate_value_dimension_instantiation(j, checked_dims, checked_inputs)
    validate_ragged_runtime_invocation(
        j.intent,
        checked_inputs,
        dims=checked_dims,
    )
    evaluator = _Evaluator(
        j,
        checked_dims,
        broadcast_axes or {},
        guarded_loads,
        conditional_effect_stores,
    )
    env = evaluator.sequence(j.intent.body, checked_inputs, allow_loop=True)
    output_values = {
        item.output: item.value
        for item in j_elaboration(j).output_bindings
    }
    outputs: dict[str, np.ndarray] = {}
    for tensor in j.intent.outputs:
        if tensor.name in evaluator.effect_output_state:
            effect_value = evaluator.effect_output_state[tensor.name]
            resolved_shape = _resolved_declared_shape(
                tensor.shape,
                checked_dims,
                np.asarray(effect_value).shape,
            )
            if resolved_shape is None:
                raise StructuredReferenceUnsupportedError(
                    f"structured output {tensor.name!r} shape is unresolved"
                )
            outputs[tensor.name] = evaluator._validate_value(
                effect_value,
                name=tensor.name,
                declared=(tensor.dtype, resolved_shape),
            )
            continue
        value_ref = output_values[tensor.name]
        if value_ref not in env:
            raise ValueError(
                f"structured reference did not produce output {tensor.name!r}"
            )
        output_value = env[value_ref]
        resolved_shape = _resolved_declared_shape(
            tensor.shape,
            checked_dims,
            np.asarray(output_value).shape,
        )
        if resolved_shape is None:
            raise StructuredReferenceUnsupportedError(
                f"structured output {tensor.name!r} shape is unresolved"
            )
        outputs[tensor.name] = evaluator._validate_value(
            output_value,
            name=tensor.name,
            declared=(tensor.dtype, resolved_shape),
        )
    return CPUExecutionResult(
        outputs,
        {
            "protocol": STRUCTURED_CPU_REFERENCE_PROTOCOL,
            "authority": "canonical_J_only",
            "execution_mode": "bounded_j_only_scan_tiles",
            "operation_count": evaluator.operation_count,
            "scan_count": evaluator.scan_count,
            "scatter_count": evaluator.scatter_count,
            "loop_count": evaluator.loop_count,
            "effect_if_count": evaluator.effect_if_count,
            "value_if_count": evaluator.value_if_count,
            "logical_loop_step_count": evaluator.logical_loop_steps,
            "max_scan_tile_elements": evaluator.max_scan_tile_elements,
            "zero_trip_loop_count": evaluator.zero_trip_loops,
            "access_guard_load_materialization_count": len(
                evaluator.access_guard_loads
            ),
            "conditional_effect_store_count": (
                evaluator.executed_conditional_effect_store_count
            ),
            "input_count": len(checked_inputs),
            "output_count": len(outputs),
        },
    )


__all__ = [
    "STRUCTURED_CPU_REFERENCE_PROTOCOL",
    "StructuredReferenceUnsupportedError",
    "execute_structured_reference",
    "requires_structured_reference",
]
