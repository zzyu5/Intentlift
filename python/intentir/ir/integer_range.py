"""Source-free integer range analysis over canonical J value graphs.

The analysis is deliberately conservative and target-neutral.  It computes a
sound inclusive interval for integer tensor values produced by closed Layer-A
operations, but returns unknown whenever finite-width overflow, a dynamic op,
or unsupported control could change the result.  Consumers may use a contained
interval as a proof; a non-contained over-approximation is not proof of failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import re
from typing import Any, Mapping

import numpy as np

from .canonical_j import elaborate_j, j_elaboration
from .model import IntentOp


@dataclass(frozen=True)
class IntegerInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if isinstance(self.lower, bool) or isinstance(self.upper, bool):
            raise TypeError("integer interval endpoints cannot be boolean")
        if not isinstance(self.lower, int) or not isinstance(self.upper, int):
            raise TypeError("integer interval endpoints must be integers")
        if self.lower > self.upper:
            raise ValueError("integer interval lower bound exceeds upper bound")

    def contained_in(self, lower: int, upper: int) -> bool:
        return lower <= self.lower and self.upper <= upper


def _dtype_bounds(dtype: str) -> tuple[int, int] | None:
    normalized = dtype.removeprefix("torch.").lower()
    if normalized == "index":
        # The logical index dtype uses the same signed 64-bit carrier as the
        # host/native ABI and the MLIR emitter.
        return -(1 << 63), (1 << 63) - 1
    aliases = {
        "byte": "uint8",
        "char": "int8",
        "short": "int16",
        "int": "int32",
        "long": "int64",
    }
    normalized = aliases.get(normalized, normalized)
    match = re.fullmatch(r"([iu]|int|uint)([0-9]+)", normalized)
    if match is None:
        return None
    prefix, width_text = match.groups()
    width = int(width_text)
    if width <= 0:
        return None
    unsigned = prefix in {"u", "uint"}
    if unsigned:
        return 0, (1 << width) - 1
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def _result_names(operation: IntentOp) -> tuple[str, ...]:
    if isinstance(operation.result, str):
        return (operation.result,)
    if isinstance(operation.result, list):
        return tuple(name for name in operation.result if isinstance(name, str))
    return ()


def _json_integers(value: Any) -> list[int] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            nested = _json_integers(item)
            if nested is None:
                return None
            result.extend(nested)
        return result
    return None


def _within_dtype(interval: IntegerInterval, dtype: str) -> IntegerInterval | None:
    bounds = _dtype_bounds(dtype)
    if bounds is None or not interval.contained_in(*bounds):
        return None
    return interval


def _unary_interval(fn: str, operand: IntegerInterval) -> IntegerInterval | None:
    if fn in {"identity", "logical_identity"}:
        return operand
    if fn == "neg":
        return IntegerInterval(-operand.upper, -operand.lower)
    if fn == "abs":
        if operand.lower >= 0:
            return operand
        if operand.upper <= 0:
            return IntegerInterval(-operand.upper, -operand.lower)
        return IntegerInterval(0, max(-operand.lower, operand.upper))
    return None


def _binary_interval(
    fn: str,
    left: IntegerInterval,
    right: IntegerInterval,
) -> IntegerInterval | None:
    if fn == "add":
        return IntegerInterval(left.lower + right.lower, left.upper + right.upper)
    if fn == "sub":
        return IntegerInterval(left.lower - right.upper, left.upper - right.lower)
    if fn == "mul":
        products = (
            left.lower * right.lower,
            left.lower * right.upper,
            left.upper * right.lower,
            left.upper * right.upper,
        )
        return IntegerInterval(min(products), max(products))
    if fn == "div" and not (right.lower <= 0 <= right.upper):
        def trunc_div(numerator: int, denominator: int) -> int:
            quotient = abs(numerator) // abs(denominator)
            return -quotient if (numerator < 0) != (denominator < 0) else quotient

        quotients = (
            trunc_div(left.lower, right.lower),
            trunc_div(left.lower, right.upper),
            trunc_div(left.upper, right.lower),
            trunc_div(left.upper, right.upper),
        )
        return IntegerInterval(min(quotients), max(quotients))
    if fn == "min":
        return IntegerInterval(min(left.lower, right.lower), min(left.upper, right.upper))
    if fn == "max":
        return IntegerInterval(max(left.lower, right.lower), max(left.upper, right.upper))
    return None


def infer_integer_value_intervals(
    j: Any,
    *,
    logical_inputs: Mapping[str, Any] | None = None,
    dimension_extents: Mapping[str, int] | None = None,
) -> dict[str, IntegerInterval]:
    """Infer sound element-value intervals for the supported closed J subgraph."""

    canonical = elaborate_j(j)
    value_types = j_elaboration(canonical).value_type_by_name
    intervals: dict[str, IntegerInterval] = {}
    for name, value in dict(logical_inputs or {}).items():
        value_type = value_types.get(name)
        if (
            value_type is None
            or not value_type.is_tensor
            or value_type.dtype is None
            or _dtype_bounds(value_type.dtype) is None
        ):
            continue
        array = np.asarray(value)
        if array.size == 0 or array.dtype.kind not in {"i", "u"}:
            continue
        interval = IntegerInterval(int(array.min()), int(array.max()))
        closed = _within_dtype(interval, value_type.dtype)
        if closed is not None:
            intervals[name] = closed

    preserving_ops = {
        "broadcast", "concat", "flip", "reshape", "tile", "transpose",
    }
    operation_rows = j_elaboration(canonical).typed_operation_rows
    if operation_rows is None:
        raise ValueError("integer range analysis requires canonical operations")
    operations = tuple(operation for _path, operation in operation_rows)
    ragged_offsets_by_name: dict[str, str] = {}
    ragged_indices_by_name: dict[str, str] = {}
    for operation in operations:
        results = _result_names(operation)
        if operation.op != "ragged" or len(results) != 1:
            continue
        if len(operation.operands) != 2:
            continue
        ragged_offsets_by_name[results[0]] = operation.operands[0]
        ragged_indices_by_name[results[0]] = operation.operands[1]

    resolved_dimensions = dict(dimension_extents or {})
    for _ in range(len(operations) + 1):
        changed = False
        for operation in operations:
            if operation.op == "loop" and operation.trip is not None:
                if operation.trip.is_dynamic:
                    index_name = operation.trip.index
                    upper_name = operation.trip.upper_bound
                    upper_interval = (
                        intervals.get(upper_name)
                        if isinstance(upper_name, str)
                        else None
                    )
                    if (
                        isinstance(index_name, str)
                        and index_name not in intervals
                        and upper_interval is not None
                        and upper_interval.upper > 0
                    ):
                        index_type = value_types.get(index_name)
                        if index_type is not None and index_type.dtype is not None:
                            index_interval = _within_dtype(
                                IntegerInterval(0, upper_interval.upper - 1),
                                index_type.dtype,
                            )
                            if index_interval is not None:
                                intervals[index_name] = index_interval
                                changed = True
            results = _result_names(operation)
            if len(results) != 1 or results[0] in intervals:
                continue
            result_name = results[0]
            result_type = value_types.get(result_name)
            if (
                result_type is None
                or not result_type.is_tensor
                or result_type.dtype is None
                or _dtype_bounds(result_type.dtype) is None
            ):
                continue
            interval: IntegerInterval | None = None
            if operation.op == "const":
                values = _json_integers(operation.attrs.get("value"))
                if values:
                    interval = IntegerInterval(min(values), max(values))
            elif operation.op == "iota":
                extent = operation.attrs.get("n")
                if isinstance(extent, str):
                    extent = resolved_dimensions.get(extent)
                if type(extent) is int and extent > 0:
                    interval = IntegerInterval(0, extent - 1)
            elif operation.op == "ragged_outer" and len(operation.operands) == 1:
                offsets_name = ragged_offsets_by_name.get(operation.operands[0])
                offsets_type = value_types.get(offsets_name or "")
                if offsets_type is not None and len(offsets_type.shape) == 1:
                    extent = offsets_type.shape[0]
                    if isinstance(extent, str):
                        extent = resolved_dimensions.get(extent)
                    if type(extent) is int and extent > 0:
                        interval = IntegerInterval(extent - 1, extent - 1)
            elif operation.op == "ragged_member" and len(operation.operands) == 2:
                indices_name = ragged_indices_by_name.get(operation.operands[0])
                indices_interval = intervals.get(indices_name or "")
                if indices_interval is not None:
                    # Selecting one ragged group cannot produce an index outside
                    # the descriptor's complete indices carrier.
                    interval = indices_interval
            elif operation.op == "map_unary" and len(operation.operands) == 1:
                operand = intervals.get(operation.operands[0])
                fn = operation.attrs.get("fn")
                if operand is not None and isinstance(fn, str):
                    interval = _unary_interval(fn, operand)
            elif operation.op == "map_binary" and len(operation.operands) == 2:
                left = intervals.get(operation.operands[0])
                right = intervals.get(operation.operands[1])
                fn = operation.attrs.get("fn")
                if left is not None and right is not None and isinstance(fn, str):
                    interval = _binary_interval(fn, left, right)
            elif operation.op == "select" and len(operation.operands) == 3:
                positive = intervals.get(operation.operands[1])
                negative = intervals.get(operation.operands[2])
                if positive is not None and negative is not None:
                    interval = IntegerInterval(
                        min(positive.lower, negative.lower),
                        max(positive.upper, negative.upper),
                    )
            elif operation.op == "gather" and len(operation.operands) == 2:
                data_ref, indices_ref = operation.operands
                data_interval = intervals.get(data_ref)
                index_interval = intervals.get(indices_ref)
                data_type = value_types.get(data_ref)
                axis = operation.attrs.get("axis")
                if (
                    data_interval is not None
                    and index_interval is not None
                    and data_type is not None
                    and type(axis) is int
                    and 0 <= axis < len(data_type.shape)
                ):
                    extent: Any = data_type.shape[axis]
                    if isinstance(extent, str):
                        extent = resolved_dimensions.get(extent)
                    if (
                        type(extent) is int
                        and extent > 0
                        and index_interval.contained_in(0, extent - 1)
                    ):
                        # A proven in-bounds gather can only select values from
                        # its data carrier.  The full data interval is a sound
                        # (possibly conservative) interval for its result.
                        interval = data_interval
            elif operation.op in preserving_ops and operation.operands:
                operands = [intervals.get(name) for name in operation.operands]
                if all(item is not None for item in operands):
                    closed_operands = [item for item in operands if item is not None]
                    interval = IntegerInterval(
                        min(item.lower for item in closed_operands),
                        max(item.upper for item in closed_operands),
                    )
            if interval is None:
                continue
            interval = _within_dtype(interval, result_type.dtype)
            if interval is None:
                continue
            intervals[result_name] = interval
            changed = True
        if not changed:
            break
    return intervals


def proves_index_bounds(
    j: Any,
    relation: Any,
    *,
    logical_inputs: Mapping[str, Any] | None = None,
    dimension_extents: Mapping[str, int] | None = None,
) -> bool:
    """Prove ``0 <= indices < dim(data, axis)`` from J and supplied J inputs."""

    canonical = elaborate_j(j)
    value_types = j_elaboration(canonical).value_type_by_name
    data_type = value_types.get(relation.data)
    if data_type is None or relation.axis < 0 or relation.axis >= len(data_type.shape):
        return False
    extent = data_type.shape[relation.axis]
    if isinstance(extent, str):
        extent = dict(dimension_extents or {}).get(extent)
    if type(extent) is not int or extent < 0:
        return False
    interval = infer_integer_value_intervals(
        canonical,
        logical_inputs=logical_inputs,
        dimension_extents=dimension_extents,
    ).get(relation.indices)
    return interval is not None and interval.contained_in(0, extent - 1)


__all__ = [
    "IntegerInterval",
    "infer_integer_value_intervals",
    "proves_index_bounds",
]
