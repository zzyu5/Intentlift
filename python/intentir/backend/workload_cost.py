"""Deterministic, J-derived workload cost for operational Roofline evidence.

This module counts semantic work, not target instructions or hardware memory
transactions.  Boundary traffic is exact for the concrete MaterializedJ ABI;
intermediate traffic is deliberately excluded so the result is a documented
logical/minimum-traffic Roofline input rather than a profiler substitute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from intentir.backend.materialize import MaterializedJ
from intentir.ir.canonical_j import j_elaboration
from intentir.ir.model import IntentOp, walk_intent_ops


J_WORKLOAD_COST_PROTOCOL = "intentir.backend.j-workload-cost.v1"

_FLOAT_DTYPES = frozenset({"f16", "bf16", "f32", "f64"})
_DTYPE_BITS = {
    # Ranked memref/tensor ABI stores one i1/bool value per byte on the current
    # CUDA and RVV runners.  This is concrete boundary traffic, not packed-wire
    # information theory.
    "i1": 8,
    "bool": 8,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "u8": 8,
    "u16": 16,
    "u32": 32,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "f64": 64,
}


def _dtype(dtype: str) -> str:
    return dtype.removeprefix("torch.").lower()


def _numel(shape: Iterable[Any], extents: dict[str, int]) -> int:
    result = 1
    for dimension in shape:
        value = extents[dimension] if isinstance(dimension, str) else dimension
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"non-concrete workload dimension {dimension!r}")
        result *= value
    return result


def _tensor_bytes(dtype: str, shape: Iterable[Any], extents: dict[str, int]) -> int:
    bits = _DTYPE_BITS.get(_dtype(dtype))
    if bits is None:
        raise ValueError(f"unsupported workload-cost dtype {dtype!r}")
    return math.ceil(_numel(shape, extents) * bits / 8)


@dataclass
class _Counts:
    floating_point_ops: int = 0
    comparison_ops: int = 0
    special_function_ops: int = 0
    integer_logic_ops: int = 0
    type_conversion_ops: int = 0
    select_ops: int = 0
    data_movement_elements: int = 0
    atomic_rmw_ops: int = 0
    control_iterations: int = 0
    unsupported: list[str] = field(default_factory=list)
    dynamic_branches: int = 0

    def add(self, other: "_Counts", multiplier: int = 1) -> None:
        for name in (
            "floating_point_ops",
            "comparison_ops",
            "special_function_ops",
            "integer_logic_ops",
            "type_conversion_ops",
            "select_ops",
            "data_movement_elements",
            "atomic_rmw_ops",
            "control_iterations",
            "dynamic_branches",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name) * multiplier)
        self.unsupported.extend(other.unsupported)


@dataclass(frozen=True)
class JWorkloadCost:
    boundary_input_bytes: int | None
    boundary_output_bytes: int | None
    floating_point_ops: int
    comparison_ops: int
    special_function_ops: int
    integer_logic_ops: int
    type_conversion_ops: int
    select_ops: int
    data_movement_elements: int
    atomic_rmw_ops: int
    control_iterations: int
    dynamic_branches: int
    unsupported_semantic_ops: tuple[str, ...]

    @property
    def boundary_io_bytes(self) -> int | None:
        if self.boundary_input_bytes is None or self.boundary_output_bytes is None:
            return None
        return self.boundary_input_bytes + self.boundary_output_bytes

    @property
    def arithmetic_intensity_flop_per_byte(self) -> float | None:
        if self.boundary_io_bytes is None:
            return None
        return (
            self.floating_point_ops / self.boundary_io_bytes
            if self.boundary_io_bytes
            else float("inf")
        )

    @property
    def complete(self) -> bool:
        return (
            self.boundary_input_bytes is not None
            and self.boundary_output_bytes is not None
            and not self.unsupported_semantic_ops
            and self.dynamic_branches == 0
        )

    def to_record(self) -> dict[str, Any]:
        caveats = [
            "boundary bytes exclude intermediate/cache-line/transaction traffic",
            "one add/sub/mul/div counts as one FLOP; FMA would count as two",
        ]
        if self.special_function_ops:
            caveats.append(
                "transcendentals are reported separately and are not assigned an arbitrary FLOP weight"
            )
        if self.comparison_ops:
            caveats.append("comparisons are reported separately from FLOPs")
        return {
            "protocol": J_WORKLOAD_COST_PROTOCOL,
            "complete": self.complete,
            "boundary_input_bytes": self.boundary_input_bytes,
            "boundary_output_bytes": self.boundary_output_bytes,
            "boundary_io_bytes": self.boundary_io_bytes,
            "floating_point_ops": self.floating_point_ops,
            "comparison_ops": self.comparison_ops,
            "special_function_ops": self.special_function_ops,
            "integer_logic_ops": self.integer_logic_ops,
            "type_conversion_ops": self.type_conversion_ops,
            "select_ops": self.select_ops,
            "data_movement_elements": self.data_movement_elements,
            "atomic_rmw_ops": self.atomic_rmw_ops,
            "control_iterations": self.control_iterations,
            "dynamic_branches": self.dynamic_branches,
            "unsupported_semantic_ops": list(self.unsupported_semantic_ops),
            "arithmetic_intensity_flop_per_byte": (
                self.arithmetic_intensity_flop_per_byte
            ),
            "traffic_interpretation": "certified-boundary-logical-minimum",
            "caveats": caveats,
        }


def _contraction_flops(
    operation: IntentOp,
    value_types: dict[str, Any],
    extents: dict[str, int],
) -> int | None:
    subscripts = operation.attrs.get("subscripts")
    if not isinstance(subscripts, str) or "->" not in subscripts:
        return None
    inputs_text, output_text = subscripts.replace(" ", "").split("->", 1)
    input_labels = inputs_text.split(",")
    if len(input_labels) != len(operation.operands):
        return None
    label_extents: dict[str, int] = {}
    for labels, operand in zip(input_labels, operation.operands, strict=True):
        value_type = value_types.get(operand)
        if value_type is None or len(labels) != len(value_type.shape):
            return None
        for label, dimension in zip(labels, value_type.shape, strict=True):
            extent = _numel((dimension,), extents)
            previous = label_extents.setdefault(label, extent)
            if previous != extent:
                return None
    if any(label not in label_extents for label in output_text):
        return None
    output_elements = math.prod(label_extents[label] for label in output_text)
    reduction_labels = set("".join(input_labels)) - set(output_text)
    reduction_elements = math.prod(
        (label_extents[label] for label in reduction_labels), start=1
    )
    return output_elements * max(1, 2 * reduction_elements - 1)


def derive_j_workload_cost(materialized: MaterializedJ) -> JWorkloadCost:
    if not isinstance(materialized, MaterializedJ):
        raise TypeError("derive_j_workload_cost requires MaterializedJ")
    extents = materialized.extents
    elaboration = j_elaboration(materialized.j)
    value_types = elaboration.value_type_by_name
    runtime_inputs = set(materialized.runtime_input_names)
    boundary_shape_issues: list[str] = []

    def boundary_bytes(items: Iterable[Any], *, kind: str) -> int | None:
        total = 0
        dynamic = False
        for item in items:
            if any(dimension == "?" for dimension in item.shape):
                boundary_shape_issues.append(
                    f"dynamic_{kind}_shape:{item.name}"
                )
                dynamic = True
                continue
            total += _tensor_bytes(item.dtype, item.shape, extents)
        return None if dynamic else total

    input_bytes = boundary_bytes(
        (
            item
            for item in materialized.j.intent.inputs
            if item.name in runtime_inputs
        ),
        kind="input",
    )
    output_bytes = boundary_bytes(materialized.j.intent.outputs, kind="output")
    constexpr = dict(materialized.constexpr_bindings)

    def result_type(operation: IntentOp) -> Any | None:
        return (
            value_types.get(operation.result)
            if isinstance(operation.result, str)
            else None
        )

    def ragged_field_elements(
        operation: IntentOp,
        field_name: str,
    ) -> int | None:
        if not operation.operands:
            return None
        descriptor = value_types.get(operation.operands[0])
        if descriptor is None or not descriptor.is_ragged:
            return None
        field = descriptor.field(field_name)
        if not field.is_tensor or field.dtype is None:
            return None
        return _numel(field.shape, extents)

    def body_spans_axis(body: list[IntentOp], axis: str) -> bool:
        """Whether the functional body already materializes the loop axis.

        Some lifted Triton loops retain execution recurrence for replay while
        their typed functional operations already produce tensors containing
        the full logical trip axis.  Multiplying such tensor work by the trip
        count a second time turns linear elementwise work into a quadratic cost.
        """

        return any(
            (value_type := result_type(operation)) is not None
            and axis in value_type.shape
            for operation in walk_intent_ops(body)
        )

    def visit(body: list[IntentOp]) -> _Counts:
        counts = _Counts()
        for operation in body:
            if operation.op == "loop":
                if operation.trip is None:
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:loop_trip"
                    )
                    continue
                if operation.trip.is_dynamic:
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:dynamic_loop_trip"
                    )
                    counts.add(visit(operation.body))
                    continue
                if operation.trip.axis not in extents:
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:loop_trip"
                    )
                    continue
                trip_count = extents[operation.trip.axis]
                counts.control_iterations += trip_count
                body_counts = visit(operation.body)
                counts.add(
                    body_counts,
                    1
                    if body_spans_axis(operation.body, operation.trip.axis)
                    else trip_count,
                )
                continue
            if operation.op in {"effect_if", "value_if"}:
                assert operation.positive is not None and operation.negative is not None
                predicate = constexpr.get(operation.predicate or "")
                if isinstance(predicate, bool):
                    arm = operation.positive if predicate else operation.negative
                    counts.add(visit(arm.body))
                else:
                    counts.dynamic_branches += 1
                    positive = visit(operation.positive.body)
                    negative = visit(operation.negative.body)
                    counts.add(positive)
                    counts.add(negative)
                continue
            if operation.op == "ragged":
                offsets = value_types.get(operation.operands[0]) if len(
                    operation.operands
                ) == 2 else None
                indices = value_types.get(operation.operands[1]) if len(
                    operation.operands
                ) == 2 else None
                if (
                    offsets is None
                    or indices is None
                    or not offsets.is_tensor
                    or not indices.is_tensor
                ):
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:ragged_shape"
                    )
                    continue
                counts.data_movement_elements += _numel(offsets.shape, extents)
                counts.data_movement_elements += _numel(indices.shape, extents)
                continue
            if operation.op == "ragged_outer":
                offsets = ragged_field_elements(operation, "offsets")
                if offsets is None:
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:ragged_outer_shape"
                    )
                    continue
                counts.data_movement_elements += 1
                continue
            if operation.op == "ragged_member":
                offsets = ragged_field_elements(operation, "offsets")
                if offsets is None:
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:ragged_member_shape"
                    )
                    continue
                counts.data_movement_elements += 2
                continue
            value_type = result_type(operation)
            if value_type is not None and (
                not value_type.is_tensor or value_type.dtype is None
            ):
                counts.unsupported.append(
                    f"{operation.node_id or 'anonymous'}:{operation.op}_record_type"
                )
                continue
            if value_type is not None and any(
                dimension == "?" for dimension in value_type.shape
            ):
                counts.unsupported.append(
                    f"{operation.node_id or 'anonymous'}:dynamic_result_shape"
                )
                continue
            elements = (
                _numel(value_type.shape, extents) if value_type is not None else 0
            )
            dtype = _dtype(value_type.dtype) if value_type is not None else ""
            if operation.op == "map_binary":
                function = operation.attrs.get("fn")
                if function in {"lt", "le", "gt", "ge", "eq", "ne", "max"}:
                    counts.comparison_ops += elements
                elif function in {"and", "or", "xor"} or dtype not in _FLOAT_DTYPES:
                    counts.integer_logic_ops += elements
                elif function in {
                    "add", "sub", "mul", "div", "floordiv", "remainder",
                    "copysign",
                }:
                    counts.floating_point_ops += elements
                else:
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:map_binary:{function}"
                    )
            elif operation.op == "map_unary":
                function = operation.attrs.get("fn")
                if function == "abs":
                    if dtype in _FLOAT_DTYPES:
                        counts.floating_point_ops += elements
                    else:
                        counts.integer_logic_ops += elements
                elif function in {
                    "exp",
                    "log",
                    "log2",
                    "sqrt",
                    "rsqrt",
                    "tanh",
                    "sigmoid",
                    "cos",
                    "sin",
                    "tan",
                    "exp2",
                    "atan",
                    "erf",
                    "acos",
                    "asin",
                    "cbrt",
                    "ceil",
                    "floor",
                    "round",
                    "roundeven",
                    "trunc",
                    "log10",
                    "log1p",
                    "expm1",
                    "erfc",
                }:
                    counts.special_function_ops += elements
                elif function in {"isnan", "isinf", "isfinite"}:
                    counts.comparison_ops += elements
                elif function == "logical_not":
                    counts.integer_logic_ops += elements
                elif function == "identity":
                    pass
                elif function == "sign":
                    if dtype in _FLOAT_DTYPES:
                        counts.floating_point_ops += elements
                    else:
                        counts.integer_logic_ops += elements
                elif function in {"neg", "recip"}:
                    if dtype in _FLOAT_DTYPES:
                        counts.floating_point_ops += elements
                    else:
                        counts.integer_logic_ops += elements
                else:
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:map_unary:{function}"
                    )
            elif operation.op == "reduce":
                operand_type = value_types.get(operation.operands[0])
                axes = operation.attrs.get("axes", [])
                if operand_type is None or not isinstance(axes, list):
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:reduce_shape"
                    )
                    continue
                normalized_axes = [
                    axis if axis >= 0 else len(operand_type.shape) + axis
                    for axis in axes
                ]
                reduction_elements = math.prod(
                    _numel((operand_type.shape[axis],), extents)
                    for axis in normalized_axes
                )
                combinations = elements * max(0, reduction_elements - 1)
                function = operation.attrs.get("fn")
                if function == "sum" and _dtype(operand_type.dtype) in _FLOAT_DTYPES:
                    counts.floating_point_ops += combinations
                elif function in {"max", "min"}:
                    counts.comparison_ops += combinations
                else:
                    counts.integer_logic_ops += combinations
            elif operation.op == "scan":
                operand_type = value_types.get(operation.operands[0])
                axis = operation.attrs.get("axis")
                if operand_type is None or not isinstance(axis, int):
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:scan_shape"
                    )
                    continue
                axis = axis if axis >= 0 else len(operand_type.shape) + axis
                lanes = _numel(operand_type.shape, extents) // _numel(
                    (operand_type.shape[axis],), extents
                )
                combinations = lanes * max(
                    0, _numel((operand_type.shape[axis],), extents) - 1
                )
                if operation.attrs.get("fn") == "sum" and _dtype(
                    operand_type.dtype
                ) in _FLOAT_DTYPES:
                    counts.floating_point_ops += combinations
                else:
                    counts.integer_logic_ops += combinations
            elif operation.op == "contraction":
                flops = _contraction_flops(operation, value_types, extents)
                if flops is None:
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:contraction"
                    )
                else:
                    counts.floating_point_ops += flops
            elif operation.op == "cast":
                counts.type_conversion_ops += elements
            elif operation.op == "select":
                counts.select_ops += elements
            elif operation.op in {"gather", "pad", "reshape"}:
                counts.data_movement_elements += elements
            elif operation.op == "scatter":
                if len(operation.operands) != 3:
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:scatter_shape"
                    )
                    continue
                source_type = value_types.get(operation.operands[2])
                if (
                    source_type is None
                    or not source_type.is_tensor
                    or source_type.dtype is None
                ):
                    counts.unsupported.append(
                        f"{operation.node_id or 'anonymous'}:scatter_shape"
                    )
                    continue
                source_elements = _numel(source_type.shape, extents)
                counts.data_movement_elements += source_elements
                if operation.attrs.get("atomic") is not None:
                    counts.atomic_rmw_ops += source_elements
            elif operation.op == "iota":
                counts.integer_logic_ops += elements
            elif operation.op != "const":
                counts.unsupported.append(
                    f"{operation.node_id or 'anonymous'}:{operation.op}"
                )
        return counts

    counts = visit(materialized.j.intent.body)
    return JWorkloadCost(
        boundary_input_bytes=input_bytes,
        boundary_output_bytes=output_bytes,
        floating_point_ops=counts.floating_point_ops,
        comparison_ops=counts.comparison_ops,
        special_function_ops=counts.special_function_ops,
        integer_logic_ops=counts.integer_logic_ops,
        type_conversion_ops=counts.type_conversion_ops,
        select_ops=counts.select_ops,
        data_movement_elements=counts.data_movement_elements,
        atomic_rmw_ops=counts.atomic_rmw_ops,
        control_iterations=counts.control_iterations,
        dynamic_branches=counts.dynamic_branches,
        unsupported_semantic_ops=tuple(
            sorted(set((*boundary_shape_issues, *counts.unsupported)))
        ),
    )


__all__ = [
    "J_WORKLOAD_COST_PROTOCOL",
    "JWorkloadCost",
    "derive_j_workload_cost",
]
