"""Derive a closed paired-last-axis rotation view from one accepted J.

The view is ephemeral compiler state.  It is emitted only when Layer A proves
the two-output rotation arithmetic and Layer B proves two disjoint stores that
cover the low/high halves of each inout tensor.  No kernel identifier, Source
path, C value, target, or sampled runtime value participates in the match.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from intentir.ir.canonical_j import ContractExpr, elaborate_j, j_elaboration
from intentir.ir.model import IntentFunction, IntentOp
from intentir.verify.gate1.model import ParameterMode, SemanticArchive


PAIRED_ROTATION_VIEW_VERSION = "intentir.paired-last-axis-rotation.v1"


@dataclass(frozen=True)
class PairedRotationView:
    data_inputs: tuple[str, str]
    cosine_input: str
    sine_input: str
    layout: str
    sections: tuple[int, ...] = ()

    def input_indices(self, function: IntentFunction) -> tuple[int, int, int, int]:
        positions = {item.name: index for index, item in enumerate(function.inputs)}
        names = (*self.data_inputs, self.cosine_input, self.sine_input)
        if any(name not in positions for name in names):
            raise ValueError("paired rotation input was removed from the runtime ABI")
        return tuple(positions[name] for name in names)  # type: ignore[return-value]


def _op_results(operation: IntentOp) -> tuple[str, ...]:
    if isinstance(operation.result, str):
        return (operation.result,)
    if isinstance(operation.result, list):
        return tuple(item for item in operation.result if isinstance(item, str))
    return ()


def _top_level_node_operations(
    j: SemanticArchive,
) -> tuple[dict[str, IntentOp], dict[str, str]] | None:
    typed_by_ref = j_elaboration(j).construction_records.typed_operation_by_ref(j.node_ids)
    if typed_by_ref is None:
        return None
    by_node: dict[str, IntentOp] = {}
    result_nodes: dict[str, str] = {}
    for node_ref, path in j.node_ids.items():
        matched = re.fullmatch(r"body\[([0-9]+)\]", path)
        if matched is None:
            return None
        operation = typed_by_ref.get(node_ref)
        if operation is None:
            return None
        by_node[node_ref] = operation
        for result in _op_results(operation):
            result_nodes[result] = node_ref
    return by_node, result_nodes


def _top_level_operations(j: SemanticArchive) -> tuple[IntentOp, ...] | None:
    typed_rows = j_elaboration(j).typed_operation_rows
    if typed_rows is None:
        return None
    return tuple(
        operation
        for path, operation in typed_rows
        if re.fullmatch(r"body\[\d+\]", path)
    )


def _fn(operation: IntentOp, name: str) -> bool:
    return operation.op == "map_binary" and operation.attrs.get("fn") == name


def _binary_inputs(operation: IntentOp) -> tuple[str, str] | None:
    if len(operation.operands) != 2:
        return None
    lhs, rhs = operation.operands
    if not isinstance(lhs, str) or not isinstance(rhs, str):
        return None
    return lhs, rhs


def _dependency_roots(
    value: str,
    operations: Mapping[str, IntentOp],
    input_names: frozenset[str],
    active: frozenset[str] = frozenset(),
) -> frozenset[str]:
    if value in input_names:
        return frozenset({value})
    if value in active:
        return frozenset()
    operation = operations.get(value)
    if operation is None:
        return frozenset()
    roots: set[str] = set()
    for operand in operation.operands:
        if isinstance(operand, str):
            roots.update(
                _dependency_roots(
                    operand,
                    operations,
                    input_names,
                    active | {value},
                )
            )
    return frozenset(roots)


def _flatten_and(expression: ContractExpr | None) -> tuple[ContractExpr, ...]:
    if expression is None:
        return ()
    if expression.op != "and":
        return (expression,)
    return tuple(
        nested for item in expression.args for nested in _flatten_and(item)
    )


def _ref(expression: ContractExpr) -> str | None:
    return expression.value if expression.op == "ref" else None


def _int(expression: ContractExpr) -> int | None:
    return (
        expression.value
        if expression.op == "int"
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
        else None
    )


def _offset_axis(
    component: ContractExpr,
) -> tuple[str, ContractExpr] | None:
    if component.op != "add" or len(component.args) != 2:
        return None
    lhs, rhs = component.args
    if (axis := _ref(lhs)) is not None:
        return axis, rhs
    if (axis := _ref(rhs)) is not None:
        return axis, lhs
    return None


def _is_half_expression(
    expression: ContractExpr,
    dimension: str,
    value_dimensions: Mapping[str, str],
) -> bool:
    if expression.op == "floordiv" and len(expression.args) == 2:
        lhs, rhs = expression.args
        return _ref(lhs) == dimension and _int(rhs) == 2
    if (
        expression.op == "call"
        and expression.value == "ceil_div"
        and len(expression.args) == 2
    ):
        lhs, rhs = expression.args
        value_ref = _ref(lhs)
        return (
            value_ref is not None
            and value_dimensions.get(value_ref) == dimension
            and _int(rhs) == 2
        )
    return False


def _predicate_has_half_bound(
    expression: ContractExpr | None,
    axis: str,
    dimension: str,
    value_dimensions: Mapping[str, str],
) -> bool:
    return any(
        term.op == "lt"
        and len(term.args) == 2
        and _ref(term.args[0]) == axis
        and _is_half_expression(term.args[1], dimension, value_dimensions)
        for term in _flatten_and(expression)
    )


def _shape(
    value: Sequence[int | str], extents: Mapping[str, int]
) -> tuple[int, ...] | None:
    result: list[int] = []
    for extent in value:
        if isinstance(extent, int):
            resolved = extent
        elif extent in extents:
            resolved = int(extents[extent])
        else:
            return None
        if resolved <= 0:
            return None
        result.append(resolved)
    return tuple(result)


def _const_value(operation: IntentOp | None, value: int) -> bool:
    return bool(
        operation is not None
        and operation.op == "const"
        and operation.attrs.get("value") == value
    )


def _commutative_operands(
    operation: IntentOp | None,
    fn: str,
    lhs: str,
    rhs: str,
) -> bool:
    operands = _binary_inputs(operation) if operation is not None else None
    return bool(
        operation is not None
        and _fn(operation, fn)
        and operands is not None
        and {operands[0], operands[1]} == {lhs, rhs}
    )


def _interleaved_rotation_view(
    j: SemanticArchive,
    concrete_extents: Mapping[str, int],
    inout: tuple[str, str],
    tensors: Mapping[str, object],
    operations: Mapping[str, IntentOp],
    node_operations: Mapping[str, IntentOp],
) -> PairedRotationView | None:
    """Recognize the exact even/odd Llama-style paired rotation graph.

    The projection is deliberately closed and structural: it verifies the
    shared ``2*k``/``2*k+1`` index producers, both output equations for both
    inout tensors, the angle planes and the dynamic imaginary sign.  It is not
    selected by a kernel identifier or Source path.
    """

    body = _top_level_operations(j)
    if body is None or len(body) != 38:
        return None
    tensor_map = {item.name: item for item in j.intent.inputs}
    q_shape = _shape(tensor_map[inout[0]].shape, concrete_extents)
    k_shape = _shape(tensor_map[inout[1]].shape, concrete_extents)
    if (
        q_shape is None
        or k_shape is None
        or len(q_shape) != 4
        or len(k_shape) != 4
        or q_shape[:2] != k_shape[:2]
        or q_shape[3] != k_shape[3]
        or q_shape[3] % 2
    ):
        return None
    half = q_shape[3] // 2
    remaining = [item for item in j.intent.inputs if item.name not in inout]
    angle_candidates = [
        item
        for item in remaining
        if _shape(item.shape, concrete_extents) == (q_shape[1], half, 2)
        and item.dtype == tensor_map[inout[0]].dtype
    ]
    sign_candidates = [
        item
        for item in remaining
        if _shape(item.shape, concrete_extents) == () and item.dtype == "f32"
    ]
    if len(angle_candidates) != 1 or len(sign_candidates) != 1:
        return None
    angle = angle_candidates[0].name
    sign = sign_candidates[0].name

    results = {
        result: operation
        for operation in body
        for result in _op_results(operation)
    }
    iotas = [operation for operation in body if operation.op == "iota"]
    if len(iotas) != 1 or not isinstance(iotas[0].result, str):
        return None
    pair = iotas[0].result
    even_candidates = [
        operation
        for operation in body
        if _commutative_operands(
            operation,
            "mul",
            pair,
            next(
                (
                    result
                    for result, producer in results.items()
                    if _const_value(producer, 2)
                ),
                "",
            ),
        )
    ]
    if len(even_candidates) != 1 or not isinstance(even_candidates[0].result, str):
        return None
    even = even_candidates[0].result
    one_values = {
        result for result, producer in results.items() if _const_value(producer, 1)
    }
    odd_candidates = [
        operation
        for operation in body
        if isinstance(operation.result, str)
        and operation.op == "map_binary"
        and operation.attrs.get("fn") == "add"
        and (operands := _binary_inputs(operation)) is not None
        and even in operands
        and any(value in operands for value in one_values)
    ]
    if len(odd_candidates) != 1:
        return None
    odd = odd_candidates[0].result
    assert isinstance(odd, str)

    angle_gathers = [
        operation
        for operation in body
        if operation.op == "gather"
        and operation.operands
        and operation.operands[0] == angle
        and operation.attrs.get("axis") == 2
        and operation.attrs.get("mode") == "take"
    ]
    if len(angle_gathers) != 2:
        return None
    cosine: str | None = None
    sine_raw: str | None = None
    for gather in angle_gathers:
        if not isinstance(gather.result, str) or len(gather.operands) != 2:
            return None
        index = results.get(gather.operands[1])
        if _const_value(index, 0):
            cosine = gather.result
        elif _const_value(index, 1):
            sine_raw = gather.result
    if cosine is None or sine_raw is None:
        return None
    sine_casts = [
        operation
        for operation in body
        if operation.op == "cast"
        and operation.operands == [sine_raw]
        and operation.attrs.get("to_dtype") == "f32"
    ]
    if len(sine_casts) != 1 or not isinstance(sine_casts[0].result, str):
        return None
    sine_scaled_candidates = [
        operation
        for operation in body
        if _commutative_operands(operation, "mul", sine_casts[0].result, sign)
    ]
    if len(sine_scaled_candidates) != 1 or not isinstance(
        sine_scaled_candidates[0].result, str
    ):
        return None
    sine_scaled = sine_scaled_candidates[0].result

    authored = {item.access_id: item for item in j.boundary.accesses}
    structured = j_elaboration(j).access_by_id
    for data_ref in inout:
        data_gathers = [
            operation
            for operation in body
            if operation.op == "gather"
            and operation.operands
            and operation.operands[0] == data_ref
            and operation.attrs.get("axis") == 3
            and operation.attrs.get("mode") == "take"
        ]
        if len(data_gathers) != 2:
            return None
        by_index = {
            operation.operands[1]: operation.result for operation in data_gathers
        }
        even_value = by_index.get(even)
        odd_value = by_index.get(odd)
        if not isinstance(even_value, str) or not isinstance(odd_value, str):
            return None
        even_casts = [
            operation
            for operation in body
            if operation.op == "cast"
            and operation.operands == [even_value]
            and operation.attrs.get("to_dtype") == "f32"
        ]
        odd_casts = [
            operation
            for operation in body
            if operation.op == "cast"
            and operation.operands == [odd_value]
            and operation.attrs.get("to_dtype") == "f32"
        ]
        cosine_casts = [
            operation
            for operation in body
            if operation.op == "cast"
            and operation.operands == [cosine]
            and operation.attrs.get("to_dtype") == "f32"
        ]
        if len(even_casts) != 1 or len(odd_casts) != 1 or not cosine_casts:
            return None
        even_f32 = even_casts[0].result
        odd_f32 = odd_casts[0].result
        cosine_f32_values = {
            operation.result
            for operation in cosine_casts
            if isinstance(operation.result, str)
        }
        if not isinstance(even_f32, str) or not isinstance(odd_f32, str) or not cosine_f32_values:
            return None

        low_cos = next(
            (
                operation.result
                for operation in body
                if any(
                    _commutative_operands(operation, "mul", even_f32, cosine_f32)
                    for cosine_f32 in cosine_f32_values
                )
            ),
            None,
        )
        high_sin = next(
            (
                operation.result
                for operation in body
                if _commutative_operands(operation, "mul", odd_f32, sine_scaled)
            ),
            None,
        )
        high_cos = next(
            (
                operation.result
                for operation in body
                if _commutative_operands(operation, "mul", odd_value, cosine)
            ),
            None,
        )
        low_sin = next(
            (
                operation.result
                for operation in body
                if _commutative_operands(operation, "mul", even_f32, sine_scaled)
            ),
            None,
        )
        if not all(isinstance(item, str) for item in (low_cos, high_sin, high_cos, low_sin)):
            return None
        high_cos_casts = [
            operation.result
            for operation in body
            if operation.op == "cast"
            and operation.operands == [high_cos]
            and operation.attrs.get("to_dtype") == "f32"
            and isinstance(operation.result, str)
        ]
        low_outputs = [
            operation
            for operation in body
            if operation.op == "map_binary"
            and operation.attrs.get("fn") == "sub"
            and _binary_inputs(operation) == (low_cos, high_sin)
        ]
        high_outputs = [
            operation
            for operation in body
            if operation.op == "map_binary"
            and operation.attrs.get("fn") == "add"
            and isinstance(operation.result, str)
            and high_cos_casts
            and _commutative_operands(
                operation, "add", low_sin, high_cos_casts[0]
            )
        ]
        if len(low_outputs) != 1 or len(high_outputs) != 1:
            return None
        low_cast = next(
            (
                operation
                for operation in body
                if operation.op == "cast"
                and operation.operands == [low_outputs[0].result]
                and operation.attrs.get("to_dtype") == tensor_map[data_ref].dtype
            ),
            None,
        )
        high_cast = next(
            (
                operation
                for operation in body
                if operation.op == "cast"
                and operation.operands == [high_outputs[0].result]
                and operation.attrs.get("to_dtype") == tensor_map[data_ref].dtype
            ),
            None,
        )
        if low_cast is None or high_cast is None:
            return None
        stored_nodes = {
            access.node_ref
            for access in j.boundary.accesses
            if access.tensor == data_ref
            and getattr(access.kind, "value", access.kind) == "store"
        }
        expected_nodes = {
            node_ref
            for node_ref, operation in node_operations.items()
            if operation is low_cast or operation is high_cast
        }
        if stored_nodes != expected_nodes or len(stored_nodes) != 2:
            return None
        for access_id, parsed in structured.items():
            access = authored[access_id]
            if access.node_ref not in stored_nodes:
                continue
            if parsed.predicate is None or len(parsed.index.components) != 4:
                return None

    return PairedRotationView(inout, angle, sign, "interleaved")


def _plane_sections(
    j: SemanticArchive,
    cosine: str,
    half: int,
    scalar_bindings: Mapping[str, bool | int | float],
) -> tuple[int, int, int] | None:
    structured = j_elaboration(j).access_by_id
    authored = {item.access_id: item for item in j.boundary.accesses}
    planes: dict[int, ContractExpr] = {}
    plane_axes: set[str] = set()
    for access_id, parsed in structured.items():
        access = authored[access_id]
        if access.tensor != cosine or getattr(access.kind, "value", access.kind) != "load":
            continue
        if len(parsed.index.components) != 4:
            continue
        plane = _int(parsed.index.components[0])
        plane_axis = _ref(parsed.index.components[-1])
        if (
            plane in {0, 1, 2}
            and plane_axis is not None
            and parsed.predicate is not None
        ):
            planes[int(plane)] = parsed.predicate
            plane_axes.add(plane_axis)
    if set(planes) != {0, 1, 2} or len(plane_axes) != 1:
        return None
    axis = next(iter(plane_axes))

    section0_ref: str | None = None
    for term in _flatten_and(planes[0]):
        if term.op == "lt" and len(term.args) == 2 and _ref(term.args[0]) == axis:
            section0_ref = _ref(term.args[1])
    if section0_ref is None:
        return None
    section1_ref: str | None = None
    for term in _flatten_and(planes[1]):
        if term.op != "lt" or len(term.args) != 2 or _ref(term.args[0]) != axis:
            continue
        upper = term.args[1]
        if upper.op != "add" or len(upper.args) != 2:
            continue
        refs = [_ref(item) for item in upper.args]
        if section0_ref in refs:
            section1_ref = refs[1 - refs.index(section0_ref)]
    if section1_ref is None:
        return None
    first = scalar_bindings.get(section0_ref)
    second = scalar_bindings.get(section1_ref)
    if (
        isinstance(first, bool)
        or not isinstance(first, int)
        or isinstance(second, bool)
        or not isinstance(second, int)
    ):
        return None
    third = half - first - second
    if first <= 0 or second <= 0 or third <= 0:
        return None
    return first, second, third


def derive_paired_rotation_view(
    j: SemanticArchive,
    concrete_extents: Mapping[str, int],
    scalar_bindings: Mapping[str, bool | int | float],
) -> PairedRotationView | None:
    """Return a paired rotation view only after both A and B close the pattern."""

    j = elaborate_j(j)
    node_maps = _top_level_node_operations(j)
    if node_maps is None:
        return None
    node_operations, result_nodes = node_maps
    body = _top_level_operations(j)
    if body is None:
        return None
    operations = {
        result: operation
        for operation in body
        for result in _op_results(operation)
    }
    tensors = {item.name: item for item in j.intent.inputs}
    inout = tuple(
        item.logical_ref
        for item in j.signature.parameters
        if item.mode is ParameterMode.INOUT
        and isinstance(item.logical_ref, str)
        and item.logical_ref in tensors
    )
    if len(inout) != 2 or {item.name for item in j.intent.outputs} != set(inout):
        return None
    data_shapes = [_shape(tensors[name].shape, concrete_extents) for name in inout]
    if any(shape is None or len(shape) != 4 for shape in data_shapes):
        return None
    q_shape, k_shape = data_shapes
    assert q_shape is not None and k_shape is not None
    if (
        q_shape[0] != k_shape[0]
        or q_shape[1] != k_shape[1]
        or q_shape[3] != k_shape[3]
        or q_shape[3] % 2 != 0
    ):
        return None
    interleaved = _interleaved_rotation_view(
        j,
        concrete_extents,
        inout,
        tensors,
        operations,
        node_operations,
    )
    if interleaved is not None:
        return interleaved
    last_dimension = tensors[inout[0]].shape[-1]
    if not isinstance(last_dimension, str) or tensors[inout[1]].shape[-1] != last_dimension:
        return None
    half = q_shape[3] // 2
    value_dimensions = {
        item.value_ref: item.dimension_ref for item in j.value_dimension_relations
    }

    structured = j_elaboration(j).access_by_id
    authored = {item.access_id: item for item in j.boundary.accesses}
    stores: dict[str, list[tuple[str, str, ContractExpr]]] = {name: [] for name in inout}
    node_accesses: dict[str, list[tuple[str, str]]] = {}
    for access_id, parsed in structured.items():
        access = authored[access_id]
        kind = getattr(access.kind, "value", access.kind)
        if access.node_ref is not None:
            node_accesses.setdefault(access.node_ref, []).append((access.tensor, kind))
        if kind != "store" or access.tensor not in stores or access.node_ref is None:
            continue
        if len(parsed.index.components) != 4:
            return None
        stores[access.tensor].append(
            (access.node_ref, access_id, parsed.index.components[-1])
        )

    input_names = frozenset(tensors)
    cosine: str | None = None
    sine: str | None = None
    for data_ref in inout:
        items = stores[data_ref]
        if len(items) != 2:
            return None
        low: tuple[str, str, ContractExpr] | None = None
        high: tuple[str, str, ContractExpr] | None = None
        for item in items:
            if _ref(item[2]) is not None:
                low = item
            elif _offset_axis(item[2]) is not None:
                high = item
        if low is None or high is None:
            return None
        axis = _ref(low[2])
        shifted = _offset_axis(high[2])
        if axis is None or shifted is None or shifted[0] != axis:
            return None
        if not _is_half_expression(shifted[1], last_dimension, value_dimensions):
            return None
        for _node, access_id, _component in (low, high):
            if not _predicate_has_half_bound(
                structured[access_id].predicate,
                axis,
                last_dimension,
                value_dimensions,
            ):
                return None

        low_op = node_operations.get(low[0])
        high_op = node_operations.get(high[0])
        if low_op is None or high_op is None or not _fn(low_op, "sub") or not _fn(high_op, "add"):
            return None
        low_inputs = _binary_inputs(low_op)
        high_inputs = _binary_inputs(high_op)
        if low_inputs is None or high_inputs is None:
            return None
        low_muls = [operations.get(value) for value in low_inputs]
        high_muls = [operations.get(value) for value in high_inputs]
        if any(operation is None or not _fn(operation, "mul") for operation in (*low_muls, *high_muls)):
            return None

        trig_sequence: list[str] = []
        data_sequence: list[str] = []
        for multiply in (*low_muls, *high_muls):
            assert multiply is not None
            operands = _binary_inputs(multiply)
            if operands is None:
                return None
            dependencies = [
                _dependency_roots(value, operations, input_names) for value in operands
            ]
            data_positions = [index for index, roots in enumerate(dependencies) if data_ref in roots]
            if len(data_positions) != 1:
                return None
            data_pos = data_positions[0]
            trig_pos = 1 - data_pos
            trig_roots = dependencies[trig_pos] - set(inout)
            if len(trig_roots) != 1:
                return None
            trig_sequence.append(next(iter(trig_roots)))

            data_value = operands[data_pos]
            source_node = result_nodes.get(data_value)
            if data_value == data_ref:
                source_node = result_nodes.get(_op_results(multiply)[0])
            if source_node is None:
                return None
            candidates = [
                structured[access.access_id]
                for access in j.boundary.accesses
                if access.node_ref == source_node
                and access.tensor == data_ref
                and getattr(access.kind, "value", access.kind) == "load"
            ]
            if len(candidates) != 1:
                return None
            component = candidates[0].index.components[-1]
            if _ref(component) == axis:
                data_sequence.append("low")
            else:
                shifted_load = _offset_axis(component)
                if (
                    shifted_load is None
                    or shifted_load[0] != axis
                    or not _is_half_expression(
                        shifted_load[1], last_dimension, value_dimensions
                    )
                ):
                    return None
                data_sequence.append("high")

        if data_sequence != ["low", "high", "high", "low"]:
            return None
        if trig_sequence[0] != trig_sequence[2] or trig_sequence[1] != trig_sequence[3]:
            return None
        if cosine is None:
            cosine, sine = trig_sequence[0], trig_sequence[1]
        elif (cosine, sine) != (trig_sequence[0], trig_sequence[1]):
            return None

    if cosine is None or sine is None or cosine == sine:
        return None
    if cosine in inout or sine in inout or cosine not in tensors or sine not in tensors:
        return None
    cosine_shape = _shape(tensors[cosine].shape, concrete_extents)
    sine_shape = _shape(tensors[sine].shape, concrete_extents)
    if cosine_shape is None or cosine_shape != sine_shape:
        return None
    if cosine_shape == (1, q_shape[1], q_shape[3]):
        return PairedRotationView(tuple(inout), cosine, sine, "shared_position")
    if cosine_shape == (3, q_shape[0], q_shape[1], q_shape[3]):
        sections = _plane_sections(
            j,
            cosine,
            half,
            scalar_bindings,
        )
        if sections is None:
            return None
        return PairedRotationView(
            tuple(inout), cosine, sine, "three_plane", sections
        )
    return None


__all__ = [
    "PAIRED_ROTATION_VIEW_VERSION",
    "PairedRotationView",
    "derive_paired_rotation_view",
]
