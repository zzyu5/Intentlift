"""Shared deterministic ``J -> intent MLIR`` materialization entry."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from intentir.ir.canonical_j import elaborate_j, j_elaboration
from intentir.ir.dimension_relation import resolve_dimension_relations
from intentir.ir.model import (
    IntentFunction,
    IntentOp,
    IntentReturn,
    loop_result_names,
    validate_intent_function_wiring,
    walk_intent_ops,
)
from intentir.ir.value_dimension import resolve_value_dimension_equalities

from .emit_mlir import BroadcastAxes, IndexBoundsProjection, emit_module

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from intentir.verify.gate1.model import SemanticArchive


FLAT_MATERIALIZATION = "intentir.materialize.flat-graph.v1"
LOGICAL_FULL_AXIS_STEP_MATERIALIZATION = (
    "intentir.materialize.logical-full-axis-step.v1"
)
STRUCTURED_SCAN_LOOP_MATERIALIZATION = (
    "intentir.materialize.structured-scan-loop.v1"
)
DYNAMIC_VALUE_LOOP_MATERIALIZATION = "intentir.materialize.dynamic-value-loop"
@dataclass(frozen=True)
class JModuleMaterialization:
    """One deterministic Intent-MLIR encoding and its registered form."""

    mlir: str
    form: str
    consumed_structures: tuple[str, ...] = ()


@dataclass(frozen=True)
class JFunctionMaterialization:
    """One registered target-neutral function view of canonical J."""

    function: IntentFunction
    form: str
    consumed_structures: tuple[str, ...] = ()
    loop_schedule_tile: int | None = None


def _walk(body: Sequence[IntentOp]) -> tuple[IntentOp, ...]:
    return tuple(walk_intent_ops(body))


def _value_handle_names(function: IntentFunction) -> set[str]:
    """Collect every logical/SSA handle that an ephemeral view must avoid."""

    names = {
        tensor.name for tensor in [*function.inputs, *function.outputs]
    }
    for binding in function.returns:
        names.update((binding.output, binding.value))
    for operation in _walk(function.body):
        results = (
            operation.result
            if isinstance(operation.result, list)
            else [operation.result]
        )
        names.update(
            value for value in results if isinstance(value, str)
        )
        names.update(
            value for value in operation.operands if isinstance(value, str)
        )
        if isinstance(operation.predicate, str):
            names.add(operation.predicate)
        for scan in operation.scan:
            names.update((scan.name, scan.source))
        for state in operation.carry:
            names.add(state.state_id)
            names.update(
                value
                for value in (
                    state.body_argument,
                    state.next_value,
                    state.result_value,
                    state.init_value,
                )
                if isinstance(value, str)
            )
        if operation.trip is not None:
            names.update(
                value
                for value in (
                    operation.trip.axis,
                    operation.trip.upper_bound,
                    operation.trip.index,
                )
                if isinstance(value, str)
            )
        for arm in (operation.positive, operation.negative):
            if arm is not None and hasattr(arm, "yield_value"):
                yielded = arm.yield_value
                if isinstance(yielded, str):
                    names.add(yielded)
    return names


def _fresh_ssa_handle(occupied: set[str]) -> str:
    """Allocate the lowest unused canonical ``v*`` handle."""

    index = 0
    while True:
        candidate = f"v{index}"
        if candidate not in occupied:
            occupied.add(candidate)
            return candidate
        index += 1


def _canonical_return_function(
    j: SemanticArchive,
) -> tuple[IntentFunction, tuple[str, ...]]:
    """Project the one elaborated output binding into the MLIR printer view."""

    bindings = j_elaboration(j).output_bindings
    inputs = {item.name: item for item in j.intent.inputs}
    outputs = {item.name: item for item in j.intent.outputs}
    body = list(j.intent.body)
    returns: list[IntentReturn] = []
    consumed: list[str] = []
    used_names = _value_handle_names(
        IntentFunction(
            name=j.intent.name,
            inputs=list(j.intent.inputs),
            outputs=list(j.intent.outputs),
            body=body,
            returns=[],
        )
    )
    for binding in bindings:
        value = binding.value
        if value in inputs:
            output = outputs[binding.output]
            if (
                inputs[value].dtype != output.dtype
                or inputs[value].shape != output.shape
            ):
                raise ValueError(
                    "direct input/output binding requires identical logical types"
                )
            result = _fresh_ssa_handle(used_names)
            body.append(
                IntentOp(
                    op="map_unary",
                    result=result,
                    operands=[value],
                    attrs={"fn": "identity"},
                    type={"dtype": output.dtype, "shape": list(output.shape)},
                )
            )
            returns.append(IntentReturn(binding.output, result))
            consumed.append(
                f"output_binding:{binding.output}:{value}:identity_materialized"
            )
        else:
            returns.append(IntentReturn(binding.output, value))
    function = IntentFunction(
        name=j.intent.name,
        inputs=list(j.intent.inputs),
        outputs=list(j.intent.outputs),
        body=body,
        returns=returns,
    )
    validate_intent_function_wiring(function)
    return function, tuple(consumed)


def _logical_full_axis_step_function(
    j: SemanticArchive,
    dims: Mapping[str, int],
) -> IntentFunction | None:
    """Inline the closed full-axis-step denotation of one logical loop.

    Canonical structured-reference semantics evaluates a logical loop once over
    the complete logical tensor domain when its trip extent is positive.  Scan
    names therefore denote their complete source tensors in this registered
    representation; they are not implementation tiles.  Explicit carry
    initializers become either aliases of dominating J values or typed tensor
    constants, while a null-initialized carry remains legal only when the body
    does not read it.  This deterministic projection uses J alone: it does not
    consult C, a kernel name, or source realization details.

    Other loop forms remain unsupported until their complete J denotation can be
    represented without inventing an initializer or schedule.
    """

    loop_indices = [
        index for index, operation in enumerate(j.intent.body) if operation.op == "loop"
    ]
    if len(loop_indices) != 1:
        return None
    loop_index = loop_indices[0]
    outer_prefix = j.intent.body[:loop_index]
    outer_suffix = j.intent.body[loop_index + 1 :]
    if any(
        operation.op == "loop"
        for operation in _walk([*outer_prefix, *outer_suffix])
    ):
        return None
    loop = j.intent.body[loop_index]
    if loop.trip is None or loop.trip.is_dynamic or loop.trip.axis not in dims:
        raise ValueError("materialize J loop requires a resolved logical trip extent")
    extent = dims[loop.trip.axis]
    if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
        raise ValueError(
            "null-initialized logical full-axis loop requires a positive extent"
        )
    if not loop.carry:
        return None
    if any(operation.op == "loop" for operation in _walk(loop.body)):
        return None

    inputs = {item.name: item for item in j.intent.inputs}
    aliases: dict[str, str] = {}
    for scan in loop.scan:
        source = inputs.get(scan.source)
        if source is None:
            # A top-level single-loop form has no other dominating producer.
            return None
        source_shape = [
            dims[item] if isinstance(item, str) and item in dims else item
            for item in source.shape
        ]
        if (
            scan.axis < 0
            or scan.axis >= len(source_shape)
            or source_shape[scan.axis] != extent
        ):
            raise ValueError(
                "logical loop scan extent differs from its semantic trip axis"
            )
        aliases[scan.name] = scan.source

    body_references = {
        operand
        for operation in _walk(loop.body)
        for operand in operation.operands
    }
    dominating_types: dict[str, tuple[str, list[object]]] = {
        name: (tensor.dtype, list(tensor.shape)) for name, tensor in inputs.items()
    }
    for operation in _walk(outer_prefix):
        results = (
            operation.result
            if isinstance(operation.result, list)
            else [operation.result]
        )
        if any(isinstance(result, str) for result in results):
            if (
                operation.type is None
                or len(results) != 1
                or operation.type.get("kind") in {"record", "ragged"}
            ):
                return None
            result = results[0]
            if not isinstance(result, str):
                return None
            dominating_types[result] = (
                operation.type["dtype"],
                list(operation.type.get("shape", [])),
            )

    carry_prefix: list[IntentOp] = []
    for state in loop.carry:
        if (
            state.type is None
            or state.type.kind != "tensor"
            or state.type.dtype is None
        ):
            return None
        init = state.init_value
        if init is None:
            if state.body_argument in body_references:
                return None
            continue
        if isinstance(init, str) and init in dominating_types:
            init_dtype, init_shape = dominating_types[init]
            if (
                init_dtype != state.type.dtype
                or init_shape != state.type.shape
            ):
                raise ValueError(
                    "logical loop carry initializer type differs from its state type"
                )
            aliases[state.body_argument] = init
            continue
        if isinstance(init, bool) or isinstance(init, (int, float)):
            carry_prefix.append(
                IntentOp(
                    op="const",
                    result=state.body_argument,
                    attrs={
                        "value": init,
                        "dtype": state.type.dtype,
                        "shape": list(state.type.shape),
                    },
                    type=state.type.to_json(),
                )
            )
            continue
        return None

    def resolve(name: str) -> str:
        visited: set[str] = set()
        while name in aliases:
            if name in visited:
                raise ValueError("logical loop alias cycle")
            visited.add(name)
            name = aliases[name]
        return name

    def substitute_body(body: Sequence[IntentOp]) -> list[IntentOp]:
        result: list[IntentOp] = []
        for original in body:
            operation = copy.deepcopy(original)
            operation.operands = [resolve(name) for name in operation.operands]
            if operation.predicate is not None:
                operation.predicate = resolve(operation.predicate)
            operation.body = substitute_body(operation.body)
            for arm in (operation.positive, operation.negative):
                if arm is not None:
                    arm.body = substitute_body(arm.body)
                    if hasattr(arm, "yield_value"):
                        arm.yield_value = resolve(arm.yield_value)
            result.append(operation)
        return result

    body = [
        *substitute_body(outer_prefix),
        *carry_prefix,
        *substitute_body(loop.body),
    ]
    result_to_next = {
        item.result_value: resolve(item.next_value) for item in loop.carry
    }
    if not isinstance(loop.result, list) or set(loop.result) != set(result_to_next):
        raise ValueError("logical loop results differ from its carry bindings")
    # Only results exported by the current lexical sequence are visible after
    # materialization.  A recursive walk would incorrectly promote a result
    # owned by an effect_if/value_if arm to the loop/function scope.
    produced = set(inputs)
    produced.update(
        name
        for operation in body
        for name in loop_result_names(operation)
    )
    missing_next = sorted(set(result_to_next.values()) - produced)
    if missing_next:
        raise ValueError(
            "logical loop body does not define next carry value(s): "
            + ", ".join(missing_next)
        )
    aliases.update(result_to_next)
    body.extend(substitute_body(outer_suffix))
    produced = set(inputs)
    produced.update(
        name
        for operation in body
        for name in loop_result_names(operation)
    )
    returns: list[IntentReturn] = []
    for binding in j_elaboration(j).output_bindings:
        value = resolve(binding.value)
        if value not in produced:
            raise ValueError(
                "logical loop output binding does not name a materialized value"
            )
        returns.append(IntentReturn(binding.output, value))
    function = IntentFunction(
        name=j.intent.name,
        inputs=list(j.intent.inputs),
        outputs=list(j.intent.outputs),
        body=body,
        returns=returns,
    )
    validate_intent_function_wiring(function)
    return function


def materialize_j_function(
    j: SemanticArchive,
    dims: Mapping[str, int],
    *,
    loop_schedule_tile: int | None = None,
) -> JFunctionMaterialization:
    """Choose a logical function form without an implicit hardware policy.

    With no target Layer-C binding the materializer uses the complete semantic
    axis.  A bounded scan tile is an implementation/resource choice and must be
    supplied explicitly by a validated target plugin; it is never inferred
    from a process-wide temporary-memory budget.
    """

    j = elaborate_j(j)
    if any(op.op == "loop" for op in j.intent.body):
        loops = [op for op in j.intent.body if op.op == "loop"]
        if len(loops) != 1:
            raise NotImplementedError(
                "canonical materializer requires exactly one top-level loop"
            )
        if loops[0].trip is not None and loops[0].trip.is_dynamic:
            if loop_schedule_tile is not None:
                raise ValueError(
                    "dynamic value loops do not accept a semantic scan tile"
                )
            # ``intent.loop`` still has an explicit init operand even when the
            # trip count is supplied at runtime.  The J wire permits a literal
            # carry initializer, but the MLIR emitter can only reference SSA
            # values.  Materialize those literals in the dominating prefix of
            # the ephemeral function view; otherwise a valid dynamic loop
            # reaches ``emit_module`` with ``init_value == 0`` (or another
            # literal) and fails late with an unavailable SSA binding.  This
            # is a J-only ABI adaptation: it does not alter canonical J or
            # invent a target schedule.
            function = IntentFunction(
                name=j.intent.name,
                inputs=list(j.intent.inputs),
                outputs=list(j.intent.outputs),
                body=copy.deepcopy(j.intent.body),
                returns=[
                    IntentReturn(binding.output, binding.value)
                    for binding in j_elaboration(j).output_bindings
                ],
            )
            loop_index = next(
                index for index, operation in enumerate(function.body)
                if operation.op == "loop"
            )
            loop = function.body[loop_index]
            prefix = function.body[:loop_index]
            dominating_names = {
                tensor.name for tensor in function.inputs
            }
            dominating_names.update(
                result
                for operation in _walk(prefix)
                for result in (
                    operation.result
                    if isinstance(operation.result, list)
                    else [operation.result]
                )
                if isinstance(result, str)
            )
            occupied = _value_handle_names(function)

            def fresh_initializer(body_argument: str) -> str:
                del body_argument
                candidate = _fresh_ssa_handle(occupied)
                dominating_names.add(candidate)
                return candidate

            initializers: list[IntentOp] = []
            for state in loop.carry:
                initializer = state.init_value
                if isinstance(initializer, (bool, int, float)):
                    if (
                        state.type is None
                        or state.type.kind != "tensor"
                        or state.type.dtype is None
                    ):
                        raise ValueError(
                            "dynamic value loop literal carry initializer "
                            f"for {state.body_argument!r} requires a typed "
                            "tensor carry state"
                        )
                    name = fresh_initializer(state.body_argument)
                    initializers.append(
                        IntentOp(
                            op="const",
                            result=name,
                            attrs={
                                "value": initializer,
                                "dtype": state.type.dtype,
                                "shape": list(state.type.shape),
                            },
                            type=state.type.to_json(),
                        )
                    )
                    state.init_value = name
                    continue
                if initializer is None:
                    raise ValueError(
                        "dynamic value loop requires an explicit carry "
                        f"initializer for {state.body_argument!r}"
                    )
                if not isinstance(initializer, str):
                    raise ValueError(
                        "dynamic value loop carry initializer must be a "
                        "dominating value or scalar literal"
                    )
                if initializer not in dominating_names:
                    raise ValueError(
                        "dynamic value loop carry initializer "
                        f"{initializer!r} is not a dominating J value"
                    )
            function.body[loop_index:loop_index] = initializers
            validate_intent_function_wiring(function)
            consumed = tuple(
                f"loop:{op.node_id or 'anonymous'}:dynamic_value_loop"
                for op in loops
            )
            return JFunctionMaterialization(
                function,
                DYNAMIC_VALUE_LOOP_MATERIALIZATION,
                consumed,
            )
        extent = dims[loops[0].trip.axis] if loops[0].trip is not None else 0
        tile = loop_schedule_tile if loop_schedule_tile is not None else extent
        if isinstance(tile, bool) or not isinstance(tile, int) or tile <= 0:
            raise ValueError("structured scan tile must be a positive integer")
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            raise ValueError("structured scan requires a positive trip extent")
        if tile > extent:
            raise ValueError("structured scan tile exceeds its semantic trip extent")
        if 0 < tile < extent:
            if extent % tile != 0:
                raise NotImplementedError(
                    "structured scan materialization does not yet support a tail tile"
                )
            function = IntentFunction(
                name=j.intent.name,
                inputs=list(j.intent.inputs),
                outputs=list(j.intent.outputs),
                body=copy.deepcopy(j.intent.body),
                returns=[
                    IntentReturn(binding.output, binding.value)
                    for binding in j_elaboration(j).output_bindings
                ],
            )
            form = STRUCTURED_SCAN_LOOP_MATERIALIZATION
            consumed = tuple(
                f"loop:{op.node_id or 'anonymous'}:structured_scan_tile:{tile}"
                for op in loops
            )
            return JFunctionMaterialization(
                function,
                form,
                consumed,
                loop_schedule_tile=tile,
            )
        function = _logical_full_axis_step_function(j, dims)
        if function is None:
            raise NotImplementedError(
                "canonical materializer does not yet encode this structured loop form"
            )
        form = LOGICAL_FULL_AXIS_STEP_MATERIALIZATION
        consumed = tuple(
            f"loop:{op.node_id or 'anonymous'}:logical_full_axis_step"
            for op in loops
        )
    else:
        function, consumed = _canonical_return_function(j)
        form = FLAT_MATERIALIZATION
    return JFunctionMaterialization(function, form, consumed)


def materialize_j_module(
    j: SemanticArchive,
    dims: Mapping[str, int],
    *,
    fold: int = 0,
    func_name: str | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
) -> JModuleMaterialization:
    """Materialize canonical J without reading C or choosing target schedules."""

    j = elaborate_j(j)
    resolved_dims = resolve_dimension_relations(j, dims)
    value_dimension_equalities = resolve_value_dimension_equalities(
        j, resolved_dims
    )
    logical = materialize_j_function(j, resolved_dims)
    mlir = emit_module(
        logical.function,
        resolved_dims,
        fold=fold,
        func_name=func_name,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
        value_dimension_equalities=value_dimension_equalities,
        loop_schedule_tile=logical.loop_schedule_tile,
    )
    return JModuleMaterialization(
        mlir,
        logical.form,
        logical.consumed_structures,
    )


def emit_j_module(
    j: SemanticArchive,
    dims: Mapping[str, int],
    *,
    fold: int = 0,
    func_name: str | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
) -> str:
    """Emit MLIR only after the unique deterministic J elaboration boundary."""

    return materialize_j_module(
        j,
        dims,
        fold=fold,
        func_name=func_name,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
    ).mlir


__all__ = [
    "FLAT_MATERIALIZATION",
    "DYNAMIC_VALUE_LOOP_MATERIALIZATION",
    "JFunctionMaterialization",
    "JModuleMaterialization",
    "LOGICAL_FULL_AXIS_STEP_MATERIALIZATION",
    "STRUCTURED_SCAN_LOOP_MATERIALIZATION",
    "emit_j_module",
    "materialize_j_function",
    "materialize_j_module",
]
