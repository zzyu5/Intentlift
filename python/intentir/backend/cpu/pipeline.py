"""Canonical J -> scalar MLIR -> LLVM CPU reference execution.

This is the scalar leg of source-anchored Gate2.  After Gate1 closes, it lowers
the accepted ``SemanticArchive`` J through the
backend-neutral Intent-to-Linalg path, compiles one value-independent shared
object, and invokes one ranked-memref entry for all logical outputs.  It
contains no source-runtime, JSON-interpreter, CUDA/ROCm, or production-target
fallback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...frontend.canonical_j import emit_j_module
from ...frontend.emit_mlir import IndexBoundsProjection
from ...ir.canonical_j import elaborate_j
from ...ir.dimension_relation import resolve_dimension_relations
from ...ir.layer_c import layer_c_content_digest
from ...ir.layer_c import (
    ImplementationChoice,
    LayerCInterface,
    TargetIdentity,
    TargetLayerCPlugin,
    TargetLowering,
    TargetPipeline,
    TargetRealizationContract,
)
from ...ir.model import IntentFunction, walk_intent_ops
from ...ir.intent_op_contract import RAGGED_OPS
from ...ir.value_dimension import (
    RuntimeShapeResolutionError,
    resolve_runtime_output_shapes as _resolve_runtime_output_shapes,
    validate_ragged_runtime_invocation,
    validate_value_dimension_instantiation,
)
from .abi import (
    CPU_ENTRY_SYMBOL,
    CPUABIValidationError,
    CPUBackendUnsupportedError,
    mlir_element_type,
    numpy_dtype,
    validate_array,
)
from .compiler import compile_cpu_module
from .runtime import execute_isolated
from ..protocol import (
    BackendTarget,
    TargetBindingContext,
    TargetPluginArtifact,
    bind_target,
    deserialize_target_artifact,
    deserialize_target_plugin,
    serialize_target_artifact,
    serialize_target_plugin,
)


CPU_COMPUTE_SYMBOL = "intentir_cpu_compute_v1"
CPU_REFERENCE_CANDIDATE_KEY = "cpu.reference.scalar"
CPU_REFERENCE_CAPABILITIES = (
    "cpu.native.scalar",
    "cpu.ranked_memref_abi",
)


def cpu_reference_target() -> BackendTarget:
    """Return the explicit target identity for the scalar CPU reference path."""

    return BackendTarget("cpu", "reference", ("native",))


def cpu_layer_c_interface(j: Any) -> LayerCInterface:
    """CPU reference lowering consumes no target-specific E_C value."""

    del j
    return LayerCInterface()


def cpu_reference_candidate_digest() -> str:
    """Digest the repository-owned scalar reference implementation identity."""

    return layer_c_content_digest(
        {
            "backend_kind": "cpu",
            "candidate_key": CPU_REFERENCE_CANDIDATE_KEY,
            "implementation": "native_ranked_memref_scalar",
        }
    )


def _cpu_reference_contract(
    choice: ImplementationChoice,
) -> TargetRealizationContract:
    return TargetRealizationContract(
        pipeline=(TargetPipeline(stage="scalar_reference", depth=1),),
        lowering=TargetLowering(
            backend_kind=choice.backend_kind,
            candidate_key=choice.candidate_key,
            candidate_digest=choice.candidate_digest,
        ),
    )


def build_cpu_reference_target_plugin(
    j: Any,
    *,
    target: BackendTarget | None = None,
) -> TargetLayerCPlugin:
    """Bind the canonical J to the only supported CPU reference candidate."""

    from ...verify.gate1.model import SemanticArchive

    if not isinstance(j, SemanticArchive):
        raise CPUBackendUnsupportedError(
            "CPU target binding requires canonical SemanticArchive J"
        )
    target_value = target or cpu_reference_target()
    if target_value != cpu_reference_target():
        raise CPUBackendUnsupportedError(
            "CPU reference lowering requires the cpu/reference target identity"
        )
    choice = ImplementationChoice(
        backend_kind="cpu",
        candidate_key=CPU_REFERENCE_CANDIDATE_KEY,
        candidate_digest=cpu_reference_candidate_digest(),
    )
    return bind_target(
        TargetBindingContext(
            j=elaborate_j(j),
            target=target_value,
            interface=cpu_layer_c_interface(j),
            capabilities=CPU_REFERENCE_CAPABILITIES,
            e_c=cpu_layer_c_interface(j),
        ),
        bindings=(),
        choice=choice,
        required_capabilities=CPU_REFERENCE_CAPABILITIES,
        contract=_cpu_reference_contract(choice),
    )


def build_cpu_reference_target_artifact(
    j: Any,
    *,
    target: BackendTarget | None = None,
) -> TargetPluginArtifact:
    """Build the complete empty-``E_C`` CPU target handoff."""

    return TargetPluginArtifact(
        e_c=LayerCInterface(),
        plugin=build_cpu_reference_target_plugin(j, target=target),
        e_c_mapping=(),
    )


def _cpu_target_artifact(
    j: Any,
    target_plugin: TargetLayerCPlugin | TargetPluginArtifact | None,
) -> TargetPluginArtifact:
    """Cross the public target-artifact boundary before CPU compilation.

    The scalar CPU implementation has no target-specific extension consumer.
    Consequently a non-empty shared ``E_C`` is rejected explicitly rather than
    being silently dropped or reinterpreted as a CPU value.
    """

    from ...verify.gate1.model import SemanticArchive

    if not isinstance(j, SemanticArchive):
        raise CPUBackendUnsupportedError(
            "CPU target binding requires canonical SemanticArchive J"
        )
    canonical = elaborate_j(j)
    target_value = cpu_reference_target()
    choice = ImplementationChoice(
        backend_kind="cpu",
        candidate_key=CPU_REFERENCE_CANDIDATE_KEY,
        candidate_digest=cpu_reference_candidate_digest(),
    )
    context = TargetBindingContext(
        j=canonical,
        target=target_value,
        interface=cpu_layer_c_interface(canonical),
        capabilities=CPU_REFERENCE_CAPABILITIES,
        e_c=cpu_layer_c_interface(canonical),
    )
    if target_plugin is None:
        return build_cpu_reference_target_artifact(canonical, target=target_value)
    if isinstance(target_plugin, TargetPluginArtifact):
        payload = serialize_target_artifact(target_plugin)
    elif isinstance(target_plugin, TargetLayerCPlugin):
        payload = {
            "e_c": LayerCInterface().to_dict(),
            "plugin": serialize_target_plugin(target_plugin),
            "e_c_mapping": [],
        }
    else:
        raise CPUBackendUnsupportedError(
            "CPU target input must be TargetLayerCPlugin or TargetPluginArtifact"
        )
    try:
        return deserialize_target_artifact(
            context,
            payload,
            expected_e_c=LayerCInterface(),
            expected_choice=choice,
            expected_bindings=(),
            expected_contract=_cpu_reference_contract(choice),
            expected_e_c_mapping=(),
        )
    except (TypeError, ValueError) as exc:
        raise CPUBackendUnsupportedError(
            "CPU target artifact validation failed: " + str(exc)
        ) from exc


def validate_cpu_reference_target_plugin(
    j: Any,
    plugin: TargetLayerCPlugin | TargetPluginArtifact,
    *,
    target: BackendTarget | None = None,
) -> TargetLayerCPlugin:
    """Validate a CPU plugin at the same public boundary used by other backends."""

    from ...verify.gate1.model import SemanticArchive

    if not isinstance(j, SemanticArchive):
        raise CPUBackendUnsupportedError(
            "CPU target validation requires canonical SemanticArchive J"
        )
    target_value = target or cpu_reference_target()
    choice = ImplementationChoice(
        backend_kind="cpu",
        candidate_key=CPU_REFERENCE_CANDIDATE_KEY,
        candidate_digest=cpu_reference_candidate_digest(),
    )
    if target_value != cpu_reference_target():
        raise CPUBackendUnsupportedError(
            "CPU reference lowering requires the cpu/reference target identity"
        )
    return _cpu_target_artifact(j, plugin).plugin

# This is an executor conformance boundary, not a generation rule catalog.  It
# mirrors the operations actually eliminated by ConvertIntentToLinalg plus
# constants emitted directly as arith.constant.  Anything else is rejected
# before an external compiler can accidentally turn "unsupported" into an
# opaque tool failure.
CPU_LOWERABLE_OPS = frozenset(
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
        "select",
        "transpose",
    }
    | RAGGED_OPS
)


@dataclass(frozen=True)
class CPUExecutionResult:
    """All logical outputs and execution metadata for one call."""

    outputs: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


def _walk_ops(body):
    yield from walk_intent_ops(body)


def validate_cpu_capability(fn: IntentFunction) -> None:
    """Fail closed when ``E_J`` has no complete scalar lowering."""

    operations = tuple(_walk_ops(fn.body))
    if any(op.op == "loop" for op in operations):
        raise CPUBackendUnsupportedError(
            "scalar CPU execution does not yet lower structured IntentIR loops "
            "for semantic recurrence"
        )
    unsupported = sorted({op.op for op in operations} - CPU_LOWERABLE_OPS)
    if unsupported:
        if "scatter" in unsupported:
            raise CPUBackendUnsupportedError(
                "scalar CPU execution rejects logical update until duplicate, "
                "conflict, order, and result relations are typed in J"
            )
        raise CPUBackendUnsupportedError(
            "scalar CPU execution has no complete lowering for IntentOp(s): "
            + ", ".join(unsupported)
        )


def _tensor_type(shape: Sequence[int], element_type: str) -> str:
    if shape:
        return f"tensor<{'x'.join(str(extent) for extent in shape)}x{element_type}>"
    return f"tensor<{element_type}>"


def _memref_type(shape: Sequence[int], element_type: str) -> str:
    if shape:
        return f"memref<{'x'.join(str(extent) for extent in shape)}x{element_type}>"
    return f"memref<{element_type}>"


def _checked_dims(dims: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(dims, Mapping) or not all(
        isinstance(name, str)
        and name
        and isinstance(extent, int)
        and not isinstance(extent, bool)
        and extent >= 0
        for name, extent in dims.items()
    ):
        raise CPUABIValidationError(
            "CPU execution dims must map names to non-negative integers"
        )
    return dict(dims)


def _infer_contracts(
    fn: IntentFunction,
    dims: Mapping[str, int],
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None,
) -> tuple[
    list[tuple[int, ...]],
    list[str],
    list[tuple[int, ...]],
    list[str],
]:
    from ...frontend.emit_mlir import _infer_body, _resolve

    extents = dict(dims)
    input_shapes = [tuple(_resolve(tensor.shape, extents)) for tensor in fn.inputs]
    input_element_types = [mlir_element_type(tensor.dtype) for tensor in fn.inputs]

    env = {
        tensor.name: list(shape)
        for tensor, shape in zip(fn.inputs, input_shapes, strict=True)
    }
    env.update(
        {
            tensor.name: _resolve(tensor.shape, extents)
            for tensor in fn.outputs
        }
    )
    ragged_types = {}
    _infer_body(
        fn.body,
        env,
        0,
        extents,
        broadcast_axes,
        ragged_types=ragged_types,
    )
    output_shapes = [tuple(env[tensor.name]) for tensor in fn.outputs]
    output_element_types = [
        mlir_element_type(tensor.dtype) for tensor in fn.outputs
    ]
    return (
        input_shapes,
        input_element_types,
        output_shapes,
        output_element_types,
    )


def resolve_runtime_output_shapes(
    fn: IntentFunction,
    declared_shapes: Sequence[Sequence[object]],
    inputs: Sequence[np.ndarray],
    *,
    dims: Mapping[str, int] | None = None,
) -> list[tuple[int, ...]]:
    """CPU-ABI-compatible wrapper around the backend-neutral J resolver."""

    try:
        return _resolve_runtime_output_shapes(
            fn,
            declared_shapes,
            inputs,
            dims=dims,
        )
    except RuntimeShapeResolutionError as exc:
        raise CPUABIValidationError(str(exc)) from exc


_runtime_ragged_output_shapes = resolve_runtime_output_shapes


def _append_cpu_entry(
    kernel_mlir: str,
    *,
    input_shapes: Sequence[Sequence[int]],
    input_element_types: Sequence[str],
    output_shapes: Sequence[Sequence[int]],
    output_element_types: Sequence[str],
) -> str:
    module = kernel_mlir.rstrip()
    if not module.endswith("}"):
        raise ValueError("CPU compute emitter did not produce a complete module")

    compute_declaration = f"func.func @{CPU_COMPUTE_SYMBOL}("
    if module.count(compute_declaration) != 1:
        raise ValueError("CPU compute module has an ambiguous entry declaration")
    module = module.replace(
        compute_declaration,
        f"func.func private @{CPU_COMPUTE_SYMBOL}(",
        1,
    )

    input_types = [
        _tensor_type(shape, element_type)
        for shape, element_type in zip(
            input_shapes, input_element_types, strict=True
        )
    ]
    output_tensor_types = [
        _tensor_type(shape, element_type)
        for shape, element_type in zip(
            output_shapes, output_element_types, strict=True
        )
    ]
    output_memref_types = [
        _memref_type(shape, element_type)
        for shape, element_type in zip(
            output_shapes, output_element_types, strict=True
        )
    ]

    arguments = [
        *(f"%input{index}: {ty}" for index, ty in enumerate(input_types)),
        *(
            f"%output{index}: {ty}"
            for index, ty in enumerate(output_memref_types)
        ),
    ]
    call_arguments = ", ".join(
        f"%input{index}" for index in range(len(input_types))
    )
    call_argument_types = ", ".join(input_types)
    if len(output_tensor_types) == 1:
        result_binding = "%result"
        result_type = output_tensor_types[0]
    else:
        result_binding = f"%result:{len(output_tensor_types)}"
        result_type = f"({', '.join(output_tensor_types)})"

    lines = [module[:-1].rstrip()]
    lines.append(
        f"  func.func @{CPU_ENTRY_SYMBOL}({', '.join(arguments)}) "
        "attributes {llvm.emit_c_interface} {"
    )
    lines.append(
        f"    {result_binding} = call @{CPU_COMPUTE_SYMBOL}({call_arguments}) "
        f": ({call_argument_types}) -> {result_type}"
    )
    for index, (tensor_type, memref_type) in enumerate(
        zip(output_tensor_types, output_memref_types, strict=True)
    ):
        value = "%result" if len(output_tensor_types) == 1 else f"%result#{index}"
        lines.append(
            "    bufferization.materialize_in_destination "
            f"{value} in writable %output{index} : "
            f"({tensor_type}, {memref_type}) -> ()"
        )
    lines.extend(("    return", "  }", "}"))
    return "\n".join(lines) + "\n"


def emit_cpu_module(
    j: "SemanticArchive",
    dims: Mapping[str, int],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
    target_plugin: TargetLayerCPlugin | TargetPluginArtifact | None = None,
) -> str:
    """Emit one scalar module from the accepted canonical J only."""

    from ...verify.gate1.model import SemanticArchive

    if not isinstance(j, SemanticArchive):
        raise CPUBackendUnsupportedError(
            "CPU production emission requires canonical SemanticArchive J"
        )
    j = elaborate_j(j)
    target_artifact = _cpu_target_artifact(j, target_plugin)
    plugin = target_artifact.plugin
    if plugin.contract is None:
        raise CPUBackendUnsupportedError(
            "CPU reference lowering requires its typed target realization contract"
        )
    fn = j.intent
    if not fn.outputs:
        raise ValueError("CPU execution requires at least one logical output")
    validate_cpu_capability(fn)
    checked_dims = _checked_dims(dims)
    checked_dims = resolve_dimension_relations(j, checked_dims)
    (
        input_shapes,
        input_element_types,
        output_shapes,
        output_element_types,
    ) = _infer_contracts(fn, checked_dims, broadcast_axes)
    kernel = emit_j_module(
        j,
        checked_dims,
        func_name=CPU_COMPUTE_SYMBOL,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
    )
    if plugin.target != TargetIdentity.from_target(cpu_reference_target()):
        raise CPUBackendUnsupportedError("validated CPU target changed during emission")
    return _append_cpu_entry(
        kernel,
        input_shapes=input_shapes,
        input_element_types=input_element_types,
        output_shapes=output_shapes,
        output_element_types=output_element_types,
    )


def execute_on_cpu_outputs(
    j: "SemanticArchive",
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
    target_plugin: TargetLayerCPlugin | TargetPluginArtifact | None = None,
) -> CPUExecutionResult:
    """Compile once and execute all canonical-J outputs with one native call."""

    from ...verify.gate1.model import SemanticArchive

    if not isinstance(j, SemanticArchive):
        raise CPUBackendUnsupportedError(
            "CPU production execution requires canonical SemanticArchive J"
        )
    j = elaborate_j(j)
    target_artifact = _cpu_target_artifact(j, target_plugin)
    plugin = target_artifact.plugin
    fn = j.intent
    if len(inputs) != len(fn.inputs):
        raise CPUABIValidationError(
            "CPU execution input arity mismatch: "
            f"expected {len(fn.inputs)}, got {len(inputs)}"
        )
    if not fn.outputs:
        raise CPUABIValidationError(
            "CPU execution requires at least one logical output"
        )
    validate_cpu_capability(fn)
    checked_dims = _checked_dims(dims)
    checked_dims = resolve_dimension_relations(j, checked_dims)
    validate_value_dimension_instantiation(
        j,
        checked_dims,
        {
            tensor.name: value
            for tensor, value in zip(fn.inputs, inputs, strict=True)
        },
    )
    validate_ragged_runtime_invocation(fn, inputs, dims=checked_dims)
    (
        input_shapes,
        _input_element_types,
        output_shapes,
        _output_element_types,
    ) = _infer_contracts(fn, checked_dims, broadcast_axes)
    output_shapes = resolve_runtime_output_shapes(
        fn,
        output_shapes,
        inputs,
        dims=checked_dims,
    )

    checked_inputs = [
        validate_array(
            value,
            shape=shape,
            dtype=tensor.dtype,
            context=f"CPU input {tensor.name!r}",
        )
        for tensor, shape, value in zip(
            fn.inputs, input_shapes, inputs, strict=True
        )
    ]
    output_storage = [
        np.empty(shape, dtype=numpy_dtype(tensor.dtype), order="C")
        for tensor, shape in zip(fn.outputs, output_shapes, strict=True)
    ]
    for tensor, shape, value in zip(
        fn.outputs, output_shapes, output_storage, strict=True
    ):
        validate_array(
            value,
            shape=shape,
            dtype=tensor.dtype,
            context=f"CPU output {tensor.name!r}",
            writable=True,
        )

    module = emit_cpu_module(
        j,
        checked_dims,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
        target_plugin=target_artifact,
    )
    compiled = compile_cpu_module(module)
    try:
        # Keep the artifact-backed evidence alive before the temporary shared
        # object is removed below.  The read and evidence construction are
        # inside this guard as well, so a failed read cannot leak the compiler's
        # temporary .so.
        shared_object = compiled.shared_object.read_bytes()
        target_realization = plugin.contract.consumption_evidence(
            (
                plugin.contract.record_consumption_from_artifact(
                    "pipeline:0",
                    consumer="cpu.native",
                    phase="mlir_opt_pipeline",
                    artifact_kind="llvm_mlir",
                    artifact=compiled.llvm_mlir,
                ),
                plugin.contract.record_consumption_from_artifact(
                    "lowering",
                    consumer="cpu.native",
                    phase="abi_entry",
                    artifact_kind="shared_object",
                    artifact=shared_object,
                ),
            )
        )
        runtime = execute_isolated(compiled, checked_inputs, output_storage)
    finally:
        compiled.shared_object.unlink(missing_ok=True)
    logical_outputs = {
        tensor.name: value
        for tensor, value in zip(fn.outputs, runtime.outputs, strict=True)
    }
    metadata = {
        **compiled.evidence,
        **runtime.metadata,
        "j_digest": layer_c_content_digest(j.to_json()),
        "compute_symbol": CPU_COMPUTE_SYMBOL,
        "input_count": len(checked_inputs),
        "output_count": len(logical_outputs),
        "module_depends_on_runtime_values": False,
        "target_plugin_digest": plugin.content_digest,
        "target_artifact_digest": target_artifact.content_digest,
        "e_c_interface_digest": target_artifact.e_c_digest,
        "e_c_mapping_digest": target_artifact.e_c_mapping_digest,
        "target": plugin.target.to_dict(),
        "target_realization": target_realization,
    }
    return CPUExecutionResult(logical_outputs, metadata)


def run_on_cpu_outputs(
    j: "SemanticArchive",
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    target_plugin: TargetLayerCPlugin | TargetPluginArtifact | None = None,
) -> dict[str, np.ndarray]:
    """Return canonical-J outputs through the native memref ABI path."""

    return dict(
        execute_on_cpu_outputs(
            j,
            dims,
            inputs,
            broadcast_axes=broadcast_axes,
            target_plugin=target_plugin,
        ).outputs
    )


def run_on_cpu(
    j: "SemanticArchive",
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    broadcast_axes: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    target_plugin: TargetLayerCPlugin | TargetPluginArtifact | None = None,
) -> np.ndarray:
    """Execute a single-output canonical J on the scalar CPU path."""

    from ...verify.gate1.model import SemanticArchive

    if not isinstance(j, SemanticArchive):
        raise CPUBackendUnsupportedError(
            "CPU production execution requires canonical SemanticArchive J"
        )
    if len(j.intent.outputs) != 1:
        raise ValueError(
            "run_on_cpu requires exactly one output; use run_on_cpu_outputs"
        )
    return run_on_cpu_outputs(
        j,
        dims,
        inputs,
        broadcast_axes=broadcast_axes,
        target_plugin=target_plugin,
    )[j.intent.outputs[0].name]


__all__ = [
    "CPU_LOWERABLE_OPS",
    "CPU_COMPUTE_SYMBOL",
    "CPUExecutionResult",
    "CPU_REFERENCE_CANDIDATE_KEY",
    "CPU_REFERENCE_CAPABILITIES",
    "build_cpu_reference_target_plugin",
    "build_cpu_reference_target_artifact",
    "cpu_layer_c_interface",
    "cpu_reference_candidate_digest",
    "cpu_reference_target",
    "emit_cpu_module",
    "execute_on_cpu_outputs",
    "run_on_cpu",
    "run_on_cpu_outputs",
    "validate_cpu_reference_target_plugin",
    "resolve_runtime_output_shapes",
    "validate_cpu_capability",
]
