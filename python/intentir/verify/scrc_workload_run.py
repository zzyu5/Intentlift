"""Real SCRC lifecycle helpers for declarative Triton workload adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

import numpy as np

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.value_dimension import resolve_runtime_output_shapes
from intentir.verify.candidate_fanout import CandidateFanoutSearchReport
from intentir.verify.gate1.model import (
    Frontend,
    Gate1Artifact,
    ImplementationConfig,
    ParameterKind,
    ParameterMode,
    ScheduleSlotKind,
    SemanticArchive,
)
from intentir.extraction.providers.client import ProviderError
from intentir.extraction.lifters.miniswe import MiniSWELiftAdapter
from intentir.extraction.lifters.codex_cli import CodexCLILiftAdapter
from intentir.extraction.lifters.scrc import (
    CONSISTENCY_ONLY_AUTHORING_PROFILE,
    ResponsesSCRCLiftAdapter,
)
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.adapters import (
    BoundGate1Adapter,
    NativeGate2ExecutionAdapter,
    TritonGate2ExecutionAdapter,
)
from intentir.verify.frontend_workloads import (
    PreparedFrontendDatasetWorkload,
    build_frontend_dataset_gate2_suite,
)
from intentir.verify.gate1.cuda_compiler import (
    CudaCompileCase,
    CudaCompiler,
    CudaKernelSource,
)
from intentir.verify.gate1.compiler import TritonCompiler
from intentir.verify.gate1.native_orchestrate import (
    CudaGate1Runner,
    TileLangGate1Runner,
)
from intentir.verify.gate1.orchestrate import (
    Gate1Config,
    Gate1ExecutionResult,
    Gate1Runner,
    SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX,
    canonical_artifact_digest,
)
from intentir.verify.gate1.replay import (
    MiniSWEReplayWriter,
    RecordedReplayWriter,
    ReplayAttempt,
    ReplayGeneration,
    ReplayIngressResult,
    ResponsesReplayWriter,
)
from intentir.verify.gate1.codex_replay import CodexCLIReplayWriter
from intentir.verify.gate1.tilelang_compiler import (
    TileLangCompileCase,
    TileLangCompiler,
    TileLangKernelSource,
)
from intentir.verify.gate2.model import (
    Gate2Status,
    Gate2SuiteResult,
    program_identity_digests,
)
from intentir.verify.gate2.compare import compare_outputs
from intentir.verify.gate2.orchestrate import (
    Gate2Case,
    Gate2Runner,
    Gate2Suite,
)
from intentir.verify.gate2.native import CudaRunner, TileLangRunner
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.inputs import materialize_ragged_input_values
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec
from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
    TritonRunner,
)
from intentir.verify.scrc_batch import BatchClassification
from intentir.verify.scrc_runner import (
    CallBudget,
    CandidateAttemptReport,
    CandidateLiftError,
    FeedbackKind,
    LiftFeedback,
    MainStatus,
    MAX_LIFT_CALLS,
    SCRCGateRunner,
    SCRCGate1SearchReport,
    SCRCProgressCallback,
    SCRCRunDependencies,
    SCRCRunReport,
)
from intentir.verify.typecheck import TypeCheckReport, type_check
from intentir.verify.workload_adapters import (
    BufferBinding,
    BufferInitializer,
    BufferRole,
    GeneratedTargetSourceManifest,
    GeneratedSourceLifecycleRecord,
    NoneBinding,
    PreparedTritonWorkload,
)


RecordedLiftAdapter = (
    ResponsesSCRCLiftAdapter | MiniSWELiftAdapter | CodexCLILiftAdapter
)
PreparedWorkload = PreparedTritonWorkload | PreparedFrontendDatasetWorkload
LIFT_SEMANTIC_PROBE_PROTOCOL = "intentir.lift.public-semantic-probe.v1"
LIFT_SEMANTIC_PROBE_SEED = 2026073001
LIFT_SEMANTIC_PROBE_MAX_BUFFER_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RecordedReplayWriterGroup:
    """One formal Gate1 round's independent source-blind Replay writers."""

    writers: tuple[RecordedReplayWriter, ...]

    def __post_init__(self) -> None:
        if len(self.writers) < 2:
            raise ValueError("Replay writer group requires fan-out width >= 2")
        if len({id(writer) for writer in self.writers}) != len(self.writers):
            raise ValueError("Replay writer group members must be independent")

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(
            record
            for writer in self.writers
            for record in writer.records
        )


RecordedReplayEvidenceWriter = RecordedReplayWriter | RecordedReplayWriterGroup


@dataclass(frozen=True)
class RecordedReplayRound:
    """Normal and masked Replay evidence for one formal Gate1 round.

    Native frontends require a writer independent from every normal candidate so
    the masked-J arm cannot share provider/session state with normal Replay.  The
    round wrapper is host-side bookkeeping only; it does not add a persisted
    proof field or alter Gate1's public schema.
    """

    normal: RecordedReplayEvidenceWriter
    masked: RecordedReplayWriter | None = None


RecordedReplayRoundEvidence = RecordedReplayEvidenceWriter | RecordedReplayRound


def _new_replay_writer(
    replay_backend: str,
    *,
    provider: ProviderConfig,
    effort: str | None,
    frontend: Frontend | str = Frontend.TRITON,
    compile_checker: Callable[[str, Gate1Artifact, str], Any] | None = None,
    stop_requested: Callable[[], int | None] | None = None,
) -> RecordedReplayWriter:
    """Create one writer for exactly one formal Gate1 round."""

    selected_frontend = Frontend(frontend)

    if replay_backend == "responses":
        return ResponsesReplayWriter(
            provider=provider,
            effort=effort,
            frontend=selected_frontend,
            compile_checker=compile_checker,
        )
    writer_kwargs: dict[str, Any] = {
        "provider": provider,
        "effort": effort,
        "frontend": selected_frontend,
    }
    if compile_checker is not None and replay_backend in {"miniswe", "codex"}:
        writer_kwargs["compile_checker"] = compile_checker
    if replay_backend == "codex" and (
        effort or provider.reasoning_effort
    ).casefold() in {"xhigh", "max"}:
        writer_kwargs["repair_effort"] = "high"
    if stop_requested is not None:
        writer_kwargs["stop_requested"] = stop_requested
    if replay_backend == "miniswe":
        return MiniSWEReplayWriter(**writer_kwargs)
    if replay_backend == "codex":
        return CodexCLIReplayWriter(**writer_kwargs)
    raise ValueError(f"unknown Replay backend: {replay_backend!r}")


def _prepared_frontend(prepared: PreparedWorkload) -> Frontend:
    """Return the one frontend selected by the prepared request.

    The request is the lifecycle dispatch boundary.  Checking the prepared
    record as well keeps a stale/mixed Dataset projection from silently using a
    compiler for a different source representation.
    """

    try:
        frontend = Frontend(prepared.request.frontend)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"prepared workload has no active frontend: {prepared.request.frontend!r}"
        ) from exc
    entry_frontend = str(prepared.entry.frontend)
    if entry_frontend != frontend.value:
        raise ValueError(
            "prepared workload request and entry frontends differ: "
            f"{frontend.value!r} != {entry_frontend!r}"
        )
    if frontend is Frontend.TRITON and not isinstance(
        prepared, PreparedTritonWorkload
    ):
        raise TypeError("Triton requests require a PreparedTritonWorkload")
    if frontend in {Frontend.CUDA, Frontend.TILELANG} and not isinstance(
        prepared, PreparedFrontendDatasetWorkload
    ):
        raise TypeError(
            f"{frontend.value} requests require a PreparedFrontendDatasetWorkload"
        )
    if frontend is Frontend.CUTE:
        raise ValueError(
            "CuTe has direct Gate1 capability only; no registered Dataset "
            "workload preparation or three-way Gate2 path"
        )
    return frontend


def _native_replay_compile_checker(
    frontend: Frontend,
    compiler: CudaCompiler | TileLangCompiler,
    case: CudaCompileCase | TileLangCompileCase,
) -> Callable[[str, Gate1Artifact, str], ReplayIngressResult]:
    """Build a compile-only checker for native Replay construction probes.

    Replay agents receive only the pass/fail result through their host tool.  A
    native compiler still needs the frozen Harness ABI/launch case, so that
    binding is captured here and is never copied into the source-blind packet.
    Formal Gate1 recompiles and observes the exact bytes independently.
    """

    if frontend not in {Frontend.CUDA, Frontend.TILELANG}:
        raise ValueError("native Replay compile checker requires CUDA or TileLang")

    def check(
        code: str,
        artifact: Gate1Artifact,
        function_name: str,
    ) -> ReplayIngressResult:
        del artifact  # The ingress packet already validated the artifact binding.
        try:
            if frontend is Frontend.CUDA:
                if not isinstance(compiler, CudaCompiler) or not isinstance(
                    case, CudaCompileCase
                ):
                    raise TypeError("CUDA Replay checker received a mismatched case")
                source = CudaKernelSource.from_replay(code, function_name)
            else:
                if not isinstance(compiler, TileLangCompiler) or not isinstance(
                    case, TileLangCompileCase
                ):
                    raise TypeError(
                        "TileLang Replay checker received a mismatched case"
                    )
                source = TileLangKernelSource.from_replay(code, function_name)
            result = compiler.compile(source, case)
        except Exception as exc:
            return ReplayIngressResult(
                "fail",
                diagnostics=(
                    {
                        "rule_code": "native_replay_compile_unavailable",
                        "path": "<native-frontend-compile>",
                        "anonymous_handles": [],
                        "line": None,
                        "detail": f"{type(exc).__name__}: {str(exc)[:2048]}",
                    },
                ),
            )
        if not result.ok:
            return ReplayIngressResult(
                "fail",
                diagnostics=(
                    {
                        "rule_code": "native_replay_compile_failed",
                        "path": "<native-frontend-compile>",
                        "anonymous_handles": [],
                        "line": None,
                        "detail": str(result.diagnostic)[:2048]
                        or "native compiler rejected Replay",
                    },
                ),
            )
        return ReplayIngressResult(
            "pass",
            code=code,
            entry_function_name=None,
        )

    return check


def _validate_replay_writer_contract(
    writer: Any,
    *,
    frontend: Frontend,
) -> RecordedReplayWriter:
    """Validate one writer at the shared Replay/frontend ingress boundary.

    A custom factory is an extension point for provider/session ownership, not
    a way to bypass the selected frontend's compiler.  Keeping this check here
    makes every normal and masked writer obey the same source representation
    and native compile-only preflight contract.
    """

    if not callable(getattr(writer, "generate", None)):
        raise TypeError("Replay writer must expose a callable generate method")
    if not hasattr(writer, "records"):
        raise TypeError("Replay writer must expose records for lifecycle evidence")
    try:
        writer_frontend = Frontend(getattr(writer, "frontend"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "Replay writer must declare the selected frontend explicitly"
        ) from exc
    if writer_frontend is not frontend:
        raise ValueError(
            "Replay writer frontend differs from the prepared workload: "
            f"{writer_frontend.value!r} != {frontend.value!r}"
        )
    if frontend in {Frontend.CUDA, Frontend.TILELANG} and not callable(
        getattr(writer, "compile_checker", None)
    ):
        raise TypeError(
            f"{frontend.value} Replay writer must carry its native compile checker"
        )
    return writer


def _workload_compiler(
    frontend: Frontend,
    prepared: PreparedWorkload,
) -> tuple[Any, Any]:
    """Create the compiler and exact Gate1 case for one prepared frontend."""

    case = prepared.request.binding.gate1_case
    if frontend is Frontend.TRITON:
        return TritonCompiler(timeout=240.0), case
    if frontend is Frontend.CUDA:
        if not isinstance(prepared, PreparedFrontendDatasetWorkload) or not isinstance(
            case, CudaCompileCase
        ):
            raise TypeError("CUDA workload does not carry a CudaCompileCase")
        return CudaCompiler(timeout=240.0), case
    if frontend is Frontend.TILELANG:
        if not isinstance(prepared, PreparedFrontendDatasetWorkload) or not isinstance(
            case, TileLangCompileCase
        ):
            raise TypeError("TileLang workload does not carry a TileLangCompileCase")
        return TileLangCompiler(timeout=300.0), case
    raise ValueError(f"no workload compiler is registered for {frontend.value!r}")


def _build_gate1_adapter(
    frontend: Frontend,
    *,
    compiler: Any,
    writer: Any,
    masked_writer: RecordedReplayWriter | None,
    run_masked_j: bool,
) -> BoundGate1Adapter:
    """Bind one frontend-specific Gate1 runner to a fresh Replay writer set."""

    config = Gate1Config(run_masked_j=run_masked_j)
    if frontend is Frontend.TRITON:
        if masked_writer is not None:
            raise ValueError("Triton Gate1 owns masked Replay on its normal writer")
        if not isinstance(compiler, TritonCompiler):
            raise TypeError("Triton workload requires TritonCompiler")
        return BoundGate1Adapter(
            Gate1Runner(compiler=compiler, writer=writer, config=config)
        )
    if frontend is Frontend.CUDA:
        if not isinstance(compiler, CudaCompiler):
            raise TypeError("CUDA workload requires CudaCompiler")
        return BoundGate1Adapter(
            CudaGate1Runner(
                compiler=compiler,
                writer=writer,
                masked_writer=masked_writer,
                config=config,
            )
        )
    if frontend is Frontend.TILELANG:
        if not isinstance(compiler, TileLangCompiler):
            raise TypeError("TileLang workload requires TileLangCompiler")
        return BoundGate1Adapter(
            TileLangGate1Runner(
                compiler=compiler,
                writer=writer,
                masked_writer=masked_writer,
                config=config,
            )
        )
    raise ValueError(f"no Gate1 runner is registered for {frontend.value!r}")


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    aliases = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "index": "int64",
        "u8": "uint8",
        "u16": "uint16",
        "u32": "uint32",
        "uint8": "uint8",
        "uint16": "uint16",
        "uint32": "uint32",
        "ui8": "uint8",
        "ui16": "uint16",
        "ui32": "uint32",
    }
    normalized = aliases.get(dtype.removeprefix("torch."), dtype)
    if normalized == "bf16":
        # NumPy has no portable bfloat16; Gate2's torch leg performs the final
        # dtype conversion from finite float32 Harness inputs.
        normalized = "float32"
    try:
        return np.dtype(normalized)
    except TypeError as exc:
        raise ValueError(f"unsupported Harness input dtype {dtype!r}") from exc


def _input_value(
    rng: np.random.Generator,
    buffer: BufferBinding,
) -> np.ndarray:
    dtype = _numpy_dtype(buffer.dtype)
    physical_shape = (
        (_physical_storage_size(buffer.shape, buffer.strides),)
        if buffer.strides is not None
        else buffer.shape
    )
    if buffer.initializer is BufferInitializer.ZEROS:
        physical = np.zeros(physical_shape, dtype=dtype)
        return _logical_buffer_value(physical, buffer)
    if buffer.initializer is BufferInitializer.STANDARD_NORMAL:
        if dtype.kind not in {"f", "c"}:
            raise ValueError("standard-normal initializer requires a floating dtype")
        physical = rng.standard_normal(size=physical_shape).astype(dtype)
        return _logical_buffer_value(physical, buffer)
    if buffer.initializer is not BufferInitializer.RANDOM:
        raise ValueError("input/inout buffer lacks a supported initializer")
    if dtype.kind == "b":
        physical = rng.integers(0, 2, size=physical_shape).astype(dtype)
        return _logical_buffer_value(physical, buffer)
    if dtype.kind in {"i", "u"}:
        low = 0 if dtype.kind == "u" else -3
        physical = rng.integers(low, 4, size=physical_shape).astype(dtype)
        return _logical_buffer_value(physical, buffer)
    physical = rng.normal(0.0, 0.25, size=physical_shape).astype(dtype)
    return _logical_buffer_value(physical, buffer)


def _physical_storage_size(
    shape: tuple[int, ...],
    strides: tuple[int, ...] | None,
) -> int:
    if strides is None:
        return int(np.prod(shape, dtype=np.int64)) if shape else 1
    return 1 + sum(
        (extent - 1) * stride
        for extent, stride in zip(shape, strides, strict=True)
    )


def _logical_buffer_value(physical: np.ndarray, buffer: BufferBinding) -> np.ndarray:
    if buffer.strides is None:
        return physical.reshape(buffer.shape)
    if not buffer.shape:
        return physical.reshape(())
    offsets = np.zeros(buffer.shape, dtype=np.int64)
    for axis, (extent, stride) in enumerate(
        zip(buffer.shape, buffer.strides, strict=True)
    ):
        values = np.arange(extent, dtype=np.int64)
        values = values.reshape(
            (1,) * axis
            + (extent,)
            + (1,) * (len(buffer.shape) - axis - 1)
        )
        offsets += values * stride
    return physical.reshape(-1)[offsets].copy()


def _bind_shape(
    declared: list[int | str],
    actual: tuple[int, ...],
    dimensions: dict[str, int],
    *,
    tensor: str,
    dynamic_output: bool = False,
) -> None:
    if len(declared) != len(actual):
        raise ValueError(f"J rank for {tensor!r} differs from the Harness buffer")
    for expected, extent in zip(declared, actual, strict=True):
        if expected == "?":
            if not dynamic_output:
                raise ValueError(
                    f"J data-dependent extent for {tensor!r} requires an "
                    "output-only binding; dynamic input/inout buffers are "
                    "outside the current ABI"
                )
            continue
        if isinstance(expected, int):
            if expected != extent:
                raise ValueError(
                    f"J static extent for {tensor!r} differs from Harness"
                )
            continue
        previous = dimensions.setdefault(expected, extent)
        if previous != extent:
            raise ValueError(f"J symbolic extent {expected!r} is inconsistent")


def _bind_logical_scalar_input(
    *,
    parameter_name: str,
    logical_ref: str | None,
    parameter_dtype: str | None,
    value: bool | int | float | str,
    j_inputs: Mapping[str, Any],
    j_outputs: Mapping[str, Any],
    values: dict[str, Any],
    input_bindings: dict[str, str],
    dimensions: dict[str, int],
) -> None:
    if logical_ref is None:
        return
    if logical_ref in j_inputs:
        tensor = j_inputs[logical_ref]
        if tensor.shape:
            raise ValueError("scalar parameter must bind a rank-0 J input")
        if parameter_dtype is not None and tensor.dtype != parameter_dtype:
            raise ValueError("scalar parameter dtype differs from its J input")
        key = f"logical_{parameter_name}"
        values[key] = np.asarray(value, dtype=_numpy_dtype(tensor.dtype))
        if input_bindings.setdefault(logical_ref, key) != key:
            raise ValueError("multiple Harness scalars bind one J input")
        return
    if logical_ref in j_outputs:
        raise ValueError("value scalar cannot bind a J output tensor")
    if not isinstance(value, (bool, int)):
        raise ValueError("dimension-bound scalar must be integral")
    previous = dimensions.setdefault(logical_ref, int(value))
    if previous != int(value):
        raise ValueError("scalar dimension is inconsistent")


def _build_workload_execution_case(
    prepared: PreparedTritonWorkload,
    j: SemanticArchive,
    c: ImplementationConfig,
    *,
    seed: int,
    case_id: str,
    test_mode: str,
) -> Gate2Case:
    """Bind one explicit execution valuation without assigning it gate authority."""

    adapter = prepared.adapter
    mapping = dict(prepared.source_to_opaque)
    buffers = {mapping[item.formal]: item for item in adapter.buffers}
    scalars = {mapping[item.formal]: item for item in adapter.scalars}
    none_arguments = {
        mapping[item.formal]: item for item in adapter.none_arguments
    }
    semantics = {
        mapping[name]: value for name, value in adapter.semantic_constexprs
    }
    j_inputs = {item.name: item for item in j.intent.inputs}
    j_outputs = {item.name: item for item in j.intent.outputs}
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {}
    dimensions: dict[str, int] = {}
    input_bindings: dict[str, str] = {}
    output_bindings: dict[str, str] = {}
    observed_output_shapes: dict[str, tuple[int, ...]] = {}
    arguments: list[TritonArgument] = []

    for parameter in j.signature.parameters:
        if parameter.kind is ParameterKind.CONSTEXPR:
            if parameter.name not in semantics:
                raise ValueError(
                    f"J constexpr {parameter.name!r} lacks a Harness binding"
                )
            value = semantics[parameter.name]
            _bind_logical_scalar_input(
                parameter_name=parameter.name,
                logical_ref=parameter.logical_ref,
                parameter_dtype=parameter.dtype,
                value=value,
                j_inputs=j_inputs,
                j_outputs=j_outputs,
                values=values,
                input_bindings=input_bindings,
                dimensions=dimensions,
            )
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.SCALAR,
                    value=value,
                )
            )
            continue
        if parameter.kind is ParameterKind.SCALAR:
            binding = scalars.get(parameter.name)
            if binding is None:
                raise ValueError(
                    f"J scalar {parameter.name!r} lacks a Harness binding"
                )
            value_key = f"scalar_{parameter.name}"
            values[value_key] = binding.value
            _bind_logical_scalar_input(
                parameter_name=parameter.name,
                logical_ref=parameter.logical_ref,
                parameter_dtype=parameter.dtype,
                value=binding.value,
                j_inputs=j_inputs,
                j_outputs=j_outputs,
                values=values,
                input_bindings=input_bindings,
                dimensions=dimensions,
            )
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.SCALAR,
                    value_key=value_key,
                )
            )
            continue
        if parameter.kind is ParameterKind.OPAQUE:
            binding = none_arguments.get(parameter.name)
            if binding is None:
                raise ValueError(
                    f"J opaque {parameter.name!r} lacks a Harness None binding"
                )
            if parameter.dtype != binding.dtype:
                raise ValueError(
                    "J opaque dtype differs from the Harness None binding"
                )
            arguments.append(
                TritonArgument(parameter.name, ArgumentRole.NONE)
            )
            continue
        if parameter.kind not in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            raise ValueError(
                f"unsupported real-workload J parameter kind {parameter.kind.value}"
            )
        buffer = buffers.get(parameter.name)
        if buffer is None:
            raise ValueError(
                f"J pointer {parameter.name!r} lacks a Harness buffer binding"
            )
        logical_ref = parameter.logical_ref
        if logical_ref is None:
            raise ValueError("real-workload pointer must anchor a J tensor")
        tensor = j_inputs.get(logical_ref) or j_outputs.get(logical_ref)
        if tensor is None:
            raise ValueError("J pointer logical_ref is not an input/output tensor")
        if tensor.dtype != buffer.dtype:
            raise ValueError("J tensor dtype differs from the Harness buffer")
        _bind_shape(
            tensor.shape,
            buffer.shape,
            dimensions,
            tensor=logical_ref,
            dynamic_output=buffer.role is BufferRole.OUTPUT,
        )
        if buffer.role is BufferRole.INPUT:
            if logical_ref not in j_inputs or parameter.mode is ParameterMode.OUTPUT:
                raise ValueError("Harness input buffer is not a J input")
            key = f"input_{parameter.name}"
            values[key] = _input_value(rng, buffer)
            if input_bindings.setdefault(logical_ref, key) != key:
                raise ValueError("multiple Harness inputs bind one J tensor")
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.INPUT,
                    value_key=key,
                    dtype=buffer.dtype,
                    shape=buffer.shape,
                    strides=buffer.strides,
                )
            )
        elif buffer.role is BufferRole.OUTPUT:
            if logical_ref not in j_outputs or parameter.mode is ParameterMode.INPUT:
                raise ValueError("Harness output buffer is not a J output")
            key = buffer.output_key or f"output_{parameter.name}"
            if output_bindings.setdefault(logical_ref, key) != key:
                raise ValueError("multiple Harness outputs bind one J tensor")
            observed_output_shapes[logical_ref] = tuple(buffer.shape)
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.OUTPUT,
                    dtype=buffer.dtype,
                    shape=buffer.shape,
                    output_key=key,
                    strides=buffer.strides,
                )
            )
        else:
            if logical_ref not in j_inputs or logical_ref not in j_outputs:
                raise ValueError("Harness inout buffer must anchor J input and output")
            input_key = f"input_{parameter.name}"
            output_key = buffer.output_key or f"output_{parameter.name}"
            values[input_key] = _input_value(rng, buffer)
            input_bindings[logical_ref] = input_key
            output_bindings[logical_ref] = output_key
            observed_output_shapes[logical_ref] = tuple(buffer.shape)
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.INOUT,
                    value_key=input_key,
                    dtype=buffer.dtype,
                    output_key=output_key,
                    shape=buffer.shape,
                    strides=buffer.strides,
                )
            )

    if set(input_bindings) != set(j_inputs):
        raise ValueError("Harness input bindings differ from accepted J inputs")
    if set(output_bindings) != set(j_outputs):
        raise ValueError("Harness output bindings differ from accepted J outputs")
    values = materialize_ragged_input_values(
        j,
        values,
        input_bindings,
    )
    runtime_inputs = [
        np.asarray(values[input_bindings[item.name]])
        for item in j.intent.inputs
    ]
    resolved_output_shapes = resolve_runtime_output_shapes(
        j.intent,
        [item.shape for item in j.intent.outputs],
        runtime_inputs,
        dims=dimensions,
    )
    for output, resolved_shape in zip(
        j.intent.outputs, resolved_output_shapes, strict=True
    ):
        observed_shape = observed_output_shapes.get(output.name)
        if observed_shape is None:
            raise ValueError(
                f"Harness did not provide a concrete output shape for {output.name!r}"
            )
        if observed_shape != resolved_shape:
            raise ValueError(
                f"Harness output shape for {output.name!r} differs from the "
                f"J-derived runtime shape: {observed_shape} != {resolved_shape}"
            )
    refeval = ScalarLoweringSpec.create(
        j,
        dims=dimensions,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    schedule: dict[str, Any] = {}
    for slot in c.slots:
        launch_name = (
            slot.role
            if slot.kind is ScheduleSlotKind.COMPILER
            else slot.name
        )
        if launch_name in schedule:
            raise ValueError("C slots collide at one Triton launch option")
        schedule[launch_name] = c.values[slot.name]
    source_program_sha256, source_execution_program_sha256 = program_identity_digests(
        prepared.request.source.gate_source,
        frontend="triton",
    )
    execution_lineage = prepared.manifest.execution_lineage()
    execution_lineage.update(
        {
            "source_program_sha256": source_program_sha256,
            "source_execution_program_sha256": source_execution_program_sha256,
        }
    )
    execution = TritonExecutionSpec(
        tuple(arguments),
        adapter.grid,
        schedule,
        metadata=execution_lineage,
    )
    output_keys = tuple(
        output_bindings[item.name] for item in j.intent.outputs
    )
    case = Gate2Case(
        case_id,
        refeval.j,
        values,
        execution,
        refeval,
        output_keys,
        {
            "test_mode": test_mode,
            "adapter_protocol": adapter.protocol,
            "input_profile": adapter.input_profile,
            "frontend": "triton",
            **execution_lineage,
        },
    )
    return case


def build_workload_jc_suite(
    prepared: PreparedTritonWorkload,
    j: SemanticArchive,
    c: ImplementationConfig,
) -> Gate2Suite:
    """Bind one accepted J/C pair to the workload's frozen hidden valuation."""

    if not isinstance(j, SemanticArchive):
        raise TypeError("workload J must be a SemanticArchive")
    if not isinstance(c, ImplementationConfig):
        raise TypeError("workload C must be an ImplementationConfig")
    prepared = prepared.with_contract(j, c)
    case = _build_workload_execution_case(
        prepared,
        j,
        c,
        seed=prepared.adapter.seed,
        case_id="finite-hidden-1",
        test_mode="finite_hidden_after_gate1",
    )
    return Gate2Suite(
        prepared.adapter.case_id,
        (case,),
        prepared.request.binding.c_values,
    )


def build_workload_gate2_suite(
    prepared: PreparedTritonWorkload,
    artifact: Gate1Artifact,
) -> Gate2Suite:
    """Generate the hidden finite case only after Gate1 accepts the Replay."""

    return build_workload_jc_suite(prepared, artifact.j, artifact.c)


def run_workload_lift_semantic_probe(
    prepared: PreparedWorkload,
    artifact: Gate1Artifact,
) -> dict[str, Any]:
    """Run a public, non-certifying Source-vs-J probe for the Lift owner.

    The probe uses a fixed public seed distinct from the adapter's hidden Gate2
    seed.  Source executes with the candidate C, while the reference leg receives
    J and logical input bindings only.  No Replay, pi relation, Gate observation,
    or hidden input enters this diagnostic.
    """

    frontend = _prepared_frontend(prepared)
    if artifact.c.ordered_value_items() != prepared.request.binding.c_values:
        return {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "unavailable",
            "diagnostic": "candidate C values differ from the frozen Harness binding",
            "authority": "public diagnostic only; not CandidateCheck or certification",
        }

    if frontend is not Frontend.TRITON:
        try:
            suite = build_frontend_dataset_gate2_suite(
                prepared,  # type: ignore[arg-type]
                artifact,
                seed=LIFT_SEMANTIC_PROBE_SEED,
                case_id="finite-public-lift-probe-1",
                test_mode="finite_public_before_gate1",
            )
            case = suite.cases[0]

            def clone_values() -> dict[str, Any]:
                return {
                    name: value.copy() if isinstance(value, np.ndarray) else value
                    for name, value in case.values.items()
                }

            if frontend is Frontend.CUDA:
                source_runner = CudaRunner()
            else:
                source_runner = TileLangRunner()
            source = source_runner.run(
                "source",
                prepared.request.source.gate_source,
                case.triton,
                clone_values(),
            )
            reference = ScalarLoweringRunner().run(
                case.refeval, clone_values()
            )
            if not source.ok or not reference.ok:
                failed = source if not source.ok else reference
                return {
                    "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
                    "status": "unavailable",
                    "frontend": frontend.value,
                    "failed_leg": failed.leg,
                    "unsupported": failed.unsupported,
                    "diagnostic": failed.error,
                    "authority": (
                        "public diagnostic only; native execution infrastructure or "
                        "unsupported reference does not reject or certify the candidate"
                    ),
                }
            agreement = compare_outputs(
                "source",
                source.outputs,
                "j",
                reference.outputs,
                j=case.j,
                input_bindings=case.refeval.input_binding_map,
                output_bindings=case.refeval.output_binding_map,
                inputs=case.values,
                policy=DEFAULT_GATE2_POLICY,
            )
            return {
                "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
                "status": "pass" if agreement.passed else "diverged",
                "frontend": frontend.value,
                "case_id": case.case_id,
                "seed": LIFT_SEMANTIC_PROBE_SEED,
                "comparison": agreement.to_json(),
                "authority": (
                    "public native Source-vs-J diagnostic only; no artifact field is "
                    "generated, repaired, accepted, or certified"
                ),
            }
        except Exception as exc:
            return {
                "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
                "status": "unavailable",
                "frontend": frontend.value,
                "failed_leg": "native_source_or_j",
                "diagnostic": f"{type(exc).__name__}: {exc}",
                "authority": (
                    "public diagnostic only; native probe failure does not reject "
                    "or certify the candidate"
                ),
            }

    public_buffer_bytes = sum(
        int(np.prod(binding.shape, dtype=np.int64))
        * _numpy_dtype(binding.dtype).itemsize
        for binding in prepared.adapter.buffers
    )
    if public_buffer_bytes > LIFT_SEMANTIC_PROBE_MAX_BUFFER_BYTES:
        return {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "unavailable",
            "diagnostic": (
                "public Source-vs-J diagnostic skipped because model-level buffers "
                "exceed the bounded Lift feedback budget"
            ),
            "buffer_bytes": public_buffer_bytes,
            "buffer_budget_bytes": LIFT_SEMANTIC_PROBE_MAX_BUFFER_BYTES,
            "formal_gate2_input_unchanged": True,
            "authority": (
                "public diagnostic only; the model-level Harness and hidden Gate2 "
                "remain unchanged and this skip cannot accept or certify a candidate"
            ),
        }
    seed = LIFT_SEMANTIC_PROBE_SEED
    if seed == prepared.adapter.seed:
        seed += 1
    try:
        case = _build_workload_execution_case(
            prepared,
            artifact.j,
            artifact.c,
            seed=seed,
            case_id="finite-public-lift-probe-1",
            test_mode="finite_public_before_gate1",
        )

        def clone_values() -> dict[str, Any]:
            return {
                name: value.copy() if isinstance(value, np.ndarray) else value
                for name, value in case.values.items()
            }

        source = TritonRunner().run(
            "source",
            prepared.request.source.gate_source,
            case.triton,
            clone_values(),
        )
        reference = ScalarLoweringRunner().run(case.refeval, clone_values())
        if not source.ok or not reference.ok:
            failed = source if not source.ok else reference
            return {
                "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
                "status": "unavailable",
                "failed_leg": failed.leg,
                "unsupported": failed.unsupported,
                "diagnostic": failed.error,
                "authority": (
                    "public diagnostic only; unsupported execution does not reject "
                    "or certify the candidate"
                ),
            }
        agreement = compare_outputs(
            "source",
            source.outputs,
            "j",
            reference.outputs,
            j=case.j,
            input_bindings=case.refeval.input_binding_map,
            output_bindings=case.refeval.output_binding_map,
            inputs=case.values,
            policy=DEFAULT_GATE2_POLICY,
        )
        return {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "pass" if agreement.passed else "diverged",
            "case_id": case.case_id,
            "seed": seed,
            "comparison": agreement.to_json(),
            "authority": (
                "public Source-vs-J diagnostic only; no artifact field is generated, "
                "repaired, accepted, or certified"
            ),
        }
    except Exception as exc:
        return {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "unavailable",
            "failed_leg": "j",
            "phase": "j_reference_construction",
            "diagnostic": f"{type(exc).__name__}: {exc}",
            "authority": (
                "public diagnostic only; probe failure does not reject or certify "
                "the candidate"
            ),
        }


def run_prepared_workload_consistency(
    prepared: PreparedWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    max_lift_calls: int = MAX_LIFT_CALLS,
) -> WorkloadConsistencyResult:
    """Run only the real Source-to-J consistency path for one workload.

    This is intentionally a diagnostic-only path for the current development
    objective.  It retains the model-authored artifact boundary and the real
    public Source-vs-J execution probe, while omitting source-blind Replay,
    formal Gate1, masked-J, and hidden Gate2.  A rejected candidate may use the
    existing value-free continuation contract for at most ``max_lift_calls``
    logical calls; an empty initial provider turn remains terminal and is never
    retried without a prior draft or artifact.
    """

    if not isinstance(max_lift_calls, int) or isinstance(max_lift_calls, bool):
        raise TypeError("max_lift_calls must be an integer")
    if not 1 <= max_lift_calls <= MAX_LIFT_CALLS:
        raise ValueError(
            f"max_lift_calls must be in [1, {MAX_LIFT_CALLS}]"
        )

    consistency_started = time.monotonic()
    stage_metrics: dict[str, dict[str, Any]] = {}

    def record_stage(
        name: str,
        started: float,
        status: str,
        **details: Any,
    ) -> None:
        entry = stage_metrics.setdefault(
            name,
            {"elapsed_ms": 0.0, "runs": 0},
        )
        entry["elapsed_ms"] += max(
            0.0,
            (time.monotonic() - started) * 1000.0,
        )
        entry["runs"] += 1
        entry["status"] = status
        entry.update(details)

    def completed_metrics(status: str) -> dict[str, Mapping[str, Any]]:
        result = {
            name: dict(value) for name, value in stage_metrics.items()
        }
        result["consistency_total"] = {
            "elapsed_ms": max(
                0.0,
                (time.monotonic() - consistency_started) * 1000.0,
            ),
            "runs": 1,
            "status": status,
        }
        return result

    frontend = _prepared_frontend(prepared)
    # Consistency-only is the fast semantic path: the model still has to author
    # a complete value-free artifact, but the initial direct Lift uses the
    # provider's no-reasoning mode so it can spend the bounded turn on the
    # Source-to-J artifact itself.  An explicit caller effort remains
    # authoritative for controlled experiments and formal workflows keep their
    # configured reasoning effort.
    resolved_effort = str(effort or "none")
    lift = ResponsesSCRCLiftAdapter(
        provider=provider,
        compiler_view=prepared.compiler_view,
        effort=resolved_effort,
        authoring_profile=CONSISTENCY_ONLY_AUTHORING_PROFILE,
        source_symbol=getattr(
            prepared.source_slice,
            "function",
            prepared.entry.symbol,
        ),
    )

    feedback: LiftFeedback | None = None
    previous_artifact: Gate1Artifact | None = None
    physical_provider_calls = 0
    artifact_digest: str | None = None
    artifact_schema: str | None = None
    typecheck_report: TypeCheckReport | None = None
    semantic_probe: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] = {}
    lift_metadata: Mapping[str, Any] = {}

    def make_result(status: str, lift_calls: int) -> WorkloadConsistencyResult:
        return WorkloadConsistencyResult(
            status=status,
            model=provider.model,
            reasoning_effort=resolved_effort,
            frontend=frontend.value,
            lift_calls=lift_calls,
            physical_provider_calls=physical_provider_calls,
            artifact_digest=artifact_digest,
            artifact_schema=artifact_schema,
            typecheck=typecheck_report,
            semantic_probe=semantic_probe,
            diagnostics=diagnostics,
            lift_metadata=lift_metadata,
            stage_metrics=completed_metrics(status),
        )

    for lift_call in range(1, max_lift_calls + 1):
        lift_started = time.monotonic()
        try:
            lift_result = lift.lift(prepared.request, feedback)
        except CandidateLiftError as exc:
            record_stage(
                "lift",
                lift_started,
                "rejected",
                attempt=lift_call,
            )
            physical_provider_calls += exc.physical_calls
            failure_diagnostics = dict(exc.diagnostics)
            failure_diagnostics.setdefault("failure_class", "lift_rejected")
            failure_diagnostics["logical_lift_call"] = lift_call
            failure_diagnostics["terminal"] = exc.terminal
            diagnostics = failure_diagnostics

            candidate_draft = exc.candidate_draft
            can_continue = (
                not exc.terminal
                and lift_call < max_lift_calls
                and (candidate_draft is not None or previous_artifact is not None)
            )
            if not can_continue:
                return make_result("lift_rejected", lift_call)

            feedback = LiftFeedback(
                FeedbackKind.LIFT,
                failure_diagnostics,
                previous_artifact=(
                    None if candidate_draft is not None else previous_artifact
                ),
                previous_candidate=candidate_draft,
            )
            if candidate_draft is not None:
                previous_artifact = None
            continue

        record_stage(
            "lift",
            lift_started,
            "finished",
            attempt=lift_call,
        )
        physical_provider_calls += lift_result.physical_calls
        artifact = canonicalize_gate1_artifact(lift_result.artifact)
        bind_started = time.monotonic()
        prepared.request.binding.validate_artifact(artifact)
        record_stage(
            "bind_source",
            bind_started,
            "finished",
            attempt=lift_call,
        )
        previous_artifact = artifact
        artifact_digest = canonical_artifact_digest(artifact)
        artifact_schema = artifact.wire_schema
        lift_metadata = {
            key: lift_result.metadata[key]
            for key in (
                "model",
                "response_model",
                "elapsed_ms",
                "lift_timeout_seconds",
                "artifact_schema",
            )
            if key in lift_result.metadata
        }
        worker_stage_metrics = lift_result.metadata.get("stage_metrics")
        if isinstance(worker_stage_metrics, Mapping):
            lift_metadata = {
                **dict(lift_metadata),
                "stage_metrics": dict(worker_stage_metrics),
            }

        typecheck_started = time.monotonic()
        typecheck_report = type_check(
            artifact.j,
            artifact.c,
            artifact.pi,
            require_program_axes=True,
        )
        record_stage(
            "typecheck",
            typecheck_started,
            typecheck_report.status.value,
            attempt=lift_call,
        )
        if not typecheck_report.passed:
            diagnostics = {
                "failure_class": "typecheck_failed",
                "logical_lift_call": lift_call,
                "typecheck": typecheck_report.to_json(),
            }
            if lift_call == max_lift_calls:
                return make_result("typecheck_failed", lift_call)
            feedback = LiftFeedback(
                FeedbackKind.TYPECHECK,
                diagnostics,
                previous_artifact=artifact,
            )
            continue

        semantic_probe_started = time.monotonic()
        semantic_probe = run_workload_lift_semantic_probe(prepared, artifact)
        probe_status = semantic_probe.get("status")
        record_stage(
            "semantic_probe",
            semantic_probe_started,
            str(probe_status or "unavailable"),
            attempt=lift_call,
        )
        if probe_status == "pass":
            return make_result("consistent", lift_call)
        if probe_status == "diverged" and lift_call < max_lift_calls:
            diagnostics = {
                "diagnostic": "public Source-vs-J semantic probe found a divergence",
                "failure_class": "lift_semantic_probe_diverged",
                "logical_lift_call": lift_call,
                "semantic_probe": semantic_probe,
                "authority": (
                    "diagnostic-only public Source-vs-J comparison; it never edits "
                    "or certifies the artifact"
                ),
            }
            feedback = LiftFeedback(
                FeedbackKind.LIFT,
                diagnostics,
                previous_artifact=artifact,
            )
            continue

        if probe_status == "diverged":
            diagnostics = {
                "failure_class": "lift_semantic_probe_diverged",
                "logical_lift_call": lift_call,
            }
            return make_result("semantic_diverged", lift_call)

        diagnostics = {
            "failure_class": "semantic_probe_unavailable",
            "logical_lift_call": lift_call,
        }
        return make_result("semantic_probe_unavailable", lift_call)

    raise RuntimeError("consistency path exhausted without a terminal result")


def classify_scrc_report(report: SCRCRunReport) -> BatchClassification:
    """Project one canonical lifecycle report into the batch taxonomy."""

    if report.main_status is MainStatus.SCRC_CERTIFIED:
        return BatchClassification.SCRC_CERTIFIED
    if report.main_status is MainStatus.GATE2_FAILED:
        statuses = {
            case.status for case in (report.gate2.cases if report.gate2 else ())
        }
        if statuses == {Gate2Status.REFERENCE_UNSUPPORTED}:
            return BatchClassification.REFERENCE_UNSUPPORTED
        return BatchClassification.GATE2_FAILED

    if not report.attempts:
        return BatchClassification.LIFT_RESPONSE_REJECTED
    terminal = report.attempts[-1]
    if terminal.formal_round is None:
        if (
            terminal.feedback is not None
            and terminal.feedback.diagnostics.get("failure_class")
            == "lift_semantic_probe_diverged"
        ):
            return BatchClassification.LIFT_RESPONSE_REJECTED
        if terminal.typecheck is not None:
            return BatchClassification.TYPECHECK_FAILED
        return BatchClassification.LIFT_RESPONSE_REJECTED
    gate1 = terminal.gate1
    if gate1 is None:
        return BatchClassification.NO_CLOSURE
    diagnostic = gate1.normal.diagnostic or ""
    if diagnostic.startswith(
        SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX
    ):
        return BatchClassification.OBSERVER_UNSUPPORTED
    if diagnostic.startswith("harness binding failed"):
        return BatchClassification.HARNESS_INVALID
    if gate1.normal.generation is None:
        return BatchClassification.REPLAY_RESPONSE_REJECTED
    if (
        gate1.normal.compile_result is not None
        and not gate1.normal.compile_result.ok
    ):
        return BatchClassification.REPLAY_COMPILE_FAILED
    return BatchClassification.NO_CLOSURE


def _attempts_diagnostic(attempts: tuple[Any, ...]) -> str:
    if attempts:
        terminal = attempts[-1]
        if terminal.feedback is not None:
            diagnostics = terminal.feedback.diagnostics
            candidate_check = diagnostics.get("candidate_check")
            if isinstance(candidate_check, Mapping):
                issues = candidate_check.get("diagnostics")
                if isinstance(issues, list):
                    rendered: list[str] = []
                    for issue in issues:
                        if not isinstance(issue, Mapping):
                            continue
                        rule_code = issue.get("rule_code") or issue.get(
                            "code", "candidate_check"
                        )
                        path = issue.get("path", "<candidate>")
                        observed = issue.get("observed_refs_or_types")
                        if isinstance(observed, list):
                            detail = " | ".join(str(item) for item in observed)
                        elif observed is None:
                            detail = ""
                        else:
                            detail = str(observed)
                        rendered.append(
                            f"{rule_code}@{path}: {detail}"
                            if detail
                            else f"{rule_code}@{path}"
                        )
                    if rendered:
                        # CandidateCheck already bounds each diagnostic, but
                        # keep the batch summary compact when a provider or a
                        # future checker supplies many independent issues.
                        summary = "; ".join(rendered)
                        if len(summary) > 8192:
                            return summary[:8180] + "...<truncated>"
                        return summary
            if isinstance(diagnostics.get("diagnostic"), str):
                return diagnostics["diagnostic"]
            typecheck = diagnostics.get("typecheck")
            if isinstance(typecheck, Mapping):
                items = typecheck.get("diagnostics")
                if isinstance(items, list):
                    return "; ".join(
                        f"{item.get('code')}@{item.get('path')}: "
                        f"{item.get('detail')}"
                        for item in items
                        if isinstance(item, Mapping)
                    )
        if terminal.gate1 is not None:
            if terminal.gate1.normal.diagnostic:
                return terminal.gate1.normal.diagnostic
            comparison = terminal.gate1.normal.comparison
            if comparison is not None and not comparison.hard_pass:
                failed = [
                    name
                    for name, passed in comparison.hard_passes.items()
                    if not passed
                ]
                return "Gate1 hard mismatch: " + ", ".join(failed)
    return ""


def scrc_report_diagnostic(report: SCRCRunReport) -> str:
    if report.gate2 is not None:
        failures = [
            case
            for case in report.gate2.cases
            if case.status is not Gate2Status.PASS
        ]
        if failures:
            case = failures[0]
            return case.source.error or case.replay.error or case.reference.error
    return _attempts_diagnostic(report.attempts)


def scrc_gate1_search_diagnostic(report: SCRCGate1SearchReport) -> str:
    return _attempts_diagnostic(report.attempts)


def scrc_attempt_stage_counts(
    report: SCRCRunReport | SCRCGate1SearchReport,
) -> dict[str, int]:
    stages: Counter[str] = Counter()
    for attempt in report.attempts:
        if attempt.formal_round is None:
            if (
                attempt.feedback is not None
                and attempt.feedback.diagnostics.get("failure_class")
                == "lift_semantic_probe_diverged"
            ):
                stages["semantic_probe_diverged"] += 1
                continue
            stages[
                "typecheck_failed"
                if attempt.typecheck is not None
                else "lift_response_rejected"
            ] += 1
            continue
        gate1 = attempt.gate1
        if gate1 is None:
            stages["formal_gate1_missing"] += 1
        elif (gate1.normal.diagnostic or "").startswith(
            SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX
        ):
            stages["observer_unsupported"] += 1
        elif (gate1.normal.diagnostic or "").startswith(
            "harness binding failed"
        ):
            stages["harness_invalid"] += 1
        elif gate1.normal.generation is None:
            stages["replay_response_rejected"] += 1
        elif (
            gate1.normal.compile_result is not None
            and not gate1.normal.compile_result.ok
        ):
            stages["replay_compile_failed"] += 1
        elif gate1.passed:
            stages["replay_closed"] += 1
        else:
            stages["gate1_hard_mismatch"] += 1
    return dict(sorted(stages.items()))


@dataclass(frozen=True)
class WorkloadExecution:
    report: SCRCRunReport
    lift: RecordedLiftAdapter
    replay_writers: tuple[RecordedReplayRoundEvidence, ...]
    gate2_attempts: tuple[Gate2SuiteResult, ...] = ()


@dataclass(frozen=True)
class WorkloadGate1Search:
    report: SCRCGate1SearchReport
    runner: SCRCGateRunner
    lift: RecordedLiftAdapter
    replay_writers: tuple[RecordedReplayRoundEvidence, ...]


@dataclass(frozen=True)
class WorkloadConsistencyResult:
    """Diagnostic-only result for the current Source-to-J objective.

    This result deliberately stops before source-blind Replay, formal Gate1,
    and Gate2.  It still requires the real model-authored artifact to pass
    CandidateCheck/BindSource, TypeCheck, and the public Source-vs-J execution
    probe.  A passing result is evidence of semantic agreement only; it is not
    an SCRC closure or a substitute for the complete lifecycle.
    """

    status: str
    model: str
    reasoning_effort: str
    frontend: str
    lift_calls: int
    physical_provider_calls: int
    artifact_digest: str | None = None
    artifact_schema: str | None = None
    typecheck: TypeCheckReport | None = None
    semantic_probe: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    lift_metadata: Mapping[str, Any] = field(default_factory=dict)
    stage_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {
            "consistent",
            "lift_rejected",
            "typecheck_failed",
            "semantic_diverged",
            "semantic_probe_unavailable",
        }
        if self.status not in allowed:
            raise ValueError(
                "unknown workload consistency status: " + repr(self.status)
            )
        if self.lift_calls < 0 or self.physical_provider_calls < 0:
            raise ValueError("consistency call counts must be non-negative")

    @property
    def passed(self) -> bool:
        return self.status == "consistent"

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic_only": True,
            "closure": False,
            "status": self.status,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "frontend": self.frontend,
            "lift_calls": self.lift_calls,
            "physical_provider_calls": self.physical_provider_calls,
            "artifact_digest": self.artifact_digest,
            "artifact_schema": self.artifact_schema,
            "typecheck": (
                self.typecheck.to_json() if self.typecheck is not None else None
            ),
            "semantic_probe": (
                dict(self.semantic_probe)
                if self.semantic_probe is not None
                else None
            ),
            "diagnostics": dict(self.diagnostics),
            "lift_metadata": dict(self.lift_metadata),
            "stage_metrics": {
                str(key): dict(value)
                for key, value in self.stage_metrics.items()
            },
        }


class WorkloadExecutionError(RuntimeError):
    def __init__(
        self,
        cause: BaseException,
        lift: RecordedLiftAdapter,
        replay_writers: tuple[RecordedReplayRoundEvidence, ...],
        *,
        stage_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.cause = cause
        self.lift = lift
        self.replay_writers = replay_writers
        evidence = getattr(cause, "evidence", None)
        self.evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
        inherited_metrics = self.evidence.get("stage_metrics")
        merged_metrics: dict[str, Mapping[str, Any]] = {}
        if isinstance(inherited_metrics, Mapping):
            merged_metrics.update(
                {
                    str(name): value
                    for name, value in inherited_metrics.items()
                    if isinstance(value, Mapping)
                }
            )
        if stage_metrics:
            merged_metrics.update(stage_metrics)
        self.stage_metrics = {
            str(name): dict(value)
            for name, value in merged_metrics.items()
            if isinstance(value, Mapping)
        }
        super().__init__(f"{type(cause).__name__}: {cause}")


def _build_workload_runtime(
    prepared: PreparedWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    replay_provider: ProviderConfig | None = None,
    replay_effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "responses",
    lift_workflow: str = "single",
    replay_backend: str = "responses",
    replay_width: int = 1,
    replay_writer_factory: Callable[[], RecordedReplayWriter] | None = None,
    stop_requested: Callable[[], int | None] | None = None,
) -> tuple[
    SCRCGateRunner,
    RecordedLiftAdapter,
    list[RecordedReplayRoundEvidence],
]:
    if not isinstance(replay_width, int) or isinstance(replay_width, bool):
        raise TypeError("replay_width must be an integer")
    if not 1 <= replay_width <= 4:
        raise ValueError("replay_width must be in [1, 4]")
    if lift_workflow != "single":
        raise ValueError(
            "lift_workflow must be 'single'; staged multi-writer Lift was retired"
        )
    frontend = _prepared_frontend(prepared)
    lift_class: type[RecordedLiftAdapter]
    if lift_backend == "responses":
        lift_class = ResponsesSCRCLiftAdapter
    elif lift_backend == "miniswe":
        lift_class = MiniSWELiftAdapter
    elif lift_backend == "codex":
        lift_class = CodexCLILiftAdapter
    else:
        raise ValueError(f"unknown Lift backend: {lift_backend!r}")
    lift_kwargs: dict[str, Any] = {
        "provider": provider,
        "compiler_view": prepared.compiler_view,
        "effort": effort,
        "source_symbol": getattr(
            prepared.source_slice,
            "function",
            prepared.entry.symbol,
        ),
    }
    if lift_backend in {"miniswe", "codex"}:
        lift_kwargs["workflow"] = lift_workflow
    if lift_backend == "codex":
        lift_kwargs["semantic_probe"] = lambda artifact: (
            run_workload_lift_semantic_probe(prepared, artifact)
        )
    if lift_backend == "codex" and (
        effort or provider.reasoning_effort
    ).casefold() in {"xhigh", "max"}:
        lift_kwargs["repair_effort"] = "high"
        lift_kwargs["stop_requested"] = stop_requested
    lift = lift_class(
        **lift_kwargs,
    )
    effective_replay_provider = replay_provider or provider
    effective_replay_effort = effort if replay_effort is None else replay_effort
    writers: list[RecordedReplayRoundEvidence] = []
    compiler, _case = _workload_compiler(frontend, prepared)
    native_checker = (
        _native_replay_compile_checker(frontend, compiler, _case)
        if frontend in {Frontend.CUDA, Frontend.TILELANG}
        else None
    )

    def new_writer() -> RecordedReplayWriter:
        writer = (
            replay_writer_factory()
            if replay_writer_factory is not None
            else _new_replay_writer(
                replay_backend,
                provider=effective_replay_provider,
                effort=effective_replay_effort,
                frontend=frontend,
                compile_checker=native_checker,
                stop_requested=stop_requested,
            )
        )
        return _validate_replay_writer_contract(writer, frontend=frontend)

    def gate1_factory(_formal_round: int) -> BoundGate1Adapter:
        round_writers = tuple(new_writer() for _ in range(replay_width))
        writer_evidence: RecordedReplayEvidenceWriter
        if replay_width == 1:
            writer_evidence = round_writers[0]
        else:
            writer_evidence = RecordedReplayWriterGroup(round_writers)
        if frontend is Frontend.TRITON:
            writers.append(writer_evidence)
            return _build_gate1_adapter(
                frontend,
                compiler=compiler,
                writer=(
                    round_writers[0]
                    if replay_width == 1
                    else round_writers
                ),
                masked_writer=None,
                run_masked_j=run_masked_j,
            )
        masked_writer = new_writer() if run_masked_j else None
        writers.append(RecordedReplayRound(writer_evidence, masked_writer))
        return _build_gate1_adapter(
            frontend,
            compiler=compiler,
            writer=(
                round_writers[0]
                if replay_width == 1
                else round_writers
            ),
            masked_writer=masked_writer,
            run_masked_j=run_masked_j,
        )

    if frontend is Frontend.TRITON:
        gate2 = TritonGate2ExecutionAdapter(
            Gate2Runner(),
            lambda artifact, _gate1, _binding: build_workload_gate2_suite(
                prepared, artifact  # type: ignore[arg-type]
            ),
        )
    elif frontend is Frontend.CUDA:
        gate2 = NativeGate2ExecutionAdapter(
            "cuda",
            Gate2Runner(
                source_runner=CudaRunner(compiler),
                replay_runner=CudaRunner(compiler),
            ),
            lambda artifact, _gate1, _binding: build_frontend_dataset_gate2_suite(
                prepared,  # type: ignore[arg-type]
                artifact,
            ),
        )
    elif frontend is Frontend.TILELANG:
        gate2 = NativeGate2ExecutionAdapter(
            "tilelang",
            Gate2Runner(
                source_runner=TileLangRunner(),
                replay_runner=TileLangRunner(),
            ),
            lambda artifact, _gate1, _binding: build_frontend_dataset_gate2_suite(
                prepared,  # type: ignore[arg-type]
                artifact,
            ),
        )
    else:  # pragma: no cover - _prepared_frontend closes this set
        raise ValueError(f"no Gate2 adapter is registered for {frontend.value!r}")

    # Responses and MiniSWE return a checked artifact directly, so the shared
    # runner performs the same public Source-vs-J probe before any Replay/Gate1
    # call. Codex owns an equivalent same-thread probe inside its persistent
    # session; do not execute that expensive native/reference probe twice.
    semantic_probe = (
        (
            lambda artifact: run_workload_lift_semantic_probe(prepared, artifact)
        )
        if lift_backend in {"responses", "miniswe"}
        else None
    )
    runner = SCRCGateRunner(
        SCRCRunDependencies(
            lift,
            gate1_factory,
            gate2,
            semantic_probe=semantic_probe,
        )
    )
    return runner, lift, writers


def search_prepared_workload_gate1(
    prepared: PreparedWorkload,
    *,
    provider: ProviderConfig,
    initial_feedback: LiftFeedback | None = None,
    effort: str | None = None,
    replay_provider: ProviderConfig | None = None,
    replay_effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "responses",
    lift_workflow: str = "single",
    replay_backend: str = "responses",
    replay_width: int = 1,
    replay_writer_factory: Callable[[], RecordedReplayWriter] | None = None,
    retry_initial_artifact: bool = False,
    max_gate1_rounds: int | None = None,
    max_lift_calls: int | None = None,
    max_pre_replay_rejections: int | None = None,
    stop_requested: Callable[[], int | None] | None = None,
    progress_callback: SCRCProgressCallback | None = None,
) -> WorkloadGate1Search:
    """Run one independent branch only through TypeCheck and Gate1."""

    runner, lift, writers = _build_workload_runtime(
        prepared,
        provider=provider,
        effort=effort,
        replay_provider=replay_provider,
        replay_effort=replay_effort,
        run_masked_j=run_masked_j,
        lift_backend=lift_backend,
        lift_workflow=lift_workflow,
        replay_backend=replay_backend,
        replay_width=replay_width,
        replay_writer_factory=replay_writer_factory,
        stop_requested=stop_requested,
    )
    request = prepared.request
    if max_gate1_rounds is not None:
        request = replace(request, max_gate1_rounds=max_gate1_rounds)
    if max_lift_calls is not None:
        request = replace(request, max_lift_calls=max_lift_calls)
    if max_pre_replay_rejections is not None:
        request = replace(
            request,
            max_pre_replay_rejections=max_pre_replay_rejections,
        )
    elif lift_backend not in {"miniswe", "codex"}:
        # Legacy one-shot writers do not expose a cumulative owner-time terminal
        # signal, so retain their historical safety override.
        request = replace(request, max_pre_replay_rejections=3)
    search_started = time.monotonic()
    try:
        report = runner.search_gate1(
            request,
            initial_feedback=initial_feedback,
            retry_initial_artifact=retry_initial_artifact,
            stop_requested=stop_requested,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        stage_metrics: dict[str, dict[str, Any]] = {
            "search_total": {
                "elapsed_ms": max(
                    0.0, (time.monotonic() - search_started) * 1000.0
                ),
                "runs": 1,
                "status": "infrastructure_error",
            }
        }
        if isinstance(exc, ProviderError):
            stage_metrics["lift"] = {
                "elapsed_ms": max(0.0, float(exc.elapsed_ms)),
                "runs": 1,
                "status": "infrastructure_error",
                "attempts": exc.attempts,
                "physical_provider_calls": exc.attempts,
                "failure_class": exc.failure_class,
            }
        raise WorkloadExecutionError(
            exc,
            lift,
            tuple(writers),
            stage_metrics=stage_metrics,
        ) from exc
    return WorkloadGate1Search(report, runner, lift, tuple(writers))


def complete_prepared_workload_gate2(
    prepared: PreparedWorkload,
    search: WorkloadGate1Search,
) -> WorkloadExecution:
    """Execute hidden Gate2 once for the preselected Gate1-closed branch."""

    if not search.report.closed:
        raise ValueError("cannot execute Gate2 for a branch without Gate1 closure")
    gate2_started = time.monotonic()
    try:
        report = search.runner.complete_gate2(
            prepared.request, search.report
        )
    except Exception as exc:
        stage_metrics = {
            "gate2": {
                "elapsed_ms": max(
                    0.0, (time.monotonic() - gate2_started) * 1000.0
                ),
                "runs": 1,
                "status": "infrastructure_error",
            },
            "gate2_total": {
                "elapsed_ms": max(
                    0.0, (time.monotonic() - gate2_started) * 1000.0
                ),
                "runs": 1,
                "status": "infrastructure_error",
            },
        }
        raise WorkloadExecutionError(
            exc,
            search.lift,
            search.replay_writers,
            stage_metrics=stage_metrics,
        ) from exc
    return WorkloadExecution(
        report,
        search.lift,
        search.replay_writers,
        (report.gate2,) if report.gate2 is not None else (),
    )


def gate2_exposes_logical_j_pass_implementation_failure(
    gate2: Gate2SuiteResult | None,
) -> bool:
    """Recognize logical-J pass with an implementation-side Gate2 failure.

    The failure may belong to Replay consumption or to an incomplete pi realization;
    Source/J alone cannot distinguish them.  This predicate is host-side lifecycle
    control only.  It never enters a Replay packet or prompt and cannot select
    another J candidate.  Every failed case must have a successful Source/J
    comparison and matching failures on both Replay legs.
    """

    if gate2 is None or gate2.passed:
        return False
    failed = tuple(case for case in gate2.cases if not case.passed)
    if not failed:
        return False
    return all(
        case.source.ok
        and case.replay.ok
        and case.reference.ok
        and case.source_j is not None
        and case.source_j.passed
        and case.source_replay is not None
        and not case.source_replay.passed
        and case.replay_j is not None
        and not case.replay_j.passed
        for case in failed
    )


def _rebase_fixed_artifact_retry_attempt(
    attempt: CandidateAttemptReport,
    *,
    formal_round_offset: int,
) -> CandidateAttemptReport:
    formal_round = (
        formal_round_offset + attempt.formal_round
        if attempt.formal_round is not None
        else None
    )
    feedback = attempt.feedback
    if feedback is not None and feedback.formal_round is not None:
        feedback = replace(
            feedback,
            formal_round=formal_round_offset + feedback.formal_round,
        )
    return replace(
        attempt,
        formal_round=formal_round,
        feedback=feedback,
        lift_metadata={
            **dict(attempt.lift_metadata),
            "fixed_artifact_fresh_replay": True,
            "formal_round_offset": formal_round_offset,
        },
    )


def retry_prepared_workload_gate1_fixed_artifact(
    prepared: PreparedWorkload,
    artifact: Gate1Artifact,
    *,
    provider: ProviderConfig,
    formal_round_offset: int,
    effort: str | None = None,
    replay_provider: ProviderConfig | None = None,
    replay_effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "responses",
    lift_workflow: str = "single",
    replay_backend: str = "responses",
    replay_width: int = 1,
    replay_writer_factory: Callable[[], RecordedReplayWriter] | None = None,
    stop_requested: Callable[[], int | None] | None = None,
    progress_callback: SCRCProgressCallback | None = None,
) -> WorkloadGate1Search:
    """Run exactly one fresh formal Replay/Gate1 round over an exact artifact.

    No Lift model call is permitted.  The synthetic Gate1 feedback contains no
    Source, Gate2, prior Replay, or repair diagnostic; it exists only to activate
    the runner's exact-artifact transaction.  A failed Gate1 round returns normally
    and the caller may start another independent fresh episode within the original K.
    """

    if not isinstance(formal_round_offset, int) or formal_round_offset < 1:
        raise ValueError("formal_round_offset must be a positive integer")
    canonical = canonicalize_gate1_artifact(artifact)
    neutral_feedback = LiftFeedback(
        FeedbackKind.GATE1,
        {
            "status": "fixed_artifact_fresh_replay",
            "authority": (
                "host lifecycle marker only; never supplied to Replay and contains "
                "no Source, Gate2, prior Replay, or semantic repair information"
            ),
        },
        formal_round_offset,
        canonical,
    )

    def offset_progress(
        stage: str,
        attempt: int | None,
        formal_round: int | None,
        status: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            stage,
            attempt,
            (
                formal_round_offset + formal_round
                if formal_round is not None
                else None
            ),
            status,
        )

    search = search_prepared_workload_gate1(
        prepared,
        provider=provider,
        initial_feedback=neutral_feedback,
        effort=effort,
        replay_provider=replay_provider,
        replay_effort=replay_effort,
        run_masked_j=run_masked_j,
        lift_backend=lift_backend,
        lift_workflow=lift_workflow,
        replay_backend=replay_backend,
        replay_width=replay_width,
        replay_writer_factory=replay_writer_factory,
        retry_initial_artifact=True,
        max_gate1_rounds=1,
        # Exact-artifact retry is intentionally one-shot before Replay.  A current
        # TypeCheck rejection must return without ever falling through to Lift.
        max_pre_replay_rejections=1,
        stop_requested=stop_requested,
        progress_callback=offset_progress,
    )
    if search.report.call_budget.lift_calls != 0 or search.lift.records:
        raise RuntimeError("fixed-artifact Replay retry unexpectedly invoked Lift")
    rebased_attempts = tuple(
        _rebase_fixed_artifact_retry_attempt(
            attempt,
            formal_round_offset=formal_round_offset,
        )
        for attempt in search.report.attempts
    )
    rebased_report = replace(
        search.report,
        attempts=rebased_attempts,
        accepted_attempt=(
            formal_round_offset + search.report.accepted_attempt
            if search.report.accepted_attempt is not None
            else None
        ),
    )
    return replace(search, report=rebased_report)


def _add_call_budgets(left: CallBudget, right: CallBudget) -> CallBudget:
    return CallBudget(
        lift_calls=left.lift_calls + right.lift_calls,
        lift_physical_calls=left.lift_physical_calls + right.lift_physical_calls,
        typecheck_calls=left.typecheck_calls + right.typecheck_calls,
        formal_gate1_rounds=(
            left.formal_gate1_rounds + right.formal_gate1_rounds
        ),
        replay_normal_calls=left.replay_normal_calls + right.replay_normal_calls,
        replay_masked_calls=left.replay_masked_calls + right.replay_masked_calls,
        replay_physical_calls=(
            left.replay_physical_calls + right.replay_physical_calls
        ),
        gate2_cases=left.gate2_cases + right.gate2_cases,
        source_executions=left.source_executions + right.source_executions,
        replay_executions=left.replay_executions + right.replay_executions,
        refeval_executions=left.refeval_executions + right.refeval_executions,
    )


def _merge_stage_metrics(
    *groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Accumulate stage timings while retaining the latest stage status.

    A Gate1 search owns the timings for its Lift/BindSource/TypeCheck/Replay
    stages.  Wrapper-level no-closure and fixed-artifact retry paths must carry
    those timings into the persisted ``SCRCRunReport`` instead of rebuilding a
    report with the default empty mapping.  ``elapsed_ms`` and ``runs`` are
    additive across independent searches; other fields describe the latest
    observation for that stage.
    """

    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for name, raw in group.items():
            if not isinstance(raw, Mapping):
                continue
            current = merged.setdefault(str(name), {})
            for key, value in raw.items():
                if key in {"elapsed_ms", "runs"}:
                    previous = current.get(key)
                    if isinstance(previous, (int, float)) and isinstance(
                        value, (int, float)
                    ):
                        current[key] = previous + value
                    elif key not in current:
                        current[key] = value
                else:
                    current[key] = value
    return merged


def _gate2_only_budget(gate2: Gate2SuiteResult) -> CallBudget:
    case_count = len(gate2.cases)
    return CallBudget(
        lift_calls=0,
        lift_physical_calls=0,
        typecheck_calls=0,
        formal_gate1_rounds=0,
        replay_normal_calls=0,
        replay_masked_calls=0,
        replay_physical_calls=0,
        gate2_cases=case_count,
        source_executions=case_count,
        replay_executions=case_count,
        refeval_executions=case_count,
    )


def merge_fixed_artifact_replay_retry_execution(
    base: WorkloadExecution,
    retry_searches: tuple[WorkloadGate1Search, ...],
    retry_executions: tuple[WorkloadExecution, ...],
) -> WorkloadExecution:
    """Merge exact-artifact retry evidence without changing candidate selection."""

    attempts = list(base.report.attempts)
    writers = list(base.replay_writers)
    budget = base.report.call_budget
    for search in retry_searches:
        attempts.extend(search.report.attempts)
        writers.extend(search.replay_writers)
        budget = _add_call_budgets(budget, search.report.call_budget)
    gate2_attempts = list(base.gate2_attempts)
    if not gate2_attempts and base.report.gate2 is not None:
        gate2_attempts.append(base.report.gate2)
    for execution in retry_executions:
        if execution.report.gate2 is None:
            raise ValueError("retry execution lacks Gate2 evidence")
        gate2_attempts.append(execution.report.gate2)
        budget = _add_call_budgets(
            budget,
            _gate2_only_budget(execution.report.gate2),
        )
    stage_metrics = _merge_stage_metrics(
        base.report.stage_metrics,
        *(search.report.stage_metrics for search in retry_searches),
        *(
            {
                key: value
                for key, value in execution.report.stage_metrics.items()
                if key in {"gate2", "gate2_total"}
            }
            for execution in retry_executions
        ),
    )
    final = retry_executions[-1] if retry_executions else base
    report = SCRCRunReport(
        main_status=final.report.main_status,
        attempts=tuple(attempts),
        accepted_attempt=final.report.accepted_attempt,
        gate2=final.report.gate2,
        j_necessity_status=final.report.j_necessity_status,
        call_budget=budget,
        stage_metrics=stage_metrics,
    )
    return WorkloadExecution(
        report,
        base.lift,
        tuple(writers),
        tuple(gate2_attempts),
    )


def execute_prepared_workload(
    prepared: PreparedWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    replay_provider: ProviderConfig | None = None,
    replay_effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "responses",
    lift_workflow: str = "single",
    replay_backend: str = "responses",
    max_lift_calls: int | None = None,
    progress_callback: SCRCProgressCallback | None = None,
) -> WorkloadExecution:
    """Run one independent real case through the canonical full lifecycle."""

    search = search_prepared_workload_gate1(
        prepared,
        provider=provider,
        effort=effort,
        replay_provider=replay_provider,
        replay_effort=replay_effort,
        run_masked_j=run_masked_j,
        lift_backend=lift_backend,
        lift_workflow=lift_workflow,
        replay_backend=replay_backend,
        max_lift_calls=max_lift_calls,
        progress_callback=progress_callback,
    )
    if search.report.closed:
        return complete_prepared_workload_gate2(prepared, search)
    report = SCRCRunReport(
        MainStatus.NO_CLOSURE,
        search.report.attempts,
        None,
        None,
        None,
        search.report.call_budget,
        stage_metrics=search.report.stage_metrics,
    )
    return WorkloadExecution(report, search.lift, search.replay_writers)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _case_slug(case_id: str) -> str:
    identity = case_id.removeprefix("high_performance:")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", identity).strip("_").lower()
    if not slug:
        raise ValueError("workload id does not form a run directory name")
    return slug


def _next_evidence_dir(
    output_root: str | Path,
    case_id: str,
    *,
    prefix: str,
) -> Path:
    parent = Path(output_root) / _case_slug(case_id)
    parent.mkdir(parents=True, exist_ok=True)
    indexes = [
        int(path.name.removeprefix(prefix))
        for path in parent.glob(f"{prefix}[0-9][0-9][0-9]")
        if path.name.removeprefix(prefix).isdigit()
    ]
    index = max(indexes, default=0) + 1
    while True:
        destination = parent / f"{prefix}{index:03d}"
        try:
            # mkdir is the allocation primitive.  If another worker/process
            # wins the same number after our directory scan, advance without
            # reusing or overwriting any evidence root.
            destination.mkdir()
        except FileExistsError:
            index += 1
            continue
        return destination


def next_workload_run_dir(output_root: str | Path, case_id: str) -> Path:
    return _next_evidence_dir(output_root, case_id, prefix="run-")


def next_candidate_fanout_dir(
    output_root: str | Path,
    case_id: str,
) -> Path:
    return _next_evidence_dir(output_root, case_id, prefix="fanout-")


def _read_workload_lineage_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"parent Gate1 evidence lacks {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"parent Gate1 {label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"parent Gate1 {label} must be a JSON object")
    return value


_GENERATED_SOURCE_MANIFEST_KEY = "generated_target_source_manifest"
_GENERATED_SOURCE_MANIFEST_DIGEST_KEY = (
    "generated_target_source_manifest_digest"
)


def _prepared_manifest_lineage(prepared: PreparedWorkload) -> dict[str, Any]:
    """Return the shared generated-source manifest lineage for every frontend."""

    manifest = getattr(prepared, "manifest", None)
    if manifest is None:
        return {}
    return manifest.execution_lineage()


def _execution_manifest_lineage(
    prepared: PreparedWorkload,
) -> dict[str, Any]:
    """Return only the manifest identity that was available at execution time."""

    lineage = _prepared_manifest_lineage(prepared)
    keys = {
        _GENERATED_SOURCE_MANIFEST_KEY,
        _GENERATED_SOURCE_MANIFEST_DIGEST_KEY,
        "execution_generated_target_source_manifest",
        "execution_generated_target_source_manifest_digest",
        "source_slice_sha256",
        "source_location",
        "device_entry_symbol",
        "lineage_required",
    }
    return {key: lineage[key] for key in keys if key in lineage}


def _workload_source_projection(prepared: PreparedWorkload) -> dict[str, Any]:
    projection = {
        **prepared.source_slice.to_json(),
        "source_location": prepared.source_location,
        "lift_source_kind": prepared.request.source.lift_source_kind,
        "lift_source_sha256": hashlib.sha256(
            prepared.request.source.text.encode("utf-8")
        ).hexdigest(),
    }
    projection.update(_prepared_manifest_lineage(prepared))
    # The compatibility ``source_*`` lineage keys describe the execution
    # snapshot.  Keep the currently selected source projection explicit as
    # well, so a later re-ingress cannot make the persisted projection appear
    # to be the old source while its final manifest points at new bytes.
    projection.update(
        {
            "selected_source_slice_sha256": hashlib.sha256(
                prepared.source_slice.code.encode("utf-8")
            ).hexdigest(),
            "selected_source_location": prepared.source_location,
            "selected_device_entry_symbol": prepared.source_slice.function,
        }
    )
    return projection


def _bind_prepared_manifest(
    prepared: PreparedWorkload,
    artifact: Gate1Artifact,
) -> PreparedWorkload:
    with_contract = getattr(prepared, "with_contract", None)
    if callable(with_contract):
        return with_contract(artifact.j, artifact.c)
    return prepared


def _attach_gate2_lifecycle(
    prepared: PreparedWorkload,
    gate2: Gate2SuiteResult,
) -> PreparedWorkload:
    """Attach only runner-verified lifecycle records to the final manifest."""

    if not gate2.cases:
        return prepared
    records: list[GeneratedSourceLifecycleRecord] = []
    for case in gate2.cases:
        raw_records = case.metadata.get("lifecycle_records")
        if raw_records is None:
            continue
        if not isinstance(raw_records, list):
            raise ValueError("Gate2 lifecycle_records must be a JSON array")
        current = tuple(
            GeneratedSourceLifecycleRecord.from_dict(item)
            for item in raw_records
        )
        if records and current != tuple(records):
            raise ValueError(
                "Gate2 cases in one suite produced different lifecycle records"
            )
        records = list(current)
    if not records:
        return prepared
    manifest = getattr(prepared, "manifest", None)
    if manifest is None:
        raise ValueError("Gate2 lifecycle evidence requires a prepared manifest")
    expected_slice = manifest.source_slice_sha256
    if records[0].source_slice_sha256 != expected_slice:
        raise ValueError(
            "Gate2 lifecycle source slice differs from the prepared manifest"
        )
    with_records = getattr(prepared, "with_lifecycle_records", None)
    if not callable(with_records):
        raise TypeError("prepared workload cannot accept lifecycle records")
    return with_records(records)


def _refresh_workload_manifest_lineage(
    run_dir: Path,
    prepared: PreparedWorkload,
) -> None:
    """Refresh only the manifest fields in the existing run lineage."""

    if not _prepared_manifest_lineage(prepared):
        return
    source_projection_path = run_dir / "source_projection.json"
    request_path = run_dir / "request.json"
    if not source_projection_path.is_file() or not request_path.is_file():
        raise ValueError(
            "workload evidence must be initialized before manifest lineage is refreshed"
        )
    request = _read_workload_lineage_object(request_path, "request.json")
    _write_json(source_projection_path, _workload_source_projection(prepared))
    request.update(_prepared_manifest_lineage(prepared))
    _write_json(request_path, request)


def validate_workload_gate1_continuation_parent(
    parent_evidence: str | Path,
    prepared: PreparedWorkload,
    artifact: Gate1Artifact,
) -> Path:
    """Bind a Gate1 continuation to one exact parent case/source/artifact.

    This is lineage validation only.  It fails closed before any provider call
    when the parent evidence lacks the exact artifact/request/source content
    needed to identify its current run.
    """

    parent = Path(parent_evidence).expanduser().resolve()
    if not parent.is_dir():
        raise ValueError("parent Gate1 evidence directory does not exist")

    _read_workload_lineage_object(parent / "gate1.json", "gate1.json")
    parent_artifact_payload = _read_workload_lineage_object(
        parent / "artifact.json",
        "artifact.json",
    )
    try:
        parent_artifact = canonicalize_gate1_artifact(
            Gate1Artifact.from_json(parent_artifact_payload)
        )
    except Exception as exc:
        raise ValueError(
            "parent Gate1 artifact cannot be decoded under the current contract: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if parent_artifact.to_json() != canonicalize_gate1_artifact(artifact).to_json():
        raise ValueError("parent Gate1 artifact bytes denote a different artifact")

    lineage_root: Path | None = None
    required_names = ("request.json", "source_projection.json", "source.py")
    for candidate in (parent, *tuple(parent.parents)[:4]):
        present = tuple((candidate / name).is_file() for name in required_names)
        if not any(present):
            continue
        if not all(present):
            raise ValueError("parent Gate1 lineage receipts are incomplete")
        lineage_root = candidate
        break
    if lineage_root is None:
        raise ValueError("parent Gate1 evidence lacks request/source lineage")

    request = _read_workload_lineage_object(
        lineage_root / "request.json", "request.json"
    )
    expected_request = {
        "case_id": prepared.entry.case_id,
        "frontend": prepared.entry.frontend,
        "c0": prepared.request.binding.values,
        "grid": list(prepared.adapter.grid),
    }
    missing_request = sorted(set(expected_request) - set(request))
    if missing_request:
        raise ValueError(
            "parent Gate1 request lacks lineage field(s): "
            + ", ".join(missing_request)
        )
    mismatched_request = sorted(
        key for key, value in expected_request.items() if request[key] != value
    )
    if mismatched_request:
        raise ValueError(
            "parent Gate1 request differs from selected workload: "
            + ", ".join(mismatched_request)
        )
    parent_source_to_opaque = request.get("source_to_opaque")
    if not isinstance(parent_source_to_opaque, Mapping):
        raise ValueError("parent Gate1 request has no source_to_opaque mapping")
    if any(
        not isinstance(source, str)
        or not isinstance(opaque, str)
        for source, opaque in parent_source_to_opaque.items()
    ):
        raise ValueError("parent Gate1 source_to_opaque mapping is malformed")
    expected_source_to_opaque = dict(prepared.source_to_opaque)
    missing_abi = sorted(
        source
        for source in expected_source_to_opaque
        if source not in parent_source_to_opaque
    )
    mismatched_abi = sorted(
        source
        for source, expected in expected_source_to_opaque.items()
        if parent_source_to_opaque.get(source) != expected
    )
    unexpected_abi = sorted(
        set(parent_source_to_opaque) - set(expected_source_to_opaque)
    )
    if missing_abi or mismatched_abi or unexpected_abi:
        details: list[str] = []
        if missing_abi:
            details.append("missing ABI source name(s): " + ", ".join(missing_abi))
        if mismatched_abi:
            details.append(
                "mismatched ABI source name(s): " + ", ".join(mismatched_abi)
            )
        if unexpected_abi:
            details.append(
                "unexpected ABI source name(s): " + ", ".join(unexpected_abi)
            )
        raise ValueError(
            "parent Gate1 source_to_opaque mapping differs from selected workload: "
            + "; ".join(details)
        )
    if artifact.c.values != request["c0"]:
        raise ValueError("input artifact C differs from parent Gate1 request c0")

    bound_prepared = _bind_prepared_manifest(prepared, artifact)
    # Gate1 parent identity is the execution snapshot.  Gate2 may have
    # appended lifecycle records to the same run and therefore changed the
    # final manifest digest; comparing that future digest here would create a
    # false claim that Gate1 executed against it.
    expected_manifest_lineage = _execution_manifest_lineage(bound_prepared)
    manifest_keys = set(expected_manifest_lineage)
    missing_manifest = sorted(manifest_keys - set(request))
    if missing_manifest:
        raise ValueError(
            "parent Gate1 request has incomplete generated-source manifest "
            "lineage: " + ", ".join(missing_manifest)
        )
    for key, expected in expected_manifest_lineage.items():
        if request[key] != expected:
            raise ValueError(
                "parent Gate1 generated-source manifest differs from the "
                f"selected workload: {key}"
            )
    final_manifest_payload = request.get(
        "final_generated_target_source_manifest"
    )
    final_manifest_digest = request.get(
        "final_generated_target_source_manifest_digest"
    )
    if (final_manifest_payload is None) != (final_manifest_digest is None):
        raise ValueError(
            "parent Gate1 request has an incomplete final manifest lineage"
        )
    if final_manifest_payload is not None:
        try:
            final_manifest = GeneratedTargetSourceManifest.from_dict(
                final_manifest_payload
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "parent Gate1 final manifest lineage is malformed"
            ) from exc
        if final_manifest.digest != final_manifest_digest:
            raise ValueError(
                "parent Gate1 final manifest digest does not match its payload"
            )

    projection = _read_workload_lineage_object(
        lineage_root / "source_projection.json", "source_projection.json"
    )
    expected_projection = _workload_source_projection(bound_prepared)
    projection_lineage_keys = set(_prepared_manifest_lineage(bound_prepared))
    projection_lineage_keys.update(
        {
            "selected_source_slice_sha256",
            "selected_source_location",
            "selected_device_entry_symbol",
        }
    )
    for key, expected in expected_projection.items():
        if key in projection_lineage_keys:
            continue
        if projection.get(key) != expected:
            raise ValueError(
                "parent Gate1 source projection differs from selected workload"
            )
    for key, expected in _execution_manifest_lineage(bound_prepared).items():
        if projection.get(key) != expected:
            raise ValueError(
                "parent Gate1 source projection execution lineage differs"
            )
    expected_source_slice_sha256 = hashlib.sha256(
        bound_prepared.source_slice.code.encode("utf-8")
    ).hexdigest()
    recorded_source_slice_sha256 = projection.get("selected_source_slice_sha256")
    if recorded_source_slice_sha256 != expected_source_slice_sha256:
        raise ValueError(
            "parent Gate1 source projection differs from selected workload"
        )
    try:
        parent_source = (lineage_root / "source.py").read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"parent Gate1 source.py is unreadable: {exc}") from exc
    if parent_source != prepared.request.source.text:
        raise ValueError("parent Gate1 source differs from selected workload")
    return lineage_root


def initialize_workload_evidence(
    run_dir: Path,
    prepared: PreparedWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None,
    replay_provider: ProviderConfig | None = None,
    replay_effort: str | None = None,
    run_masked_j: bool,
    lift_backend: str = "responses",
    lift_workflow: str = "single",
    replay_backend: str = "responses",
    max_lift_calls: int | None = None,
    search_metadata: Mapping[str, Any] | None = None,
) -> None:
    effective_max_lift_calls = (
        prepared.request.max_lift_calls
        if max_lift_calls is None
        else max_lift_calls
    )
    if (
        not isinstance(effective_max_lift_calls, int)
        or effective_max_lift_calls < 1
        or effective_max_lift_calls > MAX_LIFT_CALLS
    ):
        raise ValueError(
            f"max_lift_calls must be in [1, {MAX_LIFT_CALLS}]"
        )
    effective_replay_provider = replay_provider or provider
    effective_replay_effort = effort if replay_effort is None else replay_effort
    (run_dir / "source.py").write_text(
        prepared.request.source.text, encoding="utf-8"
    )
    _write_json(run_dir / "adapter.json", prepared.adapter.to_json())
    _write_json(
        run_dir / "source_projection.json",
        _workload_source_projection(prepared),
    )
    request_record = {
        "case_id": prepared.entry.case_id,
        "provider": provider.name,
        "model": provider.model,
        "reasoning_effort": effort or provider.reasoning_effort,
        "replay_provider": effective_replay_provider.name,
        "replay_model": effective_replay_provider.model,
        "replay_reasoning_effort": (
            effective_replay_effort or effective_replay_provider.reasoning_effort
        ),
        "lift_backend": lift_backend,
        "lift_workflow": lift_workflow,
        "max_lift_calls": effective_max_lift_calls,
        "replay_backend": replay_backend,
        "frontend": prepared.entry.frontend,
        "observer_profile": prepared.entry.observer_profile,
        "q_policy": prepared.entry.q_policy,
        "c0": prepared.request.binding.values,
        "grid": list(prepared.adapter.grid),
        "run_masked_j": run_masked_j,
        "source_to_opaque": dict(prepared.source_to_opaque),
        **_prepared_manifest_lineage(prepared),
    }
    if search_metadata is not None:
        request_record["candidate_search"] = dict(search_metadata)
    _write_json(run_dir / "request.json", request_record)


def _replay_receipt(record: ReplayGeneration | ReplayAttempt) -> dict[str, Any]:
    result: dict[str, Any] = {
        "recorded": True,
        "backend_protocol": record.backend_protocol,
        "model": record.model,
        "response_model": record.response_model,
        "response_id": record.response_id,
        "usage": dict(record.usage),
        "attempts": record.attempts,
        "physical_provider_calls": record.physical_provider_calls,
        "transport_retries": record.transport_retries,
        "elapsed_ms": record.elapsed_ms,
    }
    if isinstance(record, ReplayGeneration):
        result.update(
            {
                "status": "generated",
                "context_receipt": dict(record.context_receipt),
                "function_name": record.function_name,
            }
        )
    else:
        result.update(
            {
                "status": "rejected",
                "diagnostic": record.diagnostic,
            }
        )
    return result


_TRAJECTORY_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "password",
        "secret",
    }
)


def _canonical_replay_trajectory_bytes(
    trajectory: Mapping[str, Any],
) -> bytes:
    return json.dumps(
        trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _assert_sanitized_replay_trajectory(
    value: Any,
    *,
    key: str = "",
) -> None:
    normalized = key.casefold().replace("-", "_")
    if (
        normalized in _TRAJECTORY_SECRET_KEYS
        or normalized.endswith("_api_key")
    ) and value != "<redacted>":
        raise ValueError("Replay trajectory contains an unredacted secret field")
    if isinstance(value, Mapping):
        for item_key, item in value.items():
            _assert_sanitized_replay_trajectory(item, key=str(item_key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_sanitized_replay_trajectory(item)


def _verified_replay_trajectory_bytes(
    record: ReplayGeneration | ReplayAttempt,
) -> bytes | None:
    if not record.trajectory:
        return None
    _assert_sanitized_replay_trajectory(record.trajectory)
    encoded = _canonical_replay_trajectory_bytes(record.trajectory)
    if record.raw_response and record.raw_response != encoded:
        raise ValueError("Replay raw_response differs from canonical trajectory bytes")
    return encoded


def _save_replay_record_evidence(
    directory: Path,
    *,
    prefix: str,
    exact_source_name: str,
    record: ReplayGeneration | ReplayAttempt,
) -> None:
    """Persist exact source, one canonical receipt, and verified trajectory."""

    trajectory_bytes = _verified_replay_trajectory_bytes(record)
    (directory / exact_source_name).write_text(
        record.response_text,
        encoding="utf-8",
    )
    if isinstance(record, ReplayGeneration) and record.code != record.response_text:
        (directory / f"{prefix}_normalized.py").write_text(
            record.code,
            encoding="utf-8",
        )
    _write_json(directory / f"{prefix}_receipt.json", _replay_receipt(record))
    if trajectory_bytes is not None:
        # Write the parsed sanitized value only after its canonical byte hash has
        # been bound to the record receipt.
        _write_json(directory / f"{prefix}_trajectory.json", record.trajectory)


def save_workload_execution(
    run_dir: Path,
    prepared: PreparedWorkload,
    execution: WorkloadExecution,
) -> None:
    report = execution.report
    bound_prepared = prepared
    if report.accepted_attempt is not None:
        accepted_attempts = tuple(
            attempt
            for attempt in report.attempts
            if attempt.formal_round == report.accepted_attempt
        )
        if len(accepted_attempts) != 1:
            raise ValueError(
                "report does not identify exactly one accepted workload attempt"
            )
        accepted_artifact = accepted_attempts[0].artifact
        if accepted_artifact is None:
            raise ValueError("accepted workload attempt lacks its Gate1 artifact")
        bound_prepared = _bind_prepared_manifest(prepared, accepted_artifact)
    if report.gate2 is not None:
        bound_prepared = _attach_gate2_lifecycle(bound_prepared, report.gate2)
    _refresh_workload_manifest_lineage(run_dir, bound_prepared)
    report_payload = report.to_json()
    report_payload["manifest_lineage"] = _prepared_manifest_lineage(
        bound_prepared
    )
    _write_json(run_dir / "report.json", report_payload)
    _save_attempt_evidence(
        run_dir,
        report.attempts,
        execution.lift,
        execution.replay_writers,
    )
    gate2_attempts = execution.gate2_attempts
    if not gate2_attempts and report.gate2 is not None:
        gate2_attempts = (report.gate2,)
    if gate2_attempts and report.gate2 is not gate2_attempts[-1]:
        if report.gate2 is None or report.gate2.to_json() != gate2_attempts[-1].to_json():
            raise ValueError("final Gate2 report differs from retry evidence")
    if len(gate2_attempts) > 1:
        for index, gate2_attempt in enumerate(gate2_attempts, start=1):
            _write_json(
                run_dir / f"gate2-attempt-{index:02d}.json",
                gate2_attempt.to_json(),
            )
    if report.gate2 is not None:
        gate2_payload = report.gate2.to_json()
        gate2_payload["manifest_lineage"] = _prepared_manifest_lineage(
            bound_prepared
        )
        _write_json(run_dir / "gate2.json", gate2_payload)
    classification = classify_scrc_report(report)
    summary = [
        f"# {prepared.entry.case_id} {run_dir.name}",
        "",
        f"- classification: `{classification.value}`",
        f"- main_status: `{report.main_status.value}`",
        f"- accepted_attempt: `{report.accepted_attempt}`",
        "- j_necessity_status: `"
        + (
            report.j_necessity_status.value
            if report.j_necessity_status is not None
            else "NOT_RUN"
        )
        + "`",
        "- call_budget: `"
        + json.dumps(report.call_budget.to_json(), sort_keys=True)
        + "`",
    ]
    if report.gate2 is not None:
        summary.append(
            "- gate2_status_counts: `"
            + json.dumps(report.gate2.status_counts, sort_keys=True)
            + "`"
        )
    if gate2_attempts:
        summary.append(f"- gate2_attempts: `{len(gate2_attempts)}`")
    summary.extend(
        [
            "",
            "本目录中的 digest/receipt 仅是普通运行追溯信息，不参与 TypeCheck、Gate1 或 Gate2 判定。",
            "",
        ]
    )
    (run_dir / "SUMMARY.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )


def _save_attempt_evidence(
    run_dir: Path,
    attempts: tuple[Any, ...],
    lift: RecordedLiftAdapter,
    replay_writers: tuple[RecordedReplayRoundEvidence, ...],
) -> None:
    for index, attempt in enumerate(attempts, start=1):
        attempt_dir = run_dir / f"attempt-{index:02d}"
        attempt_dir.mkdir()
        if attempt.artifact is not None:
            _write_json(attempt_dir / "artifact.json", attempt.artifact.to_json())
        interface = attempt.lift_metadata.get("layer_c_interface")
        if isinstance(interface, Mapping):
            _write_json(attempt_dir / "layer_c_interface.json", interface)
        if attempt.typecheck is not None:
            _write_json(
                attempt_dir / "typecheck.json", attempt.typecheck.to_json()
            )
        if attempt.gate1 is not None:
            _write_json(attempt_dir / "gate1.json", attempt.gate1.to_json())
        lift_call = getattr(attempt, "lift_call", index)
        if 1 <= lift_call <= len(lift.records):
            record = lift.records[lift_call - 1]
            if record.extraction is not None:
                provenance = record.extraction.provenance
                _write_json(
                    attempt_dir / "lift_receipt.json",
                    {
                        "model": provenance.model,
                        "response_model": provenance.response_model,
                        "response_id": provenance.response_id,
                        "usage": dict(provenance.usage),
                        "attempts": provenance.attempts,
                        "elapsed_ms": provenance.elapsed_ms,
                        "backend": provenance.backend,
                        "termination_status": provenance.termination_status,
                    },
                )
                (attempt_dir / "lift_response.txt").write_text(
                    provenance.response_text, encoding="utf-8"
                )
                if provenance.trajectory is not None:
                    _write_json(
                        attempt_dir / "lift_trajectory.json",
                        provenance.trajectory,
                    )
            else:
                (attempt_dir / "lift_rejection.txt").write_text(
                    (record.diagnostic or "Lift response rejected") + "\n",
                    encoding="utf-8",
                )
                for rejected_index, rejected in enumerate(
                    record.rejected_attempts, start=1
                ):
                    (attempt_dir / f"lift_rejected_{rejected_index:02d}.txt").write_text(
                        rejected.response_text, encoding="utf-8"
                    )
                    _write_json(
                        attempt_dir
                        / f"lift_rejected_{rejected_index:02d}_receipt.json",
                        {
                            "model": rejected.model,
                            "response_model": rejected.response_model,
                            "response_id": rejected.response_id,
                            "usage": dict(rejected.usage),
                            "attempts": rejected.attempts,
                            "elapsed_ms": rejected.elapsed_ms,
                            "backend": rejected.backend,
                            "termination_status": rejected.termination_status,
                            "diagnostic": rejected.diagnostic,
                        },
                    )
                    if rejected.trajectory is not None:
                        _write_json(
                            attempt_dir
                            / f"lift_rejected_{rejected_index:02d}_trajectory.json",
                            rejected.trajectory,
                        )
            if record.backend_evidence is not None:
                worker = dict(record.backend_evidence)
                trajectory = worker.pop("trajectory", None)
                _write_json(attempt_dir / "lift_worker.json", worker)
                if trajectory is not None and not (
                    record.extraction is not None
                    and record.extraction.provenance.trajectory is not None
                ):
                    _write_json(attempt_dir / "lift_trajectory.json", trajectory)
            if record.continuation_candidate is not None:
                _write_json(
                    attempt_dir / "unaccepted_lift_continuation.json",
                    dict(record.continuation_candidate),
                )
    formal_attempts = [
        (run_dir / f"attempt-{index:02d}", attempt)
        for index, attempt in enumerate(attempts, start=1)
        if attempt.formal_round is not None
    ]
    for (attempt_dir, attempt), writer in zip(
        formal_attempts, replay_writers, strict=True
    ):
        record_entries: list[tuple[str, ReplayGeneration | ReplayAttempt]] = []
        masked_writer: RecordedReplayWriter | None = None
        if isinstance(writer, RecordedReplayRound):
            masked_writer = writer.masked
            writer = writer.normal
        if isinstance(writer, RecordedReplayWriterGroup):
            if attempt.gate1 is None:
                raise ValueError("Replay writer fan-out lacks Gate1 evidence")
            selected = attempt.gate1.selected_normal_index
            if not 0 <= selected < len(writer.writers):
                raise ValueError("Gate1 selected Replay index is out of range")
            selected_records = list(writer.writers[selected].records)
            for record_index, record in enumerate(selected_records):
                prefix = (
                    "replay"
                    if record_index == 0
                    else "masked_j_replay"
                    if record_index == 1
                    else f"replay_selected_extra_{record_index + 1:02d}"
                )
                record_entries.append((prefix, record))
            for writer_index, candidate_writer in enumerate(writer.writers):
                if writer_index == selected:
                    continue
                for record_index, record in enumerate(candidate_writer.records):
                    suffix = "" if record_index == 0 else f"_{record_index + 1:02d}"
                    record_entries.append(
                        (
                            f"replay_candidate_{writer_index + 1:02d}{suffix}",
                            record,
                        )
                    )
            if not record_entries and attempt.gate1.normal.generation is not None:
                record_entries.append(
                    ("replay", attempt.gate1.normal.generation)
                )
        else:
            records = list(writer.records)
            if not records and attempt.gate1 is not None:
                normal = attempt.gate1.normal.generation
                masked = (
                    attempt.gate1.masked_j.generation
                    if attempt.gate1.masked_j is not None
                    else None
                )
                records.extend(
                    record for record in (normal, masked) if record is not None
                )
            for record_index, record in enumerate(records, start=1):
                prefix = (
                    "replay"
                    if record_index == 1
                    else "masked_j_replay"
                    if record_index == 2
                    else f"replay_extra_{record_index:02d}"
                )
                record_entries.append((prefix, record))
        if masked_writer is not None:
            for record_index, record in enumerate(masked_writer.records, start=1):
                prefix = (
                    "masked_j_replay"
                    if record_index == 1
                    else f"masked_j_replay_extra_{record_index:02d}"
                )
                record_entries.append((prefix, record))
            if (
                not masked_writer.records
                and attempt.gate1 is not None
                and attempt.gate1.masked_j is not None
                and attempt.gate1.masked_j.generation is not None
            ):
                record_entries.append(
                    ("masked_j_replay", attempt.gate1.masked_j.generation)
                )
        for prefix, record in record_entries:
            exact_name = (
                f"{prefix}.py"
                if isinstance(record, ReplayGeneration)
                else f"{prefix}_rejected.txt"
            )
            _save_replay_record_evidence(
                attempt_dir,
                prefix=prefix,
                exact_source_name=exact_name,
                record=record,
            )


def save_workload_gate1_search(
    run_dir: Path,
    prepared: PreparedWorkload,
    search: WorkloadGate1Search,
    *,
    selected_for_gate2: bool,
) -> None:
    report = search.report
    _write_json(run_dir / "gate1_search.json", report.to_json())
    _save_attempt_evidence(
        run_dir,
        report.attempts,
        search.lift,
        search.replay_writers,
    )
    summary = [
        f"# {prepared.entry.case_id} {run_dir.name}",
        "",
        "- phase: `pre_gate2_candidate_search`",
        f"- status: `{report.status.value}`",
        f"- accepted_attempt: `{report.accepted_attempt}`",
        f"- selected_for_gate2: `{str(selected_for_gate2).lower()}`",
        "- call_budget: `"
        + json.dumps(report.call_budget.to_json(), sort_keys=True)
        + "`",
        "",
        "该目录是候选搜索 evidence，不是第四个证明 artifact。未被选择的 Gate1-closed "
        "candidate 不执行 Gate2。",
        "",
    ]
    (run_dir / "SUMMARY.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )


def save_candidate_fanout_report(
    fanout_dir: Path,
    report: CandidateFanoutSearchReport,
    *,
    winner_result: Mapping[str, Any] | None,
    lift_backend: str = "responses",
    replay_backend: str = "responses",
    max_lift_calls: int = MAX_LIFT_CALLS,
) -> None:
    if (
        type(max_lift_calls) is not int
        or not 1 <= max_lift_calls <= MAX_LIFT_CALLS
    ):
        raise ValueError(
            f"max_lift_calls must be in [1, {MAX_LIFT_CALLS}]"
        )
    payload = report.to_json()
    payload["lift_backend"] = lift_backend
    payload["replay_backend"] = replay_backend
    payload["max_lift_calls"] = max_lift_calls
    payload["winner_result"] = (
        dict(winner_result) if winner_result is not None else None
    )
    _write_json(fanout_dir / "fanout-report.json", payload)
    summary = [
        f"# {report.case_id} {fanout_dir.name}",
        "",
        f"- candidate_count: `{len(report.branches)}`",
        f"- selection_rule: `{report.selection_rule}`",
        f"- lift_backend: `{lift_backend}`",
        f"- replay_backend: `{replay_backend}`",
        f"- max_lift_calls: `{max_lift_calls}`",
        "- winner_branch_id: `"
        + (str(report.winner.branch_id) if report.winner is not None else "None")
        + "`",
        "- pre_gate2_status_counts: `"
        + json.dumps(report.counts, sort_keys=True)
        + "`",
    ]
    if winner_result is not None:
        summary.append(
            f"- winner_main_status: `{winner_result.get('main_status')}`"
        )
    summary.extend(
        [
            "",
            "所有 branch 使用同一 Source/Harness/c0，但拥有独立 Lift、Replay、TypeCheck、"
            "Gate1 与 relift history。Gate2 只对预先选定的 winner 执行。",
            "",
        ]
    )
    (fanout_dir / "SUMMARY.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )


def save_workload_failure(
    run_dir: Path,
    error: BaseException,
    *,
    evidence: Mapping[str, Any] | None = None,
    runtime: WorkloadExecutionError | None = None,
) -> None:
    if runtime is None and isinstance(error, WorkloadExecutionError):
        runtime = error
    cause = runtime.cause if runtime is not None else error
    partial_evidence = dict(runtime.evidence) if runtime is not None else {}
    if evidence is not None:
        partial_evidence.update(evidence)
    if runtime is not None and runtime.stage_metrics:
        partial_evidence["stage_metrics"] = {
            str(name): dict(value)
            for name, value in runtime.stage_metrics.items()
        }
    _write_json(
        run_dir / "failure.json",
        {
            "failure_class": type(cause).__name__,
            "diagnostic": str(cause),
            "partial_evidence": partial_evidence,
        },
    )
    if runtime is not None:
        for index, record in enumerate(runtime.lift.records, start=1):
            call_dir = run_dir / f"lift-call-{index:02d}"
            call_dir.mkdir()
            if record.extraction is not None:
                extraction = record.extraction
                _write_json(
                    call_dir / "artifact.json", extraction.artifact.to_json()
                )
                _write_json(
                    call_dir / "layer_c_interface.json",
                    extraction.layer_c_interface_payload,
                )
                (call_dir / "response.txt").write_text(
                    extraction.provenance.response_text, encoding="utf-8"
                )
                _write_json(
                    call_dir / "receipt.json",
                    {
                        "model": extraction.provenance.model,
                        "response_model": extraction.provenance.response_model,
                        "response_id": extraction.provenance.response_id,
                        "usage": dict(extraction.provenance.usage),
                        "attempts": extraction.provenance.attempts,
                        "elapsed_ms": extraction.provenance.elapsed_ms,
                        "backend": extraction.provenance.backend,
                        "termination_status": extraction.provenance.termination_status,
                    },
                )
                if extraction.provenance.trajectory is not None:
                    _write_json(
                        call_dir / "trajectory.json",
                        extraction.provenance.trajectory,
                    )
            else:
                (call_dir / "rejection.txt").write_text(
                    (record.diagnostic or "Lift response rejected") + "\n",
                    encoding="utf-8",
                )
                for rejected_index, rejected in enumerate(
                    record.rejected_attempts, start=1
                ):
                    (call_dir / f"response-{rejected_index:02d}.txt").write_text(
                        rejected.response_text, encoding="utf-8"
                    )
                    if rejected.trajectory is not None:
                        _write_json(
                            call_dir / f"trajectory-{rejected_index:02d}.json",
                            rejected.trajectory,
                        )
            if record.backend_evidence is not None:
                worker = dict(record.backend_evidence)
                trajectory = worker.pop("trajectory", None)
                _write_json(call_dir / "worker.json", worker)
                if trajectory is not None and not (call_dir / "trajectory.json").exists():
                    _write_json(call_dir / "trajectory.json", trajectory)
            if record.continuation_candidate is not None:
                _write_json(
                    call_dir / "unaccepted_lift_continuation.json",
                    dict(record.continuation_candidate),
                )
        for writer_index, writer in enumerate(
            runtime.replay_writers, start=1
        ):
            call_dir = run_dir / f"gate1-call-{writer_index:02d}"
            call_dir.mkdir()
            normal_writer = (
                writer.normal
                if isinstance(writer, RecordedReplayRound)
                else writer
            )
            records = list(normal_writer.records)
            masked_writer = (
                writer.masked if isinstance(writer, RecordedReplayRound) else None
            )
            if masked_writer is not None:
                records.extend(masked_writer.records)
            for record_index, record in enumerate(records, start=1):
                prefix = (
                    "replay"
                    if record_index == 1
                    else "masked_j_replay"
                    if record_index == 2
                    else f"replay_extra_{record_index:02d}"
                )
                exact_name = (
                    f"{prefix}.py"
                    if isinstance(record, ReplayGeneration)
                    else f"{prefix}_rejected.txt"
                )
                _save_replay_record_evidence(
                    call_dir,
                    prefix=prefix,
                    exact_source_name=exact_name,
                    record=record,
                )
    (run_dir / "SUMMARY.md").write_text(
        f"# {run_dir.parent.name} {run_dir.name}\n\n"
        "- classification: `infrastructure_failed`\n"
        f"- exception: `{type(cause).__name__}: {cause}`\n",
        encoding="utf-8",
    )


__all__ = [
    "LIFT_SEMANTIC_PROBE_PROTOCOL",
    "LIFT_SEMANTIC_PROBE_SEED",
    "WorkloadExecution",
    "WorkloadExecutionError",
    "WorkloadGate1Search",
    "WorkloadConsistencyResult",
    "build_workload_gate2_suite",
    "build_workload_jc_suite",
    "classify_scrc_report",
    "complete_prepared_workload_gate2",
    "execute_prepared_workload",
    "gate2_exposes_logical_j_pass_implementation_failure",
    "initialize_workload_evidence",
    "next_candidate_fanout_dir",
    "next_workload_run_dir",
    "retry_prepared_workload_gate1_fixed_artifact",
    "run_workload_lift_semantic_probe",
    "run_prepared_workload_consistency",
    "save_candidate_fanout_report",
    "save_workload_execution",
    "save_workload_failure",
    "save_workload_gate1_search",
    "merge_fixed_artifact_replay_retry_execution",
    "scrc_gate1_search_diagnostic",
    "scrc_attempt_stage_counts",
    "scrc_report_diagnostic",
    "search_prepared_workload_gate1",
    "validate_workload_gate1_continuation_parent",
]
