"""Real SCRC lifecycle helpers for declarative Triton workload adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

import numpy as np

from intentir.ir.canonical_j import j_elaboration
from intentir.ir.value_dimension import resolve_runtime_output_shapes
from intentir.verify.candidate_fanout import CandidateFanoutSearchReport
from intentir.verify.gate1.model import (
    Gate1Artifact,
    ImplementationConfig,
    ParameterKind,
    ParameterMode,
    ScheduleSlotKind,
    SemanticArchive,
)
from intentir.extraction.providers.client import ProviderError
from intentir.extraction.lifters.miniswe import (
    MiniSWELiftAdapter,
    _safe_preparation_receipt,
)
from intentir.extraction.lifters.direct_agent import (
    DIRECT_AGENT_LIFT_BACKEND,
    DirectAgentLiftAdapter,
)
from intentir.extraction.lifters.compiler_observation import (
    SAFE_SOURCE_PREPARATION_REASONS,
    source_preparation_failure_stage,
)
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.adapters import BoundGate1Adapter, TritonGate2ExecutionAdapter
from intentir.verify.gate1.compiler import TritonCompiler
from intentir.verify.gate1.orchestrate import (
    Gate1Config,
    Gate1Runner,
    SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX,
)
from intentir.verify.gate1.replay import (
    MiniSWEReplayWriter,
    RESPONSES_REPLAY_BACKEND,
    RecordedReplayWriter,
    ReplayAttempt,
    ReplayGeneration,
    ResponsesReplayWriter,
)
from intentir.verify.gate1.codex_replay import CodexCLIReplayWriter
from intentir.verify.gate2.model import (
    Gate2Status,
    Gate2SuiteResult,
    program_identity_digests,
)
from intentir.verify.gate2.compare import compare_outputs
from intentir.verify.gate2.inputs import materialize_ragged_input_values
from intentir.verify.gate2.orchestrate import (
    Gate2Case,
    Gate2Runner,
    Gate2Suite,
)
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec
from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
    TritonRunner,
)
from intentir.verify.scrc_batch import BatchClassification
from intentir.verify.scrc_runner import (
    MainStatus,
    MAX_LIFT_CALLS,
    SCRCGateRunner,
    SCRCGate1SearchReport,
    SCRCProgressCallback,
    SCRCRunDependencies,
    SCRCRunReport,
)
from intentir.verify.workload_adapters import (
    BufferBinding,
    BufferInitializer,
    BufferRole,
    GeneratedSourceLifecycleRecord,
    PreparedTritonWorkload,
)


# MCP remains the default formal SCRC Lift route.  The direct-Agent route is
# available only through the explicit ``lift_backend="direct-agent"`` choice
# for isolated validation; it does not alter the default lifecycle.
RecordedLiftAdapter = MiniSWELiftAdapter
PreparedWorkload = PreparedTritonWorkload
LIFT_SEMANTIC_PROBE_PROTOCOL = "intentir.lift.public-semantic-probe.v1"
LIFT_SEMANTIC_PROBE_SEED = 2026073001
LIFT_SEMANTIC_PROBE_MAX_BUFFER_BYTES = 64 * 1024 * 1024
_REPAIRABLE_DENSE_DOMAIN_LOWER_BOUND = re.compile(
    r"^(?:NotImplementedError:\s*)?functional dense domain "
    r"'(?P<domain>d[0-9]+)' lacks a lower zero bound$"
)


def _repairable_dense_domain_lower_bound(
    diagnostic: str | None,
) -> dict[str, Any] | None:
    """Recognize one directly expressible J-domain omission, and nothing else."""

    if not isinstance(diagnostic, str):
        return None
    match = _REPAIRABLE_DENSE_DOMAIN_LOWER_BOUND.fullmatch(diagnostic.strip())
    if match is None:
        return None
    domain = match.group("domain")
    return {
        "repairable": True,
        "rule_code": "j_dense_domain_lower_zero_bound_missing",
        "path": f"j.boundary.domains.{domain}",
    }


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
class RecordedReplayRoundEvidence:
    """All fresh Replay writer invocations owned by one formal Gate1 round.

    A source-blind Replay generation rejection may be retried before the round
    is consumed.  Those independent writer invocations belong to the same
    eventual formal round, rather than becoming synthetic candidate attempts.
    """

    formal_round: int
    writers: tuple[RecordedReplayEvidenceWriter, ...]
    consumed: bool

    def __post_init__(self) -> None:
        if self.formal_round < 1:
            raise ValueError("Replay evidence formal round must be positive")
        if not self.writers:
            raise ValueError("Replay evidence round requires at least one writer")
        member_ids = [
            id(member)
            for writer in self.writers
            for member in (
                writer.writers
                if isinstance(writer, RecordedReplayWriterGroup)
                else (writer,)
            )
        ]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("Replay evidence round reused a writer invocation")

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(record for writer in self.writers for record in writer.records)


def _replay_round_evidence(
    invocations: list[tuple[int, RecordedReplayEvidenceWriter]],
    attempts: tuple[Any, ...],
    *,
    formal_round_offset: int = 0,
) -> tuple[RecordedReplayRoundEvidence, ...]:
    """Bind Replay writer ownership to the formal-round lifecycle only.

    An invocation whose round was never consumed (for example, a terminal
    Replay response rejection) is retained as an unconsumed owner.  Its source
    or response is intentionally never attached to an attempt directory.
    """

    if formal_round_offset < 0:
        raise ValueError("Replay evidence formal-round offset cannot be negative")
    consumed_rounds = {
        attempt.formal_round
        for attempt in attempts
        if attempt.formal_round is not None
    }
    grouped: dict[int, list[RecordedReplayEvidenceWriter]] = {}
    for local_round, writer in invocations:
        formal_round = formal_round_offset + local_round
        grouped.setdefault(formal_round, []).append(writer)
    if len(consumed_rounds) != sum(
        attempt.formal_round is not None for attempt in attempts
    ):
        raise ValueError("formal Gate1 evidence rounds must be unique")
    missing = consumed_rounds.difference(grouped)
    if missing:
        raise ValueError("consumed formal Gate1 round lacks Replay writer ownership")
    return tuple(
        RecordedReplayRoundEvidence(
            formal_round=formal_round,
            writers=tuple(writers),
            consumed=formal_round in consumed_rounds,
        )
        for formal_round, writers in sorted(grouped.items())
    )


def _new_replay_writer(
    replay_backend: str,
    *,
    provider: ProviderConfig,
    effort: str | None,
    stop_requested: Callable[[], int | None] | None = None,
) -> RecordedReplayWriter:
    """Create one writer for exactly one formal Gate1 round."""

    if replay_backend == "responses":
        return ResponsesReplayWriter(provider=provider, effort=effort)
    writer_kwargs: dict[str, Any] = {
        "provider": provider,
        "effort": effort,
    }
    if stop_requested is not None:
        writer_kwargs["stop_requested"] = stop_requested
    if replay_backend == "miniswe":
        return MiniSWEReplayWriter(**writer_kwargs)
    if replay_backend == "codex":
        return CodexCLIReplayWriter(**writer_kwargs)
    raise ValueError(f"unknown Replay backend: {replay_backend!r}")


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
    elaboration = j_elaboration(j)
    canonical_outputs = elaboration.output_bindings
    canonical_output_by_name = elaboration.output_by_name

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
        output_binding = canonical_output_by_name.get(logical_ref)
        binding_type = output_binding.output_type if output_binding is not None else tensor
        if binding_type.dtype != buffer.dtype:
            raise ValueError("J tensor dtype differs from the Harness buffer")
        _bind_shape(
            binding_type.shape,
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
    if tuple(item.output for item in canonical_outputs) != tuple(
        item.name for item in j.intent.outputs
    ):
        raise ValueError(
            "J output declarations differ from canonical output ABI order"
        )
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
        [list(item.output_type.shape) for item in canonical_outputs],
        runtime_inputs,
        dims=dimensions,
        canonical_output_bindings=canonical_outputs,
    )
    for output, resolved_shape in zip(
        (j_outputs[item.output] for item in canonical_outputs),
        resolved_output_shapes,
        strict=True,
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
    output_keys = tuple(output_bindings[item.output] for item in canonical_outputs)
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
    prepared: PreparedTritonWorkload,
    artifact: Gate1Artifact,
) -> dict[str, Any]:
    """Run a public, non-certifying Source-vs-J probe for the Lift owner.

    The probe uses a fixed public seed distinct from the adapter's hidden Gate2
    seed.  Source executes with the candidate C, while the reference leg receives
    J and logical input bindings only.  No Replay, pi relation, Gate observation,
    or hidden input enters this diagnostic.
    """

    if artifact.c.ordered_value_items() != prepared.request.binding.c_values:
        return {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "unavailable",
            "rule_code": "semantic_probe_c_binding_mismatch",
            "path": "<candidate-c>",
            "authority": "public diagnostic only; not CandidateCheck or certification",
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
            "rule_code": "semantic_probe_buffer_budget_exceeded",
            "path": "<public-buffer-budget>",
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
    except Exception as exc:
        return {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "unavailable",
            "rule_code": "semantic_probe_case_construction_failed",
            "path": "<public-case-construction>",
            "exception_type": type(exc).__name__,
            "authority": (
                "public diagnostic only; probe failure does not reject or certify "
                "the candidate"
            ),
        }

    def clone_values() -> dict[str, Any]:
        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in case.values.items()
        }

    try:
        source = TritonRunner().run(
            "source",
            prepared.request.source.gate_source,
            case.triton,
            clone_values(),
        )
    except Exception as exc:
        return {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "unavailable",
            "rule_code": "semantic_probe_source_execution_exception",
            "path": "<public-source-execution>",
            "exception_type": type(exc).__name__,
            "authority": (
                "public diagnostic only; probe failure does not reject or certify "
                "the candidate"
            ),
        }
    if not source.ok:
        failure_class = source.metadata.get("failure_class")
        return {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "unavailable",
            "rule_code": "semantic_probe_source_execution_failed",
            "path": "<public-source-execution>",
            "failed_leg": "source",
            "unsupported": source.unsupported,
            **(
                {"failure_class": failure_class}
                if isinstance(failure_class, str)
                and re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", failure_class)
                is not None
                else {}
            ),
            "authority": (
                "public diagnostic only; unsupported execution does not reject "
                "or certify the candidate"
            ),
        }

    try:
        reference = ScalarLoweringRunner().run(case.refeval, clone_values())
    except Exception as exc:
        return {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "unavailable",
            "rule_code": "semantic_probe_reference_execution_exception",
            "path": "<public-j-reference-execution>",
            "exception_type": type(exc).__name__,
            "authority": (
                "public diagnostic only; probe failure does not reject or certify "
                "the candidate"
            ),
        }
    if not reference.ok:
        repairable = (
            _repairable_dense_domain_lower_bound(reference.error)
            if reference.unsupported else None
        )
        failure_class = reference.metadata.get("failure_class")
        result = {
            "protocol": LIFT_SEMANTIC_PROBE_PROTOCOL,
            "status": "unavailable",
            "rule_code": (
                "semantic_probe_reference_repairable_domain"
                if repairable is not None
                else "semantic_probe_reference_unsupported"
                if reference.unsupported
                else "semantic_probe_reference_execution_failed"
            ),
            "path": "<public-j-reference-execution>",
            "failed_leg": "j",
            "unsupported": reference.unsupported,
            **(
                {"failure_class": failure_class}
                if isinstance(failure_class, str)
                and re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", failure_class)
                is not None
                else {}
            ),
            "authority": (
                "public diagnostic only; unsupported execution does not reject "
                "or certify the candidate"
            ),
        }
        if repairable is not None:
            result.update(repairable)
        return result

    try:
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
            "rule_code": "semantic_probe_comparison_failed",
            "path": "<public-source-j-comparison>",
            "exception_type": type(exc).__name__,
            "authority": (
                "public diagnostic only; probe failure does not reject or certify "
                "the candidate"
            ),
        }


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
    preparation_failures = [
        attempt
        for attempt in report.attempts
        if (
            attempt.feedback is not None
            and attempt.feedback.diagnostics.get("failure_class")
            == "lift_provider_preparation"
        )
    ]
    # A provider transport/preparation failure is not a candidate or semantic
    # failure.  Preserve that phase in the batch result when no later attempt
    # reached TypeCheck or Gate1; otherwise the later semantic phase remains
    # authoritative.
    if preparation_failures and all(
        attempt.typecheck is None and attempt.gate1 is None
        for attempt in report.attempts
    ):
        return BatchClassification.LIFT_PROVIDER_PREPARATION
    terminal = report.attempts[-1]
    if terminal.formal_round is None:
        if terminal.gate1 is not None:
            if terminal.gate1.normal.generation is None:
                return BatchClassification.REPLAY_RESPONSE_REJECTED
            return BatchClassification.NO_CLOSURE
        if (
            terminal.feedback is not None
            and terminal.feedback.diagnostics.get("failure_class")
            in {
                "lift_semantic_probe_diverged",
                "lift_semantic_probe_repairable_unsupported",
            }
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
        for attempt in reversed(attempts):
            if attempt.feedback is None:
                continue
            diagnostics = attempt.feedback.diagnostics
            if (
                diagnostics.get("failure_class")
                == "lift_provider_preparation"
                and isinstance(diagnostics.get("diagnostic"), str)
            ):
                return diagnostics["diagnostic"]
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
            if attempt.gate1 is not None:
                if attempt.gate1.normal.generation is None:
                    stages["replay_response_rejected"] += 1
                else:
                    stages["no_closure"] += 1
                continue
            failure_class = (
                attempt.feedback.diagnostics.get("failure_class")
                if attempt.feedback is not None
                else None
            )
            if failure_class in {
                "lift_provider_preparation",
                "lift_agent_preparation",
            }:
                stages[str(failure_class)] += 1
                continue
            if failure_class == "lift_semantic_probe_diverged":
                stages["semantic_probe_diverged"] += 1
                continue
            if failure_class == "lift_semantic_probe_repairable_unsupported":
                stages["semantic_probe_repairable_unsupported"] += 1
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
    _replay_writer_registry: list[tuple[int, RecordedReplayEvidenceWriter]]
    _replay_rounds: tuple[RecordedReplayRoundEvidence, ...]

    @property
    def replay_writers(self) -> tuple[RecordedReplayRoundEvidence, ...]:
        """Replay owners grouped by their real formal-round lineage."""

        return self._replay_rounds


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
    lift_backend: str = "miniswe",
    lift_workflow: str = "single",
    replay_backend: str = "responses",
    replay_width: int = 1,
    replay_writer_factory: Callable[[], RecordedReplayWriter] | None = None,
    stop_requested: Callable[[], int | None] | None = None,
) -> tuple[
    SCRCGateRunner,
    RecordedLiftAdapter,
    list[tuple[int, RecordedReplayEvidenceWriter]],
]:
    if not isinstance(replay_width, int) or isinstance(replay_width, bool):
        raise TypeError("replay_width must be an integer")
    if not 1 <= replay_width <= 4:
        raise ValueError("replay_width must be in [1, 4]")
    if lift_workflow != "single":
        raise ValueError(
            "lift_workflow must be 'single'; staged multi-writer Lift was retired"
        )
    if prepared.request.frontend != "triton":
        raise ValueError(
            "shared SCRC runtime is Triton/TTIR-only; CUDA and TileLang must "
            "use their native Gate runners, and CuTe must use CuteGate1Runner"
        )
    if lift_backend == "miniswe":
        lift_class: type[Any] = MiniSWELiftAdapter
    elif lift_backend == DIRECT_AGENT_LIFT_BACKEND:
        lift_class = DirectAgentLiftAdapter
    else:
        raise ValueError(
            "unsupported SCRC Lift backend; choose the default 'miniswe' "
            "or explicitly opt into 'direct-agent'"
        )
    # Lift and source-blind Replay have separate provider owners even when
    # both are configured to use the same endpoint/model.  ProviderConfig is
    # immutable, but distinct instances make the lifecycle boundary explicit
    # and prevent a future stateful transport from being shared accidentally.
    lift_provider = replace(provider)
    effective_replay_provider = replace(replay_provider or provider)
    lift_kwargs: dict[str, Any] = {
        "provider": lift_provider,
        "compiler_view": prepared.compiler_view,
        "effort": effort,
        "source_symbol": prepared.request.source.lift_source_symbol
        or getattr(
            prepared.source_slice,
            "function",
            prepared.entry.symbol,
        ),
        "source_kind": prepared.request.source.lift_source_kind,
        "source_parameter_order": prepared.request.source.lift_parameter_order,
        "source_to_opaque": dict(
            prepared.request.source.lift_source_to_opaque
        ),
    }
    if lift_class is MiniSWELiftAdapter:
        lift_kwargs["workflow"] = lift_workflow
    lift = lift_class(
        **lift_kwargs,
    )
    effective_replay_effort = effort if replay_effort is None else replay_effort
    writers: list[tuple[int, RecordedReplayEvidenceWriter]] = []
    compiler = TritonCompiler(timeout=240.0)

    def new_writer() -> RecordedReplayWriter:
        return (
            replay_writer_factory()
            if replay_writer_factory is not None
            else _new_replay_writer(
                replay_backend,
                provider=effective_replay_provider,
                effort=effective_replay_effort,
                stop_requested=stop_requested,
            )
        )

    def gate1_factory(formal_round: int) -> BoundGate1Adapter:
        if formal_round < 1:
            raise ValueError("Gate1 formal round must be positive")
        round_writers = tuple(new_writer() for _ in range(replay_width))
        writer_evidence: RecordedReplayEvidenceWriter
        if replay_width == 1:
            writer_evidence = round_writers[0]
        else:
            writer_evidence = RecordedReplayWriterGroup(round_writers)
        # Generation-only retries receive a newly constructed adapter for the
        # same formal round.  Preserve each independent writer as an ordered
        # invocation under that round; do not invent another candidate or
        # attach it to a later Lift attempt.
        writers.append((formal_round, writer_evidence))

        return BoundGate1Adapter(
            Gate1Runner(
                compiler=compiler,
                writer=(
                    round_writers[0]
                    if replay_width == 1
                    else round_writers
                ),
                config=Gate1Config(run_masked_j=run_masked_j),
            ),
        )

    gate2 = TritonGate2ExecutionAdapter(
        Gate2Runner(),
        lambda artifact, _gate1, _binding: build_workload_gate2_suite(
            prepared, artifact
        ),
    )

    # MCP Lift returns a checked artifact directly, so the shared runner keeps
    # the public Source-vs-J probe before Replay/Gate1.
    semantic_probe = lambda artifact: run_workload_lift_semantic_probe(
        prepared, artifact
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
    effort: str | None = None,
    replay_provider: ProviderConfig | None = None,
    replay_effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "miniswe",
    lift_workflow: str = "single",
    replay_backend: str = "responses",
    replay_width: int = 1,
    replay_writer_factory: Callable[[], RecordedReplayWriter] | None = None,
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
    search_started = time.monotonic()
    try:
        report = runner.search_gate1(
            request,
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
                "provider_failure_origin": getattr(
                    exc, "failure_origin", "unknown"
                ),
            }
            provider_evidence: dict[str, Any] = {
                "failure_class": "lift_provider_preparation",
                "provider_failure_class": exc.failure_class,
                "provider_failure_origin": getattr(
                    exc, "failure_origin", "unknown"
                ),
                "candidate_present": False,
            }
            provider_error = getattr(exc, "safe_error_envelope", None)
            if isinstance(provider_error, Mapping) and provider_error:
                provider_evidence["provider_error"] = dict(provider_error)
            existing_evidence = getattr(exc, "evidence", None)
            exc.evidence = {
                **(
                    dict(existing_evidence)
                    if isinstance(existing_evidence, Mapping)
                    else {}
                ),
                **provider_evidence,
            }
        # A later provider failure must not erase the safe rule-level outcome
        # of earlier fresh Lift calls. Never copy rejected candidate text or
        # the backend's free-form diagnostic into infrastructure evidence.
        prior_rejections: list[dict[str, Any]] = []
        for record in getattr(lift, "records", ()):
            backend = getattr(record, "backend_evidence", None)
            if not isinstance(backend, Mapping):
                continue
            rules: list[dict[str, str]] = []
            for source in ("candidate_check", "source_reconcile"):
                check = backend.get(source)
                diagnostics = check.get("diagnostics") if isinstance(check, Mapping) else None
                if not isinstance(diagnostics, list):
                    continue
                for item in diagnostics:
                    if not isinstance(item, Mapping):
                        continue
                    code, path = item.get("rule_code"), item.get("path")
                    if not isinstance(code, str) or re.fullmatch(r"[a-z0-9_]{1,96}", code) is None:
                        continue
                    rule = {"rule_code": code}
                    if isinstance(path, str) and re.fullmatch(r"(?:j|pi)\.[A-Za-z0-9_.\[\]]{1,120}|<(?:candidate|canonical-constructor)>", path):
                        rule["path"] = path
                    rules.append(rule)
                    if len(rules) >= 12:
                        break
                if len(rules) >= 12:
                    break
            if rules:
                prior_rejections.append({"lift_call": record.relift_round, "rules": rules})
            if len(prior_rejections) >= 3:
                break
        if prior_rejections:
            existing_evidence = getattr(exc, "evidence", None)
            exc.evidence = {
                **(dict(existing_evidence) if isinstance(existing_evidence, Mapping) else {}),
                "prior_lift_rejections": prior_rejections,
            }
        raise WorkloadExecutionError(
            exc,
            lift,
            tuple(writers),
            stage_metrics=stage_metrics,
        ) from exc
    return WorkloadGate1Search(
        report,
        runner,
        lift,
        writers,
        _replay_round_evidence(writers, report.attempts),
    )


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


def execute_prepared_workload(
    prepared: PreparedWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    replay_provider: ProviderConfig | None = None,
    replay_effort: str | None = None,
    run_masked_j: bool = True,
    lift_backend: str = "miniswe",
    lift_workflow: str = "single",
    replay_backend: str = "responses",
    max_gate1_rounds: int | None = None,
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
        max_gate1_rounds=max_gate1_rounds,
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


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"workload evidence lacks {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"workload {label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"workload {label} must be a JSON object")
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
    request = _read_json_object(request_path, "request.json")
    _write_json(source_projection_path, _workload_source_projection(prepared))
    request.update(_prepared_manifest_lineage(prepared))
    _write_json(request_path, request)


def initialize_workload_evidence(
    run_dir: Path,
    prepared: PreparedWorkload,
    *,
    provider: ProviderConfig,
    effort: str | None,
    replay_provider: ProviderConfig | None = None,
    replay_effort: str | None = None,
    run_masked_j: bool,
    lift_backend: str = "miniswe",
    lift_workflow: str = "single",
    replay_backend: str = "responses",
    max_gate1_rounds: int | None = None,
    max_lift_calls: int | None = None,
    search_metadata: Mapping[str, Any] | None = None,
) -> None:
    effective_max_lift_calls = (
        prepared.request.max_lift_calls
        if max_lift_calls is None
        else max_lift_calls
    )
    effective_max_gate1_rounds = (
        prepared.request.max_gate1_rounds
        if max_gate1_rounds is None
        else max_gate1_rounds
    )
    if (
        not isinstance(effective_max_lift_calls, int)
        or effective_max_lift_calls < 1
        or effective_max_lift_calls > MAX_LIFT_CALLS
    ):
        raise ValueError(
            f"max_lift_calls must be in [1, {MAX_LIFT_CALLS}]"
        )
    if (
        not isinstance(effective_max_gate1_rounds, int)
        or effective_max_gate1_rounds < 1
    ):
        raise ValueError("max_gate1_rounds must be positive")
    # Keep the evidence record aligned with the independent Replay owner used
    # by the runtime.  No response continuation or agent history is shared.
    effective_replay_provider = replace(replay_provider or provider)
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
        "provider": "configured-provider",
        "model": "configured-model",
        "reasoning_effort": effort or provider.reasoning_effort,
        "replay_provider": "configured-provider",
        "replay_model": "configured-model",
        "replay_reasoning_effort": (
            effective_replay_effort or effective_replay_provider.reasoning_effort
        ),
        "lift_backend": lift_backend,
        "lift_workflow": lift_workflow,
        "max_lift_calls": effective_max_lift_calls,
        "max_gate1_rounds": effective_max_gate1_rounds,
        "replay_backend": replay_backend,
        "frontend": prepared.entry.frontend,
        "observer_profile": prepared.entry.observer_profile,
        "q_policy": prepared.entry.q_policy,
        "c0": prepared.request.binding.values,
        "grid": list(prepared.adapter.grid),
        "run_masked_j": run_masked_j,
        "source_to_opaque": dict(prepared.source_to_opaque),
        "source_facts_digest": prepared.request.source_facts.digest,
        **_prepared_manifest_lineage(prepared),
    }
    if search_metadata is not None:
        request_record["candidate_search"] = dict(search_metadata)
    _write_json(run_dir / "request.json", request_record)


def _replay_receipt(record: ReplayGeneration | ReplayAttempt) -> dict[str, Any]:
    result: dict[str, Any] = {
        "recorded": True,
        "backend_protocol": record.backend_protocol,
        "model": "<redacted-model>",
        "response_model": "<redacted-model>",
        "response_id": "<redacted>",
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
                "ingress_diagnostics": [
                    dict(item) for item in record.ingress_diagnostics
                ],
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
    # Responses stores the unpersisted raw HTTP response in ``raw_response``;
    # its optional sanitized trajectory is a different evidence payload. Agent
    # backends instead intentionally place canonical trajectory bytes in that
    # field, so retain byte equality there. Comparing the two Response payload
    # classes would turn a completed Gate1 result into a post-hoc evidence
    # persistence failure without affecting any Gate decision.
    if (
        record.backend_protocol != RESPONSES_REPLAY_BACKEND
        and record.raw_response
        and record.raw_response != encoded
    ):
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
        prepared=prepared,
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
    *,
    prepared: PreparedWorkload | None = None,
) -> None:
    for index, attempt in enumerate(attempts, start=1):
        attempt_dir = run_dir / f"attempt-{index:02d}"
        attempt_dir.mkdir()
        if attempt.artifact is not None:
            _write_json(attempt_dir / "artifact.json", attempt.artifact.to_json())
            if (
                prepared is not None
                and attempt.typecheck is not None
                and attempt.typecheck.passed
            ):
                # Keep the exact contract lineage beside the artifact that
                # produced it.  The run-level request may later be refreshed
                # by another candidate in the same search.
                attempt_prepared = _bind_prepared_manifest(
                    prepared,
                    attempt.artifact,
                )
                _write_json(
                    attempt_dir / "manifest_lineage.json",
                    _execution_manifest_lineage(attempt_prepared),
                )
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
                        "model": "<redacted-model>",
                        "response_model": "<redacted-model>",
                        "response_id": "<redacted>",
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
            if record.backend_evidence is not None:
                worker = dict(record.backend_evidence)
                trajectory = worker.pop("trajectory", None)
                _write_json(attempt_dir / "lift_worker.json", worker)
                if trajectory is not None and not (
                    record.extraction is not None
                    and record.extraction.provenance.trajectory is not None
                ):
                    _write_json(attempt_dir / "lift_trajectory.json", trajectory)
    formal_attempts = {
        attempt.formal_round: (run_dir / f"attempt-{index:02d}", attempt)
        for index, attempt in enumerate(attempts, start=1)
        if attempt.formal_round is not None
    }
    if len(formal_attempts) != sum(
        attempt.formal_round is not None for attempt in attempts
    ):
        raise ValueError("formal Gate1 attempt rounds must be unique")
    owners = {owner.formal_round: owner for owner in replay_writers}
    if len(owners) != len(replay_writers):
        raise ValueError("Replay evidence ownership rounds must be unique")
    missing_owners = set(formal_attempts).difference(owners)
    if missing_owners:
        raise ValueError("formal Gate1 attempt lacks Replay writer ownership")
    unexpected_consumed = {
        owner.formal_round
        for owner in replay_writers
        if owner.consumed and owner.formal_round not in formal_attempts
    }
    if unexpected_consumed:
        raise ValueError("consumed Replay owner lacks its formal Gate1 attempt")
    unconsumed = [owner for owner in replay_writers if not owner.consumed]
    if unconsumed:
        # Do not attach an unconsumed response to any candidate directory.  It
        # never reached a formal comparison and therefore has no candidate
        # lineage.  This small receipt records only safe ownership metadata.
        _write_json(
            run_dir / "unconsumed_replay_owners.json",
            {
                "owners": [
                    {
                        "formal_round": owner.formal_round,
                        "writer_invocations": len(owner.writers),
                        "record_count": len(owner.records),
                    }
                    for owner in unconsumed
                ]
            },
        )

    def append_writer_records(
        record_entries: list[tuple[str, ReplayGeneration | ReplayAttempt]],
        attempt: Any,
        writer: RecordedReplayEvidenceWriter,
        *,
        allow_gate1_fallback: bool,
        prefix_namespace: str,
    ) -> None:
        def name(prefix: str) -> str:
            return f"{prefix_namespace}_{prefix}" if prefix_namespace else prefix

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
                record_entries.append((name(prefix), record))
            for writer_index, candidate_writer in enumerate(writer.writers):
                if writer_index == selected:
                    continue
                for record_index, record in enumerate(candidate_writer.records):
                    suffix = "" if record_index == 0 else f"_{record_index + 1:02d}"
                    record_entries.append(
                        (
                            name(
                                f"replay_candidate_{writer_index + 1:02d}{suffix}"
                            ),
                            record,
                        )
                    )
            if (
                allow_gate1_fallback
                and not record_entries
                and attempt.gate1.normal.generation is not None
            ):
                record_entries.append(
                    (name("replay"), attempt.gate1.normal.generation)
                )
        else:
            records = list(writer.records)
            if allow_gate1_fallback and not records and attempt.gate1 is not None:
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
                record_entries.append((name(prefix), record))

    for formal_round, (attempt_dir, attempt) in sorted(formal_attempts.items()):
        owner = owners[formal_round]
        if not owner.consumed:
            raise ValueError("formal Gate1 attempt has an unconsumed Replay owner")
        record_entries: list[tuple[str, ReplayGeneration | ReplayAttempt]] = []
        for invocation_index, writer in enumerate(owner.writers, start=1):
            append_writer_records(
                record_entries,
                attempt,
                writer,
                allow_gate1_fallback=invocation_index == len(owner.writers),
                prefix_namespace=(
                    ""
                    if invocation_index == len(owner.writers)
                    else f"replay_generation_retry_{invocation_index:02d}"
                ),
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
    # A failed Gate1 candidate is still a source-bound, TypeChecked J/C
    # artifact. Preserve that binding in this run's evidence; TypeCheck-failed
    # drafts are intentionally not bound.
    bound_prepared = prepared
    for attempt in reversed(report.attempts):
        if (
            attempt.artifact is not None
            and attempt.typecheck is not None
            and attempt.typecheck.passed
        ):
            bound_prepared = _bind_prepared_manifest(
                prepared,
                attempt.artifact,
            )
            break
    _refresh_workload_manifest_lineage(run_dir, bound_prepared)
    _write_json(run_dir / "gate1_search.json", report.to_json())
    _save_attempt_evidence(
        run_dir,
        report.attempts,
        search.lift,
        search.replay_writers,
        prepared=prepared,
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
        "该目录是候选搜索 evidence，不是第四个证明 artifact。Gate1-closed candidate "
        "只表示具备 Gate2 资格；fan-out 会继续为该 candidate 执行 Gate2，"
        "直到某个 candidate 完整通过三边比较或所有候选耗尽。",
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
    lift_backend: str = "miniswe",
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
            "Gate1、Gate2 与 relift history。Gate1 闭合只使 branch 进入 Gate2；"
            "只有 Gate2 三边全部通过的 branch 才是 winner，并触发同 Kernel 其余 branch 停止。",
            "",
        ]
    )
    (fanout_dir / "SUMMARY.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )


_SAFE_FAILURE_INGRESS_FIELDS = frozenset(
    {
        "rule_code",
        "path",
        "provider_failure_class",
        "provider_failure_origin",
        "failure_class",
        "physical_provider_calls",
        "exception_type",
        "failure_stage",
        "returned_type",
        "line",
    }
)

_SAFE_SOURCE_FAILURE_CLASSES = frozenset({
    "source_preparation_transfer_ledger_unavailable",
    "source_preparation_authorization_unavailable",
})
_SAFE_SOURCE_FAILURE_REASONS = SAFE_SOURCE_PREPARATION_REASONS

_SAFE_HOST_PREPARATION_RULES = frozenset({
    "source_host_launch_contract_malformed",
    "source_host_launch_digest_mismatch",
    "source_host_launch_digest_unavailable",
    "source_host_shape_contract_mismatch",
    "source_host_shape_scalar_mismatch",
    "source_host_stride_binding_ambiguous",
    "source_host_stride_contract_malformed",
    "source_host_stride_instantiation_mismatch",
    "source_host_stride_type_mismatch",
    "source_host_tile_binding_mismatch",
    "source_host_tile_contract_malformed",
    "source_host_tile_helper_unbound",
    "source_host_tile_instantiation_mismatch",
})
_SAFE_PROVIDER_FAILURE_CLASSES = frozenset({
    "provider_authentication_error", "provider_connection_aborted",
    "provider_connection_broken_pipe", "provider_connection_closed",
    "provider_connection_refused", "provider_connection_reset",
    "provider_content_filter", "provider_dns_error", "provider_http_error",
    "provider_malformed_candidate", "provider_network_unreachable",
    "provider_no_candidate", "provider_output_limit", "provider_overloaded",
    "provider_protocol_error", "provider_rate_limit", "provider_response_failed",
    "provider_response_incomplete", "provider_server_error",
    "provider_sse_error_event", "provider_stream_incomplete", "provider_timeout",
    "provider_tls_error", "provider_transport_error", "provider_usage_limit",
})
_SAFE_PROVIDER_FAILURE_ORIGINS = frozenset({
    "completed_envelope", "completed_text", "http_error", "http_server_error",
    "network_transport", "responses_json", "sse_completed_envelope_invalid",
    "sse_completed_response_incomplete", "sse_completion_missing",
    "sse_error_event", "sse_response_failed", "sse_response_incomplete",
    "sse_protocol", "sse_terminal_event", "unknown",
    "sse_read_error",
})
_SAFE_LIFT_FAILURE_CLASSES = frozenset({
    "lift_agent_step_limit",
    "lift_agent_cost_limit",
    "lift_agent_time_limit",
    "lift_agent_format_limit",
    "lift_agent_stagnation",
    "lift_agent_preparation",
    "lift_agent_no_candidate",
    "lift_authoring_preparation",
    "lift_mcp_host_internal_error",
    "lift_mcp_application_error",
    "lift_mcp_no_progress",
    "lift_provider_preparation",
    "lift_source_harness_preparation",
    "lift_agent_infrastructure",
    "lift_adapter_contract_violation",
})
_SAFE_PROVIDER_SELECTION_FAILURE_CLASSES = frozenset({
    "provider_configuration_unavailable",
})

_SAFE_STAGE_METRIC_STRING = re.compile(r"[A-Za-z0-9_.:-]{1,96}\Z")
_SAFE_STAGE_METRIC_FIELDS = frozenset(
    {
        "elapsed_ms",
        "runs",
        "status",
        "attempt",
        "formal_round",
        "attempts",
        "physical_provider_calls",
        "failure_class",
        "candidate_present",
        "candidate_check_started",
        "typecheck_started",
        "replay_started",
        "gate1_started",
        "gate2_started",
        "phase",
        "phase_digest",
        "next_legal_action",
        "last_completed_step",
        "completed_step_count",
        "transfer_read_generation",
        "graph_failure_count",
        "access_failure_count",
        "output_failure_count",
        "boundary_failure_count",
        "mcp_phase_failure_count",
        "last_failure_class",
        "last_rule_code",
        "progress_changed",
    }
)

_SAFE_EXTERNAL_ERROR_KEYS = frozenset(
    {
        "event_type",
        "event",
        "error",
        "response_error",
        "response_status",
        "http_status",
        "exception_type",
        "message",
        "last_event_type",
        "terminal_event_received",
        "code",
        "detail",
        "reason",
        "param",
        "request_id",
    }
)
_SENSITIVE_EXTERNAL_ERROR_KEY = re.compile(
    r"(?i)(?:api[_ -]?key|access[_ -]?token|auth(?:orization)?|bearer|cookie|credential|password|secret|token)"
)
_MAX_EXTERNAL_ERROR_TEXT = 8192
_MAX_EXTERNAL_ERROR_ITEMS = 32


def _redact_external_error_text(value: str) -> str:
    """Preserve bounded external error text while masking credentials."""

    value = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1<redacted>", value)
    value = re.sub(
        r"(?i)((?:authorization|api[_ -]?key|token|secret|password)\s*[:=]\s*)[^\s,;]+",
        r"\1<redacted>",
        value,
    )
    value = re.sub(
        r"(?i)(https?://[^\s?]+)\?[^\s]+",
        r"\1?<redacted>",
        value,
    )
    value = re.sub(r"(?i)https?://[^\s\"'<>]+", "<redacted-url>", value)
    value = re.sub(
        r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]{1,5})?(?![0-9])",
        "<redacted-address>",
        value,
    )
    value = re.sub(
        r"(?<![A-Za-z0-9_])/(?:[^\s\"']+/){1,}[^\s\"']*",
        "<redacted-path>",
        value,
    )
    return value[:_MAX_EXTERNAL_ERROR_TEXT]


def _safe_external_error_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _SENSITIVE_EXTERNAL_ERROR_KEY.search(key):
        return "<redacted>"
    if isinstance(value, str):
        return _redact_external_error_text(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): _safe_external_error_value(item_value, key=str(item_key))
            for item_key, item_value in list(value.items())[:_MAX_EXTERNAL_ERROR_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _safe_external_error_value(item)
            for item in value[:_MAX_EXTERNAL_ERROR_ITEMS]
        ]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return _redact_external_error_text(str(value))


def _safe_external_error_envelope(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    projected = {
        key: _safe_external_error_value(value[key], key=key)
        for key in _SAFE_EXTERNAL_ERROR_KEYS
        if key in value
    }
    return projected or None


def _safe_stage_metrics(
    raw_metrics: Mapping[str, Any] | None,
) -> dict[str, dict[str, int | float | str | bool]]:
    """Project timings/counts and controlled status labels only.

    Stage metrics are included in infrastructure receipts, so they must not
    become an indirect channel for exception text, artifacts, or provider
    bodies.  Keep only the bounded scalar fields owned by the runner.
    """

    if not isinstance(raw_metrics, Mapping):
        return {}
    metrics: dict[str, dict[str, int | float | str | bool]] = {}
    for name, raw in raw_metrics.items():
        if (
            not isinstance(name, str)
            or _SAFE_STAGE_METRIC_STRING.fullmatch(name) is None
            or not isinstance(raw, Mapping)
        ):
            continue
        retained: dict[str, int | float | str | bool] = {}
        for field in _SAFE_STAGE_METRIC_FIELDS:
            value = raw.get(field)
            if isinstance(value, str):
                if _SAFE_STAGE_METRIC_STRING.fullmatch(value) is not None:
                    retained[field] = value
            elif isinstance(value, bool):
                retained[field] = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                retained[field] = value
        if retained:
            metrics[name] = retained
    return metrics


def _safe_gate1_arm_failure(raw_arm: Any) -> dict[str, Any] | None:
    """Project one normal or masked-J Replay arm without authored content."""

    if not isinstance(raw_arm, Mapping):
        return None
    safe_arm: dict[str, Any] = {
        "generation_present": raw_arm.get("generation") is not None,
        "comparison_present": raw_arm.get("comparison") is not None,
    }
    physical_calls = raw_arm.get("physical_calls")
    if (
        isinstance(physical_calls, int)
        and not isinstance(physical_calls, bool)
        and physical_calls >= 0
    ):
        safe_arm["physical_calls"] = physical_calls
    raw_diagnostics = raw_arm.get("ingress_diagnostics")
    if isinstance(raw_diagnostics, list):
        diagnostics = []
        for raw in raw_diagnostics:
            if not isinstance(raw, Mapping):
                continue
            projected = {
                key: value
                for key, value in raw.items()
                if key in _SAFE_FAILURE_INGRESS_FIELDS
                and (
                    isinstance(value, str)
                    or (isinstance(value, int) and not isinstance(value, bool))
                )
            }
            if projected:
                diagnostics.append(projected)
        if diagnostics:
            safe_arm["ingress_diagnostics"] = diagnostics
    return safe_arm


_SAFE_GATE1_CATEGORIES = frozenset(
    {
        "interface",
        "op_multiset",
        "dataflow",
        "reduction",
        "predicate_effect",
        "schedule-only",
    }
)


def safe_gate1_comparison_summary(report: SCRCRunReport) -> list[dict[str, Any]]:
    """Return bounded Gate1 verdict classes, never fingerprint differences.

    A normal lifecycle may finish with NO_CLOSURE, so infrastructure-failure
    metadata alone cannot explain which exact-comparison obligation failed.
    Keep only fixed comparator category names and booleans from formal rounds;
    neither the Replay text nor fingerprint items cross this boundary.
    """

    rounds: list[dict[str, Any]] = []
    for attempt in report.attempts:
        if attempt.formal_round is None or attempt.gate1 is None:
            continue
        gate1 = attempt.gate1
        entry: dict[str, Any] = {
            "formal_round": attempt.formal_round,
            "status": gate1.status.value,
        }
        for name in ("normal", "masked_j"):
            arm = getattr(gate1, name)
            if arm is None:
                continue
            comparison = arm.comparison
            arm_summary: dict[str, Any] = {
                "comparison_present": comparison is not None,
            }
            compile_result = arm.compile_result
            if compile_result is not None and not compile_result.ok:
                # These two values are closed worker classifications; do not
                # serialize the compiler diagnostic because it can contain
                # authored Replay text.
                arm_summary["compile_failure_class"] = (
                    compile_result.failure_class
                    or "triton_frontend_compile_failed"
                )
                if compile_result.exception_type is not None:
                    arm_summary["compile_exception_type"] = (
                        compile_result.exception_type
                    )
            if comparison is not None:
                arm_summary["hard_pass"] = comparison.hard_pass
                arm_summary["failed_categories"] = sorted(
                    category
                    for category, passed in comparison.hard_passes.items()
                    if category in _SAFE_GATE1_CATEGORIES and not passed
                )
            entry[name] = arm_summary
        rounds.append(entry)
    return rounds


def safe_gate2_failure_summary(report: SCRCRunReport) -> list[dict[str, Any]]:
    """Project Gate2 failures without exposing execution error text."""

    gate2 = report.gate2
    if gate2 is None:
        return []
    summaries: list[dict[str, Any]] = []
    for case in gate2.cases:
        if case.status is Gate2Status.PASS:
            continue
        entry: dict[str, Any] = {
            "case_id": case.case_id,
            "status": case.status.value,
        }
        for leg_name, result in (
            ("source", case.source),
            ("replay", case.replay),
            ("reference", case.reference),
        ):
            metadata = result.metadata
            failure_class = metadata.get("failure_class")
            if not isinstance(failure_class, str) or re.fullmatch(
                r"[A-Za-z0-9_.:-]{1,96}", failure_class
            ) is None:
                continue
            leg: dict[str, Any] = {"failure_class": failure_class}
            phase = metadata.get("phase")
            exception_type = metadata.get("exception_type")
            if isinstance(phase, str) and re.fullmatch(
                r"[A-Za-z0-9_.:-]{1,96}", phase
            ) is not None:
                leg["phase"] = phase
            if isinstance(exception_type, str) and re.fullmatch(
                r"[A-Za-z0-9_.:-]{1,96}", exception_type
            ) is not None:
                leg["exception_type"] = exception_type
            entry[leg_name] = leg
        summaries.append(entry)
    return summaries


def safe_workload_failure_metadata(
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project infrastructure evidence to public, source-blind metadata.

    Failure evidence can retain an artifact internally so a caller may recover
    its local lifecycle bookkeeping, but neither the CLI nor a persisted
    failure summary may expose an artifact, Source, TTIR, candidate text,
    Replay text, complete provider body, trajectory, or unconstrained
    diagnostic. A provider SSE error envelope is the sole diagnostic
    exception: it is copied recursively with credentials redacted and is
    never used as repair input.
    """

    if not isinstance(evidence, Mapping):
        return {}
    safe: dict[str, Any] = {}
    prior = evidence.get("prior_lift_rejections")
    if isinstance(prior, list):
        safe_prior: list[dict[str, Any]] = []
        for item in prior[:3]:
            if not isinstance(item, Mapping):
                continue
            call, rules = item.get("lift_call"), item.get("rules")
            if (
                not isinstance(call, int)
                or isinstance(call, bool)
                or not 1 <= call <= 3
                or not isinstance(rules, list)
            ):
                continue
            safe_rules: list[dict[str, str]] = []
            for rule in rules[:12]:
                if not isinstance(rule, Mapping):
                    continue
                code, path = rule.get("rule_code"), rule.get("path")
                if not isinstance(code, str) or re.fullmatch(r"[a-z0-9_]{1,96}", code) is None:
                    continue
                row = {"rule_code": code}
                if isinstance(path, str) and re.fullmatch(r"(?:j|pi)\.[A-Za-z0-9_.\[\]]{1,120}|<(?:candidate|canonical-constructor)>", path):
                    row["path"] = path
                safe_rules.append(row)
            if safe_rules:
                safe_prior.append({"lift_call": call, "rules": safe_rules})
        if safe_prior:
            safe["prior_lift_rejections"] = safe_prior
    for key in (
        "attempted_lift_call",
        "attempted_formal_round",
        "formal_gate1_rounds_consumed",
        "replay_normal_calls",
        "replay_masked_calls",
        "replay_physical_calls",
        "model_calls",
        "physical_provider_calls",
        "transport_retries",
    ):
        value = evidence.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            safe[key] = value
    candidate_present = evidence.get("candidate_present")
    if isinstance(candidate_present, bool):
        safe["candidate_present"] = candidate_present
    lifecycle = evidence.get("lifecycle")
    if isinstance(lifecycle, Mapping):
        safe_lifecycle: dict[str, Any] = {}
        for key in (
            "phase",
            "stop_reason",
        ):
            value = lifecycle.get(key)
            if isinstance(value, str) and _SAFE_STAGE_METRIC_STRING.fullmatch(value):
                safe_lifecycle[key] = value
        for key in (
            "attempt",
            "model_calls",
            "physical_provider_calls",
            "transport_retries",
        ):
            value = lifecycle.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                safe_lifecycle[key] = value
        for key in (
            "candidate_present",
            "candidate_check_started",
            "typecheck_started",
            "replay_started",
            "gate1_started",
            "gate2_started",
        ):
            value = lifecycle.get(key)
            if isinstance(value, bool):
                safe_lifecycle[key] = value
        if safe_lifecycle:
            safe["lifecycle"] = safe_lifecycle
    failure_class = evidence.get("failure_class")
    failure_stage = evidence.get("failure_stage")
    if isinstance(failure_stage, str) and re.fullmatch(
        r"[a-z][a-z0-9_]{0,63}", failure_stage
    ):
        safe["failure_stage"] = failure_stage
    for key in ("exception_type", "origin_exception_type", "returned_type"):
        value = evidence.get(key)
        if isinstance(value, str) and re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,63}", value
        ):
            safe[key] = value
    origin_frame = evidence.get("origin_frame")
    if isinstance(origin_frame, Mapping):
        safe_frame: dict[str, Any] = {}
        for key in ("file", "function"):
            value = origin_frame.get(key)
            if isinstance(value, str) and re.fullmatch(
                r"[A-Za-z0-9_.-]{1,96}", value
            ):
                safe_frame[key] = value
        line = origin_frame.get("line")
        if isinstance(line, int) and not isinstance(line, bool) and line > 0:
            safe_frame["line"] = line
        if safe_frame:
            safe["origin_frame"] = safe_frame
    if isinstance(failure_class, str) and (
        failure_class in {"source_preparation", "lift_provider_preparation"}
        or failure_class in _SAFE_LIFT_FAILURE_CLASSES
        or failure_class in _SAFE_PROVIDER_SELECTION_FAILURE_CLASSES
    ):
        safe["failure_class"] = failure_class
        # A pre-candidate MCP failure has no candidate or repair base, but its
        # host-owned preparation receipt is still useful diagnostic metadata.
        # Reuse the same bounded projection that feeds the next fresh owner;
        # never copy the authoring declaration, Source, provider response, or
        # trajectory into the durable failure summary.
        if failure_class in _SAFE_LIFT_FAILURE_CLASSES:
            raw_preparation = evidence.get("preparation_receipt")
            if not isinstance(raw_preparation, Mapping):
                authoring = evidence.get("authoring")
                preparation_failures = (
                    authoring.get("preparation_failures")
                    if isinstance(authoring, Mapping)
                    else None
                )
                if isinstance(preparation_failures, list) and preparation_failures:
                    raw_preparation = preparation_failures[-1]
            preparation = _safe_preparation_receipt(raw_preparation)
            if preparation is not None:
                safe["preparation_receipt"] = preparation
        if failure_class == "source_preparation":
            source_class = evidence.get("source_failure_class")
            if isinstance(source_class, str) and source_class in _SAFE_SOURCE_FAILURE_CLASSES:
                safe["source_failure_class"] = source_class
            source_reason = evidence.get("source_failure_reason")
            if (
                isinstance(source_reason, str)
                and source_reason in _SAFE_SOURCE_FAILURE_REASONS
            ):
                safe["source_failure_reason"] = source_reason
                stage = source_preparation_failure_stage(source_reason)
                if stage is not None:
                    safe["source_failure_stage"] = stage
        elif failure_class == "lift_provider_preparation":
            provider_class = evidence.get("provider_failure_class")
            if (
                isinstance(provider_class, str)
                and provider_class in _SAFE_PROVIDER_FAILURE_CLASSES
            ):
                safe["provider_failure_class"] = provider_class
            exception_type = evidence.get("provider_exception_type")
            if isinstance(exception_type, str) and re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_]{0,63}", exception_type
            ):
                safe["provider_exception_type"] = exception_type
            provider_origin = evidence.get("provider_failure_origin")
            if (
                isinstance(provider_origin, str)
                and provider_origin in _SAFE_PROVIDER_FAILURE_ORIGINS
            ):
                safe["provider_failure_origin"] = provider_origin
            provider_error = _safe_external_error_envelope(
                evidence.get("provider_error")
            )
            if provider_error is not None:
                # This is the bounded external SSE error envelope after
                # credential redaction. It is diagnostic-only and is never
                # fed back into Lift, CandidateCheck, Replay, or repair.
                safe["provider_error"] = provider_error
        elif failure_class in _SAFE_PROVIDER_SELECTION_FAILURE_CLASSES:
            safe["provider_failure_class"] = failure_class
    for failure_key in (
        "lift_generation_failures",
        "replay_generation_failures",
    ):
        raw_failures = evidence.get(failure_key)
        if not isinstance(raw_failures, list):
            continue
        failures = []
        for raw in raw_failures:
            if not isinstance(raw, Mapping):
                continue
            projected = {
                key: value
                for key, value in raw.items()
                if key in _SAFE_FAILURE_INGRESS_FIELDS
                and (
                    isinstance(value, str)
                    or (isinstance(value, int) and not isinstance(value, bool))
                )
            }
            if projected:
                failures.append(projected)
        if failures:
            safe[failure_key] = failures
    metrics = _safe_stage_metrics(evidence.get("stage_metrics"))
    if metrics:
        safe["stage_metrics"] = metrics
    gate1 = evidence.get("gate1")
    if isinstance(gate1, Mapping):
        safe_gate1: dict[str, Any] = {}
        status = gate1.get("status")
        if isinstance(status, str):
            safe_gate1["status"] = status
        llm_calls = gate1.get("llm_calls")
        if isinstance(llm_calls, Mapping):
            calls = {
                key: value
                for key, value in llm_calls.items()
                if key in {"normal", "masked_j"}
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            }
            if calls:
                safe_gate1["llm_calls"] = calls
        for arm_name in ("normal", "masked_j"):
            safe_arm = _safe_gate1_arm_failure(gate1.get(arm_name))
            if safe_arm is not None:
                safe_gate1[arm_name] = safe_arm
        if safe_gate1:
            safe["gate1"] = safe_gate1
    return safe


def safe_workload_failure_diagnostic(error: BaseException) -> dict[str, str]:
    """Return a public failure code without serializing exception text."""

    cause = error.cause if isinstance(error, WorkloadExecutionError) else error
    raw_evidence = (
        error.evidence
        if isinstance(error, WorkloadExecutionError)
        else getattr(cause, "evidence", None)
    )
    safe_evidence = safe_workload_failure_metadata(raw_evidence)
    evidence_failure = safe_evidence.get("failure_class")
    if isinstance(evidence_failure, str):
        rule = evidence_failure
    elif getattr(cause, "failure_class", None) in _SAFE_PROVIDER_SELECTION_FAILURE_CLASSES:
        rule = str(getattr(cause, "failure_class"))
    elif isinstance(cause, ProviderError):
        provider_class = getattr(cause, "failure_class", None)
        rule = (
            provider_class
            if isinstance(provider_class, str)
            and provider_class in _SAFE_PROVIDER_FAILURE_CLASSES
            else "provider_failure"
        )
    else:
        message = str(cause)
        rule = (
            message
            if isinstance(cause, ValueError)
            and message in _SAFE_HOST_PREPARATION_RULES
            else "workload_infrastructure_failure"
        )
    return {
        "rule_code": rule,
        "cause_class": type(cause).__name__,
    }


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
    partial_evidence = safe_workload_failure_metadata(
        runtime.evidence if runtime is not None else None
    )
    if evidence is not None:
        partial_evidence.update(safe_workload_failure_metadata(evidence))
    if runtime is not None and runtime.stage_metrics:
        runtime_metrics = _safe_stage_metrics(runtime.stage_metrics)
        if runtime_metrics:
            partial_evidence["stage_metrics"] = runtime_metrics
    diagnostic = safe_workload_failure_diagnostic(cause)
    _write_json(
        run_dir / "failure.json",
        {
            "failure_class": diagnostic["rule_code"],
            "exception_class": type(cause).__name__,
            **diagnostic,
            "partial_evidence": partial_evidence,
        },
    )


__all__ = [
    "LIFT_SEMANTIC_PROBE_PROTOCOL",
    "LIFT_SEMANTIC_PROBE_SEED",
    "WorkloadExecution",
    "WorkloadExecutionError",
    "WorkloadGate1Search",
    "build_workload_gate2_suite",
    "build_workload_jc_suite",
    "classify_scrc_report",
    "complete_prepared_workload_gate2",
    "execute_prepared_workload",
    "initialize_workload_evidence",
    "next_candidate_fanout_dir",
    "next_workload_run_dir",
    "run_workload_lift_semantic_probe",
    "safe_workload_failure_diagnostic",
    "safe_workload_failure_metadata",
    "save_candidate_fanout_report",
    "save_workload_execution",
    "save_workload_failure",
    "save_workload_gate1_search",
    "scrc_gate1_search_diagnostic",
    "scrc_attempt_stage_counts",
    "scrc_report_diagnostic",
    "search_prepared_workload_gate1",
]
