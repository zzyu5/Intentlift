#!/usr/bin/env python3
"""Run one real SCRC Triton canary through the active unified gate core."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import numpy as np

from intentir.extraction.lifters.scrc import (
    ResponsesSCRCLiftAdapter,
    SCRCCompilerArgument,
    SCRCCompilerView,
    anonymize_source,
)
from intentir.extraction.providers.provider import ProviderConfig, get_provider
from intentir.verify.adapters import (
    BoundGate1Adapter,
    TritonGate2ExecutionAdapter,
)
from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.model import (
    Gate1Artifact,
    ParameterKind,
    ParameterMode,
)
from intentir.verify.gate1.orchestrate import Gate1Config, Gate1Runner
from intentir.verify.gate1.replay import ResponsesReplayWriter
from intentir.verify.gate2.orchestrate import (
    Gate2Case,
    Gate2Runner,
    Gate2Suite,
)
from intentir.verify.gate2.scalar import ScalarLoweringSpec
from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
)
from intentir.verify.scrc_runner import (
    HarnessBindingInstance,
    MainStatus,
    SCRCGateRunner,
    SCRCRunDependencies,
    SCRCRunReport,
    SCRCRunRequest,
    SourceProgram,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = Path("/tmp/intentir-scrc-task11/canary")


EXACT_SOURCE = """import triton
import triton.language as tl

@triton.jit
def source_exact_vector_add(X, Y, Out, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    left = tl.load(X + offsets)
    right = tl.load(Y + offsets)
    tl.store(Out + offsets, left + right)
"""


MASKED_SOURCE = """import triton
import triton.language as tl

@triton.jit
def source_masked_copy(X, Y, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(X + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, value, mask=mask)
"""


REDUCTION_SOURCE = """import triton
import triton.language as tl

@triton.jit
def source_masked_row_square_sum(X, Y, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(X + row * N + offsets, mask=mask, other=0.0)
    squared = value * value
    total = tl.sum(squared, axis=0)
    tl.store(Y + row, total)
"""


@dataclass(frozen=True)
class CanarySpec:
    name: str
    source: str
    source_function: str
    compiler_view: SCRCCompilerView
    input_factory: Callable[[], tuple[np.ndarray, ...]]

    @property
    def grid(self) -> tuple[int, ...]:
        return self.compiler_view.grid


def _exact_inputs() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(7)
    return (
        rng.normal(size=128).astype(np.float32),
        rng.normal(size=128).astype(np.float32),
    )


def _masked_inputs() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(11)
    return (rng.normal(size=1000).astype(np.float32),)


def _reduction_inputs() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(19)
    return (rng.normal(size=(2, 1000)).astype(np.float32),)


CANARIES: dict[str, CanarySpec] = {
    "exact_vector_add": CanarySpec(
        "exact_vector_add",
        EXACT_SOURCE,
        "source_exact_vector_add",
        SCRCCompilerView(
            schedule_values={"BLOCK_SIZE": 128},
            arguments=(
                SCRCCompilerArgument("X", "tensor", "f32", (128,)),
                SCRCCompilerArgument("Y", "tensor", "f32", (128,)),
                SCRCCompilerArgument("Out", "tensor", "f32", (128,)),
            ),
            grid=(1,),
            grid_contract=("1",),
            parameter_order=("X", "Y", "Out", "BLOCK_SIZE"),
            constexpr_parameters=("BLOCK_SIZE",),
        ),
        _exact_inputs,
    ),
    "masked_copy": CanarySpec(
        "masked_copy",
        MASKED_SOURCE,
        "source_masked_copy",
        SCRCCompilerView(
            schedule_values={"BLOCK_SIZE": 1024},
            arguments=(
                SCRCCompilerArgument("X", "tensor", "f32", (1000,)),
                SCRCCompilerArgument("Y", "tensor", "f32", (1000,)),
                SCRCCompilerArgument("N", "scalar", "i32", value=1000),
            ),
            grid=(1,),
            grid_contract=("1",),
            parameter_order=("X", "Y", "N", "BLOCK_SIZE"),
            constexpr_parameters=("BLOCK_SIZE",),
        ),
        _masked_inputs,
    ),
    "masked_row_square_sum": CanarySpec(
        "masked_row_square_sum",
        REDUCTION_SOURCE,
        "source_masked_row_square_sum",
        SCRCCompilerView(
            schedule_values={"BLOCK_SIZE": 1024},
            arguments=(
                SCRCCompilerArgument("X", "tensor", "f32", (2, 1000)),
                SCRCCompilerArgument("Y", "tensor", "f32", (2,)),
                SCRCCompilerArgument("N", "scalar", "i32", value=1000),
            ),
            grid=(2,),
            grid_contract=("2",),
            parameter_order=("X", "Y", "N", "BLOCK_SIZE"),
            constexpr_parameters=("BLOCK_SIZE",),
        ),
        _reduction_inputs,
    ),
}


def _runtime_argument(argument: SCRCCompilerArgument) -> RuntimeArgument:
    runtime_dtype = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }.get(argument.dtype, argument.dtype)
    if argument.kind == "tensor":
        return RuntimeArgument.tensor(argument.shape, runtime_dtype)
    if argument.kind == "scalar":
        assert argument.value is not None
        return RuntimeArgument.scalar(argument.value, runtime_dtype)
    if argument.kind == "none":
        return RuntimeArgument.none(runtime_dtype)
    raise ValueError("Triton canaries do not support opaque runtime arguments")


def prepare_request(spec: CanarySpec) -> tuple[SCRCRunRequest, dict[str, str]]:
    anonymous, _source_binding, mapping = anonymize_source(
        spec.source,
        source_symbol=spec.source_function,
    )
    arguments = {
        mapping[item.name]: _runtime_argument(item)
        for item in spec.compiler_view.arguments
    }
    schedule = {
        mapping[name]: value
        for name, value in spec.compiler_view.schedule_values.items()
    }
    c0 = {
        mapping[name]: value
        for name, value in spec.compiler_view.schedule_values.items()
    }
    gate_source = KernelSource.from_code(anonymous, "scrc_k0", trusted=True)
    binding = HarnessBindingInstance.create(
        c0,
        gate1_case=CompileCase(arguments, spec.grid, schedule),
        gate2_context={"test_mode": "finite", "seed": 7},
    )
    return (
        SCRCRunRequest(
            SourceProgram(spec.source, gate_source),
            binding,
            "triton",
        ),
        mapping,
    )


def _shape(
    declared: list[int | str],
    dimensions: Mapping[str, int],
) -> tuple[int, ...]:
    result: list[int] = []
    for extent in declared:
        if isinstance(extent, int):
            result.append(extent)
        elif extent in dimensions:
            result.append(dimensions[extent])
        else:
            raise ValueError(f"Gate2 cannot resolve symbolic extent {extent!r}")
    return tuple(result)


def _build_gate2_suite(
    spec: CanarySpec,
    artifact: Gate1Artifact,
    binding: HarnessBindingInstance,
) -> Gate2Suite:
    concrete_inputs = spec.input_factory()
    logical_inputs = artifact.j.intent.inputs
    if len(logical_inputs) != len(concrete_inputs):
        raise ValueError(
            "J input arity differs from the canary's logical input contract"
        )

    dimensions: dict[str, int] = {}
    values: dict[str, Any] = {}
    input_bindings: dict[str, str] = {}
    for index, (tensor, value) in enumerate(
        zip(logical_inputs, concrete_inputs, strict=True)
    ):
        if value.ndim != len(tensor.shape):
            raise ValueError("J input rank differs from the canary input")
        key = f"input{index}"
        values[key] = value
        input_bindings[tensor.name] = key
        for declared, actual in zip(tensor.shape, value.shape, strict=True):
            if isinstance(declared, int) and declared != actual:
                raise ValueError("J static input extent differs from the canary")
            if isinstance(declared, str):
                previous = dimensions.setdefault(declared, int(actual))
                if previous != actual:
                    raise ValueError("J symbolic dimensions are inconsistent")

    output_bindings = {
        tensor.name: f"output{index}"
        for index, tensor in enumerate(artifact.j.intent.outputs)
    }
    if not output_bindings:
        raise ValueError("Gate2 canary requires at least one logical output")

    input_names = set(input_bindings)
    output_names = set(output_bindings)
    arguments: list[TritonArgument] = []
    for parameter in artifact.j.signature.parameters:
        ref = parameter.logical_ref
        if parameter.kind is ParameterKind.CONSTEXPR:
            if ref not in dimensions:
                raise ValueError("semantic constexpr is not bound by a J dimension")
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.SCALAR,
                    value=int(dimensions[ref]),
                )
            )
            continue
        if parameter.kind is ParameterKind.SCALAR:
            if ref not in dimensions:
                raise ValueError("runtime scalar is not bound by a J dimension")
            key = f"scalar_{parameter.name}"
            values[key] = int(dimensions[ref])
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.SCALAR,
                    value_key=key,
                )
            )
            continue
        if parameter.kind not in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            raise ValueError("Gate2 canary does not support opaque J parameters")
        if ref in input_names and ref in output_names:
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.INOUT,
                    value_key=input_bindings[ref],
                    dtype=parameter.dtype,
                    output_key=output_bindings[ref],
                )
            )
        elif ref in input_names:
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.INPUT,
                    value_key=input_bindings[ref],
                    dtype=parameter.dtype,
                )
            )
        elif ref in output_names or parameter.mode is ParameterMode.OUTPUT:
            if ref not in output_names:
                raise ValueError("J output signature ref is not a logical output")
            tensor = next(item for item in artifact.j.intent.outputs if item.name == ref)
            arguments.append(
                TritonArgument(
                    parameter.name,
                    ArgumentRole.OUTPUT,
                    dtype=parameter.dtype,
                    shape=_shape(tensor.shape, dimensions),
                    output_key=output_bindings[ref],
                )
            )
        else:
            raise ValueError(
                f"J signature parameter {parameter.name!r} has no Gate2 binding"
            )

    scalar = ScalarLoweringSpec.create(
        artifact.j,
        dims=dimensions,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    execution = TritonExecutionSpec(
        tuple(arguments),
        spec.grid,
        artifact.c.values,
    )
    case = Gate2Case(
        "finite-seed-7",
        scalar.j,
        values,
        execution,
        scalar,
        tuple(output_bindings.values()),
        {"test_mode": "finite", "seed": 7},
    )
    return Gate2Suite(spec.name, (case,), binding.c_values)


def _next_run_dir(spec: CanarySpec) -> Path:
    parent = TASK_ROOT / "runs" / spec.name
    parent.mkdir(parents=True, exist_ok=True)
    indexes = [
        int(path.name.removeprefix("run-"))
        for path in parent.glob("run-[0-9][0-9][0-9]")
        if path.name.removeprefix("run-").isdigit()
    ]
    path = parent / f"run-{(max(indexes, default=0) + 1):03d}"
    path.mkdir()
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _save_rejected_lift_record(
    directory: Path,
    record: Any,
    *,
    diagnostic_filename: str,
) -> None:
    (directory / diagnostic_filename).write_text(
        (record.diagnostic or "unknown Lift rejection") + "\n",
        encoding="utf-8",
    )
    for index, attempt in enumerate(record.rejected_attempts, start=1):
        rejection_dir = directory / f"rejection-{index:02d}"
        rejection_dir.mkdir(exist_ok=True)
        _write_json(
            rejection_dir / "receipt.json",
            {
                "model": attempt.model,
                "response_model": attempt.response_model,
                "response_id": attempt.response_id,
                "usage": dict(attempt.usage),
                "attempts": attempt.attempts,
                "elapsed_ms": attempt.elapsed_ms,
                "request_sha256": attempt.request_sha256,
                "response_sha256": attempt.response_sha256,
                "raw_response_sha256": attempt.raw_response_sha256,
            },
        )
        (rejection_dir / "response.txt").write_text(
            attempt.response_text,
            encoding="utf-8",
        )


def _save_lift_records_on_failure(
    run_dir: Path,
    lift: ResponsesSCRCLiftAdapter,
) -> None:
    for index, record in enumerate(lift.records, start=1):
        call_dir = run_dir / f"lift-call-{index:02d}"
        call_dir.mkdir(exist_ok=True)
        if record.extraction is None:
            _save_rejected_lift_record(
                call_dir,
                record,
                diagnostic_filename="rejection.txt",
            )
            continue
        extraction = record.extraction
        _write_json(call_dir / "artifact.json", extraction.artifact.to_json())
        _write_json(
            call_dir / "layer_c_interface.json",
            extraction.layer_c_interface_payload,
        )
        provenance = extraction.provenance
        _write_json(
            call_dir / "receipt.json",
            {
                "model": provenance.model,
                "response_model": provenance.response_model,
                "response_id": provenance.response_id,
                "usage": dict(provenance.usage),
                "attempts": provenance.attempts,
                "elapsed_ms": provenance.elapsed_ms,
            },
        )
        (call_dir / "response.txt").write_text(
            provenance.response_text,
            encoding="utf-8",
        )


def _save_report(
    run_dir: Path,
    spec: CanarySpec,
    provider: ProviderConfig,
    report: SCRCRunReport,
    lift: ResponsesSCRCLiftAdapter,
) -> None:
    payload = report.to_json()
    _write_json(run_dir / "report.json", payload)
    for index, attempt in enumerate(report.attempts, start=1):
        attempt_dir = run_dir / f"attempt-{index:02d}"
        attempt_dir.mkdir()
        if attempt.artifact is not None:
            _write_json(attempt_dir / "artifact.json", attempt.artifact.to_json())
        interface = attempt.lift_metadata.get("layer_c_interface")
        if isinstance(interface, Mapping):
            _write_json(attempt_dir / "layer_c_interface.json", interface)
        if attempt.gate1 is not None:
            _write_json(attempt_dir / "gate1.json", attempt.gate1.to_json())
            if attempt.gate1.normal.generation is not None:
                (attempt_dir / "replay.py").write_text(
                    attempt.gate1.normal.generation.code,
                    encoding="utf-8",
                )
            if (
                attempt.gate1.masked_j is not None
                and attempt.gate1.masked_j.generation is not None
            ):
                (attempt_dir / "masked_j_replay.py").write_text(
                    attempt.gate1.masked_j.generation.code,
                    encoding="utf-8",
                )
        if index <= len(lift.records):
            record = lift.records[index - 1]
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
                    },
                )
                (attempt_dir / "lift_response.txt").write_text(
                    provenance.response_text,
                    encoding="utf-8",
                )
            elif record.diagnostic is not None:
                _save_rejected_lift_record(
                    attempt_dir,
                    record,
                    diagnostic_filename="lift_rejection.txt",
                )
    if report.gate2 is not None:
        _write_json(run_dir / "gate2.json", report.gate2.to_json())
    summary = [
        f"# {spec.name} {run_dir.name}",
        "",
        f"- provider: `{provider.name}` / `{provider.model}`",
        f"- main_status: `{report.main_status.value}`",
        f"- accepted_attempt: `{report.accepted_attempt}`",
        f"- j_necessity_status: `{report.j_necessity_status.value if report.j_necessity_status else 'NOT_RUN'}`",
        f"- call_budget: `{json.dumps(report.call_budget.to_json(), sort_keys=True)}`",
    ]
    if report.gate2 is not None:
        summary.append(
            "- gate2_status_counts: `"
            + json.dumps(report.gate2.status_counts, sort_keys=True)
            + "`"
        )
    summary.extend(
        [
            "",
            "本目录中的 receipt/digest 仅用于普通追溯，不参与候选合法性或认证。",
            "",
        ]
    )
    (run_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def run_canary(
    spec: CanarySpec,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    run_masked_j: bool = True,
) -> tuple[Path, SCRCRunReport]:
    run_dir = _next_run_dir(spec)
    request, mapping = prepare_request(spec)
    (run_dir / "source.py").write_text(spec.source, encoding="utf-8")
    _write_json(
        run_dir / "request.json",
        {
            "canary": spec.name,
            "provider": provider.name,
            "model": provider.model,
            "reasoning_effort": effort or provider.reasoning_effort,
            "frontend": "triton",
            "test_mode": "finite",
            "c0": request.binding.values,
            "grid": list(spec.grid),
            "run_masked_j": run_masked_j,
            "source_to_opaque": mapping,
        },
    )

    lift = ResponsesSCRCLiftAdapter(
        provider=provider,
        compiler_view=spec.compiler_view,
        effort=effort,
        source_symbol=spec.source_function,
    )
    writers: list[ResponsesReplayWriter] = []
    compiler = TritonCompiler(timeout=240.0)

    def gate1_factory(_formal_round: int) -> BoundGate1Adapter:
        writer = ResponsesReplayWriter(
            provider=provider,
            effort=effort,
        )
        writers.append(writer)
        return BoundGate1Adapter(
            Gate1Runner(
                compiler=compiler,
                writer=writer,
                config=Gate1Config(run_masked_j=run_masked_j),
            )
        )

    gate2 = TritonGate2ExecutionAdapter(
        Gate2Runner(),
        lambda artifact, _gate1, binding: _build_gate2_suite(
            spec, artifact, binding
        ),
    )
    runner = SCRCGateRunner(
        SCRCRunDependencies(lift, gate1_factory, gate2)
    )
    try:
        report = runner.run(request)
    except Exception as exc:
        _save_lift_records_on_failure(run_dir, lift)
        evidence = getattr(exc, "evidence", None)
        if isinstance(evidence, Mapping) and evidence:
            _write_json(run_dir / "partial_evidence.json", dict(evidence))
        _write_json(
            run_dir / "failure.json",
            {
                "failure_class": type(exc).__name__,
                "diagnostic": str(exc),
                "lift_records": len(lift.records),
                "replay_writers": len(writers),
            },
        )
        (run_dir / "SUMMARY.md").write_text(
            f"# {spec.name} {run_dir.name}\n\n"
            f"- terminal: `INFRASTRUCTURE_FAILED`\n"
            f"- exception: `{type(exc).__name__}: {exc}`\n",
            encoding="utf-8",
        )
        raise
    _save_report(run_dir, spec, provider, report, lift)
    return run_dir, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary", choices=tuple(CANARIES), required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument(
        "--skip-masked-j",
        action="store_true",
        help="skip the independent masked-J diagnostic",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    provider = get_provider(args.provider)
    try:
        run_dir, report = run_canary(
            CANARIES[args.canary],
            provider=provider,
            effort=args.effort,
            run_masked_j=not args.skip_masked_j,
        )
    except Exception as exc:
        print(f"SCRC canary infrastructure failure: {type(exc).__name__}: {exc}")
        return 3
    print(f"SCRC canary run: {run_dir}")
    print(f"main_status: {report.main_status.value}")
    return 0 if report.main_status is MainStatus.SCRC_CERTIFIED else 2


if __name__ == "__main__":
    sys.exit(main())
