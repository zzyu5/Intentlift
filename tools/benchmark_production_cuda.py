#!/usr/bin/env python3
"""Lower, execute, search, and benchmark accepted J+Source-prior snapshots.

The measured implementation is always a cubin emitted from the snapshot's
materialized Intent MLIR through a registered C++ MLIR pass.  Python owns only
snapshot transport, candidate orchestration, CUDA Driver launch binding, and
event timing.  The comparison baseline is the exact declarative Triton workload
adapter already used by SCRC, not a newly authored substitute kernel.

The snapshot's compatibility ``ImplementationConfig`` is used only as an
erasable search-order prior in this legacy benchmark tool.  It is not a
``TargetLayerCPlugin`` and does not cross the public production Lower boundary.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.cuda.benchmark import (
    PreparedGraphLaunch,
    PreparedSingleLaunch,
    do_bench_cudagraph_us,
    do_bench_us,
)
from intentir.backend.cuda.search import (
    canonical_cuda_candidate,
    compile_cuda_target_plugin,
    search_cuda,
)
from intentir.backend.cuda.qd import (
    CudaOpaqueExecution,
    build_seeded_execution_domain,
)
from intentir.backend.cuda.production_cases import (
    PRODUCTION_CASE_IDS,
    PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    constexpr_input_bindings,
    prepare_production_workload,
    profile_production_snapshot,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.backend.ablation import SearchAxes
from intentir.backend.search_and_close import SearchAndCloseController
from intentir.ir.production_snapshot import (
    ProductionSnapshot,
    load_production_snapshot,
)
from intentir.ir.canonical_j import j_elaboration
from intentir.verify.gate2.compare import compare_outputs
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.gate2.triton import PreparedTritonInvocation
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.scrc_workload_run import build_workload_jc_suite


REPORT_PROTOCOL = "intentir.production.cuda-performance.v1"

CASE_IDS = PRODUCTION_CASE_IDS


def _torch_dtype(torch: Any, dtype: str) -> Any:
    aliases = {
        "bool": "bool",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
    }
    return getattr(torch, aliases[dtype.removeprefix("torch.").lower()])


def _shape(shape: list[int | str], dims: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(dims[item]) if isinstance(item, str) else item for item in shape)


def _device_record(torch: Any) -> dict[str, Any]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_index": index,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "target_chip": backend_compile.CHIP,
    }


def _tensor_inputs(
    torch: Any,
    snapshot: ProductionSnapshot,
    case: Any,
    runtime_input_names: tuple[str, ...],
) -> list[Any]:
    bindings = case.refeval.input_binding_map
    tensors = {item.name: item for item in snapshot.j.intent.inputs}
    result = []
    for name in runtime_input_names:
        tensor = tensors[name]
        raw = np.ascontiguousarray(np.asarray(case.values[bindings[tensor.name]]))
        declared_shape = _shape(tensor.shape, case.refeval.dims_map)
        raw = raw.reshape(declared_shape if declared_shape else ())
        result.append(
            torch.as_tensor(raw, dtype=_torch_dtype(torch, tensor.dtype), device="cuda")
        )
    return result


def _tensor_outputs(
    torch: Any,
    snapshot: ProductionSnapshot,
    dims: Mapping[str, int],
) -> list[Any]:
    output_bindings = j_elaboration(snapshot.j).output_bindings
    if not output_bindings:
        raise ValueError("accepted J has no canonical output bindings")
    return [
        torch.empty(
            _shape(binding.output_type.shape, dims),
            dtype=_torch_dtype(torch, binding.output_type.dtype),
            device="cuda",
        )
        for binding in output_bindings
    ]


def _one_kernel(
    *,
    name: str,
    snapshot_dir: Path,
    budget: int,
    warmup_ms: int,
    repeat_ms: int,
    timing: str,
    graph_repetitions: int,
    measurement_rounds: int,
    search_prior: str,
    b_search_mode: str,
    axes: SearchAxes | None = None,
    fixed_lowering_path_kind: str | None = None,
    qd_search_seeds: tuple[int, ...] = (),
    qd_cert_seeds: tuple[int, ...] = (),
    replay_program: KernelSource | None = None,
    require_backend_closure: bool = False,
    workload_profile: str = PRODUCTION_WORKLOAD_PROFILE_CLOSED,
) -> dict[str, Any]:
    import torch

    base_snapshot, snapshot_path = load_production_snapshot(snapshot_dir, name)
    prepared_workload = prepare_production_workload(
        name,
        profile=workload_profile,
    )
    snapshot, workload_valuation = profile_production_snapshot(
        base_snapshot,
        prepared_workload,
        workload_profile,
    )
    if snapshot.c is None and (axes is None or axes.baseline_kind is None):
        raise ValueError("performance search requires the optional C prior")
    suite = build_workload_jc_suite(prepared_workload, snapshot.j, snapshot.c)
    case = suite.cases[0]
    dims = case.refeval.dims_map
    constexpr_bindings = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    materialized = materialize_j(
        snapshot.j,
        dims,
        constexpr_bindings=constexpr_bindings,
    )
    target = BackendTarget("cuda", backend_compile.CHIP)
    state = attach_c(
        materialized,
        snapshot.c
        if (
            axes.use_c_prior
            if axes is not None
            else search_prior == "with-c"
        )
        else None,
        target,
    )
    neutral_state = attach_c(materialized, None, target)
    qd_controller = None
    if qd_search_seeds or qd_cert_seeds:
        if not qd_search_seeds or not qd_cert_seeds:
            raise ValueError("Q/D search and certification seeds are both required")
        d_search = build_seeded_execution_domain(
            prepared_workload,
            snapshot.j,
            snapshot.c,
            domain_id=f"{name}.D_search",
            seeds=qd_search_seeds,
        )
        d_cert = build_seeded_execution_domain(
            prepared_workload,
            snapshot.j,
            snapshot.c,
            domain_id=f"{name}.D_cert",
            seeds=qd_cert_seeds,
        )
        opaque = CudaOpaqueExecution(
            j=snapshot.j,
            materialized=materialized,
            source=prepared_workload.request.source.gate_source,
            replay=replay_program,
        )
        qd_controller = SearchAndCloseController(
            j=snapshot.j,
            target=state.target.to_dict(),
            policy=DEFAULT_GATE2_POLICY,
            d_search=d_search,
            d_cert=d_cert,
            runners=opaque.runners(),
        )
    inputs = _tensor_inputs(
        torch,
        snapshot,
        case,
        materialized.runtime_input_names,
    )
    outputs = _tensor_outputs(torch, snapshot, dims)
    tensors = [*inputs, *outputs]

    launch_records: dict[str, Any] = {}
    prepared_graphs: dict[str, PreparedGraphLaunch] = {}

    def measure(thunk) -> float:
        samples = []
        for _ in range(measurement_rounds):
            torch.cuda.synchronize()
            if timing == "graph":
                samples.append(
                    do_bench_cudagraph_us(
                        thunk, repetitions=graph_repetitions
                    )
                )
            else:
                samples.append(
                    do_bench_us(
                        thunk,
                        warmup_ms=warmup_ms,
                        repeat_ms=repeat_ms,
                    )
                )
            torch.cuda.synchronize()
        return float(statistics.median(samples))

    reference = ScalarLoweringRunner().run(case.refeval, case.values)
    if not reference.ok:
        raise RuntimeError(f"canonical J reference failed: {reference.error}")

    def measure_source() -> tuple[Any, float, list[float]]:
        source_program = prepared_workload.request.source.gate_source
        with PreparedTritonInvocation(
            source_program, case.triton, case.values
        ) as source:
            source.launch()
            torch.cuda.synchronize()
            source_actual = source.copied_outputs()
            source_j = compare_outputs(
                "source-triton",
                source_actual,
                "canonical-J",
                reference.outputs,
                j=snapshot.j,
                input_bindings=case.refeval.input_binding_map,
                output_bindings=case.refeval.output_binding_map,
                inputs=case.values,
                policy=DEFAULT_GATE2_POLICY,
            )
            if not source_j.passed:
                raise RuntimeError("source Triton result differs from canonical J")
            # Keep the standalone Source diagnostic at the same outer-batch
            # count as paired target confirmation.  Formal Source/IntentIR
            # ratios still come from the target arm's paired-alternating
            # same-run record below; E0 is an absolute Source diagnostic only.
            samples = [measure(source.thunk()) for _ in range(4)]
        return source_j, float(statistics.median(samples)), samples

    def confirm_target_against_source(
        target_thunk,
    ) -> tuple[Any, float, float, list[float], list[float], list[dict[str, Any]]]:
        """Confirm performance with fresh, equal-count, alternating pairs."""

        source_program = prepared_workload.request.source.gate_source
        with PreparedTritonInvocation(
            source_program, case.triton, case.values
        ) as source:
            source.launch()
            torch.cuda.synchronize()
            source_actual = source.copied_outputs()
            source_j = compare_outputs(
                "source-triton",
                source_actual,
                "canonical-J",
                reference.outputs,
                j=snapshot.j,
                input_bindings=case.refeval.input_binding_map,
                output_bindings=case.refeval.output_binding_map,
                inputs=case.values,
                policy=DEFAULT_GATE2_POLICY,
            )
            if not source_j.passed:
                raise RuntimeError("source Triton result differs from canonical J")

            source_thunk = source.thunk()
            # One untimed pair establishes both graphs and a common thermal state.
            source_thunk()
            target_thunk()
            torch.cuda.synchronize()

            target_samples: list[float] = []
            source_samples: list[float] = []
            paired_batches: list[dict[str, Any]] = []
            for pair_index in range(4):
                ordered = (
                    (("source", source_thunk), ("target", target_thunk))
                    if pair_index % 2 == 0
                    else (("target", target_thunk), ("source", source_thunk))
                )
                pair: dict[str, float] = {}
                for label, thunk in ordered:
                    pair[label] = measure(thunk)
                source_us = pair["source"]
                target_us = pair["target"]
                source_samples.append(source_us)
                target_samples.append(target_us)
                paired_batches.append(
                    {
                        "pair_index": pair_index,
                        "order": [label for label, _thunk in ordered],
                        "source_us": source_us,
                        "target_us": target_us,
                        "speedup_source_over_target": source_us / target_us,
                    }
                )

        return (
            source_j,
            float(statistics.median(target_samples)),
            float(statistics.median(source_samples)),
            target_samples,
            source_samples,
            paired_batches,
        )

    def common_record() -> dict[str, Any]:
        return {
            "kernel": name,
            "case_id": CASE_IDS[name],
            "workload_profile": workload_profile,
            "workload_valuation": workload_valuation,
            "snapshot": {
                "path": str(snapshot_path),
            },
            "extents": dims,
            "materialization": materialized.to_record(),
            "search_prior": {
                "mode": (
                    "with-c" if axes is not None and axes.use_c_prior
                    else "without-c" if axes is not None
                    else search_prior
                ),
                "c_attached_to_backend_search": (
                    axes.use_c_prior
                    if axes is not None
                    else search_prior == "with-c"
                ),
                "accepted_c_used_only_to_reconstruct_exact_workload": True,
            },
            "b_search_mode": b_search_mode,
            "ablation_axes": axes.to_dict() if axes is not None else None,
            "execution_domain_closure_required": require_backend_closure,
        }

    if axes is not None and axes.baseline_kind == "source":
        source_j, source_latency, source_samples = measure_source()
        return {
            **common_record(),
            "baseline": {
                "kind": "source",
                "candidate_budget": 0,
                "compiler_search_performed": False,
            },
            "correctness": {
                "source_j": source_j.to_json(),
                "policy": DEFAULT_GATE2_POLICY.policy_id,
            },
            "performance": {
                "selected_us": source_latency,
                "mlir_us": None,
                "source_triton_us": source_latency,
                "mlir_over_triton": None,
                "speedup_over_triton": 1.0,
                "meets_or_beats_triton": True,
                "warmup_ms": warmup_ms,
                "repeat_ms": repeat_ms,
                "timing": timing,
                "graph_repetitions": graph_repetitions,
                "measurement_rounds": measurement_rounds,
                "selection_confirmation": {
                    "source_batch_us": source_samples,
                    "batch_aggregation": "median",
                    "samples_per_batch": measurement_rounds,
                    "batch_count": len(source_samples),
                    "comparison_protocol": "standalone-source-diagnostic.v1",
                },
            },
        }

    if axes is not None and axes.baseline_kind == "canonical":
        candidate = canonical_cuda_candidate(neutral_state)
        program = compile_cuda_target_plugin(neutral_state, candidate)
        concrete_state = candidate.concrete_state(program)
        qd_search = (
            qd_controller.filter_candidate(program, concrete_state)
            if qd_controller is not None
            else None
        )
        if qd_search is not None and not qd_search.passed:
            raise RuntimeError("canonical target failed target-to-RefJ on D_search")
        prepared = PreparedSingleLaunch(program, materialized.intent_mlir)
        target_thunk = prepared.thunk(tensors)
        target_thunk()
        torch.cuda.synchronize()
        actual = {
            runtime_key: tensor.detach().cpu().numpy().copy()
            for runtime_key, tensor in zip(case.output_keys, outputs, strict=True)
        }
        target_j = compare_outputs(
            "canonical-mlir-cubin",
            actual,
            "canonical-J",
            reference.outputs,
            j=snapshot.j,
            input_bindings=case.refeval.input_binding_map,
            output_bindings=case.refeval.output_binding_map,
            inputs=case.values,
            policy=DEFAULT_GATE2_POLICY,
        )
        if not target_j.passed:
            raise RuntimeError("canonical compiled MLIR differs from canonical J")
        closure = None
        closure_error = None
        if qd_controller is not None:
            try:
                closure = qd_controller.close_winner(program, concrete_state)
                if not closure.passed:
                    closure_error = "canonical target failed a held-out direct edge"
            except Exception as error:
                closure_error = f"{type(error).__name__}: {error}"
        if require_backend_closure and (closure is None or not closure.passed):
            raise RuntimeError(
                "canonical target lacks formal backend closure: "
                + (closure_error or "no closure result")
            )
        (
            source_j,
            target_latency,
            source_latency,
            target_samples,
            source_samples,
            paired_batches,
        ) = confirm_target_against_source(target_thunk)
        kernel = program.kernels[0]
        return {
            **common_record(),
            "baseline": {
                "kind": "canonical",
                "candidate_budget": 0,
                "compiler_search_performed": False,
                "candidate": candidate.to_dict(),
                "concrete_state": concrete_state.to_dict(),
                "qd_search": qd_search.to_dict() if qd_search is not None else None,
                "backend_closure": closure.to_dict() if closure is not None else None,
                "backend_closure_error": closure_error,
            },
            "winner": {
                "candidate": candidate.to_dict(),
                "lowering": program.to_record(),
                "launch": prepared.record.to_dict(),
                "cubin_size_bytes": len(kernel.cubin),
            },
            "correctness": {
                "mlir_j": target_j.to_json(),
                "source_j": source_j.to_json(),
                "policy": DEFAULT_GATE2_POLICY.policy_id,
            },
            "performance": {
                "selected_us": target_latency,
                "mlir_us": target_latency,
                "source_triton_us": source_latency,
                "mlir_over_triton": target_latency / source_latency,
                "speedup_over_triton": source_latency / target_latency,
                "meets_or_beats_triton": target_latency <= source_latency,
                "warmup_ms": warmup_ms,
                "repeat_ms": repeat_ms,
                "timing": timing,
                "graph_repetitions": graph_repetitions,
                "measurement_rounds": measurement_rounds,
                "selection_confirmation": {
                    "mlir_batch_us": target_samples,
                    "source_batch_us": source_samples,
                    "batch_aggregation": "median",
                    "samples_per_batch": measurement_rounds,
                    "comparison_protocol": "paired-alternating.v1",
                    "warmup_pair_discarded": True,
                    "paired_batches": paired_batches,
                },
            },
        }

    def benchmark(program, candidate) -> float:
        if program.selection.pipeline_kind == "specialized":
            prepared = PreparedSingleLaunch(program, materialized.intent_mlir)
        else:
            prepared = PreparedGraphLaunch(
                program,
                materialized.intent_mlir,
                inputs,
            )
            prepared_graphs[candidate.candidate_id] = prepared
        launch_records[candidate.candidate_id] = prepared.record.to_dict()
        thunk = (
            prepared.thunk(tensors)
            if isinstance(prepared, PreparedSingleLaunch)
            else prepared.thunk()
        )
        return measure(thunk)

    search = search_cuda(
        state,
        benchmark=benchmark,
        benchmark_key=(
            f"{name}:profile={workload_profile}:shape={sorted(dims.items())}:"
            f"timing={timing}:graph-repetitions={graph_repetitions}"
            f":measurement-rounds={measurement_rounds}"
        ),
        budget=budget,
        include_generic=(
            axes.search_structural_states if axes is not None else True
        ),
        enable_b_transformations=(
            axes.enable_local_b_transformations
            if axes is not None
            else b_search_mode == "contract-guided"
        ),
        axes=axes,
        fixed_lowering_path_kind=fixed_lowering_path_kind,
        qd_controller=qd_controller,
        require_backend_closure=require_backend_closure,
    )
    if search.winner is None or search.winner_program is None:
        diagnostics = [
            {
                "candidate_id": attempt.candidate.candidate_id,
                "status": attempt.status,
                "reason": attempt.reason,
                "qd_search": (
                    attempt.qd_search.to_dict()
                    if attempt.qd_search is not None
                    else None
                ),
            }
            for attempt in search.attempts
        ]
        raise RuntimeError(
            "measured search produced no executable winner: "
            + json.dumps(diagnostics, sort_keys=True)
        )
    winner = search.winner_program
    if winner.selection.pipeline_kind == "specialized":
        mlir_prepared = PreparedSingleLaunch(winner, materialized.intent_mlir)
        mlir_thunk = mlir_prepared.thunk(tensors)
    else:
        mlir_prepared = prepared_graphs.get(search.winner.candidate_id)
        if mlir_prepared is None:
            mlir_prepared = PreparedGraphLaunch(
                winner,
                materialized.intent_mlir,
                inputs,
            )
        mlir_thunk = mlir_prepared.thunk()
    mlir_thunk()
    torch.cuda.synchronize()

    actual_tensors = (
        outputs
        if isinstance(mlir_prepared, PreparedSingleLaunch)
        else [
            mlir_prepared.outputs[f"out{index}"]
            for index in range(len(case.output_keys))
        ]
    )
    actual = {
        runtime_key: tensor.detach().cpu().numpy().copy()
        for runtime_key, tensor in zip(
            case.output_keys, actual_tensors, strict=True
        )
    }
    mlir_j = compare_outputs(
        "mlir-cubin",
        actual,
        "canonical-J",
        reference.outputs,
        j=snapshot.j,
        input_bindings=case.refeval.input_binding_map,
        output_bindings=case.refeval.output_binding_map,
        inputs=case.values,
        policy=DEFAULT_GATE2_POLICY,
    )
    if not mlir_j.passed:
        raise RuntimeError("compiled MLIR result differs from canonical J")

    search_record = search.to_record()

    (
        source_j,
        mlir_latency,
        source_latency,
        mlir_samples,
        source_samples,
        paired_batches,
    ) = confirm_target_against_source(mlir_thunk)

    kernel = winner.kernels[0]
    cubin_records = [
        {
            "kernel_name": item.kernel_name,
            "binary_name": item.binary_name,
            "size_bytes": len(item.cubin),
        }
        for item in winner.kernels
    ]
    return {
        **common_record(),
        "search": search_record,
        "winner": {
            "candidate": search.winner.to_dict(),
            "lowering": winner.to_record(),
            "launch": mlir_prepared.record.to_dict(),
            "cubin_size_bytes": len(kernel.cubin),
            "cubins": cubin_records,
        },
        "correctness": {
            "mlir_j": mlir_j.to_json(),
            "source_j": source_j.to_json(),
            "policy": DEFAULT_GATE2_POLICY.policy_id,
        },
        "performance": {
            "selected_us": mlir_latency,
            "mlir_us": mlir_latency,
            "source_triton_us": source_latency,
            "mlir_over_triton": mlir_latency / source_latency,
            "speedup_over_triton": source_latency / mlir_latency,
            "meets_or_beats_triton": mlir_latency <= source_latency,
            "warmup_ms": warmup_ms,
            "repeat_ms": repeat_ms,
            "timing": timing,
            "graph_repetitions": graph_repetitions,
            "measurement_rounds": measurement_rounds,
            "selection_confirmation": {
                "mlir_batch_us": mlir_samples,
                "source_batch_us": source_samples,
                "batch_aggregation": "median",
                "samples_per_batch": measurement_rounds,
                "comparison_protocol": "paired-alternating.v1",
                "warmup_pair_discarded": True,
                "paired_batches": paired_batches,
            },
            # These are copied from the same search run.  They are the
            # primary internal attribution fields; separately launched arm
            # latencies remain useful drift diagnostics only.
            "schedule_spine_ablation": search_record[
                "schedule_spine_ablation"
            ],
            "c_fixed_schedule_ablation": search_record[
                "c_fixed_schedule_ablation"
            ],
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument(
        "--kernels",
        nargs="*",
        choices=sorted(CASE_IDS),
        default=sorted(CASE_IDS),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--search-budget", type=int, default=96)
    parser.add_argument(
        "--workload-profile",
        choices=(
            PRODUCTION_WORKLOAD_PROFILE_CLOSED,
            PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
        ),
        default=PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    )
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument(
        "--timing", choices=("graph", "event"), default="graph"
    )
    parser.add_argument("--graph-repetitions", type=int, default=100)
    parser.add_argument("--measurement-rounds", type=int, default=3)
    parser.add_argument(
        "--search-prior",
        choices=("with-c", "without-c"),
        default="with-c",
        help="Attach or erase C for candidate ordering; J=A+B is unchanged.",
    )
    parser.add_argument(
        "--b-search-mode",
        choices=("contract-guided", "schedule-only"),
        default="contract-guided",
        help=(
            "E3 enables B-conditioned MLIR transformations; E2 keeps the "
            "same complete B contract but searches schedule-only candidates."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.measurement_rounds <= 0:
        raise ValueError("measurement-rounds must be positive")
    if args.graph_repetitions <= 0:
        raise ValueError("graph-repetitions must be positive")
    records: list[dict[str, Any]] = []
    started = time.time_ns()
    for name in args.kernels:
        try:
            record = _one_kernel(
                name=name,
                snapshot_dir=args.snapshot_dir,
                budget=args.search_budget,
                warmup_ms=args.warmup_ms,
                repeat_ms=args.repeat_ms,
                timing=args.timing,
                graph_repetitions=args.graph_repetitions,
                measurement_rounds=args.measurement_rounds,
                search_prior=args.search_prior,
                b_search_mode=args.b_search_mode,
                workload_profile=args.workload_profile,
            )
            records.append(record)
            perf = record["performance"]
            print(
                f"{name}: {perf['mlir_us']:.4f} us vs "
                f"{perf['source_triton_us']:.4f} us "
                f"({perf['mlir_over_triton']:.3f}x)",
                flush=True,
            )
        except Exception as exc:
            records.append(
                {
                    "kernel": name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"{name}: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break
    successful = [item for item in records if "performance" in item]
    report = {
        "protocol": REPORT_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "device": _device_record(torch),
        "toolchain": {
            "intent_opt": str(backend_compile.INTENT_OPT),
            "mlir_opt": str(backend_compile.MLIR_OPT),
            "cuda_bin": backend_compile.CUDA_BIN,
        },
        "workload_profile": args.workload_profile,
        "summary": {
            "requested": len(args.kernels),
            "successful": len(successful),
            "failed": len(records) - len(successful),
            "meets_or_beats_triton": sum(
                item["performance"]["meets_or_beats_triton"]
                for item in successful
            ),
        },
        "kernels": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if len(successful) == len(args.kernels) else 1


if __name__ == "__main__":
    raise SystemExit(main())
