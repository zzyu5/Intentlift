#!/usr/bin/env python3
"""Compile production J snapshots into a portable target-cubin search bundle."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.ablation import (
    ExperimentArm,
    SearchAxes,
    standard_ablation_plan,
)
from intentir.backend.cuda.benchmark import extract_portable_launch
from intentir.backend.cuda.portable_bundle import (
    cuda_compute_capability_for_chip,
    encode_triton_spec,
    encode_values,
)
from intentir.backend.cuda.production_cases import (
    FORMAL_PRODUCTION_CASE_IDS,
    PRODUCTION_CASE_IDS,
    PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S512,
    PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S2048,
    PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE,
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE,
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM,
    comparison_policy_for_workload,
    constexpr_input_bindings,
    profile_production_snapshot,
    prepare_production_workload,
)
from intentir.backend.cuda.search import (
    admit_cuda_candidate_paths,
    compile_cuda_target_plugin,
    derive_cuda_contract_view,
    enumerate_cuda_candidates,
    project_cuda_c_fixed_candidate,
    schedule_only_cuda_path_identity,
)
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.measurement_spec import (
    MEASUREMENT_PROTOCOL,
)
from intentir.backend.portable_workload import encode_workload_recipe
from intentir.backend.protocol import BackendTarget
from intentir.backend.workload_cost import derive_j_workload_cost
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.ir.value_dimension import resolve_runtime_output_shapes
from intentir.verify.gate1._triton_worker import _normalize_deprecated_triton_dot
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.scrc_workload_run import build_workload_jc_suite


BUNDLE_PROTOCOL = "intentir.production.cuda-search-bundle.v3"


def _shape(shape: list[int | str], dims: dict[str, int]) -> list[int]:
    resolved: list[int] = []
    for item in shape:
        if item == "?":
            raise ValueError("input shape cannot contain a data-dependent extent")
        if isinstance(item, str):
            if item not in dims:
                raise ValueError(f"input shape has an unbound J extent {item!r}")
            resolved.append(int(dims[item]))
        else:
            resolved.append(int(item))
    return resolved


def _fixed_path_for_axes(
    state: Any,
    axes: SearchAxes | None,
) -> tuple[str | None, str | None]:
    if axes is None or axes.search_structural_states:
        return None, None
    return schedule_only_cuda_path_identity(state)


def _entry(
    name: str,
    snapshot_dir: Path,
    budget: int,
    axes: SearchAxes | None = None,
    workload_profile: str = PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    measure_c_oracle: bool = False,
    rebuild_values_on_target: bool = False,
) -> dict[str, Any]:
    base_snapshot, path = load_production_snapshot(snapshot_dir, name)
    if base_snapshot.c is None:
        raise ValueError("portable search bundle requires accepted C")
    prepared = prepare_production_workload(name, profile=workload_profile)
    comparison_policy = comparison_policy_for_workload(name, workload_profile)
    snapshot, workload_valuation = profile_production_snapshot(
        base_snapshot,
        prepared,
        workload_profile,
    )
    if snapshot.c is None:
        raise AssertionError("workload valuation erased required C")
    suite = build_workload_jc_suite(prepared, snapshot.j, snapshot.c)
    comparison_policy.validate_j(snapshot.j)
    case = suite.cases[0]
    dims = case.refeval.dims_map
    runtime_inputs = [
        np.asarray(case.values[case.refeval.input_binding_map[item.name]])
        for item in snapshot.j.intent.inputs
    ]
    runtime_output_shapes = resolve_runtime_output_shapes(
        snapshot.j.intent,
        [item.shape for item in snapshot.j.intent.outputs],
        runtime_inputs,
        dims=dims,
    )
    case_output_shapes = {
        argument.output_key or argument.name: tuple(argument.shape or ())
        for argument in case.triton.arguments
        if argument.role.value in {"output", "inout"}
    }
    for output, shape in zip(
        snapshot.j.intent.outputs, runtime_output_shapes, strict=True
    ):
        output_key = case.refeval.output_binding_map[output.name]
        if case_output_shapes.get(output_key) != shape:
            raise ValueError(
                f"Harness output shape for {output.name!r} differs from the "
                f"J-derived runtime shape: {case_output_shapes.get(output_key)} != {shape}"
            )
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
    workload_cost = derive_j_workload_cost(materialized)
    state = attach_c(
        materialized,
        snapshot.c,
        target,
    )
    unseeded_state = attach_c(
        materialized,
        None,
        target,
    )
    seeded_contract = derive_cuda_contract_view(state)
    unseeded_contract = derive_cuda_contract_view(unseeded_state)
    if seeded_contract.to_dict() != unseeded_contract.to_dict():
        raise AssertionError("ContractView changed when C was erased")
    contract_view = unseeded_contract
    seeded_kind, seeded_fast_path = _fixed_path_for_axes(state, axes)
    unseeded_kind, unseeded_fast_path = _fixed_path_for_axes(
        unseeded_state, axes
    )
    if (seeded_kind, seeded_fast_path) != (
        unseeded_kind,
        unseeded_fast_path,
    ):
        raise AssertionError("C changed the fixed concrete lowering path")
    seeded_universe = tuple(
        candidate
        for candidate in enumerate_cuda_candidates(
            state,
            axes=axes,
            fixed_lowering_path_kind=seeded_kind,
            fixed_fast_path_id=seeded_fast_path,
        )
        if seeded_fast_path is None or not candidate.generic
    )
    unseeded_universe = tuple(
        candidate
        for candidate in enumerate_cuda_candidates(
            unseeded_state,
            axes=axes,
            fixed_lowering_path_kind=unseeded_kind,
            fixed_fast_path_id=unseeded_fast_path,
        )
        if unseeded_fast_path is None or not candidate.generic
    )
    if {item.candidate_id for item in seeded_universe} != {
        item.candidate_id for item in unseeded_universe
    }:
        raise AssertionError("C changed the candidate universe")
    seeded_candidates, seeded_path_admission = admit_cuda_candidate_paths(
        state,
        seeded_universe,
    )
    unseeded_candidates, unseeded_path_admission = admit_cuda_candidate_paths(
        unseeded_state,
        unseeded_universe,
    )
    if {item.candidate_id for item in seeded_candidates} != {
        item.candidate_id for item in unseeded_candidates
    }:
        raise AssertionError("C changed hard path admission")

    def requested_schedule_spine(
        spine_state: Any,
        admitted_candidates: tuple[Any, ...],
    ) -> list[str]:
        if axes is not None and not axes.search_structural_states:
            return [item.candidate_id for item in admitted_candidates]
        schedule_axes = next(
            item
            for item in standard_ablation_plan(budget)
            if item.arm is ExperimentArm.E2_SCHEDULE
        )
        schedule_axes = replace(
            schedule_axes,
            # ``axes=None`` is the production full-universe export, not an
            # absent experiment definition.  Its schedule spine retains the
            # normal C-prior ordering while hard admission remains C-free.
            use_c_prior=axes.use_c_prior if axes is not None else True,
        )
        spine_kind, spine_fast_path = _fixed_path_for_axes(
            spine_state, schedule_axes
        )
        spine_ids = {
            item.candidate_id
            for item in enumerate_cuda_candidates(
                spine_state,
                axes=schedule_axes,
                fixed_lowering_path_kind=spine_kind,
                fixed_fast_path_id=spine_fast_path,
            )
        }
        return [
            item.candidate_id
            for item in admitted_candidates
            if item.candidate_id in spine_ids
        ]

    seeded_schedule_spine = requested_schedule_spine(
        state, seeded_candidates
    )
    unseeded_schedule_spine = requested_schedule_spine(
        unseeded_state, unseeded_candidates
    )

    candidates_by_id = {item.candidate_id: item for item in seeded_candidates}
    compiled: dict[str, dict[str, Any]] = {}

    def compile_once(candidate: Any) -> dict[str, Any]:
        existing = compiled.get(candidate.candidate_id)
        if existing is not None:
            return existing
        started = time.perf_counter_ns()
        try:
            program = compile_cuda_target_plugin(unseeded_state, candidate)
            concrete_state = candidate.concrete_state(program)
            extracted = extract_portable_launch(program, materialized.intent_mlir)
            result = {
                "status": "compiled_state",
                "candidate_id": candidate.candidate_id,
                "admitted": {
                    "candidate": candidate.to_dict(),
                    "compile_ms": (
                        time.perf_counter_ns() - started
                    )
                    / 1_000_000.0,
                    "lowering": program.to_record(),
                    "concrete_state": concrete_state.to_dict(),
                    "launch": extracted.to_dict(),
                },
            }
        except Exception as exc:
            result = {
                "status": "compile_or_transport_failed",
                "candidate_id": candidate.candidate_id,
                "rejection": {
                    "candidate_id": candidate.candidate_id,
                    "candidate": candidate.to_dict(),
                    "status": "compile_or_transport_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "budget_consumed": True,
                },
            }
        compiled[candidate.candidate_id] = result
        return result

    # Path admission proves that the registered C++ matcher recognizes J, but
    # an exact schedule can still fail the hard target limits of that matcher.
    # C-Fixed must therefore project C over the compiler-formed canonical
    # schedules, not over requested states that never become executables.  This
    # is hard admission only: no workload cost, profiler or timing participates.
    concrete_schedule_spine = []
    c_fixed_schedule_rejections: list[dict[str, Any]] = []
    for candidate_id in seeded_schedule_spine:
        candidate = candidates_by_id[candidate_id]
        outcome = compile_once(candidate)
        if outcome["status"] == "compiled_state":
            concrete_schedule_spine.append(candidate)
        else:
            c_fixed_schedule_rejections.append(outcome["rejection"])
    if not concrete_schedule_spine:
        reasons = "; ".join(
            f"{item['candidate_id']}: {item['reason']}"
            for item in c_fixed_schedule_rejections
        )
        raise RuntimeError(
            f"{name} has no compiler-formed canonical schedule"
            + (f": {reasons}" if reasons else "")
        )
    c_fixed_candidate, c_fixed_projection = project_cuda_c_fixed_candidate(
        state,
        tuple(concrete_schedule_spine),
    )
    c_fixed_projection = {
        **c_fixed_projection,
        "exact_target_schedule_admission": True,
        "rejected_requested_schedule_count": len(c_fixed_schedule_rejections),
        "rejected_requested_schedule_ids": [
            item["candidate_id"] for item in c_fixed_schedule_rejections
        ],
    }

    visit_budget = axes.candidate_budget if axes is not None else budget

    def visit_order(ordered: tuple[Any, ...]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        concrete_ids: list[str] = []
        budget_consumed = 0
        for candidate in ordered:
            if budget_consumed >= visit_budget:
                break
            outcome = compile_once(candidate)
            if outcome["status"] != "compiled_state":
                budget_consumed += 1
                trace.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": outcome["status"],
                        "budget_consumed": True,
                    }
                )
                continue
            budget_consumed += 1
            concrete_ids.append(candidate.candidate_id)
            trace.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": "admitted_state",
                    "budget_consumed": True,
                }
            )
        return {
            "budget_consumed": budget_consumed,
            "trace": trace,
            "concrete_ids": concrete_ids,
        }

    with_c_visit = visit_order(
        (c_fixed_candidate,)
        if axes is not None and axes.arm is ExperimentArm.C_FIXED
        else seeded_candidates
    )
    if axes is not None and axes.arm is ExperimentArm.C_FIXED:
        without_c_visit = {
            "budget_consumed": 0,
            "trace": [],
            "concrete_ids": [],
        }
        if (
            with_c_visit["budget_consumed"] != 1
            or len(with_c_visit["concrete_ids"]) != 1
        ):
            raise RuntimeError(
                "C-Fixed must form exactly one concrete target-projected state"
            )
    else:
        without_c_visit = visit_order(unseeded_candidates)

    # C-Fixed is a formal main bar even when exporting E2/E3.  Carry its exact
    # state in the same bundle, and fail rather than silently substitute another
    # schedule when that mechanical projection cannot compile or transport.
    c_fixed_outcome = compile_once(c_fixed_candidate)
    if c_fixed_outcome["status"] != "compiled_state":
        reason = c_fixed_outcome["rejection"]["reason"]
        raise RuntimeError(f"C-Fixed projection failed: {reason}")

    oracle_trace: list[dict[str, Any]] = []
    oracle_concrete_ids: list[str] = []
    if measure_c_oracle:
        if axes is not None and not axes.search_structural_states:
            raise ValueError(
                "complete C oracle requires an expanded IntentIR search arm"
            )
        for candidate in seeded_candidates:
            outcome = compile_once(candidate)
            oracle_trace.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": outcome["status"],
                    "budget_consumed": False,
                    "oracle_only": (
                        candidate.candidate_id
                        not in with_c_visit["concrete_ids"]
                        and candidate.candidate_id
                        not in without_c_visit["concrete_ids"]
                    ),
                }
            )
            if outcome["status"] == "compiled_state":
                oracle_concrete_ids.append(candidate.candidate_id)

    # The portable bundle carries the candidate-id union of both equal-budget
    # visits. Each explicitly enumerated candidate consumes one budget slot.
    selected_ids = [
        c_fixed_candidate.candidate_id,
        *with_c_visit["concrete_ids"],
        *without_c_visit["concrete_ids"],
        *oracle_concrete_ids,
    ]
    admitted: list[dict[str, Any]] = []
    selected_seen: set[str] = set()
    for candidate_id in selected_ids:
        if candidate_id in selected_seen:
            continue
        selected_seen.add(candidate_id)
        outcome = compiled[candidate_id]
        if outcome["status"] == "compiled_state":
            admitted.append(outcome["admitted"])

    admitted_ids = {
        item["candidate"]["candidate_id"] for item in admitted
    }

    def concrete_order(requested_ids: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for requested_id in requested_ids:
            if requested_id not in admitted_ids:
                continue
            if requested_id not in seen:
                seen.add(requested_id)
                result.append(requested_id)
        return result

    with_c_order = concrete_order(with_c_visit["concrete_ids"])
    without_c_order = concrete_order(without_c_visit["concrete_ids"])
    with_c_full_order = concrete_order(
        [item.candidate_id for item in seeded_candidates]
    )
    without_c_full_order = concrete_order(
        [item.candidate_id for item in unseeded_candidates]
    )
    if measure_c_oracle and set(with_c_full_order) != set(without_c_full_order):
        raise AssertionError("CUDA C oracle orders differ in concrete-state universe")
    with_c_schedule_spine_order = concrete_order(seeded_schedule_spine)
    without_c_schedule_spine_order = concrete_order(
        unseeded_schedule_spine
    )
    use_c_prior = axes.use_c_prior if axes is not None else True
    active_order = with_c_order if use_c_prior else without_c_order
    active_schedule_spine_order = (
        with_c_schedule_spine_order
        if use_c_prior
        else without_c_schedule_spine_order
    )
    if not with_c_order or not with_c_schedule_spine_order:
        raise RuntimeError(f"{name} has no concrete target projection of C")
    c_fixed_projection = {
        **c_fixed_projection,
        "executed_without_search": bool(
            axes is not None and axes.arm is ExperimentArm.C_FIXED
        ),
    }
    rejected_by_id = {
        candidate_id: outcome["rejection"]
        for candidate_id, outcome in compiled.items()
        if outcome["status"] == "compile_or_transport_failed"
    }
    rejected = list(rejected_by_id.values())
    if not admitted:
        raise RuntimeError(f"{name} has no portable executable candidate")

    workload_payload: dict[str, Any]
    if rebuild_values_on_target:
        rebuilt_recipe = encode_workload_recipe(
            prepared.adapter,
            prepared.source_to_opaque,
            case.triton.arguments,
            case.values,
        )
        workload_payload = {"workload_recipe": rebuilt_recipe}
    else:
        reference = ScalarLoweringRunner().run(case.refeval, case.values)
        if not reference.ok:
            raise RuntimeError(f"canonical J reference failed: {reference.error}")
        else:
            reference_outputs = reference.outputs
        for output, shape in zip(
            snapshot.j.intent.outputs, runtime_output_shapes, strict=True
        ):
            output_key = case.refeval.output_binding_map[output.name]
            actual_shape = tuple(np.asarray(reference_outputs[output_key]).shape)
            if actual_shape != shape:
                raise RuntimeError(
                    f"canonical J output shape for {output.name!r} differs from "
                    f"the J-derived runtime shape: {actual_shape} != {shape}"
                )
        workload_payload = {
            "values": encode_values(case.values),
            "reference_outputs": encode_values(reference_outputs),
        }
    source = prepared.request.source.gate_source
    source_code, source_origin = source.read_code()
    source_code, _compat = _normalize_deprecated_triton_dot(source_code)
    source_payload: dict[str, Any]
    source_payload = {
        "source_triton": {
            "function": source.function,
            "code": source_code,
            "origin_at_export": source_origin,
            "execution_spec": encode_triton_spec(case.triton),
        }
    }
    baseline_policy = str(workload_valuation["baseline_policy"])
    return {
        "kernel": name,
        "case_id": prepared.adapter.case_id,
        "workload_profile": workload_profile,
        "baseline_policy": baseline_policy,
        "j": snapshot.j.to_json(),
        "c": snapshot.c.to_json() if snapshot.c is not None else None,
        "snapshot_path": str(path),
        "workload_valuation": workload_valuation,
        "comparison_policy": comparison_policy.to_json(),
        "materialization": materialized.to_record(),
        "contract_view": contract_view.to_dict(),
        "workload_cost": workload_cost.to_record(),
        "ablation_axes": axes.to_dict() if axes is not None else None,
        "fixed_concrete_path": {
            "lowering_path_kind": seeded_kind,
            "fast_path_id": seeded_fast_path,
        },
        "search_orders": {
            "budget": visit_budget,
            "budget_unit": "admitted candidate or compile failure",
            "with_c_budget_consumed": with_c_visit["budget_consumed"],
            "without_c_budget_consumed": without_c_visit["budget_consumed"],
            "with_c_attempt_trace": with_c_visit["trace"],
            "without_c_attempt_trace": without_c_visit["trace"],
            "with_c_concrete_order": with_c_order,
            "without_c_concrete_order": without_c_order,
            "c_oracle_transport_complete": measure_c_oracle,
            "c_oracle_attempt_trace": oracle_trace,
            "with_c_full_concrete_order": (
                with_c_full_order if measure_c_oracle else []
            ),
            "without_c_full_concrete_order": (
                without_c_full_order if measure_c_oracle else []
            ),
            "with_c_schedule_spine_concrete_order": (
                with_c_schedule_spine_order
            ),
            "without_c_schedule_spine_concrete_order": (
                without_c_schedule_spine_order
            ),
            "active_order": "with_c" if use_c_prior else "without_c",
            "active_concrete_order": active_order,
            "active_schedule_spine_concrete_order": (
                active_schedule_spine_order
            ),
            "transported_union_count": len(admitted),
            "candidate_universe_equal": True,
            "contract_view_equal": True,
            "hard_admission_equal": True,
            "c_fixed_exactly_one": bool(
                axes is not None and axes.arm is ExperimentArm.C_FIXED
            ),
            "c_fixed_projection": c_fixed_projection,
            "with_c_hard_admission": list(seeded_path_admission),
            "without_c_hard_admission": list(unseeded_path_admission),
        },
        "extents": dims,
        "input_order": list(materialized.runtime_input_names),
        "input_specs": [
            {
                "name": item.name,
                "dtype": item.dtype,
                "shape": _shape(item.shape, dims),
                "runtime": item.name in materialized.runtime_input_names,
            }
            for item in snapshot.j.intent.inputs
        ],
        "output_specs": [
            {
                "name": item.name,
                "dtype": item.dtype,
                "shape": list(shape),
                "shape_contract": list(item.shape),
            }
            for item, shape in zip(
                snapshot.j.intent.outputs, runtime_output_shapes, strict=True
            )
        ],
        "input_bindings": case.refeval.input_binding_map,
        "output_bindings": case.refeval.output_binding_map,
        "output_keys": list(case.output_keys),
        **workload_payload,
        **source_payload,
        "candidates": admitted,
        "rejected_candidate_count": len(rejected),
        "rejected_candidates": rejected,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--kernels",
        nargs="*",
        choices=sorted(PRODUCTION_CASE_IDS),
        default=sorted(FORMAL_PRODUCTION_CASE_IDS),
    )
    parser.add_argument("--search-budget", type=int, default=96)
    parser.add_argument(
        "--workload-profile",
        choices=(
            PRODUCTION_WORKLOAD_PROFILE_CLOSED,
            PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S512,
            PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S2048,
            PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
            PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM,
            PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE,
            PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE,
        ),
        default=PRODUCTION_WORKLOAD_PROFILE_CLOSED,
    )
    parser.add_argument(
        "--arm",
        choices=[
            item.value
            for item in ExperimentArm
            if item
            not in {ExperimentArm.E0_SOURCE, ExperimentArm.E1_CANONICAL}
        ],
        help=(
            "select one machine-specified search arm; omitted preserves the "
            "full production candidate universe"
        ),
    )
    parser.add_argument(
        "--measure-c-oracle",
        action="store_true",
        help=(
            "transport the complete expanded concrete-state universe for the "
            "formal C-Fixed/Schedule-Only/IntentIR performance envelopes and "
            "the candidates-to-95%% anytime diagnostic; bounded-order budgets "
            "remain unchanged"
        ),
    )
    parser.add_argument(
        "--rebuild-values-on-target",
        action="store_true",
        help=(
            "transport the closed workload initializer/seed recipe and rebuild "
            "values plus canonical RefJ on the target instead of embedding "
            "large tensor payloads"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    axes = None
    if args.arm is not None:
        axes = next(
            item
            for item in standard_ablation_plan(args.search_budget)
            if item.arm.value == args.arm
        )
    started = time.time_ns()
    if (
        args.workload_profile == PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE
        and args.kernels != ["masked_copy"]
    ):
        raise ValueError(
            "roofline-large-j-only-v1 currently requires exactly "
            "--kernels masked_copy"
        )
    entries = []
    for name in args.kernels:
        entry = _entry(
            name,
            args.snapshot_dir,
            args.search_budget,
            axes,
            args.workload_profile,
            args.measure_c_oracle,
            args.rebuild_values_on_target,
        )
        entries.append(entry)
        print(
            f"{name}: {len(entry['candidates'])} target cubins",
            flush=True,
        )
    baseline_policies = {str(entry["baseline_policy"]) for entry in entries}
    bundle = {
        "protocol": BUNDLE_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "target": {
            "backend": "cuda",
            "chip": backend_compile.CHIP,
            "compute_capability": list(
                cuda_compute_capability_for_chip(backend_compile.CHIP)
            ),
            "binary_format": "elf-cubin",
            "load_policy": "exact-compute-capability",
        },
        "workload_profile": (
            next(iter({entry["workload_profile"] for entry in entries}))
            if len({entry["workload_profile"] for entry in entries}) == 1
            else "mixed_by_kernel"
        ),
        "baseline_policy": (
            next(iter(baseline_policies))
            if len(baseline_policies) == 1
            else "mixed_by_kernel"
        ),
        "ablation_axes": axes.to_dict() if axes is not None else None,
        "measurement_spec_protocol": MEASUREMENT_PROTOCOL,
        "toolchain": {
            "intent_opt": str(backend_compile.INTENT_OPT),
            "mlir_opt": str(backend_compile.MLIR_OPT),
            "cuda_bin": backend_compile.CUDA_BIN,
        },
        "kernels": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
