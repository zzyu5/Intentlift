#!/usr/bin/env python3
"""Compile accepted production J snapshots into native RVV object bundles."""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from intentir.backend.ablation import ABLATION_PROTOCOL
from intentir.backend.cuda.portable_bundle import encode_values
from intentir.backend.cuda.production_cases import (
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
from intentir.backend.materialize import attach_c, materialize_j
from intentir.backend.measurement_spec import (
    MEASUREMENT_PROTOCOL,
)
from intentir.backend.protocol import BackendTarget
from intentir.backend.riscv.compile import lower_j_to_riscv
from intentir.backend.riscv.contract import derive_riscv_contract_view
from intentir.backend.riscv.native_baselines import native_rvv_baseline_record
from intentir.backend.riscv.search import (
    build_riscv_target_artifact,
    plan_riscv_search,
)
from intentir.backend.workload_cost import derive_j_workload_cost
from intentir.backend.protocol import LoweringRequest
from intentir.ir.canonical_j import j_elaboration
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.ir.value_dimension import (
    resolve_runtime_output_shapes,
    validate_ragged_runtime_invocation,
)
from intentir.verify.gate1._triton_worker import _normalize_deprecated_triton_dot
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import ResultRelationKind
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.gate2.triton import PreparedTritonInvocation
from intentir.verify.scrc_workload_run import build_workload_jc_suite


BUNDLE_PROTOCOL = "intentir.production.rvv-search-bundle.v4"
TARGET = BackendTarget("riscv", "rv64gcv_zfh_zvfh", ("vlen128",))


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


def _output_rules(snapshot: Any, comparison_policy: Any) -> dict[str, Any]:
    relations = {
        item.output: item.kind for item in snapshot.j.boundary.result_relations
    }
    result = {}
    for binding in j_elaboration(snapshot.j).output_bindings:
        kind = relations.get(binding.output, ResultRelationKind.EXACT)
        rule = comparison_policy.rule_for(kind, binding.output_type.dtype)
        result[binding.output] = {
            "comparison_class": kind.value,
            "dtype": rule.dtype,
            "tolerance": (
                rule.tolerance.to_json()
                if rule.tolerance is not None
                else None
            ),
        }
    return result


def _canonical_reference_outputs(
    prepared: Any,
    case: Any,
) -> tuple[dict[str, Any], str, str | None]:
    """Build target-independent Q data from RefJ, or the same closed Source.

    A few accepted legacy artifacts still exceed the old functional-boundary
    projector at a natural valuation even though their Source/J closure and
    real target lowering are valid.  That projector is being replaced by the
    shared typed ContractExpr semantics.  Until then, use the original closed
    Source only as an offline correctness oracle; RVV target code, search and
    timing remain entirely MLIR/native.
    """

    reference = ScalarLoweringRunner().run(case.refeval, case.values)
    if reference.ok:
        return dict(reference.outputs), "refj", None
    source_program = prepared.request.source.gate_source
    source_code, _source_origin = source_program.read_code()
    source_code, _frontend_compat = _normalize_deprecated_triton_dot(source_code)
    source_program = KernelSource.from_code(
        source_code,
        source_program.function,
        trusted=True,
    )
    try:
        with PreparedTritonInvocation(
            source_program,
            case.triton,
            case.values,
        ) as invocation:
            invocation.reset_mutable_inputs()
            invocation.launch()
            outputs = invocation.copied_outputs()
    except Exception as error:
        raise RuntimeError(
            f"canonical J reference failed: {reference.error}; "
            f"closed Source fallback failed: {type(error).__name__}: {error}"
        ) from error
    required = set(case.refeval.output_binding_map.values())
    if not required.issubset(outputs):
        raise RuntimeError(
            "closed Source fallback omitted canonical output bindings: "
            + ", ".join(sorted(required - set(outputs)))
        )
    return outputs, "source", reference.error


def _entry(
    name: str,
    snapshot_dir: Path,
    workload_profile: str = PRODUCTION_WORKLOAD_PROFILE_CLOSED,
) -> dict[str, Any]:
    base_snapshot, path = load_production_snapshot(snapshot_dir, name)
    if base_snapshot.c is None:
        raise ValueError("RVV bundle requires accepted C")
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
    canonical_outputs = j_elaboration(snapshot.j).output_bindings
    if not canonical_outputs:
        raise ValueError("accepted J has no canonical output bindings")
    runtime_inputs = [
        np.asarray(case.values[case.refeval.input_binding_map[item.name]])
        for item in snapshot.j.intent.inputs
    ]
    validate_ragged_runtime_invocation(
        snapshot.j.intent,
        runtime_inputs,
        dims=dims,
    )
    runtime_output_shapes = resolve_runtime_output_shapes(
        snapshot.j.intent,
        [list(item.output_type.shape) for item in canonical_outputs],
        runtime_inputs,
        dims=dims,
        canonical_output_bindings=canonical_outputs,
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
    prior_state = attach_c(materialized, snapshot.c, TARGET)
    neutral_state = attach_c(materialized, None, TARGET)
    prior_contract = derive_riscv_contract_view(prior_state)
    contract = derive_riscv_contract_view(neutral_state)
    if prior_contract.to_dict() != contract.to_dict():
        raise AssertionError("RISC-V ContractView changed when C was erased")
    workload_cost = derive_j_workload_cost(materialized)
    prior_search_plan = plan_riscv_search(
        prior_state,
        enable_b_transformations=True,
    )
    neutral_search_plan = plan_riscv_search(
        neutral_state,
        enable_b_transformations=True,
    )
    if {
        item.candidate_id for item in prior_search_plan.universe
    } != {
        item.candidate_id for item in neutral_search_plan.universe
    }:
        raise AssertionError("C changed the RISC-V candidate universe")
    search_plan = prior_search_plan
    candidates = []
    rejected = []
    for candidate in search_plan.universe:
        started = time.perf_counter_ns()
        try:
            artifact = build_riscv_target_artifact(
                neutral_state,
                candidate,
            )
            lowering = lower_j_to_riscv(
                LoweringRequest(
                    j=neutral_state.base.j,
                    target=TARGET,
                    variant="specialized",
                    c=artifact,
                    constexpr_bindings=neutral_state.base.constexpr_bindings,
                ),
                neutral_state.base.concrete_extents_map,
            )
            compiled = lowering.artifact
        except Exception as error:
            rejected.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate": candidate.to_dict(),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        compiled_record = compiled.to_record()
        if not compiled_record["schedule_realization"]["requested_equals_actual"]:
            rejected.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate": candidate.to_dict(),
                    "reason": (
                        "requested RVV schedule was not realized by MLIR: "
                        f"requested={compiled_record['requested_schedule']} "
                        f"actual={compiled_record['schedule']}"
                    ),
                }
            )
            continue
        candidates.append(
            {
                "candidate": compiled_record,
                "compile_ms": (
                    time.perf_counter_ns() - started
                )
                / 1_000_000.0,
                "object_base64": base64.b64encode(
                    compiled.object_bytes
                ).decode("ascii"),
            }
        )
    if not candidates:
        reasons = list(
            dict.fromkeys(item["reason"] for item in rejected)
        )
        raise RuntimeError(
            f"{name} has no compiled RVV candidates: "
            + " | ".join(reasons[:3])
        )
    available_ids = {
        item["candidate"]["candidate_id"] for item in candidates
    }

    def concrete_order(requested_order: list[str] | tuple[str, ...]) -> list[str]:
        result = []
        seen = set()
        for requested_id in requested_order:
            candidate_id = requested_id
            if (
                candidate_id is None
                or candidate_id not in available_ids
                or candidate_id in seen
            ):
                continue
            seen.add(candidate_id)
            result.append(candidate_id)
        return result

    erase_c_order = concrete_order(search_plan.erase_c_order)
    c_prior_order = concrete_order(search_plan.c_prior_order)
    schedule_only_erase_c_order = concrete_order(
        search_plan.schedule_only_erase_c_order
    )
    schedule_only_c_prior_order = concrete_order(
        search_plan.schedule_only_c_prior_order
    )
    c_fixed_candidate_id = search_plan.c_fixed_candidate_id
    if c_fixed_candidate_id not in available_ids:
        c_fixed_rejection = next(
            (
                item["reason"]
                for item in rejected
                if item["candidate_id"] == c_fixed_candidate_id
            ),
            "candidate was not present in the compiled universe",
        )
        raise RuntimeError(
            "mechanical RVV C-Fixed projection did not compile; refusing to "
            f"substitute another schedule: {c_fixed_rejection}"
        )
    candidate_by_id = {
        item["candidate"]["candidate_id"]: item["candidate"] for item in candidates
    }
    schedule_only_ids = set(schedule_only_c_prior_order)
    if schedule_only_ids != set(schedule_only_erase_c_order):
        raise AssertionError("C changed the concrete Schedule-Only universe")
    c_fixed_candidate = candidate_by_id[c_fixed_candidate_id]
    canonical_lowering_path = str(
        c_fixed_candidate["transformation"]["lowering_path_kind"]
    )
    if any(
        str(candidate_by_id[candidate_id]["transformation"]["lowering_path_kind"])
        != canonical_lowering_path
        for candidate_id in schedule_only_ids
    ):
        raise AssertionError("Schedule-Only escaped the canonical lowering path")
    reference_outputs, reference_kind, canonical_j_reference_error = (
        _canonical_reference_outputs(prepared, case)
    )
    for output, shape in zip(canonical_outputs, runtime_output_shapes, strict=True):
        output_key = case.refeval.output_binding_map[output.output]
        if output_key not in reference_outputs:
            raise RuntimeError(
                f"canonical reference omitted output {output.output!r}"
            )
        actual_shape = tuple(np.asarray(reference_outputs[output_key]).shape)
        if actual_shape != shape:
            raise RuntimeError(
                f"canonical reference output shape for {output.output!r} differs "
                f"from the J-derived runtime shape: {actual_shape} != {shape}"
            )
    return {
        "kernel": name,
        "case_id": PRODUCTION_CASE_IDS[name],
        "workload_profile": workload_profile,
        "workload_valuation": workload_valuation,
        "comparison_policy": comparison_policy.to_json(),
        # RVV has no Triton Source executable.  Use a separately authored native
        # implementation that shares inputs/ABI but no IntentIR lowering state.
        "baseline_policy": "native_rvv_and_refj",
        "target_reference": reference_kind,
        "canonical_j_reference_error": canonical_j_reference_error,
        "native_baseline": native_rvv_baseline_record(name),
        "snapshot_path": str(path),
        "materialization": materialized.to_record(),
        "contract_view": contract.to_dict(),
        "workload_cost": workload_cost.to_record(),
        "extents": dims,
        "input_order": list(materialized.runtime_input_names),
        "input_bindings": case.refeval.input_binding_map,
        "output_bindings": case.refeval.output_binding_map,
        "input_specs": {
            item.name: {
                "dtype": item.dtype,
                "shape": _shape(item.shape, dims),
            }
            for item in snapshot.j.intent.inputs
            if item.name in materialized.runtime_input_names
        },
        "output_specs": [
            {
                "name": item.output,
                "dtype": item.output_type.dtype,
                "shape": list(shape),
                "shape_contract": list(item.output_type.shape),
            }
            for item, shape in zip(canonical_outputs, runtime_output_shapes, strict=True)
        ],
        "output_rules": _output_rules(snapshot, comparison_policy),
        "values": encode_values(case.values),
        "reference_outputs": encode_values(reference_outputs),
        "search_plan": search_plan.to_record(),
        "concrete_search_plan": {
            "candidate_universe_count": len(candidates),
            "schedule_only_erase_c_order": schedule_only_erase_c_order,
            "schedule_only_c_prior_order": schedule_only_c_prior_order,
            "erase_c_order": erase_c_order,
            "c_prior_order": c_prior_order,
            "c_fixed_requested_candidate_id": (
                search_plan.c_fixed_candidate_id
            ),
            "c_fixed_candidate_id": c_fixed_candidate_id,
            "c_fixed_exactly_one": True,
            "c_projection": dict(search_plan.c_projection),
            "canonical_lowering_path": canonical_lowering_path,
            "same_universe_for_erase_c_and_c_prior": True,
            "same_schedule_only_universe_for_erase_c_and_c_prior": True,
            "budget_unit": "admitted_candidate",
        },
        "candidate_order": c_prior_order,
        "schedule_only_candidate_ids": [
            candidate_id
            for candidate_id in schedule_only_c_prior_order
        ],
        "candidates": candidates,
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
        default=sorted(PRODUCTION_CASE_IDS),
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time_ns()
    entries = []
    failures = []
    if (
        args.workload_profile == PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE
        and args.kernels != ["masked_copy"]
    ):
        raise ValueError(
            "roofline-large-j-only-v1 currently requires exactly "
            "--kernels masked_copy"
        )
    for name in args.kernels:
        try:
            entry = _entry(
                name,
                args.snapshot_dir,
                args.workload_profile,
            )
            entries.append(entry)
            print(
                f"{name}: {len(entry['candidates'])} native RVV objects",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {
                    "kernel": name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"{name}: FAILED: {type(error).__name__}: {error}",
                flush=True,
            )
    baseline_policies = {str(entry["baseline_policy"]) for entry in entries}
    payload = {
        "protocol": BUNDLE_PROTOCOL,
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "target": TARGET.to_dict(),
        "experiment_spec_protocol": ABLATION_PROTOCOL,
        "measurement_spec_protocol": MEASUREMENT_PROTOCOL,
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
        "correctness_policy": DEFAULT_GATE2_POLICY.to_json(),
        "true_mlir_pipeline": True,
        "python_target_kernel_emission": False,
        "summary": {
            "requested": len(args.kernels),
            "compiled": len(entries),
            "failed": len(failures),
        },
        "kernels": entries,
        "failures": failures,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
