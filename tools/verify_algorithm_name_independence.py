#!/usr/bin/env python3
"""Verify that target consumers select from Intent structure, not operator names.

The intervention deliberately assigns every formal J a misleading function name
borrowed from a different lowering structure.  CUDA selection, CUDA's admitted
Full candidate space, and the RVV search plan must remain unchanged.  A 6x6
CUDA cross-structure matrix supplies the negative control: under one identical
function name, only the structurally matching whole-graph pass may be admitted.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from intentir.backend.ablation import ExperimentArm, SearchAxes
from intentir.backend.cuda.production_cases import (
    FORMAL_PRODUCTION_CASE_IDS,
    PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    constexpr_input_bindings,
    prepare_production_workload,
    profile_production_snapshot,
)
from intentir.backend.cuda.search import (
    admit_cuda_candidate_paths,
    enumerate_cuda_candidates,
    schedule_only_cuda_path_identity,
)
from intentir.backend.cuda.selection import LoweringTarget, select_variant
from intentir.backend.materialize import MaterializedJ, attach_c, materialize_j
from intentir.backend.protocol import BackendTarget
from intentir.backend.riscv.search import plan_riscv_search
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.gate1.model import ImplementationConfig
from intentir.verify.scrc_workload_run import build_workload_jc_suite


_SYMBOL = re.compile(
    r"\bfunc\.func\s+(?:(?:public|private)\s+)?@"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.$]*)"
)

_PATH_BY_GROUP = {
    "elementwise": "cuda.elementwise.row.vectorized.v1",
    "row": "cuda.row_reduction.normalization_loss.read_once.v1",
    "padded_row": "cuda.padded_row_reduction.masked_short_dot.v1",
    "contraction": "cuda.contraction.tiled.v1",
    "splitk": "cuda.splitk.reduce-varargs.v1",
    "attention": "cuda.flash_forward.structured_scan.direct.v1",
}

_GROUP_BY_CASE = {
    "cross_entropy": "row",
    "dyt": "elementwise",
    "flash_attention_forward": "attention",
    "flash_bwd_preprocess": "padded_row",
    "flash_layer_norm_1pass": "row",
    "fused_add_rms_norm": "row",
    "flash_linear": "contraction",
    "geglu": "elementwise",
    "group_norm": "row",
    "jsd": "row",
    "kl_div": "row",
    "layer_norm": "row",
    "llama4_rope": "elementwise",
    "liger_rope": "elementwise",
    "liger_rms_norm": "row",
    "poly_norm": "row",
    "qwen2vl_mrope": "elementwise",
    "softmax": "row",
    "sparsemax": "row",
    "splitk_reduce_varargs": "splitk",
    "swiglu": "elementwise",
    "tvd": "elementwise",
    "xformers_rms_norm": "row",
    "xformers_rms_norm_add": "row",
}

_MISLEADING_NAME_BY_GROUP = {
    "elementwise": "liger_rms_norm",
    "row": "swiglu",
    "padded_row": "flash_linear",
    "contraction": "splitk_reduce_varargs",
    "splitk": "flash_attention_forward",
    "attention": "liger_rms_norm",
}

_REPRESENTATIVE_BY_GROUP = {
    "elementwise": "swiglu",
    "row": "liger_rms_norm",
    "padded_row": "flash_bwd_preprocess",
    "contraction": "flash_linear",
    "splitk": "splitk_reduce_varargs",
    "attention": "flash_attention_forward",
}


def _full_axes() -> SearchAxes:
    return SearchAxes(
        arm=ExperimentArm.E3_STATE_SEARCH,
        candidate_budget=256,
        use_c_prior=True,
        search_structural_states=True,
        enable_local_b_transformations=True,
        search_reduction_topology=True,
        search_fma=True,
        search_precision=True,
        search_approximation=True,
    )


def _rename_entry_symbol(intent_mlir: str, new_name: str) -> tuple[str, str]:
    match = _SYMBOL.search(intent_mlir)
    if match is None:
        raise ValueError("materialized module has no bare func.func symbol")
    old_name = match.group("name")
    token = re.compile(rf"@{re.escape(old_name)}(?![A-Za-z0-9_.$])")
    renamed, replacements = token.subn(f"@{new_name}", intent_mlir)
    if replacements < 1:
        raise AssertionError("entry symbol rename made no replacement")
    return renamed, old_name


def _load_case(
    catalog: Path,
    case_name: str,
) -> tuple[MaterializedJ, ImplementationConfig]:
    snapshot, _path = load_production_snapshot(catalog, case_name)
    if snapshot.j is None or snapshot.c is None:
        raise ValueError(
            f"{case_name}: compatibility snapshot lacks J or Source/search prior"
        )
    prepared = prepare_production_workload(
        case_name,
        profile=PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    )
    snapshot, _valuation = profile_production_snapshot(
        snapshot,
        prepared,
        PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE,
    )
    if snapshot.c is None:
        raise ValueError(f"{case_name}: profiled snapshot lacks C")
    case = build_workload_jc_suite(prepared, snapshot.j, snapshot.c).cases[0]
    constexpr = constexpr_input_bindings(
        snapshot.j,
        case.refeval.input_binding_map,
        case.values,
    )
    materialized = materialize_j(
        snapshot.j,
        case.refeval.dims_map,
        constexpr_bindings=constexpr,
    )
    return materialized, snapshot.c


def _candidate_record(candidate: Any) -> dict[str, Any]:
    record = candidate.to_dict()
    # JSON normalization makes Mapping implementations and tuple/list spelling
    # irrelevant without erasing any search-relevant field.
    return json.loads(json.dumps(record, sort_keys=True))


def _cuda_full_space(
    materialized: MaterializedJ,
    config: ImplementationConfig,
) -> dict[str, Any]:
    state = attach_c(
        materialized,
        config,
        BackendTarget("cuda", "sm_120a"),
    )
    raw = enumerate_cuda_candidates(state, axes=_full_axes())
    admitted, path_records = admit_cuda_candidate_paths(state, raw)
    admitted_paths = tuple(
        record["path_id"] for record in path_records if record["admitted"]
    )
    return {
        "raw_candidate_count": len(raw),
        "admitted_candidate_count": len(admitted),
        "admitted_paths": admitted_paths,
        "admitted_candidates": tuple(_candidate_record(item) for item in admitted),
    }


def _rvv_plan(
    materialized: MaterializedJ,
    config: ImplementationConfig,
) -> dict[str, Any]:
    state = attach_c(
        materialized,
        config,
        BackendTarget("riscv", "rv64gcv_zfh_zvfh", ("vlen128",)),
    )
    plan = plan_riscv_search(state, enable_b_transformations=True)
    return plan.to_record()


def verify(catalog: Path) -> dict[str, Any]:
    if set(_GROUP_BY_CASE) != set(FORMAL_PRODUCTION_CASE_IDS):
        raise AssertionError("name-independence map must cover the exact formal cohort")

    loaded = {
        name: _load_case(catalog, name) for name in FORMAL_PRODUCTION_CASE_IDS
    }
    per_case: list[dict[str, Any]] = []
    cuda_path_invariant = 0
    rvv_plan_invariant = 0

    for case_name in FORMAL_PRODUCTION_CASE_IDS:
        group = _GROUP_BY_CASE[case_name]
        materialized, config = loaded[case_name]
        misleading_name = _MISLEADING_NAME_BY_GROUP[group]
        misleading_ir, original_symbol = _rename_entry_symbol(
            materialized.intent_mlir,
            misleading_name,
        )
        misleading = replace(materialized, intent_mlir=misleading_ir)

        original_cuda_state = attach_c(
            materialized,
            config,
            BackendTarget("cuda", "sm_120a"),
        )
        misleading_cuda_state = attach_c(
            misleading,
            config,
            BackendTarget("cuda", "sm_120a"),
        )
        original_path = schedule_only_cuda_path_identity(original_cuda_state)
        misleading_path = schedule_only_cuda_path_identity(misleading_cuda_state)
        expected_path = _PATH_BY_GROUP[group]
        path_equal = original_path == misleading_path
        expected_selected = original_path[1] == expected_path
        if path_equal and expected_selected:
            cuda_path_invariant += 1

        original_rvv = _rvv_plan(materialized, config)
        misleading_rvv = _rvv_plan(misleading, config)
        rvv_equal = original_rvv == misleading_rvv
        if rvv_equal:
            rvv_plan_invariant += 1

        per_case.append(
            {
                "case": case_name,
                "structure": group,
                "original_symbol": original_symbol,
                "misleading_symbol": misleading_name,
                "original_cuda_path": list(original_path),
                "misleading_cuda_path": list(misleading_path),
                "expected_cuda_path": expected_path,
                "cuda_path_invariant": path_equal and expected_selected,
                "rvv_search_plan_invariant": rvv_equal,
                "rvv_candidate_count": original_rvv["candidate_universe_count"],
            }
        )

    full_space_records: list[dict[str, Any]] = []
    full_space_invariant = 0
    for group, case_name in _REPRESENTATIVE_BY_GROUP.items():
        materialized, config = loaded[case_name]
        misleading_ir, _old = _rename_entry_symbol(
            materialized.intent_mlir,
            _MISLEADING_NAME_BY_GROUP[group],
        )
        misleading = replace(materialized, intent_mlir=misleading_ir)
        original_space = _cuda_full_space(materialized, config)
        misleading_space = _cuda_full_space(misleading, config)
        equal = original_space == misleading_space
        if equal:
            full_space_invariant += 1
        full_space_records.append(
            {
                "structure": group,
                "case": case_name,
                "invariant": equal,
                "raw_candidate_count": original_space["raw_candidate_count"],
                "admitted_candidate_count": original_space[
                    "admitted_candidate_count"
                ],
                "admitted_path_count": len(original_space["admitted_paths"]),
                "admitted_paths": list(original_space["admitted_paths"]),
            }
        )

    matrix: list[dict[str, Any]] = []
    diagonal_accepts = 0
    off_diagonal_rejects = 0
    target = LoweringTarget("cuda", "sm_120a")
    for graph_group, case_name in _REPRESENTATIVE_BY_GROUP.items():
        materialized, _config = loaded[case_name]
        anonymous_ir, _old = _rename_entry_symbol(
            materialized.intent_mlir,
            "same_operator_name_for_all_graphs",
        )
        for pass_group, path_id in _PATH_BY_GROUP.items():
            selection = select_variant(
                anonymous_ir,
                target=target,
                only_fast_path=path_id,
            )
            admitted = selection.fast_path_id == path_id
            expected = graph_group == pass_group
            if expected and admitted:
                diagonal_accepts += 1
            if not expected and not admitted:
                off_diagonal_rejects += 1
            matrix.append(
                {
                    "graph_structure": graph_group,
                    "case": case_name,
                    "pass_structure": pass_group,
                    "path_id": path_id,
                    "admitted": admitted,
                    "expected": expected,
                }
            )

    summary = {
        "formal_case_count": len(FORMAL_PRODUCTION_CASE_IDS),
        "cuda_misleading_name_path_invariance": {
            "passed": cuda_path_invariant,
            "total": len(FORMAL_PRODUCTION_CASE_IDS),
        },
        "rvv_misleading_name_search_plan_invariance": {
            "passed": rvv_plan_invariant,
            "total": len(FORMAL_PRODUCTION_CASE_IDS),
        },
        "cuda_full_candidate_space_invariance": {
            "passed": full_space_invariant,
            "total": len(_REPRESENTATIVE_BY_GROUP),
        },
        "cuda_cross_structure_matrix": {
            "diagonal_admitted": diagonal_accepts,
            "diagonal_total": len(_REPRESENTATIVE_BY_GROUP),
            "off_diagonal_rejected": off_diagonal_rejects,
            "off_diagonal_total": (
                len(_REPRESENTATIVE_BY_GROUP)
                * (len(_PATH_BY_GROUP) - 1)
            ),
        },
    }
    expected_summary = {
        "cuda_misleading_name_path_invariance": len(FORMAL_PRODUCTION_CASE_IDS),
        "rvv_misleading_name_search_plan_invariance": len(
            FORMAL_PRODUCTION_CASE_IDS
        ),
        "cuda_full_candidate_space_invariance": len(_REPRESENTATIVE_BY_GROUP),
        "diagonal_admitted": len(_REPRESENTATIVE_BY_GROUP),
        "off_diagonal_rejected": len(_REPRESENTATIVE_BY_GROUP) * 5,
    }
    observed_summary = {
        "cuda_misleading_name_path_invariance": cuda_path_invariant,
        "rvv_misleading_name_search_plan_invariance": rvv_plan_invariant,
        "cuda_full_candidate_space_invariance": full_space_invariant,
        "diagonal_admitted": diagonal_accepts,
        "off_diagonal_rejected": off_diagonal_rejects,
    }
    if observed_summary != expected_summary:
        raise AssertionError(
            "algorithm/name-independence intervention failed: "
            f"expected={expected_summary}, observed={observed_summary}"
        )
    return {
        "protocol": "intentir.backend.algorithm-name-independence.v1",
        "cohort": "formal production-jc-v4 24-J set",
        "targets": ["cuda:sm_120a", "riscv:rv64gcv_zfh_zvfh:vlen128"],
        "intervention": (
            "replace each entry function symbol with an operator name from a "
            "different lowering structure; keep J, C, target, and MLIR body fixed"
        ),
        "summary": summary,
        "per_case": per_case,
        "cuda_full_space_representatives": full_space_records,
        "cuda_cross_structure_matrix": matrix,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("corpus/production-jc-v4"),
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = verify(args.catalog)
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
