#!/usr/bin/env python3
"""Continue SCRC from one exact TypeChecked, Gate1-failed artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.extraction.providers.provider import get_provider
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.ttir import fingerprint_ttir
from intentir.verify.scrc_runner import (
    FeedbackKind,
    LiftFeedback,
    MainStatus,
    MAX_LIFT_CALLS,
    _compact_gate1_observer_diff,
)
from intentir.verify.scrc_workload_run import (
    complete_prepared_workload_gate2,
    initialize_workload_evidence,
    next_workload_run_dir,
    save_workload_execution,
    save_workload_failure,
    save_workload_gate1_search,
    search_prepared_workload_gate1,
    validate_workload_gate1_continuation_parent,
)
from intentir.verify.typecheck import type_check
from intentir.verify.workload_adapters import (
    TRITON_WORKLOAD_ADAPTERS,
    prepare_triton_workload,
)


RANKED_CONTINUATION_PROTOCOL = (
    "intentir.scrc.ranked-gate1-continuation.v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="one active workload case id")
    parser.add_argument(
        "--parent-evidence",
        required=True,
        type=Path,
        help="attempt directory containing artifact.json and gate1.json",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument(
        "--lift-backend",
        choices=("responses", "miniswe", "codex"),
        default="responses",
    )
    parser.add_argument(
        "--replay-backend",
        choices=("responses", "miniswe", "codex"),
        default="miniswe",
    )
    parser.add_argument("--max-gate1-rounds", type=int, default=6)
    parser.add_argument(
        "--max-lift-calls",
        type=int,
        default=MAX_LIFT_CALLS,
        help=(
            "maximum new logical Lift invocations in this continuation "
            f"(1-{MAX_LIFT_CALLS})"
        ),
    )
    parser.add_argument(
        "--max-pre-replay-rejections",
        type=int,
        default=None,
        help=(
            "optional pre-Replay rejection cap; default 0 for single MiniSWE "
            "owner continuations and 3 for legacy writers"
        ),
    )
    parser.add_argument(
        "--replay-width",
        type=int,
        choices=range(1, 5),
        default=1,
        metavar="{1,2,3,4}",
        help="independent source-blind Replay workers per formal Gate1 round",
    )
    parser.add_argument("--run-masked-j", action="store_true")
    parser.add_argument(
        "--retry-parent-gate1",
        action="store_true",
        help="run fresh formal Replay fan-out on the exact parent before Lift",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/intentir-scrc-task11/ranked-continuations"),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"parent evidence lacks {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"parent {label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"parent {label} must be a JSON object")
    return value


def _formal_round(parent: Path) -> int:
    """Recover the saved formal round when its enclosing report is present."""

    match = re.fullmatch(r"attempt-(\d+)", parent.name)
    search_path = parent.parent / "gate1_search.json"
    if match is None or not search_path.is_file():
        return 1
    search = _read_object(search_path, "gate1_search.json")
    attempts = search.get("attempts")
    index = int(match.group(1)) - 1
    if not isinstance(attempts, list) or not 0 <= index < len(attempts):
        return 1
    attempt = attempts[index]
    formal_round = (
        attempt.get("formal_round") if isinstance(attempt, Mapping) else None
    )
    return formal_round if isinstance(formal_round, int) and formal_round > 0 else 1


def _load_seed(parent_path: Path, prepared: Any) -> LiftFeedback:
    parent = parent_path.expanduser().resolve()
    if not parent.is_dir():
        raise ValueError("parent evidence directory does not exist")
    artifact_payload = _read_object(parent / "artifact.json", "artifact.json")
    try:
        artifact = canonicalize_gate1_artifact(
            Gate1Artifact.from_json(artifact_payload)
        )
    except Exception as exc:
        raise ValueError(
            f"parent artifact decode rejected: {type(exc).__name__}: {exc}"
        ) from exc
    validate_workload_gate1_continuation_parent(parent, prepared, artifact)
    checked = type_check(
        artifact.j,
        artifact.c,
        artifact.pi,
        require_program_axes=artifact.wire_schema == artifact.SCHEMA,
    )
    if not checked.passed:
        # A strengthened closed checker can invalidate an older formal parent.
        # Preserve the exact model-authored artifact as the working draft and
        # let the Lift owner repair it before any new Replay/Gate execution.
        return LiftFeedback(
            FeedbackKind.TYPECHECK,
            {"typecheck": checked.to_json()},
            None,
            artifact,
            None,
        )

    gate1 = _read_object(parent / "gate1.json", "gate1.json")
    if gate1.get("status") != "no_closure":
        raise ValueError("continuation requires an exact Gate1 no_closure parent")
    normal = gate1.get("normal")
    comparison = normal.get("comparison") if isinstance(normal, Mapping) else None
    source_compile = gate1.get("source_compile")
    replay_compile = normal.get("compile") if isinstance(normal, Mapping) else None
    if not isinstance(comparison, Mapping) or comparison.get("hard_pass") is not False:
        raise ValueError("parent Gate1 lacks one failed normal comparison")
    rank = comparison.get("rank_tuple")
    if not isinstance(rank, list):
        raise ValueError("parent Gate1 comparison lacks rank_tuple")
    source_ttir = (
        source_compile.get("ttir") if isinstance(source_compile, Mapping) else None
    )
    replay_ttir = (
        replay_compile.get("ttir") if isinstance(replay_compile, Mapping) else None
    )
    if not isinstance(source_ttir, str) or not isinstance(replay_ttir, str):
        raise ValueError("parent Gate1 lacks source/replay TTIR for feedback")

    class SavedComparison:
        def to_dict(self) -> dict[str, Any]:
            return dict(comparison)

    observer_diff = _compact_gate1_observer_diff(
        SavedComparison(),
        source_fingerprint=fingerprint_ttir(source_ttir),
        replay_fingerprint=fingerprint_ttir(replay_ttir),
    )
    return LiftFeedback(
        FeedbackKind.GATE1,
        {
            "status": gate1["status"],
            "diagnostic": normal.get("diagnostic"),
            "observer_diff": observer_diff,
        },
        _formal_round(parent),
        artifact,
        tuple(rank),
    )


def _emit(value: Mapping[str, Any], *, json_mode: bool) -> None:
    encoded = json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    if json_mode:
        print(encoded)
    else:
        print(
            f"{value['case_id']}: {value['status']} -> "
            f"{value['evidence_path']}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.max_pre_replay_rejections is None:
        args.max_pre_replay_rejections = (
            0
            if args.lift_backend == "miniswe"
            else 3
        )
    if args.max_gate1_rounds < 1:
        print("max-gate1-rounds must be positive", file=sys.stderr)
        return 2
    if not 1 <= args.max_lift_calls <= MAX_LIFT_CALLS:
        print(
            f"max-lift-calls must be in [1, {MAX_LIFT_CALLS}]",
            file=sys.stderr,
        )
        return 2
    if args.max_pre_replay_rejections < 0:
        print("max-pre-replay-rejections must be non-negative", file=sys.stderr)
        return 2

    run_dir: Path | None = None
    try:
        adapter = TRITON_WORKLOAD_ADAPTERS.get(args.case)
        if adapter is None:
            raise ValueError("case has no active Triton workload adapter")
        prepared = prepare_triton_workload(
            adapter,
            dataset_root=args.dataset_root,
        )
        seed = _load_seed(args.parent_evidence, prepared)
        provider = get_provider(args.provider)
        run_dir = next_workload_run_dir(args.run_root, args.case)
        rank = list(seed.selection_rank or ())
        initialize_workload_evidence(
            run_dir,
            prepared,
            provider=provider,
            effort=args.effort,
            run_masked_j=args.run_masked_j,
            lift_backend=args.lift_backend,
            replay_backend=args.replay_backend,
            max_lift_calls=args.max_lift_calls,
            search_metadata={
                "protocol": RANKED_CONTINUATION_PROTOCOL,
                "parent_evidence": str(args.parent_evidence.expanduser().resolve()),
                "preserves_previous_artifact": True,
                "initial_selection_rank": rank,
                "failed_parent_policy": "branch-owner-latest-candidate-v1",
                "replay_width": args.replay_width,
                "max_lift_calls": args.max_lift_calls,
                "replay_selection": "exact-pass-then-gate1-rank-v1",
                "retry_parent_gate1": args.retry_parent_gate1,
            },
        )

        def progress(
            stage: str,
            attempt: int | None,
            formal_round: int | None,
            status: str,
        ) -> None:
            print(
                json.dumps(
                    {
                        "case_id": args.case,
                        "stage": stage,
                        "attempt": attempt,
                        "formal_round": formal_round,
                        "status": status,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        search = search_prepared_workload_gate1(
            prepared,
            provider=provider,
            initial_feedback=seed,
            effort=args.effort,
            run_masked_j=args.run_masked_j,
            lift_backend=args.lift_backend,
            replay_backend=args.replay_backend,
            replay_width=args.replay_width,
            retry_initial_artifact=args.retry_parent_gate1,
            max_gate1_rounds=args.max_gate1_rounds,
            max_lift_calls=args.max_lift_calls,
            max_pre_replay_rejections=args.max_pre_replay_rejections,
            progress_callback=progress,
        )
        if search.report.closed:
            execution = complete_prepared_workload_gate2(prepared, search)
            save_workload_execution(run_dir, prepared, execution)
            status = execution.report.main_status.value
        else:
            save_workload_gate1_search(
                run_dir,
                prepared,
                search,
                selected_for_gate2=False,
            )
            status = search.report.status.value
    except Exception as exc:
        if run_dir is not None:
            save_workload_failure(run_dir, exc)
        print(
            f"ranked Gate1 continuation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    payload = {
        "protocol": RANKED_CONTINUATION_PROTOCOL,
        "case_id": args.case,
        "parent_evidence": str(args.parent_evidence.expanduser().resolve()),
        "evidence_path": str(run_dir),
        "status": status,
        "max_lift_calls": args.max_lift_calls,
    }
    _emit(payload, json_mode=args.json)
    return 0 if status == MainStatus.SCRC_CERTIFIED.value else 1


if __name__ == "__main__":
    raise SystemExit(main())
