#!/usr/bin/env python3
"""Run one registered Triton, TileLang, or CUDA workload through full SCRC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

def _parser() -> argparse.ArgumentParser:
    from intentir.verify.scrc_runner import MAX_LIFT_CALLS

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", help="registered real-workload case id")
    selection.add_argument(
        "--list",
        action="store_true",
        help="list registered cases without loading a provider",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument(
        "--provider-config",
        type=Path,
        default=None,
        help="provider JSON path; API keys are never copied into evidence",
    )
    parser.add_argument("--effort", default=None)
    parser.add_argument(
        "--lift-backend",
        choices=("responses", "miniswe", "codex"),
        default="responses",
    )
    parser.add_argument(
        "--consistency-only",
        action="store_true",
        help=(
            "stop after real Source-to-J consistency; skip Replay, Gate1, "
            "masked-J, and Gate2; use none Lift effort unless --effort overrides"
        ),
    )
    parser.add_argument(
        "--replay-backend",
        choices=("responses", "miniswe", "codex"),
        default="miniswe",
    )
    parser.add_argument(
        "--no-masked-j",
        action="store_true",
        help="diagnostic mode: skip the J-necessity masked Replay arm",
    )
    parser.add_argument(
        "--max-lift-calls",
        type=int,
        default=MAX_LIFT_CALLS,
        help=(
            "maximum logical Lift invocations for this branch "
            f"(1-{MAX_LIFT_CALLS})"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "artifacts" / "scrc-frontends",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _emit(payload: dict[str, object], *, json_mode: bool) -> None:
    if json_mode:
        print(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def _listed_cases(dataset_root: str | None = None) -> dict[str, object]:
    from intentir.verify.workload_manifest import build_workload_manifest

    manifest = build_workload_manifest()
    rows = {entry.case_id: entry for entry in manifest.entries}

    try:
        from intentir.verify.workload_registry import (
            frontend_dataset_harness_projection,
        )

        projection = frontend_dataset_harness_projection(dataset_root=dataset_root)
    except (ModuleNotFoundError, OSError, ValueError) as exc:
        projection = None
        unavailable = [
            {
                "case_id": "dataset:TianchenIR-MultiFrontend",
                "frontend": "native",
                "status": "case_unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        ]
    else:
        unavailable = [item.to_json() for item in projection.diagnostics]
    if projection is not None:
        rows.update(projection.harnesses)
    cases = [
        {
            "case_id": case_id,
            "frontend": getattr(rows[case_id], "frontend", "triton"),
        }
        for case_id in sorted(rows)
    ]
    return {
        "count": len(cases),
        "cases": cases,
        "native_unavailable_count": len(unavailable),
        "native_unavailable": unavailable,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.list:
        _emit(_listed_cases(args.dataset_root), json_mode=args.json)
        return 0

    from intentir.extraction.providers.provider import get_provider
    from intentir.verify.scrc_runner import MAX_LIFT_CALLS, MainStatus
    from intentir.verify.scrc_workload_run import (
        WorkloadExecutionError,
        classify_scrc_report,
        execute_prepared_workload,
        initialize_workload_evidence,
        next_workload_run_dir,
        run_prepared_workload_consistency,
        save_workload_failure,
        save_workload_execution,
    )
    from intentir.verify.workload_registry import prepare_registered_scrc_workload

    assert args.case is not None
    if not 1 <= args.max_lift_calls <= MAX_LIFT_CALLS:
        print(
            f"--max-lift-calls must be in [1, {MAX_LIFT_CALLS}]",
            file=sys.stderr,
        )
        return 2
    run_dir: Path | None = None
    frontend_started = time.monotonic()
    try:
        prepare_started = time.monotonic()
        prepared = prepare_registered_scrc_workload(
            args.case,
            dataset_root=args.dataset_root,
        )
        prepare_elapsed_ms = (time.monotonic() - prepare_started) * 1000.0
        provider_started = time.monotonic()
        provider = get_provider(args.provider, args.provider_config)
        provider_elapsed_ms = (time.monotonic() - provider_started) * 1000.0
        frontend_setup_metrics = {
            "prepare_registered_workload": {
                "elapsed_ms": prepare_elapsed_ms,
                "runs": 1,
                "status": "finished",
            },
            "provider_setup": {
                "elapsed_ms": provider_elapsed_ms,
                "runs": 1,
                "status": "finished",
            },
        }

        if args.consistency_only:
            if args.lift_backend != "responses":
                raise ValueError(
                    "--consistency-only currently requires --lift-backend responses"
                )
            consistency = run_prepared_workload_consistency(
                prepared,
                provider=provider,
                effort=args.effort,
                max_lift_calls=args.max_lift_calls,
            )
            payload = {
                "case_id": args.case,
                "frontend": prepared.request.frontend,
                "mode": "consistency-only",
                **consistency.to_json(),
            }
            payload["stage_metrics"] = {
                **frontend_setup_metrics,
                **dict(consistency.stage_metrics),
                "frontend_total": {
                    "elapsed_ms": (time.monotonic() - frontend_started) * 1000.0,
                    "runs": 1,
                    "status": "passed" if consistency.passed else "failed",
                },
            }
            _emit(payload, json_mode=args.json)
            return 0 if consistency.passed else 1

        run_dir = next_workload_run_dir(args.run_root, args.case)
        initialize_workload_evidence(
            run_dir,
            prepared,
            provider=provider,
            effort=args.effort,
            run_masked_j=not args.no_masked_j,
            lift_backend=args.lift_backend,
            replay_backend=args.replay_backend,
            max_lift_calls=args.max_lift_calls,
        )
        execution = execute_prepared_workload(
            prepared,
            provider=provider,
            effort=args.effort,
            run_masked_j=not args.no_masked_j,
            lift_backend=args.lift_backend,
            replay_backend=args.replay_backend,
            max_lift_calls=args.max_lift_calls,
        )
        save_workload_execution(run_dir, prepared, execution)
    except Exception as exc:
        runtime_error = exc if isinstance(exc, WorkloadExecutionError) else None
        payload = {
            "failure_class": "workload_infrastructure",
            "error": f"{type(exc).__name__}: {exc}",
            "max_lift_calls": args.max_lift_calls,
        }
        if run_dir is not None:
            try:
                save_workload_failure(
                    run_dir,
                    exc,
                    evidence=(
                        runtime_error.evidence if runtime_error is not None else None
                    ),
                    runtime=runtime_error,
                )
            except Exception as save_error:  # pragma: no cover - defensive path
                (run_dir / "failure_save_error.txt").write_text(
                    f"{type(save_error).__name__}: {save_error}\n",
                    encoding="utf-8",
                )
            payload["evidence_path"] = str(run_dir)
            if isinstance(exc, WorkloadExecutionError):
                payload["evidence"] = exc.evidence
                payload["stage_metrics"] = {
                    str(name): dict(value)
                    for name, value in exc.stage_metrics.items()
                }
            (run_dir / "infrastructure_error.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif isinstance(exc, WorkloadExecutionError):
            payload["evidence"] = exc.evidence
            payload["stage_metrics"] = {
                str(name): dict(value)
                for name, value in exc.stage_metrics.items()
            }
        if args.json:
            _emit(payload, json_mode=True)
        else:
            print(
                f"SCRC frontend run failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return 2

    classification = classify_scrc_report(execution.report)
    payload = {
        "case_id": args.case,
        "frontend": prepared.request.frontend,
        "classification": classification.value,
        "main_status": execution.report.main_status.value,
        "evidence_path": str(run_dir),
        "lift_backend": args.lift_backend,
        "replay_backend": args.replay_backend,
        "max_lift_calls": args.max_lift_calls,
        "stage_metrics": {
            **frontend_setup_metrics,
            **dict(execution.report.stage_metrics),
            "frontend_total": {
                "elapsed_ms": (time.monotonic() - frontend_started) * 1000.0,
                "runs": 1,
                "status": (
                    "passed"
                    if execution.report.main_status is MainStatus.SCRC_CERTIFIED
                    else "failed"
                ),
            },
        },
    }
    _emit(payload, json_mode=args.json)
    return 0 if execution.report.main_status is MainStatus.SCRC_CERTIFIED else 1


if __name__ == "__main__":
    raise SystemExit(main())
