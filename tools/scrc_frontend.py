#!/usr/bin/env python3
"""Run one registered Triton/TTIR workload through the shared SCRC lifecycle."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def _display_run_path(path: Path, root: Path) -> str:
    """Return a stable relative run reference without exposing host paths."""

    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return "run-root://external"
    relative_text = relative.as_posix()
    return "run-root://" + (relative_text or ".")


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
        "--no-provider-fallback",
        action="store_true",
        help="stop after the configured provider failure instead of starting a fresh fallback lifecycle",
    )
    parser.add_argument(
        "--lift-backend",
        choices=("miniswe", "direct-agent"),
        default="miniswe",
        help=(
            "Lift authoring backend; default is MCP/miniswe, while "
            "direct-agent is an explicit isolated fallback"
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
        "--max-gate1-rounds",
        type=int,
        default=2,
        help=(
            "maximum formal Gate1 Replay rounds: one initial writer and at "
            "most one independent fresh retry (1-2)"
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


def _listed_cases() -> dict[str, object]:
    """List only cases executable by this shared Triton/TTIR entry point.

    CUDA, CuTe, and TileLang remain discoverable through their own Dataset and
    Gate runner paths.  Including them here would advertise cases that this
    module deliberately refuses to execute and would recreate the old
    cross-frontend routing ambiguity.
    """

    from intentir.verify.workload_manifest import build_workload_manifest

    manifest = build_workload_manifest()
    rows = {entry.case_id: entry for entry in manifest.entries}
    cases = [
        {"case_id": case_id, "frontend": "triton"}
        for case_id in sorted(rows)
    ]
    return {
        "scope": "shared-scrc-triton-ttir",
        "count": len(cases),
        "cases": cases,
        "native_frontends": {
            "cuda": "CudaCompiler + CudaGate1Runner + native Gate2",
            "cute": "CuteCompiler + CuteGate1Runner (independent CuTe path)",
            "tilelang": "TileLangCompiler + TileLangGate1Runner + native Gate2",
        },
        "native_dataset_check": (
            "run tools/frontend_dataset.py --validate separately; its cases "
            "are not accepted by the shared Triton/TTIR runner"
        ),
    }


def _provider_failure_from_evidence(evidence: dict[str, object]) -> bool:
    """Switch only on a safe, explicit Lift or Replay provider stop."""

    if (
        evidence.get("failure_class") == "lift_provider_preparation"
        and evidence.get("candidate_present") is False
        and isinstance(evidence.get("provider_failure_class"), str)
    ):
        return True
    failures = evidence.get("replay_generation_failures")
    return isinstance(failures, list) and any(
        isinstance(item, dict)
        and item.get("rule_code") == "replay_provider_generation_failed"
        and isinstance(item.get("provider_failure_class"), str)
        for item in failures
    )


def _run_once(args: argparse.Namespace, *, environment_fallback: bool = False) -> tuple[int, bool]:
    from intentir.extraction.providers.provider import (
        get_environment_provider,
        get_provider,
    )
    from intentir.verify.scrc_runner import MAX_LIFT_CALLS, MainStatus
    from intentir.verify.scrc_workload_run import (
        WorkloadExecutionError,
        classify_scrc_report,
        execute_prepared_workload,
        initialize_workload_evidence,
        next_workload_run_dir,
        safe_gate1_comparison_summary,
        safe_gate2_failure_summary,
        safe_workload_failure_diagnostic,
        safe_workload_failure_metadata,
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
        return 2, False
    if not 1 <= args.max_gate1_rounds <= 2:
        print("--max-gate1-rounds must be in [1, 2]", file=sys.stderr)
        return 2, False
    run_dir: Path | None = None
    active_provider_name: str | None = None
    frontend_started = time.monotonic()
    try:
        prepare_started = time.monotonic()
        prepared = prepare_registered_scrc_workload(
            args.case,
            dataset_root=args.dataset_root,
        )
        if prepared.request.frontend != "triton":
            raise ValueError(
                "shared SCRC frontend accepts Triton/TTIR only; "
                f"{prepared.request.frontend!r} must use its native frontend Gate runner"
            )
        prepare_elapsed_ms = (time.monotonic() - prepare_started) * 1000.0
        provider_started = time.monotonic()
        if environment_fallback:
            injected = get_environment_provider()
            provider = replace(
                injected,
                name="openai",
                model="gpt-5.6-terra",
                reasoning_effort="medium",
            )
        else:
            provider = get_provider(args.provider, args.provider_config)
        active_provider_name = provider.name
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
            max_gate1_rounds=args.max_gate1_rounds,
        )
        stage_started: dict[tuple[str, int | None, int | None], float] = {}
        progress_lock = threading.Lock()
        progress_stop = threading.Event()
        active_stage: dict[str, object] | None = None

        def progress(
            stage: str,
            attempt: int | None,
            formal_round: int | None,
            status: str,
            details: dict[str, object] | None = None,
        ) -> None:
            nonlocal active_stage
            key = (stage, attempt, formal_round)
            now = time.monotonic()
            with progress_lock:
                if status == "started":
                    stage_started[key] = now
                    active_stage = {
                        "stage": stage,
                        "attempt": attempt,
                        "formal_round": formal_round,
                        "started": now,
                    }
                elapsed_ms = (now - stage_started.get(key, now)) * 1000.0
                payload = {
                    "event": "scrc_progress",
                    "case_id": args.case,
                    "stage": stage,
                    "attempt": attempt,
                    "formal_round": formal_round,
                    "status": status,
                    "stage_elapsed_ms": round(elapsed_ms, 3),
                }
                if details:
                    payload["diagnostic"] = details
                print(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if status != "started" and active_stage is not None:
                    if (
                        active_stage["stage"],
                        active_stage["attempt"],
                        active_stage["formal_round"],
                    ) == key:
                        active_stage = None

        def emit_waiting_progress() -> None:
            while not progress_stop.wait(15.0):
                with progress_lock:
                    if active_stage is None:
                        continue
                    elapsed_ms = (time.monotonic() - active_stage["started"]) * 1000.0
                    print(
                        json.dumps(
                            {
                                "event": "scrc_progress",
                                "case_id": args.case,
                                "stage": active_stage["stage"],
                                "attempt": active_stage["attempt"],
                                "formal_round": active_stage["formal_round"],
                                "status": "waiting",
                                "stage_elapsed_ms": round(elapsed_ms, 3),
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

        heartbeat = threading.Thread(
            target=emit_waiting_progress,
            name="scrc-progress-heartbeat",
            daemon=True,
        )
        heartbeat.start()
        try:
            execution = execute_prepared_workload(
                prepared,
                provider=provider,
                effort=args.effort,
                run_masked_j=not args.no_masked_j,
                lift_backend=args.lift_backend,
                replay_backend=args.replay_backend,
                max_lift_calls=args.max_lift_calls,
                max_gate1_rounds=args.max_gate1_rounds,
                progress_callback=progress,
            )
        finally:
            progress_stop.set()
            heartbeat.join(timeout=1.0)
        save_workload_execution(run_dir, prepared, execution)
    except Exception as exc:
        runtime_error = exc if isinstance(exc, WorkloadExecutionError) else None
        safe_diagnostic = safe_workload_failure_diagnostic(exc)
        payload = {
            "failure_class": safe_diagnostic["rule_code"],
            "failure_kind": "workload_infrastructure",
            **safe_diagnostic,
            "max_lift_calls": args.max_lift_calls,
            "max_gate1_rounds": args.max_gate1_rounds,
        }
        if run_dir is None:
            payload["failure_stage"] = "workload_preparation"
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
                payload["failure_save_error_class"] = type(save_error).__name__
            payload["evidence_path"] = _display_run_path(run_dir, args.run_root)
            if isinstance(exc, WorkloadExecutionError):
                payload["evidence"] = safe_workload_failure_metadata(
                    exc.evidence
                )
                payload["stage_metrics"] = safe_workload_failure_metadata(
                    {"stage_metrics": exc.stage_metrics}
                ).get("stage_metrics", {})
            (run_dir / "infrastructure_error.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif isinstance(exc, WorkloadExecutionError):
            payload["evidence"] = safe_workload_failure_metadata(exc.evidence)
            payload["stage_metrics"] = safe_workload_failure_metadata(
                {"stage_metrics": exc.stage_metrics}
            ).get("stage_metrics", {})
        if args.json:
            _emit(payload, json_mode=True)
        else:
            print(
                "SCRC frontend run failed: "
                + safe_diagnostic["rule_code"]
                + " ("
                + safe_diagnostic["cause_class"]
                + ")",
                file=sys.stderr,
            )
        safe_evidence = (
            safe_workload_failure_metadata(runtime_error.evidence)
            if runtime_error is not None
            else {}
        )
        provider_failure = (
            active_provider_name == "cislc"
            and _provider_failure_from_evidence(safe_evidence)
        )
        return 2, provider_failure

    classification = classify_scrc_report(execution.report)
    payload = {
        "case_id": args.case,
        "frontend": prepared.request.frontend,
        "classification": classification.value,
        "main_status": execution.report.main_status.value,
        "evidence_path": _display_run_path(run_dir, args.run_root),
        "lift_backend": args.lift_backend,
        "replay_backend": args.replay_backend,
        "max_lift_calls": args.max_lift_calls,
        "max_gate1_rounds": args.max_gate1_rounds,
        "gate1_comparison_summary": safe_gate1_comparison_summary(execution.report),
        "gate2_failure_summary": safe_gate2_failure_summary(execution.report),
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
    return (0 if execution.report.main_status is MainStatus.SCRC_CERTIFIED else 1), False


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.list:
        _emit(_listed_cases(), json_mode=args.json)
        return 0

    exit_code, provider_failure = _run_once(args)
    if not provider_failure or args.no_provider_fallback:
        return exit_code

    from intentir.extraction.providers.provider import get_environment_provider

    try:
        get_environment_provider()
    except ValueError:
        print(
            "OpenAI environment transport unavailable; cislc failure is terminal",
            file=sys.stderr,
        )
        return exit_code

    # A failed cislc turn is terminal for its lifecycle.  The fallback starts
    # a new owner in a separate external root, with no candidate or session
    # inherited from the first run.  There is no third transport or retry.
    fallback_root = Path(tempfile.mkdtemp(prefix="intentir-openai-fallback-"))
    fallback_args = argparse.Namespace(**vars(args))
    fallback_args.run_root = fallback_root
    fallback_args.provider = "openai"
    fallback_args.provider_config = None
    fallback_args.effort = "medium"
    print(
        json.dumps(
            {
                "event": "provider_fallback_started",
                "provider": "fallback",
                "configuration": "process-injected",
                "run_root": "run-root://fallback",
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    fallback_exit_code, _ = _run_once(fallback_args, environment_fallback=True)
    return fallback_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
