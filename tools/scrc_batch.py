#!/usr/bin/env python3
"""List, validate, or dry-run the active 28-kernel SCRC workload catalog."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import BoundedSemaphore, Lock, Thread
from time import monotonic
from typing import Any, Callable, TextIO

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.verify.scrc_batch import (
    SequentialSCRCBatchRunner,
    dry_run_executor,
)
from intentir.verify.scrc_runner import MAX_LIFT_CALLS
from intentir.verify.workload_manifest import build_workload_manifest


SCRC_PROGRESS_PROTOCOL = "intentir.scrc.progress.v1"
_PROGRESS_FIELDS = frozenset(
    {
        "protocol",
        "event",
        "case_id",
        "ordinal",
        "branch_id",
        "stage",
        "attempt",
        "formal_round",
        "elapsed",
        "status",
    }
)
_PROGRESS_EVENTS = frozenset(
    {
        "case_started",
        "stage_transition",
        "gate2_started",
        "gate2_finished",
        "case_finished",
        "heartbeat",
    }
)
_PROGRESS_STAGES = frozenset(
    {
        "case",
        "prepare",
        "candidate_search",
        "lift",
        "typecheck",
        "replay",
        "gate1",
        "gate2_wait",
        "gate2",
    }
)


@dataclass(frozen=True)
class _ProgressState:
    case_id: str
    ordinal: int
    branch_id: int | None
    stage: str
    attempt: int | None
    formal_round: int | None
    status: str


class _LiveProgressEmitter:
    """Asynchronous, source-free JSONL progress side channel for live runs."""

    _SENTINEL = object()

    def __init__(
        self,
        *,
        enabled: bool,
        stream: TextIO,
        heartbeat_interval: float = 60.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.enabled = bool(enabled)
        self._stream = stream
        self._heartbeat_interval = heartbeat_interval
        self._clock = clock
        self._state_lock = Lock()
        self._write_lock = Lock()
        self._queue: Queue[Any] = Queue()
        self._case_starts: dict[str, float] = {}
        self._active: dict[tuple[str, int | None], _ProgressState] = {}
        self._closed = False
        self._started = False
        self._thread = Thread(
            target=self._run,
            name="scrc-progress",
            daemon=True,
        )

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        try:
            self._thread.start()
        except Exception:
            self.enabled = False
            return
        self._started = True

    def close(self) -> None:
        if not self.enabled or not self._started or self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(self._SENTINEL)
            self._thread.join(timeout=5.0)
        except Exception:
            return

    @staticmethod
    def _validate_positive(value: int | None, field: str) -> None:
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"progress {field} must be a positive integer")

    @staticmethod
    def _validate_status(status: str) -> None:
        if (
            not isinstance(status, str)
            or not status
            or len(status) > 64
            or not status.replace("_", "").isalnum()
        ):
            raise ValueError("progress status must be one closed token")

    def _payload(
        self,
        *,
        event: str,
        state: _ProgressState,
        elapsed: float,
    ) -> dict[str, Any]:
        if event not in _PROGRESS_EVENTS:
            raise ValueError("unknown progress event")
        if state.stage not in _PROGRESS_STAGES:
            raise ValueError("unknown progress stage")
        self._validate_positive(state.ordinal, "ordinal")
        self._validate_positive(state.branch_id, "branch_id")
        self._validate_positive(state.attempt, "attempt")
        self._validate_positive(state.formal_round, "formal_round")
        self._validate_status(state.status)
        payload: dict[str, Any] = {
            "protocol": SCRC_PROGRESS_PROTOCOL,
            "event": event,
            "case_id": state.case_id,
            "ordinal": state.ordinal,
            "stage": state.stage,
            "elapsed": round(max(0.0, elapsed), 3),
            "status": state.status,
        }
        if state.branch_id is not None:
            payload["branch_id"] = state.branch_id
        if state.attempt is not None:
            payload["attempt"] = state.attempt
        if state.formal_round is not None:
            payload["formal_round"] = state.formal_round
        if not set(payload) <= _PROGRESS_FIELDS:
            raise AssertionError("progress payload escaped its closed field set")
        return payload

    def _enqueue_locked(self, payloads: tuple[dict[str, Any], ...]) -> None:
        if not self._closed and payloads:
            self._queue.put_nowait(payloads)

    def _safe_update(
        self,
        update: Callable[[float], tuple[dict[str, Any], ...]],
    ) -> None:
        if not self.enabled:
            return
        try:
            with self._state_lock:
                self._enqueue_locked(update(self._clock()))
        except Exception:
            # Progress construction and queueing are outside certification.
            return

    def case_started(self, case_id: str, ordinal: int) -> None:
        def update(now: float) -> tuple[dict[str, Any], ...]:
            self._case_starts[case_id] = now
            state = _ProgressState(
                case_id,
                ordinal,
                None,
                "prepare",
                None,
                None,
                "started",
            )
            self._active[(case_id, None)] = state
            return (self._payload(event="case_started", state=state, elapsed=0.0),)

        self._safe_update(update)

    def stage_transition(
        self,
        case_id: str,
        ordinal: int,
        *,
        branch_id: int | None,
        stage: str,
        attempt: int | None = None,
        formal_round: int | None = None,
        status: str,
    ) -> None:
        def update(now: float) -> tuple[dict[str, Any], ...]:
            started = self._case_starts.setdefault(case_id, now)
            state = _ProgressState(
                case_id,
                ordinal,
                branch_id,
                stage,
                attempt,
                formal_round,
                status,
            )
            self._active[(case_id, branch_id)] = state
            return (
                self._payload(
                    event="stage_transition",
                    state=state,
                    elapsed=now - started,
                ),
            )

        self._safe_update(update)

    def callback(
        self,
        case_id: str,
        ordinal: int,
        branch_id: int,
    ) -> Callable[[str, int | None, int | None, str], None] | None:
        if not self.enabled:
            return None

        def observe(
            stage: str,
            attempt: int | None,
            formal_round: int | None,
            status: str,
        ) -> None:
            self.stage_transition(
                case_id,
                ordinal,
                branch_id=branch_id,
                stage=stage,
                attempt=attempt,
                formal_round=formal_round,
                status=status,
            )

        return observe

    def finish_branch(self, case_id: str, branch_id: int) -> None:
        if not self.enabled:
            return
        try:
            with self._state_lock:
                self._active.pop((case_id, branch_id), None)
        except Exception:
            return

    def gate2_started(
        self,
        case_id: str,
        ordinal: int,
        *,
        branch_id: int,
        formal_round: int,
    ) -> None:
        self._gate2_event(
            "gate2_started",
            case_id,
            ordinal,
            branch_id=branch_id,
            formal_round=formal_round,
            status="started",
        )

    def gate2_finished(
        self,
        case_id: str,
        ordinal: int,
        *,
        branch_id: int,
        formal_round: int,
        status: str,
    ) -> None:
        self._gate2_event(
            "gate2_finished",
            case_id,
            ordinal,
            branch_id=branch_id,
            formal_round=formal_round,
            status=status,
        )

    def _gate2_event(
        self,
        event: str,
        case_id: str,
        ordinal: int,
        *,
        branch_id: int,
        formal_round: int,
        status: str,
    ) -> None:
        def update(now: float) -> tuple[dict[str, Any], ...]:
            started = self._case_starts.setdefault(case_id, now)
            state = _ProgressState(
                case_id,
                ordinal,
                branch_id,
                "gate2",
                None,
                formal_round,
                status,
            )
            self._active[(case_id, None)] = state
            return (
                self._payload(
                    event=event,
                    state=state,
                    elapsed=now - started,
                ),
            )

        self._safe_update(update)

    def case_finished(self, case_id: str, ordinal: int, *, status: str) -> None:
        def update(now: float) -> tuple[dict[str, Any], ...]:
            started = self._case_starts.pop(case_id, now)
            state = _ProgressState(
                case_id,
                ordinal,
                None,
                "case",
                None,
                None,
                status,
            )
            for key in tuple(self._active):
                if key[0] == case_id:
                    self._active.pop(key, None)
            return (
                self._payload(
                    event="case_finished",
                    state=state,
                    elapsed=now - started,
                ),
            )

        self._safe_update(update)

    def _heartbeat_payloads(self, now: float) -> tuple[dict[str, Any], ...]:
        by_case: dict[str, list[_ProgressState]] = {}
        for state in self._active.values():
            by_case.setdefault(state.case_id, []).append(state)
        selected: list[_ProgressState] = []
        for states in by_case.values():
            branches = [state for state in states if state.branch_id is not None]
            selected.extend(branches or states)
        selected.sort(
            key=lambda state: (
                state.ordinal,
                state.branch_id if state.branch_id is not None else 0,
            )
        )
        return tuple(
            self._payload(
                event="heartbeat",
                state=state,
                elapsed=now - self._case_starts.get(state.case_id, now),
            )
            for state in selected
        )

    def emit_heartbeat(self) -> None:
        def update(now: float) -> tuple[dict[str, Any], ...]:
            return self._heartbeat_payloads(now)

        self._safe_update(update)

    def _write_payloads(self, payloads: tuple[dict[str, Any], ...]) -> None:
        if not payloads:
            return
        try:
            encoded = "".join(
                json.dumps(
                    payload,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
                for payload in payloads
            )
            with self._write_lock:
                self._stream.write(encoded)
                self._stream.flush()
        except Exception:
            # Broken stderr must not alter any case lifecycle or certificate.
            return

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self._heartbeat_interval)
            except Empty:
                try:
                    with self._state_lock:
                        payloads = self._heartbeat_payloads(self._clock())
                except Exception:
                    continue
                self._write_payloads(payloads)
                continue
            if item is self._SENTINEL:
                return
            self._write_payloads(item)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--validate-manifest", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="explicit case id; repeat to select a registry-ordered case set",
    )
    parser.add_argument("--tier", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--provider", default=None)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "optional model override on the selected provider endpoint; the "
            "actual model is preserved in evidence"
        ),
    )
    parser.add_argument("--effort", default=None)
    parser.add_argument(
        "--replay-model",
        default=None,
        help=(
            "optional source-blind Replay model override; defaults to the Lift "
            "model selected by --model"
        ),
    )
    parser.add_argument(
        "--replay-effort",
        default=None,
        help="optional Replay reasoning effort; defaults to --effort",
    )
    parser.add_argument(
        "--lift-backend",
        choices=("responses", "miniswe", "codex"),
        default="responses",
        help="candidate Lift implementation; independent from Replay backend",
    )
    parser.add_argument(
        "--lift-workflow",
        choices=("single",),
        default="single",
        help="one persistent source-aware Lift owner per candidate branch",
    )
    parser.add_argument(
        "--replay-backend",
        choices=("responses", "miniswe", "codex"),
        default="responses",
        help="source-blind Replay implementation; independent from Lift backend",
    )
    parser.add_argument(
        "--replay-width",
        type=int,
        default=1,
        help=(
            "independent source-blind Replay workers for the same typed J in "
            "each formal Gate1 round"
        ),
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="independent same-case Lift/Replay branches before one Gate2 winner",
    )
    parser.add_argument(
        "--case-workers",
        type=int,
        default=1,
        help="maximum independent case lifecycles executed concurrently",
    )
    parser.add_argument(
        "--max-branch-workers",
        type=int,
        default=8,
        help=(
            "global bound G for case_workers * candidates * replay_width"
        ),
    )
    parser.add_argument(
        "--gate2-workers",
        type=int,
        default=1,
        help="maximum selected winners executing hidden Gate2 concurrently",
    )
    parser.add_argument(
        "--gate1-rounds",
        type=int,
        default=3,
        help="bounded formal relift/Replay/Gate1 rounds per candidate branch",
    )
    parser.add_argument(
        "--max-lift-calls",
        type=int,
        default=MAX_LIFT_CALLS,
        help=(
            "maximum logical Lift invocations per candidate branch "
            f"(1-{MAX_LIFT_CALLS})"
        ),
    )
    parser.add_argument(
        "--pre-replay-rejections",
        type=int,
        default=None,
        help=(
            "optional pre-Replay rejection cap; default 0 (owner-time governed) "
            "for persistent MiniSWE/Codex Lift and 3 for legacy writers"
        ),
    )
    parser.add_argument("--skip-masked-j", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable live JSONL progress on stderr",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/intentir-scrc-task11"),
        help="evidence root (defaults outside the repository)",
    )
    return parser


def _emit(value: dict, *, output: Path | None, json_mode: bool) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    if json_mode or output is None:
        print(encoded, end="")


def main(argv: list[str] | None = None) -> int:
    argument_vector = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argument_vector)
    if args.pre_replay_rejections is None:
        args.pre_replay_rejections = (
            0
            if args.lift_backend in {"miniswe", "codex"}
            and args.lift_workflow == "single"
            else 3
        )
    concurrency_options = (
        "--candidates",
        "--case-workers",
        "--max-branch-workers",
        "--gate2-workers",
        "--replay-width",
        "--gate1-rounds",
        "--max-lift-calls",
        "--pre-replay-rejections",
    )
    explicitly_scheduled = any(
        token == option or token.startswith(option + "=")
        for token in argument_vector
        for option in concurrency_options
    )
    if explicitly_scheduled and not args.run:
        print(
            "case/branch/Gate2 concurrency options are only valid with --run",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.candidates <= 5:
        print("--candidates must be in [1, 5]", file=sys.stderr)
        return 2
    if not 1 <= args.replay_width <= 4:
        print("--replay-width must be in [1, 4]", file=sys.stderr)
        return 2
    bounded_options = (
        ("--case-workers", args.case_workers),
        ("--max-branch-workers", args.max_branch_workers),
        ("--gate2-workers", args.gate2_workers),
    )
    for option, value in bounded_options:
        if not 1 <= value <= 32:
            print(f"{option} must be in [1, 32]", file=sys.stderr)
            return 2
    if not 1 <= args.gate1_rounds <= 12:
        print("--gate1-rounds must be in [1, 12]", file=sys.stderr)
        return 2
    if not 1 <= args.max_lift_calls <= MAX_LIFT_CALLS:
        print(
            f"--max-lift-calls must be in [1, {MAX_LIFT_CALLS}]",
            file=sys.stderr,
        )
        return 2
    if not 0 <= args.pre_replay_rejections <= 12:
        print(
            "--pre-replay-rejections must be in [0, 12]",
            file=sys.stderr,
        )
        return 2
    if args.candidates != 1 and not args.run:
        print("--candidates > 1 is only valid with --run", file=sys.stderr)
        return 2
    if (
        args.run
        and args.case_workers * args.candidates * args.replay_width
        > args.max_branch_workers
    ):
        print(
            "--run requires case_workers * candidates * replay_width "
            "<= max_branch_workers",
            file=sys.stderr,
        )
        return 2
    if args.run and args.gate2_workers > args.case_workers:
        print("--gate2-workers must be <= --case-workers", file=sys.stderr)
        return 2
    manifest = build_workload_manifest()
    if args.case is not None and args.tier is not None:
        print("--case and --tier are mutually exclusive", file=sys.stderr)
        return 2
    if args.case is not None and len(args.case) != len(set(args.case)):
        print("duplicate --case values are not accepted", file=sys.stderr)
        return 2
    if args.validate_manifest and (args.case is not None or args.tier is not None):
        print(
            "--validate-manifest always validates the complete 28-item projection; "
            "--case/--tier are not accepted",
            file=sys.stderr,
        )
        return 2
    if args.run and args.case is None and args.tier is None:
        print(
            "--run requires --case or --tier; live batch does not implicitly run all 28",
            file=sys.stderr,
        )
        return 2
    try:
        if args.case is None:
            entries = manifest.select(tier=args.tier)
        else:
            entries = tuple(
                sorted(
                    (
                        manifest.select(case_id=case_id)[0]
                        for case_id in args.case
                    ),
                    key=lambda entry: entry.ordinal,
                )
            )
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.list:
        value = {
            "protocol": manifest.to_json()["protocol"],
            "total": len(entries),
            "entries": [entry.to_json() for entry in entries],
        }
        if args.json or args.output is not None:
            _emit(value, output=args.output, json_mode=args.json)
        else:
            for entry in entries:
                print(
                    f"{entry.ordinal:02d}\t{entry.case_id}\t"
                    f"{entry.representative_tier}\t{entry.source_module}:"
                    f"{entry.symbol}"
                )
        return 0

    if args.validate_manifest:
        validation = manifest.validate(dataset_root=args.dataset_root)
        value = validation.to_json()
        _emit(value, output=args.output, json_mode=args.json)
        return 0 if validation.passed else 2

    if args.dry_run:
        report = SequentialSCRCBatchRunner().run(
            entries,
            dry_run_executor(dataset_root=args.dataset_root),
            mode="dry_run",
        )
    else:
        # Live-only dependencies stay out of list/validate/dry-run imports.
        from intentir.extraction.providers.provider import get_provider
        from intentir.verify.scrc_batch import (
            BatchCaseResult,
            BatchClassification,
            ConcurrentSCRCBatchRunner,
        )
        from intentir.verify.candidate_fanout import (
            CANDIDATE_FANOUT_PROTOCOL,
            CANDIDATE_SELECTION_RULE,
            CandidateBranchOutcome,
            CandidateBranchStatus,
            ConcurrentCandidateFanoutRunner,
        )
        from intentir.verify.gate1.orchestrate import (
            SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX,
        )
        from intentir.verify.scrc_workload_run import (
            WorkloadExecution,
            classify_scrc_report,
            complete_prepared_workload_gate2,
            gate2_exposes_logical_j_pass_implementation_failure,
            initialize_workload_evidence,
            merge_fixed_artifact_replay_retry_execution,
            next_candidate_fanout_dir,
            next_workload_run_dir,
            save_candidate_fanout_report,
            save_workload_execution,
            save_workload_failure,
            scrc_attempt_stage_counts,
            scrc_gate1_search_diagnostic,
            scrc_report_diagnostic,
            search_prepared_workload_gate1,
            retry_prepared_workload_gate1_fixed_artifact,
            save_workload_gate1_search,
            WorkloadExecutionError,
        )
        from intentir.verify.scrc_runner import MainStatus, SCRCRunReport
        from intentir.verify.workload_adapters import (
            TRITON_WORKLOAD_ADAPTERS,
            prepare_triton_workload,
        )

        provider = get_provider(args.provider)
        if args.model is not None:
            if not isinstance(args.model, str) or not args.model.strip():
                print("--model must be a non-empty model name", file=sys.stderr)
                return 2
            provider = replace(provider, model=args.model.strip())
        replay_provider = provider
        if args.replay_model is not None:
            if (
                not isinstance(args.replay_model, str)
                or not args.replay_model.strip()
            ):
                print("--replay-model must be a non-empty model name", file=sys.stderr)
                return 2
            replay_provider = replace(provider, model=args.replay_model.strip())
        replay_effort = (
            args.effort if args.replay_effort is None else args.replay_effort
        )
        run_root = args.run_root.expanduser().resolve()
        gate2_slots = BoundedSemaphore(args.gate2_workers)
        scheduling = {
            "case_workers": args.case_workers,
            "candidates": args.candidates,
            "max_branch_workers": args.max_branch_workers,
            "gate2_workers": args.gate2_workers,
            "gate1_rounds": args.gate1_rounds,
            "max_lift_calls": args.max_lift_calls,
            "pre_replay_rejections": args.pre_replay_rejections,
        }
        progress = _LiveProgressEmitter(
            enabled=not args.no_progress,
            stream=sys.stderr,
        )
        progress.start()

        def complete_gate2_with_progress(
            entry,
            prepared,
            search,
            *,
            branch_id,
            formal_round,
        ):
            def execute_gate2_once(current_search, current_round):
                progress.stage_transition(
                    entry.case_id,
                    entry.ordinal,
                    branch_id=None,
                    stage="gate2_wait",
                    formal_round=current_round,
                    status="waiting",
                )
                with gate2_slots:
                    progress.gate2_started(
                        entry.case_id,
                        entry.ordinal,
                        branch_id=branch_id,
                        formal_round=current_round,
                    )
                    try:
                        current_execution = complete_prepared_workload_gate2(
                            prepared, current_search
                        )
                    except Exception:
                        progress.gate2_finished(
                            entry.case_id,
                            entry.ordinal,
                            branch_id=branch_id,
                            formal_round=current_round,
                            status="infrastructure_failed",
                        )
                        raise
                    progress.gate2_finished(
                        entry.case_id,
                        entry.ordinal,
                        branch_id=branch_id,
                        formal_round=current_round,
                        status=current_execution.report.main_status.value.lower(),
                    )
                    return current_execution

            execution = execute_gate2_once(search, formal_round)
            base_execution = execution
            gate2 = execution.report.gate2
            if not gate2_exposes_logical_j_pass_implementation_failure(gate2):
                return execution

            accepted = [
                attempt
                for attempt in search.report.attempts
                if attempt.formal_round == search.report.accepted_attempt
            ]
            if len(accepted) != 1 or accepted[0].artifact is None:
                return execution
            artifact = accepted[0].artifact
            consumed_rounds = search.report.call_budget.formal_gate1_rounds
            retry_searches = []
            retry_executions = []
            callback = progress.callback(
                entry.case_id,
                entry.ordinal,
                branch_id,
            )
            next_round_offset = consumed_rounds
            while next_round_offset < args.gate1_rounds:
                retry_search = retry_prepared_workload_gate1_fixed_artifact(
                    prepared,
                    artifact,
                    provider=provider,
                    formal_round_offset=next_round_offset,
                    effort=args.effort,
                    replay_provider=replay_provider,
                    replay_effort=replay_effort,
                    run_masked_j=not args.skip_masked_j,
                    lift_backend=args.lift_backend,
                    lift_workflow=args.lift_workflow,
                    replay_backend=args.replay_backend,
                    replay_width=args.replay_width,
                    progress_callback=callback,
                )
                retry_searches.append(retry_search)
                replay_rounds = (
                    retry_search.report.call_budget.formal_gate1_rounds
                )
                if replay_rounds == 0:
                    break
                next_round_offset += replay_rounds
                if not retry_search.report.closed:
                    continue
                execution = execute_gate2_once(
                    retry_search,
                    retry_search.report.accepted_attempt,
                )
                retry_executions.append(execution)
                if not gate2_exposes_logical_j_pass_implementation_failure(
                    execution.report.gate2
                ):
                    break

            return merge_fixed_artifact_replay_retry_execution(
                base_execution,
                tuple(retry_searches),
                tuple(retry_executions),
            )

        def execution_result(
            entry,
            prepared,
            execution,
            evidence_path,
            *,
            extra_details=None,
        ):
            result = execution.report
            classification = classify_scrc_report(result)
            gate2_counts = (
                result.gate2.status_counts if result.gate2 is not None else {}
            )
            details = {
                "accepted_attempt": result.accepted_attempt,
                "attempt_stage_counts": scrc_attempt_stage_counts(result),
                "gate2_status_counts": gate2_counts,
                "observer_profile": entry.observer_profile,
                "q_policy": entry.q_policy,
            }
            if extra_details is not None:
                details.update(extra_details)
            return BatchCaseResult(
                entry.case_id,
                entry.ordinal,
                classification,
                entry.representative_tier,
                diagnostic=scrc_report_diagnostic(result),
                source_file=prepared.source_location,
                main_status=result.main_status.value,
                j_necessity_status=(
                    result.j_necessity_status.value
                    if result.j_necessity_status is not None
                    else None
                ),
                evidence_path=evidence_path,
                call_budget=result.call_budget.to_json(),
                details=details,
            )

        def failed_execution_result(
            entry,
            prepared,
            run_dir,
            evidence_path,
            exc,
        ):
            runtime = exc if isinstance(exc, WorkloadExecutionError) else None
            evidence = getattr(exc, "evidence", None)
            save_workload_failure(
                run_dir,
                exc,
                evidence=evidence if isinstance(evidence, dict) else None,
                runtime=runtime,
            )
            cause = runtime.cause if runtime is not None else exc
            return BatchCaseResult(
                entry.case_id,
                entry.ordinal,
                BatchClassification.INFRASTRUCTURE_FAILED,
                entry.representative_tier,
                diagnostic=f"{type(cause).__name__}: {cause}",
                source_file=prepared.source_location,
                evidence_path=evidence_path,
            )

        def single_candidate_executor(entry, prepared):
            run_dir = next_workload_run_dir(run_root, entry.case_id)
            evidence_path = "run-root://" + run_dir.relative_to(run_root).as_posix()
            initialize_workload_evidence(
                run_dir,
                prepared,
                provider=provider,
                effort=args.effort,
                replay_provider=replay_provider,
                replay_effort=replay_effort,
                run_masked_j=not args.skip_masked_j,
                lift_backend=args.lift_backend,
                lift_workflow=args.lift_workflow,
                replay_backend=args.replay_backend,
                max_lift_calls=args.max_lift_calls,
            )
            callback = progress.callback(entry.case_id, entry.ordinal, 1)
            try:
                search = search_prepared_workload_gate1(
                    prepared,
                    provider=provider,
                    effort=args.effort,
                    replay_provider=replay_provider,
                    replay_effort=replay_effort,
                    run_masked_j=not args.skip_masked_j,
                    lift_backend=args.lift_backend,
                    lift_workflow=args.lift_workflow,
                    replay_backend=args.replay_backend,
                    replay_width=args.replay_width,
                    max_gate1_rounds=args.gate1_rounds,
                    max_lift_calls=args.max_lift_calls,
                    max_pre_replay_rejections=args.pre_replay_rejections,
                    progress_callback=callback,
                )
            except Exception as exc:
                return failed_execution_result(
                    entry, prepared, run_dir, evidence_path, exc
                )
            finally:
                progress.finish_branch(entry.case_id, 1)
            try:
                if search.report.closed:
                    execution = complete_gate2_with_progress(
                        entry,
                        prepared,
                        search,
                        branch_id=1,
                        formal_round=search.report.accepted_attempt,
                    )
                else:
                    execution = WorkloadExecution(
                        SCRCRunReport(
                            MainStatus.NO_CLOSURE,
                            search.report.attempts,
                            None,
                            None,
                            None,
                            search.report.call_budget,
                            stage_metrics=search.report.stage_metrics,
                        ),
                        search.lift,
                        search.replay_writers,
                    )
            except Exception as exc:
                return failed_execution_result(
                    entry, prepared, run_dir, evidence_path, exc
                )
            save_workload_execution(run_dir, prepared, execution)
            return execution_result(entry, prepared, execution, evidence_path)

        def fanout_details(fanout):
            budget_complete = all(
                item.status is not CandidateBranchStatus.INFRASTRUCTURE_FAILED
                and item.runtime is not None
                for item in fanout.branches
            )
            return {
                "candidate_count": len(fanout.branches),
                "candidate_selection_rule": fanout.selection_rule,
                "winner_branch_id": fanout.winner_branch_id,
                "winner_formal_round": fanout.winner_formal_round,
                "candidate_status_counts": fanout.counts,
                "candidate_observed_call_budget": fanout.aggregate_call_budget,
                "candidate_call_budget_complete": budget_complete,
            }

        def fanout_no_closure_result(entry, prepared, fanout, fanout_dir):
            evidence_path = (
                "run-root://" + fanout_dir.relative_to(run_root).as_posix()
            )
            details = {
                **fanout_details(fanout),
                "observer_profile": entry.observer_profile,
                "q_policy": entry.q_policy,
            }
            complete_no_closure = all(
                item.status is CandidateBranchStatus.NO_CLOSURE
                and item.runtime is not None
                for item in fanout.branches
            )
            if not complete_no_closure:
                diagnostics = [
                    f"branch {item.branch_id} {item.status.value}: "
                    f"{item.diagnostic or 'no diagnostic'}"
                    for item in fanout.branches
                    if item.status
                    is not CandidateBranchStatus.NO_CLOSURE
                    or item.runtime is None
                ]
                diagnostic = "; ".join(diagnostics) or (
                    "candidate wave did not produce a complete NO_CLOSURE result"
                )
                return BatchCaseResult(
                    entry.case_id,
                    entry.ordinal,
                    BatchClassification.INFRASTRUCTURE_FAILED,
                    entry.representative_tier,
                    diagnostic=diagnostic,
                    source_file=prepared.source_location,
                    evidence_path=evidence_path,
                    call_budget={},
                    details={
                        **details,
                        "aggregate_call_budget_complete": False,
                    },
                )
            diagnostics = [
                f"branch {item.branch_id}: {item.diagnostic}"
                for item in fanout.branches
                if item.diagnostic
            ]
            observer_unsupported = all(
                bool(item.diagnostic)
                and item.diagnostic.startswith(
                    SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX
                )
                for item in fanout.branches
            )
            return BatchCaseResult(
                entry.case_id,
                entry.ordinal,
                (
                    BatchClassification.OBSERVER_UNSUPPORTED
                    if observer_unsupported
                    else BatchClassification.NO_CLOSURE
                ),
                entry.representative_tier,
                diagnostic="; ".join(diagnostics),
                source_file=prepared.source_location,
                main_status=MainStatus.NO_CLOSURE.value,
                evidence_path=evidence_path,
                call_budget=fanout.aggregate_call_budget,
                details={
                    **details,
                    "aggregate_call_budget_complete": True,
                },
            )

        def fanout_executor(entry, prepared):
            fanout_dir = next_candidate_fanout_dir(run_root, entry.case_id)
            branch_dirs = {
                branch_id: fanout_dir / f"candidate-{branch_id:02d}"
                for branch_id in range(1, args.candidates + 1)
            }
            for branch_id, branch_dir in branch_dirs.items():
                branch_dir.mkdir()
                initialize_workload_evidence(
                    branch_dir,
                    prepared,
                    provider=provider,
                    effort=args.effort,
                    replay_provider=replay_provider,
                    replay_effort=replay_effort,
                    run_masked_j=not args.skip_masked_j,
                    lift_backend=args.lift_backend,
                    lift_workflow=args.lift_workflow,
                    replay_backend=args.replay_backend,
                    max_lift_calls=args.max_lift_calls,
                    search_metadata={
                        "protocol": CANDIDATE_FANOUT_PROTOCOL,
                        "branch_id": branch_id,
                        "candidate_count": args.candidates,
                        "replay_width": args.replay_width,
                        "max_lift_calls": args.max_lift_calls,
                        "selection_rule": CANDIDATE_SELECTION_RULE,
                    },
                )

            def candidate_executor(candidate_entry, branch_id, stop_requested):
                branch_dir = branch_dirs[branch_id]
                evidence_path = (
                    "run-root://" + branch_dir.relative_to(run_root).as_posix()
                )
                callback = progress.callback(
                    candidate_entry.case_id,
                    candidate_entry.ordinal,
                    branch_id,
                )
                try:
                    search = search_prepared_workload_gate1(
                        prepared,
                        provider=provider,
                        effort=args.effort,
                        replay_provider=replay_provider,
                        replay_effort=replay_effort,
                        run_masked_j=not args.skip_masked_j,
                        lift_backend=args.lift_backend,
                        lift_workflow=args.lift_workflow,
                        replay_backend=args.replay_backend,
                        replay_width=args.replay_width,
                        max_gate1_rounds=args.gate1_rounds,
                        max_lift_calls=args.max_lift_calls,
                        max_pre_replay_rejections=(
                            args.pre_replay_rejections
                        ),
                        stop_requested=stop_requested,
                        progress_callback=callback,
                    )
                except Exception as exc:
                    runtime = (
                        exc if isinstance(exc, WorkloadExecutionError) else None
                    )
                    evidence = getattr(exc, "evidence", None)
                    save_workload_failure(
                        branch_dir,
                        exc,
                        evidence=(evidence if isinstance(evidence, dict) else None),
                        runtime=runtime,
                    )
                    cause = runtime.cause if runtime is not None else exc
                    return CandidateBranchOutcome(
                        branch_id,
                        candidate_entry.case_id,
                        candidate_entry.ordinal,
                        CandidateBranchStatus.INFRASTRUCTURE_FAILED,
                        diagnostic=f"{type(cause).__name__}: {cause}",
                        evidence_path=evidence_path,
                    )
                finally:
                    progress.finish_branch(candidate_entry.case_id, branch_id)
                if search.report.superseded:
                    status = CandidateBranchStatus.SUPERSEDED
                elif search.report.closed:
                    status = CandidateBranchStatus.GATE1_CLOSED
                else:
                    status = CandidateBranchStatus.NO_CLOSURE
                return CandidateBranchOutcome(
                    branch_id,
                    candidate_entry.case_id,
                    candidate_entry.ordinal,
                    status,
                    diagnostic=scrc_gate1_search_diagnostic(search.report),
                    evidence_path=evidence_path,
                    call_budget=search.report.call_budget.to_json(),
                    details={
                        "accepted_attempt": search.report.accepted_attempt,
                        "attempt_stage_counts": scrc_attempt_stage_counts(
                            search.report
                        ),
                        "observer_profile": entry.observer_profile,
                        "stop_reason": search.report.stop_reason,
                        "winner_branch_id": search.report.winner_branch_id,
                        "stop_observed_at": search.report.stop_observed_at,
                    },
                    formal_round=(
                        search.report.accepted_attempt
                        if search.report.closed else None
                    ),
                    runtime=(search, branch_dir),
                )

            fanout = ConcurrentCandidateFanoutRunner().run(
                entry,
                candidate_executor,
                candidates=args.candidates,
            )
            winner = fanout.winner
            for branch in fanout.branches:
                if branch.runtime is None or branch is winner:
                    continue
                search, branch_dir = branch.runtime
                save_workload_gate1_search(
                    branch_dir,
                    prepared,
                    search,
                    selected_for_gate2=False,
                )

            winner_result = None
            if winner is not None:
                search, winner_dir = winner.runtime
                evidence_path = winner.evidence_path
                try:
                    execution = complete_gate2_with_progress(
                        entry,
                        prepared,
                        search,
                        branch_id=winner.branch_id,
                        formal_round=winner.formal_round,
                    )
                except Exception as exc:
                    winner_result = failed_execution_result(
                        entry,
                        prepared,
                        winner_dir,
                        evidence_path,
                        exc,
                    )
                    winner_result = replace(
                        winner_result,
                        call_budget={},
                        details={
                            **fanout_details(fanout),
                            "aggregate_call_budget_complete": False,
                        },
                    )
                else:
                    save_workload_execution(winner_dir, prepared, execution)
                    winner_result = execution_result(
                        entry,
                        prepared,
                        execution,
                        evidence_path,
                        extra_details=fanout_details(fanout),
                    )
                    details = dict(winner_result.details)
                    budget_complete = details["candidate_call_budget_complete"]
                    aggregate_budget = {}
                    if budget_complete:
                        aggregate_budget = dict(fanout.aggregate_call_budget)
                        for key, full_value in winner_result.call_budget.items():
                            aggregate_budget[key] = (
                                aggregate_budget.get(key, 0)
                                + full_value
                                - winner.call_budget.get(key, 0)
                            )
                    winner_result = replace(
                        winner_result,
                        call_budget=aggregate_budget,
                        details={
                            **details,
                            "aggregate_call_budget_complete": budget_complete,
                        },
                    )
            case_result = (
                winner_result
                if winner_result is not None
                else fanout_no_closure_result(
                    entry, prepared, fanout, fanout_dir
                )
            )
            save_candidate_fanout_report(
                fanout_dir,
                fanout,
                winner_result=(
                    winner_result.to_json() if winner_result is not None else None
                ),
                lift_backend=args.lift_backend,
                replay_backend=args.replay_backend,
                max_lift_calls=args.max_lift_calls,
            )
            return case_result

        def live_executor(entry):
            result = None
            progress.case_started(entry.case_id, entry.ordinal)
            try:
                adapter = TRITON_WORKLOAD_ADAPTERS.get(entry.case_id)
                if adapter is None:
                    result = BatchCaseResult(
                        entry.case_id,
                        entry.ordinal,
                        BatchClassification.HARNESS_INVALID,
                        entry.representative_tier,
                        diagnostic="no real SCRC Harness adapter is registered",
                    )
                    return result
                try:
                    prepared = prepare_triton_workload(
                        adapter, dataset_root=args.dataset_root
                    )
                except Exception as exc:
                    result = BatchCaseResult(
                        entry.case_id,
                        entry.ordinal,
                        BatchClassification.HARNESS_INVALID,
                        entry.representative_tier,
                        diagnostic=f"{type(exc).__name__}: {exc}",
                    )
                    return result
                progress.stage_transition(
                    entry.case_id,
                    entry.ordinal,
                    branch_id=None,
                    stage="prepare",
                    status="finished",
                )
                progress.stage_transition(
                    entry.case_id,
                    entry.ordinal,
                    branch_id=None,
                    stage="candidate_search",
                    status="started",
                )
                if args.candidates == 1:
                    result = single_candidate_executor(entry, prepared)
                else:
                    result = fanout_executor(entry, prepared)
                return result
            finally:
                progress.case_finished(
                    entry.case_id,
                    entry.ordinal,
                    status=(
                        result.classification.value
                        if result is not None
                        else "infrastructure_failed"
                    ),
                )

        try:
            report = ConcurrentSCRCBatchRunner(
                case_workers=args.case_workers
            ).run(
                entries,
                live_executor,
                mode="live",
                scheduling=scheduling,
            )
        finally:
            progress.close()
    value = report.to_json()
    _emit(value, output=args.output, json_mode=args.json)
    accepted = {"ready"} if args.dry_run else {"scrc_certified"}
    return 0 if set(report.counts) <= accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
