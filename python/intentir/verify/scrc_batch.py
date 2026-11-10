"""Bounded batch shells around independent SCRC workload cases."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .workload_manifest import (
    SCRCWorkloadEntry,
    SourceResolutionStatus,
)
from .scrc_runner import MAX_LIFT_CALLS


SCRC_BATCH_PROTOCOL = "intentir.scrc.batch.v3"
MAX_CASE_WORKERS = 32

_SCHEDULING_KEYS = (
    "case_workers",
    "candidates",
    "max_branch_workers",
    "gate2_workers",
    "gate1_rounds",
    "max_lift_calls",
    "pre_replay_rejections",
)


class BatchClassification(str, Enum):
    READY = "ready"
    MANIFEST_INVALID = "manifest_invalid"
    CASE_UNAVAILABLE = "case_unavailable"
    HARNESS_INVALID = "harness_invalid"
    OBSERVER_UNSUPPORTED = "observer_unsupported"
    LIFT_PROVIDER_PREPARATION = "lift_provider_preparation"
    LIFT_RESPONSE_REJECTED = "lift_response_rejected"
    TYPECHECK_FAILED = "typecheck_failed"
    REPLAY_RESPONSE_REJECTED = "replay_response_rejected"
    REPLAY_COMPILE_FAILED = "replay_compile_failed"
    NO_CLOSURE = "no_closure"
    REFERENCE_UNSUPPORTED = "reference_unsupported"
    GATE2_FAILED = "gate2_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    SCRC_CERTIFIED = "scrc_certified"


@dataclass(frozen=True)
class BatchCaseResult:
    case_id: str
    ordinal: int
    classification: BatchClassification
    representative_tier: str
    diagnostic: str = ""
    source_file: str | None = None
    main_status: str | None = None
    j_necessity_status: str | None = None
    evidence_path: str | None = None
    call_budget: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "classification": self.classification.value,
            "representative_tier": self.representative_tier,
            "diagnostic": self.diagnostic,
            "source_file": self.source_file,
            "main_status": self.main_status,
            "j_necessity_status": self.j_necessity_status,
            "evidence_path": self.evidence_path,
            "call_budget": dict(self.call_budget),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class BatchReport:
    mode: str
    cases: tuple[BatchCaseResult, ...]
    scheduling: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scheduling = _validate_scheduling(self.scheduling)
        # Keep an owned plain dict: callers cannot mutate the report through
        # their input mapping, while standard dataclass consumers such as
        # dataclasses.asdict remain supported.
        object.__setattr__(self, "scheduling", scheduling)

    @property
    def counts(self) -> dict[str, int]:
        result = Counter(item.classification.value for item in self.cases)
        return dict(sorted(result.items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": SCRC_BATCH_PROTOCOL,
            "mode": self.mode,
            "total": len(self.cases),
            "status_counts": self.counts,
            "scheduling": dict(self.scheduling),
            "cases": [item.to_json() for item in self.cases],
        }


CaseExecutor = Callable[[SCRCWorkloadEntry], BatchCaseResult]


def _validate_case_workers(case_workers: int) -> int:
    if type(case_workers) is not int or not 1 <= case_workers <= MAX_CASE_WORKERS:
        raise ValueError(
            f"case_workers must be an integer in [1, {MAX_CASE_WORKERS}]"
        )
    return case_workers


def _validate_scheduling(
    scheduling: Mapping[str, int] | None,
) -> dict[str, int]:
    if scheduling is None:
        return {}
    if not isinstance(scheduling, Mapping):
        raise ValueError("scheduling must be a mapping")
    if not scheduling:
        return {}
    supplied = set(scheduling)
    expected = set(_SCHEDULING_KEYS)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            "scheduling must contain exactly "
            f"{list(_SCHEDULING_KEYS)!r}; missing={missing!r}, extra={extra!r}"
        )
    normalized: dict[str, int] = {}
    for key in _SCHEDULING_KEYS:
        value = scheduling[key]
        minimum = 0 if key == "pre_replay_rejections" else 1
        if type(value) is not int or value < minimum:
            requirement = "non-negative" if minimum == 0 else "positive"
            raise ValueError(
                f"scheduling.{key} must be a {requirement} integer"
            )
        normalized[key] = value
    if normalized["max_lift_calls"] > MAX_LIFT_CALLS:
        raise ValueError(
            f"scheduling.max_lift_calls must be in [1, {MAX_LIFT_CALLS}]"
        )
    if (
        normalized["case_workers"] * normalized["candidates"]
        > normalized["max_branch_workers"]
    ):
        raise ValueError(
            "scheduling requires case_workers * candidates <= "
            "max_branch_workers"
        )
    return normalized


def _infrastructure_failure(
    entry: SCRCWorkloadEntry,
    exc: Exception,
) -> BatchCaseResult:
    return BatchCaseResult(
        entry.case_id,
        entry.ordinal,
        BatchClassification.INFRASTRUCTURE_FAILED,
        entry.representative_tier,
        diagnostic=f"{type(exc).__name__}: {exc}",
    )


def _validated_case_result(
    entry: SCRCWorkloadEntry,
    executor: CaseExecutor,
) -> BatchCaseResult:
    try:
        result = executor(entry)
        if not isinstance(result, BatchCaseResult):
            raise TypeError("batch executor must return BatchCaseResult")
        if result.case_id != entry.case_id or result.ordinal != entry.ordinal:
            raise ValueError("batch executor returned a result for another case")
        return result
    except Exception as exc:  # case-local isolation is part of the contract
        return _infrastructure_failure(entry, exc)


class ConcurrentSCRCBatchRunner:
    """Run independent case lifecycles with a bounded outer worker pool."""

    def __init__(self, *, case_workers: int) -> None:
        self.case_workers = _validate_case_workers(case_workers)

    def run(
        self,
        entries: Sequence[SCRCWorkloadEntry],
        executor: CaseExecutor,
        *,
        mode: str,
        scheduling: Mapping[str, int] | None = None,
    ) -> BatchReport:
        scheduling_values = _validate_scheduling(scheduling)
        if scheduling_values and (
            scheduling_values["case_workers"] != self.case_workers
        ):
            raise ValueError(
                "scheduling.case_workers must equal runner case_workers"
            )
        indexed_entries = tuple(enumerate(entries))
        if not indexed_entries:
            return BatchReport(mode, (), scheduling_values)

        completed: dict[int, BatchCaseResult] = {}
        # A one-case live run is intentionally executed by the process main
        # thread.  The Responses model's fail-closed SSE deadline uses the
        # main-thread interval timer; wrapping this degenerate case in a
        # ThreadPoolExecutor silently disables that deadline because Python
        # signals cannot interrupt a worker thread.  Multi-case batches keep
        # their bounded parallel isolation below.
        if self.case_workers == 1 and len(indexed_entries) == 1:
            index, entry = indexed_entries[0]
            completed[index] = _validated_case_result(entry, executor)
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.case_workers, len(indexed_entries)),
                thread_name_prefix="scrc-case",
            ) as pool:
                future_to_index = {
                    pool.submit(_validated_case_result, entry, executor): index
                    for index, entry in indexed_entries
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    # _validated_case_result contains the case-local exception
                    # boundary.  This fallback covers executor-level failures such
                    # as a broken Future implementation without cancelling peers.
                    try:
                        completed[index] = future.result()
                    except Exception as exc:  # pragma: no cover - defensive only
                        completed[index] = _infrastructure_failure(
                            indexed_entries[index][1], exc
                        )

        # Completion order is proof-irrelevant.  Registry ordinals determine
        # the aggregate order; declaration position only breaks malformed ties
        # deterministically without changing a valid registry.
        ordered = tuple(
            result
            for _index, result in sorted(
                completed.items(),
                key=lambda item: (
                    indexed_entries[item[0]][1].ordinal,
                    item[0],
                ),
            )
        )
        return BatchReport(mode, ordered, scheduling_values)


class SequentialSCRCBatchRunner:
    """Compatibility shell for the bounded runner with ``case_workers=1``."""

    def run(
        self,
        entries: Sequence[SCRCWorkloadEntry],
        executor: CaseExecutor,
        *,
        mode: str,
        scheduling: Mapping[str, int] | None = None,
    ) -> BatchReport:
        return ConcurrentSCRCBatchRunner(case_workers=1).run(
            entries,
            executor,
            mode=mode,
            scheduling=scheduling,
        )


def dry_run_executor(
    *,
    dataset_root: str | Path | None = None,
) -> CaseExecutor:
    def execute(entry: SCRCWorkloadEntry) -> BatchCaseResult:
        source = entry.resolve_source(dataset_root=dataset_root)
        classification = {
            SourceResolutionStatus.RESOLVED: BatchClassification.READY,
            SourceResolutionStatus.CASE_UNAVAILABLE: (
                BatchClassification.CASE_UNAVAILABLE
            ),
            SourceResolutionStatus.MANIFEST_INVALID: (
                BatchClassification.MANIFEST_INVALID
            ),
        }[source.status]
        return BatchCaseResult(
            entry.case_id,
            entry.ordinal,
            classification,
            entry.representative_tier,
            diagnostic=source.diagnostic,
            source_file=source.location or None,
            details={
                "source_module": entry.source_module,
                "symbol": entry.symbol,
                "frontend": entry.frontend,
                "observer_profile": entry.observer_profile,
                "q_policy": entry.q_policy,
                "adapter_readiness": entry.adapter_readiness,
            },
        )

    return execute


__all__ = [
    "BatchCaseResult",
    "BatchClassification",
    "BatchReport",
    "ConcurrentSCRCBatchRunner",
    "MAX_CASE_WORKERS",
    "SCRC_BATCH_PROTOCOL",
    "SequentialSCRCBatchRunner",
    "dry_run_executor",
]
