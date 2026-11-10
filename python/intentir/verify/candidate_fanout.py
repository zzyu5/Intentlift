"""Bounded same-case candidate concurrency through full Gate2 selection."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Mapping

from .workload_manifest import SCRCWorkloadEntry


CANDIDATE_FANOUT_PROTOCOL = "intentir.scrc.candidate-fanout.v4"
CANDIDATE_SELECTION_RULE = "first_observed_full_closure_cooperative_stop"
# The public batch runner deliberately keeps this wave bounded.  The runner
# itself used to accept a much larger implementation-only range, which made it
# easy for the CLI and the actual executor to disagree about the intended
# resource envelope.  Eight is the normal wave; twelve is the explicit upper
# bound for a larger wave.
DEFAULT_CANDIDATE_FANOUT = 8
MAX_CANDIDATE_FANOUT = 12


class CandidateBranchStatus(str, Enum):
    GATE1_CLOSED = "gate1_closed"
    GATE1_CLOSED_NOT_SELECTED = "gate1_closed_not_selected"
    GATE2_CLOSED = "gate2_closed"
    GATE2_CLOSED_NOT_SELECTED = "gate2_closed_not_selected"
    GATE2_FAILED = "gate2_failed"
    NO_CLOSURE = "no_closure"
    SUPERSEDED = "superseded"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


@dataclass(frozen=True)
class CandidateBranchOutcome:
    branch_id: int
    case_id: str
    ordinal: int
    status: CandidateBranchStatus
    diagnostic: str = ""
    evidence_path: str | None = None
    call_budget: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)
    formal_round: int | None = None
    runtime: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.branch_id, int) or self.branch_id < 1:
            raise ValueError("candidate branch_id must be positive")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("candidate case_id must be non-empty")
        if not isinstance(self.ordinal, int) or self.ordinal < 1:
            raise ValueError("candidate ordinal must be positive")
        if not isinstance(self.status, CandidateBranchStatus):
            object.__setattr__(self, "status", CandidateBranchStatus(self.status))
        closed_statuses = {
            CandidateBranchStatus.GATE1_CLOSED,
            CandidateBranchStatus.GATE1_CLOSED_NOT_SELECTED,
            CandidateBranchStatus.GATE2_CLOSED,
            CandidateBranchStatus.GATE2_CLOSED_NOT_SELECTED,
            CandidateBranchStatus.GATE2_FAILED,
        }
        if self.status in closed_statuses:
            if not isinstance(self.formal_round, int) or self.formal_round < 1:
                raise ValueError(
                    "Gate1-closed candidate must identify a positive formal_round"
                )
        elif self.formal_round is not None:
            raise ValueError(
                "only a Gate1-closed candidate may identify formal_round"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "status": self.status.value,
            "diagnostic": self.diagnostic,
            "evidence_path": self.evidence_path,
            "call_budget": dict(self.call_budget),
            "details": dict(self.details),
            "formal_round": self.formal_round,
        }

    @property
    def selection_coordinate(self) -> tuple[int, int]:
        if self.formal_round is None:
            raise ValueError(
                "only a Gate1-closed candidate has a selection coordinate"
            )
        return (self.branch_id, self.formal_round)


@dataclass(frozen=True)
class CandidateFanoutSearchReport:
    case_id: str
    ordinal: int
    branches: tuple[CandidateBranchOutcome, ...]
    winner_branch_id: int | None = None
    winner_formal_round: int | None = None
    terminal_failure_branch_id: int | None = None
    selection_rule: str = CANDIDATE_SELECTION_RULE

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("candidate fan-out requires at least one branch")
        branch_ids = [item.branch_id for item in self.branches]
        if branch_ids != list(range(1, len(self.branches) + 1)):
            raise ValueError("candidate branches must be ordered and gap-free")
        if any(
            item.case_id != self.case_id or item.ordinal != self.ordinal
            for item in self.branches
        ):
            raise ValueError("candidate branch belongs to another workload")
        selected = [
            item.branch_id
            for item in self.branches
            if item.status is CandidateBranchStatus.GATE2_CLOSED
        ]
        closed = [
            item
            for item in self.branches
            if item.status
            in {
                CandidateBranchStatus.GATE1_CLOSED,
                CandidateBranchStatus.GATE1_CLOSED_NOT_SELECTED,
            }
        ]
        if self.winner_branch_id is None:
            if selected or self.winner_formal_round is not None:
                raise ValueError("fan-out report omitted its selected winner")
        elif selected != [self.winner_branch_id]:
            raise ValueError("fan-out report winner differs from selected branch")
        elif self.winner.formal_round != self.winner_formal_round:
            raise ValueError("fan-out report winner formal round differs")
        elif any(
            item.status is CandidateBranchStatus.GATE2_CLOSED_NOT_SELECTED
            and item.details.get("winner_branch_id")
            not in {None, self.winner_branch_id}
            for item in closed
        ):
            raise ValueError("fan-out branch records another first-closure winner")
        if self.terminal_failure_branch_id is not None:
            if not isinstance(self.terminal_failure_branch_id, int) or self.terminal_failure_branch_id < 1:
                raise ValueError("terminal Gate2 failure branch id must be positive")
            terminal = next(
                (item for item in self.branches if item.branch_id == self.terminal_failure_branch_id),
                None,
            )
            if terminal is None or terminal.status is not CandidateBranchStatus.GATE2_FAILED:
                raise ValueError("terminal Gate2 failure must identify a failed branch")

    @property
    def winner(self) -> CandidateBranchOutcome | None:
        if self.winner_branch_id is None:
            return None
        return next(
            item for item in self.branches
            if item.branch_id == self.winner_branch_id
        )

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(item.status.value for item in self.branches)
        return dict(sorted(counts.items()))

    @property
    def aggregate_call_budget(self) -> dict[str, int]:
        totals: Counter[str] = Counter()
        for branch in self.branches:
            totals.update(branch.call_budget)
        return dict(sorted(totals.items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CANDIDATE_FANOUT_PROTOCOL,
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "candidate_count": len(self.branches),
            "selection_rule": self.selection_rule,
            "winner_branch_id": self.winner_branch_id,
            "winner_formal_round": self.winner_formal_round,
            "terminal_failure_branch_id": self.terminal_failure_branch_id,
            "status_counts": self.counts,
            "aggregate_call_budget": self.aggregate_call_budget,
            "branches": [item.to_json() for item in self.branches],
        }


CandidateStopRequested = Callable[[], int | None]
CandidateBranchExecutor = Callable[
    [SCRCWorkloadEntry, int, CandidateStopRequested], CandidateBranchOutcome
]


class ConcurrentCandidateFanoutRunner:
    """Select the first full closure and stop sibling searches cooperatively."""

    def run(
        self,
        entry: SCRCWorkloadEntry,
        executor: CandidateBranchExecutor,
        *,
        candidates: int,
    ) -> CandidateFanoutSearchReport:
        if (
            not isinstance(candidates, int)
            or not 1 <= candidates <= MAX_CANDIDATE_FANOUT
        ):
            raise ValueError(
                "candidates must be an integer in "
                f"[1, {MAX_CANDIDATE_FANOUT}]"
            )
        completed: dict[int, CandidateBranchOutcome] = {}
        winner_lock = Lock()
        winner_branch_id: int | None = None
        terminal_failure_branch_id: int | None = None

        def stop_requested() -> int | None:
            with winner_lock:
                if winner_branch_id is not None:
                    return winner_branch_id
                if terminal_failure_branch_id is not None:
                    return -terminal_failure_branch_id
                return None

        def observe_first_closure(outcome: CandidateBranchOutcome) -> None:
            nonlocal winner_branch_id, terminal_failure_branch_id
            with winner_lock:
                if (
                    outcome.status is CandidateBranchStatus.GATE2_FAILED
                    and winner_branch_id is None
                    and terminal_failure_branch_id is None
                ):
                    terminal_failure_branch_id = outcome.branch_id
                elif (
                    outcome.status is CandidateBranchStatus.GATE2_CLOSED
                    and winner_branch_id is None
                    and terminal_failure_branch_id is None
                ):
                    winner_branch_id = outcome.branch_id

        with ThreadPoolExecutor(
            max_workers=candidates,
            thread_name_prefix="scrc-candidate",
        ) as pool:
            future_to_branch = {
                pool.submit(executor, entry, branch_id, stop_requested): branch_id
                for branch_id in range(1, candidates + 1)
            }
            for future in as_completed(future_to_branch):
                branch_id = future_to_branch[future]
                try:
                    outcome = future.result()
                    if outcome.branch_id != branch_id:
                        raise ValueError(
                            "candidate executor returned another branch id"
                        )
                    if outcome.case_id != entry.case_id:
                        raise ValueError(
                            "candidate executor returned another case id"
                        )
                    if outcome.ordinal != entry.ordinal:
                        raise ValueError(
                            "candidate executor returned another ordinal"
                        )
                    if (
                        outcome.status
                        is CandidateBranchStatus.GATE1_CLOSED_NOT_SELECTED
                    ):
                        raise ValueError(
                            "candidate executor returned an output-only status"
                        )
                except Exception as exc:
                    outcome = CandidateBranchOutcome(
                        branch_id,
                        entry.case_id,
                        entry.ordinal,
                        CandidateBranchStatus.INFRASTRUCTURE_FAILED,
                        diagnostic=f"{type(exc).__name__}: {exc}",
                    )
                completed[branch_id] = outcome
                observe_first_closure(outcome)

        # Siblings observe the terminal signal at closed-stage boundaries. A
        # Gate2 failure is terminal for this workload: it cannot create Lift
        # feedback or authorize another candidate wave. In-flight work may
        # finish its current bounded operation, but no later lifecycle starts.
        winner = (
            completed[winner_branch_id]
            if winner_branch_id is not None
            else None
        )
        if winner is not None:
            for branch_id, outcome in tuple(completed.items()):
                if (
                    outcome.status is CandidateBranchStatus.GATE2_CLOSED
                    and branch_id != winner.branch_id
                ):
                    completed[branch_id] = CandidateBranchOutcome(
                        outcome.branch_id,
                        outcome.case_id,
                        outcome.ordinal,
                        CandidateBranchStatus.GATE2_CLOSED_NOT_SELECTED,
                        diagnostic=outcome.diagnostic,
                        evidence_path=outcome.evidence_path,
                        call_budget=outcome.call_budget,
                        details={
                            **dict(outcome.details),
                            "winner_branch_id": winner.branch_id,
                            "winner_formal_round": winner.formal_round,
                        },
                        formal_round=outcome.formal_round,
                        runtime=outcome.runtime,
                    )
        branches = tuple(
            completed[branch_id] for branch_id in range(1, candidates + 1)
        )
        return CandidateFanoutSearchReport(
            entry.case_id,
            entry.ordinal,
            branches,
            winner_branch_id=(winner.branch_id if winner is not None else None),
            winner_formal_round=(
                winner.formal_round if winner is not None else None
            ),
            terminal_failure_branch_id=terminal_failure_branch_id,
        )


__all__ = [
    "CANDIDATE_FANOUT_PROTOCOL",
    "CANDIDATE_SELECTION_RULE",
    "DEFAULT_CANDIDATE_FANOUT",
    "MAX_CANDIDATE_FANOUT",
    "CandidateBranchOutcome",
    "CandidateBranchStatus",
    "CandidateFanoutSearchReport",
    "ConcurrentCandidateFanoutRunner",
]
