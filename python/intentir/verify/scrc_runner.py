"""One active SCRC Gate runner joining Lift, TypeCheck, Gate1 and Gate2.

The runner is deliberately frontend-neutral.  Frontend compilers and source
aware providers are injected as small adapters; the lifecycle and terminal
status rules live here exactly once.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
import json
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from intentir.extraction.lifters.scrc import (
    normalize_scrc_candidate_delta,
    relift_repair_instruction,
    safe_expected_constraints,
    safe_source_repair_locators,
)
from intentir.ir.canonical_construction import (
    CanonicalConstruction,
    CanonicalHandleAllocator,
    SourceFacts,
)
from intentir.ir.source_projection_contracts import (
    gate1_observer_obligation_diagnostic,
    safe_semantic_obligation_guidance,
)
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from .gate1.model import (
    Gate1Artifact,
    Gate1Status,
    JNecessityStatus,
)
from .gate1.orchestrate import (
    Gate1ExecutionResult,
    SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX,
)
from .gate2.model import Gate2SuiteResult
from .typecheck import TypeCheckReport, TypeCheckStatus, type_check


MAX_GATE1_ROUNDS = 3
# A source-blind Replay response can be rejected before any formal TTIR
# comparison. Give that same TypeCheck-valid artifact one fresh writer in the
# same formal round. This is not another Gate1 round and never consumes Lift.
MAX_REPLAY_GENERATION_RETRIES = 1
# One logical Lift invocation produces one complete candidate artifact (or one
# rejected candidate transaction).  This is deliberately separate from the
# formal Replay/Gate1-round budget and from an adapter's physical provider
# retry count.  It is the shared temporary cap for every branch lifecycle.
# A candidate branch is intentionally bounded to three logical Lift
# invocations. The request, batch, and frontend validators all
# import this single bound so one Kernel cannot silently fall back to a long
# trial-and-error search.  Provider-internal transport retries remain separate
# telemetry and do not create another logical Lift state.
MAX_LIFT_CALLS = 3
# Zero means that pre-Replay construction is governed by the Lift owner's
# terminal resource signal rather than a semantic-attempt count.  A positive
# value remains available as an explicit diagnostic override.
MAX_PRE_REPLAY_REJECTIONS = 0
UNRESOLVED_GATE1_DIAGNOSTIC = "unresolved_gate1"
MAX_GATE1_FEEDBACK_EXAMPLES_PER_CATEGORY = 8
MAX_GATE1_FEEDBACK_PATTERNS_PER_CATEGORY = 16
MAX_GATE1_FEEDBACK_NODE_WITNESSES = 10
MAX_GATE1_FEEDBACK_EDGES_PER_NODE = 6
MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS = 512
MAX_GATE1_FEEDBACK_LOCAL_FRONTIER_EXAMPLES = 10
MAX_GATE1_FEEDBACK_STRUCTURAL_SUMMARIES = 8
# A provider exception can contain the complete rejected operation object.  It
# is useful for a human evidence record, but duplicating that unstructured
# string in every relift packet overwhelms the structured path/rule diagnostics
# that carry the actionable information.  Artifact JSON is not passed through
# this limit; only diagnostic strings are bounded.
MAX_RELIFT_DIAGNOSTIC_STRING_CHARS = 2048
MAX_RELIFT_GUIDANCE_ITEMS = 32
GATE1_LOCAL_FRONTIER_PROTOCOL = "intentir.gate1-local-frontier.v1"
_OBSERVER_NODE_FINGERPRINT = re.compile(
    r"(?<=[\[:])[0-9a-f]{12,64}(?=\])"
)
_GATE1_FEEDBACK_CATEGORIES = (
    "interface",
    "op_multiset",
    "dataflow",
    "reduction",
    "predicate_effect",
)

_SAFE_REPLAY_FAILURE_FIELDS = frozenset(
    {
        "rule_code",
        "path",
        "provider_failure_class",
        "provider_failure_origin",
        "failure_class",
        "physical_provider_calls",
        "exception_type",
        "line",
    }
)


def _replay_arms(gate1: Gate1ExecutionResult) -> tuple[Any, ...]:
    normal = gate1.normal_candidates or (gate1.normal,)
    return (*normal, *((gate1.masked_j,) if gate1.masked_j is not None else ()))


def _safe_replay_generation_failures(
    gate1: Gate1ExecutionResult,
    *,
    provider_only: bool = False,
) -> tuple[dict[str, str | int], ...]:
    """Project all normal/masked generation failures to closed metadata."""

    failures: list[dict[str, str | int]] = []
    for arm in _replay_arms(gate1):
        if arm.generation is not None:
            continue
        for diagnostic in arm.ingress_diagnostics:
            if not isinstance(diagnostic, Mapping):
                continue
            if provider_only and not (
                diagnostic.get("rule_code") == "replay_provider_generation_failed"
                and isinstance(diagnostic.get("provider_failure_class"), str)
            ):
                continue
            projected = {
                key: value
                for key, value in diagnostic.items()
                if key in _SAFE_REPLAY_FAILURE_FIELDS
                and (
                    isinstance(value, str)
                    or (isinstance(value, int) and not isinstance(value, bool))
                )
            }
            if projected:
                failures.append(projected)
    return tuple(failures)

def _is_replay_response_rejection(
    gate1: Gate1ExecutionResult,
) -> bool:
    """Recognize a writer-issued, source-blind response-rejection receipt.

    The classification deliberately comes from the structured receipt rather
    than exception text.  Provider transport/auth/runtime failures never issue
    this receipt and therefore remain infrastructure failures.
    """

    arms = gate1.normal_candidates or (gate1.normal,)
    rejected = [
        arm
        for arm in arms
        if arm.generation is None
        and (arm.diagnostic or "").startswith("generation failed:")
    ]
    return bool(rejected) and all(
        any(
            item.get("rule_code") == "replay_response_rejected"
            for item in arm.ingress_diagnostics
            if isinstance(item, Mapping)
        )
        for arm in rejected
    )


def _masked_replay_incomplete(gate1: Gate1ExecutionResult) -> bool:
    """Identify an enabled masked arm that supplied no comparable evidence.

    This is a Replay-owner failure, not a J/C/pi counterexample.  In
    particular, feeding it to Relift would make the Lift owner change an
    artifact that the successful normal arm has already supported.
    """

    return gate1.masked_j is not None and not gate1.masked_j.comparable

SCRCProgressCallback = Callable[
    [str, int | None, int | None, str, Mapping[str, Any] | None],
    None,
]

_SAFE_LIFT_PROGRESS_FAILURE_CLASSES = frozenset(
    {
        "candidate_rejected",
        "candidate_superseded",
        "lift_response",
        "lift_worker_no_candidate",
        "source_reconcile_rejected",
        "lift_authoring_preparation",
        "lift_agent_step_limit",
        "lift_agent_cost_limit",
        "lift_agent_time_limit",
        "lift_agent_format_limit",
        "lift_agent_stagnation",
        "lift_mcp_host_internal_error",
        "lift_mcp_application_error",
        "lift_mcp_no_progress",
        "lift_provider_preparation",
    }
)


def _safe_lift_progress_diagnostic(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one Lift rejection to bounded user-facing metadata."""

    safe: dict[str, Any] = {
        "phase": "lift",
        "candidate_check_started": isinstance(
            diagnostics.get("candidate_check_feedback"), Mapping
        ),
        "typecheck_started": False,
        "replay_started": False,
        "gate1_started": False,
        "gate2_started": False,
    }
    failure_class = diagnostics.get("failure_class")
    if (
        isinstance(failure_class, str)
        and failure_class in _SAFE_LIFT_PROGRESS_FAILURE_CLASSES
    ):
        safe["failure_class"] = failure_class
    candidate_present = diagnostics.get("candidate_present")
    if isinstance(candidate_present, bool):
        safe["candidate_present"] = candidate_present
    for key in (
        "rule_code",
        "rule_path",
        "repair_scope",
        "state_digest",
        "next_legal_action",
    ):
        value = diagnostics.get(key)
        if isinstance(value, str) and re.fullmatch(
            r"[A-Za-z0-9_.:\[\]/-]{1,256}", value
        ):
            safe[key] = value
    if isinstance(diagnostics.get("progress_changed"), bool):
        safe["progress_changed"] = diagnostics["progress_changed"]
    for source_key in ("candidate_check_feedback", "source_reconcile"):
        source = diagnostics.get(source_key)
        rows = source.get("diagnostics") if isinstance(source, Mapping) else None
        if not isinstance(rows, list):
            continue
        projected: list[dict[str, str]] = []
        for row in rows[:12]:
            if not isinstance(row, Mapping):
                continue
            rule_code = row.get("rule_code")
            path = row.get("path")
            if not isinstance(rule_code, str) or re.fullmatch(
                r"[a-z0-9_.:-]{1,96}", rule_code
            ) is None:
                continue
            item = {"rule_code": rule_code}
            if isinstance(path, str) and re.fullmatch(
                r"(?:j|pi)\.[A-Za-z0-9_.\[\]]{1,120}|<(?:candidate|canonical-constructor)>",
                path,
            ):
                item["path"] = path
            projected.append(item)
        if projected:
            safe[source_key] = {"diagnostics": projected}
    return safe


def _notify_progress(
    callback: SCRCProgressCallback | None,
    stage: str,
    attempt: int | None,
    formal_round: int | None,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Best-effort runtime observation that can never alter a gate verdict."""

    if callback is None:
        return
    try:
        callback(stage, attempt, formal_round, status, details)
    except Exception:
        # Progress is deliberately outside the certificate and evidence TCB.
        # Broken output streams or observers cannot reject or accept a case.
        return


def _feedback_item(value: Any) -> Any:
    """Remove non-actionable observer node ids from relift-only diagnostics."""

    if isinstance(value, str):
        return _OBSERVER_NODE_FINGERPRINT.sub("node", value)
    if isinstance(value, list):
        return [_feedback_item(item) for item in value]
    if isinstance(value, tuple):
        return [_feedback_item(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _feedback_item(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _prune_relift_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Retain the current frontier without recursively replaying old envelopes."""

    copied = json.loads(json.dumps(dict(value), allow_nan=False))
    envelope_keys = {UNRESOLVED_GATE1_DIAGNOSTIC}

    def visit(item: Any, depth: int) -> Any:
        if isinstance(item, str):
            marker = "...<truncated>"
            if len(item) <= MAX_RELIFT_DIAGNOSTIC_STRING_CHARS:
                return item
            return (
                item[: MAX_RELIFT_DIAGNOSTIC_STRING_CHARS - len(marker)]
                + marker
            )
        if isinstance(item, list):
            return [visit(child, depth + 1) for child in item]
        if not isinstance(item, dict):
            return item
        result: dict[str, Any] = {}
        for key, child in item.items():
            key = str(key)
            if depth > 0 and key in envelope_keys:
                continue
            result[key] = visit(child, depth + 1)
        return result

    result = visit(copied, 0)
    if not isinstance(result, dict):
        raise TypeError("relift diagnostics must remain an object")
    return result


def _guidance_key(item: Mapping[str, Any]) -> tuple[str, str]:
    obligation_id = item.get("obligation_id")
    if isinstance(obligation_id, str) and obligation_id:
        return ("obligation", obligation_id)
    return (
        str(item.get("rule_code", "")),
        str(item.get("path", "")),
    )


def _guidance_item(
    *,
    source: str,
    stage: int,
    rule_code: Any,
    path: Any,
    repair_locator: Any = None,
    repair_locators: Any = None,
    observed_refs_or_types: Any = None,
    expected_constraints: Any = None,
    obligation: Any = None,
) -> dict[str, Any] | None:
    if not isinstance(rule_code, str) or not rule_code:
        return None
    if not isinstance(path, str) or not path:
        path = "<candidate>"
    row = {
        "source": source,
        "stage": stage,
        "rule_code": rule_code[:MAX_RELIFT_DIAGNOSTIC_STRING_CHARS],
        "path": path[:MAX_RELIFT_DIAGNOSTIC_STRING_CHARS],
    }
    safe_obligation = safe_semantic_obligation_guidance(obligation)
    if safe_obligation is not None:
        obligation_stage = safe_obligation.pop("stage")
        row.update(safe_obligation)
        row["obligation_stage"] = obligation_stage
    instruction = relift_repair_instruction(rule_code, path)
    if instruction is not None:
        row["repair_instruction"] = instruction
    safe_constraints = safe_expected_constraints(
        expected_constraints, accept_unmarked=True
    )
    safe_constraints.extend(safe_expected_constraints(observed_refs_or_types))
    if safe_constraints:
        row["expected_constraints"] = safe_constraints
    locators = safe_source_repair_locators(
        [
            *(
                repair_locators
                if isinstance(repair_locators, (list, tuple))
                else ([repair_locators] if repair_locators is not None else [])
            ),
            repair_locator,
        ]
    )
    if locators:
        row["repair_locators"] = locators
    return row


def _diagnostic_guidance_items(
    diagnostics: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Extract a bounded, backend-neutral open-guidance frontier.

    This reads diagnostics only.  It never infers a semantic repair or edits a
    candidate.  The same compact rows are carried by Responses, MiniSWE, and
    Codex so their transport differences do not change relift meaning.
    """

    if not isinstance(diagnostics, Mapping):
        return []
    rows: list[dict[str, Any]] = []

    def append_rows(
        payload: Any,
        *,
        source: str,
        stage: int,
    ) -> None:
        if not isinstance(payload, Mapping):
            return
        raw = payload.get("diagnostics")
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            row = _guidance_item(
                source=source,
                stage=stage,
                rule_code=item.get("rule_code", item.get("code")),
                path=item.get("path"),
                repair_locator=item.get("repair_locator"),
                repair_locators=item.get("repair_locators"),
                observed_refs_or_types=item.get("observed_refs_or_types"),
                expected_constraints=item.get("expected_constraints"),
                obligation=item,
            )
            if row is not None and row.get("status") not in {
                "satisfied",
                "preflight_passed",
            }:
                rows.append(row)

    append_rows(
        diagnostics.get("candidate_check"),
        source="candidate_check",
        stage=0,
    )
    append_rows(
        diagnostics.get("source_reconcile"),
        source="source_reconcile",
        stage=1,
    )
    append_rows(
        diagnostics.get("typecheck"),
        source="typecheck",
        stage=1,
    )

    observer = diagnostics.get("observer_diff")
    categories = (
        observer.get("categories")
        if isinstance(observer, Mapping)
        else None
    )
    if isinstance(categories, list):
        for category in categories:
            if not isinstance(category, Mapping):
                continue
            # A passing Gate1 category is evidence that the corresponding
            # obligation is no longer open.  It must not be re-emitted as a
            # fresh repair item on the next relift.
            if category.get("passed") is True:
                continue
            name = category.get("category")
            if not isinstance(name, str) or not name:
                continue
            row = _guidance_item(
                source="gate1",
                stage=2,
                rule_code=f"gate1.{name}",
                path=f"observer_diff.categories[{name}]",
                obligation=category,
            )
            if row is not None:
                rows.append(row)

    semantic_probe = diagnostics.get("semantic_probe")
    if isinstance(semantic_probe, Mapping) and (
        semantic_probe.get("status") == "diverged"
        or semantic_probe.get("repairable") is True
    ):
        row = _guidance_item(
            source="semantic_probe",
            stage=1,
            rule_code=(
                semantic_probe.get("rule_code")
                if semantic_probe.get("repairable") is True
                else "semantic_probe_diverged"
            ),
            path=(
                semantic_probe.get("path")
                if semantic_probe.get("repairable") is True
                else "semantic_probe"
            ),
        )
        if row is not None:
            rows.append(row)

    if not rows:
        diagnostic = diagnostics.get("diagnostic")
        if isinstance(diagnostic, str) and diagnostic:
            row = _guidance_item(
                source="lifecycle",
                stage=0,
                rule_code=str(diagnostics.get("failure_class", "lift")),
                path="<candidate>",
            )
            if row is not None:
                rows.append(row)

    unique: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _guidance_key(row)
        prior = seen.get(key)
        if prior is not None:
            locators = safe_source_repair_locators(
                [
                    *prior.get("repair_locators", ()),
                    prior.get("repair_locator"),
                    *row.get("repair_locators", ()),
                    row.get("repair_locator"),
                ]
            )
            prior.pop("repair_locator", None)
            if locators:
                prior["repair_locators"] = locators
            constraints = safe_expected_constraints(
                [
                    *(prior.get("expected_constraints", ())),
                    *(row.get("expected_constraints", ())),
                    *(prior.get("observed_refs_or_types", ())),
                    *(row.get("observed_refs_or_types", ())),
                ],
                accept_unmarked=True,
            )
            if constraints:
                prior["expected_constraints"] = constraints
            continue
        seen[key] = row
        unique.append(row)
        if len(unique) >= MAX_RELIFT_GUIDANCE_ITEMS:
            break
    return unique


def _merge_relift_guidance(
    previous: "LiftFeedback | None",
    diagnostics: Mapping[str, Any] | None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Merge a new frontier without forgetting unresolved prior obligations.

    Absence from one diagnostic packet is not proof that an old obligation was
    repaired.  Resolution is recorded only when the lifecycle has crossed the
    corresponding successful prerequisite, or when Gate1 explicitly reports
    that the same category passed.  This keeps one backend from accidentally
    dropping a still-open Source/J or observer obligation just because a later
    packet focused on a different failure.
    """

    current = _diagnostic_guidance_items(diagnostics)
    previous_open = [
        dict(item)
        for item in getattr(previous, "open_guidance", ())
        if isinstance(item, Mapping)
    ]
    previous_resolved = [
        dict(item)
        for item in getattr(previous, "resolved_guidance", ())
        if isinstance(item, Mapping)
    ]
    previous_resolved_keys = {
        _guidance_key(item) for item in previous_resolved
    }
    current_with_regressions: list[dict[str, Any]] = []
    for item in current:
        key = _guidance_key(item)
        if key in previous_resolved_keys:
            # Keep the historical resolved row, but make a real later failure
            # actionable instead of silently hiding it.  This is a regression
            # marker, not a replacement of the resolved history.
            item = dict(item)
            item["status"] = "regressed"
            item["regresses"] = {
                "rule_code": key[0],
                "path": key[1],
            }
        current_with_regressions.append(item)
    current = current_with_regressions
    current_by_key = {_guidance_key(item): item for item in current}
    resolved_by_progress: set[tuple[str, str]] = set()

    def collect_satisfied_obligations(value: Any, *, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(value, Mapping):
            obligation = safe_semantic_obligation_guidance(value)
            if obligation is not None and obligation.get("status") == "satisfied":
                resolved_by_progress.add(
                    ("obligation", obligation["obligation_id"])
                )
            for child in value.values():
                collect_satisfied_obligations(child, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_satisfied_obligations(child, depth=depth + 1)

    # An obligation is resolved only by an explicit host validator receipt.
    # Reaching a later lifecycle stage or omitting a previous diagnostic is
    # not evidence that the exact Source relation was discharged.
    collect_satisfied_obligations(diagnostics)

    observer = diagnostics.get("observer_diff") if isinstance(diagnostics, Mapping) else None
    categories = observer.get("categories") if isinstance(observer, Mapping) else None
    if isinstance(categories, list):
        for category in categories:
            if not isinstance(category, Mapping) or category.get("passed") is not True:
                continue
            name = category.get("category")
            if isinstance(name, str) and name:
                resolved_by_progress.add(
                    (f"gate1.{name}", f"observer_diff.categories[{name}]")
                )
    semantic_probe = diagnostics.get("semantic_probe") if isinstance(diagnostics, Mapping) else None
    if isinstance(semantic_probe, Mapping) and semantic_probe.get("status") == "pass":
        resolved_by_progress.update(
            _guidance_key(item)
            for item in previous_open
            if item.get("source") == "semantic_probe"
        )

    open_rows: list[dict[str, Any]] = list(current)
    resolved_rows: list[dict[str, Any]] = list(previous_resolved)
    resolved_keys = {_guidance_key(item) for item in resolved_rows}
    open_keys = {_guidance_key(item) for item in open_rows}

    for item in previous_open:
        key = _guidance_key(item)
        if key in current_by_key or key in open_keys:
            continue
        if key in resolved_by_progress:
            resolved = dict(item)
            resolved["status"] = "resolved"
            resolved["resolution"] = "host_validator_receipt"
            if key not in resolved_keys:
                resolved_rows.append(resolved)
                resolved_keys.add(key)
        else:
            open_rows.append(item)
            open_keys.add(key)

    def dedupe(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        result: list[dict[str, Any]] = []
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = _guidance_key(row)
            prior = seen.get(key)
            if prior is not None:
                locators = safe_source_repair_locators(
                    [
                        *prior.get("repair_locators", ()),
                        prior.get("repair_locator"),
                        row.get("repair_locators", row.get("repair_locator")),
                    ]
                )
                prior.pop("repair_locator", None)
                if locators:
                    prior["repair_locators"] = locators
                constraints = safe_expected_constraints(
                    [
                        *(prior.get("expected_constraints", ())),
                        *(row.get("expected_constraints", ())),
                        *(prior.get("observed_refs_or_types", ())),
                        *(row.get("observed_refs_or_types", ())),
                    ],
                    accept_unmarked=True,
                )
                if constraints:
                    prior["expected_constraints"] = constraints
                continue
            seen[key] = row
            result.append(row)
            # The lifecycle state is authoritative and must retain every
            # prior open/resolved obligation across the bounded three-round
            # search.  Provider-facing projections apply their own compact
            # limit; truncating here would make a later backend forget work.
        return tuple(result)

    return dedupe(open_rows), dedupe(resolved_rows)


def _gate1_category_node_count(differences: Sequence[Any]) -> int:
    """Count affected structural nodes/items without exposing observer hashes."""

    fingerprints: set[str] = set()
    structural_items: set[str] = set()
    for difference in differences:
        item = getattr(difference, "item", None)
        tokens = _observer_fingerprints(item)
        if tokens:
            fingerprints.update(tokens)
            continue
        try:
            structural_items.add(
                json.dumps(
                    item,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    default=str,
                )
            )
        except (TypeError, ValueError):
            structural_items.add(repr(item))
    return len(fingerprints) or len(structural_items)


def _gate1_category_summary(comparison: Any) -> list[dict[str, Any]]:
    """Return the fixed five-category Gate1 diagnostic summary.

    The summary is derived from the existing hard components.  It is a
    diagnostic projection only: no score, rank, hard verdict, or candidate
    selection decision is recomputed here.
    """

    components = {
        str(component.category): component
        for component in getattr(comparison, "components", ())
    }
    summary: list[dict[str, Any]] = []
    for category in _GATE1_FEEDBACK_CATEGORIES:
        component = components.get(category)
        differences = tuple(getattr(component, "differences", ()))
        passed = bool(component.passed) if component is not None else False
        summary.append(
            {
                "category": category,
                "passed": passed,
                "applicable": (
                    bool(component.applicable) if component is not None else False
                ),
                "difference_count": len(differences),
                "node_count": _gate1_category_node_count(differences),
                **gate1_observer_obligation_diagnostic(
                    category,
                    passed=passed,
                ),
            }
        )
    return summary


def _observer_fingerprints(value: Any) -> tuple[str, ...]:
    """Return observer-only recursive node fingerprints in deterministic order."""

    if isinstance(value, str):
        return tuple(_OBSERVER_NODE_FINGERPRINT.findall(value))
    if isinstance(value, (list, tuple)):
        return tuple(
            fingerprint
            for item in value
            for fingerprint in _observer_fingerprints(item)
        )
    if isinstance(value, Mapping):
        return tuple(
            fingerprint
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            for fingerprint in _observer_fingerprints(item)
        )
    return ()


def _observer_node_aliases(
    differences: list[Mapping[str, Any]],
) -> dict[str, str]:
    """Pseudonymize node identity while preserving within-feedback incidence."""

    sides: dict[str, set[str]] = {}
    for difference in differences:
        side = (
            "source"
            if difference.get("kind") == "missing"
            else "replay"
            if difference.get("kind") == "unexpected"
            else "shared"
        )
        for fingerprint in _observer_fingerprints(difference.get("item")):
            sides.setdefault(fingerprint, set()).add(side)

    counters = {"source": 0, "replay": 0, "shared": 0}
    aliases: dict[str, str] = {}
    for fingerprint in sorted(sides):
        observed_sides = sides[fingerprint]
        only_side = (
            next(iter(observed_sides)) if len(observed_sides) == 1 else None
        )
        side = (
            only_side
            if only_side in {"source", "replay"}
            else "shared"
        )
        aliases[fingerprint] = f"{side}_node_{counters[side]}"
        counters[side] += 1
    return aliases


def _pseudonymize_observer_nodes(
    value: Any,
    aliases: Mapping[str, str],
) -> Any:
    """Replace opaque hashes with repeatable side-qualified local aliases."""

    if isinstance(value, str):
        return _OBSERVER_NODE_FINGERPRINT.sub(
            lambda match: aliases.get(match.group(0), "node"),
            value,
        )
    if isinstance(value, list):
        return [_pseudonymize_observer_nodes(item, aliases) for item in value]
    if isinstance(value, tuple):
        return [_pseudonymize_observer_nodes(item, aliases) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _pseudonymize_observer_nodes(item, aliases)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _observer_node_witnesses(
    differences: list[Mapping[str, Any]],
    aliases: Mapping[str, str],
    diagnostic_signatures: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Expose bounded incident-edge witnesses for the first divergent nodes."""

    incident: dict[str, list[Mapping[str, Any]]] = {}
    labels: dict[str, str] = {}
    primary_consumers: list[str] = []
    ordered_tokens: list[str] = []
    for difference in differences:
        item = difference.get("item")
        if (
            difference.get("category") != "dataflow"
            or not isinstance(item, (list, tuple))
            or len(item) != 4
            or item[0] != "edge"
        ):
            continue
        # Prefer the consumer: one changed consumer hash otherwise causes a
        # cascade of indistinguishable family-only missing/unexpected edges.
        ordered_endpoints = (item[2], item[1])
        edge_tokens: list[str] = []
        for endpoint_index, endpoint in enumerate(ordered_endpoints):
            endpoint_tokens = _observer_fingerprints(endpoint)
            for token in endpoint_tokens:
                if token not in edge_tokens:
                    edge_tokens.append(token)
                if token not in ordered_tokens:
                    ordered_tokens.append(token)
                if (
                    endpoint_index == 0
                    and token not in primary_consumers
                    and len(primary_consumers) < 2
                ):
                    primary_consumers.append(token)
                labels.setdefault(
                    token,
                    str(_pseudonymize_observer_nodes(endpoint, aliases)),
                )
        for token in edge_tokens:
            incident.setdefault(token, []).append(difference)

    # Start with the first source/replay consumer pair, then walk its upstream
    # producers before spending the witness budget on an unrelated
    # lexicographically early edge.  Recursive observer hashes make the
    # earliest producer divergence much more useful than a broad family list;
    # each selected witness still retains both incoming and outgoing edges.
    seeds = list(primary_consumers)
    seed_index = 0
    while seed_index < len(seeds) and len(seeds) < MAX_GATE1_FEEDBACK_NODE_WITNESSES:
        token = seeds[seed_index]
        seed_index += 1
        incoming_neighbors: list[str] = []
        for edge in incident.get(token, ()):
            item = edge.get("item")
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                continue
            producers = _observer_fingerprints(item[1])
            consumers = _observer_fingerprints(item[2])
            if token in consumers:
                for neighbor in producers:
                    if neighbor != token and neighbor not in incoming_neighbors:
                        incoming_neighbors.append(neighbor)
        for neighbor in incoming_neighbors:
            if neighbor in seeds:
                continue
            seeds.append(neighbor)
            if len(seeds) >= MAX_GATE1_FEEDBACK_NODE_WITNESSES:
                break
    for token in ordered_tokens:
        if len(seeds) >= MAX_GATE1_FEEDBACK_NODE_WITNESSES:
            break
        if token not in seeds:
            seeds.append(token)

    witnesses: list[dict[str, Any]] = []
    for token in seeds:
        edges = incident.get(token, ())
        witness: dict[str, Any] = {
            "node": labels[token],
            "incident_differences": [
                {
                    key: _pseudonymize_observer_nodes(edge[key], aliases)
                    for key in (
                        "kind",
                        "item",
                        "expected_count",
                        "actual_count",
                    )
                    if key in edge
                }
                for edge in edges[:MAX_GATE1_FEEDBACK_EDGES_PER_NODE]
            ],
            "truncated_count": max(
                0, len(edges) - MAX_GATE1_FEEDBACK_EDGES_PER_NODE
            ),
        }
        signatures = diagnostic_signatures.get(token, {})
        source_signature = signatures.get("source")
        replay_signature = signatures.get("replay")
        if source_signature is not None and source_signature == replay_signature:
            witness["local_signature"] = source_signature
        else:
            if source_signature is not None:
                witness["source_local_signature"] = source_signature
            if replay_signature is not None:
                witness["replay_local_signature"] = replay_signature
        witnesses.append(witness)
    return witnesses


def _observer_diagnostic_signatures(
    source_fingerprint: Any | None,
    replay_fingerprint: Any | None,
) -> dict[str, dict[str, str]]:
    """Index ephemeral normalized signatures by their opaque node digest."""

    result: dict[str, dict[str, str]] = {}
    for side, fingerprint in (
        ("source", source_fingerprint),
        ("replay", replay_fingerprint),
    ):
        entries = getattr(fingerprint, "diagnostic_node_signatures", ())
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
            ):
                continue
            signature = entry[1][:MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS]
            for token in _observer_fingerprints(entry[0]):
                result.setdefault(token, {}).setdefault(side, signature)
    return result


def _observer_diagnostic_profiles(
    source_fingerprint: Any | None,
    replay_fingerprint: Any | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index diagnostic-only immediate profiles by opaque comparison token."""

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for side, fingerprint in (
        ("source", source_fingerprint),
        ("replay", replay_fingerprint),
    ):
        entries = getattr(fingerprint, "diagnostic_node_profiles", ())
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 4
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
                or not isinstance(entry[2], (list, tuple))
                or not isinstance(entry[3], (list, tuple))
                or not all(
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], str)
                    for item in entry[2]
                )
                or not all(isinstance(item, str) for item in entry[3])
            ):
                continue
            profile = _feedback_item(
                {
                    "local_signature": entry[1][
                        :MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS
                    ],
                    "inputs": [
                        {
                            "role": role,
                            "producer": producer[
                                :MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS
                            ],
                        }
                        for role, producer in entry[2]
                    ],
                    "context": [
                        item[:MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS]
                        for item in entry[3]
                    ],
                }
            )
            for token in _observer_fingerprints(entry[0]):
                result.setdefault(token, {})[side] = profile
    return {
        token: profiles
        for token, profiles in result.items()
        if not (
            profiles.get("source") is not None
            and profiles.get("source") == profiles.get("replay")
        )
    }


def _observer_diagnostic_representation_profiles(
    source_fingerprint: Any | None,
    replay_fingerprint: Any | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index hash-free origins of representation carriers contracted by Gate1."""

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for side, fingerprint in (
        ("source", source_fingerprint),
        ("replay", replay_fingerprint),
    ):
        entries = getattr(
            fingerprint,
            "diagnostic_representation_profiles",
            (),
        )
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
                or not isinstance(entry[2], (list, tuple))
                or not all(isinstance(item, str) for item in entry[2])
            ):
                continue
            profile = _feedback_item(
                {
                    "result_contract": entry[1][
                        :MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS
                    ],
                    "origins": [
                        item[:MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS]
                        for item in entry[2]
                    ],
                }
            )
            for token in _observer_fingerprints(entry[0]):
                result.setdefault(token, {})[side] = profile
    return {
        token: profiles
        for token, profiles in result.items()
        if not (
            profiles.get("source") is not None
            and profiles.get("source") == profiles.get("replay")
        )
    }


def _observer_category_node_profiles(
    differences: list[Mapping[str, Any]],
    aliases: Mapping[str, str],
    diagnostic_profiles: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Explain nodes named by one hard category without exposing Source text.

    The hard observer already names these opaque nodes in its difference.  This
    sidecar merely joins each directly named node to the observer's non-recursive
    local profile.  It is excluded from comparison, ranking, and Replay input.
    """

    witnesses: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for difference in differences:
        kind = str(difference.get("kind", "unknown"))
        preferred_side = (
            "source"
            if kind == "missing"
            else "replay"
            if kind == "unexpected"
            else None
        )
        for token in _observer_fingerprints(difference.get("item")):
            profiles = diagnostic_profiles.get(token, {})
            sides = (
                (preferred_side,)
                if preferred_side in profiles
                else tuple(sorted(profiles))
            )
            for side in sides:
                if side is None or (side, token) in seen:
                    continue
                profile = profiles.get(side)
                if profile is None:
                    continue
                seen.add((side, token))
                witnesses.append(
                    {
                        "side": side,
                        "kind": kind,
                        "difference_role": _observer_difference_role(difference),
                        "node": aliases.get(token, f"{side}_node"),
                        **profile,
                    }
                )
                if len(witnesses) >= MAX_GATE1_FEEDBACK_NODE_WITNESSES:
                    return witnesses
    return witnesses


def _observer_local_profile_counter(
    fingerprint: Any | None,
) -> tuple[Counter[str], dict[str, dict[str, Any]]]:
    """Index non-recursive diagnostic node profiles without observer hashes."""

    counts: Counter[str] = Counter()
    payloads: dict[str, dict[str, Any]] = {}
    entries = getattr(fingerprint, "diagnostic_node_profiles", ())
    if not isinstance(entries, (list, tuple)):
        return counts, payloads
    for entry in entries:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 4
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], str)
            or not isinstance(entry[2], (list, tuple))
            or not isinstance(entry[3], (list, tuple))
            or not all(
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
                for item in entry[2]
            )
            or not all(isinstance(item, str) for item in entry[3])
        ):
            continue
        profile: dict[str, Any] = {
            "local_signature": entry[1][
                :MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS
            ],
            "inputs": [
                {
                    "role": role,
                    "producer": producer[
                        :MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS
                    ],
                }
                for role, producer in entry[2]
            ],
            "context": [
                item[:MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS]
                for item in entry[3]
            ],
        }
        # Be defensive if a future observer profile embeds an opaque digest.
        # The diagnostic stays hash-free even though the hard observer does not.
        profile = _feedback_item(profile)
        key = json.dumps(
            profile,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        counts[key] += 1
        payloads.setdefault(key, profile)
    return counts, payloads


def _observer_local_dataflow_frontier(
    source_fingerprint: Any | None,
    replay_fingerprint: Any | None,
) -> dict[str, Any] | None:
    """Return a bounded non-recursive frontier for a failed hard dataflow diff.

    This sidecar never participates in comparison or candidate selection.  It
    removes the recursive-hash cascade from the diagnostic view so Lift can see
    which immediate producer/consumer profiles actually differ.
    """

    source, source_payloads = _observer_local_profile_counter(source_fingerprint)
    replay, replay_payloads = _observer_local_profile_counter(replay_fingerprint)
    if not source and not replay:
        return None

    source_signatures = Counter(
        source_payloads[key]["local_signature"]
        for key, count in source.items()
        for _ in range(count)
    )
    replay_signatures = Counter(
        replay_payloads[key]["local_signature"]
        for key, count in replay.items()
        for _ in range(count)
    )
    differences: list[dict[str, Any]] = []
    for key in source.keys() | replay.keys():
        expected = source[key]
        actual = replay[key]
        if expected == actual:
            continue
        profile = source_payloads.get(key) or replay_payloads[key]
        assert profile is not None
        differences.append(
            {
                "kind": "missing" if actual < expected else "unexpected",
                "profile": profile,
                "expected_count": expected,
                "actual_count": actual,
                "_paired_signature": bool(
                    source_signatures[profile["local_signature"]]
                    and replay_signatures[profile["local_signature"]]
                ),
            }
        )

    differences.sort(
        key=lambda item: (
            0 if item["_paired_signature"] else 1,
            len(item["profile"].get("context", ())),
            item["profile"]["local_signature"],
            0 if item["kind"] == "missing" else 1,
            json.dumps(item["profile"], sort_keys=True, ensure_ascii=False),
        )
    )
    examples = []
    for difference in differences[:MAX_GATE1_FEEDBACK_LOCAL_FRONTIER_EXAMPLES]:
        example = dict(difference)
        example.pop("_paired_signature", None)
        examples.append(example)
    return {
        "protocol": GATE1_LOCAL_FRONTIER_PROTOCOL,
        "authority": (
            "diagnostic-only non-recursive profiles excluded from Gate1 "
            "membership, artifact construction, and candidate selection"
        ),
        "source_profile_count": sum(source.values()),
        "replay_profile_count": sum(replay.values()),
        "difference_count": len(differences),
        "examples": examples,
        "truncated_count": max(0, len(differences) - len(examples)),
        "diagnostic": (
            "These are immediate local producer/consumer differences after "
            "removing recursive node-hash fallout. Empty differences with a "
            "hard dataflow failure indicate same-signature occurrence wiring "
            "or higher-level region composition, not a Gate1 pass."
        ),
    }


def _observer_edge_endpoint(value: Any) -> str:
    """Collapse an observer node label to a hash-free structural family."""

    normalized = str(_feedback_item(value))
    if normalized.startswith("region["):
        payload = normalized.removeprefix("region[").split("]", 1)[0]
        return "region[" + payload.split(":", 1)[0] + "]"
    if normalized.startswith("arg["):
        return normalized
    return normalized.split("[", 1)[0]


def _observer_edge_pattern(difference: Mapping[str, Any]) -> tuple[str, ...] | None:
    item = difference.get("item")
    if (
        not isinstance(item, (list, tuple))
        or len(item) != 4
        or item[0] != "edge"
    ):
        return None
    return (
        str(difference.get("kind", "unknown")),
        _observer_edge_endpoint(item[1]),
        _observer_edge_endpoint(item[2]),
        str(_feedback_item(item[3])),
    )


def _observer_difference_role(difference: Mapping[str, Any]) -> str:
    """Classify one difference by its structural observer role."""

    item = difference.get("item")
    if isinstance(item, (list, tuple)) and item:
        if len(item) == 4 and item[0] == "edge":
            return str(_feedback_item(item[3]))
        return str(_feedback_item(item[0]))
    return "item"


def _observer_role_priority(role: str) -> tuple[int, str]:
    """Put causal control/effect witnesses ahead of bulk operand fallout."""

    if role == "effect.order":
        return (0, role)
    if role in {
        "condition",
        "positive",
        "negative",
        "true",
        "false",
        "positive_yield",
        "negative_yield",
    }:
        return (1, role)
    if role in {"region.child", "region.member"}:
        return (2, role)
    if role.startswith("region_result") or role.startswith("function.return"):
        return (3, role)
    if role.startswith("operand"):
        return (8, role)
    return (5, role)


def _select_bounded_observer_differences(
    differences: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Select causal-role-diverse differences for bounded feedback views."""

    ordered = sorted(
        enumerate(differences),
        key=lambda pair: (
            _observer_role_priority(_observer_difference_role(pair[1])),
            0 if pair[1].get("kind") == "missing" else 1,
            pair[0],
        ),
    )
    selected: list[Mapping[str, Any]] = []
    represented: set[tuple[str, str]] = set()
    for _index, difference in ordered:
        key = (
            str(difference.get("kind", "unknown")),
            _observer_difference_role(difference),
        )
        if key in represented:
            continue
        represented.add(key)
        selected.append(difference)
        if len(selected) >= MAX_GATE1_FEEDBACK_EXAMPLES_PER_CATEGORY:
            break
    if len(selected) < MAX_GATE1_FEEDBACK_EXAMPLES_PER_CATEGORY:
        selected_ids = {id(item) for item in selected}
        for difference in differences:
            if id(difference) in selected_ids:
                continue
            selected.append(difference)
            if len(selected) >= MAX_GATE1_FEEDBACK_EXAMPLES_PER_CATEGORY:
                break
    return selected


def _bounded_observer_examples(
    differences: list[Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Choose bounded examples by causal-role coverage, not input accident."""

    selected = _select_bounded_observer_differences(differences)
    return [
        {
            key: _pseudonymize_observer_nodes(difference[key], aliases)
            for key in ("kind", "item", "expected_count", "actual_count")
            if key in difference
        }
        for difference in selected
    ]


def _observer_structural_label(
    value: Any,
    *,
    side: str,
    diagnostic_signatures: Mapping[str, Mapping[str, str]],
) -> str:
    """Render one observer endpoint without recursive hashes or opaque aliases."""

    if not isinstance(value, str):
        return str(_feedback_item(value))
    tokens = _observer_fingerprints(value)
    if len(tokens) == 1:
        signature = diagnostic_signatures.get(tokens[0], {}).get(side)
        if isinstance(signature, str) and signature:
            return signature[:MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS]
    if value.startswith("region["):
        return "region"
    if value.startswith("arg["):
        return value
    # A structural family is more useful than ``operation[node]`` when the
    # observer profile is unavailable.  The bounded generic replacement keeps
    # the diagnostic hash-free as a fallback.
    return _OBSERVER_NODE_FINGERPRINT.sub("node", value).split("[", 1)[0]


def _observer_structural_payload(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _observer_structural_item(
    category: str,
    item: Any,
    *,
    side: str,
    diagnostic_signatures: Mapping[str, Mapping[str, str]],
) -> str:
    """Describe one comparison item as an actionable structural fact."""

    if isinstance(item, (list, tuple)) and item:
        kind = str(item[0])
        if kind == "edge" and len(item) == 4:
            source = _observer_structural_label(
                item[1], side=side, diagnostic_signatures=diagnostic_signatures
            )
            target = _observer_structural_label(
                item[2], side=side, diagnostic_signatures=diagnostic_signatures
            )
            return f"edge {source} --{item[3]}--> {target}"
        if kind == "dot" and len(item) == 2:
            payload = _observer_structural_payload(item[1])
            if isinstance(payload, Mapping):
                operation = payload.get("operation", "dot")
                signature = payload.get("signature")
                if isinstance(signature, str) and signature:
                    return f"{operation}: {signature}"
                return str(operation)
        if kind == "reduction" and len(item) == 2:
            payload = _observer_structural_payload(item[1])
            if isinstance(payload, Mapping):
                return (
                    "reduction "
                    f"axis={payload.get('axis')} "
                    f"combiners={payload.get('combiners')} "
                    f"inputs={payload.get('input_count')} "
                    f"results={payload.get('result_count')}"
                )
        if kind in {"loop", "structured_if", "terminal_fork"} and len(item) == 2:
            payload = _observer_structural_payload(item[1])
            if isinstance(payload, Mapping):
                if kind == "loop":
                    return (
                        "loop "
                        f"block_args={payload.get('block_argument_count')} "
                        f"init={payload.get('init_values')} "
                        f"yield={payload.get('yielded_values')}"
                    )
                if kind == "structured_if":
                    return (
                        "structured_if "
                        f"predicate={payload.get('predicate')} "
                        f"positive={payload.get('positive_arm')} "
                        f"negative={payload.get('negative_arm')}"
                    )
                return (
                    "terminal_fork "
                    f"predicate={payload.get('predicate')} "
                    f"true={payload.get('true_arm')} "
                    f"false={payload.get('false_arm')}"
                )
        if kind in {"memory", "mask", "effect"}:
            if kind == "memory" and len(item) == 2:
                payload = _observer_structural_payload(item[1])
                if isinstance(payload, Mapping):
                    return (
                        "memory "
                        f"kind={payload.get('kind')} "
                        f"masked={payload.get('masked')} "
                        f"predicate={payload.get('predicate')}"
                    )
            return f"{kind} {item[1]}" if len(item) == 2 else kind
        if category == "interface":
            if kind == "schema_version" and len(item) == 2:
                return f"interface schema={item[1]}"
            if kind == "argument" and len(item) >= 3:
                return f"interface argument[{item[1]}] {_observer_structural_payload(item[2])}"
            if kind == "result" and len(item) >= 3:
                return f"interface result[{item[1]}] {item[2]}"
        return _feedback_item(item).__repr__()
    if category == "op_multiset":
        return f"operation {item}"
    return _feedback_item(item).__repr__()


def _observer_structural_summary(
    difference: Mapping[str, Any],
    diagnostic_signatures: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    kind = str(difference.get("kind", "unknown"))
    category = str(difference.get("category", "unknown"))
    side = "source" if kind == "missing" else "replay"
    structure = _observer_structural_item(
        category,
        difference.get("item"),
        side=side,
        diagnostic_signatures=diagnostic_signatures,
    )
    expected = difference.get("expected_count")
    actual = difference.get("actual_count")
    if kind == "missing":
        message = (
            f"Source has {structure}; Replay is missing it "
            f"(source={expected}, replay={actual})"
        )
    elif kind == "unexpected":
        message = (
            f"Replay has {structure}; Source is missing it "
            f"(source={expected}, replay={actual})"
        )
    else:
        message = f"Source/Replay differ in {structure}"
    return {
        "kind": kind,
        "category": category,
        "structure": structure[:MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS],
        "expected_count": expected,
        "actual_count": actual,
        "message": message[:MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS],
    }


def _observer_effect_trace_index(
    fingerprint: Any | None,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Index the diagnostic-only expansion of synthetic effect traces."""

    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    entries = getattr(fingerprint, "diagnostic_effect_traces", ())
    if not isinstance(entries, (list, tuple)):
        return result
    for entry in entries:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 3
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], str)
            or not isinstance(entry[2], (list, tuple))
            or not all(isinstance(item, str) for item in entry[2])
        ):
            continue
        result[entry[0]] = (entry[1], tuple(entry[2]))
    return result


def _observer_effect_order_witnesses(
    differences: list[Mapping[str, Any]],
    aliases: Mapping[str, str],
    source_fingerprint: Any | None,
    replay_fingerprint: Any | None,
) -> list[dict[str, Any]]:
    """Explain divergent effect traces without changing Gate1 membership."""

    indexes = {
        "source": _observer_effect_trace_index(source_fingerprint),
        "replay": _observer_effect_trace_index(replay_fingerprint),
    }
    witnesses: list[dict[str, Any]] = []
    for difference in differences:
        item = difference.get("item")
        kind = str(difference.get("kind", "unknown"))
        side = (
            "source"
            if kind == "missing"
            else "replay"
            if kind == "unexpected"
            else None
        )
        if (
            side is None
            or difference.get("category") != "dataflow"
            or not isinstance(item, (list, tuple))
            or len(item) != 4
            or item[0] != "edge"
            or item[3] != "effect.order"
            or not isinstance(item[2], str)
        ):
            continue
        expansion = indexes[side].get(item[2])
        if expansion is None:
            continue
        region, events = expansion
        witnesses.append(
            {
                "side": side,
                "kind": kind,
                "region": _pseudonymize_observer_nodes(region, aliases),
                "trace": _pseudonymize_observer_nodes(item[2], aliases),
                "ordered_events": _pseudonymize_observer_nodes(
                    list(events), aliases
                ),
            }
        )
        if len(witnesses) >= MAX_GATE1_FEEDBACK_EXAMPLES_PER_CATEGORY:
            break
    return witnesses


def _compact_gate1_observer_diff(
    comparison: Any,
    *,
    source_fingerprint: Any | None = None,
    replay_fingerprint: Any | None = None,
) -> dict[str, Any]:
    """Expose a bounded frontier plus exact, category-local Gate1 evidence.

    The bounded examples remain the cheap first read.  Exact differences stay in
    their already-labelled category so the host can derive bounded rule-level
    guidance for a fresh Lift owner without injecting the bulk object into its
    prompt.  This function performs no attribution and proposes no artifact repair.
    """

    raw = comparison.to_dict()
    differences = raw.get("differences")
    if not isinstance(differences, list):
        return _feedback_item(raw)
    mapped_differences = [
        difference for difference in differences if isinstance(difference, Mapping)
    ]
    node_aliases = _observer_node_aliases(mapped_differences)
    diagnostic_signatures = _observer_diagnostic_signatures(
        source_fingerprint,
        replay_fingerprint,
    )
    diagnostic_profiles = _observer_diagnostic_profiles(
        source_fingerprint,
        replay_fingerprint,
    )
    diagnostic_representation_profiles = (
        _observer_diagnostic_representation_profiles(
            source_fingerprint,
            replay_fingerprint,
        )
    )

    grouped: dict[str, dict[str, Any]] = {}
    for difference in mapped_differences:
        category = str(difference.get("category", "unknown"))
        group = grouped.setdefault(
            category,
            {
                "category": category,
                "difference_count": 0,
                "_differences": [],
                "_edge_patterns": {},
            },
        )
        group["difference_count"] += 1
        group["_differences"].append(difference)
        pattern = _observer_edge_pattern(difference)
        if pattern is not None:
            patterns = group["_edge_patterns"]
            patterns[pattern] = patterns.get(pattern, 0) + 1

    categories = []
    for category in sorted(grouped):
        group = grouped[category]
        category_differences = group.pop("_differences")
        group["examples"] = _bounded_observer_examples(
            category_differences,
            node_aliases,
        )
        bounded_differences = _pseudonymize_observer_nodes(
            category_differences[:MAX_GATE1_FEEDBACK_EXAMPLES_PER_CATEGORY],
            node_aliases,
        )
        group["differences"] = bounded_differences
        group["differences_truncated_count"] = max(
            0,
            len(category_differences) - len(bounded_differences),
        )
        structural_differences = _select_bounded_observer_differences(
            category_differences
        )[:MAX_GATE1_FEEDBACK_STRUCTURAL_SUMMARIES]
        group["structural_summaries"] = [
            _observer_structural_summary(
                difference,
                diagnostic_signatures,
            )
            for difference in structural_differences
        ]
        group["structural_summaries_truncated_count"] = max(
            0,
            len(category_differences) - len(structural_differences),
        )
        raw_patterns = group.pop("_edge_patterns")
        if raw_patterns:
            group["edge_patterns"] = [
                {
                    "kind": pattern[0],
                    "source": pattern[1],
                    "target": pattern[2],
                    "role": pattern[3],
                    "count": count,
                }
                for pattern, count in sorted(
                    raw_patterns.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:MAX_GATE1_FEEDBACK_PATTERNS_PER_CATEGORY]
            ]
        if category == "dataflow":
            local_frontier = _observer_local_dataflow_frontier(
                source_fingerprint,
                replay_fingerprint,
            )
            if local_frontier is not None:
                group["local_frontier"] = local_frontier
            witnesses = _observer_node_witnesses(
                mapped_differences,
                node_aliases,
                diagnostic_signatures,
            )
            if witnesses:
                group["node_witnesses"] = witnesses
                group["node_alias_scope"] = "this_feedback_only"
            effect_order_witnesses = _observer_effect_order_witnesses(
                mapped_differences,
                node_aliases,
                source_fingerprint,
                replay_fingerprint,
            )
            if effect_order_witnesses:
                group["effect_order_witnesses"] = effect_order_witnesses
            representation_frontier = _observer_category_node_profiles(
                category_differences,
                node_aliases,
                diagnostic_representation_profiles,
            )
            if representation_frontier:
                group["representation_frontier"] = representation_frontier
        else:
            node_profiles = _observer_category_node_profiles(
                category_differences,
                node_aliases,
                diagnostic_profiles,
            )
            if node_profiles:
                group["node_profiles"] = node_profiles
                group["node_alias_scope"] = "this_feedback_only"
        group["truncated_count"] = max(
            0, group["difference_count"] - len(group["examples"])
        )
        group.update(
            gate1_observer_obligation_diagnostic(
                category,
                passed=False,
            )
        )
        categories.append(group)
    return {
        "hard_pass": bool(raw.get("hard_pass", False)),
        "hard_passes": _feedback_item(raw.get("hard_passes", {})),
        "primary_difference_category": raw.get("primary_difference_category"),
        "category_summary": _gate1_category_summary(comparison),
        "categories": categories,
    }


def _gate1_repair_cues(
    observer_diff: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Attach short, category-local repair cues to Gate1 feedback.

    The full bounded observer frontier remains the only evidence.  These cues
    do not classify a mismatch as semantic, edit an artifact, or soften Gate1;
    they keep the next source-aware Lift turn focused on the first causal
    category instead of rereading all downstream fallout.
    """

    raw_categories = observer_diff.get("categories")
    if not isinstance(raw_categories, list):
        return None
    present = {
        item.get("category")
        for item in raw_categories
        if isinstance(item, Mapping) and isinstance(item.get("category"), str)
    }
    cues: list[dict[str, str]] = []
    category_cues = (
        "interface",
        "predicate_effect",
        "op_multiset",
        "dataflow",
        "reduction",
    )
    for category in category_cues:
        if category in present:
            cue = relift_repair_instruction(f"gate1.{category}", "")
            if cue is None:
                continue
            cues.append({"category": category, "cue": cue})
    if not cues:
        return None
    primary = observer_diff.get("primary_difference_category")
    return {
        "primary_category": primary if isinstance(primary, str) else None,
        "ordered": cues,
        "authority": (
            "diagnostic-only guidance derived from the bounded Gate1 observer "
            "frontier; it never edits J/C/pi or turns a hard mismatch into a pass"
        ),
    }


class MainStatus(str, Enum):
    SCRC_CERTIFIED = "SCRC_CERTIFIED"
    NO_CLOSURE = "NO_CLOSURE"
    GATE2_FAILED = "GATE2_FAILED"


class Gate1SearchStatus(str, Enum):
    GATE1_CLOSED = "GATE1_CLOSED"
    NO_CLOSURE = "NO_CLOSURE"
    SUPERSEDED = "SUPERSEDED"


class FeedbackKind(str, Enum):
    LIFT = "lift"
    TYPECHECK = "typecheck"
    GATE1 = "gate1"


class SCRCInfrastructureError(RuntimeError):
    """An adapter/environment failure, never relabeled as a candidate failure."""

    def __init__(
        self,
        message: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.evidence = dict(evidence or {})
        super().__init__(message)


class CandidateLiftError(RuntimeError):
    """One model output was not a complete artifact, or its owner is exhausted."""

    def __init__(
        self,
        diagnostics: Mapping[str, Any],
        *,
        physical_calls: int = 1,
        terminal: bool = False,
        candidate_delta: Mapping[str, Any] | None = None,
    ) -> None:
        self.diagnostics = dict(diagnostics)
        if not isinstance(physical_calls, int) or physical_calls < 0:
            raise ValueError("Lift physical_calls must be non-negative")
        if not isinstance(terminal, bool):
            raise TypeError("CandidateLiftError.terminal must be boolean")
        self.physical_calls = physical_calls
        self.terminal = terminal
        self.candidate_delta = normalize_scrc_candidate_delta(candidate_delta)
        super().__init__(str(self.diagnostics.get("diagnostic", "Lift rejected")))


@dataclass(frozen=True)
class HarnessBindingInstance:
    """One actual c0 and its frontend execution coordinates.

    The object stays outside the proof artifact and Replay packet.  Adapters
    may read ``gate1_case``/``gate2_context`` only when compiling or executing.
    """

    c_values: tuple[tuple[str, Any], ...]
    gate1_case: Any
    gate2_context: Any = None

    @classmethod
    def create(
        cls,
        c_values: Mapping[str, Any],
        *,
        gate1_case: Any,
        gate2_context: Any = None,
        slot_order: Sequence[str] | None = None,
    ) -> "HarnessBindingInstance":
        if not isinstance(c_values, Mapping):
            raise TypeError("Harness c0 values must be a mapping")
        names = tuple(c_values)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Harness c0 slot names must be non-empty strings")
        if slot_order is None:
            # A mapping's insertion order is not an ABI declaration.  Keep
            # the optional convenience path deterministic as well: it is
            # valid only for the canonical C namespace and is materialized
            # through the same allocator order used by SourceFacts/C.
            ordered_names = CanonicalHandleAllocator.canonical_names(
                "c", len(names)
            )
            if set(names) != set(ordered_names):
                raise ValueError(
                    "Harness c0 values require explicit canonical slot_order"
                )
            ordered = tuple((name, c_values[name]) for name in ordered_names)
        else:
            if isinstance(slot_order, (str, bytes)):
                raise TypeError("Harness c0 slot_order must be a sequence of names")
            ordered_names = tuple(slot_order)
            if any(
                not isinstance(name, str) or not name for name in ordered_names
            ):
                raise ValueError("Harness c0 slot_order names must be non-empty strings")
            if len(ordered_names) != len(set(ordered_names)):
                raise ValueError("Harness c0 slot_order must be unique")
            if set(ordered_names) != set(names):
                raise ValueError(
                    "Harness c0 slot_order must cover exactly the supplied bindings"
                )
            ordered = tuple((name, c_values[name]) for name in ordered_names)
        if any(not isinstance(name, str) or not name for name, _ in ordered):
            raise ValueError("Harness c0 slot names must be non-empty strings")
        return cls(ordered, gate1_case, gate2_context)

    @property
    def values(self) -> dict[str, Any]:
        return dict(self.c_values)

    def validate_artifact(self, artifact: Gate1Artifact) -> None:
        if artifact.c.values != self.values:
            raise SCRCInfrastructureError(
                "Lift artifact C differs from the Harness-provided c0",
                evidence={"artifact": artifact.to_json()},
            )


@dataclass(frozen=True)
class SourceProgram:
    """Privileged Source material shared by Lift and frontend execution only.

    ``text`` is the selected source slice used by the Lift owner. Native
    frontends may need a language-specific slice (for example one CUDA device
    entry) while ``gate_source`` retains the real module/file consumed by the
    compiler and observer. The optional fields keep that distinction explicit
    instead of sending C++ through the Python AST anonymizer.
    """

    text: str
    gate_source: Any
    lift_source_kind: str = "python"
    lift_source_symbol: str | None = None
    lift_parameter_order: tuple[str, ...] = ()
    lift_constexpr_parameters: tuple[str, ...] = ()
    lift_source_to_opaque: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("SourceProgram.text must be non-empty")
        if self.lift_source_kind not in {"python", "cuda"}:
            raise ValueError(
                "SourceProgram.lift_source_kind must be 'python' or 'cuda'"
            )
        parameter_order = tuple(self.lift_parameter_order)
        constexpr_parameters = tuple(self.lift_constexpr_parameters)
        if any(
            not isinstance(name, str) or not name.isidentifier()
            for name in (*parameter_order, *constexpr_parameters)
        ):
            raise ValueError("SourceProgram Lift parameter names must be identifiers")
        if len(parameter_order) != len(set(parameter_order)):
            raise ValueError("SourceProgram Lift parameter order must be unique")
        if len(constexpr_parameters) != len(set(constexpr_parameters)):
            raise ValueError("SourceProgram Lift constexpr parameters must be unique")
        if not set(constexpr_parameters) <= set(parameter_order):
            raise ValueError(
                "SourceProgram Lift constexpr parameters must be in parameter order"
            )
        mapping = tuple(self.lift_source_to_opaque)
        if len({source for source, _opaque in mapping}) != len(mapping):
            raise ValueError("SourceProgram Lift source mapping must be one-to-one")
        if len({opaque for _source, opaque in mapping}) != len(mapping):
            raise ValueError("SourceProgram Lift opaque mapping must be one-to-one")
        if any(
            not isinstance(source, str)
            or not source.isidentifier()
            or not isinstance(opaque, str)
            or not opaque.isidentifier()
            for source, opaque in mapping
        ):
            raise ValueError("SourceProgram Lift mappings must contain identifiers")
        if self.lift_source_kind == "cuda":
            if not isinstance(self.lift_source_symbol, str) or not self.lift_source_symbol.isidentifier():
                raise ValueError(
                    "CUDA SourceProgram Lift slice requires a source symbol"
                )
            if not parameter_order:
                raise ValueError(
                    "CUDA SourceProgram Lift slice requires parameter order"
                )
            mapped_sources = {source for source, _opaque in mapping}
            if not set(parameter_order) <= mapped_sources:
                raise ValueError(
                    "CUDA SourceProgram Lift slice mapping must cover every source parameter"
                )
        object.__setattr__(self, "lift_parameter_order", parameter_order)
        object.__setattr__(self, "lift_constexpr_parameters", constexpr_parameters)
        object.__setattr__(self, "lift_source_to_opaque", mapping)


@dataclass(frozen=True)
class LiftFeedback:
    kind: FeedbackKind
    diagnostics: Mapping[str, Any]
    formal_round: int | None = None
    open_guidance: tuple[Mapping[str, Any], ...] = ()
    resolved_guidance: tuple[Mapping[str, Any], ...] = ()
    candidate_delta: Mapping[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for field_name in ("open_guidance", "resolved_guidance"):
            value = getattr(self, field_name)
            if value is None:
                value = ()
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
                raise TypeError(f"{field_name} must be a sequence")
            normalized: list[dict[str, Any]] = []
            for item in value:
                if not isinstance(item, Mapping):
                    raise TypeError(f"{field_name} items must be mappings")
                copied = json.loads(
                    json.dumps(
                        dict(item),
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
                if not isinstance(copied, dict):
                    raise TypeError(f"{field_name} items must be objects")
                normalized.append(copied)
            object.__setattr__(self, field_name, tuple(normalized))
        if self.candidate_delta is not None:
            object.__setattr__(
                self,
                "candidate_delta",
                normalize_scrc_candidate_delta(self.candidate_delta),
            )
    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "formal_round": self.formal_round,
            "diagnostics": dict(self.diagnostics),
            "open_guidance": [
                dict(item) for item in self.open_guidance
            ],
            "resolved_guidance": [
                dict(item) for item in self.resolved_guidance
            ],
            "candidate_delta": (
                dict(self.candidate_delta)
                if self.candidate_delta is not None else None
            ),
        }


@dataclass(frozen=True)
class LiftCallResult:
    """Execution sidecar for one complete artifact; not a proof object."""

    artifact: Gate1Artifact
    physical_calls: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, Gate1Artifact):
            raise TypeError("LiftCallResult requires complete canonical (J,C,pi)")
        if not isinstance(self.physical_calls, int) or self.physical_calls < 1:
            raise ValueError("LiftCallResult.physical_calls must be positive")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("LiftCallResult.metadata must be a mapping")


@dataclass(frozen=True)
class SCRCRunRequest:
    """Source and one real Harness binding instance (the same c0).

    ``max_lift_calls`` bounds logical Lift invocations for this branch.  It is
    independent of physical provider retries inside one invocation and of the
    formal Replay/Gate1-round budget.  ``max_pre_replay_rejections == 0`` still
    delegates that narrower rejection policy to the branch-local Lift owner's
    terminal resource signal, but it cannot bypass the Lift cap.
    """

    source: SourceProgram
    binding: HarnessBindingInstance
    frontend: str
    source_facts: SourceFacts
    construction: CanonicalConstruction
    max_gate1_rounds: int = MAX_GATE1_ROUNDS
    max_lift_calls: int = MAX_LIFT_CALLS
    max_pre_replay_rejections: int = MAX_PRE_REPLAY_REJECTIONS

    def __post_init__(self) -> None:
        if not isinstance(self.frontend, str) or not self.frontend:
            raise ValueError("SCRC request frontend must be a non-empty id")
        if not isinstance(self.source, SourceProgram):
            raise TypeError("SCRC request requires one SourceProgram")
        if not isinstance(self.binding, HarnessBindingInstance):
            raise TypeError("SCRC request requires one HarnessBindingInstance")
        if not isinstance(self.source_facts, SourceFacts):
            raise TypeError("SCRC request requires one canonical SourceFacts record")
        if not isinstance(self.construction, CanonicalConstruction):
            raise TypeError("SCRC request requires one canonical construction")
        if self.construction.source_facts.digest != self.source_facts.digest:
            raise ValueError("SCRC request construction differs from SourceFacts")
        if self.source_facts.frontend != self.frontend:
            raise ValueError("SCRC request SourceFacts frontend differs from request")
        if self.source_facts.source_kind != self.source.lift_source_kind:
            raise ValueError("SCRC request SourceFacts source kind differs from SourceProgram")
        if (
            self.source.lift_source_symbol is not None
            and self.source_facts.source_symbol != self.source.lift_source_symbol
        ):
            raise ValueError("SCRC request SourceFacts symbol differs from SourceProgram")
        if not isinstance(self.max_gate1_rounds, int) or self.max_gate1_rounds < 1:
            raise ValueError("max_gate1_rounds must be positive")
        if (
            not isinstance(self.max_lift_calls, int)
            or self.max_lift_calls < 1
            or self.max_lift_calls > MAX_LIFT_CALLS
        ):
            raise ValueError(
                f"max_lift_calls must be in [1, {MAX_LIFT_CALLS}]"
            )
        if (
            not isinstance(self.max_pre_replay_rejections, int)
            or self.max_pre_replay_rejections < 0
        ):
            raise ValueError("max_pre_replay_rejections must be non-negative")


class LiftAdapter(Protocol):
    def lift(
        self,
        request: SCRCRunRequest,
        feedback: LiftFeedback | None,
    ) -> LiftCallResult: ...


class Gate1AttemptAdapter(Protocol):
    def run(
        self,
        artifact: Gate1Artifact,
        request: SCRCRunRequest,
    ) -> Gate1ExecutionResult: ...


class Gate2ExecutionAdapter(Protocol):
    def execute(
        self,
        artifact: Gate1Artifact,
        gate1: Gate1ExecutionResult,
        request: SCRCRunRequest,
    ) -> Gate2SuiteResult: ...


@dataclass(frozen=True)
class CandidateAttemptReport:
    lift_call: int
    formal_round: int | None
    typecheck_status: TypeCheckStatus
    artifact: Gate1Artifact | None = None
    typecheck: TypeCheckReport | None = None
    gate1: Gate1ExecutionResult | None = None
    feedback: LiftFeedback | None = None
    lift_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "lift_call": self.lift_call,
            "formal_round": self.formal_round,
            "typecheck_status": self.typecheck_status.value,
            "artifact": self.artifact.to_json() if self.artifact is not None else None,
            "typecheck": (
                self.typecheck.to_json() if self.typecheck is not None else None
            ),
            "gate1": self.gate1.to_json() if self.gate1 is not None else None,
            "feedback": self.feedback.to_json() if self.feedback is not None else None,
            "lift_metadata": dict(self.lift_metadata),
        }


@dataclass(frozen=True)
class CallBudget:
    lift_calls: int
    lift_physical_calls: int
    typecheck_calls: int
    formal_gate1_rounds: int
    replay_normal_calls: int
    replay_masked_calls: int
    replay_physical_calls: int
    gate2_cases: int
    source_executions: int
    replay_executions: int
    refeval_executions: int

    def to_json(self) -> dict[str, int]:
        return {
            "lift_calls": self.lift_calls,
            "lift_physical_calls": self.lift_physical_calls,
            "typecheck_calls": self.typecheck_calls,
            "formal_gate1_rounds": self.formal_gate1_rounds,
            "replay_normal_calls": self.replay_normal_calls,
            "replay_masked_calls": self.replay_masked_calls,
            "replay_physical_calls": self.replay_physical_calls,
            "gate2_cases": self.gate2_cases,
            "source_executions": self.source_executions,
            "replay_executions": self.replay_executions,
            "refeval_executions": self.refeval_executions,
        }


@dataclass(frozen=True)
class SCRCGate1SearchReport:
    """Pre-Gate2 branch result used by bounded candidate fan-out."""

    status: Gate1SearchStatus
    attempts: tuple[CandidateAttemptReport, ...]
    accepted_attempt: int | None
    j_necessity_status: JNecessityStatus | None
    call_budget: CallBudget
    stop_reason: str | None = None
    winner_branch_id: int | None = None
    stop_observed_at: str | None = None
    stage_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return self.status is Gate1SearchStatus.GATE1_CLOSED

    @property
    def superseded(self) -> bool:
        return self.status is Gate1SearchStatus.SUPERSEDED

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "attempts": [item.to_json() for item in self.attempts],
            "accepted_attempt": self.accepted_attempt,
            "j_necessity_status": (
                self.j_necessity_status.value
                if self.j_necessity_status is not None else None
            ),
            "call_budget": self.call_budget.to_json(),
            "stop_reason": self.stop_reason,
            "winner_branch_id": self.winner_branch_id,
            "stop_observed_at": self.stop_observed_at,
            "stage_metrics": {
                str(key): dict(value)
                for key, value in self.stage_metrics.items()
            },
        }


@dataclass(frozen=True)
class SCRCRunReport:
    main_status: MainStatus
    attempts: tuple[CandidateAttemptReport, ...]
    accepted_attempt: int | None
    gate2: Gate2SuiteResult | None
    j_necessity_status: JNecessityStatus | None
    call_budget: CallBudget
    stage_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "main_status": self.main_status.value,
            "attempts": [item.to_json() for item in self.attempts],
            "accepted_attempt": self.accepted_attempt,
            "gate2": self.gate2.to_json() if self.gate2 is not None else None,
            "j_necessity_status": (
                self.j_necessity_status.value
                if self.j_necessity_status is not None else None
            ),
            "call_budget": self.call_budget.to_json(),
            "stage_metrics": {
                str(key): dict(value)
                for key, value in self.stage_metrics.items()
            },
        }


@dataclass(frozen=True)
class SCRCRunDependencies:
    lift: LiftAdapter
    gate1_factory: Callable[[int], Gate1AttemptAdapter]
    gate2: Gate2ExecutionAdapter
    type_checker: Callable[..., TypeCheckReport] = type_check
    semantic_probe: Callable[[Gate1Artifact], Mapping[str, Any]] | None = None


class SCRCGateRunner:
    """Run one bounded lifecycle; no local runner may own a second lifecycle."""

    def __init__(self, dependencies: SCRCRunDependencies) -> None:
        self.dependencies = dependencies
        self._owned_closed_searches: set[int] = set()

    @staticmethod
    def _run_semantic_probe(
        callback: Callable[[Gate1Artifact], Mapping[str, Any]],
        artifact: Gate1Artifact,
    ) -> dict[str, Any]:
        """Run the public Source-vs-J diagnostic without granting it authority."""

        try:
            raw = callback(artifact)
            if not isinstance(raw, Mapping):
                raise TypeError("semantic probe must return a mapping")
            result = dict(raw)
            if result.get("status") not in {"pass", "diverged", "unavailable"}:
                raise ValueError(
                    "semantic probe status must be pass, diverged, or unavailable"
                )
            json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            result = {
                "protocol": "intentir.lift.public-semantic-probe.v1",
                "status": "unavailable",
                "rule_code": "semantic_probe_callback_failed",
                "path": "<semantic-probe-callback>",
                "exception_type": type(exc).__name__,
            }
        result.setdefault(
            "authority",
            "public diagnostic only; not CandidateCheck or certification",
        )
        return result

    @classmethod
    def _gate1_feedback(
        cls,
        result: Gate1ExecutionResult,
        formal_round: int,
        prior_feedback: LiftFeedback | None = None,
    ) -> LiftFeedback:
        normal = result.normal
        diagnostics: dict[str, Any] = {
            "status": result.status.value,
            "diagnostic": normal.diagnostic,
        }
        if normal.comparison is not None:
            observer_diff = _compact_gate1_observer_diff(
                normal.comparison,
                source_fingerprint=result.source_fingerprint,
                replay_fingerprint=normal.fingerprint,
            )
            diagnostics["observer_diff"] = observer_diff
            diagnostics["category_summary"] = observer_diff["category_summary"]
            repair_cues = _gate1_repair_cues(observer_diff)
            if repair_cues is not None:
                diagnostics["repair_cues"] = repair_cues
        diagnostics = _prune_relift_diagnostics(diagnostics)
        open_guidance, resolved_guidance = _merge_relift_guidance(
            prior_feedback,
            diagnostics,
        )
        return LiftFeedback(
            kind=FeedbackKind.GATE1,
            diagnostics=diagnostics,
            formal_round=formal_round,
            open_guidance=open_guidance,
            resolved_guidance=resolved_guidance,
        )

    @staticmethod
    def _unresolved_gate1(
        feedback: LiftFeedback | None,
    ) -> dict[str, Any] | None:
        if feedback is None:
            return None
        if feedback.kind is FeedbackKind.GATE1:
            return {
                "formal_round": feedback.formal_round,
                "diagnostics": dict(feedback.diagnostics),
            }
        retained = feedback.diagnostics.get(UNRESOLVED_GATE1_DIAGNOSTIC)
        if not isinstance(retained, Mapping):
            return None
        formal_round = retained.get("formal_round")
        diagnostics = retained.get("diagnostics")
        if (
            not isinstance(formal_round, int)
            or formal_round < 1
            or not isinstance(diagnostics, Mapping)
        ):
            return None
        return {
            "formal_round": formal_round,
            "diagnostics": _prune_relift_diagnostics(dict(diagnostics)),
        }

    @classmethod
    def _retain_gate1_diagnostic(
        cls,
        diagnostics: Mapping[str, Any],
        feedback: LiftFeedback | None,
    ) -> dict[str, Any]:
        combined = dict(diagnostics)
        # CandidateCheck/Decode failures often happen before the new response
        # reaches SourceReconcile. Keep only the immutable Source frontier and
        # the current bounded CandidateCheck diagnostics. Rejected candidate
        # history is deliberately not carried across relift turns; the compact
        # guidance frontier is the only cumulative provider context.
        if feedback is not None:
            for key in ("source_reconcile",):
                if key not in combined and key in feedback.diagnostics:
                    combined[key] = feedback.diagnostics[key]
        unresolved = cls._unresolved_gate1(feedback)
        if unresolved is not None:
            combined[UNRESOLVED_GATE1_DIAGNOSTIC] = unresolved
        return _prune_relift_diagnostics(combined)

    def search_gate1(
        self,
        request: SCRCRunRequest,
        *,
        stop_requested: Callable[[], int | None] | None = None,
        progress_callback: SCRCProgressCallback | None = None,
    ) -> SCRCGate1SearchReport:
        """Search one fresh isolated branch through TypeCheck and Gate1."""

        report = self._search_gate1_transaction(
            request,
            stop_requested=stop_requested,
            progress_callback=progress_callback,
        )
        if report.closed:
            self._owned_closed_searches.add(id(report))
        return report

    def _search_gate1_transaction(
        self,
        request: SCRCRunRequest,
        *,
        replay_retry_artifact: Gate1Artifact | None = None,
        stop_requested: Callable[[], int | None] | None = None,
        progress_callback: SCRCProgressCallback | None = None,
    ) -> SCRCGate1SearchReport:
        """Execute one branch-local Gate1 search transaction.

        The private artifact path is used only by the owning runner to
        generate another source-blind Replay for the artifact that its current
        search just closed.  Public callers cannot inject persisted feedback or
        an artifact from another run/root.
        """

        search_started = time.monotonic()
        stage_metrics: dict[str, dict[str, Any]] = {}

        def record_stage(
            name: str,
            started: float,
            status: str,
            **details: Any,
        ) -> None:
            entry = stage_metrics.setdefault(
                name,
                {"elapsed_ms": 0.0, "runs": 0},
            )
            entry["elapsed_ms"] += max(
                0.0,
                (time.monotonic() - started) * 1000.0,
            )
            entry["runs"] += 1
            entry["status"] = status
            entry.update(details)

        def completed_metrics(status: str) -> dict[str, Mapping[str, Any]]:
            result = {
                name: dict(value) for name, value in stage_metrics.items()
            }
            result["search_total"] = {
                "elapsed_ms": max(
                    0.0,
                    (time.monotonic() - search_started) * 1000.0,
                ),
                "runs": 1,
                "status": status,
            }
            return result

        if replay_retry_artifact is not None:
            replay_retry_artifact = canonicalize_gate1_artifact(
                replay_retry_artifact
            )
            request.binding.validate_artifact(replay_retry_artifact)

        attempts: list[CandidateAttemptReport] = []
        feedback: LiftFeedback | None = None
        lift_calls = 0
        lift_physical_calls = 0
        typecheck_calls = 0
        pre_replay_rejections = 0
        formal_rounds = 0
        replay_normal_calls = 0
        replay_masked_calls = 0
        replay_physical_calls = 0
        terminal_stop_reason: str | None = None
        terminal_stop_observed_at: str | None = None
        gate1_adapters: list[Gate1AttemptAdapter] = []
        pending_replay_artifact = replay_retry_artifact

        def budget() -> CallBudget:
            return CallBudget(
                lift_calls=lift_calls,
                lift_physical_calls=lift_physical_calls,
                typecheck_calls=typecheck_calls,
                formal_gate1_rounds=formal_rounds,
                replay_normal_calls=replay_normal_calls,
                replay_masked_calls=replay_masked_calls,
                replay_physical_calls=replay_physical_calls,
                gate2_cases=0,
                source_executions=0,
                replay_executions=0,
                refeval_executions=0,
            )

        def superseded(observed_at: str) -> SCRCGate1SearchReport | None:
            winner = stop_requested() if stop_requested is not None else None
            if winner is None:
                return None
            return SCRCGate1SearchReport(
                status=Gate1SearchStatus.SUPERSEDED,
                attempts=tuple(attempts),
                accepted_attempt=None,
                j_necessity_status=None,
                call_budget=budget(),
                stop_reason="first_gate1_closure",
                winner_branch_id=winner,
                stop_observed_at=observed_at,
                stage_metrics=completed_metrics("superseded"),
            )

        while formal_rounds < request.max_gate1_rounds:
            stopped = superseded("before_lift")
            if stopped is not None:
                return stopped
            if (
                pending_replay_artifact is None
                and lift_calls >= request.max_lift_calls
            ):
                terminal_stop_reason = "lift_call_cap"
                terminal_stop_observed_at = "before_lift"
                break
            lift_started = time.monotonic()
            if pending_replay_artifact is not None:
                artifact = canonicalize_gate1_artifact(
                    pending_replay_artifact
                )
                pending_replay_artifact = None
                attempt_lift_call = 0
                lift_metadata: Mapping[str, Any] = {
                    "current_artifact_gate1_retry": True,
                    "model_calls": 0,
                }
                record_stage(
                    "lift",
                    lift_started,
                    "current_artifact_retry",
                    attempt=0,
                )
            else:
                lift_calls += 1
                _notify_progress(
                    progress_callback,
                    "lift",
                    lift_calls,
                    None,
                    "started",
                )
                try:
                    lift_result = self.dependencies.lift.lift(request, feedback)
                except CandidateLiftError as exc:
                    record_stage(
                        "lift",
                        lift_started,
                        "rejected",
                        attempt=lift_calls,
                    )
                    lift_physical_calls += exc.physical_calls
                    if not exc.terminal:
                        pre_replay_rejections += 1
                    _notify_progress(
                        progress_callback,
                        "lift",
                        lift_calls,
                        None,
                        "rejected",
                        _safe_lift_progress_diagnostic(exc.diagnostics),
                    )
                    # A decoded candidate rejected before the structural
                    # boundary is diagnostic-only and never becomes authoring
                    # state for the next fresh lifecycle.
                    merged_diagnostics = self._retain_gate1_diagnostic(
                        exc.diagnostics,
                        feedback,
                    )
                    open_guidance, resolved_guidance = _merge_relift_guidance(
                        feedback,
                        merged_diagnostics,
                    )
                    feedback = LiftFeedback(
                        kind=FeedbackKind.LIFT,
                        diagnostics=merged_diagnostics,
                        open_guidance=open_guidance,
                        resolved_guidance=resolved_guidance,
                        candidate_delta=exc.candidate_delta,
                    )
                    attempts.append(
                        CandidateAttemptReport(
                            lift_call=lift_calls,
                            formal_round=None,
                            typecheck_status=TypeCheckStatus.NOT_RUN,
                            feedback=feedback,
                        )
                    )
                    stopped = superseded("after_lift")
                    if stopped is not None:
                        return stopped
                    if exc.terminal or (
                        request.max_pre_replay_rejections > 0
                        and pre_replay_rejections
                        >= request.max_pre_replay_rejections
                    ):
                        terminal_stop_reason = (
                            str(exc.diagnostics.get("failure_class"))
                            if exc.terminal
                            and isinstance(
                                exc.diagnostics.get("failure_class"), str
                            )
                            else "pre_replay_rejection_cap"
                        )
                        terminal_stop_observed_at = "after_lift"
                        break
                    continue
                except SCRCInfrastructureError as exc:
                    record_stage(
                        "lift",
                        lift_started,
                        "infrastructure_error",
                        attempt=lift_calls,
                    )
                    evidence = dict(exc.evidence)
                    evidence.setdefault(
                        "failure_class", "lift_adapter_infrastructure"
                    )
                    evidence.setdefault("failure_stage", "lift_adapter_call")
                    evidence.setdefault(
                        "stage_metrics",
                        completed_metrics("infrastructure_error"),
                    )
                    origin = exc.__cause__ or exc.__context__
                    if origin is not None:
                        origin_type = type(origin).__name__
                        if re.fullmatch(
                            r"[A-Za-z][A-Za-z0-9_]{0,63}", origin_type
                        ):
                            evidence.setdefault(
                                "origin_exception_type", origin_type
                            )
                    exc.evidence = evidence
                    _notify_progress(
                        progress_callback,
                        "lift",
                        lift_calls,
                        None,
                        "infrastructure_error",
                        {
                            "phase": evidence["failure_stage"],
                            "failure_class": evidence["failure_class"],
                            "candidate_present": False,
                        },
                    )
                    raise
                if not isinstance(lift_result, LiftCallResult):
                    raise SCRCInfrastructureError(
                        "Lift adapter must return LiftCallResult",
                        evidence={
                            "failure_class": "lift_adapter_contract_violation",
                            "failure_stage": "lift_return_validation",
                            "returned_type": type(lift_result).__name__,
                        },
                    )
                lift_physical_calls += lift_result.physical_calls
                record_stage(
                    "lift",
                    lift_started,
                    "finished",
                    attempt=lift_calls,
                )
                _notify_progress(
                    progress_callback,
                    "lift",
                    lift_calls,
                    None,
                    "finished",
                )
                artifact = canonicalize_gate1_artifact(lift_result.artifact)
                attempt_lift_call = lift_calls
                lift_metadata = lift_result.metadata
            stopped = superseded("after_lift")
            if stopped is not None:
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=attempt_lift_call,
                        formal_round=None,
                        typecheck_status=TypeCheckStatus.NOT_RUN,
                        artifact=artifact,
                        lift_metadata=lift_metadata,
                    )
                )
                return superseded("after_lift") or stopped
            bind_started = time.monotonic()
            request.binding.validate_artifact(artifact)
            record_stage(
                "bind_source",
                bind_started,
                "finished",
                attempt=attempt_lift_call,
            )
            typecheck_calls += 1
            typecheck_started = time.monotonic()
            _notify_progress(
                progress_callback,
                "typecheck",
                attempt_lift_call,
                None,
                "started",
            )
            report = self.dependencies.type_checker(
                artifact.j,
                artifact.c,
                artifact.pi,
                require_program_axes=True,
            )
            record_stage(
                "typecheck",
                typecheck_started,
                report.status.value,
                attempt=attempt_lift_call,
            )
            _notify_progress(
                progress_callback,
                "typecheck",
                attempt_lift_call,
                None,
                report.status.value,
            )
            if not report.passed:
                pre_replay_rejections += 1
                typecheck_diagnostics = {"typecheck": report.to_json()}
                open_guidance, resolved_guidance = _merge_relift_guidance(
                    feedback,
                    typecheck_diagnostics,
                )
                # TypeCheck rejects this candidate. Expose only its structural
                # delta and diagnostics to the next fresh lifecycle.
                feedback = LiftFeedback(
                    kind=FeedbackKind.TYPECHECK,
                    diagnostics=self._retain_gate1_diagnostic(
                        typecheck_diagnostics,
                        feedback,
                    ),
                    open_guidance=open_guidance,
                    resolved_guidance=resolved_guidance,
                )
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=attempt_lift_call,
                        formal_round=None,
                        typecheck_status=TypeCheckStatus.FAIL,
                        artifact=artifact,
                        typecheck=report,
                        feedback=feedback,
                        lift_metadata=lift_metadata,
                    )
                )
                stopped = superseded("after_typecheck")
                if stopped is not None:
                    return stopped
                if (
                    request.max_pre_replay_rejections > 0
                    and pre_replay_rejections
                    >= request.max_pre_replay_rejections
                ):
                    terminal_stop_reason = "pre_replay_rejection_cap"
                    terminal_stop_observed_at = "after_typecheck"
                    break
                continue

            semantic_probe = self.dependencies.semantic_probe
            if semantic_probe is not None:
                semantic_probe_started = time.monotonic()
                _notify_progress(
                    progress_callback,
                    "semantic_probe",
                    attempt_lift_call,
                    None,
                    "started",
                )
                probe_result = self._run_semantic_probe(semantic_probe, artifact)
                lift_metadata = {
                    **dict(lift_metadata),
                    "semantic_probe": probe_result,
                }
                _notify_progress(
                    progress_callback,
                    "semantic_probe",
                    attempt_lift_call,
                    None,
                    str(probe_result.get("status", "unavailable")),
                )
                record_stage(
                    "semantic_probe",
                    semantic_probe_started,
                    str(probe_result.get("status", "unavailable")),
                    attempt=attempt_lift_call,
                )
                probe_status = probe_result.get("status")
                repairable_probe = probe_result.get("repairable") is True
                if probe_status == "diverged" or repairable_probe:
                    pre_replay_rejections += 1
                    if repairable_probe:
                        diagnostic = (
                            "public Source-vs-J semantic probe found an incomplete, "
                            "directly expressible J reference domain"
                        )
                        failure_class = "lift_semantic_probe_repairable_unsupported"
                    else:
                        diagnostic = (
                            "public Source-vs-J semantic probe found a divergence"
                        )
                        failure_class = "lift_semantic_probe_diverged"
                    probe_diagnostics = {
                        "diagnostic": diagnostic,
                        "failure_class": failure_class,
                        "semantic_probe": probe_result,
                        "authority": (
                            "diagnostic-only public Source-vs-J comparison; "
                            "it never edits or certifies the artifact"
                        ),
                    }
                    open_guidance, resolved_guidance = _merge_relift_guidance(
                        feedback,
                        probe_diagnostics,
                    )
                    feedback = LiftFeedback(
                        kind=FeedbackKind.LIFT,
                        diagnostics=self._retain_gate1_diagnostic(
                            probe_diagnostics,
                            feedback,
                        ),
                        open_guidance=open_guidance,
                        resolved_guidance=resolved_guidance,
                    )
                    attempts.append(
                        CandidateAttemptReport(
                            lift_call=attempt_lift_call,
                            formal_round=None,
                            typecheck_status=TypeCheckStatus.PASS,
                            artifact=artifact,
                            typecheck=report,
                            feedback=feedback,
                            lift_metadata=lift_metadata,
                        )
                    )
                    stopped = superseded("after_semantic_probe")
                    if stopped is not None:
                        return stopped
                    if (
                        request.max_pre_replay_rejections > 0
                        and pre_replay_rejections
                        >= request.max_pre_replay_rejections
                    ):
                        terminal_stop_reason = "pre_replay_rejection_cap"
                        terminal_stop_observed_at = "after_semantic_probe"
                        break
                    continue

            stopped = superseded("before_replay")
            if stopped is not None:
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=attempt_lift_call,
                        formal_round=None,
                        typecheck_status=TypeCheckStatus.PASS,
                        artifact=artifact,
                        typecheck=report,
                        lift_metadata=lift_metadata,
                    )
                )
                return superseded("before_replay") or stopped

            next_formal_round = formal_rounds + 1
            _notify_progress(
                progress_callback,
                "replay",
                None,
                next_formal_round,
                "started",
            )
            replay_generation_retries = 0
            replay_response_rejected_before_formal = False
            while True:
                gate1_adapter = self.dependencies.gate1_factory(
                    next_formal_round
                )
                if any(gate1_adapter is previous for previous in gate1_adapters):
                    raise SCRCInfrastructureError(
                        "gate1_factory reused an attempt adapter; Replay must be fresh"
                    )
                gate1_adapters.append(gate1_adapter)
                replay_gate1_started = time.monotonic()
                gate1 = gate1_adapter.run(artifact, request)
                record_stage(
                    "replay_gate1",
                    replay_gate1_started,
                    gate1.status.value,
                    formal_round=next_formal_round,
                )
                normal_arms_for_progress = (
                    gate1.normal_candidates
                    if gate1.normal_candidates
                    else (gate1.normal,)
                )
                replay_was_generated = any(
                    arm.generation is not None
                    for arm in normal_arms_for_progress
                )
                replay_generation_failed = (
                    not replay_was_generated
                    and any(
                        (arm.diagnostic or "").startswith("generation failed:")
                        for arm in normal_arms_for_progress
                    )
                )
                # Count every actual writer invocation, including a response
                # rejected before formal K. Formal rounds remain unchanged.
                replay_normal_calls += gate1.normal_calls
                replay_masked_calls += gate1.masked_j_calls
                for arm in (*normal_arms_for_progress, gate1.masked_j):
                    if arm is None:
                        continue
                    replay_physical_calls += (
                        arm.physical_calls
                        if arm.physical_calls
                        else (
                            arm.generation.attempts
                            if arm.generation is not None
                            else 0
                        )
                    )
                # Provider stop conditions have precedence over the one fresh
                # writer retry and over any relift decision.  In particular,
                # normal Replay may fail before comparison, while masked-J may
                # fail only after the normal formal comparison has completed;
                # both are infrastructure stops and neither may be retried or
                # flattened into an ordinary NO_CLOSURE verdict.
                provider_stop_failures = _safe_replay_generation_failures(
                    gate1,
                    provider_only=True,
                )
                if provider_stop_failures:
                    normal_formal_comparison_ran = any(
                        arm.comparison is not None
                        for arm in normal_arms_for_progress
                    )
                    consumed_rounds = (
                        next_formal_round
                        if normal_formal_comparison_ran
                        else formal_rounds
                    )
                    _notify_progress(
                        progress_callback,
                        "replay",
                        None,
                        next_formal_round,
                        "provider_stop",
                    )
                    _notify_progress(
                        progress_callback,
                        "gate1",
                        None,
                        next_formal_round,
                        "infrastructure_failed",
                    )
                    raise SCRCInfrastructureError(
                        "Gate1 Replay provider stop condition; no retry or relift "
                        "was attempted",
                        evidence={
                            "attempted_formal_round": next_formal_round,
                            "formal_gate1_rounds_consumed": consumed_rounds,
                            "replay_normal_calls": replay_normal_calls,
                            "replay_masked_calls": replay_masked_calls,
                            "replay_physical_calls": replay_physical_calls,
                            "replay_generation_failures": list(
                                provider_stop_failures
                            ),
                            "candidate_present": True,
                            "artifact": artifact.to_json(),
                            "typecheck": report.to_json(),
                            "gate1": gate1.to_json(),
                        },
                    )
                # If every authored Replay failed before a TTIR comparison,
                # Gate1 learned nothing about J/C/pi. Retry the same exact
                # artifact with one fresh source-blind writer. This retry is
                # not a formal round and never falls through to Lift.
                if replay_generation_failed:
                    if replay_generation_retries < MAX_REPLAY_GENERATION_RETRIES:
                        replay_generation_retries += 1
                        _notify_progress(
                            progress_callback,
                            "replay",
                            None,
                            next_formal_round,
                            "retrying",
                        )
                        _notify_progress(
                            progress_callback,
                            "gate1",
                            None,
                            next_formal_round,
                            "retrying",
                        )
                        continue
                    _notify_progress(
                        progress_callback,
                        "replay",
                        None,
                        next_formal_round,
                        "not_run",
                    )
                    if _is_replay_response_rejection(gate1):
                        _notify_progress(
                            progress_callback,
                            "gate1",
                            None,
                            next_formal_round,
                            "replay_response_rejected",
                        )
                        attempts.append(
                            CandidateAttemptReport(
                                lift_call=attempt_lift_call,
                                formal_round=None,
                                typecheck_status=TypeCheckStatus.PASS,
                                artifact=artifact,
                                typecheck=report,
                                gate1=gate1,
                                feedback=None,
                                lift_metadata=lift_metadata,
                            )
                        )
                        stopped = superseded("after_replay_generation")
                        if stopped is not None:
                            return stopped
                        # A rejected Replay response never consumed formal K
                        # and supplies neither a repair base nor candidate
                        # feedback.  It must not, however, terminate a
                        # fresh-Lift budget early: the next loop iteration
                        # invokes a new Lift lifecycle with no inherited
                        # candidate, Replay source, or trajectory.
                        replay_response_rejected_before_formal = True
                        break
                    _notify_progress(
                        progress_callback,
                        "gate1",
                        None,
                        next_formal_round,
                        "infrastructure_failed",
                    )
                    raise SCRCInfrastructureError(
                        "Gate1 Replay generation failed before a formal comparison; "
                        "formal K was not consumed",
                        evidence={
                            "attempted_formal_round": next_formal_round,
                            "formal_gate1_rounds_consumed": formal_rounds,
                            "replay_normal_calls": replay_normal_calls,
                            "replay_masked_calls": replay_masked_calls,
                            "replay_physical_calls": replay_physical_calls,
                            "replay_generation_failures": list(
                                _safe_replay_generation_failures(gate1)
                            ),
                            "artifact": artifact.to_json(),
                            "typecheck": report.to_json(),
                            "gate1": gate1.to_json(),
                        },
                    )
                # The final non-generation-failure result below is the only
                # result that can consume this formal Gate1 round.
                replay_compile_failed_before_comparison = (
                    replay_was_generated
                    and not any(
                        arm.comparison is not None
                        for arm in normal_arms_for_progress
                    )
                    and any(
                        arm.generation is not None
                        and arm.compile_result is not None
                        and not arm.compile_result.ok
                        for arm in normal_arms_for_progress
                    )
                    and not any(
                        arm.compile_result is not None and arm.compile_result.ok
                        for arm in normal_arms_for_progress
                    )
                )
                break
            if replay_response_rejected_before_formal:
                # The ``break`` above leaves the generation-retry loop.  This
                # ``continue`` belongs to the outer Lift search loop: formal K
                # remains unconsumed and the next candidate starts from a new
                # legal Lift lifecycle.
                continue
            formal_rounds = next_formal_round
            _notify_progress(
                progress_callback,
                "replay",
                None,
                formal_rounds,
                "finished" if replay_was_generated else "not_run",
            )
            _notify_progress(
                progress_callback,
                "gate1",
                None,
                formal_rounds,
                gate1.status.value,
            )
            if gate1.status is Gate1Status.COMPILE_FAILED:
                raise SCRCInfrastructureError(
                    "Gate1 source compilation failed; it is not a closure verdict",
                    evidence={
                        "formal_round": formal_rounds,
                        "artifact": artifact.to_json(),
                        "typecheck": report.to_json(),
                        "gate1": gate1.to_json(),
                    },
                )
            if _masked_replay_incomplete(gate1):
                _notify_progress(
                    progress_callback,
                    "replay",
                    None,
                    formal_rounds,
                    "masked_incomplete",
                )
                _notify_progress(
                    progress_callback,
                    "gate1",
                    None,
                    formal_rounds,
                    "masked_replay_incomplete",
                )
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=attempt_lift_call,
                        formal_round=formal_rounds,
                        typecheck_status=TypeCheckStatus.PASS,
                        artifact=artifact,
                        typecheck=report,
                        gate1=gate1,
                        feedback=None,
                        lift_metadata=lift_metadata,
                    )
                )
                stopped = superseded("after_gate1")
                if stopped is not None:
                    return stopped
                return SCRCGate1SearchReport(
                    status=Gate1SearchStatus.NO_CLOSURE,
                    attempts=tuple(attempts),
                    accepted_attempt=None,
                    j_necessity_status=None,
                    call_budget=budget(),
                    stop_reason="masked_replay_incomplete",
                    stop_observed_at="after_gate1",
                    stage_metrics=completed_metrics("no_closure"),
                )
            if gate1.status is not Gate1Status.REPLAY_CLOSED:
                source_observer_unsupported = (
                    gate1.normal.diagnostic or ""
                ).startswith(SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX)
                if source_observer_unsupported:
                    attempts.append(
                        CandidateAttemptReport(
                            lift_call=attempt_lift_call,
                            formal_round=formal_rounds,
                            typecheck_status=TypeCheckStatus.PASS,
                            artifact=artifact,
                            typecheck=report,
                            gate1=gate1,
                            feedback=None,
                            lift_metadata=lift_metadata,
                        )
                    )
                    stopped = superseded("after_gate1")
                    if stopped is not None:
                        return stopped
                    return SCRCGate1SearchReport(
                        status=Gate1SearchStatus.NO_CLOSURE,
                        attempts=tuple(attempts),
                        accepted_attempt=None,
                        j_necessity_status=None,
                        call_budget=budget(),
                        stop_reason="observer_unsupported",
                        stop_observed_at="after_gate1",
                        stage_metrics=completed_metrics("no_closure"),
                    )

                candidate_feedback = self._gate1_feedback(
                    gate1,
                    formal_rounds,
                    feedback,
                )
                attempts.append(
                    CandidateAttemptReport(
                        lift_call=attempt_lift_call,
                        formal_round=formal_rounds,
                        typecheck_status=TypeCheckStatus.PASS,
                        artifact=artifact,
                        typecheck=report,
                        gate1=gate1,
                        feedback=candidate_feedback,
                        lift_metadata=lift_metadata,
                    )
                )
                # The next Lift call receives only sanitized rule-level
                # guidance. The current artifact remains evidence for this
                # attempt and is never a new authoring base.
                feedback = candidate_feedback
                if replay_compile_failed_before_comparison:
                    pending_replay_artifact = artifact
                stopped = superseded("after_gate1")
                if stopped is not None:
                    return stopped
                continue

            if gate1.normal.generation is None:
                raise SCRCInfrastructureError(
                    "Gate1 reported closure without the exact Replay generation",
                    evidence={
                        "formal_round": formal_rounds,
                        "artifact": artifact.to_json(),
                        "typecheck": report.to_json(),
                        "gate1": gate1.to_json(),
                    },
                )

            attempts.append(
                CandidateAttemptReport(
                    lift_call=attempt_lift_call,
                    formal_round=formal_rounds,
                    typecheck_status=TypeCheckStatus.PASS,
                    artifact=artifact,
                    typecheck=report,
                    gate1=gate1,
                    lift_metadata=lift_metadata,
                )
            )
            return SCRCGate1SearchReport(
                status=Gate1SearchStatus.GATE1_CLOSED,
                attempts=tuple(attempts),
                accepted_attempt=formal_rounds,
                j_necessity_status=(
                    gate1.j_necessity_status
                    if gate1.masked_j is not None else None
                ),
                call_budget=budget(),
                stage_metrics=completed_metrics("gate1_closed"),
            )

        return SCRCGate1SearchReport(
            status=Gate1SearchStatus.NO_CLOSURE,
            attempts=tuple(attempts),
            accepted_attempt=None,
            j_necessity_status=None,
            call_budget=budget(),
            stop_reason=terminal_stop_reason,
            stop_observed_at=terminal_stop_observed_at,
            stage_metrics=completed_metrics("no_closure"),
        )

    def retry_current_gate1_replay(
        self,
        request: SCRCRunRequest,
        search: SCRCGate1SearchReport,
        *,
        progress_callback: SCRCProgressCallback | None = None,
    ) -> SCRCGate1SearchReport:
        """Run one fresh Replay/Gate1 round for this runner's closed search.

        This is an in-memory continuation of the current Kernel lifecycle, not
        a persisted-parent API.  It accepts the closed search report rather
        than an artifact so callers cannot select an older candidate or use a
        rejected draft as the Replay base.  No Lift call is permitted.
        """

        if id(search) not in self._owned_closed_searches:
            raise ValueError(
                "Replay retry requires a Gate1-closed report produced by this "
                "current runner"
            )
        accepted = self._accepted_search_attempt(search)
        artifact = accepted.artifact
        assert artifact is not None
        retry_request = replace(
            request,
            max_gate1_rounds=1,
            max_pre_replay_rejections=1,
        )
        retried = self._search_gate1_transaction(
            retry_request,
            replay_retry_artifact=artifact,
            progress_callback=progress_callback,
        )
        if retried.call_budget.lift_calls != 0:
            raise SCRCInfrastructureError(
                "current-artifact Replay retry unexpectedly invoked Lift"
            )
        return retried

    @staticmethod
    def _accepted_search_attempt(
        search: SCRCGate1SearchReport,
    ) -> CandidateAttemptReport:
        if not search.closed or search.accepted_attempt is None:
            raise ValueError("Gate2 requires one Gate1-closed search result")
        accepted = [
            attempt
            for attempt in search.attempts
            if attempt.formal_round == search.accepted_attempt
        ]
        if len(accepted) != 1:
            raise SCRCInfrastructureError(
                "Gate1 search does not identify exactly one accepted attempt"
            )
        attempt = accepted[0]
        if (
            attempt.artifact is None
            or attempt.typecheck is None
            or not attempt.typecheck.passed
            or attempt.gate1 is None
            or not attempt.gate1.passed
            or attempt.gate1.normal.generation is None
        ):
            raise SCRCInfrastructureError(
                "Gate1 search accepted attempt lacks certified pre-Gate2 evidence"
            )
        return attempt

    def complete_gate2(
        self,
        request: SCRCRunRequest,
        search: SCRCGate1SearchReport,
    ) -> SCRCRunReport:
        """Run Gate2 exactly once for a winner selected without Gate2 evidence."""

        gate2_total_started = time.monotonic()
        accepted = self._accepted_search_attempt(search)
        artifact = accepted.artifact
        typecheck_report = accepted.typecheck
        gate1 = accepted.gate1
        assert artifact is not None
        assert typecheck_report is not None
        assert gate1 is not None
        gate2_started = time.monotonic()
        try:
            gate2 = self.dependencies.gate2.execute(artifact, gate1, request)
        except Exception as exc:
            raise SCRCInfrastructureError(
                "Gate2 adapter failed after Gate1 closure: "
                f"{type(exc).__name__}: {exc}",
                evidence={
                    "artifact": artifact.to_json(),
                    "typecheck": typecheck_report.to_json(),
                    "gate1": gate1.to_json(),
                    "attempts": [item.to_json() for item in search.attempts],
                    "formal_round": search.accepted_attempt,
                    "stage_metrics": dict(search.stage_metrics),
                },
            ) from exc
        stage_metrics = {
            name: dict(value) for name, value in search.stage_metrics.items()
        }
        gate2_elapsed_ms = (time.monotonic() - gate2_started) * 1000.0
        stage_metrics["gate2"] = {
            "elapsed_ms": gate2_elapsed_ms,
            "runs": 1,
            "status": "passed" if gate2.passed else "failed",
            "cases": len(gate2.cases),
        }
        stage_metrics["gate2_total"] = {
            "elapsed_ms": (time.monotonic() - gate2_total_started) * 1000.0,
            "runs": 1,
            "status": "passed" if gate2.passed else "failed",
        }
        budget = search.call_budget
        case_count = len(gate2.cases)
        return SCRCRunReport(
            main_status=(
                MainStatus.SCRC_CERTIFIED
                if gate2.passed else MainStatus.GATE2_FAILED
            ),
            attempts=search.attempts,
            accepted_attempt=search.accepted_attempt,
            gate2=gate2,
            j_necessity_status=search.j_necessity_status,
            call_budget=CallBudget(
                lift_calls=budget.lift_calls,
                lift_physical_calls=budget.lift_physical_calls,
                typecheck_calls=budget.typecheck_calls,
                formal_gate1_rounds=budget.formal_gate1_rounds,
                replay_normal_calls=budget.replay_normal_calls,
                replay_masked_calls=budget.replay_masked_calls,
                replay_physical_calls=budget.replay_physical_calls,
                gate2_cases=case_count,
                source_executions=case_count,
                replay_executions=case_count,
                refeval_executions=case_count,
            ),
            stage_metrics=stage_metrics,
        )

    def run(
        self,
        request: SCRCRunRequest,
        *,
        progress_callback: SCRCProgressCallback | None = None,
    ) -> SCRCRunReport:
        search = self.search_gate1(
            request,
            progress_callback=progress_callback,
        )
        if search.closed:
            return self.complete_gate2(request, search)
        return SCRCRunReport(
            main_status=MainStatus.NO_CLOSURE,
            attempts=search.attempts,
            accepted_attempt=None,
            gate2=None,
            j_necessity_status=None,
            call_budget=search.call_budget,
            stage_metrics=dict(search.stage_metrics),
        )


__all__ = [
    "CallBudget",
    "CandidateLiftError",
    "CandidateAttemptReport",
    "FeedbackKind",
    "Gate1SearchStatus",
    "Gate1AttemptAdapter",
    "Gate2ExecutionAdapter",
    "HarnessBindingInstance",
    "LiftAdapter",
    "LiftCallResult",
    "LiftFeedback",
    "MainStatus",
    "MAX_GATE1_ROUNDS",
    "MAX_LIFT_CALLS",
    "MAX_PRE_REPLAY_REJECTIONS",
    "SCRCGateRunner",
    "SCRCGate1SearchReport",
    "SCRCInfrastructureError",
    "SCRCProgressCallback",
    "SCRCRunDependencies",
    "SCRCRunReport",
    "SCRCRunRequest",
    "SourceProgram",
]
