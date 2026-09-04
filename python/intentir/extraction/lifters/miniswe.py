"""mini-swe-agent Lift backend; SCRC verification remains entirely external."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from intentir.extraction.providers.provider import (
    ProviderConfig,
    get_provider,
)

from .scrc import (
    CANDIDATE_CHECK_PROTOCOL,
    SCRCExtraction,
    SCRCLiftAttempt,
    SCRCLiftRequest,
    SCRCProvenance,
    SCRCReliftContext,
    SCRCReliftRecord,
    SOURCE_AWARE_LIFT_TURN_TIMEOUT_SECONDS,
    SOURCE_AWARE_LIFT_XHIGH_TURN_TIMEOUT_SECONDS,
    _unbound_lift_artifact_payload,
    _value_free_lift_candidate_payload,
    decode_scrc_lift_artifact,
    check_scrc_lift_candidate,
    prepare_scrc_lift,
    resolve_source_aware_lift_timeout,
    scrc_candidate_check_context,
)
from .scrc_schema import scrc_lift_core_model_json_schema
from .prompt import SCHEMA as INTENTIR_OPERATION_REFERENCE


MINISWE_LIFT_BACKEND = "minisweagent.intentir-lift.v4"
MINISWE_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v70"
# Lift does not own a framework wall-clock deadline.  The persistent owner
# history is retained so the next bounded logical Lift can resume without
# duplicating a transport retry.
MINISWE_LIFT_TURN_TIMEOUT_SECONDS = SOURCE_AWARE_LIFT_TURN_TIMEOUT_SECONDS
MINISWE_LIFT_XHIGH_TURN_TIMEOUT_SECONDS = (
    SOURCE_AWARE_LIFT_XHIGH_TURN_TIMEOUT_SECONDS
)
# One logical Lift invocation still has a model-step budget.  The branch-level
# runner owns the separate maximum of three logical Lift invocations; neither
# budget is a wall-clock deadline.
LIFT_MODEL_STEP_LIMIT = 3
LIFT_INVOCATION_WALL_TIME_SECONDS: float | None = None
MAX_RELIFT_CHECK_FEEDBACK_ITEMS = 20
MAX_RELIFT_CHECK_FEEDBACK_CHARS = 2048
MAX_FEEDBACK_FRONTIER_EXAMPLES = 6
_FEEDBACK_CATEGORY_PRIORITY = {
    "interface": 0,
    "reduction": 1,
    "predicate_effect": 2,
    "op_multiset": 3,
    "dataflow": 4,
}


def _lift_invocation_limits(
    worker_config: Mapping[str, Any] | None = None,
) -> tuple[int, float | None]:
    """Resolve one logical Lift invocation's step/time policy.

    The YAML remains the normal configuration surface.  A zero or null wall
    time means unbounded; only the model-step budget remains capped at three.
    An explicit positive finite wall time is retained as an opt-in caller
    override and is never silently narrowed.
    """

    try:
        from minisweagent.run.intentir_lift import load_intentir_lift_config

        raw_agent = load_intentir_lift_config().get("agent", {})
    except ImportError:
        # Keep the later MiniSWE import as the authoritative dependency check.
        # A missing or malformed installed config must not silently turn into a
        # different hard-coded runtime contract.
        raw_agent = {}
    agent = dict(raw_agent) if isinstance(raw_agent, Mapping) else {}
    if isinstance(worker_config, Mapping):
        overrides = worker_config.get("agent")
        if isinstance(overrides, Mapping):
            agent.update(overrides)

    raw_steps = agent.get("step_limit", LIFT_MODEL_STEP_LIMIT)
    if raw_steps == 0:
        raw_steps = LIFT_MODEL_STEP_LIMIT
    if type(raw_steps) is not int or raw_steps < 1:
        raise ValueError("Lift agent step_limit must be a positive integer")

    raw_wall = agent.get("wall_time_limit_seconds", LIFT_INVOCATION_WALL_TIME_SECONDS)
    if raw_wall is None or raw_wall == 0:
        resolved_wall = None
    elif (
        isinstance(raw_wall, bool)
        or not isinstance(raw_wall, (int, float))
        or not math.isfinite(float(raw_wall))
        or raw_wall <= 0
    ):
        raise ValueError(
            "Lift agent wall_time_limit_seconds must be null, zero, or a "
            "finite positive number"
        )
    else:
        resolved_wall = float(raw_wall)
    return min(raw_steps, LIFT_MODEL_STEP_LIMIT), resolved_wall


def _lift_turn_timeout(
    effort: str,
    requested: float | None = None,
) -> float | None:
    """Resolve one provider turn under the source-aware timeout contract."""

    return resolve_source_aware_lift_timeout(effort, requested)


def _safe_relift_candidate(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a bounded value-free draft, or no continuation on malformed data."""

    if value is None or not isinstance(value, Mapping):
        return None
    try:
        return _value_free_lift_candidate_payload(value)
    except (TypeError, ValueError, RecursionError, json.JSONDecodeError):
        return None


def _bounded_feedback_items(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [copy.deepcopy(item) for item in value[:limit]]


def _is_provider_timeout(error: BaseException | None) -> bool:
    if error is None:
        return False
    try:
        import litellm
    except ImportError:
        return False
    return isinstance(error, litellm.exceptions.Timeout)


def _feedback_frontier(
    *,
    failure_kind: str | None,
    relift_round: int | None,
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Project closed diagnostics into a small, diagnostic-only first view.

    The exact feedback remains mounted separately.  This projection only drops
    bulk recursive witnesses and orders already-labelled categories; it never
    infers a cause, chooses a repair, or constructs an artifact field.
    """

    if diagnostics is None:
        return None
    closed = dict(diagnostics)
    observer = closed.get("observer_diff")
    if not isinstance(observer, Mapping):
        unresolved = closed.get("unresolved_gate1")
        unresolved_diagnostics = (
            unresolved.get("diagnostics")
            if isinstance(unresolved, Mapping)
            else None
        )
        observer = (
            unresolved_diagnostics.get("observer_diff")
            if isinstance(unresolved_diagnostics, Mapping)
            else None
        )

    categories: list[dict[str, Any]] = []
    raw_categories = observer.get("categories") if isinstance(observer, Mapping) else None
    if isinstance(raw_categories, list):
        for index, category in enumerate(raw_categories):
            if not isinstance(category, Mapping):
                continue
            name = str(category.get("category", "unknown"))
            item: dict[str, Any] = {
                "category": name,
                "difference_count": category.get("difference_count"),
                "examples": _bounded_feedback_items(
                    category.get("examples"), MAX_FEEDBACK_FRONTIER_EXAMPLES
                ),
                "full_query": (
                    "python /input/tools/context_query.py feedback-category " + name
                ),
            }
            if name == "dataflow":
                local = category.get("local_frontier")
                if isinstance(local, Mapping):
                    item["local_frontier"] = {
                        key: copy.deepcopy(local[key])
                        for key in ("protocol", "difference_count", "diagnostic")
                        if key in local
                    }
                    item["local_frontier"]["examples"] = _bounded_feedback_items(
                        local.get("examples"), MAX_FEEDBACK_FRONTIER_EXAMPLES
                    )
                effect_order = _bounded_feedback_items(
                    category.get("effect_order_witnesses"),
                    MAX_FEEDBACK_FRONTIER_EXAMPLES,
                )
                if effect_order:
                    item["effect_order_witnesses"] = effect_order
            item["source_category_index"] = index
            categories.append(item)
        categories.sort(
            key=lambda item: (
                _FEEDBACK_CATEGORY_PRIORITY.get(str(item["category"]), 10),
                str(item["category"]),
            )
        )

    frontier: dict[str, Any] = {
        "protocol": "intentir.lift-feedback-frontier.v1",
        "authority": (
            "deterministic bounded projection of closed diagnostics; "
            "diagnostic-only, no semantic inference or repair authority"
        ),
        "failure_kind": failure_kind,
        "relift_round": relift_round,
        "status": closed.get("status"),
        "diagnostic": closed.get("diagnostic"),
        "full_feedback_path": "/input/feedback.json",
        "categories": categories,
    }
    if isinstance(observer, Mapping):
        frontier["hard_passes"] = copy.deepcopy(observer.get("hard_passes", {}))
        frontier["primary_difference_category"] = observer.get(
            "primary_difference_category"
        )
    for key in ("repair_progress", "candidate_check_feedback"):
        if key in closed:
            frontier[key] = copy.deepcopy(closed[key])
    semantic_probe = closed.get("semantic_probe")
    if isinstance(semantic_probe, Mapping):
        frontier["semantic_probe"] = {
            key: copy.deepcopy(semantic_probe[key])
            for key in (
                "protocol",
                "status",
                "frontend",
                "case_id",
                "seed",
                "failed_leg",
                "unsupported",
                "diagnostic",
            )
            if key in semantic_probe
        }
        frontier["semantic_probe"]["full_feedback_path"] = (
            "/input/feedback.json"
        )
    if "unresolved_gate1" in closed:
        frontier["unresolved_gate1_query"] = (
            "python /input/tools/context_query.py feedback "
            "--path diagnostics.unresolved_gate1"
        )
    return frontier


def _usage_and_response_identity(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, int], str | None, str | None]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen_usage = False
    response_id: str | None = None
    response_model: str | None = None
    messages = trajectory.get("messages", ())
    if not isinstance(messages, list):
        return {}, None, None
    for message in messages:
        if not isinstance(message, Mapping) or message.get("object") != "response":
            continue
        if isinstance(message.get("id"), str):
            response_id = message["id"]
        if isinstance(message.get("model"), str):
            response_model = message["model"]
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            continue
        seen_usage = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
    return (totals if seen_usage else {}), response_id, response_model


def _trajectory_bytes(trajectory: Mapping[str, Any]) -> bytes:
    return json.dumps(
        trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _continued_lift_history(result: Any) -> list[dict[str, Any]]:
    """Keep valid conversation context while dropping the local terminal record."""

    messages = result.trajectory.get("messages", ())
    if not isinstance(messages, list):
        raise TypeError("Lift trajectory messages must be an array")
    # Only trim the outer sequence here. ``DefaultAgent.run_continued`` performs
    # the single defensive deepcopy before mutation, so copying every historical
    # payload again at each formal relift would add avoidable O(history) work.
    history = list(messages)
    while history and history[-1].get("role") == "exit":
        history.pop()
    if result.exit_status in {
        "Submitted",
        "SubmittedCheckpoint",
        "InvalidSubmission",
    } and history:
        # Submission is represented by a final tool call whose environment-side
        # success interrupted the old worker before a tool result was appended.
        # It cannot be replayed as an unmatched call in the continued Responses
        # conversation. The exact candidate remains in the branch-local
        # persistent Lift workspace.
        terminal_response = history[-1]
        extra = terminal_response.get("extra")
        actions = extra.get("actions", ()) if isinstance(extra, Mapping) else ()
        if not isinstance(actions, list) or not actions:
            raise ValueError(
                "submitted Lift history lacks its unmatched terminal tool call"
            )
        history.pop()
    if not history or any(item.get("role") == "exit" for item in history):
        raise ValueError("Lift continuation history is empty or contains exit state")
    return history


def _round_trajectory(
    trajectory: Mapping[str, Any],
    continued_from_messages: int,
) -> dict[str, Any]:
    """Return per-invocation evidence while the private owner keeps full context."""

    value = copy.deepcopy(dict(trajectory))
    messages = value.get("messages")
    if not isinstance(messages, list):
        raise TypeError("Lift trajectory messages must be an array")
    if not 0 <= continued_from_messages <= len(messages):
        raise ValueError("invalid Lift continuation prefix length")
    value["messages"] = messages[continued_from_messages:]
    return value


def _sanitize_trajectory(value: Any, secret: str, *, key: str = "") -> Any:
    normalized_key = key.casefold().replace("-", "_")
    if normalized_key in {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "password",
        "secret",
    } or normalized_key.endswith("_api_key"):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_trajectory(
                item,
                secret,
                key=str(item_key),
            )
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_trajectory(item, secret) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_trajectory(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "<redacted>")
    return value


def _worker_evidence(
    result: Any,
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    info = trajectory.get("info", {})
    candidate_check = (
        info.get("candidate_check", {}) if isinstance(info, Mapping) else {}
    )
    agent_runtime = (
        info.get("agent_runtime", {}) if isinstance(info, Mapping) else {}
    )
    return {
        "backend": MINISWE_LIFT_BACKEND,
        "knowledge_profile": MINISWE_LIFT_KNOWLEDGE,
        "exit_status": result.exit_status,
        "model_turns": result.model_calls,
        "physical_provider_calls": result.physical_provider_calls,
        "transport_retries": result.transport_retries,
        "elapsed_ms": result.elapsed_ms,
        "fresh_session": True,
        "workflow": "single",
        "stage_metrics": dict(getattr(result, "stage_metrics", {})),
        "candidate_check": (
            dict(candidate_check)
            if isinstance(candidate_check, Mapping)
            else {}
        ),
        "agent_runtime": (
            dict(agent_runtime) if isinstance(agent_runtime, Mapping) else {}
        ),
        "trajectory": trajectory,
    }


def _last_candidate_check_feedback(
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the last evaluated bounded, source-free diagnostic summary."""

    candidate_check = evidence.get("candidate_check")
    if not isinstance(candidate_check, Mapping):
        return None
    results = candidate_check.get("results")
    if not isinstance(results, list):
        return None
    for result in reversed(results):
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "fail"
            or result.get("evaluated") is not True
        ):
            continue
        protocol = result.get("protocol")
        rule_diagnostics = result.get("rule_diagnostics")
        if not isinstance(rule_diagnostics, list):
            rule_diagnostics = result.get("rule_paths")
        if protocol != CANDIDATE_CHECK_PROTOCOL or not isinstance(
            rule_diagnostics, list
        ):
            continue
        summary: list[dict[str, Any]] = []
        for item in rule_diagnostics[:MAX_RELIFT_CHECK_FEEDBACK_ITEMS]:
            if not isinstance(item, Mapping):
                continue
            rule_code = item.get("rule_code")
            path = item.get("path")
            if (
                not isinstance(rule_code, str)
                or not isinstance(path, str)
                or rule_code == "check_budget_exhausted"
            ):
                continue
            bounded_item: dict[str, Any] = {
                "rule_code": rule_code[:MAX_RELIFT_CHECK_FEEDBACK_CHARS],
                "path": path[:MAX_RELIFT_CHECK_FEEDBACK_CHARS],
            }
            observed = item.get("observed_refs_or_types")
            if isinstance(observed, list):
                bounded_observed = [
                    value[:MAX_RELIFT_CHECK_FEEDBACK_CHARS]
                    for value in observed[:MAX_RELIFT_CHECK_FEEDBACK_ITEMS]
                    if isinstance(value, str)
                ]
                if bounded_observed:
                    bounded_item["observed_refs_or_types"] = bounded_observed
            summary.append(bounded_item)
        if summary:
            return {"protocol": protocol, "diagnostics": summary}
    return None


def _input_files(request: SCRCLiftRequest) -> dict[str, str]:
    packet = json.loads(request.user_input)
    anonymous_source = packet.get("anonymous_source")
    if not isinstance(anonymous_source, str) or not anonymous_source:
        raise ValueError("prepared Lift packet lacks anonymous Source")
    harness_view = packet.get("harness_compiler_view", {})
    feedback = packet.get("relift")
    feedback_for_mount = copy.deepcopy(feedback)
    feedback_frontier = None
    if isinstance(feedback, Mapping):
        feedback_frontier = _feedback_frontier(
            failure_kind=(
                feedback.get("failure_kind")
                if isinstance(feedback.get("failure_kind"), str)
                else None
            ),
            relift_round=(
                feedback.get("relift_round")
                if isinstance(feedback.get("relift_round"), int)
                and not isinstance(feedback.get("relift_round"), bool)
                else None
            ),
            diagnostics=(
                feedback.get("diagnostics")
                if isinstance(feedback.get("diagnostics"), Mapping)
                else None
            ),
        )
    if isinstance(feedback_for_mount, dict):
        # The persistent owner already retains candidate.json.  Carrying the
        # complete prior artifact again in feedback.json duplicates tens of KiB
        # on every continuation and competes with the focused Gate frontier.
        # Removing this transport copy does not alter or derive artifact bytes.
        feedback_for_mount.pop("previous_artifact", None)
        diagnostics = feedback_for_mount.get("diagnostics")
        # Relift consumes the bounded typed Gate1 frontier, never the previous
        # source-blind implementation.  Strip legacy nested payloads as well so
        # old retained feedback cannot reopen a Replay -> Lift information path.
        if isinstance(diagnostics, dict):
            diagnostics.pop("replay_counterexample", None)
            unresolved = diagnostics.get("unresolved_gate1")
            nested = (
                unresolved.get("diagnostics")
                if isinstance(unresolved, dict)
                else None
            )
            if isinstance(nested, dict):
                nested.pop("replay_counterexample", None)
    from minisweagent import package_dir

    validator = (
        Path(package_dir) / "config" / "intentir_validate_candidate.py"
    ).read_text(encoding="utf-8")
    preflight = (
        Path(package_dir) / "config" / "intentir_lift_preflight.md"
    ).read_text(encoding="utf-8")
    schema_query = (
        Path(package_dir) / "config" / "intentir_schema_query.py"
    ).read_text(encoding="utf-8")
    context_query = (
        Path(package_dir) / "config" / "intentir_context_query.py"
    ).read_text(encoding="utf-8")
    minimal_example = (
        Path(package_dir) / "config" / "intentir_lift_example.json"
    ).read_text(encoding="utf-8")
    readme = """# IntentIR Lift input

This directory is the complete allowlisted input for one branch-local Lift
owner. Its conversation and persistent workspace survive across formal Gate1
continuations for the same Source; Replay remains a separate fresh process.

- `source.py`: alpha-renamed Source available only to Lift.
- `harness_view.json`: value-free anonymous ABI, shape relations and typed
  Layer-C declarations. It contains no runtime scalar/tensor valuation, C value,
  launch/grid, schedule, target or compiler metadata.
- Trusted Source compilation/TTIR is host-private and is neither mounted in this
  workspace nor used as a J authorship/Lift-acceptance authority.
- `feedback.json`: null on the first round, otherwise only the explicitly
  allowed previous parser/TypeCheck/public Source-vs-J/Gate1/Gate2 feedback,
  previous candidate, or the last worker's bounded source-free CandidateCheck
  diagnostic summary.
- `feedback_frontier.json`: deterministic bounded projection of the same closed
  feedback. Read this first; it orders already-labelled hard categories and
  omits bulk recursive witnesses without inferring a cause or repair. Query the
  exact `feedback.json` only for a named unresolved category/path.
- Formal Gate1 continuations expose only bounded typed observer differences.
  Replay source, Replay TTIR, Replay context, and physical implementation choices
  remain unavailable to Lift; use Source plus the typed frontier to decide whether
  J/C is incomplete, otherwise request a fresh source-blind Replay unchanged.
- `inspect_lift_context(view="feedback_category", query="CATEGORY")` (or the
  matching `context_query.py feedback-category CATEGORY`) returns just that
  exact bounded category, including local/effect-order witnesses when present.
- `schema/artifact.schema.json`: compact machine-only closed JSON envelope for
  `(J,C,pi)`; field prose remains in the focused contract/preflight references.
- `knowledge/lift_contract.md`: semantic ownership and construction rules.
- `knowledge/intent_ops.md`: global source-neutral IntentIR operation reference.
- `knowledge/preflight.md`: short source-neutral handle/ownership checklist.
- `knowledge/minimal_example.json`: source-neutral wire/ownership example for one
  integral rank-0 scalar explicitly equated to a logical dimension by a J-owned
  value_dimension_relations entry, explicit sum-then-div stages, one exact cast
  refinement, a local reduction axis, and explicit signature-carrier routes to
  both transfers; replace every case-specific semantic, handle, shape, value,
  topology, and multiplicity.
- `tools/validate_candidate.py`: syntax/static-schema validator only.
- `intentir_candidate_check /output/candidate.json`: read-only
  Decode/TypeCheck diagnostics for the complete candidate; calls are audited
  telemetry and are governed by the owner session's active wall time;
  it never repairs or certifies the artifact.
- `intentir_candidate_commit /output/candidate.json`: record only exact bytes
  already returned PASS by CandidateCheck as the LLM owner's conditional
  fallback; fixed code does not parse or modify fields.
- `intentir_candidate_rollback /output/candidate.json`: restore only those exact
  LLM-committed bytes.
- `intentir_lift_submit /output/candidate.json` or
  `intentir_lift_submit --committed`: direct structured delivery without a
  stdout marker. Direct submit is the normal fast path: it runs CandidateCheck
  on exact current bytes when needed and exits immediately on PASS; external
  Decode/TypeCheck and the public Source-vs-J diagnostic still run independently
  before Replay/Gate1.
- `submit_candidate(candidate_text)`: preferred first-candidate tool. It writes
  the complete exact LLM-authored text and invokes the same preflight in one
  action; it does not construct, normalize, or select a field.
- `apply_patch`: replace one exact, uniquely occurring text span in the existing
  `/output/candidate.json`. The model supplies both strings; fixed code performs
  no handle allocation, reordering, inference, or semantic edit. Set
  `submit=true` to invoke the existing preflight after that exact replacement.
- `tools/schema_query.py`: compact schema lookup (`top`, `catalog`, `op`,
  `section`, `bundle`) so raw 1400-line schema never needs to be paged end-to-end.
  The schema path defaults to the mounted artifact schema. Prefer one exact
  `bundle op:NAME ... section:NAME ...` call once the required operations and
  relations are known instead of spending one provider turn per leaf.
- `tools/context_query.py`: bounded lookup for bootstrap, exact external
  feedback, paged Source, Harness JSON paths, and focused contract/preflight
  matches. Its `harness-abi` command joins the already-authoritative p*/c*,
  bound-non-C constexpr, and shape-coordinate declarations into one compact
  ledger. Its `candidate-abi` command compares the current draft with only
  Harness-authored mechanical ABI facts and never proposes semantic fields.
  Its paged `candidate-control` views project only the current LLM-authored
  control/reduction nodes, output/effect facts, exact node-operand/access joins,
  and pi time/control topology.
  Its syntax-only `source-outline`, literal `source-find` search, and paged `candidate-nodes`/single
  `candidate-node` views expose only Source and J content already visible to
  this worker; they never parse semantics, construct fields, or edit the draft.
  `source-outline` also lists literal descendant-return presence, ordered
  load/store/atomic call sites, and names that were lexically bound before a loop
  and written inside it. These remain navigation checklists, not inferred
  effects, carriers, or frontend tuple order.
  No Source-TTIR, compiler-observer, launch/grid, schedule, target, or physical
  realization query exists in the Lift workspace because those facts cannot
  author or reject J=A+B.
  Its `feedback --path`, `candidate --path`, and `candidate-find` commands are
  bounded read-only inspection over already-mounted data and the current draft.
  `bootstrap` copies an explicitly supplied previous artifact only when no
  candidate exists. In the persistent owner workspace it preserves the current
  candidate unchanged; no command constructs or repairs artifact fields.

When the initial user packet says `exact_inputs_embedded=false`, start with this
single bounded command so long inputs cannot hide feedback. Otherwise exact
Source/Harness/feedback/minimal-example bytes are already present; the longer
source-neutral contract/preflight remain exact mounted files with hash receipts
and should be queried only for relevant rules. This inventory call is optional:

`python /input/tools/context_query.py bootstrap`

Then inspect Source, Harness, and the syntax-only outline with independent calls
that may share one model response:
`python /input/tools/context_query.py source` and
`python /input/tools/context_query.py harness-abi`, plus
`python /input/tools/context_query.py source-outline --start 1 --count 32`.
On repair rounds, inspect
`python /input/tools/context_query.py feedback-frontier` before editing the candidate
already retained in the workspace. Use its `feedback-category`, paged `source`, Harness-path,
and `knowledge` subcommands for narrower views.
The continuation message never re-embeds candidate.json: the same owner reads
the exact retained bytes from `/output/candidate.json`, avoiding duplicate
10--100 KiB artifact context on every formal relift.
For a long Source, run
`python /input/tools/context_query.py source-outline --start 1 --count 32`
and page it before deciding that no loop, branch, or early return exists. The
outline contains syntax facts only; it never classifies or constructs J. For
every loop, inspect each `lexically_preexisting_writes` item and decide the
complete carrier set from Source. For each selected reachable helper, preserve
its literal memory-call order in authored scope.effects and pi order; never use
J node declaration order as an effect-order substitute.
Use `feedback --path diagnostics.KEY` for an exact unresolved Gate1 or
CandidateCheck section; do not probe category names or parse the full file with
ad-hoc scripts. Use `candidate --path j.PATH` and `candidate-find HANDLE` to
inspect a large current draft without rewriting it.
Gate1 `dataflow.node_witnesses` use `source_node_*`, `replay_node_*`, and
`shared_node_*` aliases only to preserve repeated observer identity within that
one feedback. Compare their normalized local signatures and incident
producer/consumer roles to localize the earliest wiring divergence; never copy
an observer alias into J/C/pi.
When present, inspect `dataflow.local_frontier` first. It removes recursive
observer-hash fallout and lists bounded immediate producer/consumer profile
differences; it is diagnostic-only, cannot accept a candidate, and does not
prescribe any J/C/pi field.
After writing or editing a complete draft, inspect the bounded public-ABI diff
and the authored control/effect/output ledger without paging the whole artifact:

`python /input/tools/context_query.py candidate-abi`

`python /input/tools/context_query.py candidate-control`

For a graph where one logical tensor is viewed through different indices at
different consumers, use `candidate-control --section operands`; it only joins
the draft's exact node operands with accesses already attached to that node and
does not infer that any empty entry should be filled.

For multi-output/effect cases follow with `candidate-control --section outputs`;
for loop, carrier, terminal-control, route, or order obligations page the `pi`
section. These are read-only views of exact LLM-authored fields, never a composer.

Do not page through the raw schema. Use the example plus compact queries such as:

`python /input/tools/schema_query.py /input/schema/artifact.schema.json top`

`python /input/tools/schema_query.py /input/schema/artifact.schema.json catalog`

Query exact `op NAME` or leaf `section NAME` only when needed. Useful leaf
sections include `access`, `effect`, `result_relation`, `authorization`,
`scope`, `c_slot`, `pi_instance`, `pi_route`, `pi_program_axis`, `pi_iteration`, and
`pi_refinement`. Create the
first complete candidate early, then let validator errors drive further lookup.
On a continuation, keep editing the current complete candidate in the persistent
    workspace; the external previous-artifact receipt is intentionally omitted
    because the exact working candidate is already retained. Use `repair_progress`
    when present and make the
Source-justified coordinated J/C/pi changes needed by the external
counterexample; do not blindly rebuild or mechanically resubmit the same
artifact. Run `python /input/tools/context_query.py candidate-diff` before
submission. This is only a read-only diff of two LLM-authored candidates, never
a repair. The repaired full object is still a new Lift submission and will be
checked externally again.
If current Lift/TypeCheck diagnostics contain `unresolved_gate1`, the enclosed
Gate1 counterexample remains an open obligation across the intermediate
construction failure; address both without inventing fields.
Use `python /input/tools/context_query.py source-find TEXT --context 3` to
locate literal Source occurrences without paging the whole file. For a large
draft, use `candidate-nodes --start 1 --count 40` and `candidate-node n0` to
inspect the graph that you authored. These commands are read-only discovery,
not semantic extraction or repair.
If bootstrap copied a `continuation_draft`, it is only the adjacent prior
worker's complete LLM-written JSON draft, not a decoded or accepted artifact.
Continue its focused repair and rerun validator/CandidateCheck yourself.
For the first complete candidate, prefer the `submit_candidate` tool so writing
and preflight occur in this same model response. For an existing ready file, use:

`intentir_lift_submit /output/candidate.json`

Submission runs the source-free CandidateCheck preflight on unchecked exact
bytes. It exits this Lift invocation on PASS and returns short diagnostics on
failure. The static validator and explicit diagnostic command remain available
when useful, but are not mandatory network turns:

`intentir_candidate_check /output/candidate.json`

This host virtual command must be the entire shell action exactly as shown. Do
not prepend `set`, an assignment, a pipe, or another shell command; such a
prefixed action runs inside the sandbox, where it is not an executable.
If it fails, edit the candidate yourself from the closed diagnostics and rerun
within the branch-owner active wall-time budget. The formal Runner independently repeats Decode
and TypeCheck after submission; CandidateCheck is not an acceptance path.
If the exact submitted bytes have no prior CandidateCheck PASS, submission
automatically invokes the same read-only preflight and returns FAIL diagnostics
to this session instead of delivering the candidate. It still never edits the
object, and a PASS still has no authority outside this worker.
Old tool calls and observations may appear as deterministic hash receipts in the
model packet. Re-read the persistent files for exact content; the runtime never
summarizes or interprets J/C/pi semantics during compaction.
"""
    files = {
        "README.md": readme,
        "source.py": anonymous_source,
        # These files are read through JSON-aware lookup tools.  Keep the
        # mounted transport compact; the semantic values are unchanged and
        # the focused query commands provide readable bounded projections.
        "harness_view.json": json.dumps(
            harness_view,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        "feedback.json": json.dumps(
            feedback_for_mount,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        "feedback_frontier.json": json.dumps(
            feedback_frontier,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        "schema/artifact.schema.json": json.dumps(
            scrc_lift_core_model_json_schema(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        "knowledge/lift_contract.md": request.instructions,
        "knowledge/intent_ops.md": INTENTIR_OPERATION_REFERENCE,
        "knowledge/preflight.md": preflight,
        "knowledge/minimal_example.json": minimal_example,
        "tools/validate_candidate.py": validator,
        "tools/schema_query.py": schema_query,
        "tools/context_query.py": context_query,
    }
    return files


@dataclass
class MiniSWELiftAdapter:
    """One branch-local source-aware owner across successive LiftAdapter calls."""

    provider: ProviderConfig | None = None
    compiler_view: Any = None
    effort: str | None = None
    source_symbol: str | None = None
    workflow: str = "single"
    worker_config: Mapping[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None
    session_wall_time_limit_seconds: float | None = None
    stop_requested: Callable[[], int | None] | None = field(
        default=None, repr=False
    )
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)
    _session_id: str = field(default_factory=lambda: uuid.uuid4().hex, init=False)
    _session_history: list[dict[str, Any]] | None = field(default=None, init=False)
    _session_source_sha256: str | None = field(default=None, init=False)
    _session_active_elapsed_ms: float = field(default=0.0, init=False)
    _session_model_turns: int = field(default=0, init=False)
    _session_physical_provider_calls: int = field(default=0, init=False)
    _session_transport_retries: int = field(default=0, init=False)
    _session_workspace: Any = field(default=None, init=False, repr=False)
    _session_input_snapshot: dict[str, str | bytes] | None = field(
        default=None, init=False, repr=False
    )
    _session_model: Any = field(default=None, init=False, repr=False)
    _session_model_identity: tuple[str, str, str] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def close(self) -> None:
        workspace = self._session_workspace
        self._session_workspace = None
        self._session_input_snapshot = None
        self._session_model = None
        self._session_model_identity = None
        if workspace is not None:
            workspace.cleanup()

    def _create_or_refresh_session_workspace(
        self,
        files: Mapping[str, str | bytes],
        *,
        mutable_paths: frozenset[str],
    ) -> bool:
        """Keep one owner's tool/knowledge snapshot stable across host edits.

        Source and Harness bytes are request-owned and must remain identical.
        Tool, schema, example, and knowledge bytes are session-owned: a branch
        continues with the exact snapshot mounted when that owner was created,
        even if another host task updates the repository concurrently.  Only
        the explicitly closed feedback files are refreshed.
        """

        from minisweagent.environments.intentir_lift import IntentIRLiftWorkspace

        incoming = dict(files)
        if self._session_workspace is None:
            self._session_input_snapshot = dict(incoming)
            self._session_workspace = IntentIRLiftWorkspace.create(incoming)
            return True

        snapshot = self._session_input_snapshot
        if snapshot is None:
            raise RuntimeError("persistent Lift workspace lacks its input snapshot")
        if set(incoming) != set(snapshot):
            raise ValueError("persistent Lift input file set changed")
        for relative in ("source.py", "harness_view.json"):
            if incoming.get(relative) != snapshot.get(relative):
                raise ValueError(
                    f"persistent Lift request input changed: {relative}"
                )
        if not mutable_paths.issubset(incoming):
            missing = sorted(mutable_paths - set(incoming))
            raise ValueError(
                "persistent Lift mutable input is missing: " + ", ".join(missing)
            )

        refreshed = dict(snapshot)
        for relative in mutable_paths:
            refreshed[relative] = incoming[relative]
        self._session_workspace.refresh_inputs(
            refreshed,
            mutable_paths=mutable_paths,
        )
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _latest_rejected_candidate(
        self,
        diagnostics: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        """Return only the adjacent JSON object named by parser feedback."""

        if (
            not self._records
            or not isinstance(diagnostics, Mapping)
            or diagnostics.get("failure_class")
            not in {"lift_response", "candidate_check_rejected"}
        ):
            return None
        record = self._records[-1]
        if (
            record.relift_round != len(self._records)
            or record.extraction is not None
            or len(record.rejected_attempts) != 1
        ):
            return None
        attempt = record.rejected_attempts[0]
        if diagnostics.get("diagnostic") != attempt.diagnostic:
            return None

        def reject_nonfinite(token: str) -> None:
            raise ValueError(f"non-finite JSON number: {token}")

        try:
            candidate = json.loads(
                attempt.response_text,
                parse_constant=reject_nonfinite,
            )
            if not isinstance(candidate, dict):
                return None
            json.dumps(candidate, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            return None
        return candidate

    def _latest_continuation_candidate(
        self,
        diagnostics: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        """Return one adjacent LLM-written draft without accepting or editing it."""

        if (
            not self._records
            or not isinstance(diagnostics, Mapping)
            or diagnostics.get("failure_class") != "lift_worker_no_candidate"
        ):
            return None
        receipt = diagnostics.get("continuation_draft")
        if not isinstance(receipt, Mapping):
            return None
        expected_sha256 = receipt.get("sha256")
        record = self._records[-1]
        candidate = record.continuation_candidate
        if (
            record.relift_round != len(self._records)
            or record.extraction is not None
            or not isinstance(candidate, Mapping)
            or not isinstance(expected_sha256, str)
        ):
            return None
        encoded = json.dumps(
            candidate,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != expected_sha256:
            return None
        return dict(candidate)

    @staticmethod
    def _read_continuation_candidate(workspace: Any) -> dict[str, Any] | None:
        """Read a complete worker draft under the existing output policy."""

        def reject_nonfinite(token: str) -> None:
            raise ValueError(f"non-finite JSON number: {token}")

        try:
            candidate = json.loads(
                workspace.read_candidate(),
                parse_constant=reject_nonfinite,
            )
            if not isinstance(candidate, dict) or set(candidate) != {
                "schema",
                "j",
                "c",
                "pi",
            }:
                return None
            json.dumps(candidate, sort_keys=True, allow_nan=False)
        except (OSError, TypeError, ValueError, UnicodeError, RecursionError):
            return None
        return candidate

    def _relift_context(self, feedback: Any) -> SCRCReliftContext | None:
        if feedback is None:
            return None
        kind = getattr(getattr(feedback, "kind", None), "value", None)
        diagnostics = getattr(feedback, "diagnostics", None)
        previous_artifact = getattr(feedback, "previous_artifact", None)
        previous_candidate = getattr(feedback, "previous_candidate", None)
        if previous_candidate is not None:
            previous_candidate = _safe_relift_candidate(previous_candidate)
        if previous_candidate is None and kind == "lift":
            previous_candidate = self._latest_rejected_candidate(diagnostics)
        if previous_candidate is None and kind == "lift":
            previous_candidate = self._latest_continuation_candidate(
                diagnostics
            )
        previous_artifact_payload = (
            _unbound_lift_artifact_payload(previous_artifact.to_json())
            if hasattr(previous_artifact, "to_json")
            else None
        )
        # A persistent owner already has the exact current draft in its output
        # directory.  Do not duplicate it in the next model packet.  A fresh
        # owner keeps one value-free copy in the context so bootstrap/seed can
        # restore the same authored bytes without constructing an artifact.
        if self._session_workspace is not None and (
            self._read_continuation_candidate(self._session_workspace) is not None
        ):
            previous_candidate = None
            previous_artifact_payload = None
        return SCRCReliftContext(
            len(self._records) + 1,
            kind,
            diagnostics if isinstance(diagnostics, Mapping) else {},
            previous_artifact=previous_artifact_payload,
            previous_candidate=previous_candidate,
        )

    @staticmethod
    def _seed_previous_candidate(
        workspace: Any,
        context: SCRCReliftContext | None,
    ) -> None:
        """Copy one prior authored draft into a fresh owner workspace exactly."""

        if context is None:
            return
        target = workspace.candidate_path
        if target.exists():
            return
        payload = context.previous_candidate
        if payload is None:
            payload = context.previous_artifact
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            raise TypeError("relift seed must be a JSON object")
        target.write_text(
            json.dumps(
                dict(payload),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _continued_worker_config(self) -> dict[str, Any]:
        config = copy.deepcopy(dict(self.worker_config))
        if self.session_wall_time_limit_seconds in (None, 0):
            remaining_seconds = None
        elif (
            isinstance(self.session_wall_time_limit_seconds, bool)
            or not isinstance(self.session_wall_time_limit_seconds, (int, float))
            or not math.isfinite(float(self.session_wall_time_limit_seconds))
            or self.session_wall_time_limit_seconds < 0
        ):
            raise ValueError(
                "Lift owner session wall-time limit must be null, zero, or a "
                "finite non-negative number"
            )
        else:
            remaining_seconds = self.session_wall_time_limit_seconds - (
                self._session_active_elapsed_ms / 1000.0
            )
            if remaining_seconds <= 0:
                from intentir.verify.scrc_runner import CandidateLiftError

                raise CandidateLiftError(
                    {
                        "diagnostic": "branch-local Lift owner wall-time exhausted",
                        "failure_class": "lift_owner_time_exhausted",
                        "session_id": self._session_id,
                        "active_elapsed_ms": self._session_active_elapsed_ms,
                    },
                    physical_calls=0,
                    terminal=True,
                )
        step_limit, invocation_wall_time = _lift_invocation_limits(
            self.worker_config
        )
        agent = config.get("agent")
        agent_config = dict(agent) if isinstance(agent, Mapping) else {}
        # Keep the three-step work bound.  Wall time is zero when both the
        # invocation and owner policies are intentionally unbounded.
        agent_config["step_limit"] = step_limit
        wall_limits = [
            value
            for value in (remaining_seconds, invocation_wall_time)
            if value is not None
        ]
        agent_config["wall_time_limit_seconds"] = (
            max(1, math.ceil(min(wall_limits))) if wall_limits else 0
        )
        config["agent"] = agent_config
        return config

    def _continuation_message(self, feedback: Any, round_index: int) -> str:
        kind = getattr(getattr(feedback, "kind", None), "value", None)
        diagnostics = getattr(feedback, "diagnostics", None)
        closed_diagnostics = dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
        frontier = _feedback_frontier(
            failure_kind=kind,
            relift_round=round_index - 1,
            diagnostics=closed_diagnostics,
        )
        diagnostic_bytes = json.dumps(
            closed_diagnostics,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        candidate_text: str | None = None
        workspace = self._session_workspace
        if workspace is not None:
            try:
                candidate_text = workspace.read_candidate()
            except (OSError, TypeError, ValueError, UnicodeError):
                candidate_text = None
        candidate_bytes = (
            candidate_text.encode("utf-8") if candidate_text is not None else b""
        )
        envelope = {
            "protocol": "intentir.lift-owner-continuation.v1",
            "owner_session": self._session_id,
            "lift_invocation": round_index,
            "failure_kind": kind,
            "feedback_path": "/input/feedback.json",
            "feedback_frontier_path": "/input/feedback_frontier.json",
            "candidate_path": "/output/candidate.json",
            "exact_feedback_embedded": False,
            "feedback_frontier_embedded": frontier is not None,
            # The same owner already has the exact bytes in its persistent
            # workspace. Re-embedding a 10--100 KiB candidate on every formal
            # relift duplicates context without adding any information.
            "exact_candidate_embedded": False,
        }
        envelope["closed_feedback_receipt"] = {
            "bytes": len(diagnostic_bytes),
            "sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
        }
        envelope["current_candidate_receipt"] = (
            {
                "bytes": len(candidate_bytes),
                "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            }
            if candidate_text is not None
            else None
        )
        current_state = {
            "feedback_frontier": frontier,
            "closed_feedback": None,
            "candidate_text": None,
            "fallback": {
                "feedback": (
                    "read /input/feedback_frontier.json first; query only a named "
                    "category/path from /input/feedback.json when needed"
                ),
                "candidate": (
                    "reread /output/candidate.json from the persistent workspace"
                ),
            },
        }
        instructions = (
            "The same branch-local source-aware Lift task continues. "
            "Use the embedded diagnostic-only feedback frontier and current "
            "candidate immediately when present. Resolve direct hard categories "
            "before paging recursive dataflow witnesses; query exact feedback only "
            "for one named unresolved category/path. "
            "Modify the current candidate and resubmit one complete (J,C,pi). "
            "This is not a new "
            "writer and not a request for a patch-only artifact."
        )
        return (
            json.dumps(envelope, sort_keys=True, ensure_ascii=False)
            + "\n<persistent-workspace-mirror>\n"
            + json.dumps(current_state, sort_keys=True, ensure_ascii=False)
            + "\n</persistent-workspace-mirror>\n"
            + instructions
        )

    def _record_session_result(self, result: Any) -> None:
        self._session_active_elapsed_ms += float(result.elapsed_ms)
        self._session_model_turns += int(result.model_calls)
        self._session_physical_provider_calls += int(
            result.physical_provider_calls
        )
        self._session_transport_retries += int(result.transport_retries)
        self._session_history = _continued_lift_history(result)

    def _session_evidence(
        self,
        evidence: Mapping[str, Any],
        *,
        fresh_session: bool,
        result: Any,
    ) -> dict[str, Any]:
        value = dict(evidence)
        value.update(
            {
                "fresh_session": fresh_session,
                "persistent_owner": True,
                "owner_session_id": self._session_id,
                "continued_from_messages": result.continued_from_messages,
                "session_active_elapsed_ms": self._session_active_elapsed_ms,
                "session_model_turns": self._session_model_turns,
                "session_physical_provider_calls": (
                    self._session_physical_provider_calls
                ),
                "session_transport_retries": self._session_transport_retries,
                "session_wall_time_limit_seconds": (
                    self.session_wall_time_limit_seconds
                ),
            }
        )
        return value

    def lift(self, request: Any, feedback: Any) -> Any:
        source_program = getattr(request, "source", None)
        source = getattr(source_program, "text", None)
        if not isinstance(source, str) or not source:
            raise TypeError("MiniSWE Lift requires SourceProgram.text")
        if self.workflow != "single":
            raise ValueError(
                "MiniSWE Lift has one supported workflow: 'single'; staged "
                "planner/builder/reviewer ownership was retired by Task13"
            )
        source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        if self._session_source_sha256 is None:
            self._session_source_sha256 = source_sha256
        elif self._session_source_sha256 != source_sha256:
            raise ValueError("one Lift owner session cannot cross Source programs")
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        timeout = _lift_turn_timeout(
            effective_effort,
            self.model_timeout_seconds,
        )
        relift_context = self._relift_context(feedback)
        prepared = prepare_scrc_lift(
            source,
            compiler_view=self.compiler_view,
            relift_context=relift_context,
            source_symbol=(
                getattr(source_program, "lift_source_symbol", None)
                or self.source_symbol
            ),
            source_kind=getattr(source_program, "lift_source_kind", "python"),
            source_parameter_order=tuple(
                getattr(source_program, "lift_parameter_order", ())
            ),
            source_to_opaque=(
                dict(getattr(source_program, "lift_source_to_opaque", ()))
                or None
            ),
        )
        round_index = len(self._records) + 1
        fresh_owner_session = self._session_history is None
        effective_worker_config = self._continued_worker_config()
        configured_owner_remaining = effective_worker_config.get("agent", {}).get(
            "wall_time_limit_seconds"
        )
        if (
            timeout is not None
            and isinstance(configured_owner_remaining, (int, float))
            and not isinstance(configured_owner_remaining, bool)
            and configured_owner_remaining > 0
        ):
            timeout = min(timeout, float(configured_owner_remaining))
        continuation_history = (
            self._session_history
            if self.workflow == "single" and not fresh_owner_session
            else None
        )
        continuation_message = (
            self._continuation_message(feedback, round_index)
            if continuation_history is not None
            else None
        )
        candidate_check_context = scrc_candidate_check_context(prepared)
        try:
            from minisweagent.run.intentir_lift import (
                IntentIRLiftRunError,
                create_intentir_lift_model,
                run_intentir_lift,
            )
            from minisweagent.exceptions import MiniSWERequestCancelled
        except ImportError as exc:
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                "MiniSWE Lift backend requires `pip install -e '.[lift-agent,verify]'`",
                evidence={"backend": MINISWE_LIFT_BACKEND},
            ) from exc

        input_files = _input_files(prepared)
        try:
            fresh_workspace = self._create_or_refresh_session_workspace(
                input_files,
                mutable_paths=frozenset(
                    {
                        "feedback.json",
                        "feedback_frontier.json",
                    }
                ),
            )
        except ImportError as exc:
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                "MiniSWE Lift backend requires the lift-agent workspace dependency",
                evidence={"backend": MINISWE_LIFT_BACKEND},
            ) from exc
        workspace = self._session_workspace
        if workspace is None:
            raise RuntimeError("persistent MiniSWE Lift workspace was not created")
        if fresh_workspace:
            self._seed_previous_candidate(workspace, relift_context)
        result = None
        try:
            model_identity = (
                provider.model,
                provider.base_url,
                effective_effort.casefold(),
            )
            if self._session_model is None:
                self._session_model = create_intentir_lift_model(
                    model_name=provider.model,
                    api_base=provider.base_url,
                    api_key=provider.api_key,
                    reasoning_effort=effective_effort,
                    timeout_seconds=timeout,
                    prompt_cache_key=f"intentir-lift-{self._session_id}",
                )
                self._session_model_identity = model_identity
            elif self._session_model_identity != model_identity:
                raise ValueError(
                    "one Lift owner session cannot change provider model, endpoint, "
                    "or reasoning effort"
                )
            model = self._session_model
            set_cancel_requested = getattr(model, "set_cancel_requested", None)
            if callable(set_cancel_requested):
                set_cancel_requested(self.stop_requested)
            model_kwargs = getattr(
                getattr(model, "config", None), "model_kwargs", None
            )
            if isinstance(model_kwargs, dict):
                # Keep the model config JSON-safe; the Responses model converts
                # an explicit None at the final LiteLLM call boundary.
                model_kwargs["timeout"] = timeout
            run_kwargs: dict[str, Any] = {
                "model": model,
                "workspace": workspace,
                "config_overrides": effective_worker_config,
                "candidate_checker": lambda candidate_text: (
                    check_scrc_lift_candidate(
                        candidate_text,
                        candidate_check_context,
                    )
                ),
            }
            run_kwargs.update(
                {
                    "continuation_history": continuation_history,
                    "continuation_message": continuation_message,
                }
            )
            result = run_intentir_lift(
                **run_kwargs,
            )
            self._record_session_result(result)
        except IntentIRLiftRunError as exc:
            result = exc.result
            self._record_session_result(result)
            safe_trajectory = _round_trajectory(
                _sanitize_trajectory(result.trajectory, provider.api_key),
                result.continued_from_messages,
            )
            evidence = self._session_evidence(
                _worker_evidence(result, safe_trajectory),
                fresh_session=fresh_owner_session,
                result=result,
            )
            continuation_candidate = self._read_continuation_candidate(
                workspace
            )
            continuation_candidate = _safe_relift_candidate(
                continuation_candidate
            )
            continuation_receipt = None
            if continuation_candidate is not None:
                continuation_bytes = json.dumps(
                    continuation_candidate,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode()
                continuation_receipt = {
                    "sha256": hashlib.sha256(
                        continuation_bytes
                    ).hexdigest(),
                    "bytes": len(continuation_bytes),
                }
                evidence["continuation_draft"] = dict(
                    continuation_receipt
                )
            diagnostic = str(exc)
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=diagnostic,
                    backend_evidence=evidence,
                    continuation_candidate=continuation_candidate,
                )
            )
            if isinstance(exc.cause, MiniSWERequestCancelled):
                from intentir.verify.scrc_runner import CandidateLiftError

                raise CandidateLiftError(
                    {
                        "diagnostic": diagnostic,
                        "failure_class": "candidate_superseded",
                        "winner_branch_id": exc.cause.winner_branch_id,
                    },
                    physical_calls=result.physical_provider_calls,
                    candidate_draft=continuation_candidate,
                ) from exc
            if _is_provider_timeout(exc.cause):
                from intentir.verify.scrc_runner import CandidateLiftError

                raise CandidateLiftError(
                    {
                        "diagnostic": diagnostic,
                        "failure_class": "lift_provider_timeout",
                        "retryable": True,
                        "continuation": (
                            "the persistent Lift owner history is retained; the "
                            "next logical Lift may resume it"
                        ),
                    },
                    physical_calls=result.physical_provider_calls,
                    terminal=False,
                    candidate_draft=continuation_candidate,
                ) from exc
            if result.exit_status in {
                "LimitsExceeded",
                "TimeExceeded",
                "OrchestrationStall",
                "RepeatedFormatError",
            }:
                from intentir.verify.scrc_runner import CandidateLiftError

                rejected_diagnostics: dict[str, Any] = {
                    "diagnostic": (
                        "Lift owner reached its bounded invocation limit before "
                        "delivering a complete candidate"
                    ),
                    "failure_class": "lift_worker_no_candidate",
                    "termination_status": result.exit_status,
                    "resource_limited": True,
                    "retryable": True,
                    "continuation": (
                        "the persistent Lift owner workspace and conversation "
                        "remain available to the next logical Lift invocation"
                    ),
                }
                candidate_check_feedback = _last_candidate_check_feedback(evidence)
                if candidate_check_feedback is not None:
                    rejected_diagnostics["candidate_check_feedback"] = (
                        candidate_check_feedback
                    )
                if continuation_receipt is not None:
                    rejected_diagnostics["continuation_draft"] = dict(
                        continuation_receipt
                    )
                raise CandidateLiftError(
                    rejected_diagnostics,
                    physical_calls=result.physical_provider_calls,
                    terminal=False,
                    candidate_draft=continuation_candidate,
                ) from exc
            if (
                exc.cause is not None
                and result.exit_status != "InvalidSubmission"
            ) or result.model_calls < 1:
                from intentir.verify.scrc_runner import SCRCInfrastructureError

                raise SCRCInfrastructureError(
                    diagnostic,
                    evidence={
                        key: value
                        for key, value in evidence.items()
                        if key != "trajectory"
                    },
                ) from exc
            from intentir.verify.scrc_runner import CandidateLiftError

            rejected_diagnostics: dict[str, Any] = {
                "diagnostic": diagnostic,
                "failure_class": "lift_worker_no_candidate",
            }
            candidate_check_feedback = _last_candidate_check_feedback(evidence)
            if candidate_check_feedback is not None:
                rejected_diagnostics["candidate_check_feedback"] = (
                    candidate_check_feedback
                )
            if continuation_receipt is not None:
                rejected_diagnostics["continuation_draft"] = dict(
                    continuation_receipt
                )
            raise CandidateLiftError(
                rejected_diagnostics,
                physical_calls=result.physical_provider_calls,
                terminal=result.exit_status == "TimeExceeded",
                candidate_draft=continuation_candidate,
            ) from exc
        except Exception as exc:
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    backend_evidence={
                        "backend": MINISWE_LIFT_BACKEND,
                        "fresh_session": fresh_owner_session,
                        "persistent_owner": True,
                        "owner_session_id": self._session_id,
                    },
                )
            )
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            if isinstance(exc, SCRCInfrastructureError):
                raise
            raise SCRCInfrastructureError(
                f"MiniSWE Lift infrastructure failed: {type(exc).__name__}: {exc}",
                evidence={"backend": MINISWE_LIFT_BACKEND},
            ) from exc
        assert result is not None and result.candidate_text is not None
        safe_trajectory = _round_trajectory(
            _sanitize_trajectory(result.trajectory, provider.api_key),
            result.continued_from_messages,
        )
        usage, response_id, response_model = _usage_and_response_identity(
            safe_trajectory
        )
        raw_trajectory = _trajectory_bytes(safe_trajectory)
        worker_evidence = self._session_evidence(
            _worker_evidence(result, safe_trajectory),
            fresh_session=fresh_owner_session,
            result=result,
        )
        try:
            artifact = decode_scrc_lift_artifact(result.candidate_text, prepared)
        except Exception as exc:
            attempt = SCRCLiftAttempt(
                model=provider.model,
                diagnostic="host-private BindSource rejected the accepted Lift draft",
                response_text=result.candidate_text,
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                response_sha256=hashlib.sha256(
                    result.candidate_text.encode()
                ).hexdigest(),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                elapsed_ms=result.elapsed_ms,
                backend=MINISWE_LIFT_BACKEND,
                termination_status=result.exit_status,
                trajectory=safe_trajectory,
            )
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=attempt.diagnostic,
                    rejected_attempts=(attempt,),
                    backend_evidence=worker_evidence,
                )
            )
            from intentir.verify.scrc_runner import CandidateLiftError

            raise CandidateLiftError(
                {
                    "diagnostic": attempt.diagnostic,
                    "failure_class": "bind_source_rejected",
                    "authority": (
                        "host-private BindSource/Gate boundary; no Source "
                        "realization diagnostic is returned to the Lift owner"
                    ),
                },
                physical_calls=result.physical_provider_calls,
                terminal=True,
            ) from exc

        provenance = SCRCProvenance(
            model=provider.model,
            prompt_sha256=prepared.prompt_sha256,
            response_id=response_id,
            response_model=response_model,
            usage=usage,
            response_sha256=hashlib.sha256(
                result.candidate_text.encode()
            ).hexdigest(),
            raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
            response_text=result.candidate_text,
            raw_response=raw_trajectory,
            attempts=result.model_calls,
            elapsed_ms=result.elapsed_ms,
            backend=MINISWE_LIFT_BACKEND,
            termination_status=result.exit_status,
            trajectory=safe_trajectory,
        )
        extraction = SCRCExtraction(
            artifact,
            provenance,
            prepared.harness_binding,
            source,
        )
        self._records.append(
            SCRCReliftRecord(
                round_index,
                extraction=extraction,
                backend_evidence=worker_evidence,
            )
        )
        from intentir.verify.scrc_runner import LiftCallResult

        return LiftCallResult(
            artifact,
            physical_calls=result.physical_provider_calls,
            metadata={
                "backend": MINISWE_LIFT_BACKEND,
                "knowledge_profile": MINISWE_LIFT_KNOWLEDGE,
                "model": provider.model,
                "response_model": response_model,
                "response_id": response_id,
                "usage": usage,
                "elapsed_ms": result.elapsed_ms,
                "agent_steps": result.model_calls,
                "physical_provider_calls": result.physical_provider_calls,
                "transport_retries": result.transport_retries,
                "fresh_session": fresh_owner_session,
                "fresh_workspace": fresh_workspace,
                "persistent_workspace": True,
                "persistent_owner": True,
                "owner_session_id": self._session_id,
                "continued_from_messages": result.continued_from_messages,
                "session_active_elapsed_ms": self._session_active_elapsed_ms,
                "session_model_turns": self._session_model_turns,
                "session_physical_provider_calls": (
                    self._session_physical_provider_calls
                ),
                "workflow": self.workflow,
                "artifact_schema": artifact.wire_schema,
                "layer_c_interface": extraction.layer_c_interface_payload,
                "layer_c_interface_digest": (
                    extraction.layer_c_interface_digest
                ),
                "decode_protocol": candidate_check_context.decode_protocol,
                "typecheck_protocol": candidate_check_context.typecheck_protocol,
                "candidate_check": worker_evidence["candidate_check"],
                "agent_runtime": worker_evidence["agent_runtime"],
                "stage_metrics": worker_evidence["stage_metrics"],
            },
        )


__all__ = [
    "MINISWE_LIFT_BACKEND",
    "MINISWE_LIFT_KNOWLEDGE",
    "MiniSWELiftAdapter",
]
