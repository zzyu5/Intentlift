"""One active SCRC Gate runner joining Lift, TypeCheck, Gate1 and Gate2.

The runner is deliberately frontend-neutral.  Frontend compilers and source
aware providers are injected as small adapters; the lifecycle and terminal
status rules live here exactly once.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.layer_c import (
    LayerCInterface,
    binding_interface_digest,
    source_layer_c_interface,
)
from .gate1.model import Gate1Artifact, Gate1Status, JNecessityStatus
from .gate1.orchestrate import (
    Gate1ExecutionResult,
    SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX,
)
from .gate2.model import Gate2SuiteResult
from .typecheck import TypeCheckReport, TypeCheckStatus, type_check


MAX_GATE1_ROUNDS = 3
# One logical Lift invocation produces one complete candidate artifact (or one
# rejected candidate transaction).  This is deliberately separate from the
# formal Replay/Gate1-round budget and from an adapter's physical provider
# retry count.  It is the shared temporary cap for every branch lifecycle.
MAX_LIFT_CALLS = 3
# Zero means that pre-Replay construction is governed by the Lift owner's
# terminal resource signal rather than a semantic-attempt count.  A positive
# value remains available as an explicit legacy/diagnostic override.
MAX_PRE_REPLAY_REJECTIONS = 0
UNRESOLVED_GATE1_DIAGNOSTIC = "unresolved_gate1"
UNRESOLVED_GATE2_DIAGNOSTIC = "unresolved_gate2"
MAX_GATE1_FEEDBACK_EXAMPLES_PER_CATEGORY = 8
MAX_GATE1_FEEDBACK_PATTERNS_PER_CATEGORY = 16
MAX_GATE1_FEEDBACK_NODE_WITNESSES = 10
MAX_GATE1_FEEDBACK_EDGES_PER_NODE = 6
MAX_GATE1_FEEDBACK_LOCAL_SIGNATURE_CHARS = 512
MAX_GATE1_FEEDBACK_LOCAL_FRONTIER_EXAMPLES = 10
MAX_GATE1_REPAIR_CHANGED_PATHS = 48
GATE1_REPAIR_PROGRESS_PROTOCOL = "intentir.gate1-relift-progress.v1"
GATE1_LOCAL_FRONTIER_PROTOCOL = "intentir.gate1-local-frontier.v1"
_OBSERVER_NODE_FINGERPRINT = re.compile(
    r"(?<=[\[:])[0-9a-f]{12,64}(?=\])"
)

SCRCProgressCallback = Callable[
    [str, int | None, int | None, str],
    None,
]


def _notify_progress(
    callback: SCRCProgressCallback | None,
    stage: str,
    attempt: int | None,
    formal_round: int | None,
    status: str,
) -> None:
    """Best-effort runtime observation that can never alter a gate verdict."""

    if callback is None:
        return
    try:
        callback(stage, attempt, formal_round, status)
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
    envelope_keys = {UNRESOLVED_GATE1_DIAGNOSTIC, UNRESOLVED_GATE2_DIAGNOSTIC}

    def visit(item: Any, depth: int) -> Any:
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


def _changed_json_paths(
    before: Any,
    after: Any,
) -> tuple[list[str], int]:
    """Return a bounded mechanical diff between two LLM-authored artifacts."""

    paths: list[str] = []
    total = 0

    def record(path: str) -> None:
        nonlocal total
        total += 1
        if len(paths) < MAX_GATE1_REPAIR_CHANGED_PATHS:
            paths.append(path or "<root>")

    def visit(left: Any, right: Any, path: str) -> None:
        if type(left) is not type(right):
            record(path)
            return
        if isinstance(left, Mapping):
            left_keys = {str(key) for key in left}
            right_keys = {str(key) for key in right}
            for key in sorted(left_keys | right_keys):
                child = f"{path}.{key}" if path else key
                if key not in left_keys or key not in right_keys:
                    record(child)
                else:
                    visit(left[key], right[key], child)
            return
        if isinstance(left, list):
            common = min(len(left), len(right))
            for index in range(common):
                visit(left[index], right[index], f"{path}.{index}")
            for index in range(common, max(len(left), len(right))):
                record(f"{path}.{index}")
            return
        if left != right:
            record(path)

    visit(before, after, "")
    return paths, total


def _observer_category_counts(
    diagnostics: Mapping[str, Any],
) -> dict[str, int]:
    observer = diagnostics.get("observer_diff")
    categories = observer.get("categories") if isinstance(observer, Mapping) else None
    if not isinstance(categories, list):
        return {}
    result: dict[str, int] = {}
    for category in categories:
        if not isinstance(category, Mapping):
            continue
        name = category.get("category")
        count = category.get("difference_count")
        if isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool):
            result[name] = count
    return dict(sorted(result.items()))


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


def _bounded_observer_examples(
    differences: list[Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Choose bounded examples by causal-role coverage, not input accident."""

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
    return [
        {
            key: _pseudonymize_observer_nodes(difference[key], aliases)
            for key in ("kind", "item", "expected_count", "actual_count")
            if key in difference
        }
        for difference in selected
    ]


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

    The bounded examples remain the cheap first read.  The exact differences are
    retained in their already-labelled category so a persistent Lift owner can
    inspect the same counterexample evidence as a human/Codex diagnosis without
    injecting the bulk object into its prompt.  This function still performs no
    attribution and proposes no artifact repair.
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
        group["differences"] = _pseudonymize_observer_nodes(
            category_differences,
            node_aliases,
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
        categories.append(group)
    return {
        "hard_pass": bool(raw.get("hard_pass", False)),
        "hard_passes": _feedback_item(raw.get("hard_passes", {})),
        "primary_difference_category": raw.get("primary_difference_category"),
        "categories": categories,
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
    GATE2 = "gate2"


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
        candidate_draft: Mapping[str, Any] | None = None,
    ) -> None:
        self.diagnostics = dict(diagnostics)
        if not isinstance(physical_calls, int) or physical_calls < 0:
            raise ValueError("Lift physical_calls must be non-negative")
        if not isinstance(terminal, bool):
            raise TypeError("CandidateLiftError.terminal must be boolean")
        self.physical_calls = physical_calls
        self.terminal = terminal
        if candidate_draft is not None:
            if not isinstance(candidate_draft, Mapping):
                raise TypeError("CandidateLiftError.candidate_draft must be a mapping")
            draft = json.loads(
                json.dumps(dict(candidate_draft), sort_keys=True, allow_nan=False)
            )
            if not isinstance(draft, dict):
                raise TypeError("CandidateLiftError.candidate_draft must be an object")
            c = draft.get("c")
            if isinstance(c, Mapping) and "values" in c and c.get("values") != {}:
                raise ValueError(
                    "CandidateLiftError.candidate_draft must be value-free"
                )
            self.candidate_draft = draft
        else:
            self.candidate_draft = None
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
            ordered = tuple(c_values.items())
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

    ``text`` is the selected source projection used by the Lift owner.  Native
    frontends may need a language-specific projection (for example one CUDA
    device entry) while ``gate_source`` retains the real module/file consumed
    by the compiler and observer.  The optional fields keep that distinction
    explicit instead of sending C++ through the Python AST anonymizer.
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
                    "CUDA SourceProgram Lift projection requires a source symbol"
                )
            if not parameter_order:
                raise ValueError(
                    "CUDA SourceProgram Lift projection requires parameter order"
                )
            mapped_sources = {source for source, _opaque in mapping}
            if not set(parameter_order) <= mapped_sources:
                raise ValueError(
                    "CUDA SourceProgram Lift mapping must cover every source parameter"
                )
        object.__setattr__(self, "lift_parameter_order", parameter_order)
        object.__setattr__(self, "lift_constexpr_parameters", constexpr_parameters)
        object.__setattr__(self, "lift_source_to_opaque", mapping)


@dataclass(frozen=True)
class LiftFeedback:
    kind: FeedbackKind
    diagnostics: Mapping[str, Any]
    formal_round: int | None = None
    previous_artifact: Gate1Artifact | None = None
    selection_rank: tuple[int | float, ...] | None = None
    previous_candidate: Mapping[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.previous_candidate is not None:
            if self.previous_artifact is not None:
                raise ValueError(
                    "LiftFeedback cannot carry both a previous artifact and a "
                    "rejected candidate"
                )
            if not isinstance(self.previous_candidate, Mapping):
                raise TypeError("previous_candidate must be a mapping")
            candidate = json.loads(
                json.dumps(
                    dict(self.previous_candidate),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            if not isinstance(candidate, dict):
                raise TypeError("previous_candidate must be an object")
            c = candidate.get("c")
            if isinstance(c, Mapping) and "values" in c and c.get("values") != {}:
                raise ValueError("previous_candidate must be value-free")
            object.__setattr__(self, "previous_candidate", candidate)
        if self.selection_rank is None:
            return
        if self.kind is not FeedbackKind.GATE1:
            raise ValueError("selection_rank is private Gate1 search state")
        if (
            len(self.selection_rank) != 6
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in self.selection_rank
            )
        ):
            raise ValueError("selection_rank must be one Gate1 rank tuple")

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "formal_round": self.formal_round,
            "diagnostics": dict(self.diagnostics),
            "previous_artifact": (
                self.previous_artifact.to_json()
                if self.previous_artifact is not None else None
            ),
            "selection_rank": (
                list(self.selection_rank)
                if self.selection_rank is not None else None
            ),
            "previous_candidate": (
                dict(self.previous_candidate)
                if self.previous_candidate is not None else None
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
        payload = self.metadata.get("layer_c_interface")
        digest = self.metadata.get("layer_c_interface_digest")
        if not isinstance(payload, Mapping):
            raise TypeError(
                "LiftCallResult requires a standalone Layer-C interface sidecar"
            )
        try:
            interface = LayerCInterface.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "LiftCallResult Layer-C interface sidecar is malformed"
            ) from exc
        expected = source_layer_c_interface(self.artifact.j, self.artifact.c)
        if interface != expected:
            raise ValueError(
                "LiftCallResult Layer-C interface differs from Source-C declarations"
            )
        expected_digest = binding_interface_digest(interface)
        if digest != expected_digest:
            raise ValueError(
                "LiftCallResult Layer-C interface digest is missing or stale"
            )


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
                "diagnostic": f"{type(exc).__name__}: {exc}",
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
        artifact: Gate1Artifact,
        prior_feedback: LiftFeedback | None = None,
    ) -> LiftFeedback:
        normal = result.normal
        diagnostics: dict[str, Any] = {
            "status": result.status.value,
            "diagnostic": normal.diagnostic,
        }
        if normal.comparison is not None:
            diagnostics["observer_diff"] = _compact_gate1_observer_diff(
                normal.comparison,
                source_fingerprint=result.source_fingerprint,
                replay_fingerprint=normal.fingerprint,
            )
        progress = cls._gate1_repair_progress(
            artifact,
            diagnostics,
            cls._failed_gate1_rank(result),
            prior_feedback,
        )
        if progress is not None:
            diagnostics["repair_progress"] = progress
        unresolved_gate2 = cls._unresolved_gate2(prior_feedback)
        if unresolved_gate2 is not None:
            diagnostics[UNRESOLVED_GATE2_DIAGNOSTIC] = unresolved_gate2
        diagnostics = _prune_relift_diagnostics(diagnostics)
        return LiftFeedback(
            FeedbackKind.GATE1,
            diagnostics,
            formal_round,
            artifact,
            cls._failed_gate1_rank(result),
        )

    @classmethod
    def _gate1_repair_progress(
        cls,
        artifact: Gate1Artifact,
        diagnostics: Mapping[str, Any],
        current_rank: tuple[int | float, ...] | None,
        prior_feedback: LiftFeedback | None,
    ) -> dict[str, Any] | None:
        """Describe repair progress without constructing or editing semantics."""

        if prior_feedback is None or prior_feedback.previous_artifact is None:
            return None
        parent = prior_feedback.previous_artifact
        parent_payload = parent.to_json()
        current_payload = artifact.to_json()
        changed_paths, changed_path_count = _changed_json_paths(
            parent_payload,
            current_payload,
        )
        component_changed = {
            component: parent_payload[component] != current_payload[component]
            for component in ("j", "c", "pi")
        }
        parent_counts = _observer_category_counts(prior_feedback.diagnostics)
        current_counts = _observer_category_counts(diagnostics)
        category_delta = {
            category: current_counts.get(category, 0) - parent_counts.get(category, 0)
            for category in sorted(set(parent_counts) | set(current_counts))
        }
        parent_rank = prior_feedback.selection_rank
        rank_relation = "not_comparable"
        if current_rank is not None and parent_rank is not None:
            rank_relation = (
                "improved"
                if current_rank > parent_rank
                else "unchanged"
                if current_rank == parent_rank
                else "regressed"
            )
        if not any(component_changed.values()):
            diagnostic = (
                "The canonical J/C/pi artifact is unchanged, so the prior "
                "Gate1 counterexample remains unresolved. Re-read the bounded "
                "observer evidence and author a Source-justified semantic or "
                "realization change before resubmitting."
            )
        elif rank_relation in {"regressed", "unchanged"}:
            diagnostic = (
                "The authored artifact changed, but the Gate1 diagnostic rank "
                "did not improve. Use the component/path and category deltas to "
                "choose a different Source-justified repair hypothesis."
            )
        else:
            diagnostic = (
                "The authored artifact changed. Continue from the selected "
                "parent and use the remaining Gate1 categories as open "
                "obligations; fixed code has not prescribed a repair."
            )
        return {
            "protocol": GATE1_REPAIR_PROGRESS_PROTOCOL,
            "parent_formal_round": prior_feedback.formal_round,
            "component_changed": component_changed,
            "artifact_changed": bool(changed_path_count),
            "changed_path_count": changed_path_count,
            "changed_paths": changed_paths,
            "changed_paths_omitted": max(0, changed_path_count - len(changed_paths)),
            "parent_category_counts": parent_counts,
            "current_category_counts": current_counts,
            "category_count_delta": category_delta,
            "rank_relation": rank_relation,
            "diagnostic": diagnostic,
            "authority": (
                "diagnostic-only comparison of LLM-authored artifacts and Gate1 "
                "evidence; it never edits J/C/pi or certifies a candidate"
            ),
        }

    @staticmethod
    def _failed_gate1_rank(
        result: Gate1ExecutionResult,
    ) -> tuple[int | float, ...] | None:
        """Return Gate1's diagnostic-only failed-candidate ordering.

        The rank never accepts a candidate.  It only chooses which complete,
        already TypeChecked LLM artifact and its exact counterexample become
        the parent of the next Lift call.  A result without a comparable
        observer verdict keeps the ordinary latest-candidate repair path.
        """

        comparison = result.normal.comparison
        if comparison is None:
            return None
        rank = getattr(comparison, "rank_tuple", None)
        if rank is None:
            raw = comparison.to_dict()
            rank = raw.get("rank_tuple") if isinstance(raw, Mapping) else None
        if (
            not isinstance(rank, (list, tuple))
            or len(rank) != 6
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in rank
            )
        ):
            return None
        return tuple(rank)

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

    @staticmethod
    def _unresolved_gate2(
        feedback: LiftFeedback | None,
    ) -> dict[str, Any] | None:
        if feedback is None:
            return None
        if feedback.kind is FeedbackKind.GATE2:
            return {"diagnostics": dict(feedback.diagnostics)}
        retained = feedback.diagnostics.get(UNRESOLVED_GATE2_DIAGNOSTIC)
        if not isinstance(retained, Mapping):
            return None
        diagnostics = retained.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            return None
        return {"diagnostics": _prune_relift_diagnostics(dict(diagnostics))}

    @classmethod
    def _retain_gate1_diagnostic(
        cls,
        diagnostics: Mapping[str, Any],
        feedback: LiftFeedback | None,
    ) -> dict[str, Any]:
        combined = dict(diagnostics)
        unresolved = cls._unresolved_gate1(feedback)
        if unresolved is not None:
            combined[UNRESOLVED_GATE1_DIAGNOSTIC] = unresolved
        unresolved_gate2 = cls._unresolved_gate2(feedback)
        if unresolved_gate2 is not None:
            combined[UNRESOLVED_GATE2_DIAGNOSTIC] = unresolved_gate2
        return _prune_relift_diagnostics(combined)

    def search_gate1(
        self,
        request: SCRCRunRequest,
        *,
        initial_feedback: LiftFeedback | None = None,
        retry_initial_artifact: bool = False,
        stop_requested: Callable[[], int | None] | None = None,
        progress_callback: SCRCProgressCallback | None = None,
    ) -> SCRCGate1SearchReport:
        """Search one isolated branch through TypeCheck and Gate1 only.

        ``initial_feedback`` resumes from an externally preserved TypeCheck,
        Gate1, or J-side Gate2 failure.  By default it is context for the first
        new Lift call only.  ``retry_initial_artifact`` may first run current
        TypeCheck and one fresh formal Gate1 round over an exact Gate1-failed
        parent; it never edits the artifact, and all new Replay/Gate work is
        counted in this search.
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

        if initial_feedback is not None:
            if not isinstance(initial_feedback, LiftFeedback):
                raise TypeError("initial_feedback must be LiftFeedback")
            if initial_feedback.kind not in {
                FeedbackKind.TYPECHECK,
                FeedbackKind.GATE1,
                FeedbackKind.GATE2,
            }:
                raise ValueError(
                    "initial_feedback must describe a TypeCheck, Gate1, or "
                    "J-side Gate2 failure"
                )
            if initial_feedback.kind is FeedbackKind.GATE1:
                if (
                    not isinstance(initial_feedback.formal_round, int)
                    or initial_feedback.formal_round < 1
                ):
                    raise ValueError(
                        "initial Gate1 feedback requires a positive formal_round"
                    )
            elif initial_feedback.formal_round is not None:
                raise ValueError(
                    "initial TypeCheck/Gate2 feedback has no formal Gate1 round"
                )
            if initial_feedback.previous_artifact is None:
                raise ValueError(
                    "initial Gate1 feedback requires the exact previous artifact"
                )
            request.binding.validate_artifact(
                initial_feedback.previous_artifact
            )
        if retry_initial_artifact and (
            initial_feedback is None
            or initial_feedback.kind is not FeedbackKind.GATE1
        ):
            raise ValueError(
                "retry_initial_artifact requires exact Gate1 initial feedback"
            )

        attempts: list[CandidateAttemptReport] = []
        feedback = initial_feedback
        last_artifact = (
            initial_feedback.previous_artifact
            if initial_feedback is not None else None
        )
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
        pending_initial_artifact = (
            initial_feedback.previous_artifact
            if retry_initial_artifact and initial_feedback is not None
            else None
        )

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
                pending_initial_artifact is None
                and lift_calls >= request.max_lift_calls
            ):
                terminal_stop_reason = "lift_call_cap"
                terminal_stop_observed_at = "before_lift"
                break
            lift_started = time.monotonic()
            if pending_initial_artifact is not None:
                artifact = canonicalize_gate1_artifact(
                    pending_initial_artifact
                )
                pending_initial_artifact = None
                attempt_lift_call = 0
                lift_metadata: Mapping[str, Any] = {
                    "initial_artifact_gate1_retry": True,
                    "model_calls": 0,
                }
                initial_interface = source_layer_c_interface(
                    artifact.j, artifact.c
                )
                lift_metadata = {
                    **lift_metadata,
                    "layer_c_interface": initial_interface.to_dict(),
                    "layer_c_interface_digest": binding_interface_digest(
                        initial_interface
                    ),
                }
                last_artifact = artifact
                record_stage(
                    "lift",
                    lift_started,
                    "initial_artifact_retry",
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
                    )
                    previous_candidate = exc.candidate_draft
                    if previous_candidate is None and (
                        feedback is not None
                        and feedback.kind is FeedbackKind.LIFT
                    ):
                        previous_candidate = feedback.previous_candidate
                    feedback = LiftFeedback(
                        FeedbackKind.LIFT,
                        self._retain_gate1_diagnostic(exc.diagnostics, feedback),
                        None,
                        None if previous_candidate is not None else last_artifact,
                        previous_candidate=previous_candidate,
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
                if not isinstance(lift_result, LiftCallResult):
                    raise SCRCInfrastructureError(
                        "Lift adapter must return LiftCallResult"
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
                last_artifact = artifact
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
                feedback = LiftFeedback(
                    FeedbackKind.TYPECHECK,
                    self._retain_gate1_diagnostic(
                        {"typecheck": report.to_json()}, feedback
                    ),
                    None,
                    artifact,
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
                if probe_result.get("status") == "diverged":
                    pre_replay_rejections += 1
                    feedback = LiftFeedback(
                        FeedbackKind.LIFT,
                        self._retain_gate1_diagnostic(
                            {
                                "diagnostic": (
                                    "public Source-vs-J semantic probe found a "
                                    "divergence"
                                ),
                                "failure_class": "lift_semantic_probe_diverged",
                                "semantic_probe": probe_result,
                                "authority": (
                                    "diagnostic-only public Source-vs-J comparison; "
                                    "it never edits or certifies the artifact"
                                ),
                            },
                            feedback,
                        ),
                        None,
                        artifact,
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
                    last_artifact = artifact
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
            gate1_adapter = self.dependencies.gate1_factory(next_formal_round)
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
                arm.generation is not None for arm in normal_arms_for_progress
            )
            replay_generation_failed = (
                not replay_was_generated
                and any(
                    (arm.diagnostic or "").startswith("generation failed:")
                    for arm in normal_arms_for_progress
                )
            )
            # If every authored Replay failed before a TTIR comparison, Gate1 has
            # learned nothing about J/C/pi.  A fresh source-blind Replay should retry
            # the same TypeCheck-valid artifact; sending this compiler-only failure
            # through Lift would waste a semantic rewrite and can corrupt a correct J.
            replay_compile_failed_before_comparison = (
                replay_was_generated
                and not any(
                    arm.comparison is not None for arm in normal_arms_for_progress
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
            if replay_generation_failed:
                _notify_progress(
                    progress_callback,
                    "replay",
                    None,
                    next_formal_round,
                    "not_run",
                )
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
                        "artifact": artifact.to_json(),
                        "typecheck": report.to_json(),
                        "gate1": gate1.to_json(),
                    },
                )
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
            replay_normal_calls += gate1.normal_calls
            replay_masked_calls += gate1.masked_j_calls
            normal_arms = normal_arms_for_progress
            for arm in (*normal_arms, gate1.masked_j):
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
                    artifact,
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
                # The same branch-local Lift owner continues from the exact
                # candidate it just authored. Comparator rank remains diagnostic
                # evidence only; fixed code never substitutes an older "best"
                # parent behind the owner's conversation.
                feedback = candidate_feedback
                last_artifact = artifact
                if replay_compile_failed_before_comparison:
                    pending_initial_artifact = artifact
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
