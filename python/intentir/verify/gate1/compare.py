"""Pure-code Gate1 judge for normalized TTIR fingerprints.

Gate1 compares the accepted algorithm structure, not every physical compiler
artifact.  Semantic operations, semantic dataflow, interface, reductions and
memory effects remain hard evidence.  Compiler-only nodes and structural SSA
edges are retained as non-blocking diagnostics.  Numerical equivalence stays
in Gate2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any, Mapping

from .ttir import TTIRFingerprint, fingerprint_ttir

__all__ = [
    "ComponentComparison",
    "FingerprintDifference",
    "TTIR_SCORE_SEMANTICS",
    "TTIRComparator",
    "TTIRComparison",
    "compare_fingerprints",
    "compare_ttir",
]


TTIR_SCORE_SEMANTICS = "active_evidence_weighted_v2"


_CATEGORY_WEIGHTS = {
    "interface": 0.15,
    "op_multiset": 0.18,
    "dataflow": 0.22,
    "reduction": 0.18,
    "predicate_effect": 0.17,
    "output_topology": 0.10,
}

# These operations materialize a compiler representation of an already
# observed address/lane expression.  They are not allowed to introduce a new
# J operation, but their count and placement must not make an otherwise equal
# semantic graph fail Gate1.  Typed conversions remain semantic and therefore
# stay in the hard operation view.
_REALIZATION_ONLY_OPS = frozenset(
    {
        "tt.addptr",
        "tt.make_range",
        "tt.splat",
        "arith.index_cast",
        "arith.index_castui",
    }
)


def _op_name(label: str) -> str:
    return label.split("[", 1)[0]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _display(item: Any, limit: int = 280) -> str:
    rendered = _stable_json(item)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


@dataclass(frozen=True)
class FingerprintDifference:
    category: str
    kind: str
    item: Any
    expected_count: int
    actual_count: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "kind": self.kind,
            "item": self.item,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "message": self.message,
        }


@dataclass(frozen=True)
class ComponentComparison:
    category: str
    hard: bool
    passed: bool
    score: float | None
    applicable: bool
    differences: tuple[FingerprintDifference, ...]

    @property
    def name(self) -> str:
        return self.category

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "hard": self.hard,
            "passed": self.passed,
            "score": self.score,
            "applicable": self.applicable,
            "differences": [difference.to_dict() for difference in self.differences],
        }


@dataclass(frozen=True)
class TTIRComparison:
    """Itemized deterministic verdict; lexicographically larger rank is better."""

    hard_pass: bool
    score: float
    rank_tuple: tuple[int, int, float, int, int, int]
    components: tuple[ComponentComparison, ...]

    @property
    def passed(self) -> bool:
        return self.hard_pass

    @property
    def replay_closed(self) -> bool:
        return self.hard_pass

    @property
    def hard_passes(self) -> dict[str, bool]:
        return {
            component.category: component.passed
            for component in self.components
            if component.hard
        }

    @property
    def active_categories(self) -> tuple[str, ...]:
        """Hard categories that contain source or replay evidence."""

        return tuple(
            component.category
            for component in self.components
            if component.hard and component.applicable
        )

    @property
    def inactive_categories(self) -> tuple[str, ...]:
        """Hard categories whose source and replay evidence are both empty."""

        return tuple(
            component.category
            for component in self.components
            if component.hard and not component.applicable
        )

    @property
    def active_weight(self) -> float:
        return sum(_CATEGORY_WEIGHTS[category] for category in self.active_categories)

    @property
    def differences(self) -> tuple[FingerprintDifference, ...]:
        return tuple(
            difference
            for component in self.components
            for difference in component.differences
        )

    @property
    def primary_difference_category(self) -> str | None:
        failed = {
            component.category
            for component in self.components
            if not component.passed
        }
        # Dedicated semantic fields take precedence over collateral graph/op
        # differences.  This gives mutations a useful primary root cause:
        # reduce combiner -> reduction, fill/mask -> predicate/effect, and a
        # rewired graph with unchanged ops -> dataflow.
        for category in (
            "interface",
            "output_topology",
            "reduction",
            "predicate_effect",
            "op_multiset",
            "dataflow",
            "schedule-only",
        ):
            if category in failed:
                return category
        return None

    def component(self, category: str) -> ComponentComparison:
        for component in self.components:
            if component.category == category:
                return component
        raise KeyError(category)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_pass": self.hard_pass,
            "passed": self.passed,
            "score": self.score,
            "score_semantics": TTIR_SCORE_SEMANTICS,
            "active_categories": list(self.active_categories),
            "inactive_categories": list(self.inactive_categories),
            "active_weight": self.active_weight,
            "rank_tuple": list(self.rank_tuple),
            "primary_difference_category": self.primary_difference_category,
            "hard_passes": self.hard_passes,
            "components": [component.to_dict() for component in self.components],
            "differences": [difference.to_dict() for difference in self.differences],
        }


def _counter_similarity(
    expected: Counter[Any], actual: Counter[Any]
) -> float | None:
    """Return multiset Jaccard similarity, or ``None`` without evidence.

    Two empty counters are exactly equal for the hard verdict, but they do not
    provide positive similarity evidence and therefore must not receive a
    fixed category weight in the aggregate diagnostic score.
    """

    keys = expected.keys() | actual.keys()
    union = sum(max(expected[key], actual[key]) for key in keys)
    if union == 0:
        return None
    intersection = sum(min(expected[key], actual[key]) for key in keys)
    return intersection / union


def _compare_counter(
    category: str,
    expected: Counter[Any],
    actual: Counter[Any],
    *,
    hard: bool,
) -> ComponentComparison:
    differences: list[FingerprintDifference] = []
    for item in sorted(expected.keys() | actual.keys(), key=_stable_json):
        expected_count = expected[item]
        actual_count = actual[item]
        if expected_count == actual_count:
            continue
        kind = "missing" if actual_count < expected_count else "unexpected"
        direction = "缺少" if kind == "missing" else "多出"
        differences.append(
            FingerprintDifference(
                category=category,
                kind=kind,
                item=item,
                expected_count=expected_count,
                actual_count=actual_count,
                message=(
                    f"{category} {direction} {_display(item)}："
                    f"源={expected_count}，重放={actual_count}"
                ),
            )
        )
    similarity = _counter_similarity(expected, actual)
    return ComponentComparison(
        category=category,
        hard=hard,
        passed=not differences,
        score=similarity,
        applicable=similarity is not None,
        differences=tuple(differences),
    )


def _interface_counter(fingerprint: TTIRFingerprint) -> Counter[Any]:
    interface = fingerprint.interface
    items: Counter[Any] = Counter({("schema_version", fingerprint.schema_version): 1})
    for argument in interface.arguments:
        items[("argument", argument.position, _stable_json(argument.to_dict()))] += 1
    for position, result in enumerate(interface.results):
        items[("result", position, result)] += 1
    for key, value in interface.python:
        items[("python", key, value)] += 1
    return items


def _dataflow_counter(fingerprint: TTIRFingerprint) -> Counter[Any]:
    """Return semantic dataflow, excluding compiler scaffolding edges."""

    items: Counter[Any] = Counter()
    for edge in fingerprint.dataflow_edges:
        if len(edge) != 3:
            continue
        source, target, relation = edge
        # Region membership, representation anchors and effect ordering are
        # observer scaffolding.  Memory effects and semantic producer edges
        # are represented by the hard effect/reduction/output views below.
        if relation in {"region.member", "effect.order", "representation"}:
            continue
        if _op_name(source) in _REALIZATION_ONLY_OPS:
            continue
        if _op_name(target) in _REALIZATION_ONLY_OPS:
            continue
        items[("edge", *edge)] += 1
    for dot in fingerprint.dots:
        items[("dot", _stable_json(dot.to_dict()))] += 1
    for loop in fingerprint.loops:
        items[("loop", _stable_json(loop.to_dict()))] += 1
    for structured_if in fingerprint.structured_ifs:
        items[("structured_if", _stable_json(structured_if.to_dict()))] += 1
    for terminal_fork in fingerprint.terminal_forks:
        items[("terminal_fork", _stable_json(terminal_fork.to_dict()))] += 1
    return items


def _semantic_op_counter(fingerprint: TTIRFingerprint) -> Counter[Any]:
    """Keep logical operations hard while ignoring representation-only nodes."""

    return Counter(
        {
            name: count
            for name, count in fingerprint.op_multiset
            if name not in _REALIZATION_ONLY_OPS
        }
    )


def _reduction_counter(fingerprint: TTIRFingerprint) -> Counter[Any]:
    return Counter(
        ("reduction", _stable_json(reduction.to_dict()))
        for reduction in fingerprint.reductions
    )


def _predicate_effect_counter(fingerprint: TTIRFingerprint) -> Counter[Any]:
    items: Counter[Any] = Counter()
    for effect in fingerprint.memory_effects:
        items[("memory", _stable_json(effect.to_dict()))] += 1
    for predicate in fingerprint.mask_predicates:
        items[("mask", predicate)] += 1
    for effect, count in fingerprint.effect_multiset:
        items[("effect", effect)] += count
    return items


def _output_topology_counter(fingerprint: TTIRFingerprint) -> Counter[Any]:
    """Compare observed output ABI/store topology without partition proof."""

    items: Counter[Any] = Counter()
    for argument in fingerprint.interface.arguments:
        if argument.access in {"output", "inout"}:
            items[("output-argument", argument.position, _stable_json(argument.to_dict()))] += 1
    for effect in fingerprint.memory_effects:
        if effect.kind == "store":
            items[("store", _stable_json(effect.to_dict()))] += 1
    return items


def _schedule_counter(fingerprint: TTIRFingerprint) -> Counter[Any]:
    return Counter(("schedule", key, value) for key, value in fingerprint.schedule_view)


def compare_fingerprints(
    source: TTIRFingerprint, replay: TTIRFingerprint
) -> TTIRComparison:
    """Compare two fingerprints exactly on hard fields and report all deltas."""

    if not isinstance(source, TTIRFingerprint) or not isinstance(replay, TTIRFingerprint):
        raise TypeError("compare_fingerprints expects two TTIRFingerprint values")

    components = (
        _compare_counter(
            "interface", _interface_counter(source), _interface_counter(replay), hard=True
        ),
        _compare_counter(
            "op_multiset",
            _semantic_op_counter(source),
            _semantic_op_counter(replay),
            hard=True,
        ),
        _compare_counter(
            "dataflow", _dataflow_counter(source), _dataflow_counter(replay), hard=True
        ),
        _compare_counter(
            "reduction",
            _reduction_counter(source),
            _reduction_counter(replay),
            hard=True,
        ),
        _compare_counter(
            "predicate_effect",
            _predicate_effect_counter(source),
            _predicate_effect_counter(replay),
            hard=True,
        ),
        _compare_counter(
            "output_topology",
            _output_topology_counter(source),
            _output_topology_counter(replay),
            hard=True,
        ),
        _compare_counter(
            "op_multiset-realization",
            Counter(dict(source.op_multiset)),
            Counter(dict(replay.op_multiset)),
            hard=False,
        ),
        _compare_counter(
            "dataflow-realization",
            Counter(("edge", *edge) for edge in source.dataflow_edges),
            Counter(("edge", *edge) for edge in replay.dataflow_edges),
            hard=False,
        ),
        _compare_counter(
            "schedule-only",
            _schedule_counter(source),
            _schedule_counter(replay),
            hard=False,
        ),
    )
    hard_components = [component for component in components if component.hard]
    hard_pass = all(component.passed for component in hard_components)
    active_components = [
        component for component in hard_components if component.applicable
    ]
    active_weight = sum(
        _CATEGORY_WEIGHTS[component.category]
        for component in active_components
    )
    score = (
        sum(
            component.score * _CATEGORY_WEIGHTS[component.category]
            for component in active_components
            if component.score is not None
        )
        / active_weight
        if active_weight
        else 0.0
    )
    score = min(1.0, max(0.0, score))
    hard_difference_count = sum(
        len(component.differences) for component in hard_components
    )
    schedule_component = components[-1]
    rank_tuple = (
        int(hard_pass),
        sum(component.passed for component in hard_components),
        round(score, 12),
        -hard_difference_count,
        int(schedule_component.passed),
        -len(schedule_component.differences),
    )
    return TTIRComparison(hard_pass, score, rank_tuple, components)


def compare_ttir(
    source_ttir: str,
    replay_ttir: str,
    *,
    source_python_interface: Mapping[str, Any] | None = None,
    replay_python_interface: Mapping[str, Any] | None = None,
    source_schedule: Mapping[str, Any] | None = None,
    replay_schedule: Mapping[str, Any] | None = None,
    source_external_math: Mapping[str, Any] | None = None,
    replay_external_math: Mapping[str, Any] | None = None,
) -> TTIRComparison:
    """Parse and compare two Triton TTIR texts in one call."""

    source = fingerprint_ttir(
        source_ttir,
        python_interface=source_python_interface,
        schedule=source_schedule,
        external_math=source_external_math,
    )
    replay = fingerprint_ttir(
        replay_ttir,
        python_interface=replay_python_interface,
        schedule=replay_schedule,
        external_math=replay_external_math,
    )
    return compare_fingerprints(source, replay)


class TTIRComparator:
    """Small state-free OO facade for orchestrator dependency injection."""

    def compare(
        self, source: TTIRFingerprint, replay: TTIRFingerprint
    ) -> TTIRComparison:
        return compare_fingerprints(source, replay)

    def compare_text(self, source_ttir: str, replay_ttir: str, **kwargs: Any) -> TTIRComparison:
        return compare_ttir(source_ttir, replay_ttir, **kwargs)
