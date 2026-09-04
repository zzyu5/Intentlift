"""Trusted external comparison policy Q for Gate2.

J declares only a closed comparison class.  Q supplies numeric thresholds and
NaN handling, so a Lift candidate cannot choose its own passing criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping

from intentir.ir.canonical_j import j_elaboration
from intentir.verify.gate1.model import ResultRelationKind, SemanticArchive


GATE2_POLICY_PROTOCOL = "intentir.scrc.gate2-q.v2"

_DTYPE_ALIASES = {
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
    "bool": "i1",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "index": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "ui8": "u8",
    "ui16": "u16",
    "ui32": "u32",
}


def canonical_dtype(value: str) -> str:
    normalized = value.removeprefix("torch.").lower()
    return _DTYPE_ALIASES.get(normalized, normalized)


class ComparisonPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class NumericTolerance:
    rtol: float
    atol: float
    equal_nan: bool = False

    def __post_init__(self) -> None:
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"Gate2 Q {name} must be finite and non-negative")
        if not isinstance(self.equal_nan, bool):
            raise TypeError("Gate2 Q equal_nan must be boolean")

    def to_json(self) -> dict[str, float | bool]:
        return {
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "equal_nan": self.equal_nan,
        }


@dataclass(frozen=True)
class CloseRule:
    comparison_class: ResultRelationKind
    dtype: str
    tolerance: NumericTolerance | None

    @property
    def exact(self) -> bool:
        return self.comparison_class is ResultRelationKind.EXACT


@dataclass(frozen=True)
class ComparisonPolicy:
    policy_id: str
    tolerances: tuple[tuple[str, NumericTolerance], ...]
    mixed_precision_tolerances: tuple[tuple[str, NumericTolerance], ...] = ()
    protocol: str = GATE2_POLICY_PROTOCOL

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_.-]*", self.policy_id) is None:
            raise ValueError("Gate2 Q policy_id must be a stable lowercase id")
        normalized = tuple(
            (canonical_dtype(dtype), tolerance)
            for dtype, tolerance in self.tolerances
        )
        keys = [dtype for dtype, _ in normalized]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("Gate2 Q dtype rules must be unique and sorted")
        if any(
            not isinstance(tolerance, NumericTolerance)
            for _, tolerance in normalized
        ):
            raise TypeError("Gate2 Q entries must be NumericTolerance values")
        mixed_precision = tuple(
            (canonical_dtype(dtype), tolerance)
            for dtype, tolerance in self.mixed_precision_tolerances
        )
        mixed_keys = [dtype for dtype, _ in mixed_precision]
        if mixed_keys != sorted(mixed_keys) or len(mixed_keys) != len(set(mixed_keys)):
            raise ValueError(
                "Gate2 Q mixed-precision rules must be unique and sorted"
            )
        if any(
            not isinstance(tolerance, NumericTolerance)
            for _, tolerance in mixed_precision
        ):
            raise TypeError(
                "Gate2 Q mixed-precision entries must be NumericTolerance values"
            )
        object.__setattr__(self, "tolerances", normalized)
        object.__setattr__(self, "mixed_precision_tolerances", mixed_precision)
        if self.protocol != GATE2_POLICY_PROTOCOL:
            raise ValueError("unsupported Gate2 Q protocol")

    @classmethod
    def create(
        cls,
        policy_id: str,
        tolerances: Iterable[tuple[str, NumericTolerance]],
        *,
        mixed_precision_tolerances: Iterable[
            tuple[str, NumericTolerance]
        ] = (),
    ) -> "ComparisonPolicy":
        return cls(
            policy_id,
            tuple(sorted(tolerances, key=lambda item: item[0])),
            tuple(
                sorted(
                    mixed_precision_tolerances,
                    key=lambda item: item[0],
                )
            ),
        )

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ComparisonPolicy":
        if not isinstance(value, Mapping):
            raise ComparisonPolicyError("Gate2 Q JSON must be an object")
        required = {"protocol", "policy_id", "tolerances"}
        optional = {
            "mixed_precision_compute_tolerances",
            "mixed_precision_selection",
        }
        if not required.issubset(value) or set(value) - required - optional:
            raise ComparisonPolicyError(
                "Gate2 Q JSON has missing or unknown policy fields"
            )
        if value["protocol"] != GATE2_POLICY_PROTOCOL:
            raise ComparisonPolicyError("unsupported Gate2 Q protocol")

        def parse_tolerances(
            raw: object, *, context: str
        ) -> list[tuple[str, NumericTolerance]]:
            if not isinstance(raw, Mapping):
                raise ComparisonPolicyError(f"Gate2 Q {context} must be an object")
            parsed: list[tuple[str, NumericTolerance]] = []
            for dtype, record in raw.items():
                if not isinstance(dtype, str) or not isinstance(record, Mapping):
                    raise ComparisonPolicyError(
                        f"Gate2 Q {context} entry is invalid"
                    )
                if set(record) != {"rtol", "atol", "equal_nan"}:
                    raise ComparisonPolicyError(
                        f"Gate2 Q {context} requires rtol, atol, and equal_nan"
                    )
                parsed.append(
                    (
                        dtype,
                        NumericTolerance(
                            record["rtol"],
                            record["atol"],
                            record["equal_nan"],
                        ),
                    )
                )
            return parsed

        mixed = value.get("mixed_precision_compute_tolerances", {})
        return cls.create(
            str(value["policy_id"]),
            parse_tolerances(value["tolerances"], context="tolerances"),
            mixed_precision_tolerances=parse_tolerances(
                mixed,
                context="mixed-precision tolerances",
            ),
        )

    def rule_for(
        self,
        comparison_class: ResultRelationKind | str,
        dtype: str,
    ) -> CloseRule:
        kind = ResultRelationKind(comparison_class)
        dtype = canonical_dtype(dtype)
        if kind is ResultRelationKind.EXACT:
            return CloseRule(kind, dtype, None)
        if kind is ResultRelationKind.ONE_OF_UPDATES:
            raise ComparisonPolicyError(
                "set-valued result relations are outside the active Gate2 main path"
            )
        if not dtype.startswith(("f", "bf")):
            raise ComparisonPolicyError(
                "numeric tolerance is allowed only for floating outputs"
            )
        tolerance = dict(self.tolerances).get(dtype)
        if tolerance is None:
            raise ComparisonPolicyError(
                f"Gate2 Q has no tolerance rule for dtype {dtype!r}"
            )
        return CloseRule(kind, dtype, tolerance)

    def validate_j(self, j: SemanticArchive) -> None:
        explicit = {item.output: item for item in j.boundary.result_relations}
        for output in j.intent.outputs:
            relation = explicit.get(output.name)
            kind = (
                relation.kind
                if relation is not None
                else ResultRelationKind.EXACT
            )
            self.rule_for_output(j, output.name, kind, output.dtype)

    def rule_for_output(
        self,
        j: SemanticArchive,
        output: str,
        comparison_class: ResultRelationKind | str,
        dtype: str,
    ) -> CloseRule:
        """Select Q from one output's closed J-only numeric dependency path.

        J continues to declare only the result-relation class.  This method does
        not inspect Source, Replay, C, pi, backend identity, or observed errors.
        It merely selects one externally frozen Q profile when the accepted J
        explicitly routes a lower-precision contraction/reduction into this
        output.  Exact relations remain exact.
        """

        base = self.rule_for(comparison_class, dtype)
        if base.exact:
            return base
        compute_dtype = _output_low_precision_compute_dtype(j, output, dtype)
        if compute_dtype is None:
            return base
        tolerance = dict(self.mixed_precision_tolerances).get(compute_dtype)
        if tolerance is None:
            raise ComparisonPolicyError(
                "Gate2 Q has no mixed-precision tolerance rule for compute dtype "
                f"{compute_dtype!r}"
            )
        return CloseRule(base.comparison_class, base.dtype, tolerance)

    def to_json(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "policy_id": self.policy_id,
            "tolerances": {
                dtype: tolerance.to_json()
                for dtype, tolerance in self.tolerances
            },
            "mixed_precision_compute_tolerances": {
                dtype: tolerance.to_json()
                for dtype, tolerance in self.mixed_precision_tolerances
            },
            "mixed_precision_selection": (
                "per-output closed J dependency; lower-precision operand of "
                "contraction/reduce/scan only"
            ),
        }


def _operation_results(operation: Mapping[str, Any]) -> tuple[str, ...]:
    result = operation.get("result")
    if isinstance(result, str):
        return (result,)
    if isinstance(result, list):
        return tuple(item for item in result if isinstance(item, str))
    return ()


def _operation_dependencies(operation: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value not in result:
            result.append(value)

    for value in operation.get("operands", ()):
        add(value)
    if operation.get("op") in {"effect_if", "value_if"}:
        add(operation.get("predicate"))
    if operation.get("op") == "value_if":
        for polarity in ("positive", "negative"):
            arm = operation.get(polarity)
            if isinstance(arm, Mapping):
                add(arm.get("yield_value"))
    if operation.get("op") == "loop":
        trip = operation.get("trip")
        if isinstance(trip, Mapping) and "upper_bound" in trip:
            add(trip.get("upper_bound"))
        for item in operation.get("scan", ()):
            if isinstance(item, Mapping):
                add(item.get("from"))
        for item in operation.get("carry", ()):
            if isinstance(item, Mapping):
                add(item.get("init_value"))
                add(item.get("next_value"))
        for value in operation.get("yield", ()):
            add(value)
    return tuple(result)


def _walk_operations(
    body: Any,
    *,
    prefix: str = "body",
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if not isinstance(body, list):
        return
    for index, operation in enumerate(body):
        if not isinstance(operation, Mapping):
            continue
        path = f"{prefix}[{index}]"
        yield path, operation
        yield from _walk_operations(
            operation.get("body"), prefix=f"{path}.body"
        )
        if operation.get("op") in {"effect_if", "value_if"}:
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                if isinstance(arm, Mapping):
                    yield from _walk_operations(
                        arm.get("body"),
                        prefix=f"{path}.{polarity}.body",
                    )


_FLOAT_PRECISION_RANK = {"bf16": 0, "f16": 1, "f32": 2, "f64": 3}
_MIXED_PRECISION_OPS = frozenset({"contraction", "reduce", "scan"})


def _output_low_precision_compute_dtype(
    j: SemanticArchive,
    output: str,
    output_dtype: str,
) -> str | None:
    """Return the least precise floating compute operand on one output path."""

    output_dtype = canonical_dtype(output_dtype)
    output_rank = _FLOAT_PRECISION_RANK.get(output_dtype)
    if output_rank is None:
        return None

    intent = j.intent.to_json()["intent_function"]
    operations = dict(_walk_operations(intent.get("body", [])))
    path_to_node = {path: node for node, path in j.node_ids.items()}
    nodes: dict[str, Mapping[str, Any]] = dict(operations)
    producers: dict[str, Mapping[str, Any]] = {}
    for path, operation in operations.items():
        node = path_to_node.get(path)
        if node is not None:
            nodes[node] = operation
        for result in _operation_results(operation):
            producers[result] = operation

    seeds: list[str] = []
    for binding in j_elaboration(j).output_bindings:
        if binding.output == output:
            seeds.append(binding.value)
    for access in j.boundary.accesses:
        if (
            access.kind.value == "store"
            and access.tensor == output
            and access.node_ref is not None
        ):
            producer = nodes.get(access.node_ref)
            if producer is not None:
                seeds.extend(_operation_results(producer))

    value_types = j_elaboration(j).value_type_by_name
    dependency_ops: list[Mapping[str, Any]] = []
    pending = list(dict.fromkeys(seeds))
    visited: set[str] = set()
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        operation = producers.get(value)
        if operation is None:
            continue
        dependency_ops.append(operation)
        pending.extend(_operation_dependencies(operation))

    candidates: list[str] = []
    for operation in dependency_ops:
        if operation.get("op") not in _MIXED_PRECISION_OPS:
            continue
        for operand in _operation_dependencies(operation):
            value_type = value_types.get(operand)
            if (
                value_type is None
                or not value_type.is_tensor
                or value_type.dtype is None
            ):
                continue
            operand_dtype = canonical_dtype(value_type.dtype)
            operand_rank = _FLOAT_PRECISION_RANK.get(operand_dtype)
            if operand_rank is not None and operand_rank < output_rank:
                candidates.append(operand_dtype)
    if not candidates:
        return None
    return min(candidates, key=lambda item: _FLOAT_PRECISION_RANK[item])


DEFAULT_GATE2_POLICY = ComparisonPolicy.create(
    "intentir.gate2.default.v2",
    (
        ("bf16", NumericTolerance(2.0e-2, 2.0e-3, equal_nan=True)),
        # The model-level Flash attention Out is an f16 materialized result;
        # its source kernel has a documented scratch-store/low-precision
        # execution path whose repeated launches differ by about 2e-3.  Q
        # must cover that external numerical behavior; J still declares only
        # the relation class and cannot select this threshold.
        ("f16", NumericTolerance(2.0e-2, 3.0e-3, equal_nan=True)),
        # Matching NaN locations denote the same floating result for partial
        # functions such as sqrt/rsqrt/log.  Unequal NaN masks remain a hard
        # failure in compare_outputs, so this cannot hide a domain disagreement.
        ("f32", NumericTolerance(1.0e-5, 1.0e-6, equal_nan=True)),
        ("f64", NumericTolerance(1.0e-12, 1.0e-12, equal_nan=True)),
    ),
    mixed_precision_tolerances=(
        ("bf16", NumericTolerance(2.0e-2, 2.0e-3, equal_nan=True)),
        # Iterated f16 contractions/reductions (notably long attention
        # recurrences) can be nondeterministic even when the same Source is
        # launched twice.  Keep this in external Q: J still declares only
        # ``tolerance`` and cannot choose its own threshold.
        ("f16", NumericTolerance(2.0e-2, 3.0e-3, equal_nan=True)),
        ("f32", NumericTolerance(1.0e-5, 1.0e-6, equal_nan=True)),
    ),
)


TF32_TENSORCORE_GATE2_POLICY = ComparisonPolicy.create(
    "intentir.gate2.tf32-tensorcore.v1",
    (
        ("bf16", NumericTolerance(2.0e-2, 2.0e-3, equal_nan=True)),
        ("f16", NumericTolerance(2.0e-2, 3.0e-3, equal_nan=True)),
        ("f32", NumericTolerance(2.0e-2, 3.0e-3, equal_nan=True)),
        ("f64", NumericTolerance(1.0e-12, 1.0e-12, equal_nan=True)),
    ),
    mixed_precision_tolerances=(
        ("bf16", NumericTolerance(2.0e-2, 2.0e-3, equal_nan=True)),
        ("f16", NumericTolerance(2.0e-2, 3.0e-3, equal_nan=True)),
        ("f32", NumericTolerance(2.0e-2, 3.0e-3, equal_nan=True)),
    ),
)


_GATE2_POLICIES = {
    DEFAULT_GATE2_POLICY.policy_id: DEFAULT_GATE2_POLICY,
    TF32_TENSORCORE_GATE2_POLICY.policy_id: TF32_TENSORCORE_GATE2_POLICY,
}


def gate2_policy_for(policy_id: str) -> ComparisonPolicy:
    try:
        return _GATE2_POLICIES[policy_id]
    except KeyError as exc:
        raise ComparisonPolicyError(
            f"unknown trusted Gate2 Q {policy_id!r}"
        ) from exc


__all__ = [
    "CloseRule",
    "ComparisonPolicy",
    "ComparisonPolicyError",
    "DEFAULT_GATE2_POLICY",
    "GATE2_POLICY_PROTOCOL",
    "NumericTolerance",
    "TF32_TENSORCORE_GATE2_POLICY",
    "canonical_dtype",
    "gate2_policy_for",
]
