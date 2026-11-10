"""Deterministic, symmetric ``Close_Q`` comparisons for Gate2."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from intentir.ir.functional_boundary import static_index_bounds_status
from intentir.ir.integer_range import proves_index_bounds
from intentir.ir.value_dimension import validate_ragged_runtime_invocation
from intentir.ir.canonical_j import j_elaboration

from intentir.verify.gate1.model import (
    DuplicatePolicy,
    LogicalUpdateSpec,
    ResultRelationKind,
    ResultRelationSpec,
    SemanticArchive,
)

from .model import OutputAgreement, PairwiseAgreement
from .policy import ComparisonPolicy, CloseRule, canonical_dtype


GATE2_COMPARISON_PROTOCOL = "external_q_symmetric_pairwise_v5"

_DTYPE_ALIASES = {
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
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
}


def _dtype_key(value: str) -> str:
    normalized = value.removeprefix("torch.").lower()
    return canonical_dtype(_DTYPE_ALIASES.get(normalized, normalized))


def _exact_element_domain(left: np.ndarray, right: np.ndarray) -> bool:
    exact_kinds = {"b", "i", "u"}
    return left.dtype.kind in exact_kinds and right.dtype.kind in exact_kinds


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return float("inf")
    if _exact_element_domain(left, right):
        if bool(np.array_equal(left, right)):
            return 0.0
        return float(
            max(
                abs(int(lhs) - int(rhs))
                for lhs, rhs in zip(left.flat, right.flat, strict=True)
            )
        )
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return float("inf")
    finite = np.isfinite(left) & np.isfinite(right)
    if not bool(np.any(finite)):
        return 0.0
    lhs = left[finite].astype(np.float64, copy=False)
    rhs = right[finite].astype(np.float64, copy=False)
    return float(np.max(np.abs(lhs - rhs)))


def close_q(
    left: np.ndarray,
    right: np.ndarray,
    rule: CloseRule,
) -> bool:
    """Apply a symmetric tolerance using max(|left|, |right|) as scale."""

    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return False
    if rule.exact or _exact_element_domain(lhs, rhs):
        return bool(np.array_equal(lhs, rhs))
    assert rule.tolerance is not None
    lhs_nan = np.isnan(lhs)
    rhs_nan = np.isnan(rhs)
    if not bool(np.array_equal(lhs_nan, rhs_nan)):
        return False
    if bool(np.any(lhs_nan)) and not rule.tolerance.equal_nan:
        return False
    non_nan = ~(lhs_nan | rhs_nan)
    # Equal infinities of the same sign pass; unequal or finite-vs-infinite do not.
    lhs_inf = np.isinf(lhs) & non_nan
    rhs_inf = np.isinf(rhs) & non_nan
    if not bool(np.array_equal(lhs_inf, rhs_inf)):
        return False
    if bool(np.any(lhs_inf)) and not bool(np.array_equal(lhs[lhs_inf], rhs[rhs_inf])):
        return False
    finite = np.isfinite(lhs) & np.isfinite(rhs)
    if not bool(np.any(finite)):
        return True
    lhs_finite = lhs[finite].astype(np.float64, copy=False)
    rhs_finite = rhs[finite].astype(np.float64, copy=False)
    error = np.abs(lhs_finite - rhs_finite)
    scale = np.maximum(np.abs(lhs_finite), np.abs(rhs_finite))
    bound = rule.tolerance.atol + rule.tolerance.rtol * scale
    return bool(np.all(error <= bound))


def _j_inputs_and_dimensions(
    j: SemanticArchive,
    input_bindings: Mapping[str, str],
    values: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    expected = {item.name for item in j.intent.inputs}
    if set(input_bindings) != expected:
        raise ValueError("Gate2 input bindings differ from canonical J inputs")
    try:
        result = {
            name: np.asarray(values[runtime_name])
            for name, runtime_name in input_bindings.items()
        }
    except KeyError as exc:
        raise ValueError(f"Gate2 is missing input {exc.args[0]!r}") from exc
    dimension_extents: dict[str, int] = {}
    for tensor in j.intent.inputs:
        value = result[tensor.name]
        actual_dtype = _dtype_key(value.dtype.name)
        expected_dtype = _dtype_key(tensor.dtype)
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"Gate2 input {tensor.name!r} dtype {actual_dtype!r} does not "
                f"match J dtype {expected_dtype!r}"
            )
        if value.ndim != len(tensor.shape):
            raise ValueError(
                f"Gate2 input {tensor.name!r} rank differs from canonical J"
            )
        for axis, (declared, actual) in enumerate(
            zip(tensor.shape, value.shape, strict=True)
        ):
            if isinstance(declared, int) and declared != actual:
                raise ValueError(
                    f"Gate2 input {tensor.name!r} extent at axis {axis} "
                    "differs from canonical J"
                )
            if isinstance(declared, str):
                previous = dimension_extents.setdefault(declared, int(actual))
                if previous != actual:
                    raise ValueError(
                        f"Gate2 symbolic extent {declared!r} is inconsistent"
                    )
    return result, dimension_extents


def _j_inputs(
    j: SemanticArchive,
    input_bindings: Mapping[str, str],
    values: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    return _j_inputs_and_dimensions(j, input_bindings, values)[0]


def _active_update_groups(
    update: LogicalUpdateSpec,
    j_inputs: Mapping[str, np.ndarray],
    *,
    require_values: bool,
) -> tuple[np.ndarray, dict[tuple[int, ...], list[np.generic]]]:
    missing = [
        ref
        for ref in (update.base, update.indices)
        if ref not in j_inputs
    ]
    if require_values and update.values not in j_inputs:
        missing.append(update.values)
    if missing:
        raise NotImplementedError(
            "Gate2 update relation currently requires external J input ref(s): "
            + ", ".join(sorted(set(missing)))
        )

    base = np.asarray(j_inputs[update.base])
    indices = np.asarray(j_inputs[update.indices])
    if indices.dtype.kind not in {"i", "u"}:
        raise ValueError("logical update indices must be integer-typed")
    if update.axis >= base.ndim or indices.ndim != base.ndim:
        raise ValueError("logical update rank/axis violates canonical J")
    values = (
        np.asarray(j_inputs[update.values])
        if require_values
        else np.zeros(indices.shape, dtype=base.dtype)
    )
    if values.shape != indices.shape:
        raise ValueError("logical update indices and values shapes differ")
    groups: dict[tuple[int, ...], list[np.generic]] = {}
    for coordinate in np.ndindex(indices.shape):
        destination = list(coordinate)
        index = int(indices[coordinate])
        if index < 0 or index >= base.shape[update.axis]:
            raise ValueError("logical update violates its index-bounds relation")
        destination[update.axis] = index
        if any(
            position < 0 or position >= base.shape[axis]
            for axis, position in enumerate(destination)
        ):
            raise ValueError("logical update non-axis coordinate is out of bounds")
        groups.setdefault(tuple(destination), []).append(values[coordinate])
    return base, groups


def validate_j_inputs(
    j: SemanticArchive,
    input_bindings: Mapping[str, str],
    values: Mapping[str, Any],
) -> None:
    """Validate typed input-domain and uniqueness preconditions before running."""

    j_inputs, dimension_extents = _j_inputs_and_dimensions(
        j, input_bindings, values
    )
    validate_ragged_runtime_invocation(
        j.intent,
        j_inputs,
        dims=dimension_extents,
    )
    for item in j.boundary.index_bounds:
        static_status = static_index_bounds_status(j, item)
        if static_status is True:
            continue
        if static_status is False:
            raise ValueError(
                f"Gate2 index-bounds precondition failed at {item.node_ref!r}"
            )
        if proves_index_bounds(
            j,
            item,
            logical_inputs=j_inputs,
            dimension_extents=dimension_extents,
        ):
            continue
        if item.data not in j_inputs or item.indices not in j_inputs:
            # Canonical internal gather/scatter values do not exist until the
            # J-only reference graph executes.  The structured evaluator checks
            # their exact runtime dtype, axis and bounds at the authored op; do
            # not misclassify that closed check as a missing input precondition.
            continue
        data = np.asarray(j_inputs[item.data])
        indices = np.asarray(j_inputs[item.indices])
        if item.axis >= data.ndim or indices.dtype.kind not in {"i", "u"}:
            raise ValueError("Gate2 indexed-domain input violates canonical J")
        if bool(np.any(indices < 0)) or bool(
            np.any(indices >= data.shape[item.axis])
        ):
            raise ValueError(
                f"Gate2 index-bounds precondition failed at {item.node_ref!r}"
            )
    for update in j.boundary.updates:
        if update.duplicate_policy is not DuplicatePolicy.FORBID:
            continue
        if update.base not in j_inputs or update.indices not in j_inputs:
            # The structured scatter evaluator enforces the same authored
            # duplicate policy over computed internal indices.
            continue
        _base, groups = _active_update_groups(
            update, j_inputs, require_values=False
        )
        if any(len(candidates) != 1 for candidates in groups.values()):
            raise ValueError(
                f"Gate2 unique-index precondition failed at {update.node_ref!r}"
            )


def compare_outputs(
    left_name: str,
    left: Mapping[str, np.ndarray],
    right_name: str,
    right: Mapping[str, np.ndarray],
    *,
    j: SemanticArchive,
    input_bindings: Mapping[str, str],
    output_bindings: Mapping[str, str],
    inputs: Mapping[str, Any],
    policy: ComparisonPolicy,
) -> PairwiseAgreement:
    """Compare two observations using J's class and external policy Q."""

    output_records = j_elaboration(j).output_bindings
    expected_outputs = {item.output for item in output_records}
    if set(output_bindings) != expected_outputs:
        raise ValueError("Gate2 output bindings differ from canonical J outputs")
    explicit = {item.output: item for item in j.boundary.result_relations}
    validate_j_inputs(j, input_bindings, inputs)
    observations: list[OutputAgreement] = []

    for output in output_records:
        logical_key = output.output
        runtime_key = output_bindings[logical_key]
        relation = explicit.get(
            logical_key,
            ResultRelationSpec(logical_key, ResultRelationKind.EXACT),
        )
        rule = policy.rule_for_output(
            j,
            logical_key,
            relation.kind,
            output.output_type.dtype,
        )
        missing = []
        if runtime_key not in left:
            missing.append(left_name)
        if runtime_key not in right:
            missing.append(right_name)
        if missing:
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    f"output {runtime_key!r} missing from {', '.join(missing)}",
                    relation.kind.value,
                    relation.kind is not ResultRelationKind.ONE_OF_UPDATES,
                    policy.policy_id,
                    (
                        float(rule.tolerance.rtol)
                        if rule.tolerance is not None else None
                    ),
                    (
                        float(rule.tolerance.atol)
                        if rule.tolerance is not None else None
                    ),
                    (
                        rule.tolerance.equal_nan
                        if rule.tolerance is not None else None
                    ),
                )
            )
            continue

        lhs = np.asarray(left[runtime_key])
        rhs = np.asarray(right[runtime_key])
        expected_dtype = _dtype_key(output.output_type.dtype)
        observed_dtypes = (_dtype_key(lhs.dtype.name), _dtype_key(rhs.dtype.name))
        if observed_dtypes != (expected_dtype, expected_dtype):
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    "output dtype differs from canonical J: "
                    f"{observed_dtypes} != {(expected_dtype, expected_dtype)}",
                    relation.kind.value,
                    relation.kind is not ResultRelationKind.ONE_OF_UPDATES,
                    policy.policy_id,
                    (
                        float(rule.tolerance.rtol)
                        if rule.tolerance is not None else None
                    ),
                    (
                        float(rule.tolerance.atol)
                        if rule.tolerance is not None else None
                    ),
                    (
                        rule.tolerance.equal_nan
                        if rule.tolerance is not None else None
                    ),
                )
            )
            continue

        if lhs.shape != rhs.shape:
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    f"shape mismatch: {lhs.shape} != {rhs.shape}",
                    relation.kind.value,
                    True,
                    policy.policy_id,
                )
            )
            continue
        if lhs.dtype != rhs.dtype:
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    f"dtype mismatch: {lhs.dtype} != {rhs.dtype}",
                    relation.kind.value,
                    True,
                    policy.policy_id,
                )
            )
            continue
        passed = close_q(lhs, rhs, rule)
        observations.append(
            OutputAgreement(
                runtime_key,
                passed,
                _max_abs_error(lhs, rhs),
                "",
                relation.kind.value,
                True,
                policy.policy_id,
                (
                    float(rule.tolerance.rtol)
                    if rule.tolerance is not None else None
                ),
                (
                    float(rule.tolerance.atol)
                    if rule.tolerance is not None else None
                ),
                (
                    rule.tolerance.equal_nan
                    if rule.tolerance is not None else None
                ),
            )
        )

    return PairwiseAgreement(
        left_name,
        right_name,
        all(item.passed for item in observations),
        tuple(observations),
        policy.policy_id,
    )


__all__ = [
    "GATE2_COMPARISON_PROTOCOL",
    "close_q",
    "compare_outputs",
    "validate_j_inputs",
]
