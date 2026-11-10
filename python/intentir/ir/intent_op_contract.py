"""Single structural contract for canonical Layer-A ``IntentOp`` JSON.

The contract deliberately recognizes three structural discriminators:
``op == "loop"``, ``op == "effect_if"``, and ``op == "value_if"``.  Ordinary
semantic operations use one shared, closed opcode set so an unknown spelling
cannot pass structural validation and fail only in a later backend.  The scalar
function sets below are closed because a spelling that cannot reach a real
lowering is not a semantic capability.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import AbstractSet, Any


MAP_UNARY_FNS = frozenset(
    {
        "identity", "exp", "log", "log2", "sqrt", "rsqrt", "rsqrt_ftz", "neg", "recip",
        "abs", "tanh", "sigmoid", "cos", "sin", "tan", "exp2", "atan", "erf",
        "acos", "asin", "cbrt", "ceil", "floor", "round", "roundeven",
        "trunc", "sign", "log10", "log1p", "expm1", "erfc",
        "isnan", "isinf", "isfinite", "logical_not",
    }
)
MAP_UNARY_PREDICATE_FNS = frozenset(
    {"isnan", "isinf", "isfinite", "logical_not"}
)
MAP_UNARY_NUMERIC_FNS = frozenset({"identity", "neg", "abs", "sign"})
MAP_UNARY_FLOAT_FNS = MAP_UNARY_FNS - MAP_UNARY_PREDICATE_FNS - MAP_UNARY_NUMERIC_FNS

MAP_BINARY_ARITHMETIC_FNS = frozenset(
    {"add", "sub", "mul", "div", "max", "min"}
)
MAP_BINARY_COMPARISON_FNS = frozenset(
    {"gt", "ge", "lt", "le", "eq", "ne"}
)
MAP_BINARY_LOGICAL_FNS = frozenset(
    {"logical_and", "logical_or", "logical_xor", "and", "or", "xor"}
)
MAP_BINARY_FLOATING_FNS = frozenset({"pow", "atan2", "copysign"})
MAP_BINARY_MATCHING_NUMERIC_FNS = frozenset({"floordiv", "remainder"})
MAP_BINARY_SHIFT_FNS = frozenset({"left_shift", "right_shift"})
MAP_BINARY_PREDICATE_FNS = (
    MAP_BINARY_COMPARISON_FNS | MAP_BINARY_LOGICAL_FNS | {"isclose"}
)
MAP_BINARY_FNS = frozenset(
    MAP_BINARY_ARITHMETIC_FNS
    | MAP_BINARY_COMPARISON_FNS
    | MAP_BINARY_LOGICAL_FNS
    | MAP_BINARY_FLOATING_FNS
    | MAP_BINARY_MATCHING_NUMERIC_FNS
    | MAP_BINARY_SHIFT_FNS
    | {"isclose"}
)

# Logical J dtypes are intentionally a closed set.  Physical source/runtime
# spellings (for example ``float32`` or ``cuda_tensor_map``) belong to the
# Layer-C/source manifests and must not leak into the Layer-A value contract.
CANONICAL_DTYPES = frozenset(
    {
        "f16", "bf16", "f32", "f64",
        "i1", "i8", "i16", "i32", "i64",
        "u8", "u16", "u32",
    }
)
LOGICAL_DTYPES = CANONICAL_DTYPES
FLOATING_DTYPES = frozenset({"f16", "bf16", "f32", "f64"})
INTEGER_DTYPES = frozenset(
    {"i1", "i8", "i16", "i32", "i64", "u8", "u16", "u32"}
)


def canonicalize_dtype(value: object, *, context: str = "dtype") -> str:
    """Validate and canonicalize one logical J dtype spelling."""

    if not isinstance(value, str):
        _fail(context, "must be a closed logical dtype string")
    dtype = value
    if dtype not in CANONICAL_DTYPES:
        _fail(
            context,
            "must be one of " + ", ".join(sorted(CANONICAL_DTYPES)),
        )
    return dtype


def is_closed_dtype(value: object) -> bool:
    return isinstance(value, str) and value in CANONICAL_DTYPES


def is_closed_integer_dtype(value: object, *, include_boolean: bool = True) -> bool:
    if not isinstance(value, str) or value not in CANONICAL_DTYPES:
        return False
    canonical = value
    return canonical in INTEGER_DTYPES and (
        include_boolean or canonical != "i1"
    )


def is_closed_floating_dtype(value: object) -> bool:
    return (
        isinstance(value, str)
        and value in FLOATING_DTYPES
    )


def is_consumer_reduction_identity(fn: object, dtype: object, value: object) -> bool:
    """Return whether ``value`` is a closed identity for a padded consumer.

    FunctionalBoundary and TypeCheck must agree on the outside-domain value
    attached to a reduce/scan operand. Keep this predicate beside the canonical
    dtype contract so those projections cannot drift into different acceptance
    sets. Operand dtype equality and producer provenance remain separate.
    """

    if fn == "sum":
        return value == "ZERO" or (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == 0.0
        )
    if fn == "prod":
        return value == "ONE" or (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == 1.0
        )
    if fn == "any":
        return value is False or value == "false"
    if fn == "all":
        return value is True or value == "true"
    if not is_closed_floating_dtype(dtype):
        return False
    if fn == "max":
        return isinstance(value, str) and value in {"NEG_INF", "-inf"}
    if fn == "min":
        return isinstance(value, str) and value in {"POS_INF", "+inf", "inf"}
    return False


def is_closed_boolean_dtype(value: object) -> bool:
    return (
        isinstance(value, str)
        and value == "i1"
    )


def is_closed_numeric_dtype(value: object) -> bool:
    """Return whether ``value`` is a closed logical numeric dtype."""

    return is_closed_floating_dtype(value) or is_closed_integer_dtype(value)


def _canonical_dtype_or_none(value: object) -> str | None:
    try:
        return canonicalize_dtype(value)
    except (TypeError, ValueError):
        return None


def promote_dtype(lhs: object, rhs: object) -> str | None:
    """Promote two closed logical dtypes without losing signed values.

    The result is a logical J dtype, not a NumPy or backend storage spelling.
    Integer promotion deliberately chooses a wider signed type when a signed
    and unsigned operand have equal width; returning the unsigned type in that
    case would silently truncate negative values from the signed operand.
    ``None`` is accepted for partial inference and an invalid closed dtype
    returns ``None``.
    """

    left = _canonical_dtype_or_none(lhs) if lhs is not None else None
    right = _canonical_dtype_or_none(rhs) if rhs is not None else None
    if left is None:
        return right
    if right is None:
        return left
    if left == right:
        return left

    if left in FLOATING_DTYPES or right in FLOATING_DTYPES:
        floats = {
            dtype for dtype in (left, right) if dtype in FLOATING_DTYPES
        }
        if "f64" in floats:
            return "f64"
        if "f32" in floats or floats == {"f16", "bf16"}:
            return "f32"
        # Preserve the established project convention for a single low-
        # precision float combined with an integer operand.
        return next(iter(floats)) if floats else None

    if not is_closed_integer_dtype(left) or not is_closed_integer_dtype(right):
        return None
    widths = {
        "i1": 1,
        "i8": 8,
        "i16": 16,
        "i32": 32,
        "i64": 64,
        "u8": 8,
        "u16": 16,
        "u32": 32,
    }
    signed = {"i1", "i8", "i16", "i32", "i64"}
    left_width = widths[left]
    right_width = widths[right]
    left_signed = left in signed
    right_signed = right in signed
    if left_signed == right_signed:
        return left if left_width >= right_width else right

    signed_dtype, signed_width = (
        (left, left_width) if left_signed else (right, right_width)
    )
    unsigned_width = right_width if left_signed else left_width
    if signed_width > unsigned_width:
        return signed_dtype
    for candidate in ("i8", "i16", "i32", "i64"):
        if widths[candidate] > unsigned_width:
            return candidate
    return None


def map_unary_result_dtype(fn: object, input_dtype: object) -> str | None:
    """Return the closed result dtype for one ``map_unary`` function."""

    if not isinstance(fn, str) or fn not in MAP_UNARY_FNS:
        return None
    dtype = _canonical_dtype_or_none(input_dtype)
    if dtype is None:
        return None
    if fn in {"isnan", "isinf", "isfinite"}:
        return "i1" if dtype in FLOATING_DTYPES else None
    if fn == "logical_not":
        return "i1" if dtype == "i1" else None
    if fn == "rsqrt_ftz":
        return "f32" if dtype == "f32" else None
    if fn in MAP_UNARY_NUMERIC_FNS:
        return dtype if is_closed_numeric_dtype(dtype) else None
    return dtype if dtype in FLOATING_DTYPES else None


def map_binary_result_dtype(
    fn: object, lhs_dtype: object, rhs_dtype: object
) -> str | None:
    """Return the closed result dtype for one ``map_binary`` function.

    This is the single dtype contract shared by TypeCheck, MLIR emission and
    reference execution.  Backends may require already matching MLIR operand
    types, but they must materialize the same promotion before invoking their
    scalar operation.
    """

    if not isinstance(fn, str) or fn not in MAP_BINARY_FNS:
        return None
    lhs = _canonical_dtype_or_none(lhs_dtype)
    rhs = _canonical_dtype_or_none(rhs_dtype)
    if lhs is None or rhs is None:
        return None

    if fn in MAP_BINARY_COMPARISON_FNS:
        return "i1" if is_closed_numeric_dtype(lhs) and is_closed_numeric_dtype(rhs) else None
    if fn in MAP_BINARY_LOGICAL_FNS:
        return "i1" if is_closed_numeric_dtype(lhs) and is_closed_numeric_dtype(rhs) else None

    promoted = promote_dtype(lhs, rhs)
    if fn == "isclose":
        return "i1" if promoted in FLOATING_DTYPES else None
    if fn in MAP_BINARY_ARITHMETIC_FNS:
        return promoted if promoted is not None else None
    if fn in {"pow", "atan2"}:
        return promoted if lhs in FLOATING_DTYPES and rhs in FLOATING_DTYPES else None
    if fn in MAP_BINARY_MATCHING_NUMERIC_FNS:
        return lhs if lhs == rhs and is_closed_numeric_dtype(lhs) else None
    if fn in MAP_BINARY_SHIFT_FNS:
        return lhs if lhs == rhs and is_closed_integer_dtype(lhs) else None
    if fn == "copysign":
        return lhs if lhs == rhs and lhs in FLOATING_DTYPES else None
    return None


def _broadcast_projection(
    shape: list[object] | None,
    axes: Sequence[int],
    target_shape: Sequence[object] | None = None,
    *,
    unit_dimensions: AbstractSet[str] = frozenset(),
) -> tuple[list[object] | None, tuple[int, ...]]:
    """Project one operand through the canonical J broadcast contract.

    This is an IR-level shape rule, not an MLIR lowering detail.  A declared
    broadcast axis may already be represented by a singleton source extent or
    may be absent from a lower-rank operand; only the latter requires rank
    insertion.  Keeping the alignment here gives canonical J, TypeCheck,
    emitters, and reference execution one fail-closed implementation.
    """

    if shape is None:
        return None, ()
    target_rank = (
        len(target_shape)
        if target_shape is not None
        else len(shape) + len(axes)
    )
    normalized: list[int] = []
    for raw_axis in axes:
        if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
            raise NotImplementedError(
                "broadcast axes currently require integer positions"
            )
        axis = raw_axis if raw_axis >= 0 else raw_axis + target_rank
        if axis < 0 or axis >= target_rank:
            raise ValueError(
                f"broadcast axis {raw_axis} is outside target rank {target_rank}"
            )
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        raise ValueError("broadcast axes must be unique")

    def is_unit(extent: object) -> bool:
        return extent == 1 or (
            isinstance(extent, str) and extent in unit_dimensions
        )

    if not axes:
        if target_shape is not None and not shape:
            # A scalar has an empty logical input map.  Keep that fact in the
            # returned insertion set so consumers that ask for source-axis
            # positions still receive an empty tuple.
            return [1] * target_rank, tuple(range(target_rank))
        if target_shape is not None and len(shape) == target_rank:
            target = list(target_shape)
            if any(
                source_extent != target_extent and not is_unit(source_extent)
                for source_extent, target_extent in zip(shape, target, strict=True)
            ):
                raise ValueError(
                    f"operand shape {shape} cannot broadcast to consumer shape "
                    f"{target} without authored rank insertion"
                )
        return list(shape), ()

    if target_shape is None:
        inserted = set(normalized)
        source = iter(shape)
        try:
            projected = [
                1 if index in inserted else next(source)
                for index in range(target_rank)
            ]
        except StopIteration as exc:
            raise ValueError(
                f"broadcast axes {list(axes)} over-specify operand rank {len(shape)}"
            ) from exc
        if next(source, None) is not None:
            raise ValueError(
                f"broadcast axes {list(axes)} cannot project operand rank {len(shape)}"
            )
        return projected, tuple(sorted(inserted))

    target = list(target_shape)
    if len(shape) > target_rank:
        raise ValueError(
            f"broadcast axes {list(axes)} cannot project rank {len(shape)} "
            f"operand to consumer rank {target_rank}"
        )
    if not shape:
        # A scalar uses an empty affine map and broadcasts across the complete
        # consumer domain without materializing a semantic rank insertion.
        return [1] * target_rank, tuple(range(target_rank))

    axis_set = set(normalized)
    memo: dict[tuple[int, int], tuple[int, ...] | None] = {}

    def same_extent(left: object, right: object) -> bool:
        return left == right or (is_unit(left) and is_unit(right))

    def align(source_index: int, target_index: int) -> tuple[int, ...] | None:
        key = (source_index, target_index)
        if key in memo:
            return memo[key]
        if target_index == target_rank:
            result = () if source_index == len(shape) else None
            memo[key] = result
            return result

        if target_index in axis_set:
            # Prefer retaining an authored singleton at this position.  If it
            # cannot complete the suffix, treat the axis as a missing rank.
            if source_index < len(shape) and is_unit(shape[source_index]):
                suffix = align(source_index + 1, target_index + 1)
                if suffix is not None:
                    memo[key] = suffix
                    return suffix
            suffix = align(source_index, target_index + 1)
            if suffix is not None:
                result = (target_index, *suffix)
                memo[key] = result
                return result
        elif (
            source_index < len(shape)
            and same_extent(shape[source_index], target[target_index])
        ):
            suffix = align(source_index + 1, target_index + 1)
            if suffix is not None:
                memo[key] = suffix
                return suffix

        memo[key] = None
        return None

    inserted_axes = align(0, 0)
    if inserted_axes is None:
        raise ValueError(
            f"operand shape {shape} cannot align with consumer shape {target} "
            f"using broadcast axes {list(axes)}; existing broadcast "
            "dimensions must be size one and missing dimensions must be "
            "declared broadcast axes"
        )
    projected = [
        1 if index in axis_set else target_extent
        for index, target_extent in enumerate(target)
    ]
    return projected, inserted_axes

# Keep the Ragged operation vocabulary in the same structural contract used by
# the JSON parser, type checker, and backend capability probes.  Backend code
# may decide whether a target can lower these operations, but it must not grow
# a second, potentially stale spelling/arity table.
RAGGED_OPS = frozenset({"ragged", "ragged_outer", "ragged_member"})
RAGGED_ARITY = {
    "ragged": 2,
    "ragged_outer": 1,
    "ragged_member": 2,
}
# Ragged carriers are an ABI-visible integer vector, not an arbitrary integer
# tensor.  Keep this intersection of the actual CPU/CUDA/RVV storage paths
# closed so a parser cannot admit a carrier that a later backend cannot carry.
# ``index`` is the logical spelling; concrete MLIR materialization uses i64.
RAGGED_CARRIER_DTYPES = frozenset(
    {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "index"}
)

# Keep the canonical Layer-A opcode vocabulary in one place.  ``const``,
# ``record`` and ``record_get`` are J-level value operations even though they
# do not each have a one-to-one Intent dialect op.  ``yield`` is deliberately
# absent: it is the MLIR loop terminator and is represented by the JSON loop
# ``yield`` list, not as an operation in a J body.
CANONICAL_J_OPS = frozenset(
    {
        "const",
        "map_unary",
        "map_binary",
        "cast",
        "select",
        "iota",
        "scan",
        "arg_reduce",
        "sort",
        "gather",
        "scatter",
        "ragged",
        "ragged_outer",
        "ragged_member",
        "nonzero",
        "masked_select",
        "masked_scatter",
        "reshape",
        "transpose",
        "concat",
        "pad",
        "tile",
        "flip",
        "triu",
        "rand",
        "pool",
        "upsample",
        "polar",
        "conv",
        "reduce",
        "contraction",
        "record",
        "record_get",
        "loop",
        "effect_if",
        "value_if",
    }
)
# ``atomic_scatter`` is a concrete Intent/MLIR lowering form for the canonical
# J spelling ``scatter`` with ``atomic=\"cas\"``.  It remains named here for
# backend capability and internal lowering code, but it is intentionally not
# accepted by the canonical J parser or exposed in provider-facing schema.
INTERNAL_INTENT_OPS = frozenset({"atomic_scatter"})
KNOWN_INTENT_OPS = CANONICAL_J_OPS | INTERNAL_INTENT_OPS
STRUCTURED_INTENT_OPS = frozenset({"loop", "effect_if", "value_if"})

# Operand cardinality is semantic syntax, not an implementation detail of a
# particular lowering.  Keep the fixed (or bounded) leaf cardinalities next
# to the canonical J vocabulary so the structural parser, provider-facing
# graph boundary, and later canonical construction all consult the same
# authority.  ``record`` remains deliberately variadic: its exact cardinality
# is tied to its authored record-field type and is checked with that type.
# Likewise, ``concat`` is variadic but cannot be empty.
_CANONICAL_LEAF_OPERAND_ARITIES: dict[str, tuple[int, int | None]] = {
    "const": (0, 0),
    "iota": (0, 0),
    "rand": (0, 0),
    "map_unary": (1, 1),
    "cast": (1, 1),
    "scan": (1, 1),
    "arg_reduce": (1, 1),
    "sort": (1, 1),
    "nonzero": (1, 1),
    "reshape": (1, 1),
    "transpose": (1, 1),
    "pad": (1, 1),
    "tile": (1, 1),
    "flip": (1, 1),
    "triu": (1, 1),
    "pool": (1, 1),
    "upsample": (1, 1),
    "record_get": (1, 1),
    "reduce": (1, 1),
    "ragged_outer": (1, 1),
    "map_binary": (2, 2),
    "contraction": (2, 2),
    "gather": (2, 2),
    "ragged": (2, 2),
    "ragged_member": (2, 2),
    "masked_select": (2, 2),
    "polar": (2, 2),
    "select": (3, 3),
    "masked_scatter": (3, 3),
    "conv": (2, 3),
    "concat": (1, None),
    "record": (0, None),
}


def operation_operand_arity_contract(
    op: str,
    attrs: Mapping[str, Any] | None = None,
) -> tuple[int, int | None] | None:
    """Return the closed operand-cardinality range for one ordinary J op.

    ``None`` means the operation is structured or unknown to this leaf
    projection.  The ``scatter`` CAS form is the only canonical leaf whose
    arity depends on one declared attribute: it carries data, index, compare,
    and replacement instead of the ordinary data/index/source triplet.
    Attribute spelling and semantic validity are still checked by their own
    shared contracts; this helper intentionally exposes only cardinality.
    """

    if not isinstance(op, str):
        return None
    if op == "scatter":
        if isinstance(attrs, Mapping) and attrs.get("atomic") == "cas":
            return (4, 4)
        return (3, 3)
    return _CANONICAL_LEAF_OPERAND_ARITIES.get(op)


def operation_operands_match_arity(
    op: str,
    operands: object,
    attrs: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether operands meet the canonical leaf arity contract.

    Consumers must use this helper instead of duplicating the operation
    cardinality table in an MCP boundary, parser, or lowering.
    """

    if not isinstance(operands, list):
        return False
    arity = operation_operand_arity_contract(op, attrs)
    if arity is None:
        return True
    minimum, maximum = arity
    return len(operands) >= minimum and (
        maximum is None or len(operands) <= maximum
    )


def is_ragged_carrier_dtype(value: object) -> bool:
    """Return whether ``value`` is one of the closed logical Ragged dtypes."""

    return isinstance(value, str) and value in RAGGED_CARRIER_DTYPES


def is_non_boolean_integer_dtype(value: object) -> bool:
    """Return whether ``value`` is a loop-bound integer dtype.

    MLIR represents booleans as ``i1`` and accepts them in the general integer
    type family, but a boolean is not a valid loop trip or induction carrier.
    Keep this predicate in the shared IntentOp contract so the Python model,
    canonical J elaborator, and backend-facing type checks do not drift.
    """

    return is_closed_integer_dtype(value, include_boolean=False)


ATOMIC_KINDS = frozenset({"add", "cas"})
ATOMIC_ORDERINGS = frozenset(
    {"relaxed", "acquire", "release", "acq_rel", "seq_cst"}
)
ATOMIC_MEMORY_SCOPES = frozenset(
    {"work_item", "subgroup", "workgroup", "device", "system"}
)
ATOMIC_CAS_RETRIES = frozenset({"single_attempt", "until_success"})


COMMON_FIELDS = frozenset(
    {"id", "op", "result", "operands", "attrs", "type"}
)
LOOP_ONLY_FIELDS = frozenset(
    {"result", "scope_id", "carry", "trip", "scan", "body", "yield"}
)
EFFECT_IF_ONLY_FIELDS = frozenset(
    {"predicate", "positive", "negative"}
)
VALUE_IF_STRUCTURAL_FIELDS = frozenset(
    {"predicate", "positive", "negative"}
)
VALUE_IF_FIELDS = frozenset(
    {"result", "type", *VALUE_IF_STRUCTURAL_FIELDS}
)
COMMON_REQUIRED_FIELDS = frozenset({"op"})
LOOP_REQUIRED_FIELDS = frozenset(
    {"result", "scope_id", "carry", "trip", "scan", "body", "yield"}
)
EFFECT_IF_REQUIRED_FIELDS = frozenset(
    {"predicate", "positive", "negative"}
)
VALUE_IF_REQUIRED_FIELDS = frozenset(
    {"result", "type", "predicate", "positive", "negative"}
)


class IntentOpContractError(ValueError):
    """An IntentOp violates the shared wire/parser structural shape."""


def _fail(path: str, message: str) -> None:
    raise IntentOpContractError(f"{path} {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(path, "must be an object with string keys")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _handle(value: Any, path: str, prefixes: str) -> str:
    """Validate one closed logical/SSA handle used by an IntentOp.

    Layer A has two value namespaces: ``t*`` for logical function-boundary
    tensors (including rank-0 logical values), and ``v*`` for operation and
    region SSA values.  Physical ``p*`` parameters never occur in an op graph.
    Keeping this check in the shared structural contract makes the wire schema
    and the strict Gate1 parser reject the same namespace mistake.
    """

    result = _nonempty_string(value, path)
    if re.fullmatch(fr"[{prefixes}][0-9]+", result) is None:
        names = " or ".join(f"{prefix}*" for prefix in prefixes)
        _fail(
            path,
            f"must be a {names} handle; dimensions, node ids, physical "
            "parameters, C slots, and literals are not SSA operands",
        )
    return result


def _value_handle(value: Any, path: str) -> str:
    return _handle(value, path, "tv")


def _ssa_handle(value: Any, path: str) -> str:
    return _handle(value, path, "v")


SCALAR_LITERAL_TOKENS = frozenset({"-inf", "inf", "nan"})
_PHYSICAL_PARAMETER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])p[0-9]+(?![A-Za-z0-9_])"
)


def is_scalar_literal(value: Any) -> bool:
    """Return whether ``value`` is one of the closed scalar literal forms.

    This vocabulary deliberately mirrors ``validate_const_literal``: finite
    JSON numbers/booleans and the three explicit floating special strings.
    Source names, physical handles, symbolic aliases such as ``NEG_INF`` or
    ``zeros``, and numeric strings are not Layer-A literals.
    """

    if isinstance(value, bool) or isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if not isinstance(value, str):
        return False
    return value in SCALAR_LITERAL_TOKENS


def is_scalar_literal_or_value_reference(value: Any) -> bool:
    """Return whether a loop initializer is a closed scalar/reference value.

    Loop ``init_value`` is the one IntentOp field that intentionally accepts
    either a scalar literal or a value reference.  Keeping this predicate in
    the shared contract prevents the JSON schema, strict Gate1 parser, and
    typed callers from disagreeing about arbitrary strings such as ``p8`` or
    a Source parameter name.  Visibility of ``t*``/``v*`` references remains a
    lexical/semantic check owned by the caller.
    """

    if isinstance(value, str) and re.fullmatch(r"(?:t|v)[0-9]+", value) is not None:
        return True
    return is_scalar_literal(value)


def _dimension_handle(value: Any, path: str) -> str:
    return _handle(value, path, "x")


def _string_array(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
    unique: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    result = [
        _nonempty_string(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    ]
    if nonempty and not result:
        _fail(path, "must be a non-empty array")
    if unique and len(result) != len(set(result)):
        _fail(path, "must contain unique values")
    return result


def _nonnegative_integer(value: Any, path: str) -> int:
    # JSON Schema defines 1 and 1.0 as the same mathematical integer.  Mirror
    # that wire semantics, then let the typed model canonicalize it back to int.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a non-negative integer")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        _fail(path, "must be a non-negative integer")
    result = int(value)
    if result < 0:
        _fail(path, "must be a non-negative integer")
    return result


def _json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _json_value(item, f"{path}.{key}")
        return
    _fail(path, "must contain only finite JSON values")


def _exact_keys(
    value: Any,
    path: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
) -> dict[str, Any]:
    item = _object(value, path)
    missing = sorted(required - set(item))
    unknown = sorted(set(item) - allowed)
    if missing:
        _fail(path, "is missing required key(s): " + ", ".join(missing))
    if unknown:
        _fail(path, "has unknown key(s): " + ", ".join(unknown))
    return item


def _shape(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    normalized: list[Any] = []
    for index, dimension in enumerate(value):
        item_path = f"{path}[{index}]"
        if isinstance(dimension, str):
            _nonempty_string(dimension, item_path)
            if _PHYSICAL_PARAMETER_TOKEN.search(dimension) is not None:
                _fail(
                    item_path,
                    "cannot reference physical parameter p* in a logical shape",
                )
            normalized.append(dimension)
        else:
            normalized.append(_nonnegative_integer(dimension, item_path))
    return tuple(normalized)


@dataclass(frozen=True)
class ConstLiteral:
    """Validated payload for one logical ``const`` operation.

    ``values`` is flattened only for a list literal.  A scalar literal keeps a
    one-element tuple and sets ``splat`` so consumers can materialize it over
    the declared result shape without confusing a scalar with a rank-one
    vector.  The validator deliberately returns the authored special-float
    tokens; CPU and MLIR consumers map those tokens to their own exact runtime
    representation.
    """

    dtype: str | None
    shape: tuple[Any, ...]
    values: tuple[Any, ...]
    splat: bool


_CONST_SPECIAL_FLOATS = {
    "-inf": -math.inf,
    "inf": math.inf,
    "nan": math.nan,
}


def _const_literal_shape(value: Any, path: str) -> tuple[int, ...]:
    """Return the concrete rectangular shape of a nested JSON list literal."""

    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child_shapes = [
        _const_literal_shape(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    ]
    first = child_shapes[0]
    if any(shape != first for shape in child_shapes[1:]):
        _fail(path, "must be a rectangular nested list literal")
    return (len(value), *first)


def const_literal_shape(value: Any, *, context: str = "const.value") -> tuple[int, ...]:
    """Return a list literal's concrete shape, rejecting ragged nesting."""

    return _const_literal_shape(value, context)


def _flatten_const_literal(value: Any, path: str) -> tuple[Any, ...]:
    if isinstance(value, list):
        flattened: list[Any] = []
        for index, item in enumerate(value):
            flattened.extend(_flatten_const_literal(item, f"{path}[{index}]"))
        return tuple(flattened)
    return (value,)


def _validate_const_scalar(
    value: Any,
    dtype: str | None,
    path: str,
) -> Any:
    """Validate and normalize one scalar const literal."""

    if dtype is None:
        if isinstance(value, bool):
            return value
        if type(value) is int:
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if isinstance(value, str) and value in _CONST_SPECIAL_FLOATS:
            _fail(
                path,
                "special floating literals require an explicitly known floating dtype",
            )
        _fail(
            path,
            "must be a finite number, boolean, or one of '-inf', 'inf', 'nan'",
        )

    canonical = canonicalize_dtype(dtype, context=f"{path}.dtype")
    if canonical == "i1":
        if isinstance(value, bool):
            return value
        if type(value) is int and value in (0, 1):
            return value
        if type(value) is float and math.isfinite(value) and value in (0.0, 1.0):
            return int(value)
        _fail(path, "i1 literals must be boolean or the integer values 0/1")

    if canonical in INTEGER_DTYPES:
        if isinstance(value, bool):
            _fail(path, f"{canonical} literals must not be boolean")
        if type(value) is int:
            integer = value
        elif type(value) is float and math.isfinite(value) and value.is_integer():
            integer = int(value)
        else:
            _fail(path, f"{canonical} literals must be finite integer values")
        width = 64 if canonical == "i64" else int(canonical[1:])
        if canonical.startswith("u"):
            lower, upper = 0, (1 << width) - 1
        else:
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if integer < lower or integer > upper:
            _fail(
                path,
                f"{canonical} literal {integer} is outside [{lower}, {upper}]",
            )
        return integer

    if canonical in FLOATING_DTYPES:
        if isinstance(value, bool):
            _fail(path, f"{canonical} literals must be numeric, not boolean")
        if type(value) is int:
            return value
        if type(value) is float:
            if not math.isfinite(value):
                _fail(path, "floating numeric literals must be finite")
            return value
        if isinstance(value, str) and value in _CONST_SPECIAL_FLOATS:
            return value
        _fail(
            path,
            f"{canonical} literals must be finite numbers or '-inf', 'inf', 'nan'",
        )

    # ``canonicalize_dtype`` above makes this unreachable, but keep the
    # function total if the closed dtype sets change independently.
    _fail(path, f"unsupported const dtype {canonical!r}")


def validate_const_literal(
    attrs: Mapping[str, Any],
    *,
    result_type: Mapping[str, Any] | None = None,
    context: str = "const",
) -> ConstLiteral:
    """Validate one ``const`` operation across all J consumers.

    The operation has no operands, exactly one SSA result, and a closed attrs
    object containing ``value`` plus optional ``dtype`` and ``shape``.  A
    scalar may splat to a declared (including symbolic) result shape.  A list
    literal represents a concrete dense tensor and therefore has to be
    rectangular, static, and shape-identical to the authored/result shape.
    """

    if not isinstance(attrs, Mapping):
        _fail(f"{context}.attrs", "must be an object")
    unknown = sorted(set(attrs) - {"value", "dtype", "shape"})
    if unknown:
        _fail(
            f"{context}.attrs",
            "has unsupported key(s): " + ", ".join(unknown),
        )
    if "value" not in attrs:
        _fail(f"{context}.attrs", "is missing required key 'value'")

    result_dtype: str | None = None
    result_shape: tuple[Any, ...] | None = None
    if result_type is not None:
        if not isinstance(result_type, Mapping):
            _fail(f"{context}.type", "must be a tensor value type")
        _value_type(dict(result_type), f"{context}.type")
        if result_type.get("kind", "tensor") != "tensor":
            _fail(f"{context}.type", "must be a tensor value type")
        result_dtype = canonicalize_dtype(
            result_type.get("dtype"),
            context=f"{context}.type.dtype",
        )
        result_shape = _shape(
            result_type.get("shape"),
            f"{context}.type.shape",
        )

    authored_dtype: str | None = None
    if "dtype" in attrs:
        authored_dtype = canonicalize_dtype(
            attrs["dtype"],
            context=f"{context}.attrs.dtype",
        )
    if (
        authored_dtype is not None
        and result_dtype is not None
        and authored_dtype != result_dtype
    ):
        _fail(
            f"{context}.attrs.dtype",
            f"does not match result dtype {result_dtype!r}",
        )
    dtype = authored_dtype or result_dtype

    authored_shape: tuple[Any, ...] | None = None
    if "shape" in attrs:
        authored_shape = _shape(attrs["shape"], f"{context}.attrs.shape")

    value = attrs["value"]
    literal_shape = _const_literal_shape(value, f"{context}.attrs.value")
    is_list = isinstance(value, list)
    if is_list:
        effective_shape = (
            authored_shape if authored_shape is not None else literal_shape
        )
        if any(not isinstance(item, int) for item in effective_shape):
            _fail(
                f"{context}.attrs.value",
                "nested list literals require a fully static shape",
            )
        if effective_shape != literal_shape:
            _fail(
                f"{context}.attrs.shape",
                f"does not match the literal shape {list(literal_shape)!r}",
            )
    else:
        effective_shape = authored_shape
        if effective_shape is None:
            effective_shape = result_shape or ()

    if (
        result_shape is not None
        and (is_list or authored_shape is not None)
        and tuple(effective_shape) != result_shape
    ):
        _fail(
            f"{context}.attrs.shape",
            f"does not match result shape {list(result_shape)!r}",
        )

    values = tuple(
        _validate_const_scalar(item, dtype, f"{context}.attrs.value[{index}]")
        for index, item in enumerate(_flatten_const_literal(value, f"{context}.attrs.value"))
    )
    return ConstLiteral(
        dtype=dtype,
        shape=tuple(effective_shape),
        values=values,
        splat=not is_list,
    )


def _value_type(value: Any, path: str) -> None:
    """Validate a tensor, named record, or ragged descriptor type."""

    item = _object(value, path)
    if item.get("kind") in {"record", "ragged"}:
        kind = item["kind"]
        record = _exact_keys(
            item,
            path,
            allowed=frozenset({"kind", "fields"}),
            required=frozenset({"kind", "fields"}),
        )
        fields = record["fields"]
        if not isinstance(fields, list) or not fields:
            _fail(f"{path}.fields", "must be a non-empty array")
        names: set[str] = set()
        for index, field in enumerate(fields):
            field_path = f"{path}.fields[{index}]"
            field_item = _exact_keys(
                field,
                field_path,
                allowed=frozenset({"name", "type"}),
                required=frozenset({"name", "type"}),
            )
            name = _nonempty_string(field_item["name"], f"{field_path}.name")
            if name in names:
                _fail(f"{path}.fields", f"contains duplicate name {name!r}")
            if kind == "record":
                field_type = field_item["type"]
                if (
                    not isinstance(field_type, Mapping)
                    or field_type.get("kind", "tensor") != "tensor"
                ):
                    _fail(
                        f"{field_path}.type",
                        "record fields must be tensor value types",
                    )
            if kind == "ragged":
                field_type = field_item["type"]
                if (
                    not isinstance(field_type, dict)
                    or field_type.get("kind", "tensor") != "tensor"
                    or not isinstance(field_type.get("shape"), list)
                    or len(field_type["shape"]) != 1
                    or field_type["shape"][0] == "?"
                    or not is_ragged_carrier_dtype(field_type.get("dtype"))
                ):
                    _fail(
                        f"{field_path}.type",
                        "must be a rank-1 Ragged carrier tensor with a closed "
                        "carrier dtype",
                    )
            names.add(name)
            _value_type(field_item["type"], f"{field_path}.type")
        if kind == "ragged" and [
            field_item["name"] for field_item in fields
        ] != ["offsets", "indices"]:
            _fail(
                f"{path}.fields",
                "must contain offsets followed by indices fields",
            )
        return
    tensor = _exact_keys(
        item,
        path,
        allowed=frozenset({"dtype", "shape"}),
        required=frozenset({"dtype", "shape"}),
    )
    canonicalize_dtype(tensor["dtype"], context=f"{path}.dtype")
    _shape(tensor["shape"], f"{path}.shape")


def _region_boundary_value_type(value: Any, path: str) -> None:
    """Validate a value type that may cross a structured-region boundary.

    Ragged descriptors are first-class local operation values, but their MLIR
    tuple representation is not a loop/value-if/function ABI.  Keep the wire
    contract aligned with the current lowering contract: tensor and record
    carries remain expressible here; a Ragged descriptor must be consumed
    inside its defining region instead of being exported by it.
    """

    _value_type(value, path)
    if isinstance(value, dict) and value.get("kind") == "ragged":
        _fail(
            path,
            "Ragged descriptors are region-local and cannot cross a loop "
            "carry/result or value_if result boundary",
        )


def _result(value: Any, path: str, *, loop: bool) -> None:
    if loop:
        results = _string_array(value, path, nonempty=True, unique=True)
        for index, result in enumerate(results):
            _ssa_handle(result, f"{path}[{index}]")
        return
    if value is None:
        return
    if isinstance(value, str):
        _ssa_handle(value, path)
        return
    results = _string_array(value, path, unique=True)
    for index, result in enumerate(results):
        _ssa_handle(result, f"{path}[{index}]")


def _loop_state(value: Any, path: str) -> None:
    fields = frozenset(
        {
            "state_id",
            "init_value",
            "body_argument",
            "next_value",
            "result_value",
            "type",
        }
    )
    item = _exact_keys(value, path, allowed=fields, required=fields)
    _handle(item["state_id"], f"{path}.state_id", "s")
    for key in ("body_argument", "next_value", "result_value"):
        _ssa_handle(item[key], f"{path}.{key}")
    init = item["init_value"]
    if not is_scalar_literal_or_value_reference(init):
        _fail(
            f"{path}.init_value",
            "must be an authored scalar literal or a visible value reference",
        )
    _json_value(init, f"{path}.init_value")
    _region_boundary_value_type(item["type"], f"{path}.type")


def _trip(value: Any, path: str) -> None:
    item = _object(value, path)
    if "axis" in item:
        item = _exact_keys(
            item,
            path,
            allowed=frozenset({"axis"}),
            required=frozenset({"axis"}),
        )
        _dimension_handle(item["axis"], f"{path}.axis")
        return
    item = _exact_keys(
        item,
        path,
        allowed=frozenset({"upper_bound", "index"}),
        required=frozenset({"upper_bound", "index"}),
    )
    _value_handle(item["upper_bound"], f"{path}.upper_bound")
    _ssa_handle(item["index"], f"{path}.index")


def _ragged_operation(value: Mapping[str, Any], path: str) -> None:
    op = value.get("op")
    if op not in RAGGED_OPS:
        return
    operands = value.get("operands", [])
    result = value.get("result")
    attrs = value.get("attrs", {})
    expected_arity = RAGGED_ARITY[op]
    if not isinstance(operands, list) or len(operands) != expected_arity:
        _fail(
            f"{path}.operands",
            f"{op} requires exactly {expected_arity} operands",
        )
    for index, operand in enumerate(operands):
        _value_handle(operand, f"{path}.operands[{index}]")
    _ssa_handle(result, f"{path}.result")
    if not isinstance(attrs, dict) or attrs:
        _fail(f"{path}.attrs", f"{op} does not accept attrs")
    if "type" not in value or value.get("type") is None:
        _fail(
            f"{path}.type",
            f"{op} requires an explicit closed result type",
        )
    declared = value["type"]
    if op == "ragged":
        if not isinstance(declared, dict) or declared.get("kind") != "ragged":
            _fail(f"{path}.type", "ragged requires a ragged result type")
        _value_type(declared, f"{path}.type")
        return
    if not isinstance(declared, dict):
        _fail(f"{path}.type", f"{op} requires a tensor result type")
    _value_type(declared, f"{path}.type")
    shape = declared.get("shape")
    if not isinstance(shape, list):  # defensive; _value_type already checks it
        _fail(f"{path}.type.shape", f"{op} requires a closed result shape")
    expected_shape = [] if op == "ragged_outer" else ["?"]
    if shape != expected_shape:
        _fail(
            f"{path}.type.shape",
            f"{op} result shape must be {expected_shape!r}",
        )
    if not is_ragged_carrier_dtype(declared.get("dtype")):
        _fail(
            f"{path}.type.dtype",
            f"{op} result must use a closed Ragged carrier dtype",
        )


def _const_operation(value: Mapping[str, Any], path: str) -> None:
    """Apply the operation-level shape/arity part of the const contract."""

    if value.get("op") != "const":
        return
    if not isinstance(value.get("result"), str):
        _fail(f"{path}.result", "const requires exactly one SSA result")
    operands = value.get("operands")
    if operands is not None and (not isinstance(operands, list) or operands):
        _fail(f"{path}.operands", "const requires zero operands")
    attrs = value.get("attrs")
    result_type = value.get("type")
    validate_const_literal(
        attrs if isinstance(attrs, Mapping) else {},
        result_type=result_type if isinstance(result_type, Mapping) else None,
        context=path,
    )


def _scan(value: Any, path: str) -> None:
    item = _exact_keys(
        value,
        path,
        allowed=frozenset({"name", "from", "axis"}),
        required=frozenset({"name", "from", "axis"}),
    )
    _ssa_handle(item["name"], f"{path}.name")
    _value_handle(item["from"], f"{path}.from")
    _nonnegative_integer(item["axis"], f"{path}.axis")


def _effect_if_arm(value: Any, path: str, *, recursive: bool) -> None:
    item = _exact_keys(
        value,
        path,
        allowed=frozenset({"scope_id", "body"}),
        required=frozenset({"scope_id", "body"}),
    )
    _handle(item["scope_id"], f"{path}.scope_id", "q")
    body = item["body"]
    if not isinstance(body, list):
        _fail(f"{path}.body", "must be an array")
    if recursive:
        for index, operation in enumerate(body):
            validate_intent_op_structure(
                operation,
                path=f"{path}.body[{index}]",
                recursive=True,
            )


def _value_if_arm(value: Any, path: str, *, recursive: bool) -> None:
    item = _exact_keys(
        value,
        path,
        allowed=frozenset({"scope_id", "body", "yield_value"}),
        required=frozenset({"scope_id", "body", "yield_value"}),
    )
    _handle(item["scope_id"], f"{path}.scope_id", "q")
    _value_handle(item["yield_value"], f"{path}.yield_value")
    body = item["body"]
    if not isinstance(body, list):
        _fail(f"{path}.body", "must be an array")
    if recursive:
        for index, operation in enumerate(body):
            validate_intent_op_structure(
                operation,
                path=f"{path}.body[{index}]",
                recursive=True,
            )


def validate_intent_op_structure(
    value: Any,
    *,
    path: str = "IntentOp",
    recursive: bool = True,
) -> None:
    """Validate only the shape shared by wire and parser.

    Visibility, type equality, scope ownership and storage lifetime remain the
    full semantic parser's responsibility.
    """

    item = _object(value, path)
    op = _nonempty_string(item.get("op"), f"{path}.op")
    if op not in CANONICAL_J_OPS:
        _fail(
            f"{path}.op",
            "must be one of the closed IntentIR operations: "
            + ", ".join(sorted(CANONICAL_J_OPS)),
        )
    is_loop = op == "loop"
    is_effect_if = op == "effect_if"
    is_value_if = op == "value_if"
    structural_fields = (
        LOOP_ONLY_FIELDS if is_loop
        else EFFECT_IF_ONLY_FIELDS if is_effect_if
        else VALUE_IF_FIELDS if is_value_if
        else frozenset()
    )
    structural_required = (
        LOOP_REQUIRED_FIELDS if is_loop
        else EFFECT_IF_REQUIRED_FIELDS if is_effect_if
        else VALUE_IF_REQUIRED_FIELDS if is_value_if
        else frozenset()
    )
    # Structured regions are discriminated shapes, not ordinary operations
    # with a few extra fields.  In particular, a loop does not expose the
    # ordinary ``operands``/``attrs``/``type`` fields: its carry state and
    # explicit result/yield lists are the complete boundary contract.  Keep
    # effect_if/value_if on the same closed-region path so a caller cannot
    # serialize a field that another parser silently ignores.
    common_allowed = (
        frozenset({"id", "op"})
        if is_loop or is_effect_if or is_value_if
        else COMMON_FIELDS
    )
    allowed = common_allowed | structural_fields
    required = COMMON_REQUIRED_FIELDS | structural_required
    missing = sorted(required - set(item))
    unknown = sorted(set(item) - allowed)
    if missing:
        _fail(path, "is missing required key(s): " + ", ".join(missing))
    if unknown:
        region_only = LOOP_ONLY_FIELDS | EFFECT_IF_ONLY_FIELDS | VALUE_IF_FIELDS
        if not is_loop and not is_effect_if and not is_value_if and set(unknown) <= region_only:
            _fail(
                path,
                "has region-only key(s) on an ordinary op: " + ", ".join(unknown),
            )
        _fail(path, "has unknown key(s): " + ", ".join(unknown))

    if "id" in item:
        _nonempty_string(item["id"], f"{path}.id")
    if "result" in item:
        _result(item["result"], f"{path}.result", loop=is_loop)
    if "operands" in item:
        operands = _string_array(item["operands"], f"{path}.operands")
        for index, operand in enumerate(operands):
            _value_handle(operand, f"{path}.operands[{index}]")
    else:
        operands = []
    if "attrs" in item:
        _object(item["attrs"], f"{path}.attrs")
        _json_value(item["attrs"], f"{path}.attrs")
    attrs = item.get("attrs")
    if not is_loop and not is_effect_if and not is_value_if and not operation_operands_match_arity(
        op,
        operands,
        attrs if isinstance(attrs, Mapping) else None,
    ):
        arity = operation_operand_arity_contract(
            op,
            attrs if isinstance(attrs, Mapping) else None,
        )
        assert arity is not None
        minimum, maximum = arity
        if maximum is None:
            expectation = f"at least {minimum} operands"
        elif minimum == maximum:
            expectation = f"exactly {minimum} operands"
        else:
            expectation = f"between {minimum} and {maximum} operands"
        _fail(f"{path}.operands", f"{op} requires {expectation}")
    if "type" in item and item["type"] is not None:
        type_path = f"{path}.type"
        # Ordinary operation types are part of the same closed Layer-A value
        # contract as function inputs, loop carries, and value_if results.
        # Keeping this as an object/JSON-only check left a structural-only
        # caller able to smuggle a physical parameter such as ``p8`` through
        # an ordinary op shape; the typed parser caught it later, but shared
        # schema validation and its consumers then disagreed.  Use the same
        # recursive validator here so dtype, shape, record, and Ragged rules
        # are identical at every entry point.
        if not is_loop and not is_effect_if and not is_value_if:
            _value_type(item["type"], type_path)
        else:
            _object(item["type"], type_path)
            _json_value(item["type"], type_path)

    if not is_loop and not is_effect_if and not is_value_if:
        _ragged_operation(item, path)
        _const_operation(item, path)

    if is_effect_if:
        _value_handle(item["predicate"], f"{path}.predicate")
        _effect_if_arm(
            item["positive"], f"{path}.positive", recursive=recursive
        )
        _effect_if_arm(
            item["negative"], f"{path}.negative", recursive=recursive
        )
        return
    if is_value_if:
        _ssa_handle(item["result"], f"{path}.result")
        _region_boundary_value_type(item["type"], f"{path}.type")
        _value_handle(item["predicate"], f"{path}.predicate")
        _value_if_arm(item["positive"], f"{path}.positive", recursive=recursive)
        _value_if_arm(item["negative"], f"{path}.negative", recursive=recursive)
        return
    if not is_loop:
        return
    _handle(item["scope_id"], f"{path}.scope_id", "q")
    _trip(item["trip"], f"{path}.trip")
    carry = item["carry"]
    if not isinstance(carry, list) or not carry:
        _fail(f"{path}.carry", "must be a non-empty array")
    for index, state in enumerate(carry):
        _loop_state(state, f"{path}.carry[{index}]")
    scan = item["scan"]
    if not isinstance(scan, list):
        _fail(f"{path}.scan", "must be an array")
    for index, scan_item in enumerate(scan):
        _scan(scan_item, f"{path}.scan[{index}]")
    yields = _string_array(item["yield"], f"{path}.yield", nonempty=True, unique=True)
    for index, yielded in enumerate(yields):
        _ssa_handle(yielded, f"{path}.yield[{index}]")
    body = item["body"]
    if not isinstance(body, list) or not body:
        _fail(f"{path}.body", "must be a non-empty array")
    if recursive:
        for index, operation in enumerate(body):
            validate_intent_op_structure(
                operation, path=f"{path}.body[{index}]", recursive=True
            )


def _closed_schema(
    properties: dict[str, Any], required: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def intent_op_json_schema() -> dict[str, Any]:
    """Project the shared contract into the recursive Responses schema."""

    string = {"type": "string", "minLength": 1}
    # Operation identity is the canonical lexical node namespace.  It is
    # distinct from the v* SSA result namespace: J node_ids, boundary
    # node_ref, the allocator, and every downstream consumer resolve the same
    # n* identity.  A permissive string here admitted v*/op* ids that the
    # canonical allocator necessarily cannot own.
    node_handle = {"type": "string", "pattern": r"^n[0-9]+$"}
    value_handle = {"type": "string", "pattern": r"^(?:t|v)[0-9]+$"}
    ssa_handle = {"type": "string", "pattern": r"^v[0-9]+$"}
    dimension_handle = {"type": "string", "pattern": r"^x[0-9]+$"}
    physical_parameter_token = {
        "type": "string",
        "pattern": r"(^|[^A-Za-z0-9_])p[0-9]+($|[^A-Za-z0-9_])",
    }
    dimension_string = {
        "type": "string",
        "minLength": 1,
        "not": physical_parameter_token,
    }
    dimension = {
        "anyOf": [
            {"type": "integer", "minimum": 0},
            copy.deepcopy(dimension_string),
        ]
    }
    generic_result = {
        "anyOf": [
            copy.deepcopy(ssa_handle),
            {
                "type": "array",
                "items": copy.deepcopy(ssa_handle),
                "uniqueItems": True,
            },
            {"type": "null"},
        ]
    }
    common = {
        "id": copy.deepcopy(node_handle),
        "op": copy.deepcopy(string),
        "result": generic_result,
        "operands": {
            "type": "array",
            "items": copy.deepcopy(value_handle),
        },
        "attrs": {
            "type": "object",
            "description": (
                "Known pointwise operation attributes use the closed key fn: "
                "map_binary must carry fn, never function, and map_unary uses "
                "the closed unary spelling logical_not for boolean complement. "
                "A contraction attrs object contains only its canonical "
                "subscripts field; do not add accumulation_dtype or other "
                "implementation metadata."
            ),
        },
        "type": {"anyOf": [{"type": "object"}, {"type": "null"}]},
    }
    tensor_type = _closed_schema(
        {
            "dtype": {
                "type": "string",
                "enum": sorted(LOGICAL_DTYPES),
            },
            "shape": {"type": "array", "items": dimension},
        },
        ("dtype", "shape"),
    )
    record_field = _closed_schema(
        {
            "name": copy.deepcopy(string),
            "type": copy.deepcopy(tensor_type),
        },
        ("name", "type"),
    )
    ragged_field_type = _closed_schema(
        {
            "dtype": {
                "type": "string",
                "enum": sorted(RAGGED_CARRIER_DTYPES),
            },
            "shape": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "string", "minLength": 1, "pattern": r"^(?!\?$).+"},
                    ]
                },
            },
        },
        ("dtype", "shape"),
    )
    ragged_offsets_field = _closed_schema(
        {
            "name": {"type": "string", "const": "offsets"},
            "type": ragged_field_type,
        },
        ("name", "type"),
    )
    ragged_indices_field = _closed_schema(
        {
            "name": {"type": "string", "const": "indices"},
            "type": ragged_field_type,
        },
        ("name", "type"),
    )
    ragged_type = _closed_schema(
        {
            "kind": {"type": "string", "enum": ["ragged"]},
            "fields": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "prefixItems": [
                    ragged_offsets_field,
                    ragged_indices_field,
                ],
                "items": False,
            },
        },
        ("kind", "fields"),
    )
    record_type = _closed_schema(
        {
            "kind": {"type": "string", "enum": ["record"]},
            "fields": {
                "type": "array",
                "minItems": 1,
                "items": record_field,
            },
        },
        ("kind", "fields"),
    )
    # Loop carries and value_if results are region boundaries.  Their current
    # ABI is tensor/record only; Ragged remains available in ordinary op.type
    # and may be created/queried locally inside either region.
    region_value_type = {
        "anyOf": [copy.deepcopy(tensor_type), copy.deepcopy(record_type)]
    }
    state = _closed_schema(
        {
            "state_id": {"type": "string", "pattern": r"^s[0-9]+$"},
            "init_value": {
                "anyOf": [
                    copy.deepcopy(value_handle),
                    {
                        "type": "string",
                        "enum": sorted(SCALAR_LITERAL_TOKENS),
                    },
                    {"type": "number"},
                    {"type": "boolean"},
                ]
            },
            "body_argument": copy.deepcopy(ssa_handle),
            "next_value": copy.deepcopy(ssa_handle),
            "result_value": copy.deepcopy(ssa_handle),
            "type": region_value_type,
        },
        (
            "state_id",
            "init_value",
            "body_argument",
            "next_value",
            "result_value",
            "type",
        ),
    )
    trip = {
        "anyOf": [
            _closed_schema(
                {"axis": copy.deepcopy(dimension_handle)}, ("axis",)
            ),
            _closed_schema(
                {
                    "upper_bound": copy.deepcopy(value_handle),
                    "index": copy.deepcopy(ssa_handle),
                },
                ("upper_bound", "index"),
            ),
        ]
    }
    scan = _closed_schema(
        {
            "name": copy.deepcopy(ssa_handle),
            "from": copy.deepcopy(value_handle),
            "axis": {"type": "integer", "minimum": 0},
        },
        ("name", "from", "axis"),
    )
    const_literal = {
        "anyOf": [
            {"type": "number"},
            {"type": "boolean"},
            {
                "type": "string",
                "enum": ["-inf", "inf", "nan"],
            },
            {
                "type": "array",
                "items": {"$ref": "#/$defs/const_literal"},
            },
        ]
    }
    const_attrs = _closed_schema(
        {
            "value": {"$ref": "#/$defs/const_literal"},
            "dtype": {"type": "string", "enum": sorted(LOGICAL_DTYPES)},
            "shape": {"type": "array", "items": dimension},
        },
        ("value",),
    )
    closed_dtype = {
        "type": "string",
        "enum": sorted(LOGICAL_DTYPES),
    }
    non_boolean_integer_dtype = {
        "type": "string",
        "enum": sorted(
            dtype for dtype in LOGICAL_DTYPES
            if dtype not in {"i1", "bool"}
        ),
    }
    integer = {"type": "integer"}
    nonnegative_integer = {"type": "integer", "minimum": 0}
    positive_integer = {"type": "integer", "minimum": 1}
    integer_array = {"type": "array", "items": copy.deepcopy(integer)}
    nonnegative_integer_array = {
        "type": "array",
        "items": copy.deepcopy(nonnegative_integer),
    }
    positive_integer_array = {
        "type": "array",
        "items": copy.deepcopy(positive_integer),
    }
    dimension_array = {
        "type": "array",
        "items": copy.deepcopy(dimension),
    }
    scalar_literal = {
        "anyOf": [
            {"type": "number"},
            {"type": "boolean"},
            {"type": "string", "enum": ["-inf", "inf", "nan"]},
        ]
    }

    def attrs_object(
        properties: dict[str, Any],
        required: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return _closed_schema(properties, required)

    def op_attrs(
        op: str,
        attrs: dict[str, Any],
        *,
        required: bool,
    ) -> dict[str, Any]:
        condition: dict[str, Any] = {
            "if": {
                "properties": {"op": {"const": op}},
                "required": ["op"],
            },
            "then": {"properties": {"attrs": attrs}},
        }
        if required:
            condition["then"]["required"] = ["attrs"]
        return condition

    no_attrs = attrs_object({})
    fn_unary = attrs_object(
        {"fn": {"type": "string", "enum": sorted(MAP_UNARY_FNS)}},
        ("fn",),
    )
    fn_binary = attrs_object(
        {"fn": {"type": "string", "enum": sorted(MAP_BINARY_FNS)}},
        ("fn",),
    )
    cast_attrs = attrs_object(
        {"to_dtype": copy.deepcopy(closed_dtype)},
        ("to_dtype",),
    )
    reduce_attrs = attrs_object(
        {
            "fn": {
                "type": "string",
                "enum": ["sum", "mean", "max", "min", "prod", "any", "all"],
            },
            "axes": copy.deepcopy(integer_array),
            "assoc": {
                "type": "string",
                "enum": ["associative", "float_nondeterministic"],
            },
            "keepdims": {"type": "boolean"},
        },
        ("fn", "axes", "assoc"),
    )
    scatter_attrs = attrs_object(
        {
            "axis": copy.deepcopy(integer),
            "mode": {"type": "string", "const": "along_axis"},
            "reduce": {"type": "string", "enum": ["none", "add"]},
            "atomic": {"type": "string", "enum": ["add", "cas"]},
            "ordering": {
                "type": "string",
                "enum": sorted(ATOMIC_ORDERINGS),
            },
            "memory_scope": {
                "type": "string",
                "enum": sorted(ATOMIC_MEMORY_SCOPES),
            },
            "cas_retry": {
                "type": "string",
                "enum": sorted(ATOMIC_CAS_RETRIES),
            },
        },
        ("axis", "mode", "reduce"),
    )
    sort_attrs = attrs_object(
        {
            "descending": {"type": "boolean"},
            "axis": copy.deepcopy(integer),
            "k": copy.deepcopy(positive_integer),
            "return": {"type": "string", "const": "indices"},
        },
        ("descending", "axis"),
    )
    pool_attrs = attrs_object(
        {
            "fn": {"type": "string", "enum": ["avg", "max"]},
            "kH": copy.deepcopy(positive_integer),
            "kW": copy.deepcopy(positive_integer),
            "sH": copy.deepcopy(positive_integer),
            "sW": copy.deepcopy(positive_integer),
            "padH": copy.deepcopy(nonnegative_integer),
            "padW": copy.deepcopy(nonnegative_integer),
        },
        ("fn", "kH", "kW", "sH", "sW", "padH", "padW"),
    )
    conv_attrs = attrs_object(
        {
            "stride": copy.deepcopy(positive_integer_array),
            "padding": copy.deepcopy(nonnegative_integer_array),
            "dilation": copy.deepcopy(positive_integer_array),
            "groups": copy.deepcopy(positive_integer),
        },
        ("stride", "padding", "dilation", "groups"),
    )
    non_loop = _closed_schema(
        {
            **copy.deepcopy(common),
            "op": {
                "type": "string",
                "enum": sorted(CANONICAL_J_OPS - STRUCTURED_INTENT_OPS),
            },
        },
        # The MCP graph boundary and all canonical consumers require an
        # explicit operand array. Zero-input leaves (const/iota/rand) use an
        # empty array; omitting the field is not the same wire shape.
        ("id", "op", "operands"),
    )
    ragged_outer_result_type = _closed_schema(
        {
            "dtype": {
                "type": "string",
                "enum": sorted(RAGGED_CARRIER_DTYPES),
            },
            "shape": {"type": "array", "maxItems": 0},
        },
        ("dtype", "shape"),
    )
    ragged_member_result_type = _closed_schema(
        {
            "dtype": {
                "type": "string",
                "enum": sorted(RAGGED_CARRIER_DTYPES),
            },
            "shape": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "prefixItems": [{"const": "?"}],
                "items": False,
            },
        },
        ("dtype", "shape"),
    )
    non_loop["allOf"] = [
        {
            "if": {
                "properties": {"op": {"const": "const"}},
                "required": ["op"],
            },
            "then": {
                "required": ["result", "attrs"],
                "properties": {
                    "result": copy.deepcopy(ssa_handle),
                    "operands": {"type": "array", "maxItems": 0},
                    "attrs": copy.deepcopy(const_attrs),
                },
            },
        },
        {
            "if": {
                "properties": {"op": {"const": "ragged"}},
                "required": ["op"],
            },
            "then": {
                "required": ["type"],
                "properties": {"type": copy.deepcopy(ragged_type)},
            },
        },
        {
            "if": {
                "properties": {"op": {"const": "ragged_outer"}},
                "required": ["op"],
            },
            "then": {
                "required": ["type"],
                "properties": {"type": copy.deepcopy(ragged_outer_result_type)},
            },
        },
        {
            "if": {
                "properties": {"op": {"const": "ragged_member"}},
                "required": ["op"],
            },
            "then": {
                "required": ["type"],
                "properties": {"type": copy.deepcopy(ragged_member_result_type)},
            },
        },
        op_attrs("map_unary", fn_unary, required=True),
        op_attrs("map_binary", fn_binary, required=True),
        op_attrs("cast", cast_attrs, required=True),
        op_attrs("reduce", reduce_attrs, required=True),
        op_attrs(
            "contraction",
            attrs_object(
                {"subscripts": {"type": "string", "pattern": r"^[A-Za-z]+,[A-Za-z]+->[A-Za-z]+$"}},
                ("subscripts",),
            ),
            required=True,
        ),
        op_attrs(
            "iota",
            attrs_object(
                {
                    "n": copy.deepcopy(dimension),
                    "dtype": copy.deepcopy(non_boolean_integer_dtype),
                },
                ("n",),
            ),
            required=True,
        ),
        op_attrs(
            "reshape",
            attrs_object({"shape": copy.deepcopy(dimension_array)}, ("shape",)),
            required=True,
        ),
        op_attrs(
            "transpose",
            attrs_object({"perm": copy.deepcopy(integer_array)}, ("perm",)),
            required=True,
        ),
        op_attrs(
            "gather",
            attrs_object(
                {
                    "axis": copy.deepcopy(integer),
                    "mode": {"type": "string", "enum": ["along_axis", "take"]},
                },
                ("axis", "mode"),
            ),
            required=True,
        ),
        op_attrs("scatter", scatter_attrs, required=True),
        op_attrs(
            "concat",
            attrs_object({"axis": copy.deepcopy(integer)}, ("axis",)),
            required=True,
        ),
        op_attrs(
            "pad",
            attrs_object(
                {
                    "lo": copy.deepcopy(dimension_array),
                    "hi": copy.deepcopy(dimension_array),
                    "value": copy.deepcopy(scalar_literal),
                },
                ("lo", "hi", "value"),
            ),
            required=True,
        ),
        op_attrs(
            "tile",
            attrs_object({"reps": copy.deepcopy(nonnegative_integer_array)}, ("reps",)),
            required=True,
        ),
        op_attrs(
            "flip",
            attrs_object({"axes": copy.deepcopy(integer_array)}, ("axes",)),
            required=True,
        ),
        op_attrs(
            "triu",
            attrs_object({"k": copy.deepcopy(integer)}, ("k",)),
            required=True,
        ),
        op_attrs("select", no_attrs, required=False),
        op_attrs("record", attrs_object({"fields": {"type": "array", "items": {"type": "string", "minLength": 1}}}, ("fields",)), required=True),
        op_attrs("record_get", attrs_object({"field": {"type": "string", "minLength": 1}}, ("field",)), required=True),
        op_attrs("scan", attrs_object({"fn": {"type": "string", "enum": ["sum", "max", "min"]}, "axis": copy.deepcopy(integer)}, ("fn", "axis")), required=True),
        op_attrs("arg_reduce", attrs_object({"fn": {"type": "string", "enum": ["argmax", "argmin"]}, "axis": copy.deepcopy(integer)}, ("fn", "axis")), required=True),
        op_attrs("sort", sort_attrs, required=True),
        op_attrs("pool", pool_attrs, required=True),
        op_attrs("upsample", attrs_object({"mode": {"type": "string", "enum": ["nearest", "bicubic"]}, "scale": copy.deepcopy(positive_integer_array)}, ("mode", "scale")), required=True),
        op_attrs("conv", conv_attrs, required=True),
        op_attrs("rand", attrs_object({"seed": copy.deepcopy(integer), "shape": copy.deepcopy(dimension_array), "dtype": copy.deepcopy(closed_dtype)}, ("seed", "shape")), required=True),
        op_attrs("nonzero", no_attrs, required=False),
        op_attrs("masked_select", no_attrs, required=False),
        op_attrs("masked_scatter", no_attrs, required=False),
        op_attrs("polar", no_attrs, required=False),
        op_attrs("ragged", no_attrs, required=False),
        op_attrs("ragged_outer", no_attrs, required=False),
        op_attrs("ragged_member", no_attrs, required=False),
    ]
    loop = _closed_schema(
        {
            "id": copy.deepcopy(node_handle),
            "op": {"type": "string", "enum": ["loop"]},
            "result": {
                "type": "array",
                "items": copy.deepcopy(ssa_handle),
                "minItems": 1,
                "uniqueItems": True,
            },
            "scope_id": {"type": "string", "pattern": r"^q[0-9]+$"},
            "carry": {
                "type": "array",
                "items": state,
                "minItems": 1,
            },
            "trip": trip,
            "scan": {"type": "array", "items": scan},
            "body": {
                "type": "array",
                "items": {"$ref": "#/$defs/intent_op"},
                "minItems": 1,
            },
            "yield": {
                "type": "array",
                "items": copy.deepcopy(ssa_handle),
                "minItems": 1,
                "uniqueItems": True,
            },
        },
        (
            "id",
            "op",
            "result",
            "scope_id",
            "carry",
            "trip",
            "scan",
            "body",
            "yield",
        ),
    )
    arm = _closed_schema(
        {
            "scope_id": {"type": "string", "pattern": r"^q[0-9]+$"},
            "body": {
                "type": "array",
                "items": {"$ref": "#/$defs/intent_op"},
            },
        },
        ("scope_id", "body"),
    )
    effect_if = _closed_schema(
        {
            "id": copy.deepcopy(node_handle),
            "op": {"type": "string", "enum": ["effect_if"]},
            "predicate": copy.deepcopy(value_handle),
            "positive": copy.deepcopy(arm),
            "negative": copy.deepcopy(arm),
        },
        ("id", "op", "predicate", "positive", "negative"),
    )
    value_arm = _closed_schema(
        {
            "scope_id": {"type": "string", "pattern": r"^q[0-9]+$"},
            "body": {
                "type": "array",
                "items": {"$ref": "#/$defs/intent_op"},
            },
            "yield_value": copy.deepcopy(value_handle),
        },
        ("scope_id", "body", "yield_value"),
    )
    value_if = _closed_schema(
        {
            "id": copy.deepcopy(node_handle),
            "op": {"type": "string", "enum": ["value_if"]},
            "result": copy.deepcopy(ssa_handle),
            "type": copy.deepcopy(region_value_type),
            "predicate": copy.deepcopy(value_handle),
            "positive": copy.deepcopy(value_arm),
            "negative": copy.deepcopy(value_arm),
        },
        ("id", "op", "result", "type", "predicate", "positive", "negative"),
    )
    return {
        "$defs": {"const_literal": const_literal},
        "anyOf": [non_loop, loop, effect_if, value_if],
    }


def render_intent_op_wire_grammar() -> str:
    common = ", ".join(sorted(COMMON_FIELDS))
    loop_only = ", ".join(sorted(LOOP_ONLY_FIELDS))
    required = ", ".join(sorted(LOOP_REQUIRED_FIELDS))
    return (
        "IntentOp is one global discriminated shape. Ordinary operations expose "
        f"common fields: {common}. If and only if op is exactly 'loop', it "
        "uses a separate region shape with only optional id, op, and "
        f"loop-only fields: {loop_only}; all of these loop fields are "
        f"required: {required}. Every non-loop op must omit every loop-only "
        "field, and a loop must omit ordinary operands, attrs, and type; null "
        "or an empty placeholder does not count as omission. A "
        "loop has a non-empty result list, carry list, body and yield list. "
        "A static loop trip uses {axis}; a dynamic trip uses "
        "{upper_bound,index} and binds its scalar induction value in the body. "
        "Logical boundary tensor references use t*, operation and region SSA "
        "values use v*, and symbolic trip dimensions use x*; physical p* "
        "parameters never appear in Layer-A value positions. "
        "The dynamic upper_bound and induction tensor must be rank-0, "
        "non-boolean integer values (i1/bool is not a loop bound). "
        "If and only if op is exactly 'effect_if', it has exactly id(optional), "
        "op, predicate, positive and negative. Both arms are explicit closed "
        "{scope_id,body} objects and either body may be empty. The conditional "
        "itself has no result, operands, attrs, type or yield; arm-local values "
        "may feed arm effects but never escape to the parent continuation. "
        "If and only if op is exactly 'value_if', it has exactly id(optional), "
        "op, one string result, one closed type, predicate, positive and "
        "negative. Both arms are explicit closed {scope_id,body,yield_value} "
        "objects; either body may be empty, but each yield_value is required. "
        "value_if has no operands or attrs and is distinct from select and "
        "effect_if. Ragged descriptors are valid local ordinary operation values "
        "but ragged, ragged_outer, and ragged_member each require an explicit "
        "closed result type; ragged descriptors cannot be carried through "
        "loop/result or value_if result boundaries. Consume them inside the "
        "defining region. "
        "This is a structural rule, not an operation catalog."
    )


def operation_attr_contract(op: str) -> dict[str, Any]:
    """Return the schema-derived attrs contract for one ordinary operation.

    The JSON schema is the single wire authority.  TypeCheck and schema/MCP
    projections must not each maintain a hand-written set of allowed or
    required operation attributes.  This helper exposes only the small
    structural projection needed by those consumers; operation-specific
    semantic validation remains in their respective layers.
    """

    if not isinstance(op, str) or not op:
        raise TypeError("operation name must be a non-empty string")
    schema = intent_op_json_schema()
    saw_operation = False
    for variant in schema.get("anyOf", ()):
        if not isinstance(variant, Mapping):
            continue
        variant_operation_names = (
            variant.get("properties", {})
            .get("op", {})
            .get("enum", ())
            if isinstance(variant.get("properties"), Mapping)
            else ()
        )
        if op in variant_operation_names:
            saw_operation = True
            # loop/effect_if/value_if are complete structured operation
            # branches.  They intentionally have no attrs object; represent
            # that fact as the same closed empty attrs contract instead of
            # making consumers special-case a missing operation contract.
            if op in STRUCTURED_INTENT_OPS:
                return {
                    "allowed": frozenset(),
                    "required": frozenset(),
                    "properties": {},
                }
        for condition in variant.get("allOf", ()):
            if not isinstance(condition, Mapping):
                continue
            discriminator = (
                condition.get("if", {})
                .get("properties", {})
                .get("op", {})
                if isinstance(condition.get("if"), Mapping)
                else {}
            )
            if not isinstance(discriminator, Mapping) or discriminator.get("const") != op:
                continue
            attrs = (
                condition.get("then", {})
                .get("properties", {})
                .get("attrs")
                if isinstance(condition.get("then"), Mapping)
                else None
            )
            if not isinstance(attrs, Mapping):
                return {
                    "allowed": frozenset(),
                    "required": frozenset(),
                    "properties": {},
                }
            properties = attrs.get("properties", {})
            if not isinstance(properties, Mapping):
                properties = {}
            required = attrs.get("required", ())
            if not isinstance(required, list):
                required = []
            return {
                "allowed": frozenset(str(name) for name in properties),
                "required": frozenset(str(name) for name in required),
                "properties": copy.deepcopy(dict(properties)),
            }
    if saw_operation:
        return {
            "allowed": frozenset(),
            "required": frozenset(),
            "properties": {},
        }
    raise KeyError(f"operation has no schema attrs contract: {op}")


def operation_attr_enum(op: str, attr: str) -> frozenset[Any]:
    """Return one schema-owned enum for an operation attribute.

    Enum consumers such as TypeCheck must use the same leaf that is exposed by
    schema lookup and the provider-facing operation profile.  Semantic checks
    may still impose relationships between attrs, but they must not duplicate
    the closed spelling set here.
    """

    if not isinstance(attr, str) or not attr:
        raise TypeError("operation attribute name must be non-empty")
    contract = operation_attr_contract(op)
    property_schema = contract["properties"].get(attr)
    if not isinstance(property_schema, Mapping):
        raise KeyError(f"operation attribute has no schema contract: {op}.{attr}")
    if "const" in property_schema:
        return frozenset({property_schema["const"]})
    values = property_schema.get("enum")
    if not isinstance(values, list) or not values:
        raise ValueError(f"operation attribute has no closed enum: {op}.{attr}")
    return frozenset(values)


__all__ = [
    "COMMON_FIELDS",
    "COMMON_REQUIRED_FIELDS",
    "ConstLiteral",
    "EFFECT_IF_ONLY_FIELDS",
    "EFFECT_IF_REQUIRED_FIELDS",
    "IntentOpContractError",
    "LOOP_ONLY_FIELDS",
    "LOOP_REQUIRED_FIELDS",
    "MAP_BINARY_ARITHMETIC_FNS",
    "MAP_BINARY_COMPARISON_FNS",
    "MAP_BINARY_FLOATING_FNS",
    "MAP_BINARY_FNS",
    "MAP_BINARY_LOGICAL_FNS",
    "MAP_BINARY_MATCHING_NUMERIC_FNS",
    "MAP_BINARY_PREDICATE_FNS",
    "MAP_BINARY_SHIFT_FNS",
    "MAP_UNARY_FLOAT_FNS",
    "MAP_UNARY_FNS",
    "MAP_UNARY_NUMERIC_FNS",
    "MAP_UNARY_PREDICATE_FNS",
    "CANONICAL_J_OPS",
    "INTERNAL_INTENT_OPS",
    "KNOWN_INTENT_OPS",
    "RAGGED_ARITY",
    "RAGGED_CARRIER_DTYPES",
    "RAGGED_OPS",
    "STRUCTURED_INTENT_OPS",
    "VALUE_IF_FIELDS",
    "VALUE_IF_STRUCTURAL_FIELDS",
    "CANONICAL_DTYPES",
    "FLOATING_DTYPES",
    "INTEGER_DTYPES",
    "LOGICAL_DTYPES",
    "canonicalize_dtype",
    "const_literal_shape",
    "SCALAR_LITERAL_TOKENS",
    "is_closed_boolean_dtype",
    "is_closed_dtype",
    "is_closed_floating_dtype",
    "is_consumer_reduction_identity",
    "is_closed_integer_dtype",
    "is_closed_numeric_dtype",
    "VALUE_IF_REQUIRED_FIELDS",
    "intent_op_json_schema",
    "operation_attr_contract",
    "operation_attr_enum",
    "is_non_boolean_integer_dtype",
    "is_scalar_literal",
    "is_scalar_literal_or_value_reference",
    "is_ragged_carrier_dtype",
    "map_binary_result_dtype",
    "map_unary_result_dtype",
    "operation_operand_arity_contract",
    "operation_operands_match_arity",
    "promote_dtype",
    "render_intent_op_wire_grammar",
    "validate_const_literal",
    "validate_intent_op_structure",
]
