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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


MAP_UNARY_FNS = frozenset(
    {
        "identity", "exp", "log", "log2", "sqrt", "rsqrt", "neg", "recip",
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
# ``bool`` and ``index`` remain accepted wire aliases for compatibility, but
# all internal consumers can canonicalize them to ``i1`` and ``i64``.
CANONICAL_DTYPES = frozenset(
    {
        "f16", "bf16", "f32", "f64",
        "i1", "i8", "i16", "i32", "i64",
        "u8", "u16", "u32",
    }
)
DTYPE_ALIASES = {"bool": "i1", "index": "i64"}
LOGICAL_DTYPES = CANONICAL_DTYPES | frozenset(DTYPE_ALIASES)
FLOATING_DTYPES = frozenset({"f16", "bf16", "f32", "f64"})
INTEGER_DTYPES = frozenset(
    {"i1", "i8", "i16", "i32", "i64", "u8", "u16", "u32"}
)


def canonicalize_dtype(value: object, *, context: str = "dtype") -> str:
    """Validate and canonicalize one logical J dtype spelling."""

    if not isinstance(value, str):
        _fail(context, "must be a closed logical dtype string")
    dtype = DTYPE_ALIASES.get(value, value)
    if dtype not in CANONICAL_DTYPES:
        _fail(
            context,
            "must be one of " + ", ".join(sorted(LOGICAL_DTYPES)),
        )
    return dtype


def is_closed_dtype(value: object) -> bool:
    return isinstance(value, str) and value in LOGICAL_DTYPES


def is_closed_integer_dtype(value: object, *, include_boolean: bool = True) -> bool:
    if not isinstance(value, str) or value not in LOGICAL_DTYPES:
        return False
    canonical = DTYPE_ALIASES.get(value, value)
    return canonical in INTEGER_DTYPES and (
        include_boolean or canonical != "i1"
    )


def is_closed_floating_dtype(value: object) -> bool:
    return (
        isinstance(value, str)
        and DTYPE_ALIASES.get(value, value) in FLOATING_DTYPES
    )


def is_closed_boolean_dtype(value: object) -> bool:
    return (
        isinstance(value, str)
        and DTYPE_ALIASES.get(value, value) == "i1"
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

# Keep the Layer-A opcode vocabulary in one place.  ``const``, ``record`` and
# ``record_get`` are J-level value operations even though they do not each have
# a one-to-one Intent dialect op.  ``yield`` is deliberately absent: it is the
# MLIR loop terminator and is represented by the JSON loop ``yield`` list, not
# as an operation in a J body.  The TableGen operations and the J-only value
# operations therefore share this contract without making the MLIR terminator
# look like a liftable semantic node.
KNOWN_INTENT_OPS = frozenset(
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
        "atomic_scatter",
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
STRUCTURED_INTENT_OPS = frozenset({"loop", "effect_if", "value_if"})


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
        _fail(path, f"must be a {names} handle")
    return result


def _value_handle(value: Any, path: str) -> str:
    return _handle(value, path, "tv")


def _ssa_handle(value: Any, path: str) -> str:
    return _handle(value, path, "v")


SCALAR_LITERAL_TOKENS = frozenset(
    {"NEG_INF", "POS_INF", "ZERO", "ONE", "zeros", "ones", "false", "true"}
)
_NUMERIC_LITERAL_STRING = re.compile(
    r"^[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?$"
)
_PHYSICAL_PARAMETER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])p[0-9]+(?![A-Za-z0-9_])"
)


def is_scalar_literal(value: Any) -> bool:
    """Return whether ``value`` is one of the closed scalar literal forms."""

    if isinstance(value, bool) or isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if not isinstance(value, str):
        return False
    if value in SCALAR_LITERAL_TOKENS:
        return True
    if _NUMERIC_LITERAL_STRING.fullmatch(value) is None:
        return False
    try:
        return math.isfinite(float(value))
    except ValueError:  # pragma: no cover - guarded by the closed grammar
        return False


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
    if op not in KNOWN_INTENT_OPS:
        _fail(
            f"{path}.op",
            "must be one of the closed IntentIR operations: "
            + ", ".join(sorted(KNOWN_INTENT_OPS)),
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
    if "attrs" in item:
        _object(item["attrs"], f"{path}.attrs")
        _json_value(item["attrs"], f"{path}.attrs")
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
        "id": copy.deepcopy(string),
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
    value_type = {
        "anyOf": [
            copy.deepcopy(tensor_type),
            copy.deepcopy(record_type),
            ragged_type,
        ]
    }
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
                    {
                        "type": "string",
                        "pattern": _NUMERIC_LITERAL_STRING.pattern,
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
    non_loop = _closed_schema(
        {
            **copy.deepcopy(common),
            "op": {
                "type": "string",
                "enum": sorted(KNOWN_INTENT_OPS - STRUCTURED_INTENT_OPS),
            },
        },
        ("op",),
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
    ]
    loop = _closed_schema(
        {
            "id": copy.deepcopy(string),
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
            "id": copy.deepcopy(string),
            "op": {"type": "string", "enum": ["effect_if"]},
            "predicate": copy.deepcopy(value_handle),
            "positive": copy.deepcopy(arm),
            "negative": copy.deepcopy(arm),
        },
        ("op", "predicate", "positive", "negative"),
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
            "id": copy.deepcopy(string),
            "op": {"type": "string", "enum": ["value_if"]},
            "result": copy.deepcopy(ssa_handle),
            "type": copy.deepcopy(region_value_type),
            "predicate": copy.deepcopy(value_handle),
            "positive": copy.deepcopy(value_arm),
            "negative": copy.deepcopy(value_arm),
        },
        ("op", "result", "type", "predicate", "positive", "negative"),
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
    "KNOWN_INTENT_OPS",
    "RAGGED_ARITY",
    "RAGGED_CARRIER_DTYPES",
    "RAGGED_OPS",
    "STRUCTURED_INTENT_OPS",
    "VALUE_IF_FIELDS",
    "VALUE_IF_STRUCTURAL_FIELDS",
    "CANONICAL_DTYPES",
    "DTYPE_ALIASES",
    "FLOATING_DTYPES",
    "INTEGER_DTYPES",
    "LOGICAL_DTYPES",
    "canonicalize_dtype",
    "const_literal_shape",
    "SCALAR_LITERAL_TOKENS",
    "is_closed_boolean_dtype",
    "is_closed_dtype",
    "is_closed_floating_dtype",
    "is_closed_integer_dtype",
    "is_closed_numeric_dtype",
    "VALUE_IF_REQUIRED_FIELDS",
    "intent_op_json_schema",
    "is_non_boolean_integer_dtype",
    "is_scalar_literal",
    "is_scalar_literal_or_value_reference",
    "is_ragged_carrier_dtype",
    "map_binary_result_dtype",
    "map_unary_result_dtype",
    "promote_dtype",
    "render_intent_op_wire_grammar",
    "validate_const_literal",
    "validate_intent_op_structure",
]
