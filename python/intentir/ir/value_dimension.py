"""Static, materialization, and runtime checks for J-owned scalar equalities.

The relation itself is authored in J.  The helpers in this module never infer
one from tensor shapes, Harness values, C, or pi; they only validate and
project an already declared relation for J consumers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import numpy as np

from .intent_op_contract import (
    RAGGED_OPS,
    is_closed_integer_dtype,
    is_ragged_carrier_dtype,
)
from .dimension_relation import declared_dimension_handles
from .model import (
    intent_op_producers,
    walk_intent_op_scopes,
)


_INTEGRAL_DTYPES = frozenset(
    {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "index"}
)
_RUNTIME_RAGGED_DTYPES = frozenset(
    dtype for dtype in ("i8", "i16", "i32", "i64", "u8", "u16", "u32")
)
_RAGGED_NUMPY_DTYPES = {
    "i1": np.bool_,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
}


class ValueDimensionRelationError(ValueError):
    """A declared relation or its public valuation is not closed."""


class RuntimeShapeResolutionError(ValueError):
    """A finite J invocation cannot be assigned a closed runtime shape."""


class RaggedRuntimeError(RuntimeShapeResolutionError):
    """A runtime offsets/indices carrier violates the Ragged contract."""


def _dtype_key(value: str) -> str:
    aliases = {
        "bool": "i1",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
        "ui8": "u8",
        "ui16": "u16",
        "ui32": "u32",
    }
    spelling = value.casefold().removeprefix("torch.")
    return aliases.get(spelling, spelling)


def validate_value_dimension_relations(j: Any) -> None:
    """Validate only model-authored relation declarations, without valuations."""

    inputs = {item.name: item for item in j.intent.inputs}
    dimensions = declared_dimension_handles(j)
    seen_relations: set[tuple[str, str, str]] = set()
    for index, relation in enumerate(j.value_dimension_relations):
        path = f"j.value_dimension_relations[{index}]"
        if relation.relation != "eq":
            raise ValueDimensionRelationError(
                f"{path}.relation must be the closed equality 'eq'"
            )
        value = inputs.get(relation.value_ref)
        if (
            value is None
            or re.fullmatch(r"t[0-9]+", relation.value_ref) is None
            or value.shape
            or not is_closed_integer_dtype(
                _dtype_key(value.dtype), include_boolean=False
            )
        ):
            raise ValueDimensionRelationError(
                f"{path}.value_ref must name an integral rank-0 J input t*"
            )
        if (
            relation.dimension_ref not in dimensions
            or re.fullmatch(r"x[0-9]+", relation.dimension_ref) is None
        ):
            raise ValueDimensionRelationError(
                f"{path}.dimension_ref must name a declared symbolic x* dimension"
            )
        key = (relation.value_ref, relation.dimension_ref, relation.relation)
        if key in seen_relations:
            raise ValueDimensionRelationError(
                f"{path} duplicates an earlier value/dimension equality"
            )
        seen_relations.add(key)


def _closed_integer(value: Any, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueDimensionRelationError(f"{context} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    raise ValueDimensionRelationError(f"{context} must be an integer")


def _dtype_bounds(dtype: str) -> tuple[int, int]:
    key = _dtype_key(dtype)
    if key == "index":
        return -(1 << 63), (1 << 63) - 1
    if key not in _INTEGRAL_DTYPES:
        raise ValueDimensionRelationError(
            f"logical scalar dtype {dtype!r} is not integral"
        )
    bits = int(key[1:])
    if key[0] == "u":
        return 0, (1 << bits) - 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _check_representable(value: int, dtype: str, *, context: str) -> None:
    lower, upper = _dtype_bounds(dtype)
    if not lower <= value <= upper:
        raise ValueDimensionRelationError(
            f"{context}={value} is not representable by logical scalar dtype "
            f"{dtype!r}"
        )


def resolve_value_dimension_equalities(
    j: Any,
    dimensions: Mapping[str, int],
) -> tuple[tuple[int, int], ...]:
    """Resolve authored equalities to stable ``(input arg index, value)`` pairs.

    This is the only projection needed by production materializers.  Multiple
    dimensions may equal the same scalar input; equal valuations collapse to
    one pair, while conflicting valuations fail closed.
    """

    validate_value_dimension_relations(j)
    input_indices = {item.name: index for index, item in enumerate(j.intent.inputs)}
    input_types = {item.name: item.dtype for item in j.intent.inputs}
    resolved: dict[int, int] = {}
    for relation in j.value_dimension_relations:
        if relation.dimension_ref not in dimensions:
            raise ValueDimensionRelationError(
                "value/dimension equality lacks dimension valuation "
                f"{relation.dimension_ref!r}"
            )
        value = _closed_integer(
            dimensions[relation.dimension_ref],
            context=f"dimension {relation.dimension_ref!r}",
        )
        _check_representable(
            value,
            input_types[relation.value_ref],
            context=f"dimension {relation.dimension_ref}",
        )
        arg_index = input_indices[relation.value_ref]
        previous = resolved.setdefault(arg_index, value)
        if previous != value:
            raise ValueDimensionRelationError(
                f"logical value {relation.value_ref!r} is equated to conflicting "
                f"dimension valuations {previous} and {value}"
            )
    return tuple(sorted(resolved.items()))


def _rank_zero_integral(value: Any, *, ref: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueDimensionRelationError(
            f"logical value {ref!r} must be one integral rank-0 invocation value"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.shape != () or not is_closed_integer_dtype(
            _dtype_key(str(value.dtype)), include_boolean=False
        ):
            raise ValueDimensionRelationError(
                f"logical value {ref!r} must be one integral rank-0 invocation value"
            )
        return int(value.item())

    # Torch tensors, including CUDA tensors, cannot in general be passed to
    # np.asarray.  Use their typed scalar API without importing torch or
    # accepting arbitrary ``item()``-convertible objects.
    value_type = type(value)
    if value_type.__module__.startswith("torch") and value_type.__name__ == "Tensor":
        try:
            rank = int(value.dim())
            dtype = _dtype_key(str(value.dtype))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueDimensionRelationError(
                f"logical value {ref!r} must be one integral rank-0 invocation value"
            ) from exc
        if rank != 0 or not is_closed_integer_dtype(
            dtype, include_boolean=False
        ):
            raise ValueDimensionRelationError(
                f"logical value {ref!r} must be one integral rank-0 invocation value"
            )
        return int(value.item())

    raise ValueDimensionRelationError(
        f"logical value {ref!r} must be one integral rank-0 invocation value"
    )


def validate_value_dimension_instantiation(
    j: Any,
    dimensions: Mapping[str, int],
    logical_values: Mapping[str, Any],
) -> None:
    """Check actual invocation values against all authored equality relations."""

    resolved = dict(resolve_value_dimension_equalities(j, dimensions))
    input_indices = {item.name: index for index, item in enumerate(j.intent.inputs)}
    input_types = {item.name: item.dtype for item in j.intent.inputs}
    for relation in j.value_dimension_relations:
        if relation.value_ref not in logical_values:
            raise ValueDimensionRelationError(
                f"value/dimension equality lacks logical value "
                f"{relation.value_ref!r}"
            )
        value = _rank_zero_integral(
            logical_values[relation.value_ref], ref=relation.value_ref
        )
        _check_representable(
            value,
            input_types[relation.value_ref],
            context=f"logical value {relation.value_ref}",
        )
        dimension = resolved[input_indices[relation.value_ref]]
        if value != dimension:
            raise ValueDimensionRelationError(
                "J value/dimension equality mismatch: "
                f"{relation.value_ref}={value}, "
                f"{relation.dimension_ref}={dimension}"
            )


def _runtime_tensor(value: Any, *, context: str) -> Any:
    """Return a tensor-like runtime value without importing a target runtime.

    CPU callers pass NumPy arrays and CUDA callers pass Torch tensors, including
    CUDA-resident tensors.  Converting the latter through ``np.asarray`` would
    either fail or implicitly move data across devices, so the small adapter is
    intentionally based on the public shape/dtype/index/item protocol only.
    Plain array-like values are normalized to NumPy for the CPU-side workload
    path.
    """

    if isinstance(value, np.ndarray):
        return value
    value_type = type(value)
    if value_type.__module__.startswith("torch") and value_type.__name__ == "Tensor":
        return value
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise RaggedRuntimeError(f"{context} must be a runtime tensor") from exc


def _runtime_shape(value: Any, *, context: str) -> tuple[int, ...]:
    tensor = _runtime_tensor(value, context=context)
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise RaggedRuntimeError(f"{context} has no runtime shape")
    try:
        result = tuple(int(extent) for extent in shape)
    except (TypeError, ValueError) as exc:
        raise RaggedRuntimeError(f"{context} has an invalid runtime shape") from exc
    if any(extent < 0 for extent in result):
        raise RaggedRuntimeError(f"{context} has a negative runtime extent")
    return result


def _runtime_dtype_kind(value: Any, *, context: str) -> str:
    key = _runtime_dtype_key(value, context=context)
    if key in {"i1", "i8", "i16", "i32", "i64"}:
        return "i"
    if key in {"u8", "u16", "u32", "u64"}:
        return "u"
    return ""


def _runtime_dtype_key(value: Any, *, context: str) -> str:
    tensor = _runtime_tensor(value, context=context)
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        raise RaggedRuntimeError(f"{context} has no runtime dtype")
    spelling = getattr(dtype, "name", None)
    if not isinstance(spelling, str) or not spelling:
        spelling = str(dtype)
    key = _dtype_key(spelling)
    # ``index`` has no NumPy storage dtype; every host/device invocation uses
    # the signed i64 carrier at this boundary.
    return "i64" if key == "index" else key


def _validate_runtime_ragged_dtype(
    value: Any,
    *,
    expected: str | None,
    context: str,
) -> str:
    actual = _runtime_dtype_key(value, context=context)
    if actual not in _RUNTIME_RAGGED_DTYPES:
        raise RaggedRuntimeError(
            f"{context} must use one of the closed Ragged carrier dtypes "
            "i8/i16/i32/i64/u8/u16/u32"
        )
    if expected is not None:
        expected_key = _dtype_key(expected)
        if expected_key == "index":
            expected_key = "i64"
        if not is_ragged_carrier_dtype(expected):
            raise RaggedRuntimeError(
                f"{context} expected dtype {expected!r} is not a closed "
                "Ragged carrier dtype"
            )
        if actual != expected_key:
            raise RaggedRuntimeError(
                f"{context} dtype {actual!r} differs from J dtype "
                f"{expected!r}"
            )
    return actual


def _runtime_numel(value: Any, shape: tuple[int, ...]) -> int:
    tensor = _runtime_tensor(value, context="runtime tensor")
    numel = getattr(tensor, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except (TypeError, ValueError) as exc:
            raise RaggedRuntimeError("runtime tensor has an invalid element count") from exc
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _runtime_integer_at(value: Any, index: int | None, *, context: str) -> int:
    tensor = _runtime_tensor(value, context=context)
    try:
        item = tensor.item() if index is None else tensor[index]
        if hasattr(item, "item"):
            item = item.item()
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        raise RaggedRuntimeError(f"{context} cannot be read as an integer") from exc
    try:
        return _closed_integer(item, context=context)
    except ValueDimensionRelationError as exc:
        raise RaggedRuntimeError(str(exc)) from exc


def _runtime_any(value: Any, *, context: str) -> bool:
    try:
        reduced = value.any()
        if hasattr(reduced, "item"):
            reduced = reduced.item()
        return bool(reduced)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise RaggedRuntimeError(f"{context} cannot be checked") from exc


def validate_ragged_runtime_carrier(
    offsets: Any,
    indices: Any,
    *,
    context: str = "ragged",
    offsets_dtype: str | None = None,
    indices_dtype: str | None = None,
) -> tuple[int, int]:
    """Validate one finite offsets/indices carrier shared by all backends.

    The returned pair is ``(outer_group_count, member_index_count)``.  This
    function is deliberately value-aware only at the runtime boundary: the
    descriptor shape and dtype remain J-owned, while these monotonicity and
    terminal-offset checks protect every host/reference/device consumer from
    interpreting malformed storage differently.
    """

    offsets = _runtime_tensor(offsets, context=f"{context}.offsets")
    indices = _runtime_tensor(indices, context=f"{context}.indices")
    offsets_shape = _runtime_shape(offsets, context=f"{context}.offsets")
    indices_shape = _runtime_shape(indices, context=f"{context}.indices")
    if len(offsets_shape) != 1 or len(indices_shape) != 1:
        raise RaggedRuntimeError(
            f"{context} fields must both be rank-1 tensors"
        )
    _validate_runtime_ragged_dtype(
        offsets,
        expected=offsets_dtype,
        context=f"{context}.offsets",
    )
    _validate_runtime_ragged_dtype(
        indices,
        expected=indices_dtype,
        context=f"{context}.indices",
    )
    if offsets_shape[0] == 0:
        raise RaggedRuntimeError(
            f"{context}.offsets must contain at least one value"
        )
    if _runtime_integer_at(offsets, 0, context=f"{context}.offsets[0]") != 0:
        raise RaggedRuntimeError(f"{context}.offsets must start at zero")
    if _runtime_any(
        offsets[1:] < offsets[:-1],
        context=f"{context}.offsets monotonicity",
    ):
        raise RaggedRuntimeError(f"{context}.offsets must be non-decreasing")
    end = _runtime_integer_at(
        offsets, -1, context=f"{context}.offsets[-1]"
    )
    index_count = _runtime_numel(indices, indices_shape)
    if end != index_count:
        raise RaggedRuntimeError(
            f"{context}.offsets[-1] must equal the number of indices"
        )
    if _runtime_any(offsets < 0, context=f"{context}.offsets non-negativity"):
        raise RaggedRuntimeError(f"{context}.offsets must not contain negative values")
    return offsets_shape[0] - 1, index_count


def ragged_outer_count(
    offsets: Any,
    indices: Any,
    *,
    context: str = "ragged",
) -> int:
    """Validate a carrier and return its number of outer groups."""

    outer_count, _index_count = validate_ragged_runtime_carrier(
        offsets, indices, context=context
    )
    return outer_count


def ragged_member_length(
    offsets: Any,
    indices: Any,
    selector: Any,
    *,
    context: str = "ragged_member",
) -> int:
    """Validate a carrier and return one selected member vector's length."""

    outer_count, _index_count = validate_ragged_runtime_carrier(
        offsets, indices, context=context
    )
    selector_shape = _runtime_shape(selector, context=f"{context}.selector")
    if selector_shape != ():
        raise RaggedRuntimeError(
            f"{context}.selector must be a rank-0 closed Ragged carrier tensor"
        )
    _validate_runtime_ragged_dtype(
        selector,
        expected=None,
        context=f"{context}.selector",
    )
    outer = _runtime_integer_at(
        selector, None, context=f"{context}.selector"
    )
    if outer < 0 or outer >= outer_count:
        raise RaggedRuntimeError(
            f"{context}.selector {outer} is outside the offsets tensor"
        )
    offsets = _runtime_tensor(offsets, context=f"{context}.offsets")
    start = _runtime_integer_at(
        offsets, outer, context=f"{context}.offsets[{outer}]"
    )
    end = _runtime_integer_at(
        offsets, outer + 1, context=f"{context}.offsets[{outer + 1}]"
    )
    length = end - start
    if length < 0:  # Defensive; the carrier validator already checks this.
        raise RaggedRuntimeError(f"{context}.offsets must be non-decreasing")
    return length


def validate_ragged_runtime_invocation(
    fn: Any,
    inputs: Sequence[Any] | Mapping[str, Any],
    *,
    dims: Mapping[str, int] | None = None,
) -> None:
    """Validate every Ragged carrier reachable in one finite J invocation.

    The function is intentionally backend-neutral.  It accepts either the
    ordered ``IntentFunction.inputs`` sequence used by CPU/RVV or a logical-name
    mapping used by Gate2.  A carrier must be a direct invocation input or a
    fully materializable ``const`` producer; computed carriers are rejected
    rather than being interpreted differently by different backends.  Both
    arms of a conditional and all nested regions are walked, so an inactive
    branch cannot bypass the common runtime contract.
    """

    def result_name(operation: Any) -> str | None:
        value = getattr(operation, "result", None)
        return value if isinstance(value, str) else None

    def operation_attr(operation: Any, name: str, default: Any = None) -> Any:
        attrs = getattr(operation, "attrs", {})
        return attrs.get(name, default) if isinstance(attrs, Mapping) else default

    body = getattr(fn, "body", None)
    if not isinstance(body, Sequence):
        raise RaggedRuntimeError("runtime Ragged validation requires a J body")
    fn_inputs = tuple(getattr(fn, "inputs", ()) or ())
    scoped_operations = tuple(
        walk_intent_op_scopes(
            body,
            visible=(
                tensor.name
                for tensor in fn_inputs
                if isinstance(getattr(tensor, "name", None), str)
            ),
        )
    )
    operations = tuple(item[1] for item in scoped_operations)
    if not any(getattr(operation, "op", None) in RAGGED_OPS for operation in operations):
        return

    if isinstance(inputs, Mapping):
        input_values = {
            tensor.name: inputs[tensor.name]
            for tensor in fn_inputs
            if tensor.name in inputs
        }
    else:
        if len(inputs) != len(fn_inputs):
            raise RaggedRuntimeError(
                "runtime Ragged validation input arity differs from J"
            )
        input_values = {
            tensor.name: value
            for tensor, value in zip(fn_inputs, inputs, strict=True)
        }
    input_dtypes = {
        tensor.name: tensor.dtype
        for tensor in fn_inputs
        if isinstance(getattr(tensor, "name", None), str)
    }
    def operation_dtype(operation: Any) -> str | None:
        if operation is None:
            return None
        operation_type = getattr(operation, "type", None)
        if (
            isinstance(operation_type, Mapping)
            and isinstance(operation_type.get("dtype"), str)
        ):
            return operation_type["dtype"]
        return None

    checked_dims = dict(dims or {})
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(extent, bool)
        or not isinstance(extent, (int, np.integer))
        or int(extent) < 0
        for name, extent in checked_dims.items()
    ):
        raise RaggedRuntimeError(
            "runtime Ragged validation dims must be non-negative J bindings"
        )

    def declared_field_dtype(operation: Any, field_index: int) -> str | None:
        operation_type = getattr(operation, "type", None)
        fields = (
            operation_type.get("fields")
            if isinstance(operation_type, Mapping)
            else None
        )
        if not isinstance(fields, list) or len(fields) != 2:
            return None
        field = fields[field_index]
        field_type = field.get("type") if isinstance(field, Mapping) else None
        dtype = field_type.get("dtype") if isinstance(field_type, Mapping) else None
        return dtype if isinstance(dtype, str) else None

    def constant_value(name: str, operation: Any, *, expected: str | None) -> Any:
        raw_shape = operation_attr(operation, "shape", [])
        raw_value = operation_attr(operation, "value", None)
        dtype = expected or operation_dtype(operation) or operation_attr(
            operation, "dtype", None
        )
        try:
            return materialize_ragged_constant(
                raw_shape,
                raw_value,
                dtype=dtype,
                dims=checked_dims,
                context=f"constant carrier {name!r}",
            )
        except (TypeError, ValueError, OverflowError, RuntimeShapeResolutionError) as exc:
            raise RaggedRuntimeError(
                f"constant carrier {name!r} cannot be materialized"
            ) from exc

    def value_for(
        name: str,
        *,
        expected: str | None,
        context: str,
        visible_bindings: Mapping[str, Any],
    ) -> Any:
        if name in input_values:
            return input_values[name]
        operation = visible_bindings.get(name)
        if operation is not None and getattr(operation, "op", None) == "const":
            return constant_value(name, operation, expected=expected)
        raise RaggedRuntimeError(
            f"{context} requires a direct input or determinable const carrier "
            f"for {name!r}"
        )

    descriptors: dict[int, tuple[Any, Any, str | None, str | None]] = {}
    for _path, operation, visible_bindings in scoped_operations:
        op_name = getattr(operation, "op", None)
        operands = getattr(operation, "operands", ())
        if op_name == "ragged":
            if len(operands) != 2 or result_name(operation) is None:
                raise RaggedRuntimeError(
                    "runtime Ragged validation found an incomplete ragged op"
                )
            offsets_name, indices_name = operands
            offsets_producer = visible_bindings.get(offsets_name)
            indices_producer = visible_bindings.get(indices_name)
            offsets_dtype = declared_field_dtype(operation, 0) or input_dtypes.get(
                offsets_name
            ) or operation_dtype(offsets_producer)
            indices_dtype = declared_field_dtype(operation, 1) or input_dtypes.get(
                indices_name
            ) or operation_dtype(indices_producer)
            offsets = value_for(
                offsets_name,
                expected=offsets_dtype,
                context="ragged offsets",
                visible_bindings=visible_bindings,
            )
            indices = value_for(
                indices_name,
                expected=indices_dtype,
                context="ragged indices",
                visible_bindings=visible_bindings,
            )
            validate_ragged_runtime_carrier(
                offsets,
                indices,
                context=f"runtime {result_name(operation)}",
                offsets_dtype=offsets_dtype,
                indices_dtype=indices_dtype,
            )
            descriptors[id(operation)] = (
                offsets,
                indices,
                offsets_dtype,
                indices_dtype,
            )
            continue
        if op_name not in {"ragged_outer", "ragged_member"}:
            continue
        descriptor_producer = (
            visible_bindings.get(operands[0]) if operands else None
        )
        descriptor = (
            descriptors.get(id(descriptor_producer))
            if descriptor_producer is not None
            else None
        )
        if descriptor is None:
            raise RaggedRuntimeError(
                f"runtime {op_name} requires a preceding local ragged descriptor"
            )
        offsets, indices, offsets_dtype, indices_dtype = descriptor
        validate_ragged_runtime_carrier(
            offsets,
            indices,
            context=f"runtime {operands[0]}",
            offsets_dtype=offsets_dtype,
            indices_dtype=indices_dtype,
        )
        if op_name == "ragged_member":
            if len(operands) != 2:
                raise RaggedRuntimeError(
                    "runtime ragged_member requires descriptor and selector"
            )
            selector_name = operands[1]
            selector_dtype = input_dtypes.get(selector_name) or operation_dtype(
                visible_bindings.get(selector_name)
            )
            selector = value_for(
                selector_name,
                expected=selector_dtype,
                context="ragged selector",
                visible_bindings=visible_bindings,
            )
            selector_shape = _runtime_shape(selector, context="runtime ragged selector")
            if selector_shape != ():
                raise RaggedRuntimeError(
                    "runtime ragged selector must be rank-0"
                )
            _validate_runtime_ragged_dtype(
                selector,
                expected=selector_dtype,
                context="runtime ragged selector",
            )
            ragged_member_length(
                offsets,
                indices,
                selector,
                context=f"runtime {result_name(operation)}",
            )


def _resolve_runtime_extent(
    extent: Any,
    dimensions: Mapping[str, int],
    *,
    context: str,
) -> int | None:
    if extent == "?":
        return None
    if isinstance(extent, bool):
        raise RuntimeShapeResolutionError(f"{context} has a boolean extent")
    if isinstance(extent, (int, np.integer)):
        value = int(extent)
    elif isinstance(extent, str) and extent in dimensions:
        value = int(dimensions[extent])
    else:
        raise RuntimeShapeResolutionError(
            f"{context} has an unresolved extent {extent!r}"
        )
    if value < 0:
        raise RuntimeShapeResolutionError(f"{context} has a negative extent")
    return value


def materialize_ragged_constant(
    shape: Sequence[Any],
    value: Any,
    *,
    dtype: str | None = None,
    dims: Mapping[str, int] | None = None,
    context: str = "constant carrier",
) -> np.ndarray:
    """Materialize one J-authored Ragged carrier/selector constant.

    Shape resolution and dtype spelling belong to the shared runtime boundary;
    Gate2, CPU/reference validation, and dynamic-output resolution must not keep
    independent NumPy type tables.  ``index`` is materialized as the i64
    carrier, while an unknown dtype is left to NumPy and subsequently rejected
    by the closed Ragged runtime validator.
    """

    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
        raise RuntimeShapeResolutionError(f"{context} has a non-list shape")
    resolved_shape: list[int] = []
    dimensions = dict(dims or {})
    for axis, extent in enumerate(shape):
        resolved = _resolve_runtime_extent(
            extent,
            dimensions,
            context=f"{context} shape[{axis}]",
        )
        if resolved is None:
            raise RuntimeShapeResolutionError(
                f"{context} has an unresolved dynamic shape"
            )
        resolved_shape.append(resolved)
    dtype_key = _dtype_key(dtype) if isinstance(dtype, str) else ""
    dtype_key = "i64" if dtype_key == "index" else dtype_key
    numpy_dtype = _RAGGED_NUMPY_DTYPES.get(dtype_key)
    if numpy_dtype is not None:
        return np.full(tuple(resolved_shape), value, dtype=numpy_dtype)
    return np.full(tuple(resolved_shape), value)


def resolve_runtime_output_shapes(
    fn: Any,
    declared_shapes: Sequence[Sequence[Any]],
    inputs: Sequence[Any],
    *,
    dims: Mapping[str, int] | None = None,
    canonical_output_bindings: Sequence[Any] | None = None,
) -> list[tuple[int, ...]]:
    """Resolve J output shapes for one finite invocation.

    A ``?`` is a data-dependent extent, never a symbolic J dimension.  The
    only currently closed dynamic result is a direct ``ragged_member`` return;
    its length is derived through :func:`ragged_member_length`, so CPU, CUDA
    host orchestration, and Gate2 workload construction share the same carrier
    contract.  ``canonical_output_bindings`` is required so output/value/ABI
    ownership comes from the canonical construction record rather than a
    second scan of ``IntentFunction.returns``.  This module intentionally has
    no backend imports.
    """

    if len(fn.inputs) != len(inputs):
        raise RuntimeShapeResolutionError(
            "runtime output shape resolution input arity differs from J"
        )
    checked_dims = dict(dims or {})
    if any(
        not isinstance(name, str)
        or not name
        or name == "?"
        or isinstance(extent, bool)
        or not isinstance(extent, (int, np.integer))
        or int(extent) < 0
        for name, extent in checked_dims.items()
    ):
        raise RuntimeShapeResolutionError(
            "runtime output shape resolution dims must be non-negative J bindings"
        )

    # Every runtime shape consumer shares the same Ragged carrier boundary.
    # This runs before any output-shape fast path, so a static ragged_outer
    # result cannot bypass validation merely because it has no ``?`` extent.
    validate_ragged_runtime_invocation(fn, inputs, dims=checked_dims)

    input_values = {
        tensor.name: _runtime_tensor(value, context=f"input {tensor.name!r}")
        for tensor, value in zip(fn.inputs, inputs, strict=True)
    }
    producer_by_result = intent_op_producers(
        fn.body,
        visible=(
            tensor.name
            for tensor in fn.inputs
            if isinstance(getattr(tensor, "name", None), str)
        ),
    )
    scoped_operations = tuple(
        walk_intent_op_scopes(
            fn.body,
            visible=(
                tensor.name
                for tensor in fn.inputs
                if isinstance(getattr(tensor, "name", None), str)
            ),
        )
    )
    visible_by_operation = {
        id(operation): visible
        for _path, operation, visible in scoped_operations
    }

    def tensor_value(
        name: str,
        *,
        visible_bindings: Mapping[str, Any],
    ) -> Any:
        if name in input_values:
            return input_values[name]
        producer = visible_bindings.get(name)
        if producer is not None and producer.op == "const":
            value = producer.attrs.get("value")
            shape = producer.attrs.get("shape", [])
            operation_type = getattr(producer, "type", None)
            dtype = (
                operation_type.get("dtype")
                if isinstance(operation_type, Mapping)
                else None
            )
            try:
                return materialize_ragged_constant(
                    shape,
                    value,
                    dtype=dtype,
                    dims=checked_dims,
                    context=f"constant carrier {name!r}",
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeShapeResolutionError(
                    f"constant carrier {name!r} cannot be materialized"
                ) from exc
        raise RuntimeShapeResolutionError(
            "runtime Ragged output resolution requires carrier tensor input "
            f"{name!r}"
        )

    if canonical_output_bindings is None:
        raise RuntimeShapeResolutionError(
            "runtime output shape resolution requires the canonical output "
            "binding sequence"
        )
    bindings = tuple(canonical_output_bindings)
    output_by_name = {
        item.name: item for item in fn.outputs
        if isinstance(getattr(item, "name", None), str)
    }
    ordered: list[Any] = []
    value_for_output: dict[str, str] = {}
    seen_outputs: set[str] = set()
    previous_abi_index = -1
    for binding in bindings:
        output = getattr(binding, "output", None)
        value = getattr(binding, "value", None)
        abi_index = getattr(binding, "abi_index", None)
        output_type = getattr(binding, "output_type", None)
        output_shape = getattr(output_type, "shape", None)
        output_dtype = getattr(output_type, "dtype", None)
        function_output = output_by_name.get(output)
        if (
            not isinstance(output, str)
            or not isinstance(value, str)
            or output in seen_outputs
            or isinstance(abi_index, bool)
            or not isinstance(abi_index, int)
            or abi_index != previous_abi_index + 1
            or not isinstance(output_shape, tuple)
            or not isinstance(output_dtype, str)
            or function_output is None
            or tuple(function_output.shape) != output_shape
            or function_output.dtype != output_dtype
        ):
            raise RuntimeShapeResolutionError(
                "runtime output shape resolution received an incomplete "
                "canonical output binding sequence"
            )
        seen_outputs.add(output)
        previous_abi_index = abi_index
        ordered.append(binding)
        value_for_output[output] = value
    if tuple(item.output for item in ordered) != tuple(
        item.name for item in fn.outputs
    ):
        raise RuntimeShapeResolutionError(
            "canonical output binding order differs from IntentFunction outputs"
        )
    output_order = tuple(ordered)
    resolved_outputs: list[tuple[int, ...]] = []
    for output, declared in zip(output_order, declared_shapes, strict=True):
        shape = list(declared)
        output_name = output.output
        output_shape = output.output_type.shape
        if len(shape) != len(output_shape):
            raise RuntimeShapeResolutionError(
                f"runtime output shape rank differs for {output_name!r}"
            )
        has_dynamic_extent = False
        for axis, extent in enumerate(shape):
            resolved = _resolve_runtime_extent(
                extent,
                checked_dims,
                context=f"runtime output {output_name!r} shape[{axis}]",
            )
            if resolved is None:
                has_dynamic_extent = True
            else:
                shape[axis] = resolved
        if not has_dynamic_extent:
            resolved_outputs.append(tuple(int(extent) for extent in shape))
            continue

        value_name = value_for_output.get(output_name)
        producer = producer_by_result.get(value_name)
        if (
            producer is None
            or producer.op != "ragged_member"
            or len(producer.operands) != 2
        ):
            raise RuntimeShapeResolutionError(
                "compiled dynamic output is not produced by ragged_member: "
                f"{output_name!r}"
            )
        member_bindings = visible_by_operation.get(id(producer), {})
        descriptor = member_bindings.get(producer.operands[0])
        if (
            descriptor is None
            or descriptor.op != "ragged"
            or len(descriptor.operands) != 2
        ):
            raise RuntimeShapeResolutionError(
                "compiled dynamic output has no local ragged producer: "
                f"{output_name!r}"
            )
        descriptor_bindings = visible_by_operation.get(id(descriptor), {})
        offsets = tensor_value(
            descriptor.operands[0], visible_bindings=descriptor_bindings
        )
        indices = tensor_value(
            descriptor.operands[1], visible_bindings=descriptor_bindings
        )
        selector = tensor_value(
            producer.operands[1], visible_bindings=member_bindings
        )
        try:
            length = ragged_member_length(
                offsets,
                indices,
                selector,
                context=f"runtime output {output_name!r}",
            )
        except RaggedRuntimeError as exc:
            raise RuntimeShapeResolutionError(str(exc)) from exc
        shape = [length if extent == "?" else int(extent) for extent in shape]
        resolved_outputs.append(tuple(shape))
    return resolved_outputs


__all__ = [
    "materialize_ragged_constant",
    "RaggedRuntimeError",
    "RuntimeShapeResolutionError",
    "ValueDimensionRelationError",
    "ragged_member_length",
    "ragged_outer_count",
    "resolve_value_dimension_equalities",
    "resolve_runtime_output_shapes",
    "validate_ragged_runtime_invocation",
    "validate_ragged_runtime_carrier",
    "validate_value_dimension_instantiation",
    "validate_value_dimension_relations",
]
