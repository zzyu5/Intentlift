"""Canonical in-memory model for the Layer-A value graph in ``J``.

The persisted contract is intentionally smaller than the downstream
``intent`` MLIR dialect.  Layer B lives only in the enclosing canonical-J
boundary, while scheduling and source realization live in C and pi.  This
module therefore rejects legacy per-op correctness and scheduling fields
instead of silently preserving them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

from .intent_op_contract import (
    IntentOpContractError,
    canonicalize_dtype,
    is_non_boolean_integer_dtype,
    is_ragged_carrier_dtype,
    is_scalar_literal,
    is_scalar_literal_or_value_reference,
    validate_const_literal,
    validate_intent_op_structure,
)


def _canonical_dimension(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _closed_keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} has forbidden fields: {unknown}")


_LOGICAL_TENSOR_HANDLE = re.compile(r"t[0-9]+")
_VALUE_HANDLE = re.compile(r"(?:t|v)[0-9]+")
_SSA_HANDLE = re.compile(r"v[0-9]+")
_DIMENSION_HANDLE = re.compile(r"x[0-9]+")
_LOOP_STATE_HANDLE = re.compile(r"s[0-9]+")
_SCOPE_HANDLE = re.compile(r"q[0-9]+")
_PHYSICAL_PARAMETER_HANDLE = re.compile(r"p[0-9]+")
_PHYSICAL_PARAMETER_TOKEN = re.compile(r"(?<![A-Za-z0-9_])p[0-9]+(?![A-Za-z0-9_])")


def _require_handle(
    value: Any,
    pattern: re.Pattern[str],
    context: str,
    description: str,
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{context} must be a {description} handle")
    return value


def _reject_physical_parameter_reference(value: Any, context: str) -> None:
    """Keep physical signature handles out of typed Layer-A references.

    The JSON parser applies the complete structural contract before creating
    these dataclasses.  Typed callers, however, can construct region objects
    directly.  This deliberately narrow guard closes only the namespace hole
    for exact ``p*`` handles; it does not impose a second full structural
    validator on internal, not-yet-serialized objects.
    """

    if isinstance(value, str) and _PHYSICAL_PARAMETER_HANDLE.fullmatch(value):
        raise ValueError(
            f"{context} cannot reference physical parameter {value!r}; "
            "p* handles belong only to J.signature.parameters"
        )


def _reject_physical_parameter_references(
    values: Any,
    context: str,
) -> None:
    """Apply :func:`_reject_physical_parameter_reference` to one sequence."""

    if isinstance(values, (list, tuple)):
        for index, value in enumerate(values):
            _reject_physical_parameter_reference(value, f"{context}[{index}]")
    else:
        _reject_physical_parameter_reference(values, context)


def _validate_shape_dimension(value: Any, context: str) -> Any:
    """Validate one Layer-A shape extent without rejecting legacy symbols.

    Ordinary J callers still use source-neutral symbols such as ``M`` and
    ``Sq`` in addition to the canonical ``x*`` dimensions.  The namespace
    boundary that matters here is narrower: a physical signature parameter
    must never be smuggled into a logical tensor shape.  ContractExpr objects
    are retained for typed internal callers; their referenced names receive
    the same physical-parameter guard.
    """

    if isinstance(value, str):
        if not value:
            raise ValueError(f"{context} must be non-empty")
        if _PHYSICAL_PARAMETER_TOKEN.search(value) is not None:
            raise ValueError(
                f"{context} cannot reference physical parameter p* in a logical shape"
            )
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"{context} must be non-negative")
        return _canonical_dimension(value)

    # Avoid importing canonical_j here: canonical_j imports this module.  A
    # typed ContractExpr is identified by its stable public surface and kept
    # opaque to this model; only its symbolic references are checked.
    references = getattr(value, "references", None)
    if references is not None and callable(getattr(value, "to_text", None)):
        for reference in references:
            if isinstance(reference, str) and _PHYSICAL_PARAMETER_HANDLE.fullmatch(
                reference
            ):
                raise ValueError(
                    f"{context} cannot reference physical parameter {reference!r}"
                )
        return value

    raise TypeError(f"{context} must be an integer, symbol, or ContractExpr")


def _validate_shape(value: Any, context: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a shape sequence")
    return [
        _validate_shape_dimension(dimension, f"{context}[{index}]")
        for index, dimension in enumerate(value)
    ]


@dataclass(frozen=True)
class ValueType:
    """Closed Layer-A tensor, named-record, or ragged value type.

    Tensor types retain the historical ``dtype``/``shape`` spelling.  Records
    use an ordered list of named fields so a record can carry several tensor
    values through a reduction, scan, or loop without turning field identity
    into positional convention.  A ragged value has exactly two tensor fields:
    one rank-1 offsets vector and one rank-1 member-index vector.
    """

    kind: str
    dtype: str | None = None
    shape: tuple[Any, ...] = ()
    fields: tuple[tuple[str, "ValueType"], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"tensor", "record", "ragged"}:
            raise ValueError(
                "ValueType.kind must be 'tensor', 'record', or 'ragged'"
            )
        if self.kind == "tensor":
            if (
                isinstance(self.fields, (str, bytes))
                or not isinstance(self.fields, Sequence)
                or len(self.fields) != 0
            ):
                raise ValueError("ValueType tensor types cannot have fields")
            object.__setattr__(
                self,
                "dtype",
                canonicalize_dtype(self.dtype, context="ValueType.dtype"),
            )
            object.__setattr__(
                self,
                "shape",
                tuple(_validate_shape(self.shape, "ValueType.shape")),
            )
            object.__setattr__(self, "fields", ())
            return

        if self.dtype is not None:
            raise ValueError(
                f"ValueType.{self.kind} types must not have a tensor dtype"
            )
        if (
            isinstance(self.shape, (str, bytes))
            or not isinstance(self.shape, Sequence)
            or len(self.shape) != 0
        ):
            raise ValueError(
                f"ValueType.{self.kind} types must not have a tensor shape"
            )
        object.__setattr__(self, "shape", ())

        if (
            isinstance(self.fields, (str, bytes))
            or not isinstance(self.fields, Sequence)
            or not self.fields
        ):
            raise ValueError(
                f"ValueType.{self.kind}.fields must be a non-empty sequence"
            )
        normalized_fields: list[tuple[str, ValueType]] = []
        names: set[str] = set()
        for index, field_item in enumerate(self.fields):
            if (
                not isinstance(field_item, (tuple, list))
                or len(field_item) != 2
            ):
                raise TypeError(
                    f"ValueType.{self.kind}.fields[{index}] must be a "
                    "(name, ValueType) pair"
                )
            name, field_type = field_item
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"ValueType.{self.kind}.fields[{index}].name must be non-empty"
                )
            if name in names:
                raise ValueError(
                    f"ValueType.{self.kind}.fields contains duplicate name {name!r}"
                )
            if not isinstance(field_type, ValueType):
                raise TypeError(
                    f"ValueType.{self.kind}.fields[{index}].type must be a ValueType"
                )
            if self.kind == "record" and not field_type.is_tensor:
                raise ValueError(
                    f"ValueType.record.fields[{index}].type must be a tensor"
                )
            if self.kind == "ragged" and (
                not field_type.is_tensor
                or len(field_type.shape) != 1
                or field_type.shape[0] == "?"
                or not is_ragged_carrier_dtype(field_type.dtype)
            ):
                raise ValueError(
                    f"ValueType.ragged.fields[{index}].type must be a rank-1 "
                    "Ragged carrier tensor"
                )
            names.add(name)
            normalized_fields.append((name, field_type))
        if self.kind == "ragged" and [name for name, _ in normalized_fields] != [
            "offsets", "indices"
        ]:
            raise ValueError(
                "ValueType.ragged.fields must contain offsets followed by indices"
            )
        object.__setattr__(self, "fields", tuple(normalized_fields))

    @classmethod
    def from_json(cls, value: dict[str, Any], *, context: str = "ValueType") -> "ValueType":
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be an object")
        kind = value.get("kind", "tensor")
        if kind == "tensor":
            if "kind" in value:
                raise ValueError(
                    f"{context}.kind must be omitted for canonical tensor types"
                )
            _closed_keys(value, {"dtype", "shape"}, context)
            dtype = canonicalize_dtype(
                value.get("dtype"), context=f"{context}.dtype"
            )
            shape = value.get("shape")
            if not isinstance(dtype, str) or not dtype:
                raise ValueError(f"{context}.dtype must be a non-empty string")
            if not isinstance(shape, list):
                raise TypeError(f"{context}.shape must be an array")
            normalized = _validate_shape(shape, f"{context}.shape")
            return cls("tensor", dtype=dtype, shape=tuple(normalized))
        if kind not in {"record", "ragged"}:
            raise ValueError(
                f"{context}.kind must be 'tensor', 'record', or 'ragged'"
            )
        _closed_keys(value, {"kind", "fields"}, context)
        raw_fields = value.get("fields")
        if not isinstance(raw_fields, list) or not raw_fields:
            raise ValueError(f"{context}.fields must be a non-empty array")
        fields: list[tuple[str, ValueType]] = []
        names: set[str] = set()
        for index, raw_field in enumerate(raw_fields):
            field_context = f"{context}.fields[{index}]"
            _closed_keys(raw_field, {"name", "type"}, field_context)
            name = raw_field.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{field_context}.name must be non-empty")
            if name in names:
                raise ValueError(f"{context}.fields contains duplicate name {name!r}")
            field_type = cls.from_json(
                raw_field.get("type"), context=f"{field_context}.type"
            )
            if kind == "ragged":
                if not field_type.is_tensor or len(field_type.shape) != 1:
                    raise ValueError(
                        f"{field_context}.type must be a rank-1 tensor for ragged"
                    )
                if field_type.shape[0] == "?":
                    raise ValueError(
                        f"{field_context}.type.shape must be a concrete or "
                        "symbolic rank-1 carrier extent"
                    )
                if not is_ragged_carrier_dtype(field_type.dtype):
                    raise ValueError(
                        f"{field_context}.type.dtype must be a closed Ragged "
                        "carrier dtype"
                    )
            names.add(name)
            fields.append((name, field_type))
        if kind == "ragged" and [name for name, _ in fields] != [
            "offsets", "indices"
        ]:
            raise ValueError(
                f"{context}.fields must contain offsets followed by indices"
            )
        return cls(kind, fields=tuple(fields))

    @classmethod
    def tensor(cls, dtype: str, shape: Sequence[Any]) -> "ValueType":
        return cls(
            "tensor",
            dtype=canonicalize_dtype(dtype, context="ValueType.dtype"),
            shape=tuple(_validate_shape(shape, "ValueType.shape")),
        )

    def field(self, name: str) -> "ValueType":
        for field_name, field_type in self.fields:
            if field_name == name:
                return field_type
        raise KeyError(name)

    def to_json(self) -> dict[str, Any]:
        if self.kind == "tensor":
            return {"dtype": self.dtype, "shape": list(self.shape)}
        return {
            "kind": self.kind,
            "fields": [
                {"name": name, "type": field_type.to_json()}
                for name, field_type in self.fields
            ],
        }

    @property
    def is_tensor(self) -> bool:
        return self.kind == "tensor"

    @property
    def is_record(self) -> bool:
        return self.kind == "record"

    @property
    def is_ragged(self) -> bool:
        return self.kind == "ragged"


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: list[Any]  # entries may be ints or symbolic dim names (e.g. "Sq")

    def __post_init__(self) -> None:
        self.name = _require_handle(
            self.name,
            _LOGICAL_TENSOR_HANDLE,
            "Tensor.name",
            "logical tensor t*",
        )
        self.dtype = canonicalize_dtype(self.dtype, context=f"Tensor {self.name!r}.dtype")
        self.shape = _validate_shape(self.shape, f"Tensor {self.name!r}.shape")

    @classmethod
    def from_json(cls, d: dict) -> "Tensor":
        _closed_keys(d, {"name", "dtype", "shape"}, "Tensor")
        name = _require_handle(
            d.get("name"),
            _LOGICAL_TENSOR_HANDLE,
            "Tensor.name",
            "logical tensor t*",
        )
        value_type = ValueType.from_json(
            {"dtype": d.get("dtype"), "shape": d.get("shape")},
            context="Tensor",
        )
        return cls(
            name=name,
            dtype=value_type.dtype,
            shape=list(value_type.shape),
        )

    def to_json(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class IntentReturn:
    """One Layer-A logical result binding.

    ``value`` is an SSA value produced by the semantic graph.  ``output`` is
    the logical function result it defines.  Caller-owned destinations and
    alias/writeback behavior deliberately do not live here; those are Layer-B
    contracts in the canonical J boundary object.
    """

    output: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output",
            _require_handle(
                self.output,
                _LOGICAL_TENSOR_HANDLE,
                "IntentReturn.output",
                "logical tensor t*",
            ),
        )
        object.__setattr__(
            self,
            "value",
            _require_handle(
                self.value,
                _VALUE_HANDLE,
                "IntentReturn.value",
                "logical value t*/v*",
            ),
        )

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "IntentReturn":
        _closed_keys(d, {"output", "value"}, "IntentReturn")
        return cls(
            output=_require_handle(
                d.get("output"),
                _LOGICAL_TENSOR_HANDLE,
                "IntentReturn.output",
                "logical tensor t*",
            ),
            value=_require_handle(
                d.get("value"),
                _VALUE_HANDLE,
                "IntentReturn.value",
                "logical value t*/v*",
            ),
        )

    def to_json(self) -> dict[str, str]:
        return {"output": self.output, "value": self.value}


@dataclass
class LoopValueType:
    """Portable type carried across a semantic loop iteration.

    This is deliberately a tensor/value type, not a frontend storage decision:
    register/shared/global placement belongs outside J in realization/lowering.
    """

    dtype: str | None
    shape: list[Any] = field(default_factory=list)
    kind: str = "tensor"
    fields: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind == "ragged":
            raise ValueError(
                "LoopValueType.kind='ragged' is not supported: Ragged "
                "descriptors are region-local and cannot cross a loop carry "
                "or result ABI"
            )
        if self.kind == "tensor":
            if (
                isinstance(self.fields, (str, bytes))
                or not isinstance(self.fields, Sequence)
                or len(self.fields) != 0
            ):
                raise ValueError("LoopValueType tensor types cannot have fields")
            value_type = ValueType.tensor(self.dtype, self.shape)
            self.dtype = value_type.dtype
            self.shape = list(value_type.shape)
            self.fields = []
            return
        if self.kind != "record":
            raise ValueError(
                "LoopValueType.kind must be 'tensor' or 'record'"
            )
        if self.dtype is not None:
            raise ValueError("LoopValueType record types must not have a dtype")
        if (
            isinstance(self.shape, (str, bytes))
            or not isinstance(self.shape, Sequence)
            or len(self.shape) != 0
        ):
            raise ValueError("LoopValueType record types must not have a shape")
        value_type = ValueType.from_json(
            {"kind": "record", "fields": self.fields},
            context="LoopValueType",
        )
        self.dtype = None
        self.shape = []
        self.fields = [
            {"name": name, "type": field_type.to_json()}
            for name, field_type in value_type.fields
        ]

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "LoopValueType":
        if not isinstance(d, dict):
            raise TypeError("LoopValueType must be an object")
        if d.get("kind") == "tensor":
            raise ValueError(
                "LoopValueType.kind must be omitted for canonical tensor types"
            )
        if d.get("kind") == "ragged":
            raise ValueError(
                "LoopValueType.kind='ragged' is not supported: Ragged "
                "descriptors are region-local and cannot cross a loop carry "
                "or result ABI"
            )
        # Keep the historical tensor spelling and its default empty shape, but
        # route the actual dtype/shape validation through the same closed parser
        # used by ordinary value types.
        if "kind" not in d:
            dtype = d.get("dtype")
            shape = d.get("shape", [])
            if not isinstance(shape, list):
                raise TypeError("LoopValueType.shape must be an array")
            value_type = ValueType.from_json(
                {"dtype": dtype, "shape": shape},
                context="LoopValueType",
            )
            return cls(
                dtype=value_type.dtype,
                shape=list(value_type.shape),
            )
        value_type = ValueType.from_json(d, context="LoopValueType")
        return cls(
            dtype=value_type.dtype,
            shape=list(value_type.shape),
            kind=value_type.kind,
            fields=[
                {"name": name, "type": field_type.to_json()}
                for name, field_type in value_type.fields
            ],
        )

    def to_json(self) -> dict[str, Any]:
        if self.kind == "record":
            return {
                "kind": self.kind,
                "fields": list(self.fields),
            }
        if self.kind == "tensor":
            return {"dtype": self.dtype, "shape": list(self.shape)}
        raise ValueError("LoopValueType.kind must be 'tensor' or 'record'")

    def as_value_type(self) -> ValueType:
        return ValueType.from_json(self.to_json(), context="LoopValueType")


@dataclass
class LoopStateBinding:
    """One explicit ``init -> body arg -> next -> result`` loop binding.

    The binding is the single structured-loop wire shape.  Positional loop
    result/yield arrays are checked by ``IntentOp.from_json`` against these
    explicit fields; no historical ``name``/``init`` spelling is accepted.
    """

    state_id: str
    init_value: Any
    body_argument: str
    next_value: str
    result_value: str
    type: LoopValueType | None = None

    def __post_init__(self) -> None:
        self.state_id = _require_handle(
            self.state_id,
            _LOOP_STATE_HANDLE,
            "LoopStateBinding.state_id",
            "loop-state s*",
        )
        _reject_physical_parameter_reference(
            self.init_value, "LoopStateBinding.init_value"
        )
        _reject_physical_parameter_references(
            (self.body_argument, self.next_value, self.result_value),
            "LoopStateBinding.value_role",
        )
        if self.type is not None and not isinstance(self.type, LoopValueType):
            raise TypeError("LoopStateBinding.type must be a LoopValueType")

    @classmethod
    def from_json(
        cls,
        d: dict[str, Any],
        *,
        index: int,
        positional_result: str | None = None,
        positional_yield: str | None = None,
    ) -> "LoopStateBinding":
        _closed_keys(
            d,
            {
                "state_id", "init_value", "body_argument", "next_value",
                "result_value", "type",
            },
            "LoopStateBinding",
        )
        body_argument = d.get("body_argument")
        result_value = d.get("result_value")
        next_value = d.get("next_value")
        init_value = d.get("init_value")
        value_type = d.get("type")
        state_id = _require_handle(
            d.get("state_id"),
            _LOOP_STATE_HANDLE,
            "LoopStateBinding.state_id",
            "loop-state s*",
        )
        if not is_scalar_literal_or_value_reference(init_value):
            raise ValueError(
                "LoopStateBinding.init_value must be a scalar literal or "
                "a t*/v* value reference"
            )
        if positional_result is not None and result_value != positional_result:
            raise ValueError(
                "LoopStateBinding.result_value disagrees with loop.result"
            )
        if positional_yield is not None and next_value != positional_yield:
            raise ValueError(
                "LoopStateBinding.next_value disagrees with loop.yield"
            )
        body_argument = _require_handle(
            body_argument,
            _SSA_HANDLE,
            "LoopStateBinding.body_argument",
            "SSA v*",
        )
        next_value = _require_handle(
            next_value,
            _SSA_HANDLE,
            "LoopStateBinding.next_value",
            "SSA v*",
        )
        result_value = _require_handle(
            result_value,
            _SSA_HANDLE,
            "LoopStateBinding.result_value",
            "SSA v*",
        )
        if value_type is None:
            raise ValueError("LoopStateBinding.type is required")
        parsed_type = LoopValueType.from_json(value_type)
        return cls(
            state_id=state_id,
            init_value=init_value,
            body_argument=body_argument,
            next_value=next_value,
            result_value=result_value,
            type=parsed_type,
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "state_id": self.state_id,
            "init_value": self.init_value,
            "body_argument": self.body_argument,
            "next_value": self.next_value,
            "result_value": self.result_value,
        }
        if self.type is not None:
            if not isinstance(self.type, LoopValueType):
                raise TypeError("LoopStateBinding.type must be a LoopValueType")
            result["type"] = self.type.to_json()
        return result

@dataclass
class LoopTrip:
    """Semantic loop trip description; tiling is supplied separately by C."""

    axis: str | None = None
    upper_bound: str | None = None
    index: str | None = None

    def __post_init__(self) -> None:
        _reject_physical_parameter_references(
            (self.axis, self.upper_bound, self.index),
            "LoopTrip.reference",
        )
        if self.axis is not None:
            _require_handle(
                self.axis,
                _DIMENSION_HANDLE,
                "LoopTrip.axis",
                "symbolic dimension x*",
            )
            if self.upper_bound is not None or self.index is not None:
                raise ValueError(
                    "LoopTrip static form cannot also contain upper_bound or index"
                )
            return
        if self.upper_bound is None or self.index is None:
            raise ValueError(
                "LoopTrip dynamic form requires both upper_bound and index"
            )
        _require_handle(
            self.upper_bound,
            _VALUE_HANDLE,
            "LoopTrip.upper_bound",
            "logical value t*/v*",
        )
        _require_handle(
            self.index,
            _SSA_HANDLE,
            "LoopTrip.index",
            "SSA v*",
        )

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "LoopTrip":
        if not isinstance(d, dict):
            raise TypeError("LoopTrip must be an object")
        if "axis" in d:
            _closed_keys(d, {"axis"}, "LoopTrip")
            axis = _require_handle(
                d.get("axis"),
                _DIMENSION_HANDLE,
                "LoopTrip.axis",
                "symbolic dimension x*",
            )
            return cls(axis=axis)
        _closed_keys(d, {"upper_bound", "index"}, "LoopTrip")
        upper_bound = _require_handle(
            d.get("upper_bound"),
            _VALUE_HANDLE,
            "LoopTrip.upper_bound",
            "logical value t*/v*",
        )
        index = _require_handle(
            d.get("index"),
            _SSA_HANDLE,
            "LoopTrip.index",
            "SSA v*",
        )
        return cls(upper_bound=upper_bound, index=index)

    def to_json(self) -> dict[str, Any]:
        if self.axis is not None:
            return {"axis": self.axis}
        return {"upper_bound": self.upper_bound, "index": self.index}

    @property
    def is_dynamic(self) -> bool:
        return self.axis is None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class LoopScanInput:
    """One input sliced along a semantic loop trip axis."""

    name: str
    source: str
    axis: int

    def __post_init__(self) -> None:
        _reject_physical_parameter_references(
            (self.name, self.source),
            "LoopScanInput.reference",
        )

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "LoopScanInput":
        _closed_keys(d, {"name", "from", "axis"}, "LoopScanInput")
        name = _require_handle(
            d.get("name"),
            _SSA_HANDLE,
            "LoopScanInput.name",
            "SSA v*",
        )
        source = _require_handle(
            d.get("from"),
            _VALUE_HANDLE,
            "LoopScanInput.from",
            "logical value t*/v*",
        )
        axis = d.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
            raise ValueError("LoopScanInput.axis must be a non-negative integer")
        return cls(name=name, source=source, axis=axis)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "from": self.source, "axis": self.axis}

    def __getitem__(self, key: str) -> Any:
        return getattr(self, "source" if key == "from" else key)


@dataclass
class EffectIfArm:
    """One explicitly polarized arm of an effect-only conditional region.

    The arm binds an existing portable ``branch_region`` scope.  Its body may
    define branch-local SSA values, but the enclosing ``effect_if`` has no SSA
    result or yield.  Effect/access membership remains solely in the bound
    scope's ``effects`` list and the matching Layer-B records; it is deliberately
    not duplicated here.
    """

    scope_id: str
    body: list["IntentOp"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scope_id = _require_handle(
            self.scope_id,
            _SCOPE_HANDLE,
            "EffectIfArm.scope_id",
            "portable scope q*",
        )

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "EffectIfArm":
        _closed_keys(d, {"scope_id", "body"}, "EffectIfArm")
        scope_id = _require_handle(
            d.get("scope_id"),
            _SCOPE_HANDLE,
            "EffectIfArm.scope_id",
            "portable scope q*",
        )
        body = d.get("body")
        if not isinstance(body, list):
            raise TypeError("EffectIfArm.body must be an array")
        return cls(
            scope_id=scope_id,
            body=[IntentOp.from_json(item) for item in body],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "body": [item.to_json() for item in self.body],
        }


@dataclass
class ValueIfArm:
    """One explicit arm of a pure single-result value conditional."""

    scope_id: str
    body: list["IntentOp"] = field(default_factory=list)
    yield_value: str = ""

    def __post_init__(self) -> None:
        self.scope_id = _require_handle(
            self.scope_id,
            _SCOPE_HANDLE,
            "ValueIfArm.scope_id",
            "portable scope q*",
        )
        _reject_physical_parameter_reference(
            self.yield_value, "ValueIfArm.yield_value"
        )

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ValueIfArm":
        _closed_keys(d, {"scope_id", "body", "yield_value"}, "ValueIfArm")
        scope_id = _require_handle(
            d.get("scope_id"),
            _SCOPE_HANDLE,
            "ValueIfArm.scope_id",
            "portable scope q*",
        )
        body = d.get("body")
        if not isinstance(body, list):
            raise TypeError("ValueIfArm.body must be an array")
        return cls(
            scope_id=scope_id,
            body=[IntentOp.from_json(item) for item in body],
            yield_value=_require_handle(
                d.get("yield_value"),
                _VALUE_HANDLE,
                "ValueIfArm.yield_value",
                "logical value t*/v*",
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "body": [item.to_json() for item in self.body],
            "yield_value": self.yield_value,
        }


@dataclass
class IntentOp:
    """One Layer-A semantic node, including closed structured regions."""

    op: str
    result: Any = None  # str | list[str]
    operands: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    type: dict[str, Any] | None = None
    scope_id: str | None = None  # portable semantic scope; never a source span
    # structured-loop fields (op == "loop"); see spec/ir/structured-loop.md
    carry: list[LoopStateBinding] = field(default_factory=list)
    trip: LoopTrip | None = None
    scan: list[LoopScanInput] = field(default_factory=list)  # per-iter sliced inputs
    body: list["IntentOp"] = field(default_factory=list)
    yields: list[str] = field(default_factory=list)  # JSON key "yield"
    # binary conditional fields (op == "effect_if" or "value_if")
    predicate: str | None = None
    positive: EffectIfArm | ValueIfArm | None = None
    negative: EffectIfArm | ValueIfArm | None = None
    node_id: str | None = None  # canonical stable Layer-A node id

    def _validate_loop_shape(self, *, context: str = "IntentOp") -> None:
        """Validate the typed loop envelope before wiring is inspected.

        ``IntentOp`` is also constructible without going through
        :meth:`from_json`.  Keep that path closed in exactly the same way as
        the wire contract: a loop owns only its structured fields and must not
        carry ordinary ``operands``/``attrs``/``type`` state that a later
        consumer would silently ignore.
        """

        if self.op != "loop":
            return
        if self.operands:
            raise ValueError(f"{context} loop must not have operands")
        if self.attrs:
            raise ValueError(f"{context} loop must not have attrs")
        if self.type is not None:
            raise ValueError(f"{context} loop must not have an ordinary type")
        if self.predicate is not None or self.positive is not None or self.negative is not None:
            raise ValueError(
                f"{context} loop must not have conditional predicate/arm fields"
            )
        if not isinstance(self.result, list) or not self.result:
            raise ValueError(f"{context} loop result must be a non-empty list")
        if len(self.result) != len(set(self.result)):
            raise ValueError(f"{context} loop result values must be unique")
        for index, value in enumerate(self.result):
            _require_handle(
                value,
                _SSA_HANDLE,
                f"{context}.result[{index}]",
                "SSA v*",
            )
        if not isinstance(self.yields, list) or not self.yields:
            raise ValueError(f"{context} loop yield must be a non-empty list")
        if len(self.yields) != len(set(self.yields)):
            raise ValueError(f"{context} loop yield values must be unique")
        for index, value in enumerate(self.yields):
            _require_handle(
                value,
                _SSA_HANDLE,
                f"{context}.yield[{index}]",
                "SSA v*",
            )
        if not isinstance(self.carry, list) or not self.carry:
            raise ValueError(f"{context} loop carry must be a non-empty list")
        if any(not isinstance(item, LoopStateBinding) for item in self.carry):
            raise ValueError(
                f"{context} loop carry entries must be LoopStateBinding objects"
            )
        if not isinstance(self.trip, LoopTrip):
            raise ValueError(f"{context} loop trip must be a LoopTrip object")
        if not isinstance(self.scan, list):
            raise ValueError(f"{context} loop scan must be a list")
        if any(not isinstance(item, LoopScanInput) for item in self.scan):
            raise ValueError(
                f"{context} loop scan entries must be LoopScanInput objects"
            )
        if not isinstance(self.body, list) or not self.body:
            raise ValueError(f"{context} loop body must be a non-empty list")
        if any(not isinstance(item, IntentOp) for item in self.body):
            raise ValueError(f"{context} loop body entries must be IntentOp objects")

    def _structural_payload(self) -> dict[str, Any]:
        """Project one typed op into the shared wire-shape contract.

        ``IntentOp`` is intentionally constructible by internal typed callers,
        so JSON parsing is not the only way an operation can reach an emitter.
        Keep the direct-construction path aligned with ``from_json`` by
        validating the same fields that ``to_json`` would expose before the
        object can escape this model.
        """

        def nested_operations(values: Any) -> Any:
            if not isinstance(values, list):
                return values
            return [
                item._structural_payload()
                if isinstance(item, IntentOp)
                else item
                for item in values
            ]

        def arm_payload(arm: Any) -> Any:
            if isinstance(arm, EffectIfArm):
                return {
                    "scope_id": arm.scope_id,
                    "body": nested_operations(arm.body),
                }
            if isinstance(arm, ValueIfArm):
                return {
                    "scope_id": arm.scope_id,
                    "body": nested_operations(arm.body),
                    "yield_value": arm.yield_value,
                }
            return arm

        payload: dict[str, Any] = {"op": self.op}
        if self.node_id is not None:
            payload["id"] = self.node_id

        if self.op == "loop":
            payload.update(
                {
                    "result": self.result,
                    "scope_id": self.scope_id,
                    "carry": (
                        [
                            item.to_json()
                            if isinstance(item, LoopStateBinding)
                            else item
                            for item in self.carry
                        ]
                        if isinstance(self.carry, list)
                        else self.carry
                    ),
                    "trip": (
                        self.trip.to_json()
                        if isinstance(self.trip, LoopTrip)
                        else self.trip
                    ),
                    "scan": (
                        [
                            item.to_json()
                            if isinstance(item, LoopScanInput)
                            else item
                            for item in self.scan
                        ]
                        if isinstance(self.scan, list)
                        else self.scan
                    ),
                    "body": nested_operations(self.body),
                    "yield": self.yields,
                }
            )
            return payload

        if self.result is not None:
            payload["result"] = self.result
        if self.operands:
            payload["operands"] = self.operands
        if self.attrs:
            payload["attrs"] = self.attrs
        if self.type is not None:
            payload["type"] = self.type
        if self.scope_id is not None:
            payload["scope_id"] = self.scope_id
        if self.carry:
            payload["carry"] = [
                item.to_json() if isinstance(item, LoopStateBinding) else item
                for item in self.carry
            ]
        if self.trip is not None:
            payload["trip"] = (
                self.trip.to_json()
                if isinstance(self.trip, LoopTrip)
                else self.trip
            )
        if self.scan:
            payload["scan"] = [
                item.to_json() if isinstance(item, LoopScanInput) else item
                for item in self.scan
            ]
        if self.body:
            payload["body"] = nested_operations(self.body)
        if self.yields:
            payload["yield"] = self.yields
        if (
            self.op in {"effect_if", "value_if"}
            or self.predicate is not None
            or self.positive is not None
            or self.negative is not None
        ):
            payload["predicate"] = self.predicate
            payload["positive"] = arm_payload(self.positive)
            payload["negative"] = arm_payload(self.negative)
        return payload

    def __post_init__(self) -> None:
        """Keep direct typed construction on the same contract as JSON."""

        _reject_physical_parameter_references(
            self.result, "IntentOp.result"
        )
        _reject_physical_parameter_references(
            self.operands, "IntentOp.operands"
        )
        _reject_physical_parameter_references(
            self.yields, "IntentOp.yield"
        )
        _reject_physical_parameter_reference(
            self.predicate, "IntentOp.predicate"
        )
        validate_intent_op_structure(
            self._structural_payload(),
            path="IntentOp",
            recursive=True,
        )
        if self.op == "loop":
            self._validate_loop_shape()
            self.scope_id = _require_handle(
                self.scope_id,
                _SCOPE_HANDLE,
                "IntentOp.scope_id",
                "portable scope q*",
            )
            # Keep direct typed construction and the JSON/IntentFunction
            # entry point on the same complete loop contract.  This catches a
            # carry whose next/yield/result chain is incomplete before any
            # emitter or backend can consume it.
            validate_loop_wiring(self)

    @classmethod
    def from_json(cls, d: dict) -> "IntentOp":
        # Keep every JSON entry point on the same discriminated structural
        # contract.  ``_closed_keys`` alone cannot reject a loop-only field on
        # an ordinary op, or a partial effect_if/value_if shape, and callers
        # that construct an IntentOp directly otherwise bypass that boundary.
        validate_intent_op_structure(
            d,
            path="IntentOp",
            recursive=False,
        )
        _closed_keys(
            d,
            {
                "id", "op", "result", "operands", "attrs", "type",
                "scope_id", "carry", "trip", "scan", "body", "yield",
                "predicate", "positive", "negative",
            },
            "IntentOp",
        )
        result = d.get("result")
        result_names = list(result) if isinstance(result, list) else []
        yields = list(d.get("yield", []))
        raw_type = d.get("type")
        parsed_type = (
            ValueType.from_json(raw_type, context="IntentOp.type").to_json()
            if raw_type is not None
            else None
        )
        return cls(
            op=d["op"],
            node_id=d.get("id"),
            result=result,
            operands=list(d.get("operands", [])),
            attrs=dict(d.get("attrs", {})),
            type=parsed_type,
            scope_id=d.get("scope_id"),
            carry=[
                LoopStateBinding.from_json(
                    item,
                    index=index,
                    positional_result=(result_names[index] if index < len(result_names) else None),
                    positional_yield=(yields[index] if index < len(yields) else None),
                )
                for index, item in enumerate(d.get("carry", []))
            ],
            trip=(LoopTrip.from_json(d["trip"]) if d.get("trip") is not None else None),
            scan=[LoopScanInput.from_json(item) for item in d.get("scan", [])],
            body=[cls.from_json(x) for x in d.get("body", [])],
            yields=yields,
            predicate=d.get("predicate"),
            positive=(
                (
                    ValueIfArm.from_json(d["positive"])
                    if d.get("op") == "value_if"
                    else EffectIfArm.from_json(d["positive"])
                )
                if d.get("positive") is not None else None
            ),
            negative=(
                (
                    ValueIfArm.from_json(d["negative"])
                    if d.get("op") == "value_if"
                    else EffectIfArm.from_json(d["negative"])
                )
                if d.get("negative") is not None else None
            ),
        )

    def to_json(self) -> dict:
        # Direct callers can mutate this dataclass after construction.  Repeat
        # the shared structural check at the serialization boundary so a later
        # emitter cannot observe fields that this method would otherwise omit.
        validate_intent_op_structure(
            self._structural_payload(),
            path="IntentOp.to_json",
            recursive=True,
        )
        if self.op == "loop":
            self._validate_loop_shape(context="IntentOp.to_json")
            validate_loop_wiring(self, context="IntentOp.to_json loop")
            return {
                "op": "loop",
                **({"id": self.node_id} if self.node_id is not None else {}),
                "result": list(self.result),
                "scope_id": self.scope_id,
                "carry": [item.to_json() for item in self.carry],
                "trip": self.trip.to_json(),
                "scan": [item.to_json() for item in self.scan],
                "body": [item.to_json() for item in self.body],
                "yield": list(self.yields),
            }
        d: dict[str, Any] = {"op": self.op}
        if self.node_id is not None:
            d["id"] = self.node_id
        if self.result is not None:
            d["result"] = self.result
        if self.operands:
            d["operands"] = self.operands
        if self.attrs:
            d["attrs"] = self.attrs
        if self.type is not None:
            d["type"] = self.type
        if self.scope_id is not None:
            d["scope_id"] = self.scope_id
        if self.carry:
            d["carry"] = [item.to_json() for item in self.carry]
        if self.trip is not None:
            d["trip"] = self.trip.to_json()
        # ``scan`` is required by the canonical structured-loop wire contract,
        # and an empty scan list is semantically meaningful.  Do not collapse
        # it into field absence during a strict parse/to_json round trip.
        if self.op == "loop" or self.scan:
            d["scan"] = [item.to_json() for item in self.scan]
        if self.body:
            d["body"] = [x.to_json() for x in self.body]
        if self.yields:
            d["yield"] = self.yields
        if (
            self.op in {"effect_if", "value_if"}
            or self.predicate is not None
            or self.positive is not None
            or self.negative is not None
        ):
            d["predicate"] = self.predicate
            d["positive"] = (
                self.positive.to_json() if self.positive is not None else None
            )
            d["negative"] = (
                self.negative.to_json() if self.negative is not None else None
            )
        return d


@dataclass
class IntentFunction:
    name: str
    inputs: list[Tensor] = field(default_factory=list)
    outputs: list[Tensor] = field(default_factory=list)
    body: list[IntentOp] = field(default_factory=list)
    returns: list[IntentReturn] = field(default_factory=list)

    def _validate_tensor_declarations(self) -> None:
        """Validate dense logical tensor declaration order and inouts."""

        declarations: dict[str, tuple[str, tuple[Any, ...], str]] = {}
        for collection_name, tensors in (
            ("inputs", self.inputs),
            ("outputs", self.outputs),
        ):
            if not isinstance(tensors, list):
                raise TypeError(f"IntentFunction.{collection_name} must be an array")
            for index, tensor in enumerate(tensors):
                if not isinstance(tensor, Tensor):
                    raise TypeError(
                        f"IntentFunction.{collection_name}[{index}] must be a Tensor"
                    )
                previous = declarations.get(tensor.name)
                if previous is None:
                    expected = f"t{len(declarations)}"
                    if tensor.name != expected:
                        raise ValueError(
                            "IntentFunction logical tensor handles must be dense "
                            "in first-declaration order; expected "
                            f"{expected!r} at {collection_name}[{index}], got "
                            f"{tensor.name!r}"
                        )
                    declarations[tensor.name] = (
                        tensor.dtype,
                        tuple(tensor.shape),
                        collection_name,
                    )
                    continue
                previous_dtype, previous_shape, previous_collection = previous
                if previous_collection == collection_name:
                    raise ValueError(
                        f"IntentFunction.{collection_name} contains duplicate "
                        f"logical tensor {tensor.name!r}"
                    )
                if (
                    tensor.dtype != previous_dtype
                    or tuple(tensor.shape) != previous_shape
                ):
                    raise ValueError(
                        f"IntentFunction inout tensor {tensor.name!r} must have "
                        "identical input/output dtype and shape"
                    )

    @classmethod
    def from_json(cls, d: dict) -> "IntentFunction":
        if not isinstance(d, dict):
            raise TypeError("IntentFunction must be an object")
        if "intent_function" in d:
            _closed_keys(d, {"intent_function"}, "IntentFunction")
            fn = d["intent_function"]
        else:
            fn = d
        if not isinstance(fn, dict):
            raise TypeError("IntentFunction.intent_function must be an object")
        if "layer_c" in fn:
            raise ValueError(
                "IntentFunction no longer embeds C; pass pure C separately"
            )
        _closed_keys(
            fn,
            {"name", "inputs", "outputs", "returns", "body"},
            "IntentFunction.intent_function",
        )
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("IntentFunction.name must be a non-empty string")
        raw_inputs = fn.get("inputs")
        raw_outputs = fn.get("outputs")
        raw_returns = fn.get("returns")
        raw_body = fn.get("body")
        for value, field_name in (
            (raw_inputs, "inputs"),
            (raw_outputs, "outputs"),
            (raw_returns, "returns"),
            (raw_body, "body"),
        ):
            if not isinstance(value, list):
                raise TypeError(
                    f"IntentFunction.{field_name} must be an array"
                )

        parsed = cls(
            name=name,
            inputs=[Tensor.from_json(t) for t in raw_inputs],
            outputs=[Tensor.from_json(t) for t in raw_outputs],
            returns=[IntentReturn.from_json(item) for item in raw_returns],
            body=[IntentOp.from_json(o) for o in raw_body],
        )

        # A tensor is declared once in first-declaration order.  An inout may
        # occur in both collections, but the second occurrence is the same
        # logical t* declaration and must have an identical contract.
        parsed._validate_tensor_declarations()
        validate_intent_function_wiring(parsed)
        validate_ragged_region_boundaries(parsed.body)
        return parsed

    @classmethod
    def from_json_str(cls, s: str) -> "IntentFunction":
        return cls.from_json(json.loads(s))

    def to_json(self) -> dict:
        # Direct typed callers do not pass through ``from_json``.  Make the
        # serialization boundary perform the same complete validation so an
        # out-of-scope operand, an incomplete conditional, or a non-dense
        # logical ABI cannot be handed to MLIR/backends from an in-memory J.
        self._validate_tensor_declarations()
        validate_intent_function_wiring(self)
        validate_ragged_region_boundaries(self.body)
        function = {
            "name": self.name,
            "inputs": [t.to_json() for t in self.inputs],
            "outputs": [t.to_json() for t in self.outputs],
            "returns": [item.to_json() for item in self.returns],
            "body": [o.to_json() for o in self.body],
        }
        return {"intent_function": function}

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_json(), indent=indent)


def _intent_result_names(operation: IntentOp) -> tuple[str, ...]:
    if isinstance(operation.result, str):
        return (operation.result,)
    if isinstance(operation.result, list):
        return tuple(
            value for value in operation.result if isinstance(value, str)
        )
    return ()


def _intent_declared_type(
    operation: IntentOp,
    *,
    context: str,
) -> ValueType | None:
    if operation.type is None:
        return None
    try:
        return ValueType.from_json(operation.type, context=f"{context}.type")
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _intent_result_type(
    operation: IntentOp,
    result_index: int,
    *,
    context: str,
) -> ValueType | None:
    if operation.op == "loop":
        if result_index >= len(operation.carry):
            return None
        state_type = operation.carry[result_index].type
        return state_type.as_value_type() if state_type is not None else None
    return _intent_declared_type(operation, context=context)


def validate_intent_function_wiring(function: IntentFunction) -> None:
    """Validate one parsed function's lexical and structured-value wiring.

    This is intentionally a small value-graph validator rather than a second
    operation type inferencer.  It owns the contracts that every consumer must
    agree on: inputs dominate the top-level body, operation results are local
    ``v*`` bindings, conditional-arm definitions do not escape, loop bindings
    are local, and every return names a declared output in declaration order.
    When an operation supplies an explicit type, the validator uses it to
    check boundary compatibility; it does not invent types for an operation
    whose normal type checker has not inferred one yet.
    """

    input_types: dict[str, ValueType | None] = {
        tensor.name: ValueType.tensor(tensor.dtype, tensor.shape)
        for tensor in function.inputs
    }
    output_types: dict[str, ValueType] = {
        tensor.name: ValueType.tensor(tensor.dtype, tensor.shape)
        for tensor in function.outputs
    }

    def check_predicate(
        predicate: str | None,
        visible: Mapping[str, ValueType | None],
        context: str,
    ) -> None:
        if not isinstance(predicate, str) or predicate not in visible:
            raise ValueError(
                f"{context}.predicate must reference a visible logical value"
            )
        predicate_type = visible[predicate]
        if predicate_type is None:
            return
        if (
            not predicate_type.is_tensor
            or predicate_type.shape
            or predicate_type.dtype != "i1"
        ):
            raise ValueError(
                f"{context}.predicate must be a rank-0 i1 logical value"
            )

    def validate_sequence(
        operations: Sequence[IntentOp],
        inherited: Mapping[str, ValueType | None],
        sequence_path: str,
    ) -> tuple[dict[str, ValueType | None], dict[str, ValueType | None]]:
        visible = dict(inherited)
        definitions: dict[str, ValueType | None] = {}
        for index, operation in enumerate(operations):
            context = f"{sequence_path}[{index}]"
            names = _intent_result_names(operation)

            if operation.op in {"effect_if", "value_if"}:
                check_predicate(operation.predicate, visible, context)
                arm_definitions: dict[str, dict[str, ValueType | None]] = {}
                arm_yield_types: dict[str, ValueType | None] = {}
                for arm_name, arm in (
                    ("positive", operation.positive),
                    ("negative", operation.negative),
                ):
                    if arm is None:
                        raise ValueError(
                            f"{context}.{arm_name} must be an explicit region"
                        )
                    arm_visible, branch_definitions = validate_sequence(
                        arm.body,
                        visible,
                        f"{context}.{arm_name}.body",
                    )
                    arm_definitions[arm_name] = branch_definitions
                    if operation.op == "value_if":
                        yielded = arm.yield_value
                        if yielded not in arm_visible:
                            raise ValueError(
                                f"{context}.{arm_name}.yield_value references a "
                                "value outside that arm or its dominating scope"
                            )
                        arm_yield_types[arm_name] = arm_visible.get(yielded)

                if (
                    set(arm_definitions["positive"])
                    & set(arm_definitions["negative"])
                ):
                    overlap = sorted(
                        set(arm_definitions["positive"])
                        & set(arm_definitions["negative"])
                    )
                    raise ValueError(
                        f"{context} positive/negative arms define duplicate "
                        "branch-local values: "
                        + ", ".join(overlap)
                    )
                if operation.op == "effect_if":
                    # No arm definition is exported by an effect-only region.
                    continue

                if len(names) != 1:
                    raise ValueError(
                        f"{context}.result must contain exactly one value_if result"
                    )
                result_type = _intent_result_type(
                    operation, 0, context=context
                )
                if result_type is None:
                    raise ValueError(
                        f"{context}.type must describe the value_if result"
                    )
                positive_type = arm_yield_types["positive"]
                negative_type = arm_yield_types["negative"]
                if (
                    positive_type is not None
                    and positive_type != result_type
                ) or (
                    negative_type is not None
                    and negative_type != result_type
                ):
                    raise ValueError(
                        f"{context} arm yield types must match its result type"
                    )
                if names[0] in visible:
                    raise ValueError(
                        f"{context}.result redefines visible value {names[0]!r}"
                    )
                visible[names[0]] = result_type
                definitions[names[0]] = result_type
                continue

            if operation.op == "loop":
                validate_loop_wiring(
                    operation,
                    context=context,
                    visible_values=visible,
                )
                body_types = dict(visible)
                for scan in operation.scan:
                    source_type = visible.get(scan.source)
                    if source_type is not None and not source_type.is_tensor:
                        raise ValueError(
                            f"{context}.scan source {scan.source!r} must be a "
                            "tensor value"
                        )
                    body_types[scan.name] = source_type
                for state in operation.carry:
                    if state.type is None:
                        raise ValueError(
                            f"{context}.carry state {state.state_id!r} requires a type"
                        )
                    body_types[state.body_argument] = state.type.as_value_type()
                if operation.trip.is_dynamic:
                    assert operation.trip.index is not None
                    body_types[operation.trip.index] = visible.get(
                        operation.trip.upper_bound or ""
                    )
                _body_visible, body_definitions = validate_sequence(
                    operation.body,
                    body_types,
                    f"{context}.body",
                )
                for state in operation.carry:
                    carry_type = state.type.as_value_type()  # checked above
                    next_type = body_definitions.get(state.next_value)
                    if next_type is None:
                        raise ValueError(
                            f"{context}.carry next_value {state.next_value!r} "
                            "requires an explicitly typed body producer"
                        )
                    if next_type != carry_type:
                        raise ValueError(
                            f"{context}.carry next_value {state.next_value!r} "
                            "must have exactly the carry state type"
                        )
                for result_index, name in enumerate(names):
                    result_type = _intent_result_type(
                        operation, result_index, context=context
                    )
                    if result_type is None:
                        raise ValueError(
                            f"{context}.result {name!r} has no carry type"
                        )
                    if name in visible:
                        raise ValueError(
                            f"{context}.result redefines visible value {name!r}"
                        )
                    visible[name] = result_type
                    definitions[name] = result_type
                continue

            for operand_index, operand in enumerate(operation.operands):
                if operand not in visible:
                    raise ValueError(
                        f"{context}.operands[{operand_index}] references a "
                        f"non-visible logical value {operand!r}"
                    )
            declared_type = _intent_declared_type(operation, context=context)
            for result_index, name in enumerate(names):
                if name in visible:
                    raise ValueError(
                        f"{context}.result redefines visible value {name!r}"
                    )
                result_type = (
                    declared_type if len(names) == 1 else None
                )
                visible[name] = result_type
                definitions[name] = result_type
        return visible, definitions

    visible_after, _definitions = validate_sequence(
        function.body,
        input_types,
        "IntentFunction.body",
    )
    output_names = [tensor.name for tensor in function.outputs]
    returned_outputs: list[str] = []
    for index, binding in enumerate(function.returns):
        context = f"IntentFunction.returns[{index}]"
        if binding.output not in output_types:
            raise ValueError(
                f"{context}.output {binding.output!r} is not a declared output"
            )
        if binding.value not in visible_after:
            raise ValueError(
                f"{context}.value {binding.value!r} is not visible after the body"
            )
        value_type = visible_after[binding.value]
        output_type = output_types[binding.output]
        if value_type is not None and value_type != output_type:
            raise ValueError(
                f"{context}.value type does not match output "
                f"{binding.output!r}: expected {output_type.to_json()}, "
                f"got {value_type.to_json()} for {binding.value!r}"
            )
        returned_outputs.append(binding.output)
    if returned_outputs != output_names:
        raise ValueError(
            "IntentFunction.returns must cover declared outputs exactly once "
            "and in output declaration order"
        )


def validate_ragged_region_boundaries(
    body: Sequence[IntentOp], *, path: str = "body"
) -> None:
    """Enforce the closed Ragged descriptor region contract.

    Ragged descriptors are useful as short-lived Layer-A values: a region may
    construct one from its two carrier tensors and immediately query it with
    ``ragged_outer`` or ``ragged_member``.  The current MLIR/native ABI does
    not flatten a tuple descriptor through structured control-flow or a
    function boundary, however.  Keep that limitation explicit at the shared
    model boundary so the canonical parser, emitters, and backends cannot
    disagree about a late tuple-ABI failure.

    The check tracks lexical descriptor names so it also catches an arm or loop
    yielding a descriptor that was captured from an enclosing region.  It does
    not reject a Ragged operation inside a region when its descriptor is
    consumed before the region returns.
    """

    def result_names(operation: IntentOp) -> tuple[str, ...]:
        if isinstance(operation.result, str):
            return (operation.result,)
        if isinstance(operation.result, list):
            return tuple(
                name for name in operation.result if isinstance(name, str)
            )
        return ()

    def is_ragged_type(value_type: Any) -> bool:
        return (
            isinstance(value_type, dict)
            and value_type.get("kind") == "ragged"
        )

    def visit(
        operations: Sequence[IntentOp],
        inherited_descriptors: set[str],
        sequence_path: str,
    ) -> set[str]:
        descriptors = set(inherited_descriptors)
        for index, operation in enumerate(operations):
            operation_path = f"{sequence_path}[{index}]"
            names = result_names(operation)

            if operation.op == "loop":
                if any(
                    state.type is not None and state.type.kind == "ragged"
                    for state in operation.carry
                ):
                    raise ValueError(
                        f"{operation_path} Ragged descriptors cannot cross a "
                        "loop carry/result boundary"
                    )
                if is_ragged_type(operation.type):
                    raise ValueError(
                        f"{operation_path} cannot expose a Ragged loop result"
                    )
                body_descriptors = set(descriptors)
                if any(
                    isinstance(state.init_value, str)
                    and state.init_value in descriptors
                    for state in operation.carry
                ):
                    raise ValueError(
                        f"{operation_path} consumes a Ragged descriptor as a "
                        "loop carry initializer"
                    )
                body_descriptors = visit(
                    operation.body, body_descriptors, f"{operation_path}.body"
                )
                if any(
                    state.next_value in body_descriptors
                    for state in operation.carry
                ):
                    raise ValueError(
                        f"{operation_path} yields a Ragged descriptor across a "
                        "loop carry/result boundary"
                    )
                # Loop results are tensor/record values under the current ABI;
                # no descriptor produced in the body becomes visible here.
                continue

            if operation.op in {"effect_if", "value_if"}:
                if operation.op == "value_if" and is_ragged_type(operation.type):
                    raise ValueError(
                        f"{operation_path} Ragged descriptors cannot cross a "
                        "value_if result boundary"
                    )
                for arm_name, arm in (
                    ("positive", operation.positive),
                    ("negative", operation.negative),
                ):
                    if arm is None:
                        continue
                    arm_descriptors = set(descriptors)
                    arm_descriptors = visit(
                        arm.body,
                        arm_descriptors,
                        f"{operation_path}.{arm_name}.body",
                    )
                    if (
                        operation.op == "value_if"
                        and arm.yield_value in arm_descriptors
                    ):
                        raise ValueError(
                            f"{operation_path}.{arm_name} yields Ragged "
                            "descriptor across a value_if result boundary"
                        )
                # Neither conditional form exports arm-local descriptors.  A
                # value_if exports only its authored tensor/record result, and
                # effect_if is value-less at the parent continuation.
                continue

            if operation.op == "ragged" or is_ragged_type(operation.type):
                descriptors.update(names)

            # Ordinary operations do not currently own regions, but recurse on
            # an attached body so this helper remains aligned with the model's
            # region traversal for pre-validation objects.
            if operation.body:
                visit(operation.body, set(descriptors), f"{operation_path}.body")

        return descriptors

    visit(body, set(), path)


def intent_op_regions(
    operation: IntentOp,
) -> tuple[tuple[str, Sequence[IntentOp]], ...]:
    """Return every nested region owned by one IntentOp in canonical order.

    ``body`` is the structured region used by ordinary operations and loops.
    ``effect_if`` and ``value_if`` additionally own the positive and negative
    regions.  Keeping this relation in the Layer-A model prevents capability
    checks, node-path resolution, and reference execution from silently
    disagreeing about which operations belong to the J graph.
    """

    regions: list[tuple[str, Sequence[IntentOp]]] = [("body", operation.body)]
    if operation.op in {"effect_if", "value_if"}:
        if operation.positive is None or operation.negative is None:
            raise ValueError(
                f"{operation.op} must own both positive and negative regions"
            )
        for name, arm in (
            ("positive.body", operation.positive),
            ("negative.body", operation.negative),
        ):
            regions.append((name, arm.body))
    return tuple(regions)


def walk_intent_ops(body: Sequence[IntentOp]) -> Iterator[IntentOp]:
    """Walk a J operation sequence, including all nested conditional regions."""

    for operation in body:
        yield operation
        for _suffix, region in intent_op_regions(operation):
            yield from walk_intent_ops(region)


def walk_intent_op_scopes(
    body: Sequence[IntentOp],
    *,
    visible: Iterable[str] = (),
    prefix: str = "body",
) -> Iterator[tuple[str, IntentOp, dict[str, IntentOp | None]]]:
    """Walk J operations with the lexical bindings visible at each operation.

    The ordinary walker above is intentionally a structural view.  Runtime
    validators and workload materializers also need to know which producer a
    name denotes: a value defined in one conditional arm must not become
    visible in its sibling or in the parent continuation.  This companion
    walker carries a copy-on-entry binding environment through loop/conditional
    regions and yields ``(path, operation, visible_bindings)``.  ``None`` marks
    a function input (or another caller-provided binding); an ``IntentOp``
    value marks the operation that produced the visible name.

    Loop body arguments, scan names, and dynamic induction values are local
    bindings owned by the loop operation.  A loop exports only its authored
    result names.  A ``value_if`` exports its result, while an ``effect_if``
    exports no value; both arms are always visited and their local definitions
    are discarded on return to the parent sequence.
    """

    initial = {name: None for name in visible if isinstance(name, str)}

    def result_names(operation: IntentOp) -> tuple[str, ...]:
        if isinstance(operation.result, str):
            return (operation.result,)
        if isinstance(operation.result, list):
            return tuple(
                name for name in operation.result if isinstance(name, str)
            )
        return ()

    def visit(
        operations: Sequence[IntentOp],
        inherited: dict[str, IntentOp | None],
        sequence_prefix: str,
    ) -> Iterator[tuple[str, IntentOp, dict[str, IntentOp | None]]]:
        bindings = dict(inherited)
        for index, operation in enumerate(operations):
            path = f"{sequence_prefix}[{index}]"
            before = dict(bindings)
            yield path, operation, before

            if operation.op in {"effect_if", "value_if"}:
                # ``intent_op_regions`` also exposes the ordinary body.  It is
                # empty for a valid conditional, but visiting it here keeps
                # this scope-aware view aligned with the structural walker if
                # a caller is inspecting a not-yet-validated object.
                for suffix, region in intent_op_regions(operation):
                    yield from visit(region, before, f"{path}.{suffix}")
                if operation.op == "value_if":
                    for name in result_names(operation):
                        bindings[name] = operation
                continue

            if operation.op == "loop":
                loop_bindings = dict(before)
                for scan in operation.scan:
                    loop_bindings[scan.name] = operation
                for state in operation.carry:
                    loop_bindings[state.body_argument] = operation
                if operation.trip is not None and operation.trip.is_dynamic:
                    if operation.trip.index is not None:
                        loop_bindings[operation.trip.index] = operation
                yield from visit(
                    operation.body,
                    loop_bindings,
                    f"{path}.body",
                )
                for name in result_names(operation):
                    bindings[name] = operation
                continue

            # Ordinary operations normally have no region.  Keep the fallback
            # recursive so this helper cannot silently diverge from
            # ``walk_intent_ops`` on an object that is being diagnosed before
            # structural validation.
            if operation.body:
                yield from visit(operation.body, before, f"{path}.body")
            for name in result_names(operation):
                bindings[name] = operation

    yield from visit(body, initial, prefix)


def intent_op_producers(
    body: Sequence[IntentOp],
    *,
    visible: Iterable[str] = (),
) -> dict[str, IntentOp | None]:
    """Return the bindings exported by one lexical operation sequence.

    Unlike :func:`walk_intent_ops`, this map does not expose definitions owned
    by a loop body or conditional arm.  It is therefore safe for output-shape
    resolution and other parent-level consumers that need a result producer by
    name without accidentally selecting a same-named sibling definition.
    """

    initial = {name: None for name in visible if isinstance(name, str)}

    def result_names(operation: IntentOp) -> tuple[str, ...]:
        if isinstance(operation.result, str):
            return (operation.result,)
        if isinstance(operation.result, list):
            return tuple(
                name for name in operation.result if isinstance(name, str)
            )
        return ()

    def visit(
        operations: Sequence[IntentOp],
        inherited: dict[str, IntentOp | None],
    ) -> dict[str, IntentOp | None]:
        bindings = dict(inherited)
        for operation in operations:
            if operation.op in {"effect_if", "value_if"}:
                for _suffix, region in intent_op_regions(operation):
                    visit(region, bindings)
                if operation.op == "value_if":
                    for name in result_names(operation):
                        bindings[name] = operation
                continue
            if operation.op == "loop":
                loop_bindings = dict(bindings)
                for scan in operation.scan:
                    loop_bindings[scan.name] = operation
                for state in operation.carry:
                    loop_bindings[state.body_argument] = operation
                if operation.trip is not None and operation.trip.is_dynamic:
                    if operation.trip.index is not None:
                        loop_bindings[operation.trip.index] = operation
                visit(operation.body, loop_bindings)
            elif operation.body:
                visit(operation.body, bindings)
            for name in result_names(operation):
                bindings[name] = operation
        return bindings

    return visit(body, initial)


def loop_result_names(operation: IntentOp) -> tuple[str, ...]:
    """Return the explicitly exported result handles of one operation."""

    if isinstance(operation.result, str):
        return (operation.result,)
    if isinstance(operation.result, list):
        return tuple(
            value for value in operation.result if isinstance(value, str)
        )
    return ()


def validate_loop_wiring(
    operation: IntentOp,
    *,
    context: str = "loop",
    visible_values: Iterable[str] | Mapping[str, ValueType | None] | None = None,
    value_types: Mapping[str, ValueType | None] | None = None,
) -> dict[str, IntentOp]:
    """Validate the lexical ``carry -> body -> yield -> result`` loop wire.

    This check is deliberately independent of dtype and shape inference.  It
    is the common ownership rule consumed by canonical J, MLIR, and the CPU
    structured reference:

    ``init_value -> body_argument -> body producer -> next_value -> yield``
    and ``loop.result == carry.result_value``.  A value defined inside a
    conditional arm is not present in :func:`intent_op_producers` for the
    enclosing body sequence, so it cannot accidentally be exported as a loop
    carry update.

    The strict Gate1 wire validator performs the same checks while it still
    has JSON paths available.  Keeping this typed check here closes the gap
    for callers that receive an already parsed :class:`IntentOp` directly.
    """

    if operation.op != "loop":
        raise ValueError(f"{context} wiring check requires op='loop'")
    if operation.trip is None:
        raise ValueError(f"{context} requires a trip binding")
    if not operation.carry:
        raise ValueError(f"{context} requires at least one carry state")
    if not isinstance(operation.body, list) or not operation.body:
        raise ValueError(f"{context} requires a non-empty body")

    visible_names = (
        set(visible_values) if visible_values is not None else None
    )
    known_types: dict[str, ValueType | None] = dict(value_types or {})
    if isinstance(visible_values, Mapping):
        for name, value_type in visible_values.items():
            known_types.setdefault(name, value_type)

    trip = operation.trip
    if trip.is_dynamic:
        if (
            not isinstance(trip.upper_bound, str)
            or _VALUE_HANDLE.fullmatch(trip.upper_bound) is None
            or not isinstance(trip.index, str)
            or _SSA_HANDLE.fullmatch(trip.index) is None
        ):
            raise ValueError(
                f"{context}.trip dynamic form requires upper_bound=t*/v* "
                "and index=v*"
            )
        if visible_names is not None:
            if trip.upper_bound not in visible_names:
                raise ValueError(
                    f"{context}.trip.upper_bound must reference an outer visible "
                    f"value: {trip.upper_bound!r}"
                )
            if trip.index in visible_names:
                raise ValueError(
                    f"{context}.trip.index must be local to the loop scope: "
                    f"{trip.index!r}"
                )
        upper_type = known_types.get(trip.upper_bound)
        if upper_type is not None and (
            not upper_type.is_tensor
            or upper_type.shape
            or not is_non_boolean_integer_dtype(upper_type.dtype)
        ):
            raise ValueError(
                f"{context}.trip.upper_bound must be a rank-0 non-boolean "
                "integer logical value"
            )
    elif (
        not isinstance(trip.axis, str)
        or _DIMENSION_HANDLE.fullmatch(trip.axis) is None
    ):
        raise ValueError(f"{context}.trip.axis must be a symbolic x* dimension")

    expected_results = tuple(state.result_value for state in operation.carry)
    expected_yields = tuple(state.next_value for state in operation.carry)
    actual_results = loop_result_names(operation)
    actual_yields = tuple(operation.yields)
    if actual_results != expected_results:
        raise ValueError(
            f"{context}.result must exactly match carry.result_value order"
        )
    if actual_yields != expected_yields:
        raise ValueError(
            f"{context}.yield must exactly match carry.next_value order"
        )
    scan_names = {scan.name for scan in operation.scan}
    if len(operation.scan) != len(scan_names):
        raise ValueError(f"{context}.scan names must be unique")
    if any(
        not isinstance(scan.name, str)
        or _SSA_HANDLE.fullmatch(scan.name) is None
        or not isinstance(scan.source, str)
        or _VALUE_HANDLE.fullmatch(scan.source) is None
        for scan in operation.scan
    ):
        raise ValueError(f"{context}.scan names/from values must use v*/t* handles")
    if visible_names is not None:
        missing_sources = sorted(
            scan.source for scan in operation.scan
            if scan.source not in visible_names
        )
        if missing_sources:
            raise ValueError(
                f"{context}.scan source(s) must be outer visible values: "
                + ", ".join(missing_sources)
            )
        shadowed_scans = sorted(scan_names & visible_names)
        if shadowed_scans:
            raise ValueError(
                f"{context}.scan name(s) shadow outer values: "
                + ", ".join(shadowed_scans)
            )

    state_ids = [state.state_id for state in operation.carry]
    if any(
        not isinstance(state_id, str)
        or _LOOP_STATE_HANDLE.fullmatch(state_id) is None
        for state_id in state_ids
    ):
        raise ValueError(
            f"{context}.carry state_id values must use s* loop-state handles"
        )
    if len(state_ids) != len(set(state_ids)):
        raise ValueError(f"{context}.carry state_id values must be unique")

    role_names = [
        role
        for state in operation.carry
        for role in (
            state.body_argument,
            state.next_value,
            state.result_value,
        )
    ]
    if any(not isinstance(name, str) or not name for name in role_names):
        raise ValueError(
            f"{context}.carry body_argument/next_value/result_value must be names"
        )
    if any(_SSA_HANDLE.fullmatch(name) is None for name in role_names):
        raise ValueError(
            f"{context}.carry body_argument/next_value/result_value must use v*"
        )
    if len(role_names) != len(set(role_names)):
        raise ValueError(
            f"{context}.carry value roles must be distinct across all states"
        )
    if any(state.init_value is None for state in operation.carry):
        raise ValueError(
            f"{context}.carry init_value must be an authored literal or "
            "dominating value reference"
        )
    for state in operation.carry:
        if state.type is None:
            raise ValueError(
                f"{context}.carry state {state.state_id!r} requires an explicit type"
            )
        if state.type.kind == "ragged":
            raise ValueError(
                f"{context}.carry state {state.state_id!r} cannot carry a Ragged value"
            )
        if not is_scalar_literal_or_value_reference(state.init_value):
            raise ValueError(
                f"{context}.carry state {state.state_id!r} has an invalid init_value"
            )
        if is_scalar_literal(state.init_value):
            if state.type.kind != "tensor" or state.type.dtype is None:
                raise ValueError(
                    f"{context}.carry state {state.state_id!r} scalar init_value "
                    "requires a tensor carry type"
                )
            try:
                validate_const_literal(
                    {
                        "value": state.init_value,
                        "dtype": state.type.dtype,
                        "shape": list(state.type.shape),
                    },
                    result_type=state.type.to_json(),
                    context=(
                        f"{context}.carry state {state.state_id!r}.init_value"
                    ),
                )
            except IntentOpContractError as exc:
                raise ValueError(str(exc)) from exc
        elif isinstance(state.init_value, str):
            if _VALUE_HANDLE.fullmatch(state.init_value) is None:
                raise ValueError(
                    f"{context}.carry state {state.state_id!r} init_value must be "
                    "a logical t*/v* reference or scalar literal"
                )
            if (
                visible_names is not None
                and state.init_value not in visible_names
            ):
                raise ValueError(
                    f"{context}.carry state {state.state_id!r} init_value "
                    f"{state.init_value!r} is not outer visible"
                )
            init_type = known_types.get(state.init_value)
            if init_type is not None and init_type != state.type.as_value_type():
                raise ValueError(
                    f"{context}.carry state {state.state_id!r} type must match "
                    "its init_value type"
                )

    local_binding_names = set(scan_names)
    local_binding_names.update(state.body_argument for state in operation.carry)
    if trip.is_dynamic:
        assert trip.index is not None
        local_binding_names.add(trip.index)
    if len(local_binding_names) != (
        len(scan_names) + len(operation.carry) + (1 if trip.is_dynamic else 0)
    ):
        raise ValueError(
            f"{context} scan/carry/induction bindings must be pairwise distinct"
        )
    local_non_carry_bindings = set(scan_names)
    if trip.is_dynamic:
        assert trip.index is not None
        local_non_carry_bindings.add(trip.index)
    role_export_collisions = sorted(
        {
            name
            for state in operation.carry
            for name in (state.next_value, state.result_value)
            if name in local_non_carry_bindings
        }
    )
    if role_export_collisions:
        raise ValueError(
            f"{context}.carry next_value/result_value must be distinct from "
            "loop-local scan/induction bindings: "
            + ", ".join(role_export_collisions)
        )
    if visible_names is not None:
        shadowed_local_bindings = sorted(local_binding_names & visible_names)
        if shadowed_local_bindings:
            raise ValueError(
                f"{context} local scan/carry/induction binding(s) shadow outer "
                "values: " + ", ".join(shadowed_local_bindings)
            )
        shadowed_results = sorted(
            state.result_value
            for state in operation.carry
            if state.result_value in visible_names
        )
        if shadowed_results:
            raise ValueError(
                f"{context}.carry result_value(s) redefine outer values: "
                + ", ".join(shadowed_results)
            )

    direct_producers = {
        name: producer
        for name, producer in intent_op_producers(operation.body).items()
        if producer is not None
    }
    if scan_names & direct_producers.keys():
        overlap = ", ".join(sorted(scan_names & direct_producers.keys()))
        raise ValueError(
            f"{context}.body redefines loop scan binding(s): {overlap}"
        )
    for state in operation.carry:
        if state.body_argument in direct_producers:
            raise ValueError(
                f"{context}.body redefines carry body_argument "
                f"{state.body_argument!r}"
            )
        if state.next_value not in direct_producers:
            raise ValueError(
                f"{context}.carry next_value {state.next_value!r} must be "
                "defined by a direct producer in the loop body"
            )
        if state.result_value in direct_producers:
            raise ValueError(
                f"{context}.body must not define loop result_value "
                f"{state.result_value!r}"
            )

    def direct_producer_type(value_name: str) -> ValueType | None:
        for child_index, child in enumerate(operation.body):
            names = loop_result_names(child)
            if value_name not in names:
                continue
            result_index = names.index(value_name)
            if child.op == "loop":
                if result_index >= len(child.carry):
                    return None
                child_type = child.carry[result_index].type
                return child_type.as_value_type() if child_type is not None else None
            return _intent_result_type(
                child,
                result_index,
                context=f"{context}.body[{child_index}]",
            )
        return None

    for state_index, state in enumerate(operation.carry):
        next_type = direct_producer_type(state.next_value)
        if next_type is None:
            raise ValueError(
                f"{context}.carry next_value {state.next_value!r} requires an "
                "explicitly typed direct body producer"
            )
        if next_type != state.type.as_value_type():
            raise ValueError(
                f"{context}.carry[{state_index}].next_value "
                f"{state.next_value!r} direct producer type "
                f"{next_type.to_json()} differs from declared carry type "
                f"{state.type.as_value_type().to_json()}; next_value must have "
                "exactly the carry state type"
            )
    return direct_producers


def walk_intent_op_paths(
    body: Sequence[IntentOp],
    prefix: str = "body",
) -> Iterator[tuple[str, IntentOp]]:
    """Walk J operations together with the paths used by ``node_ids``."""

    for index, operation in enumerate(body):
        path = f"{prefix}[{index}]"
        yield path, operation
        for suffix, region in intent_op_regions(operation):
            yield from walk_intent_op_paths(region, f"{path}.{suffix}")
