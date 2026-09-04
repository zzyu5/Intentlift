"""Declarative Harness adapters for the active real Triton representatives.

These records bind actual buffers, scalars, constexpr values, and launch grids.
They deliberately contain no J operators, predicates, formulas, or Replay text.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from intentir.ir.canonical_j import elaborate_j
from intentir.ir.contract_ir import project_contract_ir
from intentir.ir.layer_c import (
    binding_interface_digest,
    layer_c_content_digest,
    source_layer_c_interface,
)
from intentir.ir.dimension_relation import (
    DIMENSION_INT64_MAX,
    DIMENSION_RELATION_CONST_OPS,
    DIMENSION_RELATION_OPS,
    DimensionRelationError,
    evaluate_dimension_relation_operation,
)
from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    SCRCCompilerView,
    anonymize_source,
)
from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
)
from intentir.verify.gate1.model import DimensionRelation, ValueDimensionRelation
from intentir.verify.gate2.model import mapping_digest
from intentir.verify.scrc_runner import (
    HarnessBindingInstance,
    SCRCRunRequest,
    SourceProgram,
)
from intentir.verify.triton_source_slice import (
    TritonSourceSlice,
    build_triton_source_slice,
    expand_triton_varargs,
    inline_imported_triton_helpers,
    requires_triton_vararg_expansion,
)
from intentir.verify.workload_manifest import (
    SCRCWorkloadEntry,
    build_workload_manifest,
)


TRITON_WORKLOAD_ADAPTER_PROTOCOL = "intentir.scrc.triton-workload-adapter.v6"


class BufferRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class BufferInitializer(str, Enum):
    RANDOM = "random"
    STANDARD_NORMAL = "standard_normal"
    ZEROS = "zeros"


@dataclass(frozen=True)
class BufferBinding:
    formal: str
    role: BufferRole
    dtype: str
    shape: tuple[int, ...]
    output_key: str | None = None
    initializer: BufferInitializer | None = None
    shape_contract: tuple[int | str, ...] | None = None
    # Physical element strides belong to the captured Harness valuation, not J.
    # Keeping them here lets Source and Replay receive the same real view while
    # RefJ continues to consume the logical tensor value.
    strides: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.formal.isidentifier():
            raise ValueError("buffer formal must be a Python identifier")
        if not isinstance(self.role, BufferRole):
            object.__setattr__(self, "role", BufferRole(self.role))
        if not self.dtype:
            raise ValueError("buffer dtype must be non-empty")
        shape = tuple(int(dim) for dim in self.shape)
        # A rank-zero tensor is a real pointer-backed scalar output/input (for
        # example dot and global reductions), not a scalar ABI argument.
        if any(dim < 0 for dim in shape):
            raise ValueError("real workload buffer dimensions must be non-negative")
        if any(dim == 0 for dim in shape) and self.role is not BufferRole.OUTPUT:
            raise ValueError(
                "zero-length real workload buffers are allowed only for outputs"
            )
        object.__setattr__(self, "shape", shape)
        shape_contract = (
            tuple(self.shape_contract)
            if self.shape_contract is not None
            else None
        )
        if shape_contract is not None:
            if len(shape_contract) != len(shape):
                raise ValueError(
                    "buffer shape contract rank differs from concrete valuation"
                )
            for contract, extent in zip(shape_contract, shape, strict=True):
                if isinstance(contract, bool) or not isinstance(contract, (int, str)):
                    raise TypeError(
                        "buffer shape contract entries must be integers or h* coordinates"
                    )
                if isinstance(contract, int):
                    if contract < 0 or contract != extent:
                        raise ValueError(
                            "static buffer shape contract differs from concrete valuation"
                        )
                    if contract == 0 and self.role is not BufferRole.OUTPUT:
                        raise ValueError(
                            "zero-length buffer shape contracts are allowed only for outputs"
                        )
                elif not contract.startswith("h") or not contract[1:].isdigit():
                    raise ValueError(
                        "buffer shape contract coordinates must use anonymous h* handles"
                    )
        object.__setattr__(self, "shape_contract", shape_contract)
        strides = tuple(int(value) for value in self.strides) if self.strides is not None else None
        if strides is not None:
            if len(strides) != len(shape):
                raise ValueError("buffer physical strides must match its rank")
            if any(value < 0 for value in strides):
                raise ValueError("captured negative strides require an explicit storage offset")
        object.__setattr__(self, "strides", strides)
        if self.role is BufferRole.INPUT and self.output_key is not None:
            raise ValueError("input-only buffer cannot declare an output key")
        if self.role is not BufferRole.INPUT and self.output_key is None:
            object.__setattr__(self, "output_key", self.formal.casefold())
        if self.role is BufferRole.OUTPUT:
            if self.initializer is not None:
                raise ValueError("output-only buffer cannot declare an initializer")
        elif self.initializer is None:
            object.__setattr__(self, "initializer", BufferInitializer.RANDOM)
        elif not isinstance(self.initializer, BufferInitializer):
            object.__setattr__(
                self, "initializer", BufferInitializer(self.initializer)
            )


@dataclass(frozen=True)
class ScalarBinding:
    formal: str
    dtype: str
    value: bool | int | float
    shape_coordinate: str | None = None

    def __post_init__(self) -> None:
        if not self.formal.isidentifier():
            raise ValueError("scalar formal must be a Python identifier")
        if not self.dtype:
            raise ValueError("scalar dtype must be non-empty")
        if not isinstance(self.value, (bool, int, float)):
            raise TypeError("runtime scalar value must be bool/int/float")
        if self.shape_coordinate is not None:
            if (
                not isinstance(self.shape_coordinate, str)
                or not self.shape_coordinate.startswith("h")
                or not self.shape_coordinate[1:].isdigit()
            ):
                raise ValueError(
                    "scalar shape-coordinate binding must use an anonymous h* handle"
                )
            if isinstance(self.value, bool) or not isinstance(self.value, int):
                raise ValueError(
                    "scalar shape-coordinate binding requires an integer valuation"
                )
            if self.dtype.casefold() != "index" and re.fullmatch(
                r"[iu][0-9]+", self.dtype.casefold()
            ) is None:
                raise ValueError(
                    "scalar shape-coordinate binding requires an integer dtype"
                )


@dataclass(frozen=True)
class NoneBinding:
    formal: str
    dtype: str = "none"

    def __post_init__(self) -> None:
        if not self.formal.isidentifier():
            raise ValueError("None formal must be a Python identifier")
        if self.dtype != "none":
            raise ValueError("None binding dtype must be exactly 'none'")


_GENERATED_SOURCE_MANIFEST_PROTOCOL = "intentir.generated-target-source-manifest"
_MANIFEST_SHA256 = re.compile(r"[0-9a-f]{64}")
_GENERATED_SOURCE_LIFECYCLE_PHASES = (
    "source_compile",
    "source_execute",
    "output_digest",
    "reingress",
)
_MANIFEST_PARAMETER_CATEGORIES = frozenset(
    {
        "buffer",
        "runtime_scalar",
        "none",
        "opaque",
        "semantic_constexpr",
        "c_slot",
    }
)
_MANIFEST_VALUE_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.:+/-]*")


def _manifest_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{context} must be a Python identifier")
    return value


def _manifest_token(value: Any, context: str) -> str:
    if not isinstance(value, str) or _MANIFEST_VALUE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a closed token")
    return value


def _manifest_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _MANIFEST_SHA256.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _manifest_scalar(value: Any, context: str) -> bool | int | float | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} must be finite")
        return value
    if isinstance(value, str):
        return _manifest_token(value, context)
    raise ValueError(f"{context} must be a closed JSON scalar")


def _manifest_names(
    values: Iterable[Any],
    *,
    context: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{context} must be an array")
    try:
        result = tuple(_manifest_identifier(item, context) for item in values)
    except TypeError as exc:
        raise ValueError(f"{context} must be an array") from exc
    if len(result) != len(set(result)):
        raise ValueError(f"{context} must contain unique names")
    return result


def _manifest_pairs(
    values: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    context: str,
) -> tuple[tuple[str, bool | int | float | str], ...]:
    if isinstance(values, Mapping):
        items = tuple(values.items())
    else:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{context} must be an object")
        try:
            items = tuple(values)
        except TypeError as exc:
            raise ValueError(f"{context} must be an object") from exc
    result: list[tuple[str, bool | int | float | str]] = []
    names: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"{context}[{index}] must be a name/value pair")
        name = _manifest_identifier(item[0], f"{context}[{index}].name")
        if name in names:
            raise ValueError(f"{context} contains duplicate name {name!r}")
        names.add(name)
        result.append(
            (name, _manifest_scalar(item[1], f"{context}.{name}"))
        )
    return tuple(result)


def _manifest_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be an object")
    return dict(value)


def _manifest_ordered_pairs(
    value: Any,
    order: tuple[str, ...],
    *,
    context: str,
) -> tuple[tuple[str, Any], ...]:
    raw = _manifest_object(value, context)
    if set(raw) != set(order):
        raise ValueError(f"{context} keys must exactly match its declared order")
    return tuple((name, raw[name]) for name in order)


def _manifest_value_equal(left: Any, right: Any) -> bool:
    """Compare JSON scalar valuations without conflating bool and int."""

    return type(left) is type(right) and left == right


def _manifest_c_slot_records(
    c: Any,
) -> tuple[tuple[str, str, str, str, bool, bool | int | float | str], ...]:
    """Read the declaration-order C slots without depending on their class.

    Gate1 keeps the historical ``ImplementationConfig`` wire, while this
    module also accepts its mapping-shaped representation.  The generated
    source manifest must compare the two at the boundary, so it intentionally
    preserves declaration order instead of sorting anonymous ``c*`` handles.
    """

    if isinstance(c, Mapping):
        slots = c.get("slots")
        values = c.get("values")
    else:
        slots = getattr(c, "slots", None)
        values = getattr(c, "values", None)
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise ValueError("C slot lineage requires an ordered slots array")
    if not isinstance(values, Mapping):
        raise ValueError("C slot lineage requires a values object")

    records: list[tuple[str, str, str, str, bool, bool | int | float | str]] = []
    declared: set[str] = set()
    for index, raw_slot in enumerate(slots):
        if isinstance(raw_slot, Mapping):
            name = raw_slot.get("name")
            kind = raw_slot.get("kind")
            role = raw_slot.get("role")
            value_type = raw_slot.get("value_type")
            required = raw_slot.get("required")
        else:
            name = getattr(raw_slot, "name", None)
            kind = getattr(raw_slot, "kind", None)
            role = getattr(raw_slot, "role", None)
            value_type = getattr(raw_slot, "value_type", None)
            required = getattr(raw_slot, "required", None)
        kind = getattr(kind, "value", kind)
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name in declared
            or kind not in {"constexpr", "compiler"}
            or not isinstance(role, str)
            or not role
            or not isinstance(value_type, str)
            or not value_type
            or not isinstance(required, bool)
        ):
            raise ValueError(f"C slot lineage declaration {index} is invalid")
        if name not in values:
            raise ValueError(f"C slot lineage is missing value for {name!r}")
        value = _manifest_scalar(values[name], f"C slot lineage value {name}")
        declared.add(name)
        records.append((name, kind, role, value_type, required, value))
    unknown = sorted(set(values) - declared)
    if unknown:
        raise ValueError(
            "C slot lineage contains values without declarations: "
            + ", ".join(unknown)
        )
    return tuple(records)


def _manifest_optional_int_sequence(
    value: Any,
    *,
    rank: int | None,
    context: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{context} must be null or an array")
    if rank is not None and len(value) != rank:
        raise ValueError(f"{context} rank differs from its tensor shape")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{context}[{index}] must be a non-negative integer")
        result.append(item)
    return tuple(result)


def _manifest_shape_contract(
    value: Any,
    *,
    rank: int | None,
    context: str,
) -> tuple[int | str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{context} must be null or an array")
    if rank is not None and len(value) != rank:
        raise ValueError(f"{context} rank differs from its tensor shape")
    result: list[int | str] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{context}[{index}] must be an integer or h* coordinate")
        if isinstance(item, int):
            if item < 0:
                raise ValueError(f"{context}[{index}] must be non-negative")
            result.append(item)
            continue
        if isinstance(item, str) and re.fullmatch(r"h[0-9]+", item):
            result.append(item)
            continue
        raise ValueError(f"{context}[{index}] must be an integer or h* coordinate")
    return tuple(result)


@dataclass(frozen=True)
class GeneratedTargetSourceArgument:
    """One explicit parameter in a generated target device-entry ABI.

    The record is source-realization metadata.  It is deliberately separate
    from J and from the target-neutral Kernel IR: a buffer's physical strides,
    a runtime scalar valuation, and a target C slot can be replayed here
    without becoming a logical operator or shape relation.
    """

    name: str
    category: str
    role: str | None
    dtype: str | None
    shape: tuple[int, ...] | None = None
    shape_contract: tuple[int | str, ...] | None = None
    strides: tuple[int, ...] | None = None
    value: bool | int | float | str | None = None
    shape_coordinate: str | None = None
    # ``dtype`` is the physical source ABI spelling for an opaque argument;
    # ``logical_dtype`` records the separate logical view used by J/Replay.
    logical_dtype: str | None = None
    output_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _manifest_identifier(self.name, "argument.name"))
        if self.category not in _MANIFEST_PARAMETER_CATEGORIES:
            raise ValueError(f"unsupported generated-source argument category {self.category!r}")
        if self.dtype is not None:
            object.__setattr__(self, "dtype", _manifest_token(self.dtype, "argument.dtype"))
        if self.shape is not None:
            shape = tuple(self.shape)
            if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape):
                raise ValueError("argument.shape must contain non-negative integers")
            object.__setattr__(self, "shape", shape)
        shape_contract = self.shape_contract
        if shape_contract is not None:
            shape_contract = tuple(shape_contract)
            if self.shape is None:
                raise ValueError("argument.shape_contract requires argument.shape")
            if len(shape_contract) != len(self.shape):
                raise ValueError("argument.shape_contract rank differs from shape")
            for index, item in enumerate(shape_contract):
                if isinstance(item, bool) or not isinstance(item, (int, str)):
                    raise ValueError("argument.shape_contract entries are invalid")
                if isinstance(item, int) and item < 0:
                    raise ValueError("argument.shape_contract entries must be non-negative")
                if isinstance(item, str) and re.fullmatch(r"h[0-9]+", item) is None:
                    raise ValueError(
                        f"argument.shape_contract[{index}] must use an h* coordinate"
                    )
            object.__setattr__(self, "shape_contract", shape_contract)
        strides = self.strides
        if strides is not None:
            strides = tuple(strides)
            if self.shape is None or len(strides) != len(self.shape):
                raise ValueError("argument.strides rank differs from shape")
            if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in strides):
                raise ValueError("argument.strides must contain non-negative integers")
            object.__setattr__(self, "strides", strides)
        if self.shape_coordinate is not None:
            if re.fullmatch(r"h[0-9]+", self.shape_coordinate) is None:
                raise ValueError("argument.shape_coordinate must use an h* coordinate")
        logical_dtype = self.logical_dtype
        if logical_dtype is not None:
            object.__setattr__(
                self, "logical_dtype", _manifest_token(logical_dtype, "argument.logical_dtype")
            )
        output_key = self.output_key
        if output_key is not None:
            object.__setattr__(
                self, "output_key", _manifest_identifier(output_key, "argument.output_key")
            )
        if self.category == "buffer":
            if self.role not in {item.value for item in BufferRole}:
                raise ValueError("buffer argument role must be input/output/inout")
            if self.dtype is None or self.shape is None or self.value is not None:
                raise ValueError("buffer argument requires dtype/shape and forbids value")
            if self.shape_coordinate is not None:
                raise ValueError("buffer argument cannot bind a scalar shape coordinate")
            if logical_dtype is not None and logical_dtype != self.dtype:
                raise ValueError("logical_dtype differs from a logical buffer dtype")
            if self.role == BufferRole.INPUT.value and output_key is not None:
                raise ValueError("input buffer cannot declare an output key")
        elif self.category == "opaque":
            # Physical opaque arguments (currently CUDA CUtensorMap) are
            # runtime-owned inputs.  The logical tensor contract is carried
            # by shape/shape_contract; the physical token is carried by dtype.
            # No scalar valuation or physical stride is implied by this class.
            if self.role != "input" or self.dtype is None or self.shape is None:
                raise ValueError(
                    "opaque argument requires input role, dtype, and shape"
                )
            if self.dtype not in {"tensor_map", "cuda_tensor_map", "opaque"}:
                raise ValueError(
                    "opaque argument dtype must be a recognized physical ABI token"
                )
            if self.value is not None or self.shape_contract is None or logical_dtype is None:
                raise ValueError(
                    "opaque argument requires logical_dtype/shape_contract and forbids value"
                )
            if output_key is not None:
                raise ValueError("opaque argument cannot declare an output key")
            if self.shape_coordinate is not None or self.strides is not None:
                raise ValueError(
                    "opaque argument forbids scalar coordinates and physical strides"
                )
        elif self.category == "runtime_scalar":
            if self.role != "value" or self.dtype is None or self.shape is not None:
                raise ValueError("runtime scalar argument has an invalid ABI record")
            if self.value is None:
                raise ValueError("runtime scalar argument requires a valuation")
            if self.strides is not None or self.shape_contract is not None:
                raise ValueError("runtime scalar argument forbids tensor layout fields")
            if output_key is not None or logical_dtype is not None:
                raise ValueError("runtime scalar argument forbids logical/output fields")
        elif self.category == "none":
            if self.role != "value" or self.dtype != "none" or self.value is not None:
                raise ValueError("None argument has an invalid ABI record")
            if any(item is not None for item in (self.shape, self.shape_contract, self.strides, self.shape_coordinate)):
                raise ValueError("None argument forbids tensor layout fields")
            if output_key is not None or logical_dtype is not None:
                raise ValueError("None argument forbids logical/output fields")
        elif self.category == "semantic_constexpr":
            if self.role != "semantic_constexpr" or self.dtype is None or self.value is None:
                raise ValueError("semantic constexpr argument has an invalid ABI record")
            if any(item is not None for item in (self.shape, self.shape_contract, self.strides)):
                raise ValueError("semantic constexpr argument forbids tensor layout fields")
            if output_key is not None or logical_dtype is not None:
                raise ValueError("semantic constexpr argument forbids logical/output fields")
        else:
            if self.role not in {"constexpr", "compiler"} or self.dtype is None or self.value is None:
                raise ValueError("C slot argument has an invalid ABI record")
            if any(item is not None for item in (self.shape, self.shape_contract, self.strides, self.shape_coordinate)):
                raise ValueError("C slot argument forbids tensor layout fields")
            if output_key is not None or logical_dtype is not None:
                raise ValueError("C slot argument forbids logical/output fields")
        if self.value is not None:
            object.__setattr__(self, "value", _manifest_scalar(self.value, "argument.value"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedTargetSourceArgument":
        keys = {
            "name", "category", "role", "dtype", "shape", "shape_contract",
            "strides", "value", "shape_coordinate", "logical_dtype", "output_key",
        }
        if not isinstance(value, Mapping) or set(value) != keys or any(not isinstance(key, str) for key in value):
            raise ValueError(
                "GeneratedTargetSourceArgument must contain exactly "
                + ", ".join(sorted(keys))
            )
        shape = value["shape"]
        if shape is not None and not isinstance(shape, list):
            raise ValueError("argument.shape must be null or an array")
        normalized_shape: tuple[int, ...] | None = None
        if shape is not None:
            checked_shape: list[int] = []
            for item in shape:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise ValueError("argument.shape must contain integers")
                checked_shape.append(item)
            normalized_shape = tuple(checked_shape)
        rank = len(normalized_shape) if normalized_shape is not None else None
        return cls(
            name=value["name"],
            category=value["category"],
            role=value["role"],
            dtype=value["dtype"],
            shape=normalized_shape,
            shape_contract=_manifest_shape_contract(
                value["shape_contract"], rank=rank, context="argument.shape_contract"
            ),
            strides=_manifest_optional_int_sequence(
                value["strides"], rank=rank, context="argument.strides"
            ),
            value=value["value"],
            shape_coordinate=value["shape_coordinate"],
            logical_dtype=value["logical_dtype"],
            output_key=value["output_key"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "role": self.role,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
            "shape_contract": (
                list(self.shape_contract) if self.shape_contract is not None else None
            ),
            "strides": list(self.strides) if self.strides is not None else None,
            "value": self.value,
            "shape_coordinate": self.shape_coordinate,
            "logical_dtype": self.logical_dtype,
            "output_key": self.output_key,
        }


_SOURCE_DIMENSION_RELATION_OPS = DIMENSION_RELATION_OPS


@dataclass(frozen=True)
class GeneratedSourceValueDimensionRelation:
    """An explicit source-side scalar-to-shape relation.

    These names belong to the generated source projection (source parameters
    and anonymous ``h*`` coordinates).  They are intentionally not J
    ``t*``/``x*`` references; the latter are carried separately after the
    accepted Gate1 artifact is bound.
    """

    value_ref: str
    dimension_ref: str
    relation: str = "eq"
    provenance: str = "explicit_shape_coordinate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_ref", _manifest_identifier(self.value_ref, "source value relation.value_ref"))
        if re.fullmatch(r"h[0-9]+", self.dimension_ref) is None:
            raise ValueError("source value relation.dimension_ref must use an h* coordinate")
        if self.relation != "eq":
            raise ValueError("source value relation.relation must be 'eq'")
        object.__setattr__(self, "provenance", _manifest_token(self.provenance, "source value relation.provenance"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedSourceValueDimensionRelation":
        keys = {"value_ref", "dimension_ref", "relation", "provenance"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(
                "GeneratedSourceValueDimensionRelation must contain exactly "
                + ", ".join(sorted(keys))
            )
        return cls(value["value_ref"], value["dimension_ref"], value["relation"], value["provenance"])

    def to_dict(self) -> dict[str, str]:
        return {
            "value_ref": self.value_ref,
            "dimension_ref": self.dimension_ref,
            "relation": self.relation,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class GeneratedSourceDimensionRelation:
    """An explicit source-side definition between anonymous shape coordinates."""

    result: str
    op: str
    arg: str
    value: int | None = None
    rhs: str | None = None
    provenance: str = "explicit_source"

    def __post_init__(self) -> None:
        for name, value in (("result", self.result), ("arg", self.arg)):
            if re.fullmatch(r"h[0-9]+", value) is None:
                raise ValueError(f"source dimension relation.{name} must use an h* coordinate")
        if self.rhs is not None and re.fullmatch(r"h[0-9]+", self.rhs) is None:
            raise ValueError("source dimension relation.rhs must use an h* coordinate")
        if self.op not in _SOURCE_DIMENSION_RELATION_OPS:
            raise ValueError("source dimension relation.op is outside the closed grammar")
        if self.op == "identity":
            if self.value is not None or self.rhs is not None:
                raise ValueError("identity source dimension relation forbids value/rhs")
        elif self.op == "mul_dims":
            if self.value is not None or self.rhs is None:
                raise ValueError("mul_dims source dimension relation requires rhs")
        else:
            if (
                self.op not in DIMENSION_RELATION_CONST_OPS
                or self.rhs is not None
                or isinstance(self.value, bool)
                or not isinstance(self.value, int)
                or self.value <= 0
                or self.value > DIMENSION_INT64_MAX
            ):
                raise ValueError(
                    f"{self.op} source dimension relation requires one positive signed-int64 value"
                )
        object.__setattr__(self, "provenance", _manifest_token(self.provenance, "source dimension relation.provenance"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedSourceDimensionRelation":
        keys = {"result", "op", "arg", "value", "rhs", "provenance"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(
                "GeneratedSourceDimensionRelation must contain exactly "
                + ", ".join(sorted(keys))
            )
        return cls(value["result"], value["op"], value["arg"], value["value"], value["rhs"], value["provenance"])

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "result": self.result,
            "op": self.op,
            "arg": self.arg,
            "value": self.value,
            "rhs": self.rhs,
            "provenance": self.provenance,
        }
        return result


@dataclass(frozen=True)
class GeneratedSourceShapeCoordinateBinding:
    """One explicit h* coordinate occurrence in a source parameter shape."""

    coordinate: str
    parameter: str
    axis: int
    extent: int

    def __post_init__(self) -> None:
        if re.fullmatch(r"h[0-9]+", self.coordinate) is None:
            raise ValueError("shape coordinate binding.coordinate must use an h* coordinate")
        object.__setattr__(self, "parameter", _manifest_identifier(self.parameter, "shape coordinate binding.parameter"))
        if isinstance(self.axis, bool) or not isinstance(self.axis, int) or self.axis < 0:
            raise ValueError("shape coordinate binding.axis must be non-negative")
        if isinstance(self.extent, bool) or not isinstance(self.extent, int) or self.extent < 0:
            raise ValueError("shape coordinate binding.extent must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedSourceShapeCoordinateBinding":
        keys = {"coordinate", "parameter", "axis", "extent"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(
                "GeneratedSourceShapeCoordinateBinding must contain exactly "
                + ", ".join(sorted(keys))
            )
        return cls(value["coordinate"], value["parameter"], value["axis"], value["extent"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate,
            "parameter": self.parameter,
            "axis": self.axis,
            "extent": self.extent,
        }


@dataclass(frozen=True)
class GeneratedSourceLayoutRelation:
    """Keep logical shape and physical layout facts in an independent record."""

    parameter: str
    physical_kind: str
    logical_dtype: str
    physical_dtype: str
    shape: tuple[int, ...]
    shape_contract: tuple[int | str, ...] | None
    strides: tuple[int, ...] | None
    layout: str | None = None
    memory_space: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter", _manifest_identifier(self.parameter, "layout relation.parameter"))
        if self.physical_kind not in {"buffer", "opaque"}:
            raise ValueError("layout relation.physical_kind must be buffer or opaque")
        object.__setattr__(self, "logical_dtype", _manifest_token(self.logical_dtype, "layout relation.logical_dtype"))
        object.__setattr__(self, "physical_dtype", _manifest_token(self.physical_dtype, "layout relation.physical_dtype"))
        shape = tuple(self.shape)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape):
            raise ValueError("layout relation.shape must contain non-negative integers")
        object.__setattr__(self, "shape", shape)
        contract = None if self.shape_contract is None else tuple(self.shape_contract)
        if contract is not None:
            if len(contract) != len(shape):
                raise ValueError("layout relation.shape_contract rank differs from shape")
            for item in contract:
                if isinstance(item, bool) or not isinstance(item, (int, str)):
                    raise ValueError("layout relation.shape_contract entries are invalid")
                if isinstance(item, int) and item < 0:
                    raise ValueError("layout relation.shape_contract entries must be non-negative")
                if isinstance(item, str) and re.fullmatch(r"h[0-9]+", item) is None:
                    raise ValueError("layout relation.shape_contract symbols must use h* coordinates")
        object.__setattr__(self, "shape_contract", contract)
        strides = None if self.strides is None else tuple(self.strides)
        if strides is not None:
            if len(strides) != len(shape) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in strides):
                raise ValueError("layout relation.strides must match shape rank and be non-negative")
        object.__setattr__(self, "strides", strides)
        if self.layout is not None:
            object.__setattr__(self, "layout", _manifest_token(self.layout, "layout relation.layout"))
        if self.memory_space is not None:
            object.__setattr__(self, "memory_space", _manifest_token(self.memory_space, "layout relation.memory_space"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedSourceLayoutRelation":
        keys = {"parameter", "physical_kind", "logical_dtype", "physical_dtype", "shape", "shape_contract", "strides", "layout", "memory_space"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(
                "GeneratedSourceLayoutRelation must contain exactly "
                + ", ".join(sorted(keys))
            )
        shape = value["shape"]
        if not isinstance(shape, list):
            raise ValueError("layout relation.shape must be an array")
        return cls(
            value["parameter"],
            value["physical_kind"],
            value["logical_dtype"],
            value["physical_dtype"],
            tuple(shape),
            None if value["shape_contract"] is None else tuple(value["shape_contract"]),
            None if value["strides"] is None else tuple(value["strides"]),
            value["layout"],
            value["memory_space"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "physical_kind": self.physical_kind,
            "logical_dtype": self.logical_dtype,
            "physical_dtype": self.physical_dtype,
            "shape": list(self.shape),
            "shape_contract": list(self.shape_contract) if self.shape_contract is not None else None,
            "strides": list(self.strides) if self.strides is not None else None,
            "layout": self.layout,
            "memory_space": self.memory_space,
        }


@dataclass(frozen=True)
class GeneratedSourceTensorMapRelation:
    """Explicit logical/physical projection for a CUtensorMap parameter."""

    physical_parameter: str
    logical_parameter: str
    logical_dtype: str
    shape_contract: tuple[int | str, ...]
    box_dim: tuple[int, int]
    swizzle: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_parameter", _manifest_identifier(self.physical_parameter, "TensorMap relation.physical_parameter"))
        object.__setattr__(self, "logical_parameter", _manifest_identifier(self.logical_parameter, "TensorMap relation.logical_parameter"))
        object.__setattr__(self, "logical_dtype", _manifest_token(self.logical_dtype, "TensorMap relation.logical_dtype"))
        contract = tuple(self.shape_contract)
        if len(contract) != 2 or any(
            isinstance(item, bool)
            or not ((isinstance(item, int) and item > 0) or (isinstance(item, str) and re.fullmatch(r"h[0-9]+", item)))
            for item in contract
        ):
            raise ValueError("TensorMap relation.shape_contract must be one positive rank-2 shape")
        box = tuple(self.box_dim)
        if len(box) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in box):
            raise ValueError("TensorMap relation.box_dim must be positive rank-2")
        if self.swizzle not in {"none", "32b", "64b", "128b"}:
            raise ValueError("TensorMap relation.swizzle is unsupported")
        object.__setattr__(self, "shape_contract", contract)
        object.__setattr__(self, "box_dim", box)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedSourceTensorMapRelation":
        keys = {"physical_parameter", "logical_parameter", "logical_dtype", "shape_contract", "box_dim", "swizzle"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(
                "GeneratedSourceTensorMapRelation must contain exactly "
                + ", ".join(sorted(keys))
            )
        return cls(
            value["physical_parameter"],
            value["logical_parameter"],
            value["logical_dtype"],
            tuple(value["shape_contract"]),
            tuple(value["box_dim"]),
            value["swizzle"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_parameter": self.physical_parameter,
            "logical_parameter": self.logical_parameter,
            "logical_dtype": self.logical_dtype,
            "shape_contract": list(self.shape_contract),
            "box_dim": list(self.box_dim),
            "swizzle": self.swizzle,
        }


@dataclass(frozen=True)
class GeneratedSourceOutputBinding:
    """Source-side output identity retained for Gate2 output re-ingress."""

    parameter: str
    output_key: str
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter", _manifest_identifier(self.parameter, "output binding.parameter"))
        object.__setattr__(self, "output_key", _manifest_identifier(self.output_key, "output binding.output_key"))
        if self.role not in {"output", "inout"}:
            raise ValueError("output binding.role must be output or inout")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedSourceOutputBinding":
        keys = {"parameter", "output_key", "role"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(
                "GeneratedSourceOutputBinding must contain exactly "
                + ", ".join(sorted(keys))
            )
        return cls(value["parameter"], value["output_key"], value["role"])

    def to_dict(self) -> dict[str, str]:
        return {
            "parameter": self.parameter,
            "output_key": self.output_key,
            "role": self.role,
        }


def _source_relation_tuple(
    values: Iterable[Any],
    relation_type: Any,
    context: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{context} must be an array")
    try:
        result = []
        for item in values:
            if isinstance(item, relation_type):
                result.append(item)
                continue
            if not isinstance(item, Mapping):
                raise ValueError(f"{context} entries must be objects")
            result.append(relation_type.from_dict(item))
    except TypeError as exc:
        raise ValueError(f"{context} must be an array") from exc
    return tuple(result)


def _j_relation_tuple(
    values: Iterable[Any],
    relation_type: Any,
    context: str,
) -> tuple[Any, ...]:
    """Normalize the strict Gate1 relation records carried by a bound J."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{context} must be an array")
    try:
        result = []
        for item in values:
            if isinstance(item, relation_type):
                result.append(item)
                continue
            if not isinstance(item, Mapping):
                raise ValueError(f"{context} entries must be objects")
            result.append(relation_type.from_json(item))
    except TypeError as exc:
        raise ValueError(f"{context} must be an array") from exc
    return tuple(result)


def _derived_source_relations(
    arguments: tuple[GeneratedTargetSourceArgument, ...],
) -> tuple[
    tuple[GeneratedSourceValueDimensionRelation, ...],
    tuple[GeneratedSourceShapeCoordinateBinding, ...],
    tuple[GeneratedSourceLayoutRelation, ...],
    tuple[GeneratedSourceOutputBinding, ...],
]:
    value_relations: list[GeneratedSourceValueDimensionRelation] = []
    shape_bindings: list[GeneratedSourceShapeCoordinateBinding] = []
    layout_relations: list[GeneratedSourceLayoutRelation] = []
    output_bindings: list[GeneratedSourceOutputBinding] = []
    for argument in arguments:
        if argument.category in {"buffer", "opaque"}:
            if argument.shape_contract is not None:
                for axis, (extent, concrete) in enumerate(
                    zip(argument.shape_contract, argument.shape, strict=True)
                ):
                    if isinstance(extent, str):
                        shape_bindings.append(
                            GeneratedSourceShapeCoordinateBinding(
                                extent,
                                argument.name,
                                axis,
                                concrete,
                            )
                        )
            logical_dtype = argument.logical_dtype or argument.dtype
            if logical_dtype is None or argument.dtype is None or argument.shape is None:
                raise ValueError(
                    f"source layout relation lacks ABI types/shape for {argument.name!r}"
                )
            layout_relations.append(
                GeneratedSourceLayoutRelation(
                    parameter=argument.name,
                    physical_kind=argument.category,
                    logical_dtype=logical_dtype,
                    physical_dtype=argument.dtype,
                    shape=argument.shape,
                    shape_contract=argument.shape_contract,
                    strides=argument.strides,
                )
            )
            if argument.output_key is not None:
                output_bindings.append(
                    GeneratedSourceOutputBinding(
                        argument.name,
                        argument.output_key,
                        argument.role or "output",
                    )
                )
        elif argument.category in {"runtime_scalar", "semantic_constexpr"}:
            if argument.shape_coordinate is not None:
                value_relations.append(
                    GeneratedSourceValueDimensionRelation(
                        argument.name,
                        argument.shape_coordinate,
                    )
                )
    return (
        tuple(value_relations),
        tuple(shape_bindings),
        tuple(layout_relations),
        tuple(output_bindings),
    )


def _validate_source_relations(
    arguments: tuple[GeneratedTargetSourceArgument, ...],
    coordinates: tuple[tuple[str, int], ...],
    value_relations: tuple[GeneratedSourceValueDimensionRelation, ...],
    dimension_relations: tuple[GeneratedSourceDimensionRelation, ...],
    shape_bindings: tuple[GeneratedSourceShapeCoordinateBinding, ...],
    layout_relations: tuple[GeneratedSourceLayoutRelation, ...],
    tensor_map_relations: tuple[GeneratedSourceTensorMapRelation, ...],
    output_bindings: tuple[GeneratedSourceOutputBinding, ...],
) -> None:
    """Admit the complete source-side relation graph against source ABI rows.

    This is deliberately independent of J.  The source projection owns source
    parameter names and anonymous ``h*`` valuations; canonical J relations are
    admitted separately when :meth:`GeneratedTargetSourceManifest.bind_contract`
    attaches an accepted Gate1 artifact.
    """

    expected_value, expected_shapes, expected_layout, expected_outputs = (
        _derived_source_relations(arguments)
    )
    if value_relations != expected_value:
        raise ValueError(
            "source value-dimension relations do not exactly match scalar ABI "
            "shape-coordinate bindings"
        )
    if shape_bindings != expected_shapes:
        raise ValueError(
            "source shape-coordinate bindings do not exactly match tensor ABI "
            "shape contracts"
        )
    if layout_relations != expected_layout:
        raise ValueError(
            "source layout relations do not exactly match source ABI arguments"
        )
    if output_bindings != expected_outputs:
        raise ValueError(
            "source output bindings do not exactly match output/inout ABI arguments"
        )

    coordinate_values = dict(coordinates)
    argument_names = {item.name for item in arguments}
    for relation in value_relations:
        if relation.value_ref not in argument_names:
            raise ValueError(
                f"source value relation references unknown parameter {relation.value_ref!r}"
            )
        if relation.dimension_ref not in coordinate_values:
            raise ValueError(
                f"source value relation references unknown coordinate {relation.dimension_ref!r}"
            )
        argument = next(
            item for item in arguments if item.name == relation.value_ref
        )
        if argument.category not in {"runtime_scalar", "semantic_constexpr"}:
            raise ValueError(
                f"source value relation {relation.value_ref!r} is not scalar ABI"
            )
        if argument.shape_coordinate != relation.dimension_ref:
            raise ValueError(
                f"source value relation for {relation.value_ref!r} is detached from its ABI row"
            )
        if argument.value != coordinate_values[relation.dimension_ref]:
            raise ValueError(
                f"source value relation valuation differs for {relation.value_ref!r}"
            )

    if len({item.result for item in dimension_relations}) != len(dimension_relations):
        raise ValueError("source dimension relations must define each result once")
    for relation in dimension_relations:
        if relation.result not in coordinate_values:
            raise ValueError(
                f"source dimension relation result {relation.result!r} lacks a coordinate valuation"
            )
        if relation.arg not in coordinate_values:
            raise ValueError(
                f"source dimension relation arg {relation.arg!r} lacks a coordinate valuation"
            )
        if relation.rhs is not None and relation.rhs not in coordinate_values:
            raise ValueError(
                f"source dimension relation rhs {relation.rhs!r} lacks a coordinate valuation"
            )
        argument_value = coordinate_values[relation.arg]
        try:
            expected = evaluate_dimension_relation_operation(
                relation.op,
                argument_value,
                value=relation.value,
                rhs_argument=(
                    coordinate_values[relation.rhs]
                    if relation.rhs is not None
                    else None
                ),
                context=f"source dimension relation {relation.result!r}",
            )
        except DimensionRelationError as exc:
            raise ValueError(str(exc)) from exc
        if coordinate_values[relation.result] != expected:
            raise ValueError(
                f"source dimension relation result valuation differs for {relation.result!r}"
            )

    opaque_arguments = {
        item.name: item for item in arguments if item.category == "opaque"
    }
    tensor_map_by_physical: dict[str, GeneratedSourceTensorMapRelation] = {}
    logical_parameters: set[str] = set()
    for relation in tensor_map_relations:
        if relation.physical_parameter in tensor_map_by_physical:
            raise ValueError("source TensorMap relations must have unique physical parameters")
        if relation.logical_parameter in logical_parameters:
            raise ValueError("source TensorMap relations must have unique logical parameters")
        tensor_map_by_physical[relation.physical_parameter] = relation
        logical_parameters.add(relation.logical_parameter)
        argument = opaque_arguments.get(relation.physical_parameter)
        if argument is None:
            raise ValueError(
                f"source TensorMap relation references non-opaque parameter {relation.physical_parameter!r}"
            )
        if relation.logical_dtype != argument.logical_dtype:
            raise ValueError(
                f"source TensorMap logical dtype differs for {relation.physical_parameter!r}"
            )
        if relation.shape_contract != argument.shape_contract:
            raise ValueError(
                f"source TensorMap shape contract differs for {relation.physical_parameter!r}"
            )
    if set(tensor_map_by_physical) != set(opaque_arguments):
        raise ValueError(
            "source TensorMap relations must cover every opaque ABI argument and no other argument"
        )

    output_parameters = {
        item.name: (item.output_key, item.role)
        for item in arguments
        if item.category == "buffer" and item.role in {"output", "inout"}
    }
    output_by_key: set[str] = set()
    for relation in output_bindings:
        expected = output_parameters.get(relation.parameter)
        if expected is None:
            raise ValueError(
                f"source output binding references a non-output parameter {relation.parameter!r}"
            )
        if expected != (relation.output_key, relation.role):
            raise ValueError(
                f"source output binding differs from ABI argument {relation.parameter!r}"
            )
        if relation.output_key in output_by_key:
            raise ValueError("source output bindings must have unique output keys")
        output_by_key.add(relation.output_key)


_SOURCE_J_DTYPE_ALIASES = {
    "bool": "i1",
    "boolean": "i1",
    "float16": "f16",
    "half": "f16",
    "float32": "f32",
    "float": "f32",
    "float64": "f64",
    "double": "f64",
    "bfloat16": "bf16",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "index": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "str": "string",
    "string": "string",
}


def _source_j_dtype_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    spelling = value.casefold().removeprefix("torch.")
    return _SOURCE_J_DTYPE_ALIASES.get(spelling, spelling)


def _validate_source_j_relation_closure(
    *,
    arguments: tuple[GeneratedTargetSourceArgument, ...],
    source_parameter_order: tuple[str, ...],
    compiler_slot_order: tuple[str, ...],
    shape_coordinate_values: tuple[tuple[str, int], ...],
    value_dimension_relations: tuple[GeneratedSourceValueDimensionRelation, ...],
    dimension_relations: tuple[GeneratedSourceDimensionRelation, ...],
    output_bindings: tuple[GeneratedSourceOutputBinding, ...],
    source_to_opaque: Mapping[str, str],
    canonical: Any,
    contract: Any,
) -> None:
    """Prove the source relation graph and accepted J use one ABI mapping.

    The source manifest and J deliberately retain different namespaces:
    source parameters/coordinates use their real names and ``h*`` while the
    accepted artifact uses anonymous ``p*``, logical ``t*`` and dimensions
    ``x*``.  Independent validation of the two sides is insufficient: two
    individually valid graphs could still describe different shapes.  This
    helper only follows the explicit source-to-anonymous mapping and exact
    ABI declarations.  It never infers a missing mapping from names, order of
    unrelated tensors, or concrete values.
    """

    mapping = dict(source_to_opaque)
    expected_mapping_keys = set(source_parameter_order) | set(compiler_slot_order)
    if set(mapping) != expected_mapping_keys:
        missing = sorted(expected_mapping_keys - set(mapping))
        extra = sorted(set(mapping) - expected_mapping_keys)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(
            "source/J relation closure requires a complete source-to-anonymous "
            "mapping" + (" (" + "; ".join(detail) + ")" if detail else "")
        )
    if len(mapping) != len(set(mapping.values())):
        raise ValueError("source/J relation closure requires a one-to-one ABI mapping")
    if any(
        not isinstance(source, str)
        or not source.isidentifier()
        or not isinstance(anonymous, str)
        or re.fullmatch(r"(?:p|c)[0-9]+", anonymous) is None
        for source, anonymous in mapping.items()
    ):
        raise ValueError("source/J relation closure mapping must use p*/c* handles")
    if any(
        not re.fullmatch(r"c[0-9]+", mapping[name])
        for name in compiler_slot_order
    ):
        raise ValueError("compiler-only source slots must map to c* handles")

    expected_p = tuple(
        mapping[name]
        for name in source_parameter_order
        if re.fullmatch(r"p[0-9]+", mapping[name])
    )
    if expected_p != tuple(f"p{index}" for index in range(len(expected_p))):
        raise ValueError("source/J relation closure requires dense p* ABI handles")
    signature_parameters = tuple(canonical.signature.parameters)
    if tuple(parameter.name for parameter in signature_parameters) != expected_p:
        raise ValueError(
            "J.signature parameter order is not the mapped source ABI order"
        )
    signature = {parameter.name: parameter for parameter in signature_parameters}
    j_tensors = {
        tensor.name: tensor
        for tensor in [*canonical.intent.inputs, *canonical.intent.outputs]
    }
    input_names = {tensor.name for tensor in canonical.intent.inputs}
    output_names = {tensor.name for tensor in canonical.intent.outputs}

    def parameter_for(source_name: str) -> Any:
        anonymous = mapping.get(source_name)
        if anonymous is None or not re.fullmatch(r"p[0-9]+", anonymous):
            raise ValueError(
                f"source relation parameter {source_name!r} has no mapped p* binding"
            )
        parameter = signature.get(anonymous)
        if parameter is None:
            raise ValueError(
                f"J.signature lacks mapped source parameter {source_name!r}"
            )
        return parameter

    coordinate_values = dict(shape_coordinate_values)
    coordinate_to_dimension: dict[str, str] = {}
    dimension_to_coordinate: dict[str, str] = {}

    def bind_coordinate(coordinate: str, dimension: Any, *, context: str) -> None:
        if coordinate not in coordinate_values:
            raise ValueError(
                f"{context} references undeclared source coordinate {coordinate!r}"
            )
        if not isinstance(dimension, str) or re.fullmatch(r"x[0-9]+", dimension) is None:
            raise ValueError(
                f"{context} must map to one symbolic J dimension x*"
            )
        previous_dimension = coordinate_to_dimension.setdefault(coordinate, dimension)
        if previous_dimension != dimension:
            raise ValueError(
                f"source coordinate {coordinate!r} maps to multiple J dimensions"
            )
        previous_coordinate = dimension_to_coordinate.setdefault(dimension, coordinate)
        if previous_coordinate != coordinate:
            raise ValueError(
                f"J dimension {dimension!r} merges distinct source coordinates"
            )

    def check_tensor_argument(
        argument: GeneratedTargetSourceArgument,
        parameter: Any,
    ) -> None:
        actual_kind = getattr(getattr(parameter, "kind", None), "value", parameter.kind)
        # ``opaque`` is the physical Source representation (currently a CUDA
        # TensorMap).  Its logical projection is still a J tensor/pointer;
        # the physical token is closed by tensor_map_relations and must never
        # become an opaque J logical value.
        if actual_kind not in {"pointer", "tensor"}:
            raise ValueError(
                f"source tensor parameter {argument.name!r} is not a J tensor/pointer"
            )
        actual_mode = getattr(getattr(parameter, "mode", None), "value", parameter.mode)
        if actual_mode != argument.role:
            raise ValueError(
                f"source tensor parameter {argument.name!r} changed J mode"
            )
        logical_ref = parameter.logical_ref
        if not isinstance(logical_ref, str) or logical_ref not in j_tensors:
            raise ValueError(
                f"source tensor parameter {argument.name!r} is not bound to a J t*"
            )
        tensor = j_tensors[logical_ref]
        expected_dtype = argument.logical_dtype or argument.dtype
        if _source_j_dtype_key(parameter.dtype) != _source_j_dtype_key(expected_dtype):
            raise ValueError(
                f"source tensor parameter {argument.name!r} changed logical dtype"
            )
        if parameter.rank != len(argument.shape or ()):
            raise ValueError(
                f"source tensor parameter {argument.name!r} changed rank"
            )
        if len(tensor.shape) != parameter.rank:
            raise ValueError(
                f"J tensor {logical_ref!r} rank differs from its physical binding"
            )
        if argument.shape_contract is None:
            return
        if len(argument.shape_contract) != len(tensor.shape):
            raise ValueError(
                f"source/J shape contract rank differs for {argument.name!r}"
            )
        for axis, (source_extent, j_extent) in enumerate(
            zip(argument.shape_contract, tensor.shape, strict=True)
        ):
            context = f"source/J shape relation {argument.name!r}[{axis}]"
            if isinstance(source_extent, str):
                bind_coordinate(source_extent, j_extent, context=context)
            elif isinstance(source_extent, int):
                if j_extent != source_extent:
                    raise ValueError(
                        f"{context} changed a static source shape extent"
                    )
            else:  # pragma: no cover - manifest construction owns this grammar
                raise ValueError(f"{context} has an invalid source extent")

    source_arguments = {argument.name: argument for argument in arguments}
    for source_name in source_parameter_order:
        argument = source_arguments.get(source_name)
        if argument is None:
            raise ValueError(
                f"source/J relation closure lacks ABI argument {source_name!r}"
            )
        anonymous = mapping[source_name]
        if re.fullmatch(r"c[0-9]+", anonymous):
            if argument.category != "c_slot":
                raise ValueError(
                    f"source runtime parameter {source_name!r} was mapped to a C slot"
                )
            continue
        parameter = parameter_for(source_name)
        if argument.category in {"buffer", "opaque"}:
            check_tensor_argument(argument, parameter)
            continue
        actual_kind = getattr(getattr(parameter, "kind", None), "value", parameter.kind)
        actual_mode = getattr(getattr(parameter, "mode", None), "value", parameter.mode)
        if argument.category in {"runtime_scalar", "semantic_constexpr"}:
            if actual_kind not in {"scalar", "constexpr"} or actual_mode != "value":
                raise ValueError(
                    f"source scalar parameter {source_name!r} is not a value binding"
                )
            if _source_j_dtype_key(parameter.dtype) != _source_j_dtype_key(argument.dtype):
                raise ValueError(
                    f"source scalar parameter {source_name!r} changed dtype"
                )
            if argument.shape_coordinate is None:
                if parameter.logical_ref is not None and re.fullmatch(
                    r"x[0-9]+", parameter.logical_ref
                ):
                    raise ValueError(
                        f"source scalar {source_name!r} has no source shape coordinate "
                        "but J binds a dimension"
                    )
                if parameter.logical_ref is not None:
                    logical_value = j_tensors.get(parameter.logical_ref)
                    if logical_value is None or tuple(logical_value.shape) != ():
                        raise ValueError(
                            f"source scalar {source_name!r} must bind a rank-0 J value"
                        )
                continue
            continue
        if argument.category == "none":
            if actual_kind != "opaque" or actual_mode != "value" or parameter.logical_ref is not None:
                raise ValueError(
                    f"source None parameter {source_name!r} changed J representation"
                )
            continue
        if argument.category == "c_slot":
            raise ValueError(
                f"source C slot {source_name!r} was mapped to a J p* parameter"
            )
        raise ValueError(f"unsupported source argument category {argument.category!r}")

    # A source scalar/semantic constexpr that carries a shape coordinate may
    # be represented directly by J's x* ABI scalar or by a rank-0 t* input.
    # In the latter case the equality must be the exact J-authored relation.
    source_value_relation_j_keys: list[tuple[str, str, str]] = []
    for relation in value_dimension_relations:
        parameter = parameter_for(relation.value_ref)
        logical_ref = parameter.logical_ref
        if isinstance(logical_ref, str) and re.fullmatch(r"x[0-9]+", logical_ref):
            bind_coordinate(
                relation.dimension_ref,
                logical_ref,
                context=f"source value relation {relation.value_ref!r}",
            )
            continue
        if not isinstance(logical_ref, str) or not re.fullmatch(r"t[0-9]+", logical_ref):
            raise ValueError(
                f"source value relation {relation.value_ref!r} is not represented "
                "by a J dimension or rank-0 t* value"
            )
        tensor = j_tensors.get(logical_ref)
        if tensor is None or tensor.shape != () or logical_ref not in input_names:
            raise ValueError(
                f"source value relation {relation.value_ref!r} must bind a rank-0 J input"
            )
        candidates = [
            item
            for item in canonical.value_dimension_relations
            if item.value_ref == logical_ref and item.relation == "eq"
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"source value relation {relation.value_ref!r} lacks one unique J equality"
            )
        candidate = candidates[0]
        bind_coordinate(
            relation.dimension_ref,
            candidate.dimension_ref,
            context=f"source value relation {relation.value_ref!r}",
        )
        source_value_relation_j_keys.append(
            (candidate.value_ref, candidate.dimension_ref, candidate.relation)
        )

    actual_value_relation_j_keys = sorted(
        (item.value_ref, item.dimension_ref, item.relation)
        for item in canonical.value_dimension_relations
    )
    if actual_value_relation_j_keys != sorted(source_value_relation_j_keys):
        raise ValueError(
            "J value_dimension_relations are not exactly the source scalar/shape "
            "relations after ABI mapping"
        )

    mapped_dimension_relation_keys: list[tuple[str, str, str, str | None, int | None]] = []
    for relation in dimension_relations:
        mapped_dimension_relation_keys.append(
            (
                coordinate_to_dimension.get(relation.result, ""),
                relation.op,
                coordinate_to_dimension.get(relation.arg, ""),
                (
                    coordinate_to_dimension.get(relation.rhs)
                    if relation.rhs is not None
                    else None
                ),
                relation.value,
            )
        )
        if not coordinate_to_dimension.get(relation.result) or not coordinate_to_dimension.get(
            relation.arg
        ) or (relation.rhs is not None and not coordinate_to_dimension.get(relation.rhs)):
            raise ValueError(
                "source dimension relation is not connected to a mapped J dimension"
            )
    actual_dimension_relation_keys = sorted(
        (
            item.result,
            item.expr.op,
            item.expr.arg,
            item.expr.rhs,
            item.expr.value,
        )
        for item in canonical.dimension_relations
    )
    if actual_dimension_relation_keys != sorted(mapped_dimension_relation_keys):
        raise ValueError(
            "J dimension_relations are not exactly the source dimension relations "
            "after ABI mapping"
        )

    if output_bindings:
        mapped_outputs = tuple(
            parameter_for(item.parameter).logical_ref
            for item in output_bindings
        )
        j_outputs = tuple(item.name for item in canonical.intent.outputs)
        if mapped_outputs != j_outputs:
            raise ValueError(
                "source output bindings do not map to J outputs in ABI order"
            )
        if any(item is None or item not in output_names for item in mapped_outputs):
            raise ValueError("source output bindings must map to declared J outputs")

    # Every J dimension reference that is part of the callable/contract must
    # be one of the source coordinates proved above.  This catches a J-only
    # shape or relation that happens to be locally valid but is detached from
    # the selected Source ABI.  Dimension constraints are included through the
    # already-typed contract projection.
    j_dimensions: set[str] = set()
    for tensor in [*canonical.intent.inputs, *canonical.intent.outputs]:
        j_dimensions.update(
            item for item in tensor.shape if isinstance(item, str) and item != "?"
        )
    j_dimensions.update(
        parameter.logical_ref
        for parameter in signature_parameters
        if isinstance(parameter.logical_ref, str)
        and re.fullmatch(r"x[0-9]+", parameter.logical_ref)
    )
    for relation in canonical.dimension_relations:
        j_dimensions.add(relation.result)
        j_dimensions.add(relation.expr.arg)
        if relation.expr.rhs is not None:
            j_dimensions.add(relation.expr.rhs)
    for relation in canonical.value_dimension_relations:
        j_dimensions.add(relation.dimension_ref)
    for program in contract.typed_dimension_constraints:
        j_dimensions.update(
            reference
            for reference in program.expression.references
            if re.fullmatch(r"x[0-9]+", reference)
        )
    mapped_dimensions = set(coordinate_to_dimension.values())
    if j_dimensions - mapped_dimensions:
        raise ValueError(
            "J dimension reference(s) are detached from source h* coordinates: "
            + ", ".join(sorted(j_dimensions - mapped_dimensions))
        )
    if set(coordinate_values) != set(coordinate_to_dimension):
        raise ValueError(
            "source h* coordinates are not all represented by the accepted J"
        )


@dataclass(frozen=True)
class GeneratedSourceLifecycleRecord:
    """One typed, byte-bound stage in a generated-source lifecycle.

    The record is deliberately small and closed.  ``artifact_digest`` always
    names real bytes produced or consumed by the stage; source text, cache keys,
    and record identifiers are not accepted as substitutes.  Source and Replay
    execution legs are aggregated by the Gate2 owner into one record per phase,
    while the per-leg digests remain in the corresponding execution receipts.
    """

    phase: str
    consumer: str
    artifact_kind: str
    artifact_digest: str
    source_slice_sha256: str
    output_digest: str | None = None
    output_binding_digest: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in _GENERATED_SOURCE_LIFECYCLE_PHASES:
            raise ValueError(
                "generated-source lifecycle phase must be one of "
                + ", ".join(_GENERATED_SOURCE_LIFECYCLE_PHASES)
            )
        object.__setattr__(
            self,
            "consumer",
            _manifest_token(self.consumer, "lifecycle.consumer"),
        )
        object.__setattr__(
            self,
            "artifact_kind",
            _manifest_token(self.artifact_kind, "lifecycle.artifact_kind"),
        )
        _manifest_digest(self.artifact_digest, "lifecycle.artifact_digest")
        _manifest_digest(
            self.source_slice_sha256,
            "lifecycle.source_slice_sha256",
        )
        if self.output_digest is not None:
            _manifest_digest(self.output_digest, "lifecycle.output_digest")
        if self.output_binding_digest is not None:
            _manifest_digest(
                self.output_binding_digest,
                "lifecycle.output_binding_digest",
            )
        if self.phase == "output_digest" and self.output_digest is None:
            raise ValueError("output_digest lifecycle phase requires output_digest")
        if self.phase == "source_execute" and self.output_binding_digest is None:
            raise ValueError(
                "source_execute lifecycle phase requires output_binding_digest"
            )
        if self.phase == "reingress":
            if self.artifact_kind != "source_slice":
                raise ValueError("reingress lifecycle artifact must be a source_slice")
            if self.artifact_digest != self.source_slice_sha256:
                raise ValueError(
                    "reingress artifact digest must equal the newly ingressed source slice digest"
                )
            if self.output_binding_digest is None:
                raise ValueError(
                    "reingress lifecycle phase requires output_binding_digest"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedSourceLifecycleRecord":
        keys = {
            "phase",
            "consumer",
            "artifact_kind",
            "artifact_digest",
            "source_slice_sha256",
            "output_digest",
            "output_binding_digest",
        }
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(
                "GeneratedSourceLifecycleRecord must contain exactly "
                + ", ".join(sorted(keys))
            )
        return cls(
            phase=value["phase"],
            consumer=value["consumer"],
            artifact_kind=value["artifact_kind"],
            artifact_digest=value["artifact_digest"],
            source_slice_sha256=value["source_slice_sha256"],
            output_digest=value["output_digest"],
            output_binding_digest=value["output_binding_digest"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "consumer": self.consumer,
            "artifact_kind": self.artifact_kind,
            "artifact_digest": self.artifact_digest,
            "source_slice_sha256": self.source_slice_sha256,
            "output_digest": self.output_digest,
            "output_binding_digest": self.output_binding_digest,
        }


def _validate_lifecycle_records(
    records: tuple[GeneratedSourceLifecycleRecord, ...],
    *,
    current_source_slice_sha256: str,
) -> None:
    phases = tuple(item.phase for item in records)
    expected = _GENERATED_SOURCE_LIFECYCLE_PHASES[: len(records)]
    if phases != expected:
        raise ValueError(
            "generated-source lifecycle records must be a fixed ordered prefix: "
            + ", ".join(_GENERATED_SOURCE_LIFECYCLE_PHASES)
        )
    if records and records[0].source_slice_sha256 != current_source_slice_sha256:
        # A changed source slice is admitted only by the explicit final
        # re-ingress stage.  The three execution records remain bound to the
        # source slice that was actually compiled and executed.
        if records[-1].phase != "reingress":
            raise ValueError(
                "lifecycle execution records must bind the manifest source slice"
            )
    if len(records) >= 2 and (
        records[1].artifact_digest != records[0].artifact_digest
    ):
        raise ValueError(
            "source_execute must retain the compiled artifact digest from source_compile"
        )
    if len(records) >= 3 and records[2].source_slice_sha256 != records[0].source_slice_sha256:
        raise ValueError(
            "output_digest must bind the source slice used for execution"
        )
    if len(records) == 4 and records[3].source_slice_sha256 != current_source_slice_sha256:
        raise ValueError(
            "reingress must bind the manifest's newly selected source slice"
        )


@dataclass(frozen=True)
class GeneratedTargetSourceManifest:
    """Strict source-output/ABI manifest used at target-source re-ingress."""

    source_sha256: str
    source_slice_sha256: str
    source_location: str
    device_entry_symbol: str
    source_parameter_order: tuple[str, ...]
    runtime_parameter_order: tuple[str, ...]
    constexpr_parameter_order: tuple[str, ...]
    arguments: tuple[GeneratedTargetSourceArgument, ...]
    semantic_constexpr_parameters: tuple[str, ...]
    c_source_constexpr_parameters: tuple[str, ...]
    compiler_slot_order: tuple[str, ...]
    c_slot_order: tuple[str, ...]
    runtime_scalar_values: tuple[tuple[str, bool | int | float | str], ...]
    semantic_constexpr_values: tuple[tuple[str, bool | int | float | str], ...]
    c_slot_values: tuple[tuple[str, bool | int | float | str], ...]
    shape_coordinate_values: tuple[tuple[str, int], ...]
    grid: tuple[int, ...]
    grid_contract: tuple[str, ...]
    grid_shape_coordinates: tuple[str | None, ...]
    # Source-side relations are explicit facts from the source/adapter view.
    # They are never substituted for the separately bound J relations below.
    value_dimension_relations: tuple[GeneratedSourceValueDimensionRelation, ...] = ()
    dimension_relations: tuple[GeneratedSourceDimensionRelation, ...] = ()
    shape_coordinate_bindings: tuple[GeneratedSourceShapeCoordinateBinding, ...] = ()
    layout_relations: tuple[GeneratedSourceLayoutRelation, ...] = ()
    tensor_map_relations: tuple[GeneratedSourceTensorMapRelation, ...] = ()
    output_bindings: tuple[GeneratedSourceOutputBinding, ...] = ()
    j_value_dimension_relations: tuple[ValueDimensionRelation, ...] = ()
    j_dimension_relations: tuple[DimensionRelation, ...] = ()
    j_digest: str | None = None
    e_c_interface_digest: str | None = None
    contract_ir_digest: str | None = None
    # A final manifest may contain lifecycle records or a re-ingressed source
    # whose source bytes differ from the bytes used for the original
    # Source/Replay/J execution.  Preserve the exact pre-execution source
    # identity in the manifest so a persisted final digest can never be
    # mistaken for the digest that the runners actually consumed.
    execution_source_sha256: str | None = None
    execution_source_slice_sha256: str | None = None
    execution_source_location: str | None = None
    execution_device_entry_symbol: str | None = None
    lifecycle_records: tuple[GeneratedSourceLifecycleRecord, ...] = ()
    protocol: str = _GENERATED_SOURCE_MANIFEST_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != _GENERATED_SOURCE_MANIFEST_PROTOCOL:
            raise ValueError("unsupported generated target source manifest protocol")
        object.__setattr__(self, "source_sha256", _manifest_digest(self.source_sha256, "source_sha256"))
        object.__setattr__(self, "source_slice_sha256", _manifest_digest(self.source_slice_sha256, "source_slice_sha256"))
        if not isinstance(self.source_location, str) or not self.source_location.strip():
            raise ValueError("source_location must be a non-empty string")
        object.__setattr__(self, "device_entry_symbol", _manifest_identifier(self.device_entry_symbol, "device_entry_symbol"))
        execution_source_sha256 = self.execution_source_sha256
        execution_source_slice_sha256 = self.execution_source_slice_sha256
        execution_source_location = self.execution_source_location
        execution_device_entry_symbol = self.execution_device_entry_symbol
        execution_identity = (
            execution_source_sha256,
            execution_source_slice_sha256,
            execution_source_location,
            execution_device_entry_symbol,
        )
        if any(value is not None for value in execution_identity) and not all(
            value is not None for value in execution_identity
        ):
            raise ValueError(
                "execution source identity must bind source, slice, location, and device entry together"
            )
        if execution_source_sha256 is not None:
            object.__setattr__(
                self,
                "execution_source_sha256",
                _manifest_digest(execution_source_sha256, "execution_source_sha256"),
            )
            object.__setattr__(
                self,
                "execution_source_slice_sha256",
                _manifest_digest(
                    execution_source_slice_sha256,
                    "execution_source_slice_sha256",
                ),
            )
            if not isinstance(execution_source_location, str) or not execution_source_location.strip():
                raise ValueError("execution_source_location must be a non-empty string")
            object.__setattr__(
                self,
                "execution_source_location",
                execution_source_location,
            )
            object.__setattr__(
                self,
                "execution_device_entry_symbol",
                _manifest_identifier(
                    execution_device_entry_symbol,
                    "execution_device_entry_symbol",
                ),
            )
        source_order = _manifest_names(self.source_parameter_order, context="source_parameter_order")
        runtime_order = _manifest_names(self.runtime_parameter_order, context="runtime_parameter_order")
        constexpr_order = _manifest_names(self.constexpr_parameter_order, context="constexpr_parameter_order")
        if set(runtime_order) & set(constexpr_order) or set(runtime_order) | set(constexpr_order) != set(source_order):
            raise ValueError("runtime and constexpr parameter orders must partition the source ABI")
        if runtime_order != tuple(name for name in source_order if name in set(runtime_order)):
            raise ValueError("runtime_parameter_order must preserve source declaration order")
        if constexpr_order != tuple(name for name in source_order if name in set(constexpr_order)):
            raise ValueError("constexpr_parameter_order must preserve source declaration order")
        object.__setattr__(self, "source_parameter_order", source_order)
        object.__setattr__(self, "runtime_parameter_order", runtime_order)
        object.__setattr__(self, "constexpr_parameter_order", constexpr_order)
        arguments = tuple(
            item if isinstance(item, GeneratedTargetSourceArgument)
            else GeneratedTargetSourceArgument.from_dict(item)
            for item in self.arguments
        )
        if tuple(item.name for item in arguments) != source_order:
            raise ValueError("manifest arguments must exactly follow source_parameter_order")
        object.__setattr__(self, "arguments", arguments)
        argument_by_name = {item.name: item for item in arguments}
        runtime_set = set(runtime_order)
        constexpr_set = set(constexpr_order)
        for name, argument in argument_by_name.items():
            if name in runtime_set:
                expected_categories = {
                    "buffer",
                    "runtime_scalar",
                    "none",
                    "opaque",
                }
            elif name in constexpr_set:
                expected_categories = {"semantic_constexpr", "c_slot"}
            else:  # pragma: no cover - source_order partition owns this
                raise ValueError(f"manifest argument {name!r} is outside the source ABI")
            if argument.category not in expected_categories:
                raise ValueError(
                    f"manifest argument {name!r} category {argument.category!r} "
                    "does not match its runtime/constexpr ABI partition"
                )
        if set(item.name for item in arguments if item.category == "buffer") - runtime_set:
            raise ValueError("buffer arguments must be runtime parameters")
        if set(item.name for item in arguments if item.category == "runtime_scalar") - runtime_set:
            raise ValueError("runtime scalar arguments must be runtime parameters")
        if set(item.name for item in arguments if item.category == "none") - runtime_set:
            raise ValueError("None arguments must be runtime parameters")
        if set(item.name for item in arguments if item.category == "opaque") - runtime_set:
            raise ValueError("opaque arguments must be runtime parameters")
        if set(item.name for item in arguments if item.category in {"semantic_constexpr", "c_slot"}) - constexpr_set:
            raise ValueError("constexpr arguments must be constexpr parameters")
        semantic_order = _manifest_names(self.semantic_constexpr_parameters, context="semantic_constexpr_parameters")
        c_source_order = _manifest_names(self.c_source_constexpr_parameters, context="c_source_constexpr_parameters")
        compiler_order = _manifest_names(self.compiler_slot_order, context="compiler_slot_order")
        c_order = _manifest_names(self.c_slot_order, context="c_slot_order")
        if (
            set(semantic_order) | set(c_source_order) != set(constexpr_order)
            or set(semantic_order) & set(c_source_order)
            or semantic_order != tuple(
                name for name in constexpr_order if name in set(semantic_order)
            )
            or c_source_order != tuple(
                name for name in constexpr_order if name in set(c_source_order)
            )
        ):
            raise ValueError("semantic and C source constexpr parameters must partition constexpr_parameter_order")
        if set(c_source_order) & set(compiler_order):
            raise ValueError("C source constexprs and compiler slots must be disjoint")
        if set(compiler_order) & set(source_order):
            raise ValueError("compiler slots must not be source parameters")
        if c_order != c_source_order + compiler_order:
            raise ValueError("c_slot_order must be C source constexprs followed by compiler slots")
        object.__setattr__(self, "semantic_constexpr_parameters", semantic_order)
        object.__setattr__(self, "c_source_constexpr_parameters", c_source_order)
        object.__setattr__(self, "compiler_slot_order", compiler_order)
        object.__setattr__(self, "c_slot_order", c_order)
        runtime_values = _manifest_pairs(self.runtime_scalar_values, context="runtime_scalar_values")
        semantic_values = _manifest_pairs(self.semantic_constexpr_values, context="semantic_constexpr_values")
        c_values = _manifest_pairs(self.c_slot_values, context="c_slot_values")
        if tuple(name for name, _value in runtime_values) != tuple(
            item.name for item in arguments if item.category == "runtime_scalar"
        ):
            raise ValueError("runtime_scalar_values must cover runtime scalar arguments in source order")
        if tuple(name for name, _value in semantic_values) != semantic_order:
            raise ValueError("semantic_constexpr_values must cover semantic constexprs in source order")
        if tuple(name for name, _value in c_values) != c_order:
            raise ValueError("c_slot_values must cover c_slot_order exactly")
        runtime_value_map = dict(runtime_values)
        semantic_value_map = dict(semantic_values)
        c_value_map = dict(c_values)
        for argument in arguments:
            if argument.category == "runtime_scalar":
                if not _manifest_value_equal(
                    runtime_value_map[argument.name], argument.value
                ):
                    raise ValueError(
                        f"runtime scalar valuation differs for {argument.name!r}"
                    )
            elif argument.category == "semantic_constexpr":
                if not _manifest_value_equal(
                    semantic_value_map[argument.name], argument.value
                ):
                    raise ValueError(
                        f"semantic constexpr valuation differs for {argument.name!r}"
                    )
            elif argument.category == "c_slot":
                if not _manifest_value_equal(
                    c_value_map[argument.name], argument.value
                ):
                    raise ValueError(
                        f"C source constexpr valuation differs for {argument.name!r}"
                    )
        object.__setattr__(self, "runtime_scalar_values", runtime_values)
        object.__setattr__(self, "semantic_constexpr_values", semantic_values)
        object.__setattr__(self, "c_slot_values", c_values)
        coordinates = _manifest_pairs(self.shape_coordinate_values, context="shape_coordinate_values")
        if any(not re.fullmatch(r"h[0-9]+", name) or not isinstance(value, int) or isinstance(value, bool) or value < 0 for name, value in coordinates):
            raise ValueError("shape_coordinate_values must contain non-negative h* valuations")
        expected_coordinates = tuple(f"h{index}" for index in range(len(coordinates)))
        if tuple(name for name, _value in coordinates) != expected_coordinates:
            raise ValueError("shape_coordinate_values must use gap-free h* order")
        object.__setattr__(self, "shape_coordinate_values", tuple((name, int(value)) for name, value in coordinates))
        coordinate_values = dict(coordinates)
        used_coordinates: set[str] = set()
        for argument in arguments:
            if argument.category in {"buffer", "opaque"} and argument.shape_contract is not None:
                for axis, (contract, extent) in enumerate(
                    zip(argument.shape_contract, argument.shape, strict=True)
                ):
                    if isinstance(contract, int):
                        if contract != extent:
                            raise ValueError(
                                f"manifest static shape contract differs for "
                                f"{argument.name!r} axis {axis}"
                            )
                        continue
                    used_coordinates.add(contract)
                    if contract not in coordinate_values:
                        raise ValueError(
                            f"manifest shape contract uses undeclared coordinate {contract!r}"
                        )
                    if coordinate_values[contract] != extent:
                        raise ValueError(
                            f"manifest shape coordinate valuation differs for {contract!r}"
                        )
            if argument.category in {"runtime_scalar", "semantic_constexpr"}:
                coordinate = argument.shape_coordinate
                if coordinate is not None:
                    used_coordinates.add(coordinate)
                    if coordinate not in coordinate_values:
                        raise ValueError(
                            f"manifest scalar uses undeclared shape coordinate {coordinate!r}"
                        )
                    if not _manifest_value_equal(
                        argument.value, coordinate_values[coordinate]
                    ):
                        raise ValueError(
                            f"manifest scalar valuation differs for {coordinate!r}"
                        )
        if used_coordinates != set(coordinate_values):
            raise ValueError(
                "manifest shape_coordinate_values must exactly cover coordinates "
                "used by tensor contracts and scalar bindings"
            )
        value_dimension_relations = _source_relation_tuple(
            self.value_dimension_relations,
            GeneratedSourceValueDimensionRelation,
            "manifest.value_dimension_relations",
        )
        dimension_relations = _source_relation_tuple(
            self.dimension_relations,
            GeneratedSourceDimensionRelation,
            "manifest.dimension_relations",
        )
        shape_coordinate_bindings = _source_relation_tuple(
            self.shape_coordinate_bindings,
            GeneratedSourceShapeCoordinateBinding,
            "manifest.shape_coordinate_bindings",
        )
        layout_relations = _source_relation_tuple(
            self.layout_relations,
            GeneratedSourceLayoutRelation,
            "manifest.layout_relations",
        )
        tensor_map_relations = _source_relation_tuple(
            self.tensor_map_relations,
            GeneratedSourceTensorMapRelation,
            "manifest.tensor_map_relations",
        )
        output_bindings = _source_relation_tuple(
            self.output_bindings,
            GeneratedSourceOutputBinding,
            "manifest.output_bindings",
        )
        _validate_source_relations(
            arguments,
            coordinates,
            value_dimension_relations,
            dimension_relations,
            shape_coordinate_bindings,
            layout_relations,
            tensor_map_relations,
            output_bindings,
        )
        object.__setattr__(self, "value_dimension_relations", value_dimension_relations)
        object.__setattr__(self, "dimension_relations", dimension_relations)
        object.__setattr__(self, "shape_coordinate_bindings", shape_coordinate_bindings)
        object.__setattr__(self, "layout_relations", layout_relations)
        object.__setattr__(self, "tensor_map_relations", tensor_map_relations)
        object.__setattr__(self, "output_bindings", output_bindings)
        j_value_dimension_relations = _j_relation_tuple(
            self.j_value_dimension_relations,
            ValueDimensionRelation,
            "manifest.j_value_dimension_relations",
        )
        j_dimension_relations = _j_relation_tuple(
            self.j_dimension_relations,
            DimensionRelation,
            "manifest.j_dimension_relations",
        )
        object.__setattr__(
            self,
            "j_value_dimension_relations",
            j_value_dimension_relations,
        )
        object.__setattr__(self, "j_dimension_relations", j_dimension_relations)
        grid = tuple(self.grid)
        if not 1 <= len(grid) <= 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid):
            raise ValueError("manifest grid must contain one to three positive dimensions")
        grid_contract = tuple(self.grid_contract)
        if len(grid_contract) != len(grid) or any(not isinstance(value, str) or not value.strip() for value in grid_contract):
            raise ValueError("manifest grid_contract must describe every grid axis")
        grid_coordinates = tuple(self.grid_shape_coordinates)
        if len(grid_coordinates) != len(grid):
            raise ValueError("manifest grid_shape_coordinates must describe every grid axis")
        coordinate_values = dict(coordinates)
        for axis, coordinate in enumerate(grid_coordinates):
            if coordinate is not None:
                if not isinstance(coordinate, str):
                    raise ValueError("manifest grid_shape_coordinates entries must be h* coordinates or null")
                if coordinate not in coordinate_values:
                    raise ValueError("manifest grid_shape_coordinates names an unknown h* coordinate")
                if grid[axis] != coordinate_values[coordinate]:
                    raise ValueError("manifest grid differs from its shape-coordinate valuation")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "grid_contract", grid_contract)
        object.__setattr__(self, "grid_shape_coordinates", grid_coordinates)
        digests = (self.j_digest, self.e_c_interface_digest, self.contract_ir_digest)
        if any(value is not None for value in digests) and not all(value is not None for value in digests):
            raise ValueError("J, E_C, and contract IR digests must be bound together")
        for name, value in zip(
            ("j_digest", "e_c_interface_digest", "contract_ir_digest"), digests,
            strict=True,
        ):
            if value is not None:
                _manifest_digest(value, name)
        if any(value is None for value in digests):
            if j_value_dimension_relations or j_dimension_relations:
                raise ValueError(
                    "unbound generated-source manifests cannot carry J relations"
                )
        lifecycle_records = tuple(
            item
            if isinstance(item, GeneratedSourceLifecycleRecord)
            else GeneratedSourceLifecycleRecord.from_dict(item)
            for item in self.lifecycle_records
        )
        _validate_lifecycle_records(
            lifecycle_records,
            current_source_slice_sha256=self.source_slice_sha256,
        )
        if execution_source_slice_sha256 is not None and lifecycle_records:
            if lifecycle_records[0].source_slice_sha256 != execution_source_slice_sha256:
                raise ValueError(
                    "lifecycle execution records do not match the preserved execution source slice"
                )
        object.__setattr__(self, "lifecycle_records", lifecycle_records)

    @property
    def bound(self) -> bool:
        return self.j_digest is not None

    @property
    def lifecycle_complete(self) -> bool:
        """Whether all four real lifecycle stages have been recorded."""

        return len(self.lifecycle_records) == len(_GENERATED_SOURCE_LIFECYCLE_PHASES)

    @property
    def output_bindings_digest(self) -> str:
        """Digest the manifest's stable logical output identity map."""

        return mapping_digest(
            {
                item.parameter: {
                    "output_key": item.output_key,
                    "role": item.role,
                }
                for item in self.output_bindings
            },
            purpose="generated-source-output-bindings",
        )

    def with_lifecycle_records(
        self,
        records: Iterable[GeneratedSourceLifecycleRecord | Mapping[str, Any]],
    ) -> "GeneratedTargetSourceManifest":
        """Return a manifest with a strictly ordered lifecycle prefix."""

        return replace(self, lifecycle_records=tuple(records))

    def execution_manifest(self) -> "GeneratedTargetSourceManifest":
        """Return the source manifest snapshot consumed by execution.

        Before re-ingress this is the current manifest with lifecycle records
        removed.  After re-ingress the preserved execution identity restores the
        original source fields while retaining the same J/E_C/ABI contract.
        """

        if self.execution_source_sha256 is None:
            return replace(self, lifecycle_records=())
        return replace(
            self,
            source_sha256=self.execution_source_sha256,
            source_slice_sha256=self.execution_source_slice_sha256,
            source_location=self.execution_source_location,
            device_entry_symbol=self.execution_device_entry_symbol,
            execution_source_sha256=None,
            execution_source_slice_sha256=None,
            execution_source_location=None,
            execution_device_entry_symbol=None,
            lifecycle_records=(),
        )

    def execution_lineage(self) -> dict[str, Any]:
        """Return the common manifest metadata consumed by Gate2 runners."""

        execution_manifest = self.execution_manifest()
        final_manifest = self if self.lifecycle_records else None
        reingress_slice = None
        if self.lifecycle_records and self.lifecycle_records[-1].phase == "reingress":
            reingress_slice = self.lifecycle_records[-1].source_slice_sha256
        return {
            # These two names are retained as compatibility aliases, but they
            # always mean the execution snapshot, never the later final write.
            "generated_target_source_manifest": execution_manifest.to_dict(),
            "generated_target_source_manifest_digest": execution_manifest.digest,
            "execution_generated_target_source_manifest": execution_manifest.to_dict(),
            "execution_generated_target_source_manifest_digest": execution_manifest.digest,
            "final_generated_target_source_manifest": (
                final_manifest.to_dict() if final_manifest is not None else None
            ),
            "final_generated_target_source_manifest_digest": (
                final_manifest.digest if final_manifest is not None else None
            ),
            "source_slice_sha256": execution_manifest.source_slice_sha256,
            "source_sha256": execution_manifest.source_sha256,
            "source_location": execution_manifest.source_location,
            "device_entry_symbol": execution_manifest.device_entry_symbol,
            "manifest_output_binding_digest": execution_manifest.output_bindings_digest,
            "source_parameter_order": list(execution_manifest.source_parameter_order),
            "constexpr_parameter_order": list(execution_manifest.constexpr_parameter_order),
            "compiler_slot_order": list(execution_manifest.compiler_slot_order),
            "c_slot_order": list(execution_manifest.c_slot_order),
            "manifest_grid": list(execution_manifest.grid),
            "manifest_grid_contract": list(execution_manifest.grid_contract),
            "reingress_source_slice_sha256": reingress_slice,
            "lineage_required": True,
        }

    def reingress_source_slice(
        self,
        *,
        source: str,
        source_slice: Any,
        source_location: str,
        compiler_view: SCRCCompilerView,
        source_to_opaque: Mapping[str, str],
        request: SCRCRunRequest | None = None,
        consumer: str = "source-reingress",
    ) -> "GeneratedTargetSourceManifest":
        """Admit one explicit, ABI-checked source re-ingress.

        Re-ingress is a new source observation, not a filename or kernel-name
        lookup.  The selected source slice must expose the same anonymous ABI
        and compiler view, while its own source and slice bytes receive fresh
        digests.  A re-ingress is admitted only after the three execution stages
        have real records, so an unexecuted source cannot masquerade as a
        round-trip result.
        """

        if len(self.lifecycle_records) != 3:
            raise ValueError(
                "source re-ingress requires source_compile, source_execute, and output_digest records"
            )
        if not isinstance(source, str) or not source:
            raise ValueError("re-ingress source must be a non-empty string")
        if not isinstance(source_location, str) or not source_location.strip():
            raise ValueError("re-ingress source location must be non-empty")
        if not hasattr(source_slice, "code") or not hasattr(source_slice, "function"):
            raise TypeError("re-ingress source slice has no source projection interface")
        selected_parameters = tuple(getattr(source_slice, "parameters", ()))
        selected_constexpr = tuple(
            getattr(source_slice, "constexpr_parameters", ())
        )
        if selected_parameters != self.source_parameter_order:
            raise ValueError(
                "re-ingress source parameter order differs from the generated-source ABI"
            )
        if selected_constexpr != self.constexpr_parameter_order:
            raise ValueError(
                "re-ingress constexpr order differs from the generated-source ABI"
            )
        mapping = dict(source_to_opaque)
        expected_mapping_keys = set(self.source_parameter_order) | set(
            self.compiler_slot_order
        )
        if set(mapping) != expected_mapping_keys:
            raise ValueError(
                "re-ingress source-to-anonymous mapping does not exactly cover the ABI"
            )
        if len(mapping) != len(set(mapping.values())) or any(
            not isinstance(name, str)
            or not name.isidentifier()
            or not isinstance(anonymous, str)
            or re.fullmatch(r"(?:p|c)[0-9]+", anonymous) is None
            for name, anonymous in mapping.items()
        ):
            raise ValueError("re-ingress source-to-anonymous mapping is not bijective")

        next_manifest = replace(
            self,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            source_slice_sha256=hashlib.sha256(
                str(source_slice.code).encode("utf-8")
            ).hexdigest(),
            source_location=source_location,
            device_entry_symbol=source_slice.function,
            execution_source_sha256=self.execution_manifest().source_sha256,
            execution_source_slice_sha256=self.execution_manifest().source_slice_sha256,
            execution_source_location=self.execution_manifest().source_location,
            execution_device_entry_symbol=self.execution_manifest().device_entry_symbol,
            lifecycle_records=(),
        )
        next_manifest.validate_source_slice(source_slice, source_location)
        next_manifest.validate_compiler_view(
            compiler_view,
            source_to_opaque=mapping,
            request=request,
        )
        reingress_slice = next_manifest.source_slice_sha256
        record = GeneratedSourceLifecycleRecord(
            phase="reingress",
            consumer=consumer,
            artifact_kind="source_slice",
            artifact_digest=reingress_slice,
            source_slice_sha256=reingress_slice,
            output_binding_digest=next_manifest.output_bindings_digest,
        )
        return next_manifest.with_lifecycle_records(
            (*self.lifecycle_records, record)
        )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_source(
        self,
        source: str,
        source_slice: "TritonSourceSlice",
        source_location: str,
    ) -> None:
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != self.source_sha256:
            raise ValueError("generated-source manifest source SHA-256 differs from the real source")
        self.validate_source_slice(source_slice, source_location)

    def validate_source_slice(
        self,
        source_slice: "TritonSourceSlice",
        source_location: str,
    ) -> None:
        if hashlib.sha256(source_slice.code.encode("utf-8")).hexdigest() != self.source_slice_sha256:
            raise ValueError("generated-source manifest source-slice SHA-256 differs from the selected slice")
        if source_location != self.source_location:
            raise ValueError("generated-source manifest source location differs")
        if source_slice.function != self.device_entry_symbol:
            raise ValueError("generated-source manifest device entry differs from source slice")
        if source_slice.parameters != self.source_parameter_order:
            raise ValueError("generated-source manifest source parameter order differs")
        if source_slice.constexpr_parameters != self.constexpr_parameter_order:
            raise ValueError("generated-source manifest constexpr parameter order differs")

    def validate_compiler_view(
        self,
        compiler_view: SCRCCompilerView,
        *,
        source_to_opaque: Mapping[str, str] | None = None,
        request: SCRCRunRequest | None = None,
    ) -> None:
        """Admit compiler metadata and Gate request bindings against this ABI.

        The source manifest is the physical/source-name view.  ``compiler_view``
        is the logical compiler view and may therefore represent a physical
        opaque source slot as a logical tensor.  The optional request check then
        verifies the anonymous Gate compile case and its exact slot order.
        """

        if not isinstance(compiler_view, SCRCCompilerView):
            raise TypeError("generated-source compiler view has an invalid type")
        if compiler_view.parameter_order != self.source_parameter_order:
            raise ValueError(
                "compiler view parameter order differs from the generated-source ABI"
            )
        if compiler_view.constexpr_parameters != self.constexpr_parameter_order:
            raise ValueError(
                "compiler view constexpr order differs from the generated-source ABI"
            )
        runtime_arguments = tuple(compiler_view.arguments)
        expected_runtime = tuple(
            item for item in self.arguments if item.name in self.runtime_parameter_order
        )
        if tuple(item.name for item in runtime_arguments) != self.runtime_parameter_order:
            raise ValueError("compiler view runtime argument order differs from source ABI")
        if len(runtime_arguments) != len(expected_runtime):
            raise ValueError("compiler view runtime argument count differs from source ABI")

        def dtype_equal(left: str | None, right: str | None) -> bool:
            aliases = {
                "bool": "i1", "boolean": "i1", "float16": "f16",
                "half": "f16", "float32": "f32", "float": "f32",
                "float64": "f64", "double": "f64", "bfloat16": "bf16",
                "int8": "i8", "int16": "i16", "int32": "i32",
                "int64": "i64", "index": "i64", "uint8": "u8",
                "uint16": "u16", "uint32": "u32", "str": "string",
                "string": "string",
            }
            if left is None or right is None:
                return left == right
            return aliases.get(left.casefold().removeprefix("torch."), left.casefold()) == aliases.get(
                right.casefold().removeprefix("torch."), right.casefold()
            )

        for manifest_argument, compiler_argument in zip(
            expected_runtime, runtime_arguments, strict=True
        ):
            if manifest_argument.category in {"buffer", "opaque"}:
                if compiler_argument.kind != "tensor":
                    raise ValueError(
                        f"compiler view argument {manifest_argument.name!r} is not a logical tensor"
                    )
                expected_dtype = (
                    manifest_argument.logical_dtype or manifest_argument.dtype
                )
                if not dtype_equal(compiler_argument.dtype, expected_dtype):
                    raise ValueError(
                        f"compiler view dtype differs for {manifest_argument.name!r}"
                    )
                if compiler_argument.shape != manifest_argument.shape:
                    raise ValueError(
                        f"compiler view shape differs for {manifest_argument.name!r}"
                    )
                if compiler_argument.shape_contract != manifest_argument.shape_contract:
                    raise ValueError(
                        f"compiler view shape contract differs for {manifest_argument.name!r}"
                    )
                if compiler_argument.mode != manifest_argument.role:
                    raise ValueError(
                        f"compiler view mode differs for {manifest_argument.name!r}"
                    )
                if compiler_argument.strides != manifest_argument.strides:
                    raise ValueError(
                        f"compiler view strides differ for {manifest_argument.name!r}"
                    )
                continue
            if manifest_argument.category == "runtime_scalar":
                if compiler_argument.kind != "scalar" or compiler_argument.mode != "value":
                    raise ValueError(
                        f"compiler view scalar ABI differs for {manifest_argument.name!r}"
                    )
                if not dtype_equal(compiler_argument.dtype, manifest_argument.dtype):
                    raise ValueError(
                        f"compiler view scalar dtype differs for {manifest_argument.name!r}"
                    )
                if not _manifest_value_equal(compiler_argument.value, manifest_argument.value):
                    raise ValueError(
                        f"compiler view scalar valuation differs for {manifest_argument.name!r}"
                    )
                if compiler_argument.shape_coordinate != manifest_argument.shape_coordinate:
                    raise ValueError(
                        f"compiler view scalar coordinate differs for {manifest_argument.name!r}"
                    )
                continue
            if manifest_argument.category == "none":
                if compiler_argument.kind != "none" or compiler_argument.mode != "value":
                    raise ValueError(
                        f"compiler view None ABI differs for {manifest_argument.name!r}"
                    )
                continue
            raise ValueError(
                f"unexpected non-runtime argument in compiler view: {manifest_argument.name!r}"
            )

        if tuple(compiler_view.schedule_values) != self.c_slot_order:
            raise ValueError(
                "compiler view schedule declaration order differs from source C ABI"
            )
        if tuple(compiler_view.schedule_value_types) != self.c_slot_order:
            raise ValueError(
                "compiler view schedule value-type order differs from source C ABI"
            )
        if tuple(compiler_view.compiler_slots) != self.compiler_slot_order:
            raise ValueError("compiler view compiler-slot order differs from source C ABI")
        if dict(compiler_view.bound_constexpr_values) != dict(self.semantic_constexpr_values):
            raise ValueError("compiler view semantic constexpr values differ from source ABI")
        expected_bound_constexpr_types = {
            item.name: item.dtype
            for item in self.arguments
            if item.category == "semantic_constexpr"
        }
        if tuple(compiler_view.bound_constexpr_value_types) != tuple(
            self.semantic_constexpr_parameters
        ):
            raise ValueError(
                "compiler view semantic constexpr type order differs from source ABI"
            )
        if set(compiler_view.bound_constexpr_value_types) != set(
            expected_bound_constexpr_types
        ):
            raise ValueError(
                "compiler view semantic constexpr types differ from source ABI"
            )
        for name, expected_type in expected_bound_constexpr_types.items():
            if not dtype_equal(
                compiler_view.bound_constexpr_value_types[name], expected_type
            ):
                raise ValueError(
                    f"compiler view semantic constexpr type differs for {name!r}"
                )
        for name, value in self.c_slot_values:
            if not _manifest_value_equal(compiler_view.schedule_values.get(name), value):
                raise ValueError(f"compiler view C-slot valuation differs for {name!r}")
        expected_bound_shape_coordinates = {
            item.name: item.shape_coordinate
            for item in self.arguments
            if item.category == "semantic_constexpr"
            and item.shape_coordinate is not None
        }
        if dict(compiler_view.bound_constexpr_shape_coordinates) != (
            expected_bound_shape_coordinates
        ):
            raise ValueError(
                "compiler view constexpr shape-coordinate bindings differ from source ABI"
            )
        if tuple(compiler_view.grid) != self.grid:
            raise ValueError("compiler view grid differs from generated-source ABI")
        if compiler_view.grid_contract != self.grid_contract:
            raise ValueError("compiler view grid contract differs from generated-source ABI")
        compiler_grid_coordinates = tuple(compiler_view.grid_shape_coordinates) or tuple(
            None for _ in compiler_view.grid
        )
        if compiler_grid_coordinates != self.grid_shape_coordinates:
            raise ValueError(
                "compiler view grid shape-coordinate bindings differ from source ABI"
            )

        if request is None:
            return
        if not isinstance(request, SCRCRunRequest):
            raise TypeError("generated-source request has an invalid type")
        mapping = dict(source_to_opaque or {})
        expected_mapping_keys = set(self.source_parameter_order) | set(
            self.compiler_slot_order
        )
        if set(mapping) != expected_mapping_keys:
            raise ValueError("source-to-anonymous mapping does not exactly cover the ABI")
        if len(mapping) != len(set(mapping.values())):
            raise ValueError("source-to-anonymous mapping is not one-to-one")
        if any(
            not isinstance(source, str)
            or not source.isidentifier()
            or not isinstance(anonymous, str)
            or re.fullmatch(r"(?:p|c)[0-9]+", anonymous) is None
            for source, anonymous in mapping.items()
        ):
            raise ValueError("source-to-anonymous mapping must use p*/c* handles")
        case = request.binding.gate1_case
        # Gate compilation receives concrete values for both semantic
        # constexprs (derived from the bound logical shape/ABI) and C-owned
        # realization slots.  Only the latter belong to C; omitting the former
        # here made an otherwise valid source request look detached from its
        # manifest.
        expected_schedule = {
            mapping[name]: value
            for name, value in (
                *self.semantic_constexpr_values,
                *self.c_slot_values,
            )
        }
        case_schedule = getattr(case, "schedule", None)
        if not isinstance(case_schedule, Mapping) or dict(case_schedule) != expected_schedule:
            raise ValueError("Gate compile schedule differs from generated-source ABI")
        case_arguments = getattr(case, "arguments", None)
        if isinstance(case_arguments, Mapping):
            expected_names = tuple(
                mapping[name] for name in self.runtime_parameter_order
            )
            if tuple(case_arguments) != expected_names:
                raise ValueError("Gate runtime argument order differs from source ABI")
            for source_name, anonymous_name in zip(
                self.runtime_parameter_order, expected_names, strict=True
            ):
                manifest_argument = next(
                    item for item in self.arguments if item.name == source_name
                )
                actual = case_arguments[anonymous_name]
                if manifest_argument.category in {"buffer", "opaque"}:
                    expected_dtype = manifest_argument.logical_dtype or manifest_argument.dtype
                    if (
                        actual.kind != "tensor"
                        or not dtype_equal(actual.dtype, expected_dtype)
                        or actual.shape != manifest_argument.shape
                        or actual.strides != manifest_argument.strides
                        or actual.value is not None
                    ):
                        raise ValueError(
                            f"Gate runtime tensor {anonymous_name!r} differs from source ABI"
                        )
                elif manifest_argument.category == "runtime_scalar":
                    if (
                        actual.kind != "scalar"
                        or not dtype_equal(actual.dtype, manifest_argument.dtype)
                        or not _manifest_value_equal(actual.value, manifest_argument.value)
                        or actual.shape is not None
                        or actual.strides is not None
                    ):
                        raise ValueError(
                            f"Gate runtime scalar {anonymous_name!r} differs from source ABI"
                        )
                elif manifest_argument.category == "none":
                    if (
                        actual.kind != "none"
                        or not dtype_equal(actual.dtype, manifest_argument.dtype)
                        or actual.shape is not None
                        or actual.strides is not None
                        or actual.value is not None
                    ):
                        raise ValueError(
                            f"Gate None runtime argument {anonymous_name!r} differs from source ABI"
                        )
        else:
            case_parameters = getattr(case, "parameters", None)
            if not isinstance(case_parameters, Sequence):
                raise ValueError("Gate compile case lacks an ordered parameter ABI")
            expected_parameters: list[tuple[str, str, int, str]] = []
            for name in self.runtime_parameter_order:
                argument = next(item for item in self.arguments if item.name == name)
                if argument.category == "opaque":
                    expected_parameters.append(
                        ("pointer", argument.logical_dtype or "", len(argument.shape or ()), "input")
                    )
                elif argument.category == "buffer":
                    expected_parameters.append(
                        ("pointer", argument.dtype or "", len(argument.shape or ()), argument.role or "")
                    )
                elif argument.category == "runtime_scalar":
                    expected_parameters.append(("scalar", argument.dtype or "", 0, "value"))
                else:
                    raise ValueError("native Gate case cannot represent a None argument")
            if len(case_parameters) != len(expected_parameters):
                raise ValueError("native Gate parameter count differs from source ABI")
            for index, (actual, expected) in enumerate(zip(case_parameters, expected_parameters, strict=True)):
                if not (
                    actual.kind == expected[0]
                    and dtype_equal(actual.dtype, expected[1])
                    and actual.rank == expected[2]
                    and actual.mode == expected[3]
                ):
                    raise ValueError(f"native Gate parameter {index} differs from source ABI")
            physical_parameters = getattr(case, "physical_parameters", None)
            if physical_parameters is not None:
                expected_physical: list[tuple[str, str, int, str]] = []
                for name in self.runtime_parameter_order:
                    argument = next(item for item in self.arguments if item.name == name)
                    if argument.category == "opaque":
                        expected_physical.append(
                            (
                                "opaque",
                                argument.dtype or "",
                                len(argument.shape or ()),
                                argument.role or "value",
                            )
                        )
                    elif argument.category == "buffer":
                        expected_physical.append(
                            (
                                "pointer",
                                argument.dtype or "",
                                len(argument.shape or ()),
                                argument.role or "",
                            )
                        )
                    elif argument.category == "runtime_scalar":
                        expected_physical.append(
                            ("scalar", argument.dtype or "", 0, "value")
                        )
                    else:
                        raise ValueError(
                            "native Gate physical ABI cannot represent a None argument"
                        )
                if len(physical_parameters) != len(expected_physical):
                    raise ValueError(
                        "native Gate physical parameter count differs from source ABI"
                    )
                for index, (actual, expected) in enumerate(
                    zip(physical_parameters, expected_physical, strict=True)
                ):
                    if not (
                        actual.kind == expected[0]
                        and dtype_equal(actual.dtype, expected[1])
                        and actual.rank == expected[2]
                        and actual.mode == expected[3]
                    ):
                        raise ValueError(
                            f"native Gate physical parameter {index} differs from source ABI"
                        )
        case_grid = getattr(case, "grid", None)
        if case_grid is not None and tuple(case_grid) != self.grid:
            raise ValueError("Gate compile grid differs from generated-source ABI")

    def require_contract(self) -> "GeneratedTargetSourceManifest":
        if not self.bound:
            raise ValueError("generated-source manifest is not bound to J/E_C/contract IR")
        return self

    def bind_contract(
        self,
        j: Any,
        c: Any,
        *,
        source_to_opaque: Mapping[str, str] | None = None,
    ) -> "GeneratedTargetSourceManifest":
        """Bind exact J/E_C/contract identity and the real C slot valuation.

        The generated source uses source-level constexpr/compiler names, while
        the accepted Gate1 artifact uses anonymous ``c*`` handles.  A digest
        match alone cannot distinguish two C valuations attached to the same
        J, so this boundary checks the ordered name mapping, slot kind/type,
        trusted compiler role, required bit, and scalar value before accepting
        the manifest for Gate2 or target-source re-ingress.
        """

        try:
            canonical = elaborate_j(j)
            interface = source_layer_c_interface(canonical, c)
            contract = project_contract_ir(canonical)
            contract_digest = contract.digest
            values = {
                "j_digest": layer_c_content_digest(canonical.to_json()),
                "e_c_interface_digest": binding_interface_digest(interface),
                "contract_ir_digest": contract_digest,
            }
        except Exception as exc:
            raise ValueError(f"cannot bind generated-source manifest to J/E_C: {exc}") from exc

        j_value_dimension_relations = tuple(canonical.value_dimension_relations)
        j_dimension_relations = tuple(canonical.dimension_relations)

        if source_to_opaque is None:
            raise ValueError(
                "binding a generated-source manifest to J requires the explicit "
                "source-to-anonymous ABI mapping"
            )
        mapping = dict(source_to_opaque)
        c_records = _manifest_c_slot_records(c)
        missing_mapping = sorted(
            name for name in self.c_slot_order if name not in mapping
        )
        if missing_mapping:
            raise ValueError(
                "generated-source manifest C slot mapping is incomplete: "
                + ", ".join(missing_mapping)
            )
        expected_c_names = tuple(mapping[name] for name in self.c_slot_order)
        if len(expected_c_names) != len(set(expected_c_names)):
            raise ValueError(
                "generated-source manifest C slot mapping is not one-to-one"
            )
        actual_c_names = tuple(item[0] for item in c_records)
        if actual_c_names != expected_c_names:
            raise ValueError(
                "generated-source manifest C slot declaration order differs from "
                "the accepted artifact"
            )
        compiler_slots = set(self.compiler_slot_order)
        expected_values = dict(self.c_slot_values)
        arguments = {item.name: item for item in self.arguments}
        trusted_points = {
            item.slot: item for item in interface.points
        }
        for source_name, actual in zip(self.c_slot_order, c_records, strict=True):
            _actual_name, kind, role, value_type, required, value = actual
            actual_name = _actual_name
            point = trusted_points.get(actual_name)
            expected_kind = "compiler" if source_name in compiler_slots else "constexpr"
            if kind != expected_kind:
                raise ValueError(
                    f"accepted C slot {source_name!r} has kind {kind!r}, "
                    f"expected {expected_kind!r}"
                )
            if expected_kind == "constexpr":
                if point is None:
                    raise ValueError(
                        f"accepted C slot {actual_name!r} lacks its trusted E_C declaration"
                    )
                if (
                    point.role != role
                    or point.value_type != value_type
                    or point.required != required
                ):
                    raise ValueError(
                        f"accepted C slot {actual_name!r} disagrees with its trusted E_C metadata"
                    )
            elif point is not None:
                raise ValueError(
                    f"compiler-only C slot {actual_name!r} must not appear in shared E_C"
                )
            argument = arguments.get(source_name)
            if expected_kind == "constexpr":
                if argument is None or argument.category != "c_slot":
                    raise ValueError(
                        f"generated-source manifest lacks C ABI metadata for {source_name!r}"
                    )
                if value_type != argument.dtype:
                    raise ValueError(
                        f"accepted C slot {source_name!r} has value_type {value_type!r}, "
                        f"expected {argument.dtype!r}"
                    )
            elif argument is not None:
                raise ValueError(
                    f"compiler-only C slot {source_name!r} must not be a source argument"
                )
            if not required:
                raise ValueError(
                    f"accepted active C slot {source_name!r} is not required"
                )
            if source_name in compiler_slots and role != source_name:
                raise ValueError(
                    f"accepted compiler slot {source_name!r} changed its trusted role"
                )
            if not _manifest_value_equal(value, expected_values[source_name]):
                raise ValueError(
                    f"accepted C slot {source_name!r} value differs from the "
                    "captured generated-source valuation"
                )
        _validate_source_j_relation_closure(
            arguments=self.arguments,
            source_parameter_order=self.source_parameter_order,
            compiler_slot_order=self.compiler_slot_order,
            shape_coordinate_values=self.shape_coordinate_values,
            value_dimension_relations=self.value_dimension_relations,
            dimension_relations=self.dimension_relations,
            output_bindings=self.output_bindings,
            source_to_opaque=mapping,
            canonical=canonical,
            contract=contract,
        )
        existing = (self.j_digest, self.e_c_interface_digest, self.contract_ir_digest)
        actual = (values["j_digest"], values["e_c_interface_digest"], values["contract_ir_digest"])
        if any(item is not None for item in existing) and existing != actual:
            raise ValueError("generated-source manifest is already bound to different J/E_C/contract IR")
        if self.bound and (
            self.j_value_dimension_relations != j_value_dimension_relations
            or self.j_dimension_relations != j_dimension_relations
        ):
            raise ValueError(
                "generated-source manifest is already bound to different J relations"
            )
        return replace(
            self,
            **values,
            j_value_dimension_relations=j_value_dimension_relations,
            j_dimension_relations=j_dimension_relations,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedTargetSourceManifest":
        keys = {
            "protocol", "source_sha256", "source_slice_sha256", "source_location",
            "device_entry_symbol", "source_parameter_order", "runtime_parameter_order",
            "constexpr_parameter_order", "arguments", "semantic_constexpr_parameters",
            "c_source_constexpr_parameters", "compiler_slot_order", "c_slot_order",
            "runtime_scalar_values", "semantic_constexpr_values", "c_slot_values",
            "shape_coordinate_values", "grid", "grid_contract", "grid_shape_coordinates",
            "value_dimension_relations", "dimension_relations",
            "shape_coordinate_bindings", "layout_relations", "tensor_map_relations",
            "output_bindings", "j_value_dimension_relations", "j_dimension_relations",
            "j_digest", "e_c_interface_digest", "contract_ir_digest",
            "execution_source_sha256", "execution_source_slice_sha256",
            "execution_source_location", "execution_device_entry_symbol",
            "lifecycle_records",
        }
        optional_keys = {
            "execution_source_sha256",
            "execution_source_slice_sha256",
            "execution_source_location",
            "execution_device_entry_symbol",
            "lifecycle_records",
        }
        required_keys = keys - optional_keys
        if (
            not isinstance(value, Mapping)
            or not required_keys <= set(value)
            or not (set(value) - required_keys) <= optional_keys
            or any(not isinstance(key, str) for key in value)
        ):
            raise ValueError(
                "GeneratedTargetSourceManifest must contain exactly "
                + ", ".join(sorted(keys))
            )
        for name in (
            "source_parameter_order", "runtime_parameter_order", "constexpr_parameter_order",
            "semantic_constexpr_parameters", "c_source_constexpr_parameters",
            "compiler_slot_order", "c_slot_order", "arguments", "grid",
            "grid_contract", "grid_shape_coordinates", "value_dimension_relations",
            "dimension_relations", "shape_coordinate_bindings", "layout_relations",
            "tensor_map_relations", "output_bindings", "j_value_dimension_relations",
            "j_dimension_relations", "lifecycle_records",
        ):
            if name in value and not isinstance(value[name], list):
                raise ValueError(f"manifest.{name} must be an array")
        arguments = tuple(
            GeneratedTargetSourceArgument.from_dict(item)
            for item in value["arguments"]
        )
        source_order = tuple(value["source_parameter_order"])
        constexpr_order = tuple(value["constexpr_parameter_order"])
        semantic_order = tuple(value["semantic_constexpr_parameters"])
        c_source_order = tuple(value["c_source_constexpr_parameters"])
        compiler_order = tuple(value["compiler_slot_order"])
        c_order = tuple(value["c_slot_order"])
        runtime_scalar_order = tuple(
            item.name for item in arguments if item.category == "runtime_scalar"
        )
        raw_coordinates = _manifest_object(
            value["shape_coordinate_values"], "manifest.shape_coordinate_values"
        )
        if any(
            not isinstance(name, str) or re.fullmatch(r"h[0-9]+", name) is None
            for name in raw_coordinates
        ):
            raise ValueError(
                "manifest.shape_coordinate_values keys must be h* coordinates"
            )
        coordinate_order = tuple(
            sorted(raw_coordinates, key=lambda name: int(name[1:]))
        )
        return cls(
            source_sha256=value["source_sha256"],
            source_slice_sha256=value["source_slice_sha256"],
            source_location=value["source_location"],
            device_entry_symbol=value["device_entry_symbol"],
            source_parameter_order=source_order,
            runtime_parameter_order=tuple(value["runtime_parameter_order"]),
            constexpr_parameter_order=constexpr_order,
            arguments=arguments,
            semantic_constexpr_parameters=semantic_order,
            c_source_constexpr_parameters=c_source_order,
            compiler_slot_order=compiler_order,
            c_slot_order=c_order,
            runtime_scalar_values=_manifest_ordered_pairs(
                value["runtime_scalar_values"],
                runtime_scalar_order,
                context="manifest.runtime_scalar_values",
            ),
            semantic_constexpr_values=_manifest_ordered_pairs(
                value["semantic_constexpr_values"],
                semantic_order,
                context="manifest.semantic_constexpr_values",
            ),
            c_slot_values=_manifest_ordered_pairs(
                value["c_slot_values"],
                c_order,
                context="manifest.c_slot_values",
            ),
            shape_coordinate_values=_manifest_ordered_pairs(
                raw_coordinates,
                coordinate_order,
                context="manifest.shape_coordinate_values",
            ),
            grid=tuple(value["grid"]),
            grid_contract=tuple(value["grid_contract"]),
            grid_shape_coordinates=tuple(value["grid_shape_coordinates"]),
            value_dimension_relations=tuple(
                GeneratedSourceValueDimensionRelation.from_dict(item)
                for item in value["value_dimension_relations"]
            ),
            dimension_relations=tuple(
                GeneratedSourceDimensionRelation.from_dict(item)
                for item in value["dimension_relations"]
            ),
            shape_coordinate_bindings=tuple(
                GeneratedSourceShapeCoordinateBinding.from_dict(item)
                for item in value["shape_coordinate_bindings"]
            ),
            layout_relations=tuple(
                GeneratedSourceLayoutRelation.from_dict(item)
                for item in value["layout_relations"]
            ),
            tensor_map_relations=tuple(
                GeneratedSourceTensorMapRelation.from_dict(item)
                for item in value["tensor_map_relations"]
            ),
            output_bindings=tuple(
                GeneratedSourceOutputBinding.from_dict(item)
                for item in value["output_bindings"]
            ),
            j_value_dimension_relations=tuple(
                ValueDimensionRelation.from_json(item)
                for item in value["j_value_dimension_relations"]
            ),
            j_dimension_relations=tuple(
                DimensionRelation.from_json(item)
                for item in value["j_dimension_relations"]
            ),
            j_digest=value["j_digest"],
            e_c_interface_digest=value["e_c_interface_digest"],
            contract_ir_digest=value["contract_ir_digest"],
            execution_source_sha256=value.get("execution_source_sha256"),
            execution_source_slice_sha256=value.get(
                "execution_source_slice_sha256"
            ),
            execution_source_location=value.get("execution_source_location"),
            execution_device_entry_symbol=value.get(
                "execution_device_entry_symbol"
            ),
            lifecycle_records=tuple(
                GeneratedSourceLifecycleRecord.from_dict(item)
                for item in value.get("lifecycle_records", [])
            ),
            protocol=value["protocol"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "source_sha256": self.source_sha256,
            "source_slice_sha256": self.source_slice_sha256,
            "source_location": self.source_location,
            "device_entry_symbol": self.device_entry_symbol,
            "source_parameter_order": list(self.source_parameter_order),
            "runtime_parameter_order": list(self.runtime_parameter_order),
            "constexpr_parameter_order": list(self.constexpr_parameter_order),
            "arguments": [item.to_dict() for item in self.arguments],
            "semantic_constexpr_parameters": list(self.semantic_constexpr_parameters),
            "c_source_constexpr_parameters": list(self.c_source_constexpr_parameters),
            "compiler_slot_order": list(self.compiler_slot_order),
            "c_slot_order": list(self.c_slot_order),
            "runtime_scalar_values": dict(self.runtime_scalar_values),
            "semantic_constexpr_values": dict(self.semantic_constexpr_values),
            "c_slot_values": dict(self.c_slot_values),
            "shape_coordinate_values": dict(self.shape_coordinate_values),
            "grid": list(self.grid),
            "grid_contract": list(self.grid_contract),
            "grid_shape_coordinates": list(self.grid_shape_coordinates),
            "value_dimension_relations": [
                item.to_dict() for item in self.value_dimension_relations
            ],
            "dimension_relations": [item.to_dict() for item in self.dimension_relations],
            "shape_coordinate_bindings": [
                item.to_dict() for item in self.shape_coordinate_bindings
            ],
            "layout_relations": [item.to_dict() for item in self.layout_relations],
            "tensor_map_relations": [
                item.to_dict() for item in self.tensor_map_relations
            ],
            "output_bindings": [item.to_dict() for item in self.output_bindings],
            "j_value_dimension_relations": [
                item.to_json() for item in self.j_value_dimension_relations
            ],
            "j_dimension_relations": [
                item.to_json() for item in self.j_dimension_relations
            ],
            "j_digest": self.j_digest,
            "e_c_interface_digest": self.e_c_interface_digest,
            "contract_ir_digest": self.contract_ir_digest,
            "execution_source_sha256": self.execution_source_sha256,
            "execution_source_slice_sha256": self.execution_source_slice_sha256,
            "execution_source_location": self.execution_source_location,
            "execution_device_entry_symbol": self.execution_device_entry_symbol,
            "lifecycle_records": [
                item.to_dict() for item in self.lifecycle_records
            ],
        }


@dataclass(frozen=True)
class PreparedTritonWorkload:
    adapter: "TritonWorkloadAdapter"
    entry: SCRCWorkloadEntry
    source_location: str
    source_slice: TritonSourceSlice
    compiler_view: SCRCCompilerView
    request: SCRCRunRequest
    source_to_opaque: Mapping[str, str]
    manifest: GeneratedTargetSourceManifest

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, GeneratedTargetSourceManifest):
            raise TypeError("PreparedTritonWorkload requires a generated-source manifest")
        self.manifest.validate_source_slice(
            self.source_slice,
            self.source_location,
        )
        mapping = dict(self.source_to_opaque)
        expected_mapping_keys = set(self.manifest.source_parameter_order) | set(
            self.manifest.compiler_slot_order
        )
        if set(mapping) != expected_mapping_keys:
            missing = sorted(expected_mapping_keys - set(mapping))
            extra = sorted(set(mapping) - expected_mapping_keys)
            raise ValueError(
                "Triton source_to_opaque mapping must exactly cover the ABI"
                + ("; missing " + ", ".join(missing) if missing else "")
                + ("; extra " + ", ".join(extra) if extra else "")
            )
        if len(mapping) != len(set(mapping.values())):
            raise ValueError("Triton source_to_opaque mapping must be one-to-one")
        if any(
            not isinstance(source, str)
            or not source.isidentifier()
            or not isinstance(opaque, str)
            or re.fullmatch(r"(?:p|c)[0-9]+", opaque) is None
            for source, opaque in mapping.items()
        ):
            raise ValueError(
                "Triton source_to_opaque mapping must use p*/c* handles"
            )
        c_values = {mapping[name] for name in self.manifest.c_slot_order}
        if c_values != set(self.request.binding.values):
            raise ValueError(
                "Triton source/C mapping differs from request c0"
            )
        self.manifest.validate_compiler_view(
            self.compiler_view,
            source_to_opaque=mapping,
            request=self.request,
        )
        object.__setattr__(self, "source_to_opaque", mapping)

    def with_contract(self, j: Any, c: Any) -> "PreparedTritonWorkload":
        """Bind the exact accepted J/C pair before Gate2 or target re-ingress."""

        return replace(
            self,
            manifest=self.manifest.bind_contract(
                j,
                c,
                source_to_opaque=self.source_to_opaque,
            ),
        )

    def with_lifecycle_records(
        self,
        records: Iterable[GeneratedSourceLifecycleRecord | Mapping[str, Any]],
    ) -> "PreparedTritonWorkload":
        return replace(
            self,
            manifest=self.manifest.with_lifecycle_records(records),
        )

    def reingress_source(
        self,
        *,
        source: str,
        source_symbol: str,
        source_location: str,
        search_roots: Iterable[str | Path] = (),
        j: Any | None = None,
        c: Any | None = None,
    ) -> "PreparedTritonWorkload":
        return reingress_triton_workload(
            self,
            source=source,
            source_symbol=source_symbol,
            source_location=source_location,
            search_roots=search_roots,
            j=j,
            c=c,
        )


def _pairs(
    values: Iterable[tuple[str, Any]],
    *,
    label: str,
) -> tuple[tuple[str, Any], ...]:
    result = tuple(values)
    names = [name for name, _value in result]
    if len(names) != len(set(names)) or any(
        not isinstance(name, str) or not name.isidentifier() for name in names
    ):
        raise ValueError(f"{label} names must be unique Python identifiers")
    return result


@dataclass(frozen=True)
class TritonWorkloadAdapter:
    case_id: str
    buffers: tuple[BufferBinding, ...]
    scalars: tuple[ScalarBinding, ...]
    semantic_constexprs: tuple[tuple[str, Any], ...]
    c_values: tuple[tuple[str, Any], ...]
    grid: tuple[int, ...]
    grid_contract: tuple[str, ...]
    seed: int
    compiler_options: tuple[tuple[str, Any], ...] = ()
    input_profile: str = "capture-seed-v1"
    none_arguments: tuple[NoneBinding, ...] = ()
    vararg_arity: int | None = None
    grid_shape_coordinates: tuple[str | None, ...] = ()
    semantic_constexpr_shape_coordinates: tuple[tuple[str, str], ...] = ()
    protocol: str = TRITON_WORKLOAD_ADAPTER_PROTOCOL

    def __post_init__(self) -> None:
        if not self.case_id.startswith(
            ("high_performance:", "scrc_canary:", "tilelang:")
        ):
            raise ValueError(
                "adapter case_id must reference the active registry or an accepted "
                "external Source namespace"
            )
        buffers = tuple(
            item if isinstance(item, BufferBinding) else BufferBinding(*item)
            for item in self.buffers
        )
        scalars = tuple(
            item if isinstance(item, ScalarBinding) else ScalarBinding(*item)
            for item in self.scalars
        )
        none_arguments = tuple(
            item if isinstance(item, NoneBinding) else NoneBinding(*item)
            for item in self.none_arguments
        )
        object.__setattr__(self, "buffers", buffers)
        object.__setattr__(self, "scalars", scalars)
        object.__setattr__(self, "none_arguments", none_arguments)
        semantics = _pairs(
            self.semantic_constexprs, label="semantic constexpr"
        )
        c_values = _pairs(self.c_values, label="C")
        compiler_options = _pairs(
            self.compiler_options, label="compiler option"
        )
        object.__setattr__(self, "semantic_constexprs", semantics)
        object.__setattr__(self, "c_values", c_values)
        object.__setattr__(self, "compiler_options", compiler_options)
        names = [item.formal for item in buffers] + [
            item.formal for item in scalars
        ] + [item.formal for item in none_arguments] + [
            name for name, _ in semantics
        ] + [name for name, _ in c_values] + [
            name for name, _ in compiler_options
        ]
        if len(names) != len(set(names)):
            raise ValueError("adapter formals must be globally unique")
        grid = tuple(int(dim) for dim in self.grid)
        if not 1 <= len(grid) <= 3 or any(dim <= 0 for dim in grid):
            raise ValueError("adapter grid must contain positive dimensions")
        object.__setattr__(self, "grid", grid)
        contracts = tuple(self.grid_contract)
        if len(contracts) != len(grid) or any(not item for item in contracts):
            raise ValueError("grid contract must describe every launch dimension")
        object.__setattr__(self, "grid_contract", contracts)
        grid_shape_coordinates = tuple(self.grid_shape_coordinates)
        if grid_shape_coordinates and len(grid_shape_coordinates) != len(grid):
            raise ValueError(
                "grid shape-coordinate bindings must describe every launch dimension"
            )
        if any(
            item is not None
            and (
                not isinstance(item, str)
                or not item.startswith("h")
                or not item[1:].isdigit()
            )
            for item in grid_shape_coordinates
        ):
            raise ValueError(
                "grid shape-coordinate bindings must use h* handles or null"
            )
        object.__setattr__(
            self, "grid_shape_coordinates", grid_shape_coordinates
        )
        shape_coordinates: dict[str, int] = {}
        coordinate_order: list[str] = []
        for buffer in buffers:
            if buffer.shape_contract is None:
                continue
            for contract, extent in zip(
                buffer.shape_contract, buffer.shape, strict=True
            ):
                if not isinstance(contract, str):
                    continue
                if contract not in shape_coordinates:
                    shape_coordinates[contract] = extent
                    coordinate_order.append(contract)
                elif shape_coordinates[contract] != extent:
                    raise ValueError(
                        "adapter shape coordinate has inconsistent valuations"
                    )
        if coordinate_order != [f"h{index}" for index in range(len(coordinate_order))]:
            raise ValueError(
                "adapter shape coordinates must use gap-free first-declaration order"
            )
        for axis, coordinate in enumerate(grid_shape_coordinates):
            if coordinate is None:
                continue
            if coordinate not in shape_coordinates:
                raise ValueError(
                    "grid shape-coordinate binding names an undeclared coordinate"
                )
            if grid[axis] != shape_coordinates[coordinate]:
                raise ValueError(
                    "grid extent differs from its explicit shape-coordinate valuation"
                )
        for scalar in scalars:
            coordinate = scalar.shape_coordinate
            if coordinate is None:
                continue
            if coordinate not in shape_coordinates:
                raise ValueError(
                    "scalar shape-coordinate binding names an undeclared coordinate"
                )
            if scalar.value != shape_coordinates[coordinate]:
                raise ValueError(
                    "scalar and tensor shape-coordinate valuations differ"
                )
        semantic_constexpr_shape_coordinates = _pairs(
            self.semantic_constexpr_shape_coordinates,
            label="semantic constexpr shape-coordinate",
        )
        semantic_values = dict(semantics)
        for name, coordinate in semantic_constexpr_shape_coordinates:
            if name not in semantic_values:
                raise ValueError(
                    "semantic constexpr shape-coordinate binding requires a semantic constexpr"
                )
            if not isinstance(coordinate, str) or coordinate not in shape_coordinates:
                raise ValueError(
                    "semantic constexpr shape-coordinate binding names an undeclared coordinate"
                )
            value = semantic_values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != shape_coordinates[coordinate]
            ):
                raise ValueError(
                    "semantic constexpr and tensor shape-coordinate valuations differ"
                )
        object.__setattr__(
            self,
            "semantic_constexpr_shape_coordinates",
            semantic_constexpr_shape_coordinates,
        )
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("adapter seed must be a non-negative integer")
        if not self.input_profile or any(
            character.isspace() for character in self.input_profile
        ):
            raise ValueError("adapter input profile must be a non-empty token")
        if self.vararg_arity is not None and (
            isinstance(self.vararg_arity, bool)
            or not isinstance(self.vararg_arity, int)
            or not 1 <= self.vararg_arity <= 64
        ):
            raise ValueError("adapter vararg arity must be in [1, 64]")
        output_keys = [
            item.output_key
            for item in buffers
            if item.role is not BufferRole.INPUT
        ]
        if not output_keys or len(output_keys) != len(set(output_keys)):
            raise ValueError("adapter outputs must have unique observable keys")

    def compiler_view(
        self,
        parameter_order: tuple[str, ...],
        constexpr_parameters: tuple[str, ...],
    ) -> SCRCCompilerView:
        runtime = {
            item.formal: SCRCCompilerArgument(
                item.formal,
                "tensor",
                item.dtype,
                item.shape,
                mode=item.role.value,
                shape_contract=item.shape_contract,
                strides=item.strides,
            )
            for item in self.buffers
        }
        runtime.update(
            {
                item.formal: SCRCCompilerArgument(
                    item.formal,
                    "scalar",
                    item.dtype,
                    value=item.value,
                    mode="value",
                    shape_coordinate=item.shape_coordinate,
                )
                for item in self.scalars
            }
        )
        runtime.update(
            {
                item.formal: SCRCCompilerArgument(
                    item.formal,
                    "none",
                    item.dtype,
                    mode="value",
                )
                for item in self.none_arguments
            }
        )
        expected_runtime = tuple(
            name for name in parameter_order if name not in constexpr_parameters
        )
        if set(expected_runtime) != set(runtime):
            missing = sorted(set(expected_runtime) - set(runtime))
            extra = sorted(set(runtime) - set(expected_runtime))
            raise ValueError(
                f"adapter runtime ABI differs from source; missing={missing}, extra={extra}"
            )
        semantics = dict(self.semantic_constexprs)
        c_values = dict(self.c_values)
        compiler_options = dict(self.compiler_options)
        if set(constexpr_parameters) != set(semantics) | set(c_values):
            raise ValueError("adapter constexpr ABI differs from source")
        # The source declaration order is the only stable order available for
        # C-source constexprs.  The adapter fields are independently
        # constructible, so relying on their mapping insertion order leaves a
        # latent split between ``schedule_values`` and the source slice when a
        # caller supplies C values in a different order.  Normalize the map
        # before it reaches every manifest/BindSource consumer; compiler
        # options remain a distinct, ordered suffix.
        ordered_c_names = tuple(
            name for name in constexpr_parameters if name in c_values
        )
        if set(ordered_c_names) != set(c_values):
            raise ValueError("adapter C values must be source constexpr parameters")
        ordered_c_values = {name: c_values[name] for name in ordered_c_names}
        ordered_compiler_options = dict(compiler_options)
        return SCRCCompilerView(
            schedule_values={**ordered_c_values, **ordered_compiler_options},
            bound_constexpr_values=semantics,
            bound_constexpr_shape_coordinates=dict(
                self.semantic_constexpr_shape_coordinates
            ),
            compiler_slots=tuple(compiler_options),
            arguments=tuple(runtime[name] for name in expected_runtime),
            grid=self.grid,
            grid_contract=self.grid_contract,
            grid_shape_coordinates=self.grid_shape_coordinates,
            parameter_order=parameter_order,
            constexpr_parameters=constexpr_parameters,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "case_id": self.case_id,
            "buffers": [
                {
                    "formal": item.formal,
                    "role": item.role.value,
                    "dtype": item.dtype,
                    "shape": list(item.shape),
                    "shape_contract": (
                        list(item.shape_contract)
                        if item.shape_contract is not None
                        else None
                    ),
                    "strides": list(item.strides) if item.strides is not None else None,
                    "output_key": item.output_key,
                    "initializer": (
                        item.initializer.value
                        if item.initializer is not None
                        else None
                    ),
                }
                for item in self.buffers
            ],
            "scalars": [
                {
                    "formal": item.formal,
                    "dtype": item.dtype,
                    "value": item.value,
                    "shape_coordinate": item.shape_coordinate,
                }
                for item in self.scalars
            ],
            "none_arguments": [
                {"formal": item.formal, "dtype": item.dtype}
                for item in self.none_arguments
            ],
            "semantic_constexprs": dict(self.semantic_constexprs),
            "semantic_constexpr_shape_coordinates": dict(
                self.semantic_constexpr_shape_coordinates
            ),
            "c_values": dict(self.c_values),
            "compiler_options": dict(self.compiler_options),
            "grid": list(self.grid),
            "grid_contract": list(self.grid_contract),
            "grid_shape_coordinates": list(self.grid_shape_coordinates),
            "seed": self.seed,
            "input_profile": self.input_profile,
            "vararg_arity": self.vararg_arity,
        }


def _runtime_argument(argument: SCRCCompilerArgument) -> RuntimeArgument:
    aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "index": "int64",
        "u8": "uint8",
        "u16": "uint16",
        "u32": "uint32",
    }
    dtype = aliases.get(argument.dtype, argument.dtype)
    if argument.kind == "tensor":
        return RuntimeArgument.tensor(
            argument.shape,
            dtype,
            strides=argument.strides,
        )
    if argument.kind == "scalar":
        assert argument.value is not None
        return RuntimeArgument.scalar(argument.value, dtype)
    if argument.kind == "none":
        return RuntimeArgument.none(dtype)
    raise ValueError("real workload adapter supports tensor/scalar/none runtime ABI")


def _canonicalize_gate_parameter_order(
    source: str,
    *,
    function: str,
    ordered_names: tuple[str, ...],
) -> str:
    """Make the anonymous Gate entry a complete positional ABI.

    The captured Source may use positional-only, keyword-only, or defaulted
    Python parameters even though the captured launch supplies every concrete
    argument.  Gate compilation needs a stable anonymous positional ABI, so
    this boundary preserves each parameter name/body use while dropping those
    Python call-surface distinctions and ordering runtime parameters before
    constexprs.  Genuine variadic call surfaces are handled by the existing
    Triton vararg expansion path; an unexpanded ``*args``/``**kwargs`` stays
    unsupported rather than guessed.
    """

    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function
    ]
    if len(matches) != 1:
        raise ValueError("anonymous Gate entry must resolve exactly once")
    target = matches[0]
    if target.args.vararg is not None or target.args.kwarg is not None:
        raise ValueError(
            "captured Gate parameter canonicalization does not support "
            "unexpanded variadic ABI"
        )
    original_arguments = tuple(
        [
            *target.args.posonlyargs,
            *target.args.args,
            *target.args.kwonlyargs,
        ]
    )
    current = tuple(argument.arg for argument in original_arguments)
    if len(current) != len(set(current)) or set(current) != set(ordered_names):
        raise ValueError("anonymous Gate parameter order differs from the captured ABI")
    if (
        current == ordered_names
        and not target.args.posonlyargs
        and not target.args.kwonlyargs
        and not target.args.defaults
        and not target.args.kw_defaults
    ):
        return source
    by_name = {argument.arg: argument for argument in original_arguments}
    target.args.posonlyargs = []
    target.args.args = [by_name[name] for name in ordered_names]
    target.args.kwonlyargs = []
    target.args.defaults = []
    target.args.kw_defaults = []
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def build_generated_source_manifest(
    *,
    source: str,
    source_slice: Any,
    source_location: str,
    view: SCRCCompilerView,
    source_arguments: Iterable[GeneratedTargetSourceArgument],
    value_dimension_relations: Iterable[GeneratedSourceValueDimensionRelation | Mapping[str, Any]] | None = None,
    dimension_relations: Iterable[GeneratedSourceDimensionRelation | Mapping[str, Any]] | None = None,
    shape_coordinate_bindings: Iterable[GeneratedSourceShapeCoordinateBinding | Mapping[str, Any]] | None = None,
    layout_relations: Iterable[GeneratedSourceLayoutRelation | Mapping[str, Any]] | None = None,
    tensor_map_relations: Iterable[GeneratedSourceTensorMapRelation | Mapping[str, Any]] | None = None,
    output_bindings: Iterable[GeneratedSourceOutputBinding | Mapping[str, Any]] | None = None,
) -> GeneratedTargetSourceManifest:
    """Build one frontend-neutral generated-source admission manifest.

    ``source_slice`` is intentionally duck-typed: the active Triton slice and
    the native Dataset projection both expose ``code``, ``function``,
    ``parameters`` and ``constexpr_parameters``.  The caller supplies the
    physical argument records so CUDA opaque ABI values can be represented
    without pretending that a CUtensorMap is a logical tensor or a Python
    scalar.
    """

    source_order = tuple(source_slice.parameters)
    constexpr_order = tuple(source_slice.constexpr_parameters)
    supplied_arguments = tuple(source_arguments)
    if tuple(item.name for item in supplied_arguments) != source_order:
        raise ValueError(
            "generated-source argument records must exactly follow the source ABI"
        )
    semantic_values = dict(view.bound_constexpr_values)
    schedule_values = dict(view.schedule_values)
    compiler_slots = tuple(view.compiler_slots)
    compiler_set = set(compiler_slots)
    runtime_order = tuple(name for name in source_order if name not in constexpr_order)
    semantic_order = tuple(
        name for name in constexpr_order
        if name in semantic_values
    )
    c_source_order = tuple(
        name for name in constexpr_order
        if name in schedule_values and name not in compiler_set
    )
    if set(semantic_order) | set(c_source_order) != set(constexpr_order):
        raise ValueError(
            "generated-source manifest cannot classify every source constexpr "
            "parameter as semantic or C-owned"
        )
    if set(schedule_values) & compiler_set != compiler_set:
        raise ValueError(
            "generated-source manifest compiler slots are absent from C values"
        )
    expected_schedule_order = (*c_source_order, *compiler_slots)
    if set(schedule_values) != set(expected_schedule_order):
        raise ValueError(
            "generated-source schedule values must cover exactly the source C "
            "constexprs and compiler slots"
        )
    if tuple(schedule_values) != expected_schedule_order:
        raise ValueError(
            "generated-source schedule declaration order must be source C "
            "constexprs followed by compiler slots"
        )
    # Keep this as the one authoritative order copied into C declarations,
    # source manifests, and BindSource requests.
    c_slot_order = tuple(schedule_values)
    c_slot_values = tuple((name, schedule_values[name]) for name in c_slot_order)
    arguments = supplied_arguments
    runtime_scalar_values = tuple(
        (item.name, item.value)
        for item in arguments
        if item.category == "runtime_scalar" and item.value is not None
    )
    semantic_constexpr_shape_coordinates = {
        item.name: item.shape_coordinate
        for item in arguments
        if item.category == "semantic_constexpr"
        and item.shape_coordinate is not None
    }
    for name in source_order:
        argument = next(item for item in arguments if item.name == name)
        if name in runtime_order:
            if argument.category not in {"buffer", "runtime_scalar", "none", "opaque"}:
                raise ValueError(
                    f"generated-source manifest runtime parameter {name!r} has an invalid category"
                )
        elif name in semantic_order:
            if argument.category != "semantic_constexpr":
                raise ValueError(
                    f"generated-source manifest semantic constexpr {name!r} is not marked semantic_constexpr"
                )
        elif name in c_source_order:
            if argument.category != "c_slot":
                raise ValueError(
                    f"generated-source manifest C constexpr {name!r} is not marked c_slot"
                )
        else:
            raise ValueError(
                f"generated-source manifest cannot classify source parameter {name!r}"
            )

    derived_value_relations, derived_shape_bindings, derived_layout_relations, derived_output_bindings = (
        _derived_source_relations(arguments)
    )
    normalized_value_relations = (
        derived_value_relations
        if value_dimension_relations is None
        else _source_relation_tuple(
            value_dimension_relations,
            GeneratedSourceValueDimensionRelation,
            "generated-source value_dimension_relations",
        )
    )
    normalized_dimension_relations = (
        ()
        if dimension_relations is None
        else _source_relation_tuple(
            dimension_relations,
            GeneratedSourceDimensionRelation,
            "generated-source dimension_relations",
        )
    )
    normalized_shape_bindings = (
        derived_shape_bindings
        if shape_coordinate_bindings is None
        else _source_relation_tuple(
            shape_coordinate_bindings,
            GeneratedSourceShapeCoordinateBinding,
            "generated-source shape_coordinate_bindings",
        )
    )
    normalized_layout_relations = (
        derived_layout_relations
        if layout_relations is None
        else _source_relation_tuple(
            layout_relations,
            GeneratedSourceLayoutRelation,
            "generated-source layout_relations",
        )
    )
    normalized_tensor_map_relations = (
        ()
        if tensor_map_relations is None
        else _source_relation_tuple(
            tensor_map_relations,
            GeneratedSourceTensorMapRelation,
            "generated-source tensor_map_relations",
        )
    )
    normalized_output_bindings = (
        derived_output_bindings
        if output_bindings is None
        else _source_relation_tuple(
            output_bindings,
            GeneratedSourceOutputBinding,
            "generated-source output_bindings",
        )
    )

    coordinate_values = tuple(view.shape_coordinates.items())
    grid_shape_coordinates = tuple(view.grid_shape_coordinates) or tuple(
        None for _ in view.grid
    )
    manifest = GeneratedTargetSourceManifest(
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source_slice_sha256=hashlib.sha256(
            source_slice.code.encode("utf-8")
        ).hexdigest(),
        source_location=source_location,
        device_entry_symbol=source_slice.function,
        source_parameter_order=source_order,
        runtime_parameter_order=runtime_order,
        constexpr_parameter_order=constexpr_order,
        arguments=arguments,
        semantic_constexpr_parameters=semantic_order,
        c_source_constexpr_parameters=c_source_order,
        compiler_slot_order=compiler_slots,
        c_slot_order=c_slot_order,
        runtime_scalar_values=tuple(runtime_scalar_values),
        semantic_constexpr_values=tuple(
            (name, semantic_values[name]) for name in semantic_order
        ),
        c_slot_values=c_slot_values,
        shape_coordinate_values=coordinate_values,
        grid=tuple(view.grid),
        grid_contract=tuple(view.grid_contract),
        grid_shape_coordinates=grid_shape_coordinates,
        value_dimension_relations=normalized_value_relations,
        dimension_relations=normalized_dimension_relations,
        shape_coordinate_bindings=normalized_shape_bindings,
        layout_relations=normalized_layout_relations,
        tensor_map_relations=normalized_tensor_map_relations,
        output_bindings=normalized_output_bindings,
    )
    manifest.validate_source(source, source_slice, source_location)
    return manifest


def _build_generated_source_manifest(
    adapter: TritonWorkloadAdapter,
    *,
    source: str,
    source_slice: TritonSourceSlice,
    source_location: str,
    view: SCRCCompilerView,
) -> GeneratedTargetSourceManifest:
    """Build the strict manifest for the ordinary Triton Dataset entrance."""

    buffers = {item.formal: item for item in adapter.buffers}
    scalars = {item.formal: item for item in adapter.scalars}
    none_arguments = {item.formal: item for item in adapter.none_arguments}
    semantic_values = dict(view.bound_constexpr_values)
    schedule_values = dict(view.schedule_values)
    compiler_slots = set(view.compiler_slots)
    semantic_coordinates = dict(adapter.semantic_constexpr_shape_coordinates)
    arguments: list[GeneratedTargetSourceArgument] = []
    for name in source_slice.parameters:
        if name in buffers:
            item = buffers[name]
            arguments.append(
                GeneratedTargetSourceArgument(
                    name=name,
                    category="buffer",
                    role=item.role.value,
                    dtype=item.dtype,
                    shape=item.shape,
                    shape_contract=item.shape_contract,
                    strides=item.strides,
                    output_key=item.output_key,
                )
            )
        elif name in scalars:
            item = scalars[name]
            arguments.append(
                GeneratedTargetSourceArgument(
                    name=name,
                    category="runtime_scalar",
                    role="value",
                    dtype=item.dtype,
                    value=item.value,
                    shape_coordinate=item.shape_coordinate,
                )
            )
        elif name in none_arguments:
            arguments.append(
                GeneratedTargetSourceArgument(
                    name=name,
                    category="none",
                    role="value",
                    dtype="none",
                )
            )
        elif name in semantic_values:
            try:
                value_type = view.bound_constexpr_value_types[name]
            except KeyError as exc:
                raise ValueError(
                    "generated-source manifest lacks a canonical Source dtype "
                    f"for semantic constexpr {name!r}"
                ) from exc
            arguments.append(
                GeneratedTargetSourceArgument(
                    name=name,
                    category="semantic_constexpr",
                    role="semantic_constexpr",
                    dtype=value_type,
                    value=semantic_values[name],
                    shape_coordinate=semantic_coordinates.get(name),
                )
            )
        elif name in schedule_values and name not in compiler_slots:
            try:
                value_type = view.schedule_value_types[name]
            except KeyError as exc:
                raise ValueError(
                    "generated-source manifest lacks a canonical Source-C dtype "
                    f"for constexpr {name!r}"
                ) from exc
            arguments.append(
                GeneratedTargetSourceArgument(
                    name=name,
                    category="c_slot",
                    role="constexpr",
                    dtype=value_type,
                    value=schedule_values[name],
                )
            )
        else:
            raise ValueError(
                f"generated-source manifest cannot classify source parameter {name!r}"
            )
    return build_generated_source_manifest(
        source=source,
        source_slice=source_slice,
        source_location=source_location,
        view=view,
        source_arguments=tuple(arguments),
    )


def _prepare_triton_workload_source(
    adapter: TritonWorkloadAdapter,
    *,
    entry: SCRCWorkloadEntry,
    source: str,
    source_location: str,
    search_roots: tuple[Path, ...],
) -> PreparedTritonWorkload:
    source_slice = build_triton_source_slice(
        source, source_symbol=entry.symbol
    )
    source_slice = inline_imported_triton_helpers(
        source_slice, search_roots=search_roots
    )
    requires_varargs = requires_triton_vararg_expansion(source_slice)
    if requires_varargs != (adapter.vararg_arity is not None):
        raise ValueError(
            "adapter vararg arity must be present exactly when Source declares "
            "VAR_ARGS_ARRAY"
        )
    if adapter.vararg_arity is not None:
        source_slice = expand_triton_varargs(
            source_slice, arity=adapter.vararg_arity
        )
    view = adapter.compiler_view(
        source_slice.parameters, source_slice.constexpr_parameters
    )
    source_slice.validate_abi(
        parameter_order=view.parameter_order,
        constexpr_parameters=view.constexpr_parameters,
    )
    anonymous, _binding, mapping = anonymize_source(
        source_slice.code,
        source_symbol=entry.symbol,
        bound_constexpr_parameters=tuple(
            name for name, _value in adapter.semantic_constexprs
        ),
    )
    runtime_parameters = tuple(
        name
        for name in source_slice.parameters
        if name not in source_slice.constexpr_parameters
    )
    anonymous = _canonicalize_gate_parameter_order(
        anonymous,
        function="scrc_k0",
        ordered_names=tuple(
            mapping[name]
            for name in (*runtime_parameters, *source_slice.constexpr_parameters)
        ),
    )
    used_opaque_names = set(mapping.values())
    next_c = 0
    for name, _value in adapter.compiler_options:
        while f"c{next_c}" in used_opaque_names:
            next_c += 1
        mapping[name] = f"c{next_c}"
        used_opaque_names.add(mapping[name])
    # ``anonymize_source`` also records every source identifier it rewrites,
    # including local temporaries, helper names, and module declarations.  The
    # public Triton ABI mapping is narrower: it is only the selected entry's
    # physical parameters plus the compiler-only C slots.  Keeping the full
    # mapping for source rewriting while exporting this ABI projection avoids
    # confusing lexical provenance with a bindable argument.
    abi_names = (*source_slice.parameters, *view.compiler_slots)
    abi_mapping = {name: mapping[name] for name in abi_names}
    arguments = {
        mapping[item.name]: _runtime_argument(item)
        for item in view.arguments
    }
    all_constexprs = {
        **dict(adapter.semantic_constexprs),
        **dict(adapter.c_values),
    }
    schedule = {
        mapping[name]: all_constexprs[name]
        for name in source_slice.constexpr_parameters
    }
    schedule.update(
        (mapping[name], value) for name, value in adapter.compiler_options
    )
    c0 = {
        mapping[name]: value for name, value in adapter.c_values
    }
    c0.update(
        (mapping[name], value) for name, value in adapter.compiler_options
    )
    gate_source = KernelSource.from_code(
        anonymous, "scrc_k0", trusted=True
    )
    request = SCRCRunRequest(
        SourceProgram(source_slice.code, gate_source),
        HarnessBindingInstance.create(
            c0,
            gate1_case=CompileCase(arguments, adapter.grid, schedule),
            slot_order=tuple(c0),
            gate2_context={
                "test_mode": "finite",
                "seed": adapter.seed,
                "adapter_protocol": adapter.protocol,
                "input_profile": adapter.input_profile,
            },
        ),
        "triton",
    )
    source_manifest = _build_generated_source_manifest(
        adapter,
        source=source,
        source_slice=source_slice,
        source_location=source_location,
        view=view,
    )
    return PreparedTritonWorkload(
        adapter,
        entry,
        source_location,
        source_slice,
        view,
        request,
        abi_mapping,
        source_manifest,
    )


def prepare_triton_workload(
    adapter: TritonWorkloadAdapter,
    *,
    dataset_root: str | Path | None = None,
) -> PreparedTritonWorkload:
    manifest = build_workload_manifest()
    entry = manifest.by_id.get(adapter.case_id)
    if entry is None:
        raise ValueError("adapter case is absent from the active 28-kernel registry")
    resolution = entry.resolve_source(dataset_root=dataset_root)
    if not resolution.resolved or resolution.path is None:
        raise ValueError(
            f"real source is unavailable for {entry.case_id}: {resolution.diagnostic}"
        )
    source = resolution.path.read_text(encoding="utf-8")
    module_root = resolution.path
    for _component in entry.source_module.split("."):
        module_root = module_root.parent
    return _prepare_triton_workload_source(
        adapter,
        entry=entry,
        source=source,
        source_location=resolution.location,
        search_roots=(module_root,),
    )


def prepare_triton_source_workload(
    adapter: TritonWorkloadAdapter,
    source_path: str | Path,
    *,
    source_symbol: str,
    source_location: str | None = None,
) -> PreparedTritonWorkload:
    """Prepare one exact certified Source outside the active registry."""

    if not adapter.case_id.startswith(("scrc_canary:", "tilelang:")):
        raise ValueError(
            "explicit Source preparation is reserved for certified canaries "
            "and accepted TileLang workload adapters"
        )
    path = Path(source_path).resolve()
    source = path.read_text(encoding="utf-8")
    entry = SCRCWorkloadEntry(
        ordinal=0,
        case_id=adapter.case_id,
        suite=(
            "tilelang"
            if adapter.case_id.startswith("tilelang:")
            else "scrc_canary"
        ),
        symbol=source_symbol,
        mechanisms=("certified_source_reuse",),
        source_module="intentir.certified_canary",
        source_alignment="exact archived accepted Source",
        lowering_status="certified",
        representative_tier="certified_canary",
        adapter_readiness="declared",
    )
    return _prepare_triton_workload_source(
        adapter,
        entry=entry,
        source=source,
        source_location=(source_location or str(path)),
        search_roots=(path.parent,),
    )


def prepare_captured_triton_workload(
    adapter: TritonWorkloadAdapter,
    *,
    source: str,
    source_symbol: str,
    source_location: str,
    search_roots: Iterable[str | Path] = (),
    suite: str = "flaggems",
    mechanisms: Iterable[str] = (),
) -> PreparedTritonWorkload:
    """Prepare one captured launch without adding it to the static corpus registry.

    The caller supplies the exact source projection and Harness bindings.  This
    boundary performs no operator, role, shape, or J inference; it only routes a
    captured real launch through the same anonymization and SCRC lifecycle as the
    registered corpus.
    """

    if not isinstance(source, str) or not source.strip():
        raise ValueError("captured Triton source must be non-empty")
    if not source_symbol.isidentifier():
        raise ValueError("captured Triton source symbol must be a Python identifier")
    if not source_location:
        raise ValueError("captured Triton source location must be non-empty")
    entry = SCRCWorkloadEntry(
        ordinal=0,
        case_id=adapter.case_id,
        suite=suite,
        symbol=source_symbol,
        mechanisms=tuple(mechanisms),
        source_module="captured_triton",
        source_alignment="captured_launch",
        lowering_status="captured",
        representative_tier="zero_shot",
        adapter_readiness="captured",
    )
    roots = tuple(Path(root).expanduser().resolve() for root in search_roots)
    return _prepare_triton_workload_source(
        adapter,
        entry=entry,
        source=source,
        source_location=source_location,
        search_roots=roots,
    )


def reingress_triton_workload(
    prepared: PreparedTritonWorkload,
    *,
    source: str,
    source_symbol: str,
    source_location: str,
    search_roots: Iterable[str | Path] = (),
    j: Any | None = None,
    c: Any | None = None,
) -> PreparedTritonWorkload:
    """Re-enter one explicit target-source slice through the same ABI checks.

    The caller supplies the complete source bytes and the device entry symbol;
    no filename, kernel-name convention, or algorithm template is consulted.
    A previously bound J/C pair must be supplied again so the new source can be
    checked against the exact accepted contract rather than merely sharing a
    case identifier.
    """

    if not isinstance(prepared, PreparedTritonWorkload):
        raise TypeError("re-ingress requires a prepared Triton workload")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("re-ingress source must be non-empty")
    if not isinstance(source_symbol, str) or not source_symbol.isidentifier():
        raise ValueError("re-ingress source_symbol must be an identifier")
    if not isinstance(source_location, str) or not source_location.strip():
        raise ValueError("re-ingress source_location must be non-empty")
    if len(prepared.manifest.lifecycle_records) != 3:
        raise ValueError(
            "Triton source re-ingress requires the three real execution lifecycle records"
        )
    if prepared.manifest.bound and (j is None or c is None):
        raise ValueError(
            "re-ingress of a bound workload requires the exact accepted J and C"
        )
    entry = replace(prepared.entry, symbol=source_symbol)
    roots = tuple(Path(root).expanduser().resolve() for root in search_roots)
    result = _prepare_triton_workload_source(
        prepared.adapter,
        entry=entry,
        source=source,
        source_location=source_location,
        search_roots=roots,
    )
    if prepared.manifest.bound:
        assert j is not None and c is not None
        result = result.with_contract(j, c)
        if (
            result.manifest.j_digest != prepared.manifest.j_digest
            or result.manifest.e_c_interface_digest
            != prepared.manifest.e_c_interface_digest
            or result.manifest.contract_ir_digest
            != prepared.manifest.contract_ir_digest
        ):
            raise ValueError(
                "re-ingress source changed the bound J/E_C/contract identity"
            )
    result_manifest = prepared.manifest.reingress_source_slice(
        source=source,
        source_slice=result.source_slice,
        source_location=source_location,
        compiler_view=result.compiler_view,
        source_to_opaque=result.source_to_opaque,
        request=result.request,
        consumer="triton-source-reingress",
    )
    return replace(result, manifest=result_manifest)


def reingress_prepared_workload(
    prepared: Any,
    *,
    source: str,
    source_slice: Any,
    source_location: str,
    compiler_view: SCRCCompilerView,
    source_to_opaque: Mapping[str, str],
    request: SCRCRunRequest,
    consumer: str,
) -> Any:
    """Re-ingress a prepared Triton/CUDA/TileLang projection uniformly.

    Frontend-specific slicers and source constructors remain outside this
    function.  Once those explicit objects exist, this boundary checks their
    source bytes, selected device entry, source ABI, anonymous mapping, C-slot
    request, and prior three execution records in one place.
    """

    manifest = getattr(prepared, "manifest", None)
    if not isinstance(manifest, GeneratedTargetSourceManifest):
        raise TypeError("prepared workload has no generated-source manifest")
    if not isinstance(request, SCRCRunRequest):
        raise TypeError("re-ingress requires a complete SCRC request")
    if request.frontend != getattr(prepared.request, "frontend", None):
        raise ValueError("re-ingress request frontend differs from prepared workload")
    if request.source.text != source_slice.code:
        raise ValueError(
            "re-ingress request lift source differs from the selected source slice"
        )
    if request.source.lift_source_symbol != source_slice.function:
        raise ValueError(
            "re-ingress request lacks the selected source device_entry_symbol"
        )
    next_manifest = manifest.reingress_source_slice(
        source=source,
        source_slice=source_slice,
        source_location=source_location,
        compiler_view=compiler_view,
        source_to_opaque=source_to_opaque,
        request=request,
        consumer=consumer,
    )
    updated = replace(
        prepared,
        source_location=source_location,
        source_slice=source_slice,
        compiler_view=compiler_view,
        request=request,
        source_to_opaque=dict(source_to_opaque),
        manifest=next_manifest,
    )
    return updated


def _buffer(
    formal: str,
    role: BufferRole,
    dtype: str,
    shape: tuple[int, ...],
    output_key: str | None = None,
    initializer: BufferInitializer | None = None,
    shape_contract: tuple[int | str, ...] | None = None,
    strides: tuple[int, ...] | None = None,
) -> BufferBinding:
    return BufferBinding(
        formal=formal,
        role=role,
        dtype=dtype,
        shape=shape,
        output_key=output_key,
        initializer=initializer,
        shape_contract=shape_contract,
        strides=strides,
    )


def _scalar(
    formal: str,
    dtype: str,
    value: Any,
    *,
    shape_coordinate: str | None = None,
) -> ScalarBinding:
    return ScalarBinding(formal, dtype, value, shape_coordinate)


TRITON_WORKLOAD_ADAPTERS: dict[str, TritonWorkloadAdapter] = {
    "high_performance:liger_dyt": TritonWorkloadAdapter(
        "high_performance:liger_dyt",
        buffers=(
            _buffer(
                "X", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "Y", BufferRole.OUTPUT, "f32", (2, 127), "y",
                shape_contract=("h0", "h1"),
            ),
            _buffer("Alpha", BufferRole.INPUT, "f32", (1,)),
            _buffer(
                "Gamma", BufferRole.INPUT, "f32", (127,),
                shape_contract=("h1",),
            ),
            _buffer(
                "Beta", BufferRole.INPUT, "f32", (127,),
                shape_contract=("h1",),
            ),
        ),
        scalars=(),
        semantic_constexprs=(("HAVE_BETA", True), ("N", 127)),
        semantic_constexpr_shape_coordinates=(("N", "h1"),),
        c_values=(("BLOCK_N", 128),),
        grid=(1, 2),
        grid_contract=("ceil_div(N,BLOCK_N)", "rows"),
        grid_shape_coordinates=(None, "h0"),
        seed=101,
    ),
    "high_performance:liger_swiglu": TritonWorkloadAdapter(
        "high_performance:liger_swiglu",
        buffers=(
            _buffer(
                "a_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "b_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "c_ptr", BufferRole.OUTPUT, "f32", (2, 127), "c",
                shape_contract=("h0", "h1"),
            ),
        ),
        scalars=(
            _scalar("stride", "i64", 127),
        ),
        semantic_constexprs=(("n_cols", 127),),
        semantic_constexpr_shape_coordinates=(("n_cols", "h1"),),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=103,
    ),
    (
        "high_performance:flash_attn.flash_attn_triton."
        "_bwd_preprocess_do_o_dot"
    ): TritonWorkloadAdapter(
        (
            "high_performance:flash_attn.flash_attn_triton."
            "_bwd_preprocess_do_o_dot"
        ),
        buffers=(
            _buffer(
                "Out", BufferRole.INPUT, "f16", (1, 2, 33, 16),
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "DO", BufferRole.INPUT, "f16", (1, 2, 33, 16),
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "Delta", BufferRole.OUTPUT, "f32", (1, 2, 64), "delta",
                shape_contract=("h0", "h1", "h4"),
            ),
        ),
        scalars=(
            _scalar("stride_ob", "i64", 1056),
            _scalar("stride_oh", "i64", 528),
            _scalar("stride_om", "i64", 16),
            _scalar("stride_dob", "i64", 1056),
            _scalar("stride_doh", "i64", 528),
            _scalar("stride_dom", "i64", 16),
            _scalar("nheads", "i32", 2, shape_coordinate="h1"),
            _scalar("seqlen_q", "i32", 33, shape_coordinate="h2"),
            _scalar(
                "seqlen_q_rounded", "i32", 64, shape_coordinate="h4"
            ),
            _scalar("headdim", "i32", 16, shape_coordinate="h3"),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_M", 32), ("BLOCK_HEADDIM", 16)),
        grid=(2, 2),
        grid_contract=("ceil_div(seqlen_q,BLOCK_M)", "batch*nheads"),
        seed=107,
    ),
    "high_performance:liger_softmax": TritonWorkloadAdapter(
        "high_performance:liger_softmax",
        buffers=(
            _buffer(
                "Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "X_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("n_cols", "i32", 127, shape_coordinate="h1"),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=109,
    ),
    "high_performance:liger_rms_norm": TritonWorkloadAdapter(
        "high_performance:liger_rms_norm",
        buffers=(
            _buffer(
                "Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "X_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "W_ptr", BufferRole.INPUT, "f32", (127,),
                shape_contract=("h1",),
            ),
            _buffer(
                "RSTD_ptr", BufferRole.OUTPUT, "f32", (2,), "rstd",
                shape_contract=("h0",),
            ),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("W_row_stride", "i64", 127),
            _scalar("RSTD_row_stride", "i64", 1),
            _scalar("n_cols", "i32", 127, shape_coordinate="h1"),
            _scalar("eps", "f32", 1.0e-6),
            _scalar("offset", "f32", 0.0),
        ),
        semantic_constexprs=(
            ("casting_mode", 0),
            ("elementwise_affine", True),
        ),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=113,
    ),
    (
        "high_performance:flash_attn.ops.triton.linear.kernel_fwd"
    ): TritonWorkloadAdapter(
        "high_performance:flash_attn.ops.triton.linear.kernel_fwd",
        buffers=(
            _buffer(
                "C", BufferRole.OUTPUT, "f16", (4096, 4096), "c",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "ACT_INPUT", BufferRole.OUTPUT, "f16", (4096, 4096), "act_input",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "A", BufferRole.INPUT, "f16", (4096, 4096),
                shape_contract=("h0", "h2"),
            ),
            _buffer(
                "B", BufferRole.INPUT, "f16", (4096, 4096),
                # The selected B_COLMAJOR specialization indexes the flat
                # buffer as B[n, k] (k + n * stride_bn).  Equal model extents
                # must not hide that logical ABI orientation from Lift.
                shape_contract=("h1", "h2"),
            ),
            _buffer(
                "bias", BufferRole.INPUT, "f16", (4096,),
                shape_contract=("h1",),
            ),
        ),
        scalars=(
            _scalar("M", "i32", 4096, shape_coordinate="h0"),
            _scalar("N", "i32", 4096, shape_coordinate="h1"),
            _scalar("K", "i32", 4096, shape_coordinate="h2"),
            _scalar("CACHE_KEY_M", "i32", 4096),
            _scalar("CACHE_KEY_N", "i32", 4096),
            _scalar("CACHE_KEY_K", "i32", 4096),
            _scalar("stride_cm", "i64", 4096),
            _scalar("stride_am", "i64", 4096),
            _scalar("stride_ak", "i64", 1),
            _scalar("stride_bn", "i64", 4096),
            _scalar("stride_bk", "i64", 1),
        ),
        semantic_constexprs=(
            ("BIAS", True),
            ("SAVE_ACT_INPUT", True),
            ("ACTIVATION", "id"),
        ),
        c_values=(
            ("BLOCK_M", 32),
            ("GROUP_M", 8),
            ("BLOCK_N", 32),
            ("BLOCK_K", 32),
            ("SPLIT_K", 1),
            ("EVEN_K", True),
            ("A_ROWMAJOR", True),
            ("B_COLMAJOR", True),
        ),
        grid=(16384,),
        grid_contract=("ceil_div(M,BLOCK_M)*ceil_div(N,BLOCK_N)",),
        seed=127,
        input_profile="flash-linear-M4096-N4096-K4096-f16-v1",
    ),
    (
        "high_performance:flash_attn.flash_attn_triton._fwd_kernel"
    ): TritonWorkloadAdapter(
        "high_performance:flash_attn.flash_attn_triton._fwd_kernel",
        buffers=(
            _buffer(
                "Q", BufferRole.INPUT, "f16", (4, 8192, 16, 128),
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "K", BufferRole.INPUT, "f16", (4, 8192, 16, 128),
                shape_contract=("h0", "h4", "h2", "h3"),
            ),
            _buffer(
                "V", BufferRole.INPUT, "f16", (4, 8192, 16, 128),
                shape_contract=("h0", "h4", "h2", "h3"),
            ),
            # BIAS_TYPE=none makes this ABI carrier intentionally unread.
            _buffer("Bias", BufferRole.INPUT, "f16", (1,)),
            _buffer(
                "Out", BufferRole.OUTPUT, "f16", (4, 8192, 16, 128), "out",
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "Lse", BufferRole.OUTPUT, "f32", (4, 16, 8192), "lse",
                shape_contract=("h0", "h2", "h5"),
            ),
            # TMP is explicit source-ABI scratch storage. The captured symbol
            # stores and reloads it, so its compiler-visible access is inout.
            _buffer(
                "TMP", BufferRole.INOUT, "f32", (4, 16, 8192), "tmp",
                shape_contract=("h0", "h2", "h5"),
            ),
        ),
        scalars=(
            _scalar("softmax_scale", "f32", 0.08838834764831845),
            _scalar("stride_qb", "i64", 16777216),
            _scalar("stride_qh", "i64", 128),
            _scalar("stride_qm", "i64", 2048),
            _scalar("stride_kb", "i64", 16777216),
            _scalar("stride_kh", "i64", 128),
            _scalar("stride_kn", "i64", 2048),
            _scalar("stride_vb", "i64", 16777216),
            _scalar("stride_vh", "i64", 128),
            _scalar("stride_vn", "i64", 2048),
            _scalar("stride_bb", "i64", 0),
            _scalar("stride_bh", "i64", 0),
            _scalar("stride_bm", "i64", 0),
            _scalar("stride_ob", "i64", 16777216),
            _scalar("stride_oh", "i64", 128),
            _scalar("stride_om", "i64", 2048),
            _scalar("nheads", "i32", 16, shape_coordinate="h2"),
            _scalar("seqlen_q", "i32", 8192, shape_coordinate="h1"),
            _scalar("seqlen_k", "i32", 8192, shape_coordinate="h4"),
            _scalar(
                "seqlen_q_rounded", "i32", 8192, shape_coordinate="h5"
            ),
            _scalar("headdim", "i32", 128, shape_coordinate="h3"),
            _scalar("CACHE_KEY_SEQLEN_Q", "i32", 256),
            _scalar("CACHE_KEY_SEQLEN_K", "i32", 256),
        ),
        semantic_constexprs=(
            ("BIAS_TYPE", "none"),
            ("IS_CAUSAL", False),
        ),
        c_values=(
            ("BLOCK_HEADDIM", 128),
            ("EVEN_M", True),
            ("EVEN_N", True),
            ("EVEN_HEADDIM", True),
            # Keep the model-level B4/H16/S8192/D128 valuation intact while
            # choosing a target-executable implementation schedule.  The
            # 128x128 tile requires 164352 bytes of shared memory on the
            # current target (limit 101376); BLOCK_* belongs to C, not J.
            ("BLOCK_M", 64),
            ("BLOCK_N", 64),
        ),
        grid=(128, 64),
        grid_contract=("ceil_div(seqlen_q,BLOCK_M)", "batch*nheads"),
        seed=131,
        input_profile="flashattention-B4-H16-S8192-D128-f16-v1",
    ),
    (
        "high_performance:flash_attn.flash_attn_triton._bwd_kernel"
    ): TritonWorkloadAdapter(
        "high_performance:flash_attn.flash_attn_triton._bwd_kernel",
        buffers=(
            _buffer(
                "Q", BufferRole.INPUT, "f16", (4, 8192, 16, 128),
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "K", BufferRole.INPUT, "f16", (4, 8192, 16, 128),
                shape_contract=("h0", "h4", "h2", "h3"),
            ),
            _buffer(
                "V", BufferRole.INPUT, "f16", (4, 8192, 16, 128),
                shape_contract=("h0", "h4", "h2", "h3"),
            ),
            # BIAS_TYPE=none makes this ABI carrier intentionally unread.
            _buffer("Bias", BufferRole.INPUT, "f16", (1,)),
            _buffer(
                "DO", BufferRole.INPUT, "f16", (4, 8192, 16, 128),
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "DQ",
                BufferRole.INOUT,
                "f32",
                (4, 8192, 16, 128),
                "dq",
                BufferInitializer.ZEROS,
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "DK", BufferRole.OUTPUT, "f32", (4, 8192, 16, 128), "dk",
                shape_contract=("h0", "h4", "h2", "h3"),
            ),
            _buffer(
                "DV", BufferRole.OUTPUT, "f32", (4, 8192, 16, 128), "dv",
                shape_contract=("h0", "h4", "h2", "h3"),
            ),
            _buffer(
                "LSE", BufferRole.INPUT, "f32", (4, 16, 8192),
                shape_contract=("h0", "h2", "h5"),
            ),
            _buffer(
                "D", BufferRole.INPUT, "f32", (4, 16, 8192),
                shape_contract=("h0", "h2", "h5"),
            ),
        ),
        scalars=(
            _scalar("softmax_scale", "f32", 0.08838834764831845),
            _scalar("stride_qb", "i64", 16777216),
            _scalar("stride_qh", "i64", 128),
            _scalar("stride_qm", "i64", 2048),
            _scalar("stride_kb", "i64", 16777216),
            _scalar("stride_kh", "i64", 128),
            _scalar("stride_kn", "i64", 2048),
            _scalar("stride_vb", "i64", 16777216),
            _scalar("stride_vh", "i64", 128),
            _scalar("stride_vn", "i64", 2048),
            _scalar("stride_bb", "i64", 0),
            _scalar("stride_bh", "i64", 0),
            _scalar("stride_bm", "i64", 0),
            _scalar("stride_dob", "i64", 16777216),
            _scalar("stride_doh", "i64", 128),
            _scalar("stride_dom", "i64", 2048),
            _scalar("stride_dqb", "i64", 16777216),
            _scalar("stride_dqh", "i64", 128),
            _scalar("stride_dqm", "i64", 2048),
            _scalar("stride_dkb", "i64", 16777216),
            _scalar("stride_dkh", "i64", 128),
            _scalar("stride_dkn", "i64", 2048),
            _scalar("stride_dvb", "i64", 16777216),
            _scalar("stride_dvh", "i64", 128),
            _scalar("stride_dvn", "i64", 2048),
            _scalar("nheads", "i32", 16, shape_coordinate="h2"),
            _scalar("seqlen_q", "i32", 8192, shape_coordinate="h1"),
            _scalar("seqlen_k", "i32", 8192, shape_coordinate="h4"),
            _scalar(
                "seqlen_q_rounded", "i32", 8192, shape_coordinate="h5"
            ),
            _scalar("headdim", "i32", 128, shape_coordinate="h3"),
            _scalar("CACHE_KEY_SEQLEN_Q", "i32", 256),
            _scalar("CACHE_KEY_SEQLEN_K", "i32", 256),
        ),
        semantic_constexprs=(
            ("BIAS_TYPE", "none"),
            ("IS_CAUSAL", False),
        ),
        c_values=(
            ("BLOCK_HEADDIM", 128),
            ("SEQUENCE_PARALLEL", False),
            ("EVEN_M", True),
            ("EVEN_N", True),
            ("EVEN_HEADDIM", True),
            # As above, this changes only the frozen implementation choice C;
            # tensor shapes and their logical shape identities remain model
            # level.  The backward kernel's first grid axis is source-fixed.
            ("BLOCK_M", 64),
            ("BLOCK_N", 64),
        ),
        grid=(1, 64),
        grid_contract=("1", "batch*nheads"),
        seed=137,
        input_profile="flashattention-B4-H16-S8192-D128-f16-v1",
    ),
    "high_performance:liger_fused_add_rms_norm": TritonWorkloadAdapter(
        "high_performance:liger_fused_add_rms_norm",
        buffers=(
            _buffer(
                "Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "S_ptr", BufferRole.OUTPUT, "f32", (2, 127), "s",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "X_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "R_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "W_ptr", BufferRole.INPUT, "f32", (127,),
                shape_contract=("h1",),
            ),
            _buffer(
                "RSTD_ptr", BufferRole.OUTPUT, "f32", (2,), "rstd",
                shape_contract=("h0",),
            ),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("S_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("R_row_stride", "i64", 127),
            _scalar("W_row_stride", "i64", 127),
            _scalar("RSTD_row_stride", "i64", 1),
            _scalar("n_cols", "i32", 127, shape_coordinate="h1"),
            _scalar("eps", "f32", 1.0e-6),
            _scalar("offset", "f32", 0.0),
        ),
        semantic_constexprs=(("casting_mode", -1),),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=139,
    ),
    "high_performance:liger_sparsemax": TritonWorkloadAdapter(
        "high_performance:liger_sparsemax",
        buffers=(
            _buffer(
                "x_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "sorted_x_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "o_ptr", BufferRole.OUTPUT, "f32", (2, 127), "o",
                shape_contract=("h0", "h1"),
            ),
        ),
        scalars=(
            _scalar("x_stride_row", "i64", 127),
            _scalar("sorted_x_stride_row", "i64", 127),
            _scalar("o_stride_row", "i64", 127),
            _scalar("n_cols", "i32", 127, shape_coordinate="h1"),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 128), ("num_warps", 4)),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=149,
    ),
    (
        "high_performance:flash_attn.flash_attn_triton."
        "_bwd_kernel_one_col_block"
    ): TritonWorkloadAdapter(
        (
            "high_performance:flash_attn.flash_attn_triton."
            "_bwd_kernel_one_col_block"
        ),
        buffers=(
            _buffer(
                "Q", BufferRole.INPUT, "f16", (128, 64),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "K", BufferRole.INPUT, "f16", (64, 64),
                shape_contract=("h2", "h1"),
            ),
            _buffer(
                "V", BufferRole.INPUT, "f16", (64, 64),
                shape_contract=("h2", "h1"),
            ),
            _buffer("Bias", BufferRole.INPUT, "f16", (1,)),
            _buffer(
                "DO", BufferRole.INPUT, "f16", (128, 64),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "DQ",
                BufferRole.INOUT,
                "f32",
                (128, 64),
                "dq",
                BufferInitializer.ZEROS,
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "DK", BufferRole.OUTPUT, "f32", (64, 64), "dk",
                shape_contract=("h2", "h1"),
            ),
            _buffer(
                "DV", BufferRole.OUTPUT, "f32", (64, 64), "dv",
                shape_contract=("h2", "h1"),
            ),
            _buffer(
                "LSE", BufferRole.INPUT, "f32", (128,),
                shape_contract=("h0",),
            ),
            _buffer(
                "D", BufferRole.INPUT, "f32", (128,),
                shape_contract=("h0",),
            ),
        ),
        scalars=(
            _scalar("start_n", "i32", 0),
            _scalar("softmax_scale", "f32", 0.125),
            _scalar("stride_qm", "i64", 64),
            _scalar("stride_kn", "i64", 64),
            _scalar("stride_vn", "i64", 64),
            _scalar("stride_bm", "i64", 0),
            _scalar("stride_dom", "i64", 64),
            _scalar("stride_dqm", "i64", 64),
            _scalar("stride_dkn", "i64", 64),
            _scalar("stride_dvn", "i64", 64),
            _scalar("seqlen_q", "i32", 128, shape_coordinate="h0"),
            _scalar("seqlen_k", "i32", 64, shape_coordinate="h2"),
            _scalar("headdim", "i32", 64, shape_coordinate="h1"),
        ),
        semantic_constexprs=(
            ("BIAS_TYPE", "none"),
            ("IS_CAUSAL", False),
        ),
        c_values=(
            ("ATOMIC_ADD", False),
            ("BLOCK_HEADDIM", 64),
            ("EVEN_M", True),
            ("EVEN_N", True),
            ("EVEN_HEADDIM", True),
            # The one-col kernel traverses all query rows in its internal
            # BLOCK_M loop, so 64 preserves the selected Sq=128 program while
            # avoiding the 108544-byte shared-memory image produced by 128 on
            # the certification GPU (101376-byte limit).  This is a reviewed
            # Harness/C valuation, not a J relation or backend repair.
            ("BLOCK_M", 64),
            ("BLOCK_N", 64),
        ),
        grid=(1,),
        grid_contract=("1",),
        seed=151,
        input_profile="flashattention-direct-tile-Sq128-Sk64-D64-f16-v1",
    ),
    "high_performance:liger_qwen2vl_mrope": TritonWorkloadAdapter(
        "high_performance:liger_qwen2vl_mrope",
        buffers=(
            _buffer(
                "q_ptr", BufferRole.INOUT, "f16", (1, 2048, 28, 128), "q",
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "k_ptr", BufferRole.INOUT, "f16", (1, 2048, 4, 128), "k",
                shape_contract=("h0", "h1", "h4", "h3"),
            ),
            _buffer(
                "cos", BufferRole.INPUT, "f16", (3, 1, 2048, 128),
                shape_contract=(3, "h0", "h1", "h3"),
            ),
            _buffer(
                "sin", BufferRole.INPUT, "f16", (3, 1, 2048, 128),
                shape_contract=(3, "h0", "h1", "h3"),
            ),
        ),
        scalars=(_scalar("sl", "i32", 2048, shape_coordinate="h1"),),
        semantic_constexprs=(
            ("bs", 1),
            ("n_qh", 28),
            ("n_kh", 4),
            ("hd", 128),
            ("mrope_section_t", 16),
            ("mrope_section_h", 24),
            ("BACKWARD_PASS", False),
        ),
        semantic_constexpr_shape_coordinates=(
            ("bs", "h0"),
            ("n_qh", "h2"),
            ("n_kh", "h4"),
            ("hd", "h3"),
        ),
        c_values=(
            ("pad_n_qh", 32),
            ("pad_n_kh", 4),
            ("pad_hd", 128),
            ("BLOCK_SIZE", 32),
        ),
        grid=(2048,),
        grid_contract=("bs*sl",),
        seed=157,
        input_profile="qwen2.5-7b-mrope-B1-T2048-Hq28-Hkv4-D128-f16-v1",
    ),
    "high_performance:liger_group_norm": TritonWorkloadAdapter(
        "high_performance:liger_group_norm",
        buffers=(
            _buffer(
                "Y_ptr", BufferRole.OUTPUT, "f32", (2, 2, 32), "y",
                shape_contract=("h0", "h1", "h2"),
            ),
            _buffer(
                "X_ptr", BufferRole.INPUT, "f32", (2, 2, 32),
                shape_contract=("h0", "h1", "h2"),
            ),
            _buffer(
                "Mean_ptr", BufferRole.OUTPUT, "f32", (2, 2), "mean",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "RSTD_ptr", BufferRole.OUTPUT, "f32", (2, 2), "rstd",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "W_ptr", BufferRole.INPUT, "f32", (4,),
                shape_contract=("h3",),
            ),
            _buffer(
                "B_ptr", BufferRole.INPUT, "f32", (4,),
                shape_contract=("h3",),
            ),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 64),
            _scalar("Y_col_stride", "i64", 32),
            _scalar("X_row_stride", "i64", 64),
            _scalar("X_col_stride", "i64", 32),
            _scalar("Mean_row_stride", "i64", 2),
            _scalar("Mean_col_stride", "i64", 1),
            _scalar("RSTD_row_stride", "i64", 2),
            _scalar("RSTD_col_stride", "i64", 1),
            _scalar("hidden_size", "i32", 32, shape_coordinate="h2"),
            _scalar("channels_per_group", "i32", 2),
            _scalar("eps", "f32", 1.0e-6),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 32),),
        grid=(2, 2),
        grid_contract=("batch", "groups"),
        grid_shape_coordinates=("h0", "h1"),
        seed=163,
    ),
    "high_performance:liger_jsd": TritonWorkloadAdapter(
        "high_performance:liger_jsd",
        buffers=(
            _buffer(
                "X_ptr", BufferRole.INPUT, "f32", (2, 64),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "Y_ptr", BufferRole.INPUT, "f32", (2, 64),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "loss_ptr", BufferRole.OUTPUT, "f32", (2, 64), "loss",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "dX_ptr", BufferRole.OUTPUT, "f32", (2, 64), "dx",
                shape_contract=("h0", "h1"),
            ),
            # HAS_LABEL=false makes this required ABI carrier unread.
            _buffer(
                "label_ptr", BufferRole.INPUT, "i64", (2,),
                shape_contract=("h0",),
            ),
        ),
        scalars=(
            _scalar("X_stride", "i64", 64),
            _scalar("Y_stride", "i64", 64),
            _scalar("loss_stride", "i64", 64),
            _scalar("dX_stride", "i64", 64),
            _scalar("n_non_ignore", "i32", 2),
            _scalar("n_cols", "i32", 64, shape_coordinate="h1"),
        ),
        semantic_constexprs=(
            ("beta", 0.5),
            ("ignore_index", -100),
            ("HAS_LABEL", False),
        ),
        c_values=(("BLOCK_SIZE", 64),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=167,
    ),
    "high_performance:liger_cross_entropy": TritonWorkloadAdapter(
        "high_performance:liger_cross_entropy",
        buffers=(
            # The runtime ignore-index early-return clears this buffer even
            # when HAS_GRADIENTS=false, so its Source effect contract is inout.
            _buffer(
                "X_ptr", BufferRole.INOUT, "f32", (2, 64), "x",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "Y_ptr",
                BufferRole.INPUT,
                "i64",
                (2,),
                initializer=BufferInitializer.ZEROS,
                shape_contract=("h0",),
            ),
            # HAS_WEIGHT=false makes this required ABI carrier unread.
            _buffer(
                "weight_ptr", BufferRole.INPUT, "f32", (64,),
                shape_contract=("h1",),
            ),
            _buffer(
                "loss_ptr", BufferRole.OUTPUT, "f32", (2,), "loss",
                shape_contract=("h0",),
            ),
            # The three RETURN_* flags below are false, so these ABI carriers
            # are deliberately inputs rather than observable uninitialized
            # outputs.
            _buffer(
                "z_loss_ptr", BufferRole.INPUT, "f32", (2,),
                shape_contract=("h0",),
            ),
            _buffer(
                "token_accuracy_ptr", BufferRole.INPUT, "f32", (2,),
                shape_contract=("h0",),
            ),
            _buffer(
                "predicted_tokens_ptr", BufferRole.INPUT, "i64", (2,),
                shape_contract=("h0",),
            ),
        ),
        scalars=(
            _scalar("X_stride", "i64", 64),
            _scalar("Y_stride", "i64", 1),
            _scalar("loss_stride", "i64", 1),
            _scalar("token_accuracy_stride", "i64", 0),
            _scalar("predicted_tokens_stride", "i64", 0),
            _scalar("n_cols", "i32", 64, shape_coordinate="h1"),
            _scalar("n_non_ignore", "f32", 2.0),
            _scalar("sum_non_ignore_weight", "f32", 2.0),
            _scalar("weight_sum", "f32", 0.0),
            _scalar("ignore_index", "i64", -100),
            _scalar("softcap", "f32", 0.0),
        ),
        semantic_constexprs=(
            ("lse_square_scale", 0.0),
            ("label_smoothing", 0.0),
            ("reduction", "mean"),
            ("RETURN_Z_LOSS", False),
            ("RETURN_TOKEN_ACCURACY", False),
            ("RETURN_PREDICTED_TOKENS", False),
            ("HAS_WEIGHT", False),
            ("HAS_SOFTCAPPING", False),
            ("HAS_GRADIENTS", False),
        ),
        c_values=(("BLOCK_SIZE", 64),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=173,
    ),
    "high_performance:liger_geglu": TritonWorkloadAdapter(
        "high_performance:liger_geglu",
        buffers=(
            _buffer(
                "a", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "b", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "c", BufferRole.OUTPUT, "f32", (2, 127), "c",
                shape_contract=("h0", "h1"),
            ),
        ),
        scalars=(_scalar("stride", "i64", 127),),
        semantic_constexprs=(("n_cols", 127),),
        semantic_constexpr_shape_coordinates=(("n_cols", "h1"),),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=179,
    ),
    "high_performance:liger_poly_norm": TritonWorkloadAdapter(
        "high_performance:liger_poly_norm",
        buffers=(
            _buffer(
                "Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "X_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer("W_ptr", BufferRole.INPUT, "f32", (3,)),
            _buffer("B_ptr", BufferRole.INPUT, "f32", (1,)),
            _buffer(
                "RSTD_ptr", BufferRole.OUTPUT, "f32", (2, 3), "rstd",
                shape_contract=("h0", 3),
            ),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("RSTD_row_stride", "i64", 3),
            _scalar("n_cols", "i32", 127, shape_coordinate="h1"),
            _scalar("eps", "f32", 1.0e-6),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=181,
    ),
    "high_performance:liger_llama4_rope": TritonWorkloadAdapter(
        "high_performance:liger_llama4_rope",
        buffers=(
            _buffer(
                "q_ptr", BufferRole.INOUT, "f16", (1, 2048, 32, 128), "q",
                shape_contract=("h0", "h1", "h2", 128),
            ),
            _buffer(
                "k_ptr", BufferRole.INOUT, "f16", (1, 2048, 8, 128), "k",
                shape_contract=("h0", "h1", "h3", 128),
            ),
            _buffer(
                "freqs_complex_ptr", BufferRole.INPUT, "f16", (2048, 64, 2),
                shape_contract=("h1", "h4", 2),
            ),
        ),
        scalars=(
            _scalar("q_row_stride", "i64", 4096),
            _scalar("k_row_stride", "i64", 1024),
            _scalar("q_head_stride", "i64", 128),
            _scalar("k_head_stride", "i64", 128),
            _scalar("freqs_row_stride", "i64", 128),
            _scalar("seq_len", "i32", 2048, shape_coordinate="h1"),
            _scalar("batch_size", "i32", 1, shape_coordinate="h0"),
            _scalar("imag_sign", "f32", 1.0),
        ),
        semantic_constexprs=(
            ("head_dim_half", 64),
            ("n_q_heads", 32),
            ("n_k_heads", 8),
        ),
        semantic_constexpr_shape_coordinates=(
            ("head_dim_half", "h4"),
            ("n_q_heads", "h2"),
            ("n_k_heads", "h3"),
        ),
        c_values=(("BLOCK_SIZE", 64),),
        grid=(2048, 32),
        grid_contract=("batch_size*seq_len", "max(n_q_heads,n_k_heads)"),
        seed=191,
        input_profile="llama4-rope-B1-T2048-Hq32-Hkv8-D128-f16-v1",
    ),
    (
        "high_performance:flash_attn.ops.triton.layer_norm."
        "_layer_norm_fwd_1pass_kernel"
    ): TritonWorkloadAdapter(
        (
            "high_performance:flash_attn.ops.triton.layer_norm."
            "_layer_norm_fwd_1pass_kernel"
        ),
        buffers=(
            _buffer(
                "X", BufferRole.INPUT, "f16", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "Y", BufferRole.OUTPUT, "f16", (2, 127), "y",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "W", BufferRole.INPUT, "f16", (127,),
                shape_contract=("h1",),
            ),
            _buffer(
                "B", BufferRole.INPUT, "f16", (127,),
                shape_contract=("h1",),
            ),
            _buffer(
                "RESIDUAL", BufferRole.INPUT, "f16", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "X1", BufferRole.INPUT, "f16", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "W1", BufferRole.INPUT, "f16", (127,),
                shape_contract=("h1",),
            ),
            _buffer(
                "B1", BufferRole.INPUT, "f16", (127,),
                shape_contract=("h1",),
            ),
            _buffer(
                "Y1", BufferRole.OUTPUT, "f16", (2, 127), "y1",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "RESIDUAL_OUT",
                BufferRole.OUTPUT,
                "f16",
                (2, 127),
                "residual_out",
                shape_contract=("h0", "h1"),
            ),
            # The fixed flags make these four required ABI carriers unread.
            _buffer(
                "ROWSCALE", BufferRole.INPUT, "f32", (2,),
                shape_contract=("h0",),
            ),
            _buffer(
                "SEEDS", BufferRole.INPUT, "i64", (2,),
                shape_contract=("h0",),
            ),
            _buffer(
                "DROPOUT_MASK", BufferRole.INPUT, "i1", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "DROPOUT_MASK1", BufferRole.INPUT, "i1", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "Mean", BufferRole.OUTPUT, "f32", (2,), "mean",
                shape_contract=("h0",),
            ),
            _buffer(
                "Rstd", BufferRole.OUTPUT, "f32", (2,), "rstd",
                shape_contract=("h0",),
            ),
        ),
        scalars=(
            _scalar("stride_x_row", "i64", 127),
            _scalar("stride_y_row", "i64", 127),
            _scalar("stride_res_row", "i64", 127),
            _scalar("stride_res_out_row", "i64", 127),
            _scalar("stride_x1_row", "i64", 127),
            _scalar("stride_y1_row", "i64", 127),
            _scalar("M", "i32", 2, shape_coordinate="h0"),
            _scalar("N", "i32", 127, shape_coordinate="h1"),
            _scalar("eps", "f32", 1.0e-6),
            _scalar("dropout_p", "f32", 0.0),
            _scalar("zero_centered_weight", "i1", False),
        ),
        semantic_constexprs=(
            ("IS_RMS_NORM", False),
            ("HAS_RESIDUAL", True),
            ("STORE_RESIDUAL_OUT", True),
            ("HAS_BIAS", True),
            ("HAS_DROPOUT", False),
            ("STORE_DROPOUT_MASK", False),
            ("HAS_ROWSCALE", False),
            ("HAS_X1", True),
            ("HAS_W1", True),
            ("HAS_B1", True),
        ),
        c_values=(("BLOCK_N", 128),),
        grid=(2,),
        grid_contract=("M",),
        grid_shape_coordinates=("h0",),
        seed=193,
    ),
    "high_performance:liger_rope": TritonWorkloadAdapter(
        "high_performance:liger_rope",
        buffers=(
            _buffer(
                "q_ptr", BufferRole.INOUT, "f16", (1, 2048, 32, 128), "q",
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "k_ptr", BufferRole.INOUT, "f16", (1, 2048, 8, 128), "k",
                shape_contract=("h0", "h1", "h4", "h3"),
            ),
            _buffer(
                "cos", BufferRole.INPUT, "f16", (1, 2048, 128),
                shape_contract=("h5", "h1", "h3"),
            ),
            _buffer(
                "sin", BufferRole.INPUT, "f16", (1, 2048, 128),
                shape_contract=("h5", "h1", "h3"),
            ),
        ),
        scalars=(
            _scalar("q_row_stride", "i32", 4096),
            _scalar("k_row_stride", "i32", 1024),
            _scalar("cos_row_stride", "i32", 128),
            _scalar("sin_row_stride", "i32", 128),
            _scalar("sl", "i32", 2048, shape_coordinate="h1"),
        ),
        semantic_constexprs=(
            ("bs", 1),
            ("cos_bs", 1),
            ("n_qh", 32),
            ("n_kh", 8),
            ("hd", 128),
            ("BACKWARD_PASS", False),
        ),
        semantic_constexpr_shape_coordinates=(
            ("bs", "h0"),
            ("cos_bs", "h5"),
            ("n_qh", "h2"),
            ("n_kh", "h4"),
            ("hd", "h3"),
        ),
        c_values=(
            ("pad_n_qh", 32),
            ("pad_n_kh", 8),
            ("pad_hd", 128),
            ("BLOCK_SIZE", 32),
        ),
        grid=(2048,),
        grid_contract=("bs*sl",),
        seed=197,
        input_profile="llama-3-8b-rope-B1-T2048-Hq32-Hkv8-D128-f16-v1",
    ),
    "high_performance:liger_layer_norm": TritonWorkloadAdapter(
        "high_performance:liger_layer_norm",
        buffers=(
            _buffer(
                "Y_ptr", BufferRole.OUTPUT, "f32", (2, 127), "y",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "X_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "W_ptr", BufferRole.INPUT, "f32", (127,),
                shape_contract=("h1",),
            ),
            _buffer(
                "B_ptr", BufferRole.INPUT, "f32", (127,),
                shape_contract=("h1",),
            ),
            _buffer(
                "Mean_ptr", BufferRole.OUTPUT, "f32", (2,), "mean",
                shape_contract=("h0",),
            ),
            _buffer(
                "RSTD_ptr", BufferRole.OUTPUT, "f32", (2,), "rstd",
                shape_contract=("h0",),
            ),
        ),
        scalars=(
            _scalar("Y_row_stride", "i64", 127),
            _scalar("X_row_stride", "i64", 127),
            _scalar("W_row_stride", "i64", 127),
            _scalar("B_row_stride", "i64", 127),
            _scalar("Mean_row_stride", "i64", 1),
            _scalar("RSTD_row_stride", "i64", 1),
            _scalar("n_cols", "i32", 127, shape_coordinate="h1"),
            _scalar("eps", "f32", 1.0e-6),
        ),
        semantic_constexprs=(),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=199,
    ),
    "high_performance:liger_kl_div": TritonWorkloadAdapter(
        "high_performance:liger_kl_div",
        buffers=(
            _buffer(
                "y_ptr", BufferRole.INPUT, "f32", (2, 64),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "gt_ptr", BufferRole.INPUT, "f32", (2, 64),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "loss_ptr", BufferRole.OUTPUT, "f32", (2,), "loss",
                shape_contract=("h0",),
            ),
        ),
        scalars=(
            _scalar("y_stride", "i64", 64),
            _scalar("gt_stride", "i64", 64),
            _scalar("loss_stride", "i64", 1),
            _scalar("n_cols", "i32", 64, shape_coordinate="h1"),
            _scalar("eps", "f32", 1.0e-6),
        ),
        semantic_constexprs=(
            ("log_target", True),
            ("reduction", 3),
        ),
        c_values=(("BLOCK_SIZE", 64),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=211,
    ),
    "high_performance:liger_tvd": TritonWorkloadAdapter(
        "high_performance:liger_tvd",
        buffers=(
            _buffer(
                "p_ptr", BufferRole.INPUT, "f32", (2, 64),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "q_ptr", BufferRole.INPUT, "f32", (2, 64),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "loss_ptr", BufferRole.OUTPUT, "f32", (2, 64), "loss",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "grads_ptr", BufferRole.OUTPUT, "f32", (2, 64), "grads",
                shape_contract=("h0", "h1"),
            ),
            # HAS_LABEL=false makes this required ABI carrier unread.
            _buffer(
                "label_ptr", BufferRole.INPUT, "i64", (2,),
                shape_contract=("h0",),
            ),
        ),
        scalars=(
            _scalar("p_stride", "i64", 64),
            _scalar("q_stride", "i64", 64),
            _scalar("loss_stride", "i64", 64),
            _scalar("grads_stride", "i64", 64),
            _scalar("n_cols", "i32", 64, shape_coordinate="h1"),
            _scalar("scale", "f32", 1.0),
        ),
        semantic_constexprs=(
            ("ignore_index", -100),
            ("HAS_LABEL", False),
            ("reduction", 0),
        ),
        c_values=(("BLOCK_SIZE", 64),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=223,
    ),
    (
        "high_performance:xformers.ops._triton.rmsnorm_kernels."
        "_rms_norm_kernel"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops._triton.rmsnorm_kernels."
            "_rms_norm_kernel"
        ),
        buffers=(
            _buffer(
                "x_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "h1_ptr", BufferRole.OUTPUT, "f32", (2, 127), "h1",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "w_ptr", BufferRole.INPUT, "f32", (127,),
                shape_contract=("h1",),
            ),
        ),
        scalars=(
            _scalar("eps", "f32", 1.0e-6),
            _scalar("stride", "i64", 127),
        ),
        semantic_constexprs=(
            ("N_COLS", 127),
            ("INCLUDE_WEIGHT", True),
        ),
        semantic_constexpr_shape_coordinates=(("N_COLS", "h1"),),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=227,
    ),
    (
        "high_performance:xformers.ops._triton.rmsnorm_kernels."
        "_rms_norm_add_kernel"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops._triton.rmsnorm_kernels."
            "_rms_norm_add_kernel"
        ),
        buffers=(
            _buffer(
                "x_ptr", BufferRole.INOUT, "f32", (2, 127), "x",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "y_ptr", BufferRole.INPUT, "f32", (2, 127),
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "h1_ptr", BufferRole.OUTPUT, "f32", (2, 127), "h1",
                shape_contract=("h0", "h1"),
            ),
            _buffer(
                "w_ptr", BufferRole.INPUT, "f32", (127,),
                shape_contract=("h1",),
            ),
        ),
        scalars=(
            _scalar("eps", "f32", 1.0e-6),
            _scalar("stride", "i64", 127),
        ),
        semantic_constexprs=(
            ("N_COLS", 127),
            ("INCLUDE_WEIGHT", True),
        ),
        semantic_constexpr_shape_coordinates=(("N_COLS", "h1"),),
        c_values=(("BLOCK_SIZE", 128),),
        grid=(2,),
        grid_contract=("rows",),
        grid_shape_coordinates=("h0",),
        seed=229,
    ),
    (
        "high_performance:xformers.ops.fmha._triton.splitk_kernels."
        "_fwd_kernel_splitK"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops.fmha._triton.splitk_kernels."
            "_fwd_kernel_splitK"
        ),
        buffers=(
            _buffer(
                "Q", BufferRole.INPUT, "f16", (1, 1, 1, 16, 256),
                shape_contract=("h0", "h1", "h2", "h3", "h4"),
            ),
            _buffer(
                "K", BufferRole.INPUT, "f16", (1, 4096, 1, 16, 256),
                shape_contract=("h0", "h5", "h2", "h3", "h4"),
            ),
            _buffer(
                "V", BufferRole.INPUT, "f16", (1, 4096, 1, 16, 256),
                shape_contract=("h0", "h5", "h2", "h3", "h4"),
            ),
            _buffer(
                "Out_splitK",
                BufferRole.OUTPUT,
                "f32",
                (1, 1, 16, 4, 32, 256),
                "out_splitk",
                shape_contract=("h0", "h2", "h3", 4, 32, "h4"),
            ),
            _buffer(
                "LSE_splitk",
                BufferRole.OUTPUT,
                "f32",
                (1, 1, 16, 4, 1),
                "lse_splitk",
                shape_contract=("h0", "h2", "h3", 4, "h1"),
            ),
        ),
        scalars=(
            _scalar("sm_scale", "f32", 0.0625),
            *(
                _scalar(name, "i64", value)
                for name, value in (
                    ("stride_qz", 4096),
                    ("stride_qm", 4096),
                    ("stride_qg", 4096),
                    ("stride_qh", 256),
                    ("stride_qk", 1),
                    ("stride_kz", 16777216),
                    ("stride_kn", 4096),
                    ("stride_kg", 4096),
                    ("stride_kh", 256),
                    ("stride_kk", 1),
                    ("stride_vz", 16777216),
                    ("stride_vn", 4096),
                    ("stride_vg", 4096),
                    ("stride_vh", 256),
                    ("stride_vk", 1),
                    ("stride_osk_z", 524288),
                    ("stride_osk_g", 524288),
                    ("stride_osk_h", 32768),
                    ("stride_osk_s", 8192),
                    ("stride_osk_m", 256),
                    ("stride_osk_k", 1),
                    ("stride_lsek_z", 64),
                    ("stride_lsek_g", 64),
                    ("stride_lsek_h", 4),
                    ("stride_lsek_s", 1),
                    ("stride_lsek_m", 1),
                    ("stride_blocktablesz", 0),
                    ("stride_blocktablesl", 0),
                    ("stride_bias_b", 0),
                    ("stride_bias_g", 0),
                    ("stride_bias_h", 0),
                    ("stride_bias_qm", 0),
                    ("stride_bias_km", 0),
                )
            ),
        ),
        semantic_constexprs=(
            ("Z", 1),
            ("N_CTX_Q", 1),
            ("N_CTX_K", 4096),
            ("BLOCK_N_PER_SPLIT", 1024),
            ("H", 16),
            ("G", 1),
            ("BLOCK_DMODEL", 256),
            ("USE_SEQ_LEN", False),
            ("PACKED_PER_VAL", 1),
            ("N_GROUPS", 1),
            ("IS_SPLITK", True),
            ("SPLIT_K_EARLY_EXIT", False),
            ("IS_CAUSAL", False),
            ("IS_LOCAL", False),
            ("NUM_QUERIES_CAUSAL", 1),
            ("USE_PAGED_ATTENTION", False),
            ("PAGE_SIZE", 0),
            ("WINDOW_LEFT", -1),
            ("WINDOW_RIGHT", -1),
            ("WRITE_LSE", True),
            ("HAS_ADDITIVE_BIAS", False),
        ),
        semantic_constexpr_shape_coordinates=(
            ("Z", "h0"),
            ("N_CTX_Q", "h1"),
            ("G", "h2"),
            ("H", "h3"),
            ("BLOCK_DMODEL", "h4"),
            ("N_CTX_K", "h5"),
        ),
        c_values=(
            ("stride_k_fp8_scale_shift_z", 0),
            ("stride_k_fp8_scale_shift_n", 0),
            ("stride_k_fp8_scale_shift_g", 0),
            ("stride_k_fp8_scale_shift_h", 0),
            ("stride_v_fp8_scale_shift_z", 0),
            ("stride_v_fp8_scale_shift_n", 0),
            ("stride_v_fp8_scale_shift_g", 0),
            ("stride_v_fp8_scale_shift_h", 0),
            ("kv_cache_blocks_per_row", 0),
            ("BOUNDS_CHECKS_N", True),
            ("BLOCK_M", 16),
            ("BLOCK_N", 64),
            ("NUM_PROGRAMS_DIM2_CONST", 4),
            ("IS_HIP", False),
        ),
        grid=(1, 16, 4),
        grid_contract=("ceil_div(N_CTX_Q,BLOCK_M)", "Z*G*H", "split_k"),
        seed=241,
        compiler_options=(("num_warps", 2), ("num_stages", 1)),
        input_profile=(
            "xformers-decode-B1-Mq1-G1-H16-KV4096-D256-"
            "split4-bm16-bn64-w2-s1-f16-v2"
        ),
        none_arguments=tuple(
            NoneBinding(name)
            for name in (
                "block_tables",
                "Seq_len",
                "Seq_starts_k",
                "Seq_starts_q",
                "Seq_starts_q_multiplier",
                "additive_bias",
                "K_fp8_scale_shift",
                "V_fp8_scale_shift",
            )
        ),
        vararg_arity=1,
    ),
    (
        "high_performance:xformers.ops.fmha._triton.splitk_kernels."
        "_splitK_reduce_varargs"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops.fmha._triton.splitk_kernels."
            "_splitK_reduce_varargs"
        ),
        buffers=(
            *(
                _buffer(
                    f"Out_splitK{index}", BufferRole.INPUT, "f32",
                    (1, 1, 16, 1, 256),
                    shape_contract=("h0", "h1", "h2", "h3", "h4"),
                )
                for index in range(4)
            ),
            *(
                _buffer(
                    f"LSE_splitK{index}", BufferRole.INPUT, "f32",
                    (1, 1, 16, 1),
                    shape_contract=("h0", "h1", "h2", "h3"),
                )
                for index in range(4)
            ),
            _buffer(
                "Out", BufferRole.OUTPUT, "f32", (1, 1, 16, 1, 256), "out",
                shape_contract=("h0", "h1", "h2", "h3", "h4"),
            ),
            _buffer(
                "LSE", BufferRole.OUTPUT, "f32", (1, 1, 16, 1), "lse",
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
        ),
        scalars=tuple(
            _scalar(f"stride_{family}{index}", "i64", value)
            for family, value in (
                ("osk_z", 4096),
                ("osk_g", 4096),
                ("osk_h", 256),
                ("osk_m", 256),
                ("osk_k", 1),
                ("lsek_z", 16),
                ("lsek_g", 16),
                ("lsek_h", 1),
                ("lsek_m", 1),
            )
            for index in range(4)
        ) + tuple(
            _scalar(name, "i64", value)
            for name, value in (
                ("stride_oz", 4096),
                ("stride_og", 4096),
                ("stride_oh", 256),
                ("stride_om", 256),
                ("stride_ok", 1),
                ("stride_lse_z", 16),
                ("stride_lse_g", 16),
                ("stride_lse_h", 1),
                ("stride_lse_m", 1),
            )
        ),
        semantic_constexprs=(
            ("head_dim", 256),
            ("H", 16),
            ("G", 1),
            ("WRITE_LSE", True),
        ),
        semantic_constexpr_shape_coordinates=(
            ("G", "h1"),
            ("H", "h2"),
            ("head_dim", "h4"),
        ),
        c_values=(("head_dim_pow_2", 256),),
        grid=(1, 16, 1),
        grid_contract=("M", "B*G*H", "1"),
        seed=233,
        input_profile="xformers-decode-B1-Mq1-G1-H16-D256-split4-f32-v1",
        vararg_arity=4,
    ),
    (
        "high_performance:xformers.ops.fmha._triton.splitk_kernels."
        "_splitK_reduce_varargs_backward"
    ): TritonWorkloadAdapter(
        (
            "high_performance:xformers.ops.fmha._triton.splitk_kernels."
            "_splitK_reduce_varargs_backward"
        ),
        buffers=(
            *(
                _buffer(
                    f"Out_splitK{index}", BufferRole.INPUT, "f32",
                    (1, 1, 16, 1, 256),
                    shape_contract=("h0", "h1", "h2", "h3", "h4"),
                )
                for index in range(4)
            ),
            *(
                _buffer(
                    f"LSE_splitK{index}", BufferRole.INPUT, "f32",
                    (1, 1, 16, 1),
                    shape_contract=("h0", "h1", "h2", "h3"),
                )
                for index in range(4)
            ),
            *(
                _buffer(
                    f"Dout_splitK{index}", BufferRole.OUTPUT, "f32",
                    (1, 1, 16, 1, 256), f"dout{index}",
                    shape_contract=("h0", "h1", "h2", "h3", "h4"),
                )
                for index in range(4)
            ),
            *(
                _buffer(
                    f"DLSE_splitK{index}", BufferRole.OUTPUT, "f32",
                    (1, 1, 16, 1), f"dlse{index}",
                    shape_contract=("h0", "h1", "h2", "h3"),
                )
                for index in range(4)
            ),
            _buffer(
                "Out", BufferRole.INPUT, "f32", (1, 1, 16, 1, 256),
                shape_contract=("h0", "h1", "h2", "h3", "h4"),
            ),
            _buffer(
                "LSE", BufferRole.INPUT, "f32", (1, 1, 16, 1),
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
            _buffer(
                "DOut", BufferRole.INPUT, "f32", (1, 1, 16, 1, 256),
                shape_contract=("h0", "h1", "h2", "h3", "h4"),
            ),
            _buffer(
                "DLSE", BufferRole.INPUT, "f32", (1, 1, 16, 1),
                shape_contract=("h0", "h1", "h2", "h3"),
            ),
        ),
        scalars=tuple(
            _scalar(f"stride_{family}{index}", "i64", value)
            for family, value in (
                ("osk_z", 4096),
                ("osk_g", 4096),
                ("osk_h", 256),
                ("osk_m", 256),
                ("osk_k", 1),
                ("lsek_z", 16),
                ("lsek_g", 16),
                ("lsek_h", 1),
                ("lsek_m", 1),
            )
            for index in range(4)
        ) + tuple(
            _scalar(name, "i64", value)
            for name, value in (
                ("stride_oz", 4096),
                ("stride_og", 4096),
                ("stride_oh", 256),
                ("stride_om", 256),
                ("stride_ok", 1),
                ("stride_lse_z", 16),
                ("stride_lse_g", 16),
                ("stride_lse_h", 1),
                ("stride_lse_m", 1),
                ("stride_doz", 4096),
                ("stride_dog", 4096),
                ("stride_doh", 256),
                ("stride_dom", 256),
                ("stride_dok", 1),
                ("stride_dlse_z", 16),
                ("stride_dlse_g", 16),
                ("stride_dlse_h", 1),
                ("stride_dlse_m", 1),
            )
        ),
        semantic_constexprs=(
            ("BLOCK_SIZE", 256),
            ("H", 16),
            ("G", 1),
        ),
        semantic_constexpr_shape_coordinates=(
            ("G", "h1"),
            ("H", "h2"),
            ("BLOCK_SIZE", "h4"),
        ),
        c_values=(),
        grid=(1, 16, 1),
        grid_contract=("M", "B*G*H", "1"),
        seed=239,
        input_profile="xformers-decode-B1-Mq1-G1-H16-D256-split4-f32-v1",
        vararg_arity=4,
    ),
}


__all__ = [
    "BufferBinding",
    "BufferInitializer",
    "BufferRole",
    "GeneratedSourceLifecycleRecord",
    "GeneratedTargetSourceArgument",
    "GeneratedTargetSourceManifest",
    "NoneBinding",
    "PreparedTritonWorkload",
    "TRITON_WORKLOAD_ADAPTER_PROTOCOL",
    "TRITON_WORKLOAD_ADAPTERS",
    "ScalarBinding",
    "TritonWorkloadAdapter",
    "build_generated_source_manifest",
    "prepare_captured_triton_workload",
    "reingress_triton_workload",
    "prepare_triton_source_workload",
    "prepare_triton_workload",
]
