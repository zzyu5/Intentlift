"""Compact, target-rebuildable payloads for production benchmark workloads.

Large natural tensors dwarf the target binaries in a portable search bundle.
The production Harness already defines those tensors by a closed initializer,
seed, dtype, shape, and (optionally) a derived-input transform.  Transport that
definition and rebuild it on the target; scalar literals remain inline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from intentir.backend.cuda.portable_bundle import decode_values, encode_values
from intentir.runtime_dtype import numpy_bfloat16_dtype

if TYPE_CHECKING:
    from intentir.verify.workload_adapters import TritonWorkloadAdapter


WORKLOAD_RECIPE_PROTOCOL = "intentir.backend.portable-workload.v1"


def _role_name(value: Any) -> str:
    return str(getattr(value, "value", value))


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    normalized = dtype.removeprefix("torch.").lower()
    if normalized == "bf16":
        return numpy_bfloat16_dtype()
    aliases: dict[str, Any] = {
        "bool": np.bool_,
        "i1": np.bool_,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "index": np.int64,
        "u8": np.uint8,
        "u16": np.uint16,
        "u32": np.uint32,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "ui8": np.uint8,
        "ui16": np.uint16,
        "ui32": np.uint32,
        "f16": np.float16,
        "f32": np.float32,
        "f64": np.float64,
    }
    try:
        return np.dtype(aliases[normalized])
    except KeyError as exc:
        raise ValueError(f"unsupported portable workload dtype {dtype!r}") from exc


def encode_workload_recipe(
    adapter: TritonWorkloadAdapter,
    source_to_opaque: Mapping[str, str],
    arguments: Sequence[Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Encode the exact Harness value construction without tensor payloads."""

    value_key_by_opaque = {
        str(argument.name): str(argument.value_key)
        for argument in arguments
        if argument.value_key is not None
    }
    runtime_key_by_formal: dict[str, str] = {}
    buffers: list[dict[str, Any]] = []
    generated_keys: set[str] = set()
    for buffer in adapter.buffers:
        if _role_name(buffer.role) not in {"input", "inout"}:
            continue
        opaque = source_to_opaque.get(buffer.formal)
        if opaque is None or opaque not in value_key_by_opaque:
            raise ValueError(
                f"portable workload buffer {buffer.formal!r} has no runtime key"
            )
        runtime_key = value_key_by_opaque[opaque]
        runtime_key_by_formal[buffer.formal] = runtime_key
        generated_keys.add(runtime_key)
        buffers.append(
            {
                "runtime_key": runtime_key,
                "dtype": buffer.dtype,
                "shape": list(buffer.shape),
                "initializer": (
                    buffer.initializer.value
                    if buffer.initializer is not None
                    else None
                ),
                "source_runtime_key": None,
                "input_transform": (
                    buffer.input_transform.value
                    if buffer.input_transform is not None
                    else None
                ),
            }
        )
    for buffer, record in zip(
        (
            item
            for item in adapter.buffers
            if _role_name(item.role) in {"input", "inout"}
        ),
        buffers,
        strict=True,
    ):
        if buffer.source_formal is not None:
            record["source_runtime_key"] = runtime_key_by_formal[
                buffer.source_formal
            ]

    literal_values = {
        name: value for name, value in values.items() if name not in generated_keys
    }
    if generated_keys | set(literal_values) != set(values):
        raise AssertionError("portable workload recipe lost a runtime value")
    return {
        "protocol": WORKLOAD_RECIPE_PROTOCOL,
        "seed": adapter.seed,
        "buffers": buffers,
        "literal_values": encode_values(literal_values),
        "value_keys": sorted(values),
    }


def decode_workload_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a production workload from its closed portable recipe."""

    if value.get("protocol") != WORKLOAD_RECIPE_PROTOCOL:
        raise ValueError("unsupported portable workload recipe")
    if set(value) != {
        "protocol",
        "seed",
        "buffers",
        "literal_values",
        "value_keys",
    }:
        raise ValueError("portable workload recipe fields differ from v1")
    records = value["buffers"]
    if not isinstance(records, list):
        raise ValueError("portable workload buffers must be a list")
    rng = np.random.default_rng(int(value["seed"]))
    result: dict[str, Any] = decode_values(value["literal_values"])

    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "runtime_key",
            "dtype",
            "shape",
            "initializer",
            "source_runtime_key",
            "input_transform",
        }:
            raise ValueError("portable workload buffer record is invalid")
        if record["source_runtime_key"] is not None:
            continue
        key = str(record["runtime_key"])
        dtype = _numpy_dtype(str(record["dtype"]))
        shape = tuple(int(item) for item in record["shape"])
        initializer = record["initializer"]
        if initializer == "zeros":
            generated = np.zeros(shape, dtype=dtype)
        elif initializer == "standard_normal":
            generated = rng.standard_normal(size=shape).astype(dtype)
        elif initializer == "random":
            if dtype.kind == "b":
                generated = rng.integers(0, 2, size=shape).astype(dtype)
            elif dtype.kind in {"i", "u"}:
                low = 0 if dtype.kind == "u" else -3
                generated = rng.integers(low, 4, size=shape).astype(dtype)
            else:
                generated = rng.normal(0.0, 0.25, size=shape).astype(dtype)
        else:
            raise ValueError(f"unsupported workload initializer {initializer!r}")
        result[key] = generated

    for record in records:
        source_key = record["source_runtime_key"]
        if source_key is None:
            continue
        key = str(record["runtime_key"])
        if record["input_transform"] == "sort_descending_last_axis":
            result[key] = np.ascontiguousarray(
                np.flip(np.sort(result[str(source_key)], axis=-1), axis=-1)
            )
        else:
            raise ValueError(
                f"unsupported workload input transform {record['input_transform']!r}"
            )

    if set(result) != set(value["value_keys"]):
        raise ValueError("rebuilt workload keys differ from the portable recipe")
    return result


__all__ = [
    "WORKLOAD_RECIPE_PROTOCOL",
    "decode_workload_recipe",
    "encode_workload_recipe",
]
