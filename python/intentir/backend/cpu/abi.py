"""Strict NumPy bindings for the CPU reference ranked-memref ABI.

This module only describes storage.  It does not implement IntentIR semantics,
convert values, or provide a fallback execution path.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Sequence

import numpy as np

from intentir.runtime_dtype import numpy_bfloat16_dtype


CPU_MEMREF_ABI_PROTOCOL = "intentir_cpu_ranked_memref_abi_v1"
CPU_ENTRY_SYMBOL = "intentir_cpu_entry_v1"
CPU_C_INTERFACE_SYMBOL = f"_mlir_ciface_{CPU_ENTRY_SYMBOL}"


class CPUBackendUnsupportedError(NotImplementedError):
    """The declared typed program is outside the scalar backend capability."""


class CPUABIValidationError(ValueError):
    """A runtime value violates the declared static CPU ABI."""


_MLIR_ELEMENT_TYPES = {
    "bool": "i1",
    "i1": "i1",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "ui8": "ui8",
    "ui16": "ui16",
    "ui32": "ui32",
    "u8": "ui8",
    "u16": "ui16",
    "u32": "ui32",
    "f16": "f16",
    "bf16": "bf16",
    "f32": "f32",
    "f64": "f64",
}

_ALIASES = {
    # ``index`` is a logical contract dtype; the native host carrier is i64.
    "index": "i64",
    "half": "f16",
    "float16": "f16",
    "bfloat16": "bf16",
    "float": "f32",
    "float32": "f32",
    "double": "f64",
    "float64": "f64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
}

_NUMPY_DTYPES = {
    "i1": np.dtype(np.bool_),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "ui8": np.dtype(np.uint8),
    "ui16": np.dtype(np.uint16),
    "ui32": np.dtype(np.uint32),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}


def mlir_element_type(dtype: str) -> str:
    """Return the canonical MLIR element type for one IntentIR dtype."""

    spelling = dtype.casefold().removeprefix("torch.")
    spelling = _ALIASES.get(spelling, spelling)
    try:
        return _MLIR_ELEMENT_TYPES[spelling]
    except KeyError as exc:
        raise CPUBackendUnsupportedError(
            f"unsupported CPU execution dtype {dtype!r}"
        ) from exc


def numpy_dtype(dtype: str) -> np.dtype:
    """Return the exact runtime dtype, including registered BF16 storage."""

    element_type = mlir_element_type(dtype)
    if element_type == "bf16":
        return numpy_bfloat16_dtype()
    return _NUMPY_DTYPES[element_type]


def validate_array(
    value: np.ndarray,
    *,
    shape: Sequence[int],
    dtype: str,
    context: str,
    writable: bool = False,
) -> np.ndarray:
    """Validate one ABI value without copying or coercing it."""

    if not isinstance(value, np.ndarray):
        raise CPUABIValidationError(f"{context} must be a NumPy ndarray")
    expected_shape = tuple(int(extent) for extent in shape)
    if any(extent < 0 for extent in expected_shape):
        raise CPUABIValidationError(f"{context} has a negative static extent")
    if value.shape != expected_shape:
        raise CPUABIValidationError(
            f"{context} shape mismatch: expected {expected_shape}, got {value.shape}"
        )
    expected_dtype = numpy_dtype(dtype)
    if value.dtype != expected_dtype:
        raise CPUABIValidationError(
            f"{context} dtype mismatch: expected {expected_dtype}, got {value.dtype}"
        )
    if not value.dtype.isnative:
        raise CPUABIValidationError(f"{context} must use native byte order")
    if not value.flags.c_contiguous:
        raise CPUABIValidationError(f"{context} must be C-contiguous")
    if not value.flags.aligned:
        raise CPUABIValidationError(
            f"{context} must satisfy NumPy dtype alignment"
        )
    if writable and not value.flags.writeable:
        raise CPUABIValidationError(f"{context} must be writable")
    return value


@lru_cache(maxsize=None)
def ranked_memref_descriptor_type(rank: int) -> type[ctypes.Structure]:
    """Create the canonical MLIR C-interface descriptor for a static rank."""

    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("ranked memref ABI rank must be a non-negative integer")

    fields: list[tuple[str, object]] = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
    ]
    if rank:
        fields.extend(
            [
                ("sizes", ctypes.c_int64 * rank),
                ("strides", ctypes.c_int64 * rank),
            ]
        )

    class RankedMemRefDescriptor(ctypes.Structure):
        _fields_ = fields

    RankedMemRefDescriptor.__name__ = f"RankedMemRefDescriptor{rank}D"
    return RankedMemRefDescriptor


def descriptor_for(value: np.ndarray) -> ctypes.Structure:
    """Build a borrowed ranked-memref descriptor for a validated ndarray."""

    descriptor_type = ranked_memref_descriptor_type(value.ndim)
    address = int(value.ctypes.data)
    common = [address, address, 0]
    if value.ndim == 0:
        return descriptor_type(*common)
    sizes = (ctypes.c_int64 * value.ndim)(*value.shape)
    strides = (ctypes.c_int64 * value.ndim)(
        *(stride // value.dtype.itemsize for stride in value.strides)
    )
    return descriptor_type(*common, sizes, strides)


__all__ = [
    "CPU_C_INTERFACE_SYMBOL",
    "CPU_ENTRY_SYMBOL",
    "CPU_MEMREF_ABI_PROTOCOL",
    "CPUABIValidationError",
    "CPUBackendUnsupportedError",
    "descriptor_for",
    "mlir_element_type",
    "numpy_dtype",
    "ranked_memref_descriptor_type",
    "validate_array",
]
