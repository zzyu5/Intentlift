#!/usr/bin/env python3
"""Link, execute, validate, and search a native RVV object bundle.

This target-side runner intentionally uses only the Python standard library.
The target computation lives in precompiled objects produced by the MLIR/LLVM
pipeline; Python owns descriptor binding, correctness audit, and timing only.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


BUNDLE_PROTOCOL = "intentir.production.rvv-search-bundle.v4"
REPORT_PROTOCOL = "intentir.production.rvv-remote-performance.v6"
NATIVE_COMPARATOR_PROTOCOL = "intentir.backend.native-output-comparator.v1"
NATIVE_BASELINE_PROTOCOL = "intentir.backend.riscv.native-baseline.v1"
NATIVE_BASELINE_SIDECAR_PROTOCOL = (
    "intentir.backend.riscv.native-baseline-sidecar.v1"
)
EXTERNAL_BASELINE_PROTOCOL = "intentir.backend.riscv.external-baseline.v1"
EXTERNAL_BASELINE_SIDECAR_PROTOCOL = (
    "intentir.backend.riscv.external-baseline-sidecar.v1"
)
_SAFE_BASELINE_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_BEST_NATIVE_KINDS = frozenset({"expert_intrinsic", "mature_library"})
RVV_TARGET_MARCH = "rv64gcv_zfh_zvfh"

_NATIVE_COMPARATOR_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <string.h>

int intentir_equal_bytes(const void *left, const void *right, uint64_t bytes) {
  return memcmp(left, right, (size_t)bytes) == 0;
}

int intentir_close_f32(const float *left, const float *right, uint64_t count,
                       double atol, double rtol, int equal_nan,
                       double *max_abs_error) {
  double maximum = 0.0;
  for (uint64_t index = 0; index < count; ++index) {
    const double a = (double)left[index];
    const double b = (double)right[index];
    if (isnan(a) || isnan(b)) {
      if (!(equal_nan && isnan(a) && isnan(b))) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    if (isinf(a) || isinf(b)) {
      if (a != b) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    const double error = fabs(a - b);
    if (error > maximum)
      maximum = error;
    if (error > atol + rtol * fmax(fabs(a), fabs(b))) {
      *max_abs_error = maximum;
      return 0;
    }
  }
  *max_abs_error = maximum;
  return 1;
}

int intentir_close_f64(const double *left, const double *right, uint64_t count,
                       double atol, double rtol, int equal_nan,
                       double *max_abs_error) {
  double maximum = 0.0;
  for (uint64_t index = 0; index < count; ++index) {
    const double a = left[index];
    const double b = right[index];
    if (isnan(a) || isnan(b)) {
      if (!(equal_nan && isnan(a) && isnan(b))) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    if (isinf(a) || isinf(b)) {
      if (a != b) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    const double error = fabs(a - b);
    if (error > maximum)
      maximum = error;
    if (error > atol + rtol * fmax(fabs(a), fabs(b))) {
      *max_abs_error = maximum;
      return 0;
    }
  }
  *max_abs_error = maximum;
  return 1;
}

static float intentir_half_to_float(uint16_t value) {
  const uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
  uint32_t exponent = ((uint32_t)value >> 10) & 0x1fu;
  uint32_t mantissa = (uint32_t)value & 0x03ffu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03ffu;
      bits = sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static float intentir_bf16_to_float(uint16_t value) {
  const uint32_t bits = (uint32_t)value << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static int intentir_close_16(const uint16_t *left, const uint16_t *right,
                             uint64_t count, double atol, double rtol,
                             int equal_nan, double *max_abs_error,
                             int bfloat) {
  double maximum = 0.0;
  for (uint64_t index = 0; index < count; ++index) {
    const double a = bfloat ? intentir_bf16_to_float(left[index])
                            : intentir_half_to_float(left[index]);
    const double b = bfloat ? intentir_bf16_to_float(right[index])
                            : intentir_half_to_float(right[index]);
    if (isnan(a) || isnan(b)) {
      if (!(equal_nan && isnan(a) && isnan(b))) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    if (isinf(a) || isinf(b)) {
      if (a != b) {
        *max_abs_error = INFINITY;
        return 0;
      }
      continue;
    }
    const double error = fabs(a - b);
    if (error > maximum)
      maximum = error;
    if (error > atol + rtol * fmax(fabs(a), fabs(b))) {
      *max_abs_error = maximum;
      return 0;
    }
  }
  *max_abs_error = maximum;
  return 1;
}

int intentir_close_f16(const uint16_t *left, const uint16_t *right,
                       uint64_t count, double atol, double rtol, int equal_nan,
                       double *max_abs_error) {
  return intentir_close_16(left, right, count, atol, rtol, equal_nan,
                           max_abs_error, 0);
}

int intentir_close_bf16(const uint16_t *left, const uint16_t *right,
                        uint64_t count, double atol, double rtol, int equal_nan,
                        double *max_abs_error) {
  return intentir_close_16(left, right, count, atol, rtol, equal_nan,
                           max_abs_error, 1);
}
"""


class KernelFailure(RuntimeError):
    def __init__(self, message: str, diagnostics: list[dict[str, Any]]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _payload_path(record: Mapping[str, Any], payload_root: Path) -> Path:
    relative = Path(str(record["data_file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RVV array payload path must stay below the bundle directory")
    path = payload_root / relative
    if not path.is_file():
        raise ValueError(f"RVV array payload is unavailable: {path}")
    return path


def _decode_payload(record: Mapping[str, Any], payload_root: Path) -> bytes:
    embedded = "data_base64" in record
    external = "data_file" in record
    if embedded == external:
        raise ValueError("RVV array payload must use exactly one transport")
    if external:
        return _payload_path(record, payload_root).read_bytes()
    return base64.b64decode(record["data_base64"], validate=True)


def _payload_size(record: Mapping[str, Any], payload_root: Path) -> int:
    if "data_file" in record:
        if "data_base64" in record:
            raise ValueError("RVV array payload mixes embedded and file transport")
        return _payload_path(record, payload_root).stat().st_size
    return len(_decode_payload(record, payload_root))


def _numel(shape: list[int]) -> int:
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def _resolve_runtime_output_specs(
    entry: Mapping[str, Any], payload_root: Path
) -> list[dict[str, Any]]:
    """Resolve runtime contracts without letting reference data define the ABI.

    ``output_specs.shape`` is the concrete native allocation contract emitted
    by the bundle exporter.  ``shape_contract`` may retain ``?`` or a symbolic
    J extent for provenance, but the transported reference can only validate
    that contract; it must never replace the native shape used for allocation.
    """

    reference_outputs = entry.get("reference_outputs")
    bindings = entry.get("output_bindings")
    raw_specs = entry.get("output_specs")
    if not isinstance(reference_outputs, Mapping):
        raise ValueError("RVV bundle has no transported reference outputs")
    if not isinstance(bindings, Mapping) or not isinstance(raw_specs, list):
        raise ValueError("RVV bundle has an invalid output ABI")
    resolved: list[dict[str, Any]] = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, Mapping):
            raise ValueError("RVV output specification is not an object")
        name = raw_spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("RVV output specification has an invalid name")
        binding = bindings.get(name)
        if not isinstance(binding, str) or binding not in reference_outputs:
            raise ValueError(f"RVV reference is missing output {name!r}")
        record = reference_outputs[binding]
        if not isinstance(record, Mapping) or not isinstance(record.get("shape"), list):
            raise ValueError(f"RVV reference shape for output {name!r} is invalid")
        reference_shape_values = record["shape"]
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in reference_shape_values
        ):
            raise ValueError(f"RVV reference shape for output {name!r} is invalid")
        reference_shape = tuple(reference_shape_values)
        if any(item < 0 for item in reference_shape):
            raise ValueError(f"RVV reference shape for output {name!r} is negative")
        declared_shape = raw_spec.get("shape")
        if not isinstance(declared_shape, list):
            raise ValueError(f"RVV output specification for {name!r} has an invalid shape")
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in declared_shape
        ):
            raise ValueError(
                f"RVV output specification for {name!r} must use a concrete integer shape"
            )
        abi_shape = tuple(declared_shape)
        if any(item < 0 for item in abi_shape):
            raise ValueError(
                f"RVV output specification for {name!r} must use non-negative concrete extents"
            )
        if abi_shape != reference_shape:
            raise ValueError(
                f"RVV reference shape for output {name!r} differs from its native ABI: "
                f"{list(reference_shape)} != {list(abi_shape)}"
            )
        contract = raw_spec.get("shape_contract", list(abi_shape))
        if not isinstance(contract, list) or len(contract) != len(abi_shape):
            raise ValueError(f"RVV output contract for {name!r} has an invalid shape")
        extents = entry.get("extents", {})
        if not isinstance(extents, Mapping):
            raise ValueError("RVV bundle extents must be an object")
        for axis, (contract_extent, actual_extent) in enumerate(
            zip(contract, abi_shape, strict=True)
        ):
            if contract_extent == "?":
                continue
            if isinstance(contract_extent, bool):
                raise ValueError(
                    f"RVV output contract for {name!r} contains a boolean extent"
                )
            if isinstance(contract_extent, int):
                expected_extent = contract_extent
            elif isinstance(contract_extent, str) and contract_extent.isidentifier():
                expected_extent = extents.get(contract_extent)
                if (
                    isinstance(expected_extent, bool)
                    or not isinstance(expected_extent, int)
                    or expected_extent < 0
                ):
                    raise ValueError(
                        f"RVV output {name!r} has an unbound symbolic extent "
                        f"{contract_extent!r}"
                    )
            else:
                raise ValueError(
                    f"RVV output contract for {name!r} has an unsupported extent "
                    f"at axis {axis}"
                )
            if expected_extent != actual_extent:
                raise ValueError(
                    f"RVV output {name!r} contract differs from its reference shape"
                )
        payload = _decode_payload(record, payload_root)
        dtype = raw_spec.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"RVV output specification for {name!r} has no dtype")
        expected_bytes = _numel(list(abi_shape)) * _dtype_size(dtype)
        if len(payload) != expected_bytes:
            raise ValueError(f"RVV reference payload size differs from output {name!r}")
        spec = dict(raw_spec)
        spec["shape"] = list(abi_shape)
        resolved.append(spec)
    return resolved


def _dtype_size(dtype: str) -> int:
    key = dtype.removeprefix("torch.").lower()
    return {
        "bool": 1,
        "i1": 1,
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "i64": 8,
        "f16": 2,
        "bf16": 2,
        "f32": 4,
        "f64": 8,
    }[key]


def _cache_size_bytes(text: str) -> int:
    value = text.strip().upper()
    multiplier = 1
    if value.endswith("K"):
        multiplier = 1024
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1024 * 1024
        value = value[:-1]
    return int(value) * multiplier


def _cpu_cache_record() -> dict[str, Any]:
    caches = []
    root = Path("/sys/devices/system/cpu/cpu0/cache")
    for index in sorted(root.glob("index*")):
        try:
            cache_type = (index / "type").read_text().strip()
            level = int((index / "level").read_text().strip())
            size_bytes = _cache_size_bytes((index / "size").read_text())
            line_bytes = int((index / "coherency_line_size").read_text().strip())
        except (OSError, ValueError):
            continue
        caches.append(
            {
                "level": level,
                "type": cache_type,
                "size_bytes": size_bytes,
                "coherency_line_bytes": line_bytes,
            }
        )
    data_caches = [
        item for item in caches if item["type"] in {"Data", "Unified"}
    ]
    last_level_bytes = max(
        (item["size_bytes"] for item in data_caches), default=0
    )
    return {"levels": caches, "last_level_data_cache_bytes": last_level_bytes}


@lru_cache(maxsize=None)
def _descriptor_type(rank: int) -> type[ctypes.Structure]:
    fields: list[tuple[str, Any]] = [
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
    return type(
        f"MemRefDescriptor{rank}",
        (ctypes.Structure,),
        {"_fields_": fields},
    )


def _descriptor_from_address(address: int, shape: list[int]) -> ctypes.Structure:
    descriptor_type = _descriptor_type(len(shape))
    result = descriptor_type()
    result.allocated = address
    result.aligned = address
    result.offset = 0
    if shape:
        stride = 1
        strides = [0] * len(shape)
        for index in range(len(shape) - 1, -1, -1):
            strides[index] = stride
            stride *= int(shape[index])
        for index, extent in enumerate(shape):
            result.sizes[index] = int(extent)
            result.strides[index] = int(strides[index])
    return result


def _descriptor(buffer: Any, shape: list[int]) -> ctypes.Structure:
    return _descriptor_from_address(ctypes.addressof(buffer), shape)


class _NativeComparator:
    def __init__(self, directory: Path):
        source = directory / "intentir_native_compare.c"
        library_path = directory / "intentir_native_compare.so"
        source.write_text(_NATIVE_COMPARATOR_SOURCE, encoding="utf-8")
        compiler = shutil.which("clang") or shutil.which("cc")
        if compiler is None:
            raise RuntimeError("native comparator requires target clang or cc")
        process = subprocess.run(
            [
                compiler,
                "-shared",
                "-O3",
                "-fPIC",
                str(source),
                "-lm",
                "-o",
                str(library_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"native comparator compile failed: {process.stderr}")
        self.library = ctypes.CDLL(str(library_path))
        self.equal_bytes = self.library.intentir_equal_bytes
        self.equal_bytes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        self.equal_bytes.restype = ctypes.c_int
        self.close_functions = {}
        for dtype in ("f16", "bf16", "f32", "f64"):
            function = getattr(self.library, f"intentir_close_{dtype}")
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
            ]
            function.restype = ctypes.c_int
            self.close_functions[dtype] = function
        self.record = {
            "protocol": NATIVE_COMPARATOR_PROTOCOL,
            "compiler": f"{Path(compiler).name} -O3 -shared -fPIC",
            "compares_all_output_elements": True,
            "python_elementwise_comparison": False,
        }

    def close(
        self,
        actual: Any,
        expected: Any,
        *,
        byte_count: int,
        element_count: int,
        dtype: str,
        rule: Mapping[str, Any],
    ) -> tuple[bool, float]:
        actual_pointer = ctypes.c_void_p(ctypes.addressof(actual))
        expected_pointer = ctypes.c_void_p(ctypes.addressof(expected))
        key = dtype.removeprefix("torch.").lower()
        tolerance = rule.get("tolerance")
        if tolerance is None or key not in self.close_functions:
            equal = bool(
                self.equal_bytes(actual_pointer, expected_pointer, byte_count)
            )
            return equal, 0.0 if equal else float("inf")
        maximum = ctypes.c_double(0.0)
        passed = bool(
            self.close_functions[key](
                actual_pointer,
                expected_pointer,
                element_count,
                float(tolerance["atol"]),
                float(tolerance["rtol"]),
                int(bool(tolerance["equal_nan"])),
                ctypes.byref(maximum),
            )
        )
        return passed, float(maximum.value)


def _link_object(directory: Path, candidate: Mapping[str, Any]) -> Path:
    candidate_id = candidate["candidate"]["candidate_id"]
    object_path = directory / f"{candidate_id}.o"
    library_path = directory / f"{candidate_id}.so"
    object_bytes = base64.b64decode(candidate["object_base64"], validate=True)
    object_path.write_bytes(object_bytes)
    process = subprocess.run(
        [
            "clang",
            "-shared",
            f"-march={RVV_TARGET_MARCH}",
            "-fopenmp",
            str(object_path),
            "-lm",
            "-o",
            str(library_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"target link failed: {process.stderr}")
    return library_path


def _validated_native_baseline(record: Any) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise RuntimeError("RVV entry lacks an independent Native baseline")
    if record.get("protocol") != NATIVE_BASELINE_PROTOCOL:
        raise RuntimeError("unsupported Native RVV baseline protocol")
    source_text = record.get("source")
    symbol = record.get("symbol")
    if not isinstance(source_text, str) or not source_text.strip():
        raise RuntimeError("Native RVV baseline source is empty")
    if not isinstance(symbol, str) or not symbol:
        raise RuntimeError("Native RVV baseline symbol is empty")
    return record


def _legacy_external_baseline(record: Any) -> dict[str, Any]:
    native = _validated_native_baseline(record)
    return {
        "protocol": EXTERNAL_BASELINE_PROTOCOL,
        "baseline_id": "llvm_rvv_c",
        "display_name": "LLVM-RVV C",
        "kind": "portable_compiler",
        "symbol": native["symbol"],
        "source": native["source"],
        "language": "c",
        "build_profile": "clang-rvv-c-strict",
        "implementation": native.get("implementation"),
        "strict_match": True,
    }


def _validated_external_baseline(record: Any) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise RuntimeError("RVV external baseline record must be an object")
    if record.get("protocol") == NATIVE_BASELINE_PROTOCOL:
        return _legacy_external_baseline(record)
    if record.get("protocol") != EXTERNAL_BASELINE_PROTOCOL:
        raise RuntimeError("unsupported RVV external baseline protocol")
    baseline_id = record.get("baseline_id")
    if not isinstance(baseline_id, str) or _SAFE_BASELINE_ID.fullmatch(baseline_id) is None:
        raise RuntimeError("RVV external baseline has an unsafe baseline_id")
    for field in ("display_name", "kind", "symbol", "source", "implementation"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise RuntimeError(f"RVV external baseline {baseline_id!r} lacks {field}")
    if record.get("language") not in {"c", "c++"}:
        raise RuntimeError(f"RVV external baseline {baseline_id!r} has invalid language")
    if record.get("build_profile") not in {
        "clang-rvv-c-strict",
        "clang-rvv-intrinsic-strict",
        "ggml-rvv-strict",
        "triton-cpu-rvv-prebuilt-strict",
    }:
        raise RuntimeError(f"RVV external baseline {baseline_id!r} has invalid build profile")
    if record.get("strict_match") is not True:
        raise RuntimeError(f"RVV external baseline {baseline_id!r} is not a strict match")
    if record.get("build_profile") == "triton-cpu-rvv-prebuilt-strict":
        library = record.get("prebuilt_library")
        if not isinstance(library, str) or not library:
            raise RuntimeError(
                f"RVV external baseline {baseline_id!r} lacks prebuilt_library"
            )
        source_identity = record.get("source_identity")
        if not isinstance(source_identity, str) or not source_identity:
            raise RuntimeError(
                f"RVV external baseline {baseline_id!r} lacks source_identity"
            )
    return record


def _load_native_baseline_sidecar(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != NATIVE_BASELINE_SIDECAR_PROTOCOL:
        raise ValueError("unsupported Native RVV baseline sidecar")
    raw_baselines = payload.get("baselines")
    if not isinstance(raw_baselines, Mapping):
        raise ValueError("Native RVV baseline sidecar lacks a baseline map")
    result: dict[str, Mapping[str, Any]] = {}
    for kernel, record in raw_baselines.items():
        if not isinstance(kernel, str) or not kernel:
            raise ValueError("Native RVV baseline sidecar has an invalid kernel name")
        result[kernel] = _validated_native_baseline(record)
    return result


def _load_external_baseline_sidecar(
    path: Path,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if protocol == NATIVE_BASELINE_SIDECAR_PROTOCOL:
        return {
            kernel: (_legacy_external_baseline(record),)
            for kernel, record in _load_native_baseline_sidecar(path).items()
        }
    if protocol != EXTERNAL_BASELINE_SIDECAR_PROTOCOL:
        raise ValueError("unsupported RVV external baseline sidecar")
    raw_baselines = payload.get("baselines")
    if not isinstance(raw_baselines, Mapping):
        raise ValueError("RVV external baseline sidecar lacks a baseline map")
    result: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for kernel, records in raw_baselines.items():
        if not isinstance(kernel, str) or not kernel:
            raise ValueError("RVV external baseline sidecar has an invalid kernel name")
        if not isinstance(records, list) or not records:
            raise ValueError(f"RVV external baseline list for {kernel!r} is empty")
        validated = tuple(_validated_external_baseline(record) for record in records)
        ids = [record["baseline_id"] for record in validated]
        if len(ids) != len(set(ids)):
            raise ValueError(f"RVV external baseline IDs for {kernel!r} are duplicated")
        result[kernel] = validated
    return result


def _attach_native_baselines(
    entries: list[Mapping[str, Any]],
    sidecar: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for raw_entry in entries:
        entry = dict(raw_entry)
        kernel = entry.get("kernel")
        if kernel in sidecar:
            entry["native_baseline"] = sidecar[kernel]
            entry["baseline_policy"] = "native_rvv_and_refj"
        _validated_native_baseline(entry.get("native_baseline"))
        result.append(entry)
    return result


def _attach_external_baselines(
    entries: list[Mapping[str, Any]],
    sidecar: Mapping[str, tuple[Mapping[str, Any], ...]],
) -> list[dict[str, Any]]:
    result = []
    for raw_entry in entries:
        entry = dict(raw_entry)
        kernel = entry.get("kernel")
        if kernel in sidecar:
            records = tuple(sidecar[kernel])
        elif isinstance(entry.get("external_baselines"), list):
            records = tuple(
                _validated_external_baseline(record)
                for record in entry["external_baselines"]
            )
        elif entry.get("native_baseline") is not None:
            records = (_legacy_external_baseline(entry["native_baseline"]),)
        else:
            raise RuntimeError(
                f"RVV entry {kernel!r} lacks an independent external baseline"
            )
        ids = [record["baseline_id"] for record in records]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"RVV entry {kernel!r} has duplicate external baseline IDs")
        entry["external_baselines"] = list(records)
        entry["baseline_policy"] = "external_rvv_and_refj"
        result.append(entry)
    return result


def _select_best_native(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choose a target-native competitor, never the portability compiler bar."""

    applicable = [
        record
        for record in records
        if record.get("status") == "measured"
        and record.get("kind") in _BEST_NATIVE_KINDS
    ]
    return (
        min(applicable, key=lambda record: float(record["latency_us"]))
        if applicable
        else None
    )


def _compile_native_baseline(
    directory: Path, entry: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:
    record = _validated_native_baseline(entry.get("native_baseline"))
    source_text = record["source"]
    symbol = record["symbol"]
    source = directory / "native_rvv_baseline.c"
    library_path = directory / "native_rvv_baseline.so"
    source.write_text(source_text, encoding="utf-8")
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        raise RuntimeError("Native RVV baseline requires target clang or cc")
    command = [
        compiler,
        "-shared",
        "-O3",
        "-fPIC",
        f"-march={RVV_TARGET_MARCH}",
        "-fopenmp",
        "-fno-fast-math",
        "-ffp-contract=off",
        str(source),
        "-lm",
        "-o",
        str(library_path),
    ]
    started = time.perf_counter_ns()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"Native RVV baseline compilation failed: {process.stderr}")
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, symbol)
    return library, function, {
        "protocol": NATIVE_BASELINE_PROTOCOL,
        "implementation": record.get("implementation"),
        "symbol": symbol,
        "compiler": compiler,
        "compile_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "flags": command[1:-3],
        "shares_intentir_candidate_or_object": False,
    }


def _compile_external_baseline(
    directory: Path,
    record: Mapping[str, Any],
    *,
    ggml_root: Path | None,
    ggml_lib_dir: Path | None,
) -> tuple[Any, Any, dict[str, Any]]:
    record = _validated_external_baseline(record)
    baseline_id = str(record["baseline_id"])
    source_suffix = ".cpp" if record["language"] == "c++" else ".c"
    source = directory / f"external_{baseline_id}{source_suffix}"
    library_path = directory / f"external_{baseline_id}.so"
    source.write_text(str(record["source"]), encoding="utf-8")
    profile = str(record["build_profile"])
    if profile == "triton-cpu-rvv-prebuilt-strict":
        compiler = shutil.which("clang") or shutil.which("cc")
        if compiler is None:
            raise RuntimeError("Triton-CPU/RVV adapter requires target clang or cc")
        prebuilt_library = Path(str(record["prebuilt_library"]))
        if not prebuilt_library.is_file():
            raise RuntimeError(
                f"Triton-CPU/RVV object does not exist: {prebuilt_library}"
            )
        command = [
            compiler,
            "-shared",
            "-O3",
            "-fPIC",
            f"-march={RVV_TARGET_MARCH}",
            "-fno-fast-math",
            "-ffp-contract=off",
            str(source),
            str(prebuilt_library),
            f"-Wl,-rpath,{prebuilt_library.parent}",
            "-lm",
            "-o",
            str(library_path),
        ]
    elif profile == "ggml-rvv-strict":
        if ggml_root is None or ggml_lib_dir is None:
            raise RuntimeError("ggml RVV baseline requires --ggml-root and --ggml-lib-dir")
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            raise RuntimeError("ggml RVV baseline requires target clang++ or c++")
        command = [
            compiler,
            "-shared",
            "-O3",
            "-std=c++17",
            "-fPIC",
            f"-march={RVV_TARGET_MARCH}",
            "-fno-fast-math",
            "-ffp-contract=off",
            str(source),
            f"-I{ggml_root / 'ggml' / 'include'}",
            f"-L{ggml_lib_dir}",
            f"-Wl,-rpath,{ggml_lib_dir}",
            "-lggml-cpu",
            "-lggml",
            "-lggml-base",
            "-lm",
            "-o",
            str(library_path),
        ]
    else:
        compiler = shutil.which("clang") or shutil.which("cc")
        if compiler is None:
            raise RuntimeError("RVV external C baseline requires target clang or cc")
        command = [
            compiler,
            "-shared",
            "-O3",
            "-fPIC",
            f"-march={RVV_TARGET_MARCH}",
            "-fopenmp",
            "-fno-fast-math",
            "-ffp-contract=off",
            str(source),
            "-lm",
            "-o",
            str(library_path),
        ]
    started = time.perf_counter_ns()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"RVV external baseline {baseline_id!r} compilation failed: {process.stderr}"
        )
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, str(record["symbol"]))
    return library, function, {
        "protocol": EXTERNAL_BASELINE_PROTOCOL,
        "baseline_id": baseline_id,
        "display_name": record["display_name"],
        "kind": record["kind"],
        "implementation": record["implementation"],
        "symbol": record["symbol"],
        "build_profile": profile,
        "compiler": compiler,
        "compile_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "flags": command[1:-3],
        "strict_match": True,
        "shares_intentir_candidate_or_object": False,
        "source_identity": record.get("source_identity"),
        "specialized_columns": record.get("specialized_columns"),
        "specialization": record.get("specialization"),
    }


class _ArgumentRotation:
    def __init__(self, argument_sets: list[list[Any]]):
        if not argument_sets:
            raise ValueError("argument rotation requires at least one replica")
        self.argument_sets = argument_sets
        self.cursor = 0

    def next(self) -> list[Any]:
        result = self.argument_sets[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.argument_sets)
        return result


def _logical_io_bytes(
    entry: Mapping[str, Any], payload_root: Path
) -> dict[str, int]:
    input_bytes = sum(
        _payload_size(
            entry["values"][entry["input_bindings"][name]], payload_root
        )
        for name in entry["input_order"]
    )
    output_bytes = sum(
        _numel(spec["shape"]) * _dtype_size(spec["dtype"])
        for spec in entry["output_specs"]
    )
    return {
        "boundary_input_bytes": input_bytes,
        "boundary_output_bytes": output_bytes,
        "boundary_io_bytes": input_bytes + output_bytes,
    }


def _argument_replicas(
    entry: Mapping[str, Any], replicas: int, payload_root: Path
) -> tuple[list[list[Any]], list[Any], list[list[Any]]]:
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    all_buffers: list[Any] = []
    output_buffers: list[list[Any]] = []
    argument_sets: list[list[Any]] = []
    for _ in range(replicas):
        descriptors = []
        replica_outputs = []
        for name in entry["input_order"]:
            value_record = entry["values"][entry["input_bindings"][name]]
            payload = _decode_payload(value_record, payload_root)
            buffer = ctypes.create_string_buffer(payload, len(payload))
            all_buffers.append(buffer)
            descriptors.append(
                _descriptor(buffer, [int(item) for item in value_record["shape"]])
            )
        for spec in entry["output_specs"]:
            size = _numel(spec["shape"]) * _dtype_size(spec["dtype"])
            buffer = ctypes.create_string_buffer(max(1, size))
            all_buffers.append(buffer)
            replica_outputs.append(buffer)
            descriptors.append(_descriptor(buffer, list(spec["shape"])))
        output_buffers.append(replica_outputs)
        argument_sets.append([ctypes.byref(item) for item in descriptors])
        # byref objects keep descriptors alive, and all_buffers keeps payloads
        # alive.  Retain the descriptor list explicitly for PyPy/CPython parity.
        all_buffers.extend(descriptors)
    return argument_sets, all_buffers, output_buffers


def _evict(eviction_buffer: Any, fill: int) -> float:
    started = time.perf_counter_ns()
    ctypes.memset(
        ctypes.addressof(eviction_buffer),
        fill & 0xFF,
        ctypes.sizeof(eviction_buffer),
    )
    return (time.perf_counter_ns() - started) / 1_000.0


def _measure(
    function: Any,
    rotation: _ArgumentRotation,
    *,
    regime: str,
    eviction_buffer: Any | None,
    warmup: int | None,
    warmup_ms: float | None,
    iterations: int,
    batches: int,
    timed_batch_ms: float | None,
) -> tuple[float, list[float], float, int, list[int]]:
    if warmup is not None and warmup_ms is not None:
        raise ValueError("choose either fixed-count or time-budgeted warmup")
    warmup_invocations = 0
    warmup_deadline_ns = (
        time.perf_counter_ns() + int(warmup_ms * 1_000_000)
        if warmup_ms is not None and warmup_ms > 0
        else None
    )
    while True:
        if warmup is not None:
            if warmup_invocations >= warmup:
                break
        elif warmup_deadline_ns is None or (
            warmup_invocations > 0
            and time.perf_counter_ns() >= warmup_deadline_ns
        ):
            break
        if regime == "cold-data":
            assert eviction_buffer is not None
            _evict(eviction_buffer, rotation.cursor)
        function(*rotation.next())
        warmup_invocations += 1
    samples = []
    prep_samples = []
    batch_invocation_counts = []
    for _ in range(batches):
        if regime == "cold-data":
            call_samples = []
            timed_us = 0.0
            for iteration in range(iterations):
                assert eviction_buffer is not None
                prep_samples.append(_evict(eviction_buffer, iteration))
                started = time.perf_counter_ns()
                function(*rotation.next())
                elapsed_us = (time.perf_counter_ns() - started) / 1_000.0
                call_samples.append(elapsed_us)
                timed_us += elapsed_us
                if (
                    timed_batch_ms is not None
                    and timed_us >= timed_batch_ms * 1_000.0
                ):
                    break
            samples.append(float(statistics.median(call_samples)))
            batch_invocation_counts.append(len(call_samples))
            continue
        started = time.perf_counter_ns()
        for _ in range(iterations):
            function(*rotation.next())
        elapsed = time.perf_counter_ns() - started
        samples.append(elapsed / iterations / 1_000.0)
        batch_invocation_counts.append(iterations)
    prep_us = float(statistics.median(prep_samples)) if prep_samples else 0.0
    return (
        float(statistics.median(samples)),
        samples,
        prep_us,
        warmup_invocations,
        batch_invocation_counts,
    )


def _bounded_candidate_orders(
    entry: Mapping[str, Any],
    search_budget: int,
    *,
    thread_count: int,
    measure_c_oracle: bool = False,
) -> dict[str, Any]:
    """Select the exact bounded timing union before any candidate executes.

    C-Fixed owns exactly one target-projected state. Main Schedule-Only and
    IntentIR evaluate their complete executable spaces. The exported concrete
    search plan already round-robin interleaves schedules across admitted MLIR
    lowering states; the runner must preserve that order exactly. The bounded
    prefixes define only the C-prior anytime table.
    """

    if search_budget <= 0:
        raise ValueError("RVV search budget must be positive")
    candidate_by_id = {
        item["candidate"]["candidate_id"]: item
        for item in entry["candidates"]
        if int(
            item["candidate"]
            .get("requested_schedule", item["candidate"]["schedule"])
            .get("thread_count", 1)
        )
        == thread_count
    }
    if not candidate_by_id:
        raise ValueError(f"RVV bundle has no candidates for {thread_count} threads")
    available = set(candidate_by_id)
    plan = entry.get("concrete_search_plan", {})

    def ordered(raw: Any) -> list[str]:
        result = []
        seen = set()
        for candidate_id in raw:
            if candidate_id in available and candidate_id not in seen:
                result.append(candidate_id)
                seen.add(candidate_id)
        return result

    c_order = ordered(plan.get("c_prior_order", tuple(candidate_by_id)))
    erase_order = ordered(plan.get("erase_c_order", tuple(candidate_by_id)))
    c_schedule_order = ordered(plan.get("schedule_only_c_prior_order", ()))
    erase_schedule_order = ordered(plan.get("schedule_only_erase_c_order", ()))
    if not c_schedule_order or not erase_schedule_order:
        raise ValueError("RVV bundle lacks explicit Schedule-Only search orders")
    if set(c_order) != available or set(erase_order) != available:
        raise ValueError("RVV C/Erase-C orders do not cover the same full universe")
    if set(c_schedule_order) != set(erase_schedule_order):
        raise ValueError("RVV C changed the concrete Schedule-Only universe")
    canonical_path = plan.get("canonical_lowering_path")
    if not isinstance(canonical_path, str) or not canonical_path:
        raise ValueError("RVV bundle lacks canonical lowering-path identity")
    for candidate_id in c_schedule_order:
        actual_path = candidate_by_id[candidate_id]["candidate"][
            "transformation"
        ].get("lowering_path_kind")
        if actual_path != canonical_path:
            raise ValueError("RVV Schedule-Only escaped the canonical lowering path")

    # Do not reconstruct this order from schedule equality.  Specialized MLIR
    # states deliberately have different schedule domains (for example the
    # widening f16->f32 dot reduction has 16 fixed lanes or scalable VL), so a
    # second interleaving here used to starve precisely those states from a
    # finite budget.
    if [item for item in c_order if item in set(c_schedule_order)] != c_schedule_order:
        raise ValueError("RVV IntentIR C order changed the E2 schedule ranking")
    if [item for item in erase_order if item in set(erase_schedule_order)] != erase_schedule_order:
        raise ValueError("RVV IntentIR Erase-C order changed the E2 schedule ranking")
    active_e2 = c_schedule_order[:search_budget]
    active_e3 = c_order[:search_budget]
    erase_e3 = erase_order[:search_budget]
    c_fixed_id = plan.get("c_fixed_candidate_id")
    if c_fixed_id not in set(c_schedule_order):
        raise ValueError("RVV C-Fixed projection is absent from Schedule-Only")
    no_retile = [
        candidate_id
        for candidate_id in c_schedule_order
        if int(candidate_by_id[candidate_id]["candidate"]["schedule"].get(
            "parallel_tile", 0
        ))
        == 0
        and int(candidate_by_id[candidate_id]["candidate"]["schedule"].get(
            "reduction_tile", 0
        ))
        == 0
    ]
    def unique(items: Any) -> list[str]:
        result = []
        seen = set()
        for candidate_id in items:
            if candidate_id in available and candidate_id not in seen:
                result.append(candidate_id)
                seen.add(candidate_id)
        return result

    # Main bars are capability envelopes over the complete executable spaces.
    # The bounded prefixes above exist only for the C-prior anytime table.
    e2_evaluation = unique(c_schedule_order)
    e3_evaluation = unique(c_order)
    if not set(e2_evaluation).issubset(set(c_order)):
        raise AssertionError("RVV E2 candidates escaped the IntentIR universe")
    # This arm isolates schedule retile: do not mix B/structural states into it.
    schedule_only_retiled = [
        candidate_id
        for candidate_id in e2_evaluation
        if int(candidate_by_id[candidate_id]["candidate"]["schedule"].get(
            "parallel_tile", 0
        ))
        != 0
        or int(candidate_by_id[candidate_id]["candidate"]["schedule"].get(
            "reduction_tile", 0
        ))
        != 0
    ]

    # A one-shot initial timing is only a coarse ranking sample.  Preserve one
    # provisional winner per concrete transformation state for the fresh,
    # alternating formal measurement; otherwise noise in one generic state can
    # remove an entire lowering path before it receives a formal sample.
    lowering_state_orders_by_key: dict[str, list[str]] = {}
    for candidate_id in e3_evaluation:
        transformation = candidate_by_id[candidate_id]["candidate"][
            "transformation"
        ]
        state_key = json.dumps(transformation, sort_keys=True, separators=(",", ":"))
        lowering_state_orders_by_key.setdefault(state_key, []).append(candidate_id)
    lowering_state_orders = list(lowering_state_orders_by_key.values())

    timed = []
    seen = set()
    timed_source = tuple(candidate_by_id)
    for candidate_id in timed_source:
        if candidate_id not in seen:
            timed.append(candidate_id)
            seen.add(candidate_id)
    return {
        "active_e3": active_e3,
        "active_e2": active_e2,
        "erase_e3": erase_e3,
        "c_order": c_order,
        "erase_order": erase_order,
        "no_retile": no_retile,
        "c_fixed": [c_fixed_id],
        "e2_evaluation": e2_evaluation,
        "e3_evaluation": e3_evaluation,
        "schedule_only_retiled": schedule_only_retiled,
        "lowering_state_orders": lowering_state_orders,
        "timed": timed,
    }


def _finalist_candidate_ids(
    measured: list[dict[str, Any]],
    arm_orders: list[list[str]],
    finalist_count: int,
) -> list[str]:
    """Choose fresh-measurement finalists without dropping an experiment arm."""

    if finalist_count <= 0:
        raise ValueError("RVV finalist count must be positive")
    measured_by_id = {
        item["candidate"]["candidate_id"]: item for item in measured
    }
    mandatory: list[str] = []
    for order in arm_orders:
        available = [
            measured_by_id[candidate_id]
            for candidate_id in order
            if candidate_id in measured_by_id
        ]
        if not available:
            continue
        candidate_id = min(
            available, key=lambda item: item["latency_us"]
        )["candidate"]["candidate_id"]
        if candidate_id not in mandatory:
            mandatory.append(candidate_id)
    ranked = [
        item["candidate"]["candidate_id"]
        for item in sorted(measured, key=lambda item: item["latency_us"])
    ]
    result: list[str] = []
    for candidate_id in (*mandatory, *ranked):
        if candidate_id not in result:
            result.append(candidate_id)
        if len(result) >= finalist_count and all(
            item in result for item in mandatory
        ):
            break
    return result


def _provisional_group_finalist_orders(
    measured: list[dict[str, Any]],
    groups: list[list[str]],
    *,
    per_group: int = 2,
) -> list[list[str]]:
    """Turn each state's top coarse samples into mandatory fresh measurements."""

    if per_group <= 0:
        raise ValueError("per-group RVV finalist count must be positive")
    measured_by_id = {
        item["candidate"]["candidate_id"]: item for item in measured
    }
    result: list[list[str]] = []
    for group in groups:
        available = sorted(
            (
                measured_by_id[candidate_id]
                for candidate_id in group
                if candidate_id in measured_by_id
            ),
            key=lambda item: item["latency_us"],
        )
        result.extend(
            [[item["candidate"]["candidate_id"]] for item in available[:per_group]]
        )
    return result


def _one_kernel(
    entry: Mapping[str, Any],
    *,
    payload_root: Path,
    regime: str,
    working_set_bytes: int,
    max_replicas: int,
    search_budget: int,
    last_level_cache_bytes: int,
    warmup: int | None,
    warmup_ms: float | None,
    iterations: int,
    batches: int,
    timed_batch_ms: float | None,
    finalist_count: int,
    thread_count: int,
    measure_c_oracle: bool,
    ggml_root: Path | None,
    ggml_lib_dir: Path | None,
) -> dict[str, Any]:
    runtime_output_specs = _resolve_runtime_output_specs(entry, payload_root)
    entry = {**entry, "output_specs": runtime_output_specs}
    logical_io = _logical_io_bytes(entry, payload_root)
    io_bytes = logical_io["boundary_io_bytes"]
    if io_bytes <= 0:
        raise ValueError("RVV benchmark has no logical boundary traffic")
    workload_cost = entry.get("workload_cost")
    certified_io_bytes = (
        workload_cost.get("boundary_io_bytes")
        if workload_cost is not None
        else None
    )
    if (
        certified_io_bytes is not None
        and int(certified_io_bytes) != io_bytes
    ):
        raise ValueError("runner logical bytes disagree with J workload-cost record")
    required_replicas = (
        max(2, math.ceil(working_set_bytes / io_bytes))
        if regime == "streaming"
        else 1
    )
    if required_replicas > max_replicas:
        raise ValueError(
            f"streaming regime needs {required_replicas} replicas, exceeding "
            f"--max-replicas={max_replicas}"
        )
    argument_sets, buffer_keepalive, output_buffer_sets = _argument_replicas(
        entry, required_replicas, payload_root
    )
    output_buffers = output_buffer_sets[0]
    descriptors = [item._obj for item in argument_sets[0]]
    eviction_buffer = (
        ctypes.create_string_buffer(working_set_bytes)
        if regime == "cold-data"
        else None
    )

    candidate_by_id = {
        item["candidate"]["candidate_id"]: item
        for item in entry["candidates"]
        if int(
            item["candidate"]
            .get("requested_schedule", item["candidate"]["schedule"])
            .get("thread_count", 1)
        )
        == thread_count
    }
    concrete_plan = entry.get("concrete_search_plan", {})
    requested_order = concrete_plan.get(
        "c_prior_order", entry.get("candidate_order", tuple(candidate_by_id))
    )
    ordered_ids = []
    seen_ids = set()
    for candidate_id in (*requested_order, *candidate_by_id):
        if candidate_id in candidate_by_id and candidate_id not in seen_ids:
            seen_ids.add(candidate_id)
            ordered_ids.append(candidate_id)
    bounded_orders = _bounded_candidate_orders(
        entry,
        search_budget,
        thread_count=thread_count,
        measure_c_oracle=measure_c_oracle,
    )
    timed_ids = set(bounded_orders["timed"])

    expected_outputs: dict[str, tuple[Any, int, int, str]] = {}
    for spec in entry["output_specs"]:
        output_name = spec["name"]
        reference_record = entry["reference_outputs"][
            entry["output_bindings"][output_name]
        ]
        payload = _decode_payload(reference_record, payload_root)
        element_count = _numel([int(item) for item in spec["shape"]])
        byte_count = element_count * _dtype_size(spec["dtype"])
        if len(payload) != byte_count:
            raise ValueError(
                f"reference payload size differs from output {output_name!r}"
            )
        expected_outputs[output_name] = (
            ctypes.create_string_buffer(max(1, len(payload))),
            element_count,
            byte_count,
            spec["dtype"],
        )

    records = []
    candidate_artifacts: dict[str, tuple[Any, Any, _ArgumentRotation]] = {}
    external_artifacts: list[dict[str, Any]] = []
    external_baseline_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="intentir-rvv-") as temporary:
        directory = Path(temporary)
        comparator = _NativeComparator(directory)
        comparator_record = comparator.record
        for raw_baseline in entry["external_baselines"]:
            baseline = _validated_external_baseline(raw_baseline)
            baseline_id = str(baseline["baseline_id"])
            try:
                library, function, compile_record = _compile_external_baseline(
                    directory,
                    baseline,
                    ggml_root=ggml_root,
                    ggml_lib_dir=ggml_lib_dir,
                )
                function.argtypes = [
                    ctypes.POINTER(type(item)) for item in descriptors
                ]
                function.restype = None
                function(*argument_sets[0])
                correctness = []
                passed = True
                for spec, buffer in zip(
                    entry["output_specs"], output_buffers, strict=True
                ):
                    output_name = spec["name"]
                    expected, element_count, byte_count, dtype = expected_outputs[
                        output_name
                    ]
                    agrees, max_error = comparator.close(
                        buffer,
                        expected,
                        byte_count=byte_count,
                        element_count=element_count,
                        dtype=dtype,
                        rule=entry["output_rules"][output_name],
                    )
                    correctness.append(
                        {
                            "output": output_name,
                            "passed": agrees,
                            "max_abs_error": max_error,
                            "comparator_protocol": NATIVE_COMPARATOR_PROTOCOL,
                        }
                    )
                    passed = passed and agrees
                baseline_record = {
                    **compile_record,
                    "status": "ready" if passed else "incorrect",
                    "correctness": correctness,
                }
                external_baseline_records.append(baseline_record)
                if passed:
                    external_artifacts.append(
                        {
                            "record": baseline_record,
                            "library": library,
                            "function": function,
                            "rotation": _ArgumentRotation(argument_sets),
                        }
                    )
            except Exception as error:
                external_baseline_records.append(
                    {
                        "protocol": EXTERNAL_BASELINE_PROTOCOL,
                        "baseline_id": baseline_id,
                        "display_name": baseline.get("display_name"),
                        "kind": baseline.get("kind"),
                        "implementation": baseline.get("implementation"),
                        "build_profile": baseline.get("build_profile"),
                        "strict_match": True,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        for candidate_id in ordered_ids:
            candidate = candidate_by_id[candidate_id]
            if candidate_id not in timed_ids:
                records.append(
                    {
                        "candidate": candidate["candidate"],
                        "status": "not_visited_due_budget",
                    }
                )
                continue
            try:
                library_path = _link_object(directory, candidate)
                library = ctypes.CDLL(str(library_path))
                symbol = candidate["candidate"]["c_interface_symbol"]
                function = getattr(library, symbol)
                function.argtypes = [
                    ctypes.POINTER(type(item)) for item in descriptors
                ]
                function.restype = None
                function(*argument_sets[0])
                correctness = []
                passed = True
                for spec, buffer in zip(
                    entry["output_specs"], output_buffers, strict=True
                ):
                    output_name = spec["name"]
                    expected, element_count, byte_count, dtype = expected_outputs[
                        output_name
                    ]
                    agrees, max_error = comparator.close(
                        buffer,
                        expected,
                        byte_count=byte_count,
                        element_count=element_count,
                        dtype=dtype,
                        rule=entry["output_rules"][output_name],
                    )
                    correctness.append(
                        {
                            "output": output_name,
                            "passed": agrees,
                            "max_abs_error": max_error,
                            "comparator_protocol": NATIVE_COMPARATOR_PROTOCOL,
                        }
                    )
                    passed = passed and agrees
                if not passed:
                    records.append(
                        {
                            "candidate": candidate["candidate"],
                            "status": "incorrect",
                            "correctness": correctness,
                        }
                    )
                    continue
                rotation = _ArgumentRotation(argument_sets)
                (
                    latency,
                    samples,
                    prep_us,
                    warmup_invocations,
                    ranking_invocations,
                ) = _measure(
                    function,
                    rotation,
                    regime=regime,
                    eviction_buffer=eviction_buffer,
                    warmup=warmup,
                    warmup_ms=warmup_ms,
                    iterations=iterations,
                    batches=1,
                    timed_batch_ms=timed_batch_ms,
                )
                candidate_artifacts[candidate_id] = (library, function, rotation)
                records.append(
                    {
                        "candidate": candidate["candidate"],
                        "status": "measured",
                        "latency_us": latency,
                        "initial_ranking_samples_us": samples,
                        "initial_data_prep_us_per_call": prep_us,
                        "initial_warmup_invocations": warmup_invocations,
                        "initial_ranking_invocations_per_batch": ranking_invocations,
                        "correctness": correctness,
                    }
                )
            except Exception as error:
                records.append(
                    {
                        "candidate": candidate["candidate"],
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    measured = [item for item in records if item["status"] == "measured"]
    if not measured:
        raise KernelFailure("no correct RVV candidate", records)
    # Exclude the sequential initial ranking sample from the formal winners.
    # Freshly remeasure each experiment arm's provisional winner plus the top
    # global states in alternating round-robin order.  This keeps expensive
    # natural workloads from spending formal batches on every losing state,
    # while no arm can disappear merely because its provisional winner was not
    # globally top-k.
    lowering_state_finalist_orders = _provisional_group_finalist_orders(
        measured,
        bounded_orders["lowering_state_orders"],
        per_group=2,
    )
    finalist_ids = _finalist_candidate_ids(
        measured,
        [
            bounded_orders["e2_evaluation"],
            bounded_orders["e3_evaluation"],
            bounded_orders["c_fixed"],
            bounded_orders["active_e3"],
            bounded_orders["erase_e3"],
            bounded_orders["no_retile"],
            bounded_orders["schedule_only_retiled"],
            *lowering_state_finalist_orders,
        ],
        finalist_count,
    )
    finalist_set = set(finalist_ids)
    finalists = [
        item
        for candidate_id in finalist_ids
        for item in measured
        if item["candidate"]["candidate_id"] == candidate_id
    ]
    fresh_samples: dict[str, list[float]] = {
        candidate_id: [] for candidate_id in finalist_ids
    }
    for round_index in range(batches):
        round_order = finalists if round_index % 2 == 0 else list(reversed(finalists))
        for item in round_order:
            candidate_id = item["candidate"]["candidate_id"]
            function = candidate_artifacts[candidate_id][1]
            rotation = candidate_artifacts[candidate_id][2]
            latency, samples, prep_us, _, invocation_counts = _measure(
                function,
                rotation,
                regime=regime,
                eviction_buffer=eviction_buffer,
                warmup=0,
                warmup_ms=None,
                iterations=iterations,
                batches=1,
                timed_batch_ms=timed_batch_ms,
            )
            assert len(samples) == 1
            fresh_samples[candidate_id].append(latency)
            item.setdefault("data_prep_samples_us", []).append(prep_us)
            item.setdefault("selection_invocations_per_batch", []).extend(
                invocation_counts
            )
    for item in finalists:
        candidate_id = item["candidate"]["candidate_id"]
        samples = fresh_samples[candidate_id]
        item["latency_us"] = float(statistics.median(samples))
        item["selection_samples_us"] = samples
        item["selection_protocol"] = "fresh-alternating-round-robin.v1"
        item["logical_boundary_gbps"] = io_bytes / item["latency_us"] / 1_000.0
    measured_by_id = {
        item["candidate"]["candidate_id"]: item for item in measured
    }

    def best(candidate_ids: list[str]) -> dict[str, Any]:
        available = [
            measured_by_id[candidate_id]
            for candidate_id in candidate_ids
            if candidate_id in measured_by_id and candidate_id in finalist_set
        ]
        if not available:
            raise KernelFailure("ablation arm has no correct RVV candidate", records)
        return min(available, key=lambda item: item["latency_us"])

    def optional_best(candidate_ids: list[str]) -> dict[str, Any] | None:
        available = [
            measured_by_id[candidate_id]
            for candidate_id in candidate_ids
            if candidate_id in measured_by_id and candidate_id in finalist_set
        ]
        return min(available, key=lambda item: item["latency_us"]) if available else None

    e2 = best(bounded_orders["e2_evaluation"])
    e3 = best(bounded_orders["e3_evaluation"])
    c_fixed = best(bounded_orders["c_fixed"])
    no_retile = optional_best(bounded_orders["no_retile"])
    schedule_only_retiled = optional_best(
        bounded_orders["schedule_only_retiled"]
    )
    erase_budget_ids = bounded_orders["erase_e3"]
    c_prior_budget_ids = bounded_orders["active_e3"]
    erase_budget_winner = best(erase_budget_ids)
    c_prior_budget_winner = best(c_prior_budget_ids)
    c_fixed_id = bounded_orders["c_fixed"][0]
    if c_fixed_id != c_fixed["candidate"]["candidate_id"]:
        raise KernelFailure("C-Fixed candidate selection changed", records)

    e3_function = candidate_artifacts[e3["candidate"]["candidate_id"]][1]
    e3_rotation = candidate_artifacts[e3["candidate"]["candidate_id"]][2]
    for artifact in external_artifacts:
        baseline_samples: list[float] = []
        full_pair_samples: list[float] = []
        for round_index in range(batches):
            pair = (
                (
                    artifact["function"],
                    artifact["rotation"],
                    baseline_samples,
                ),
                (e3_function, e3_rotation, full_pair_samples),
            )
            if round_index % 2:
                pair = tuple(reversed(pair))
            for function, rotation, destination in pair:
                latency_us, _, _, _, _ = _measure(
                    function,
                    rotation,
                    regime=regime,
                    eviction_buffer=eviction_buffer,
                    warmup=0,
                    warmup_ms=None,
                    iterations=iterations,
                    batches=1,
                    timed_batch_ms=timed_batch_ms,
                )
                destination.append(latency_us)
        baseline_latency = float(statistics.median(baseline_samples))
        full_pair_latency = float(statistics.median(full_pair_samples))
        artifact["record"].update(
            {
                "status": "measured",
                "latency_us": baseline_latency,
                "selection_samples_us": baseline_samples,
                "full_paired_samples_us": full_pair_samples,
                "full_paired_us": full_pair_latency,
                "intentir_speedup": baseline_latency / full_pair_latency,
                "comparison_protocol": "paired-alternating.v1",
            }
        )
    measured_external = [
        record
        for record in external_baseline_records
        if record.get("status") == "measured"
    ]
    best_external = _select_best_native(measured_external)
    llvm_rvv_c = next(
        (
            record
            for record in measured_external
            if record.get("baseline_id") == "llvm_rvv_c"
        ),
        None,
    )

    oracle_complete = not any(
        item["status"] == "not_visited_due_budget" for item in records
    )
    oracle_best = optional_best(bounded_orders["c_order"]) if oracle_complete else None

    def candidates_to_95pct(order: list[str]) -> int | None:
        if oracle_best is None:
            return None
        threshold_us = float(oracle_best["latency_us"]) / 0.95
        best_so_far = float("inf")
        for count, candidate_id in enumerate(order, start=1):
            measured_candidate = measured_by_id.get(candidate_id)
            if measured_candidate is not None:
                best_so_far = min(
                    best_so_far, float(measured_candidate["latency_us"])
                )
            if best_so_far <= threshold_us:
                return count
        return None

    c_candidates_to_95pct = candidates_to_95pct(bounded_orders["c_order"])
    erase_candidates_to_95pct = candidates_to_95pct(
        bounded_orders["erase_order"]
    )
    achieved_working_set_bytes = required_replicas * io_bytes
    floating_point_ops = (
        int(workload_cost["floating_point_ops"])
        if workload_cost is not None
        else 0
    )
    return {
        "kernel": entry["kernel"],
        "case_id": entry["case_id"],
        "workload_profile": entry.get("workload_profile"),
        "workload_valuation": entry.get("workload_valuation"),
        "baseline_policy": entry.get("baseline_policy"),
        "workload_signature": {
            "inputs": entry.get("input_specs"),
            "outputs": entry["output_specs"],
            "extents": entry["extents"],
        },
        "contract_view": entry["contract_view"],
        "extents": entry["extents"],
        "candidates": records,
        "winner": e3["candidate"],
        "external_baselines": external_baseline_records,
        "best_native_baseline": best_external,
        "native_baseline": llvm_rvv_c,
        "performance": {
            "c_fixed_us": c_fixed["latency_us"],
            "schedule_only_us": e2["latency_us"],
            "full_us": e3["latency_us"],
            "g_schedule": c_fixed["latency_us"] / e2["latency_us"],
            "g_full": e2["latency_us"] / e3["latency_us"],
            "llvm_rvv_c_us": (
                float(llvm_rvv_c["latency_us"]) if llvm_rvv_c is not None else None
            ),
            "best_native_us": (
                float(best_external["latency_us"])
                if best_external is not None
                else None
            ),
            "best_native_baseline_id": (
                best_external["baseline_id"] if best_external is not None else None
            ),
            "full_paired_with_best_native_us": (
                float(best_external["full_paired_us"])
                if best_external is not None
                else None
            ),
            "g_best_native": (
                float(best_external["intentir_speedup"])
                if best_external is not None
                else None
            ),
            "native_us": (
                float(llvm_rvv_c["latency_us"]) if llvm_rvv_c is not None else None
            ),
            "full_paired_with_native_us": (
                float(llvm_rvv_c["full_paired_us"])
                if llvm_rvv_c is not None
                else None
            ),
            "g_native": (
                float(llvm_rvv_c["intentir_speedup"])
                if llvm_rvv_c is not None
                else None
            ),
            "c_fixed_candidate_id": c_fixed_id,
            "schedule_only_winner_candidate_id": e2["candidate"]["candidate_id"],
            "full_winner_candidate_id": e3["candidate"]["candidate_id"],
        },
        "c_prior_efficiency": {
            "candidate_budget": search_budget,
            "budget_unit": "admitted_candidate",
            "c_prior_best_us": c_prior_budget_winner["latency_us"],
            "erase_c_best_us": erase_budget_winner["latency_us"],
            "best_at_budget_c_over_erase_c_speedup": (
                erase_budget_winner["latency_us"]
                / c_prior_budget_winner["latency_us"]
            ),
            "c_prior_candidates_to_95pct_of_final_best": c_candidates_to_95pct,
            "erase_c_candidates_to_95pct_of_final_best": (
                erase_candidates_to_95pct
            ),
            "oracle_complete": oracle_complete,
            "oracle_best_us": (
                oracle_best["latency_us"] if oracle_best is not None else None
            ),
            "unavailable_reason": (
                None
                if oracle_complete
                else "complete_latency_map_unavailable"
            ),
            "c_prior_budget_winner_candidate_id": c_prior_budget_winner[
                "candidate"
            ]["candidate_id"],
            "erase_c_budget_winner_candidate_id": erase_budget_winner[
                "candidate"
            ]["candidate_id"],
        },
        "supplementary_diagnostics": {
            "formal_role": "not_a_main_experiment_arm",
            "b_transformation_wins": bool(
                e3["candidate"]["transformation"]["b_transformations"]
            ),
            "no_retile_schedule_us": (
                no_retile["latency_us"] if no_retile is not None else None
            ),
            "retiled_schedule_us": (
                schedule_only_retiled["latency_us"]
                if schedule_only_retiled is not None
                else None
            ),
            "retiled_over_no_retile_speedup": (
                no_retile["latency_us"] / schedule_only_retiled["latency_us"]
                if no_retile is not None and schedule_only_retiled is not None
                else None
            ),
            "retiled_over_c_fixed_speedup": (
                c_fixed["latency_us"] / schedule_only_retiled["latency_us"]
                if schedule_only_retiled is not None
                else None
            ),
            "intentir_over_c_fixed_speedup": (
                c_fixed["latency_us"] / e3["latency_us"]
            ),
            "no_retile_winner_candidate_id": (
                no_retile["candidate"]["candidate_id"]
                if no_retile is not None
                else None
            ),
            "retiled_winner_candidate_id": (
                schedule_only_retiled["candidate"]["candidate_id"]
                if schedule_only_retiled is not None
                else None
            ),
        },
        "search_execution": {
            "candidate_budget": search_budget,
            "budget_unit": "admitted_candidate",
            "candidate_correctness_pass_count": len(
                [
                    item
                    for item in records
                    if item["status"] == "measured"
                ]
            ),
            "candidate_correctness_fail_count": len(
                [item for item in records if item["status"] == "incorrect"]
            ),
            "candidate_execution_failure_count": len(
                [item for item in records if item["status"] == "failed"]
            ),
            "timed_candidate_count": len(measured),
            "not_visited_candidate_count": len(
                [
                    item
                    for item in records
                    if item["status"] == "not_visited_due_budget"
                ]
            ),
            "active_e3_budget_candidate_ids": bounded_orders["active_e3"],
            "active_e2_budget_candidate_ids": bounded_orders["active_e2"],
            "erase_c_e3_budget_candidate_ids": bounded_orders["erase_e3"],
            "e3_evaluation_candidate_ids": bounded_orders["e3_evaluation"],
            "e2_evaluation_candidate_ids": bounded_orders["e2_evaluation"],
            "retiled_schedule_candidate_ids": bounded_orders[
                "schedule_only_retiled"
            ],
            "c_fixed_candidate_ids": bounded_orders["c_fixed"],
            "timed_candidate_ids": bounded_orders["timed"],
            "fresh_remeasured_lowering_state_count": len(
                bounded_orders["lowering_state_orders"]
            ),
            "fresh_remeasured_candidates_per_lowering_state": 2,
            "correctness_comparator": comparator_record,
        },
        "measurement_analysis": {
            "logical_io": {
                **logical_io,
                "e2_logical_boundary_gbps": (
                    io_bytes / e2["latency_us"] / 1_000.0
                ),
                "e3_logical_boundary_gbps": (
                    io_bytes / e3["latency_us"] / 1_000.0
                ),
                "traffic_definition": (
                    "sum of certified boundary input and output payload bytes; "
                    "not measured DRAM transactions"
                ),
            },
            "roofline_point": {
                "j_workload_cost": workload_cost,
                "arithmetic_intensity_flop_per_byte": (
                    floating_point_ops / io_bytes
                ),
                "e2_achieved_gflops": (
                    floating_point_ops / e2["latency_us"] / 1_000.0
                ),
                "e3_achieved_gflops": (
                    floating_point_ops / e3["latency_us"] / 1_000.0
                ),
                "hardware_counter_traffic": None,
                "interpretation": "J-derived logical/minimum-traffic point",
            },
            "cache_regime": {
                "name": regime,
                "buffer_policy": (
                    "rotating-certified-invocation-replicas"
                    if regime == "streaming"
                    else "eviction-before-each-single-shot"
                    if regime == "cold-data"
                    else "persistent-single-address"
                ),
                "replicas": required_replicas,
                "requested_working_set_bytes": working_set_bytes,
                "achieved_working_set_bytes": achieved_working_set_bytes,
                "eviction_bytes": (
                    working_set_bytes if regime == "cold-data" else 0
                ),
                "data_prep_in_timed_region": False,
                "streaming_target_achieved": (
                    achieved_working_set_bytes >= working_set_bytes
                    if regime == "streaming"
                    else None
                ),
                "last_level_cache_bytes": last_level_cache_bytes,
                "working_set_exceeds_last_level_cache": (
                    achieved_working_set_bytes > last_level_cache_bytes
                    if regime == "streaming" and last_level_cache_bytes
                    else working_set_bytes > last_level_cache_bytes
                    if regime == "cold-data" and last_level_cache_bytes
                    else None
                ),
            },
            "timing": "pinned-process perf_counter_ns via ctypes C ABI",
            "comparison_protocol": "fresh-alternating-round-robin.v1",
            "warmup": warmup if warmup is not None else 0,
            "warmup_policy": (
                "time_budget" if warmup_ms is not None else "fixed_iterations"
            ),
            "warmup_time_budget_ms": warmup_ms,
            "warmup_actual_invocations": {
                "minimum": min(
                    item["initial_warmup_invocations"] for item in measured
                ),
                "median": statistics.median(
                    item["initial_warmup_invocations"] for item in measured
                ),
                "maximum": max(
                    item["initial_warmup_invocations"] for item in measured
                ),
            },
            "iterations": iterations,
            "iterations_per_batch_cap": iterations,
            "target_timed_ms_per_batch": timed_batch_ms,
            "batches": batches,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--native-baseline-sidecar",
        type=Path,
        help="legacy single Native RVV source map",
    )
    parser.add_argument(
        "--external-baseline-sidecar",
        type=Path,
        help="independent LLVM-RVV, Expert RVV, and library baseline source map",
    )
    parser.add_argument("--ggml-root", type=Path)
    parser.add_argument("--ggml-lib-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", nargs="*")
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--threads", type=int, choices=(1,), default=1)
    parser.add_argument(
        "--regime",
        choices=("warm", "streaming", "cold-data"),
        default="warm",
    )
    parser.add_argument("--working-set-mib", type=int, default=128)
    parser.add_argument("--max-replicas", type=int, default=262144)
    parser.add_argument("--search-budget", type=int, default=12)
    parser.add_argument(
        "--measure-c-oracle",
        action="store_true",
        help=(
            "compatibility flag; complete concrete-space execution is mandatory "
            "for the main three bars and candidates-to-95%% is always available"
        ),
    )
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument(
        "--warmup",
        type=int,
        help="compatibility mode: fixed warmup invocation count",
    )
    warmup.add_argument(
        "--warmup-ms",
        type=float,
        help="untimed warmup time budget per candidate (default: 50 ms)",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--cold-iterations", type=int, default=16)
    parser.add_argument(
        "--cold-batch-ms",
        type=float,
        default=200.0,
        help=(
            "target timed kernel duration per cold-data batch; "
            "--cold-iterations remains the invocation cap"
        ),
    )
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--finalist-count", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    warmup_iterations = args.warmup
    warmup_ms = args.warmup_ms
    if warmup_iterations is None and warmup_ms is None:
        warmup_ms = 50.0
    if (
        args.native_baseline_sidecar is not None
        and args.external_baseline_sidecar is not None
    ):
        raise ValueError("use only one RVV external baseline sidecar option")
    if (args.ggml_root is None) != (args.ggml_lib_dir is None):
        raise ValueError("--ggml-root and --ggml-lib-dir must be provided together")
    if warmup_iterations is not None and warmup_iterations < 0:
        raise ValueError("fixed warmup count must be non-negative")
    if warmup_ms is not None and warmup_ms < 0:
        raise ValueError("warmup time budget must be non-negative")
    if args.cold_batch_ms <= 0:
        raise ValueError("cold-data batch time budget must be positive")
    if min(
        args.iterations,
        args.cold_iterations,
        args.batches,
        args.working_set_mib,
        args.max_replicas,
        args.search_budget,
        args.finalist_count,
    ) <= 0:
        raise ValueError("all measurement counts and budgets must be positive")
    affinity = set(range(args.cpu, args.cpu + args.threads))
    available_cpus = os.sched_getaffinity(0)
    if not affinity.issubset(available_cpus):
        raise ValueError(
            f"requested CPU range {sorted(affinity)} is outside available affinity"
        )
    os.sched_setaffinity(0, affinity)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_PROC_BIND"] = "CLOSE"
    cache_record = _cpu_cache_record()
    working_set_bytes = args.working_set_mib * 1024 * 1024
    last_level_cache_bytes = cache_record["last_level_data_cache_bytes"]
    if (
        args.regime != "warm"
        and last_level_cache_bytes
        and working_set_bytes <= last_level_cache_bytes
    ):
        raise ValueError(
            f"{args.regime} working set ({working_set_bytes} bytes) must exceed "
            f"detected last-level cache ({last_level_cache_bytes} bytes)"
        )
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("protocol") != BUNDLE_PROTOCOL:
        raise ValueError("unsupported RVV benchmark bundle")
    selected = set(args.kernels or [])
    entries = [
        item
        for item in bundle["kernels"]
        if not selected or item["kernel"] in selected
    ]
    sidecar_path = args.external_baseline_sidecar or args.native_baseline_sidecar
    sidecar = (
        _load_external_baseline_sidecar(sidecar_path)
        if sidecar_path is not None
        else {}
    )
    entries = _attach_external_baselines(entries, sidecar)
    effective_baseline_policy = (
        "external_rvv_and_refj" if entries else bundle.get("baseline_policy")
    )
    started = time.time_ns()
    kernels = []
    failures = []
    measured_iterations = (
        args.cold_iterations if args.regime == "cold-data" else args.iterations
    )
    for entry in entries:
        try:
            record = _one_kernel(
                entry,
                payload_root=args.bundle.resolve().parent,
                regime=args.regime,
                working_set_bytes=working_set_bytes,
                max_replicas=args.max_replicas,
                search_budget=args.search_budget,
                last_level_cache_bytes=last_level_cache_bytes,
                warmup=warmup_iterations,
                warmup_ms=warmup_ms,
                iterations=measured_iterations,
                batches=args.batches,
                timed_batch_ms=(
                    args.cold_batch_ms if args.regime == "cold-data" else None
                ),
                finalist_count=args.finalist_count,
                thread_count=args.threads,
                measure_c_oracle=args.measure_c_oracle,
                ggml_root=args.ggml_root,
                ggml_lib_dir=args.ggml_lib_dir,
            )
            kernels.append(record)
            performance = record["performance"]
            best_native_text = (
                f"{performance['g_best_native']:.3f}x"
                if performance["g_best_native"] is not None
                else "N/A"
            )
            print(
                f"{entry['kernel']}: CF={performance['c_fixed_us']:.4f} us "
                f"Schedule={performance['schedule_only_us']:.4f} us "
                f"Full={performance['full_us']:.4f} us "
                f"G_schedule={performance['g_schedule']:.3f}x "
                f"G_full={performance['g_full']:.3f}x "
                f"G_best_native={best_native_text} C="
                f"{record['c_prior_efficiency']['best_at_budget_c_over_erase_c_speedup']:.3f}x",
                flush=True,
            )
        except Exception as error:
            failure = {
                "kernel": entry["kernel"],
                "error": f"{type(error).__name__}: {error}",
            }
            if isinstance(error, KernelFailure):
                failure["candidate_diagnostics"] = error.diagnostics
            failures.append(failure)
            print(
                f"{entry['kernel']}: FAILED: {type(error).__name__}: {error}",
                flush=True,
            )
    schedule_speedups = [item["performance"]["g_schedule"] for item in kernels]
    full_speedups = [item["performance"]["g_full"] for item in kernels]
    llvm_rvv_c_speedups = [
        item["performance"]["g_native"]
        for item in kernels
        if item["performance"]["g_native"] is not None
    ]
    best_native_speedups = [
        item["performance"]["g_best_native"]
        for item in kernels
        if item["performance"]["g_best_native"] is not None
    ]
    c_speedups = [
        item["c_prior_efficiency"]["best_at_budget_c_over_erase_c_speedup"]
        for item in kernels
    ]
    c_candidates_to_95pct = [
        item["c_prior_efficiency"][
            "c_prior_candidates_to_95pct_of_final_best"
        ]
        for item in kernels
        if item["c_prior_efficiency"][
            "c_prior_candidates_to_95pct_of_final_best"
        ]
        is not None
    ]
    erase_c_candidates_to_95pct = [
        item["c_prior_efficiency"][
            "erase_c_candidates_to_95pct_of_final_best"
        ]
        for item in kernels
        if item["c_prior_efficiency"][
            "erase_c_candidates_to_95pct_of_final_best"
        ]
        is not None
    ]

    def geomean(values: list[float]) -> float | None:
        return (
            math.exp(sum(math.log(item) for item in values) / len(values))
            if values
            else None
        )

    report = {
        "protocol": REPORT_PROTOCOL,
        "bundle_protocol": bundle["protocol"],
        "target": bundle["target"],
        "workload_profile": bundle.get("workload_profile", "closed-v1"),
        "baseline_policy": effective_baseline_policy,
        "experiment_spec_protocol": bundle.get("experiment_spec_protocol"),
        "measurement_spec_protocol": bundle.get("measurement_spec_protocol"),
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "device": {
            "host": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_count": args.threads,
            "parallel_runtime": (
                "OpenMP" if args.threads > 1 else "serial MLIR loops"
            ),
            "vlen_bits": 128,
            "cache": cache_record,
        },
        "measurement": {
            "data_regime": args.regime,
            "effective_timing": (
                "monotonic-clock-single-shot"
                if args.regime == "cold-data"
                else "monotonic-clock-batch"
            ),
            "timer": "perf_counter_ns",
            "correctness_policy": bundle.get("correctness_policy"),
            "working_set_bytes": working_set_bytes,
            "warmup_policy": (
                "time_budget" if warmup_ms is not None else "fixed_iterations"
            ),
            "warmup_time_budget_ms": warmup_ms,
            "warmup_fixed_iterations": warmup_iterations,
            "warmup_iterations": (
                warmup_iterations if warmup_iterations is not None else 0
            ),
            "iterations_per_batch": measured_iterations,
            "iterations_per_batch_cap": measured_iterations,
            "target_timed_ms_per_batch": (
                args.cold_batch_ms if args.regime == "cold-data" else None
            ),
            "measurement_batches": args.batches,
            "detected_last_level_data_cache_bytes": last_level_cache_bytes,
            "cache_eviction_in_timed_region": False,
            "regime": args.regime,
            "working_set_mib": args.working_set_mib,
            "warmup": warmup_iterations if warmup_iterations is not None else 0,
            "iterations": measured_iterations,
            "batches": args.batches,
            "finalist_count": args.finalist_count,
            "search_budget": args.search_budget,
            "measure_c_oracle": args.measure_c_oracle,
            "thread_count": args.threads,
        },
        "summary": {
            "requested": len(entries),
            "successful": len(kernels),
            "failed": len(failures),
            "geomean_g_schedule": geomean(schedule_speedups),
            "geomean_g_full": geomean(full_speedups),
            "geomean_g_native": geomean(llvm_rvv_c_speedups),
            "geomean_g_llvm_rvv_c": geomean(llvm_rvv_c_speedups),
            "geomean_g_best_native": geomean(best_native_speedups),
            "best_native_applicable_kernel_count": len(best_native_speedups),
        },
        "c_prior_efficiency_summary": {
            "geomean_best_at_budget_c_over_erase_c_speedup": geomean(
                c_speedups
            ),
            "oracle_complete_kernel_count": len(c_candidates_to_95pct),
            "median_c_prior_candidates_to_95pct_of_final_best": (
                statistics.median(c_candidates_to_95pct)
                if c_candidates_to_95pct
                else None
            ),
            "median_erase_c_candidates_to_95pct_of_final_best": (
                statistics.median(erase_c_candidates_to_95pct)
                if erase_c_candidates_to_95pct
                else None
            ),
        },
        "kernels": kernels,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
