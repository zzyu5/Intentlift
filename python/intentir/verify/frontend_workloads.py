"""Prepared SCRC workloads for the external TileLang and CUDA Dataset.

This module is the source-aware Harness boundary for non-Triton frontends.  It
projects a real Dataset source into authoring material, preserves the original
frontend object for Gate1/Gate2, and binds one concrete finite invocation.  It
does not translate either frontend into J; J authorship stays outside the
Harness and Replay remains source-blind.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import importlib.util
import json
import operator
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    SCRCCompilerView,
    anonymize_source,
)
from intentir.verify.frontend_dataset import (
    FrontendDatasetKernel,
    load_frontend_dataset,
    resolve_cuda_include_dirs,
)
from intentir.verify.gate1.cuda_compiler import (
    CudaCompileCase,
    CudaKernelSource,
    FrontendParameter,
)
from intentir.verify.gate1.model import (
    Frontend,
    Gate1Artifact,
    ParameterKind,
    ParameterMode,
)
from intentir.verify.gate1.profile import observer_profile_for
from intentir.verify.gate1.tilelang_compiler import (
    TileLangCompileCase,
    TileLangKernelSource,
)
from intentir.ir.value_dimension import resolve_runtime_output_shapes
from intentir.ir.canonical_j import j_elaboration
from intentir.ir.canonical_construction import (
    CanonicalConstruction,
    CanonicalHandleAllocator,
    SourceFacts,
)
from intentir.verify.gate2.native import (
    NativeArgument,
    NativeArgumentRole,
    NativeExecutionSpec,
    NativeTensorMapSpec,
)
from intentir.verify.gate2.model import program_identity_digests
from intentir.verify.gate2.orchestrate import Gate2Case, Gate2Suite
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY, gate2_policy_for
from intentir.verify.gate2.inputs import materialize_ragged_input_values
from intentir.verify.gate2.scalar import ScalarLoweringSpec
from intentir.verify.scrc_runner import (
    HarnessBindingInstance,
    SCRCRunRequest,
    SourceProgram,
)
from intentir.verify.workload_adapters import (
    GeneratedTargetSourceArgument,
    GeneratedTargetSourceManifest,
    build_generated_source_manifest,
    reingress_prepared_workload,
)


FRONTEND_DATASET_HARNESS_PROTOCOL = "intentir.scrc.frontend-dataset-harness.v5"
FRONTEND_DATASET_SOURCE_PROTOCOL = "intentir.scrc.frontend-source-projection.v1"
FRONTEND_DATASET_PROJECTION_PROTOCOL = (
    "intentir.scrc.frontend-dataset-harness-projection.v5"
)
_DEFAULT_DATASET_DIMENSION = 4
_DEFAULT_DATASET_BLOCK_1D = (16, 1, 1)
_DEFAULT_DATASET_BLOCK_2D = (16, 16, 1)


class FrontendDatasetCaseUnavailableError(ValueError):
    """The catalog entry is valid but needs an unsupported explicit sidecar."""


@dataclass(frozen=True)
class CuTileConsumerResult:
    """Bounded result from the optional external cuTile lowering consumer."""

    ok: bool
    failure_class: str
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    diagnostic: str = ""
    source: str | None = None
    realized_ir: str | None = None
    capability_failure: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failure_class": self.failure_class,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "diagnostic": self.diagnostic,
            "source": self.source,
            "realized_ir": self.realized_ir,
            "capability_failure": self.capability_failure,
        }


def _bounded_diagnostic(value: str, *, limit: int = 8192) -> str:
    value = str(value or "")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _resolve_cutile_consumer(executable: str | Path | None) -> Path | None:
    if executable is not None:
        path = Path(executable).expanduser().resolve()
        return path if path.is_file() else None
    repo_root = Path(__file__).resolve().parents[3]
    configured = os.environ.get("INTENTIR_CUTILE_COMPILER", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        Path(item)
        for item in (
            shutil.which("intent-compile"),
            repo_root / "build" / "tools" / "intent-compile" / "intent-compile",
            repo_root / "build" / "bin" / "intent-compile",
            repo_root / "tools" / "intent-compile" / "intent-compile",
            repo_root.parent
            / "intent-dsl"
            / "build"
            / "tools"
            / "intent-compile"
            / "intent-compile",
            repo_root.parent / "intent-dsl" / "build" / "bin" / "intent-compile",
            repo_root.parent
            / "intent-dsl"
            / "tools"
            / "intent-compile"
            / "intent-compile",
        )
        if item
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def run_cutile_consumer(
    kernel_mlir: str | Path,
    *,
    executable: str | Path | None = None,
    device: int = 0,
    compute_units: int,
    shared_memory_per_unit: int,
    registers_per_unit: int,
    matrix_units: bool,
    dynamic_vector_width: bool,
    timeout: float = 120.0,
) -> CuTileConsumerResult:
    """Run the external cuTile consumer with fail-closed diagnostics.

    All generated files live in a temporary directory and are returned only as
    in-memory text.  A zero exit status is insufficient: any ``error:`` line,
    missing output, or empty output is a failure.  Missing consumer binaries are
    reported separately as capability failures.
    """

    consumer = _resolve_cutile_consumer(executable)
    if consumer is None:
        return CuTileConsumerResult(
            False,
            "capability_unavailable",
            None,
            diagnostic="cuTile intent-compile consumer is not available",
            capability_failure=True,
        )
    if timeout <= 0 or not np.isfinite(timeout):
        raise ValueError("cuTile consumer timeout must be positive and finite")
    capability_values = {
        "device": device,
        "compute-units": compute_units,
        "shared-memory-per-unit": shared_memory_per_unit,
        "registers-per-unit": registers_per_unit,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in capability_values.values()
    ):
        raise ValueError("cuTile consumer capability values must be non-negative integers")
    if not isinstance(kernel_mlir, (str, Path)):
        raise TypeError("cuTile kernel MLIR must be a path or text")
    with tempfile.TemporaryDirectory(prefix="intentir-cutile-consumer-") as raw:
        root = Path(raw)
        input_path = (
            Path(kernel_mlir).expanduser().resolve()
            if isinstance(kernel_mlir, Path)
            else Path(kernel_mlir)
        )
        if not input_path.is_file():
            input_path = root / "kernel.mlir"
            input_path.write_text(str(kernel_mlir), encoding="utf-8")
        ir_path = root / "realized.mlir"
        source_path = root / "generated.py"
        command = [
            str(consumer),
            str(input_path),
            "--target=cutile",
            f"--device={device}",
            f"--compute-units={compute_units}",
            f"--shared-memory-per-unit={shared_memory_per_unit}",
            f"--registers-per-unit={registers_per_unit}",
            f"--matrix-units={'true' if matrix_units else 'false'}",
            f"--dynamic-vector-width={'true' if dynamic_vector_width else 'false'}",
            f"--ir-output={ir_path}",
            f"--source-output={source_path}",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _bounded_diagnostic(str(exc.stdout or ""))
            stderr = _bounded_diagnostic(str(exc.stderr or ""))
            return CuTileConsumerResult(
                False,
                "consumer_timeout",
                None,
                stdout=stdout,
                stderr=stderr,
                diagnostic=f"cuTile consumer timed out after {timeout:.1f}s",
            )
        except OSError as exc:
            return CuTileConsumerResult(
                False,
                "capability_unavailable",
                None,
                diagnostic=f"cuTile consumer could not start: {exc}",
                capability_failure=True,
            )

        stdout = _bounded_diagnostic(completed.stdout)
        stderr = _bounded_diagnostic(completed.stderr)
        diagnostic_stream = f"{completed.stdout}\n{completed.stderr}"
        if "error:" in diagnostic_stream.casefold():
            return CuTileConsumerResult(
                False,
                "consumer_diagnostic",
                completed.returncode,
                stdout=stdout,
                stderr=stderr,
                diagnostic="cuTile consumer emitted an error diagnostic",
            )
        if completed.returncode != 0:
            return CuTileConsumerResult(
                False,
                "consumer_exit",
                completed.returncode,
                stdout=stdout,
                stderr=stderr,
                diagnostic=(
                    f"cuTile consumer exited with status {completed.returncode}"
                ),
            )
        if not ir_path.is_file() or ir_path.stat().st_size == 0:
            return CuTileConsumerResult(
                False,
                "realized_ir_missing_or_empty",
                completed.returncode,
                stdout=stdout,
                stderr=stderr,
                diagnostic="cuTile consumer produced no non-empty realized MLIR",
            )
        if not source_path.is_file() or source_path.stat().st_size == 0:
            return CuTileConsumerResult(
                False,
                "source_missing_or_empty",
                completed.returncode,
                stdout=stdout,
                stderr=stderr,
                diagnostic="cuTile consumer produced no non-empty source output",
            )
        try:
            realized_ir = ir_path.read_text(encoding="utf-8")
            source = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return CuTileConsumerResult(
                False,
                "output_unreadable",
                completed.returncode,
                stdout=stdout,
                stderr=stderr,
                diagnostic=f"cuTile consumer output is unreadable: {exc}",
            )
        return CuTileConsumerResult(
            True,
            "ok",
            completed.returncode,
            stdout=stdout,
            stderr=stderr,
            source=source,
            realized_ir=realized_ir,
        )


@dataclass(frozen=True)
class DatasetTensorMapDescriptor:
    """Pinned logical-tensor to physical-CUtensorMap binding."""

    parameter: str
    semantic_name: str
    logical_dtype: str
    shape_contract: tuple[int | str, ...]
    box_dim: tuple[int, int]
    swizzle: str = "128b"

    def __post_init__(self) -> None:
        if not self.parameter.isidentifier() or not self.semantic_name.isidentifier():
            raise ValueError("TensorMap sidecar names must be identifiers")
        if self.logical_dtype not in {"f16", "bf16", "f32"}:
            raise ValueError("TensorMap sidecar dtype is outside the native profile")
        shape = tuple(self.shape_contract)
        if len(shape) != 2 or any(
            isinstance(value, bool)
            or not (
                (isinstance(value, int) and value > 0)
                or (isinstance(value, str) and value.isidentifier())
            )
            for value in shape
        ):
            raise ValueError("TensorMap sidecar requires one positive rank-2 shape")
        box = tuple(int(value) for value in self.box_dim)
        if len(box) != 2 or any(value <= 0 for value in box):
            raise ValueError("TensorMap sidecar box_dim must be positive rank-2")
        if self.swizzle not in {"none", "32b", "64b", "128b"}:
            raise ValueError("TensorMap sidecar swizzle is unsupported")
        object.__setattr__(self, "shape_contract", shape)
        object.__setattr__(self, "box_dim", box)

    def to_json(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "semantic_name": self.semantic_name,
            "logical_dtype": self.logical_dtype,
            "shape": list(self.shape_contract),
            "box_dim": list(self.box_dim),
            "swizzle": self.swizzle,
        }


@dataclass(frozen=True)
class FrontendDatasetHarness:
    """One explicit finite binding for a Dataset kernel."""

    case_id: str
    frontend: str
    dimensions: tuple[tuple[str, int], ...]
    scalar_values: tuple[tuple[str, bool | int | float], ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    grid_contract: tuple[str, str, str]
    seed: int
    factory_kwargs: tuple[tuple[str, bool | int | float | str], ...] = ()
    cuda_arch: str = "sm_120"
    source_sha256: str = ""
    descriptor_sha256: str = ""
    tensor_map_descriptors: tuple[DatasetTensorMapDescriptor, ...] = ()
    tensor_map_sidecar_sha256: str = ""
    physical_block: tuple[int, int, int] | None = None
    dynamic_shared_memory_bytes: int = 0
    gate2_policy: str = DEFAULT_GATE2_POLICY.policy_id
    protocol: str = FRONTEND_DATASET_HARNESS_PROTOCOL

    def __post_init__(self) -> None:
        if self.frontend not in {"cuda", "tilelang"}:
            raise ValueError("Dataset Harness frontend must be cuda or tilelang")
        if not self.case_id.startswith(f"{self.frontend}:"):
            raise ValueError("Dataset Harness case_id and frontend differ")
        dimensions = tuple((str(name), int(value)) for name, value in self.dimensions)
        if (
            len({name for name, _value in dimensions}) != len(dimensions)
            or any(not name.isidentifier() or value <= 0 for name, value in dimensions)
        ):
            raise ValueError("Dataset dimensions require unique names and positive values")
        scalar_values = tuple(self.scalar_values)
        if (
            len({name for name, _value in scalar_values}) != len(scalar_values)
            or any(
                not isinstance(name, str) or not name.isidentifier()
                for name, _value in scalar_values
            )
        ):
            raise ValueError("Dataset scalar bindings must be unique")
        for name, value in scalar_values:
            if not isinstance(value, (bool, int, float)):
                raise TypeError(f"Dataset scalar {name!r} is not a numeric scalar")
            if isinstance(value, float) and (
                value != value or value in {float("inf"), float("-inf")}
            ):
                raise ValueError(f"Dataset scalar {name!r} is non-finite")
        for label, values in (("grid", self.grid), ("block", self.block)):
            normalized = tuple(int(value) for value in values)
            if len(normalized) != 3 or any(value <= 0 for value in normalized):
                raise ValueError(f"Dataset {label} must contain three positive values")
            object.__setattr__(self, label, normalized)
        contract = tuple(str(value) for value in self.grid_contract)
        if len(contract) != 3 or any(not value for value in contract):
            raise ValueError("Dataset grid contract must describe all three axes")
        kwargs = tuple(self.factory_kwargs)
        if (
            len({name for name, _value in kwargs}) != len(kwargs)
            or any(
                not isinstance(name, str) or not name.isidentifier()
                for name, _value in kwargs
            )
        ):
            raise ValueError("TileLang factory kwargs must be unique")
        for name, value in kwargs:
            if not isinstance(value, (bool, int, float, str)):
                raise TypeError(f"TileLang factory kwarg {name!r} is not a JSON scalar")
            if isinstance(value, float) and (
                value != value or value in {float("inf"), float("-inf")}
            ):
                raise ValueError(f"TileLang factory kwarg {name!r} is non-finite")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Dataset Harness seed must be non-negative")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.source_sha256):
            raise ValueError("Dataset Harness requires a pinned source SHA-256")
        if self.descriptor_sha256 and not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.descriptor_sha256
        ):
            raise ValueError("Dataset Harness descriptor SHA-256 is malformed")
        descriptors = tuple(
            item
            if isinstance(item, DatasetTensorMapDescriptor)
            else DatasetTensorMapDescriptor(**item)
            for item in self.tensor_map_descriptors
        )
        if len({item.parameter for item in descriptors}) != len(descriptors):
            raise ValueError("Dataset TensorMap sidecar parameters must be unique")
        if descriptors and self.frontend != "cuda":
            raise ValueError("only a CUDA Harness may expose TensorMap descriptors")
        if descriptors and not self.tensor_map_sidecar_sha256:
            raise ValueError("Dataset TensorMap descriptors require one sidecar digest")
        if self.tensor_map_sidecar_sha256 and not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.tensor_map_sidecar_sha256
        ):
            raise ValueError("Dataset TensorMap sidecar SHA-256 is malformed")
        physical_block = self.physical_block
        if physical_block is not None:
            physical_block = tuple(int(value) for value in physical_block)
            if len(physical_block) != 3 or any(value <= 0 for value in physical_block):
                raise ValueError("Dataset physical block must contain three positive extents")
        if (
            not isinstance(self.dynamic_shared_memory_bytes, int)
            or self.dynamic_shared_memory_bytes < 0
        ):
            raise ValueError("Dataset dynamic shared memory must be non-negative")
        gate2_policy_for(self.gate2_policy)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "scalar_values", scalar_values)
        object.__setattr__(self, "grid_contract", contract)
        object.__setattr__(self, "factory_kwargs", kwargs)
        object.__setattr__(self, "tensor_map_descriptors", descriptors)
        object.__setattr__(self, "physical_block", physical_block)

    @property
    def dimension_map(self) -> dict[str, int]:
        return dict(self.dimensions)

    @property
    def scalar_map(self) -> dict[str, bool | int | float]:
        return dict(self.scalar_values)

    @property
    def factory_kwarg_map(self) -> dict[str, bool | int | float | str]:
        return dict(self.factory_kwargs)

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "case_id": self.case_id,
            "frontend": self.frontend,
            "dimensions": self.dimension_map,
            "scalar_values": self.scalar_map,
            "grid": list(self.grid),
            "block": list(self.block),
            "grid_contract": list(self.grid_contract),
            "seed": self.seed,
            "factory_kwargs": self.factory_kwarg_map,
            "cuda_arch": self.cuda_arch,
            "source_sha256": self.source_sha256,
            "descriptor_sha256": self.descriptor_sha256,
            "tensor_map_descriptors": [item.to_json() for item in self.tensor_map_descriptors],
            "tensor_map_sidecar_sha256": self.tensor_map_sidecar_sha256,
            "physical_block": list(self.physical_block) if self.physical_block else None,
            "dynamic_shared_memory_bytes": self.dynamic_shared_memory_bytes,
            "gate2_policy": self.gate2_policy,
        }


@dataclass(frozen=True)
class FrontendDatasetHarnessDiagnostic:
    """A closed reason why one catalog entry is not projected as a Harness.

    Diagnostics intentionally contain only catalog identity and a bounded
    mechanical reason.  They never contain source bytes, generated J, or a
    semantic guess about the kernel.
    """

    case_id: str
    frontend: str
    status: str
    reason: str

    def __post_init__(self) -> None:
        if self.frontend not in {"cuda", "tilelang", "native"}:
            raise ValueError("Dataset diagnostic frontend must be native")
        if self.status not in {"ready", "case_unavailable", "harness_invalid"}:
            raise ValueError("unsupported Dataset Harness diagnostic status")
        if not self.reason or len(self.reason) > 2048:
            raise ValueError("Dataset diagnostic reason must be bounded")

    def to_json(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "frontend": self.frontend,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FrontendDatasetHarnessProjection:
    """Deterministic ready/unavailable view of the external Dataset."""

    harnesses: Mapping[str, FrontendDatasetHarness]
    diagnostics: tuple[FrontendDatasetHarnessDiagnostic, ...]
    protocol: str = FRONTEND_DATASET_PROJECTION_PROTOCOL

    def __post_init__(self) -> None:
        harnesses = dict(self.harnesses)
        diagnostics = tuple(self.diagnostics)
        if set(harnesses) & {item.case_id for item in diagnostics}:
            raise ValueError("a Dataset case cannot be both ready and diagnosed")
        object.__setattr__(self, "harnesses", harnesses)
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "ready": [
                harness.to_json()
                for _case_id, harness in sorted(self.harnesses.items())
            ],
            "diagnostics": [item.to_json() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class DatasetRuntimeParameter:
    source_name: str
    semantic_name: str
    kind: str
    dtype: str
    rank: int
    mode: str
    shape: tuple[int, ...] = ()
    value: bool | int | float | None = None
    shape_contract: tuple[int | str, ...] = ()
    semantic_mode: str | None = None
    # Physical layout and observable output identity are source/Harness facts.
    # They are intentionally not folded into J's logical tensor type.
    strides: tuple[int, ...] | None = None
    output_key: str | None = None
    # For an opaque physical ABI slot, ``dtype`` is the physical token and this
    # field is the logical tensor dtype projected to J/Replay.
    logical_dtype: str | None = None

    def __post_init__(self) -> None:
        if not self.source_name.isidentifier() or not self.semantic_name.isidentifier():
            raise ValueError("Dataset parameter names must be identifiers")
        if self.kind not in {"pointer", "scalar", "opaque"}:
            raise ValueError("unsupported Dataset parameter kind")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("Dataset parameter dtype must be non-empty")
        shape = tuple(int(value) for value in self.shape)
        shape_contract: tuple[int | str, ...] = tuple(self.shape_contract)
        strides = (
            tuple(int(value) for value in self.strides)
            if self.strides is not None
            else None
        )
        if strides is not None:
            if self.kind != "pointer":
                raise ValueError("Dataset strides are valid only for pointer ABI")
            if len(strides) != len(shape) or any(value < 0 for value in strides):
                raise ValueError(
                    "Dataset pointer strides must match rank and be non-negative"
                )
        output_key = self.output_key
        if output_key is not None and (
            not isinstance(output_key, str) or not output_key.isidentifier()
        ):
            raise ValueError("Dataset output_key must be an identifier")
        logical_dtype = self.logical_dtype
        if self.kind == "pointer":
            if self.rank != len(shape) or not shape or any(value < 0 for value in shape):
                raise ValueError("Dataset pointer shape must match its non-negative rank")
            if any(value == 0 for value in shape) and self.mode != "output":
                raise ValueError(
                    "zero-length Dataset pointers are allowed only for outputs"
                )
            if len(shape_contract) != self.rank:
                raise ValueError(
                    "Dataset pointer source shape contract must match its rank"
                )
            for extent in shape_contract:
                if isinstance(extent, bool):
                    raise TypeError("Dataset source shape contract cannot be boolean")
                if isinstance(extent, int):
                    if extent < 0:
                        raise ValueError(
                            "Dataset static source shape extents must be non-negative"
                        )
                    if extent == 0 and self.mode != "output":
                        raise ValueError(
                            "zero-length Dataset shape contracts are allowed only for outputs"
                        )
                elif not isinstance(extent, str) or not extent.isidentifier():
                    raise ValueError(
                        "Dataset symbolic source shape extents must be identifiers"
                    )
            if self.mode not in {"input", "output", "inout"}:
                raise ValueError("Dataset pointer requires an effect mode")
            if self.value is not None:
                raise ValueError("Dataset pointer cannot carry a scalar value")
            if logical_dtype is not None:
                if _normalize_dataset_dtype(logical_dtype) != _normalize_dataset_dtype(
                    self.dtype
                ):
                    raise ValueError(
                        "Dataset pointer logical_dtype differs from its dtype"
                    )
                logical_dtype = _normalize_dataset_dtype(logical_dtype)
            if self.mode == "input":
                if output_key is not None:
                    raise ValueError("Dataset input pointer cannot declare output_key")
            elif output_key is None:
                # The source formal is stable within the captured ABI and is a
                # deterministic fallback when an older descriptor omitted the
                # explicit field.  It is not a J output name.
                output_key = self.source_name.casefold()
        elif self.kind == "scalar":
            if (
                self.rank != 0
                or shape
                or shape_contract
                or self.mode != "value"
                or self.value is None
            ):
                raise ValueError("Dataset scalar requires one fixed runtime value")
            if strides is not None or output_key is not None or logical_dtype is not None:
                raise ValueError("Dataset scalar forbids layout/output fields")
        else:
            try:
                physical_dtype = _normalize_physical_opaque_dtype(self.dtype)
                logical_dtype = _normalize_dataset_dtype(logical_dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Dataset opaque TensorMap requires a physical token and logical dtype"
                ) from exc
            if (
                self.rank != 2
                or len(shape) != 2
                or len(shape_contract) != 2
                or any(value <= 0 for value in shape)
                or self.mode != "value"
                or self.value is not None
                or self.semantic_mode != "input"
                or output_key is not None
                or strides is not None
                or logical_dtype is None
            ):
                raise ValueError(
                    "Dataset opaque TensorMap requires one rank-2 logical input sidecar"
                )
            object.__setattr__(self, "dtype", physical_dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "shape_contract", shape_contract)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "output_key", output_key)
        object.__setattr__(self, "logical_dtype", logical_dtype)

    def frontend_parameter(self) -> FrontendParameter:
        return FrontendParameter(self.kind, self.dtype, self.rank, self.mode)

    def compiler_argument(
        self,
        shape_coordinates: Mapping[str, str],
    ) -> SCRCCompilerArgument:
        if self.kind == "pointer":
            return SCRCCompilerArgument(
                self.source_name,
                "tensor",
                self.dtype,
                self.shape,
                mode=self.mode,
                shape_contract=tuple(
                    shape_coordinates[extent]
                    if isinstance(extent, str)
                    else extent
                    for extent in self.shape_contract
                ),
                strides=self.strides,
            )
        if self.kind == "scalar":
            return SCRCCompilerArgument(
                self.source_name,
                "scalar",
                self.dtype,
                value=self.value,
                mode="value",
                shape_coordinate=shape_coordinates.get(self.semantic_name),
            )
        return SCRCCompilerArgument(
            self.source_name, "opaque", self.dtype, mode="value"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "semantic_name": self.semantic_name,
            "kind": self.kind,
            "dtype": self.dtype,
            "rank": self.rank,
            "mode": self.mode,
            "shape": list(self.shape),
            "shape_contract": list(self.shape_contract),
            "value": self.value,
            "semantic_mode": self.semantic_mode,
            "strides": list(self.strides) if self.strides is not None else None,
            "output_key": self.output_key,
            "logical_dtype": self.logical_dtype,
        }


def _logical_abi_parameters(
    parameters: tuple[DatasetRuntimeParameter, ...],
    descriptors: tuple[DatasetTensorMapDescriptor, ...],
) -> tuple[DatasetRuntimeParameter, ...]:
    """Project physical TensorMap slots to the logical tensor ABI seen by J.

    Source execution retains the original opaque parameters.  Gate1, the
    compiler view, J, and Replay instead observe the logical tensors pinned by
    the descriptor sidecar.  Parameter names and positions remain unchanged so
    alpha-normalization and the Source-to-opaque mapping stay exact.
    """

    by_parameter = {item.parameter: item for item in descriptors}
    consumed: set[str] = set()
    result: list[DatasetRuntimeParameter] = []
    for parameter in parameters:
        if parameter.kind != "opaque":
            result.append(parameter)
            continue
        descriptor = by_parameter.get(parameter.source_name)
        if (
            descriptor is None
            or descriptor.semantic_name != parameter.semantic_name
            or descriptor.shape_contract != parameter.shape_contract
            or parameter.semantic_mode != "input"
        ):
            raise ValueError("TensorMap physical ABI lacks one exact logical projection")
        consumed.add(parameter.source_name)
        result.append(
            DatasetRuntimeParameter(
                parameter.source_name,
                parameter.semantic_name,
                "pointer",
                descriptor.logical_dtype,
                parameter.rank,
                "input",
                parameter.shape,
                shape_contract=parameter.shape_contract,
                strides=parameter.strides,
            )
        )
    if consumed != set(by_parameter):
        raise ValueError("TensorMap logical ABI mapping contains unused descriptors")
    return tuple(result)


@dataclass(frozen=True)
class DatasetSourceProjection:
    code: str
    function: str
    parameters: tuple[str, ...]
    frontend: str
    source_kind: str
    origin_location: str
    protocol: str = FRONTEND_DATASET_SOURCE_PROTOCOL
    constexpr_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.function.isidentifier():
            raise ValueError("Dataset source projection requires code and one function")
        if len(self.parameters) != len(set(self.parameters)) or any(
            not name.isidentifier() for name in self.parameters
        ):
            raise ValueError("Dataset source projection parameters must be unique")
        constexpr_parameters = tuple(self.constexpr_parameters)
        if len(constexpr_parameters) != len(set(constexpr_parameters)) or any(
            not name.isidentifier() for name in constexpr_parameters
        ):
            raise ValueError(
                "Dataset source projection constexpr parameters must be unique identifiers"
            )
        if not set(constexpr_parameters) <= set(self.parameters):
            raise ValueError(
                "Dataset source projection constexpr parameters must be in its ABI"
            )
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "constexpr_parameters", constexpr_parameters)

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "function": self.function,
            "parameters": list(self.parameters),
            "constexpr_parameters": list(self.constexpr_parameters),
            "frontend": self.frontend,
            "source_kind": self.source_kind,
            "origin_location": self.origin_location,
        }


@dataclass(frozen=True)
class DatasetWorkloadEntry:
    case_id: str
    frontend: str
    symbol: str
    location: str
    observer_profile: str
    q_policy: str


@dataclass(frozen=True)
class PreparedFrontendDatasetWorkload:
    adapter: FrontendDatasetHarness
    entry: DatasetWorkloadEntry
    source_location: str
    source_slice: DatasetSourceProjection
    compiler_view: SCRCCompilerView
    request: SCRCRunRequest
    source_to_opaque: Mapping[str, str]
    runtime_parameters: tuple[DatasetRuntimeParameter, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    manifest: GeneratedTargetSourceManifest

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, GeneratedTargetSourceManifest):
            raise TypeError(
                "PreparedFrontendDatasetWorkload requires a generated-source manifest"
            )
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
                "Dataset source_to_opaque mapping must exactly cover the ABI"
                + ("; missing " + ", ".join(missing) if missing else "")
                + ("; extra " + ", ".join(extra) if extra else "")
            )
        if len(mapping) != len(set(mapping.values())):
            raise ValueError("Dataset source_to_opaque mapping must be one-to-one")
        if any(
            not isinstance(source, str)
            or not source.isidentifier()
            or not isinstance(opaque, str)
            or re.fullmatch(r"(?:p|c)[0-9]+", opaque) is None
            for source, opaque in mapping.items()
        ):
            raise ValueError(
                "Dataset source_to_opaque mapping must use p*/c* handles"
            )
        self.manifest.validate_compiler_view(
            self.compiler_view,
            source_to_opaque=mapping,
            request=self.request,
        )
        if set(mapping[name] for name in self.manifest.c_slot_order) != set(
            self.request.binding.values
        ):
            raise ValueError(
                "Dataset source/C mapping differs from request c0"
            )
        object.__setattr__(self, "source_to_opaque", mapping)
        object.__setattr__(self, "runtime_parameters", tuple(self.runtime_parameters))
        object.__setattr__(self, "grid", tuple(self.grid))
        object.__setattr__(self, "block", tuple(self.block))

    def with_contract(
        self,
        j: Any,
        c: Any,
        *,
        e_c: Any | None = None,
    ) -> "PreparedFrontendDatasetWorkload":
        """Bind the exact accepted J/C pair before native Gate2 or re-ingress."""

        return replace(
            self,
            manifest=self.manifest.bind_contract(
                j,
                c,
                e_c=e_c,
                source_to_opaque=self.source_to_opaque,
            ),
        )

    def with_lifecycle_records(
        self,
        records: tuple[Any, ...] | list[Any],
    ) -> "PreparedFrontendDatasetWorkload":
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
        j: Any | None = None,
        c: Any | None = None,
    ) -> "PreparedFrontendDatasetWorkload":
        """Admit one explicit generated CUDA or TileLang source projection.

        The complete source is sent to the frontend compiler, while the
        selected device entry is independently projected for the Lift/source
        ABI.  The frozen Dataset case supplies the compiler parameters,
        launch geometry, C slots, and logical TensorMap bindings; none of those
        facts are inferred from a filename or kernel name.
        """

        if not isinstance(source, str) or not source.strip():
            raise ValueError("re-ingress source must be non-empty")
        if not isinstance(source_symbol, str) or not source_symbol.isidentifier():
            raise ValueError("re-ingress source_symbol must be an identifier")
        if not isinstance(source_location, str) or not source_location.strip():
            raise ValueError("re-ingress source_location must be non-empty")
        if (j is None) != (c is None):
            raise ValueError("re-ingress J and C must be supplied together")

        prepared = self
        if prepared.manifest.bound:
            if j is None or c is None:
                raise ValueError(
                    "re-ingress of a bound Dataset workload requires the exact accepted J and C"
                )
            rebound = prepared.with_contract(j, c)
            if (
                rebound.manifest.j_digest != prepared.manifest.j_digest
                or rebound.manifest.e_c_interface_digest
                != prepared.manifest.e_c_interface_digest
                or rebound.manifest.contract_ir_digest
                != prepared.manifest.contract_ir_digest
            ):
                raise ValueError(
                    "re-ingress source changed the bound J/E_C/contract identity"
                )
            prepared = rebound
        elif j is not None:
            assert c is not None
            prepared = prepared.with_contract(j, c)

        if prepared.adapter.frontend == "cuda":
            projection = _cuda_lift_projection(
                source,
                source_symbol,
                source_kind=prepared.source_slice.source_kind,
                origin_location=source_location,
            )
            gate_source: Any = CudaKernelSource.from_code(
                source,
                source_symbol,
            )
            lift_source_kind = "cuda"
            lift_mapping = tuple(prepared.source_to_opaque.items())
        elif prepared.adapter.frontend == "tilelang":
            gate_source = TileLangKernelSource.from_code(
                source,
                source_symbol,
            )
            projection = _tilelang_source_projection(
                source,
                source_symbol,
                source_location=source_location,
                source_kind=prepared.source_slice.source_kind,
            )
            # TileLang PrimFunc source is a Python/TIR source projection at the
            # Lift boundary.  Keep the original kind so the same anonymizer
            # and compiler-view contract are used for initial and re-ingressed
            # observations.
            lift_source_kind = prepared.request.source.lift_source_kind
            if lift_source_kind != "python":
                raise ValueError(
                    "TileLang re-ingress requires the Python/TIR Lift source kind"
                )
            lift_mapping = ()
        else:
            raise ValueError(
                f"unsupported Dataset re-ingress frontend {prepared.adapter.frontend!r}"
            )

        request_source = SourceProgram(
            projection.code,
            gate_source,
            lift_source_kind=lift_source_kind,
            lift_source_symbol=projection.function,
            lift_parameter_order=projection.parameters,
            lift_constexpr_parameters=projection.constexpr_parameters,
            lift_source_to_opaque=lift_mapping,
        )
        source_facts = SourceFacts.from_compiler_view(
            prepared.compiler_view,
            frontend=prepared.request.frontend,
            source_kind=request_source.lift_source_kind,
            source_symbol=projection.function,
            source_to_opaque=prepared.source_to_opaque,
            source_digest=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            source_slice_digest=hashlib.sha256(
                projection.code.encode("utf-8")
            ).hexdigest(),
        )
        construction = CanonicalConstruction.from_source_facts(
            source_facts,
            source_text=projection.code,
        )
        request = SCRCRunRequest(
            request_source,
            prepared.request.binding,
            prepared.request.frontend,
            source_facts,
            construction,
            max_gate1_rounds=prepared.request.max_gate1_rounds,
            max_lift_calls=prepared.request.max_lift_calls,
            max_pre_replay_rejections=prepared.request.max_pre_replay_rejections,
        )
        return reingress_prepared_workload(
            prepared,
            source=source,
            source_slice=projection,
            source_location=source_location,
            compiler_view=prepared.compiler_view,
            source_to_opaque=prepared.source_to_opaque,
            request=request,
            consumer=f"{prepared.adapter.frontend}-source-reingress",
        )


def _safe_integer_expression(expression: Any, values: Mapping[str, int]) -> int:
    if isinstance(expression, bool):
        raise ValueError("launch expressions cannot be boolean")
    if isinstance(expression, int):
        result = expression
    else:
        text = str(expression)
        if len(text) > 256:
            raise ValueError("launch expression exceeds the 256-character limit")
        tree = ast.parse(text, mode="eval")
        if sum(1 for _node in ast.walk(tree)) > 128:
            raise ValueError("launch expression is too complex")
        binary = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift,
        }

        def evaluate(node: ast.AST) -> int:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                if abs(node.value) > 2**31:
                    raise ValueError("launch integer exceeds the 32-bit limit")
                return int(node.value)
            if isinstance(node, ast.Name) and node.id in values:
                return int(values[node.id])
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and type(node.op) in binary:
                value = int(
                    binary[type(node.op)](
                        evaluate(node.left), evaluate(node.right)
                    )
                )
                if abs(value) > 2**31:
                    raise ValueError("launch result exceeds the 32-bit limit")
                return value
            raise ValueError(f"unsupported launch expression {expression!r}")

        result = evaluate(tree)
    if result <= 0:
        raise ValueError(f"launch expression must be positive, got {result}")
    return result


def _launch_from_metadata(
    metadata: Mapping[str, Any], dimensions: Mapping[str, int]
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[str, str, str]]:
    launch = metadata.get("launch")
    if not isinstance(launch, Mapping):
        raise ValueError("generated Dataset kernel lacks launch metadata")
    grid_expressions = tuple(str(launch.get(f"blockIdx.{axis}", "1")) for axis in "xyz")
    block_expressions = tuple(str(launch.get(f"threadIdx.{axis}", "1")) for axis in "xyz")
    grid = tuple(_safe_integer_expression(value, dimensions) for value in grid_expressions)
    block = tuple(_safe_integer_expression(value, dimensions) for value in block_expressions)
    return grid, block, grid_expressions  # type: ignore[return-value]


def _resolve_shape(raw: Any, dimensions: Mapping[str, int]) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("Dataset pointer parameter requires shape metadata")
    return tuple(_safe_integer_expression(value, dimensions) for value in raw)


def _source_shape_contract(raw: Any) -> tuple[int | str, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("Dataset pointer parameter requires shape metadata")
    result: list[int | str] = []
    for extent in raw:
        if isinstance(extent, bool):
            raise TypeError("Dataset source shape contract cannot contain booleans")
        if isinstance(extent, int):
            if extent <= 0:
                raise ValueError("Dataset static source shape extent must be positive")
            result.append(extent)
            continue
        if not isinstance(extent, str) or not extent.isidentifier():
            raise ValueError(
                "Dataset source shape contract supports integers and symbolic identifiers"
            )
        result.append(extent)
    return tuple(result)


def _parameter_alias(name: str) -> str:
    """Return a conservative ABI spelling alias for source/metadata matching.

    TileLang's public PrimFunc uses carriers such as ``x_handle`` while the
    exported CUDA ABI records ``x`` (and a few exports use ``x_ptr``).  Only
    these mechanically declared carrier suffixes are removed.  No operator,
    tensor role, J reference, or semantic relation is inferred here.
    """

    pieces = name.lower().split("_")
    while pieces and pieces[-1] in {"handle", "ptr"}:
        pieces.pop()
    return "_".join(pieces)


def _ordered_abi_rows(metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Validate and return the public ABI in an unambiguous position order."""

    abi = metadata.get("cuda_abi")
    raw_rows = abi.get("parameters") if isinstance(abi, Mapping) else None
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Dataset kernel lacks structured CUDA ABI metadata")
    positioned: list[tuple[int, Mapping[str, Any]]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Dataset ABI parameters must all be objects")
        position = row.get("position")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("Dataset ABI parameter positions must be integers")
        positioned.append((position, row))
    positions = sorted(position for position, _row in positioned)
    if positions != list(range(len(positioned))):
        raise ValueError("Dataset ABI parameter positions must be unique and contiguous")
    return tuple(row for _position, row in sorted(positioned, key=lambda item: item[0]))


def _metadata_rows_in_source_order(
    source_names: tuple[str, ...], rows: list[Mapping[str, Any]]
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Bijection-match materialized carriers, then retain source order."""

    by_alias: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        metadata_name = str(row.get("name") or row.get("semantic_name") or "")
        if not metadata_name:
            raise ValueError("Dataset ABI parameter lacks a public name")
        by_alias.setdefault(_parameter_alias(metadata_name), []).append(row)
    ordered: list[tuple[str, Mapping[str, Any]]] = []
    used: set[str] = set()
    for source_name in source_names:
        candidates = [
            row
            for row in by_alias.get(_parameter_alias(source_name), ())
            if str(row.get("name") or row.get("semantic_name") or "") not in used
        ]
        if len(candidates) != 1:
            raise ValueError(
                "materialized TileLang ABI cannot be uniquely aligned with "
                f"source parameter {source_name!r}"
            )
        row = candidates[0]
        metadata_name = str(row.get("name") or row.get("semantic_name") or "")
        used.add(metadata_name)
        ordered.append((source_name, row))
    if len(used) != len(source_names):
        raise ValueError("materialized TileLang ABI contains an unlisted carrier")
    return tuple(ordered)


def _metadata_parameters(
    metadata: Mapping[str, Any],
    *,
    dimensions: Mapping[str, int],
    scalars: Mapping[str, bool | int | float],
    source_names: tuple[str, ...] | None = None,
    include_runtime_scalars: bool,
    use_logical_interface_dtypes: bool = False,
    use_logical_interface: bool = False,
) -> tuple[DatasetRuntimeParameter, ...]:
    if use_logical_interface:
        rows = list(_metadata_interface_rows(metadata))
    else:
        rows = list(_ordered_abi_rows(metadata))
    if not include_runtime_scalars:
        rows = [item for item in rows if item.get("kind") == "pointer"]
    if source_names is not None:
        source_rows = _metadata_rows_in_source_order(source_names, rows)
        if len(source_rows) != len(rows):
            raise ValueError("materialized Dataset ABI differs from generated metadata")
    else:
        source_rows = tuple((str(row.get("name") or ""), row) for row in rows)

    # A physical CUDA ABI may contain ``opaque`` CUtensorMap rows while its
    # logical interface contains ordinary rank-2 pointers.  Resolve those rows
    # by their explicit semantic name; never align them by position or infer a
    # dtype from the physical token.
    logical_rows_by_source: dict[str, Mapping[str, Any]] = {}
    need_logical_rows = use_logical_interface_dtypes or any(
        str(row.get("kind") or "") == "opaque" for _name, row in source_rows
    )
    if not use_logical_interface and need_logical_rows:
        logical_rows = _metadata_interface_rows(metadata)
        for source_name, physical in source_rows:
            logical = _logical_row_for_physical(physical, logical_rows)
            physical_kind = str(physical.get("kind") or "")
            logical_kind = str(logical.get("kind") or "")
            physical_shape = _source_shape_contract(physical.get("shape"))
            logical_shape = _source_shape_contract(logical.get("shape"))
            physical_rank = int(physical.get("rank", len(physical_shape)))
            logical_rank = int(logical.get("rank", len(logical_shape)))
            if physical_shape != logical_shape or physical_rank != logical_rank:
                raise ValueError(
                    "Dataset logical interface differs from its physical ABI shape"
                )
            if physical_kind == "opaque":
                if logical_kind != "pointer" or str(logical.get("mode") or "") != "input":
                    raise ValueError(
                        "Dataset opaque ABI must project to a logical input pointer"
                    )
            elif logical_kind != physical_kind or str(logical.get("mode") or "value") != str(
                physical.get("mode") or "value"
            ):
                raise ValueError(
                    "Dataset logical interface differs from its physical ABI kind/mode"
                )
            logical_rows_by_source[source_name] = logical

    result: list[DatasetRuntimeParameter] = []
    for source_name, row in source_rows:
        semantic_name = str(row.get("semantic_name") or row.get("name") or "")
        kind = str(row.get("kind") or "")
        logical_row = logical_rows_by_source.get(source_name)
        value = None
        shape: tuple[int, ...] = ()
        shape_contract: tuple[int | str, ...] = ()
        rank = int(row.get("rank", 0))
        if kind in {"pointer", "opaque"}:
            shape_contract = _source_shape_contract(row.get("shape"))
            shape = _resolve_shape(shape_contract, dimensions)
        elif kind == "scalar":
            if semantic_name in scalars:
                value = scalars[semantic_name]
            elif semantic_name in dimensions:
                value = dimensions[semantic_name]
            else:
                raise ValueError(f"Dataset scalar {semantic_name!r} lacks a Harness value")

        if kind == "opaque":
            dtype = _normalize_physical_opaque_dtype(row.get("dtype"))
            logical_dtype = _normalize_dataset_dtype(
                (logical_row or {}).get("dtype")
            )
        elif logical_row is not None:
            dtype = _normalize_dataset_dtype(logical_row.get("dtype"))
            logical_dtype = None
        else:
            dtype = _normalize_dataset_dtype(row.get("dtype"))
            logical_dtype = None
        output_key = row.get("output_key")
        result.append(
            DatasetRuntimeParameter(
                source_name=source_name,
                semantic_name=semantic_name,
                kind=kind,
                dtype=dtype,
                rank=rank,
                mode=str(row.get("mode") or "value"),
                shape=shape,
                value=value,
                shape_contract=shape_contract,
                semantic_mode=(
                    str(row.get("semantic_mode"))
                    if row.get("semantic_mode") is not None
                    else (
                        str(logical_row.get("semantic_mode"))
                        if logical_row is not None
                        and logical_row.get("semantic_mode") is not None
                        else None
                    )
                ),
                strides=_metadata_row_strides(row, rank=rank)
                if kind == "pointer"
                else None,
                output_key=str(output_key) if output_key is not None else None,
                logical_dtype=logical_dtype,
            )
        )
    return tuple(result)


def _shape_coordinate_mapping(
    parameters: tuple[DatasetRuntimeParameter, ...],
) -> dict[str, str]:
    """Assign source-grounded shape identities without exposing source names.

    Coordinates follow first tensor-declaration order and are independent of
    their current valuations.  Equal source symbols share one coordinate;
    distinct symbols remain distinct even when their finite values coincide.
    """

    coordinates: dict[str, str] = {}
    for parameter in parameters:
        # A physical opaque slot has no shape coordinate until the pinned
        # sidecar projects it into the separate logical ABI view.
        if parameter.kind != "pointer":
            continue
        for extent in parameter.shape_contract:
            if isinstance(extent, str) and extent not in coordinates:
                coordinates[extent] = f"h{len(coordinates)}"
    return coordinates


def _grid_shape_coordinates(
    grid_contract: tuple[str, str, str],
    shape_coordinates: Mapping[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Bind only axes that are exactly one authored shape coordinate.

    Matching by concrete extent would conflate distinct symbolic dimensions;
    compound launch expressions therefore remain unbound expressions.
    """

    return tuple(shape_coordinates.get(expression) for expression in grid_contract)  # type: ignore[return-value]


def _materialize_tilelang_projection(
    kernel: FrontendDatasetKernel,
    factory_kwargs: Mapping[str, bool | int | float | str],
) -> DatasetSourceProjection:
    # Python's process-randomized ``hash`` made this ephemeral import name
    # change on every run without carrying any proof meaning.  A short stable
    # digest keeps module isolation while making preparation/replay receipts
    # reproducible and avoiding a needless per-run hash churn.
    module_digest = hashlib.sha256(
        f"{kernel.path.resolve()}:{kernel.symbol}".encode("utf-8")
    ).hexdigest()[:16]
    module_name = f"_intentir_dataset_lift_{module_digest}"
    spec = importlib.util.spec_from_file_location(module_name, kernel.path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load TileLang Dataset module {kernel.path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        factory = getattr(module, kernel.symbol)
        prim_func = factory(**dict(factory_kwargs))
        code = str(prim_func.script())
    finally:
        sys.modules.pop(module_name, None)
    marker = "# from tvm.script import tir as T"
    if marker not in code:
        raise ValueError("TileLang PrimFunc script lacks its public TIR import marker")
    code = code.replace(marker, "import tilelang.language as T", 1)
    tree = ast.parse(code)
    functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("materialized TileLang source must contain exactly one PrimFunc")
    function = functions[0]
    parameters = tuple(argument.arg for argument in function.args.args)
    return DatasetSourceProjection(
        code if code.endswith("\n") else code + "\n",
        function.name,
        parameters,
        "tilelang",
        "materialized_tir",
        kernel.location,
    )


def _tilelang_source_projection(
    code: str,
    symbol: str,
    *,
    source_location: str,
    source_kind: str,
) -> DatasetSourceProjection:
    """Validate and project one explicitly selected generated PrimFunc.

    Re-ingress must consume the supplied source bytes.  Calling the original
    Dataset factory here would silently select a new source, so this helper
    only parses the already generated module after ``TileLangKernelSource``
    has applied its static policy.
    """

    tree = ast.parse(code)
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if len(functions) != 1 or functions[0].name != symbol:
        raise ValueError(
            "TileLang re-ingress source must contain exactly the selected PrimFunc"
        )
    function = functions[0]
    parameters = tuple(argument.arg for argument in function.args.args)
    return DatasetSourceProjection(
        code if code.endswith("\n") else code + "\n",
        function.name,
        parameters,
        "tilelang",
        source_kind,
        source_location,
    )


def _generated_companion(
    tilelang: FrontendDatasetKernel,
    kernels: tuple[FrontendDatasetKernel, ...],
) -> FrontendDatasetKernel:
    matches = [
        item
        for item in kernels
        if item.frontend == "cuda"
        and item.metadata.get("origin") == "tilelang_export"
        and item.metadata.get("origin_case_id") == tilelang.case_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"TileLang Dataset case {tilelang.case_id!r} requires exactly one CUDA ABI companion"
        )
    return matches[0]


def _literal_factory_kwargs(
    kernel: FrontendDatasetKernel,
) -> tuple[tuple[str, bool | int | float | str], ...]:
    """Read only literal factory defaults; never import or execute Dataset code."""

    try:
        tree = ast.parse(kernel.path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError(f"TileLang factory source is not statically readable: {exc}") from exc
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == kernel.symbol
    ]
    if len(functions) != 1:
        raise ValueError("TileLang Dataset factory definition is not unique")
    function = functions[0]
    positional = [*function.args.posonlyargs, *function.args.args]
    positional_defaults = [
        *([None] * (len(positional) - len(function.args.defaults))),
        *function.args.defaults,
    ]
    keyword_only = list(function.args.kwonlyargs)
    keyword_defaults = list(function.args.kw_defaults)
    arguments = [*zip(positional, positional_defaults), *zip(keyword_only, keyword_defaults)]
    result: list[tuple[str, bool | int | float | str]] = []
    for argument, default in arguments:
        if default is None:
            raise ValueError(
                f"TileLang factory argument {argument.arg!r} lacks a literal default"
            )
        try:
            value = ast.literal_eval(default)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(
                f"TileLang factory default {argument.arg!r} is not a literal"
            ) from exc
        if not isinstance(value, (bool, int, float, str)):
            raise ValueError(
                f"TileLang factory default {argument.arg!r} is not a JSON scalar"
            )
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            raise ValueError(f"TileLang factory default {argument.arg!r} is non-finite")
        result.append((argument.arg, value))
    return tuple(result)


def _snapshot_canonical_shapes(dataset_root: Path) -> dict[str, dict[str, int]]:
    """Load only concrete invocation extents from optional export snapshots."""

    result: dict[str, dict[str, int]] = {}
    snapshot_root = dataset_root / "cuda" / "ops" / "snapshots"
    for path in sorted(snapshot_root.glob("*.cuda_snapshot.json")):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if not isinstance(loaded, Mapping):
            continue
        name = loaded.get("name")
        canonical = loaded.get("canonical_shapes")
        if not isinstance(name, str) or not isinstance(canonical, Mapping):
            continue
        values: dict[str, int] = {}
        for key, value in canonical.items():
            if (
                isinstance(key, str)
                and key.isidentifier()
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            ):
                values[key] = value
        if values:
            result[name] = values
    return result


def _ordered_dimension_symbols(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect authored symbolic extents without interpreting their meaning."""

    seen: set[str] = set()
    ordered: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.isidentifier() and value not in seen:
            seen.add(value)
            ordered.append(value)

    rows = _ordered_abi_rows(metadata)
    for row in rows:
        for extent in row.get("shape") or ():
            add(extent)
        if row.get("kind") == "scalar":
            add(row.get("semantic_name") or row.get("name"))
    launch = metadata.get("launch")
    if not isinstance(launch, Mapping):
        raise ValueError("generated Dataset kernel lacks launch metadata")
    for axis in ("x", "y", "z"):
        for stage in ("blockIdx", "threadIdx"):
            expression = launch.get(f"{stage}.{axis}", "1")
            if isinstance(expression, bool):
                raise ValueError("launch expressions cannot be boolean")
            if isinstance(expression, int):
                continue
            try:
                tree = ast.parse(str(expression), mode="eval")
            except SyntaxError as exc:
                raise ValueError(
                    f"launch expression {expression!r} is not a complete expression"
                ) from exc
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    add(node.id)
    return tuple(ordered)


def _dataset_descriptor_values(
    metadata: Mapping[str, Any],
    *,
    canonical_shapes: Mapping[str, int],
) -> tuple[
    tuple[tuple[str, int], ...],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[str, str, str],
    tuple[tuple[str, bool | int | float], ...],
]:
    symbols = _ordered_dimension_symbols(metadata)
    dimensions = tuple(
        (
            symbol,
            int(canonical_shapes.get(symbol, _DEFAULT_DATASET_DIMENSION)),
        )
        for symbol in symbols
    )
    if any(value <= 0 for _name, value in dimensions):
        raise ValueError("Dataset canonical shape values must be positive")
    dimension_map = dict(dimensions)
    grid, block, contract = _launch_from_metadata(metadata, dimension_map)
    scalar_values: list[tuple[str, bool | int | float]] = []
    rows = _ordered_abi_rows(metadata)
    for row in rows:
        if row.get("kind") != "scalar":
            continue
        name = str(row.get("semantic_name") or row.get("name") or "")
        if not name:
            raise ValueError("Dataset scalar parameter lacks a public name")
        scalar_values.append((name, dimension_map.get(name, _DEFAULT_DATASET_DIMENSION)))
    return dimensions, grid, block, contract, tuple(scalar_values)


def _literal_handwritten_io(kernel: FrontendDatasetKernel) -> Mapping[str, Any]:
    """Read the adjacent CUDA IO dictionary with ``ast.literal_eval`` only."""

    metadata_path = kernel.path.with_suffix(".py")
    try:
        tree = ast.parse(metadata_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError(
            f"handwritten CUDA IO metadata is not statically readable: {exc}"
        ) from exc
    candidates: list[Mapping[str, Any]] = []
    for node in tree.body:
        value_node: ast.expr | None = None
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Assign):
            value_node = node.value
            names = tuple(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value_node = node.value
            names = (node.target.id,)
        if value_node is None or not any(name.endswith("_io") for name in names):
            continue
        try:
            loaded = ast.literal_eval(value_node)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(loaded, Mapping) and loaded.get("kernel_name") == kernel.symbol:
            candidates.append(loaded)
    if len(candidates) != 1:
        raise ValueError(
            "handwritten CUDA kernel requires exactly one literal adjacent IO descriptor"
        )
    return candidates[0]


def _matching_parenthesis(code: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(code)):
        character = code[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("CUDA kernel declaration has unbalanced parentheses")


def _split_cuda_parameters(parameters: str) -> tuple[str, ...]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(parameters):
        if character in "(<[":
            depth += 1
        elif character in ")>]":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(parameters[start:index].strip())
            start = index + 1
    tail = parameters[start:].strip()
    if tail:
        result.append(tail)
    if not result or any(not item for item in result):
        raise ValueError("CUDA kernel parameter list is empty or malformed")
    return tuple(result)


def _matching_brace(code: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(code)):
        character = code[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("CUDA kernel definition has unbalanced braces")


def _cuda_definition(
    code: str, symbol: str
) -> tuple[tuple[str, ...], str]:
    matches: list[tuple[tuple[str, ...], str]] = []
    for match in re.finditer(rf"\b{re.escape(symbol)}\s*\(", code):
        header = code[max(0, match.start() - 512) : match.start()]
        if "__global__" not in header:
            continue
        opening = code.find("(", match.start())
        closing = _matching_parenthesis(code, opening)
        suffix = code[closing + 1 :].lstrip()
        if not suffix.startswith("{"):
            continue
        body_opening = code.find("{", closing + 1)
        body_closing = _matching_brace(code, body_opening)
        matches.append(
            (
                _split_cuda_parameters(code[opening + 1 : closing]),
                code[body_opening + 1 : body_closing],
            )
        )
    if len(matches) != 1:
        raise ValueError(
            "handwritten CUDA source must contain exactly one selected __global__ definition"
        )
    return matches[0]


def _cuda_declaration_name(declaration: str) -> str:
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", declaration)
    if match is None:
        raise ValueError(f"unsupported CUDA parameter declaration {declaration!r}")
    return match.group(1)


def _normalize_dataset_dtype(dtype: Any) -> str:
    aliases = {
        "bool": "i1",
        "boolean": "i1",
        "float16": "f16",
        "half": "f16",
        "__half": "f16",
        "float32": "f32",
        "float64": "f64",
        "double": "f64",
        "bfloat16": "bf16",
        "__nv_bfloat16": "bf16",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "index": "i64",
        "uint": "u32",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
    }
    value = aliases.get(str(dtype), str(dtype))
    if not re.fullmatch(
        r"(?:bf16|f16|f32|f64|i1|i8|i16|i32|i64|u8|u16|u32)",
        value,
    ):
        raise ValueError(f"unsupported Dataset ABI dtype {dtype!r}")
    return value


def _normalize_physical_opaque_dtype(dtype: Any) -> str:
    """Normalize the closed physical spelling for an opaque ABI value.

    CUDA metadata has used both CUDA's ``CUtensorMap`` spelling and the
    source-neutral ``tensor_map`` token.  These are physical ABI names, not
    logical tensor dtypes, so they must never enter ``_normalize_dataset_dtype``.
    """

    spelling = str(dtype or "").strip().casefold().replace(" ", "")
    aliases = {
        "tensormap": "cuda_tensor_map",
        "cutensormap": "cuda_tensor_map",
        "cuda_tensormap": "cuda_tensor_map",
        "cuda_tensor_map": "cuda_tensor_map",
        "tensor_map": "tensor_map",
        "opaque": "opaque",
    }
    result = aliases.get(spelling)
    if result is None:
        raise ValueError(f"unsupported physical opaque ABI dtype {dtype!r}")
    return result


def _metadata_row_strides(
    row: Mapping[str, Any],
    *,
    rank: int,
) -> tuple[int, ...] | None:
    """Read physical element strides without guessing a layout."""

    raw = row.get("strides")
    if raw is None and isinstance(row.get("layout"), Mapping):
        raw = row["layout"].get("strides")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != rank:
        raise ValueError("Dataset ABI strides must match the declared rank")
    result = tuple(int(value) for value in raw)
    if any(isinstance(value, bool) or value < 0 for value in result):
        raise ValueError("Dataset ABI strides must be non-negative integers")
    return result


def _cuda_declaration_kind_dtype(declaration: str) -> tuple[str, str]:
    """Project one CUDA declaration to its physical kind and scalar dtype."""

    normalized = re.sub(r"\b(const|volatile|restrict|__restrict__)\b", " ", declaration)
    lowered = normalized.casefold()
    if "cutensormap" in lowered:
        return "opaque", "cuda_tensor_map"
    kind = "pointer" if "*" in normalized else "scalar"
    type_aliases = (
        ("__nv_bfloat16", "bf16"),
        ("bfloat16", "bf16"),
        ("__half", "f16"),
        ("half", "f16"),
        ("float", "f32"),
        ("double", "f64"),
        ("int8_t", "i8"),
        ("int16_t", "i16"),
        ("int32_t", "i32"),
        ("int64_t", "i64"),
        ("uint8_t", "u8"),
        ("uint16_t", "u16"),
        ("uint32_t", "u32"),
        ("int", "i32"),
        ("unsigned", "u32"),
        ("bool", "i1"),
    )
    for spelling, dtype in type_aliases:
        if re.search(rf"\b{re.escape(spelling)}\b", normalized):
            return kind, dtype
    raise ValueError(f"CUDA declaration dtype is outside the closed profile: {declaration!r}")


def _metadata_interface_rows(
    metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    interface = metadata.get("interface")
    raw_rows = interface.get("parameters") if isinstance(interface, Mapping) else None
    if not isinstance(raw_rows, list) or not raw_rows or not all(
        isinstance(item, Mapping) for item in raw_rows
    ):
        raise ValueError("Dataset metadata lacks a structured logical interface")
    return tuple(raw_rows)


def _logical_row_for_physical(
    physical: Mapping[str, Any],
    logical_rows: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    """Resolve one physical row through its explicit semantic name only."""

    semantic_name = str(
        physical.get("semantic_name") or physical.get("name") or ""
    )
    if not semantic_name:
        raise ValueError("Dataset physical ABI row lacks a semantic_name")
    candidates = [
        row
        for row in logical_rows
        if str(row.get("name") or row.get("semantic_name") or "")
        in {semantic_name, _parameter_alias(semantic_name)}
        or _parameter_alias(
            str(row.get("name") or row.get("semantic_name") or "")
        )
        == _parameter_alias(semantic_name)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Dataset physical ABI row cannot be uniquely aligned with its "
            f"logical interface parameter {semantic_name!r}"
        )
    return candidates[0]


def _cuda_pointer_mode(name: str, declaration: str, body: str) -> str:
    """Conservatively project direct subscript reads/writes for one pointer.

    The handwritten corpus has no explicit effect-mode field.  We accept only
    direct ``p[index]`` accesses and reject bare pointer escapes/calls.  Thus a
    mutable read-only pointer remains an input instead of being mislabeled as
    output merely because its declaration omitted ``const``.
    """

    if "*" not in declaration:
        raise ValueError(f"CUDA tensor parameter {name!r} is not a pointer")
    access_pattern = re.compile(rf"\b{re.escape(name)}\s*\[[^\]]+\]")
    accesses = list(access_pattern.finditer(body))
    remainder = access_pattern.sub("", body)
    if re.search(rf"\b{re.escape(name)}\b", remainder):
        raise ValueError(
            f"CUDA pointer {name!r} escapes direct subscript effect analysis"
        )
    read = False
    write = False
    for access in accesses:
        suffix = body[access.end() :].lstrip()
        assignment = re.match(r"(?P<op>(?:<<|>>|[+\-*/%&|^])?=)(?!=)", suffix)
        if assignment is None:
            read = True
            continue
        write = True
        if assignment.group("op") != "=":
            read = True
    if re.search(r"\bconst\b", declaration):
        if write:
            raise ValueError(f"const CUDA pointer {name!r} is written")
        if not read:
            raise ValueError(f"const CUDA pointer {name!r} has no direct read")
        return "input"
    if read and write:
        return "inout"
    if read:
        return "input"
    if write:
        return "output"
    raise ValueError(f"CUDA pointer {name!r} has no direct readable effect")


def _handwritten_runtime_metadata(
    kernel: FrontendDatasetKernel,
) -> Mapping[str, Any]:
    """Project source ABI plus literal IO metadata into a finite call descriptor.

    Handwritten CUDA sources do not have a fixed launch in Dataset.  The generic
    bounded policy uses one block and dimensions of four.  Rank-1 buffers get a
    16-thread block; rank-2-or-higher buffers get a 16x16 block.  This is a
    Harness invocation choice, not a kernel-family or semantic inference.
    """

    io = _literal_handwritten_io(kernel)
    arg_names = io.get("arg_names")
    tensors = io.get("tensors")
    scalars = io.get("scalars")
    if (
        not isinstance(arg_names, list)
        or not all(isinstance(name, str) and name.isidentifier() for name in arg_names)
        or len(arg_names) != len(set(arg_names))
        or not isinstance(tensors, Mapping)
        or not isinstance(scalars, Mapping)
    ):
        raise ValueError("handwritten CUDA IO descriptor is malformed")
    code = kernel.path.read_text(encoding="utf-8")
    declarations, body = _cuda_definition(code, kernel.symbol)
    declaration_names = tuple(_cuda_declaration_name(item) for item in declarations)
    if tuple(arg_names) != declaration_names:
        raise ValueError("handwritten CUDA IO descriptor differs from source ABI order")
    rows: list[dict[str, Any]] = []
    pointer_ranks: list[int] = []
    for position, (name, declaration) in enumerate(
        zip(arg_names, declarations, strict=True)
    ):
        if name in tensors:
            tensor = tensors[name]
            if not isinstance(tensor, Mapping):
                raise ValueError("handwritten CUDA tensor metadata must be an object")
            shape = tensor.get("shape")
            rank = tensor.get("rank")
            if (
                not isinstance(shape, list)
                or not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank != len(shape)
                or rank <= 0
            ):
                raise ValueError("handwritten CUDA tensor rank/shape metadata differs")
            source_shape = list(_source_shape_contract(shape))
            pointer_ranks.append(rank)
            rows.append(
                {
                    "declaration": declaration,
                    "dtype": _normalize_dataset_dtype(tensor.get("dtype")),
                    "kind": "pointer",
                    "mode": _cuda_pointer_mode(name, declaration, body),
                    "name": name,
                    "position": position,
                    "rank": rank,
                    "semantic_name": name,
                    "shape": source_shape,
                }
            )
            continue
        if name in scalars:
            if "*" in declaration:
                raise ValueError(f"CUDA scalar parameter {name!r} is a pointer")
            rows.append(
                {
                    "declaration": declaration,
                    "dtype": _normalize_dataset_dtype(scalars[name]),
                    "kind": "scalar",
                    "mode": "value",
                    "name": name,
                    "position": position,
                    "rank": 0,
                    "semantic_name": name,
                }
            )
            continue
        raise ValueError(f"handwritten CUDA parameter {name!r} lacks IO metadata")
    if set(tensors) | set(scalars) != set(arg_names):
        raise ValueError("handwritten CUDA IO descriptor contains extra parameters")
    block = (
        _DEFAULT_DATASET_BLOCK_2D
        if pointer_ranks and max(pointer_ranks) >= 2
        else _DEFAULT_DATASET_BLOCK_1D
    )
    return {
        "origin": "handwritten",
        "harness_protocol": FRONTEND_DATASET_PROJECTION_PROTOCOL,
        "cuda_abi": {"parameters": rows},
        "launch": {
            "blockIdx.x": "1",
            "blockIdx.y": "1",
            "blockIdx.z": "1",
            "threadIdx.x": str(block[0]),
            "threadIdx.y": str(block[1]),
            "threadIdx.z": str(block[2]),
        },
        "target": {"name": "cuda"},
    }


def _runtime_metadata(kernel: FrontendDatasetKernel) -> Mapping[str, Any]:
    if kernel.frontend != "cuda":
        raise ValueError("runtime CUDA metadata requested for a non-CUDA kernel")
    if kernel.metadata.get("origin") == "handwritten":
        return _handwritten_runtime_metadata(kernel)
    return kernel.metadata


def _stable_dataset_source_mapping(
    source_parameters: tuple[str, ...],
    compiler_slots: tuple[str, ...],
) -> dict[str, str]:
    """Allocate one stable p*/c* namespace for every Dataset ABI name."""

    source_parameters = tuple(source_parameters)
    compiler_slots = tuple(compiler_slots)
    if len(source_parameters) != len(set(source_parameters)):
        raise ValueError("Dataset source ABI parameters must be unique")
    if len(compiler_slots) != len(set(compiler_slots)):
        raise ValueError("Dataset compiler slots must be unique")
    overlap = set(source_parameters) & set(compiler_slots)
    if overlap:
        raise ValueError(
            "Dataset source parameters and compiler slots overlap: "
            + ", ".join(sorted(overlap))
        )
    return dict(
        CanonicalHandleAllocator.abi_mapping(source_parameters, compiler_slots)
    )


def _dataset_manifest_arguments(
    parameters: tuple[DatasetRuntimeParameter, ...],
    compiler_view: SCRCCompilerView,
) -> tuple[GeneratedTargetSourceArgument, ...]:
    """Project physical Dataset parameters into the shared source manifest."""

    by_compiler_name = {item.name: item for item in compiler_view.arguments}
    result: list[GeneratedTargetSourceArgument] = []
    for parameter in parameters:
        compiler_argument = by_compiler_name.get(parameter.source_name)
        if compiler_argument is None:
            raise ValueError(
                f"Dataset compiler view lacks parameter {parameter.source_name!r}"
            )
        if parameter.kind == "pointer":
            if compiler_argument.kind != "tensor":
                raise ValueError(
                    f"Dataset pointer {parameter.source_name!r} lacks tensor ABI mapping"
                )
            result.append(
                GeneratedTargetSourceArgument(
                    name=parameter.source_name,
                    category="buffer",
                    role=parameter.mode,
                    dtype=parameter.dtype,
                    shape=parameter.shape,
                    shape_contract=compiler_argument.shape_contract,
                    strides=parameter.strides,
                    output_key=parameter.output_key,
                )
            )
        elif parameter.kind == "opaque":
            if compiler_argument.kind != "tensor":
                raise ValueError(
                    f"Dataset opaque {parameter.source_name!r} lacks logical shape projection"
                )
            result.append(
                GeneratedTargetSourceArgument(
                    name=parameter.source_name,
                    category="opaque",
                    role="input",
                    dtype=parameter.dtype,
                    shape=parameter.shape,
                    shape_contract=compiler_argument.shape_contract,
                    logical_dtype=parameter.logical_dtype,
                )
            )
        elif parameter.kind == "scalar":
            if compiler_argument.kind != "scalar" or compiler_argument.value is None:
                raise ValueError(
                    f"Dataset scalar {parameter.source_name!r} lacks a compiler valuation"
                )
            result.append(
                GeneratedTargetSourceArgument(
                    name=parameter.source_name,
                    category="runtime_scalar",
                    role="value",
                    dtype=parameter.dtype,
                    value=parameter.value,
                    shape_coordinate=compiler_argument.shape_coordinate,
                )
            )
        else:  # pragma: no cover - DatasetRuntimeParameter closes the set
            raise ValueError("unsupported Dataset parameter kind")
    return tuple(result)


def _dataset_tensor_map_relations(
    parameters: tuple[DatasetRuntimeParameter, ...],
    descriptors: tuple[DatasetTensorMapDescriptor, ...],
    compiler_view: SCRCCompilerView,
) -> tuple[Any, ...]:
    """Build source-side TensorMap relations in the anonymous h* view."""

    if not descriptors:
        return ()
    from intentir.verify.workload_adapters import GeneratedSourceTensorMapRelation

    by_parameter = {item.source_name: item for item in parameters}
    by_descriptor = {item.parameter: item for item in descriptors}
    by_compiler_name = {item.name: item for item in compiler_view.arguments}
    if set(by_descriptor) != {
        item.source_name for item in parameters if item.kind == "opaque"
    }:
        raise ValueError(
            "Dataset TensorMap descriptors must exactly cover physical opaque parameters"
        )
    relations: list[GeneratedSourceTensorMapRelation] = []
    for physical_name, descriptor in (
        (item.parameter, item) for item in descriptors
    ):
        parameter = by_parameter.get(physical_name)
        compiler_argument = by_compiler_name.get(physical_name)
        if (
            parameter is None
            or parameter.kind != "opaque"
            or parameter.logical_dtype != descriptor.logical_dtype
            or parameter.shape_contract != descriptor.shape_contract
            or compiler_argument is None
            or compiler_argument.kind != "tensor"
            or compiler_argument.shape_contract is None
            or compiler_argument.dtype != descriptor.logical_dtype
        ):
            raise ValueError(
                f"Dataset TensorMap logical/physical ABI differs for {physical_name!r}"
            )
        relations.append(
            GeneratedSourceTensorMapRelation(
                physical_parameter=physical_name,
                logical_parameter=parameter.semantic_name,
                logical_dtype=descriptor.logical_dtype,
                shape_contract=compiler_argument.shape_contract,
                box_dim=descriptor.box_dim,
                swizzle=descriptor.swizzle,
            )
        )
    return tuple(relations)


def _validate_cuda_metadata_admission(
    kernel: FrontendDatasetKernel,
    metadata: Mapping[str, Any],
    physical_parameters: tuple[DatasetRuntimeParameter, ...],
    *,
    logical_parameters: tuple[DatasetRuntimeParameter, ...] = (),
    tensor_map_descriptors: tuple[DatasetTensorMapDescriptor, ...] = (),
) -> None:
    """Close CUDA declaration, metadata, and logical sidecar ABI together."""

    code = kernel.path.read_text(encoding="utf-8")
    declarations, _body = _cuda_definition(code, kernel.symbol)
    declaration_names = tuple(_cuda_declaration_name(item) for item in declarations)
    expected_names = tuple(item.source_name for item in physical_parameters)
    if declaration_names != expected_names:
        raise ValueError(
            f"CUDA declaration order differs from Dataset metadata for {kernel.case_id}"
        )
    rows = _ordered_abi_rows(metadata)
    if len(rows) != len(physical_parameters):
        raise ValueError("CUDA declaration and metadata ABI arities differ")
    descriptors = {item.parameter: item for item in tensor_map_descriptors}
    for declaration, row, parameter in zip(
        declarations, rows, physical_parameters, strict=True
    ):
        if str(row.get("name") or row.get("semantic_name") or "") != parameter.source_name:
            raise ValueError("CUDA metadata row order/name differs from declaration order")
        recorded_declaration = row.get("declaration")
        if recorded_declaration is not None and str(recorded_declaration).strip() != declaration.strip():
            raise ValueError(
                f"CUDA declaration text differs from metadata for {parameter.source_name!r}"
            )
        declaration_kind, declaration_dtype = _cuda_declaration_kind_dtype(declaration)
        if declaration_kind != parameter.kind or declaration_dtype != parameter.dtype:
            raise ValueError(
                f"CUDA declaration kind/dtype differs for {parameter.source_name!r}"
            )
        if str(row.get("kind") or "") != parameter.kind:
            raise ValueError(f"CUDA metadata kind differs for {parameter.source_name!r}")
        row_dtype = (
            _normalize_physical_opaque_dtype(row.get("dtype"))
            if parameter.kind == "opaque"
            else _normalize_dataset_dtype(row.get("dtype"))
        )
        if row_dtype != parameter.dtype:
            raise ValueError(f"CUDA metadata dtype differs for {parameter.source_name!r}")
        if str(row.get("mode") or "value") != parameter.mode:
            raise ValueError(f"CUDA metadata mode differs for {parameter.source_name!r}")
        if int(row.get("rank", -1)) != parameter.rank:
            raise ValueError(f"CUDA metadata rank differs for {parameter.source_name!r}")
        if parameter.kind in {"pointer", "opaque"}:
            if _source_shape_contract(row.get("shape")) != parameter.shape_contract:
                raise ValueError(
                    f"CUDA metadata shape contract differs for {parameter.source_name!r}"
                )
            if parameter.kind == "pointer" and _metadata_row_strides(
                row, rank=parameter.rank
            ) != parameter.strides:
                raise ValueError(
                    f"CUDA metadata strides differ for {parameter.source_name!r}"
                )
        if parameter.output_key is not None and row.get("output_key") is not None:
            if str(row["output_key"]) != parameter.output_key:
                raise ValueError(
                    f"CUDA metadata output identity differs for {parameter.source_name!r}"
                )
        if parameter.kind == "opaque":
            descriptor = descriptors.get(parameter.source_name)
            if (
                descriptor is None
                or descriptor.semantic_name != parameter.semantic_name
                or descriptor.logical_dtype != parameter.logical_dtype
                or descriptor.shape_contract != parameter.shape_contract
            ):
                raise ValueError(
                    f"CUDA TensorMap descriptor differs for {parameter.source_name!r}"
                )

    if logical_parameters:
        logical_by_semantic = {item.semantic_name: item for item in logical_parameters}
        consumed: set[str] = set()
        for parameter in physical_parameters:
            logical = logical_by_semantic.get(parameter.semantic_name)
            if logical is None:
                raise ValueError(
                    f"CUDA physical parameter {parameter.source_name!r} lacks logical projection"
                )
            if parameter.kind == "opaque":
                expected_kind = "pointer"
                expected_mode = "input"
                expected_dtype = parameter.logical_dtype
            else:
                expected_kind = parameter.kind
                expected_mode = parameter.mode
                expected_dtype = parameter.dtype
            if (
                logical.kind != expected_kind
                or logical.mode != expected_mode
                or logical.dtype != expected_dtype
                or logical.rank != parameter.rank
                or logical.shape != parameter.shape
                or logical.shape_contract != parameter.shape_contract
            ):
                raise ValueError(
                    f"CUDA logical interface differs for {parameter.source_name!r}"
                )
            consumed.add(logical.semantic_name)
        if consumed != {item.semantic_name for item in logical_parameters}:
            raise ValueError("CUDA logical interface contains an unpaired parameter")


def _cuda_lift_projection(
    code: str,
    symbol: str,
    *,
    source_kind: str,
    origin_location: str,
) -> DatasetSourceProjection:
    """Return the one verified CUDA device entry used by the Lift owner.

    The complete CUDA module remains the Gate compiler input.  Lift receives a
    source projection containing only the selected ``__global__`` entry, and
    the same projection object is carried into the manifest.  Keeping the
    bytes, declaration order, and source-slice digest on one object prevents a
    full-module digest from being paired with a different Lift input.
    """

    declarations, body = _cuda_definition(code, symbol)
    projected = (
        "__global__ void "
        + symbol
        + "("
        + ", ".join(declarations)
        + ") {\n"
        + body
        + "\n}\n"
    )
    return DatasetSourceProjection(
        projected,
        symbol,
        tuple(_cuda_declaration_name(item) for item in declarations),
        "cuda",
        source_kind,
        origin_location,
    )


def prepare_frontend_dataset_workload(
    adapter: FrontendDatasetHarness,
    *,
    dataset_root: str | Path | None = None,
) -> PreparedFrontendDatasetWorkload:
    catalog = load_frontend_dataset(dataset_root)
    validation = catalog.validate()
    if not validation.passed:
        raise ValueError(
            "multi-frontend Dataset validation failed: "
            + "; ".join(validation.errors)
        )
    kernel = catalog.by_id.get(adapter.case_id)
    if kernel is None:
        raise ValueError(f"Dataset case {adapter.case_id!r} is unavailable")
    if kernel.frontend != adapter.frontend:
        raise ValueError("Dataset catalog frontend differs from its Harness")
    source_code = kernel.path.read_text(encoding="utf-8")
    actual_sha256 = hashlib.sha256(kernel.path.read_bytes()).hexdigest()
    if actual_sha256 != adapter.source_sha256.lower():
        raise ValueError(
            f"Dataset source digest differs for {adapter.case_id}: "
            f"expected {adapter.source_sha256}, got {actual_sha256}"
        )
    dimensions = adapter.dimension_map
    source_schedule = {
        f"launch_block_{axis}": extent
        for axis, extent in zip(("x", "y", "z"), adapter.block, strict=True)
        if extent != 1
    }
    if adapter.frontend == "tilelang":
        metadata_kernel = _generated_companion(kernel, catalog.kernels)
        projection = _materialize_tilelang_projection(
            kernel, adapter.factory_kwarg_map
        )
        parameters = _metadata_parameters(
            metadata_kernel.metadata,
            dimensions=dimensions,
            scalars=adapter.scalar_map,
            source_names=projection.parameters,
            include_runtime_scalars=False,
            use_logical_interface=True,
        )
        gate_parameters = _logical_abi_parameters(
            parameters, adapter.tensor_map_descriptors
        )
        physical_parameters = _metadata_parameters(
            metadata_kernel.metadata,
            dimensions=dimensions,
            scalars=adapter.scalar_map,
            include_runtime_scalars=True,
            use_logical_interface_dtypes=True,
        )
        physical_sidecar = _tensor_map_sidecars().get(metadata_kernel.case_id)
        _validate_cuda_metadata_admission(
            metadata_kernel,
            metadata_kernel.metadata,
            physical_parameters,
            logical_parameters=parameters,
            tensor_map_descriptors=(
                physical_sidecar.descriptors if physical_sidecar is not None else ()
            ),
        )
        gate_source: Any = TileLangKernelSource.from_factory_file(
            kernel.path,
            kernel.symbol,
            factory_kwargs=adapter.factory_kwarg_map,
        )
        metadata = metadata_kernel.metadata
    else:
        metadata = _runtime_metadata(kernel)
        parameters = _metadata_parameters(
            metadata,
            dimensions=dimensions,
            scalars=adapter.scalar_map,
            include_runtime_scalars=True,
        )
        gate_parameters = _logical_abi_parameters(
            parameters, adapter.tensor_map_descriptors
        )
        _validate_cuda_metadata_admission(
            kernel,
            metadata,
            parameters,
            logical_parameters=gate_parameters,
            tensor_map_descriptors=adapter.tensor_map_descriptors,
        )
        projection = DatasetSourceProjection(
            source_code,
            kernel.symbol,
            tuple(item.source_name for item in parameters),
            "cuda",
            kernel.source_kind,
            kernel.location,
        )
        gate_source = CudaKernelSource.from_file(kernel.path, kernel.symbol)
    if (
        adapter.descriptor_sha256
        and _descriptor_sha256(metadata) != adapter.descriptor_sha256.lower()
    ):
        raise ValueError(
            f"Dataset invocation descriptor differs for {adapter.case_id}"
        )
    launch_metadata = (
        _metadata_with_tensor_map_launch(metadata, adapter.grid_contract)
        if adapter.tensor_map_sidecar_sha256
        else metadata
    )
    discovered_grid, discovered_block, discovered_contract = _launch_from_metadata(
        launch_metadata, dimensions
    )
    if discovered_grid != adapter.grid or discovered_block != adapter.block:
        raise ValueError(
            "Dataset Harness launch differs from the captured compiler metadata"
        )
    if discovered_contract != adapter.grid_contract:
        raise ValueError("Dataset Harness grid contract differs from metadata")
    shape_coordinates = _shape_coordinate_mapping(gate_parameters)
    compiler_view = SCRCCompilerView(
        schedule_values=source_schedule,
        compiler_slots=tuple(source_schedule),
        arguments=tuple(
            item.compiler_argument(shape_coordinates) for item in gate_parameters
        ),
        grid=adapter.grid,
        grid_contract=adapter.grid_contract,
        grid_shape_coordinates=_grid_shape_coordinates(
            adapter.grid_contract,
            shape_coordinates,
        ),
        parameter_order=tuple(item.source_name for item in parameters),
    )
    source_to_opaque = _stable_dataset_source_mapping(
        projection.parameters,
        tuple(compiler_view.compiler_slots),
    )
    # Harness execution receives the same c* bindings declared by SourceFacts;
    # it must not independently allocate a schedule namespace.
    opaque_schedule = {
        source_to_opaque[name]: value for name, value in source_schedule.items()
    }
    if adapter.frontend == "tilelang":
        gate_case: Any = TileLangCompileCase(
            tuple(item.frontend_parameter() for item in gate_parameters),
            adapter.grid,
            adapter.block,
            opaque_schedule,
        )
    else:
        gate_case = CudaCompileCase(
            tuple(item.frontend_parameter() for item in gate_parameters),
            adapter.grid,
            adapter.block,
            opaque_schedule,
            adapter.cuda_arch,
            include_dirs=resolve_cuda_include_dirs(kernel),
            physical_parameters=tuple(
                item.frontend_parameter() for item in parameters
            ),
        )
    if adapter.frontend == "tilelang":
        _anonymous, _binding, python_mapping = anonymize_source(
            projection.code,
            source_symbol=projection.function,
        )
        expected_runtime_mapping = {
            name: source_to_opaque[name] for name in projection.parameters
        }
        if any(
            python_mapping.get(name) != opaque
            for name, opaque in expected_runtime_mapping.items()
        ):
            raise ValueError(
                "materialized TileLang source ABI does not match the stable Dataset mapping"
            )
        lift_source = projection.code
        lift_source_kind = "python"
    else:
        projection = _cuda_lift_projection(
            source_code,
            projection.function,
            source_kind=kernel.source_kind,
            origin_location=kernel.location,
        )
        lift_source = projection.code
        lift_source_kind = "cuda"
    manifest = build_generated_source_manifest(
        source=source_code,
        source_slice=projection,
        source_location=kernel.location,
        view=compiler_view,
        source_arguments=_dataset_manifest_arguments(parameters, compiler_view),
        tensor_map_relations=(
            _dataset_tensor_map_relations(
                parameters,
                adapter.tensor_map_descriptors,
                compiler_view,
            )
            if adapter.frontend == "cuda"
            else ()
        ),
    )
    runtime_mapping = {
        item.source_name: source_to_opaque[item.source_name]
        for item in parameters
    }
    profile = observer_profile_for(Frontend(adapter.frontend))
    entry = DatasetWorkloadEntry(
        kernel.case_id,
        kernel.frontend,
        kernel.symbol,
        kernel.location,
        f"{profile.profile_id}@{profile.profile_version}",
        adapter.gate2_policy,
    )
    request_source = SourceProgram(
        lift_source,
        gate_source,
        lift_source_kind=lift_source_kind,
        lift_source_symbol=projection.function,
        lift_parameter_order=projection.parameters,
        lift_constexpr_parameters=projection.constexpr_parameters,
        lift_source_to_opaque=tuple(source_to_opaque.items())
        if lift_source_kind == "cuda"
        else (),
    )
    source_facts = SourceFacts.from_compiler_view(
        compiler_view,
        frontend=adapter.frontend,
        source_kind=lift_source_kind,
        source_symbol=projection.function,
        source_to_opaque=source_to_opaque,
        source_digest=hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
        source_slice_digest=hashlib.sha256(lift_source.encode("utf-8")).hexdigest(),
    )
    construction = CanonicalConstruction.from_source_facts(
        source_facts,
        source_text=lift_source,
    )
    request = SCRCRunRequest(
        request_source,
        HarnessBindingInstance.create(
            opaque_schedule,
            gate1_case=gate_case,
            slot_order=tuple(opaque_schedule),
            gate2_context={
                "test_mode": "finite",
                "seed": adapter.seed,
                "adapter_protocol": adapter.protocol,
                "dataset_case_id": adapter.case_id,
            },
        ),
        adapter.frontend,
        source_facts,
        construction,
    )
    return PreparedFrontendDatasetWorkload(
        adapter,
        entry,
        kernel.location,
        projection,
        compiler_view,
        request,
        runtime_mapping,
        parameters,
        adapter.grid,
        adapter.block,
        manifest,
    )


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    aliases = {
        "bf16": "float32",
        "f16": "float16",
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
    try:
        return np.dtype(aliases.get(dtype, dtype))
    except TypeError as exc:
        raise ValueError(f"unsupported Dataset tensor dtype {dtype!r}") from exc


def _random_input(
    rng: np.random.Generator, parameter: DatasetRuntimeParameter
) -> np.ndarray:
    return _random_array(rng, parameter.dtype, parameter.shape)


def _random_array(
    rng: np.random.Generator,
    dtype_name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    dtype = _numpy_dtype(dtype_name)
    if dtype.kind == "b":
        return rng.integers(0, 2, size=shape).astype(dtype)
    if dtype.kind in {"i", "u"}:
        # Non-negative bounded integer fixtures are valid arithmetic/mask data
        # and, unlike the old negative sample range, are also memory-safe for
        # index tensors.  This is dtype-level fixture policy; no parameter name
        # or kernel family is inspected.
        return rng.integers(0, 4, size=shape).astype(dtype)
    return rng.normal(0.0, 0.25, size=shape).astype(dtype)


def _bind_shape(
    declared: list[int | str],
    actual: tuple[int, ...],
    dimensions: dict[str, int],
    *,
    dynamic_output: bool = False,
) -> None:
    if len(declared) != len(actual):
        raise ValueError("accepted J rank differs from the Dataset Harness")
    for expected, value in zip(declared, actual, strict=True):
        if expected == "?":
            if not dynamic_output:
                raise ValueError(
                    "J data-dependent extent requires an output-only binding; "
                    "dynamic input/inout buffers are outside the current ABI"
                )
            continue
        if isinstance(expected, int):
            if expected != value:
                raise ValueError("accepted J static extent differs from Dataset")
        elif dimensions.setdefault(expected, value) != value:
            raise ValueError("accepted J symbolic extent is inconsistent")


def build_frontend_dataset_gate2_suite(
    prepared: PreparedFrontendDatasetWorkload,
    artifact: Gate1Artifact,
    *,
    seed: int | None = None,
    case_id: str = "finite-hidden-1",
    test_mode: str = "finite_hidden_after_gate1",
) -> Gate2Suite:
    """Create one finite native Source/Replay/J case from an explicit valuation."""

    prepared = prepared.with_contract(artifact.j, artifact.c)
    mapping = dict(prepared.source_to_opaque)
    runtime = {mapping[item.source_name]: item for item in prepared.runtime_parameters}
    j_inputs = {item.name: item for item in artifact.j.intent.inputs}
    j_outputs = {item.name: item for item in artifact.j.intent.outputs}
    rng = np.random.default_rng(prepared.adapter.seed if seed is None else seed)
    values: dict[str, Any] = {}
    dimensions: dict[str, int] = {}
    input_bindings: dict[str, str] = {}
    output_bindings: dict[str, str] = {}
    observed_output_shapes: dict[str, tuple[int, ...]] = {}
    elaboration = j_elaboration(artifact.j)
    canonical_outputs = elaboration.output_bindings
    canonical_output_by_name = elaboration.output_by_name
    # ``arguments`` is the logical ABI consumed by Replay.  Source arguments
    # may replace selected logical inputs with physical CUtensorMap values.
    arguments: list[NativeArgument] = []
    source_arguments: list[NativeArgument] = []
    tensor_maps = {
        item.parameter: item for item in prepared.adapter.tensor_map_descriptors
    }

    for parameter in artifact.j.signature.parameters:
        binding = runtime.get(parameter.name)
        if binding is None:
            raise ValueError(f"accepted J parameter {parameter.name!r} lacks Dataset ABI")
        if parameter.kind is ParameterKind.CONSTEXPR:
            raise ValueError("native Dataset frontends do not expose constexpr parameters")
        if parameter.kind is ParameterKind.SCALAR:
            if binding.kind != "scalar" or binding.value is None:
                raise ValueError("accepted J scalar differs from Dataset ABI")
            key = f"scalar_{parameter.name}"
            values[key] = binding.value
            logical_ref = parameter.logical_ref
            if logical_ref in j_inputs:
                tensor = j_inputs[logical_ref]
                if tensor.shape:
                    raise ValueError("Dataset scalar may bind only a rank-0 J tensor")
                logical_key = f"logical_{parameter.name}"
                values[logical_key] = np.asarray(
                    binding.value, dtype=_numpy_dtype(tensor.dtype)
                )
                input_bindings[logical_ref] = logical_key
            elif logical_ref in j_outputs:
                raise ValueError("Dataset scalar cannot bind a J output")
            elif logical_ref is not None:
                if isinstance(binding.value, bool) or not isinstance(binding.value, int):
                    raise ValueError("J dimension-bound Dataset scalar must be integral")
                previous = dimensions.setdefault(logical_ref, int(binding.value))
                if previous != int(binding.value):
                    raise ValueError(
                        "J dimension-bound Dataset scalar is inconsistent"
                    )
            native_scalar = NativeArgument(
                NativeArgumentRole.SCALAR,
                value_key=key,
                dtype=binding.dtype,
            )
            arguments.append(native_scalar)
            source_arguments.append(native_scalar)
            continue
        if parameter.kind is ParameterKind.OPAQUE:
            raise ValueError(
                "accepted J exposes a physical opaque ABI; TensorMap must be logicalized"
            )
        if parameter.kind not in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            raise ValueError("unsupported accepted J Dataset parameter kind")
        if binding.kind == "opaque":
            descriptor = tensor_maps.get(binding.source_name)
            logical_ref = parameter.logical_ref
            tensor = j_inputs.get(logical_ref)
            if (
                parameter.kind is not ParameterKind.POINTER
                or parameter.mode is not ParameterMode.INPUT
                or parameter.rank != 2
                or descriptor is None
                or descriptor.semantic_name != binding.semantic_name
                or descriptor.shape_contract != binding.shape_contract
                or tensor is None
                or parameter.dtype != descriptor.logical_dtype
                or tensor.dtype != descriptor.logical_dtype
            ):
                raise ValueError(
                    "accepted J logical pointer differs from TensorMap sidecar"
                )
            _bind_shape(tensor.shape, binding.shape, dimensions)
            key = f"input_{parameter.name}"
            values[key] = _random_array(
                rng, descriptor.logical_dtype, binding.shape
            )
            input_bindings[logical_ref] = key
            arguments.append(
                NativeArgument(
                    NativeArgumentRole.INPUT,
                    value_key=key,
                    dtype=descriptor.logical_dtype,
                    shape=binding.shape,
                    strides=binding.strides,
                )
            )
            source_arguments.append(
                NativeArgument(
                    NativeArgumentRole.TENSOR_MAP,
                    value_key=key,
                    dtype=descriptor.logical_dtype,
                    shape=binding.shape,
                    tensor_map=NativeTensorMapSpec(
                        descriptor.box_dim,
                        swizzle=descriptor.swizzle,
                    ),
                )
            )
            continue
        if binding.kind != "pointer" or parameter.mode.value != binding.mode:
            raise ValueError("accepted J pointer mode differs from Dataset ABI")
        logical_ref = parameter.logical_ref
        tensor = (
            j_inputs[logical_ref]
            if logical_ref in j_inputs
            else j_outputs.get(logical_ref)
        )
        if tensor is None:
            raise ValueError("Dataset pointer must anchor a J input/output tensor")
        output_binding = canonical_output_by_name.get(logical_ref)
        binding_type = output_binding.output_type if output_binding is not None else tensor
        if binding_type.dtype != binding.dtype:
            raise ValueError("accepted J tensor dtype differs from Dataset ABI")
        _bind_shape(
            binding_type.shape,
            binding.shape,
            dimensions,
            dynamic_output=parameter.mode is ParameterMode.OUTPUT,
        )
        if parameter.mode is ParameterMode.INPUT:
            if logical_ref not in j_inputs:
                raise ValueError("Dataset input pointer does not bind a J input")
            key = f"input_{parameter.name}"
            values[key] = _random_input(rng, binding)
            input_bindings[logical_ref] = key
            arguments.append(
                NativeArgument(
                    NativeArgumentRole.INPUT,
                    value_key=key,
                    dtype=binding.dtype,
                    shape=binding.shape,
                    strides=binding.strides,
                )
            )
        elif parameter.mode is ParameterMode.OUTPUT:
            if logical_ref not in j_outputs:
                raise ValueError("Dataset output pointer does not bind a J output")
            internal_key = f"output_{parameter.name}"
            output_key = binding.output_key
            if output_key is None:
                raise ValueError(
                    f"Dataset output {binding.source_name!r} lacks stable output identity"
                )
            output_bindings[logical_ref] = output_key
            observed_output_shapes[logical_ref] = tuple(binding.shape)
            arguments.append(
                NativeArgument(
                    NativeArgumentRole.OUTPUT,
                    dtype=binding.dtype,
                    shape=binding.shape,
                    # ``internal_key`` is intentionally not used as the
                    # observable name; Gate2 output identity comes from the
                    # source manifest binding.
                    value_key=internal_key,
                    output_key=output_key,
                    strides=binding.strides,
                )
            )
        else:
            if logical_ref not in j_inputs or logical_ref not in j_outputs:
                raise ValueError("Dataset inout pointer must bind one J inout tensor")
            input_key = f"input_{parameter.name}"
            output_key = binding.output_key
            if output_key is None:
                raise ValueError(
                    f"Dataset inout {binding.source_name!r} lacks stable output identity"
                )
            values[input_key] = _random_input(rng, binding)
            input_bindings[logical_ref] = input_key
            output_bindings[logical_ref] = output_key
            observed_output_shapes[logical_ref] = tuple(binding.shape)
            arguments.append(
                NativeArgument(
                    NativeArgumentRole.INOUT,
                    value_key=input_key,
                    dtype=binding.dtype,
                    shape=binding.shape,
                    output_key=output_key,
                    strides=binding.strides,
                )
            )
        source_arguments.append(arguments[-1])
    if set(input_bindings) != set(j_inputs):
        raise ValueError("Dataset Harness input bindings differ from accepted J")
    if set(output_bindings) != set(j_outputs):
        raise ValueError("Dataset Harness output bindings differ from accepted J")
    if tuple(item.output for item in canonical_outputs) != tuple(
        item.name for item in artifact.j.intent.outputs
    ):
        raise ValueError(
            "J output declarations differ from canonical output ABI order"
        )
    values = materialize_ragged_input_values(
        artifact.j,
        values,
        input_bindings,
    )
    runtime_inputs = [
        np.asarray(values[input_bindings[item.name]])
        for item in artifact.j.intent.inputs
    ]
    resolved_output_shapes = resolve_runtime_output_shapes(
        artifact.j.intent,
        [list(item.output_type.shape) for item in canonical_outputs],
        runtime_inputs,
        dims=dimensions,
        canonical_output_bindings=canonical_outputs,
    )
    for output, resolved_shape in zip(
        (j_outputs[item.output] for item in canonical_outputs),
        resolved_output_shapes,
        strict=True,
    ):
        observed_shape = observed_output_shapes.get(output.name)
        if observed_shape is None:
            raise ValueError(
                f"Dataset did not provide a concrete output shape for {output.name!r}"
            )
        if observed_shape != resolved_shape:
            raise ValueError(
                f"Dataset output shape for {output.name!r} differs from the "
                f"J-derived runtime shape: {observed_shape} != {resolved_shape}"
            )
    refeval = ScalarLoweringSpec.create(
        artifact.j,
        dims=dimensions,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    source_program_sha256, source_execution_program_sha256 = program_identity_digests(
        prepared.request.source.gate_source,
        frontend=prepared.adapter.frontend,
    )
    execution_lineage = prepared.manifest.execution_lineage()
    execution_lineage.update(
        {
            "source_program_sha256": source_program_sha256,
            "source_execution_program_sha256": source_execution_program_sha256,
        }
    )
    execution = NativeExecutionSpec(
        tuple(arguments),
        prepared.request.binding.gate1_case,
        prepared.grid,
        prepared.block,
        metadata={
            "adapter_protocol": prepared.adapter.protocol,
            "dataset_case_id": prepared.adapter.case_id,
            **execution_lineage,
        },
        source_arguments=tuple(source_arguments),
        source_block=prepared.adapter.physical_block,
        source_shared_mem=(
            prepared.adapter.dynamic_shared_memory_bytes
            if prepared.adapter.tensor_map_descriptors
            else None
        ),
    )
    output_keys = tuple(output_bindings[item.output] for item in canonical_outputs)
    case = Gate2Case(
        case_id,
        refeval.j,
        values,
        execution,  # Gate2Case retains the historical field name `triton`.
        refeval,
        output_keys,
        {
            "test_mode": test_mode,
            "adapter_protocol": prepared.adapter.protocol,
            "frontend": prepared.adapter.frontend,
            **execution_lineage,
        },
    )
    return Gate2Suite(
        prepared.adapter.case_id,
        (case,),
        prepared.request.binding.c_values,
    )


FRONTEND_DATASET_HARNESSES: dict[str, FrontendDatasetHarness] = {
    "tilelang:add2d": FrontendDatasetHarness(
        "tilelang:add2d",
        "tilelang",
        (("M", 4),),
        (),
        (4, 1, 1),
        (16, 1, 1),
        ("M", "1", "1"),
        401,
        source_sha256="3beb7fe73950ba50986bb0e8ee7af608a4c92c9d11c6710306414bcb42584733",
    ),
    "cuda:tilelang_add2d": FrontendDatasetHarness(
        "cuda:tilelang_add2d",
        "cuda",
        (("M", 4),),
        (("M", 4),),
        (4, 1, 1),
        (16, 1, 1),
        ("M", "1", "1"),
        409,
        source_sha256="2ffca814f59525234c27ee419e6856dd6488f94d1d32354977407ae43b772a81",
    ),
}


def _stable_dataset_seed(case_id: str, frontend: str) -> int:
    digest = hashlib.sha256(
        f"{FRONTEND_DATASET_PROJECTION_PROTOCOL}:{frontend}:{case_id}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _descriptor_sha256(metadata: Mapping[str, Any]) -> str:
    """Pin the ABI/launch facts consumed by a projected Harness."""

    payload = {
        "cuda_abi": metadata.get("cuda_abi"),
        "interface": metadata.get("interface"),
        "launch": metadata.get("launch"),
        "target": metadata.get("target"),
        "include_roots": metadata.get("include_roots"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _DatasetTensorMapSidecar:
    cuda_arch: str
    gate2_policy: str
    source_sha256: str
    dimensions: tuple[tuple[str, int], ...]
    grid_contract: tuple[str, str, str]
    physical_block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
    descriptors: tuple[DatasetTensorMapDescriptor, ...]
    digest: str


@lru_cache(maxsize=1)
def _tensor_map_sidecars() -> Mapping[str, _DatasetTensorMapSidecar]:
    path = Path(__file__).with_name("frontend_tensor_map_sidecars.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "intentir.frontend-dataset.cuda-tensor-map-sidecars.v1":
        raise ValueError("unsupported CUDA TensorMap sidecar protocol")
    profile = payload.get("profile")
    expected_profile = {
        "cuda_arch": "sm_120a",
        "gate2_policy": "intentir.gate2.tf32-tensorcore.v1",
        "descriptor_rank": 2,
        "element_strides": [1, 1],
        "interleave": "none",
        "swizzle": "per_descriptor",
        "l2_promotion": "128b",
        "float_oob_fill": "none",
    }
    if profile != expected_profile:
        raise ValueError("CUDA TensorMap sidecar profile differs from the closed runner")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Mapping):
        raise ValueError("CUDA TensorMap sidecar cases must be an object")
    result: dict[str, _DatasetTensorMapSidecar] = {}
    for case_id, raw in raw_cases.items():
        if not isinstance(case_id, str) or not case_id.startswith("cuda:"):
            raise ValueError("CUDA TensorMap sidecar case id is malformed")
        if not isinstance(raw, Mapping):
            raise ValueError("CUDA TensorMap sidecar case must be an object")
        source_sha256 = str(raw.get("source_sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
            raise ValueError("CUDA TensorMap sidecar source digest is malformed")
        raw_dimensions = raw.get("dimensions")
        if not isinstance(raw_dimensions, Mapping):
            raise ValueError("CUDA TensorMap sidecar dimensions must be an object")
        dimensions = tuple(
            sorted((str(name), int(value)) for name, value in raw_dimensions.items())
        )
        if any(not name.isidentifier() or value <= 0 for name, value in dimensions):
            raise ValueError("CUDA TensorMap sidecar dimensions are malformed")
        grid_contract = tuple(str(value) for value in raw.get("grid_contract", ()))
        if len(grid_contract) != 3 or any(not value for value in grid_contract):
            raise ValueError("CUDA TensorMap sidecar grid contract is malformed")
        for expression in grid_contract:
            _safe_integer_expression(expression, dict(dimensions))
        physical_block = tuple(int(value) for value in raw.get("physical_block", ()))
        if len(physical_block) != 3 or any(value <= 0 for value in physical_block):
            raise ValueError("CUDA TensorMap sidecar physical block is malformed")
        shared = raw.get("dynamic_shared_memory_bytes")
        if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
            raise ValueError("CUDA TensorMap sidecar shared-memory size is malformed")
        raw_descriptors = raw.get("descriptors")
        if not isinstance(raw_descriptors, list) or not raw_descriptors:
            raise ValueError("CUDA TensorMap sidecar requires descriptor rows")
        descriptors = tuple(
            DatasetTensorMapDescriptor(
                parameter=str(item.get("parameter") or ""),
                semantic_name=str(item.get("semantic_name") or ""),
                logical_dtype=str(item.get("logical_dtype") or ""),
                shape_contract=tuple(item.get("shape") or ()),
                box_dim=tuple(item.get("box_dim") or ()),
                swizzle=str(item.get("swizzle") or ""),
            )
            for item in raw_descriptors
            if isinstance(item, Mapping)
        )
        if len(descriptors) != len(raw_descriptors):
            raise ValueError("CUDA TensorMap sidecar descriptor row is malformed")
        canonical = {"profile": profile, "case_id": case_id, "case": raw}
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        result[case_id] = _DatasetTensorMapSidecar(
            str(profile["cuda_arch"]),
            str(profile["gate2_policy"]),
            source_sha256,
            dimensions,
            grid_contract,  # type: ignore[arg-type]
            physical_block,  # type: ignore[arg-type]
            shared,
            descriptors,
            digest,
        )
    return result


def _metadata_with_tensor_map_launch(
    metadata: Mapping[str, Any],
    grid_contract: tuple[str, str, str],
) -> dict[str, Any]:
    result = dict(metadata)
    launch = dict(metadata.get("launch") or {})
    for axis, expression in zip(("x", "y", "z"), grid_contract, strict=True):
        launch[f"blockIdx.{axis}"] = expression
    result["launch"] = launch
    return result


def _validate_tensor_map_sidecar(
    kernel: FrontendDatasetKernel,
    metadata: Mapping[str, Any],
    sidecar: _DatasetTensorMapSidecar,
) -> None:
    actual_sha256 = hashlib.sha256(kernel.path.read_bytes()).hexdigest()
    if actual_sha256 != sidecar.source_sha256:
        raise ValueError("CUDA TensorMap sidecar source digest differs")
    rows = [
        row
        for row in _ordered_abi_rows(metadata)
        if str(row.get("kind")) == "opaque"
    ]
    descriptors = {item.parameter: item for item in sidecar.descriptors}
    if {str(row.get("name")) for row in rows} != set(descriptors):
        raise ValueError("CUDA TensorMap sidecar does not cover the opaque ABI")
    interface = metadata.get("interface")
    logical_rows = interface.get("parameters") if isinstance(interface, Mapping) else None
    if not isinstance(logical_rows, list):
        raise ValueError("CUDA TensorMap sidecar lacks a logical interface")
    logical_by_name = {
        str(row.get("name")): row for row in logical_rows if isinstance(row, Mapping)
    }
    for row in rows:
        name = str(row.get("name"))
        descriptor = descriptors[name]
        semantic_name = str(row.get("semantic_name") or "")
        logical = logical_by_name.get(semantic_name)
        if (
            descriptor.semantic_name != semantic_name
            or descriptor.shape_contract != _source_shape_contract(row.get("shape"))
            or logical is None
            or descriptor.logical_dtype
            != _normalize_dataset_dtype(logical.get("dtype"))
        ):
            raise ValueError("CUDA TensorMap sidecar differs from Dataset metadata")


def _canonical_for_kernel(
    kernel: FrontendDatasetKernel,
    snapshots: Mapping[str, Mapping[str, int]],
) -> Mapping[str, int]:
    # Snapshot names are an optional invocation-value cache.  Exact stem
    # matching is the only accepted association; aliases are deliberately not
    # guessed from operator names.
    if kernel.frontend == "cuda" and kernel.metadata.get("origin") != "tilelang_export":
        return {}
    return snapshots.get(kernel.path.stem, {})


def _project_tilelang_harness(
    kernel: FrontendDatasetKernel,
    companion: FrontendDatasetKernel | None,
    *,
    snapshots: Mapping[str, Mapping[str, int]],
) -> FrontendDatasetHarness:
    if companion is None:
        raise FrontendDatasetCaseUnavailableError(
            "TileLang Dataset case lacks its generated CUDA ABI companion"
        )
    abi = companion.metadata.get("cuda_abi")
    rows = abi.get("parameters") if isinstance(abi, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("TileLang Dataset case lacks structured CUDA ABI metadata")
    has_tensor_maps = any(
        isinstance(row, Mapping) and row.get("kind") == "opaque" for row in rows
    )
    sidecar = _tensor_map_sidecars().get(companion.case_id) if has_tensor_maps else None
    if has_tensor_maps and sidecar is None:
        raise FrontendDatasetCaseUnavailableError(
            "TileLang Dataset opaque companion lacks an explicit descriptor sidecar"
        )
    if sidecar is not None:
        _validate_tensor_map_sidecar(companion, companion.metadata, sidecar)
    factory_kwargs = _literal_factory_kwargs(kernel)
    canonical_shapes = dict(_canonical_for_kernel(kernel, snapshots))
    if sidecar is not None:
        canonical_shapes.update(dict(sidecar.dimensions))
    descriptor_metadata = (
        _metadata_with_tensor_map_launch(companion.metadata, sidecar.grid_contract)
        if sidecar is not None
        else companion.metadata
    )
    dimensions, grid, block, contract, _scalars = _dataset_descriptor_values(
        descriptor_metadata,
        canonical_shapes=canonical_shapes,
    )
    return FrontendDatasetHarness(
        kernel.case_id,
        "tilelang",
        dimensions,
        (),
        grid,
        block,
        contract,
        _stable_dataset_seed(kernel.case_id, "tilelang"),
        factory_kwargs=factory_kwargs,
        source_sha256=hashlib.sha256(kernel.path.read_bytes()).hexdigest(),
        descriptor_sha256=_descriptor_sha256(companion.metadata),
        tensor_map_sidecar_sha256=sidecar.digest if sidecar is not None else "",
        gate2_policy=(
            sidecar.gate2_policy
            if sidecar is not None
            else DEFAULT_GATE2_POLICY.policy_id
        ),
    )


def _project_cuda_harness(
    kernel: FrontendDatasetKernel,
    *,
    snapshots: Mapping[str, Mapping[str, int]],
) -> FrontendDatasetHarness:
    metadata = _runtime_metadata(kernel)
    abi = metadata.get("cuda_abi")
    rows = abi.get("parameters") if isinstance(abi, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("CUDA Dataset case lacks structured ABI metadata")
    has_tensor_maps = any(
        isinstance(row, Mapping) and row.get("kind") == "opaque" for row in rows
    )
    sidecar = _tensor_map_sidecars().get(kernel.case_id) if has_tensor_maps else None
    if has_tensor_maps and sidecar is None:
        raise FrontendDatasetCaseUnavailableError(
            "CUDA Dataset opaque ABI lacks an explicit descriptor sidecar"
        )
    if sidecar is not None:
        _validate_tensor_map_sidecar(kernel, metadata, sidecar)
    canonical_shapes = dict(_canonical_for_kernel(kernel, snapshots))
    if sidecar is not None:
        canonical_shapes.update(dict(sidecar.dimensions))
    descriptor_metadata = (
        _metadata_with_tensor_map_launch(metadata, sidecar.grid_contract)
        if sidecar is not None
        else metadata
    )
    dimensions, grid, block, contract, scalar_values = _dataset_descriptor_values(
        descriptor_metadata,
        canonical_shapes=canonical_shapes,
    )
    target = metadata.get("target")
    arch = (
        sidecar.cuda_arch
        if sidecar is not None
        else (
            str(target.get("arch"))
            if isinstance(target, Mapping) and target.get("arch")
            else "sm_120"
        )
    )
    return FrontendDatasetHarness(
        kernel.case_id,
        "cuda",
        dimensions,
        scalar_values,
        grid,
        block,
        contract,
        _stable_dataset_seed(kernel.case_id, "cuda"),
        cuda_arch=arch,
        source_sha256=hashlib.sha256(kernel.path.read_bytes()).hexdigest(),
        descriptor_sha256=_descriptor_sha256(metadata),
        tensor_map_descriptors=sidecar.descriptors if sidecar is not None else (),
        tensor_map_sidecar_sha256=sidecar.digest if sidecar is not None else "",
        physical_block=sidecar.physical_block if sidecar is not None else None,
        dynamic_shared_memory_bytes=(
            sidecar.dynamic_shared_memory_bytes if sidecar is not None else 0
        ),
        gate2_policy=(
            sidecar.gate2_policy
            if sidecar is not None
            else DEFAULT_GATE2_POLICY.policy_id
        ),
    )


def _diagnostic(
    kernel: FrontendDatasetKernel,
    exc: Exception,
) -> FrontendDatasetHarnessDiagnostic:
    reason = f"{type(exc).__name__}: {exc}".strip()
    if len(reason) > 2048:
        reason = reason[:2045] + "..."
    status = (
        "case_unavailable"
        if isinstance(exc, FrontendDatasetCaseUnavailableError)
        else "harness_invalid"
    )
    return FrontendDatasetHarnessDiagnostic(kernel.case_id, kernel.frontend, status, reason)


def project_frontend_dataset_harnesses(
    dataset_root: str | Path | None = None,
) -> FrontendDatasetHarnessProjection:
    """Project every mechanically complete Dataset entry into a Harness.

    The projection is deliberately conservative: it consumes only exported
    ABI/launch facts, literal factory defaults, and optional concrete snapshot
    extents.  Opaque descriptors, malformed launch expressions, missing
    companions, and non-literal factory defaults remain explicit unavailable
    cases instead of receiving semantic guesses.
    """

    from intentir.verify.frontend_dataset import load_frontend_dataset

    catalog = load_frontend_dataset(dataset_root)
    validation = catalog.validate()
    if not validation.passed:
        raise ValueError(
            "multi-frontend Dataset validation failed: "
            + "; ".join(validation.errors)
        )
    snapshots = _snapshot_canonical_shapes(catalog.root)
    generated_by_origin = {
        item.metadata.get("origin_case_id"): item
        for item in catalog.kernels
        if item.frontend == "cuda"
        and item.metadata.get("origin") == "tilelang_export"
    }
    ready: dict[str, FrontendDatasetHarness] = {}
    diagnostics: list[FrontendDatasetHarnessDiagnostic] = []

    for kernel in catalog.select(frontend="tilelang"):
        try:
            companion = generated_by_origin.get(kernel.case_id)
            ready[kernel.case_id] = _project_tilelang_harness(
                kernel,
                companion,
                snapshots=snapshots,
            )
        except Exception as exc:  # fail closed per case, never fabricate a binding
            diagnostics.append(_diagnostic(kernel, exc))

    for kernel in catalog.select(frontend="cuda"):
        try:
            ready[kernel.case_id] = _project_cuda_harness(
                kernel,
                snapshots=snapshots,
            )
        except Exception as exc:  # fail closed per case, never fabricate a binding
            diagnostics.append(_diagnostic(kernel, exc))

    diagnostics.sort(key=lambda item: (item.frontend, item.case_id))
    return FrontendDatasetHarnessProjection(ready, tuple(diagnostics))


__all__ = [
    "DatasetRuntimeParameter",
    "DatasetSourceProjection",
    "DatasetWorkloadEntry",
    "CuTileConsumerResult",
    "FRONTEND_DATASET_HARNESSES",
    "FRONTEND_DATASET_HARNESS_PROTOCOL",
    "FRONTEND_DATASET_PROJECTION_PROTOCOL",
    "FRONTEND_DATASET_SOURCE_PROTOCOL",
    "FrontendDatasetCaseUnavailableError",
    "FrontendDatasetHarness",
    "FrontendDatasetHarnessDiagnostic",
    "FrontendDatasetHarnessProjection",
    "PreparedFrontendDatasetWorkload",
    "build_frontend_dataset_gate2_suite",
    "project_frontend_dataset_harnesses",
    "run_cutile_consumer",
    "prepare_frontend_dataset_workload",
]
