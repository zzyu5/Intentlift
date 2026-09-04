"""Actual final-J replay Triton execution adapter for Gate2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.triton_compat import normalize_deprecated_triton_dot

from .model import (
    ExecutionBinding,
    ExecutionResult,
    artifact_sha256,
    execution_spec_contract,
    output_binding_digest,
    output_digest,
    program_identity_digests,
)


class ArgumentRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    SCALAR = "scalar"
    NONE = "none"


@dataclass(frozen=True)
class TritonArgument:
    name: str
    role: ArgumentRole
    value_key: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    value: int | float | bool | None = None
    output_key: str | None = None
    strides: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid Triton argument name: {self.name!r}")
        if not isinstance(self.role, ArgumentRole):
            object.__setattr__(self, "role", ArgumentRole(self.role))
        if self.role in {ArgumentRole.INPUT, ArgumentRole.INOUT} and not self.value_key:
            raise ValueError(f"{self.role.value} argument requires value_key")
        if self.role is ArgumentRole.OUTPUT:
            if self.shape is None or self.dtype is None:
                raise ValueError("output argument requires shape and dtype")
            if not self.output_key:
                object.__setattr__(self, "output_key", self.name)
        if self.strides is not None:
            strides = tuple(int(value) for value in self.strides)
            if self.shape is not None and len(strides) != len(self.shape):
                raise ValueError("Triton argument strides must match its rank")
            if any(value < 0 for value in strides):
                raise ValueError("negative Triton strides require a storage offset")
            object.__setattr__(self, "strides", strides)
        if self.role is ArgumentRole.SCALAR and self.value_key is None and self.value is None:
            raise ValueError("scalar argument requires value_key or fixed value")
        if self.role is ArgumentRole.NONE and any(
            item is not None
            for item in (
                self.value_key,
                self.dtype,
                self.shape,
                self.value,
                self.output_key,
                self.strides,
            )
        ):
            raise ValueError("none argument forbids value/dtype/shape/output fields")


def _storage_size(shape: tuple[int, ...], strides: tuple[int, ...]) -> int:
    if len(shape) != len(strides):
        raise ValueError("physical tensor shape/stride ranks differ")
    if not shape:
        return 1
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides, strict=True))


def _physical_offsets(shape: tuple[int, ...], strides: tuple[int, ...]) -> np.ndarray:
    if not shape:
        return np.asarray(0, dtype=np.int64)
    offsets = np.zeros(shape, dtype=np.int64)
    for axis, (extent, stride) in enumerate(zip(shape, strides, strict=True)):
        axis_values = np.arange(extent, dtype=np.int64)
        axis_values = axis_values.reshape((1,) * axis + (extent,) + (1,) * (len(shape) - axis - 1))
        offsets += axis_values * stride
    return offsets


def _input_tensor(
    torch: Any,
    raw: np.ndarray,
    *,
    dtype: Any,
    device: str,
    strides: tuple[int, ...] | None,
) -> Any:
    if strides is None:
        return torch.as_tensor(raw, dtype=dtype, device=device)
    shape = tuple(int(value) for value in raw.shape)
    offsets = _physical_offsets(shape, strides)
    storage = np.zeros(_storage_size(shape, strides), dtype=raw.dtype)
    storage[offsets.reshape(-1)] = raw.reshape(-1)
    if not np.array_equal(storage[offsets], raw, equal_nan=True):
        raise ValueError("logical input values conflict under captured overlapping strides")
    base = torch.as_tensor(storage, dtype=dtype, device=device)
    return torch.as_strided(base, size=shape, stride=strides)


def _output_tensor(
    torch: Any,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    device: str,
    strides: tuple[int, ...] | None,
) -> Any:
    if strides is None:
        return torch.empty(shape, dtype=dtype, device=device)
    base = torch.empty(_storage_size(shape, strides), dtype=dtype, device=device)
    return torch.as_strided(base, size=shape, stride=strides)


@dataclass(frozen=True)
class TritonExecutionSpec:
    arguments: tuple[TritonArgument, ...]
    grid: tuple[int, ...]
    schedule: Mapping[str, int | float | bool | str] = field(default_factory=dict)
    device: str = "cuda"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_binding: ExecutionBinding | None = None

    def __post_init__(self) -> None:
        if not self.arguments:
            raise ValueError("Triton execution spec requires arguments")
        names = [argument.name for argument in self.arguments]
        if len(names) != len(set(names)):
            raise ValueError("duplicate Triton argument names")
        schedule = dict(self.schedule)
        invalid_schedule_names = sorted(
            repr(name)
            for name in schedule
            if not isinstance(name, str) or not name.isidentifier()
        )
        if invalid_schedule_names:
            raise ValueError(
                "invalid Triton schedule names: "
                + ", ".join(invalid_schedule_names)
            )
        overlap = sorted(set(names) & set(schedule))
        if overlap:
            raise ValueError(
                "Triton arguments and schedule overlap: " + ", ".join(overlap)
            )
        object.__setattr__(self, "schedule", schedule)
        metadata = dict(self.metadata)
        if any(not isinstance(name, str) or not name for name in metadata):
            raise ValueError("Triton execution metadata keys must be non-empty strings")
        object.__setattr__(self, "metadata", metadata)
        if not 1 <= len(self.grid) <= 3 or any(int(dim) <= 0 for dim in self.grid):
            raise ValueError("grid must contain one to three positive dimensions")
        if self.execution_binding is not None and not isinstance(
            self.execution_binding, ExecutionBinding
        ):
            raise TypeError("Triton execution_binding must be an ExecutionBinding")


def _torch_dtype(torch: Any, dtype: str) -> Any:
    key = dtype.removeprefix("torch.").lower()
    aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "index": "int64",
        "i1": "bool",
        "u8": "uint8",
        "u16": "uint16",
        "u32": "uint32",
        "ui8": "uint8",
        "ui16": "uint16",
        "ui32": "uint32",
    }
    key = aliases.get(key, key)
    try:
        return getattr(torch, key)
    except AttributeError as exc:
        raise ValueError(f"unsupported torch dtype {dtype!r}") from exc


def _load_module(
    source: KernelSource, temporary: Path
) -> tuple[ModuleType, str, dict[str, Any]]:
    code, origin = source.read_code()
    normalized, frontend_compat = normalize_deprecated_triton_dot(code)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    module_name = f"_intentir_gate2_{source.function}_{digest}_{id(source):x}"
    if source.kind == "module_file" and not frontend_compat["rewrite_count"]:
        assert source.path is not None
        path = Path(source.path).resolve()
    else:
        path = temporary / f"{module_name}.py"
        path.write_text(normalized, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Triton module from {origin}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module, module_name, frontend_compat


def _launch_by_name(
    kernel: Any,
    grid: tuple[int, ...],
    runtime_arguments: Mapping[str, Any],
    schedule: Mapping[str, int | float | bool | str],
) -> None:
    """Bind a complete Triton ABI by name, including interleaved constexprs."""

    runtime = dict(runtime_arguments)
    frozen_schedule = dict(schedule)
    overlap = sorted(set(runtime) & set(frozen_schedule))
    if overlap:
        raise ValueError(
            "Triton runtime arguments and schedule overlap: "
            + ", ".join(overlap)
        )
    declared = getattr(kernel, "arg_names", None)
    if not isinstance(declared, (list, tuple)) or not all(
        isinstance(name, str) and name.isidentifier() for name in declared
    ):
        raise TypeError("Triton replay kernel has no valid arg_names interface")
    declared_names = list(declared)
    if len(declared_names) != len(set(declared_names)):
        raise ValueError("Triton replay kernel has duplicate arg_names")
    unknown_runtime = sorted(set(runtime) - set(declared_names))
    if unknown_runtime:
        raise ValueError(
            "runtime arguments not in Triton replay interface: "
            + ", ".join(unknown_runtime)
        )
    missing = sorted(
        set(declared_names) - set(runtime) - set(frozen_schedule)
    )
    if missing:
        raise ValueError(
            "missing Triton replay interface arguments: " + ", ".join(missing)
        )
    # Triton may canonicalize an interleaved Python signature by placing
    # runtime arguments before constexpr arguments.  Launches below are fully
    # name-keyed, so dictionary/declaration order is not part of the ABI.  The
    # completeness, overlap, and unknown-name checks above remain authoritative.
    launch_kwargs = {**runtime, **frozen_schedule}
    kernel[grid](**launch_kwargs)


def _triton_artifact_parts(value: Any) -> tuple[tuple[str, bytes], ...]:
    """Extract actual backend image bytes from Triton's compiled result."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        parts: list[tuple[str, bytes]] = []
        for item in value:
            parts.extend(_triton_artifact_parts(item))
        return tuple(parts)
    kernel_image = getattr(value, "kernel", None)
    if isinstance(kernel_image, (bytes, bytearray, memoryview)):
        asm = getattr(value, "asm", None)
        kind = "binary"
        if isinstance(asm, Mapping):
            for candidate in ("cubin", "hsaco", "amdgcn", "ptx"):
                if candidate in asm:
                    kind = candidate
                    break
        return ((kind, bytes(kernel_image)),)
    asm = getattr(value, "asm", None)
    if isinstance(asm, Mapping):
        for candidate in ("cubin", "hsaco", "amdgcn", "ptx"):
            payload = asm.get(candidate)
            if isinstance(payload, (bytes, bytearray, memoryview)):
                return ((candidate, bytes(payload)),)
            if isinstance(payload, str) and payload:
                return ((candidate, payload.encode("utf-8")),)
    return ()


def _triton_artifact_metadata(value: Any) -> dict[str, Any]:
    parts = _triton_artifact_parts(value)
    if not parts:
        raise RuntimeError(
            "Triton warmup did not expose a real compiled backend artifact"
        )
    entries = [
        {
            "ordinal": index,
            "kind": kind,
            "digest": artifact_sha256(payload),
            "size": len(payload),
        }
        for index, (kind, payload) in enumerate(parts)
    ]
    if len(entries) == 1:
        artifact_digest = entries[0]["digest"]
        artifact_kind = f"triton.{entries[0]['kind']}"
    else:
        # Autotuner warmup can produce more than one real candidate.  Keep every
        # per-candidate byte digest and bind the deterministic ordered set as the
        # lifecycle artifact identity.
        encoded = json.dumps(
            [(item["kind"], item["digest"], item["size"]) for item in entries],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        artifact_digest = artifact_sha256(encoded)
        artifact_kind = "triton.autotuned_artifacts"
    return {
        "artifact_digest": artifact_digest,
        "artifact_kind": artifact_kind,
        "compiled_artifacts": entries,
    }


class TritonRunner:
    """Load and launch the final-J replay executable independently."""

    def run(
        self,
        leg: str,
        program: KernelSource,
        spec: TritonExecutionSpec,
        values: Mapping[str, Any],
    ) -> ExecutionResult:
        try:
            import torch
            import triton  # noqa: F401

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            binding = spec.execution_binding
            if binding is not None:
                raw_program_digest, executed_program_digest = program_identity_digests(
                    program,
                    frontend="triton",
                )
                binding.validate_runtime(
                    leg=leg,
                    program_source_sha256=raw_program_digest,
                    program_execution_sha256=executed_program_digest,
                    program_entry=program.function,
                    contract=execution_spec_contract(spec, leg=leg),
                )
            with tempfile.TemporaryDirectory(prefix=f"intentir-gate2-{leg}-") as tmp:
                module, module_name, frontend_compat = _load_module(
                    program, Path(tmp)
                )
                try:
                    kernel = getattr(module, program.function)
                    launch_args: dict[str, Any] = {}
                    outputs: dict[str, Any] = {}
                    for argument in spec.arguments:
                        if argument.role is ArgumentRole.NONE:
                            launch_args[argument.name] = None
                            continue
                        if argument.role is ArgumentRole.SCALAR:
                            scalar = (
                                values[argument.value_key]
                                if argument.value_key is not None
                                else argument.value
                            )
                            if isinstance(scalar, np.generic):
                                scalar = scalar.item()
                            launch_args[argument.name] = scalar
                            continue
                        dtype = _torch_dtype(torch, argument.dtype or "float32")
                        if argument.role is ArgumentRole.OUTPUT:
                            tensor = _output_tensor(
                                torch,
                                shape=tuple(argument.shape or ()),
                                dtype=dtype,
                                device=spec.device,
                                strides=argument.strides,
                            )
                            outputs[argument.output_key or argument.name] = tensor
                        else:
                            raw = np.asarray(values[argument.value_key])
                            tensor = _input_tensor(
                                torch,
                                raw,
                                dtype=dtype,
                                device=spec.device,
                                strides=argument.strides,
                            )
                            if argument.role is ArgumentRole.INOUT:
                                outputs[argument.output_key or argument.name] = tensor
                        launch_args[argument.name] = tensor
                    # Warmup is the compile boundary.  It must complete before
                    # launch so the evidence is taken from Triton's actual
                    # CompiledKernel rather than from source text or a cache
                    # label.
                    compiled = kernel.warmup(
                        grid=tuple(int(dim) for dim in spec.grid),
                        **{**launch_args, **dict(spec.schedule)},
                    )
                    artifact_metadata = _triton_artifact_metadata(compiled)
                    _launch_by_name(
                        kernel,
                        tuple(int(dim) for dim in spec.grid),
                        launch_args,
                        spec.schedule,
                    )
                    torch.cuda.synchronize(spec.device)
                    copied = {
                        key: tensor.detach().cpu().numpy().copy()
                        for key, tensor in outputs.items()
                    }
                    source_code, _source_origin = program.read_code()
                    normalized_source, _ = normalize_deprecated_triton_dot(
                        source_code
                    )
                    return ExecutionResult(
                        leg,
                        True,
                        copied,
                        metadata={
                            **dict(spec.metadata),
                            "function": program.function,
                            "grid": list(spec.grid),
                            "binding_protocol": (
                                "triton_name_keyed_complete_abi_v1"
                            ),
                            "frontend_compat": frontend_compat,
                            "program_source_sha256": hashlib.sha256(
                                source_code.encode("utf-8")
                            ).hexdigest(),
                            "program_execution_sha256": hashlib.sha256(
                                normalized_source.encode("utf-8")
                            ).hexdigest(),
                            **artifact_metadata,
                            **(
                                {
                                    "execution_binding": binding.to_json(),
                                    "execution_binding_digest": binding.digest,
                                }
                                if binding is not None
                                else {}
                            ),
                            "output_digest": output_digest(copied),
                            "output_binding_digest": output_binding_digest(copied),
                        },
                    )
                finally:
                    sys.modules.pop(module_name, None)
        except Exception as exc:
            return ExecutionResult(leg, False, error=f"{type(exc).__name__}: {exc}")


class PreparedTritonInvocation:
    """Load and bind once, then expose a no-synchronization launch closure."""

    def __init__(
        self,
        program: KernelSource,
        spec: TritonExecutionSpec,
        values: Mapping[str, Any],
    ) -> None:
        import torch

        self._temporary = tempfile.TemporaryDirectory(
            prefix="intentir-triton-prepared-"
        )
        module, self._module_name, self.frontend_compat = _load_module(
            program, Path(self._temporary.name)
        )
        self._kernel = getattr(module, program.function)
        self._grid = tuple(int(dim) for dim in spec.grid)
        self._schedule = dict(spec.schedule)
        self._device = spec.device
        self._launch_args: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self._mutable_input_resets: list[tuple[Any, Any]] = []
        for argument in spec.arguments:
            if argument.role is ArgumentRole.NONE:
                self._launch_args[argument.name] = None
                continue
            if argument.role is ArgumentRole.SCALAR:
                scalar = (
                    values[argument.value_key]
                    if argument.value_key is not None
                    else argument.value
                )
                if isinstance(scalar, np.ndarray) and scalar.size == 1:
                    scalar = scalar.item()
                elif isinstance(scalar, np.generic):
                    scalar = scalar.item()
                self._launch_args[argument.name] = scalar
                continue
            dtype = _torch_dtype(torch, argument.dtype or "float32")
            if argument.role is ArgumentRole.OUTPUT:
                tensor = _output_tensor(
                    torch,
                    shape=tuple(argument.shape or ()),
                    dtype=dtype,
                    device=spec.device,
                    strides=argument.strides,
                )
                self.outputs[argument.output_key or argument.name] = tensor
            else:
                raw = np.asarray(values[argument.value_key])
                tensor = _input_tensor(
                    torch,
                    raw,
                    dtype=dtype,
                    device=spec.device,
                    strides=argument.strides,
                )
                if argument.role is ArgumentRole.INOUT:
                    self.outputs[argument.output_key or argument.name] = tensor
                    self._mutable_input_resets.append((tensor, tensor.clone()))
            self._launch_args[argument.name] = tensor

    def launch(self) -> None:
        _launch_by_name(
            self._kernel,
            self._grid,
            self._launch_args,
            self._schedule,
        )

    def thunk(self):
        return self.launch

    def reset_mutable_inputs(self) -> None:
        for destination, pristine in self._mutable_input_resets:
            destination.copy_(pristine)

    def copied_outputs(self) -> dict[str, np.ndarray]:
        return {
            key: tensor.detach().cpu().numpy().copy()
            for key, tensor in self.outputs.items()
        }

    def close(self) -> None:
        module_name = getattr(self, "_module_name", None)
        if module_name is not None:
            sys.modules.pop(module_name, None)
            self._module_name = None
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> "PreparedTritonInvocation":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = [
    "ArgumentRole",
    "PreparedTritonInvocation",
    "TritonArgument",
    "TritonExecutionSpec",
    "TritonRunner",
]
