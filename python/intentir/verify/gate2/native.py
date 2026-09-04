"""CUDA and TileLang execution adapters for the three-way Gate2 protocol."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

from intentir.verify.gate1.cuda_compiler import (
    CudaCompileCase,
    CudaCompiler,
    CudaKernelSource,
)
from intentir.verify.gate1.tilelang_compiler import TileLangKernelSource

from .model import (
    ExecutionBinding,
    ExecutionResult,
    artifact_sha256,
    execution_spec_contract,
    output_binding_digest,
    output_digest,
    program_identity_digests,
)


class NativeArgumentRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    SCALAR = "scalar"
    TENSOR_MAP = "tensor_map"


@dataclass(frozen=True)
class NativeTensorMapSpec:
    """Closed rank-2 row-major CUDA TensorMap construction profile.

    Dimensions are in TMA coordinate order (contiguous/minor first), not the
    logical row-major tensor order.  The tensor base address and global extents
    remain runtime values supplied by the logical J input.
    """

    box_dim: tuple[int, int]
    element_strides: tuple[int, int] = (1, 1)
    interleave: str = "none"
    swizzle: str = "128b"
    l2_promotion: str = "128b"
    float_oob_fill: str = "none"

    def __post_init__(self) -> None:
        box = tuple(int(value) for value in self.box_dim)
        strides = tuple(int(value) for value in self.element_strides)
        if len(box) != 2 or any(value <= 0 for value in box):
            raise ValueError("native TensorMap box_dim must contain two positive extents")
        if strides != (1, 1):
            raise ValueError("native TensorMap profile requires unit element strides")
        if self.interleave != "none" or self.swizzle not in {
            "none",
            "32b",
            "64b",
            "128b",
        }:
            raise ValueError("native TensorMap profile has an unsupported swizzle")
        if self.l2_promotion != "128b" or self.float_oob_fill != "none":
            raise ValueError("native TensorMap cache/OOB profile is unsupported")
        object.__setattr__(self, "box_dim", box)
        object.__setattr__(self, "element_strides", strides)


@dataclass(frozen=True)
class NativeArgument:
    role: NativeArgumentRole
    value_key: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    value: int | float | bool | None = None
    output_key: str | None = None
    tensor_map: NativeTensorMapSpec | None = None
    strides: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, NativeArgumentRole):
            object.__setattr__(self, "role", NativeArgumentRole(self.role))
        if self.role in {NativeArgumentRole.INPUT, NativeArgumentRole.INOUT} and not self.value_key:
            raise ValueError(f"{self.role.value} native argument requires value_key")
        if self.role is NativeArgumentRole.OUTPUT:
            if self.shape is None or self.dtype is None:
                raise ValueError("native output requires shape and dtype")
            if self.output_key is None:
                raise ValueError("native output requires an observable output_key")
        if self.role is NativeArgumentRole.SCALAR and self.value_key is None and self.value is None:
            raise ValueError("native scalar requires value_key or fixed value")
        if self.role is NativeArgumentRole.TENSOR_MAP:
            if not self.value_key or self.dtype is None or self.shape is None:
                raise ValueError("native TensorMap requires one logical input tensor")
            if self.tensor_map is None:
                raise ValueError("native TensorMap requires a construction spec")
        elif self.tensor_map is not None:
            raise ValueError("only a TensorMap native argument may carry a descriptor spec")
        if self.strides is not None:
            strides = tuple(int(value) for value in self.strides)
            if self.shape is None or len(strides) != len(self.shape):
                raise ValueError("native tensor strides must match its shape rank")
            if any(value < 0 for value in strides):
                raise ValueError("native tensor strides must be non-negative")
            object.__setattr__(self, "strides", strides)
        if self.role in {NativeArgumentRole.SCALAR, NativeArgumentRole.TENSOR_MAP} and self.strides is not None:
            raise ValueError("native scalar/TensorMap arguments cannot carry tensor strides")


@dataclass(frozen=True)
class NativeExecutionSpec:
    arguments: tuple[NativeArgument, ...]
    compile_case: Any
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_mem: int = 0
    device: str = "cuda"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_arguments: tuple[NativeArgument, ...] | None = None
    source_grid: tuple[int, int, int] | None = None
    source_block: tuple[int, int, int] | None = None
    source_shared_mem: int | None = None
    execution_binding: ExecutionBinding | None = None

    def __post_init__(self) -> None:
        if not self.arguments:
            raise ValueError("native execution requires ordered arguments")
        for label, dimensions in (("grid", self.grid), ("block", self.block)):
            dims = tuple(int(item) for item in dimensions)
            if len(dims) != 3 or any(item <= 0 for item in dims):
                raise ValueError(f"native {label} must contain three positive dimensions")
            object.__setattr__(self, label, dims)
        if not isinstance(self.shared_mem, int) or self.shared_mem < 0:
            raise ValueError("native shared memory must be a non-negative integer")
        metadata = dict(self.metadata)
        if any(not isinstance(name, str) or not name for name in metadata):
            raise ValueError("native execution metadata keys must be non-empty strings")
        object.__setattr__(self, "metadata", metadata)
        source_arguments = self.source_arguments
        if source_arguments is not None:
            source_arguments = tuple(source_arguments)
            if not source_arguments:
                raise ValueError("native Source argument override cannot be empty")
            object.__setattr__(self, "source_arguments", source_arguments)
        for field_name in ("source_grid", "source_block"):
            dimensions = getattr(self, field_name)
            if dimensions is None:
                continue
            dims = tuple(int(item) for item in dimensions)
            if len(dims) != 3 or any(item <= 0 for item in dims):
                raise ValueError(f"native {field_name} must contain three positive dimensions")
            object.__setattr__(self, field_name, dims)
        if self.source_shared_mem is not None and (
            not isinstance(self.source_shared_mem, int) or self.source_shared_mem < 0
        ):
            raise ValueError("native Source shared memory must be non-negative")
        if self.execution_binding is not None and not isinstance(
            self.execution_binding, ExecutionBinding
        ):
            raise TypeError("native execution_binding must be an ExecutionBinding")

    def arguments_for(self, leg: str) -> tuple[NativeArgument, ...]:
        if leg == "source" and self.source_arguments is not None:
            return self.source_arguments
        return self.arguments

    def launch_for(
        self, leg: str
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        if leg != "source":
            return self.grid, self.block, self.shared_mem
        return (
            self.source_grid or self.grid,
            self.source_block or self.block,
            self.shared_mem if self.source_shared_mem is None else self.source_shared_mem,
        )


def _torch_dtype(torch: Any, dtype: str) -> Any:
    aliases = {
        "bf16": "bfloat16",
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
        "ui8": "uint8",
        "ui16": "uint16",
        "ui32": "uint32",
    }
    key = aliases.get(dtype.removeprefix("torch.").lower(), dtype.removeprefix("torch.").lower())
    try:
        return getattr(torch, key)
    except AttributeError as exc:
        raise ValueError(f"unsupported native tensor dtype {dtype!r}") from exc


def _scalar_ctype(dtype: str | None, value: Any) -> type[Any]:
    key = str(dtype or "").removeprefix("torch.").lower()
    mapping: dict[str, type[Any]] = {
        "bool": ctypes.c_bool,
        "f32": ctypes.c_float,
        "float32": ctypes.c_float,
        "f64": ctypes.c_double,
        "float64": ctypes.c_double,
        "i1": ctypes.c_bool,
        "i8": ctypes.c_int8,
        "i16": ctypes.c_int16,
        "i32": ctypes.c_int32,
        "i64": ctypes.c_int64,
        "index": ctypes.c_int64,
        "int32": ctypes.c_int32,
        "int64": ctypes.c_int64,
        "u8": ctypes.c_uint8,
        "u16": ctypes.c_uint16,
        "u32": ctypes.c_uint32,
        "u64": ctypes.c_uint64,
    }
    if key in mapping:
        return mapping[key]
    if isinstance(value, bool):
        return ctypes.c_bool
    if isinstance(value, int):
        return ctypes.c_int32
    if isinstance(value, float):
        return ctypes.c_float
    raise ValueError(f"unsupported native scalar dtype {dtype!r}")


def _storage_size(shape: tuple[int, ...], strides: tuple[int, ...]) -> int:
    if len(shape) != len(strides):
        raise ValueError("native tensor shape/stride ranks differ")
    if not shape:
        return 1
    return 1 + sum(
        (extent - 1) * stride
        for extent, stride in zip(shape, strides, strict=True)
    )


def _physical_offsets(
    shape: tuple[int, ...], strides: tuple[int, ...]
) -> np.ndarray:
    if not shape:
        return np.asarray(0, dtype=np.int64)
    offsets = np.zeros(shape, dtype=np.int64)
    for axis, (extent, stride) in enumerate(zip(shape, strides, strict=True)):
        axis_values = np.arange(extent, dtype=np.int64)
        axis_values = axis_values.reshape(
            (1,) * axis + (extent,) + (1,) * (len(shape) - axis - 1)
        )
        offsets += axis_values * stride
    return offsets


def _native_input_tensor(
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
        raise ValueError("logical native input values conflict under captured strides")
    base = torch.as_tensor(storage, dtype=dtype, device=device)
    return torch.as_strided(base, size=shape, stride=strides)


def _native_output_tensor(
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


def _prepare_torch_arguments(
    spec: NativeExecutionSpec,
    values: Mapping[str, Any],
    leg: str,
) -> tuple[list[Any], dict[str, Any]]:
    import torch

    arguments: list[Any] = []
    outputs: dict[str, Any] = {}
    for index, argument in enumerate(spec.arguments_for(leg)):
        if argument.role is NativeArgumentRole.SCALAR:
            value = values[argument.value_key] if argument.value_key is not None else argument.value
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            if isinstance(value, np.generic):
                value = value.item()
            arguments.append(value)
            continue
        dtype = _torch_dtype(torch, argument.dtype or "f32")
        if argument.role is NativeArgumentRole.OUTPUT:
            tensor = _native_output_tensor(
                torch,
                shape=tuple(argument.shape or ()),
                dtype=dtype,
                device=spec.device,
                strides=argument.strides,
            )
            outputs[argument.output_key or f"output_{index}"] = tensor
        else:
            tensor = _native_input_tensor(
                torch,
                np.asarray(values[argument.value_key]),
                dtype=dtype,
                device=spec.device,
                strides=argument.strides,
            )
            if argument.role is NativeArgumentRole.INOUT:
                tensor = tensor.clone()
                outputs[argument.output_key or f"output_{index}"] = tensor
        arguments.append(tensor)
    return arguments, outputs


def _copy_outputs(outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: tensor.detach().cpu().numpy().copy()
        for key, tensor in outputs.items()
    }


def _encode_tensor_map(driver: Any, tensor: Any, spec: NativeTensorMapSpec) -> Any:
    """Encode one by-value CUtensorMap while retaining its Python owner."""

    if tensor.ndim != 2 or not tensor.is_contiguous():
        raise ValueError("native TensorMap requires one contiguous rank-2 tensor")
    dtype_key = str(tensor.dtype).removeprefix("torch.")
    dtype_names = {
        "float16": "CU_TENSOR_MAP_DATA_TYPE_FLOAT16",
        "float32": "CU_TENSOR_MAP_DATA_TYPE_FLOAT32",
        "bfloat16": "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
    }
    dtype_name = dtype_names.get(dtype_key)
    if dtype_name is None:
        raise ValueError(f"unsupported native TensorMap dtype {tensor.dtype}")
    rows, columns = (int(value) for value in tensor.shape)
    major_stride_bytes = int(tensor.stride(0)) * int(tensor.element_size())
    swizzles = {
        "none": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        "32b": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
        "64b": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
        "128b": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
    }
    error, tensor_map = driver.cuTensorMapEncodeTiled(
        getattr(driver.CUtensorMapDataType, dtype_name),
        2,
        int(tensor.data_ptr()),
        (driver.cuuint64_t(columns), driver.cuuint64_t(rows)),
        (driver.cuuint64_t(major_stride_bytes),),
        tuple(driver.cuuint32_t(value) for value in spec.box_dim),
        tuple(driver.cuuint32_t(value) for value in spec.element_strides),
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzles[spec.swizzle],
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(error) != 0 or tensor_map is None:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed with {error}")
    return tensor_map


def _driver_check(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    error = result[0]
    if int(error) != 0:
        raise RuntimeError(f"CUDA driver {operation} failed with {error}")
    return result[1:]


def _cuda_module_image(ptx: str, compiler: CudaCompiler) -> bytes:
    """Assemble architecture-specific PTX before driver loading.

    PTX targets suffixed with ``a`` contain architecture-specific instructions
    (for example TMA/WGMMA).  The CUDA driver in the experiment environment
    rejects their textual JIT image but accepts the cubin produced by the same
    toolkit's ptxas.  Ordinary PTX keeps the existing driver-JIT path.
    """

    match = re.search(r"(?m)^\s*\.target\s+(sm_[0-9]+a)\b", ptx)
    if match is None:
        return ptx.encode("utf-8")
    nvcc = getattr(compiler, "nvcc", None)
    ptxas = Path(str(nvcc)).with_name("ptxas") if nvcc else None
    if ptxas is None or not ptxas.is_file():
        raise RuntimeError("architecture-specific PTX requires sibling ptxas")
    with tempfile.TemporaryDirectory(prefix="intentir-native-ptxas-") as raw:
        root = Path(raw)
        source = root / "kernel.ptx"
        image = root / "kernel.cubin"
        source.write_text(ptx, encoding="utf-8")
        completed = subprocess.run(
            [str(ptxas), f"-arch={match.group(1)}", str(source), "-o", str(image)],
            capture_output=True,
            text=True,
            timeout=120.0,
            check=False,
        )
        if completed.returncode != 0 or not image.is_file():
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"ptxas failed: {diagnostic}")
        return image.read_bytes()


class CudaRunner:
    """Compile to PTX and launch through the CUDA driver on isolated tensors."""

    def __init__(self, compiler: CudaCompiler | None = None) -> None:
        self.compiler = compiler or CudaCompiler()

    def run(
        self,
        leg: str,
        program: CudaKernelSource,
        spec: NativeExecutionSpec,
        values: Mapping[str, Any],
    ) -> ExecutionResult:
        module = None
        try:
            import torch
            from cuda.bindings import driver

            if not isinstance(spec.compile_case, CudaCompileCase):
                raise TypeError("CUDA Gate2 requires a CudaCompileCase")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            raw_program_digest, executed_program_digest = program_identity_digests(
                program,
                frontend="cuda",
            )
            binding = spec.execution_binding
            if binding is not None:
                binding.validate_runtime(
                    leg=leg,
                    program_source_sha256=raw_program_digest,
                    program_execution_sha256=executed_program_digest,
                    program_entry=program.function,
                    contract=execution_spec_contract(spec, leg=leg),
                )
            compiled = self.compiler.compile(program, spec.compile_case)
            if not compiled.ok or compiled.ptx is None or compiled.entry is None:
                raise RuntimeError(compiled.diagnostic or "CUDA Gate2 compile failed")
            device = torch.device(spec.device)
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(device_index)
            # Materializing one tensor ensures PyTorch has retained the primary context.
            torch.empty((), device=spec.device)
            _driver_check(driver.cuInit(0), "cuInit")
            module_image = _cuda_module_image(compiled.ptx, self.compiler)
            module, = _driver_check(
                driver.cuModuleLoadData(module_image),
                "cuModuleLoadData",
            )
            function, = _driver_check(
                driver.cuModuleGetFunction(module, compiled.entry.encode("utf-8")),
                "cuModuleGetFunction",
            )
            definitions = spec.arguments_for(leg)
            grid, block, shared_mem = spec.launch_for(leg)
            if shared_mem:
                _driver_check(
                    driver.cuFuncSetAttribute(
                        function,
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        int(shared_mem),
                    ),
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                )
            arguments, outputs = _prepare_torch_arguments(spec, values, leg)
            packed_values: list[Any] = []
            packed_types: list[type[Any] | None] = []
            for definition, value in zip(definitions, arguments, strict=True):
                if definition.role is NativeArgumentRole.SCALAR:
                    ctype = _scalar_ctype(definition.dtype, value)
                    packed_values.append(ctype(value))
                    packed_types.append(ctype)
                elif definition.role is NativeArgumentRole.TENSOR_MAP:
                    assert definition.tensor_map is not None
                    packed_values.append(
                        _encode_tensor_map(driver, value, definition.tensor_map)
                    )
                    # cuda.bindings treats a CUDA structure exposing getPtr()
                    # with ctype=None as by-value kernel argument storage.
                    packed_types.append(None)
                else:
                    packed_values.append(ctypes.c_void_p(value.data_ptr()))
                    packed_types.append(ctypes.c_void_p)
            _driver_check(
                driver.cuLaunchKernel(
                    function,
                    *grid,
                    *block,
                    shared_mem,
                    0,
                    (tuple(packed_values), tuple(packed_types)),
                    0,
                ),
                "cuLaunchKernel",
            )
            torch.cuda.synchronize(device_index)
            copied = _copy_outputs(outputs)
            artifact_kind = (
                "cuda.cubin"
                if module_image[:4] == b"\x7fELF"
                else "cuda.ptx"
            )
            return ExecutionResult(
                leg,
                True,
                copied,
                metadata={
                    "frontend": "cuda",
                    "entry": compiled.entry,
                    "grid": list(grid),
                    "block": list(block),
                    "program_source_sha256": raw_program_digest,
                    "program_execution_sha256": executed_program_digest,
                    **dict(spec.metadata),
                    "artifact_kind": artifact_kind,
                    "artifact_digest": artifact_sha256(module_image),
                    "artifact_size": len(module_image),
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
        except Exception as exc:
            return ExecutionResult(leg, False, error=f"{type(exc).__name__}: {exc}")
        finally:
            if module is not None:
                try:
                    from cuda.bindings import driver

                    driver.cuModuleUnload(module)
                except Exception:
                    pass


def _load_tilelang_prim_func(
    program: TileLangKernelSource,
    temporary: Path,
    *,
    schedule: Mapping[str, int | float | bool | str] | None = None,
) -> tuple[Any, str | None]:
    if program.kind == "factory_file":
        assert program.path is not None
        path = Path(program.path).resolve()
    else:
        assert program.code is not None
        path = temporary / "replay.py"
        path.write_text(program.code, encoding="utf-8")
    module_name = f"_intentir_gate2_tilelang_{id(program):x}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load TileLang module {path}")
    module = importlib.util.module_from_spec(module_spec)
    module.__dict__.update(dict(schedule or {}))
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
        value = getattr(module, program.function)
        if program.kind == "factory_file":
            value = value(**dict(program.factory_kwargs))
        return value, module_name
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def _tilelang_launch_contract(kernel: Any) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    """Read the launch that TileLang actually emitted for its device kernel."""

    adapter = getattr(kernel, "adapter", None)
    grid_info = getattr(adapter, "grid_info", None)
    block_info = getattr(adapter, "block_info", None)
    shared_info = getattr(adapter, "dynamic_smem_buf", None)
    if not isinstance(grid_info, Mapping) or not isinstance(block_info, Mapping):
        raise RuntimeError(
            "TileLang compiler did not expose its generated grid/block contract"
        )
    if set(grid_info) != set(block_info) or len(grid_info) != 1:
        raise RuntimeError(
            "TileLang Gate2 requires exactly one generated device launch"
        )
    function_name = next(iter(grid_info))
    grid = tuple(int(item) for item in grid_info[function_name])
    block = tuple(int(item) for item in block_info[function_name])
    if len(grid) != 3 or len(block) != 3 or any(item <= 0 for item in (*grid, *block)):
        raise RuntimeError("TileLang emitted an invalid grid/block contract")
    if isinstance(shared_info, Mapping):
        shared = shared_info.get(function_name)
    else:
        shared = shared_info
    shared_mem = 0 if shared is None else int(shared)
    if shared_mem < 0:
        raise RuntimeError("TileLang emitted a negative dynamic shared-memory contract")
    return grid, block, shared_mem


class TileLangRunner:
    """Compile a PrimFunc to a callable JITKernel and execute ordered tensors."""

    def run(
        self,
        leg: str,
        program: TileLangKernelSource,
        spec: NativeExecutionSpec,
        values: Mapping[str, Any],
    ) -> ExecutionResult:
        module_name: str | None = None
        try:
            import tilelang
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            with tempfile.TemporaryDirectory(prefix=f"intentir-tilelang-gate2-{leg}-") as raw:
                raw_program_digest, executed_program_digest = program_identity_digests(
                    program,
                    frontend="tilelang",
                )
                binding = spec.execution_binding
                if binding is not None:
                    binding.validate_runtime(
                        leg=leg,
                        program_source_sha256=raw_program_digest,
                        program_execution_sha256=executed_program_digest,
                        program_entry=program.function,
                        contract=execution_spec_contract(spec, leg=leg),
                    )
                schedule = dict(getattr(spec.compile_case, "schedule", {}) or {})
                prim_func, module_name = _load_tilelang_prim_func(
                    program,
                    Path(raw),
                    schedule=schedule,
                )
                target = str(getattr(spec.compile_case, "target", "cuda"))
                kernel = tilelang.compile(prim_func, target=target)
                arguments, outputs = _prepare_torch_arguments(spec, values, leg)
                artifact_path = Path(raw) / "tilelang_kernel.so"
                artifact_kind = "tilelang.runtime_library"
                try:
                    kernel.export_library(str(artifact_path))
                except Exception as library_error:
                    artifact_path = Path(raw) / "tilelang_kernel.ptx"
                    try:
                        kernel.export_ptx(str(artifact_path))
                    except Exception as ptx_error:
                        raise RuntimeError(
                            "TileLang did not expose an exportable runtime library or PTX: "
                            f"library={library_error}; ptx={ptx_error}"
                        ) from ptx_error
                    artifact_kind = "tilelang.ptx"
                if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
                    raise RuntimeError(
                        "TileLang compile produced no non-empty runtime artifact"
                    )
                artifact_bytes = artifact_path.read_bytes()
                actual_grid, actual_block, actual_shared_mem = _tilelang_launch_contract(
                    kernel
                )
                expected_grid, expected_block, expected_shared_mem = spec.launch_for(leg)
                if (
                    actual_grid != expected_grid
                    or actual_block != expected_block
                    or actual_shared_mem != expected_shared_mem
                ):
                    raise RuntimeError(
                        "TileLang generated launch differs from the bound Source/Replay "
                        f"contract: actual={(actual_grid, actual_block, actual_shared_mem)!r} "
                        f"expected={(expected_grid, expected_block, expected_shared_mem)!r}"
                    )
                kernel(*arguments)
                torch.cuda.synchronize(spec.device)
                copied = _copy_outputs(outputs)
                return ExecutionResult(
                    leg,
                    True,
                    copied,
                    metadata={
                        "frontend": "tilelang",
                        "grid": list(actual_grid),
                        "block": list(actual_block),
                        "shared_mem": actual_shared_mem,
                        "launch_contract_verified": True,
                        "program_source_sha256": raw_program_digest,
                        "program_execution_sha256": executed_program_digest,
                        **dict(spec.metadata),
                        "artifact_kind": artifact_kind,
                        "artifact_digest": artifact_sha256(artifact_bytes),
                        "artifact_size": len(artifact_bytes),
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
        except Exception as exc:
            return ExecutionResult(leg, False, error=f"{type(exc).__name__}: {exc}")
        finally:
            if module_name is not None:
                sys.modules.pop(module_name, None)


__all__ = [
    "CudaRunner",
    "NativeArgument",
    "NativeArgumentRole",
    "NativeExecutionSpec",
    "NativeTensorMapSpec",
    "TileLangRunner",
]
