"""Prepared no-sync launcher for already-compiled single-launch candidates.

This is the measured-search adapter over the existing compiler/orchestrator.
It deliberately does not compile, allocate, synchronize, or decide candidate
legality inside the timed closure.  ``triton.testing.do_bench`` (or another
event-based timer) therefore measures the same launch interval as a Triton
baseline.
"""

from __future__ import annotations

import base64
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.backend.cuda import compile as backend_compile
from intentir.backend.cuda import orchestrate
from intentir.backend.cuda.launch import load_module
from intentir.backend.cuda.pipeline import LoweredProgram
from intentir.backend.cuda.selection import read_intent_module


EXTRACTED_LAUNCH_PROTOCOL = "intentir.backend.cuda.extracted-launch.v1"
EXTRACTED_GRAPH_LAUNCH_PROTOCOL = (
    "intentir.backend.cuda.extracted-graph-launch.v1"
)


@dataclass(frozen=True)
class PreparedLaunchRecord:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_smem_bytes: int
    argument_count: int
    tensor_argument_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": list(self.grid),
            "block": list(self.block),
            "dynamic_smem_bytes": self.dynamic_smem_bytes,
            "argument_count": self.argument_count,
            "tensor_argument_count": self.tensor_argument_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreparedLaunchRecord":
        return cls(
            grid=tuple(int(item) for item in value["grid"]),
            block=tuple(int(item) for item in value["block"]),
            dynamic_smem_bytes=int(value["dynamic_smem_bytes"]),
            argument_count=int(value["argument_count"]),
            tensor_argument_count=int(value["tensor_argument_count"]),
        )


@dataclass(frozen=True)
class ExtractedSingleLaunch:
    """Portable cubin plus the launch ABI extracted from compiler host IR."""

    cubin: bytes
    kernel_name: str
    binary_name: str
    plan: tuple[tuple[Any, ...], ...]
    record: PreparedLaunchRecord

    def __post_init__(self) -> None:
        if not self.cubin.startswith(b"\x7fELF"):
            raise ValueError("extracted launch cubin is not ELF")
        if not self.kernel_name or not self.binary_name:
            raise ValueError("extracted launch requires compiler symbols")
        if len(self.plan) != self.record.argument_count:
            raise ValueError("extracted launch plan length mismatch")
    @property
    def binary_size_bytes(self) -> int:
        return len(self.cubin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": EXTRACTED_LAUNCH_PROTOCOL,
            "kernel_name": self.kernel_name,
            "binary_name": self.binary_name,
            "cubin_base64": base64.b64encode(self.cubin).decode("ascii"),
            "plan": [list(item) for item in self.plan],
            "launch": self.record.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExtractedSingleLaunch":
        if value.get("protocol") != EXTRACTED_LAUNCH_PROTOCOL:
            raise ValueError("unsupported extracted launch protocol")
        cubin = base64.b64decode(value["cubin_base64"], validate=True)
        result = cls(
            cubin=cubin,
            kernel_name=str(value["kernel_name"]),
            binary_name=str(value["binary_name"]),
            plan=tuple(tuple(item) for item in value["plan"]),
            record=PreparedLaunchRecord.from_dict(value["launch"]),
        )
        return result


@dataclass(frozen=True)
class ExtractedGraphKernel:
    """One cubin identity in a portable compiler-emitted launch graph."""

    cubin: bytes
    kernel_name: str
    binary_name: str

    def __post_init__(self) -> None:
        if not self.cubin.startswith(b"\x7fELF"):
            raise ValueError("extracted graph kernel cubin is not ELF")
        if not self.kernel_name or not self.binary_name:
            raise ValueError("extracted graph kernel requires compiler symbols")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "binary_name": self.binary_name,
            "cubin_base64": base64.b64encode(self.cubin).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExtractedGraphKernel":
        result = cls(
            cubin=base64.b64decode(value["cubin_base64"], validate=True),
            kernel_name=str(value["kernel_name"]),
            binary_name=str(value["binary_name"]),
        )
        return result


@dataclass(frozen=True)
class ExtractedGraphLaunchRecord:
    kernel_count: int
    launch_count: int
    intermediate_buffer_count: int

    def __post_init__(self) -> None:
        if self.kernel_count <= 0 or self.launch_count <= 0:
            raise ValueError("extracted graph launch counts must be positive")
        if self.intermediate_buffer_count < 0:
            raise ValueError("intermediate buffer count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "compiler-emitted-graph",
            "kernel_count": self.kernel_count,
            "launch_count": self.launch_count,
            "intermediate_buffer_count": self.intermediate_buffer_count,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ExtractedGraphLaunchRecord":
        if value.get("kind") != "compiler-emitted-graph":
            raise ValueError("unsupported extracted graph launch record")
        return cls(
            kernel_count=int(value["kernel_count"]),
            launch_count=int(value["launch_count"]),
            intermediate_buffer_count=int(value["intermediate_buffer_count"]),
        )


@dataclass(frozen=True)
class ExtractedGraphLaunch:
    """Portable cubins plus the compiler host graph that launches them.

    The host MLIR is not executable Python emission.  It is the lowered MLIR
    record that already drives ``orchestrate.prepare_program`` locally: launch
    order, temporary buffers, constants, and ABI pointer flow are recovered from
    it on the target machine while every target kernel remains a frozen cubin.
    """

    kernels: tuple[ExtractedGraphKernel, ...]
    source_mlir: str
    binary_mlir: str
    record: ExtractedGraphLaunchRecord

    def __post_init__(self) -> None:
        if not self.kernels:
            raise ValueError("extracted graph launch requires at least one cubin")
        if len(self.kernels) != self.record.kernel_count:
            raise ValueError("extracted graph kernel count mismatch")
        if len({item.binary_name for item in self.kernels}) != len(self.kernels):
            raise ValueError("extracted graph binary symbols must be unique")
        if not self.binary_mlir:
            raise ValueError("extracted graph launch requires lowered host MLIR")

    @property
    def binary_size_bytes(self) -> int:
        return sum(len(item.cubin) for item in self.kernels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": EXTRACTED_GRAPH_LAUNCH_PROTOCOL,
            "source_mlir": self.source_mlir,
            "binary_mlir": self.binary_mlir,
            "kernels": [item.to_dict() for item in self.kernels],
            "launch": self.record.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExtractedGraphLaunch":
        if value.get("protocol") != EXTRACTED_GRAPH_LAUNCH_PROTOCOL:
            raise ValueError("unsupported extracted graph launch protocol")
        result = cls(
            kernels=tuple(
                ExtractedGraphKernel.from_dict(item) for item in value["kernels"]
            ),
            source_mlir=str(value["source_mlir"]),
            binary_mlir=str(value["binary_mlir"]),
            record=ExtractedGraphLaunchRecord.from_dict(value["launch"]),
        )
        return result


def extract_single_launch(
    program: LoweredProgram,
    intent_module: str,
) -> ExtractedSingleLaunch:
    """Extract a target-portable launch bundle without loading its cubin."""

    if len(program.kernels) != 1:
        raise ValueError("prepared benchmark requires exactly one compiled kernel")
    source = read_intent_module(intent_module)
    inputs = orchestrate._parse_intent_signature(source)
    results = orchestrate._parse_intent_results(source)
    inplace = orchestrate._parse_intent_inplace(source, len(results))
    fresh_outputs = sum(
        1 for item in (inplace or [-1] * len(results)) if item < 0
    )
    tensor_count = len(inputs) + fresh_outputs

    kernel = program.kernels[0]
    _name, host = orchestrate._host_func_text(kernel.binary_mlir)
    _func, _ordered, pointer_blockargs = (
        orchestrate._parse_signature_blockargs(host)
    )
    aligned = orchestrate._map_input_aligned_ptrs(pointer_blockargs)
    constants = orchestrate._parse_constants(host)
    scalar_inputs: dict[str, tuple[int, str]] = {}
    for match in re.finditer(
        r"(%\w+)\s*=\s*llvm\.load\s+(%\w+)\s*:\s*!llvm\.ptr\s*->\s*"
        r"(i1|i8|i16|i32|i64|ui8|ui16|ui32|f16|bf16|f32|f64)",
        host,
    ):
        result_ssa, pointer_ssa, scalar_type = match.groups()
        if pointer_ssa in aligned:
            scalar_inputs[result_ssa] = (aligned[pointer_ssa], scalar_type)
    sites = orchestrate._parse_launches(host)
    if len(sites) != 1:
        raise ValueError(
            "prepared benchmark requires one compiler-emitted launch site"
        )
    site = sites[0]
    if (
        site.binary_module != kernel.binary_name
        or site.func_symbol != kernel.kernel_name
    ):
        raise ValueError("compiled cubin and host launch symbols disagree")

    class _ConstantView:
        pass

    constant_view = _ConstantView()
    constant_view.constants = constants
    grid = tuple(
        orchestrate._resolve_const_index(constant_view, item)
        for item in site.grid_ssa
    )
    block = tuple(
        orchestrate._resolve_const_index(constant_view, item)
        for item in site.block_ssa
    )
    shared = (
        int(constants[site.dynamic_shared_ssa][0])
        if site.dynamic_shared_ssa is not None
        else 0
    )

    plan: list[tuple[Any, ...]] = []
    for operand in site.args:
        if operand.ty.startswith("!llvm.ptr"):
            if operand.ssa not in aligned:
                raise ValueError(
                    "single-launch candidate captures a non-ABI pointer"
                )
            plan.append(("ptr", aligned[operand.ssa]))
            continue
        constant = constants.get(operand.ssa)
        if constant is None:
            dynamic = scalar_inputs.get(operand.ssa)
            if dynamic is None or dynamic[1] != operand.ty:
                raise ValueError(
                    f"unresolved scalar launch operand {operand.ssa!r}"
                )
            plan.append((f"input_{operand.ty}", dynamic[0]))
            continue
        raw, _constant_type = constant
        if operand.ty in {"i64", "index"}:
            plan.append(("i64", int(raw)))
        elif operand.ty == "i32":
            plan.append(("i32", int(raw)))
        elif operand.ty == "ui8":
            plan.append(("u8", int(raw)))
        elif operand.ty == "ui16":
            plan.append(("u16", int(raw)))
        elif operand.ty == "ui32":
            plan.append(("u32", int(raw)))
        elif operand.ty == "f32":
            plan.append(("f32", orchestrate._parse_f32(raw)))
        elif operand.ty == "f64":
            plan.append(("f64", orchestrate._parse_f64(raw)))
        elif operand.ty == "f16":
            plan.append(("f16", orchestrate._parse_f16_bits(raw)))
        elif operand.ty == "bf16":
            plan.append(("bf16", orchestrate._parse_bf16_bits(raw)))
        else:
            raise NotImplementedError(
                f"unhandled scalar launch type {operand.ty!r}"
            )
    record = PreparedLaunchRecord(
        grid=grid,
        block=block,
        dynamic_smem_bytes=shared,
        argument_count=len(plan),
        tensor_argument_count=tensor_count,
    )
    return ExtractedSingleLaunch(
        cubin=kernel.cubin,
        kernel_name=kernel.kernel_name,
        binary_name=kernel.binary_name,
        plan=tuple(plan),
        record=record,
    )


def extract_graph_launch(
    program: LoweredProgram,
    intent_module: str,
) -> ExtractedGraphLaunch:
    """Extract an arbitrary compiler-emitted CUDA launch graph for transport."""

    source = read_intent_module(intent_module)
    binary_modules = {item.binary_mlir for item in program.kernels}
    if len(binary_modules) != 1:
        raise ValueError("compiled graph kernels do not share one host MLIR module")
    binary_mlir = next(iter(binary_modules))
    _name, host = orchestrate._host_func_text(binary_mlir)
    sites = orchestrate._parse_launches(host)
    if not sites:
        raise ValueError("compiled graph contains no launch sites")
    compiled_symbols = {
        (item.binary_name, item.kernel_name) for item in program.kernels
    }
    launched_symbols = {
        (item.binary_module, item.func_symbol) for item in sites
    }
    if launched_symbols != compiled_symbols:
        missing = sorted(launched_symbols - compiled_symbols)
        unused = sorted(compiled_symbols - launched_symbols)
        raise ValueError(
            "compiled graph launch/cubin symbols disagree: "
            f"missing={missing}, unused={unused}"
        )
    intent_inputs = orchestrate._parse_intent_signature(source)
    intent_results = orchestrate._parse_intent_results(source)
    intent_inplace = orchestrate._parse_intent_inplace(source, len(intent_results))
    parsed_host = orchestrate._parse_host_module(
        binary_mlir,
        intent_inputs,
        intent_results,
        intent_inplace,
    )
    intermediate_count = len(parsed_host.loose_buffers)
    return ExtractedGraphLaunch(
        kernels=tuple(
            ExtractedGraphKernel(
                cubin=item.cubin,
                kernel_name=item.kernel_name,
                binary_name=item.binary_name,
            )
            for item in program.kernels
        ),
        source_mlir=source,
        binary_mlir=binary_mlir,
        record=ExtractedGraphLaunchRecord(
            kernel_count=len(program.kernels),
            launch_count=len(sites),
            intermediate_buffer_count=intermediate_count,
        ),
    )


PortableExtractedLaunch = ExtractedSingleLaunch | ExtractedGraphLaunch


def extract_portable_launch(
    program: LoweredProgram,
    intent_module: str,
) -> PortableExtractedLaunch:
    """Choose the compact single-launch record or the general graph record."""

    if len(program.kernels) == 1:
        _name, host = orchestrate._host_func_text(program.kernels[0].binary_mlir)
        if len(orchestrate._parse_launches(host)) == 1:
            try:
                return extract_single_launch(program, intent_module)
            except ValueError:
                # A fused one-kernel Linalg state may still allocate a
                # compiler-owned temporary and pass that non-ABI pointer to the
                # launch.  The compact record intentionally cannot represent
                # it; transport the exact compiler host graph instead of
                # rejecting a valid production MLIR state.
                pass
    return extract_graph_launch(program, intent_module)


def portable_launch_from_dict(
    value: Mapping[str, Any],
) -> PortableExtractedLaunch:
    protocol = value.get("protocol")
    if protocol == EXTRACTED_LAUNCH_PROTOCOL:
        return ExtractedSingleLaunch.from_dict(value)
    if protocol == EXTRACTED_GRAPH_LAUNCH_PROTOCOL:
        return ExtractedGraphLaunch.from_dict(value)
    raise ValueError(f"unsupported portable CUDA launch protocol {protocol!r}")


class PreparedSingleLaunch:
    """Bind one persisted cubin to its compiler-emitted launch plan."""

    def __init__(self, program: LoweredProgram, intent_module: str):
        self._bind(extract_single_launch(program, intent_module))

    @classmethod
    def from_extracted(
        cls, extracted: ExtractedSingleLaunch
    ) -> "PreparedSingleLaunch":
        instance = cls.__new__(cls)
        instance._bind(extracted)
        return instance

    def _bind(self, extracted: ExtractedSingleLaunch) -> None:
        function = load_module(extracted.cubin).function(extracted.kernel_name)
        shared = extracted.record.dynamic_smem_bytes
        orchestrate._set_dynamic_shared_memory(function, shared)
        self._launcher = orchestrate.PreparedLauncher(
            function,
            extracted.record.grid,
            extracted.record.block,
            extracted.plan,
            shared=shared,
        )
        self._tensor_count = extracted.record.tensor_argument_count
        self.record = extracted.record
        self.extracted = extracted

    def thunk(self, tensors: Sequence[Any]):
        """Return a zero-argument no-sync closure over persistent tensors."""

        bound = tuple(tensors)
        if len(bound) != self._tensor_count:
            raise ValueError(
                "prepared launch tensor count mismatch: "
                f"expected {self._tensor_count}, got {len(bound)}"
            )
        self._launcher.bind_input_scalars(bound)

        def launch() -> None:
            self._launcher.launch(bound)

        return launch


class PreparedGraphLaunch:
    """Bind a generic/fused multi-launch MLIR program once for fair timing."""

    def __init__(
        self,
        program: LoweredProgram,
        intent_module: str,
        input_tensors: Sequence[Any],
    ):
        source = read_intent_module(intent_module)
        self._bind(
            source=source,
            kernels=program.kernels,
            input_tensors=input_tensors,
        )

    @classmethod
    def from_extracted(
        cls,
        extracted: ExtractedGraphLaunch,
        input_tensors: Sequence[Any],
    ) -> "PreparedGraphLaunch":
        instance = cls.__new__(cls)
        kernels = tuple(
            backend_compile.CompiledKernel(
                cubin=item.cubin,
                kernel_name=item.kernel_name,
                binary_name=item.binary_name,
                binary_mlir=extracted.binary_mlir,
            )
            for item in extracted.kernels
        )
        instance._bind(
            source=extracted.source_mlir,
            kernels=kernels,
            input_tensors=input_tensors,
        )
        if instance.record.launch_count != extracted.record.launch_count:
            raise ValueError("prepared graph launch count differs from record")
        if instance.record.kernel_count != extracted.record.kernel_count:
            raise ValueError("prepared graph kernel count differs from record")
        if (
            instance.record.intermediate_buffer_count
            != extracted.record.intermediate_buffer_count
        ):
            raise ValueError("prepared graph intermediate-buffer count differs from record")
        instance.extracted = extracted
        return instance

    def _bind(
        self,
        *,
        source: str,
        kernels: Sequence[backend_compile.CompiledKernel],
        input_tensors: Sequence[Any],
    ) -> None:
        signature = orchestrate._parse_intent_signature(source)
        bound = tuple(input_tensors)
        if len(bound) != len(signature):
            raise ValueError(
                "prepared graph input count mismatch: "
                f"expected {len(signature)}, got {len(bound)}"
            )
        values = {
            item.name: tensor for item, tensor in zip(signature, bound, strict=True)
        }
        with tempfile.TemporaryDirectory(
            prefix="intentir-prepared-graph-"
        ) as directory:
            path = f"{directory}/input.mlir"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
            prepared = orchestrate.prepare_program(
                path,
                values,
                compile_all=lambda _path: list(kernels),
            )
        self._prepared = prepared
        self.record = prepared.record
        self.outputs = prepared.outputs

    def thunk(self):
        return self._prepared.thunk()


def do_bench_us(
    thunk,
    *,
    warmup_ms: int = 25,
    repeat_ms: int = 100,
) -> float:
    """Measure a no-sync closure with Triton's event-based benchmark helper."""

    from triton.testing import do_bench

    result = do_bench(thunk, warmup=warmup_ms, rep=repeat_ms)
    return float(result) * 1000.0


def do_bench_cudagraph_us(thunk, *, repetitions: int = 100) -> float:
    """Measure steady-state device time with host launch overhead amortized.

    Both the compiled MLIR cubin and its comparison baseline must use this same
    helper.  The callable is captured on PyTorch's capture stream, replayed as
    one CUDA graph, and reported as median microseconds per invocation by
    Triton's established benchmark utility.
    """

    if repetitions <= 0:
        raise ValueError("CUDA-graph benchmark repetitions must be positive")
    from triton.testing import do_bench_cudagraph

    result = do_bench_cudagraph(
        thunk,
        rep=repetitions,
        return_mode="median",
    )
    return float(result) * 1000.0


__all__ = [
    "EXTRACTED_GRAPH_LAUNCH_PROTOCOL",
    "EXTRACTED_LAUNCH_PROTOCOL",
    "ExtractedGraphKernel",
    "ExtractedGraphLaunch",
    "ExtractedGraphLaunchRecord",
    "ExtractedSingleLaunch",
    "PortableExtractedLaunch",
    "PreparedLaunchRecord",
    "PreparedGraphLaunch",
    "PreparedSingleLaunch",
    "do_bench_cudagraph_us",
    "do_bench_us",
    "extract_graph_launch",
    "extract_portable_launch",
    "extract_single_launch",
    "portable_launch_from_dict",
]
