"""Direct compiler for the Gate2 scalar CPU reference module.

Every call lowers and compiles the supplied MLIR into a fresh shared object.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..toolchain.tools import (
    CLANG,
    INTENT_OPT,
    LLVM_LIB,
    MLIR_OPT,
    MLIR_TRANSLATE,
)
from ...ir.intent_op_contract import RAGGED_OPS
from .abi import CPU_MEMREF_ABI_PROTOCOL


CPU_COMPILER_PROTOCOL = "intentir_cpu_mlir_shared_object_v2"
CPU_EXECUTION_RECORD_SCHEMA = "intentir_cpu_execution_record_v2"

CPU_LOWERING_PIPELINE = (
    "-one-shot-bufferize=bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map",
    "-buffer-deallocation-pipeline",
    "-convert-bufferization-to-memref",
    "-convert-linalg-to-loops",
    "-convert-math-to-libm",
    "-convert-scf-to-cf",
    "-convert-cf-to-llvm",
    "-expand-strided-metadata",
    "-finalize-memref-to-llvm",
    "-convert-func-to-llvm",
    "-convert-arith-to-llvm",
    "-reconcile-unrealized-casts",
)

_CLANG_FLAGS = (
    "-O2",
    "-fPIC",
    "-shared",
    "-x",
    "ir",
    "-",
    "-lm",
    "-Wl,-z,defs",
)


def _c_runner_utils() -> Path:
    try:
        return (LLVM_LIB / "libmlir_c_runner_utils.so").resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "required MLIR C runner utility library is unavailable: "
            f"{LLVM_LIB / 'libmlir_c_runner_utils.so'}"
        ) from exc


def _resolve_tool(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        found = shutil.which(value)
        if found is None:
            raise RuntimeError(f"required CPU lowering tool is unavailable: {value}")
        candidate = Path(found)
    try:
        return candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"required CPU lowering tool is unavailable: {candidate}"
        ) from exc


def _run_text(argv: Sequence[str], source: str, *, stage: str) -> str:
    completed = subprocess.run(
        list(argv),
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        diagnostic = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"{stage} failed: {diagnostic[:2000]}")
    return completed.stdout


def _has_ragged_ops(source: str) -> bool:
    return any(f"intent.{name}" in source for name in RAGGED_OPS)


def _validate_ragged_lowering(source: str, lowered: str, *, stage: str) -> None:
    if not _has_ragged_ops(source):
        return
    if _has_ragged_ops(lowered) or "tuple<" in lowered:
        raise RuntimeError(
            f"{stage} left a Ragged descriptor or Intent Ragged operation in "
            "the CPU native ABI"
        )


@dataclass(frozen=True)
class CompiledCPUModule:
    shared_object: Path
    toolchain: tuple[tuple[str, str], ...]
    # Keep the actual compiler-stage outputs available until the execution
    # boundary has recorded which target realization record consumed them.
    # The final shared object alone cannot prove that the preceding pipeline
    # record was consumed independently.
    intent_mlir: str = ""
    llvm_mlir: str = ""
    llvm_ir: str = ""

    @property
    def evidence(self) -> dict[str, Any]:
        return {
            "schema": CPU_EXECUTION_RECORD_SCHEMA,
            "compiler_protocol": CPU_COMPILER_PROTOCOL,
            "abi_protocol": CPU_MEMREF_ABI_PROTOCOL,
            "lowering_pipeline": list(CPU_LOWERING_PIPELINE),
            "toolchain": dict(self.toolchain),
        }


def compile_cpu_module(source: str) -> CompiledCPUModule:
    """Lower one static MLIR module and compile a fresh shared object."""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("CPU compilation requires non-empty MLIR source")

    intent_opt = _resolve_tool(str(INTENT_OPT))
    mlir_opt = _resolve_tool(str(MLIR_OPT))
    mlir_translate = _resolve_tool(str(MLIR_TRANSLATE))
    clang = _resolve_tool(str(CLANG))
    c_runner_utils = _c_runner_utils()

    intent_mlir = _run_text(
        [str(intent_opt), "-convert-intent-to-linalg", "-expand-rsqrt-for-libm"],
        source,
        stage="intent-opt",
    )
    _validate_ragged_lowering(source, intent_mlir, stage="intent-opt")
    llvm_mlir = _run_text(
        [str(mlir_opt), *CPU_LOWERING_PIPELINE],
        intent_mlir,
        stage="mlir-opt",
    )
    _validate_ragged_lowering(source, llvm_mlir, stage="mlir-opt")
    llvm_ir = _run_text(
        [str(mlir_translate), "--mlir-to-llvmir"],
        llvm_mlir,
        stage="mlir-translate",
    )

    descriptor, output_name = tempfile.mkstemp(
        prefix="intentir-cpu-module-", suffix=".so"
    )
    os.close(descriptor)
    shared_object = Path(output_name)
    runtime_link_flags = (
        "-x",
        "none",
        str(c_runner_utils),
        f"-Wl,-rpath,{c_runner_utils.parent}",
    )
    try:
        completed = subprocess.run(
            [
                str(clang),
                *_CLANG_FLAGS,
                *runtime_link_flags,
                "-o",
                str(shared_object),
            ],
            input=llvm_ir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"clang failed: {diagnostic[:2000]}")
        if not shared_object.is_file() or shared_object.stat().st_size == 0:
            raise RuntimeError("clang produced no CPU shared object")
    except Exception:
        shared_object.unlink(missing_ok=True)
        raise

    return CompiledCPUModule(
        shared_object=shared_object,
        toolchain=(
            ("intent_opt", str(intent_opt)),
            ("mlir_opt", str(mlir_opt)),
            ("mlir_translate", str(mlir_translate)),
            ("clang", str(clang)),
            ("mlir_c_runner_utils", str(c_runner_utils)),
        ),
        intent_mlir=intent_mlir,
        llvm_mlir=llvm_mlir,
        llvm_ir=llvm_ir,
    )


__all__ = [
    "CPU_COMPILER_PROTOCOL",
    "CPU_EXECUTION_RECORD_SCHEMA",
    "CPU_LOWERING_PIPELINE",
    "CompiledCPUModule",
    "compile_cpu_module",
]
