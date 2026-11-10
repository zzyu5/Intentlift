"""Opaque Gate2-compatible runners for CUDA SearchAndClose.

All target computation comes from a persisted ``LoweredProgram`` produced by
the C++/MLIR pipeline.  Python only binds J logical inputs, launches the cubin,
and returns observations to the frozen Q comparator.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

import numpy as np

from intentir.runtime_dtype import torch_tensor_from_numpy, torch_tensor_to_numpy
from intentir.backend.materialize import MaterializedJ
from intentir.backend.search_and_close import (
    ExecutionDomain,
    ExecutionDomainCase,
    OpaqueExecutionRunners,
)
from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import ImplementationConfig, SemanticArchive
from intentir.verify.gate2.model import ExecutionResult
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec
from intentir.verify.gate2.triton import TritonRunner
from intentir.verify.scrc_workload_run import build_workload_jc_suite
from intentir.verify.workload_adapters import PreparedTritonWorkload
from intentir.ir.canonical_j import j_elaboration

from .pipeline import LoweredProgram, execute_cuda_artifact
from .portable_bundle import logical_runtime_array


def build_seeded_execution_domain(
    prepared: PreparedTritonWorkload,
    j: SemanticArchive,
    c: ImplementationConfig,
    *,
    domain_id: str,
    seeds: Iterable[int],
) -> ExecutionDomain:
    """Instantiate one signature-identical domain with independent input seeds."""

    cases = []
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or any(seed < 0 for seed in seed_values):
        raise ValueError("Q/D seeds must be non-empty non-negative integers")
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("Q/D seeds must be unique")
    for index, seed in enumerate(seed_values):
        seeded = replace(
            prepared,
            adapter=replace(prepared.adapter, seed=seed),
        )
        suite = build_workload_jc_suite(seeded, j, c)
        case = replace(
            suite.cases[0],
            case_id=f"{domain_id}-seed-{seed}-{index}",
            metadata={
                **dict(suite.cases[0].metadata),
                "backend_domain_id": domain_id,
                "backend_domain_seed": seed,
            },
        )
        cases.append(ExecutionDomainCase.from_gate2_case(case))
    return ExecutionDomain(domain_id, tuple(cases))


def _torch_dtype(torch: Any, dtype: str) -> Any:
    aliases = {
        "bool": "bool",
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
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
    }
    key = aliases.get(dtype.removeprefix("torch.").lower())
    if key is None:
        raise ValueError(f"unsupported CUDA Q/D dtype {dtype!r}")
    return getattr(torch, key)


def _logical_runtime_array(value: Any, *, rank: int, name: str) -> np.ndarray:
    """Compatibility alias for the shared portable-runtime ABI normalizer."""

    return logical_runtime_array(value, rank=rank, name=name)


class CudaOpaqueExecution:
    """Bind opaque Source/Replay/RefJ plus an already-compiled CUDA state."""

    def __init__(
        self,
        *,
        j: SemanticArchive,
        materialized: MaterializedJ,
        source: KernelSource,
        replay: KernelSource | None,
    ) -> None:
        self.j = j
        self.materialized = materialized
        self.source_program = source
        self.replay_program = replay
        self.triton_runner = TritonRunner()
        self.refeval_runner = ScalarLoweringRunner()
        self._tensors = {item.name: item for item in j.intent.inputs}

    @staticmethod
    def _gate2_case(case: ExecutionDomainCase):
        context = case.opaque_context
        if context is None or not hasattr(context, "triton"):
            raise ValueError("CUDA Q/D case lacks opaque Gate2 harness context")
        return context

    def source(
        self, case: ExecutionDomainCase, values: Mapping[str, Any]
    ) -> ExecutionResult:
        context = self._gate2_case(case)
        return self.triton_runner.run(
            "source",
            self.source_program,
            context.triton,
            values,
        )

    def replay(
        self, case: ExecutionDomainCase, values: Mapping[str, Any]
    ) -> ExecutionResult:
        if self.replay_program is None:
            return ExecutionResult(
                "replay",
                False,
                error="opaque Replay runner is not configured",
            )
        context = self._gate2_case(case)
        return self.triton_runner.run(
            "replay",
            self.replay_program,
            context.triton,
            values,
        )

    def ref_j(
        self, case: ExecutionDomainCase, values: Mapping[str, Any]
    ) -> ExecutionResult:
        context = self._gate2_case(case)
        if not isinstance(context.refeval, ScalarLoweringSpec):
            return ExecutionResult(
                "ref_j",
                False,
                error="canonical J reference is unavailable",
                unsupported=True,
            )
        result = self.refeval_runner.run(context.refeval, values)
        return ExecutionResult(
            "ref_j",
            result.ok,
            result.outputs,
            result.error,
            result.unsupported,
            result.metadata,
        )

    def target(
        self,
        executable: Any,
        case: ExecutionDomainCase,
        values: Mapping[str, Any],
    ) -> ExecutionResult:
        if not isinstance(executable, LoweredProgram):
            return ExecutionResult(
                "target",
                False,
                error="CUDA target runner requires LoweredProgram",
            )
        try:
            import torch

            runtime_names = set(self.materialized.runtime_input_names)
            inputs: dict[str, Any] = {}
            for logical_name, runtime_key in case.input_bindings:
                if logical_name not in runtime_names:
                    continue
                tensor = self._tensors[logical_name]
                raw = _logical_runtime_array(
                    values[runtime_key],
                    rank=len(tensor.shape),
                    name=logical_name,
                )
                inputs[logical_name] = torch_tensor_from_numpy(
                    torch,
                    raw,
                    dtype=_torch_dtype(torch, tensor.dtype),
                    device="cuda",
                    clone=True,
                )
            outputs = execute_cuda_artifact(
                executable,
                self.materialized.intent_mlir,
                inputs,
            )
            logical_outputs = j_elaboration(self.j).output_names
            copied = {
                case.output_binding_map[logical_name]: torch_tensor_to_numpy(
                    torch, outputs[f"out{index}"]
                )
                for index, logical_name in enumerate(logical_outputs)
            }
            return ExecutionResult(
                "target",
                True,
                copied,
                metadata={
                    "true_mlir_pipeline": True,
                    "python_target_kernel_emission": False,
                },
            )
        except Exception as exc:
            return ExecutionResult(
                "target",
                False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def runners(self) -> OpaqueExecutionRunners:
        return OpaqueExecutionRunners(
            ref_j=self.ref_j,
            target=self.target,
            source=self.source,
            replay=(self.replay if self.replay_program is not None else None),
        )


__all__ = [
    "CudaOpaqueExecution",
    "build_seeded_execution_domain",
]
