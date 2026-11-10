"""Verification-neutral canonical-J -> CUDA lowering entry points.

Gate, paper, command-line, and future target frontends may share this module,
but none of their verdicts participate in lowering authorization.  The formal
J-facing entry accepts only a ``SemanticArchive``; raw Intent dialect MLIR is
handled by the lower consumer below.  This module emits/probes/compiles that
exact program and returns executable cubins plus a neutral lowering record.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import intentir.backend.cuda.compile as backend_compile
import intentir.backend.cuda.orchestrate as backend_orchestrate
from intentir.backend.cuda.launch_config import (
    KernelLaunchConfig,
    parse_launch_configs,
)
from intentir.backend.cuda.selection import (
    LoweringTarget,
    RAGGED_FAST_PATH_ID,
    VariantSelection,
    read_intent_module,
    residual_intent_ops,
    select_variant,
)
from intentir.backend.cuda.schedules import ScheduleDecision
from intentir.frontend.canonical_j import emit_j_module
from intentir.ir.canonical_j import elaborate_j
from intentir.verify.gate1.model import SemanticArchive


@dataclass(frozen=True)
class MLIRTransformationRecord:
    """Facts attached by the C++ pass to the concrete post-pass MLIR state."""

    lowering_path_kind: str
    reduction_topology: str
    transformation_trace: tuple[str, ...]
    fma_count: int
    approximate_div_count: int
    shuffle_count: int
    barrier_count: int
    all_reduce_count: int
    block_threads: int
    grid_blocks: int
    grid_stride_loop_count: int
    grid_stride_unroll: int
    logical_tile_elements: int
    rows_per_block: int
    warps_per_row: int
    d_partitions: int
    accumulator_chains: int
    stage_count: int
    reduction_parts: int
    predicated_value_if_count: int
    folded_constexpr_select_count: int
    host_value_capture_count: int
    epilogue_load_motion_count: int
    prefetched_epilogue_input_count: int
    prefix_packet_width: int
    approximation_mode: str
    vector_width: int = 1
    warps_n: int = 1
    group_m: int = 8
    launch_plans: tuple[KernelLaunchConfig, ...] = ()

    @property
    def observation_changing(self) -> bool:
        return (
            self.fma_count > 0
            or self.approximate_div_count > 0
            or self.approximation_mode != "none"
            or self.accumulator_chains > 1
            or self.reduction_topology
            in {
                "subgroup_shuffle",
                "workgroup_hierarchical",
                "subgroup_feature",
                "subgroup_rows",
                "multi_warp_d_partition",
                "cross_cta_d_partition",
                "multi_stage_partial_finalize",
                "sorted_prefix_early_stop",
                "sorted_prefix_early_stop_cross_multiply",
                "sorted_prefix_multirow_warp_early_stop",
                "sorted_prefix_multirow_warp_scan_wide_epilogue",
                "sorted_prefix_multirow_warp_scan_wide_epilogue_cross_multiply",
                "sorted_prefix_warp_scan_wide_epilogue",
                "sorted_prefix_warp_scan_wide_epilogue_cross_multiply",
                "two_stage_cross_row_finalize",
                "raw_moment_hierarchical",
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "lowering_path_kind": self.lowering_path_kind,
            "reduction_topology": self.reduction_topology,
            "transformation_trace": list(self.transformation_trace),
            "fma_count": self.fma_count,
            "approximate_div_count": self.approximate_div_count,
            "shuffle_count": self.shuffle_count,
            "barrier_count": self.barrier_count,
            "all_reduce_count": self.all_reduce_count,
            "block_threads": self.block_threads,
            "grid_blocks": self.grid_blocks,
            "grid_stride_loop_count": self.grid_stride_loop_count,
            "grid_stride_unroll": self.grid_stride_unroll,
            "logical_tile_elements": self.logical_tile_elements,
            "rows_per_block": self.rows_per_block,
            "warps_per_row": self.warps_per_row,
            "d_partitions": self.d_partitions,
            "accumulator_chains": self.accumulator_chains,
            "stage_count": self.stage_count,
            "reduction_parts": self.reduction_parts,
            "predicated_value_if_count": self.predicated_value_if_count,
            "folded_constexpr_select_count": self.folded_constexpr_select_count,
            "host_value_capture_count": self.host_value_capture_count,
            "epilogue_load_motion_count": self.epilogue_load_motion_count,
            "prefetched_epilogue_input_count": (
                self.prefetched_epilogue_input_count
            ),
            "prefix_packet_width": self.prefix_packet_width,
            "approximation_mode": self.approximation_mode,
            "vector_width": self.vector_width,
            "warps_n": self.warps_n,
            "group_m": self.group_m,
            "launch_plans": [item.to_dict() for item in self.launch_plans],
            "observation_changing": self.observation_changing,
        }


def _transformation_record(mlir: str) -> MLIRTransformationRecord | None:
    """Decode C++-authored structural attrs; never infer from function names."""

    def string_attr(name: str) -> str | None:
        match = re.search(
            rf"{re.escape(name)}\s*=\s*\"([^\"]+)\"",
            mlir,
        )
        return match.group(1) if match is not None else None

    def integer_attr(name: str) -> int | None:
        match = re.search(
            rf"{re.escape(name)}\s*=\s*(-?[0-9]+)\s*:\s*i64",
            mlir,
        )
        return int(match.group(1)) if match is not None else None

    path = string_attr("intent.backend.lowering_path_kind")
    topology = string_attr("intent.backend.reduction_topology")
    approximation_mode = (
        string_attr("intent.backend.approximation_mode") or "none"
    )
    trace_match = re.search(
        r"intent\.backend\.transformation_trace\s*=\s*\[([^\]]*)\]",
        mlir,
    )
    if path is None and topology is None and trace_match is None:
        return None
    if path is None or topology is None or trace_match is None:
        raise ValueError("C++ concrete-state record is incomplete")
    trace = tuple(re.findall(r'\"([^\"]+)\"', trace_match.group(1)))
    counts = {
        name: integer_attr(f"intent.backend.{name}")
        for name in (
            "fma_count",
            "shuffle_count",
            "barrier_count",
            "all_reduce_count",
            "block_threads",
            "predicated_value_if_count",
            "folded_constexpr_select_count",
        )
    }
    rows_per_block = integer_attr("intent.backend.rows_per_block")
    if rows_per_block is None:
        rows_per_block = 1
    warps_per_row = integer_attr("intent.backend.warps_per_row")
    if warps_per_row is None:
        warps_per_row = 1
    d_partitions = integer_attr("intent.backend.d_partitions")
    if d_partitions is None:
        d_partitions = 1
    accumulator_chains = integer_attr("intent.backend.accumulator_chains")
    if accumulator_chains is None:
        accumulator_chains = 1
    stage_count = integer_attr("intent.backend.stage_count")
    if stage_count is None:
        stage_count = 1
    reduction_parts = integer_attr("intent.backend.reduction_parts")
    if reduction_parts is None:
        reduction_parts = 1
    grid_blocks = integer_attr("intent.backend.grid_blocks")
    if grid_blocks is None:
        grid_blocks = 0
    grid_stride_loop_count = integer_attr(
        "intent.backend.grid_stride_loop_count"
    )
    if grid_stride_loop_count is None:
        grid_stride_loop_count = 0
    grid_stride_unroll = integer_attr("intent.backend.grid_stride_unroll")
    if grid_stride_unroll is None:
        grid_stride_unroll = 1
    logical_tile_elements = integer_attr("intent.backend.logical_tile_elements")
    if logical_tile_elements is None:
        logical_tile_elements = 0
    vector_width = integer_attr("intent.backend.vector_width")
    if vector_width is None:
        vector_width = 1
    warps_n = integer_attr("intent.backend.warps_n")
    if warps_n is None:
        warps_n = 1
    group_m = integer_attr("intent.backend.group_m")
    if group_m is None:
        group_m = 8
    approximate_div_count = integer_attr(
        "intent.backend.approximate_div_count"
    )
    if approximate_div_count is None:
        approximate_div_count = 0
    host_value_capture_count = integer_attr(
        "intent.backend.host_value_capture_count"
    )
    if host_value_capture_count is None:
        host_value_capture_count = 0
    epilogue_load_motion_count = integer_attr(
        "intent.backend.epilogue_load_motion_count"
    )
    if epilogue_load_motion_count is None:
        epilogue_load_motion_count = 0
    prefetched_epilogue_input_count = integer_attr(
        "intent.backend.prefetched_epilogue_input_count"
    )
    if prefetched_epilogue_input_count is None:
        prefetched_epilogue_input_count = 0
    prefix_packet_width = integer_attr("intent.backend.prefix_packet_width")
    if prefix_packet_width is None:
        prefix_packet_width = 0
    if (
        not trace
        or rows_per_block <= 0
        or warps_per_row <= 0
        or d_partitions not in {1, 2, 4, 8}
        or accumulator_chains not in {1, 2, 4}
        or stage_count <= 0
        or reduction_parts <= 0
        or grid_blocks < 0
        or grid_stride_loop_count not in {0, 1}
        or grid_stride_unroll not in {1, 2, 4, 8}
        or logical_tile_elements < 0
        or vector_width not in {1, 2, 4, 8}
        or warps_n not in {1, 2, 4, 8}
        or group_m not in {1, 2, 4, 8, 16, 32}
        or prefix_packet_width not in {0, 1, 2, 4}
        or prefetched_epilogue_input_count < 0
        or any(value is None or value < 0 for value in counts.values())
    ):
        raise ValueError("C++ concrete-state record has invalid trace/counts")
    assert all(value is not None for value in counts.values())
    return MLIRTransformationRecord(
        lowering_path_kind=path,
        reduction_topology=topology,
        transformation_trace=trace,
        fma_count=int(counts["fma_count"]),
        approximate_div_count=approximate_div_count,
        shuffle_count=int(counts["shuffle_count"]),
        barrier_count=int(counts["barrier_count"]),
        all_reduce_count=int(counts["all_reduce_count"]),
        block_threads=int(counts["block_threads"]),
        grid_blocks=grid_blocks,
        grid_stride_loop_count=grid_stride_loop_count,
        grid_stride_unroll=grid_stride_unroll,
        logical_tile_elements=logical_tile_elements,
        rows_per_block=rows_per_block,
        warps_per_row=warps_per_row,
        d_partitions=d_partitions,
        accumulator_chains=accumulator_chains,
        stage_count=stage_count,
        reduction_parts=reduction_parts,
        predicated_value_if_count=int(counts["predicated_value_if_count"]),
        folded_constexpr_select_count=int(
            counts["folded_constexpr_select_count"]
        ),
        host_value_capture_count=host_value_capture_count,
        epilogue_load_motion_count=epilogue_load_motion_count,
        prefetched_epilogue_input_count=prefetched_epilogue_input_count,
        prefix_packet_width=prefix_packet_width,
        approximation_mode=approximation_mode,
        vector_width=vector_width,
        warps_n=warps_n,
        group_m=group_m,
        launch_plans=tuple(parse_launch_configs(mlir)),
    )


@dataclass(frozen=True)
class LoweredProgram:
    """One compiled CUDA program and the structural decision that produced it."""

    selection: VariantSelection
    kernels: tuple[backend_compile.CompiledKernel, ...]
    transformation_record: MLIRTransformationRecord | None = None
    pipeline_id: str = "cuda.pipeline.direct.v1"
    # The exact MLIR handed to the selected CUDA pipeline.  It is retained as
    # a stage artifact so a target split/materialization record cannot be
    # certified merely by repeating the final cubin digest.
    realization_mlir: str = ""

    def __post_init__(self) -> None:
        if not self.kernels:
            raise ValueError("lowering must produce at least one CUDA kernel")
        for kernel in self.kernels:
            if not kernel.cubin.startswith(b"\x7fELF"):
                raise ValueError(
                    f"compiled kernel {kernel.kernel_name!r} is not an ELF cubin"
                )
            residuals = residual_intent_ops(kernel.binary_mlir)
            if residuals:
                raise ValueError(
                    f"compiled kernel {kernel.kernel_name!r} retains Intent ops: "
                    f"{list(residuals)}"
                )
        if (
            self.transformation_record is not None
            and self.selection.pipeline_kind != "specialized"
        ):
            raise ValueError("generic lowering cannot carry a transformation record")
        if not self.pipeline_id:
            raise ValueError("lowering pipeline_id must be non-empty")
        if not isinstance(self.realization_mlir, str) or not self.realization_mlir.strip():
            raise ValueError("lowering must retain its non-empty realization MLIR")
    def to_record(self) -> dict[str, object]:
        """Return the structural compilation result without embedding binary bytes."""

        return {
            "selection": self.selection.to_dict(),
            "pipeline_id": self.pipeline_id,
            "transformation_record": (
                self.transformation_record.to_dict()
                if self.transformation_record is not None
                else None
            ),
            "kernel_count": len(self.kernels),
            "realization_mlir_size_bytes": len(self.realization_mlir.encode("utf-8")),
            "kernels": [
                {
                    "binary_name": kernel.binary_name,
                    "kernel_name": kernel.kernel_name,
                    "cubin_size_bytes": len(kernel.cubin),
                    "cubin_is_elf": kernel.cubin.startswith(b"\x7fELF"),
                }
                for kernel in self.kernels
            ],
        }


def _existing_path(intent_module: str | os.PathLike[str]) -> Path | None:
    try:
        path = Path(intent_module)
        return path if path.is_file() else None
    except OSError:
        return None


def _compile_generic(
    intent_module: str | os.PathLike[str], source: str
) -> tuple[backend_compile.CompiledKernel, ...]:
    path = _existing_path(intent_module)
    if path is not None:
        return tuple(backend_compile.compile_all_kernels(path))
    with tempfile.TemporaryDirectory(prefix="intentir-lowering-") as directory:
        source_path = Path(directory) / "input.mlir"
        source_path.write_text(source, encoding="utf-8")
        return tuple(backend_compile.compile_all_kernels(source_path))


def lower_to_cuda(
    intent_module: str | os.PathLike[str],
    *,
    target: LoweringTarget | None = None,
    pass_overrides: Mapping[str, str] | None = None,
    only_fast_path: str | None = None,
    schedule_decision: ScheduleDecision | None = None,
    allow_specialized: bool = True,
    require_specialized: bool = False,
) -> LoweredProgram:
    """Compile accepted Intent dialect MLIR through generic or direct CUDA.

    ``only_fast_path`` is useful for an explicit tuning/ablation candidate.
    With ``require_specialized=True`` a structural probe miss is reported as a
    rejected search candidate instead of silently compiling the generic
    fallback under a specialized candidate id.
    """

    source = read_intent_module(intent_module)
    selected_target = target or LoweringTarget()
    if selected_target.chip != backend_compile.CHIP:
        raise ValueError(
            "lowering target must match the configured CUDA backend: "
            f"requested {selected_target.chip!r}, configured "
            f"{backend_compile.CHIP!r}"
        )
    effective_overrides = dict(pass_overrides or {})
    if schedule_decision is not None:
        if only_fast_path not in (None, schedule_decision.fast_path_id):
            raise ValueError(
                "schedule decision and only_fast_path identify different variants"
            )
        existing = effective_overrides.get(schedule_decision.fast_path_id)
        if existing not in (None, schedule_decision.pass_argument):
            raise ValueError(
                "schedule decision conflicts with an explicit pass override"
            )
        only_fast_path = schedule_decision.fast_path_id
        effective_overrides[only_fast_path] = schedule_decision.pass_argument
    selection = select_variant(
        source,
        target=selected_target,
        pass_overrides=effective_overrides,
        only_fast_path=only_fast_path,
        allow_specialized=allow_specialized,
    )
    if backend_compile._has_ragged_ops(source):
        ragged_attempts = tuple(
            attempt
            for attempt in selection.attempts
            if attempt.fast_path_id == RAGGED_FAST_PATH_ID
        )
        if len(ragged_attempts) != 1 or not ragged_attempts[0].direct_hit:
            reason = (
                ragged_attempts[0].reason
                if ragged_attempts
                else "descriptor-aware Ragged probe was not executed"
            )
            raise ValueError(
                "CUDA Ragged capability boundary rejected the module: " + reason
            )
    if require_specialized and selection.pipeline_kind != "specialized":
        raise ValueError(
            "required specialized candidate failed whole-graph admission"
        )

    transformation_record: MLIRTransformationRecord | None = None
    if selection.pipeline_kind == "specialized":
        assert selection.pass_argument is not None
        specialized = backend_compile._run(
            [backend_compile.INTENT_OPT, selection.pass_argument], stdin=source
        )
        transformation_record = _transformation_record(specialized)
        kernels = tuple(
            backend_compile.lower_memref_module_to_kernels(
                specialized, scf_to_cf=selection.scf_to_cf
            )
        )
    else:
        kernels = _compile_generic(intent_module, source)

    return LoweredProgram(
        selection=selection,
        kernels=kernels,
        transformation_record=transformation_record,
        pipeline_id=(
            selection.pipeline_id
        ),
        realization_mlir=(specialized if selection.pipeline_kind == "specialized" else source),
    )


def lower_to_cuda_fused(
    intent_module: str | os.PathLike[str],
    *,
    tile: tuple[int, ...] = (1, 256),
    target: LoweringTarget | None = None,
    require_single_kernel: bool = True,
) -> LoweredProgram:
    """Compile a real upstream Linalg fusion+tiling state to CUDA.

    This is the existing MLIR fusion pipeline from ``compile.py`` exposed as a
    production search candidate.  It consumes the same immutable Intent module;
    no Python implementation of the kernel is generated.
    """

    source = read_intent_module(intent_module)
    if backend_compile._has_ragged_ops(source):
        raise ValueError(
            "CUDA Ragged graphs require the descriptor-aware lowering; "
            "the fused elementwise pipeline is not ABI-compatible"
        )
    selected_target = target or LoweringTarget()
    if selected_target.chip != backend_compile.CHIP:
        raise ValueError(
            "lowering target must match the configured CUDA backend: "
            f"requested {selected_target.chip!r}, configured {backend_compile.CHIP!r}"
        )
    selection = select_variant(
        source,
        target=selected_target,
        allow_specialized=False,
    )
    path = _existing_path(intent_module)
    if path is not None:
        kernels = tuple(backend_compile.compile_all_kernels_fused(path, tile))
    else:
        with tempfile.TemporaryDirectory(
            prefix="intentir-lowering-fused-"
        ) as directory:
            source_path = Path(directory) / "input.mlir"
            source_path.write_text(source, encoding="utf-8")
            kernels = tuple(
                backend_compile.compile_all_kernels_fused(source_path, tile)
            )
    if require_single_kernel and len(kernels) != 1:
        raise ValueError(
            "Linalg fusion did not collapse the complete graph to one kernel: "
            f"observed {len(kernels)} kernels"
        )
    return LoweredProgram(
        selection=selection,
        kernels=kernels,
        pipeline_id="cuda.pipeline.linalg-elementwise-fused.v1",
        realization_mlir=source,
    )


def lower_function_to_cuda(
    function: SemanticArchive,
    concrete_extents: Mapping[str, int],
    *,
    target: LoweringTarget | None = None,
    pass_overrides: Mapping[str, str] | None = None,
    only_fast_path: str | None = None,
    schedule_decision: ScheduleDecision | None = None,
    allow_specialized: bool = True,
    require_specialized: bool = False,
    fold: int = 0,
) -> LoweredProgram:
    """Emit the accepted canonical J and compile it to the generic CUDA path.

    This named compatibility entry is still a formal J boundary: callers may
    not pass a bare ``IntentFunction`` and thereby skip Layer-B elaboration.
    Raw MLIR remains available through :func:`lower_to_cuda`, which is the
    explicit lower-consumer boundary.
    """

    if not isinstance(function, SemanticArchive):
        raise TypeError(
            "CUDA production lowering requires canonical SemanticArchive J"
        )
    j = elaborate_j(function)
    mlir = emit_j_module(
        j,
        dict(concrete_extents),
        fold=fold,
    )
    return lower_to_cuda(
        mlir,
        target=target,
        pass_overrides=pass_overrides,
        only_fast_path=only_fast_path,
        schedule_decision=schedule_decision,
        allow_specialized=allow_specialized,
        require_specialized=require_specialized,
    )


def execute_cuda_artifact(
    program: LoweredProgram,
    intent_module: str | os.PathLike[str],
    inputs: Mapping[str, Any],
    *,
    output_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute already-compiled cubins without invoking the compiler again.

    ``intent_module`` supplies the logical ABI used by the host orchestrator.
    The compiler callback passed to
    :func:`intentir.backend.cuda.orchestrate.run_program` returns only the
    persisted kernels, so execution cannot silently rebuild or substitute a
    different binary.
    """

    source = read_intent_module(intent_module)
    path = _existing_path(intent_module)
    if path is not None:
        return backend_orchestrate.run_program(
            path,
            dict(inputs),
            compile_all=lambda _path: list(program.kernels),
            output_aliases=(
                dict(output_aliases) if output_aliases is not None else None
            ),
        )
    with tempfile.TemporaryDirectory(prefix="intentir-execution-") as directory:
        source_path = Path(directory) / "input.mlir"
        source_path.write_text(source, encoding="utf-8")
        return backend_orchestrate.run_program(
            source_path,
            dict(inputs),
            compile_all=lambda _path: list(program.kernels),
            output_aliases=(
                dict(output_aliases) if output_aliases is not None else None
            ),
        )


__all__ = [
    "LoweredProgram",
    "MLIRTransformationRecord",
    "execute_cuda_artifact",
    "lower_function_to_cuda",
    "lower_to_cuda",
    "lower_to_cuda_fused",
]
