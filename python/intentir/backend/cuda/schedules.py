"""Structural Layer-C schedules for direct CUDA lowering passes.

The resolver consumes accepted Intent dialect MLIR plus accepted Layer-C data.
It has no dependency on paper evidence or Gate verdicts.  All policies are
function-name independent and return explicit pass arguments that must still
pass the C++ whole-graph probe in :mod:`intentir.backend.cuda.selection`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import intentir.backend.cuda.orchestrate as backend_orchestrate
from intentir.ir.layer_c import resolve_layer_c_scalar
from intentir.backend.cuda.selection import read_intent_module


ROW_FAST_PATH_ID = "cuda.row_reduction.normalization_loss.read_once.v1"
ELEMENTWISE_FAST_PATH_ID = "cuda.elementwise.row.vectorized.v1"
PADDED_ROW_FAST_PATH_ID = "cuda.padded_row_reduction.masked_short_dot.v1"
_BLOCK_THREADS = frozenset({32, 64, 128, 256, 512, 1024})
_INTEGER_DTYPES = frozenset({
    "i8", "i16", "i32", "i64", "u8", "u16", "u32",
})


@dataclass(frozen=True)
class RowSchedule:
    block_threads: int
    tile_cols: int
    vector_width: int = 0
    fuse_fma: bool = False
    approximate_div: bool = False
    leader_finalize: bool = False
    defer_epilogue_col_loads: bool = True
    promote_rank0_scalars: bool = True
    rows_per_block: int = 1
    reduction_parts: int = 1
    materialize_shared_row_output: bool = True
    cross_row_finalize: bool = False
    raw_moment_variance: bool = False
    compact_reduction_topology: bool = True
    accumulator_chains: int = 1

    def __post_init__(self) -> None:
        if type(self.block_threads) is not int or self.block_threads not in _BLOCK_THREADS:
            raise ValueError(
                "row schedule block_threads must be a supported warp multiple"
            )
        if (
            type(self.tile_cols) is not int
            or self.tile_cols <= 0
            or self.tile_cols > 256
            or self.tile_cols & (self.tile_cols - 1)
        ):
            raise ValueError(
                "row schedule tile_cols must be a positive power of two <= 256"
            )
        if type(self.vector_width) is not int or self.vector_width not in {0, 1, 4}:
            raise ValueError("row schedule vector_width must be 0, 1, or 4")
        if type(self.fuse_fma) is not bool:
            raise ValueError("row schedule fuse_fma must be a bool")
        if type(self.approximate_div) is not bool:
            raise ValueError("row schedule approximate_div must be a bool")
        if type(self.leader_finalize) is not bool:
            raise ValueError("row schedule leader_finalize must be a bool")
        if type(self.defer_epilogue_col_loads) is not bool:
            raise ValueError("row schedule defer_epilogue_col_loads must be a bool")
        if type(self.promote_rank0_scalars) is not bool:
            raise ValueError("row schedule promote_rank0_scalars must be a bool")
        if type(self.materialize_shared_row_output) is not bool:
            raise ValueError(
                "row schedule materialize_shared_row_output must be a bool"
            )
        if type(self.cross_row_finalize) is not bool:
            raise ValueError("row schedule cross_row_finalize must be a bool")
        if type(self.raw_moment_variance) is not bool:
            raise ValueError("row schedule raw_moment_variance must be a bool")
        if type(self.compact_reduction_topology) is not bool:
            raise ValueError("row schedule compact_reduction_topology must be a bool")
        if type(self.rows_per_block) is not int or self.rows_per_block not in {
            1,
            2,
            4,
            8,
            16,
            32,
        }:
            raise ValueError(
                "row schedule rows_per_block must be 1, 2, 4, 8, 16, or 32"
            )
        if self.rows_per_block > 1 and self.block_threads != 32:
            raise ValueError(
                "multi-row schedule requires 32 logical threads per row"
            )
        if (
            type(self.reduction_parts) is not int
            or self.reduction_parts < 1
            or self.reduction_parts > 128
            or self.reduction_parts & (self.reduction_parts - 1)
        ):
            raise ValueError(
                "row schedule reduction_parts must be a power of two in [1, 128]"
            )
        if self.reduction_parts > 1 and self.rows_per_block != 1:
            raise ValueError(
                "multi-stage reduction cannot also assign multiple rows per block"
            )
        if self.reduction_parts > 1 and self.vector_width == 4:
            raise ValueError(
                "multi-stage reduction currently requires scalar feature loads"
            )
        if self.cross_row_finalize and (
            self.rows_per_block != 1 or self.reduction_parts != 1
        ):
            raise ValueError(
                "cross-row finalize requires one row per block and one row stage"
            )
        if not self.compact_reduction_topology and (
            self.rows_per_block > 1 or self.leader_finalize
        ):
            raise ValueError(
                "multi-row and leader-finalize schedules require the compact "
                "reduction topology"
            )
        if (
            type(self.accumulator_chains) is not int
            or self.accumulator_chains not in {1, 2, 4}
        ):
            raise ValueError("row schedule accumulator_chains must be 1, 2, or 4")
        if self.accumulator_chains > 1 and (
            self.rows_per_block != 1 or self.reduction_parts != 1
        ):
            raise ValueError(
                "multi-accumulator reduction requires one row and one stage"
            )

    @property
    def num_warps(self) -> int:
        return self.block_threads // 32

    @property
    def pass_argument(self) -> str:
        return (
            "-convert-intent-row-reduction-to-gpu="
            f"block-threads={self.block_threads} tile-cols={self.tile_cols} "
            f"vector-width={self.vector_width} "
            f"fuse-fma={'true' if self.fuse_fma else 'false'} "
            f"approximate-div={'true' if self.approximate_div else 'false'} "
            f"leader-finalize={'true' if self.leader_finalize else 'false'} "
            "promote-rank0-scalars="
            f"{'true' if self.promote_rank0_scalars else 'false'} "
            "predicate-pure-value-if=true "
            "defer-epilogue-col-loads="
            f"{'true' if self.defer_epilogue_col_loads else 'false'} "
            f"rows-per-block={self.rows_per_block} "
            f"reduction-parts={self.reduction_parts} "
            "materialize-shared-row-output="
            f"{'true' if self.materialize_shared_row_output else 'false'} "
            "cross-row-finalize="
            f"{'true' if self.cross_row_finalize else 'false'} "
            "raw-moment-variance="
            f"{'true' if self.raw_moment_variance else 'false'} "
            "compact-reduction-topology="
            f"{'true' if self.compact_reduction_topology else 'false'} "
            f"accumulator-chains={self.accumulator_chains}"
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "block_threads": self.block_threads,
            "tile_cols": self.tile_cols,
            "num_warps": self.num_warps,
            "vector_width": self.vector_width,
            "fuse_fma": self.fuse_fma,
            "approximate_div": self.approximate_div,
            "leader_finalize": self.leader_finalize,
            "promote_rank0_scalars": self.promote_rank0_scalars,
            "defer_epilogue_col_loads": self.defer_epilogue_col_loads,
            "rows_per_block": self.rows_per_block,
            "reduction_parts": self.reduction_parts,
            "materialize_shared_row_output": self.materialize_shared_row_output,
            "cross_row_finalize": self.cross_row_finalize,
            "raw_moment_variance": self.raw_moment_variance,
            "compact_reduction_topology": self.compact_reduction_topology,
            "accumulator_chains": self.accumulator_chains,
        }


@dataclass(frozen=True)
class ElementwiseSchedule:
    block_threads: int
    flatten_contiguous: bool = True
    vector_width: int = 4
    promote_uniform_scalars: bool = False
    tanh_approximation: str = "stable-ex2"
    grid_stride_unroll: int = 1
    logical_tile_elements: int = 0
    paired_head_reuse: bool = False
    paired_head_group_size: int = 4
    paired_direct_plane: bool = False

    def __post_init__(self) -> None:
        if type(self.block_threads) is not int or self.block_threads not in _BLOCK_THREADS:
            raise ValueError(
                "elementwise block_threads must be a supported warp multiple"
            )
        if type(self.flatten_contiguous) is not bool:
            raise ValueError("elementwise flatten_contiguous must be boolean")
        if type(self.vector_width) is not int or self.vector_width not in {1, 2, 4}:
            raise ValueError("elementwise vector_width must be 1, 2, or 4")
        if type(self.promote_uniform_scalars) is not bool:
            raise ValueError("promote_uniform_scalars must be boolean")
        if type(self.paired_head_reuse) is not bool:
            raise ValueError("paired_head_reuse must be boolean")
        if (
            type(self.paired_head_group_size) is not int
            or self.paired_head_group_size not in {2, 4, 8, 16}
        ):
            raise ValueError("paired_head_group_size must be 2, 4, 8, or 16")
        if type(self.paired_direct_plane) is not bool:
            raise ValueError("paired_direct_plane must be boolean")
        if self.tanh_approximation not in {"stable-ex2", "native-ptx"}:
            raise ValueError(
                "elementwise tanh_approximation must be stable-ex2 or native-ptx"
            )
        if self.grid_stride_unroll not in {1, 2, 4, 8}:
            raise ValueError("elementwise grid_stride_unroll must be 1, 2, 4, or 8")
        if (
            type(self.logical_tile_elements) is not int
            or self.logical_tile_elements < 0
            or self.logical_tile_elements > 8192
            or (
                self.logical_tile_elements
                and self.logical_tile_elements
                & (self.logical_tile_elements - 1)
            )
        ):
            raise ValueError(
                "elementwise logical_tile_elements must be zero or a power of two <= 8192"
            )
        if self.flatten_contiguous and self.vector_width not in {2, 4}:
            raise ValueError("flattened elementwise vector_width must be 2 or 4")
        if self.flatten_contiguous and self.grid_stride_unroll != 1:
            raise ValueError("flattened elementwise schedule requires unroll=1")
        if self.flatten_contiguous and self.logical_tile_elements != 0:
            raise ValueError("flattened elementwise schedule requires logical tile=0")
        if (self.paired_head_reuse or self.paired_direct_plane) and (
            self.flatten_contiguous
            or self.vector_width not in {1, 2}
            or self.grid_stride_unroll != 1
            or self.logical_tile_elements != 0
            or self.tanh_approximation != "stable-ex2"
        ):
            raise ValueError(
                "paired structural states require width 1/2, non-flattened, stable elementwise schedule"
            )
        if self.paired_direct_plane and self.vector_width != 1:
            raise ValueError("paired direct-plane lowering requires vector_width=1")

    @property
    def pass_argument(self) -> str:
        argument = (
            "-convert-intent-elementwise-to-gpu="
            f"block-threads={self.block_threads}"
        )
        if self.flatten_contiguous:
            argument += (
                " flatten-contiguous=true "
                f"flat-vector-width={self.vector_width}"
            )
        else:
            argument += f" row-vector-width={self.vector_width}"
        if self.promote_uniform_scalars:
            argument += " promote-uniform-scalars=true"
        argument += f" grid-stride-unroll={self.grid_stride_unroll}"
        argument += f" logical-tile-elements={self.logical_tile_elements}"
        if self.paired_head_reuse:
            argument += (
                " paired-head-reuse=true"
                f" paired-head-group-size={self.paired_head_group_size}"
            )
        if self.paired_direct_plane:
            argument += " paired-direct-plane=true"
        argument += f" tanh-approximation={self.tanh_approximation}"
        return argument

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "block_threads": self.block_threads,
            "flatten_contiguous": self.flatten_contiguous,
            "vector_width": self.vector_width,
            "promote_uniform_scalars": self.promote_uniform_scalars,
            "tanh_approximation": self.tanh_approximation,
            "grid_stride_unroll": self.grid_stride_unroll,
            "logical_tile_elements": self.logical_tile_elements,
            "paired_head_reuse": self.paired_head_reuse,
            "paired_head_group_size": self.paired_head_group_size,
            "paired_direct_plane": self.paired_direct_plane,
        }


@dataclass(frozen=True)
class PaddedRowSchedule:
    """Launch geometry for the structural masked-short-dot/pad path."""

    block_threads: int
    vectorize_full_features: bool = True
    promote_rank0_scalars: bool = True
    reduction_topology: str = "serial-row"
    rows_per_subgroup: int = 1
    lanes_per_row: int = 32

    def __post_init__(self) -> None:
        if type(self.block_threads) is not int or self.block_threads not in _BLOCK_THREADS:
            raise ValueError(
                "padded-row schedule block_threads must be a supported warp multiple"
            )
        if type(self.vectorize_full_features) is not bool:
            raise ValueError(
                "padded-row vectorize_full_features must be boolean"
            )
        if type(self.promote_rank0_scalars) is not bool:
            raise ValueError(
                "padded-row promote_rank0_scalars must be boolean"
            )
        if self.reduction_topology not in {"serial-row", "subgroup-feature"}:
            raise ValueError(
                "padded-row reduction_topology must be serial-row or "
                "subgroup-feature"
            )
        if type(self.rows_per_subgroup) is not int or self.rows_per_subgroup not in {
            1,
            2,
            4,
            8,
        }:
            raise ValueError("padded-row rows_per_subgroup must be 1, 2, 4, or 8")
        if self.reduction_topology == "serial-row" and self.rows_per_subgroup != 1:
            raise ValueError("serial-row requires rows_per_subgroup=1")
        if self.lanes_per_row not in {4, 8, 16, 32}:
            raise ValueError("padded-row lanes_per_row must be 4, 8, 16, or 32")
        if self.reduction_topology == "serial-row" and self.lanes_per_row != 32:
            raise ValueError("serial-row requires lanes_per_row=32")

    @property
    def pass_argument(self) -> str:
        return (
            "-convert-intent-padded-row-reduction-to-gpu="
            f"block-threads={self.block_threads} "
            "vectorize-full-features="
            f"{'true' if self.vectorize_full_features else 'false'} "
            "promote-rank0-scalars="
            f"{'true' if self.promote_rank0_scalars else 'false'} "
            "reduction-topology="
            f"{self.reduction_topology} "
            f"rows-per-subgroup={self.rows_per_subgroup} "
            f"lanes-per-row={self.lanes_per_row}"
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "block_threads": self.block_threads,
            "num_warps": self.block_threads // 32,
            "vectorize_full_features": self.vectorize_full_features,
            "promote_rank0_scalars": self.promote_rank0_scalars,
            "reduction_topology": self.reduction_topology,
            "rows_per_subgroup": self.rows_per_subgroup,
            "lanes_per_row": self.lanes_per_row,
        }


REGISTERED_DEFAULT_SCHEDULES: dict[str, dict[str, int | bool]] = {
    "elementwise": ElementwiseSchedule(64, False, 1).to_dict(),
    "row_reduction": RowSchedule(1024, 32).to_dict(),
    "padded_row_reduction": PaddedRowSchedule(64, False, True).to_dict(),
}


@dataclass(frozen=True)
class ScheduleDecision:
    fast_path_id: str
    schedule: RowSchedule | ElementwiseSchedule | PaddedRowSchedule
    observed_block_threads: int
    source: str
    optimization: Mapping[str, Any] | None = None

    @property
    def pass_argument(self) -> str:
        return self.schedule.pass_argument

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast_path_id": self.fast_path_id,
            "schedule": self.schedule.to_dict(),
            "pass_argument": self.pass_argument,
            "observed_block_threads": self.observed_block_threads,
            "source": self.source,
            "optimization": (
                dict(self.optimization) if self.optimization is not None else None
            ),
        }


def _observed_block_threads(layer_c: Mapping[str, Any]) -> tuple[int, str]:
    resolved = resolve_layer_c_scalar(layer_c, "num_warps")
    num_warps = resolved.value
    if isinstance(num_warps, bool) or not isinstance(num_warps, int):
        raise ValueError("Layer-C num_warps must be an integer")
    block_threads = num_warps * 32
    if block_threads not in _BLOCK_THREADS:
        raise ValueError(f"unsupported Layer-C num_warps={num_warps}")
    return block_threads, resolved.path


def resolve_row_schedule(
    intent_module: str,
    layer_c: Mapping[str, Any],
    *,
    schedule_override: RowSchedule | None = None,
) -> ScheduleDecision:
    """Resolve the preserved structural row policies on accepted MLIR."""

    text = read_intent_module(intent_module)
    observed_block_threads, source = _observed_block_threads(layer_c)
    selected = RowSchedule(observed_block_threads, 32)
    optimization: dict[str, Any] | None = None
    inputs = backend_orchestrate._parse_intent_signature(text)
    results = backend_orchestrate._parse_intent_results(text)
    row_inputs = [
        item for item in inputs if item.dtype_str == "f32" and len(item.shape) == 2
    ]
    col_inputs = [
        item for item in inputs if item.dtype_str == "f32" and len(item.shape) == 1
    ]
    scalar_inputs = [
        item
        for item in inputs
        if item.dtype_str == "f32" and tuple(item.shape) == ()
    ]
    integer_scalar_inputs = [
        item
        for item in inputs
        if item.dtype_str in _INTEGER_DTYPES and tuple(item.shape) == ()
    ]
    per_row_results = [
        item for item in results if item[1] == "f32" and len(item[0]) == 1
    ]
    same_row_shape = bool(
        len(row_inputs) == 2
        and tuple(row_inputs[0].shape) == tuple(row_inputs[1].shape)
    )
    row_width = (
        row_inputs[0].shape[1]
        if same_row_shape and len(row_inputs[0].shape) == 2
        else None
    )

    single_log_row_reduce = bool(
        observed_block_threads == 256
        and len(inputs) == 3
        and len(scalar_inputs) == 1
        and len(results) == 1
        and len(per_row_results) == 1
        and same_row_shape
        and isinstance(row_width, int)
        and row_width >= 4096
        and text.count("intent.reduce") == 1
        and text.count('fn = "log"') == 1
        and all(
            marker not in text
            for marker in (
                "intent.gather",
                "intent.scan",
                "intent.loop",
                "intent.transpose",
            )
        )
    )
    if single_log_row_reduce:
        selected = RowSchedule(64, 64)
        source = "structural.single_log_row_reduce.v2"
        optimization = {
            "policy_id": "cuda.row.single_log_reduce.occupancy.v2",
            "kind": "deterministic_structural_schedule",
            "function_name_used": False,
            "source_symbol_used": False,
            "python_graph_rewrite": False,
            "observed_block_threads": observed_block_threads,
            "selected_block_threads": selected.block_threads,
            "selected_tile_cols": selected.tile_cols,
        }

    centered_uniform_select = bool(
        observed_block_threads == 256
        and len(row_inputs) == 1
        and row_inputs[0].shape[1] == 4096
        and len(scalar_inputs) == 1
        and len(integer_scalar_inputs) == 1
        and len(results) == 3
        and len(per_row_results) == 2
        and text.count("intent.reduce") == 2
        and text.count("intent.select") == 1
        and text.count("intent.contraction") == 2
        and text.count('fn = "sqrt"') == 1
        and text.count('fn = "recip"') == 1
        and all(
            marker not in text
            for marker in ("intent.gather", "intent.scan", "intent.loop")
        )
    )
    if centered_uniform_select:
        selected = RowSchedule(256, 32, vector_width=1, fuse_fma=True)
        source = "structural.centered-uniform-select.retune.v1"
        optimization = {
            "policy_id": "cuda.row.centered-uniform-select.sm120-retune.v1",
            "kind": "measured_structural_schedule",
            "function_name_used": False,
            "source_symbol_used": False,
            "python_graph_rewrite": False,
            "observed_block_threads": observed_block_threads,
            "selected_block_threads": selected.block_threads,
            "selected_vector_width": selected.vector_width,
            "selected_fuse_fma": selected.fuse_fma,
        }

    row_widths = {
        item.shape[1]
        for item in row_inputs
        if len(item.shape) == 2 and isinstance(item.shape[1], int)
    }
    small_register_rmsnorm = bool(
        observed_block_threads == 128
        and 1 <= len(row_inputs) <= 2
        and row_widths == {1024}
        and len(col_inputs) == 1
        and tuple(col_inputs[0].shape) == (1024,)
        and len(scalar_inputs) == 1
        and len(integer_scalar_inputs) == 0
        and len(results) in {1, 2}
        and text.count("intent.reduce") == 1
        and text.count('fn = "rsqrt"') == 1
        and all(
            marker not in text
            for marker in (
                "intent.gather",
                "intent.scan",
                "intent.loop",
                "intent.transpose",
            )
        )
    )
    if small_register_rmsnorm:
        selected = RowSchedule(
            128,
            32,
            defer_epilogue_col_loads=False,
        )
        source = "structural.small-register-rmsnorm.preload-col.v1"
        optimization = {
            "policy_id": "cuda.row.small-register-rmsnorm.preload-col.v1",
            "kind": "measured_structural_schedule",
            "function_name_used": False,
            "source_symbol_used": False,
            "python_graph_rewrite": False,
            "observed_block_threads": observed_block_threads,
            "selected_defer_epilogue_col_loads": False,
        }

    if schedule_override is not None:
        selected = schedule_override
        source = "offline.layer_c_retune.v1"
        optimization = {
            "policy_id": "cuda.row.offline-layer-c-retune.v1",
            "kind": "explicit_offline_schedule_candidate",
            "function_name_used": False,
            "source_symbol_used": False,
            "python_graph_rewrite": False,
            "observed_block_threads": observed_block_threads,
            "candidate": selected.to_dict(),
        }
    return ScheduleDecision(
        fast_path_id=ROW_FAST_PATH_ID,
        schedule=selected,
        observed_block_threads=observed_block_threads,
        source=source,
        optimization=optimization,
    )


def resolve_elementwise_schedule(
    intent_module: str,
    layer_c: Mapping[str, Any],
    *,
    schedule_override: ElementwiseSchedule | None = None,
) -> ScheduleDecision:
    """Resolve row or flattened schedules for a pure-elementwise graph."""

    text = read_intent_module(intent_module)
    observed_block_threads, source = _observed_block_threads(layer_c)
    inputs = backend_orchestrate._parse_intent_signature(text)
    results = backend_orchestrate._parse_intent_results(text)
    if not results:
        raise ValueError("elementwise schedule requires at least one result")
    result_shape, result_dtype = results[0]
    if result_dtype != "f32" or len(result_shape) != 2:
        raise ValueError("elementwise schedule requires a rank-2 f32 result")
    if any(shape != result_shape or dtype != result_dtype for shape, dtype in results):
        raise ValueError("elementwise results must share one shape and dtype")
    m, n = (int(result_shape[0]), int(result_shape[1]))
    all_inputs_equal_rows = bool(inputs) and all(
        item.dtype_str == "f32" and tuple(item.shape) == tuple(result_shape)
        for item in inputs
    )
    has_explicit_strides = any(
        marker in text for marker in ("intent.strides", "intent.result_strides")
    )
    has_indexed_body = any(
        marker in text for marker in ("intent.iota", "intent.select")
    )
    flat_packet = 128 * 4
    flat_eligible = bool(
        all_inputs_equal_rows
        and not has_explicit_strides
        and not has_indexed_body
        and (m * n) % flat_packet == 0
        and (m * n) >= flat_packet * 170
    )

    optimization: dict[str, Any] | None = None
    if schedule_override is not None:
        if schedule_override.flatten_contiguous and not flat_eligible:
            raise ValueError(
                "flatten-contiguous override requires a structurally eligible graph"
            )
        selected = schedule_override
        source = "explicit.layer_c_schedule_ablation.v1"
        optimization = {
            "policy_id": "cuda.elementwise.explicit-schedule-ablation.v1",
            "kind": "explicit_layer_c_schedule_ablation",
            "function_name_used": False,
            "source_symbol_used": False,
            "python_graph_rewrite": False,
            "structural_flat_eligibility": flat_eligible,
            "observed_block_threads": observed_block_threads,
            "candidate": selected.to_dict(),
        }
    elif flat_eligible:
        selected = ElementwiseSchedule(128, True, 4)
        source = "structural.flat_equal_shape_contiguous.v1"
        optimization = {
            "policy_id": "cuda.elementwise.flat.vec4.four_warp.v1",
            "kind": "deterministic_structural_schedule",
            "function_name_used": False,
            "source_symbol_used": False,
            "python_graph_rewrite": False,
            "observed_block_threads": observed_block_threads,
            "selected_block_threads": selected.block_threads,
            "selected_vector_width": selected.vector_width,
        }
    else:
        block_threads = min(observed_block_threads, n)
        while block_threads > 1 and n % block_threads != 0:
            block_threads //= 2
        if block_threads not in _BLOCK_THREADS:
            raise ValueError(
                "elementwise row schedule requires a supported warp multiple"
            )
        selected = ElementwiseSchedule(block_threads, False, 1)
    return ScheduleDecision(
        fast_path_id=ELEMENTWISE_FAST_PATH_ID,
        schedule=selected,
        observed_block_threads=observed_block_threads,
        source=source,
        optimization=optimization,
    )


def pass_override(decision: ScheduleDecision) -> dict[str, str]:
    return {decision.fast_path_id: decision.pass_argument}


__all__ = [
    "ELEMENTWISE_FAST_PATH_ID",
    "ElementwiseSchedule",
    "PADDED_ROW_FAST_PATH_ID",
    "PaddedRowSchedule",
    "REGISTERED_DEFAULT_SCHEDULES",
    "ROW_FAST_PATH_ID",
    "RowSchedule",
    "ScheduleDecision",
    "pass_override",
    "resolve_elementwise_schedule",
    "resolve_row_schedule",
]
