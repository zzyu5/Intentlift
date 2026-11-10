"""True MLIR -> LLVM -> native RVV object compilation.

Python orchestrates registered C++/upstream MLIR passes and LLVM tools.  It
does not emit target kernel source.  The generated object originates from the
same deterministic Intent MLIR M0 consumed by the CUDA backend.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, replace
from typing import Any, Mapping

from intentir.backend.contract_capabilities import B_CAP_READONLY_INPUTS
from intentir.backend.materialize import (
    BackendSearchState,
    MaterializedJ,
    materialize_j,
)
from intentir.backend.protocol import LoweringRequest
from intentir.backend.protocol import (
    TargetBindingContext,
    TargetPluginArtifact,
    deserialize_target_artifact,
    serialize_target_artifact,
)
from intentir.backend.riscv.contract import (
    derive_riscv_contract_view,
    riscv_allowed_authorizations,
)
from intentir.backend.toolchain.tools import (
    INTENT_OPT,
    LLVM_BIN,
    MLIR_OPT,
    MLIR_TRANSLATE,
)
from intentir.ir.layer_c import LayerCInterface, TargetLayerCPlugin, layer_c_content_digest
from intentir.ir.intent_op_contract import RAGGED_OPS


OPT = os.environ.get("LLVM_OPT", str(LLVM_BIN / "opt"))
LLC = os.environ.get("LLC", str(LLVM_BIN / "llc"))
TRIPLE = os.environ.get("INTENT_RISCV_TRIPLE", "riscv64-unknown-linux-gnu")
MATTR = os.environ.get("INTENT_RISCV_MATTR", "+v,+zfh,+zvfh")

RVV_B_FUSION = "riscv.b.readonly_producer_consumer_fusion.v2"
RVV_B_FUSED_SOFTMAX = "riscv.b.readonly_fused_softmax.v1"
RVV_B_GATHER_COMPOSITION = "riscv.b.readonly_gather_composition.v1"


def _has_ragged_ops(mlir: str) -> bool:
    return any(f"intent.{name}" in mlir for name in RAGGED_OPS)


def ragged_requires_scalar_schedule(mlir: str) -> bool:
    """Return whether the module exposes the dynamic Ragged member ABI."""

    return "intent.ragged_member" in mlir


def _run_text(command: list[str], *, stdin: str) -> str:
    process = subprocess.run(
        command,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{command[0]} failed with exit {process.returncode}:\n"
            f"{process.stderr}"
        )
    return process.stdout


def _run_bytes(command: list[str], *, stdin: str) -> bytes:
    process = subprocess.run(
        command,
        input=stdin.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{command[0]} failed with exit {process.returncode}:\n"
            + process.stderr.decode("utf-8", errors="replace")
        )
    return process.stdout


@dataclass(frozen=True)
class RiscvTransformationState:
    fuse_producers: bool
    reassociate_reductions: bool = False
    explicit_vector_reduction: bool = False
    joint_vector_reduction: bool = False
    fused_softmax: bool = False
    rowmask_zero_fill: bool = False
    composed_indexed_gather: bool = False
    fuse_sibling_outputs: bool = False
    approximate_exp: bool = False
    mixed_precision_dot_reduction: bool = False
    paired_rotation: bool = False
    paired_rotation_head_angle_reuse: bool = False
    streaming_attention: bool = False

    def __post_init__(self) -> None:
        if type(self.fuse_producers) is not bool:
            raise TypeError("fuse_producers must be bool")
        if type(self.reassociate_reductions) is not bool:
            raise TypeError("reassociate_reductions must be bool")
        if type(self.explicit_vector_reduction) is not bool:
            raise TypeError("explicit_vector_reduction must be bool")
        if type(self.joint_vector_reduction) is not bool:
            raise TypeError("joint_vector_reduction must be bool")
        if type(self.fused_softmax) is not bool:
            raise TypeError("fused_softmax must be bool")
        if type(self.rowmask_zero_fill) is not bool:
            raise TypeError("rowmask_zero_fill must be bool")
        if type(self.composed_indexed_gather) is not bool:
            raise TypeError("composed_indexed_gather must be bool")
        if type(self.fuse_sibling_outputs) is not bool:
            raise TypeError("fuse_sibling_outputs must be bool")
        if type(self.approximate_exp) is not bool:
            raise TypeError("approximate_exp must be bool")
        if type(self.mixed_precision_dot_reduction) is not bool:
            raise TypeError("mixed_precision_dot_reduction must be bool")
        if type(self.paired_rotation) is not bool:
            raise TypeError("paired_rotation must be bool")
        if type(self.paired_rotation_head_angle_reuse) is not bool:
            raise TypeError("paired_rotation_head_angle_reuse must be bool")
        if type(self.streaming_attention) is not bool:
            raise TypeError("streaming_attention must be bool")
        if self.paired_rotation_head_angle_reuse and not self.paired_rotation:
            raise ValueError(
                "paired-rotation head-angle reuse requires paired_rotation"
            )
        if self.explicit_vector_reduction and not self.reassociate_reductions:
            raise ValueError(
                "explicit float vector reduction requires the admitted "
                "reassociation state"
            )
        if self.joint_vector_reduction and not (
            self.fuse_producers
            and self.reassociate_reductions
            and self.explicit_vector_reduction
        ):
            raise ValueError(
                "joint RVV vector reduction requires producer fusion, "
                "reassociation, and explicit vector reduction"
            )
        if self.fused_softmax and not self.reassociate_reductions:
            raise ValueError(
                "fused RVV Softmax requires the admitted reduction "
                "reassociation state"
            )
        if self.fused_softmax and (
            self.fuse_producers
            or self.explicit_vector_reduction
            or self.joint_vector_reduction
        ):
            raise ValueError(
                "fused RVV Softmax is a complete structural state and cannot "
                "be combined with generic fusion or row-reduction states"
            )
        if self.approximate_exp and not self.fused_softmax:
            raise ValueError(
                "approximate RVV exp is only defined inside the fused Softmax state"
            )
        if self.mixed_precision_dot_reduction and not (
            self.fuse_producers and self.reassociate_reductions
        ):
            raise ValueError(
                "mixed-precision RVV dot reduction requires producer fusion "
                "and admitted reduction reassociation"
            )
        if self.mixed_precision_dot_reduction and (
            self.explicit_vector_reduction
            or self.joint_vector_reduction
            or self.fused_softmax
            or self.rowmask_zero_fill
            or self.composed_indexed_gather
            or self.fuse_sibling_outputs
        ):
            raise ValueError(
                "mixed-precision RVV dot reduction is a complete structural "
                "state and cannot be combined with another specialized state"
            )
        if self.rowmask_zero_fill and (
            self.fuse_producers
            or self.reassociate_reductions
            or self.explicit_vector_reduction
            or self.joint_vector_reduction
            or self.fused_softmax
        ):
            raise ValueError(
                "row-mask zero-fill is a complete structural state and cannot "
                "be combined with other RVV transformations"
            )
        if self.composed_indexed_gather and (
            self.fuse_producers
            or self.reassociate_reductions
            or self.explicit_vector_reduction
            or self.joint_vector_reduction
            or self.fused_softmax
            or self.rowmask_zero_fill
        ):
            raise ValueError(
                "composed indexed gather is a complete structural state and "
                "cannot be combined with other RVV transformations"
            )
        if self.fuse_sibling_outputs and not self.fuse_producers:
            raise ValueError(
                "sibling-output fusion requires producer fusion within each root"
            )
        if self.fuse_sibling_outputs and (
            self.fused_softmax
            or self.rowmask_zero_fill
            or self.composed_indexed_gather
        ):
            raise ValueError(
                "sibling-output fusion cannot be combined with a complete "
                "specialized structural state"
            )
        if self.paired_rotation and (
            self.fuse_producers
            or self.reassociate_reductions
            or self.explicit_vector_reduction
            or self.joint_vector_reduction
            or self.fused_softmax
            or self.rowmask_zero_fill
            or self.composed_indexed_gather
            or self.fuse_sibling_outputs
            or self.approximate_exp
            or self.mixed_precision_dot_reduction
        ):
            raise ValueError(
                "paired rotation is a complete direct-loop structural state"
            )
        if self.streaming_attention and not self.reassociate_reductions:
            raise ValueError(
                "streaming attention requires admitted float reduction "
                "reassociation"
            )
        if self.streaming_attention and (
            self.fuse_producers
            or self.explicit_vector_reduction
            or self.joint_vector_reduction
            or self.fused_softmax
            or self.rowmask_zero_fill
            or self.composed_indexed_gather
            or self.fuse_sibling_outputs
            or self.approximate_exp
            or self.mixed_precision_dot_reduction
            or self.paired_rotation
        ):
            raise ValueError(
                "streaming attention is a complete direct-loop structural state"
            )

    @property
    def lowering_path_kind(self) -> str:
        if self.streaming_attention:
            return "streaming_attention_vector"
        if self.paired_rotation:
            return (
                "paired_rotation_head_angle_reuse"
                if self.paired_rotation_head_angle_reuse
                else "paired_rotation_contiguous"
            )
        if self.composed_indexed_gather:
            return "composed_indexed_gather"
        if self.rowmask_zero_fill:
            return "rowmask_zero_fill"
        if self.fused_softmax:
            return (
                "fused_softmax_vector_approximate_exp"
                if self.approximate_exp
                else "fused_softmax_hybrid"
            )
        if self.mixed_precision_dot_reduction:
            return "mixed_precision_dot_vector_reduction"
        if self.fuse_sibling_outputs:
            return "producer_consumer_sibling_fused"
        if self.joint_vector_reduction:
            return "producer_consumer_fused_joint_vector_reduction"
        if self.explicit_vector_reduction:
            return (
                "producer_consumer_fused_vector_reduction"
                if self.fuse_producers
                else "explicit_vector_reduction"
            )
        return "producer_consumer_fused" if self.fuse_producers else "materialized"

    @property
    def transformation_trace(self) -> tuple[str, ...]:
        structural = (
            "rvv.attention.streaming_vector.v1"
            if self.streaming_attention
            else "rvv.elementwise.paired_rotation.head_angle_reuse.v1"
            if self.paired_rotation_head_angle_reuse
            else "rvv.elementwise.paired_rotation.contiguous.v1"
            if self.paired_rotation
            else "rvv.gather.composed_indexed.v1"
            if self.composed_indexed_gather
            else "rvv.elementwise.rowmask.zero_fill.v1"
            if self.rowmask_zero_fill
            else "rvv.vector.fused_softmax_hybrid.v1"
            if self.fused_softmax
            else "rvv.linalg.producer_consumer_fusion.v2"
            if self.fuse_producers
            else "rvv.linalg.materialized.v1"
        )
        numeric = (
            ("rvv.numeric.explicit_reduction_reassociation.v1",)
            if self.reassociate_reductions
            else ()
        )
        approximation = (
            ("rvv.numeric.execution_domain_approximate_exp.v1",)
            if self.approximate_exp
            else ()
        )
        vector = (
            ("rvv.vector.explicit_row_reduction.v1",)
            if self.explicit_vector_reduction
            else ()
        )
        joint = (
            ("rvv.vector.joint_row_statistics.v1",)
            if self.joint_vector_reduction
            else ()
        )
        mixed_dot = (
            ("rvv.vector.mixed_precision_dot_reduction.v1",)
            if self.mixed_precision_dot_reduction
            else ()
        )
        sibling = (
            ("rvv.linalg.sibling_output_fusion.v1",)
            if self.fuse_sibling_outputs
            else ()
        )
        return (
            structural,
            *numeric,
            *approximation,
            *vector,
            *joint,
            *mixed_dot,
            *sibling,
        )

    @property
    def b_transformations(self) -> tuple[str, ...]:
        if self.composed_indexed_gather:
            return (RVV_B_GATHER_COMPOSITION,)
        if self.fused_softmax:
            return (RVV_B_FUSED_SOFTMAX,)
        return (RVV_B_FUSION,) if self.fuse_producers else ()

    @property
    def required_b_capabilities(self) -> tuple[str, ...]:
        return (
            (B_CAP_READONLY_INPUTS,)
            if (
                self.fuse_producers
                or self.fused_softmax
                or self.composed_indexed_gather
            )
            else ()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lowering_path_kind": self.lowering_path_kind,
            "fuse_producers": self.fuse_producers,
            "reassociate_reductions": self.reassociate_reductions,
            "explicit_vector_reduction": self.explicit_vector_reduction,
            "joint_vector_reduction": self.joint_vector_reduction,
            "fused_softmax": self.fused_softmax,
            "rowmask_zero_fill": self.rowmask_zero_fill,
            "composed_indexed_gather": self.composed_indexed_gather,
            "fuse_sibling_outputs": self.fuse_sibling_outputs,
            "approximate_exp": self.approximate_exp,
            "mixed_precision_dot_reduction": self.mixed_precision_dot_reduction,
            "paired_rotation": self.paired_rotation,
            "paired_rotation_head_angle_reuse": (
                self.paired_rotation_head_angle_reuse
            ),
            "streaming_attention": self.streaming_attention,
            "transformation_trace": list(self.transformation_trace),
            "b_transformations": list(self.b_transformations),
            "required_b_capabilities": list(self.required_b_capabilities),
        }


@dataclass(frozen=True)
class RiscvSchedule:
    opt_level: int
    vectorize: bool
    vector_bits: int = 128
    parallel_tile: int = 0
    reduction_tile: int = 0
    thread_count: int = 1
    scalable_vectorization: bool = False
    vector_lanes: int = 4

    def __post_init__(self) -> None:
        if self.opt_level not in {2, 3}:
            raise ValueError("RVV opt_level must be 2 or 3")
        if type(self.vectorize) is not bool:
            raise TypeError("vectorize must be bool")
        if type(self.scalable_vectorization) is not bool:
            raise TypeError("scalable_vectorization must be bool")
        if self.vector_bits not in {0, 128}:
            raise ValueError("RVV vector_bits must be 0 or native VLEN=128")
        if self.scalable_vectorization and (
            not self.vectorize or self.vector_bits != 0
        ):
            raise ValueError(
                "RVV scalable vectorization requires vectorize=true and "
                "vector_bits=0"
            )
        if self.vector_bits == 0 and not self.scalable_vectorization:
            raise ValueError(
                "RVV vector_bits=0 is reserved for scalable vectorization"
            )
        if (
            type(self.vector_lanes) is not int
            or self.vector_lanes < 1
            or self.vector_lanes > 16
            or self.vector_lanes & (self.vector_lanes - 1)
        ):
            raise ValueError("RVV vector_lanes must be a power of two in [1, 16]")
        if self.thread_count != 1:
            raise ValueError("RVV production search is single-thread only")
        for name, value in (
            ("parallel_tile", self.parallel_tile),
            ("reduction_tile", self.reduction_tile),
        ):
            if type(value) is not int:
                raise TypeError(f"RVV {name} must be int")
            if value < 0 or value > 1024:
                raise ValueError(f"RVV {name} must be in [0, 1024]")
            if value and value & (value - 1):
                raise ValueError(f"RVV {name} must be zero or a power of two")

    def to_dict(self) -> dict[str, Any]:
        return {
            "opt_level": self.opt_level,
            "vectorize": self.vectorize,
            "vector_bits": self.vector_bits,
            "vector_policy": (
                "scalable"
                if self.scalable_vectorization
                else "fixed"
                if self.vectorize
                else "scalar"
            ),
            "scalable_vectorization": self.scalable_vectorization,
            "parallel_tile": self.parallel_tile,
            "reduction_tile": self.reduction_tile,
            "thread_count": self.thread_count,
            "vector_lanes": self.vector_lanes,
        }


def ragged_schedule_supported(schedule: RiscvSchedule) -> bool:
    """Return the one native schedule shape closed for dynamic Ragged output."""

    return (
        isinstance(schedule, RiscvSchedule)
        and not schedule.vectorize
        and not schedule.scalable_vectorization
        and schedule.parallel_tile == 0
        and schedule.reduction_tile == 0
        and schedule.thread_count == 1
    )


@dataclass(frozen=True)
class RiscvLoweringCandidate:
    transformation: RiscvTransformationState
    schedule: RiscvSchedule

    @property
    def candidate_id(self) -> str:
        schedule = self.schedule
        path = self.transformation.lowering_path_kind
        return (
            f"rvv.{path}.o{schedule.opt_level}."
            f"vec{int(schedule.vectorize)}.vlen{schedule.vector_bits}."
            f"scalable{int(schedule.scalable_vectorization)}."
            f"lanes{schedule.vector_lanes}."
            f"pt{schedule.parallel_tile}.rt{schedule.reduction_tile}."
            f"thr{schedule.thread_count}."
            f"reassoc{int(self.transformation.reassociate_reductions)}."
            f"explicitvec{int(self.transformation.explicit_vector_reduction)}."
            f"jointvec{int(self.transformation.joint_vector_reduction)}"
            f".fusedsoftmax{int(self.transformation.fused_softmax)}"
            f".approxexp{int(self.transformation.approximate_exp)}"
            f".mixeddot{int(self.transformation.mixed_precision_dot_reduction)}"
            + (
                ".streamattn1"
                if self.transformation.streaming_attention
                else ""
            )
            + (".rowmaskzero1" if self.transformation.rowmask_zero_fill else "")
            + (
                ".composedgather1"
                if self.transformation.composed_indexed_gather
                else ""
            )
            + (
                ".siblingfuse1"
                if self.transformation.fuse_sibling_outputs
                else ""
            )
            + (
                ".pairedrotation1.headreuse"
                f"{int(self.transformation.paired_rotation_head_angle_reuse)}"
                if self.transformation.paired_rotation
                else ""
            )
        )

    def _content(self) -> dict[str, Any]:
        return {
            "transformation": self.transformation.to_dict(),
            "schedule": self.schedule.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **self._content()}


@dataclass(frozen=True)
class RiscvTransformationRecord:
    linalg_ops_before: int
    linalg_ops_after: int
    parallel_tile_requested: int = 0
    reduction_tile_requested: int = 0
    parallel_tile_applied: int = 0
    reduction_tile_applied: int = 0
    linalg_tiled_ops: int = 0
    generated_tile_loops: int = 0
    fused_producers: int = 0
    fusion_outer_parallel_only_ops: int = 0
    fusion_multi_use_rejections: int = 0
    openmp_parallel_ops: int = 0
    reassociated_float_combines: int = 0
    explicit_vector_reductions: int = 0
    explicit_vector_reduction_width: int = 0
    explicit_vector_reduction_loops: int = 0
    explicit_vector_reduction_scalable: int = 0
    explicit_vector_max_reductions: int = 0
    joint_vector_reduction_groups: int = 0
    joint_vector_reductions: int = 0
    joint_vector_reduction_root_loads: int = 0
    joint_raw_moment_variances: int = 0
    fused_softmax_count: int = 0
    fused_softmax_input_traversals: int = 0
    fused_softmax_full_row_traversals: int = 0
    fused_softmax_removed_materializations: int = 0
    fused_softmax_vector_reductions: int = 0
    fused_softmax_scalar_reductions: int = 0
    fused_softmax_vector_width: int = 0
    fused_softmax_scalable: int = 0
    fused_softmax_approximate_exp: int = 0
    mixed_dot_reductions: int = 0
    mixed_dot_reduction_width: int = 0
    mixed_dot_reduction_scalable: int = 0
    rowmask_lowerings: int = 0
    rowmask_zero_fills: int = 0
    rowmask_mask_hoists: int = 0
    rowmask_vector_width: int = 0
    composed_gather_lowerings: int = 0
    composed_gather_chain_depth: int = 0
    composed_gather_vector_width: int = 0
    sibling_output_fusions: int = 0
    sibling_output_eliminated_traversals: int = 0
    paired_rotation_lowerings: int = 0
    paired_rotation_head_angle_reuses: int = 0
    paired_rotation_vector_width: int = 0
    paired_rotation_scalable: int = 0
    streaming_attention_lowerings: int = 0
    streaming_attention_vector_width: int = 0
    streaming_attention_score_scratch: int = 0
    streaming_attention_removed_materializations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "linalg_ops_before": self.linalg_ops_before,
            "linalg_ops_after": self.linalg_ops_after,
            "parallel_tile_requested": self.parallel_tile_requested,
            "reduction_tile_requested": self.reduction_tile_requested,
            "parallel_tile_applied": self.parallel_tile_applied,
            "reduction_tile_applied": self.reduction_tile_applied,
            "linalg_tiled_ops": self.linalg_tiled_ops,
            "generated_tile_loops": self.generated_tile_loops,
            "fused_producers": self.fused_producers,
            "fusion_outer_parallel_only_ops": self.fusion_outer_parallel_only_ops,
            "fusion_multi_use_rejections": self.fusion_multi_use_rejections,
            "openmp_parallel_ops": self.openmp_parallel_ops,
            "reassociated_float_combines": self.reassociated_float_combines,
            "explicit_vector_reductions": self.explicit_vector_reductions,
            "explicit_vector_reduction_width": self.explicit_vector_reduction_width,
            "explicit_vector_reduction_loops": self.explicit_vector_reduction_loops,
            "explicit_vector_reduction_scalable": (
                self.explicit_vector_reduction_scalable
            ),
            "explicit_vector_max_reductions": self.explicit_vector_max_reductions,
            "joint_vector_reduction_groups": self.joint_vector_reduction_groups,
            "joint_vector_reductions": self.joint_vector_reductions,
            "joint_vector_reduction_root_loads": (
                self.joint_vector_reduction_root_loads
            ),
            "joint_raw_moment_variances": self.joint_raw_moment_variances,
            "fused_softmax_count": self.fused_softmax_count,
            "fused_softmax_input_traversals": (
                self.fused_softmax_input_traversals
            ),
            "fused_softmax_full_row_traversals": (
                self.fused_softmax_full_row_traversals
            ),
            "fused_softmax_removed_materializations": (
                self.fused_softmax_removed_materializations
            ),
            "fused_softmax_vector_reductions": (
                self.fused_softmax_vector_reductions
            ),
            "fused_softmax_scalar_reductions": (
                self.fused_softmax_scalar_reductions
            ),
            "fused_softmax_vector_width": self.fused_softmax_vector_width,
            "fused_softmax_scalable": self.fused_softmax_scalable,
            "fused_softmax_approximate_exp": (
                self.fused_softmax_approximate_exp
            ),
            "mixed_dot_reductions": self.mixed_dot_reductions,
            "mixed_dot_reduction_width": self.mixed_dot_reduction_width,
            "mixed_dot_reduction_scalable": self.mixed_dot_reduction_scalable,
            "rowmask_lowerings": self.rowmask_lowerings,
            "rowmask_zero_fills": self.rowmask_zero_fills,
            "rowmask_mask_hoists": self.rowmask_mask_hoists,
            "rowmask_vector_width": self.rowmask_vector_width,
            "composed_gather_lowerings": self.composed_gather_lowerings,
            "composed_gather_chain_depth": self.composed_gather_chain_depth,
            "composed_gather_vector_width": self.composed_gather_vector_width,
            "sibling_output_fusions": self.sibling_output_fusions,
            "sibling_output_eliminated_traversals": (
                self.sibling_output_eliminated_traversals
            ),
            "paired_rotation_lowerings": self.paired_rotation_lowerings,
            "paired_rotation_head_angle_reuses": (
                self.paired_rotation_head_angle_reuses
            ),
            "paired_rotation_vector_width": self.paired_rotation_vector_width,
            "paired_rotation_scalable": self.paired_rotation_scalable,
            "streaming_attention_lowerings": (
                self.streaming_attention_lowerings
            ),
            "streaming_attention_vector_width": (
                self.streaming_attention_vector_width
            ),
            "streaming_attention_score_scratch": (
                self.streaming_attention_score_scratch
            ),
            "streaming_attention_removed_materializations": (
                self.streaming_attention_removed_materializations
            ),
        }


def validate_riscv_schedule_realization(
    candidate: RiscvLoweringCandidate,
    record: RiscvTransformationRecord,
) -> None:
    """Require the compiler's applied schedule to equal the requested one.

    The requested schedule is part of the candidate identity and the applied
    values are read back from the actual structured MLIR produced by the
    tiling pass.  Letting these values diverge would make the target contract
    describe one realization while the object and consumption evidence
    describe another one.
    """

    if not isinstance(candidate, RiscvLoweringCandidate):
        raise TypeError("RISC-V schedule validation requires a lowering candidate")
    if not isinstance(record, RiscvTransformationRecord):
        raise TypeError("RISC-V schedule validation requires a transformation record")
    requested = candidate.schedule
    mismatches: list[str] = []
    if requested.parallel_tile != record.parallel_tile_applied:
        mismatches.append(
            "parallel_tile requested="
            f"{requested.parallel_tile} applied={record.parallel_tile_applied}"
        )
    if requested.reduction_tile != record.reduction_tile_applied:
        mismatches.append(
            "reduction_tile requested="
            f"{requested.reduction_tile} applied={record.reduction_tile_applied}"
        )
    if mismatches:
        raise RuntimeError(
            "RISC-V requested/applied schedule mismatch: "
            + "; ".join(mismatches)
        )


@dataclass(frozen=True)
class RiscvCompiledObject:
    candidate: RiscvLoweringCandidate
    symbol: str
    object_bytes: bytes
    rvv_instruction_count: int
    rvv_reduction_instruction_count: int
    scalable_vector_ir_count: int
    rvv_vsetvl_modes: tuple[str, ...]
    transformation_record: RiscvTransformationRecord
    # Actual stage outputs are retained for per-record consumption evidence.
    fusion_mlir: str = ""
    memory_layout_mlir: str = ""
    parallel_tiling_mlir: str = ""
    reduction_tiling_mlir: str = ""
    lowered_mlir: str = ""
    llvm_ir: str = ""
    optimized_llvm_ir: str = ""
    assembly: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, RiscvLoweringCandidate):
            raise TypeError("RISC-V compiled object has an invalid candidate")
        if not isinstance(self.object_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("RISC-V compiled object must contain object bytes")
        if not self.object_bytes:
            raise ValueError("RISC-V compiled object must not be empty")
        for field_name in (
            "fusion_mlir",
            "memory_layout_mlir",
            "parallel_tiling_mlir",
            "reduction_tiling_mlir",
            "lowered_mlir",
            "llvm_ir",
            "optimized_llvm_ir",
            "assembly",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"RISC-V compiled object is missing real {field_name} output"
                )
        validate_riscv_schedule_realization(
            self.candidate,
            self.transformation_record,
        )

    @property
    def c_interface_symbol(self) -> str:
        return f"_mlir_ciface_{self.symbol}"

    def to_record(self) -> dict[str, Any]:
        requested_schedule = self.candidate.schedule.to_dict()
        actual_schedule = {
            **requested_schedule,
            "parallel_tile": self.transformation_record.parallel_tile_applied,
            "reduction_tile": self.transformation_record.reduction_tile_applied,
        }
        return {
            "candidate_id": self.candidate.candidate_id,
            "lowering_path_kind": (
                self.candidate.transformation.lowering_path_kind
            ),
            "transformation": self.candidate.transformation.to_dict(),
            "schedule": actual_schedule,
            "requested_schedule": requested_schedule,
            "schedule_realization": {
                "parallel_tile_applied": (
                    self.transformation_record.parallel_tile_applied
                ),
                "reduction_tile_applied": (
                    self.transformation_record.reduction_tile_applied
                ),
                "requested_equals_actual": requested_schedule == actual_schedule,
            },
            "transformation_record": self.transformation_record.to_dict(),
            "symbol": self.symbol,
            "c_interface_symbol": self.c_interface_symbol,
            "object_size_bytes": len(self.object_bytes),
            "lowered_mlir_size_bytes": len(self.lowered_mlir.encode("utf-8")),
            "llvm_ir_size_bytes": len(self.llvm_ir.encode("utf-8")),
            "optimized_llvm_ir_size_bytes": len(
                self.optimized_llvm_ir.encode("utf-8")
            ),
            "assembly_size_bytes": len(self.assembly.encode("utf-8")),
            "fusion_mlir_size_bytes": len(self.fusion_mlir.encode("utf-8")),
            "memory_layout_mlir_size_bytes": len(
                self.memory_layout_mlir.encode("utf-8")
            ),
            "parallel_tiling_mlir_size_bytes": len(
                self.parallel_tiling_mlir.encode("utf-8")
            ),
            "reduction_tiling_mlir_size_bytes": len(
                self.reduction_tiling_mlir.encode("utf-8")
            ),
            "rvv_instruction_count": self.rvv_instruction_count,
            "rvv_reduction_instruction_count": (
                self.rvv_reduction_instruction_count
            ),
            "scalable_vector_ir_count": self.scalable_vector_ir_count,
            "rvv_vsetvl_modes": list(self.rvv_vsetvl_modes),
            "true_mlir_pipeline": True,
            "python_target_kernel_emission": False,
        }


class RiscvJLoweringError(ValueError):
    """Raised at the public J-to-RVV target-plugin boundary."""


@dataclass(frozen=True)
class RiscvJLowering:
    """One RISC-V lowering produced from immutable J and target Layer C."""

    request: LoweringRequest
    artifact: RiscvCompiledObject
    boundary_consumption: Mapping[str, Any]
    concrete_extents: tuple[tuple[str, int], ...]
    materializer_version: str
    materialization_form: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, LoweringRequest):
            raise RiscvJLoweringError(
                "RISC-V J lowering requires a LoweringRequest"
            )
        if not isinstance(self.artifact, RiscvCompiledObject):
            raise RiscvJLoweringError(
                "RISC-V J lowering requires a compiled RVV object"
            )
        if not self.boundary_consumption.get("supported", False):
            raise RiscvJLoweringError(
                "RISC-V lowering lacks complete target realization consumption"
            )
        if self.request.c is not None and self.request.c.contract is not None:
            try:
                self.request.c.contract.validate_consumption_evidence(
                    self.boundary_consumption
                )
            except (TypeError, ValueError) as exc:
                raise RiscvJLoweringError(str(exc)) from exc

    @property
    def concrete_extents_map(self) -> dict[str, int]:
        return dict(self.concrete_extents)

    def to_record(self) -> dict[str, Any]:
        target_artifact = None
        if self.request.c is not None:
            if self.request.e_c is None:
                raise RiscvJLoweringError(
                    "RISC-V target plugin record is missing its shared E_C"
                )
            target_artifact = TargetPluginArtifact(
                e_c=self.request.e_c,
                plugin=self.request.c,
                e_c_mapping=self.request.e_c_mapping,
            )
        return {
            "target": self.request.target.to_dict(),
            "variant": self.request.variant,
            "j_digest": layer_c_content_digest(self.request.j.to_json()),
            "layer_c_interface_digest": self.request.c.interface_digest
            if self.request.c is not None
            else None,
            "constexpr_bindings": dict(self.request.constexpr_bindings),
            "concrete_extents": self.concrete_extents_map,
            "materializer_version": self.materializer_version,
            "materialization_form": self.materialization_form,
            "boundary_consumption": dict(self.boundary_consumption),
            "e_c_interface_digest": (
                target_artifact.e_c_digest if target_artifact is not None else None
            ),
            "e_c_mapping_digest": (
                target_artifact.e_c_mapping_digest
                if target_artifact is not None
                else None
            ),
            "target_plugin_digest": (
                self.request.c.content_digest if self.request.c is not None else None
            ),
            "target_artifact_digest": (
                target_artifact.content_digest if target_artifact is not None else None
            ),
            # RISC-V stores the typed target-consumption evidence directly in
            # ``boundary_consumption`` (unlike CUDA, whose Layer-B diagnostic
            # wraps it under ``target_realization``).  Preserve that evidence
            # at the public record key instead of emitting a misleading null.
            "target_realization": dict(self.boundary_consumption),
            "riscv": self.artifact.to_record(),
        }


def _function_symbol(intent_mlir: str) -> str:
    match = re.search(r"func\.func\s+@([A-Za-z_][A-Za-z0-9_.$]*)", intent_mlir)
    if match is None:
        raise ValueError("Intent MLIR has no func.func symbol")
    return match.group(1)


def _module_i64_attr(mlir: str, name: str, *, default: int = 0) -> int:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*([0-9]+)\s*:\s*i64\b",
        mlir,
    )
    return int(match.group(1)) if match is not None else default


def _linalg_op_count(mlir: str) -> int:
    return len(re.findall(r"\blinalg\.[A-Za-z0-9_]+", mlir))


@dataclass(frozen=True)
class _RiscvStageArtifacts:
    """Real compiler outputs for the independently declared target stages."""

    fusion_mlir: str
    memory_layout_mlir: str
    parallel_tiling_mlir: str
    reduction_tiling_mlir: str
    lowered_mlir: str
    llvm_ir: str
    optimized_llvm_ir: str
    assembly: str


def _tile_linalg_stage(
    source: str,
    *,
    parallel_tile: int,
    reduction_tile: int,
    fuse_producers: bool,
) -> str:
    """Apply one real iterator-kind tiling stage to structured MLIR."""

    if parallel_tile == 0 and reduction_tile == 0:
        return source
    return _run_text(
        [
            INTENT_OPT,
            (
                "-tile-linalg-by-iterator-kind="
                f"parallel-tile={parallel_tile} "
                f"reduction-tile={reduction_tile} "
                f"fuse-producers={'true' if fuse_producers else 'false'}"
            ),
            "-canonicalize",
            "-cse",
        ],
        stdin=source,
    )


def _lower_to_llvm_ir(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
) -> tuple[_RiscvStageArtifacts, RiscvTransformationRecord]:
    source = state.base.intent_mlir
    schedule = candidate.schedule
    has_ragged = _has_ragged_ops(source)
    has_dynamic_ragged_member = ragged_requires_scalar_schedule(source)
    if has_dynamic_ragged_member:
        if candidate.transformation != RiscvTransformationState(False):
            raise ValueError(
                "RISC-V Ragged lowering uses the shared carrier-tensor path and "
                "does not admit unrelated structural transformation states"
            )
        if not ragged_schedule_supported(schedule):
            raise ValueError(
                "RISC-V Ragged lowering requires the scalar untiled native "
                "schedule; vectorized or tiled schedules are not ABI-closed"
            )
    if candidate.transformation.composed_indexed_gather:
        if not schedule.vectorize or schedule.scalable_vectorization:
            raise ValueError(
                "composed indexed gather requires a fixed-width vector schedule"
            )
        source = _run_text(
            [
                INTENT_OPT,
                (
                    "-lower-gather-chain-to-loops="
                    "require-lowering=true "
                    f"vector-width={schedule.vector_lanes}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=source,
        )
    if (
        candidate.transformation.explicit_vector_reduction
        and not schedule.scalable_vectorization
    ):
        raise ValueError(
            "explicit RVV vector lowering requires a scalable-vector schedule"
        )
    linalg = _run_text(
        [
            INTENT_OPT,
            (
                "-convert-intent-to-linalg="
                "reassociate-float-reductions="
                f"{'true' if candidate.transformation.reassociate_reductions else 'false'}"
            ),
            "-expand-rsqrt-for-libm",
        ],
        stdin=source,
    )
    if has_ragged and _has_ragged_ops(linalg):
        raise RuntimeError(
            "RISC-V lowering left a high-level Ragged operation after the "
            "shared Intent-to-Linalg conversion"
        )
    if candidate.transformation.fused_softmax:
        fused_structured = _run_text(
            [
                INTENT_OPT,
                (
                    "-fuse-linalg-softmax="
                    f"vector-width={schedule.vector_lanes} scalable-vector="
                    f"{'true' if schedule.scalable_vectorization else 'false'} "
                    "approximate-exp="
                    f"{'true' if candidate.transformation.approximate_exp else 'false'} "
                    "require-fusion=true"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=linalg,
        )
    elif candidate.transformation.fuse_producers:
        fused_structured = _run_text(
            [MLIR_OPT, "--linalg-fuse-elementwise-ops", "--canonicalize"],
            stdin=linalg,
        )
    else:
        fused_structured = linalg
    fusion_changed_ir = bool(
        candidate.transformation.fuse_producers
        and _linalg_op_count(fused_structured) < _linalg_op_count(linalg)
    )
    if candidate.transformation.fuse_sibling_outputs:
        before_sibling_fusion = fused_structured
        fused_structured = _run_text(
            [
                INTENT_OPT,
                "-fuse-linalg-sibling-outputs=require-fusion=true",
                "-canonicalize",
                "-cse",
            ],
            stdin=fused_structured,
        )
        fusion_changed_ir = fusion_changed_ir or (
            _linalg_op_count(fused_structured)
            < _linalg_op_count(before_sibling_fusion)
        )
    pre_schedule_structured = fused_structured
    if candidate.transformation.mixed_precision_dot_reduction:
        pre_schedule_structured = _run_text(
            [
                INTENT_OPT,
                (
                    "-vectorize-linalg-mixed-precision-dot-reduction="
                    f"vector-width={schedule.vector_lanes} "
                    "scalable-vector="
                    f"{'true' if schedule.scalable_vectorization else 'false'} "
                    "require-vectorization=true"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=pre_schedule_structured,
        )
    if candidate.transformation.joint_vector_reduction:
        pre_schedule_structured = _run_text(
            [
                MLIR_OPT,
                "--linalg-fold-unit-extent-dims",
                "--canonicalize",
                "--cse",
            ],
            stdin=pre_schedule_structured,
        )
    if candidate.transformation.explicit_vector_reduction:
        joint_options = (
            " joint-reductions=true require-joint-reductions=true"
            if candidate.transformation.joint_vector_reduction
            else ""
        )
        pre_schedule_structured = _run_text(
            [
                INTENT_OPT,
                (
                    "-vectorize-linalg-row-reduction="
                    f"vector-width={schedule.vector_lanes} scalable-vector=true "
                    f"require-vectorization=true{joint_options}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=pre_schedule_structured,
        )
    if candidate.transformation.joint_vector_reduction:
        before_second_fusion = pre_schedule_structured
        pre_schedule_structured = _run_text(
            [MLIR_OPT, "--linalg-fuse-elementwise-ops", "--canonicalize", "--cse"],
            stdin=pre_schedule_structured,
        )
        fusion_changed_ir = fusion_changed_ir or (
            _linalg_op_count(pre_schedule_structured)
            < _linalg_op_count(before_second_fusion)
        )
    fusion_mlir = pre_schedule_structured

    # Parallel and reduction records refer to different iterator-kind
    # applications.  When both are requested, retain both as real compiler
    # stages: the reduction pass consumes the MLIR produced by the preceding
    # parallel-only pass.  TileLinalgByIteratorKind preserves the earlier
    # schedule attributes and cumulative counters across this split.
    if schedule.parallel_tile:
        parallel_tiling_mlir = _tile_linalg_stage(
            pre_schedule_structured,
            parallel_tile=schedule.parallel_tile,
            reduction_tile=0,
            fuse_producers=candidate.transformation.fuse_producers,
        )
        structured = parallel_tiling_mlir
    else:
        parallel_tiling_mlir = pre_schedule_structured
        structured = pre_schedule_structured
    if schedule.reduction_tile:
        reduction_tiling_mlir = _tile_linalg_stage(
            structured,
            parallel_tile=0,
            reduction_tile=schedule.reduction_tile,
            # Producer fusion belongs to the first structural stage.  Repeating
            # it in the reduction stage could duplicate producer slices.
            fuse_producers=(
                candidate.transformation.fuse_producers
                and not schedule.parallel_tile
            ),
        )
        structured = reduction_tiling_mlir
    else:
        reduction_tiling_mlir = pre_schedule_structured
    record = RiscvTransformationRecord(
        linalg_ops_before=_linalg_op_count(linalg),
        linalg_ops_after=_linalg_op_count(structured),
        parallel_tile_requested=schedule.parallel_tile,
        reduction_tile_requested=schedule.reduction_tile,
        parallel_tile_applied=_module_i64_attr(
            structured, "intent.schedule.parallel_tile"
        ),
        reduction_tile_applied=_module_i64_attr(
            structured, "intent.schedule.reduction_tile"
        ),
        linalg_tiled_ops=_module_i64_attr(
            structured, "intent.schedule.linalg_tiled_op_count"
        ),
        generated_tile_loops=_module_i64_attr(
            structured, "intent.schedule.generated_loop_count"
        ),
        fused_producers=max(
            _module_i64_attr(structured, "intent.schedule.fused_producer_count"),
            int(fusion_changed_ir),
        ),
        fusion_outer_parallel_only_ops=_module_i64_attr(
            structured, "intent.schedule.fusion_outer_parallel_only_count"
        ),
        fusion_multi_use_rejections=_module_i64_attr(
            structured, "intent.schedule.fusion_multi_use_rejection_count"
        ),
        reassociated_float_combines=len(
            re.findall(r"fastmath<[^>]*\breassoc\b[^>]*>", linalg)
        ),
        explicit_vector_reductions=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_reduction_count"
        ),
        explicit_vector_reduction_width=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_reduction_width"
        ),
        explicit_vector_reduction_loops=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_reduction_loop_count"
        ),
        explicit_vector_reduction_scalable=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_reduction_scalable_count"
        ),
        explicit_vector_max_reductions=_module_i64_attr(
            structured, "intent.rvv.explicit_vector_max_reduction_count"
        ),
        joint_vector_reduction_groups=_module_i64_attr(
            structured, "intent.rvv.joint_vector_reduction_group_count"
        ),
        joint_vector_reductions=_module_i64_attr(
            structured, "intent.rvv.joint_vector_reduction_count"
        ),
        joint_vector_reduction_root_loads=_module_i64_attr(
            structured, "intent.rvv.joint_vector_reduction_root_load_count"
        ),
        joint_raw_moment_variances=_module_i64_attr(
            structured, "intent.rvv.joint_raw_moment_variance_count"
        ),
        fused_softmax_count=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_count"
        ),
        fused_softmax_input_traversals=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_input_traversal_count"
        ),
        fused_softmax_full_row_traversals=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_full_row_traversal_count"
        ),
        fused_softmax_removed_materializations=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_removed_materialization_count"
        ),
        fused_softmax_vector_reductions=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_vector_reduction_count"
        ),
        fused_softmax_scalar_reductions=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_scalar_reduction_count"
        ),
        fused_softmax_vector_width=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_vector_width"
        ),
        fused_softmax_scalable=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_scalable_count"
        ),
        fused_softmax_approximate_exp=_module_i64_attr(
            structured, "intent.rvv.fused_softmax_approximate_exp_count"
        ),
        mixed_dot_reductions=_module_i64_attr(
            structured, "intent.rvv.mixed_dot_reduction_count"
        ),
        mixed_dot_reduction_width=_module_i64_attr(
            structured, "intent.rvv.mixed_dot_reduction_width"
        ),
        mixed_dot_reduction_scalable=_module_i64_attr(
            structured, "intent.rvv.mixed_dot_reduction_scalable_count"
        ),
        sibling_output_fusions=_module_i64_attr(
            structured, "intent.rvv.sibling_output_fusion_count"
        ),
        sibling_output_eliminated_traversals=_module_i64_attr(
            structured, "intent.rvv.sibling_output_eliminated_traversals"
        ),
    )
    validate_riscv_schedule_realization(candidate, record)
    if (
        candidate.transformation.reassociate_reductions
        and record.reassociated_float_combines == 0
    ):
        raise RuntimeError(
            "requested reduction reassociation matched no explicit "
            "float_nondeterministic combine"
        )
    if candidate.transformation.fuse_producers and record.fused_producers == 0:
        raise RuntimeError(
            "requested RVV producer-consumer fusion produced no fused Linalg "
            "operation"
        )
    if candidate.transformation.fuse_sibling_outputs and (
        record.sibling_output_fusions != 1
        or record.sibling_output_eliminated_traversals != 1
    ):
        raise RuntimeError(
            "requested RVV sibling-output fusion produced no valid multi-output "
            "elementwise operation"
        )
    if (
        candidate.transformation.explicit_vector_reduction
        and record.explicit_vector_reductions == 0
    ):
        raise RuntimeError(
            "requested explicit RVV vector reduction matched no supported "
            "Linalg row reduction"
        )
    if candidate.transformation.joint_vector_reduction and (
        record.joint_vector_reduction_groups == 0
        or record.joint_vector_reductions < 2
        or record.joint_vector_reduction_root_loads
        != record.joint_vector_reduction_groups
    ):
        raise RuntimeError(
            "requested joint RVV vector reduction produced no valid "
            "same-root statistics group"
        )
    if candidate.transformation.fused_softmax and (
        record.fused_softmax_count != 1
        or record.fused_softmax_input_traversals != 2
        or record.fused_softmax_full_row_traversals != 3
        or record.fused_softmax_removed_materializations < 4
        or record.fused_softmax_vector_reductions
        != (2 if candidate.transformation.approximate_exp else 1)
        or record.fused_softmax_scalar_reductions
        != (0 if candidate.transformation.approximate_exp else 1)
        or record.fused_softmax_vector_width != schedule.vector_lanes
        or record.fused_softmax_scalable
        != int(schedule.scalable_vectorization)
        or record.fused_softmax_approximate_exp
        != int(candidate.transformation.approximate_exp)
    ):
        raise RuntimeError(
            "requested fused RVV Softmax produced no valid hybrid "
            "three-traversal vector state"
        )
    if candidate.transformation.mixed_precision_dot_reduction and (
        record.mixed_dot_reductions != 1
        or record.mixed_dot_reduction_width != schedule.vector_lanes
        or record.mixed_dot_reduction_scalable
        != int(schedule.scalable_vectorization)
    ):
        raise RuntimeError(
            "requested mixed-precision RVV dot produced no widening vector reduction"
        )
    parallel_runtime = schedule.thread_count > 1
    buffer_front = [
        # Intent functions expose logical tensor values.  This backend owns the
        # target ABI and the benchmark/runtime binds those values to dense
        # row-major buffers, so use the same identity-layout boundary
        # conversion as the CPU and CUDA pipelines.  Leaving the default
        # fully-dynamic strided layout prevents the downstream LLVM loop
        # vectorizer from proving unit-stride RVV accesses even though the
        # concrete buffers are contiguous.
        (
            "--one-shot-bufferize=bufferize-function-boundaries "
            "function-boundary-type-conversion=identity-layout-map "
            "allow-return-allocs-from-loops=true"
        ),
        "--buffer-results-to-out-params",
        "--canonicalize",
        "--cse",
    ]
    bufferized = _run_text([MLIR_OPT, *buffer_front], stdin=structured)
    if candidate.transformation.paired_rotation:
        specialized_buffer = _run_text(
            [
                INTENT_OPT,
                (
                    "-lower-paired-rotation-to-loops="
                    "require-lowering=true head-angle-reuse="
                    f"{'true' if candidate.transformation.paired_rotation_head_angle_reuse else 'false'} "
                    "vector-width="
                    f"{candidate.schedule.vector_lanes if candidate.schedule.vectorize else 1} "
                    "scalable-vector="
                    f"{'true' if candidate.schedule.scalable_vectorization else 'false'}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=bufferized,
        )
    elif candidate.transformation.streaming_attention:
        specialized_buffer = _run_text(
            [
                INTENT_OPT,
                (
                    "-lower-flash-attention-to-loops="
                    "require-lowering=true vector-width="
                    f"{candidate.schedule.vector_lanes}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=bufferized,
        )
    elif candidate.transformation.rowmask_zero_fill:
        rowmask_vector_width = 4 if schedule.vectorize else 1
        specialized_buffer = _run_text(
            [
                INTENT_OPT,
                (
                    "-lower-row-mask-to-loops="
                    "zero-fill=true require-lowering=true "
                    f"vector-width={rowmask_vector_width}"
                ),
                "-canonicalize",
                "-cse",
            ],
            stdin=bufferized,
        )
    else:
        specialized_buffer = bufferized
    memref = _run_text(
        [
            MLIR_OPT,
            (
                "--convert-linalg-to-parallel-loops"
                if parallel_runtime
                else "--convert-linalg-to-loops"
            ),
            "--canonicalize",
            "--cse",
        ],
        stdin=specialized_buffer,
    )
    record = replace(
        record,
        linalg_ops_after=_linalg_op_count(specialized_buffer),
        rowmask_lowerings=_module_i64_attr(
            specialized_buffer, "intent.rvv.rowmask_lowering_count"
        ),
        rowmask_zero_fills=_module_i64_attr(
            specialized_buffer, "intent.rvv.rowmask_zero_fill_count"
        ),
        rowmask_mask_hoists=_module_i64_attr(
            specialized_buffer, "intent.rvv.rowmask_mask_hoist_count"
        ),
        rowmask_vector_width=_module_i64_attr(
            specialized_buffer, "intent.rvv.rowmask_vector_width"
        ),
        composed_gather_lowerings=_module_i64_attr(
            specialized_buffer, "intent.rvv.composed_gather_count"
        ),
        composed_gather_chain_depth=_module_i64_attr(
            specialized_buffer, "intent.rvv.composed_gather_chain_depth"
        ),
        composed_gather_vector_width=_module_i64_attr(
            specialized_buffer, "intent.rvv.composed_gather_vector_width"
        ),
        paired_rotation_lowerings=_module_i64_attr(
            specialized_buffer, "intent.rvv.paired_rotation_lowering_count"
        ),
        paired_rotation_head_angle_reuses=_module_i64_attr(
            specialized_buffer,
            "intent.rvv.paired_rotation_head_angle_reuse_count",
        ),
        paired_rotation_vector_width=_module_i64_attr(
            specialized_buffer, "intent.rvv.paired_rotation_vector_width"
        ),
        paired_rotation_scalable=_module_i64_attr(
            specialized_buffer, "intent.rvv.paired_rotation_scalable_count"
        ),
        streaming_attention_lowerings=_module_i64_attr(
            specialized_buffer, "intent.rvv.flash_attention_streaming_count"
        ),
        streaming_attention_vector_width=_module_i64_attr(
            specialized_buffer, "intent.rvv.flash_attention_vector_width"
        ),
        streaming_attention_score_scratch=_module_i64_attr(
            specialized_buffer, "intent.rvv.flash_attention_score_scratch_count"
        ),
        streaming_attention_removed_materializations=_module_i64_attr(
            specialized_buffer,
            "intent.rvv.flash_attention_removed_materialization_count",
        ),
    )
    validate_riscv_schedule_realization(candidate, record)
    if candidate.transformation.paired_rotation and (
        record.paired_rotation_lowerings != 1
        or record.paired_rotation_head_angle_reuses
        != int(candidate.transformation.paired_rotation_head_angle_reuse)
        or record.paired_rotation_vector_width
        != (candidate.schedule.vector_lanes if candidate.schedule.vectorize else 1)
        or record.paired_rotation_scalable
        != int(candidate.schedule.scalable_vectorization)
    ):
        raise RuntimeError(
            "requested RVV paired rotation produced no exact direct-loop state"
        )
    if candidate.transformation.streaming_attention and (
        record.streaming_attention_lowerings != 1
        or record.streaming_attention_vector_width != candidate.schedule.vector_lanes
        or record.streaming_attention_score_scratch != 1
        or record.streaming_attention_removed_materializations < 20
    ):
        raise RuntimeError(
            "requested RVV streaming attention produced no exact direct-loop state"
        )
    if candidate.transformation.rowmask_zero_fill and (
        record.rowmask_lowerings != 1
        or record.rowmask_zero_fills != 1
        or record.rowmask_mask_hoists != 1
        or record.rowmask_vector_width != (4 if schedule.vectorize else 1)
    ):
        raise RuntimeError(
            "requested RVV row-mask zero-fill produced no valid hoisted "
            "typed-broadcast loop"
        )
    if candidate.transformation.composed_indexed_gather and (
        record.composed_gather_lowerings != 1
        or record.composed_gather_chain_depth != 2
        or record.composed_gather_vector_width != schedule.vector_lanes
    ):
        raise RuntimeError(
            "requested composed RVV gather produced no valid direct indexed "
            "vector state"
        )
    forwarded = _run_text(
        [
            INTENT_OPT,
            "-forward-buffer-results-to-out-params",
            "-lower-disjoint-memref-copy-to-loops",
        ],
        stdin=memref,
    )
    cleaned = _run_text(
        [MLIR_OPT, "--buffer-deallocation-pipeline", "--canonicalize", "--cse"],
        stdin=forwarded,
    )
    parallelized = (
        _run_text(
            [MLIR_OPT, "--convert-scf-to-openmp", "--canonicalize", "--cse"],
            stdin=cleaned,
        )
        if parallel_runtime
        else cleaned
    )
    record = replace(
        record,
        openmp_parallel_ops=len(re.findall(r"\bomp\.parallel\b", parallelized)),
    )
    terminal_pipeline = [
        MLIR_OPT,
        "--llvm-request-c-wrappers",
    ]
    if parallel_runtime:
        terminal_pipeline.append("--convert-openmp-to-llvm")
    terminal_pipeline.extend(
        [
            "--expand-strided-metadata",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--convert-math-to-libm",
            *(
                ["--convert-vector-to-llvm"]
                if (
                    candidate.transformation.explicit_vector_reduction
                    or candidate.transformation.fused_softmax
                    or candidate.transformation.mixed_precision_dot_reduction
                    or (
                        candidate.transformation.rowmask_zero_fill
                        and schedule.vectorize
                    )
                    or candidate.transformation.composed_indexed_gather
                    or (
                        candidate.transformation.paired_rotation
                        and schedule.vectorize
                    )
                    or candidate.transformation.streaming_attention
                )
                else []
            ),
            "--convert-arith-to-llvm",
            "--convert-index-to-llvm",
            "--finalize-memref-to-llvm",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
        ]
    )
    lowered = _run_text(
        terminal_pipeline,
        stdin=parallelized,
    )
    if re.search(
        r"^\s*(?:%[^=]+\s*=\s*)?(?:intent|linalg)\.",
        lowered,
        flags=re.MULTILINE,
    ):
        raise RuntimeError("RVV lowering left high-level Intent/Linalg ops")
    if has_ragged and (
        any(f"intent.{name}" in lowered for name in RAGGED_OPS)
        or "tuple<" in lowered
    ):
        raise RuntimeError(
            "RISC-V Ragged lowering left a tuple descriptor in the native ABI"
        )
    llvm_ir = _run_text(
        [MLIR_TRANSLATE, "--mlir-to-llvmir"],
        stdin=lowered,
    )
    return (
        _RiscvStageArtifacts(
            fusion_mlir=fusion_mlir,
            memory_layout_mlir=bufferized,
            parallel_tiling_mlir=parallel_tiling_mlir,
            reduction_tiling_mlir=reduction_tiling_mlir,
            lowered_mlir=lowered,
            llvm_ir=llvm_ir,
            optimized_llvm_ir="",
            assembly="",
        ),
        record,
    )


def compile_riscv_object(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
    *,
    capabilities: tuple[str, ...] | None = None,
) -> RiscvCompiledObject:
    """Compile one internal RVV candidate using freshly derived authority.

    ``capabilities`` is retained for existing callers only.  It is never an
    authority input: when provided it must exactly equal the contract view
    derived again at this boundary.
    """

    if not isinstance(state, BackendSearchState):
        raise TypeError("RVV compiler requires BackendSearchState")
    if not isinstance(candidate, RiscvLoweringCandidate):
        raise TypeError("RVV compiler requires RiscvLoweringCandidate")
    from intentir.backend.riscv.contract import derive_riscv_contract_view

    contract = derive_riscv_contract_view(state)
    derived_capabilities = contract.capabilities
    if capabilities is not None:
        if not isinstance(capabilities, tuple) or any(
            not isinstance(item, str) for item in capabilities
        ):
            raise TypeError("RVV compatibility capabilities must be tuple[str, ...]")
        if capabilities != derived_capabilities:
            raise ValueError(
                "RVV caller capabilities differ from the internally derived contract"
            )
    missing = set(candidate.transformation.required_b_capabilities) - set(
        derived_capabilities
    )
    if missing:
        raise ValueError(
            "RVV candidate lacks required B capabilities: "
            + ", ".join(sorted(missing))
        )
    # The low-level function remains available to the bounded search driver,
    # but it still admits only a candidate reconstructed by the current
    # J/target plan.  This closes a direct-call path around target Layer C.
    from intentir.backend.riscv.search import plan_riscv_search

    plan = plan_riscv_search(state, enable_b_transformations=True)
    if sum(item == candidate for item in plan.universe) != 1:
        raise ValueError(
            "RVV compiler candidate is absent or ambiguous in the rebuilt search plan"
        )
    stage_artifacts, transformation_record = _lower_to_llvm_ir(state, candidate)
    lowered = stage_artifacts.lowered_mlir
    raw_llvm_ir = stage_artifacts.llvm_ir
    # Keep the production entrypoint fail-closed even when an alternate caller
    # supplies a pre-lowered module.  A requested structural/numeric state is
    # not a distinct candidate unless the compiler record proves it happened.
    if (
        candidate.transformation.reassociate_reductions
        and transformation_record.reassociated_float_combines == 0
    ):
        raise RuntimeError(
            "requested reduction reassociation matched no explicit "
            "float_nondeterministic combine"
        )
    if (
        candidate.transformation.fuse_producers
        and transformation_record.fused_producers == 0
    ):
        raise RuntimeError(
            "requested RVV producer-consumer fusion produced no fused Linalg "
            "operation"
        )
    if candidate.transformation.fuse_sibling_outputs and (
        transformation_record.sibling_output_fusions != 1
        or transformation_record.sibling_output_eliminated_traversals != 1
    ):
        raise RuntimeError(
            "requested RVV sibling-output fusion produced no multi-output "
            "elementwise state"
        )
    if (
        candidate.transformation.explicit_vector_reduction
        and transformation_record.explicit_vector_reductions == 0
    ):
        raise RuntimeError(
            "requested explicit RVV vector reduction produced no vectorized "
            "row reduction"
        )
    if candidate.transformation.joint_vector_reduction and (
        transformation_record.joint_vector_reduction_groups == 0
        or transformation_record.joint_vector_reductions < 2
        or transformation_record.joint_vector_reduction_root_loads
        != transformation_record.joint_vector_reduction_groups
    ):
        raise RuntimeError(
            "requested joint RVV vector reduction produced no valid "
            "same-root statistics group"
        )
    if candidate.transformation.fused_softmax and (
        transformation_record.fused_softmax_count != 1
        or transformation_record.fused_softmax_input_traversals != 2
        or transformation_record.fused_softmax_full_row_traversals != 3
        or transformation_record.fused_softmax_removed_materializations < 4
        or transformation_record.fused_softmax_vector_reductions
        != (2 if candidate.transformation.approximate_exp else 1)
        or transformation_record.fused_softmax_scalar_reductions
        != (0 if candidate.transformation.approximate_exp else 1)
        or transformation_record.fused_softmax_vector_width
        != candidate.schedule.vector_lanes
        or transformation_record.fused_softmax_scalable
        != int(candidate.schedule.scalable_vectorization)
        or transformation_record.fused_softmax_approximate_exp
        != int(candidate.transformation.approximate_exp)
    ):
        raise RuntimeError(
            "requested fused RVV Softmax produced no valid hybrid "
            "three-traversal vector state"
        )
    if candidate.transformation.mixed_precision_dot_reduction and (
        transformation_record.mixed_dot_reductions != 1
        or transformation_record.mixed_dot_reduction_width
        != candidate.schedule.vector_lanes
        or transformation_record.mixed_dot_reduction_scalable
        != int(candidate.schedule.scalable_vectorization)
    ):
        raise RuntimeError(
            "requested mixed-precision RVV dot produced no widening vector reduction"
        )
    if candidate.transformation.rowmask_zero_fill and (
        transformation_record.rowmask_lowerings != 1
        or transformation_record.rowmask_zero_fills != 1
        or transformation_record.rowmask_mask_hoists != 1
        or transformation_record.rowmask_vector_width
        != (4 if candidate.schedule.vectorize else 1)
    ):
        raise RuntimeError(
            "requested RVV row-mask zero-fill produced no valid hoisted "
            "typed-broadcast loop"
        )
    if candidate.transformation.composed_indexed_gather and (
        transformation_record.composed_gather_lowerings != 1
        or transformation_record.composed_gather_chain_depth != 2
        or transformation_record.composed_gather_vector_width
        != candidate.schedule.vector_lanes
    ):
        raise RuntimeError(
            "requested composed RVV gather produced no direct indexed vector "
            "state"
        )
    if candidate.transformation.paired_rotation and (
        transformation_record.paired_rotation_lowerings != 1
        or transformation_record.paired_rotation_head_angle_reuses
        != int(candidate.transformation.paired_rotation_head_angle_reuse)
        or transformation_record.paired_rotation_vector_width
        != (candidate.schedule.vector_lanes if candidate.schedule.vectorize else 1)
        or transformation_record.paired_rotation_scalable
        != int(candidate.schedule.scalable_vectorization)
    ):
        raise RuntimeError(
            "requested RVV paired rotation produced no exact direct-loop state"
        )
    if candidate.transformation.streaming_attention and (
        transformation_record.streaming_attention_lowerings != 1
        or transformation_record.streaming_attention_vector_width
        != candidate.schedule.vector_lanes
        or transformation_record.streaming_attention_score_scratch != 1
        or transformation_record.streaming_attention_removed_materializations < 20
    ):
        raise RuntimeError(
            "requested RVV streaming attention produced no exact direct-loop state"
        )
    schedule = candidate.schedule
    vector_shape_flags = (
        [
            "--riscv-v-vector-bits-min=-1",
            "--riscv-v-vector-bits-max=0",
            "--scalable-vectorization=preferred",
        ]
        if schedule.scalable_vectorization
        else [
            f"--riscv-v-vector-bits-min={schedule.vector_bits}",
            f"--riscv-v-vector-bits-max={schedule.vector_bits}",
            "--scalable-vectorization=off",
        ]
    )
    opt_command = [
        OPT,
        f"-O{schedule.opt_level}",
        "-S",
        f"-mtriple={TRIPLE}",
        f"-mattr={MATTR}",
        *vector_shape_flags,
        f"-vectorize-loops={'true' if schedule.vectorize else 'false'}",
        f"-vectorize-slp={'true' if schedule.vectorize else 'false'}",
        "-o",
        "-",
    ]
    optimized = _run_text(opt_command, stdin=raw_llvm_ir)
    llc_common = [
        LLC,
        f"-mtriple={TRIPLE}",
        f"-mattr={MATTR}",
        *vector_shape_flags,
        f"-O{schedule.opt_level}",
        "-relocation-model=pic",
    ]
    assembly = _run_text([*llc_common, "-filetype=asm", "-o", "-"], stdin=optimized)
    object_bytes = _run_bytes(
        [*llc_common, "-filetype=obj", "-o", "-"],
        stdin=optimized,
    )
    rvv_instruction_count = len(
        re.findall(
            r"^\s+v(?:set|l|s|f|m|n|w|r)[a-z0-9_.]*\s",
            assembly,
            flags=re.MULTILINE,
        )
    )
    rvv_reduction_instruction_count = len(
        re.findall(
            r"^\s+vfred[a-z0-9_.]*\s",
            assembly,
            flags=re.MULTILINE,
        )
    )
    if (
        (
            candidate.transformation.explicit_vector_reduction
            and rvv_reduction_instruction_count == 0
        )
        or (
            candidate.transformation.fused_softmax
            and rvv_reduction_instruction_count < 1
        )
        or (
            candidate.transformation.mixed_precision_dot_reduction
            and rvv_reduction_instruction_count < 1
        )
        or (
            candidate.transformation.paired_rotation
            and schedule.vectorize
            and rvv_instruction_count == 0
        )
        or (
            candidate.transformation.streaming_attention
            and rvv_reduction_instruction_count < 1
        )
    ):
        raise RuntimeError(
            "requested explicit RVV vector state produced insufficient RVV "
            "target instructions"
        )
    scalable_vector_ir_count = len(re.findall(r"<vscale\s+x\s+", optimized))
    if schedule.scalable_vectorization and scalable_vector_ir_count == 0:
        raise RuntimeError(
            "requested scalable RVV vectorization produced no scalable-vector "
            "LLVM IR"
        )
    rvv_vsetvl_modes = tuple(
        sorted(
            {
                match.group(1).replace(" ", "")
                for match in re.finditer(
                    r"^\s+vset(?:i?vli|vl)\s+[^\n]*?\b(e(?:8|16|32|64)\s*,\s*m(?:f[248]|[1248]))\b",
                    assembly,
                    flags=re.MULTILINE,
                )
            }
        )
    )
    stage_artifacts = replace(
        stage_artifacts,
        optimized_llvm_ir=optimized,
        assembly=assembly,
    )
    return RiscvCompiledObject(
        candidate=candidate,
        symbol=_function_symbol(state.base.intent_mlir),
        object_bytes=object_bytes,
        rvv_instruction_count=rvv_instruction_count,
        rvv_reduction_instruction_count=rvv_reduction_instruction_count,
        scalable_vector_ir_count=scalable_vector_ir_count,
        rvv_vsetvl_modes=rvv_vsetvl_modes,
        transformation_record=transformation_record,
        fusion_mlir=stage_artifacts.fusion_mlir,
        memory_layout_mlir=stage_artifacts.memory_layout_mlir,
        parallel_tiling_mlir=stage_artifacts.parallel_tiling_mlir,
        reduction_tiling_mlir=stage_artifacts.reduction_tiling_mlir,
        lowered_mlir=stage_artifacts.lowered_mlir,
        llvm_ir=stage_artifacts.llvm_ir,
        optimized_llvm_ir=stage_artifacts.optimized_llvm_ir,
        assembly=stage_artifacts.assembly,
    )


def compile_riscv_target_plugin(
    state: BackendSearchState,
    plugin: TargetLayerCPlugin | TargetPluginArtifact,
) -> RiscvCompiledObject:
    """J-facing fail-closed boundary for one target Layer-C RVV artifact.

    Resolution regenerates the contract, target-specific search plan, and
    candidate universe.  Only the exact internal candidate matched by both the
    opaque key and canonical digest reaches the compiler; there is no generic
    or nearest-candidate fallback.
    """

    if not isinstance(state, BackendSearchState):
        raise TypeError("RVV plugin compilation requires BackendSearchState")
    if not isinstance(plugin, (TargetLayerCPlugin, TargetPluginArtifact)):
        raise TypeError(
            "RVV plugin compilation requires TargetLayerCPlugin or "
            "TargetPluginArtifact"
        )
    from intentir.backend.riscv.search import (
        resolve_riscv_target_plugin,
        riscv_layer_c_interface,
    )

    before = layer_c_content_digest(state.base.j.to_json())
    contract_view = derive_riscv_contract_view(state)
    target_interface = riscv_layer_c_interface(state.base.j)
    if isinstance(plugin, TargetPluginArtifact):
        artifact_payload = serialize_target_artifact(plugin)
        shared_e_c = plugin.e_c
    else:
        # A plugin-only request has no producer-side shared E_C.  The target
        # interface may still contain target-only declarations; those belong
        # to C_target and must not synthesize a Source-C value.
        artifact_payload = serialize_target_artifact(
            TargetPluginArtifact(
                e_c=LayerCInterface(),
                plugin=plugin,
                e_c_mapping=(),
            )
        )
        shared_e_c = LayerCInterface()
    target_artifact = deserialize_target_artifact(
        TargetBindingContext(
            j=state.base.j,
            target=state.target,
            interface=target_interface,
            capabilities=contract_view.capabilities,
            allowed_authorizations=riscv_allowed_authorizations(state),
            e_c=shared_e_c,
        ),
        artifact_payload,
    )
    target_plugin = target_artifact.plugin
    candidate = resolve_riscv_target_plugin(state, target_plugin)
    compiled = compile_riscv_object(state, candidate)
    after = layer_c_content_digest(state.base.j.to_json())
    if after != before or target_plugin.j_digest != before:
        raise RuntimeError("RVV plugin lowering mutated or detached from canonical J")
    return compiled


def _riscv_consumption_phase(record_id: str) -> str:
    """Map each typed contract record to the real RVV compiler phase."""

    if record_id == "lowering":
        return "riscv_lowering"
    field_name = record_id.split(":", 1)[0]
    phases = {
        "memory_layout": "riscv_memory_layout",
        "parallel_mapping": "riscv_parallel_mapping",
        "pipeline": "riscv_pipeline",
        "fusion": "riscv_fusion",
        "split": "riscv_reduction_split",
    }
    try:
        return phases[field_name]
    except KeyError as exc:
        raise RiscvJLoweringError(
            f"RISC-V has no consumption phase for record {record_id!r}"
        ) from exc


def _riscv_consumption_applications(
    contract: Any,
    artifact: RiscvCompiledObject,
) -> tuple[Any, ...]:
    """Create one independent application from actual RVV stage outputs."""

    artifacts = {
        "pipeline": ("riscv_lowered_mlir", artifact.lowered_mlir),
        "parallel_mapping": (
            "riscv_parallel_tiling_mlir",
            artifact.parallel_tiling_mlir,
        ),
        "split": ("riscv_reduction_tiling_mlir", artifact.reduction_tiling_mlir),
        "fusion": ("riscv_fusion_mlir", artifact.fusion_mlir),
        "memory_layout": (
            "riscv_memory_layout_mlir",
            artifact.memory_layout_mlir,
        ),
        "lowering": ("riscv_object", artifact.object_bytes),
    }
    applications: list[Any] = []
    for record_id in contract.declared_record_ids():
        field_name = "lowering" if record_id == "lowering" else record_id.split(":", 1)[0]
        try:
            artifact_kind, stage_artifact = artifacts[field_name]
        except KeyError as exc:
            raise RiscvJLoweringError(
                f"RISC-V has no independent artifact for target realization record "
                f"{record_id!r}"
            ) from exc
        if not stage_artifact:
            raise RiscvJLoweringError(
                f"RISC-V compiler did not produce the {field_name} stage artifact"
            )
        applications.append(
            contract.record_consumption_from_artifact(
                record_id,
                consumer="riscv.native",
                phase=_riscv_consumption_phase(record_id),
                artifact_kind=artifact_kind,
                artifact=stage_artifact,
            )
        )
    return tuple(applications)


def _deserialize_riscv_target_artifact(
    request: LoweringRequest,
    context: TargetBindingContext,
) -> TargetPluginArtifact:
    """Reparse the complete ``E_C``/``C_target`` handoff at RVV ingress."""

    if request.c is None:
        raise RiscvJLoweringError(
            "RISC-V target-artifact ingress requires a target Layer-C plugin"
        )
    # A plugin-only request is compatible only when the shared E_C is empty.
    # For a non-empty interface the producer must provide the explicit
    # TargetPluginArtifact; the consumer must not invent its lineage.
    if request.e_c is None:
        if context.e_c.points:
            raise RiscvJLoweringError(
                "RISC-V lowering requires an explicit TargetPluginArtifact for "
                "non-empty shared E_C"
            )
        e_c = context.e_c
        e_c_mapping = ()
    else:
        e_c = request.e_c
        e_c_mapping = request.e_c_mapping
    try:
        envelope = TargetPluginArtifact(
            e_c=e_c,
            plugin=request.c,
            e_c_mapping=e_c_mapping,
        )
        return deserialize_target_artifact(
            context,
            serialize_target_artifact(envelope),
            expected_e_c=e_c,
            expected_e_c_mapping=e_c_mapping,
        )
    except (TypeError, ValueError) as exc:
        raise RiscvJLoweringError(
            "RISC-V target artifact failed public ingress validation: " + str(exc)
        ) from exc


def lower_j_to_riscv(
    request: LoweringRequest,
    concrete_extents: Mapping[str, int],
) -> RiscvJLowering:
    """Lower J through a neutral M0 and one validated RVV target plugin.

    Source/Harness ``ImplementationConfig`` values are intentionally not
    accepted at this boundary.  The request carries only target Layer C; the
    temporary search state has no C priors, so the same J-facing contract is
    used by direct production lowering and by the native compiler.
    """

    if not isinstance(request, LoweringRequest):
        raise RiscvJLoweringError(
            "RISC-V lowering requires LoweringRequest"
        )
    if request.target.family != "riscv":
        raise RiscvJLoweringError("RISC-V backend requires a RISC-V target")
    if request.c is None:
        raise RiscvJLoweringError(
            "RISC-V lowering requires a target Layer-C plugin"
        )

    j_digest_before = layer_c_content_digest(request.j.to_json())
    try:
        materialized: MaterializedJ = materialize_j(
            request.j,
            concrete_extents,
            constexpr_bindings=dict(request.constexpr_bindings),
        )
    except (TypeError, ValueError) as exc:
        raise RiscvJLoweringError(str(exc)) from exc

    neutral_state = BackendSearchState(
        base=materialized,
        target=request.target,
        priors=(),
        ignored_slots=(),
    )
    try:
        from intentir.backend.riscv.search import riscv_layer_c_interface

        contract_view = derive_riscv_contract_view(neutral_state)
        target_interface = riscv_layer_c_interface(neutral_state.base.j)
        shared_e_c = (
            request.e_c if request.e_c is not None else LayerCInterface()
        )
        target_artifact = _deserialize_riscv_target_artifact(
            request,
            TargetBindingContext(
                j=neutral_state.base.j,
                target=neutral_state.target,
                interface=target_interface,
                capabilities=contract_view.capabilities,
                allowed_authorizations=riscv_allowed_authorizations(
                    neutral_state
                ),
                e_c=shared_e_c,
            ),
        )
        request = replace(
            request,
            c=target_artifact,
            e_c=None,
            e_c_mapping=(),
        )
        compiled = compile_riscv_target_plugin(neutral_state, target_artifact)
    except RiscvJLoweringError:
        raise
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RiscvJLoweringError(str(exc)) from exc

    j_digest_after = layer_c_content_digest(request.j.to_json())
    if j_digest_after != j_digest_before or request.c.j_digest != j_digest_before:
        raise RiscvJLoweringError(
            "RISC-V lowering mutated J or detached target Layer C from canonical J"
        )
    contract = request.c.contract
    if contract is None:
        raise RiscvJLoweringError(
            "RISC-V target plugin is missing its realization contract"
        )
    applications = _riscv_consumption_applications(
        contract,
        compiled,
    )
    try:
        boundary_consumption = contract.consumption_evidence(applications)
    except (TypeError, ValueError) as exc:
        raise RiscvJLoweringError(
            "RISC-V target Layer-C consumption is not exact: " + str(exc)
        ) from exc
    return RiscvJLowering(
        request=request,
        artifact=compiled,
        boundary_consumption=boundary_consumption,
        concrete_extents=materialized.concrete_extents,
        materializer_version=materialized.materializer_version,
        materialization_form=materialized.form,
    )


# Compatibility spelling for the initial target-plugin design draft.
compile_riscv_plugin_object = compile_riscv_target_plugin


__all__ = [
    "MATTR",
    "RVV_B_FUSED_SOFTMAX",
    "RVV_B_FUSION",
    "RiscvCompiledObject",
    "RiscvJLowering",
    "RiscvJLoweringError",
    "RiscvLoweringCandidate",
    "RiscvSchedule",
    "RiscvTransformationRecord",
    "RiscvTransformationState",
    "ragged_requires_scalar_schedule",
    "ragged_schedule_supported",
    "TRIPLE",
    "compile_riscv_plugin_object",
    "compile_riscv_object",
    "compile_riscv_target_plugin",
    "lower_j_to_riscv",
]
