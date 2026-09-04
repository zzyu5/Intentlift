"""Target-aware structural selection for the real CUDA lowering backend.

This module is deliberately independent of every verification pipeline.  Its
input is already-accepted Intent dialect MLIR; selection probes the registered
C++ conversion passes and chooses a specialized path only when one pass
consumes the complete graph into one or more compiler-emitted ``gpu.launch``
operations. Otherwise the generic Intent -> Linalg -> CUDA path remains the
mandatory fallback.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import intentir.backend.cuda.compile as backend_compile
from intentir.ir.intent_op_contract import RAGGED_OPS


_INTENT_OP_NAMES = (
    "arg_reduce",
    "atomic_scatter",
    "cast",
    "concat",
    "const",
    "contraction",
    "conv",
    "flip",
    "gather",
    "iota",
    "loop",
    "map",
    "map_binary",
    "map_unary",
    "masked_scatter",
    "masked_select",
    "nonzero",
    "pad",
    "polar",
    "pool",
    "rand",
    *sorted(RAGGED_OPS),
    "reduce",
    "scan",
    "scatter",
    "select",
    "sort",
    "tile",
    "transpose",
    "triu",
    "upsample",
    "yield",
)
_INTENT_OP_ALTERNATION = "|".join(
    re.escape(name)
    for name in sorted(_INTENT_OP_NAMES, key=lambda item: (-len(item), item))
)
_INTENT_OPERATION = re.compile(
    rf'(?<![@A-Za-z0-9_.$])"?(intent\.(?:{_INTENT_OP_ALTERNATION}))\b"?'
)
_GPU_LAUNCH = re.compile(
    r"(?<![A-Za-z0-9_.$])gpu\.launch(?![A-Za-z0-9_.$])"
)
_PORTABLE_BLOCK_CUDA_CHIPS = ("sm_90a", "sm_120a")


def residual_intent_ops(mlir: str) -> tuple[str, ...]:
    """Return executable Intent operations that remain in an MLIR module."""

    return tuple(sorted(set(_INTENT_OPERATION.findall(mlir))))


def gpu_launch_count(mlir: str) -> int:
    return len(_GPU_LAUNCH.findall(mlir))


@dataclass(frozen=True)
class LoweringTarget:
    """Backend target facts that may participate in variant selection."""

    backend: str = "cuda"
    chip: str = field(default_factory=lambda: backend_compile.CHIP)
    features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.backend != "cuda":
            raise ValueError(f"unsupported lowering backend {self.backend!r}")
        if not isinstance(self.chip, str) or not self.chip.strip():
            raise ValueError("lowering target chip must be a non-empty string")
        normalized = frozenset(
            item.strip()
            for item in self.features
            if isinstance(item, str) and item.strip()
        )
        if len(normalized) != len(self.features):
            raise ValueError("lowering target features must be non-empty strings")
        object.__setattr__(self, "chip", self.chip.strip())
        object.__setattr__(self, "features", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "chip": self.chip,
            "features": sorted(self.features),
        }


@dataclass(frozen=True)
class PassDescriptor:
    fast_path_id: str
    pass_argument: str
    supported_chips: tuple[str, ...]
    description: str
    scf_to_cf: bool = False
    required_markers: tuple[str, ...] = ()
    expanded_only: bool = False
    schedule: Mapping[str, int | bool | str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeAttempt:
    fast_path_id: str
    pass_argument: str
    target_match: bool
    status: str
    direct_hit: bool
    residual_ops: tuple[str, ...] = ()
    gpu_launch_count: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "fast_path_id": self.fast_path_id,
            "pass_argument": self.pass_argument,
            "target_match": self.target_match,
            "status": self.status,
            "direct_hit": self.direct_hit,
            "residual_ops": list(self.residual_ops),
            "gpu_launch_count": self.gpu_launch_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VariantSelection:
    pipeline_kind: str
    target: LoweringTarget
    fast_path_id: str | None
    pass_argument: str | None
    scf_to_cf: bool
    attempts: tuple[ProbeAttempt, ...]
    generic_fallback: bool = True

    def __post_init__(self) -> None:
        if self.pipeline_kind not in {"generic", "specialized"}:
            raise ValueError(f"invalid pipeline kind {self.pipeline_kind!r}")
        if self.pipeline_kind == "specialized":
            if not self.fast_path_id or not self.pass_argument:
                raise ValueError("specialized selection requires an id and pass")
        elif self.fast_path_id is not None or self.pass_argument is not None:
            raise ValueError("generic selection cannot carry a specialized pass")

    @property
    def pipeline_id(self) -> str:
        """Return the exact compiler pipeline selected by this probe.

        Ragged lowering remains in the generic candidate class because it is a
        mandatory ABI consumer rather than an optional tunable specialization.
        Its descriptor-aware pass still has a distinct pipeline identity, and
        that identity must survive the selection boundary instead of being
        collapsed into the ordinary tensor-to-Linalg fallback.
        """

        if any(
            attempt.fast_path_id == RAGGED_FAST_PATH_ID
            for attempt in self.attempts
        ):
            return RAGGED_FAST_PATH_ID
        return (
            "cuda.pipeline.direct-specialized.v1"
            if self.pipeline_kind == "specialized"
            else "cuda.pipeline.linalg-materialized.v1"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pipeline_kind": self.pipeline_kind,
            "pipeline_id": self.pipeline_id,
            "target": self.target.to_dict(),
            "fast_path_id": self.fast_path_id,
            "pass_argument": self.pass_argument,
            "scf_to_cf": self.scf_to_cf,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "generic_fallback": self.generic_fallback,
        }


def _flash_structured_direct_fast_path_id(block_n: int) -> str:
    if block_n == 128:
        return "cuda.flash_forward.structured_scan.direct.v1"
    return f"cuda.flash_forward.structured_scan.direct.bn{block_n}.v1"


def _flash_certified_full_axis_fast_path_id(block_n: int) -> str:
    if block_n == 128:
        return "cuda.flash_forward.certified_full_axis.mma.v1"
    return f"cuda.flash_forward.certified_full_axis.bn{block_n}.mma.v1"


def _flash_structured_direct_descriptors(
    block_n: int,
) -> tuple[PassDescriptor, ...]:
    """Registered direct-lowering schedules for one materialized scan step.

    ``block_n`` is not inferred from a kernel name or from C: the C++ whole-graph
    matcher requires it to equal the ``intent.loop`` tile formed by deterministic
    J materialization.  A distinct path id keeps CF/Schedule-Only tied to the
    descriptor that actually matched instead of losing the selected block-N when
    several descriptors share one public id.
    """

    block_ms = (16, 32, 64) if block_n == 256 else (64, 128, 32, 16)
    stages = (1,) if block_n == 256 else (1, 2)
    fast_path_id = _flash_structured_direct_fast_path_id(block_n)
    return tuple(
        PassDescriptor(
            fast_path_id=fast_path_id,
            pass_argument=(
                "-convert-intent-flash-mma-to-gpu="
                f"num-stages={num_stages} block-m={block_m} "
                f"certified-full-axis-only=true block-n={block_n} "
                "materialize-tmp-roundtrip=true"
            ),
            supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
            description=(
                "deterministic direct lowering of a certified structured "
                f"FlashAttention scan with a {block_n}-key materialized step"
            ),
            scf_to_cf=True,
            schedule={
                "tile_rows": block_m,
                "tile_cols": block_n,
                "pipeline_stages": num_stages,
                "num_warps": block_m // 16,
                "direct_v_matrix_load": False,
                "direct_v_row_padding": 0,
                "materialize_tmp_roundtrip": True,
                "paired_b_matrix_loads": False,
                "vectorized_q_staging": False,
                "stream_p_fragments": False,
            },
        )
        for block_m in block_ms
        for num_stages in stages
    )


def _flash_certified_full_axis_descriptors(
    block_n: int,
) -> tuple[PassDescriptor, ...]:
    """Expanded compiler-lowering states for one admitted scan-step width."""

    if block_n == 256:
        # A 256-key step already consumes 64 KiB for one natural K/V stage at
        # D=64.  Keep the registered space within the real SM90/SM120 shared-
        # memory envelope: one stage and direct V matrix loads (no extra Vt).
        direct_options = (
            (True, False, 8),
            (True, True, 8),
            (True, False, 0),
            (True, True, 0),
        )
        block_ms = (16, 32, 64)
        stages = (1,)
        fragment_columns = (0, 64)
    else:
        direct_options = (
            (True, False, 8),
            (True, True, 8),
            (True, False, 0),
            (True, True, 0),
            (False, False, 0),
        )
        block_ms = (64, 128, 32, 16)
        stages = (1, 2)
        # A score fragment must be strictly narrower than block-N.
        fragment_columns = (0, 32) if block_n == 64 else (0, 64)

    fast_path_id = _flash_certified_full_axis_fast_path_id(block_n)
    return tuple(
        PassDescriptor(
            fast_path_id=fast_path_id,
            pass_argument=(
                "-convert-intent-flash-mma-to-gpu="
                f"num-stages={num_stages} block-m={block_m} "
                f"certified-full-axis-only=true block-n={block_n} "
                + ("direct-v-matrix-load=true " if direct_v else "")
                + (
                    f"direct-v-row-padding={direct_v_padding} "
                    if direct_v_padding
                    else ""
                )
                + ("paired-b-matrix-loads=true " if paired_b else "")
                + ("vectorized-q-staging=true " if vector_q else "")
                + ("stream-p-fragments=true " if stream_p else "")
                + (
                    f"score-fragment-columns={score_fragment_columns}"
                    if score_fragment_columns
                    else ""
                )
            ),
            supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
            description=(
                "certified batch/head-explicit FlashAttention scan with "
                f"a {block_n}-key step and register-resident online-softmax state"
            ),
            scf_to_cf=True,
            expanded_only=True,
            schedule={
                "tile_rows": block_m,
                "tile_cols": block_n,
                "pipeline_stages": num_stages,
                "num_warps": block_m // 16,
                "direct_v_matrix_load": direct_v,
                "direct_v_row_padding": direct_v_padding,
                "materialize_tmp_roundtrip": False,
                "paired_b_matrix_loads": paired_b,
                "vectorized_q_staging": vector_q,
                "stream_p_fragments": stream_p,
                "score_fragment_columns": score_fragment_columns,
            },
        )
        for direct_v, paired_b, direct_v_padding in direct_options
        for block_m in block_ms
        for vector_q in (False, True)
        for stream_p in (False, True)
        for score_fragment_columns in fragment_columns
        for num_stages in stages
        if score_fragment_columns == 0 or stream_p
    )


DEFAULT_PASSES: tuple[PassDescriptor, ...] = (
    PassDescriptor(
        fast_path_id="cuda.xformers.splitk-forward.flat-rank5.mma.v2",
        pass_argument=(
            "-convert-intent-loop-contraction-to-gpu="
            "num-stages=1 captured-flat-only=true block-m=16 block-n=32 "
            "fast-f32-lse-log=true"
        ),
        supported_chips=("sm_120a",),
        description="flat rank-5 xFormers split-K forward MMA",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.flat_rank4.mma.v1",
        pass_argument="-convert-intent-flash-mma-to-gpu=num-stages=2",
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="flat rank-4 FlashAttention forward MMA",
        scf_to_cf=True,
    ),
    *_flash_structured_direct_descriptors(128),
    *_flash_structured_direct_descriptors(64),
    *_flash_structured_direct_descriptors(256),
    *(
        PassDescriptor(
            fast_path_id="cuda.flash_forward.hopper_hybrid_pv.v1",
            pass_argument=(
                "-convert-intent-flash-mma-to-gpu="
                "num-stages=1 block-m=64 block-n=128 "
                "certified-full-axis-only=true hopper-warpgroup=true "
                "hopper-hybrid-pv=true"
            ),
            supported_chips=("sm_90a",),
            description=(
                "Hopper FlashAttention lowering with WGMMA QK and a "
                "register-resident mma.sync PV recurrence"
            ),
            scf_to_cf=True,
            expanded_only=True,
            schedule={
                "tile_rows": 64,
                "tile_cols": 128,
                "pipeline_stages": 1,
                "num_warps": 4,
                "direct_v_matrix_load": True,
                "direct_v_row_padding": 8,
                "materialize_tmp_roundtrip": False,
                "paired_b_matrix_loads": False,
                "vectorized_q_staging": False,
                "stream_p_fragments": True,
                "score_fragment_columns": 0,
                "hopper_warpgroup": True,
                "hopper_hybrid_pv": True,
            },
        ),
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.hopper_register_qk.v1",
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=64 block-n=128 "
            "certified-full-axis-only=true hopper-warpgroup=true "
            "hopper-hybrid-pv=true hopper-register-qk=true"
        ),
        supported_chips=("sm_90a",),
        description=(
            "Hopper FlashAttention with register-consumed NVVM WGMMA QK and "
            "register-resident mma.sync PV"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 64,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": True,
            "direct_v_row_padding": 8,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": False,
            "stream_p_fragments": True,
            "score_fragment_columns": 0,
            "hopper_warpgroup": True,
            "hopper_hybrid_pv": True,
            "hopper_register_qk": True,
        },
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.hopper_register_qk_async_kv.v1",
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=64 block-n=128 "
            "certified-full-axis-only=true hopper-warpgroup=true "
            "hopper-hybrid-pv=true hopper-register-qk=true "
            "hopper-async-kv=true"
        ),
        supported_chips=("sm_90a",),
        description=(
            "register-consumed Hopper WGMMA QK with 16-byte asynchronous K/V "
            "global-to-shared staging"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 64,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": True,
            "direct_v_row_padding": 8,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": False,
            "stream_p_fragments": True,
            "score_fragment_columns": 0,
            "hopper_warpgroup": True,
            "hopper_hybrid_pv": True,
            "hopper_register_qk": True,
            "hopper_async_kv": True,
        },
    ),
    PassDescriptor(
        fast_path_id=(
            "cuda.flash_forward.hopper_register_qk_async_kv_normalized.v1"
        ),
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=64 block-n=128 "
            "certified-full-axis-only=true hopper-warpgroup=true "
            "hopper-hybrid-pv=true hopper-register-qk=true "
            "hopper-async-kv=true normalized-softmax-carry=true"
        ),
        supported_chips=("sm_90a",),
        description=(
            "register-consumed Hopper WGMMA QK and asynchronous K/V staging "
            "with a normalized online-softmax carry and epilogue LSE rebuild"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 64,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": True,
            "direct_v_row_padding": 8,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": False,
            "stream_p_fragments": True,
            "score_fragment_columns": 0,
            "hopper_warpgroup": True,
            "hopper_hybrid_pv": True,
            "hopper_register_qk": True,
            "hopper_async_kv": True,
            "normalized_softmax_carry": True,
        },
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.hopper_register_qk_pv.v1",
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=64 block-n=128 "
            "certified-full-axis-only=true hopper-warpgroup=true "
            "hopper-hybrid-pv=true hopper-register-qk=true "
            "hopper-register-pv=true"
        ),
        supported_chips=("sm_90a",),
        description=(
            "Hopper FlashAttention with register-consumed NVVM WGMMA QK and "
            "WGMMA PV carrying O entirely in registers"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 64,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": False,
            "direct_v_row_padding": 0,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": False,
            "stream_p_fragments": True,
            "score_fragment_columns": 0,
            "hopper_warpgroup": True,
            "hopper_hybrid_pv": True,
            "hopper_register_qk": True,
            "hopper_register_pv": True,
        },
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.hopper_register_qk_pv_async_kv.v1",
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=64 block-n=128 "
            "certified-full-axis-only=true hopper-warpgroup=true "
            "hopper-hybrid-pv=true hopper-register-qk=true "
            "hopper-register-pv=true hopper-async-kv=true"
        ),
        supported_chips=("sm_90a",),
        description=(
            "Hopper FlashAttention with register-consumed WGMMA QK/PV and "
            "16-byte asynchronous K/V staging"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 64,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": False,
            "direct_v_row_padding": 0,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": False,
            "stream_p_fragments": True,
            "score_fragment_columns": 0,
            "hopper_warpgroup": True,
            "hopper_hybrid_pv": True,
            "hopper_register_qk": True,
            "hopper_register_pv": True,
            "hopper_async_kv": True,
        },
    ),
    PassDescriptor(
        fast_path_id=(
            "cuda.flash_forward.hopper_register_qk_pv_async_kv_normalized.v1"
        ),
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=64 block-n=128 "
            "certified-full-axis-only=true hopper-warpgroup=true "
            "hopper-hybrid-pv=true hopper-register-qk=true "
            "hopper-register-pv=true hopper-async-kv=true "
            "normalized-softmax-carry=true"
        ),
        supported_chips=("sm_90a",),
        description=(
            "register-consumed Hopper WGMMA QK/PV and asynchronous K/V with "
            "a normalized online-softmax carry"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 64,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": False,
            "direct_v_row_padding": 0,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": False,
            "stream_p_fragments": True,
            "score_fragment_columns": 0,
            "hopper_warpgroup": True,
            "hopper_hybrid_pv": True,
            "hopper_register_qk": True,
            "hopper_register_pv": True,
            "hopper_async_kv": True,
            "normalized_softmax_carry": True,
        },
    ),
    *(
        PassDescriptor(
            fast_path_id=(
                "cuda.flash_forward.hopper_warpgroup_async_kv_normalized.v1"
                if normalized
                else "cuda.flash_forward.hopper_warpgroup_async_kv.v1"
            ),
            pass_argument=(
                "-convert-intent-flash-mma-to-gpu="
                "num-stages=1 block-m=128 block-n=128 "
                "certified-full-axis-only=true hopper-warpgroup=true "
                "hopper-async-kv=true"
                + (" normalized-softmax-carry=true" if normalized else "")
            ),
            supported_chips=("sm_90a",),
            description=(
                "two-query-half Hopper warpgroup FlashAttention with one "
                "128-thread block per 128 queries and asynchronous K/V staging"
            ),
            scf_to_cf=True,
            expanded_only=True,
            schedule={
                "tile_rows": 128,
                "tile_cols": 128,
                "pipeline_stages": 1,
                "num_warps": 4,
                "direct_v_matrix_load": False,
                "direct_v_row_padding": 0,
                "materialize_tmp_roundtrip": False,
                "paired_b_matrix_loads": False,
                "vectorized_q_staging": False,
                "stream_p_fragments": False,
                "score_fragment_columns": 0,
                "hopper_warpgroup": True,
                "hopper_async_kv": True,
                "normalized_softmax_carry": normalized,
            },
        )
        for normalized in (False, True)
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.normalized_softmax_carry.mma.v1",
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=64 block-n=128 "
            "certified-full-axis-only=true direct-v-matrix-load=true "
            "direct-v-row-padding=8 vectorized-q-staging=true "
            "normalized-softmax-carry=true"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description=(
            "certified FlashAttention scan with a standard normalized online-"
            "softmax carry and one epilogue LSE reconstruction"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 64,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": True,
            "direct_v_row_padding": 8,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": True,
            "stream_p_fragments": False,
            "score_fragment_columns": 0,
            "normalized_softmax_carry": True,
        },
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.compact_query_warps.mma.v1",
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=128 block-n=128 "
            "certified-full-axis-only=true direct-v-matrix-load=true "
            "direct-v-row-padding=8 vectorized-q-staging=true "
            "normalized-softmax-carry=true compact-query-warp-mapping=true"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description=(
            "certified FlashAttention scan with two M16 query fragments per "
            "warp sharing one staged K/V tile"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 128,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": True,
            "direct_v_row_padding": 8,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": True,
            "stream_p_fragments": False,
            "score_fragment_columns": 0,
            "normalized_softmax_carry": True,
            "compact_query_warp_mapping": True,
        },
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.compact_query_warps_stream_p.mma.v1",
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=128 block-n=128 "
            "certified-full-axis-only=true direct-v-matrix-load=true "
            "direct-v-row-padding=8 vectorized-q-staging=true "
            "stream-p-fragments=true normalized-softmax-carry=true "
            "compact-query-warp-mapping=true"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description=(
            "compact four-warp FlashAttention with streamed probability "
            "fragments to shorten register lifetimes"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 128,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": True,
            "direct_v_row_padding": 8,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": True,
            "stream_p_fragments": True,
            "score_fragment_columns": 0,
            "normalized_softmax_carry": True,
            "compact_query_warp_mapping": True,
        },
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.compact_query_warps_fragment64.mma.v1",
        pass_argument=(
            "-convert-intent-flash-mma-to-gpu="
            "num-stages=1 block-m=128 block-n=128 "
            "certified-full-axis-only=true direct-v-matrix-load=true "
            "direct-v-row-padding=8 vectorized-q-staging=true "
            "stream-p-fragments=true score-fragment-columns=64 "
            "normalized-softmax-carry=true compact-query-warp-mapping=true"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description=(
            "compact four-warp FlashAttention with 64-column score fragments "
            "and bounded live-register state"
        ),
        scf_to_cf=True,
        expanded_only=True,
        schedule={
            "tile_rows": 128,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
            "direct_v_matrix_load": True,
            "direct_v_row_padding": 8,
            "materialize_tmp_roundtrip": False,
            "paired_b_matrix_loads": False,
            "vectorized_q_staging": True,
            "stream_p_fragments": True,
            "score_fragment_columns": 64,
            "normalized_softmax_carry": True,
            "compact_query_warp_mapping": True,
        },
    ),
    *_flash_certified_full_axis_descriptors(128),
    *_flash_certified_full_axis_descriptors(64),
    *_flash_certified_full_axis_descriptors(256),
    *(
        PassDescriptor(
            fast_path_id="cuda.flash_forward.hopper_warpgroup.v1",
            pass_argument=(
                "-convert-intent-flash-mma-to-gpu="
                f"num-stages=1 block-m={block_m} block-n=128 "
                "certified-full-axis-only=true hopper-warpgroup=true"
            ),
            supported_chips=("sm_90a",),
            description=(
                "Hopper warpgroup FlashAttention lowering using stock NVGPU "
                "WGMMA with independently stored 64x64 fragments"
            ),
            scf_to_cf=True,
            expanded_only=True,
            schedule={
                "tile_rows": block_m,
                "tile_cols": 128,
                "pipeline_stages": 1,
                "num_warps": 4,
                "direct_v_matrix_load": False,
                "direct_v_row_padding": 0,
                "materialize_tmp_roundtrip": False,
                "paired_b_matrix_loads": False,
                "vectorized_q_staging": False,
                "stream_p_fragments": False,
                "score_fragment_columns": 0,
                "hopper_warpgroup": True,
            },
        )
        for block_m in (128, 64)
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_backward.flat_rank4.mma.v1",
        pass_argument=(
            "-convert-intent-attention-backward-to-gpu="
            "flat-shared-d-pad=8 flat-shared-pds-pad=8 flat-dq-init=prelaunch"
        ),
        supported_chips=("sm_120a",),
        description="flat rank-4 FlashAttention backward MMA",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.loop_contraction.online_softmax.v1",
        pass_argument="-convert-intent-loop-contraction-to-gpu",
        supported_chips=("sm_120a",),
        description="general fused loop-contraction/online-softmax graph",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.splitk.reduce-varargs.v1",
        pass_argument="-convert-intent-splitk-reduce-varargs-to-gpu",
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="xFormers split-K varargs reduction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.splitk.reduce-forward.v1",
        pass_argument="-convert-intent-splitk-reduce-to-gpu",
        supported_chips=("sm_120a",),
        description="split-K attention forward reduction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.splitk.reduce-backward.v1",
        pass_argument="-convert-intent-splitk-reduce-backward-to-gpu",
        supported_chips=("sm_120a",),
        description="split-K attention backward reduction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        # Keep the original public ID for schedule/CLI compatibility.  The
        # implementation now accepts matching f16 or bf16 storage.
        fast_path_id="cuda.linear.f16_f32.bias_saved.mma_sync.v1",
        pass_argument=(
            "-convert-intent-contraction-to-gpu-mma=num-stages=5 "
            "block-m=16 block-n=32 block-k=32 smem-pad-k=8"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="tuned f16/bf16 tensor-core contraction with f32 accumulate",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.contraction.mma-warp-specialized.v1",
        pass_argument="-convert-intent-contraction-to-gpu-mma-ws",
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="warp-specialized tensor-core contraction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.contraction.mma-sync.v1",
        pass_argument="-convert-intent-contraction-to-gpu-mma",
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="tensor-core contraction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.contraction.tiled.v1",
        pass_argument="-convert-intent-contraction-to-gpu",
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="tiled-K contraction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.conv.implicit-gemm-mma.v1",
        pass_argument="-convert-intent-conv-to-gpu-mma",
        supported_chips=("sm_120a",),
        description="conv2d implicit-GEMM tensor-core lowering",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.conv.direct-window.v1",
        pass_argument="-convert-intent-conv-to-gpu",
        supported_chips=("sm_120a",),
        description="direct windowed conv1d/2d/3d lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.row_scan.sparsemax.sorted_prefix.v1",
        pass_argument=(
            "-convert-intent-row-reduction-to-gpu=block-threads=256 "
            "require-sorted-prefix=true"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="sorted-prefix Sparsemax row scan",
        scf_to_cf=True,
        required_markers=(
            "intent_sorted_descending = true",
            "intent_host_preprocessing_in_kernel_timing = false",
        ),
    ),
    PassDescriptor(
        fast_path_id="cuda.padded_row_reduction.masked_short_dot.v1",
        pass_argument=(
            "-convert-intent-padded-row-reduction-to-gpu="
            "block-threads=64 vectorize-full-features=false "
            "promote-rank0-scalars=true reduction-topology=serial-row"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="masked short row dot with authored zero padding",
    ),
    PassDescriptor(
        fast_path_id="cuda.elementwise.row.vectorized.v1",
        pass_argument=(
            "-convert-intent-elementwise-to-gpu=block-threads=64 "
            "tanh-approximation=stable-ex2"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="complete static row-elementwise graph",
    ),
    PassDescriptor(
        fast_path_id="cuda.elementwise.rowmask.zero-fill.v1",
        pass_argument=(
            "-convert-intent-elementwise-to-gpu=block-threads=64 "
            "row-vector-width=1 row-mask-zero-fill=true "
            "tanh-approximation=stable-ex2"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description=(
            "uniform row-mask branch that skips data loads for false rows"
        ),
    ),
    PassDescriptor(
        fast_path_id="cuda.row_reduction.normalization_loss.read_once.v1",
        pass_argument=(
            "-convert-intent-row-reduction-to-gpu=block-threads=1024 "
            "promote-rank0-scalars=true rows-per-block=1"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="row normalization/loss reduction graph",
    ),
    PassDescriptor(
        fast_path_id="cuda.algebra.exp_shift_product_cancellation.v1",
        pass_argument=(
            "-convert-intent-elementwise-to-gpu=block-threads=256 "
            "cancel-shifted-exp=true tanh-approximation=stable-ex2"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description=(
            "execution-domain exp(x-shift)*exp(shift) cancellation followed by "
            "the ordinary elementwise MLIR lowering"
        ),
    ),
    PassDescriptor(
        fast_path_id=(
            "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1"
        ),
        pass_argument=(
            "-convert-intent-row-reduction-to-gpu=block-threads=256 "
            "allow-execution-domain-sorted-prefix=true "
            "require-sorted-prefix=true"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description=(
            "graph-structural Sparsemax sorted-prefix execution-domain state"
        ),
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.gather-scatter.block.v1",
        pass_argument="-convert-intent-gather-to-gpu",
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description=(
            "data-dependent gather/scatter lowering, including deterministic "
            "eager evaluation of an admitted two-level gather"
        ),
    ),
    PassDescriptor(
        fast_path_id="cuda.gather-chain.composed-index.v1",
        pass_argument=(
            "-convert-intent-gather-to-gpu=compose-gather-chain=true"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description=(
            "compose two typed runtime gather index maps and eliminate the "
            "intermediate gathered row"
        ),
    ),
    PassDescriptor(
        fast_path_id="cuda.layout.block.v1",
        pass_argument="-convert-intent-layout-to-gpu",
        supported_chips=("sm_120a",),
        description="concat/pad/tile/flip/triu data-movement lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.compact.prefix-sum.v1",
        pass_argument="-convert-intent-compact-to-gpu",
        supported_chips=("sm_120a",),
        description="nonzero/masked-select/masked-scatter compaction lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.rng.philox.v1",
        pass_argument="-convert-intent-rng-to-gpu",
        supported_chips=("sm_120a",),
        description="counter-based Philox RNG lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.arg-reduce.block.v1",
        pass_argument="-convert-intent-arg-reduce-to-gpu",
        supported_chips=("sm_120a",),
        description="argmax/argmin pair-reduction lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.sort.bitonic.v1",
        pass_argument="-convert-intent-sort-to-gpu",
        supported_chips=("sm_120a",),
        description="block bitonic sort/top-k lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.pool.window.v1",
        pass_argument="-convert-intent-pool-to-gpu",
        supported_chips=("sm_120a",),
        description="windowed average/max pooling lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.upsample.nearest.v1",
        pass_argument="-convert-intent-upsample-to-gpu",
        supported_chips=("sm_120a",),
        description="nearest-neighbor/bicubic upsample lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.polar.elementwise.v1",
        pass_argument="-convert-intent-polar-to-gpu",
        supported_chips=("sm_120a",),
        description="polar real/imag pair lowering",
    ),
)


RAGGED_FAST_PATH_ID = "cuda.ragged.descriptor-aware"
_RAGGED_DESCRIPTOR_PASS = PassDescriptor(
    fast_path_id=RAGGED_FAST_PATH_ID,
    pass_argument="-convert-intent-ragged-to-gpu",
    supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
    description="descriptor-aware lowering of a closed local Ragged query",
)


def read_intent_module(intent_module: str | os.PathLike[str]) -> str:
    path = Path(intent_module)
    try:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            text = str(intent_module)
    except OSError:
        text = str(intent_module)
    if not text.strip() or ("module" not in text and "func.func" not in text):
        raise ValueError("selection requires non-empty Intent dialect MLIR")
    return text


def _passes(
    only_fast_path: str | None,
    passes: Sequence[PassDescriptor],
) -> tuple[PassDescriptor, ...]:
    selected = tuple(passes)
    if only_fast_path is None:
        return tuple(
            descriptor for descriptor in selected if not descriptor.expanded_only
        )
    filtered = tuple(
        descriptor
        for descriptor in selected
        if descriptor.fast_path_id == only_fast_path
    )
    if not filtered:
        raise ValueError(f"unknown fast path {only_fast_path!r}")
    return filtered


def select_variant(
    intent_module: str | os.PathLike[str],
    *,
    target: LoweringTarget | None = None,
    pass_overrides: Mapping[str, str] | None = None,
    only_fast_path: str | None = None,
    allow_specialized: bool = True,
    passes: Sequence[PassDescriptor] = DEFAULT_PASSES,
) -> VariantSelection:
    """Select a CUDA lowering variant by target capability and C++ probe.

    No verifier result, provider response, dataset tag, function
    name, or source path is accepted by this API.  ``pass_overrides`` is the
    explicit schedule/tuning seam; every override is still required to pass the
    same whole-graph structural probe before it can be selected.
    """

    text = read_intent_module(intent_module)
    selected_target = target or LoweringTarget()
    # Ragged descriptors use the descriptor-aware CUDA C++ lowering.  The
    # existing specialized flat-tensor probes must not see the tuple, while
    # the generic compiler entry still owns the pass selection and launch ABI.
    has_ragged = any(
        f"intent.{name}" in text
        for name in RAGGED_OPS
    )
    if has_ragged:
        if only_fast_path not in (None, RAGGED_FAST_PATH_ID):
            raise ValueError(
                "CUDA Ragged modules admit only the descriptor-aware lowering"
            )
        overrides = dict(pass_overrides or {})
        unknown_overrides = sorted(set(overrides) - {RAGGED_FAST_PATH_ID})
        if unknown_overrides:
            raise ValueError(
                "Ragged pass overrides reference unknown fast paths: "
                f"{unknown_overrides}"
            )
        pass_argument = overrides.get(
            RAGGED_FAST_PATH_ID, _RAGGED_DESCRIPTOR_PASS.pass_argument
        )
        if pass_argument != _RAGGED_DESCRIPTOR_PASS.pass_argument:
            raise ValueError(
                "CUDA Ragged lowering has no tunable pass override; use the "
                "registered descriptor-aware pass"
            )
        if selected_target.chip not in _RAGGED_DESCRIPTOR_PASS.supported_chips:
            attempt = ProbeAttempt(
                fast_path_id=RAGGED_FAST_PATH_ID,
                pass_argument=pass_argument,
                target_match=False,
                status="rejected_target",
                direct_hit=False,
                reason=f"target {selected_target.chip!r} is unsupported",
            )
        else:
            try:
                probe = backend_compile._run(
                    [backend_compile.INTENT_OPT, pass_argument], stdin=text
                )
            except Exception as error:
                attempt = ProbeAttempt(
                    fast_path_id=RAGGED_FAST_PATH_ID,
                    pass_argument=pass_argument,
                    target_match=True,
                    status="probe_error",
                    direct_hit=False,
                    reason=f"{type(error).__name__}: {error}",
                )
            else:
                residuals = residual_intent_ops(probe)
                launch_count = gpu_launch_count(probe)
                direct_hit = not residuals and launch_count >= 1
                attempt = ProbeAttempt(
                    fast_path_id=RAGGED_FAST_PATH_ID,
                    pass_argument=pass_argument,
                    target_match=True,
                    status="selected" if direct_hit else "rejected_structure",
                    direct_hit=direct_hit,
                    residual_ops=residuals,
                    gpu_launch_count=launch_count,
                    reason=(
                        "descriptor-aware Ragged lowering consumed the complete graph"
                        if direct_hit
                        else "descriptor-aware Ragged lowering did not consume the complete graph"
                    ),
                )
        # This pass is required ABI lowering, not an optional flat-tensor
        # specialization.  Therefore allow_specialized=False must not bypass
        # its probe, and the result stays in the generic pipeline class.
        return VariantSelection(
            pipeline_kind="generic",
            target=selected_target,
            fast_path_id=None,
            pass_argument=None,
            scf_to_cf=False,
            attempts=(attempt,),
            generic_fallback=False,
        )
    if not allow_specialized:
        return VariantSelection(
            pipeline_kind="generic",
            target=selected_target,
            fast_path_id=None,
            pass_argument=None,
            scf_to_cf=False,
            attempts=(),
        )

    overrides = dict(pass_overrides or {})
    known_ids = {descriptor.fast_path_id for descriptor in passes}
    unknown_overrides = sorted(set(overrides) - known_ids)
    if unknown_overrides:
        raise ValueError(f"pass overrides reference unknown fast paths: {unknown_overrides}")
    for fast_path_id, pass_argument in overrides.items():
        if not isinstance(pass_argument, str) or not pass_argument.strip():
            raise ValueError(f"pass override for {fast_path_id!r} is empty")

    attempts: list[ProbeAttempt] = []
    for descriptor in _passes(only_fast_path, passes):
        pass_argument = overrides.get(
            descriptor.fast_path_id, descriptor.pass_argument
        )
        if selected_target.chip not in descriptor.supported_chips:
            attempts.append(
                ProbeAttempt(
                    fast_path_id=descriptor.fast_path_id,
                    pass_argument=pass_argument,
                    target_match=False,
                    status="rejected_target",
                    direct_hit=False,
                    reason=f"target {selected_target.chip!r} is unsupported",
                )
            )
            continue
        missing_markers = tuple(
            marker for marker in descriptor.required_markers if marker not in text
        )
        if missing_markers:
            attempts.append(
                ProbeAttempt(
                    fast_path_id=descriptor.fast_path_id,
                    pass_argument=pass_argument,
                    target_match=True,
                    status="rejected_contract",
                    direct_hit=False,
                    reason="required typed MLIR markers are absent: "
                    + ", ".join(missing_markers),
                )
            )
            continue
        try:
            probe = backend_compile._run(
                [backend_compile.INTENT_OPT, pass_argument], stdin=text
            )
        except Exception as error:
            attempts.append(
                ProbeAttempt(
                    fast_path_id=descriptor.fast_path_id,
                    pass_argument=pass_argument,
                    target_match=True,
                    status="probe_error",
                    direct_hit=False,
                    reason=f"{type(error).__name__}: {error}",
                )
            )
            continue
        residuals = residual_intent_ops(probe)
        launch_count = gpu_launch_count(probe)
        direct_hit = not residuals and launch_count >= 1
        attempt = ProbeAttempt(
            fast_path_id=descriptor.fast_path_id,
            pass_argument=pass_argument,
            target_match=True,
            status="selected" if direct_hit else "rejected_structure",
            direct_hit=direct_hit,
            residual_ops=residuals,
            gpu_launch_count=launch_count,
            reason=(
                "compiler-launch-graph whole-graph structural match"
                if direct_hit
                else "specialized pass did not consume the complete graph"
            ),
        )
        attempts.append(attempt)
        if direct_hit:
            return VariantSelection(
                pipeline_kind="specialized",
                target=selected_target,
                fast_path_id=descriptor.fast_path_id,
                pass_argument=pass_argument,
                scf_to_cf=descriptor.scf_to_cf,
                attempts=tuple(attempts),
            )

    return VariantSelection(
        pipeline_kind="generic",
        target=selected_target,
        fast_path_id=None,
        pass_argument=None,
        scf_to_cf=False,
        attempts=tuple(attempts),
    )


__all__ = [
    "DEFAULT_PASSES",
    "LoweringTarget",
    "PassDescriptor",
    "ProbeAttempt",
    "VariantSelection",
    "gpu_launch_count",
    "read_intent_module",
    "residual_intent_ops",
    "select_variant",
]
