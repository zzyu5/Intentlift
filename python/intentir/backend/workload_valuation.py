"""Target-neutral workload valuations for the closed production J corpus.

This module belongs to the benchmark harness, not to compiler lowering.  A
profile may instantiate dimensions that are already symbolic in an accepted J
and may update the matching Source ABI/C launch values.  It must never turn a J
literal into a symbol or participate in candidate/path selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from intentir.verify.workload_adapters import TritonWorkloadAdapter


PRODUCTION_CASE_IDS: dict[str, str] = {
    "cross_entropy": "high_performance:liger_cross_entropy",
    "dyt": "high_performance:liger_dyt",
    "flash_attention_forward": (
        "high_performance:flash_attn.flash_attn_triton._fwd_kernel"
    ),
    "flash_bwd_preprocess": (
        "high_performance:flash_attn.flash_attn_triton."
        "_bwd_preprocess_do_o_dot"
    ),
    "flash_layer_norm_1pass": (
        "high_performance:flash_attn.ops.triton.layer_norm."
        "_layer_norm_fwd_1pass_kernel"
    ),
    "fused_add_rms_norm": "high_performance:liger_fused_add_rms_norm",
    "flash_linear": "high_performance:flash_attn.ops.triton.linear.kernel_fwd",
    "geglu": "high_performance:liger_geglu",
    "group_norm": "high_performance:liger_group_norm",
    "jsd": "high_performance:liger_jsd",
    "kl_div": "high_performance:liger_kl_div",
    "layer_norm": "high_performance:liger_layer_norm",
    "llama4_rope": "high_performance:liger_llama4_rope",
    "liger_rope": "high_performance:liger_rope",
    "liger_rms_norm": "high_performance:liger_rms_norm",
    "poly_norm": "high_performance:liger_poly_norm",
    "qwen2vl_mrope": "high_performance:liger_qwen2vl_mrope",
    "softmax": "high_performance:liger_softmax",
    "sparsemax": "high_performance:liger_sparsemax",
    "splitk_reduce_varargs": (
        "high_performance:xformers.ops.fmha._triton.splitk_kernels."
        "_splitK_reduce_varargs"
    ),
    "swiglu": "high_performance:liger_swiglu",
    "tvd": "high_performance:liger_tvd",
    "xformers_rms_norm": (
        "high_performance:xformers.ops._triton.rmsnorm_kernels."
        "_rms_norm_kernel"
    ),
    "xformers_rms_norm_add": (
        "high_performance:xformers.ops._triton.rmsnorm_kernels."
        "_rms_norm_add_kernel"
    ),
}

# Every production entry is a registered high-performance Source workload.
FORMAL_PRODUCTION_CASE_IDS: dict[str, str] = dict(PRODUCTION_CASE_IDS)

PRODUCTION_WORKLOAD_PROFILE_CLOSED = "closed-v1"
PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE = "natural-headline-v1"
PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE = "roofline-large-j-only-v1"
PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM = "symbolic-medium-v1"
PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE = "symbolic-large-v1"
PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S512 = "flash-forward-s512-v1"
PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S2048 = "flash-forward-s2048-v1"

_ROW = "row"
_FEATURE = "feature"
_HEADS = "heads"
_LENGTH = "length"
_BATCH = "batch"
_GROUPS = "groups"
_SEQUENCE = "sequence"
_PADDED_SEQUENCE = "padded_sequence"
_HEAD_DIM = "head_dim"
_KV_HEADS = "kv_heads"
_MROPE_SECTIONS = "mrope_sections"
_M = "m"
_N = "n"
_K = "k"
_CHANNELS = "channels"
_CHANNELS_PER_GROUP = "channels_per_group"
_MROPE_SECTION_T = "mrope_section_t"
_MROPE_SECTION_H = "mrope_section_h"
_ROTARY_PAIRS = "rotary_pairs"

_PROFILE_EXTENTS: dict[str, dict[str, int]] = {
    PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S512: {
        # Keep B*S constant across the 512/1024/2048 Flash sweep.  This is the
        # usual model-serving memory envelope and, importantly, keeps the
        # certified Source's 128-key online-softmax recurrence unchanged.
        _BATCH: 16,
        _HEADS: 32,
        _SEQUENCE: 512,
        _PADDED_SEQUENCE: 512,
        _HEAD_DIM: 64,
    },
    PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S2048: {
        _BATCH: 4,
        _HEADS: 32,
        _SEQUENCE: 2048,
        _PADDED_SEQUENCE: 2048,
        _HEAD_DIM: 64,
    },
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM: {
        _ROW: 128,
        _FEATURE: 1024,
        _HEADS: 32,
        _LENGTH: 1024 * 1024,
    },
    PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE: {
        _ROW: 2048,
        _FEATURE: 4096,
        _HEADS: 256,
        _LENGTH: 16 * 1024 * 1024,
    },
    PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE: {
        _LENGTH: 16 * 1024 * 1024,
    },
}

# One source-grounded/model-grounded headline valuation per production J.  The
# values are fixed, explicit model configurations rather than artificial size
# tiers. SplitK keeps the accepted vararg arity at two while using the
# decode-shaped tensor axes.
_NATURAL_HEADLINE_EXTENTS: dict[str, dict[str, int]] = {
    "swiglu": {_ROW: 16384, _FEATURE: 11008},
    "liger_rms_norm": {_ROW: 2048, _FEATURE: 4096},
    "fused_add_rms_norm": {_ROW: 2048, _FEATURE: 4096},
    "geglu": {_ROW: 32768, _FEATURE: 11008},
    "layer_norm": {_ROW: 4096, _FEATURE: 4096},
    "softmax": {_ROW: 2048, _FEATURE: 4096},
    "dyt": {_ROW: 4096, _FEATURE: 4096},
    "sparsemax": {_ROW: 2048, _FEATURE: 4096},
    "kl_div": {_ROW: 4096, _FEATURE: 4096},
    "poly_norm": {_ROW: 2048, _FEATURE: 4096},
    "flash_attention_forward": {
        # This is a model-scale 32-head attention workload rather than the old
        # tiny B2/H8 smoke shape.  Production J materialization has no implicit
        # scan-memory budget: a physical scan tile must come from a validated
        # target Layer-C binding, while the C-free path consumes the complete
        # semantic axis.
        _BATCH: 8,
        _HEADS: 32,
        _SEQUENCE: 1024,
        _PADDED_SEQUENCE: 1024,
        _HEAD_DIM: 64,
    },
    "flash_bwd_preprocess": {
        _BATCH: 4,
        _HEADS: 16,
        _SEQUENCE: 8192,
        _PADDED_SEQUENCE: 8192,
        _HEAD_DIM: 128,
    },
    "flash_layer_norm_1pass": {_ROW: 4096, _FEATURE: 4096},
    "xformers_rms_norm_add": {_ROW: 4096, _FEATURE: 12288},
    "splitk_reduce_varargs": {
        _BATCH: 1,
        _GROUPS: 1,
        _HEADS: 16,
        _ROW: 1,
        _HEAD_DIM: 256,
    },
    "jsd": {_ROW: 4096, _FEATURE: 4096},
    "tvd": {_ROW: 4096, _FEATURE: 4096},
    "xformers_rms_norm": {_ROW: 4096, _FEATURE: 12288},
    "cross_entropy": {_ROW: 1024, _FEATURE: 2048},
    "group_norm": {
        _BATCH: 8,
        _GROUPS: 32,
        _FEATURE: 512,
        _CHANNELS: 256,
        _CHANNELS_PER_GROUP: 8,
    },
    "liger_rope": {
        _BATCH: 4,
        _SEQUENCE: 2048,
        _HEADS: 16,
        _KV_HEADS: 8,
        _HEAD_DIM: 64,
    },
    "llama4_rope": {
        _BATCH: 8,
        _SEQUENCE: 1024,
        _HEADS: 8,
        _KV_HEADS: 4,
        _HEAD_DIM: 128,
        _ROTARY_PAIRS: 64,
    },
    "qwen2vl_mrope": {
        _BATCH: 8,
        _SEQUENCE: 1024,
        _HEADS: 8,
        _KV_HEADS: 8,
        _HEAD_DIM: 128,
        _MROPE_SECTION_T: 16,
        _MROPE_SECTION_H: 24,
    },
    "flash_linear": {_M: 1024, _N: 1024, _K: 8192},
}


@dataclass(frozen=True)
class ValueBinding:
    """Bind one scalar-like Source ABI value to a semantic extent."""

    formal: str
    role: str
    multiplier: int = 1
    factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CBinding:
    """Revalue one C slot without changing the accepted J."""

    formal: str
    role: str
    transform: Literal["identity", "next_power_of_two"] = "identity"


@dataclass(frozen=True)
class GridBinding:
    """Closed grid-expression vocabulary for workload harness revaluation."""

    op: Literal[
        "preserve",
        "extent",
        "ceil_div_extent_c",
        "ceil_div_extent_product_c",
    ]
    role: str | None = None
    c_formal: str | None = None
    factors: tuple[str, ...] = ()
    c_formals: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessBindingSpec:
    """Complete, auditable mapping from J-symbol roles to one Source adapter."""

    # Every adapter buffer must occur exactly once.  ``None`` preserves the
    # original literal dimension; a role replaces only a J-symbolic dimension.
    buffer_axes: tuple[tuple[str, tuple[str | None, ...]], ...]
    scalars: tuple[ValueBinding, ...] = ()
    semantic_constexprs: tuple[ValueBinding, ...] = ()
    c_values: tuple[CBinding, ...] = ()
    grid: tuple[GridBinding, ...] = ()
    literal_constraints: tuple[str, ...] = ()
    source_baseline: Literal["source_and_refj", "target_refj_only"] = (
        "source_and_refj"
    )
    source_unavailable_reason: str | None = None
    grid_contract_override: tuple[str, ...] | None = None


def _axes(*items: str | None) -> tuple[str | None, ...]:
    return tuple(items)


def _preserve_grid(count: int = 1) -> tuple[GridBinding, ...]:
    return tuple(GridBinding("preserve") for _ in range(count))


_HARNESS_BINDINGS: dict[str, HarnessBindingSpec] = {
    "cross_entropy": HarnessBindingSpec(
        buffer_axes=(
            ("X_ptr", _axes(_ROW, _FEATURE)),
            ("Y_ptr", _axes(_ROW)),
            ("weight_ptr", _axes(_FEATURE)),
            ("loss_ptr", _axes(_ROW)),
            ("z_loss_ptr", _axes(_ROW)),
            ("token_accuracy_ptr", _axes(_ROW)),
            ("predicted_tokens_ptr", _axes(_ROW)),
        ),
        scalars=(
            ValueBinding("X_stride", _FEATURE),
            ValueBinding("n_cols", _FEATURE),
            ValueBinding("n_non_ignore", _ROW),
            ValueBinding("sum_non_ignore_weight", _ROW),
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
        literal_constraints=(
            "Y_stride=1",
            "loss_stride=1",
            "RETURN_*=false",
            "HAS_WEIGHT=false",
            "HAS_SOFTCAPPING=false",
            "HAS_GRADIENTS=false",
        ),
    ),
    "dyt": HarnessBindingSpec(
        buffer_axes=(
            ("X", _axes(None, _FEATURE)),
            ("Y", _axes(None, _FEATURE)),
            ("Alpha", _axes(None)),
            ("Gamma", _axes(_FEATURE)),
            ("Beta", _axes(_FEATURE)),
        ),
        semantic_constexprs=(ValueBinding("N", _FEATURE),),
        c_values=(CBinding("BLOCK_N", _FEATURE, "next_power_of_two"),),
        grid=(
            GridBinding("ceil_div_extent_c", _FEATURE, "BLOCK_N"),
            GridBinding("preserve"),
        ),
        literal_constraints=("rows=2", "Alpha.elements=1"),
    ),
    "flash_attention_forward": HarnessBindingSpec(
        buffer_axes=(
            ("Q", _axes(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)),
            ("K", _axes(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)),
            ("V", _axes(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)),
            ("Bias", _axes(None)),
            ("Out", _axes(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)),
            ("Lse", _axes(_BATCH, _HEADS, _PADDED_SEQUENCE)),
            ("TMP", _axes(_BATCH, _HEADS, _PADDED_SEQUENCE)),
        ),
        scalars=(
            ValueBinding("stride_qb", _SEQUENCE, factors=(_HEADS, _HEAD_DIM)),
            ValueBinding("stride_qh", _HEAD_DIM),
            ValueBinding("stride_qm", _HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("stride_kb", _SEQUENCE, factors=(_HEADS, _HEAD_DIM)),
            ValueBinding("stride_kh", _HEAD_DIM),
            ValueBinding("stride_kn", _HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("stride_vb", _SEQUENCE, factors=(_HEADS, _HEAD_DIM)),
            ValueBinding("stride_vh", _HEAD_DIM),
            ValueBinding("stride_vn", _HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("stride_ob", _SEQUENCE, factors=(_HEADS, _HEAD_DIM)),
            ValueBinding("stride_oh", _HEAD_DIM),
            ValueBinding("stride_om", _HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("nheads", _HEADS),
            ValueBinding("seqlen_q", _SEQUENCE),
            ValueBinding("seqlen_k", _SEQUENCE),
            ValueBinding("seqlen_q_rounded", _PADDED_SEQUENCE),
            ValueBinding("headdim", _HEAD_DIM),
        ),
        grid=(
            GridBinding("ceil_div_extent_c", _SEQUENCE, "BLOCK_M"),
            GridBinding("extent", _BATCH, factors=(_HEADS,)),
        ),
        literal_constraints=(
            "BIAS_TYPE=none",
            "IS_CAUSAL=false",
            "EVEN_M/N/HEADDIM=true",
        ),
        grid_contract_override=(
            "ceil_div(seqlen_q,BLOCK_M)",
            "batch*nheads",
        ),
    ),
    "flash_bwd_preprocess": HarnessBindingSpec(
        buffer_axes=(
            ("Out", _axes(None, _HEADS, None, None)),
            ("DO", _axes(None, _HEADS, None, None)),
            ("Delta", _axes(None, _HEADS, None)),
        ),
        scalars=(
            ValueBinding("stride_ob", _HEADS, 33 * 16),
            ValueBinding("stride_dob", _HEADS, 33 * 16),
            ValueBinding("nheads", _HEADS),
        ),
        grid=(GridBinding("preserve"), GridBinding("extent", _HEADS)),
        literal_constraints=(
            "batch=1",
            "sequence=33",
            "head_dim=16",
            "rounded_sequence=64",
        ),
    ),
    "flash_layer_norm_1pass": HarnessBindingSpec(
        buffer_axes=(
            ("X", _axes(_ROW, _FEATURE)),
            ("Y", _axes(_ROW, _FEATURE)),
            ("W", _axes(_FEATURE)),
            ("B", _axes(_FEATURE)),
            ("RESIDUAL", _axes(_ROW, _FEATURE)),
            ("X1", _axes(_ROW, _FEATURE)),
            ("W1", _axes(_FEATURE)),
            ("B1", _axes(_FEATURE)),
            ("Y1", _axes(_ROW, _FEATURE)),
            ("RESIDUAL_OUT", _axes(_ROW, _FEATURE)),
            ("ROWSCALE", _axes(_ROW)),
            ("SEEDS", _axes(_ROW)),
            ("DROPOUT_MASK", _axes(_ROW, _FEATURE)),
            ("DROPOUT_MASK1", _axes(_ROW, _FEATURE)),
            ("Mean", _axes(_ROW)),
            ("Rstd", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "stride_x_row",
                "stride_y_row",
                "stride_res_row",
                "stride_res_out_row",
                "stride_x1_row",
                "stride_y1_row",
                "N",
            )
        )
        + (ValueBinding("M", _ROW),),
        c_values=(CBinding("BLOCK_N", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "fused_add_rms_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(None, _FEATURE)),
            ("S_ptr", _axes(None, _FEATURE)),
            ("X_ptr", _axes(None, _FEATURE)),
            ("R_ptr", _axes(None, _FEATURE)),
            ("W_ptr", _axes(_FEATURE)),
            ("RSTD_ptr", _axes(None)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "Y_row_stride",
                "S_row_stride",
                "X_row_stride",
                "R_row_stride",
                "W_row_stride",
                "n_cols",
            )
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2",),
    ),
    "flash_linear": HarnessBindingSpec(
        buffer_axes=(
            ("C", _axes(_M, _N)),
            ("ACT_INPUT", _axes(_M, _N)),
            ("A", _axes(_M, _K)),
            ("B", _axes(_N, _K)),
            ("bias", _axes(_N)),
        ),
        scalars=(
            ValueBinding("M", _M),
            ValueBinding("N", _N),
            ValueBinding("K", _K),
            ValueBinding("CACHE_KEY_M", _M),
            ValueBinding("CACHE_KEY_N", _N),
            ValueBinding("CACHE_KEY_K", _K),
            ValueBinding("stride_cm", _N),
            ValueBinding("stride_am", _K),
            ValueBinding("stride_bn", _K),
        ),
        c_values=(
            CBinding("BLOCK_M", _M),
            CBinding("BLOCK_N", _N),
            CBinding("BLOCK_K", _K),
        ),
        grid=_preserve_grid(),
        literal_constraints=(
            "BIAS=true",
            "SAVE_ACT_INPUT=true",
            "ACTIVATION=id",
            "SPLIT_K=1",
            "A_ROWMAJOR=true",
            "B_COLMAJOR=true",
        ),
    ),
    "geglu": HarnessBindingSpec(
        buffer_axes=tuple(
            (name, _axes(None, _FEATURE)) for name in ("a", "b", "c")
        ),
        scalars=(ValueBinding("stride", _FEATURE),),
        semantic_constexprs=(ValueBinding("n_cols", _FEATURE),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2",),
    ),
    "group_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(_BATCH, _GROUPS, _FEATURE)),
            ("X_ptr", _axes(_BATCH, _GROUPS, _FEATURE)),
            ("Mean_ptr", _axes(_BATCH, _GROUPS)),
            ("RSTD_ptr", _axes(_BATCH, _GROUPS)),
            ("W_ptr", _axes(None)),
            ("B_ptr", _axes(None)),
        ),
        scalars=(
            ValueBinding("Y_row_stride", _FEATURE, factors=(_GROUPS,)),
            ValueBinding("Y_col_stride", _FEATURE),
            ValueBinding("X_row_stride", _FEATURE, factors=(_GROUPS,)),
            ValueBinding("X_col_stride", _FEATURE),
            ValueBinding("Mean_row_stride", _GROUPS),
            ValueBinding("RSTD_row_stride", _GROUPS),
            ValueBinding("hidden_size", _FEATURE),
            ValueBinding("channels_per_group", _GROUPS),
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _BATCH), GridBinding("extent", _GROUPS)),
        literal_constraints=(
            "Mean_col_stride=1",
            "RSTD_col_stride=1",
            "eps=1e-6",
            "weights.elements=4",
        ),
    ),
    "jsd": HarnessBindingSpec(
        buffer_axes=(
            ("X_ptr", _axes(_ROW, _FEATURE)),
            ("Y_ptr", _axes(_ROW, _FEATURE)),
            ("loss_ptr", _axes(_ROW, _FEATURE)),
            ("dX_ptr", _axes(_ROW, _FEATURE)),
            ("label_ptr", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in ("X_stride", "Y_stride", "loss_stride", "dX_stride", "n_cols")
        )
        + (ValueBinding("n_non_ignore", _ROW),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "kl_div": HarnessBindingSpec(
        buffer_axes=(
            ("y_ptr", _axes(None, _FEATURE)),
            ("gt_ptr", _axes(None, _FEATURE)),
            ("loss_ptr", _axes(None)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in ("y_stride", "gt_stride", "n_cols")
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2",),
    ),
    "layer_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(_ROW, _FEATURE)),
            ("X_ptr", _axes(_ROW, _FEATURE)),
            ("W_ptr", _axes(_FEATURE)),
            ("B_ptr", _axes(_FEATURE)),
            ("Mean_ptr", _axes(_ROW)),
            ("RSTD_ptr", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "Y_row_stride",
                "X_row_stride",
                "W_row_stride",
                "B_row_stride",
                "n_cols",
            )
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "liger_rope": HarnessBindingSpec(
        buffer_axes=(
            ("q_ptr", _axes(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)),
            ("k_ptr", _axes(_BATCH, _SEQUENCE, _KV_HEADS, _HEAD_DIM)),
            ("cos", _axes(None, _SEQUENCE, _HEAD_DIM)),
            ("sin", _axes(None, _SEQUENCE, _HEAD_DIM)),
        ),
        scalars=(
            ValueBinding("q_row_stride", _HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("k_row_stride", _KV_HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("cos_row_stride", _HEAD_DIM),
            ValueBinding("sin_row_stride", _HEAD_DIM),
            ValueBinding("sl", _SEQUENCE),
        ),
        semantic_constexprs=(
            ValueBinding("bs", _BATCH),
            ValueBinding("n_qh", _HEADS),
            ValueBinding("n_kh", _KV_HEADS),
            ValueBinding("hd", _HEAD_DIM),
        ),
        c_values=(
            CBinding("pad_n_qh", _HEADS),
            CBinding("pad_n_kh", _KV_HEADS),
            CBinding("pad_hd", _HEAD_DIM),
            CBinding("BLOCK_SIZE", _HEADS),
        ),
        grid=(GridBinding("extent", _BATCH, factors=(_SEQUENCE,)),),
        literal_constraints=("cos_bs=1", "BACKWARD_PASS=false"),
        grid_contract_override=("bs*sl",),
    ),
    "llama4_rope": HarnessBindingSpec(
        buffer_axes=(
            ("q_ptr", _axes(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)),
            ("k_ptr", _axes(_BATCH, _SEQUENCE, _KV_HEADS, _HEAD_DIM)),
            ("freqs_complex_ptr", _axes(_SEQUENCE, _ROTARY_PAIRS, None)),
        ),
        scalars=(
            ValueBinding("q_row_stride", _HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("k_row_stride", _KV_HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("q_head_stride", _HEAD_DIM),
            ValueBinding("k_head_stride", _HEAD_DIM),
            ValueBinding("freqs_row_stride", _HEAD_DIM),
            ValueBinding("seq_len", _SEQUENCE),
            ValueBinding("batch_size", _BATCH),
        ),
        semantic_constexprs=(
            ValueBinding("head_dim_half", _ROTARY_PAIRS),
            ValueBinding("n_q_heads", _HEADS),
            ValueBinding("n_k_heads", _KV_HEADS),
        ),
        c_values=(
            CBinding("BLOCK_SIZE", _ROTARY_PAIRS, "next_power_of_two"),
        ),
        grid=(
            GridBinding("extent", _BATCH, factors=(_SEQUENCE,)),
            GridBinding("extent", _HEADS),
        ),
        literal_constraints=("imag_sign=1", "n_q_heads>=n_k_heads"),
        grid_contract_override=(
            "batch_size*seq_len",
            "max(n_q_heads,n_k_heads)",
        ),
    ),
    "liger_rms_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(None, _FEATURE)),
            ("X_ptr", _axes(None, _FEATURE)),
            ("W_ptr", _axes(_FEATURE)),
            ("RSTD_ptr", _axes(None)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in ("Y_row_stride", "X_row_stride", "W_row_stride", "n_cols")
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2",),
    ),
    "masked_copy": HarnessBindingSpec(
        buffer_axes=(("X", _axes(_LENGTH)), ("Y", _axes(_LENGTH))),
        scalars=(ValueBinding("N", _LENGTH),),
        grid=(GridBinding("ceil_div_extent_c", _LENGTH, "BLOCK_SIZE"),),
        source_baseline="target_refj_only",
        source_unavailable_reason=(
            "accepted Source addresses one BLOCK_SIZE tile and does not use "
            "program_id; expanded J valuations are Target--RefJ-only"
        ),
        grid_contract_override=("ceil_div(N,BLOCK_SIZE)",),
    ),
    "masked_row_square_sum": HarnessBindingSpec(
        buffer_axes=(("X", _axes(None, _FEATURE)), ("Y", _axes(None))),
        scalars=(ValueBinding("N", _FEATURE),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2",),
    ),
    "poly_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(None, _FEATURE)),
            ("X_ptr", _axes(None, _FEATURE)),
            ("W_ptr", _axes(None)),
            ("B_ptr", _axes(None)),
            ("RSTD_ptr", _axes(None, None)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in ("Y_row_stride", "X_row_stride", "n_cols")
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2", "polynomial_terms=3"),
    ),
    "qwen2vl_mrope": HarnessBindingSpec(
        buffer_axes=(
            ("q_ptr", _axes(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)),
            ("k_ptr", _axes(_BATCH, _SEQUENCE, _KV_HEADS, _HEAD_DIM)),
            ("cos", _axes(_MROPE_SECTIONS, _BATCH, _SEQUENCE, _HEAD_DIM)),
            ("sin", _axes(_MROPE_SECTIONS, _BATCH, _SEQUENCE, _HEAD_DIM)),
        ),
        scalars=(ValueBinding("sl", _SEQUENCE),),
        semantic_constexprs=(
            ValueBinding("bs", _BATCH),
            ValueBinding("n_qh", _HEADS),
            ValueBinding("n_kh", _KV_HEADS),
            ValueBinding("hd", _HEAD_DIM),
            ValueBinding("mrope_section_t", _MROPE_SECTIONS),
            ValueBinding("mrope_section_h", _MROPE_SECTIONS),
        ),
        c_values=(
            CBinding("pad_n_qh", _HEADS),
            CBinding("pad_n_kh", _KV_HEADS),
            CBinding("pad_hd", _HEAD_DIM),
            CBinding("BLOCK_SIZE", _HEADS),
        ),
        grid=(GridBinding("extent", _BATCH, factors=(_SEQUENCE,)),),
        literal_constraints=("BACKWARD_PASS=false",),
        grid_contract_override=("bs*sl",),
    ),
    "softmax": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(None, _FEATURE)),
            ("X_ptr", _axes(None, _FEATURE)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in ("Y_row_stride", "X_row_stride", "n_cols")
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2",),
    ),
    "sparsemax": HarnessBindingSpec(
        buffer_axes=tuple(
            (name, _axes(_ROW, _FEATURE))
            for name in ("x_ptr", "sorted_x_ptr", "o_ptr")
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "x_stride_row",
                "sorted_x_stride_row",
                "o_stride_row",
                "n_cols",
            )
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "splitk_reduce_varargs": HarnessBindingSpec(
        buffer_axes=(
            ("Out_splitK0", _axes(None, None, None, None, None)),
            ("Out_splitK1", _axes(None, None, None, None, None)),
            ("LSE_splitK0", _axes(None, None, None, None)),
            ("LSE_splitK1", _axes(None, None, None, None)),
            ("Out", _axes(None, None, None, None, None)),
            ("LSE", _axes(None, None, None, None)),
        ),
        grid=_preserve_grid(3),
        literal_constraints=(
            "batch=1",
            "groups=1",
            "heads=1",
            "M=2",
            "head_dim=16",
            "vararg_arity=2",
            "WRITE_LSE=true",
        ),
    ),
    "swiglu": HarnessBindingSpec(
        buffer_axes=tuple(
            (name, _axes(None, _FEATURE))
            for name in ("a_ptr", "b_ptr", "c_ptr")
        ),
        scalars=(ValueBinding("stride", _FEATURE),),
        semantic_constexprs=(ValueBinding("n_cols", _FEATURE),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2",),
    ),
    "tvd": HarnessBindingSpec(
        buffer_axes=(
            ("p_ptr", _axes(_ROW, _FEATURE)),
            ("q_ptr", _axes(_ROW, _FEATURE)),
            ("loss_ptr", _axes(_ROW, _FEATURE)),
            ("grads_ptr", _axes(_ROW, _FEATURE)),
            ("label_ptr", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in ("p_stride", "q_stride", "loss_stride", "grads_stride", "n_cols")
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "xformers_rms_norm": HarnessBindingSpec(
        buffer_axes=(
            ("x_ptr", _axes(None, _FEATURE)),
            ("h1_ptr", _axes(None, _FEATURE)),
            ("w_ptr", _axes(_FEATURE)),
        ),
        scalars=(ValueBinding("stride", _FEATURE),),
        semantic_constexprs=(ValueBinding("N_COLS", _FEATURE),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2",),
    ),
    "xformers_rms_norm_add": HarnessBindingSpec(
        buffer_axes=(
            ("x_ptr", _axes(None, _FEATURE)),
            ("y_ptr", _axes(None, _FEATURE)),
            ("h1_ptr", _axes(None, _FEATURE)),
            ("w_ptr", _axes(_FEATURE)),
        ),
        scalars=(ValueBinding("stride", _FEATURE),),
        semantic_constexprs=(ValueBinding("N_COLS", _FEATURE),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=_preserve_grid(),
        literal_constraints=("rows=2",),
    ),
}


_NATURAL_HARNESS_BINDINGS: dict[str, HarnessBindingSpec] = {
    "flash_attention_forward": _HARNESS_BINDINGS["flash_attention_forward"],
    "llama4_rope": _HARNESS_BINDINGS["llama4_rope"],
    "jsd": HarnessBindingSpec(
        buffer_axes=(
            ("X_ptr", _axes(_ROW, _FEATURE)),
            ("Y_ptr", _axes(_ROW, _FEATURE)),
            ("loss_ptr", _axes(_ROW, _FEATURE)),
            ("dX_ptr", _axes(_ROW, _FEATURE)),
            ("label_ptr", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "X_stride", "Y_stride", "loss_stride", "dX_stride", "n_cols"
            )
        )
        + (ValueBinding("n_non_ignore", _ROW),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "tvd": HarnessBindingSpec(
        buffer_axes=(
            ("p_ptr", _axes(_ROW, _FEATURE)),
            ("q_ptr", _axes(_ROW, _FEATURE)),
            ("loss_ptr", _axes(_ROW, _FEATURE)),
            ("grads_ptr", _axes(_ROW, _FEATURE)),
            ("label_ptr", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "p_stride", "q_stride", "loss_stride", "grads_stride", "n_cols"
            )
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "xformers_rms_norm": HarnessBindingSpec(
        buffer_axes=(
            ("x_ptr", _axes(_ROW, _FEATURE)),
            ("h1_ptr", _axes(_ROW, _FEATURE)),
            ("w_ptr", _axes(_FEATURE)),
        ),
        scalars=(ValueBinding("stride", _FEATURE),),
        semantic_constexprs=(ValueBinding("N_COLS", _FEATURE),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "cross_entropy": HarnessBindingSpec(
        buffer_axes=(
            ("X_ptr", _axes(_ROW, _FEATURE)),
            ("Y_ptr", _axes(_ROW)),
            ("weight_ptr", _axes(_FEATURE)),
            ("loss_ptr", _axes(_ROW)),
            ("z_loss_ptr", _axes(_ROW)),
            ("token_accuracy_ptr", _axes(_ROW)),
            ("predicted_tokens_ptr", _axes(_ROW)),
        ),
        scalars=(
            ValueBinding("X_stride", _FEATURE),
            ValueBinding("n_cols", _FEATURE),
            ValueBinding("n_non_ignore", _ROW),
            ValueBinding("sum_non_ignore_weight", _ROW),
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
        literal_constraints=(
            "Y_stride=1",
            "loss_stride=1",
            "RETURN_*=false",
            "HAS_WEIGHT=false",
            "HAS_SOFTCAPPING=false",
            "HAS_GRADIENTS=false",
        ),
    ),
    "group_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(_BATCH, _GROUPS, _FEATURE)),
            ("X_ptr", _axes(_BATCH, _GROUPS, _FEATURE)),
            ("Mean_ptr", _axes(_BATCH, _GROUPS)),
            ("RSTD_ptr", _axes(_BATCH, _GROUPS)),
            ("W_ptr", _axes(_CHANNELS)),
            ("B_ptr", _axes(_CHANNELS)),
        ),
        scalars=(
            ValueBinding("Y_row_stride", _FEATURE, factors=(_GROUPS,)),
            ValueBinding("Y_col_stride", _FEATURE),
            ValueBinding("X_row_stride", _FEATURE, factors=(_GROUPS,)),
            ValueBinding("X_col_stride", _FEATURE),
            ValueBinding("Mean_row_stride", _GROUPS),
            ValueBinding("RSTD_row_stride", _GROUPS),
            ValueBinding("hidden_size", _FEATURE),
            ValueBinding("channels_per_group", _CHANNELS_PER_GROUP),
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _BATCH), GridBinding("extent", _GROUPS)),
        literal_constraints=(
            "Mean_col_stride=1",
            "RSTD_col_stride=1",
            "eps=1e-6",
        ),
    ),
    "liger_rope": HarnessBindingSpec(
        buffer_axes=(
            ("q_ptr", _axes(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)),
            ("k_ptr", _axes(_BATCH, _SEQUENCE, _KV_HEADS, _HEAD_DIM)),
            ("cos", _axes(None, _SEQUENCE, _HEAD_DIM)),
            ("sin", _axes(None, _SEQUENCE, _HEAD_DIM)),
        ),
        scalars=(
            ValueBinding("q_row_stride", _HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("k_row_stride", _KV_HEADS, factors=(_HEAD_DIM,)),
            ValueBinding("cos_row_stride", _HEAD_DIM),
            ValueBinding("sin_row_stride", _HEAD_DIM),
            ValueBinding("sl", _SEQUENCE),
        ),
        semantic_constexprs=(
            ValueBinding("bs", _BATCH),
            ValueBinding("n_qh", _HEADS),
            ValueBinding("n_kh", _KV_HEADS),
            ValueBinding("hd", _HEAD_DIM),
        ),
        c_values=(
            CBinding("pad_n_qh", _HEADS),
            CBinding("pad_n_kh", _KV_HEADS),
            CBinding("pad_hd", _HEAD_DIM),
            CBinding("BLOCK_SIZE", _HEADS),
        ),
        grid=(GridBinding("extent", _BATCH, factors=(_SEQUENCE,)),),
        literal_constraints=("cos_bs=1", "BACKWARD_PASS=false"),
        grid_contract_override=("bs*sl",),
    ),
    "qwen2vl_mrope": HarnessBindingSpec(
        buffer_axes=(
            ("q_ptr", _axes(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)),
            ("k_ptr", _axes(_BATCH, _SEQUENCE, _KV_HEADS, _HEAD_DIM)),
            ("cos", _axes(None, _BATCH, _SEQUENCE, _HEAD_DIM)),
            ("sin", _axes(None, _BATCH, _SEQUENCE, _HEAD_DIM)),
        ),
        scalars=(ValueBinding("sl", _SEQUENCE),),
        semantic_constexprs=(
            ValueBinding("bs", _BATCH),
            ValueBinding("n_qh", _HEADS),
            ValueBinding("n_kh", _KV_HEADS),
            ValueBinding("hd", _HEAD_DIM),
            ValueBinding("mrope_section_t", _MROPE_SECTION_T),
            ValueBinding("mrope_section_h", _MROPE_SECTION_H),
        ),
        c_values=(
            CBinding("pad_n_qh", _HEADS),
            CBinding("pad_n_kh", _KV_HEADS),
            CBinding("pad_hd", _HEAD_DIM),
            CBinding("BLOCK_SIZE", _HEADS),
        ),
        grid=(GridBinding("extent", _BATCH, factors=(_SEQUENCE,)),),
        literal_constraints=("mrope_planes=3", "BACKWARD_PASS=false"),
        grid_contract_override=("bs*sl",),
    ),
    "flash_linear": HarnessBindingSpec(
        buffer_axes=(
            ("C", _axes(_M, _N)),
            ("ACT_INPUT", _axes(_M, _N)),
            ("A", _axes(_M, _K)),
            ("B", _axes(_N, _K)),
            ("bias", _axes(_N)),
        ),
        scalars=(
            ValueBinding("M", _M),
            ValueBinding("N", _N),
            ValueBinding("K", _K),
            ValueBinding("CACHE_KEY_M", _M),
            ValueBinding("CACHE_KEY_N", _N),
            ValueBinding("CACHE_KEY_K", _K),
            ValueBinding("stride_cm", _N),
            ValueBinding("stride_am", _K),
            ValueBinding("stride_bn", _K),
        ),
        grid=(
            GridBinding(
                "ceil_div_extent_product_c",
                _M,
                factors=(_N,),
                c_formals=("BLOCK_M", "BLOCK_N"),
            ),
        ),
        literal_constraints=(
            "BIAS=true",
            "SAVE_ACT_INPUT=true",
            "ACTIVATION=id",
            "SPLIT_K=1",
            "A_ROWMAJOR=true",
            "B_COLMAJOR=true",
        ),
    ),
    "swiglu": HarnessBindingSpec(
        buffer_axes=tuple(
            (name, _axes(_ROW, _FEATURE))
            for name in ("a_ptr", "b_ptr", "c_ptr")
        ),
        scalars=(ValueBinding("stride", _FEATURE),),
        semantic_constexprs=(ValueBinding("n_cols", _FEATURE),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "liger_rms_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(_ROW, _FEATURE)),
            ("X_ptr", _axes(_ROW, _FEATURE)),
            ("W_ptr", _axes(_FEATURE)),
            ("RSTD_ptr", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "Y_row_stride",
                "X_row_stride",
                "W_row_stride",
                "n_cols",
            )
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "fused_add_rms_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(_ROW, _FEATURE)),
            ("S_ptr", _axes(_ROW, _FEATURE)),
            ("X_ptr", _axes(_ROW, _FEATURE)),
            ("R_ptr", _axes(_ROW, _FEATURE)),
            ("W_ptr", _axes(_FEATURE)),
            ("RSTD_ptr", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "Y_row_stride",
                "S_row_stride",
                "X_row_stride",
                "R_row_stride",
                "W_row_stride",
                "n_cols",
            )
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "geglu": HarnessBindingSpec(
        buffer_axes=tuple(
            (name, _axes(_ROW, _FEATURE)) for name in ("a", "b", "c")
        ),
        scalars=(ValueBinding("stride", _FEATURE),),
        semantic_constexprs=(ValueBinding("n_cols", _FEATURE),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "layer_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(_ROW, _FEATURE)),
            ("X_ptr", _axes(_ROW, _FEATURE)),
            ("W_ptr", _axes(_FEATURE)),
            ("B_ptr", _axes(_FEATURE)),
            ("Mean_ptr", _axes(_ROW)),
            ("RSTD_ptr", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "Y_row_stride",
                "X_row_stride",
                "W_row_stride",
                "B_row_stride",
                "n_cols",
            )
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "softmax": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(_ROW, _FEATURE)),
            ("X_ptr", _axes(_ROW, _FEATURE)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in ("Y_row_stride", "X_row_stride", "n_cols")
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "dyt": HarnessBindingSpec(
        buffer_axes=(
            ("X", _axes(_ROW, _FEATURE)),
            ("Y", _axes(_ROW, _FEATURE)),
            ("Alpha", _axes(None)),
            ("Gamma", _axes(_FEATURE)),
            ("Beta", _axes(_FEATURE)),
        ),
        semantic_constexprs=(ValueBinding("N", _FEATURE),),
        c_values=(CBinding("BLOCK_N", _FEATURE, "next_power_of_two"),),
        grid=(
            GridBinding("ceil_div_extent_c", _FEATURE, "BLOCK_N"),
            GridBinding("extent", _ROW),
        ),
        literal_constraints=("Alpha.elements=1",),
    ),
    "sparsemax": HarnessBindingSpec(
        buffer_axes=tuple(
            (name, _axes(_ROW, _FEATURE))
            for name in ("x_ptr", "sorted_x_ptr", "o_ptr")
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "x_stride_row",
                "sorted_x_stride_row",
                "o_stride_row",
                "n_cols",
            )
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "kl_div": HarnessBindingSpec(
        buffer_axes=(
            ("y_ptr", _axes(_ROW, _FEATURE)),
            ("gt_ptr", _axes(_ROW, _FEATURE)),
            ("loss_ptr", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in ("y_stride", "gt_stride", "n_cols")
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "poly_norm": HarnessBindingSpec(
        buffer_axes=(
            ("Y_ptr", _axes(_ROW, _FEATURE)),
            ("X_ptr", _axes(_ROW, _FEATURE)),
            ("W_ptr", _axes(None)),
            ("B_ptr", _axes(None)),
            ("RSTD_ptr", _axes(_ROW, None)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in ("Y_row_stride", "X_row_stride", "n_cols")
        ),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
        literal_constraints=("polynomial_terms=3",),
    ),
    "flash_bwd_preprocess": HarnessBindingSpec(
        buffer_axes=(
            ("Out", _axes(_BATCH, _HEADS, _SEQUENCE, _HEAD_DIM)),
            ("DO", _axes(_BATCH, _HEADS, _SEQUENCE, _HEAD_DIM)),
            ("Delta", _axes(_BATCH, _HEADS, _PADDED_SEQUENCE)),
        ),
        scalars=(
            ValueBinding(
                "stride_ob", _HEADS, factors=(_SEQUENCE, _HEAD_DIM)
            ),
            ValueBinding("stride_oh", _SEQUENCE, factors=(_HEAD_DIM,)),
            ValueBinding("stride_om", _HEAD_DIM),
            ValueBinding(
                "stride_dob", _HEADS, factors=(_SEQUENCE, _HEAD_DIM)
            ),
            ValueBinding("stride_doh", _SEQUENCE, factors=(_HEAD_DIM,)),
            ValueBinding("stride_dom", _HEAD_DIM),
            ValueBinding("nheads", _HEADS),
            ValueBinding("seqlen_q", _SEQUENCE),
            ValueBinding("seqlen_q_rounded", _PADDED_SEQUENCE),
            ValueBinding("headdim", _HEAD_DIM),
        ),
        c_values=(
            CBinding("BLOCK_HEADDIM", _HEAD_DIM, "next_power_of_two"),
        ),
        grid=(
            GridBinding("ceil_div_extent_c", _SEQUENCE, "BLOCK_M"),
            GridBinding("extent", _BATCH, factors=(_HEADS,)),
        ),
    ),
    "flash_layer_norm_1pass": HarnessBindingSpec(
        buffer_axes=(
            ("X", _axes(_ROW, _FEATURE)),
            ("Y", _axes(_ROW, _FEATURE)),
            ("W", _axes(_FEATURE)),
            ("B", _axes(_FEATURE)),
            ("RESIDUAL", _axes(_ROW, _FEATURE)),
            ("X1", _axes(_ROW, _FEATURE)),
            ("W1", _axes(_FEATURE)),
            ("B1", _axes(_FEATURE)),
            ("Y1", _axes(_ROW, _FEATURE)),
            ("RESIDUAL_OUT", _axes(_ROW, _FEATURE)),
            ("ROWSCALE", _axes(_ROW)),
            ("SEEDS", _axes(_ROW)),
            ("DROPOUT_MASK", _axes(_ROW, _FEATURE)),
            ("DROPOUT_MASK1", _axes(_ROW, _FEATURE)),
            ("Mean", _axes(_ROW)),
            ("Rstd", _axes(_ROW)),
        ),
        scalars=tuple(
            ValueBinding(name, _FEATURE)
            for name in (
                "stride_x_row",
                "stride_y_row",
                "stride_res_row",
                "stride_res_out_row",
                "stride_x1_row",
                "stride_y1_row",
                "N",
            )
        )
        + (ValueBinding("M", _ROW),),
        c_values=(CBinding("BLOCK_N", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "xformers_rms_norm_add": HarnessBindingSpec(
        buffer_axes=(
            ("x_ptr", _axes(_ROW, _FEATURE)),
            ("y_ptr", _axes(_ROW, _FEATURE)),
            ("h1_ptr", _axes(_ROW, _FEATURE)),
            ("w_ptr", _axes(_FEATURE)),
        ),
        scalars=(ValueBinding("stride", _FEATURE),),
        semantic_constexprs=(ValueBinding("N_COLS", _FEATURE),),
        c_values=(CBinding("BLOCK_SIZE", _FEATURE, "next_power_of_two"),),
        grid=(GridBinding("extent", _ROW),),
    ),
    "splitk_reduce_varargs": HarnessBindingSpec(
        buffer_axes=(
            *(
                (name, _axes(_BATCH, _GROUPS, _HEADS, _ROW, _HEAD_DIM))
                for name in ("Out_splitK0", "Out_splitK1", "Out")
            ),
            *(
                (name, _axes(_BATCH, _GROUPS, _HEADS, _ROW))
                for name in ("LSE_splitK0", "LSE_splitK1", "LSE")
            ),
        ),
        scalars=(
            *(
                ValueBinding(
                    name,
                    _GROUPS,
                    factors=(_HEADS, _ROW, _HEAD_DIM),
                )
                for name in ("stride_osk_z0", "stride_osk_z1", "stride_oz")
            ),
            *(
                ValueBinding(name, _HEADS, factors=(_ROW, _HEAD_DIM))
                for name in ("stride_osk_g0", "stride_osk_g1", "stride_og")
            ),
            *(
                ValueBinding(name, _ROW, factors=(_HEAD_DIM,))
                for name in ("stride_osk_h0", "stride_osk_h1", "stride_oh")
            ),
            *(
                ValueBinding(name, _HEAD_DIM)
                for name in ("stride_osk_m0", "stride_osk_m1", "stride_om")
            ),
            *(
                ValueBinding(name, _GROUPS, factors=(_HEADS, _ROW))
                for name in ("stride_lsek_z0", "stride_lsek_z1", "stride_lse_z")
            ),
            *(
                ValueBinding(name, _HEADS, factors=(_ROW,))
                for name in ("stride_lsek_g0", "stride_lsek_g1", "stride_lse_g")
            ),
            *(
                ValueBinding(name, _ROW)
                for name in ("stride_lsek_h0", "stride_lsek_h1", "stride_lse_h")
            ),
        ),
        semantic_constexprs=(
            ValueBinding("head_dim", _HEAD_DIM),
            ValueBinding("H", _HEADS),
            ValueBinding("G", _GROUPS),
        ),
        c_values=(
            CBinding("head_dim_pow_2", _HEAD_DIM, "next_power_of_two"),
        ),
        grid=(
            GridBinding("extent", _ROW),
            GridBinding("extent", _BATCH, factors=(_GROUPS, _HEADS)),
            GridBinding("preserve"),
        ),
        literal_constraints=("vararg_arity=2", "WRITE_LSE=true"),
    ),
}


def _extent(extents: dict[str, int], role: str | None) -> int:
    if role is None or role not in extents:
        raise ValueError(f"workload profile has no positive extent for role {role!r}")
    value = int(extents[role])
    if value <= 0:
        raise ValueError(f"workload extent {role!r} must be positive")
    return value


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _binding_spec(name: str, profile: str) -> HarnessBindingSpec:
    if name not in PRODUCTION_CASE_IDS:
        raise KeyError(f"unknown production case {name!r}")
    if profile == PRODUCTION_WORKLOAD_PROFILE_CLOSED:
        return _HARNESS_BINDINGS[name]
    if profile == PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE:
        spec = _NATURAL_HARNESS_BINDINGS.get(name)
        if spec is None:
            raise ValueError(
                f"workload profile {profile!r} is not registered for {name!r}"
            )
        return spec
    if profile in {
        PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S512,
        PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S2048,
    }:
        if name != "flash_attention_forward":
            raise ValueError(
                f"workload profile {profile!r} is not registered for {name!r}"
            )
        return _NATURAL_HARNESS_BINDINGS[name]
    if profile == PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE and name != "masked_copy":
        raise ValueError(f"workload profile {profile!r} is not registered for {name!r}")
    if profile not in _PROFILE_EXTENTS:
        raise ValueError(f"unknown production workload profile {profile!r}")
    spec = _HARNESS_BINDINGS.get(name)
    if spec is None:
        raise ValueError(f"workload profile {profile!r} is not registered for {name!r}")
    return spec


def _profile_extents(name: str, profile: str) -> dict[str, int]:
    if profile == PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE:
        try:
            return _NATURAL_HEADLINE_EXTENTS[name]
        except KeyError as error:
            raise ValueError(
                f"workload profile {profile!r} is not registered for {name!r}"
            ) from error
    try:
        return _PROFILE_EXTENTS[profile]
    except KeyError as error:
        raise ValueError(f"unknown production workload profile {profile!r}") from error


def _extent_product(
    extents: dict[str, int],
    role: str | None,
    factors: tuple[str, ...],
) -> int:
    value = _extent(extents, role)
    for factor in factors:
        value *= _extent(extents, factor)
    return value


def case_name_for_id(case_id: str) -> str:
    matches = [name for name, registered in PRODUCTION_CASE_IDS.items() if registered == case_id]
    if len(matches) != 1:
        raise KeyError(f"production case_id {case_id!r} is not uniquely registered")
    return matches[0]


def workload_profile_metadata(name: str, profile: str) -> dict[str, Any]:
    """Return explicit horizontal-coverage and Source-baseline status."""

    spec = _binding_spec(name, profile)
    if profile == PRODUCTION_WORKLOAD_PROFILE_CLOSED:
        return {
            "profile": profile,
            "status": "accepted_closed_valuation",
            "revalued_roles": [],
            "literal_constraints": list(spec.literal_constraints),
            "baseline_policy": "source_and_refj",
            "source_unavailable_reason": None,
        }
    extents = _profile_extents(name, profile)
    roles = sorted(
        {
            role
            for _formal, axes in spec.buffer_axes
            for role in axes
            if role is not None
        }
    )
    return {
        "profile": profile,
        "status": "applicable",
        "revalued_roles": roles,
        "role_extents": {role: _extent(extents, role) for role in roles},
        "literal_constraints": list(spec.literal_constraints),
        "baseline_policy": spec.source_baseline,
        "source_unavailable_reason": spec.source_unavailable_reason,
    }


def apply_workload_profile(
    name: str,
    adapter: TritonWorkloadAdapter,
    profile: str,
) -> TritonWorkloadAdapter:
    """Instantiate only explicitly bound J-symbolic axes; fail closed otherwise."""

    spec = _binding_spec(name, profile)
    if adapter.case_id != PRODUCTION_CASE_IDS[name]:
        raise ValueError(f"workload adapter case_id does not match {name!r}")
    if profile == PRODUCTION_WORKLOAD_PROFILE_CLOSED:
        return adapter
    extents = _profile_extents(name, profile)

    buffer_axes = dict(spec.buffer_axes)
    actual_buffers = {item.formal for item in adapter.buffers}
    if set(buffer_axes) != actual_buffers:
        raise ValueError(
            f"workload buffer-axis map for {name!r} does not exactly match adapter"
        )
    buffers = []
    for item in adapter.buffers:
        axes = buffer_axes[item.formal]
        if len(axes) != len(item.shape):
            raise ValueError(
                f"workload axes for {name!r}/{item.formal!r} disagree with rank"
            )
        shape = tuple(
            original if role is None else _extent(extents, role)
            for original, role in zip(item.shape, axes, strict=True)
        )
        buffers.append(replace(item, shape=shape))

    def revalue_named(items: tuple[Any, ...], bindings: tuple[ValueBinding, ...], kind: str):
        by_name = {item.formal: item for item in items}
        unknown = {binding.formal for binding in bindings} - set(by_name)
        if unknown:
            raise ValueError(
                f"workload {kind} bindings for {name!r} name missing formals: "
                f"{sorted(unknown)}"
            )
        updates = {
            binding.formal: (
                _extent_product(extents, binding.role, binding.factors)
                * binding.multiplier
            )
            for binding in bindings
        }
        return tuple(
            replace(item, value=updates[item.formal])
            if item.formal in updates
            else item
            for item in items
        )

    scalars = revalue_named(adapter.scalars, spec.scalars, "scalar")
    constexpr_values = dict(adapter.semantic_constexprs)
    unknown_constexpr = {
        binding.formal for binding in spec.semantic_constexprs
    } - set(constexpr_values)
    if unknown_constexpr:
        raise ValueError(
            f"workload constexpr bindings for {name!r} name missing formals: "
            f"{sorted(unknown_constexpr)}"
        )
    for binding in spec.semantic_constexprs:
        constexpr_values[binding.formal] = (
            _extent_product(extents, binding.role, binding.factors)
            * binding.multiplier
        )
    semantic_constexprs = tuple(
        (key, constexpr_values[key]) for key, _value in adapter.semantic_constexprs
    )

    c_values = dict(adapter.c_values)
    unknown_c = {binding.formal for binding in spec.c_values} - set(c_values)
    if unknown_c:
        raise ValueError(
            f"workload C bindings for {name!r} name missing formals: {sorted(unknown_c)}"
        )
    for binding in spec.c_values:
        value = _extent(extents, binding.role)
        if binding.transform == "next_power_of_two":
            value = _next_power_of_two(value)
        elif binding.transform != "identity":
            raise ValueError(f"unknown C value transform {binding.transform!r}")
        c_values[binding.formal] = value
    ordered_c_values = tuple((key, c_values[key]) for key, _value in adapter.c_values)

    if len(spec.grid) != len(adapter.grid):
        raise ValueError(f"workload grid bindings for {name!r} disagree with rank")
    grid = []
    for original, binding in zip(adapter.grid, spec.grid, strict=True):
        if binding.op == "preserve":
            value = int(original)
        elif binding.op == "extent":
            value = _extent_product(extents, binding.role, binding.factors)
        elif binding.op == "ceil_div_extent_c":
            if binding.c_formal is None or binding.c_formal not in c_values:
                raise ValueError(
                    f"grid binding for {name!r} has no C slot {binding.c_formal!r}"
                )
            numerator = _extent_product(
                extents, binding.role, binding.factors
            )
            denominator = int(c_values[binding.c_formal])
            if denominator <= 0:
                raise ValueError("grid divisor must be positive")
            value = (numerator + denominator - 1) // denominator
        elif binding.op == "ceil_div_extent_product_c":
            if not binding.c_formals or any(
                name not in c_values for name in binding.c_formals
            ):
                raise ValueError(
                    f"grid binding for {name!r} has incomplete C divisors"
                )
            numerator = _extent_product(
                extents, binding.role, binding.factors
            )
            denominator = 1
            for c_name in binding.c_formals:
                denominator *= int(c_values[c_name])
            if denominator <= 0:
                raise ValueError("grid divisor product must be positive")
            value = (numerator + denominator - 1) // denominator
        else:
            raise ValueError(f"unknown workload grid operation {binding.op!r}")
        if value <= 0:
            raise ValueError("workload grid dimensions must be positive")
        grid.append(value)

    return replace(
        adapter,
        buffers=tuple(buffers),
        scalars=scalars,
        semantic_constexprs=semantic_constexprs,
        c_values=ordered_c_values,
        grid=tuple(grid),
        grid_contract=(spec.grid_contract_override or adapter.grid_contract),
    )


__all__ = [
    "FORMAL_PRODUCTION_CASE_IDS",
    "PRODUCTION_CASE_IDS",
    "PRODUCTION_WORKLOAD_PROFILE_CLOSED",
    "PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S512",
    "PRODUCTION_WORKLOAD_PROFILE_FLASH_FORWARD_S2048",
    "PRODUCTION_WORKLOAD_PROFILE_NATURAL_HEADLINE",
    "PRODUCTION_WORKLOAD_PROFILE_ROOFLINE_LARGE",
    "PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_LARGE",
    "PRODUCTION_WORKLOAD_PROFILE_SYMBOLIC_MEDIUM",
    "apply_workload_profile",
    "case_name_for_id",
    "workload_profile_metadata",
]
