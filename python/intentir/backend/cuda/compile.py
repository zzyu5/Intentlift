"""Compile an ``intent`` MLIR module to an sm_120a cubin.

The pass pipeline below is the empirically-verified path on **system MLIR
20.1.8** (no MLIR-21 escape hatch needed) to a launchable Blackwell cubin:

  intent-opt -convert-intent-to-linalg          # our lowering (Layer A -> linalg)
  mlir-opt   one-shot-bufferize                  # tensors -> memrefs
             convert-linalg-to-parallel-loops    # -> scf.parallel
             gpu-map-parallel-loops
             convert-parallel-loops-to-gpu        # -> gpu.launch
             gpu-kernel-outlining                 # -> gpu.module @<k>_kernel
             lower-affine
             convert-gpu-to-nvvm{bare-ptr}        # device -> nvvm  (barrier0 form)
             reconcile-unrealized-casts
             nvvm-attach-target{chip=sm_120a}
             gpu-module-to-binary                 # translate + llc + ptxas -> cubin

Notes: ``use-bare-ptr-memref-call-conv`` is the default for statically shaped
kernel arguments so dense kernel args are bare pointers (matching the CUDA
Driver-API launch on torch pointers).  Ragged lowering explicitly disables the
kernel bare-pointer calling convention: its dynamic destination is a ranked
memref, and the device receives the flattened descriptor fields
(allocated/aligned pointers, offset, sizes, and strides).  MLIR's CUDA
bare-pointer conversion requires every memref shape to be static, while the
host-side Ragged member allocation remains data-dependent.
The elementwise add path carries no barrier/shuffle, so the MLIR-20-vs-21
``gpu.barrier`` intrinsic divergence (see roadmap R5) does not bite here.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from intentir.backend.toolchain.tools import INTENT_OPT, MLIR_OPT
from intentir.ir.intent_op_contract import RAGGED_OPS

CUDA_BIN = os.environ.get("CUDA_BIN", "/usr/local/cuda-12.8/bin")
CHIP = os.environ.get("INTENT_GPU_CHIP", "sm_120a")

# Bare MLIR symbols may contain ``.`` and ``$`` after their first character.
# Live paper-pipeline functions preserve readable dotted source names, so
# artifact parsing must use the MLIR grammar instead of the narrower ``\w+``.
_MLIR_BARE_SYMBOL_RE = r"[A-Za-z_][A-Za-z0-9_.$]*"


def _has_ragged_ops(source: str) -> bool:
    return any(f"intent.{name}" in source for name in RAGGED_OPS)


def _has_structured_ragged_region(source: str) -> bool:
    """Return whether Ragged is used inside a structured Intent region.

    The dedicated CUDA Ragged pass owns the direct carrier/query ABI.  A
    structured loop must first go through the shared Intent-to-Linalg lowering
    so local Ragged producers/queries are rewritten before the loop is cloned;
    routing it to the straight-line pass was the remaining backend split.
    """

    return _has_ragged_ops(source) and re.search(
        r"\bintent\.loop\b", source
    ) is not None


def _validate_ragged_lowering(source: str, lowered: str) -> None:
    if not _has_ragged_ops(source):
        return
    if _has_ragged_ops(lowered):
        raise RuntimeError(
            "CUDA Ragged capability boundary left an Intent Ragged operation "
            "after -convert-intent-ragged-to-gpu"
        )
    if "tuple<" in lowered:
        raise RuntimeError(
            "CUDA Ragged capability boundary left a tuple descriptor in the "
            "target ABI"
        )
    if "gpu.launch" not in lowered:
        raise RuntimeError(
            "CUDA Ragged lowering produced no gpu.launch for the admitted query"
        )


def _validate_shared_ragged_lowering(source: str, lowered: str) -> None:
    """Check the common lowering's descriptor boundary before GPU passes."""

    if not _has_ragged_ops(source):
        return
    if _has_ragged_ops(lowered):
        raise RuntimeError(
            "CUDA structured Ragged lowering left a high-level Ragged operation"
        )
    if "tuple<" in lowered:
        raise RuntimeError(
            "CUDA structured Ragged lowering left a tuple descriptor in the ABI"
        )

_FRONT_PASSES = [
    "-one-shot-bufferize=bufferize-function-boundaries=true "
    "function-boundary-type-conversion=identity-layout-map "
    "allow-return-allocs-from-loops=true",
    "-convert-linalg-to-parallel-loops",
    "-gpu-map-parallel-loops",
    "-convert-parallel-loops-to-gpu",
    "-gpu-kernel-outlining",
    "-canonicalize",  # folds grid extents to literal constants
    "-cse",  # share identical typed producers before NVVM lowering
]

# NVIDIA-target preparation on the outlined device module: rewrites scalar f32
# math to native PTX intrinsics and backports LLVM-20's missing BF16
# nvgpu.mma.sync conversion.  Both run before the stock composite NVVM pipeline;
# modules without BF16 MMA make the compatibility pass a no-op.
_DEVICE_PREP_PASSES = [
    "-lower-gpu-math-to-nvvm",
    "-lower-bf16-mma-to-nvvm",
]

# Integrated device lowering (R3-proven): one pass gpu -> nvvm -> PTX. Handles
# scf.for/index/cf cleanly (the manual pass set left unrealized casts on
# reduction-trip index args). We emit PTX (`cubin-format=isa`) and run ptxas
# ourselves: the integrated `cubin-format=bin` serializer silently drops kernels
# that call an LLVM intrinsic (e.g. our ex2.approx exp), producing a cubin with
# no `.text` section. Manual ptxas on the identical PTX is correct.
def _device_pipeline(*, kernel_bare_ptr: bool = True) -> str:
    pipeline = (
        f'--gpu-lower-to-nvvm-pipeline=cubin-chip={CHIP} cubin-format=isa '
        f"kernel-bare-ptr-calling-convention={'true' if kernel_bare_ptr else 'false'}"
    )
    return pipeline


@dataclass
class CompiledKernel:
    cubin: bytes
    kernel_name: str        # the launchable gpu.func symbol
    binary_name: str        # the gpu.binary module name
    binary_mlir: str        # full lowered module (host launch_func intact for dims)


def _run(cmd: list[str], stdin: str | None = None) -> str:
    env = dict(os.environ)
    env["PATH"] = CUDA_BIN + os.pathsep + env.get("PATH", "")
    p = subprocess.run(
        cmd, input=stdin, capture_output=True, text=True, env=env
    )
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{p.stderr}")
    return p.stdout


def _unescape_mlir_bytes(s: str) -> bytes:
    out = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 2 < len(s):
            try:
                out.append(int(s[i + 1 : i + 3], 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(ord(s[i]))
        i += 1
    return bytes(out)


def lower_to_binary_mlir(intent_mlir: str | os.PathLike) -> str:
    """intent MLIR file -> module with one gpu.binary (PTX assembly) per kernel."""
    src = Path(intent_mlir).read_text()
    has_ragged = _has_ragged_ops(src)
    if has_ragged and not _has_structured_ragged_region(src):
        lowered = _run(
            [INTENT_OPT, "-convert-intent-ragged-to-gpu"],
            stdin=src,
        )
        _validate_ragged_lowering(src, lowered)
    else:
        lowered = _run([INTENT_OPT, "-convert-intent-to-linalg"], stdin=src)
        if has_ragged:
            _validate_shared_ragged_lowering(src, lowered)
    front = _run([MLIR_OPT, *_FRONT_PASSES], stdin=lowered)
    prepped = _run([INTENT_OPT, *_DEVICE_PREP_PASSES], stdin=front)
    return _run(
        [
            MLIR_OPT,
            _device_pipeline(kernel_bare_ptr=not has_ragged),
        ],
        stdin=prepped,
    )


# --------------------------------------------------------------------------- #
# Fused + tiled elementwise path (the "beat Triton via fusion" lowering).      #
# --------------------------------------------------------------------------- #
# The default `_FRONT_PASSES` emits one `linalg.generic` per intent op and maps
# every parallel iteration to its own *block* with a single thread (threads in
# (1,1,1)) — the same one-element-per-block granularity that makes the flash
# kernel slow. For a pointwise *chain* we instead:
#   1. `-linalg-fuse-elementwise-ops`  — collapse the whole chain to ONE
#      linalg.generic (one kernel, one HBM read of each input + one write),
#      which is the actual source of the win over a library that ships the chain
#      as separate kernels (each paying a full HBM round-trip).
#   2. `-scf-parallel-loop-tiling=...` — tile the innermost (contiguous) parallel
#      dim into a thread block, so the kernel is a coalesced ~256-thread grid
#      instead of one-thread-per-block. This is a Layer-C schedule choice (the
#      tile size); Layer A/B are untouched.
# tile=(1,256): rows -> grid.x, 256 contiguous columns -> threadIdx.y (coalesced).
def _fused_front_passes(tile: tuple[int, ...]) -> list[str]:
    ts = ",".join(str(t) for t in tile)
    return [
        "-linalg-fuse-elementwise-ops",
        "-one-shot-bufferize=bufferize-function-boundaries=true "
        "function-boundary-type-conversion=identity-layout-map",
        "-convert-linalg-to-parallel-loops",
        f"-scf-parallel-loop-tiling=parallel-loop-tile-sizes={ts}",
        "-gpu-map-parallel-loops",
        "-convert-parallel-loops-to-gpu",
        "-gpu-kernel-outlining",
        "-canonicalize",
        "-cse",
    ]


def lower_to_binary_mlir_fused(
    intent_mlir: str | os.PathLike, tile: tuple[int, ...] = (1, 256)
) -> str:
    """Like ``lower_to_binary_mlir`` but fuses the pointwise chain into ONE tiled
    GPU kernel (see ``_fused_front_passes``)."""
    src = Path(intent_mlir).read_text()
    if _has_ragged_ops(src):
        raise RuntimeError(
            "CUDA Ragged graphs require the descriptor-aware CUDA lowering; "
            "the elementwise fusion pipeline is not an ABI-compatible path"
        )
    linalg = _run([INTENT_OPT, "-convert-intent-to-linalg"], stdin=src)
    front = _run([MLIR_OPT, *_fused_front_passes(tile)], stdin=linalg)
    prepped = _run([INTENT_OPT, *_DEVICE_PREP_PASSES], stdin=front)
    return _run([MLIR_OPT, _device_pipeline()], stdin=prepped)


def compile_all_kernels_fused(
    intent_mlir: str | os.PathLike, tile: tuple[int, ...] = (1, 256)
) -> list[CompiledKernel]:
    """Compile every kernel through the fused+tiled pipeline for this call.

    For a pointwise chain this returns exactly one kernel; callers retain that
    returned executable explicitly for repeated launch timing.
    """
    return _extract_kernels(lower_to_binary_mlir_fused(intent_mlir, tuple(tile)))


# Device-side leg only (scf.parallel + scf.for + memref already in hand): maps
# the query parallel loop to the GPU grid, outlines, applies the exp peephole,
# and serializes. This is the back-end target the fused-flash `loop` lowering
# emits — a single kernel with scalar online-softmax carries through scf.for
# iter_args (no tensor iter_args / no bufferization of carries).
_DEVICE_FRONT_PASSES = [
    "-gpu-map-parallel-loops",
    "-convert-parallel-loops-to-gpu",
    "-gpu-kernel-outlining",
    "-canonicalize",
    "-cse",
]


def lower_memref_module_to_kernels(
    mlir: str | os.PathLike, scf_to_cf: bool = False
) -> list[CompiledKernel]:
    """A module already in scf.parallel+scf.for+memref form -> loadable cubins.

    ``scf_to_cf`` runs ``-convert-scf-to-cf`` just before the integrated NVVM
    pipeline. The default ``--gpu-lower-to-nvvm-pipeline`` lowers gpu->nvvm
    BEFORE its internal scf->cf, so an ``scf.for`` carrying ``!gpu.mma_matrix``
    iter_args (the tiled-K tensor-core accumulator of
    ``convert-intent-contraction-to-gpu``) fails to legalize — the gpu type
    converter never reaches the scf op. Pre-lowering scf to cf routes the
    fragment values through ``cf.br`` block args, which the gpu->nvvm type
    converter DOES convert. Off by default so the M1-M6 row-reduction/elementwise
    paths (scf.for with f32/index iter_args, handled fine by the composite) are
    byte-for-byte unchanged.
    """
    src = Path(mlir).read_text() if os.path.exists(str(mlir)) else str(mlir)
    front = _run([MLIR_OPT, *_DEVICE_FRONT_PASSES], stdin=src)
    prepped = _run([INTENT_OPT, *_DEVICE_PREP_PASSES], stdin=front)
    device_pipeline = _device_pipeline(kernel_bare_ptr="memref<?" not in src)
    pipeline = (
        ["-convert-scf-to-cf", device_pipeline]
        if scf_to_cf
        else [device_pipeline]
    )
    return _extract_kernels(_run([MLIR_OPT, *pipeline], stdin=prepped))


def _ptxas(ptx: bytes) -> bytes:
    """Assemble PTX to an sm cubin with the standalone ptxas (12.8)."""
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "k.ptx", Path(d) / "k.cubin"
        src.write_bytes(ptx)
        _run([str(Path(CUDA_BIN) / "ptxas"), f"-arch={CHIP}", "-O3",
              str(src), "-o", str(out)])
        return out.read_bytes()


def _extract_kernels(binary: str) -> list[CompiledKernel]:
    # gpu.func symbol per gpu.binary, recovered from the host launch sites.
    func_of = dict(
        re.findall(
            rf"gpu\.launch_func\s+@({_MLIR_BARE_SYMBOL_RE})::"
            rf"@({_MLIR_BARE_SYMBOL_RE})",
            binary,
        )
    )
    kernels: list[CompiledKernel] = []
    for m in re.finditer(
        rf'gpu\.binary @({_MLIR_BARE_SYMBOL_RE})\s*'
        r'\[#gpu\.object<#nvvm\.target<[^>]*>,'
        r'[^"]*?assembly = "((?:[^"\\]|\\.)*)"',
        binary,
    ):
        bin_name = m.group(1)
        ptx = _unescape_mlir_bytes(m.group(2))
        kernels.append(
            CompiledKernel(
                cubin=_ptxas(ptx),
                kernel_name=func_of.get(bin_name, bin_name),
                binary_name=bin_name,
                binary_mlir=binary,
            )
        )
    if not kernels:
        raise RuntimeError("no gpu.binary with PTX assembly produced")
    return kernels


def compile_all_kernels(intent_mlir: str | os.PathLike) -> list[CompiledKernel]:
    """Compile all outlined kernels in host launch order for this call."""
    return _extract_kernels(lower_to_binary_mlir(intent_mlir))


def compile_intent_to_cubin(intent_mlir: str | os.PathLike) -> CompiledKernel:
    """The single kernel of a single-kernel intent module (e.g. elementwise)."""
    return compile_all_kernels(intent_mlir)[0]
