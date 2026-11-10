"""Compile + launch the read-once row-reduction kernels, for PK vs the library.

The mature ``-convert-intent-row-reduction-to-gpu`` pass lowers a row-norm intent
func (rms_norm / softmax / ...) to ONE read-once block-cooperative ``gpu.launch``.
This module compiles that to an sm_120a cubin and launches it, resolving the
bare-ptr launch plan from the lowered host module (the outliner's capture order
is NOT the func-arg order, so we parse ``gpu.launch_func ... args(...)`` and map
each aligned-ptr back to its input index — same machinery the flash emitter uses).

MEASUREMENT (cost real debug time — encode here, do not relearn):
  * Time with ``triton.testing.do_bench`` over a NO-SYNC launch (``thunk``), the
    same way a Triton baseline is timed. do_bench manages clocks + an L2 flush
    and brackets just the kernel via CUDA events.
  * Two traps that fabricate a 4-38x phantom slowdown:
      (1) a ``torch.cuda.synchronize()`` INSIDE the timed launch (what
          ``intentir.backend.cuda.launch.launch`` does) breaks do_bench ->
          ~6800us for a 178us kernel. Use the no-sync ``thunk`` here.
      (2) a hand-rolled CUDA-event loop floors at ~730us for EVERY kernel
          (incl. torch.copy_ and liger) -> a fake "366 GB/s ceiling". It is the
          timer, not the kernel.
    Measured correctly, rms_norm/softmax sit at parity with liger (~1507 GB/s,
    peak HBM, bit-exact) — parity is the honest ceiling for these min-traffic
    library kernels (read-once, write-once: no bytes left to take out).
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

from intentir.backend.cuda.compile import (INTENT_OPT, _run,
                                       lower_memref_module_to_kernels)
from intentir.backend.cuda.launch import (
    I64Arg,
    PtrArg,
    U16Arg,
    U32Arg,
    U8Arg,
    _driver,
    _marshal,
    load_module,
)
from intentir.backend.cuda.launch_config import single_launch_config
from intentir.backend.cuda.orchestrate import (_host_func_text,
                                           _map_input_aligned_ptrs,
                                           _parse_constants, _parse_launches,
                                           _parse_signature_blockargs,
                                           _resolve_const_index)
from intentir.frontend.emit_mlir import emit_module
from intentir.ir.model import IntentFunction

_REPO = Path(__file__).resolve().parents[4]


def _scalar_arg(kind: str, value: int):
    constructors = {
        "i64": I64Arg,
        "u8": U8Arg,
        "u16": U16Arg,
        "u32": U32Arg,
    }
    try:
        return constructors[kind](value)
    except KeyError as exc:
        raise RuntimeError(f"unsupported scalar launch argument {kind!r}") from exc


def _append_scalar_plan(plan, ty: str, consts, ssa: str, *, context: str) -> None:
    kind = {
        "i64": "i64",
        "index": "i64",
        "ui8": "u8",
        "ui16": "u16",
        "ui32": "u32",
    }.get(ty)
    if kind is None:
        raise RuntimeError(f"{context}: unhandled launch-arg {ty}")
    plan.append((kind, int(consts[ssa][0])))


def _intent_mlir(name: str, dims: dict) -> str:
    spec = json.loads(
        (_REPO / "corpus" / "kernels" / name / "intent.json").read_text()
    )["intent_function"]
    return emit_module(IntentFunction.from_json(spec), dims, fold=0)


class RowReductionKernel:
    """A compiled row-reduction kernel + its resolved bare-ptr launch plan.

    ``thunk(tensors)`` returns a zero-arg, NO-SYNC launch closure ready for
    ``do_bench`` (or a single call + manual ``torch.cuda.synchronize()`` for a
    correctness check). ``tensors`` are the inputs in func-signature order
    followed by the output Y (all contiguous CUDA f32; a rank-0 scalar like eps
    is a 0-d / 1-elem tensor).
    """

    def __init__(self, name: str, dims: dict, block_threads: int,
                 pass_name: str = "convert-intent-row-reduction-to-gpu",
                 mlir_text: str | None = None):
        """Lower + compile a read-once kernel for PK.

        ``pass_name`` selects the C++ lowering pass to drive (default the
        row-reduction pass; pass ``"convert-intent-elementwise-to-gpu"`` for the
        pure-elementwise M3 activations). ``mlir_text`` is a raw intent-MLIR
        module string (M3 hand-writes the intent the same shape as the lit test,
        bypassing corpus extraction); when ``None`` the corpus ``intent.json``
        for ``name`` is emitted instead.
        """
        src = mlir_text if mlir_text is not None else _intent_mlir(name, dims)
        pass_options = f"block-threads={block_threads}"
        if pass_name == "convert-intent-row-reduction-to-gpu":
            # This low-level PK driver launches the outlined cubin directly and
            # therefore supplies pointer arguments only; it intentionally
            # bypasses the compiled host stub that performs rank-0 capture.
            # Production search exercises both capture states through
            # execute_cuda_artifact, while this harness retains the compatible
            # device-load state explicitly instead of depending on a pass
            # default it cannot execute.
            pass_options += " promote-rank0-scalars=false"
        rr = _run(
            [INTENT_OPT, f"-{pass_name}={pass_options}"],
            stdin=src,
        )
        kernels = lower_memref_module_to_kernels(rr)
        if len(kernels) != 1:
            raise RuntimeError(
                f"row_reduction: expected one kernel for {name}, got {len(kernels)}"
            )
        ck = kernels[0]
        self.func = load_module(ck.cubin).function(ck.kernel_name)

        _n, ftext = _host_func_text(ck.binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)
        if len(sites) != 1:
            raise RuntimeError(
                f"row_reduction: expected one gpu.launch_func, got {len(sites)}"
            )
        site = sites[0]

        class _Shim:
            constants = consts

        self.grid = tuple(_resolve_const_index(_Shim, s) for s in site.grid_ssa)
        self.block = tuple(_resolve_const_index(_Shim, s) for s in site.block_ssa)
        # M5 tiled+smem path: a workgroup (dynamic shared) row cache. Resolve the
        # dynamic_shared_memory_size constant and, when it exceeds the 48 KiB
        # default static cap, opt the kernel into a larger dynamic-shared budget.
        self.shared = 0
        if site.dynamic_shared_ssa is not None:
            self.shared = int(consts[site.dynamic_shared_ssa][0])
            # Opt the kernel into the larger dynamic-shared budget: the 48 KiB
            # default cap is for static shared, and the gpu.all_reduce already
            # uses a little static shared, so even a 48 KiB cache needs the
            # opt-in (max-dynamic-shared up to the sm_120 ~100 KiB ceiling).
            cuda = _driver()
            attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            cuda.cuFuncSetAttribute(self.func, attr, self.shared)
        # M6 (R-RUNTIME-1): PREFER the pass-emitted structured launch plan for
        # grid/block/smem instead of the regex-scraped values. `rr` is the
        # pass-output MLIR carrying the `intent.launch_plan` attribute. We ASSERT
        # the structured plan reproduces exactly the regex-recovered geometry
        # (equivalence proof across the PK suite) before trusting it; the regex
        # path stays as a validated fallback for kernels whose pass has not yet
        # been taught to emit a plan (plan is None).
        plan = single_launch_config(rr)
        if plan is not None:
            plan.assert_matches(self.grid, self.block, self.shared,
                                where=f"({name} {pass_name})")
            self.grid, self.block = plan.grid, plan.block
            self.shared = plan.smem
        # plan: per launch arg, ("ptr", input_index) or a typed scalar literal.
        # Scalar args appear when -canonicalize hoists a constant out of the
        # launch region so the outliner captures it (e.g. a folded `1`).
        self.plan = []
        for op in site.args:
            if op.ty.startswith("!llvm.ptr"):
                self.plan.append(("ptr", aligned[op.ssa]))
            else:
                _append_scalar_plan(
                    self.plan, op.ty, consts, op.ssa, context="row_reduction"
                )
        self._cuda = _driver()

    def thunk(self, tensors):
        """A zero-arg NO-SYNC launch closure binding `tensors` (kept alive)."""
        import torch

        args = [
            PtrArg(tensors[i].data_ptr())
            if kind == "ptr"
            else _scalar_arg(kind, i)
            for kind, i in self.plan
        ]
        keep, argp = _marshal(args)
        cuda, func, grid, block = self._cuda, self.func, self.grid, self.block
        shared = self.shared
        addr = int(ctypes.addressof(argp))
        stream = int(torch.cuda.current_stream().cuda_stream)

        def go(_keep=keep, _argp=argp):  # keep the marshaled buffer alive
            cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2],
                                block[0], block[1], block[2],
                                shared, stream, addr, 0)

        return go


class CompactKernel:
    """A compiled MULTI-LAUNCH stream-compaction kernel (C8 dynamic-shape output).

    ``-convert-intent-compact-to-gpu`` lowers ``intent.nonzero`` /
    ``masked_select`` / ``masked_scatter`` to THREE ``gpu.launch``es (count, scan,
    scatter) that share a device scratch ``memref<[B]xi64>`` (a ``memref.alloc``
    that lowers to a host ``malloc``, passed to the count/scan/scatter kernels as a
    bare pointer). This driver loads the three outlined cubins, allocates the
    scratch buffer, and launches the three in order on the current stream (no sync
    between — stream order is a correct barrier).

    ``thunk(tensors)`` binds the func-signature tensors (inputs + destination
    buffers, same order as ``RowReductionKernel``) and returns a NO-SYNC closure.
    """

    def __init__(self, name: str, block_threads: int, mlir_text: str):
        import re

        rr = _run(
            [INTENT_OPT,
             f"-convert-intent-compact-to-gpu=block-threads={block_threads}"],
            stdin=mlir_text,
        )
        kernels = lower_memref_module_to_kernels(rr)
        by_binary = {k.binary_name: k for k in kernels}
        _n, ftext = _host_func_text(kernels[0].binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)  # ptr ssa -> input index
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)

        class _Shim:
            constants = consts

        # Discover the scratch malloc buffer(s): a bare `%r = llvm.call @malloc(%s)`
        # whose size operand traces to a `getelementptr %p[N] ..., <elem>`.
        defs: dict[str, str] = {}
        for m in re.finditer(r"(%\w+)\s*=\s*([^\n]+)", ftext):
            defs.setdefault(m.group(1), m.group(2).strip())

        def _trace_size(ssa: str):
            cur, seen = ssa, set()
            while cur in defs and cur not in seen:
                seen.add(cur)
                rhs = defs[cur]
                gm = re.match(
                    r"llvm\.getelementptr\s+%\w+\[(\d+)\][^,]*,\s*([A-Za-z0-9_]+)",
                    rhs)
                if gm:
                    return int(gm.group(1)), gm.group(2)
                pm = re.match(r"llvm\.ptrtoint\s+(%\w+)", rhs)
                if pm:
                    cur = pm.group(1)
                    continue
                break
            return None

        import torch

        _dtype = {
            "i64": torch.int64,
            "index": torch.int64,
            "i32": torch.int32,
            "f32": torch.float32,
        }
        self._scratch: dict[str, "torch.Tensor"] = {}
        for m in re.finditer(r"(%\w+)\s*=\s*llvm\.call\s+@malloc\((%\w+)\)", ftext):
            res, size_ssa = m.group(1), m.group(2)
            info = _trace_size(size_ssa)
            if info is None:
                raise RuntimeError(
                    f"compact: cannot size scratch malloc {res}")
            n_elems, elem = info
            self._scratch[res] = torch.empty(
                (n_elems,), dtype=_dtype.get(elem, torch.int64), device="cuda")

        # Build one launch plan per site (in stream order).
        self._sites = []
        for site in sites:
            ck = by_binary.get(site.binary_module)
            if ck is None:
                raise RuntimeError(
                    f"compact: no cubin for @{site.binary_module}")
            func = load_module(ck.cubin).function(site.func_symbol)
            grid = tuple(_resolve_const_index(_Shim, s) for s in site.grid_ssa)
            block = tuple(_resolve_const_index(_Shim, s) for s in site.block_ssa)
            shared = (int(consts[site.dynamic_shared_ssa][0])
                      if site.dynamic_shared_ssa is not None else 0)
            plan = []
            for op in site.args:
                if op.ty.startswith("!llvm.ptr"):
                    if op.ssa in aligned:
                        plan.append(("in", aligned[op.ssa]))
                    elif op.ssa in self._scratch:
                        plan.append(("scratch", op.ssa))
                    else:
                        raise RuntimeError(
                            f"compact: ptr operand {op.ssa} is neither input "
                            f"nor scratch")
                else:
                    _append_scalar_plan(
                        plan, op.ty, consts, op.ssa, context="compact"
                    )
            self._sites.append((func, grid, block, shared, plan))
        self._cuda = _driver()

    def thunk(self, tensors):
        """A zero-arg NO-SYNC closure launching all three phases in stream order.

        The current stream is read INSIDE ``go`` (not baked here) so a CUDA-graph
        capture on ``do_bench_cudagraph``'s capture stream records all three
        launches — baking it would send them to the default stream, uncaptured.
        """
        cuda = self._cuda
        packed = []
        for func, grid, block, shared, plan in self._sites:
            args = []
            for kind, v in plan:
                if kind == "in":
                    args.append(PtrArg(tensors[v].data_ptr()))
                elif kind == "scratch":
                    args.append(PtrArg(self._scratch[v].data_ptr()))
                else:
                    args.append(_scalar_arg(kind, v))
            keep, argp = _marshal(args)
            packed.append((func, grid, block, shared, keep, argp,
                           int(ctypes.addressof(argp))))

        def go(_p=packed):  # keep marshaled buffers alive
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
            for func, grid, block, shared, _keep, _argp, addr in _p:
                cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2],
                                    block[0], block[1], block[2],
                                    shared, stream, addr, 0)

        return go


class MatmulKernel:
    """A compiled tensor-core matmul kernel + its resolved bare-ptr launch plan.

    Drives ``-convert-intent-contraction-to-gpu`` on a raw intent-MLIR module
    (one ``intent.contraction{"mk,nk->mn"}``, A/W f16 or bf16 -> Y f32) and
    compiles the resulting tiled-K tensor-core ``gpu.launch`` to an sm_120a
    cubin. Unlike the
    row-reduction pass this pass has no ``block-threads`` option (the WMMA tile
    geometry is fixed: BM=BN=64, BK=16, 4 warps), and it needs ``scf_to_cf=True``
    so the gpu->nvvm pipeline can legalize the ``scf.for`` carrying
    ``!gpu.mma_matrix`` iter_args (the K-loop accumulator fragments).

    ``thunk(tensors)`` returns a zero-arg NO-SYNC launch closure ready for
    ``do_bench``; ``tensors`` are [A, W, Y] (matching 16-bit A/W storage and an
    f32 Y, all contiguous CUDA).
    """

    def __init__(self, mlir_text: str, num_stages: int | None = None,
                 block_m: int | None = None, block_n: int | None = None,
                 mma: bool = False, warp_spec: bool = False,
                 num_producer_warps: int | None = None,
                 block_k: int | None = None, smem_pad: int | None = None):
        # mma=True selects the nvgpu.ldmatrix + nvgpu.mma.sync path
        # (-convert-intent-contraction-to-gpu-mma); warp_spec=True selects the
        # M7g warp-specialized (producer/consumer) mma.sync path
        # (-convert-intent-contraction-to-gpu-mma-ws). The default WMMA
        # gpu.subgroup_mma_* path (-convert-intent-contraction-to-gpu) is kept
        # 100% intact and remains the default.
        opts = []
        if num_stages is not None:
            opts.append(f"num-stages={num_stages}")
        if block_m is not None:
            opts.append(f"block-m={block_m}")
        if block_n is not None:
            opts.append(f"block-n={block_n}")
        if block_k is not None:
            # block-k (BK staging step) is only a knob on the nvgpu.mma.sync path.
            if not mma:
                raise ValueError("block_k is only supported with mma=True")
            opts.append(f"block-k={block_k}")
        if smem_pad is not None:
            if not mma:
                raise ValueError("smem_pad is only supported with mma=True")
            opts.append(f"smem-pad-k={smem_pad}")
        if warp_spec and num_producer_warps is not None:
            opts.append(f"num-producer-warps={num_producer_warps}")
        if warp_spec:
            base = "-convert-intent-contraction-to-gpu-mma-ws"
        else:
            base = ("-convert-intent-contraction-to-gpu-mma" if mma
                    else "-convert-intent-contraction-to-gpu")
        pass_opt = base
        if opts:
            pass_opt += "=" + " ".join(opts)
        rr = _run([INTENT_OPT, pass_opt], stdin=mlir_text)
        kernels = lower_memref_module_to_kernels(rr, scf_to_cf=True)
        if len(kernels) != 1:
            raise RuntimeError(
                f"matmul: expected one kernel, got {len(kernels)}")
        ck = kernels[0]
        self.binary_mlir = ck.binary_mlir
        self.func = load_module(ck.cubin).function(ck.kernel_name)

        _n, ftext = _host_func_text(ck.binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)
        if len(sites) != 1:
            raise RuntimeError(
                f"matmul: expected one gpu.launch_func, got {len(sites)}")
        site = sites[0]

        class _Shim:
            constants = consts

        self.grid = tuple(_resolve_const_index(_Shim, s) for s in site.grid_ssa)
        self.block = tuple(_resolve_const_index(_Shim, s) for s in site.block_ssa)
        self.shared = 0
        if site.dynamic_shared_ssa is not None:
            self.shared = int(consts[site.dynamic_shared_ssa][0])
            cuda = _driver()
            attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            cuda.cuFuncSetAttribute(self.func, attr, self.shared)
        # M6: prefer the pass-emitted structured launch plan; assert == regex.
        plan = single_launch_config(rr)
        if plan is not None:
            plan.assert_matches(self.grid, self.block, self.shared, where="(matmul)")
            self.grid, self.block, self.shared = plan.grid, plan.block, plan.smem
        self.plan = []
        for op in site.args:
            if op.ty.startswith("!llvm.ptr"):
                self.plan.append(("ptr", aligned[op.ssa]))
            else:
                _append_scalar_plan(
                    self.plan, op.ty, consts, op.ssa, context="matmul"
                )
        self._cuda = _driver()

    def thunk(self, tensors):
        """A zero-arg NO-SYNC launch closure binding `tensors` (kept alive)."""
        import torch

        args = [
            PtrArg(tensors[i].data_ptr())
            if kind == "ptr"
            else _scalar_arg(kind, i)
            for kind, i in self.plan
        ]
        keep, argp = _marshal(args)
        cuda, func, grid, block = self._cuda, self.func, self.grid, self.block
        shared = self.shared
        addr = int(ctypes.addressof(argp))
        stream = int(torch.cuda.current_stream().cuda_stream)

        def go(_keep=keep, _argp=argp):  # keep the marshaled buffer alive
            cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2],
                                block[0], block[1], block[2],
                                shared, stream, addr, 0)

        return go


class ConvMMAKernel:
    """A compiled implicit-GEMM tensor-core conv2d kernel + its launch plan.

    Drives ``-convert-intent-conv-to-gpu-mma`` on a raw intent-MLIR module (one
    forward conv2d ``intent.conv``) and compiles the resulting nvgpu.ldmatrix +
    nvgpu.mma.sync (m16n8k16) implicit-GEMM ``gpu.launch`` to an sm_120a cubin.
    Same device pipeline as ``MatmulKernel`` (``scf_to_cf=True`` so the
    ``scf.for`` carrying the ``vector<2x2xf32>`` accumulator iter_args lowers),
    and the same bare-ptr launch-plan resolution.

    ``thunk(tensors)`` returns a zero-arg NO-SYNC launch closure; ``tensors`` are
    the destination-passing args ``[input, weight, (bias,) output]`` (all f32
    CUDA tensors; the pass casts to f16 internally, fp16 tensor-core input mode).
    """

    def __init__(self, mlir_text: str, block_m: int | None = None,
                 block_n: int | None = None, tf32: bool = False):
        opts = []
        if block_m is not None:
            opts.append(f"block-m={block_m}")
        if block_n is not None:
            opts.append(f"block-n={block_n}")
        if tf32:
            opts.append("tf32=true")
        pass_opt = "-convert-intent-conv-to-gpu-mma"
        if opts:
            pass_opt += "=" + " ".join(opts)
        rr = _run([INTENT_OPT, pass_opt], stdin=mlir_text)
        kernels = lower_memref_module_to_kernels(rr, scf_to_cf=True)
        if len(kernels) != 1:
            raise RuntimeError(f"conv-mma: expected one kernel, got {len(kernels)}")
        ck = kernels[0]
        self.func = load_module(ck.cubin).function(ck.kernel_name)

        _n, ftext = _host_func_text(ck.binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)
        if len(sites) != 1:
            raise RuntimeError(
                f"conv-mma: expected one gpu.launch_func, got {len(sites)}")
        site = sites[0]

        class _Shim:
            constants = consts

        self.grid = tuple(_resolve_const_index(_Shim, s) for s in site.grid_ssa)
        self.block = tuple(_resolve_const_index(_Shim, s) for s in site.block_ssa)
        self.shared = 0
        if site.dynamic_shared_ssa is not None:
            self.shared = int(consts[site.dynamic_shared_ssa][0])
            cuda = _driver()
            attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            cuda.cuFuncSetAttribute(self.func, attr, self.shared)
        # M6: prefer the pass-emitted structured launch plan; assert == regex.
        plan = single_launch_config(rr)
        if plan is not None:
            plan.assert_matches(self.grid, self.block, self.shared, where="(conv-mma)")
            self.grid, self.block, self.shared = plan.grid, plan.block, plan.smem
        self.plan = []
        for op in site.args:
            if op.ty.startswith("!llvm.ptr"):
                self.plan.append(("ptr", aligned[op.ssa]))
            else:
                _append_scalar_plan(
                    self.plan, op.ty, consts, op.ssa, context="conv-mma"
                )
        self._cuda = _driver()

    def thunk(self, tensors):
        """A zero-arg NO-SYNC launch closure binding `tensors` (kept alive)."""
        import torch

        args = [
            PtrArg(tensors[i].data_ptr())
            if kind == "ptr"
            else _scalar_arg(kind, i)
            for kind, i in self.plan
        ]
        keep, argp = _marshal(args)
        cuda, func, grid, block = self._cuda, self.func, self.grid, self.block
        shared = self.shared
        addr = int(ctypes.addressof(argp))
        stream = int(torch.cuda.current_stream().cuda_stream)

        def go(_keep=keep, _argp=argp):  # keep the marshaled buffer alive
            cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2],
                                block[0], block[1], block[2],
                                shared, stream, addr, 0)

        return go
