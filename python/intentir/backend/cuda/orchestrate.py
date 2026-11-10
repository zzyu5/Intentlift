"""Generic multi-kernel launcher for the intent IR back-end (M9 orchestration).

Today's launches are hand-specified per fixture: ``test_backend_execute.py`` types
the grid/block/args of each kernel by hand. This module makes the *whole* lowered
op-graph self-orchestrating by parsing the lowered host module that
``compile_all_kernels`` already returns, then executing every ``gpu.launch_func``
in order with the right grid/block/args resolved from the IR.

Public entry point::

    run_program(intent_mlir_path, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

What the lowered host module actually looks like
------------------------------------------------
The task brief describes the *front-pass* module (memref + arith + linalg). But
``compile_all_kernels`` runs the full device pipeline, so ``binary_mlir`` is the
**fully LLVM-dialect lowered** module. Concretely (see ``lower_to_binary_mlir``)::

    llvm.func @add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, ... ) -> !llvm.struct<...> {
      ...
      %10 = llvm.call @malloc(%9) : (i64) -> !llvm.ptr        # an output/intermediate buffer
      ...
      %16 = llvm.inttoptr %15 : i64 to !llvm.ptr              # its aligned pointer
      ...
      gpu.launch_func @add_kernel::@add_kernel
          blocks in (%4, %3, %5) threads in (%5, %5, %5) : i64
          args(%5 : i64, %6 : i64, %arg1 : !llvm.ptr, %arg8 : !llvm.ptr, %16 : !llvm.ptr)
      llvm.return %23 : !llvm.struct<...>                     # descriptor whose aligned ptr is %16
    }
    gpu.binary @add_kernel [#gpu.object<#nvvm.target<...>, bin = "...">]

So we parse, against this LLVM-dialect form:

* the **host llvm.func** signature -> input tensor names (from the *intent*
  source) and the flattened memref-descriptor block-arg layout (per memref:
  allocated-ptr, *aligned-ptr*, offset, sizes..., strides...; the launcher
  resolves both pointer fields and supplies the complete descriptor scalar
  fields to kernels);
* every ``llvm.mlir.constant(V : ty) : i64`` -> a literal value (grid/block/scalar);
* every ``llvm.call @malloc`` -> one buffer, identified in launch args by its
  ``llvm.inttoptr`` aligned-pointer result, shape from the descriptor's
  ``insertvalue ...[3, i]`` size fields;
* every ``gpu.launch_func`` -> a kernel invocation, classifying each ``args(...)``
  operand by its explicit type annotation (``i64``/``i32`` scalar-by-value,
  ``f32`` 4-byte-float-by-value, ``!llvm.ptr`` a memref pointer that is either an
  input block-arg or a malloc buffer);
* ``llvm.return`` traced through the ``insertvalue ...[1]`` (aligned-ptr) chain ->
  the buffer that is the program output.

Kernel<->cubin matching: in a multi-kernel program the *gpu.func* name can repeat
(matmul: both launches target ``@mm_kernel``); only the ``gpu.binary`` *module*
name (``mm_kernel`` vs ``mm_kernel_0``) is unique. We match each launch site by the
token before ``::`` against ``CompiledKernel.binary_name``.

Dynamic-shape output (C8 nonzero / masked_select / Ragged member): the kernel's
output extent is data-dependent (unknown at compile time), but this parser only
ever launches a concrete host-allocated descriptor. The runtime extent is
resolved on the HOST before the launch, and the complete flattened descriptor
(allocated pointer, aligned pointer, offset, sizes, and strides) is passed to the
device. Two flows, both compatible with the static-buffer machinery here:
  * two-step: reduce the mask to ``nnz`` on the host (``mask.sum()``), allocate the
    exact ``[nnz, ndim]`` / ``[nnz]`` output, then run the single compaction launch;
  * max-buffer + count: allocate a ``[Ntotal, *]`` upper-bound buffer, launch the
    compaction (the last thread writes the total ``nnz`` into a ``[1] i64`` count
    buffer), read ``count`` on the host and slice ``out[:nnz]``.
Either way the kernel writes only its valid slots and the destination leading dim
handed to ``run_program`` is concrete before launch. Ragged member uses the same
descriptor machinery; it is not a bare-pointer exception.

f32 scalar args: ``launch.py``'s ``_marshal`` only handles i32/i64/pointer; a float
would silently fall through to ``c_void_p`` (a null-ish pointer) and corrupt the
result with no error. The matmul fixture's fill kernel takes an ``f32`` value, so
this module reimplements the marshal + ``cuLaunchKernel`` locally with a
``ctypes.c_float`` branch (we may NOT edit launch.py). We still reuse launch.py's
``load_module`` / ``ensure_context`` (loading is unaffected).
"""

from __future__ import annotations

import ctypes
import numpy as np
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from intentir.backend.cuda.compile import CompiledKernel, compile_all_kernels
from intentir.backend.cuda.launch import ensure_context, load_module
from intentir.ir.value_dimension import (
    RaggedRuntimeError,
    ragged_member_length,
    validate_ragged_runtime_carrier,
)


# MLIR permits dotted/dollar-bearing symbols and also prints unnamed SSA values
# as numeric names (``%0``).  All host/source parsers in this module use these
# grammar-level patterns so a name accepted by MLIR cannot become an
# orchestration-only parse failure.
_MLIR_SSA_RE = r"%(?:[0-9]+|[A-Za-z_][A-Za-z0-9_.$]*)"
_MLIR_SYMBOL_RE = r"[A-Za-z_][A-Za-z0-9_.$]*"

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


# --------------------------------------------------------------------------- #
# Scalar kernel-argument wrappers (orchestrate-local; launch.py is untouched). #
# --------------------------------------------------------------------------- #
class _PtrArg(int):
    """Kernel arg: a raw device pointer (u64)."""


class _I64Arg(int):
    """Kernel arg passed by value as a 64-bit index/integer."""


class _I32Arg(int):
    """Kernel arg passed by value as a 32-bit integer."""


class _U8Arg(int):
    """Kernel arg passed by value as an 8-bit unsigned integer."""


class _U16Arg(int):
    """Kernel arg passed by value as a 16-bit unsigned integer."""


class _U32Arg(int):
    """Kernel arg passed by value as a 32-bit unsigned integer."""


class _F16Arg(int):
    """Kernel arg passed as one raw IEEE-754 binary16 bit pattern."""


class _BF16Arg(int):
    """Kernel arg passed as one raw bfloat16 bit pattern."""


class _F32Arg(float):
    """Kernel arg passed by value as a 32-bit float (4 raw bytes)."""


class _F64Arg(float):
    """Kernel arg passed by value as a 64-bit float (8 raw bytes)."""


def _parse_f32(s: str) -> float:
    """Parse an MLIR f32 literal: decimal (``1.0``, ``0.0e+00``) or the raw
    IEEE-754 hex form MLIR prints for special values (``0xFF800000`` = -inf)."""
    s = s.strip()
    if s.lower().startswith("0x"):
        import struct
        return struct.unpack(">f", bytes.fromhex(s[2:].rjust(8, "0")))[0]
    return float(s)


def _parse_f16_bits(s: str) -> int:
    """Parse an MLIR f16 literal into the exact 16-bit kernel ABI payload."""
    import struct

    s = s.strip()
    if s.lower().startswith("0x"):
        return int(s, 16) & 0xFFFF
    return int.from_bytes(struct.pack(">e", float(s)), "big")


def _parse_bf16_bits(s: str) -> int:
    """Parse an MLIR bf16 literal into its round-to-nearest-even bit pattern."""
    import struct

    s = s.strip()
    if s.lower().startswith("0x"):
        return int(s, 16) & 0xFFFF
    bits = int.from_bytes(struct.pack(">f", float(s)), "big")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return rounded >> 16


def _parse_f64(s: str) -> float:
    """Parse an MLIR f64 decimal or raw 64-bit IEEE-754 hex literal."""
    s = s.strip()
    if s.lower().startswith("0x"):
        import struct
        return struct.unpack(">d", bytes.fromhex(s[2:].rjust(16, "0")))[0]
    return float(s)


def _marshal(args):
    """Build the ctypes arg-pointer array for ``cuLaunchKernel``.

    Mirrors launch.py's ``_marshal`` but adds floating scalar ABI branches.
    """
    cvals = []
    for a in args:
        if isinstance(a, _F64Arg):
            cvals.append(ctypes.c_double(float(a)))
        elif isinstance(a, _F32Arg):
            cvals.append(ctypes.c_float(float(a)))
        elif isinstance(a, (_F16Arg, _BF16Arg)):
            cvals.append(ctypes.c_uint16(int(a)))
        elif isinstance(a, _I32Arg):
            cvals.append(ctypes.c_int32(int(a)))
        elif isinstance(a, _U8Arg):
            cvals.append(ctypes.c_uint8(int(a)))
        elif isinstance(a, _U16Arg):
            cvals.append(ctypes.c_uint16(int(a)))
        elif isinstance(a, _U32Arg):
            cvals.append(ctypes.c_uint32(int(a)))
        elif isinstance(a, _I64Arg):
            cvals.append(ctypes.c_int64(int(a)))
        else:  # _PtrArg or plain int -> device pointer
            cvals.append(ctypes.c_void_p(int(a)))
    arg_ptrs = (ctypes.c_void_p * len(cvals))()
    for i, cv in enumerate(cvals):
        arg_ptrs[i] = ctypes.cast(ctypes.byref(cv), ctypes.c_void_p)
    return cvals, arg_ptrs  # keep cvals alive


def _launch(func, grid, block, args, shared: int = 0) -> None:
    """Launch + synchronize on torch's current stream (local copy with f32)."""
    import torch

    from intentir.backend.cuda.launch import _driver

    cuda = _driver()
    _keep, arg_ptrs = _marshal(args)
    stream = int(torch.cuda.current_stream().cuda_stream)
    ret = cuda.cuLaunchKernel(
        func, grid[0], grid[1], grid[2], block[0], block[1], block[2],
        shared, stream, int(ctypes.addressof(arg_ptrs)), 0,
    )
    err = ret[0] if isinstance(ret, tuple) else ret
    if int(err) != 0:
        raise RuntimeError(f"cuLaunchKernel: CUDA driver error {int(err)}")
    torch.cuda.synchronize()


def _set_dynamic_shared_memory(func, shared: int) -> bool:
    """Opt a function into its persisted dynamic shared-memory requirement."""

    if shared <= 0:
        return False
    from intentir.backend.cuda.launch import _driver

    cuda = _driver()
    attribute = (
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    )
    result = cuda.cuFuncSetAttribute(func, attribute, int(shared))
    error = result[0] if isinstance(result, tuple) else result
    if int(error) != 0:
        raise RuntimeError(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) failed with "
            f"CUDA driver error {int(error)}"
        )
    return True


# --------------------------------------------------------------------------- #
# Prepared (Triton-style) launcher (R-FLASH-1).                                #
# --------------------------------------------------------------------------- #
class PreparedLauncher:
    """Pre-marshalled kernel launcher: builds the ctypes argument array ONCE and
    updates only the pointer/scalar slots in place per call, then invokes
    ``cuLaunchKernel`` directly with NO per-call Python arg rebuild and NO
    synchronize (the caller / ``do_bench`` owns timing + sync).

    ``_launch`` above rebuilds the ctypes value list, casts a fresh pointer
    array, and blocks on ``torch.cuda.synchronize()`` every call -- fine for
    correctness harnesses but it inflates sub-30us kernels (the small-shape flash
    gap in R4). Triton's launcher caches its arg buffer and does not sync per
    call; this mirrors that prepared launch closure so flash-fwd reaches parity
    at ALL shapes, not just
    the GPU-bound large ones.

    ``plan`` is the same launch-arg plan the flash kernels already build:
    ``("ptr", input_index) | ("scale",) | ("f32", value) | ("i64", value)``.
    ``launch(inputs, scale)`` refreshes the pointer slots from
    ``inputs[i].data_ptr()`` and any dynamic scale slot, then launches on
    torch's current stream.  Fixed ``f32`` entries remain immutable.
    """

    def __init__(self, func, grid, block, plan, shared: int = 0):
        from intentir.backend.cuda.launch import _driver

        self.func = func
        self.grid = tuple(int(x) for x in grid)
        self.block = tuple(int(x) for x in block)
        self.shared = int(shared)
        self._cuda = _driver()

        self._cells: list = []
        self._ptr_slots: list[tuple[int, int]] = []  # (cell_index, input_index)
        self._scale_slots: list[int] = []
        self._input_scalar_slots: list[tuple[int, int, str]] = []
        for entry in plan:
            kind = entry[0]
            if kind == "ptr":
                self._ptr_slots.append((len(self._cells), entry[1]))
                self._cells.append(ctypes.c_void_p(0))
            elif kind == "scale":
                self._scale_slots.append(len(self._cells))
                self._cells.append(ctypes.c_float(0.0))
            elif kind == "f32":
                self._cells.append(ctypes.c_float(float(entry[1])))
            elif kind == "f64":
                self._cells.append(ctypes.c_double(float(entry[1])))
            elif kind == "f16":
                self._cells.append(ctypes.c_uint16(int(entry[1])))
            elif kind == "bf16":
                self._cells.append(ctypes.c_uint16(int(entry[1])))
            elif kind == "i64":
                self._cells.append(ctypes.c_int64(int(entry[1])))
            elif kind == "i32":
                self._cells.append(ctypes.c_int32(int(entry[1])))
            elif kind == "u8":
                self._cells.append(ctypes.c_uint8(int(entry[1])))
            elif kind == "u16":
                self._cells.append(ctypes.c_uint16(int(entry[1])))
            elif kind == "u32":
                self._cells.append(ctypes.c_uint32(int(entry[1])))
            elif kind.startswith("input_"):
                scalar_kind = kind.removeprefix("input_")
                cells = {
                    "i32": ctypes.c_int32,
                    "i64": ctypes.c_int64,
                    "ui8": ctypes.c_uint8,
                    "ui16": ctypes.c_uint16,
                    "ui32": ctypes.c_uint32,
                    "f16": ctypes.c_uint16,
                    "bf16": ctypes.c_uint16,
                    "f32": ctypes.c_float,
                    "f64": ctypes.c_double,
                }
                if scalar_kind not in cells:
                    raise RuntimeError(
                        f"PreparedLauncher: unsupported input scalar {kind!r}"
                    )
                self._input_scalar_slots.append(
                    (len(self._cells), int(entry[1]), scalar_kind)
                )
                self._cells.append(cells[scalar_kind](0))
            else:
                raise RuntimeError(f"PreparedLauncher: unhandled plan entry {entry!r}")
        # Build the pointer array ONCE; the cells' addresses are stable.
        self._arg_ptrs = (ctypes.c_void_p * len(self._cells))()
        for i, cv in enumerate(self._cells):
            self._arg_ptrs[i] = ctypes.cast(ctypes.byref(cv), ctypes.c_void_p)
        self._arg_addr = int(ctypes.addressof(self._arg_ptrs))

    def bind_input_scalars(self, inputs) -> None:
        """Snapshot dynamic scalar ABI values outside the timed launch closure."""

        for cell_i, in_i, kind in self._input_scalar_slots:
            value = inputs[in_i]
            if not hasattr(value, "numel") or int(value.numel()) != 1:
                raise ValueError(
                    "PreparedLauncher: dynamic scalar input must contain one value"
                )
            if kind in {"f16", "bf16"}:
                import torch

                expected = torch.float16 if kind == "f16" else torch.bfloat16
                if value.dtype != expected:
                    raise ValueError(
                        f"PreparedLauncher: input {in_i} must have dtype {expected}"
                    )
                scalar = int(value.reshape(()).view(torch.uint16).item())
            else:
                scalar = value.item()
            self._cells[cell_i].value = scalar

    def launch(self, inputs, scale: float | None = None) -> None:
        import torch

        for cell_i, in_i in self._ptr_slots:
            self._cells[cell_i].value = int(inputs[in_i].data_ptr())
        if scale is not None:
            fs = float(scale)
            for cell_i in self._scale_slots:
                self._cells[cell_i].value = fs
        stream = int(torch.cuda.current_stream().cuda_stream)
        ret = self._cuda.cuLaunchKernel(
            self.func, self.grid[0], self.grid[1], self.grid[2],
            self.block[0], self.block[1], self.block[2],
            self.shared, stream, self._arg_addr, 0,
        )
        err = ret[0] if isinstance(ret, tuple) else ret
        if int(err) != 0:
            raise RuntimeError(f"cuLaunchKernel: CUDA driver error {int(err)}")


@dataclass(frozen=True)
class PreparedGraphRecord:
    """Allocation-free timed-launch counts for one compiled CUDA graph."""

    launch_count: int
    kernel_count: int
    intermediate_buffer_count: int
    output_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "launch_count": self.launch_count,
            "kernel_count": self.kernel_count,
            "intermediate_buffer_count": self.intermediate_buffer_count,
            "output_count": self.output_count,
        }


class PreparedGraphProgram:
    """Persistent tensors plus prepared no-sync launchers for a whole MLIR graph."""

    def __init__(self, sites, outputs, record: PreparedGraphRecord):
        self._sites = tuple(sites)
        self.outputs = dict(outputs)
        self.record = record

    def thunk(self):
        """Return an allocation-free, synchronization-free whole-graph closure."""

        sites = self._sites

        def launch() -> None:
            for launcher, tensors in sites:
                launcher.launch(tensors)

        return launch


# --------------------------------------------------------------------------- #
# MLIR type -> torch dtype.                                                    #
# --------------------------------------------------------------------------- #
def _torch_dtype(elem: str):
    import torch

    table = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f64": torch.float64,
        # ``index`` is a logical dtype; CUDA host/device launch storage uses
        # the same signed 64-bit carrier as the MLIR emitter and CPU ABI.
        "index": torch.int64,
        "i64": torch.int64,
        "i32": torch.int32,
        "i16": torch.int16,
        "i8": torch.int8,
        "i1": torch.bool,
        "ui8": torch.uint8,
        "ui16": torch.uint16,
        "ui32": torch.uint32,
        "u8": torch.uint8,
        "u16": torch.uint16,
        "u32": torch.uint32,
        "uint8": torch.uint8,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
    }
    if elem not in table:
        raise NotImplementedError(
            f"orchestrate: unhandled element type {elem!r} (extend _torch_dtype)"
        )
    return table[elem]


# --------------------------------------------------------------------------- #
# Intent-source signature parsing (input names / shapes / dtypes).            #
# --------------------------------------------------------------------------- #
@dataclass
class _IntentInput:
    name: str
    shape: tuple[int, ...]
    dtype_str: str


@dataclass(frozen=True)
class _IntentConstant:
    """One integer ``arith.constant`` tensor in the authored Intent module.

    Ragged carriers may be deterministic constants rather than function
    arguments.  Keep their exact static shape, dtype, and element sequence so
    host-side result-shape resolution uses the same carrier contract as the
    C++ lowering and the structured CPU reference.
    """

    shape: tuple[int, ...]
    dtype_str: str
    values: tuple[int, ...]


def _parse_intent_signature(intent_src: str) -> list[_IntentInput]:
    """Parse ``func.func @name(%a: tensor<8x16xf32>, ...) -> ...`` from intent IR.

    Returns inputs in positional order. Used only to recover *names* (so callers
    can key the ``inputs`` dict by name) and to sanity-check shapes/dtypes.
    """
    m = re.search(
        rf"func\.func\s+@{_MLIR_SYMBOL_RE}\s*\(([^)]*)\)", intent_src
    )
    if not m:
        raise ValueError("orchestrate: could not find a func.func signature in intent IR")
    arglist = m.group(1).strip()
    inputs: list[_IntentInput] = []
    if not arglist:
        return inputs
    # Split on top-level commas (tensor types here carry no nested commas).
    for piece in arglist.split(","):
        piece = piece.strip()
        if not piece:
            continue
        am = re.match(
            rf"({_MLIR_SSA_RE})\s*:\s*tensor<([0-9x]*)x?([a-z0-9]+)>",
            piece,
        )
        if not am:
            am2 = re.match(
                rf"({_MLIR_SSA_RE})\s*:\s*tensor<([a-z0-9]+)>", piece
            )
            if am2:  # 0-d tensor
                inputs.append(_IntentInput(am2.group(1)[1:], (), am2.group(2)))
                continue
            raise NotImplementedError(
                f"orchestrate: unhandled intent arg type in {piece!r}"
            )
        name = am.group(1)[1:]
        dims_str = am.group(2)
        elem = am.group(3)
        dims = tuple(int(d) for d in dims_str.split("x") if d != "")
        inputs.append(_IntentInput(name, dims, elem))
    return inputs


def _parse_intent_results(
    intent_src: str,
) -> list[tuple[tuple[int | None, ...], str]]:
    """Recover declared result shapes/dtypes from one emitted ``func.func``.

    Generic bufferization may return a tensor through a loose flat malloc even
    when its source ABI is keepdims (for example ``tensor<1x1x128xf32>``).  The
    host LLVM descriptor is then insufficient to recover the semantic shape, so
    output collection must consult the same P4 signature that was compiled.
    """

    match = re.search(
        rf"func\.func\s+@{_MLIR_SYMBOL_RE}\s*\([^)]*\)\s*->\s*"
        r"(\([^)]*\)|tensor<[^>]+>)",
        intent_src,
    )
    if not match:
        return []
    result_text = match.group(1).strip()
    pieces = (
        _split_top_level(result_text[1:-1])
        if result_text.startswith("(")
        else [result_text]
    )
    results: list[tuple[tuple[int | None, ...], str]] = []
    for piece in pieces:
        tensor_match = re.fullmatch(
            r"tensor<(?:(?P<dims>(?:\d+|\?)(?:x(?:\d+|\?))*)x)?"
            r"(?P<elem>[A-Za-z0-9]+)>",
            piece.strip(),
        )
        if tensor_match is None:
            raise NotImplementedError(
                f"orchestrate: unhandled intent result type {piece.strip()!r}"
            )
        dims_text = tensor_match.group("dims")
        dims = (
            tuple(None if dim == "?" else int(dim) for dim in dims_text.split("x"))
            if dims_text
            else ()
        )
        results.append((dims, tensor_match.group("elem")))
    return results


def _parse_intent_constant_tensors(
    intent_src: str,
) -> dict[str, _IntentConstant]:
    """Recover static integer tensor constants from the authored Intent IR.

    ``emit_mlir`` uses ``dense<[...]>`` for explicit vectors and ``dense<V>``
    for splats.  This parser is intentionally limited to the closed integer
    carrier types used by Ragged; a different constant kind must not silently
    become a runtime carrier.
    """

    constants: dict[str, _IntentConstant] = {}
    pattern = re.compile(
        rf"({_MLIR_SSA_RE})\s*=\s*arith\.constant\s+dense<"
        r"(?P<values>[^>]+)>\s*:\s*tensor<"
        r"(?:(?P<dims>\d+(?:x\d+)*)x)?"
        r"(?P<dtype>ui\d+|i\d+|index)>"
    )
    dtype_map = {
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "ui8": np.uint8,
        "ui16": np.uint16,
        "ui32": np.uint32,
        "index": np.int64,
    }
    for match in pattern.finditer(intent_src):
        dtype = match.group("dtype")
        if dtype not in dtype_map:
            continue
        dims_text = match.group("dims")
        shape = tuple(int(dim) for dim in dims_text.split("x")) if dims_text else ()
        raw = match.group("values").strip()
        if raw.startswith("[") and raw.endswith("]"):
            items = [item.strip() for item in raw[1:-1].split(",") if item.strip()]
        else:
            items = [raw]
        try:
            values = tuple(
                1 if item == "true" else 0 if item == "false" else int(item, 0)
                for item in items
            )
        except ValueError as exc:
            raise NotImplementedError(
                "orchestrate: Ragged constant carrier has a non-integer dense value"
            ) from exc
        expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if len(values) == 1 and expected > 1:
            values = values * expected
        if len(values) != expected:
            raise ValueError(
                "orchestrate: Ragged constant carrier dense element count does not "
                f"match its declared shape {shape}"
            )
        # Materialize once here to reject a value that cannot be represented by
        # the authored carrier width, before any CUDA launch is prepared.
        np.asarray(values, dtype=dtype_map[dtype])
        constants[match.group(1)] = _IntentConstant(shape, dtype, values)
    return constants


def _resolve_intent_result_shapes(
    intent_src: str,
    intent_inputs: list[_IntentInput],
    declared: list[tuple[tuple[int | None, ...], str]],
    input_tensor_by_index,
) -> list[tuple[tuple[int, ...], str]]:
    """Resolve the host allocation extent for a direct ragged_member result.

    The CUDA C++ pass deliberately keeps the member result dynamic in the
    source contract but emits a destination-passing output pointer.  Offsets,
    indices, and selector values may be direct input tensors or deterministic
    constant tensors; the exact allocation can be established before the
    launch without putting a tuple or a dynamic memref descriptor in the device
    ABI.
    """

    input_indices = {f"%{item.name}": index for index, item in enumerate(intent_inputs)}
    ragged_producers = {
        result: (offsets, indices)
        for result, offsets, indices in re.findall(
            rf"({_MLIR_SSA_RE})\s*=\s*intent\.ragged\s+"
            rf"({_MLIR_SSA_RE})\s*,\s*({_MLIR_SSA_RE})",
            intent_src,
        )
    }
    member_producers = {
        result: (descriptor, selector)
        for result, descriptor, selector in re.findall(
            rf"({_MLIR_SSA_RE})\s*=\s*intent\.ragged_member\s+"
            rf"({_MLIR_SSA_RE})\s*,\s*({_MLIR_SSA_RE})",
            intent_src,
        )
    }
    constant_tensors = _parse_intent_constant_tensors(intent_src)

    def input_tensor(name: str, *, role: str):
        if name not in input_indices:
            constant = constant_tensors.get(name)
            if constant is not None:
                dtype_map = {
                    "i8": np.int8,
                    "i16": np.int16,
                    "i32": np.int32,
                    "i64": np.int64,
                    "ui8": np.uint8,
                    "ui16": np.uint16,
                    "ui32": np.uint32,
                    "index": np.int64,
                }
                return np.asarray(
                    constant.values,
                    dtype=dtype_map[constant.dtype_str],
                ).reshape(constant.shape)
            raise NotImplementedError(
                "CUDA Ragged host ABI requires "
                f"{role} to be a function input or a static arith.constant carrier"
            )
        try:
            return input_tensor_by_index[input_indices[name]]
        except KeyError as exc:
            raise NotImplementedError(
                "CUDA Ragged host ABI could not bind "
                f"{role} input {name}"
            ) from exc

    # Validate every descriptor before inspecting result shapes.  In
    # particular, ragged_outer has a statically ranked scalar result and would
    # otherwise bypass the dynamic-output branch below, allowing CUDA host
    # orchestration to accept a carrier that CPU/Reference reject.
    for descriptor, (offsets_name, indices_name) in ragged_producers.items():
        offsets = input_tensor(offsets_name, role="offsets")
        indices = input_tensor(indices_name, role="indices")
        try:
            validate_ragged_runtime_carrier(
                offsets,
                indices,
                context=f"CUDA {descriptor} Ragged carrier",
            )
        except RaggedRuntimeError as exc:
            raise ValueError(str(exc)) from exc
    member_lengths: dict[str, int] = {}
    for result, (descriptor, selector_name) in member_producers.items():
        carrier = ragged_producers.get(descriptor)
        if carrier is None:
            continue
        offsets_name, indices_name = carrier
        offsets = input_tensor(offsets_name, role="offsets")
        indices = input_tensor(indices_name, role="indices")
        selector = input_tensor(selector_name, role="selector")
        try:
            member_lengths[result] = ragged_member_length(
                offsets,
                indices,
                selector,
                context=f"CUDA {result} Ragged member",
            )
        except RaggedRuntimeError as exc:
            raise ValueError(str(exc)) from exc
    if not any(None in shape for shape, _elem in declared):
        return [(tuple(int(dim) for dim in shape), elem) for shape, elem in declared]
    return_match = re.search(r"\breturn\s+([^:\n]+)\s*:", intent_src)
    returned = (
        [item.strip() for item in return_match.group(1).split(",")]
        if return_match is not None
        else []
    )

    resolved: list[tuple[tuple[int, ...], str]] = []
    for result_index, (shape, elem) in enumerate(declared):
        if None not in shape:
            resolved.append((tuple(int(dim) for dim in shape), elem))
            continue
        if len(shape) != 1 or shape[0] is not None:
            raise NotImplementedError(
                "CUDA dynamic output is only implemented for a rank-1 "
                "ragged_member result"
            )
        if result_index >= len(returned) or returned[result_index] not in member_producers:
            raise NotImplementedError(
                "CUDA dynamic output is not a direct ragged_member return"
            )
        descriptor, _selector = member_producers[returned[result_index]]
        if descriptor not in ragged_producers:
            raise NotImplementedError(
                "CUDA ragged_member output has no local ragged producer"
            )
        try:
            length = member_lengths[returned[result_index]]
        except KeyError as exc:
            raise NotImplementedError(
                "CUDA dynamic output is not a closed Ragged member carrier"
            ) from exc
        resolved.append(((length,), elem))
    return resolved


def _parse_intent_inplace(
    intent_src: str, result_count: int
) -> list[int] | None:
    """Recover the typed result->input alias contract, when present."""

    match = re.search(
        r"intent\.inplace\s*=\s*array<i64:\s*([^>]*)>", intent_src
    )
    if match is None:
        return None
    values = [
        int(piece.strip())
        for piece in match.group(1).split(",")
        if piece.strip()
    ]
    if len(values) != result_count:
        raise ValueError(
            "orchestrate: intent.inplace length does not match declared results"
        )
    if any(value < -1 for value in values):
        raise ValueError("orchestrate: intent.inplace entries must be -1 or input indices")
    return values


# --------------------------------------------------------------------------- #
# Host LLVM-func parsing.                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class _LaunchOperand:
    ssa: str          # e.g. "%5", "%arg1", "%16"
    ty: str           # e.g. "i64", "f32", "!llvm.ptr"


@dataclass
class _LaunchSite:
    binary_module: str            # token before "::"  -> matches CompiledKernel.binary_name
    func_symbol: str              # token after  "::"  -> the gpu.func name
    grid_ssa: tuple[str, str, str]
    block_ssa: tuple[str, str, str]
    args: list[_LaunchOperand]
    dynamic_shared_ssa: str | None = None  # dynamic_shared_memory_size operand ssa


@dataclass(frozen=True)
class _MemRefDescriptor:
    """One flattened ranked-memref descriptor in the host LLVM ABI.

    ``convert-gpu-to-nvvm`` with the bare-pointer convention disabled expands a
    ranked memref to allocated/aligned pointers followed by offset, sizes, and
    strides.  The descriptor is kept as one object here so launch parsing,
    allocation, and output collection cannot disagree about field ownership.
    ``value_index`` is an input index for an input descriptor and a result index
    for a destination descriptor.
    """

    ordinal: int
    role: str
    value_index: int
    rank: int
    allocated_ssa: str
    aligned_ssa: str
    offset_ssa: str
    size_ssas: tuple[str, ...]
    stride_ssas: tuple[str, ...]

    @property
    def scalar_fields(self) -> tuple[tuple[str, int | None, str], ...]:
        fields: list[tuple[str, int | None, str]] = [("offset", None, self.offset_ssa)]
        fields.extend(
            ("size", index, ssa) for index, ssa in enumerate(self.size_ssas)
        )
        fields.extend(
            ("stride", index, ssa) for index, ssa in enumerate(self.stride_ssas)
        )
        return tuple(fields)


@dataclass(frozen=True)
class _ConstGlobal:
    name: str
    shape: tuple[int, ...]
    elem: str
    value: int | float | tuple[int | float, ...]


@dataclass
class _HostModule:
    func_name: str
    sig_args: list[str]                       # ordered block-arg SSA names ("%arg0", ...)
    constants: dict[str, tuple[str, str]]     # ssa -> (raw_value, type)
    aligned_ptr_blockargs: dict[str, int]     # aligned-ptr block-arg ssa -> input index
    pointer_blockargs: dict[str, int]         # allocated/aligned input ptr -> input index
    descriptors: tuple[_MemRefDescriptor, ...]
    descriptor_scalar_fields: dict[str, tuple[int, str, int | None]]
    destination_aligned_ptrs: list[str]       # destination-passing result ptrs
    destination_result_indices: list[int]     # declared result index per destination
    malloc_buffers: dict[str, "_Buffer"]      # normalized aligned-ptr ssa -> buffer
    loose_buffers: dict[str, "_LooseBuffer"]  # normalized ptr ssa -> intermediate buffer
    launches: list[_LaunchSite]
    return_aligned_ptrs: list[str]            # aligned-ptr ssa per result, in order
    body_text: str = field(default="", repr=False)
    const_globals: dict[str, _ConstGlobal] = field(default_factory=dict)
    # llvm.mlir.addressof/zero-GEP ssa -> static scalar or dense splat


@dataclass
class _Buffer:
    aligned_ssa: str
    shape: tuple[int, ...]
    elem: str


@dataclass
class _LooseBuffer:
    """A malloc'd intermediate buffer passed to kernels as the raw aligned
    pointer, never packed into a memref-descriptor struct.

    Such buffers carry no struct shape in the host IR, only an element count
    (from the ``getelementptr`` that computes the malloc byte size). We allocate
    a flat 1-D tensor of ``n_elems`` of ``elem`` dtype — enough storage for the
    kernel's bare-ptr writes; torch's allocator is >=256-byte aligned, which
    satisfies the 64-byte alignment the host computes for the kernel pointer.

    ``aligned_ssa`` is either the legacy ``llvm.inttoptr`` result or, with
    opaque-pointer LLVM lowering, the direct ``malloc !llvm.ptr`` result.
    """

    aligned_ssa: str        # normalized kernel-arg pointer identity
    malloc_ssa: str         # the llvm.call @malloc result
    n_elems: int            # element count (from getelementptr %null[N])
    elem: str               # element type (from the same getelementptr)


def _host_func_text(binary_mlir: str) -> tuple[str, str]:
    """Return (func_name, body-up-to-launches) for the single host ``llvm.func``.

    The host function is the one containing ``gpu.launch_func`` (``@malloc`` decl
    has no body). We slice from its ``{`` to the matching closing of the gpu
    launches region via a simple brace counter, which is robust to the nested
    ``!llvm.struct<...>`` angle brackets (we only count ``{`` / ``}``).
    """
    func_starts = [
        (mm.start(), mm.group(1), mm.end())
        for mm in re.finditer(
            rf"llvm\.func\s+@({_MLIR_SYMBOL_RE})\s*\(", binary_mlir
        )
    ]
    for k, (start, name, end) in enumerate(func_starts):
        # A declaration-only func (e.g. @malloc) has no body brace before the
        # next llvm.func; reject a brace that belongs to a *later* function.
        next_start = func_starts[k + 1][0] if k + 1 < len(func_starts) else len(binary_mlir)
        # Scan every top-level brace region after the signature and return the
        # one that is the *body* (contains gpu.launch_func). A func may carry an
        # `attributes {...}` dict brace BEFORE its body brace (e.g. the M6
        # `intent.launch_plan` structured launch plan); we skip such non-body
        # regions instead of mistaking the first `{` for the body. With no
        # attributes the first region IS the body -> identical to before.
        search = end
        while True:
            open_idx = binary_mlir.find("{", search)
            if open_idx == -1 or open_idx >= next_start:
                break
            depth = 0
            i = open_idx
            close_idx = -1
            while i < len(binary_mlir):
                c = binary_mlir[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        close_idx = i
                        break
                i += 1
            if close_idx == -1:
                break
            body = binary_mlir[open_idx + 1 : close_idx]
            if "gpu.launch_func" in body:
                sig = binary_mlir[start:open_idx]
                return name, sig + "{" + body + "}"
            search = close_idx + 1  # skip this region (attributes dict); try next
    raise ValueError("orchestrate: no host llvm.func containing gpu.launch_func found")


def _parse_signature_blockargs(
    func_text: str,
) -> tuple[str, list[str], list[str], list[tuple[str, str]]]:
    """Return the host function name and its typed block arguments.

    The fourth result preserves argument order and type spelling.  It is needed
    to reconstruct the flattened ranked-memref descriptors; retaining only the
    pointer subset loses which ``i64`` values are offsets, sizes, and strides.
    """
    m = re.search(
        rf"llvm\.func\s+@({_MLIR_SYMBOL_RE})\s*\((.*?)\)\s*"
        rf"(?:->|attributes|\{{)",
        func_text,
        re.DOTALL,
    )
    if not m:
        raise ValueError("orchestrate: could not parse host llvm.func signature")
    name = m.group(1)
    sig = m.group(2)
    ordered: list[str] = []
    ptrs: list[str] = []
    typed: list[tuple[str, str]] = []
    for piece in _split_top_level(sig):
        piece = piece.strip()
        if not piece:
            continue
        am = re.match(rf"({_MLIR_SSA_RE})\s*:\s*(.+)$", piece)
        if am is None:
            raise ValueError(
                f"orchestrate: could not parse host block argument {piece!r}"
            )
        ssa = am.group(1)
        ty = am.group(2).strip()
        ordered.append(ssa)
        typed.append((ssa, ty))
        if ty.startswith("!llvm.ptr"):
            ptrs.append(ssa)
    return name, ordered, ptrs, typed


def _parse_memref_descriptors(
    typed_args: list[tuple[str, str]],
    intent_inputs: list[_IntentInput],
    intent_results: list[tuple[tuple[int | None, ...], str]],
    fresh_results: list[int],
) -> tuple[tuple[_MemRefDescriptor, ...], dict[str, int], dict[str, tuple[int, str, int | None]]]:
    """Recover every flattened ranked-memref descriptor from the host ABI.

    Descriptor order is the function-boundary order: all logical inputs, then
    fresh destination-passing results.  Each rank ``r`` descriptor must contain
    ``2 + 1 + r + r`` fields.  The exact count check is intentional: accepting a
    partial descriptor would make a later scalar look like a size/stride and
    silently corrupt a CUDA launch.
    """

    specs: list[tuple[str, int, int]] = [
        ("input", index, len(item.shape))
        for index, item in enumerate(intent_inputs)
    ]
    specs.extend(
        ("output", result_index, len(intent_results[result_index][0]))
        for result_index in fresh_results
    )
    if not specs:
        return (), {}, {}

    descriptors: list[_MemRefDescriptor] = []
    pointer_blockargs: dict[str, int] = {}
    scalar_fields: dict[str, tuple[int, str, int | None]] = {}
    cursor = 0
    for ordinal, (role, value_index, rank) in enumerate(specs):
        field_count = 2 + 1 + 2 * rank
        fields = typed_args[cursor : cursor + field_count]
        if len(fields) != field_count:
            raise NotImplementedError(
                "orchestrate: host LLVM signature ends in a partial memref "
                f"descriptor for {role} {value_index}"
            )
        pointers = fields[:2]
        if any(not ty.startswith("!llvm.ptr") for _ssa, ty in pointers):
            raise NotImplementedError(
                "orchestrate: flattened memref descriptor must begin with "
                f"allocated/aligned pointers for {role} {value_index}"
            )
        scalars = fields[2:]
        if any(ty != "i64" for _ssa, ty in scalars):
            raise NotImplementedError(
                "orchestrate: flattened memref descriptor scalar fields must "
                f"be i64 for {role} {value_index}"
            )
        allocated_ssa, aligned_ssa = pointers[0][0], pointers[1][0]
        offset_ssa = scalars[0][0]
        size_ssas = tuple(ssa for ssa, _ty in scalars[1 : 1 + rank])
        stride_ssas = tuple(ssa for ssa, _ty in scalars[1 + rank :])
        descriptor = _MemRefDescriptor(
            ordinal=ordinal,
            role=role,
            value_index=value_index,
            rank=rank,
            allocated_ssa=allocated_ssa,
            aligned_ssa=aligned_ssa,
            offset_ssa=offset_ssa,
            size_ssas=size_ssas,
            stride_ssas=stride_ssas,
        )
        descriptors.append(descriptor)
        if role == "input":
            pointer_blockargs[allocated_ssa] = value_index
            pointer_blockargs[aligned_ssa] = value_index
        for kind, index, ssa in descriptor.scalar_fields:
            scalar_fields[ssa] = (ordinal, kind, index)
        cursor += field_count
    if cursor != len(typed_args):
        raise NotImplementedError(
            "orchestrate: host LLVM signature contains arguments outside the "
            "declared ranked-memref descriptors"
        )
    return tuple(descriptors), pointer_blockargs, scalar_fields


def _map_input_aligned_ptrs(ptr_blockargs: list[str]) -> dict[str, int]:
    """Map each input's *aligned* (2nd) pointer block-arg -> input index.

    Bare-ptr-disabled host ABI flattens every memref descriptor as
    (allocated-ptr, aligned-ptr, offset, sizes..., strides...). Within the
    block-arg list the pointers appear in consecutive pairs (allocated, aligned)
    per memref, because offset/sizes/strides are i64 and interleave *between*
    descriptors, not between the two pointers of one descriptor. So the pointer
    block-args, taken two at a time, are (alloc0, align0, alloc1, align1, ...);
    every odd-indexed pointer is an aligned pointer = the value the kernel
    receives for that input.
    """
    if len(ptr_blockargs) % 2 != 0:
        raise NotImplementedError(
            "orchestrate: host signature has an odd number of !llvm.ptr block-args "
            f"({len(ptr_blockargs)}); expected (allocated, aligned) pairs per memref"
        )
    aligned: dict[str, int] = {}
    for idx in range(len(ptr_blockargs) // 2):
        aligned[ptr_blockargs[2 * idx + 1]] = idx
    return aligned


def _parse_constants(func_text: str) -> dict[str, tuple[str, str]]:
    """ssa -> (raw_value, result_type) for every ``llvm.mlir.constant``.

    LLVM dialect printing changed from ``constant(4 : index) : i64`` to forms
    such as ``constant(true) : i1`` for already-typed attributes.  Parse the
    operation structurally enough to accept both without coupling runtime ABI
    handling to one MLIR printer version.
    """

    consts: dict[str, tuple[str, str]] = {}
    for m in re.finditer(
        rf"({_MLIR_SSA_RE})\s*=\s*llvm\.mlir\.constant\(([^)]*)\)\s*:\s*(\S+)",
        func_text,
    ):
        body = m.group(2).strip()
        # The optional inner type is an attribute type, while the trailing type
        # is the actual SSA result type consumed by grid/launch/prelude code.
        raw = body.rsplit(":", 1)[0].strip() if ":" in body else body
        if raw == "true":
            raw = "1"
        elif raw == "false":
            raw = "0"
        consts[m.group(1)] = (raw, m.group(3).strip())
    return consts


def _parse_malloc_buffers(func_text: str) -> dict[str, _Buffer]:
    """Identify each struct-packed malloc buffer by its aligned-pointer SSA.

    A buffer descriptor is built by a chain of ``insertvalue``s into an
    ``llvm.struct``. The aligned pointer (struct field [1]) can be either a
    legacy ``llvm.inttoptr`` result or a direct opaque-pointer ``malloc`` result;
    the sizes are inserted at ``[3, i]``. We group insertvalues by their
    struct-SSA chain to recover each buffer's shape.
    """
    # Map each undef-struct chain: chase insertvalue dst<-src to a root undef.
    # value_inserted[ssa_result] = (src_ssa, inserted_ssa, position)
    inserts: list[tuple[str, str, str, str]] = []  # (result, src, inserted, pos)
    for m in re.finditer(
        rf"({_MLIR_SSA_RE})\s*=\s*llvm\.insertvalue\s+"
        rf"({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE})\[([0-9, ]+)\]",
        func_text,
    ):
        inserts.append((m.group(1), m.group(3), m.group(2), m.group(4).strip()))

    # Build forward chains rooted at llvm.mlir.undef structs.
    undefs = set(
        re.findall(rf"({_MLIR_SSA_RE})\s*=\s*llvm\.mlir\.undef", func_text)
    )
    # For each insertvalue, follow src back to find the originating undef root.
    src_to_result: dict[str, str] = {r: r for (r, _s, _i, _p) in inserts}
    # parent map: result -> src
    parent = {r: s for (r, s, _i, _p) in inserts}

    def root(ssa: str) -> str:
        seen = set()
        cur = ssa
        while cur in parent and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
        return cur

    # Collect, per root, the aligned ptr ([1]) and size fields ([3, i]).
    by_root: dict[str, dict] = {}
    for (result, src, inserted, pos) in inserts:
        r = root(result)
        d = by_root.setdefault(r, {"aligned": None, "sizes": {}})
        if pos == "1":
            d["aligned"] = inserted
        else:
            sm = re.match(r"3,\s*(\d+)", pos)
            if sm:
                d["sizes"][int(sm.group(1))] = inserted

    # Map size-field SSA values to integers via the constant table.
    consts = _parse_constants(func_text)

    # Reuse the per-allocation size trace used for loose buffers.  Looking at the
    # first getelementptr in the whole host function silently assigns that one
    # element type to every returned buffer; a mixed-result graph (f16 Out plus
    # f32 LSE/TMP) therefore allocated Out as f32 despite a correct P4 ABI.
    allocation_info = _parse_loose_buffers(func_text)

    buffers: dict[str, _Buffer] = {}
    # Only roots that descend from an undef AND whose aligned ptr resolves to a
    # malloc allocation are buffers; input descriptors are not rebuilt here.
    allocation_ptrs = set(allocation_info)
    for r, d in by_root.items():
        if r not in undefs:
            continue
        aligned = d["aligned"]
        if aligned is None or aligned not in allocation_ptrs:
            continue
        sizes: list[int] = []
        if d["sizes"]:
            for i in range(max(d["sizes"]) + 1):
                ssa = d["sizes"].get(i)
                if ssa is None or ssa not in consts:
                    raise NotImplementedError(
                        "orchestrate: could not resolve buffer size field "
                        f"[3,{i}] (ssa={ssa}) to a constant"
                    )
                sizes.append(int(consts[ssa][0]))
        if aligned not in allocation_info:
            raise NotImplementedError(
                f"orchestrate: could not recover allocation type for {aligned}"
            )
        elem = allocation_info[aligned].elem
        buffers[aligned] = _Buffer(aligned_ssa=aligned, shape=tuple(sizes), elem=elem)
    return buffers


def _parse_loose_buffers(func_text: str) -> dict[str, _LooseBuffer]:
    """Identify malloc'd *intermediate* buffers passed as loose aligned pointers.

    Unlike the returned output (which is packed into an ``insertvalue`` struct),
    intermediate buffers between kernels are handed to ``gpu.launch_func`` as a
    bare pointer. Legacy lowering emits an alignment chain ending in
    ``llvm.inttoptr``; opaque-pointer lowering can pass the malloc result directly::

        %m = llvm.call @malloc(%s)          # %s = byte size
        ... %a = ptrtoint %m ; alignment arithmetic ...
        %p = llvm.inttoptr %q : i64 to !llvm.ptr   # the kernel-arg pointer

    or::

        %p = llvm.call @malloc(%s) : (i64) -> !llvm.ptr
        gpu.launch_func ... args(%p : !llvm.ptr, ...)

    Returns ``{pointer_ssa -> _LooseBuffer}`` keyed by the normalized pointer the
    kernel actually receives.

    Size is taken by *tracing the malloc's size operand*, NOT by source-order
    proximity to a getelementptr: the size geps are CSE-shared across mallocs
    (e.g. two [4]-element buffers reference the same gep), so the nearest
    preceding gep is unreliable and can mis-size. We chase
    ``malloc(%s)`` -> ``%s = add %p, 64`` -> ``%p = ptrtoint %g``
    -> ``%g = getelementptr %null[N] ..., <elem>`` to read N and elem per buffer.
    Legacy malloc->inttoptr pairing is recovered by SSA def-use, not by source
    order. This matters when several allocations coexist or a direct opaque
    malloc precedes an unrelated inttoptr.
    """
    # defs[ssa] = the verbatim RHS of its single defining op (SSA is unique).
    defs: dict[str, str] = {}
    for m in re.finditer(rf"({_MLIR_SSA_RE})\s*=\s*([^\n]+)", func_text):
        defs.setdefault(m.group(1), m.group(2).strip())

    # getelementptr def -> (N, elem).  LLVM may spell the static element count
    # either as a literal ``[N]`` or as an SSA constant ``[%n]`` after CSE.
    def _gep_info(rhs: str):
        gm = re.match(
            rf"llvm\.getelementptr\s+{_MLIR_SSA_RE}\[({_MLIR_SSA_RE}|\d+)\]\s*:\s*"
            r"\([^)]*\)\s*->\s*!llvm\.ptr\s*,\s*"
            r"([A-Za-z0-9_!.<>]+)",
            rhs,
        )
        if not gm:
            return None
        count = gm.group(1)
        if count.startswith("%"):
            definition = defs.get(count, "")
            constant = re.match(
                r"llvm\.mlir\.constant\((-?\d+)\s*:\s*[^)]+\)\s*:\s*i64",
                definition,
            )
            if constant is None:
                return None
            count = constant.group(1)
        return int(count), gm.group(2).strip()

    def _trace_size(size_ssa: str):
        """size_ssa -> (n_elems, elem) by walking add/ptrtoint/getelementptr."""
        seen = set()
        cur = size_ssa
        while cur is not None and cur not in seen and cur in defs:
            seen.add(cur)
            rhs = defs[cur]
            g = _gep_info(rhs)
            if g is not None:
                return g
            # llvm.add %p, %c   -> follow the operand that is NOT the 64-const
            am = re.match(rf"llvm\.add\s+({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE})", rhs)
            if am:
                a, b = am.group(1), am.group(2)
                # prefer the ptrtoint-bearing side; fall back to first.
                cur = a
                # if a resolves to a constant and b doesn't, swap
                if a in defs and defs[a].startswith("llvm.mlir.constant") and \
                        b in defs and not defs[b].startswith("llvm.mlir.constant"):
                    cur = b
                continue
            # llvm.ptrtoint %g -> follow %g
            pm = re.match(rf"llvm\.ptrtoint\s+({_MLIR_SSA_RE})", rhs)
            if pm:
                cur = pm.group(1)
                continue
            break
        return None

    # Mallocs and inttoptrs.  Match aliases to allocations through their SSA
    # dependencies; source-order pairing silently aliases distinct allocations.
    mallocs = [(m.start(), m.group(1), m.group(2))
               for m in re.finditer(
                   rf"({_MLIR_SSA_RE})\s*=\s*llvm\.call\s+@malloc\(({_MLIR_SSA_RE})\)",
                   func_text)]
    inttoptrs = [m.group(1)
                 for m in re.finditer(
                     rf"({_MLIR_SSA_RE})\s*=\s*llvm\.inttoptr", func_text
                 )]

    def _depends_on(ssa: str, allocation_ssa: str, seen: set[str]) -> bool:
        if ssa == allocation_ssa:
            return True
        if ssa in seen or ssa not in defs:
            return False
        seen.add(ssa)
        return any(
            _depends_on(operand, allocation_ssa, seen)
            for operand in re.findall(_MLIR_SSA_RE, defs[ssa])
        )

    loose: dict[str, _LooseBuffer] = {}
    for _pos, malloc_ssa, size_ssa in mallocs:
        aliases = [
            issa
            for issa in inttoptrs
            if _depends_on(issa, malloc_ssa, set())
        ]
        if len(aliases) > 1:
            raise NotImplementedError(
                f"orchestrate: malloc {malloc_ssa} has multiple dependent "
                f"llvm.inttoptr aliases {aliases}; aligned pointer is ambiguous"
            )
        # New opaque-pointer lowering needs no ptr->int->ptr alignment roundtrip;
        # the malloc result is already the exact !llvm.ptr launch operand.
        aligned = aliases[0] if aliases else malloc_ssa
        info = _trace_size(size_ssa)
        if info is None:
            raise NotImplementedError(
                f"orchestrate: could not trace malloc {malloc_ssa} size operand "
                f"{size_ssa} to a getelementptr element count"
            )
        n_elems, elem = info
        loose[aligned] = _LooseBuffer(
            aligned_ssa=aligned, malloc_ssa=malloc_ssa, n_elems=n_elems, elem=elem
        )
    return loose


def _parse_launches(func_text: str) -> list[_LaunchSite]:
    sites: list[_LaunchSite] = []
    pat = re.compile(
        rf"gpu\.launch_func\s+@({_MLIR_SYMBOL_RE})::@({_MLIR_SYMBOL_RE})\s+"
        rf"blocks\s+in\s+\(({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE})\)\s+"
        rf"threads\s+in\s+\(({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE})\)\s*"
        r"(?::\s*\S+\s*)?"
        rf"(?:dynamic_shared_memory_size\s+({_MLIR_SSA_RE})\s*)?"  # M5 tiled smem cache
        r"args\(([^)]*)\)"
    )
    for m in pat.finditer(func_text):
        binmod, fsym = m.group(1), m.group(2)
        grid = (m.group(3), m.group(4), m.group(5))
        block = (m.group(6), m.group(7), m.group(8))
        dyn_shared = m.group(9)
        argstr = m.group(10)
        operands: list[_LaunchOperand] = []
        for piece in _split_top_level(argstr):
            piece = piece.strip()
            if not piece:
                continue
            om = re.match(rf"({_MLIR_SSA_RE})\s*:\s*(.+)$", piece)
            if not om:
                raise NotImplementedError(
                    f"orchestrate: unparsable launch operand {piece!r}"
                )
            operands.append(_LaunchOperand(om.group(1), om.group(2).strip()))
        sites.append(_LaunchSite(binmod, fsym, grid, block, operands, dyn_shared))
    return sites


def _split_top_level(s: str) -> list[str]:
    """Split on commas not nested inside <>, (), [] (defensive for arg lists)."""
    out: list[str] = []
    depth = 0
    cur = []
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _parse_return_aligned_ptrs(func_text: str) -> list[str]:
    """Trace ``llvm.return %X`` to each output's aligned pointer, in result order.

    A single-tensor result returns one flat memref descriptor
    ``struct<(ptr, ptr, i64, ...)>`` whose field [1] is the aligned ptr. A
    multi-tensor result (e.g. a loop body yielding ``m_new, l_new, O_new``)
    returns a struct-of-structs ``struct<(struct<...>, struct<...>, ...)>``: each
    top-level field [i] is itself a memref descriptor, inserted by an
    ``insertvalue ...[i]`` whose inserted value is another built struct (i.e. it
    is itself an insertvalue result), and each inner descriptor's [1] is its
    aligned ptr. Returns ``[]`` if no return ptr can be recovered.
    """
    rm = re.search(rf"llvm\.return\s+({_MLIR_SSA_RE})", func_text)
    if not rm:
        return []
    ret_ssa = rm.group(1)
    # Build insertvalue maps: parent (dst<-src), and per-result (pos, inserted).
    parent: dict[str, str] = {}
    pos_of: dict[str, str] = {}
    inserted_of: dict[str, str] = {}
    for m in re.finditer(
        rf"({_MLIR_SSA_RE})\s*=\s*llvm\.insertvalue\s+"
        rf"({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE})\[([0-9, ]+)\]",
        func_text,
    ):
        result, inserted, src, pos = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        parent[result] = src
        pos_of[result] = pos
        inserted_of[result] = inserted

    def aligned_of_struct(struct_ssa: str) -> str | None:
        """First [1] insert walking up a single descriptor's chain = aligned ptr."""
        seen: set[str] = set()
        cur = struct_ssa
        while cur is not None and cur not in seen:
            seen.add(cur)
            if pos_of.get(cur) == "1":
                return inserted_of[cur]
            cur = parent.get(cur)
        return None

    # Top-level field inserts in the returned struct's chain: pos is a single int
    # AND the inserted value is itself a built struct (an insertvalue result).
    fields: dict[int, str] = {}
    seen: set[str] = set()
    cur: str | None = ret_ssa
    while cur is not None and cur not in seen:
        seen.add(cur)
        p = pos_of.get(cur)
        if p is not None and p.isdigit() and inserted_of[cur] in pos_of:
            fields.setdefault(int(p), inserted_of[cur])
        cur = parent.get(cur)

    if fields:  # multi-output: one aligned ptr per top-level field, in order
        out: list[str] = []
        for i in sorted(fields):
            a = aligned_of_struct(fields[i])
            if a is not None:
                out.append(a)
        return out
    a = aligned_of_struct(ret_ssa)
    return [a] if a is not None else []


def _parse_const_globals(binary_mlir: str, func_text: str) -> dict[str, _ConstGlobal]:
    """Resolve scalar/dense globals plus zero-offset GEP aliases."""

    def parse_value(text: str, elem: str) -> int | float:
        if elem == "i1" and text in {"true", "false"}:
            return int(text == "true")
        return _parse_f32(text) if elem.startswith("f") else int(text, 0)

    def parse_dense_values(text: str, elem: str) -> int | float | tuple[int | float, ...]:
        text = text.strip()
        if not (text.startswith("[") and text.endswith("]")):
            return parse_value(text, elem)
        pieces = [piece.strip() for piece in text[1:-1].split(",") if piece.strip()]
        return tuple(parse_value(piece, elem) for piece in pieces)

    globals_: dict[str, _ConstGlobal] = {}
    for match in re.finditer(
        rf"llvm\.mlir\.global\s+[^@]*@({_MLIR_SYMBOL_RE})\(\s*"
        r"(true|false|[-+0-9.eExXa-fA-F]+)"
        r"(?:\s*:\s*(?:bf16|f\d+|(?:ui|si|i)\d+))?\s*\)\s*"
        r"(?:\{[^}\n]*\}\s*)?:\s*(bf16|f\d+|(?:ui|si|i)\d+)",
        binary_mlir,
    ):
        name, raw, elem = match.group(1), match.group(2), match.group(3)
        globals_[name] = _ConstGlobal(name, (), elem, parse_value(raw, elem))
    for match in re.finditer(
        rf"llvm\.mlir\.global\s+[^@]*@({_MLIR_SYMBOL_RE})\(\s*dense<"
        r"(?P<values>\[[^]]*\]|[^>]+)>\s*:\s*tensor<"
        r"(?P<dims>(?:\d+x)*)?(?P<elem>bf16|f\d+|(?:ui|si|i)\d+)>\s*\)",
        binary_mlir,
    ):
        name, raw, dims_text, elem = (
            match.group(1),
            match.group("values"),
            match.group("dims") or "",
            match.group("elem"),
        )
        shape = tuple(int(dim) for dim in dims_text.split("x") if dim)
        parsed = parse_dense_values(raw, elem)
        if isinstance(parsed, tuple):
            expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if len(parsed) == 1 and expected > 1:
                parsed = parsed * expected
            if len(parsed) != expected:
                raise ValueError(
                    f"orchestrate: constant global @{name} element count does not "
                    f"match its declared shape {shape}"
                )
        globals_[name] = _ConstGlobal(
            name, shape, elem, parsed
        )

    out: dict[str, _ConstGlobal] = {}
    for match in re.finditer(
        rf"({_MLIR_SSA_RE})\s*=\s*llvm\.mlir\.addressof\s+@({_MLIR_SYMBOL_RE})",
        func_text
    ):
        if match.group(2) in globals_:
            out[match.group(1)] = globals_[match.group(2)]
    changed = True
    while changed:
        changed = False
        for match in re.finditer(
            rf"({_MLIR_SSA_RE})\s*=\s*llvm\.getelementptr\s+"
            rf"({_MLIR_SSA_RE})\[([^]]+)\]",
            func_text,
        ):
            result, base, indices = match.group(1), match.group(2), match.group(3)
            if result in out or base not in out:
                continue
            parsed_indices = [item.strip() for item in indices.split(",")]
            if not parsed_indices or any(item != "0" for item in parsed_indices):
                continue
            out[result] = out[base]
            changed = True
    return out


def _parse_host_module(
    binary_mlir: str,
    intent_inputs: list[_IntentInput],
    intent_results: list[tuple[tuple[int | None, ...], str]] | None = None,
    intent_inplace: list[int] | None = None,
    require_descriptor_abi: bool = False,
) -> _HostModule:
    _func_name, func_text = _host_func_text(binary_mlir)
    sig_name, ordered, ptr_blockargs, typed_args = _parse_signature_blockargs(
        func_text
    )
    declared_results = intent_results or []
    result_count = len(declared_results)
    input_count = len(intent_inputs)
    destination_aligned_ptrs: list[str] = []
    destination_result_indices: list[int] = []
    fresh_results = (
        [index for index, alias in enumerate(intent_inplace) if alias < 0]
        if intent_inplace is not None
        else list(range(result_count))
    )
    descriptors: tuple[_MemRefDescriptor, ...] = ()
    aligned_map: dict[str, int] = {}
    pointer_blockargs: dict[str, int] = {}
    descriptor_scalar_fields: dict[str, tuple[int, str, int | None]] = {}
    descriptor_error: NotImplementedError | None = None
    try:
        descriptors, pointer_blockargs, descriptor_scalar_fields = (
            _parse_memref_descriptors(
                typed_args,
                intent_inputs,
                declared_results,
                fresh_results,
            )
        )
    except NotImplementedError as exc:
        descriptor_error = exc

    if descriptors:
        input_descriptors = tuple(
            item for item in descriptors if item.role == "input"
        )
        destination_descriptors = tuple(
            item for item in descriptors if item.role == "output"
        )
        if len(input_descriptors) != input_count or len(destination_descriptors) != len(
            fresh_results
        ):
            raise NotImplementedError(
                "orchestrate: flattened descriptor order does not match the "
                "declared input/destination ABI"
            )
        aligned_map = {
            item.aligned_ssa: item.value_index for item in input_descriptors
        }
        destination_aligned_ptrs = [
            item.aligned_ssa for item in destination_descriptors
        ]
        destination_result_indices = [
            item.value_index for item in destination_descriptors
        ]
    else:
        if require_descriptor_abi:
            detail = f": {descriptor_error}" if descriptor_error else ""
            raise NotImplementedError(
                "orchestrate: CUDA Ragged graph requires the complete flattened "
                f"memref-descriptor ABI{detail}"
            )
        all_aligned = _map_input_aligned_ptrs(ptr_blockargs)
        if len(all_aligned) == input_count:
            aligned_map = all_aligned
            ordered_pairs = [
                (ptr_blockargs[2 * index], ptr_blockargs[2 * index + 1])
                for index in range(input_count)
            ]
        elif fresh_results and len(all_aligned) == input_count + len(fresh_results):
            ordered_aligned = [
                ssa
                for ssa, _index in sorted(
                    all_aligned.items(), key=lambda item: item[1]
                )
            ]
            aligned_map = {
                ssa: index
                for index, ssa in enumerate(ordered_aligned[:input_count])
            }
            destination_aligned_ptrs = ordered_aligned[input_count:]
            destination_result_indices = fresh_results
            ordered_pairs = [
                (ptr_blockargs[2 * index], ptr_blockargs[2 * index + 1])
                for index in range(input_count)
            ]
        else:
            raise NotImplementedError(
                "orchestrate: host has "
                f"{len(all_aligned)} memrefs but intent signature has {input_count} "
                f"inputs and {result_count} results; cannot distinguish input and "
                "destination-passing buffers unambiguously"
            )
        pointer_blockargs = {
            pointer: index
            for index, pair in enumerate(ordered_pairs)
            for pointer in pair
        }
    struct_buffers = _parse_malloc_buffers(func_text)
    loose_buffers = _parse_loose_buffers(func_text)
    # The struct-packed (returned) buffer wins: it carries real shape+dtype, so
    # drop any loose entry that the struct path already shaped correctly.
    for ssa in struct_buffers:
        loose_buffers.pop(ssa, None)
    return _HostModule(
        func_name=sig_name,
        sig_args=ordered,
        constants=_parse_constants(func_text),
        aligned_ptr_blockargs=aligned_map,
        pointer_blockargs=pointer_blockargs,
        descriptors=descriptors,
        descriptor_scalar_fields=descriptor_scalar_fields,
        destination_aligned_ptrs=destination_aligned_ptrs,
        destination_result_indices=destination_result_indices,
        malloc_buffers=struct_buffers,
        loose_buffers=loose_buffers,
        launches=_parse_launches(func_text),
        return_aligned_ptrs=_parse_return_aligned_ptrs(func_text),
        body_text=func_text,
        const_globals=_parse_const_globals(binary_mlir, func_text),
    )


# --------------------------------------------------------------------------- #
# Execution.                                                                   #
# --------------------------------------------------------------------------- #
def _resolve_const_index(host: _HostModule, ssa: str) -> int:
    if ssa not in host.constants:
        raise NotImplementedError(
            f"orchestrate: grid/block extent {ssa} is not a literal constant"
        )
    return int(host.constants[ssa][0])


def _execute_host_scalar_prelude(
    host: _HostModule,
    input_tensor_by_index,
    buffer_tensor_by_ssa,
) -> None:
    """Execute scalar host stores emitted before the first GPU launch.

    Bufferization keeps rank-0 tensor casts on the host.  A representative
    lowering is ``load input<i32> -> sitofp -> store malloc<f32>``; ignoring
    that prelude leaves the device buffer uninitialized even though every GPU
    kernel is launched correctly.  Evaluate the deliberately small scalar SSA
    subset on CUDA tensors so no host value round-trip or original compiler
    callback is needed.  A later scalar store is hoisted only when its
    destination has not appeared in an earlier launch and its value depends
    solely on input scalars/constants; arbitrary host LLVM is still rejected.
    """

    import torch

    body = host.body_text
    first_launch = body.find("gpu.launch_func")
    definitions: dict[str, str] = {}
    for match in re.finditer(rf"({_MLIR_SSA_RE})\s*=\s*([^\n]+)", body):
        definitions[match.group(1)] = match.group(2).strip()
    stores: list[tuple[int, str, str]] = []
    for match in re.finditer(
        rf"llvm\.store\s+({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE})\s*:", body
    ):
        destination = match.group(2)
        if destination not in buffer_tensor_by_ssa:
            continue
        if first_launch >= 0 and match.start() > first_launch:
            earlier_launch_text = body[first_launch : match.start()]
            if re.search(
                rf"gpu\.launch_func[^\n]*{re.escape(destination)}\s*:",
                earlier_launch_text,
            ):
                raise NotImplementedError(
                    "orchestrate: cannot hoist scalar store whose destination "
                    f"{destination} was used by an earlier GPU launch"
                )
        stores.append((match.start(), match.group(1), destination))
    if not stores:
        return

    cache: dict[str, "torch.Tensor"] = {}
    initialized_scalar_buffers: set[str] = set()

    def resolve_load_pointer(pointer: str) -> tuple["torch.Tensor", int]:
        if pointer in host.pointer_blockargs:
            return (
                input_tensor_by_index[host.pointer_blockargs[pointer]],
                0,
            )
        if pointer in host.const_globals and pointer in buffer_tensor_by_ssa:
            return buffer_tensor_by_ssa[pointer], 0
        # A compiler-created scalar buffer is readable only after this prelude
        # has itself initialized it from inputs/constants.  This permits
        # straight-line scalar expressions while continuing to reject reads
        # from arbitrary or uninitialized malloc buffers.
        if pointer in initialized_scalar_buffers:
            return buffer_tensor_by_ssa[pointer], 0
        rhs = definitions.get(pointer)
        if rhs is None:
            raise NotImplementedError(
                f"orchestrate: scalar load pointer {pointer} has no definition"
            )
        gep = re.match(
            rf"llvm\.getelementptr\s+({_MLIR_SSA_RE})\[([^]]+)\]",
            rhs,
        )
        if gep:
            tensor, base_offset = resolve_load_pointer(gep.group(1))
            pieces = [piece.strip() for piece in gep.group(2).split(",")]
            if not pieces or any(not re.fullmatch(r"-?\d+", piece) for piece in pieces):
                raise NotImplementedError(
                    "orchestrate: scalar prelude GEP indices must be literals"
                )
            # Scalar element pointers use one linear index.  Aggregate constant
            # globals may spell the root element as [0, 0]; only zero leading
            # aggregate indices are valid in this host-scalar subset.
            indices = [int(piece) for piece in pieces]
            if len(indices) > 1 and any(index != 0 for index in indices[:-1]):
                raise NotImplementedError(
                    "orchestrate: scalar prelude aggregate GEP is not a root element"
                )
            return tensor, base_offset + indices[-1]
        raise NotImplementedError(
            f"orchestrate: unsupported scalar load pointer {pointer} = {rhs}"
        )

    def evaluate(ssa: str) -> "torch.Tensor":
        if ssa in cache:
            return cache[ssa]
        if ssa in host.constants:
            raw, ty = host.constants[ssa]
            if ty.startswith("f"):
                value = torch.tensor(
                    _parse_f32(raw), dtype=_torch_dtype(ty), device="cuda"
                )
            else:
                integer_types = {
                    "i1", "i8", "i16", "i32", "i64",
                    "ui8", "ui16", "ui32",
                }
                dtype = _torch_dtype(ty) if ty in integer_types else torch.int64
                value = torch.tensor(int(raw), dtype=dtype, device="cuda")
            cache[ssa] = value
            return value
        rhs = definitions.get(ssa)
        if rhs is None:
            raise NotImplementedError(
                f"orchestrate: scalar prelude value {ssa} has no definition"
            )
        load = re.match(rf"llvm\.load\s+({_MLIR_SSA_RE})\s*:", rhs)
        if load:
            pointer = load.group(1)
            tensor, offset = resolve_load_pointer(pointer)
            flat = tensor.reshape(-1)
            if offset < 0 or offset >= flat.numel():
                raise IndexError(
                    f"orchestrate: scalar prelude load {pointer}[{offset}] is out of bounds"
                )
            value = flat[offset]
            cache[ssa] = value
            return value

        cast = re.match(
            r"llvm\.(sitofp|uitofp|fptosi|fptoui|fpext|fptrunc|sext|zext|trunc)\s+"
            rf"({_MLIR_SSA_RE})\s*:\s*\S+\s+to\s+(\S+)",
            rhs,
        )
        if cast:
            value = evaluate(cast.group(2)).to(dtype=_torch_dtype(cast.group(3)))
            cache[ssa] = value
            return value

        binary = re.match(
            r"llvm\.(add|sub|mul|sdiv|udiv|fadd|fsub|fmul|fdiv|and|or|xor)\s+"
            rf"({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE})",
            rhs,
        )
        if binary:
            lhs = evaluate(binary.group(2))
            rhs_value = evaluate(binary.group(3))
            operation = binary.group(1)
            if operation in {"add", "fadd"}:
                value = lhs + rhs_value
            elif operation in {"sub", "fsub"}:
                value = lhs - rhs_value
            elif operation in {"mul", "fmul"}:
                value = lhs * rhs_value
            elif operation == "and":
                value = torch.bitwise_and(lhs, rhs_value)
            elif operation == "or":
                value = torch.bitwise_or(lhs, rhs_value)
            elif operation == "xor":
                value = torch.bitwise_xor(lhs, rhs_value)
            else:
                value = lhs / rhs_value
            cache[ssa] = value
            return value

        icmp = re.match(
            r'llvm\.icmp\s+"(eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge)"\s+'
            rf"({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE})",
            rhs,
        )
        if icmp:
            lhs = evaluate(icmp.group(2))
            rhs_value = evaluate(icmp.group(3))
            predicate = icmp.group(1)
            if predicate == "eq":
                value = lhs == rhs_value
            elif predicate == "ne":
                value = lhs != rhs_value
            elif predicate in {"slt", "ult"}:
                value = lhs < rhs_value
            elif predicate in {"sle", "ule"}:
                value = lhs <= rhs_value
            elif predicate in {"sgt", "ugt"}:
                value = lhs > rhs_value
            else:
                value = lhs >= rhs_value
            cache[ssa] = value
            return value

        select = re.match(
            rf"llvm\.select\s+({_MLIR_SSA_RE}),\s*({_MLIR_SSA_RE}),\s*"
            rf"({_MLIR_SSA_RE})\s*:",
            rhs,
        )
        if select:
            condition = evaluate(select.group(1))
            true_value = evaluate(select.group(2))
            false_value = evaluate(select.group(3))
            if condition.numel() != 1:
                raise NotImplementedError(
                    "orchestrate: scalar prelude select condition is not rank-0"
                )
            value = torch.where(condition.to(dtype=torch.bool), true_value, false_value)
            cache[ssa] = value
            return value
        raise NotImplementedError(
            f"orchestrate: unsupported scalar prelude definition {ssa} = {rhs}"
        )

    for _position, value_ssa, destination_ssa in stores:
        destination = buffer_tensor_by_ssa[destination_ssa].reshape(-1)
        if destination.numel() != 1:
            raise NotImplementedError(
                "orchestrate: host scalar store destination is not one element: "
                f"{destination_ssa} has {destination.numel()}"
            )
        destination[0].copy_(evaluate(value_ssa).to(dtype=destination.dtype))
        initialized_scalar_buffers.add(destination_ssa)


def run_program(
    intent_mlir_path,
    inputs,
    compile_all=compile_all_kernels,
    output_aliases=None,
    _prepare_only: bool = False,
):
    """Compile *intent_mlir_path* and execute its whole kernel graph on the GPU.

    Parameters
    ----------
    intent_mlir_path : str | PathLike
        Path to an intent-dialect ``.mlir`` module (single- or multi-kernel).
    inputs : dict[str, torch.Tensor]
        Maps each intent function input *name* (from the ``func.func`` signature,
        e.g. ``"a"``, ``"b"``) to a CUDA torch tensor. If a name is missing, the
        positional key ``"arg{i}"`` (0-based) is accepted as a fallback.
    compile_all : callable, optional
        The ``intent_mlir_path -> list[CompiledKernel]`` compiler. Defaults to the
        per-op pipeline; pass ``compile_all_kernels_fused`` (the fused+tiled
        pipeline) to execute a pointwise chain as one fused kernel. The host module
        is parsed generically, so a fused single-kernel graph runs unchanged.
    output_aliases : dict[str, str], optional
        Layer-B runtime binding from an output key (``"out0"``) to an input
        tensor name.  The returned P5 buffer SSA is bound directly to that input
        allocation before any launch; no copy kernel or source callback is used.

    Returns
    -------
    dict[str, torch.Tensor]
        The program outputs. Keyed by ``"out{i}"`` (0-based, in return order).
    """
    import torch

    ensure_context()

    intent_src = Path(intent_mlir_path).read_text()
    intent_inputs = _parse_intent_signature(intent_src)
    intent_results = _parse_intent_results(intent_src)
    intent_inplace = _parse_intent_inplace(intent_src, len(intent_results))

    kernels = compile_all(intent_mlir_path)
    if not kernels:
        raise RuntimeError("orchestrate: compile_all_kernels produced no kernels")
    binary_mlir = kernels[0].binary_mlir
    by_binary: dict[str, CompiledKernel] = {k.binary_name: k for k in kernels}

    host = _parse_host_module(
        binary_mlir,
        intent_inputs,
        intent_results,
        intent_inplace,
        require_descriptor_abi="intent.ragged" in intent_src,
    )
    if host.return_aligned_ptrs:
        output_aligned_ptrs = host.return_aligned_ptrs
    elif intent_inplace is not None:
        input_ssa_by_index = {
            index: ssa for ssa, index in host.aligned_ptr_blockargs.items()
        }
        destination_by_result = dict(
            zip(
                host.destination_result_indices,
                host.destination_aligned_ptrs,
                strict=True,
            )
        )
        output_aligned_ptrs = []
        for result_index, alias in enumerate(intent_inplace):
            if alias >= 0:
                if alias >= len(intent_inputs) or alias not in input_ssa_by_index:
                    raise ValueError(
                        "orchestrate: intent.inplace references an invalid input"
                    )
                output_aligned_ptrs.append(input_ssa_by_index[alias])
            elif result_index in destination_by_result:
                output_aligned_ptrs.append(destination_by_result[result_index])
            else:
                raise RuntimeError(
                    "orchestrate: no destination buffer for a fresh declared result"
                )
    else:
        output_aligned_ptrs = host.destination_aligned_ptrs

    # --- bind input tensors by aligned-ptr block-arg -> input index -------- #
    # aligned_ptr_blockargs: %argK (aligned ptr) -> input index i
    input_tensor_by_index: dict[int, "torch.Tensor"] = {}
    for blockarg_ssa, idx in host.aligned_ptr_blockargs.items():
        # Determine the user key for this input index.
        key = None
        if idx < len(intent_inputs):
            name = intent_inputs[idx].name
            if name in inputs:
                key = name
        if key is None:
            fallback = f"arg{idx}"
            if fallback in inputs:
                key = fallback
        if key is None:
            raise KeyError(
                f"orchestrate: no tensor supplied for input index {idx} "
                f"(tried name "
                f"{intent_inputs[idx].name!r} and 'arg{idx}')"
                if idx < len(intent_inputs)
                else f"orchestrate: no tensor supplied for input index {idx} (tried 'arg{idx}')"
            )
        t = inputs[key]
        if not t.is_cuda:
            raise ValueError(f"orchestrate: input {key!r} must be a CUDA tensor")
        if idx >= len(intent_inputs):
            raise ValueError(
                f"orchestrate: host input index {idx} exceeds the P4 signature"
            )
        declared = intent_inputs[idx]
        if tuple(t.shape) != declared.shape:
            raise ValueError(
                f"orchestrate: input {key!r} shape {tuple(t.shape)} does not "
                f"match declared {declared.shape}"
            )
        expected_dtype = _torch_dtype(declared.dtype_str)
        if t.dtype != expected_dtype:
            raise ValueError(
                f"orchestrate: input {key!r} dtype {t.dtype} does not match "
                f"declared {expected_dtype}"
            )
        input_tensor_by_index[idx] = t.contiguous()

    intent_results = _resolve_intent_result_shapes(
        intent_src,
        intent_inputs,
        intent_results,
        input_tensor_by_index,
    )

    alias_tensor_by_ssa: dict[str, "torch.Tensor"] = {}
    if output_aliases:
        if not isinstance(output_aliases, dict):
            raise TypeError("orchestrate: output_aliases must be a dict")
        for output_key, input_name in output_aliases.items():
            match = re.fullmatch(r"out(\d+)", str(output_key))
            if match is None:
                raise ValueError(
                    f"orchestrate: invalid output alias key {output_key!r}"
                )
            output_index = int(match.group(1))
            if output_index >= len(output_aligned_ptrs):
                raise ValueError(
                    f"orchestrate: output alias {output_key!r} is out of range"
                )
            input_indices = [
                index
                for index, item in enumerate(intent_inputs)
                if item.name == input_name
            ]
            if len(input_indices) != 1:
                raise ValueError(
                    f"orchestrate: output alias input {input_name!r} matched "
                    f"{len(input_indices)} ABI entries"
                )
            input_index = input_indices[0]
            tensor = input_tensor_by_index[input_index]
            if output_index < len(intent_results):
                shape, elem = intent_results[output_index]
                if tuple(tensor.shape) != tuple(shape):
                    raise ValueError(
                        f"orchestrate: {output_key} alias shape {tuple(tensor.shape)} "
                        f"does not match declared {shape}"
                    )
                expected_dtype = _torch_dtype(elem)
                if tensor.dtype != expected_dtype:
                    raise ValueError(
                        f"orchestrate: {output_key} alias dtype {tensor.dtype} "
                        f"does not match declared {expected_dtype}"
                    )
            output_ssa = output_aligned_ptrs[output_index]
            existing = alias_tensor_by_ssa.get(output_ssa)
            if existing is not None and existing.data_ptr() != tensor.data_ptr():
                raise ValueError(
                    f"orchestrate: conflicting aliases for returned buffer {output_ssa}"
                )
            alias_tensor_by_ssa[output_ssa] = tensor

    # --- allocate every malloc buffer ------------------------------------- #
    # Struct-packed buffers (the returned output) get their real shape; loose
    # intermediates get a flat 1-D buffer of the right element count. One
    # SSA->tensor entry per buffer means a pointer reused across kernels
    # (e.g. softmax's exp buffer) resolves to the same storage automatically.
    buffer_tensor_by_ssa: dict[str, "torch.Tensor"] = {}

    def register_descriptor_pointers(
        pointer_ssa: str, tensor: "torch.Tensor"
    ) -> None:
        """Bind both fields of one destination memref descriptor.

        The flattened non-bare-pointer ABI carries ``allocated`` and
        ``aligned`` as separate pointer arguments even though both describe
        the same allocation.  Output aliases and fresh destinations must
        therefore register both SSA identities; registering only the aligned
        field makes the host descriptor internally inconsistent and leaves a
        later launch operand unresolved.
        """

        buffer_tensor_by_ssa[pointer_ssa] = tensor
        matches = [
            descriptor
            for descriptor in host.descriptors
            if descriptor.role == "output"
            and pointer_ssa
            in {descriptor.allocated_ssa, descriptor.aligned_ssa}
        ]
        if len(matches) > 1:
            raise RuntimeError(
                "orchestrate: one destination pointer belongs to multiple "
                "memref descriptors"
            )
        if matches:
            descriptor = matches[0]
            buffer_tensor_by_ssa[descriptor.allocated_ssa] = tensor
            buffer_tensor_by_ssa[descriptor.aligned_ssa] = tensor

    for ssa, result_index in zip(
        host.destination_aligned_ptrs,
        host.destination_result_indices,
        strict=True,
    ):
        if ssa in alias_tensor_by_ssa:
            register_descriptor_pointers(ssa, alias_tensor_by_ssa[ssa])
            continue
        if result_index >= len(intent_results):
            raise RuntimeError(
                "orchestrate: destination-passing output has no declared P4 result"
            )
        shape, elem = intent_results[result_index]
        tensor = torch.empty(
            shape if shape else (), dtype=_torch_dtype(elem), device="cuda"
        ).contiguous()
        register_descriptor_pointers(ssa, tensor)
    for ssa, buf in host.malloc_buffers.items():
        if ssa in alias_tensor_by_ssa:
            buffer_tensor_by_ssa[ssa] = alias_tensor_by_ssa[ssa]
            continue
        dtype = _torch_dtype(buf.elem)
        buffer_tensor_by_ssa[ssa] = torch.zeros(
            *buf.shape, dtype=dtype, device="cuda"
        ).contiguous()
    for ssa, lbuf in host.loose_buffers.items():
        if ssa in alias_tensor_by_ssa:
            buffer_tensor_by_ssa[ssa] = alias_tensor_by_ssa[ssa]
            continue
        dtype = _torch_dtype(lbuf.elem)
        # torch.empty is fine: every reduce buffer is overwritten by its explicit
        # identity-fill kernel (-inf for max, 0 for sum) before first read.
        buffer_tensor_by_ssa[ssa] = torch.empty(
            (lbuf.n_elems,), dtype=dtype, device="cuda"
        ).contiguous()

    # Static scalar / dense globals become read-only CUDA buffers.  GEP aliases
    # share the same allocation instead of materializing extra copies.  This is
    # also the runtime representation of a rank-one constant Ragged carrier
    # after MLIR bufferization.
    const_tensor_by_name: dict[str, "torch.Tensor"] = {}
    for ssa, constant in host.const_globals.items():
        tensor = const_tensor_by_name.get(constant.name)
        if tensor is None:
            dtype = _torch_dtype(constant.elem)
            if isinstance(constant.value, tuple):
                tensor = torch.tensor(
                    constant.value,
                    dtype=dtype,
                    device="cuda",
                ).reshape(constant.shape).contiguous()
            else:
                shape = constant.shape if constant.shape else (1,)
                tensor = torch.full(
                    shape,
                    constant.value,
                    dtype=dtype,
                    device="cuda",
                ).contiguous()
            const_tensor_by_name[constant.name] = tensor
        buffer_tensor_by_ssa[ssa] = tensor

    # ``memref.get_global`` lowers its unused allocated-pointer field to the
    # LLVM placeholder 0xDEADBEEF while the aligned field points at the actual
    # global.  The CUDA kernel consumes the aligned field, but the flattened
    # launch ABI still contains both pointers.  Supply any valid device pointer
    # for the placeholder so the driver argument list remains complete; the
    # corresponding aligned global pointer is still passed in its own slot.
    if const_tensor_by_name:
        fallback = next(iter(const_tensor_by_name.values()))
        for match in re.finditer(
            rf"({_MLIR_SSA_RE})\s*=\s*llvm\.inttoptr\s+"
            rf"({_MLIR_SSA_RE})\s*:\s*i64\s+to\s+!llvm\.ptr",
            host.body_text,
        ):
            constant = host.constants.get(match.group(2))
            if constant is None:
                continue
            try:
                value = int(constant[0], 0)
            except ValueError:
                continue
            if value == 0xDEADBEEF:
                buffer_tensor_by_ssa.setdefault(match.group(1), fallback)

    _execute_host_scalar_prelude(
        host,
        input_tensor_by_index,
        buffer_tensor_by_ssa,
    )

    # --- map each launch operand to a concrete launch arg ------------------ #
    def _resolve_ptr(ssa: str) -> "torch.Tensor":
        if ssa in host.pointer_blockargs:
            return input_tensor_by_index[host.pointer_blockargs[ssa]]
        if ssa in buffer_tensor_by_ssa:
            return buffer_tensor_by_ssa[ssa]
        raise NotImplementedError(
            f"orchestrate: pointer operand {ssa} is neither an input descriptor "
            f"pointer, a destination descriptor pointer, a struct-packed buffer, "
            f"nor a loose malloc buffer "
            f"(unhandled host construct)"
        )

    descriptor_tensors: dict[int, "torch.Tensor"] = {}
    for descriptor in host.descriptors:
        if descriptor.role == "input":
            descriptor_tensors[descriptor.ordinal] = input_tensor_by_index[
                descriptor.value_index
            ]
        elif descriptor.aligned_ssa in buffer_tensor_by_ssa:
            descriptor_tensors[descriptor.ordinal] = buffer_tensor_by_ssa[
                descriptor.aligned_ssa
            ]

    def _descriptor_scalar(ssa: str) -> int | None:
        field = host.descriptor_scalar_fields.get(ssa)
        if field is None:
            return None
        ordinal, kind, index = field
        tensor = descriptor_tensors.get(ordinal)
        if tensor is None:
            raise RuntimeError(
                f"orchestrate: no host tensor is bound to descriptor {ordinal}"
            )
        if kind == "offset":
            # All launcher tensors are made contiguous before binding, so the
            # destination-passing descriptor has the canonical zero offset.
            return int(tensor.storage_offset())
        if kind == "size":
            assert index is not None
            return int(tensor.shape[index])
        if kind == "stride":
            assert index is not None
            return int(tensor.stride(index))
        raise RuntimeError(f"orchestrate: unknown descriptor scalar field {kind!r}")

    # --- execute each launch in order -------------------------------------- #
    scalar_load_inputs: dict[str, int] = {}
    for match in re.finditer(
        rf"({_MLIR_SSA_RE})\s*=\s*llvm\.load\s+({_MLIR_SSA_RE})\s*:\s*!llvm\.ptr\s*->\s*"
        r"(i1|i8|i16|i32|i64|ui8|ui16|ui32|f16|bf16|f32|f64)",
        host.body_text,
    ):
        result_ssa, pointer_ssa = match.group(1), match.group(2)
        if pointer_ssa in host.pointer_blockargs:
            scalar_load_inputs[result_ssa] = host.pointer_blockargs[pointer_ssa]

    def _loaded_input_scalar(ssa: str, expected_dtype):
        if ssa not in scalar_load_inputs:
            raise NotImplementedError(
                f"orchestrate: scalar launch arg {ssa} is neither a constant "
                "nor a direct rank-0 input load"
            )
        input_index = scalar_load_inputs[ssa]
        tensor = input_tensor_by_index[input_index]
        if tensor.numel() != 1 or tensor.dtype != expected_dtype:
            raise ValueError(
                "orchestrate: loaded scalar launch input has the wrong ABI: "
                f"index={input_index}, shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, expected={expected_dtype}"
            )
        return tensor.item()

    def _loaded_input_scalar_bits(ssa: str, expected_dtype) -> int:
        if ssa not in scalar_load_inputs:
            raise NotImplementedError(
                f"orchestrate: scalar launch arg {ssa} is neither a constant "
                "nor a direct rank-0 input load"
            )
        input_index = scalar_load_inputs[ssa]
        tensor = input_tensor_by_index[input_index]
        if tensor.numel() != 1 or tensor.dtype != expected_dtype:
            raise ValueError(
                "orchestrate: loaded scalar launch input has the wrong ABI: "
                f"index={input_index}, shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, expected={expected_dtype}"
            )
        return int(tensor.reshape(()).view(torch.uint16).item())

    prepared_sites = []
    for site in host.launches:
        ck = by_binary.get(site.binary_module)
        if ck is None:
            raise RuntimeError(
                f"orchestrate: no compiled cubin for gpu.binary @{site.binary_module} "
                f"(have {sorted(by_binary)})"
            )
        func = load_module(ck.cubin).function(site.func_symbol)
        # NB: ck.kernel_name == site.func_symbol; both fixtures agree, but we
        # match the *binary* module name (unique) not the func name (may repeat).

        grid = tuple(_resolve_const_index(host, s) for s in site.grid_ssa)
        block = tuple(_resolve_const_index(host, s) for s in site.block_ssa)
        shared = 0
        if site.dynamic_shared_ssa is not None:
            if site.dynamic_shared_ssa not in host.constants:
                raise NotImplementedError(
                    "orchestrate: dynamic shared-memory size is not a constant"
                )
            shared = int(host.constants[site.dynamic_shared_ssa][0])
            _set_dynamic_shared_memory(func, shared)

        launch_tensors = []
        plan = []
        for op in site.args:
            ty = op.ty
            if ty.startswith("!llvm.ptr"):
                t = _resolve_ptr(op.ssa)
                plan.append(("ptr", len(launch_tensors)))
                launch_tensors.append(t)
            elif ty in ("i64", "index"):
                descriptor_value = _descriptor_scalar(op.ssa)
                value = (
                    descriptor_value
                    if descriptor_value is not None
                    else int(host.constants[op.ssa][0])
                    if op.ssa in host.constants
                    else int(_loaded_input_scalar(op.ssa, torch.int64))
                )
                plan.append(("i64", value))
            elif ty == "i32":
                value = (
                    int(host.constants[op.ssa][0])
                    if op.ssa in host.constants
                    else int(_loaded_input_scalar(op.ssa, torch.int32))
                )
                plan.append(("i32", value))
            elif ty == "ui8":
                value = (
                    int(host.constants[op.ssa][0])
                    if op.ssa in host.constants
                    else int(_loaded_input_scalar(op.ssa, torch.uint8))
                )
                plan.append(("u8", value))
            elif ty == "ui16":
                value = (
                    int(host.constants[op.ssa][0])
                    if op.ssa in host.constants
                    else int(_loaded_input_scalar(op.ssa, torch.uint16))
                )
                plan.append(("u16", value))
            elif ty == "ui32":
                value = (
                    int(host.constants[op.ssa][0])
                    if op.ssa in host.constants
                    else int(_loaded_input_scalar(op.ssa, torch.uint32))
                )
                plan.append(("u32", value))
            elif ty == "f16":
                bits = (
                    _parse_f16_bits(host.constants[op.ssa][0])
                    if op.ssa in host.constants
                    else _loaded_input_scalar_bits(op.ssa, torch.float16)
                )
                plan.append(("f16", bits))
            elif ty == "bf16":
                bits = (
                    _parse_bf16_bits(host.constants[op.ssa][0])
                    if op.ssa in host.constants
                    else _loaded_input_scalar_bits(op.ssa, torch.bfloat16)
                )
                plan.append(("bf16", bits))
            elif ty == "f32":
                value = (
                    _parse_f32(host.constants[op.ssa][0])
                    if op.ssa in host.constants
                    else _loaded_input_scalar(op.ssa, torch.float32)
                )
                plan.append(("f32", value))
            elif ty == "f64":
                value = (
                    _parse_f64(host.constants[op.ssa][0])
                    if op.ssa in host.constants
                    else _loaded_input_scalar(op.ssa, torch.float64)
                )
                plan.append(("f64", value))
            else:
                raise NotImplementedError(
                    f"orchestrate: unhandled launch-arg type {ty!r} (operand {op.ssa})"
                )

        launcher = PreparedLauncher(func, grid, block, plan, shared=shared)
        launcher.bind_input_scalars(launch_tensors)
        prepared_sites.append((launcher, tuple(launch_tensors)))

    # --- collect outputs --------------------------------------------------- #
    # One ``out{i}`` per function result, in return order (multi-result programs,
    # e.g. a loop body yielding several carries, return them all).
    outputs: dict[str, "torch.Tensor"] = {}
    if output_aligned_ptrs:
        for i, ssa in enumerate(output_aligned_ptrs):
            if ssa in buffer_tensor_by_ssa:
                outputs[f"out{i}"] = buffer_tensor_by_ssa[ssa]
            elif ssa in host.aligned_ptr_blockargs:
                # an input is returned as-is (identity program)
                outputs[f"out{i}"] = input_tensor_by_index[host.aligned_ptr_blockargs[ssa]]
            else:
                raise NotImplementedError(
                    f"orchestrate: returned aligned ptr {ssa} is not a known buffer/input"
                )
    else:
        if intent_results:
            raise NotImplementedError(
                "orchestrate: compiled module has declared results but no "
                "explicit returned or destination-passing output pointer"
            )
        if buffer_tensor_by_ssa:
            raise NotImplementedError(
                "orchestrate: compiled module has unbound buffers but no "
                "declared output binding"
            )
    if intent_results:
        if len(outputs) != len(intent_results):
            raise RuntimeError(
                "orchestrate: recovered output count does not match P4 signature: "
                f"runtime={len(outputs)}, declared={len(intent_results)}"
            )
        for index, (shape, elem) in enumerate(intent_results):
            key = f"out{index}"
            tensor = outputs[key]
            expected_dtype = _torch_dtype(elem)
            if tensor.dtype != expected_dtype:
                raise RuntimeError(
                    f"orchestrate: output {index} dtype {tensor.dtype} does not "
                    f"match declared {expected_dtype}"
                )
            expected_numel = 1
            for dim in shape:
                expected_numel *= dim
            if tensor.numel() != expected_numel:
                raise RuntimeError(
                    f"orchestrate: output {index} has {tensor.numel()} elements, "
                    f"declared shape {shape} requires {expected_numel}"
                )
            outputs[key] = tensor.reshape(shape)
    prepared = PreparedGraphProgram(
        prepared_sites,
        outputs,
        PreparedGraphRecord(
            launch_count=len(host.launches),
            kernel_count=len(kernels),
            intermediate_buffer_count=len(host.loose_buffers),
            output_count=len(outputs),
        ),
    )
    if _prepare_only:
        return prepared
    prepared.thunk()()
    torch.cuda.synchronize()
    return outputs


def prepare_program(
    intent_mlir_path,
    inputs,
    compile_all=compile_all_kernels,
    output_aliases=None,
) -> PreparedGraphProgram:
    """Compile/bind once and return a no-allocation whole-graph launcher."""

    result = run_program(
        intent_mlir_path,
        inputs,
        compile_all=compile_all,
        output_aliases=output_aliases,
        _prepare_only=True,
    )
    if not isinstance(result, PreparedGraphProgram):
        raise RuntimeError("orchestrate did not return a prepared graph")
    return result


__all__ = [
    "PreparedGraphProgram",
    "PreparedGraphRecord",
    "prepare_program",
    "run_program",
]
