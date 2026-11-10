#ifndef INTENT_OPS
#define INTENT_OPS

include "IntentDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Tensor-level atomic ops (Layer A semantics, Layer B constraints as attrs).
// Starter set: map_binary / reduce / contraction. More ops (scan, gather,
// scatter, loop, ...) follow the same declaration and verifier pattern below.
//===----------------------------------------------------------------------===//

def Intent_MapBinaryOp : Intent_Op<"map_binary"> {
  let summary = "Elementwise binary map (Layer A).";
  let description = [{
    Elementwise binary op over two ranked tensors. `fn` selects the
    pointwise operation (add/sub/mul/div/floordiv/remainder/max/min/pow/atan2/
    left_shift/right_shift/copysign). The comparison fns
    (gt/ge/lt/le/eq/ne), logical fns (logical_and/logical_or/logical_xor and
    their and/or/xor aliases), and isclose produce an `i1`-typed result tensor
    (feeding `intent.select`); every other fn is element-type-preserving.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs, StrAttr:$fn);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_SelectOp : Intent_Op<"select"> {
  let summary = "Elementwise select (Layer A): where(cond, x, y).";
  let description = [{
    Per-element `cond ? x : y` (numpy `where`). `cond` is an `i1`-typed ranked
    tensor; `x`/`y`/result share one element type. Shapes broadcast numpy-style
    (e.g. an `[M,N]` `i1` condition selecting between two rank-0 scalars), so the
    verifier constrains element types only, not shapes. Owned by M3; reused by
    M4/M6/M7/M9 (do not re-add). Lowered via `arith.select` over the scalar
    condition produced by a `map_binary` comparison fn.
  }];
  let arguments = (ins AnyRankedTensor:$cond, AnyRankedTensor:$x,
                       AnyRankedTensor:$y);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$cond `,` $x `,` $y attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_IotaOp : Intent_Op<"iota", [Pure]> {
  let summary = "Column-index vector (Layer A): [0, 1, ..., N-1].";
  let description = [{
    Materializes the rank-1 integer index vector `0,1,...,N-1` (numpy
    `arange`), where `N` is the static extent of the result tensor. The
    canonical source of a per-column position used to build a one-hot / gather
    condition (`iota == target`) for cross_entropy's target-logit pick. No
    operands; the result type encodes both `N` and the integer width. In a
    block-cooperative lowering the value for column `col` resolves to `col`
    itself (an `arith.index_cast` of the thread's column index) — there is no
    materialized vector. Owned by M6; reused wherever a column index is needed.
  }];
  let arguments = (ins);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Intent_MapUnaryOp : Intent_Op<"map_unary"> {
  let summary = "Elementwise unary map (Layer A).";
  let description = [{
    Elementwise unary op over one ranked tensor. `fn` selects the pointwise
    operation. Most fns are shape- AND element-type-preserving
    (identity/exp/log/log2/neg/abs/sqrt/rsqrt/recip/tanh/sigmoid/cos/sin/tan/
    exp2/atan/erf/acos/asin/cbrt/ceil/floor/round/roundeven/trunc/sign/log10/
    log1p/expm1/erfc), the same
    class the transcendental peephole turns into NVVM approx intrinsics.
    Predicate fns (`isnan`) instead produce an `i1`-typed result tensor (the
    boolean condition for `intent.select`, mirroring the `map_binary` comparison
    fns); `logical_not` is the `i1`->`i1` complement. Shape-preserving throughout.
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$fn);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_CastOp : Intent_Op<"cast"> {
  let summary = "Elementwise dtype cast (Layer A).";
  let description = [{
    Element-type conversion (numpy `astype`) of one ranked tensor to `to_dtype`.
    Shape-preserving; only the element type changes. Lowers to the matching
    `arith` conversion for the (source, target) pair — `truncf`/`extf` between
    float widths, `fptosi`/`sitofp` across the float/int boundary, `trunci`/
    `extsi` between integer widths — and folds to an identity alias when source
    and target element types are already equal (the f32-only corpus case).
    The M3b-deferred op, landed by C0 so the dtype-convert (`cast`/`to`) family
    extracts + lowers.
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$to_dtype);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ScanOp : Intent_Op<"scan"> {
  let summary = "Inclusive cumulative scan along one axis (Layer A).";
  let description = [{
    Inclusive cumulative scan of `input` along `axis` with `fn`
    (`sum` = cumsum, `max` = cummax, `min` = cummin). Shape-preserving: the
    result has the same shape and element type as the input;
    `result[..., i, ...] = combine(input[..., 0..i, ...])`. `max`/`min` (C6) reuse
    the SAME Hillis-Steele block scan as `sum`, only the combine changes.
    Unlike `intent.reduce` (which collapses the axis to a single value) and
    `intent.loop` (which yields only the final carry), scan keeps the per-element
    running value — the primitive sparsemax's `cumsum` of the sorted row needs.
    Owned by M4. In a block-cooperative lowering it lowers to a single-block
    inclusive scan over a contiguous column-to-thread tiling (per-thread
    sequential prefix + a cross-thread exclusive scan of the segment totals
    through `gpu.dynamic_shared_memory` + `gpu.barrier`).
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$fn, I64Attr:$axis);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ArgReduceOp : Intent_Op<"arg_reduce"> {
  let summary = "Index of the extremum along one axis (Layer A): argmax/argmin.";
  let description = [{
    Reduce `input` along `axis` to the INTEGER INDEX of the extremum
    (torch.argmax / torch.argmin), NOT the extremum value. `fn` selects
    `argmax`/`argmin`. Ties resolve to the FIRST (lowest) index, matching torch.
    The result is an integer-typed tensor with `axis` collapsed (rank-1 `[M]` for
    a `[M,N]` row argmax, or keepdims `[M,1]`). The C6 order-statistics mechanism:
    the block reduction carries a `(best_val, best_idx)` PAIR — a manual argmax
    tree over the register tile + a shared-memory tree keeping the max/min value's
    column index — and stores the integer index. Distinct from `intent.reduce`
    (returns the value) and `intent.scan` (running value). `axis` must index a
    static extent.
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$fn, I64Attr:$axis);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_SortOp : Intent_Op<"sort"> {
  let summary = "Full sort along the last axis (Layer A): values + argsort indices.";
  let description = [{
    Sort each row of `input` along `axis` (the last axis) and produce BOTH the
    sorted `values` and the `indices` (argsort permutation) — torch.sort /
    numpy.sort + argsort. `descending` selects the direction. The two results are
    always present; a caller returns whichever it needs (values = sort, indices =
    argsort, both = sort-with-indices). When the result `axis` extent `W` is
    SMALLER than the input extent `N`, only the first `W` sorted entries are kept
    — that is `topk` (`descending` = largest-first). Quantile composes on top
    (sort ascending, then a fractional-rank gather with linear interpolation).

    The C6 HARD-tier order-statistics mechanism, the last order-stats gap: a
    single-block BLOCK BITONIC SORT. One block per row loads the row into a
    dynamic-shared buffer padded up to the next power of two `P` with a `+inf`
    (ascending) / `-inf` (descending) sentinel so the pad sorts to the END, then
    runs the bitonic compare-exchange network — `log2(P)*(log2(P)+1)/2`
    barrier-separated stages, each a `gpu.barrier` then a paired compare-swap of
    partners `i` and `i^j` under the direction bit `(i & k)==0`. For the index
    result a parallel `i32` index buffer is swapped in lockstep. Correctness-only
    (a valid permutation reproducing the sorted values; ties are unstable, like
    torch's default sort). `axis` must be the last axis and index a static extent.
  }];
  let arguments = (ins AnyRankedTensor:$input, BoolAttr:$descending,
                       I64Attr:$axis);
  let results = (outs AnyRankedTensor:$values, AnyRankedTensor:$indices);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_GatherOp : Intent_Op<"gather"> {
  let summary = "Data-dependent gather along one axis (Layer A).";
  let description = [{
    Gather elements of `data` at runtime positions read from an integer `index`
    tensor along `axis`. Two `mode`s cover the data-dependent-addressing family:

    * `"along_axis"` (torch.gather / numpy `take_along_axis`): the canonical J
      source contract accepts either an index with the same rank as `data`, or
      a compressed index with the selected axis omitted (rank `data.rank - 1`).
      In both cases the result is shaped like the source index.  The latter
      form is expanded with a unit selected axis before this Intent MLIR op is
      emitted, so the dialect verifier receives an equal-rank index and result.
      `result[i..] = data[i.. with i[axis] replaced by index[i..]]`.
    * `"take"` (index_select / embedding row-gather / numpy `take`): `index` is a
      rank-r selector inserted at `axis`; the result rank is
      `data.rank - 1 + index.rank` and
      `result[pre.., k.., post..] = data[pre.., index[k..], post..]`. An
      embedding is exactly `take` with a rank-1 `[N]` index into axis 0 of the
      `[V,D]` weight (`result[n,d] = weight[index[n], d]`).

    The C0/C1 cluster's new mechanism: the load address is a RUNTIME value (read
    from `index`), not an affine function of the iteration — the GPU lowering
    emits a data-dependent `memref.load` at the index-computed coordinate. The
    result element type equals `data`'s; `index` must be integer-typed.
  }];
  let arguments = (ins AnyRankedTensor:$data, AnyRankedTensor:$index,
                       I64Attr:$axis, StrAttr:$mode);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$data `,` $index attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ScatterOp : Intent_Op<"scatter"> {
  let summary = "Data-dependent logical update along one axis (Layer A).";
  let description = [{
    Write `src` elements into a copy of `data` at runtime positions read from an
    integer `index` tensor along `axis` (torch.scatter / scatter_add_ /
    numpy `put_along_axis`). `mode` is `"along_axis"`: `index` and `src` share a
    shape, and for each element `out[i.. with i[axis]=index[i..]] (op)= src[i..]`
    where the combine `op` is selected by `reduce`:

    * `"none"` — overwrite;
    * `"add"`  — accumulate every logical contribution.

    This Layer-A op does not by itself complete duplicate/conflict semantics.
    Canonical J binds it to typed Layer-B bounds, duplicate/order policy and an
    exact/tolerance/one-of-updates result relation. In particular floating add
    is not declared bitwise order-independent, and overwrite races are not
    assigned an arbitrary deterministic winner here.

    When `atomic` is present it is either `"add"` or `"cas"`; the operation
    must also carry one of `relaxed`, `acquire`, `release`, `acq_rel`, or
    `seq_cst` in `ordering`, and one of `work_item`, `subgroup`, `workgroup`,
    `device`, or `system` in `memory_scope`. Layer-B
    `realization_authorizations` must contain the matching `native_atomic`
    authorization for this scatter's stable J node handle.

    The result has `data`'s shape and element type. The legacy GPU lowering
    treats an unannotated scatter as an in-place update of the (caller-cloned)
    `data` buffer: a data-dependent `memref.store` (reduce `"none"`) or
    `memref.atomic_rmw addf` (reduce `"add"`) at the index-computed coordinate.
    An explicit atomic scatter is lowered through the target's scoped atomic
    operation and ordering fences instead. `index` must be integer-typed.
  }];
  let arguments = (ins AnyRankedTensor:$data, AnyRankedTensor:$index,
                       AnyRankedTensor:$src, I64Attr:$axis, StrAttr:$mode,
                       StrAttr:$reduce, OptionalAttr<StrAttr>:$atomic,
                       OptionalAttr<StrAttr>:$ordering,
                       OptionalAttr<StrAttr>:$memory_scope);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$data `,` $index `,` $src attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_AtomicScatterOp : Intent_Op<"atomic_scatter"> {
  let summary = "Explicit compare-exchange scatter with observable old value.";
  let description = [{
    Internal Layer-A form for a J scatter with `atomic = "cas"`. `compare`
    supplies the expected value and `src` supplies the replacement value for
    each indexed update. The first result is the updated data state (the CUDA
    implementation uses the data buffer in place); the second result is the
    value observed by the compare-exchange. `cas_retry` is explicit so a
    consumer cannot silently replace a retrying compare-exchange with a single
    attempt.
  }];
  let arguments = (ins AnyRankedTensor:$data, AnyRankedTensor:$index,
                       AnyRankedTensor:$compare, AnyRankedTensor:$src,
                       I64Attr:$axis, StrAttr:$mode, StrAttr:$atomic,
                       StrAttr:$ordering, StrAttr:$memory_scope,
                       OptionalAttr<StrAttr>:$cas_retry);
  let results = (outs AnyRankedTensor:$updated, AnyRankedTensor:$observed);
  let assemblyFormat =
      "$data `,` $index `,` $compare `,` $src attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Ragged descriptor ops (Layer A). A ragged value is represented in MLIR as a
// builtin tuple<offsets, indices>; both fields are rank-1 Ragged carrier
// tensors: i8/i16/i32/i64, ui8/ui16/ui32, or logical index (materialized as
// i64). bool/i1, ui64, explicit signed integer types, and other widths are not
// valid Ragged carriers.
// Keeping the descriptor as one typed result/operand preserves the J-level
// value graph while giving C++ lowerings a concrete type they can inspect and
// destructure. The verifiers below enforce the field and result contracts.
//===----------------------------------------------------------------------===//

def Intent_RaggedOp : Intent_Op<"ragged", [Pure]> {
  let summary = "Package offsets and member indices into a ragged descriptor.";
  let description = [{
    Package a rank-1 Ragged carrier `offsets` tensor and a rank-1 Ragged carrier
    `indices` tensor into one typed ragged descriptor. The descriptor is represented by
    the builtin MLIR type `tuple<offsets_type, indices_type>`; the tuple fields
    must exactly match the two operands. This operation has no attributes.
  }];
  let arguments = (ins AnyRankedTensor:$offsets, AnyRankedTensor:$indices);
  let results = (outs AnyTuple:$result);
  let assemblyFormat =
      "$offsets `,` $indices attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_RaggedOuterOp : Intent_Op<"ragged_outer", [Pure]> {
  let summary = "Return the number of outer groups in a ragged descriptor.";
  let description = [{
    Return a rank-0 Ragged carrier tensor containing the number of outer groups in a
    ragged descriptor. The result element type matches the descriptor's
    offsets element type. This operation has no attributes.
  }];
  let arguments = (ins AnyTuple:$ragged);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$ragged attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_RaggedMemberOp : Intent_Op<"ragged_member", [Pure]> {
  let summary = "Return the member-index vector for one ragged outer group.";
  let description = [{
    Select the member-index vector for one outer group from `ragged`, using a
    rank-0 Ragged carrier `index`. The result is a dynamic-length rank-1 Ragged carrier
    tensor (`tensor<?x...>`), with element type equal to the descriptor's
    indices field type. This operation has no attributes.
  }];
  let arguments = (ins AnyTuple:$ragged, AnyRankedTensor:$index);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$ragged `,` $index attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Dynamic-shape output ops (Layer A). The C8 cluster — the ONE genuinely-new
// mechanism vs every prior cluster: a data-dependent output EXTENT. The output
// row count `nnz` (number of true mask / nonzero elements) depends on the input
// VALUES, unknown at compile time. All three share the compaction primitive: an
// exclusive prefix-sum of the flattened 0/1 mask gives each true element its
// compacted output SLOT (row-major order, matching torch); the total `nnz` is
// the inclusive sum. Lowered by `convert-intent-compact-to-gpu`; the runtime
// extent is resolved host-side (a reduce over the mask) before the compaction
// launch, so the destination buffer's leading dim is concrete at launch time.
// See the operation declaration and verifier below (C8).
//===----------------------------------------------------------------------===//

def Intent_NonzeroOp : Intent_Op<"nonzero"> {
  let summary = "Row-major multi-indices of the nonzero elements (Layer A).";
  let description = [{
    Materialize the multi-indices of the nonzero elements of `cond`
    (torch.nonzero / numpy.argwhere): a rank-2 integer `[nnz, ndim]` result whose
    row `k` is the `ndim`-vector coordinate of the `k`-th nonzero element in
    row-major (C-order) scan order, where `ndim = cond.rank`. An element counts
    as nonzero when it is `!= 0` (the `i1` `true`, or any nonzero numeric value).

    The C8 dynamic-shape mechanism: the leading extent `nnz` is DATA-DEPENDENT
    (the count of nonzero elements), unknown at compile time — so the result type
    carries a dynamic leading dim `tensor<?x{ndim}xi64>`. The GPU lowering
    compacts each true element to its row-major slot (an exclusive prefix-sum of
    the flattened mask) and writes the unraveled coordinate; the host sizes the
    `[nnz, ndim]` allocation from a reduce over `cond` before the launch. Row
    order is exact (matches torch.nonzero).
  }];
  let arguments = (ins AnyRankedTensor:$cond);
  let results = (outs AnyRankedTensor:$indices);
  let assemblyFormat = "$cond attr-dict `:` functional-type($cond, $indices)";
  let hasVerifier = 1;
}

def Intent_MaskedSelectOp : Intent_Op<"masked_select"> {
  let summary = "Compact the masked-true elements into a rank-1 vector (Layer A).";
  let description = [{
    Select the elements of `data` where `mask` is true, flattened in row-major
    order into a rank-1 `[nnz]` result (torch.masked_select / numpy boolean
    index `data[mask]`). `mask` shares `data`'s shape; `nnz` is the number of true
    mask elements — the same DATA-DEPENDENT extent as `intent.nonzero`, so the
    result type is `tensor<?x{elt}>`. The GPU lowering compacts each masked-true
    element to its exclusive-prefix-sum slot (row-major, exact vs torch); the host
    sizes the `[nnz]` allocation from a reduce over `mask`.
  }];
  let arguments = (ins AnyRankedTensor:$data, AnyRankedTensor:$mask);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$data `,` $mask attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_MaskedScatterOp : Intent_Op<"masked_scatter"> {
  let summary = "Fill the masked positions of data with src, in order (Layer A).";
  let description = [{
    Copy `data`, then overwrite each position where `mask` is true with the next
    element of the rank-1 `src` vector, consumed in row-major order
    (torch.Tensor.masked_scatter): `out[coord] = mask[coord] ? src[rank(coord)] :
    data[coord]`, where `rank(coord)` is the exclusive prefix-sum of the mask (the
    count of true elements before `coord`). The result is DATA-SHAPED (same shape
    and element type as `data`) — NOT dynamic-extent — but it shares the C8
    compaction primitive: the prefix-sum of the mask indexes `src`. `mask` shares
    `data`'s shape; `src` must supply at least `nnz` elements.
  }];
  let arguments = (ins AnyRankedTensor:$data, AnyRankedTensor:$mask,
                       AnyRankedTensor:$src);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$data `,` $mask `,` $src attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Structural layout ops (Layer A). Pure DATA MOVEMENT: every output element is a
// source element chosen by a STATIC structural rule (an affine index / a
// piecewise-by-offset select) or a pad constant — no arithmetic. Only
// broadcast/reshape were implicit before; these make transpose/concat/pad/tile/
// flip/triu (permute, cat/stack/hstack/vstack, pad, repeat/tile, flip, triu)
// first-class. Lowered by
// `convert-intent-layout-to-gpu`: one block per output row, cols over threads,
// each output element's SOURCE coord computed by the rule then `memref.load`ed
// (or set to the pad constant). See the operation declaration below (C4).
//===----------------------------------------------------------------------===//

def Intent_ReshapeOp : Intent_Op<"reshape", [Pure]> {
  let summary = "Reshape a tensor without changing its element order (Layer A).";
  let description = [{
    Reinterpret the row-major element sequence of `input` with the static result
    shape carried by `shape`. The operation preserves element type and total
    element count; symbolic or runtime shape expressions are resolved by the
    Layer-A frontend before this static MLIR leg is emitted.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_TransposeOp : Intent_Op<"transpose"> {
  let summary = "Permute tensor dimensions by a static permutation (Layer A).";
  let description = [{
    Apply the numpy/torch transpose convention: result dimension `i` is input
    dimension `perm[i]`, and `out[o0,...]` reads the input coordinate obtained
    by applying the inverse permutation. `perm` must contain every axis exactly
    once. This is pure data movement and preserves the element type.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$perm);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ConcatOp : Intent_Op<"concat"> {
  let summary = "Variadic concatenation along one axis (Layer A).";
  let description = [{
    Place `inputs` end-to-end along `axis` at their cumulative offsets
    (numpy `concatenate` / torch.cat). All inputs share the result's rank and
    element type and agree on every non-`axis` extent; the result `axis` extent is
    the sum of the input `axis` extents. `hstack`/`vstack` are concat on axis 1/0;
    `stack` is concat after an implicit unsqueeze of each input at `axis`.

    The one genuinely-new bit vs. the affine layout ops is the VARIADIC + offset
    placement: `out[coord]` is owned by exactly one input, selected piecewise by
    the cumulative offsets, and loaded from that input at the offset-shifted coord.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs, I64Attr:$axis);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$inputs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_PadOp : Intent_Op<"pad"> {
  let summary = "Constant pad along every axis (Layer A).";
  let description = [{
    Pad `input` with `lo[d]` elements before and `hi[d]` after along each dim `d`,
    filling the border with the constant `value` (numpy `pad` mode="constant" /
    torch `constant_pad_nd`). Result extent `= input[d] + lo[d] + hi[d]`.
    `out[coord] = in-bounds(coord) ? input[coord - lo] : value` — a piecewise
    select on the per-dim in-bounds predicate plus the pad constant.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$lo,
                       I64ArrayAttr:$hi, F64Attr:$value);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_TileOp : Intent_Op<"tile"> {
  let summary = "Tile (repeat the whole tensor) along every axis (Layer A).";
  let description = [{
    Repeat `input` `reps[d]` times along each dim `d` (numpy `tile` / torch.tile /
    torch.Tensor.repeat). Result extent `= input[d] * reps[d]` and
    `out[coord] = input[coord mod input_extent]` — a per-dim affine `remui` back
    into the source, no guard (every output coord maps to a valid source).
  }];
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$reps);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_FlipOp : Intent_Op<"flip"> {
  let summary = "Reverse the element order along the given axes (Layer A).";
  let description = [{
    Reverse `input` along each dim in `axes` (numpy `flip` / torch.flip).
    Shape-preserving; `out[.., i, ..] = input[.., N-1-i, ..]` for each flipped
    axis (extent `N`) — a per-dim affine `N-1-i` remap of the source coord.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$axes);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_TriuOp : Intent_Op<"triu"> {
  let summary = "Upper-triangular mask above the k-th diagonal (Layer A).";
  let description = [{
    Zero the elements below the `k`-th diagonal of a rank-2 `input` (numpy `triu` /
    torch.triu): `out[i,j] = (j >= i + k) ? input[i,j] : 0`. Shape-preserving; a
    select on the iota-derived (i,j) position over the always-in-bounds load.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$k);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Stateful seeded RNG (Layer A). A counter-based, STATELESS-per-element philox
// hash: the value at output element linear-index `i` is a pure function of
// (seed, i) — no visited-order dependence, no global RNG state. `intent.rand`
// produces uniform[0,1) f32; every other RNG op (dropout / randn / normal /
// uniform(a,b) / *_like) COMPOSES rand with the existing map/select primitives
// (dropout = select(rand >= p, x/(1-p), 0); randn = Box-Muller of two rands;
// uniform(a,b) = a + (b-a)*rand). Lowered by `convert-intent-rng-to-gpu`: one
// block per output row, each element computes philox4x32-10(seed, row*Co+col)
// -> uniform -> the composed elementwise transform -> store. See spec/ir (C5).
//===----------------------------------------------------------------------===//

def Intent_RandOp : Intent_Op<"rand"> {
  let summary = "Counter-based uniform[0,1) RNG (Layer A, stateless per element).";
  let description = [{
    Materialize a tensor of uniform[0,1) f32 samples whose element at row-major
    linear index `i` is `philox4x32-10(seed, i)` mapped to `[0,1)` — a PURE
    function of the `seed` attribute and the element index (no operands, no
    global RNG state, order-independent), so two launches with the same seed are
    bit-identical and different seeds decorrelate. The seed splits into the two
    philox key words; the counter is `(i, 0, 0, 0)`; the first output word gives
    the 24-bit-mantissa uniform. The result must be f32; its shape is carried in
    the result type (like `intent.iota`'s extent). rand_like is rand at the
    input's shape; randn / normal / uniform(a,b) / dropout are COMPOSITIONS of
    rand with map_unary / map_binary / select (no new op).
  }];
  let arguments = (ins I64Attr:$seed);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Windowed-reduce / interpolation / complex (Layer A). The C7 cluster: three
// distinct memory-bound mechanisms over NCHW tensors.
//   pool     — windowed reduction (max/avg) over a kHxkW window at stride s with
//              padding; max_pool2d_with_indices ALSO returns the argmax FLAT index
//              (ih*W + iw within the (H,W) plane) as an i64.
//   upsample — nearest-neighbor or bicubic resize over NCHW; the mode selects
//              either the floor index-remap or the four-tap cubic interpolation
//              described by Intent_UpsampleOp below.
//   polar    — elementwise abs*(cos angle, sin angle): a complex tensor
//              represented as a trailing real/imag PAIR (two real results),
//              reusing the C0 cos/sin scalar emitters.
// Lowered by convert-intent-{pool,upsample,polar}-to-gpu. See spec/ir (C7).
//===----------------------------------------------------------------------===//

def Intent_PoolOp : Intent_Op<"pool"> {
  let summary = "2-D windowed pooling reduce (max/avg) over NCHW (Layer A).";
  let description = [{
    Pool a rank-4 `[N,C,H,W]` `input` to `[N,C,OH,OW]` by reducing over each
    `kH x kW` window at stride `(sH,sW)` with zero-padding `(padH,padW)`
    (torch.avg_pool2d / torch.max_pool2d). For output cell `(n,c,oh,ow)` the
    window source coord is `(ih,iw) = (oh*sH - padH + kh, ow*sW - padW + kw)` for
    `kh in [0,kH)`, `kw in [0,kW)`; out-of-bounds coords are skipped:

    * `fn = "max"` — the window maximum (out-of-bounds = -inf, so ignored). When
      the optional `indices` result is present (torch.max_pool2d_with_indices) it
      carries the argmax FLAT index `ih*W + iw` (the position WITHIN the (H,W)
      plane) as i64; ties keep the first (row-major window scan) index, matching
      torch/flag_gems (strict `>` update).
    * `fn = "avg"` — the window sum divided by `kH*kW` (torch's default
      `count_include_pad=True`: padded cells count as 0 in the sum but the divisor
      is the full window size).

    The C7 windowed-reduce mechanism: one block per `(n,c)` channel-plane row (or
    per output row), each thread reduces the kHxkW window for its output columns —
    a bounded per-cell reduce, NOT a full-axis `intent.reduce`. `indices` is only
    valid for `fn = "max"`.
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$fn,
                       I64Attr:$kH, I64Attr:$kW, I64Attr:$sH, I64Attr:$sW,
                       I64Attr:$padH, I64Attr:$padW);
  let results = (outs AnyRankedTensor:$output,
                      Optional<AnyRankedTensor>:$indices);
  let assemblyFormat =
      "$input attr-dict `:` functional-type($input, results)";
  let hasVerifier = 1;
}

def Intent_UpsampleOp : Intent_Op<"upsample"> {
  let summary = "2-D upsample (nearest / bicubic) over NCHW (Layer A).";
  let description = [{
    Resize a rank-4 `[N,C,H,W]` `input` to `[N,C,OH,OW]` under `mode` and the
    positive integer spatial scale `[sh,sw]`, with `OH = H*sh` and
    `OW = W*sw`:

    * `"nearest"` (torch.upsample_nearest2d / F.interpolate mode="nearest") —
      pure index-remap: `out[n,c,oh,ow] = input[n,c, min(floor(oh*IH/OH), IH-1),
      min(floor(ow*IW/OW), IW-1)]`, the affine `floor` of the output coord with
      the reciprocal scale `IH/OH` (matching flag_gems' `(oh*IH/OH).to(int32)`
      truncation). C4-layout-like: a `memref.load`, no arithmetic on the value.

    * `"bicubic"` (torch.upsample_bicubic2d / F.interpolate mode="bicubic",
      align_corners=False) — a 4x4 cubic-convolution neighborhood (`a=-0.75`, the
      torch/PIL kernel) at the mapped source coord
      `r = (o + 0.5)*IN/OUT - 0.5`: `out = sum_{p,q} wy[p]*wx[q]*
      input[n,c, clamp(floor(ry)-1+p), clamp(floor(rx)-1+q)]` with the four
      per-axis cubic weights from the fraction `t = r - floor(r)` (edges
      clamped). Extends the C7a nearest gather with a per-tap weighted
      accumulation. On the ENLARGING path antialias is a no-op, so this matches
      `upsample_bicubic2d_aa`; the variable-support aa DOWNSAMPLE kernel is a
      follow-up. Bilinear is a documented follow-up.
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$mode,
                       I64ArrayAttr:$scale);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$input attr-dict `:` functional-type($input, $result)";
  let hasVerifier = 1;
}

def Intent_PolarOp : Intent_Op<"polar"> {
  let summary = "Complex construction from magnitude+angle (Layer A).";
  let description = [{
    Build a complex tensor from a magnitude `abs` and a phase `angle`
    (torch.polar): `real = abs*cos(angle)`, `imag = abs*sin(angle)`. The complex
    result is represented as a trailing real/imag PAIR — two real-typed results
    `real` and `imag` of the same shape/element type as `abs` — so a caller
    compares against `torch.view_as_real(torch.polar(abs, angle))[..., 0]` and
    `[..., 1]`. Elementwise, reusing the C0 `cos`/`sin` scalar emitters (the NVVM
    `cos.approx`/`sin.approx` hardware ops) + `mul`; verified rel < 1e-3.
  }];
  let arguments = (ins AnyRankedTensor:$abs, AnyRankedTensor:$angle);
  let results = (outs AnyRankedTensor:$real, AnyRankedTensor:$imag);
  let assemblyFormat =
      "$abs `,` $angle attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ConvOp : Intent_Op<"conv"> {
  let summary = "N-D cross-correlation convolution (conv1d/2d/3d) over NC*spatial.";
  let description = [{
    Forward convolution (torch.nn.functional.conv1d/conv2d/conv3d) of a rank
    `2+R` `input` `[N, Cin, *spatial]` with a rank `2+R` `weight`
    `[Cout, Cin/groups, *kernel]` and an optional `bias` `[Cout]`, producing
    `[N, Cout, *out_spatial]`. `R` (the spatial rank, 1/2/3) is `input.rank - 2`;
    `stride`, `padding`, `dilation` are length-`R` integer arrays and `groups`
    partitions the channels. The output spatial extent along axis `j` is
    `O_j = (H_j + 2*pad_j - dil_j*(k_j - 1) - 1) / stride_j + 1`.

    Cross-correlation (no kernel flip, matching torch): for output cell
    `(n, co, *o)`,
      `out = bias[co] + sum_{ci in group(co)} sum_{*k}
             input[n, ci, *(o*stride - pad + k*dil)] * weight[co, ci_local, *k]`
    where out-of-bounds source coords are skipped (implicit zero pad). Channel
    grouping: group `g = co / (Cout/groups)` selects input channels
    `[g*(Cin/groups), (g+1)*(Cin/groups))`; `ci_local` is `ci - g*(Cin/groups)`.

    The C7 convolution mechanism = the windowed per-output-cell accumulation of
    `intent.pool` extended with an input-channel sum and a per-tap weight
    multiply (correctness-only; direct windowed-weighted-accumulation, no
    im2col). `bias` is optional; `Cin` must be divisible by `groups` and
    `Cout` must match `weight` dim 0.
  }];
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight,
                       Optional<AnyRankedTensor>:$bias,
                       I64ArrayAttr:$stride, I64ArrayAttr:$padding,
                       I64ArrayAttr:$dilation, I64Attr:$groups);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat =
      "$input `,` $weight (`,` $bias^)? attr-dict `:` "
      "functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ReduceOp : Intent_Op<"reduce"> {
  let summary = "Reduction along axes (Layer A semantics + Layer B associativity).";
  let description = [{
    Reduce `input` along `axes` with `fn` (sum/max/min/prod/any/all).
    `assoc` records reduction associativity for verification:
    `associative` (bit-exact) or `float_nondeterministic` (tolerance).
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       StrAttr:$fn,
                       I64ArrayAttr:$axes,
                       StrAttr:$assoc);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Intent_ContractionOp : Intent_Op<"contraction"> {
  let summary = "Einsum-style contraction (Layer A).";
  let description = [{
    General tensor contraction with einsum-style `subscripts`
    (e.g. "bhqd,bhkd->bhqk"). Covers matmul / batched-matmul / dot.
  }];
  let arguments = (ins AnyRankedTensor:$a, AnyRankedTensor:$b, StrAttr:$subscripts);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$a `,` $b attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Structured loop with loop-carried state (Layer A recurrence + Layer C tiling).
// FlashAttention's K-loop: carries the online-softmax state (running max, sum,
// accumulator) while scanning K/V tiles. See spec/ir/structured-loop.md.
//===----------------------------------------------------------------------===//

def Intent_YieldOp : Intent_Op<"yield", [Pure, Terminator]> {
  let summary = "Terminator of an intent.loop body; yields the next carries.";
  let description = [{
    Yields the loop-carried values for the next trip iteration (and, after the
    last iteration, the loop results). Operand count/types match the parent
    `intent.loop` carries.
  }];
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "($results^ `:` type($results))? attr-dict";
  // No-operand builder so SingleBlockImplicitTerminator can materialize the
  // implicit terminator (mirrors scf.yield).
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let hasVerifier = 1;
}

def Intent_LoopOp : Intent_Op<"loop",
    [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"YieldOp">,
     RecursiveMemoryEffects]> {
  let summary = "Structured loop with loop-carried state and scanned inputs.";
  let description = [{
    A structured loop (an explicit induction variable, loop-carried `inits`, and
    a single-block body region) that either scans `scan_inputs` tile-by-tile along
    their `scan_axes`, or uses the dynamic `upper_bound` form for a runtime-sized
    value loop. The static form is the IntentIR primitive for FlashAttention's
    K-loop: `inits` = (running max m, running sum l, accumulator Oacc); each trip
    slices a tile (`tile_hint` rows, Layer C) of K and V; the body recomputes the
    online-softmax update and `intent.yield`s the new carries. The dynamic form
    has no scan inputs, carries one rank-0 integer `upper_bound` operand, and
    exposes a rank-0 integer induction tensor as the first body argument after
    the scan slice arguments (there are no scan slice arguments in this form).
    Its induction value is the J `trip.index` binding and is not a carry.

    Body block arguments are the per-trip scan slices (one per `scan_inputs`, in
    order), followed by the dynamic induction tensor when `upper_bound` is
    present, and then the current carries (one per `inits`). Results are the
    final carries; result types equal `inits` types (carry types are invariant).

    Layer attribution: the recurrence is Layer A (algorithmic intent); `tile_hint`
    is a non-binding Layer C schedule hint (the back-end may retune it).
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$scan_inputs,
                       Variadic<AnyType>:$inits,
                       Optional<AnyRankedTensor>:$upper_bound,
                       I64ArrayAttr:$scan_axes,
                       OptionalAttr<I64Attr>:$tile_hint);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `scan` `(` ($scan_inputs^ `:` type($scan_inputs))? `)`
    `init` `(` ($inits^ `:` type($inits))? `)`
    (`upper_bound` `(` $upper_bound^ `:` type($upper_bound) `)`)?
    attr-dict (`->` type($results)^)? `do` $body
  }];
}

#endif // INTENT_OPS
