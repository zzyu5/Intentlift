"""Emit `intent`-dialect MLIR from an IntentIR JSON function.

This is the ``JSON -> intent MLIR dialect`` front-end leg. It walks the
:class:`~intentir.ir.model.IntentFunction` body generically (it is driven by the
op list, not hard-coded to FlashAttention) and prints `intent`-dialect text at
concrete shapes, doing einsum/broadcast/reduce shape inference that matches the
archived NumPy interpreter (``intentir.verify.back.legacy_v0.interpret``).

Two artifacts are produced from the *same* JSON:

* :func:`emit_loop_module` — the whole function, with the structured ``intent.loop``
  carrying the online-softmax state. Proves the JSON is representable in the real
  dialect and lowers to standard MLIR (``-convert-intent-to-linalg`` -> ``scf.for``).
* :func:`emit_body_function` — the loop *body* as a standalone ``func.func`` whose
  args are the scan tiles + carries. This is the executable artifact: a host
  driver runs it once per K-tile (carries in persistent CUDA buffers), exactly
  the non-fused tiled FlashAttention that runs on the 5090D.

Instantiation: symbolic dims are bound to concrete ints; a configurable number of
leading **batch** dims (``fold`` — B,H for the flash JSON, asserted size 1) are
folded out so a single attention head is emitted as 2-D ops (the proven
executable form). Scalar inputs (``scale``) are emitted as rank-0 tensors while
retaining their declared element type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any
from intentir.ir.intent_op_contract import (
    ATOMIC_CAS_RETRIES,
    ATOMIC_KINDS,
    ATOMIC_MEMORY_SCOPES,
    ATOMIC_ORDERINGS,
    CANONICAL_DTYPES,
    FLOATING_DTYPES,
    INTEGER_DTYPES,
    IntentOpContractError,
    MAP_BINARY_FNS,
    MAP_BINARY_PREDICATE_FNS,
    MAP_UNARY_FNS,
    MAP_UNARY_PREDICATE_FNS,
    RAGGED_OPS,
    _broadcast_projection,
    canonicalize_dtype,
    map_binary_result_dtype,
    map_unary_result_dtype,
    promote_dtype,
    validate_const_literal,
)
from intentir.ir.dim_expr import DimExprError, parse_dimension_expr
from intentir.ir.model import (
    IntentFunction,
    IntentOp,
    intent_op_producers,
    validate_loop_wiring,
    validate_ragged_region_boundaries,
    walk_intent_op_scopes,
    walk_intent_ops,
)

IndexBoundsProjection = frozenset[str]

_INVALID_MLIR_SYMBOL_CHAR = re.compile(r"[^A-Za-z0-9_]")


def _sanitize_mlir_symbol(name: str) -> str:
    """Return a deterministic bare MLIR symbol derived from ``name``.

    The emitted symbol is deliberately stricter than MLIR's bare-symbol grammar:
    NVPTX reuses it as a PTX ``.entry`` name, where punctuation such as ``.`` or
    ``$`` can make ptxas reject otherwise valid MLIR.  Keep ASCII letters,
    digits and ``_`` only, then add a leading ``_`` when needed (including the
    empty name).
    """

    symbol = _INVALID_MLIR_SYMBOL_CHAR.sub("_", name)
    if not symbol or not re.match(r"[A-Za-z_]", symbol[0]):
        symbol = "_" + symbol
    return symbol


def _resolve(shape, extents) -> list[object]:
    resolved: list[object] = []
    for value in shape:
        if value == "?":
            resolved.append("?")
        else:
            resolved.append(_resolve_int(value, extents))
    return resolved


def _fold_prefix_len(
    shape: Sequence[object],
    fold: int,
    *,
    preserve_rank_one: bool = False,
) -> int:
    """Return the number of leading batch dimensions that are actually folded.

    ``fold`` is a batch projection, not a blind list slice.  A rank smaller
    than the requested projection has no foldable prefix and therefore keeps
    its original rank.  Ragged carrier vectors are always semantic rank-one
    values, so they opt out even when ``fold == 1``.
    """

    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise ValueError("emit_mlir: fold must be a non-negative integer")
    if fold == 0 or len(shape) < fold:
        return 0
    if preserve_rank_one and len(shape) == 1:
        return 0
    return fold


def _fold_shape(
    shape: Sequence[object],
    fold: int,
    *,
    preserve_rank_one: bool = False,
) -> list[object]:
    """Project a shape through the batch fold without changing its ABI class."""

    prefix = _fold_prefix_len(
        shape,
        fold,
        preserve_rank_one=preserve_rank_one,
    )
    return list(shape[prefix:])


def _fold_axis(axis: int, rank: int, fold: int) -> int:
    """Project one authored axis through a leading-batch fold.

    ``rank`` is the authored/raw rank, never the already folded rank.  Axis
    normalization happens before projection, so negative axes keep their J
    meaning.  An axis in the folded batch prefix is not representable by the
    single-head MLIR projection and is rejected instead of being silently
    shifted to a different dimension.
    """

    if isinstance(axis, bool) or not isinstance(axis, int):
        raise ValueError("emit_mlir: axis must be an integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("emit_mlir: source rank must be a non-negative integer")
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(
            f"emit_mlir: axis {axis} is outside authored rank {rank}"
        )
    prefix = _fold_prefix_len(range(rank), fold)
    if normalized < prefix:
        raise ValueError(
            f"emit_mlir: axis {axis} selects folded batch dimension "
            f"{normalized} of authored rank {rank}"
        )
    folded_rank = rank - prefix
    projected = normalized - prefix
    if projected < 0 or projected >= folded_rank:
        raise ValueError(
            f"emit_mlir: projected axis {projected} is outside folded rank "
            f"{folded_rank}"
        )
    return projected


_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


def _resolve_int(value, extents: dict[str, int] | None = None) -> int:
    """Resolve one checked integer dimension expression."""

    if isinstance(value, bool):
        raise TypeError("boolean is not an integer extent")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return int(value)
    bindings = dict(extents or {})
    try:
        expression = parse_dimension_expr(
            value,
            dimensions=frozenset(bindings),
            context=f"integer extent expression {value!r}",
        )
        resolved = expression.evaluate(
            bindings,
            context=f"integer extent expression {value!r}",
        )
    except DimExprError as exc:
        raise ValueError(str(exc)) from exc
    if resolved < _I64_MIN or resolved > _I64_MAX:
        raise ValueError(f"integer extent expression is outside int64: {value!r}")
    return resolved


def _fold_permutation(values, fold: int, extents) -> list[int]:
    perm = [_resolve_int(value, extents) for value in values]
    prefix = _fold_prefix_len(perm, fold)
    if prefix == 0:
        return perm
    if perm[:prefix] != list(range(prefix)) or any(
        value < prefix for value in perm[prefix:]
    ):
        raise ValueError(
            "emit_mlir: cannot fold batch dimensions moved by transpose"
        )
    return [value - prefix for value in perm[prefix:]]


def _tt(shape: Sequence[object], dtype: str) -> str:
    """MLIR ranked-tensor type for a (possibly rank-0) shape."""
    # J keeps unsignedness in its logical dtype spelling.  MLIR's tensor type
    # syntax can carry that spelling, while arith lowerings bridge it to the
    # signless scalar type required by the arithmetic operation itself.
    dtype = {
        "u8": "ui8",
        "u16": "ui16",
        "u32": "ui32",
    }.get(dtype, dtype)
    if not shape:
        return f"tensor<{dtype}>"
    return "tensor<" + "x".join(str(s) for s in shape) + f"x{dtype}>"


@dataclass(frozen=True)
class _RaggedValueType:
    """Concrete MLIR type facts for one internal ragged descriptor.

    Ragged values are deliberately kept out of the ordinary tensor ``env``:
    the descriptor is a two-field value, not a tensor with an invented shape.
    Shapes are concrete for the two carrier tensors at materialization time;
    element dtypes may be filled by the dtype fixed-point pass when an older
    IntentIR artifact omitted the redundant descriptor field types.
    """

    offsets_shape: tuple[object, ...]
    offsets_dtype: str | None
    indices_shape: tuple[object, ...]
    indices_dtype: str | None


def _ragged_type_text(value: _RaggedValueType) -> str:
    if (
        value.offsets_dtype is None
        or value.indices_dtype is None
        or len(value.offsets_shape) != 1
        or len(value.indices_shape) != 1
    ):
        raise ValueError("emit_mlir: ragged descriptor type is incomplete")
    return (
        "tuple<"
        f"{_tt(list(value.offsets_shape), value.offsets_dtype)}, "
        f"{_tt(list(value.indices_shape), value.indices_dtype)}>"
    )


def _strip(group: str, fold: int) -> str:
    """Drop the leading `fold` (batch) labels from one einsum label group."""
    prefix = _fold_prefix_len(group, fold)
    return group[prefix:]


BroadcastAxes = Mapping[str, Mapping[str, Sequence[int]]]


def _broadcast(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    """numpy-style right-aligned broadcast of two shapes (None = unknown)."""
    if a is None:
        return b
    if b is None:
        return a
    n = max(len(a), len(b))
    a = [1] * (n - len(a)) + list(a)
    b = [1] * (n - len(b)) + list(b)
    return [max(x, y) for x, y in zip(a, b)]


def _broadcast_checked(
    a: list[int] | None,
    b: list[int] | None,
) -> list[int] | None:
    """Strict numpy broadcast used when J supplies explicit broadcast facts."""

    if a is None:
        return b
    if b is None:
        return a
    rank = max(len(a), len(b))
    lhs = [1] * (rank - len(a)) + list(a)
    rhs = [1] * (rank - len(b)) + list(b)
    result: list[int] = []
    for left, right in zip(lhs, rhs, strict=True):
        if left != right and left != 1 and right != 1:
            raise ValueError(
                f"explicit broadcast shapes are incompatible: {a} vs {b}"
            )
        result.append(max(left, right))
    return result


def _insert_broadcast_axes(
    shape: list[int] | None,
    axes: Sequence[int],
    target_shape: Sequence[int] | None = None,
) -> list[int] | None:
    projected, _inserted_axes = _broadcast_projection(
        shape, axes, target_shape
    )
    return projected


def _op_broadcast_axes(
    op: IntentOp,
    broadcast_axes: BroadcastAxes | None,
) -> Mapping[str, Sequence[int]]:
    if broadcast_axes is None or not isinstance(op.result, str):
        return {}
    return broadcast_axes.get(op.result, {})


def _broadcast_mul_contraction_subscripts(
    op: IntentOp,
    env: Mapping[str, list[int]],
    axes: Mapping[str, Sequence[int]],
) -> str | None:
    """Canonicalize a non-trailing rank-insertion multiply to einsum.

    ``map_binary`` plus Layer-B broadcast axes is the canonical JSON spelling,
    while the row CUDA pass also accepts the equivalent no-reduction
    contraction.  Converting only multiplication with a proven non-trailing
    insertion keeps the transformation exact and avoids forcing live LLMs to
    choose one backend-facing graph spelling.  Ordinary trailing vector
    broadcasts remain map operations.
    """

    if (
        op.op != "map_binary"
        or op.attrs.get("fn") != "mul"
        or not isinstance(op.result, str)
        or len(op.operands) != 2
        or op.result not in env
    ):
        return None
    result_shape = env[op.result]
    rank = len(result_shape)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if rank == 0 or rank > len(alphabet):
        return None
    output_labels = alphabet[:rank]
    operand_labels: list[str] = []
    saw_nontrailing_insertion = False
    for operand in op.operands:
        source_shape = env.get(operand)
        if source_shape is None or not source_shape:
            return None
        operand_axes = tuple(axes.get(operand, ()))
        if not operand_axes:
            if source_shape != result_shape:
                return None
            positions = tuple(range(rank))
        else:
            projected, inserted = _broadcast_projection(
                source_shape, operand_axes, result_shape
            )
            if projected is None:
                return None
            positions = tuple(
                index for index in range(rank) if index not in set(inserted)
            )
            if len(positions) != len(source_shape):
                return None
            if any(
                source_extent != result_shape[position]
                for source_extent, position in zip(source_shape, positions)
            ):
                return None
            leading = tuple(range(rank - len(source_shape)))
            if inserted and inserted != leading:
                saw_nontrailing_insertion = True
        operand_labels.append("".join(output_labels[index] for index in positions))
    if not saw_nontrailing_insertion:
        return None
    return f"{operand_labels[0]},{operand_labels[1]}->{output_labels}"


def _op_declared_shape(
    op: IntentOp,
    fold: int,
    extents: dict[str, int] | None,
) -> list[object] | None:
    if not isinstance(op.type, dict) or not isinstance(
        op.type.get("shape"), list
    ):
        return None
    resolved: list[object] = []
    for item in op.type["shape"]:
        # ``?`` is the closed J spelling for a data-dependent extent.  It is
        # valid only for the ragged-member result, but preserving it here lets
        # the verifier below diagnose any other misuse instead of trying to
        # interpret it as a symbolic dimension name.
        if item == "?":
            resolved.append("?")
        else:
            resolved.append(_resolve_int(item, extents or {}))
    return _fold_shape(
        resolved,
        fold,
        preserve_rank_one=op.op == "ragged_member",
    )


def _declared_ragged_type(
    op: IntentOp,
    extents: dict[str, int] | None,
    fold: int,
) -> _RaggedValueType | None:
    """Decode an authored ragged result type when one is present."""

    raw = op.type
    if not isinstance(raw, dict) or raw.get("kind") != "ragged":
        return None
    fields = raw.get("fields")
    if not isinstance(fields, list) or len(fields) != 2:
        raise ValueError("emit_mlir: ragged type requires offsets and indices fields")
    names = [item.get("name") if isinstance(item, dict) else None for item in fields]
    if names != ["offsets", "indices"]:
        raise ValueError(
            "emit_mlir: ragged type fields must be offsets followed by indices"
        )

    def field_info(item: object, field_name: str) -> tuple[tuple[object, ...], str]:
        if not isinstance(item, dict) or not isinstance(item.get("type"), dict):
            raise ValueError(
                f"emit_mlir: ragged {field_name} field lacks a tensor type"
            )
        field_type = item["type"]
        shape = field_type.get("shape")
        dtype = field_type.get("dtype")
        if not isinstance(shape, list) or len(shape) != 1:
            raise ValueError(
                f"emit_mlir: ragged {field_name} field must be rank one"
            )
        if not isinstance(dtype, str):
            raise ValueError(
                f"emit_mlir: ragged {field_name} field lacks an element dtype"
            )
        resolved = tuple(
            "?" if value == "?" else _resolve_int(value, extents or {})
            for value in shape
        )
        return (
            tuple(
                _fold_shape(resolved, fold, preserve_rank_one=True)
            ),
            _normalize_dtype(dtype, f"ragged {field_name} dtype"),
        )

    offsets_shape, offsets_dtype = field_info(fields[0], "offsets")
    indices_shape, indices_dtype = field_info(fields[1], "indices")
    if len(offsets_shape) != 1 or len(indices_shape) != 1:
        raise ValueError("emit_mlir: folded ragged fields must remain rank one")
    return _RaggedValueType(
        offsets_shape,
        offsets_dtype,
        indices_shape,
        indices_dtype,
    )


def _merge_ragged_type(
    previous: _RaggedValueType | None,
    current: _RaggedValueType,
    *,
    context: str,
) -> _RaggedValueType:
    if previous is None:
        return current

    def merge_field(
        lhs_shape: tuple[object, ...],
        lhs_dtype: str | None,
        rhs_shape: tuple[object, ...],
        rhs_dtype: str | None,
        field_name: str,
    ) -> tuple[tuple[object, ...], str | None]:
        if lhs_shape and rhs_shape and lhs_shape != rhs_shape:
            raise ValueError(
                f"emit_mlir: ragged {field_name} shape conflict in {context}: "
                f"{lhs_shape} versus {rhs_shape}"
            )
        if lhs_dtype is not None and rhs_dtype is not None and lhs_dtype != rhs_dtype:
            raise ValueError(
                f"emit_mlir: ragged {field_name} dtype conflict in {context}: "
                f"{lhs_dtype} versus {rhs_dtype}"
            )
        return (lhs_shape or rhs_shape, lhs_dtype or rhs_dtype)

    offsets_shape, offsets_dtype = merge_field(
        previous.offsets_shape,
        previous.offsets_dtype,
        current.offsets_shape,
        current.offsets_dtype,
        "offsets",
    )
    indices_shape, indices_dtype = merge_field(
        previous.indices_shape,
        previous.indices_dtype,
        current.indices_shape,
        current.indices_dtype,
        "indices",
    )
    return _RaggedValueType(
        offsets_shape,
        offsets_dtype,
        indices_shape,
        indices_dtype,
    )


def _record_ragged_type(
    ragged_types: dict[str, _RaggedValueType],
    name: str,
    value: _RaggedValueType,
    context: str,
) -> bool:
    merged = _merge_ragged_type(ragged_types.get(name), value, context=context)
    changed = ragged_types.get(name) != merged
    ragged_types[name] = merged
    return changed


def _record_shape(
    env: dict[str, list[object]],
    name: str,
    shape: Sequence[object],
    context: str,
) -> bool:
    """Record one lexical shape binding without weakening conflicts."""

    normalized = list(shape)
    previous = env.get(name)
    if previous is not None and previous != normalized:
        raise ValueError(
            f"emit_mlir: shape conflict for {name!r}: {previous} versus "
            f"{normalized} ({context})"
        )
    if previous != normalized:
        env[name] = normalized
        return True
    return False


def _prepare_loop_shape_scope(
    loop: IntentOp,
    env: Mapping[str, list[object]],
    fold: int,
    extents: dict[str, int] | None,
    schedule_tile: int | None,
    raw_ranks: Mapping[str, int] | None = None,
) -> tuple[dict[str, list[object]], dict[str, int], int, dict[str, int]]:
    """Create the loop body's lexical shape environment.

    The loop body may read dominating inputs and carry initializers, but scan
    names, carry block arguments, and body-produced values belong to the loop
    region.  Keeping this preparation in one helper makes inference and MLIR
    printing use the same region boundary.
    """

    if loop.trip is None:
        raise ValueError("emit_mlir: loop requires a semantic trip axis")
    validate_loop_wiring(loop, context="emit_mlir loop")
    resolved_extents = dict(extents or {})
    loop_raw_ranks = dict(raw_ranks or {})
    dynamic_trip = loop.trip.is_dynamic
    tile = _tile(loop, resolved_extents, schedule_tile)
    body_extents = dict(resolved_extents)
    if not dynamic_trip:
        assert loop.trip.axis is not None
        body_extents[loop.trip.axis] = tile

    local_env = dict(env)
    # A loop result is defined by the region, not a value visible inside it.
    for state in loop.carry:
        local_env.pop(state.result_value, None)

    for scan in loop.scan:
        source_shape = local_env.get(scan.source)
        source_rank = _raw_rank_for_value(
            scan.source,
            local_env,
            loop_raw_ranks,
            context="loop scan source",
        )
        _record_raw_rank(
            loop_raw_ranks,
            scan.name,
            source_rank,
            "loop scan binding",
        )
        if source_shape is None:
            continue
        axis = _fold_axis(scan.axis, source_rank, fold)
        if axis < 0 or axis >= len(source_shape):
            raise ValueError(
                "emit_mlir: loop scan axis is outside its source rank"
            )
        slice_shape = list(source_shape)
        slice_shape[axis] = tile
        _record_shape(local_env, scan.name, slice_shape, "loop scan slice")

    for state in loop.carry:
        declared = None
        if state.type is not None:
            if state.type.kind != "tensor":
                raise NotImplementedError(
                    "emit_mlir: named record loop carries require a "
                    "record-aware ABI"
                )
            _record_raw_rank(
                loop_raw_ranks,
                state.body_argument,
                len(state.type.shape),
                "loop carry body_argument",
            )
            _record_raw_rank(
                loop_raw_ranks,
                state.result_value,
                len(state.type.shape),
                "loop carry result_value",
            )
            declared = _fold_shape(
                _resolve(state.type.shape, body_extents),
                fold,
            )
        init_shape = (
            local_env.get(state.init_value)
            if isinstance(state.init_value, str)
            else None
        )
        if isinstance(state.init_value, str) and state.init_value in loop_raw_ranks:
            _record_raw_rank(
                loop_raw_ranks,
                state.body_argument,
                loop_raw_ranks[state.init_value],
                "loop carry initializer",
            )
        state_shape = declared or init_shape
        if state_shape is None:
            continue
        if init_shape is not None and declared is not None and init_shape != declared:
            raise ValueError(
                "emit_mlir: loop carry initializer shape differs from its authored "
                "state type"
            )
        _record_shape(
            local_env,
            state.body_argument,
            state_shape,
            "loop carry block argument",
        )

    if dynamic_trip:
        upper_bound = loop.trip.upper_bound
        induction = loop.trip.index
        if not isinstance(upper_bound, str) or not isinstance(induction, str):
            raise ValueError(
                "emit_mlir: dynamic loop requires upper_bound and index bindings"
            )
        upper_shape = local_env.get(upper_bound)
        if upper_shape is not None and upper_shape != []:
            raise ValueError(
                "emit_mlir: dynamic loop upper_bound must be a rank-0 tensor"
            )
        _record_shape(
            local_env,
            induction,
            [],
            "dynamic loop induction block argument",
        )

        _record_raw_rank(
            loop_raw_ranks,
            induction,
            0,
            "dynamic loop induction",
        )

    return local_env, body_extents, tile, loop_raw_ranks


def _infer_ragged_value_type(
    op: IntentOp,
    env: Mapping[str, list[object]],
    ragged_types: Mapping[str, _RaggedValueType],
    *,
    extents: dict[str, int] | None = None,
    fold: int = 0,
    dtypes: Mapping[str, str] | None = None,
) -> _RaggedValueType | None:
    if op.op == "ragged":
        if len(op.operands) != 2 or not isinstance(op.result, str):
            return None
        offsets_name, indices_name = op.operands
        declared = _declared_ragged_type(op, extents, fold)
        offsets_shape = tuple(env[offsets_name]) if offsets_name in env else ()
        indices_shape = tuple(env[indices_name]) if indices_name in env else ()
        offsets_dtype = dtypes.get(offsets_name) if dtypes is not None else None
        indices_dtype = dtypes.get(indices_name) if dtypes is not None else None
        if declared is not None:
            if offsets_shape and tuple(offsets_shape) != declared.offsets_shape:
                raise ValueError(
                    "emit_mlir: ragged offsets field shape differs from its operand"
                )
            if indices_shape and tuple(indices_shape) != declared.indices_shape:
                raise ValueError(
                    "emit_mlir: ragged indices field shape differs from its operand"
                )
            offsets_shape = offsets_shape or declared.offsets_shape
            indices_shape = indices_shape or declared.indices_shape
            offsets_dtype = offsets_dtype or declared.offsets_dtype
            indices_dtype = indices_dtype or declared.indices_dtype
        if len(offsets_shape) != 1 or len(indices_shape) != 1:
            return None
        return _RaggedValueType(
            offsets_shape,
            offsets_dtype,
            indices_shape,
            indices_dtype,
        )
    if not op.operands:
        return None
    return ragged_types.get(op.operands[0])


def _op_target_shape(
    op: IntentOp,
    env: Mapping[str, list[int]],
    fold: int,
    extents: dict[str, int] | None,
) -> list[int] | None:
    declared = _op_declared_shape(op, fold, extents)
    constrained = env.get(op.result) if isinstance(op.result, str) else None
    if (
        declared is not None
        and constrained is not None
        and declared != constrained
    ):
        raise ValueError(
            f"op result {op.result!r} declared shape {declared} conflicts "
            f"with consumer/output constraint {constrained}"
        )
    return declared if declared is not None else constrained


def _infer_op(
    op: IntentOp,
    env: dict[str, list[object]],
    fold: int,
    extents: dict[str, int] | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    ragged_types: dict[str, _RaggedValueType] | None = None,
    dtypes: Mapping[str, str] | None = None,
    raw_ranks: Mapping[str, int] | None = None,
) -> list[object] | None:
    """Output shape of one body op given known operand shapes (None if not yet
    resolvable). Mirrors interpret.py's shape rules."""
    ragged_types = ragged_types if ragged_types is not None else {}
    if op.op == "ragged":
        if op.attrs:
            raise ValueError("emit_mlir: ragged does not accept attrs")
        inferred = _infer_ragged_value_type(
            op,
            env,
            ragged_types,
            extents=extents,
            fold=fold,
            dtypes=dtypes,
        )
        if inferred is not None and isinstance(op.result, str):
            _record_ragged_type(ragged_types, op.result, inferred, "ragged")
        return None
    if op.op == "ragged_outer":
        if op.attrs:
            raise ValueError("emit_mlir: ragged_outer does not accept attrs")
        if len(op.operands) != 1:
            raise ValueError("emit_mlir: ragged_outer requires one descriptor")
        descriptor = ragged_types.get(op.operands[0])
        if descriptor is None:
            return None
        declared = _op_declared_shape(op, fold, extents)
        if declared is not None and declared != []:
            raise ValueError("emit_mlir: ragged_outer result must be rank zero")
        return []
    if op.op == "ragged_member":
        if op.attrs:
            raise ValueError("emit_mlir: ragged_member does not accept attrs")
        if len(op.operands) != 2:
            raise ValueError(
                "emit_mlir: ragged_member requires a descriptor and an index"
            )
        descriptor = ragged_types.get(op.operands[0])
        if descriptor is None:
            return None
        index_shape = env.get(op.operands[1])
        if index_shape is not None and index_shape != []:
            raise ValueError(
                "emit_mlir: ragged_member index must be a rank-0 tensor"
            )
        declared = _op_declared_shape(op, fold, extents)
        if declared is not None and declared != ["?"]:
            raise ValueError(
                "emit_mlir: ragged_member result must have dynamic rank-one shape"
            )
        return ["?"]
    if op.op == "contraction":
        a, b = op.operands
        if a not in env or b not in env:
            return None
        ins, out = op.attrs["subscripts"].split("->")
        aL, bL = (s.strip() for s in ins.split(","))
        aL, bL, oL = _strip(aL, fold), _strip(bL, fold), _strip(out.strip(), fold)
        ext: dict[str, int] = {}
        for lab, shp in ((aL, env[a]), (bL, env[b])):
            for c, d in zip(lab, shp):
                ext[c] = d
        return [ext[c] for c in oL]
    if op.op == "reduce":
        (x,) = op.operands
        if x not in env:
            return None
        shp = list(env[x])
        raw_axes = [int(a) for a in op.attrs["axes"]]
        raw_rank = _raw_rank_for_value(
            x, env, raw_ranks, context="reduce operand"
        )
        axes = [
            _fold_axis(axis, raw_rank, fold)
            for axis in raw_axes
        ]
        keepdims = list(shp)
        for ax in axes:
            keepdims[ax] = 1
        collapsed = [dim for index, dim in enumerate(shp) if index not in axes]
        # Canonical J carries a checked result type for every operation.  Use
        # that declaration to disambiguate collapsed versus keepdims reductions
        # even when the reduction is an intermediate value.  Looking only at
        # ``env`` loses the declaration until a downstream consumer happens to
        # constrain it and can silently choose the legacy keepdims default.
        declared = _op_target_shape(op, env, fold, extents)
        explicit_keepdims = op.attrs.get("keepdims")
        if explicit_keepdims is not None:
            if not isinstance(explicit_keepdims, bool):
                raise ValueError("reduce keepdims must be a boolean")
            selected = keepdims if explicit_keepdims else collapsed
            if declared is not None and declared != selected:
                raise ValueError(
                    f"reduce result {op.result!r} shape {declared} conflicts "
                    f"with keepdims={explicit_keepdims}: expected {selected}"
                )
            return list(selected)
        if declared is not None:
            if declared not in (keepdims, collapsed):
                raise ValueError(
                    f"reduce result {op.result!r} shape {declared} is neither "
                    f"keepdims {keepdims} nor collapsed {collapsed}"
                )
            return list(declared)
        if sorted(axes) == list(range(len(shp))):
            # A full reduction has no surviving semantic axis.  Rank-0 is the
            # canonical default and still broadcasts naturally to later ops;
            # keeping one unit dimension here would make scalar final-stage
            # reductions (sum(partials)/M) disagree with their ABI.
            return []
        for ax in axes:  # existing default: keepdims for broadcast consumers
            shp[ax] = 1
        return shp
    if op.op == "scan":
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "arg_reduce":
        (x,) = op.operands
        if x not in env:
            return None
        raw_axis = int(op.attrs["axis"])
        raw_rank = _raw_rank_for_value(
            x, env, raw_ranks, context="arg_reduce operand"
        )
        ax = _fold_axis(raw_axis, raw_rank, fold)
        # collapse the axis (rank-1 index out)
        return [d for i, d in enumerate(env[x]) if i != ax]
    if op.op == "sort":
        (x,) = op.operands
        if x not in env:
            return None
        shp = list(env[x])            # shape-preserving (values & indices)
        w = op.attrs.get("k")         # optional topk width along the last axis
        if w is not None:
            shp[-1] = int(w)
        return shp
    if op.op == "iota":
        return [_resolve_int(op.attrs["n"], extents)]
    if op.op == "const":
        result_type = op.type if isinstance(op.type, Mapping) else None
        try:
            literal = validate_const_literal(
                op.attrs,
                result_type=result_type,
                context="emit_mlir const",
            )
        except IntentOpContractError as exc:
            raise ValueError(str(exc)) from exc
        return _resolve(literal.shape, extents or {})
    if op.op == "rand":
        # Counter-based RNG (C5): shape carried in attrs (like const), symbolic
        # dims resolved against the extents, fold the leading batch dims out so a
        # single-head instance emits rank-2.
        return _fold_shape(
            _resolve(op.attrs.get("shape", []), extents or {}),
            fold,
        )
    if op.op == "cast":
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "reshape":
        (x,) = op.operands
        if x not in env:
            return None
        raw_shape = op.attrs.get("shape")
        if not isinstance(raw_shape, list):
            raise ValueError("emit_mlir: reshape requires attrs.shape")
        resolved = [_resolve_int(item, extents) for item in raw_shape]
        resolved = _fold_shape(resolved, fold)
        source_elements = math.prod(env[x]) if env[x] else 1
        result_elements = math.prod(resolved) if resolved else 1
        if source_elements != result_elements:
            raise ValueError(
                f"emit_mlir: reshape changes element count from "
                f"{source_elements} to {result_elements}"
            )
        declared = _op_target_shape(op, env, fold, extents)
        if declared is not None and declared != resolved:
            raise ValueError(
                f"emit_mlir: reshape result {op.result!r} declared shape "
                f"{declared} differs from attrs.shape {resolved}"
            )
        return resolved
    if op.op in {"record", "record_get"}:
        # Named records are a J-level type and have no flat tensor ABI yet.
        # Keep shape inference from guessing a tensor representation.
        return None
    if op.op == "transpose":
        (x,) = op.operands
        if x not in env:
            return None
        perm = _fold_permutation(op.attrs["perm"], fold, extents or {})
        if sorted(perm) != list(range(len(env[x]))):
            raise ValueError(
                f"transpose perm {perm!r} is invalid for rank {len(env[x])}"
            )
        inferred = [env[x][axis] for axis in perm]
        declared = env.get(op.result)
        if declared is not None and declared != inferred:
            raise ValueError(
                f"transpose result {op.result!r} shape {declared} does not "
                f"match permuted input shape {inferred}"
            )
        return inferred
    if op.op == "map_unary":
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "map_binary":
        a, b = op.operands
        axes = _op_broadcast_axes(op, broadcast_axes)
        combine = _broadcast_checked if axes else _broadcast
        declared = _op_target_shape(op, env, fold, extents)
        inferred = combine(
            _insert_broadcast_axes(
                env.get(a), axes.get(a, ()), declared
            ),
            _insert_broadcast_axes(
                env.get(b), axes.get(b, ()), declared
            ),
        )
        if declared is not None and axes and inferred != declared:
            raise ValueError(
                f"map_binary result {op.result!r} inferred shape {inferred} "
                f"conflicts with declared shape {declared}"
            )
        return inferred
    if op.op == "gather":
        d, idx = op.operands
        if d not in env or idx not in env:
            return None
        raw_axis = int(op.attrs["axis"])
        raw_rank = _raw_rank_for_value(
            d, env, raw_ranks, context="gather data operand"
        )
        axis = _fold_axis(raw_axis, raw_rank, fold)
        mode = op.attrs["mode"]
        if mode not in {"along_axis", "take"}:
            raise ValueError(
                f"emit_mlir: gather mode {mode!r} is not supported"
            )
        if mode == "along_axis":
            data_shape = list(env[d])
            index_shape = list(env[idx])
            data_rank = len(data_shape)
            index_rank = len(index_shape)
            equal_rank = index_rank == data_rank
            compressed_rank = index_rank == data_rank - 1 and index_rank > 0
            if not (equal_rank or compressed_rank):
                raise ValueError(
                    "emit_mlir: along_axis gather index rank must equal "
                    "data rank or data rank minus one"
                )
            expected_index_shape = (
                data_shape
                if equal_rank
                else data_shape[:axis] + data_shape[axis + 1 :]
            )
            if index_shape != expected_index_shape:
                raise ValueError(
                    "emit_mlir: along_axis gather index must match every "
                    f"non-axis data extent; expected {expected_index_shape}, "
                    f"got {index_shape}"
                )
            return list(env[idx])                        # shaped like index
        return list(env[d][:axis]) + list(env[idx]) + list(env[d][axis + 1:])
    if op.op in {"scatter", "atomic_scatter"}:
        d = op.operands[0]
        return list(env[d]) if d in env else None        # shaped like data
    if op.op == "nonzero":
        # Dynamic-shape output (C8): the leading nnz extent is data-dependent.
        # The host orchestration resolves the concrete allocation before launch;
        # the MLIR contract must therefore retain the dynamic extent rather than
        # silently changing it into a static capacity bound.
        (c,) = op.operands
        if c not in env:
            return None
        return ["?", len(env[c])]
    if op.op == "masked_select":
        d, m = op.operands
        if d not in env:
            return None
        return ["?"]
    if op.op == "masked_scatter":
        d = op.operands[0]
        return list(env[d]) if d in env else None         # data-shaped
    if op.op == "select":
        c, x, y = op.operands
        axes = _op_broadcast_axes(op, broadcast_axes)
        combine = _broadcast_checked if axes else _broadcast
        declared = _op_target_shape(op, env, fold, extents)
        inferred = combine(
            combine(
                _insert_broadcast_axes(
                    env.get(c), axes.get(c, ()), declared
                ),
                _insert_broadcast_axes(
                    env.get(x), axes.get(x, ()), declared
                ),
            ),
            _insert_broadcast_axes(
                env.get(y), axes.get(y, ()), declared
            ),
        )
        if declared is not None and axes and inferred != declared:
            raise ValueError(
                f"select result {op.result!r} inferred shape {inferred} "
                f"conflicts with declared shape {declared}"
            )
        return inferred
    if op.op == "value_if":
        if op.positive is None or op.negative is None:
            raise ValueError("value_if requires two explicit arms")
        if op.predicate in env and env[op.predicate] != []:
            raise ValueError("value_if predicate must be a rank-0 tensor")
        declared = _op_declared_shape(op, fold, extents)
        yields = [op.positive.yield_value, op.negative.yield_value]
        known = [env[name] for name in yields if name in env]
        if any(shape != known[0] for shape in known[1:]):
            raise ValueError("value_if arm yields have different shapes")
        if declared is not None and any(shape != declared for shape in known):
            raise ValueError(
                f"value_if result {op.result!r} shape {declared} differs "
                "from an arm yield"
            )
        return list(declared if declared is not None else known[0]) if (
            declared is not None or known
        ) else None
    if op.op == "concat":
        if any(o not in env for o in op.operands):
            return None
        raw_axis = int(op.attrs["axis"])
        raw_rank = _raw_rank_for_value(
            op.operands[0], env, raw_ranks, context="concat operand"
        )
        axis = _fold_axis(raw_axis, raw_rank, fold)
        shp = list(env[op.operands[0]])
        shp[axis] = sum(env[o][axis] for o in op.operands)
        return shp
    if op.op == "pad":
        (x,) = op.operands
        if x not in env:
            return None
        raw_lo = [_resolve_int(v, extents) for v in op.attrs["lo"]]
        raw_hi = [_resolve_int(v, extents) for v in op.attrs["hi"]]
        prefix = _fold_prefix_len(raw_lo, fold)
        expected_rank = len(env[x]) + prefix
        if len(raw_lo) != expected_rank or len(raw_hi) != expected_rank:
            raise ValueError(
                f"pad lo/hi must have {expected_rank} entries, got "
                f"{len(raw_lo)}/{len(raw_hi)}"
            )
        if any(value != 0 for value in raw_lo[:prefix] + raw_hi[:prefix]):
            raise ValueError("emit_mlir: cannot fold padded batch dimensions")
        lo, hi = raw_lo[prefix:], raw_hi[prefix:]
        if any(value < 0 for value in lo + hi):
            raise ValueError("pad lo/hi entries must be non-negative")
        inferred = [d + lo[i] + hi[i] for i, d in enumerate(env[x])]
        declared = env.get(op.result)
        if declared is not None and declared != inferred:
            raise ValueError(
                f"pad result {op.result!r} shape {declared} does not match "
                f"input+lo+hi shape {inferred}"
            )
        return inferred
    if op.op == "tile":
        (x,) = op.operands
        if x not in env:
            return None
        raw_reps = [int(v) for v in op.attrs["reps"]]
        prefix = _fold_prefix_len(raw_reps, fold)
        reps = raw_reps[prefix:]
        return [d * reps[i] for i, d in enumerate(env[x])]
    if op.op in ("flip", "triu"):
        (x,) = op.operands
        return list(env[x]) if x in env else None
    if op.op == "pool":
        (x,) = op.operands
        if x not in env:
            return None
        N, C, H, W = env[x]
        kH, kW = int(op.attrs["kH"]), int(op.attrs["kW"])
        sH, sW = int(op.attrs["sH"]), int(op.attrs["sW"])
        padH, padW = int(op.attrs["padH"]), int(op.attrs["padW"])
        OH = (H + 2 * padH - kH) // sH + 1
        OW = (W + 2 * padW - kW) // sW + 1
        return [N, C, OH, OW]
    if op.op == "upsample":
        (x,) = op.operands
        if x not in env:
            return None
        N, C, H, W = env[x]
        sh, sw = (int(v) for v in op.attrs["scale"])
        return [N, C, H * sh, W * sw]
    if op.op == "polar":
        a0 = op.operands[0]
        return list(env[a0]) if a0 in env else None
    if op.op == "conv":
        xn, wn = op.operands[0], op.operands[1]
        if xn not in env or wn not in env:
            return None
        inshape, wshape = env[xn], env[wn]
        R = len(inshape) - 2
        stride = [int(v) for v in op.attrs["stride"]]
        pad = [int(v) for v in op.attrs["padding"]]
        dil = [int(v) for v in op.attrs["dilation"]]
        out = [inshape[0], wshape[0]]
        for j in range(R):
            H, k = inshape[2 + j], wshape[2 + j]
            out.append((H + 2 * pad[j] - dil[j] * (k - 1) - 1) // stride[j] + 1)
        return out
    raise NotImplementedError(f"emit_mlir: unhandled body op {op.op!r}")


_CMP_FNS = MAP_BINARY_PREDICATE_FNS
# map_unary fns whose result is an i1 (finiteness predicates + i1 complement).
_U_TO_I1 = MAP_UNARY_PREDICATE_FNS

_FLOAT_DTYPES = FLOATING_DTYPES
_SIGNED_INTEGER_DTYPES = frozenset({"i1", "i8", "i16", "i32", "i64"})
_UNSIGNED_INTEGER_DTYPES = frozenset({"u8", "u16", "u32"})
_INTEGER_DTYPES = INTEGER_DTYPES
_DTYPES = CANONICAL_DTYPES


def _normalize_dtype(value: object, context: str) -> str:
    """Return one supported executable MLIR scalar dtype.

    ``index`` is a logical public dtype.  The host/native ABI has no separate
    NumPy storage type for it, so materialize it as the signed ``i64`` carrier
    before the rest of the emitter performs width, promotion, and constant
    range reasoning.
    """

    try:
        return canonicalize_dtype(value, context=context)
    except ValueError as exc:
        raise ValueError(f"emit_mlir: invalid dtype {value!r} for {context}") from exc


def _value_dimension_attribute(
    fn: IntentFunction,
    equalities: Sequence[tuple[int, int]],
) -> str | None:
    """Render the closed J-derived rank-0 scalar equality contract.

    The pair index is an ``IntentFunction.inputs`` index, which is exactly the
    function-argument order emitted below.  This helper accepts no J, C, pi, or
    Harness object and cannot invent an equality.
    """

    checked: list[tuple[int, int]] = []
    seen: set[int] = set()
    for ordinal, equality in enumerate(equalities):
        if (
            not isinstance(equality, (tuple, list))
            or len(equality) != 2
        ):
            raise ValueError(
                "emit_mlir: value/dimension equality "
                f"{ordinal} must be an (input index, value) pair"
            )
        argument_index, value = equality
        if (
            isinstance(argument_index, bool)
            or not isinstance(argument_index, int)
            or argument_index < 0
            or argument_index >= len(fn.inputs)
        ):
            raise ValueError(
                "emit_mlir: value/dimension equality references an invalid "
                "IntentFunction input index"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "emit_mlir: value/dimension equality value must be an integer"
            )
        if argument_index in seen:
            raise ValueError(
                "emit_mlir: duplicate value/dimension equality input index"
            )
        tensor = fn.inputs[argument_index]
        dtype = _normalize_dtype(
            tensor.dtype,
            f"value/dimension input {tensor.name!r}",
        )
        if tensor.shape or dtype not in _INTEGER_DTYPES or dtype == "i1":
            raise ValueError(
                "emit_mlir: value/dimension equality requires a rank-0 "
                "non-boolean integer input"
            )
        width = int(dtype[1:])
        if dtype in _UNSIGNED_INTEGER_DTYPES:
            minimum = 0
            maximum = (1 << width) - 1
        else:
            minimum = -(1 << (width - 1))
            maximum = (1 << (width - 1)) - 1
        if value < minimum or value > maximum:
            raise ValueError(
                "emit_mlir: value/dimension equality is not representable by "
                f"input dtype {dtype}"
            )
        seen.add(argument_index)
        checked.append((argument_index, value))
    if not checked:
        return None
    flattened = ", ".join(
        str(item)
        for equality in sorted(checked)
        for item in equality
    )
    return "intent.scalar_equalities = array<i64: " + flattened + ">"


def _function_attributes(
    fn: IntentFunction,
    *,
    value_dimension_equalities: Sequence[tuple[int, int]],
    inplace_outputs: Sequence[int],
    paired_rotation: tuple[str, tuple[int, int, int, int], tuple[int, ...]]
    | None,
) -> str:
    attributes: list[str] = []
    equality = _value_dimension_attribute(fn, value_dimension_equalities)
    if equality is not None:
        attributes.append(equality)

    if inplace_outputs:
        if len(inplace_outputs) != len(fn.outputs):
            raise ValueError(
                "emit_mlir: inplace output mapping must cover every result"
            )
        checked: list[int] = []
        for output_index, argument_index in enumerate(inplace_outputs):
            if (
                isinstance(argument_index, bool)
                or not isinstance(argument_index, int)
                or argument_index < -1
                or argument_index >= len(fn.inputs)
            ):
                raise ValueError("emit_mlir: invalid inplace input index")
            if argument_index >= 0:
                source = fn.inputs[argument_index]
                output = fn.outputs[output_index]
                if source.dtype != output.dtype or source.shape != output.shape:
                    raise ValueError(
                        "emit_mlir: inplace source/result tensor types differ"
                    )
            checked.append(argument_index)
        flattened = ", ".join(str(item) for item in checked)
        attributes.append(f"intent.inplace = array<i64: {flattened}>")

    if paired_rotation is not None:
        layout, input_indices, sections = paired_rotation
        if layout not in {"shared_position", "three_plane", "interleaved"}:
            raise ValueError("emit_mlir: unknown paired rotation layout")
        if (
            len(input_indices) != 4
            or len(set(input_indices)) != 4
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(fn.inputs)
                for index in input_indices
            )
        ):
            raise ValueError("emit_mlir: invalid paired rotation input mapping")
        if layout == "three_plane":
            if len(sections) != 3 or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in sections
            ):
                raise ValueError("emit_mlir: invalid three-plane sections")
        elif sections:
            raise ValueError(
                "emit_mlir: non-three-plane rotation has no sections"
            )
        arguments = ", ".join(str(item) for item in input_indices)
        attributes.extend(
            (
                'intent.paired_rotation_version = '
                '"intentir.paired-last-axis-rotation.v1"',
                f'intent.paired_rotation = "{layout}"',
                f"intent.paired_rotation_inputs = array<i64: {arguments}>",
            )
        )
        if sections:
            values = ", ".join(str(item) for item in sections)
            attributes.append(
                f"intent.paired_rotation_sections = array<i64: {values}>"
            )

    return "" if not attributes else " attributes {" + ", ".join(attributes) + "}"


def _promote_dtype(lhs: str | None, rhs: str | None) -> str | None:
    """Compatibility wrapper around the shared J dtype promotion contract."""

    return promote_dtype(lhs, rhs)


def _merge_operand_dtypes(
    op: IntentOp, dtypes: dict[str, str], start: int = 0
) -> str | None:
    result: str | None = None
    for name in op.operands[start:]:
        result = _promote_dtype(result, dtypes.get(name))
    return result


def _declared_op_dtype(op: IntentOp) -> str | None:
    if not isinstance(op.type, dict) or "dtype" not in op.type:
        return None
    return _normalize_dtype(op.type["dtype"], f"{op.op} result type")


def _cast_target_dtype(attrs: Mapping[str, object]) -> str:
    """Read the sole canonical cast target attribute."""

    try:
        value = attrs["to_dtype"]
    except KeyError as exc:
        raise ValueError("emit_mlir: cast requires attrs.to_dtype") from exc
    return _normalize_dtype(value, "cast target dtype")


def _result_names(op: IntentOp) -> list[str]:
    if isinstance(op.result, list):
        return list(op.result)
    return [op.result] if isinstance(op.result, str) else []


def _declared_raw_rank(op: IntentOp) -> int | None:
    """Return an authored tensor result rank before batch projection."""

    if not isinstance(op.type, dict):
        return None
    shape = op.type.get("shape")
    if isinstance(shape, list):
        return len(shape)
    return None


def _record_raw_rank(
    raw_ranks: dict[str, int],
    name: str,
    rank: int,
    context: str,
) -> bool:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(
            f"emit_mlir: {context} raw rank for {name!r} must be non-negative"
        )
    previous = raw_ranks.get(name)
    if previous is not None and previous != rank:
        raise ValueError(
            f"emit_mlir: raw rank conflict for {name!r}: {previous} versus "
            f"{rank} ({context})"
        )
    if previous is None:
        raw_ranks[name] = rank
        return True
    return False


def _raw_rank_for_value(
    name: str,
    env: Mapping[str, Sequence[object]],
    raw_ranks: Mapping[str, int] | None,
    *,
    context: str,
) -> int:
    """Resolve one value's authored rank for an axis-bearing operation."""

    if raw_ranks is not None and name in raw_ranks:
        return raw_ranks[name]
    if name in env:
        # This fallback is exact for fold=0 and for values whose authored rank
        # is smaller than the requested fold.  All normal emitter entry points
        # seed ``raw_ranks`` from J input/output declarations, so an ambiguous
        # folded rank cannot silently choose a meaning here.
        return len(env[name])
    raise ValueError(
        f"emit_mlir: cannot resolve authored rank for {name!r} ({context})"
    )


def _axis_for_value(
    op: IntentOp,
    operand: str,
    axis: object,
    env: Mapping[str, Sequence[object]],
    fold: int,
    raw_ranks: Mapping[str, int] | None,
    *,
    context: str,
) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise ValueError(f"emit_mlir: {context} axis must be an integer")
    rank = _raw_rank_for_value(operand, env, raw_ranks, context=context)
    return _fold_axis(axis, rank, fold)


def _operation_raw_result_rank(
    op: IntentOp,
    raw_ranks: Mapping[str, int],
    broadcast_axes: BroadcastAxes | None = None,
) -> int | None:
    """Infer a result's authored rank without using a folded shape."""

    # A Ragged descriptor is a tuple value, not an ordinary tensor rank.  The
    # two carrier ranks belong to its fields and must never be copied onto the
    # descriptor name as if it were a logical ``t*`` tensor.
    if op.op in RAGGED_OPS or op.op in {"record", "record_get"}:
        return None
    declared = _declared_raw_rank(op)
    if declared is not None:
        return declared
    operand_ranks = [raw_ranks.get(name) for name in op.operands]
    known = [rank for rank in operand_ranks if rank is not None]
    if not known:
        if op.op == "iota":
            return 1
        if op.op in {"const", "rand"} and isinstance(
            op.attrs.get("shape"), list
        ):
            return len(op.attrs["shape"])
        return None
    if op.op in {
        "map_unary", "cast", "scan", "sort", "scatter", "atomic_scatter",
        "masked_scatter", "concat", "pad", "tile", "flip", "triu",
        "pool", "upsample", "polar",
    }:
        return known[0]
    if op.op == "gather":
        mode = op.attrs["mode"]
        data_rank = operand_ranks[0] if operand_ranks else None
        index_rank = operand_ranks[1] if len(operand_ranks) > 1 else None
        if mode == "along_axis" and index_rank is not None:
            return index_rank
        if data_rank is not None and index_rank is not None:
            axis = op.attrs.get("axis")
            if isinstance(axis, int) and not isinstance(axis, bool):
                return data_rank - 1 + index_rank
        return data_rank
    if op.op in {"reduce", "arg_reduce"}:
        axis_values = op.attrs.get("axes")
        if axis_values is None:
            axis_values = [op.attrs.get("axis")]
        if op.attrs.get("keepdims") is True:
            return known[0]
        if isinstance(axis_values, list):
            return max(0, known[0] - len(axis_values))
        return max(0, known[0] - 1)
    if op.op in {"map_binary", "select"}:
        result = max(known)
        if broadcast_axes:
            explicit = broadcast_axes.get(op.result, {}) if isinstance(
                op.result, str
            ) else {}
            result = max(
                [
                    rank + len(explicit.get(operand, ()))
                    if rank is not None
                    else 0
                    for operand, rank in zip(
                        op.operands, operand_ranks, strict=True
                    )
                ]
                or [result]
            )
        return result
    if op.op == "reshape":
        shape = op.attrs.get("shape")
        return len(shape) if isinstance(shape, list) else known[0]
    if op.op == "iota":
        return 1
    if op.op in {"nonzero"}:
        return 2
    if op.op in {"masked_select"}:
        return 1
    return known[0]


def _loop_body_producers(
    body: Sequence[IntentOp],
) -> dict[str, IntentOp]:
    """Index only values exported by the loop body's lexical sequence."""

    return {
        name: producer
        for name, producer in intent_op_producers(body).items()
        if producer is not None
    }


def _loop_body_dtype_facts(
    body: Sequence[IntentOp], dtypes: Mapping[str, str]
) -> dict[str, str]:
    """Collect authored dtypes without assigning a default to unknown values."""

    facts = dict(dtypes)

    for _path, operation, _visible in walk_intent_op_scopes(body):
        names = _result_names(operation)
        declared = _declared_op_dtype(operation)
        if declared is not None and len(names) == 1:
            facts.setdefault(names[0], declared)
        if operation.op in {"const", "rand"} and len(names) == 1:
            if "dtype" in operation.attrs:
                facts[names[0]] = _normalize_dtype(
                    operation.attrs["dtype"],
                    f"{operation.op} dtype",
                )
    return facts


def _align_loop_body_literal_dtypes(
    loop: IntentOp,
    dtypes: Mapping[str, str],
) -> None:
    """Fill omitted loop-literal dtypes from the carry dataflow.

    Layer-A literals keep their historical f32 default when no consumer gives
    them a stronger type.  A loop carry is such a consumer: its next value must
    have the carry type, so the expected type is propagated backwards through
    value-preserving producers and select/map_binary value arms.  Authored
    attributes and result types remain authoritative and are never rewritten.
    """

    producers = _loop_body_producers(loop.body)
    facts = _loop_body_dtype_facts(loop.body, dtypes)
    expected: dict[str, str] = {}
    pending: list[str] = []

    def enqueue(name: str, dtype: str) -> None:
        normalized = _normalize_dtype(dtype, f"loop value {name!r}")
        previous = expected.get(name)
        if previous is not None:
            if previous != normalized:
                raise ValueError(
                    f"emit_mlir: loop value {name!r} is required to have both "
                    f"{previous} and {normalized} dtypes"
                )
            return
        expected[name] = normalized
        pending.append(name)

    for state in loop.carry:
        state_dtype = dtypes.get(state.next_value)
        if state_dtype is None and state.type is not None:
            if state.type.kind == "tensor" and state.type.dtype is not None:
                state_dtype = _normalize_dtype(
                    state.type.dtype,
                    "loop carry result type",
                )
        if state_dtype is not None:
            enqueue(state.next_value, state_dtype)

    comparison_fns = MAP_BINARY_PREDICATE_FNS
    preserving_ops = {
        "reshape", "transpose", "gather", "scatter", "masked_select",
        "atomic_scatter", "masked_scatter", "concat", "pad", "tile", "flip", "triu", "scan",
        "pool", "upsample",
    }

    while pending:
        value = pending.pop()
        operation = producers.get(value)
        if operation is None:
            continue
        target = expected[value]
        names = _result_names(operation)
        declared = _declared_op_dtype(operation)
        authored = facts.get(value)
        if declared is not None:
            authored = declared
        if authored is not None and authored != target:
            # Leave the authored fact untouched.  The ordinary inference pass
            # reports the precise producer/carry conflict to its caller.
            continue

        if operation.op in {"const", "rand"}:
            if "dtype" not in operation.attrs:
                operation.attrs["dtype"] = target
            facts[value] = _normalize_dtype(
                operation.attrs["dtype"],
                f"{operation.op} dtype",
            )
            continue
        if operation.op == "select":
            if len(operation.operands) == 3:
                enqueue(operation.operands[1], target)
                enqueue(operation.operands[2], target)
            continue
        if operation.op == "value_if":
            for arm in (operation.positive, operation.negative):
                if arm is not None:
                    enqueue(arm.yield_value, target)
            continue
        if operation.op == "map_binary":
            if operation.attrs.get("fn") in comparison_fns:
                continue
            operand_dtypes = [facts.get(name) for name in operation.operands]
            known = [dtype for dtype in operand_dtypes if dtype is not None]
            operand_target = target
            if operand_target is None and len(set(known)) == 1:
                operand_target = known[0]
            if operand_target is not None:
                for name, operand_dtype in zip(
                    operation.operands, operand_dtypes, strict=True
                ):
                    if operand_dtype is None:
                        enqueue(name, operand_target)
            continue
        if operation.op == "map_unary":
            if operation.attrs.get("fn") not in _U_TO_I1 and operation.operands:
                enqueue(operation.operands[0], target)
            continue
        if operation.op == "reduce":
            if operation.attrs.get("fn") not in {"all", "any"} and operation.operands:
                enqueue(operation.operands[0], target)
            continue
        if operation.op == "sort":
            if operation.operands and (len(names) == 1 or value == names[0]):
                enqueue(operation.operands[0], target)
            continue
        if operation.op in preserving_ops:
            operands = operation.operands
            if operation.op in {"gather", "scatter", "masked_select", "masked_scatter"}:
                operands = operands[:1] if operation.op == "gather" else (
                    operands[:1] if operation.op == "masked_select" else
                    [operands[0], operands[2]] if len(operands) > 2 else operands[:1]
                )
            for operand in operands:
                enqueue(operand, target)


def _infer_op_dtypes(
    op: IntentOp,
    dtypes: dict[str, str],
    ragged_types: Mapping[str, _RaggedValueType] | None = None,
) -> dict[str, str]:
    """Infer the element type(s) produced by one atomic Intent operation."""

    ragged_types = ragged_types or {}
    if op.op == "ragged":
        # A descriptor has no single element dtype.  Its two field dtypes are
        # recorded by _infer_body_dtypes; consumers recover the appropriate
        # field dtype below.
        return {}
    if op.op == "ragged_outer":
        if len(op.operands) != 1:
            return {}
        descriptor = ragged_types.get(op.operands[0])
        if descriptor is None or descriptor.offsets_dtype is None:
            return {}
        return {op.result: descriptor.offsets_dtype} if isinstance(op.result, str) else {}
    if op.op == "ragged_member":
        if len(op.operands) != 2:
            return {}
        descriptor = ragged_types.get(op.operands[0])
        if descriptor is None or descriptor.indices_dtype is None:
            return {}
        return {op.result: descriptor.indices_dtype} if isinstance(op.result, str) else {}

    names = _result_names(op)
    if not names:
        return {}
    a = op.attrs
    inferred: list[str | None]

    if op.op == "cast":
        inferred = [_cast_target_dtype(a)]
    elif op.op == "contraction":
        # Accumulator/result precision is value semantics, not Layer B.  An
        # explicit op.type wins; an externally declared result type is the next
        # source of truth, otherwise operands determine the type.
        dtype = _declared_op_dtype(op) or dtypes.get(names[0])
        inferred = [dtype or _merge_operand_dtypes(op, dtypes)]
    elif op.op == "reduce":
        inferred = [
            "i1" if a.get("fn") in ("all", "any")
            else dtypes.get(op.operands[0])
        ]
    elif op.op == "map_unary":
        fn = a.get("fn")
        if fn not in MAP_UNARY_FNS:
            raise ValueError(
                f"emit_mlir: map_unary has no closed lowering for fn {fn!r}"
            )
        input_dtype = dtypes.get(op.operands[0])
        declared_dtype = _declared_op_dtype(op) or dtypes.get(names[0])
        expected_dtype = (
            map_unary_result_dtype(fn, input_dtype)
            if input_dtype is not None
            else None
        )
        if input_dtype is not None and expected_dtype is None:
            raise ValueError(
                f"emit_mlir: map_unary fn {fn!r} has no valid closed input dtype"
            )
        if (
            expected_dtype is not None
            and declared_dtype is not None
            and declared_dtype != expected_dtype
        ):
            raise ValueError(
                f"emit_mlir: map_unary fn {fn!r} has incompatible "
                f"input/result dtypes {input_dtype!r}/{declared_dtype!r}"
            )
        if input_dtype is not None and declared_dtype is not None and (
            declared_dtype != expected_dtype
        ):
            # Keep a defensive branch for malformed partial maps where the
            # shared helper could not produce a result.
            if expected_dtype is None:
                raise ValueError(
                    f"emit_mlir: map_unary fn {fn!r} has incompatible "
                    f"input/result dtypes {input_dtype!r}/{declared_dtype!r}"
                )
        inferred = [
            declared_dtype or expected_dtype or dtypes.get(op.operands[0])
        ]
    elif op.op == "map_binary":
        fn = a.get("fn")
        if fn not in MAP_BINARY_FNS:
            raise ValueError(
                f"emit_mlir: map_binary has no closed lowering for fn {fn!r}"
            )
        operand_dtypes = [dtypes.get(name) for name in op.operands]
        operand_dtype = _merge_operand_dtypes(op, dtypes)
        context_dtype = dtypes.get(names[0])
        expected_dtype = (
            map_binary_result_dtype(fn, operand_dtypes[0], operand_dtypes[1])
            if len(operand_dtypes) == 2
            and operand_dtypes[0] is not None
            and operand_dtypes[1] is not None
            else None
        )
        if len(operand_dtypes) == 2 and all(
            dtype is not None for dtype in operand_dtypes
        ) and expected_dtype is None:
            raise ValueError(
                f"emit_mlir: map_binary fn {fn!r} has no valid closed dtype contract"
            )
        if context_dtype is not None and expected_dtype is not None and context_dtype != expected_dtype:
            raise ValueError(
                f"emit_mlir: map_binary result {names[0]!r} is required to have "
                f"dtype {expected_dtype}, got {context_dtype}"
            )
        inferred = [
            context_dtype or expected_dtype or operand_dtype
        ]
    elif op.op == "select":
        operand_dtype = _merge_operand_dtypes(op, dtypes, start=1)
        context_dtype = _declared_op_dtype(op) or dtypes.get(names[0])
        if (
            context_dtype is not None
            and operand_dtype is not None
            and context_dtype != operand_dtype
        ):
            raise ValueError(
                f"emit_mlir: select result {names[0]!r} is required to have "
                f"both {context_dtype} and {operand_dtype} dtypes"
            )
        inferred = [context_dtype or operand_dtype]
    elif op.op == "value_if":
        if op.positive is None or op.negative is None:
            raise ValueError("value_if requires two explicit arms")
        arm_dtypes = [
            dtypes.get(op.positive.yield_value),
            dtypes.get(op.negative.yield_value),
        ]
        known = [dtype for dtype in arm_dtypes if dtype is not None]
        if len(set(known)) > 1:
            raise ValueError("value_if arm yields have different dtypes")
        inferred = [_declared_op_dtype(op) or (known[0] if known else None)]
    elif op.op in {"scatter", "atomic_scatter"} and (
        op.op == "atomic_scatter" or op.attrs.get("atomic") == "cas"
    ):
        if len(names) != 2 or len(op.operands) != 4:
            raise ValueError(
                "emit_mlir: atomic compare-exchange scatter requires four operands and "
                "two results"
            )
        source_dtype = dtypes.get(op.operands[0])
        inferred = [source_dtype, source_dtype]
    elif op.op in {
        "scan", "reshape", "transpose", "gather", "scatter", "masked_select",
        "atomic_scatter", "masked_scatter", "concat", "pad", "tile", "flip", "triu",
        "pool", "upsample",
    }:
        inferred = [dtypes.get(op.operands[0])]
        if op.op == "pool" and len(names) == 2:
            inferred.append("i64")
    elif op.op == "sort":
        source = dtypes.get(op.operands[0])
        if len(names) == 2:
            inferred = [source, "i32"]
        elif a.get("return") == "indices":
            inferred = ["i32"]
        else:
            inferred = [source]
    elif op.op == "arg_reduce":
        inferred = ["i32"]
    elif op.op == "iota":
        # ``dtype`` is optional on the wire.  When it is omitted, the
        # authored result type is the only available source-of-truth for the
        # logical index width (for example ``type.dtype: index`` is the
        # canonical i64 spelling).  Falling back to i32 unconditionally here
        # makes a valid i64/index iota fail before TypeCheck and disagrees with
        # the structured reference evaluator, which already uses the result
        # type as the default.  An explicitly authored attrs.dtype remains
        # authoritative and is checked against the result type below.
        inferred = [
            _normalize_dtype(
                a["dtype"] if "dtype" in a else (
                    _declared_op_dtype(op) or "i32"
                ),
                "iota dtype",
            )
        ]
    elif op.op == "nonzero":
        inferred = ["i64"]
    elif op.op in ("const", "rand"):
        dtype = (
            a["dtype"]
            if "dtype" in a
            else _declared_op_dtype(op) or dtypes.get(names[0]) or "f32"
        )
        inferred = [_normalize_dtype(dtype, f"{op.op} dtype")]
    elif op.op in ("conv", "polar"):
        common = _merge_operand_dtypes(op, dtypes)
        inferred = [common] * len(names)
    elif op.op == "record_get":
        # The canonical elaborator resolves a record field before this emitter
        # is called.  A flat MLIR function has no first-class record ABI, so
        # leave the field dtype to the explicit result type and fail closed in
        # emit_op rather than flattening a named value implicitly.
        inferred = [None] * len(names)
    else:
        inferred = [None] * len(names)

    declared = _declared_op_dtype(op)
    if declared is not None and len(names) == 1:
        if inferred[0] is not None and inferred[0] != declared:
            raise ValueError(
                f"emit_mlir: {op.op} result {names[0]!r} declares {declared} "
                f"but its semantics infer {inferred[0]}"
            )
        inferred[0] = declared
    return {
        name: dtype for name, dtype in zip(names, inferred) if dtype is not None
    }


def _record_dtype(
    dtypes: dict[str, str], name: str, dtype: str, context: str
) -> bool:
    previous = dtypes.get(name)
    if previous is not None and previous != dtype:
        raise ValueError(
            f"emit_mlir: dtype conflict for {name!r}: {previous} versus "
            f"{dtype} ({context})"
        )
    if previous is None:
        dtypes[name] = dtype
        return True
    return False


def _infer_body_dtypes(
    body: list[IntentOp],
    dtypes: dict[str, str],
    ragged_types: dict[str, _RaggedValueType] | None = None,
) -> bool:
    ragged_types = ragged_types if ragged_types is not None else {}
    changed = False
    for op in body:
        if op.op == "loop":
            # A semantic loop is a region operation.  Its scan block arguments
            # inherit the source element type; carry block arguments and results
            # inherit the authored state type (or, for older artifacts, the
            # initializer/yield type).  All of those facts are local to the loop
            # body except the authored carry result.  In particular, a body
            # descriptor or an intermediate dtype must not become visible to a
            # parent continuation.
            loop_dtypes = dict(dtypes)
            loop_ragged_types = dict(ragged_types)
            for state in op.carry:
                loop_dtypes.pop(state.result_value, None)
            for scan in op.scan:
                source_dtype = dtypes.get(scan.source)
                if source_dtype is not None:
                    changed |= _record_dtype(
                        loop_dtypes, scan.name, source_dtype, "loop scan slice"
                    )
            if op.trip is not None and op.trip.is_dynamic:
                upper_bound = op.trip.upper_bound
                induction = op.trip.index
                if (
                    isinstance(upper_bound, str)
                    and isinstance(induction, str)
                    and upper_bound in dtypes
                ):
                    changed |= _record_dtype(
                        loop_dtypes,
                        induction,
                        dtypes[upper_bound],
                        "dynamic loop induction value",
                    )
            for state in op.carry:
                if state.type is not None and state.type.kind != "tensor":
                    raise NotImplementedError(
                        "emit_mlir: structured loop carries require tensor "
                        "values; Ragged descriptors are region-local and "
                        "cannot cross the carry ABI"
                    )
                declared = (
                    _normalize_dtype(
                        state.type.dtype, "loop carry result type"
                    )
                    if state.type is not None
                    else None
                )
                state_dtype = declared or dtypes.get(state.init_value)
                if state_dtype is not None:
                    changed |= _record_dtype(
                        loop_dtypes,
                        state.body_argument,
                        state_dtype,
                        "loop carry block argument",
                    )
                    changed |= _record_dtype(
                        loop_dtypes,
                        state.next_value,
                        state_dtype,
                        "loop carry next value",
                    )
            _align_loop_body_literal_dtypes(op, loop_dtypes)
            for _ in range(len(op.body) + len(op.carry) + 3):
                if not _infer_body_dtypes(
                    op.body, loop_dtypes, loop_ragged_types
                ):
                    break
            for state in op.carry:
                yielded_dtype = loop_dtypes.get(state.next_value)
                if yielded_dtype is not None:
                    _record_dtype(
                        loop_dtypes,
                        state.body_argument,
                        yielded_dtype,
                        "loop yielded carry",
                    )
                result_dtype = yielded_dtype or loop_dtypes.get(state.body_argument)
                if result_dtype is not None:
                    changed |= _record_dtype(
                        dtypes,
                        state.result_value,
                        result_dtype,
                        "loop result",
                    )
            continue
        if op.op in {"effect_if", "value_if"}:
            arm_scopes: list[tuple[object, dict[str, str], dict[str, _RaggedValueType]]] = []
            for arm in (op.positive, op.negative):
                if arm is None:
                    continue
                arm_dtypes = dict(dtypes)
                arm_ragged_types = dict(ragged_types)
                if op.op == "value_if" and isinstance(op.result, str):
                    arm_dtypes.pop(op.result, None)
                    arm_ragged_types.pop(op.result, None)
                for _ in range(len(arm.body) + 3):
                    if not _infer_body_dtypes(
                        arm.body, arm_dtypes, arm_ragged_types
                    ):
                        break
                arm_scopes.append((arm, arm_dtypes, arm_ragged_types))

            # Only a value_if result is exported.  Its two arm yields must agree
            # before the parent receives the merged dtype/descriptor; effect_if
            # is intentionally value-less at the region boundary.
            if op.op == "value_if":
                arm_yield_dtypes = [
                    scope_dtypes.get(arm.yield_value)
                    for arm, scope_dtypes, _scope_ragged in arm_scopes
                ]
                known = [dtype for dtype in arm_yield_dtypes if dtype is not None]
                if len(set(known)) > 1:
                    raise ValueError("value_if arm yields have different dtypes")
                declared = _declared_op_dtype(op)
                if declared is not None and any(dtype != declared for dtype in known):
                    raise ValueError(
                        "value_if arm yield dtype differs from its declared result"
                    )
                result_dtype = declared or (known[0] if known else None)
                if result_dtype is not None and isinstance(op.result, str):
                    changed |= _record_dtype(
                        dtypes, op.result, result_dtype, "value_if result"
                    )

                merged_ragged: _RaggedValueType | None = None
                for arm, _scope_dtypes, scope_ragged in arm_scopes:
                    descriptor = scope_ragged.get(arm.yield_value)
                    if descriptor is None:
                        continue
                    merged_ragged = _merge_ragged_type(
                        merged_ragged,
                        descriptor,
                        context="value_if arm result",
                    )
                if merged_ragged is not None and isinstance(op.result, str):
                    changed |= _record_ragged_type(
                        ragged_types,
                        op.result,
                        merged_ragged,
                        "value_if result",
                    )
            continue
        if op.op == "ragged" and isinstance(op.result, str):
            descriptor = ragged_types.get(op.result)
            if descriptor is not None and len(op.operands) == 2:
                refined = _RaggedValueType(
                    descriptor.offsets_shape,
                    dtypes.get(op.operands[0]) or descriptor.offsets_dtype,
                    descriptor.indices_shape,
                    dtypes.get(op.operands[1]) or descriptor.indices_dtype,
                )
                changed |= _record_ragged_type(
                    ragged_types, op.result, refined, "ragged dtype inference"
                )
        for name, dtype in _infer_op_dtypes(op, dtypes, ragged_types).items():
            changed |= _record_dtype(dtypes, name, dtype, op.op)
    return changed


def _flat_dtypes(
    fn: IntentFunction,
    ragged_types: dict[str, _RaggedValueType] | None = None,
) -> dict[str, str]:
    ragged_types = ragged_types if ragged_types is not None else {}
    dtypes = {
        tensor.name: _normalize_dtype(tensor.dtype, f"input {tensor.name!r}")
        for tensor in fn.inputs
    }
    for tensor in fn.outputs:
        _record_dtype(
            dtypes,
            tensor.name,
            _normalize_dtype(tensor.dtype, f"output {tensor.name!r}"),
            "function output ABI",
        )
    for _ in range(len(fn.body) + 3):
        if not _infer_body_dtypes(fn.body, dtypes, ragged_types):
            break
    for tensor in fn.outputs:
        declared = _normalize_dtype(tensor.dtype, f"output {tensor.name!r}")
        actual = dtypes.get(tensor.name)
        if actual is None:
            raise ValueError(
                f"emit_mlir: cannot infer dtype for output {tensor.name!r}"
            )
        if actual != declared:
            raise ValueError(
                f"emit_mlir: output {tensor.name!r} declares {declared} but "
                f"the produced SSA value has dtype {actual}; add an explicit cast"
            )
    return dtypes


def _loop_dtypes(
    fn: IntentFunction,
    loop: IntentOp,
    ragged_types: dict[str, _RaggedValueType] | None = None,
) -> dict[str, str]:
    ragged_types = ragged_types if ragged_types is not None else {}
    dtypes = {
        tensor.name: _normalize_dtype(tensor.dtype, f"input {tensor.name!r}")
        for tensor in fn.inputs
    }
    loop_position = next(
        index for index, operation in enumerate(fn.body) if operation is loop
    )
    # A dynamic loop upper bound may be a dominating producer (for example
    # ragged -> ragged_outer), not a function argument.  Infer the prefix in
    # the same lexical order as emit_module so the specialized loop emitter
    # cannot lose that binding.
    prefix = fn.body[:loop_position]
    for _ in range(len(prefix) + 3):
        if not _infer_body_dtypes(prefix, dtypes, ragged_types):
            break
    for scan in loop.scan:
        source = scan.source
        source_dtype = dtypes.get(source)
        if source_dtype is not None:
            _record_dtype(
                dtypes, scan.name, source_dtype, "loop scan slice"
            )

    if loop.trip is not None and loop.trip.is_dynamic:
        upper_bound = loop.trip.upper_bound
        induction = loop.trip.index
        if (
            isinstance(upper_bound, str)
            and isinstance(induction, str)
            and upper_bound in dtypes
        ):
            _record_dtype(
                dtypes,
                induction,
                dtypes[upper_bound],
                "dynamic loop induction value",
            )

    result_names = _result_names(loop)
    validate_loop_wiring(loop, context="emit_mlir loop")
    carries = [item.body_argument for item in loop.carry]
    for item in loop.carry:
        state_type = item.type
        if state_type is not None:
            if state_type.kind != "tensor" or state_type.dtype is None:
                raise NotImplementedError(
                    "emit_mlir: structured loop carries require tensor "
                    "values; Ragged descriptors are region-local and "
                    "cannot cross the carry ABI"
                )
            carry_dtype = _normalize_dtype(
                state_type.dtype,
                "loop carry result type",
            )
            _record_dtype(
                dtypes,
                item.body_argument,
                carry_dtype,
                "authored loop carry type",
            )
            _record_dtype(
                dtypes,
                item.result_value,
                carry_dtype,
                "authored loop result type",
            )
        init = item.init_value
        if isinstance(init, str) and init in dtypes:
            _record_dtype(
                dtypes,
                item.body_argument,
                dtypes[init],
                "loop carry init",
            )
    for item in loop.carry:
        carry_dtype = dtypes.get(item.body_argument)
        if carry_dtype is not None:
            _record_dtype(
                dtypes,
                item.next_value,
                carry_dtype,
                "authored loop next value type",
            )
    _align_loop_body_literal_dtypes(loop, dtypes)
    output_dtypes = {
        tensor.name: _normalize_dtype(tensor.dtype, f"output {tensor.name!r}")
        for tensor in fn.outputs
    }
    for result, carry in zip(result_names, carries):
        if result in output_dtypes:
            _record_dtype(dtypes, carry, output_dtypes[result], "loop result ABI")

    for _ in range(len(loop.body) + len(carries) + 3):
        changed = _infer_body_dtypes(loop.body, dtypes, ragged_types)
        for carry, yielded, result in zip(carries, loop.yields, result_names):
            yielded_dtype = dtypes.get(yielded)
            if yielded_dtype is not None:
                changed |= _record_dtype(
                    dtypes, carry, yielded_dtype, "loop yielded carry"
                )
                changed |= _record_dtype(
                    dtypes, result, yielded_dtype, "loop result"
                )
        if not changed:
            break

    required = set(carries) | set(loop.yields)
    for op in loop.body:
        required.update(op.operands)
        required.update(_result_names(op))
    missing = sorted(
        name
        for name in required
        if name not in dtypes and name not in ragged_types
    )
    if missing:
        raise ValueError(
            "emit_mlir: cannot infer loop-body dtype(s): " + ", ".join(missing)
        )
    return dtypes


_MLIR_SPECIAL_FLOAT_BITS = {
    "f16": {
        "pos_inf": "0x7C00",
        "neg_inf": "0xFC00",
        "nan": "0x7E00",
    },
    "bf16": {
        "pos_inf": "0x7F80",
        "neg_inf": "0xFF80",
        "nan": "0x7FC0",
    },
    "f32": {
        "pos_inf": "0x7F800000",
        "neg_inf": "0xFF800000",
        "nan": "0x7FC00000",
    },
    "f64": {
        "pos_inf": "0x7FF0000000000000",
        "neg_inf": "0xFFF0000000000000",
        "nan": "0x7FF8000000000000",
    },
}


def _mlir_f(v, dtype: str = "f64") -> str:
    """Format a Python float as an MLIR-parseable float literal. MLIR requires a
    decimal point before the optional exponent, so bare scientific literals like
    ``1e-05`` (which ``repr`` emits) must become ``1.0e-05``.  MLIR's textual
    dense-element parser does not accept ``inf``/``nan`` spellings, so IEEE
    specials use the exact bit-pattern form accepted by DenseElementsAttr.
    """
    value = float(v)
    if math.isnan(value) or math.isinf(value):
        bits = _MLIR_SPECIAL_FLOAT_BITS.get(dtype)
        if bits is None:
            raise ValueError(
                f"unsupported floating dtype for special literal: {dtype}"
            )
        if math.isnan(value):
            return bits["nan"]
        return bits["neg_inf" if value < 0 else "pos_inf"]
    s = repr(value)
    if "e" in s or "E" in s:
        e = "e" if "e" in s else "E"
        mant, _, exp = s.partition(e)
        if "." not in mant:
            mant += ".0"
        return f"{mant}e{exp}"
    if "." not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


def _mlir_dense_literal(
    values: Sequence[object],
    shape: Sequence[int],
    formatter,
) -> str:
    """Render a dense literal with nesting that matches its tensor shape."""

    if not shape:
        if len(values) != 1:
            raise ValueError("emit_mlir: scalar dense literal has multiple values")
        return formatter(values[0])
    position = 0

    def render(axis: int) -> str:
        nonlocal position
        extent = shape[axis]
        if extent == 0:
            return "[]"
        if axis == len(shape) - 1:
            items = []
            for _ in range(extent):
                if position >= len(values):
                    raise ValueError("emit_mlir: dense literal has too few values")
                items.append(formatter(values[position]))
                position += 1
            return "[" + ", ".join(items) + "]"
        return "[" + ", ".join(render(axis + 1) for _ in range(extent)) + "]"

    rendered = render(0)
    if position != len(values):
        raise ValueError("emit_mlir: dense literal has too many values")
    return rendered


def _mlir_dense_float(v, dtype: str) -> str:
    """Format one dense floating literal, including IEEE non-finite values."""

    value = float(v)
    if math.isfinite(value):
        return _mlir_f(value)
    normalized = dtype.removeprefix("torch.").lower()
    encodings = {
        "f16": ("0x7C00", "0xFC00", "0x7E00"),
        "float16": ("0x7C00", "0xFC00", "0x7E00"),
        "bf16": ("0x7F80", "0xFF80", "0x7FC0"),
        "bfloat16": ("0x7F80", "0xFF80", "0x7FC0"),
        "f32": ("0x7F800000", "0xFF800000", "0x7FC00000"),
        "float32": ("0x7F800000", "0xFF800000", "0x7FC00000"),
        "f64": (
            "0x7FF0000000000000",
            "0xFFF0000000000000",
            "0x7FF8000000000000",
        ),
        "float64": (
            "0x7FF0000000000000",
            "0xFFF0000000000000",
            "0x7FF8000000000000",
        ),
    }
    if normalized not in encodings:
        raise ValueError(
            f"emit_mlir: non-finite dense literal has unsupported dtype {dtype!r}"
        )
    positive_inf, negative_inf, quiet_nan = encodings[normalized]
    if math.isnan(value):
        return quiet_nan
    return negative_inf if value < 0 else positive_inf


def _infer_body(
    body: list[IntentOp],
    env: dict[str, list[object]],
    fold: int,
    extents: dict[str, int] | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    loop_schedule_tile: int | None = None,
    ragged_types: dict[str, _RaggedValueType] | None = None,
    dtypes: Mapping[str, str] | None = None,
    raw_ranks: dict[str, int] | None = None,
) -> None:
    ragged_types = ragged_types if ragged_types is not None else {}
    raw_ranks = raw_ranks if raw_ranks is not None else {}
    for _ in range(len(body) + 2):
        changed = False
        # Structural consumers can constrain an otherwise ambiguous producer.
        # Most importantly, a declared output after transpose determines the
        # transpose operand rank/shape, which lets an intermediate reduce choose
        # collapsed vs keepdims without relying on a model-specific heuristic.
        for op in reversed(body):
            if not isinstance(op.result, str) or op.result not in env:
                continue
            result_shape = env[op.result]
            if op.op in {"map_unary", "cast"} and len(op.operands) == 1:
                operand = op.operands[0]
                if operand not in env:
                    env[operand] = list(result_shape)
                    changed = True
                if op.result in raw_ranks and operand not in raw_ranks:
                    changed |= _record_raw_rank(
                        raw_ranks,
                        operand,
                        raw_ranks[op.result],
                        "reverse shape constraint",
                    )
                continue
            if op.op in {"map_binary", "select"}:
                axes = _op_broadcast_axes(op, broadcast_axes)
                if not axes:
                    continue
                for operand in op.operands:
                    if operand not in env and not axes.get(operand):
                        env[operand] = list(result_shape)
                        changed = True
                    if (
                        isinstance(op.result, str)
                        and op.result in raw_ranks
                        and operand not in raw_ranks
                        and not axes.get(operand)
                    ):
                        changed |= _record_raw_rank(
                            raw_ranks,
                            operand,
                            raw_ranks[op.result],
                            "reverse broadcast constraint",
                        )
                continue
            if (
                op.op != "transpose"
                or len(op.operands) != 1
            ):
                continue
            operand = op.operands[0]
            perm = _fold_permutation(op.attrs["perm"], fold, extents or {})
            if sorted(perm) != list(range(len(result_shape))):
                raise ValueError(
                    f"transpose perm {perm!r} is invalid for declared result "
                    f"rank {len(result_shape)}"
                )
            inferred_operand = [0] * len(result_shape)
            for out_dim, input_dim in enumerate(perm):
                inferred_operand[input_dim] = result_shape[out_dim]
            previous = env.get(operand)
            if previous is None:
                env[operand] = inferred_operand
                changed = True
            elif previous != inferred_operand:
                raise ValueError(
                    f"transpose operand {operand!r} shape {previous} conflicts "
                    f"with result constraint {inferred_operand}"
                )
            if (
                isinstance(op.result, str)
                and op.result in raw_ranks
                and operand not in raw_ranks
            ):
                changed |= _record_raw_rank(
                    raw_ranks,
                    operand,
                    raw_ranks[op.result],
                    "reverse transpose constraint",
                )
        for op in body:
            if op.op == "loop":
                loop_env, body_extents, tile, loop_raw_ranks = _prepare_loop_shape_scope(
                    op,
                    env,
                    fold,
                    extents,
                    loop_schedule_tile,
                    raw_ranks,
                )
                loop_ragged_types = dict(ragged_types)
                _infer_body(
                    op.body,
                    loop_env,
                    fold,
                    body_extents,
                    broadcast_axes,
                    loop_schedule_tile,
                    loop_ragged_types,
                    dtypes,
                    loop_raw_ranks,
                )
                for state in op.carry:
                    state_shape = loop_env.get(state.body_argument)
                    yielded_shape = loop_env.get(state.next_value)
                    if (
                        state_shape is not None
                        and yielded_shape is not None
                        and state_shape != yielded_shape
                    ):
                        raise ValueError(
                            "emit_mlir: loop yielded value shape differs from carry"
                        )
                    result_shape = yielded_shape or state_shape
                    if result_shape is not None:
                        changed |= _record_shape(
                            env,
                            state.result_value,
                            result_shape,
                            "loop result",
                        )
                    if state.body_argument in loop_raw_ranks:
                        changed |= _record_raw_rank(
                            raw_ranks,
                            state.result_value,
                            loop_raw_ranks[state.body_argument],
                            "loop result",
                        )
                continue
            if op.op in {"effect_if", "value_if"}:
                arm_scopes: list[
                    tuple[
                        object,
                        dict[str, list[object]],
                        dict[str, _RaggedValueType],
                        dict[str, int],
                    ]
                ] = []
                for arm in (op.positive, op.negative):
                    if arm is None:
                        continue
                    arm_env = dict(env)
                    arm_ragged_types = dict(ragged_types)
                    arm_raw_ranks = dict(raw_ranks)
                    if op.op == "value_if" and isinstance(op.result, str):
                        arm_env.pop(op.result, None)
                        arm_ragged_types.pop(op.result, None)
                    _infer_body(
                        arm.body,
                        arm_env,
                        fold,
                        extents,
                        broadcast_axes,
                        loop_schedule_tile,
                        arm_ragged_types,
                        dtypes,
                        arm_raw_ranks,
                    )
                    arm_scopes.append((arm, arm_env, arm_ragged_types, arm_raw_ranks))

                # A conditional's only legal value export is the value_if
                # result.  The effect_if arms are deliberately discarded at
                # the parent boundary.
                if op.op == "value_if":
                    arm_shapes = [
                        arm_env.get(arm.yield_value)
                        for arm, arm_env, _scope_ragged, _scope_raw_ranks
                        in arm_scopes
                    ]
                    known_shapes = [shape for shape in arm_shapes if shape is not None]
                    if any(shape != known_shapes[0] for shape in known_shapes[1:]):
                        raise ValueError("value_if arm yields have different shapes")
                    declared = _op_declared_shape(op, fold, extents)
                    existing = env.get(op.result) if isinstance(op.result, str) else None
                    if (
                        declared is not None
                        and existing is not None
                        and declared != existing
                    ):
                        raise ValueError(
                            f"value_if result {op.result!r} declaration conflicts "
                            "with its output constraint"
                        )
                    target_shape = declared or existing or (
                        known_shapes[0] if known_shapes else None
                    )
                    if target_shape is not None:
                        if any(shape != target_shape for shape in known_shapes):
                            raise ValueError(
                                f"value_if result {op.result!r} shape differs from "
                                "an arm yield"
                            )
                        if isinstance(op.result, str):
                            changed |= _record_shape(
                                env,
                                op.result,
                                target_shape,
                                "value_if result",
                            )

                    merged_ragged: _RaggedValueType | None = None
                    for arm, _arm_env, scope_ragged, _scope_raw_ranks in arm_scopes:
                        descriptor = scope_ragged.get(arm.yield_value)
                        if descriptor is None:
                            continue
                        merged_ragged = _merge_ragged_type(
                            merged_ragged,
                            descriptor,
                            context="value_if arm result",
                        )
                    if merged_ragged is not None and isinstance(op.result, str):
                        changed |= _record_ragged_type(
                            ragged_types,
                            op.result,
                            merged_ragged,
                            "value_if result",
                        )
                    if isinstance(op.result, str):
                        declared_rank = _declared_raw_rank(op)
                        known_ranks = [
                            scope_raw_ranks.get(arm.yield_value)
                            for arm, _arm_env, _scope_ragged, scope_raw_ranks
                            in arm_scopes
                            if scope_raw_ranks.get(arm.yield_value) is not None
                        ]
                        result_rank = declared_rank
                        if result_rank is None:
                            if len(set(known_ranks)) > 1:
                                raise ValueError(
                                    "emit_mlir: value_if arm yields have "
                                    "different authored ranks"
                                )
                            result_rank = known_ranks[0] if known_ranks else None
                        if result_rank is not None:
                            changed |= _record_raw_rank(
                                raw_ranks,
                                op.result,
                                result_rank,
                                "value_if result",
                            )
                continue
            result_names = _result_names(op)
            result_rank = _operation_raw_result_rank(
                op, raw_ranks, broadcast_axes
            )
            if result_rank is not None:
                for name in result_names:
                    if isinstance(name, str):
                        changed |= _record_raw_rank(
                            raw_ranks,
                            name,
                            result_rank,
                            f"{op.op} result",
                        )
            sh = _infer_op(
                op,
                env,
                fold,
                extents,
                broadcast_axes,
                ragged_types,
                dtypes,
                raw_ranks,
            )
            if sh is None:
                continue
            # Multi-result ops (sort -> values, indices) name every result; all
            # results share the inferred shape.
            names = op.result if isinstance(op.result, list) else [op.result]
            for nm in names:
                if env.get(nm) != sh:
                    env[nm] = sh
                    changed = True
        if not changed:
            break


class _Printer:
    def __init__(
        self,
        dtypes: dict[str, str],
        broadcast_axes: BroadcastAxes | None = None,
        proven_index_bounds: IndexBoundsProjection | None = None,
        loop_schedule_tile: int | None = None,
        ragged_types: Mapping[str, _RaggedValueType] | None = None,
        raw_ranks: Mapping[str, int] | None = None,
    ) -> None:
        self.lines: list[str] = []
        self._n = 0
        self.ssa: dict[str, str] = {}   # JSON name -> SSA name
        self.dtypes = dtypes
        self.broadcast_axes = broadcast_axes or {}
        self.proven_index_bounds = proven_index_bounds or frozenset()
        self.loop_schedule_tile = loop_schedule_tile
        self.ragged_types = dict(ragged_types or {})
        self.raw_ranks = dict(raw_ranks or {})

    def _ttn(self, name: str, env: Mapping[str, list[object]]) -> str:
        """Render one JSON value's tensor or ragged descriptor MLIR type."""
        ragged = self.ragged_types.get(name)
        if ragged is not None:
            return _ragged_type_text(ragged)
        if name not in self.dtypes:
            raise ValueError(f"emit_mlir: no dtype inferred for {name!r}")
        return _tt(env[name], self.dtypes[name])

    def _ttn_shape(self, name: str, shape: Sequence[int]) -> str:
        if name in self.ragged_types:
            raise ValueError(
                f"emit_mlir: ragged descriptor {name!r} cannot use a tensor shape"
            )
        if name not in self.dtypes:
            raise ValueError(f"emit_mlir: no dtype inferred for {name!r}")
        return _tt(list(shape), self.dtypes[name])

    def _broadcast_operand(
        self,
        name: str,
        env: Mapping[str, list[object]],
        axes: Sequence[int],
        indent: str,
        consumer_shape: Sequence[int] | None = None,
        element_dtype: str | None = None,
    ) -> tuple[str, str]:
        source_shape = env[name]
        target_shape, inserted_axes = _broadcast_projection(
            source_shape, axes, consumer_shape
        )
        assert target_shape is not None
        source_type = self._ttn(name, env)
        value = self.ssa[name]
        projected_dtype = element_dtype or self.dtypes[name]
        projected_source_type = _tt(source_shape, projected_dtype)
        if projected_dtype != self.dtypes[name]:
            self._n += 1
            promoted = f"%{name}_promote_{self._n}"
            self.lines.append(
                f'{indent}{promoted} = intent.cast {value} '
                f'{{to_dtype = "{projected_dtype}"}} : '
                f'({source_type}) -> {projected_source_type}'
            )
            value = promoted
        target_type = _tt(target_shape, projected_dtype)
        # Rank-0 tensors already broadcast through an empty affine input map;
        # no tensor.expand_shape is needed for explicit scalar axes.
        if not axes or not source_shape or target_shape == source_shape:
            return value, projected_source_type
        rank_gap = len(target_shape) - len(source_shape)
        standard_trailing_vector = (
            consumer_shape is not None
            and len(source_shape) == 1
            and inserted_axes == tuple(range(rank_gap))
            and source_shape == target_shape[rank_gap:]
        )
        if standard_trailing_vector:
            # Intent map lowering and the row fast path both understand the
            # conventional trailing rank-1 parameter broadcast directly.  Keep
            # it rank-1 so specialized row kernels can classify it as a column
            # parameter; non-trailing projections still materialize an explicit
            # expand_shape below.
            return value, projected_source_type

        inserted = set(inserted_axes)
        source_positions = [
            index for index in range(len(target_shape)) if index not in inserted
        ]
        reassociation: list[list[int]] = []
        for index, position in enumerate(source_positions):
            start = 0 if index == 0 else position
            end = (
                source_positions[index + 1]
                if index + 1 < len(source_positions)
                else len(target_shape)
            )
            reassociation.append(list(range(start, end)))
        groups = "[" + ", ".join(
            "[" + ", ".join(str(value) for value in group) + "]"
            for group in reassociation
        ) + "]"
        output_shape = ", ".join(str(value) for value in target_shape)
        self._n += 1
        result = f"%{name}_broadcast_{self._n}"
        self.lines.append(
            f"{indent}{result} = tensor.expand_shape {value} {groups} "
            f"output_shape [{output_shape}] : {projected_source_type} "
            f"into {target_type}"
        )
        return result, target_type

    def fresh(self, name: str) -> str:
        self._n += 1
        s = f"%{name}_{self._n}"
        self.ssa[name] = s
        return s

    def arg(self, name: str) -> str:
        """Clean SSA name for a function/block arg (no numeric suffix), so the
        orchestrator can key inputs by the original JSON name."""
        s = f"%{name}"
        self.ssa[name] = s
        return s

    def emit_op(self, op: IntentOp, env: Mapping[str, list[object]], fold: int,
                indent: str = "    ",
                extents: dict[str, int] | None = None,
                node_ref: str | None = None) -> None:
        a = op.attrs
        if op.op in RAGGED_OPS:
            if not isinstance(op.result, str):
                raise ValueError(f"emit_mlir: {op.op} requires one named result")
            result = self.fresh(op.result)
            operand_values = [self.ssa[name] for name in op.operands]
            operand_types = [self._ttn(name, env) for name in op.operands]
            result_type = self._ttn(op.result, env)
            if op.op == "ragged":
                if len(op.operands) != 2:
                    raise ValueError(
                        "emit_mlir: ragged requires offsets and indices operands"
                    )
                self.lines.append(
                    f"{indent}{result} = intent.ragged "
                    f"{operand_values[0]}, {operand_values[1]} : "
                    f"({operand_types[0]}, {operand_types[1]}) -> {result_type}"
                )
            elif op.op == "ragged_outer":
                if len(op.operands) != 1:
                    raise ValueError(
                        "emit_mlir: ragged_outer requires one descriptor operand"
                    )
                self.lines.append(
                    f"{indent}{result} = intent.ragged_outer {operand_values[0]} : "
                    f"({operand_types[0]}) -> {result_type}"
                )
            else:
                if len(op.operands) != 2:
                    raise ValueError(
                        "emit_mlir: ragged_member requires descriptor and index"
                    )
                self.lines.append(
                    f"{indent}{result} = intent.ragged_member "
                    f"{operand_values[0]}, {operand_values[1]} : "
                    f"({operand_types[0]}, {operand_types[1]}) -> {result_type}"
                )
            return
        if op.op == "loop":
            if op.trip is None or not op.carry:
                raise ValueError(
                    "emit_mlir: loop requires trip and carry state"
                )
            if op.trip.is_dynamic and op.scan:
                raise ValueError(
                    "emit_mlir: dynamic value loops cannot have scan inputs"
                )
            if len(op.yields) != len(op.carry):
                raise ValueError("emit_mlir: loop carry/yield arity mismatch")
            loop_env, body_extents, tile, loop_raw_ranks = _prepare_loop_shape_scope(
                op,
                env,
                fold,
                extents,
                self.loop_schedule_tile,
                self.raw_ranks,
            )
            loop_dtypes = dict(self.dtypes)
            loop_ragged_types = dict(self.ragged_types)
            for state in op.carry:
                loop_dtypes.pop(state.result_value, None)
            for scan in op.scan:
                source_dtype = self.dtypes.get(scan.source)
                if source_dtype is not None:
                    _record_dtype(
                        loop_dtypes,
                        scan.name,
                        source_dtype,
                        "loop scan slice",
                    )
            if op.trip.is_dynamic:
                upper_bound = op.trip.upper_bound
                induction = op.trip.index
                if not isinstance(upper_bound, str) or not isinstance(induction, str):
                    raise ValueError(
                        "emit_mlir: dynamic loop requires upper_bound and index bindings"
                    )
                upper_shape = loop_env.get(upper_bound)
                if upper_shape is not None and upper_shape != []:
                    raise ValueError(
                        "emit_mlir: dynamic loop upper_bound must be a rank-0 tensor"
                    )
                upper_dtype = self.dtypes.get(upper_bound)
                if upper_dtype is None:
                    raise ValueError(
                        f"emit_mlir: dynamic loop upper_bound {upper_bound!r} "
                        "has no inferred dtype"
                    )
                _record_dtype(
                    loop_dtypes,
                    induction,
                    upper_dtype,
                    "dynamic loop induction value",
                )
            for state in op.carry:
                if state.type is not None and state.type.kind != "tensor":
                    raise NotImplementedError(
                        "emit_mlir: named record loop carries require a "
                        "record-aware ABI"
                    )
                declared_dtype = (
                    _normalize_dtype(state.type.dtype, "loop carry result type")
                    if state.type is not None and state.type.dtype is not None
                    else None
                )
                state_dtype = declared_dtype or loop_dtypes.get(state.init_value)
                if state_dtype is not None:
                    _record_dtype(
                        loop_dtypes,
                        state.body_argument,
                        state_dtype,
                        "loop carry block argument",
                    )
                    _record_dtype(
                        loop_dtypes,
                        state.next_value,
                        state_dtype,
                        "loop carry next value",
                    )
            _align_loop_body_literal_dtypes(op, loop_dtypes)
            for _ in range(len(op.body) + len(op.carry) + 3):
                if not _infer_body_dtypes(
                    op.body, loop_dtypes, loop_ragged_types
                ):
                    break
            _infer_body(
                op.body,
                loop_env,
                fold,
                body_extents,
                self.broadcast_axes,
                self.loop_schedule_tile,
                loop_ragged_types,
                loop_dtypes,
                loop_raw_ranks,
            )
            scan_values: list[str] = []
            scan_types: list[str] = []
            scan_axes: list[str] = []
            for scan in op.scan:
                if scan.source not in self.ssa:
                    raise ValueError(
                        f"emit_mlir: loop scan source {scan.source!r} is unavailable"
                )
                scan_values.append(self.ssa[scan.source])
                scan_types.append(self._ttn(scan.source, env))
                scan_rank = _raw_rank_for_value(
                    scan.source,
                    env,
                    loop_raw_ranks,
                    context="loop scan source",
                )
                scan_axes.append(f"{_fold_axis(scan.axis, scan_rank, fold)} : i64")

            scan_clause = (
                f"scan({', '.join(scan_values)} : {', '.join(scan_types)})"
                if scan_values
                else "scan()"
            )

            init_values: list[str] = []
            init_types: list[str] = []
            result_types: list[str] = []
            for state in op.carry:
                if state.init_value not in self.ssa:
                    raise ValueError(
                        f"emit_mlir: loop carry init {state.init_value!r} is unavailable"
                    )
                init_values.append(self.ssa[state.init_value])
                init_types.append(self._ttn(state.init_value, env))
                result_types.append(self._ttn(state.result_value, env))

            self._n += 1
            loop_value = f"%intentir_loop_{self._n}"
            upper_bound_clause = ""
            if op.trip.is_dynamic:
                upper_bound = op.trip.upper_bound
                assert isinstance(upper_bound, str)
                if upper_bound not in self.ssa:
                    raise ValueError(
                        f"emit_mlir: dynamic loop upper_bound {upper_bound!r} "
                        "is unavailable"
                    )
                upper_bound_clause = (
                    f" upper_bound({self.ssa[upper_bound]} : "
                    f"{self._ttn(upper_bound, env)})"
                )
            tile_clause = (
                f", tile_hint = {tile} : i64" if not op.trip.is_dynamic else ""
            )
            self.lines.append(
                f"{indent}{loop_value}:{len(op.carry)} = intent.loop "
                f"{scan_clause} "
                f"init({', '.join(init_values)} : {', '.join(init_types)}) "
                f"{upper_bound_clause} "
                f"{{scan_axes = [{', '.join(scan_axes)}]{tile_clause}}} "
                f"-> {', '.join(result_types)} do {{"
            )

            outer_ssa = dict(self.ssa)
            outer_dtypes = self.dtypes
            outer_ragged_types = self.ragged_types
            outer_raw_ranks = self.raw_ranks
            self.dtypes = loop_dtypes
            self.ragged_types = loop_ragged_types
            self.raw_ranks = loop_raw_ranks
            block_args: list[str] = []
            for scan in op.scan:
                block = self.fresh(scan.name)
                block_args.append(f"{block}: {self._ttn(scan.name, loop_env)}")
            if op.trip.is_dynamic:
                assert op.trip.index is not None
                block = self.fresh(op.trip.index)
                block_args.append(
                    f"{block}: {self._ttn(op.trip.index, loop_env)}"
                )
            for state in op.carry:
                block = self.fresh(state.body_argument)
                block_args.append(
                    f"{block}: {self._ttn(state.body_argument, loop_env)}"
                )
            self.lines.append(f"{indent}^bb0({', '.join(block_args)}):")
            for child_index, child in enumerate(op.body):
                self.emit_op(
                    child,
                    loop_env,
                    fold,
                    indent=indent + "  ",
                    extents=body_extents,
                    node_ref=(
                        f"{node_ref}.body[{child_index}]"
                        if node_ref is not None
                        else None
                    ),
                )
            yielded_values = []
            yielded_types = []
            for yielded in op.yields:
                if yielded not in self.ssa:
                    raise ValueError(
                        f"emit_mlir: loop yield {yielded!r} is unavailable"
                    )
                yielded_values.append(self.ssa[yielded])
                yielded_types.append(self._ttn(yielded, loop_env))
            self.lines.append(
                f"{indent}  intent.yield {', '.join(yielded_values)} : "
                f"{', '.join(yielded_types)}"
            )
            self.lines.append(f"{indent}}}")
            self.ssa = outer_ssa
            self.dtypes = outer_dtypes
            self.ragged_types = outer_ragged_types
            self.raw_ranks = outer_raw_ranks
            for index, state in enumerate(op.carry):
                self.ssa[state.result_value] = f"{loop_value}#{index}"
                if state.body_argument in loop_raw_ranks:
                    _record_raw_rank(
                        self.raw_ranks,
                        state.result_value,
                        loop_raw_ranks[state.body_argument],
                        "loop result",
                    )
            return
        if op.op == "effect_if":
            if (
                not isinstance(op.predicate, str)
                or op.positive is None
                or op.negative is None
            ):
                raise ValueError(
                    "emit_mlir: effect_if requires a predicate and two arms"
                )
            if env.get(op.predicate) != []:
                raise ValueError(
                    "emit_mlir: effect_if predicate must be a rank-0 tensor"
                )
            if self.dtypes.get(op.predicate) != "i1":
                raise ValueError(
                    "emit_mlir: effect_if predicate must have i1 dtype"
                )
            if op.predicate not in self.ssa:
                raise ValueError(
                    f"emit_mlir: effect_if predicate {op.predicate!r} is unavailable"
                )

            condition_tensor = self.ssa[op.predicate]
            self._n += 1
            condition = f"%effect_if_condition_{self._n}"
            self.lines.append(
                f"{indent}{condition} = tensor.extract {condition_tensor}[] : "
                f"{self._ttn(op.predicate, env)}"
            )
            self.lines.append(f"{indent}scf.if {condition} {{")

            outer_ssa = dict(self.ssa)
            outer_dtypes = self.dtypes
            outer_ragged_types = self.ragged_types
            outer_raw_ranks = self.raw_ranks

            def emit_effect_arm(arm, arm_name: str) -> None:
                arm_env = dict(env)
                arm_dtypes = dict(outer_dtypes)
                arm_ragged_types = dict(outer_ragged_types)
                arm_raw_ranks = dict(outer_raw_ranks)
                _infer_body(
                    arm.body,
                    arm_env,
                    fold,
                    extents,
                    self.broadcast_axes,
                    self.loop_schedule_tile,
                    arm_ragged_types,
                    arm_dtypes,
                    arm_raw_ranks,
                )
                for _ in range(len(arm.body) + 3):
                    if not _infer_body_dtypes(
                        arm.body, arm_dtypes, arm_ragged_types
                    ):
                        break
                self.ssa = dict(outer_ssa)
                self.dtypes = arm_dtypes
                self.ragged_types = arm_ragged_types
                self.raw_ranks = arm_raw_ranks
                for child_index, child in enumerate(arm.body):
                    self.emit_op(
                        child,
                        arm_env,
                        fold,
                        indent=indent + "  ",
                        extents=extents,
                        node_ref=(
                            f"{node_ref}.{arm_name}[{child_index}]"
                            if node_ref is not None
                            else None
                        ),
                    )
                # effect_if deliberately has no result; the region terminator
                # must therefore carry no operands.  Values created above stay
                # in this arm's private SSA/dtype environments.
                self.lines.append(f"{indent}  scf.yield")

            emit_effect_arm(op.positive, "positive")
            self.lines.append(f"{indent}}} else {{")
            emit_effect_arm(op.negative, "negative")
            self.lines.append(f"{indent}}}")
            self.ssa = outer_ssa
            self.dtypes = outer_dtypes
            self.ragged_types = outer_ragged_types
            self.raw_ranks = outer_raw_ranks
            return
        if op.op == "value_if":
            if (
                not isinstance(op.result, str)
                or op.positive is None
                or op.negative is None
                or not isinstance(op.predicate, str)
            ):
                raise ValueError(
                    "emit_mlir: value_if requires one result, predicate, and two arms"
                )
            if env.get(op.predicate) != []:
                raise ValueError(
                    "emit_mlir: value_if predicate must be a rank-0 tensor"
                )
            if self.dtypes.get(op.predicate) != "i1":
                raise ValueError("emit_mlir: value_if predicate must have i1 dtype")

            condition_tensor = self.ssa[op.predicate]
            self._n += 1
            condition = f"%{op.result}_condition_{self._n}"
            self.lines.append(
                f"{indent}{condition} = tensor.extract {condition_tensor}[] : "
                f"{self._ttn(op.predicate, env)}"
            )
            self._n += 1
            result = f"%{op.result}_{self._n}"
            result_type = self._ttn(op.result, env)
            self.lines.append(
                f"{indent}{result} = scf.if {condition} -> ({result_type}) {{"
            )

            outer_ssa = dict(self.ssa)
            outer_dtypes = self.dtypes
            outer_ragged_types = self.ragged_types
            outer_raw_ranks = self.raw_ranks

            def emit_arm(arm, arm_name: str) -> None:
                arm_env = dict(env)
                arm_ragged_types = dict(outer_ragged_types)
                arm_dtypes = dict(outer_dtypes)
                arm_raw_ranks = dict(outer_raw_ranks)
                if isinstance(op.result, str):
                    arm_env.pop(op.result, None)
                    arm_ragged_types.pop(op.result, None)
                    arm_dtypes.pop(op.result, None)
                _infer_body(
                    arm.body,
                    arm_env,
                    fold,
                    extents,
                    self.broadcast_axes,
                    self.loop_schedule_tile,
                    arm_ragged_types,
                    arm_dtypes,
                    arm_raw_ranks,
                )
                for _ in range(len(arm.body) + 3):
                    if not _infer_body_dtypes(
                        arm.body, arm_dtypes, arm_ragged_types
                    ):
                        break
                self.ssa = dict(outer_ssa)
                self.dtypes = arm_dtypes
                self.ragged_types = arm_ragged_types
                self.raw_ranks = arm_raw_ranks
                for child_index, child in enumerate(arm.body):
                    self.emit_op(
                        child,
                        arm_env,
                        fold,
                        indent=indent + "  ",
                        extents=extents,
                        node_ref=(
                            f"{node_ref}.{arm_name}[{child_index}]"
                            if node_ref is not None
                            else None
                        ),
                    )
                yielded = arm.yield_value
                if yielded not in self.ssa:
                    raise ValueError(
                        f"emit_mlir: value_if {arm_name} arm yield "
                        f"{yielded!r} is unavailable"
                    )
                result_ragged = outer_ragged_types.get(op.result)
                yield_ragged = self.ragged_types.get(yielded)
                if result_ragged is not None or yield_ragged is not None:
                    if yield_ragged != result_ragged:
                        raise ValueError(
                            f"emit_mlir: value_if {arm_name} arm yield descriptor "
                            "differs from its result"
                        )
                elif arm_env.get(yielded) != env.get(op.result):
                    raise ValueError(
                        f"emit_mlir: value_if {arm_name} arm yield shape "
                        "differs from its result"
                    )
                if self.dtypes.get(yielded) != outer_dtypes.get(op.result):
                    raise ValueError(
                        f"emit_mlir: value_if {arm_name} arm yield dtype "
                        "differs from its result"
                    )
                self.lines.append(
                    f"{indent}  scf.yield {self.ssa[yielded]} : {result_type}"
                )

            emit_arm(op.positive, "positive")
            self.lines.append(f"{indent}}} else {{")
            emit_arm(op.negative, "negative")
            self.lines.append(f"{indent}}}")
            self.ssa = outer_ssa
            self.dtypes = outer_dtypes
            self.ragged_types = outer_ragged_types
            self.raw_ranks = outer_raw_ranks
            self.ssa[op.result] = result
            result_rank = _declared_raw_rank(op)
            if result_rank is None:
                result_rank = self.raw_ranks.get(op.result)
            if result_rank is not None:
                _record_raw_rank(
                    self.raw_ranks,
                    op.result,
                    result_rank,
                    "value_if result",
                )
            return
        if op.op == "cast":
            to = _cast_target_dtype(a)
            operand = op.operands[0]
            if self.dtypes[op.result] != to:
                raise ValueError(
                    f"emit_mlir: cast result {op.result!r} inferred as "
                    f"{self.dtypes[op.result]}, expected {to}"
                )
            rr = self.fresh(op.result)
            x = self.ssa[operand]
            self.lines.append(
                f'{indent}{rr} = intent.cast {x} {{to_dtype = "{to}"}} : '
                f'({self._ttn(operand, env)}) -> '
                f'{self._ttn(op.result, env)}')
            return
        if op.op == "reshape":
            if len(op.operands) != 1:
                raise ValueError("emit_mlir: reshape requires one operand")
            raw_shape = a.get("shape")
            if not isinstance(raw_shape, list):
                raise ValueError("emit_mlir: reshape requires attrs.shape")
            shape = _fold_shape(
                [_resolve_int(item, extents) for item in raw_shape],
                fold,
            )
            rr = self.fresh(op.result)
            rendered = ", ".join(str(item) for item in shape)
            operand = op.operands[0]
            self.lines.append(
                f'{indent}{rr} = intent.reshape {self.ssa[operand]} '
                f'{{shape = [{rendered}]}} : '
                f'({self._ttn(operand, env)}) -> {self._ttn(op.result, env)}'
            )
            return
        if op.op == "record":
            raise NotImplementedError(
                "emit_mlir: record construction requires a record-aware ABI"
            )
        if op.op == "record_get":
            raise NotImplementedError(
                "emit_mlir: record_get requires a record-aware ABI"
            )
        if op.op == "sort":
            # The op ALWAYS has two results (sorted values + argsort indices); a
            # JSON op may name both (list result) or just one (values-only sort,
            # or indices-only argsort via attrs {"return":"indices"}). Emit both
            # SSA results, aliasing a throwaway name for the unnamed one.
            if isinstance(op.result, list):
                vname, iname = op.result[0], op.result[1]
            elif a.get("return") == "indices":
                vname, iname = None, op.result
            else:
                vname, iname = op.result, None
            named = iname if vname is None else vname
            shp = env[named]
            self._n += 1
            vssa = self.fresh(vname) if vname else f"%_sortv_{self._n}"
            issa = self.fresh(iname) if iname else f"%_sorti_{self._n}"
            x = self.ssa[op.operands[0]]
            desc = "true" if a.get("descending", False) else "false"
            value_dtype = self.dtypes[op.operands[0]]
            vty = _tt(shp, value_dtype)
            ity = "tensor<" + "x".join(str(s) for s in shp) + "xi32>"
            raw_rank = _raw_rank_for_value(
                op.operands[0],
                env,
                self.raw_ranks,
                context="sort operand",
            )
            axis = _fold_axis(
                a.get("axis", raw_rank - 1),
                raw_rank,
                fold,
            )
            self.lines.append(
                f'{indent}{vssa}, {issa} = intent.sort {x} '
                f'{{descending = {desc}, axis = {axis} : i64}} : '
                f'({self._ttn(op.operands[0], env)}) -> ({vty}, {ity})')
            return
        if op.op == "pool":
            # avg -> one dtype-preserving output; max may carry a 2nd i64 index
            # (max_pool2d_with_indices). op.result is a list [out, indices] when
            # indices are named, else a single str.
            x = self.ssa[op.operands[0]]
            if isinstance(op.result, list):
                oname, iname = op.result[0], op.result[1]
            else:
                oname, iname = op.result, None
            oshp = env[oname]
            ossa = self.fresh(oname)
            oty = self._ttn(oname, env)
            attrs = (f'fn = "{a["fn"]}", kH = {int(a["kH"])} : i64, '
                     f'kW = {int(a["kW"])} : i64, sH = {int(a["sH"])} : i64, '
                     f'sW = {int(a["sW"])} : i64, padH = {int(a["padH"])} : i64, '
                     f'padW = {int(a["padW"])} : i64')
            if iname is not None:
                issa = self.fresh(iname)
                ity = "tensor<" + "x".join(str(s) for s in oshp) + "xi64>"
                self.lines.append(
                    f'{indent}{ossa}, {issa} = intent.pool {x} {{{attrs}}} : '
                    f'({self._ttn(op.operands[0], env)}) -> ({oty}, {ity})')
            else:
                self.lines.append(
                    f'{indent}{ossa} = intent.pool {x} {{{attrs}}} : '
                    f'({self._ttn(op.operands[0], env)}) -> {oty}')
            return
        if op.op == "polar":
            # abs,angle -> (real, imag) real/imag pair in the common float dtype.
            absv, ang = (self.ssa[o] for o in op.operands)
            rname, iname = op.result[0], op.result[1]
            rssa, issa = self.fresh(rname), self.fresh(iname)
            ta = self._ttn(op.operands[0], env)
            tb = self._ttn(op.operands[1], env)
            self.lines.append(
                f'{indent}{rssa}, {issa} = intent.polar {absv}, {ang} : '
                f'({ta}, {tb}) -> '
                f'({self._ttn(rname, env)}, {self._ttn(iname, env)})')
            return
        r = self.fresh(op.result) if isinstance(op.result, str) else None
        if op.op == "conv":
            # input, weight, optional bias -> single output. stride/padding/
            # dilation are length-R spatial arrays; groups partitions channels.
            x = self.ssa[op.operands[0]]
            w = self.ssa[op.operands[1]]
            tx, tw = (self._ttn(name, env) for name in op.operands[:2])
            st = ", ".join(str(int(v)) for v in a["stride"])
            pd = ", ".join(str(int(v)) for v in a["padding"])
            dl = ", ".join(str(int(v)) for v in a["dilation"])
            grp = int(a.get("groups", 1))
            attrs = (f'stride = [{st}], padding = [{pd}], dilation = [{dl}], '
                     f'groups = {grp} : i64')
            if len(op.operands) > 2:
                bn = op.operands[2]
                bssa, tb = self.ssa[bn], self._ttn(bn, env)
                self.lines.append(
                    f'{indent}{r} = intent.conv {x}, {w}, {bssa} {{{attrs}}} : '
                    f'({tx}, {tw}, {tb}) -> {self._ttn(op.result, env)}')
            else:
                self.lines.append(
                    f'{indent}{r} = intent.conv {x}, {w} {{{attrs}}} : '
                    f'({tx}, {tw}) -> {self._ttn(op.result, env)}')
        elif op.op == "upsample":
            x = self.ssa[op.operands[0]]
            mode = a["mode"]
            if mode not in {"nearest", "bicubic"}:
                raise ValueError(
                    "emit_mlir: upsample mode must be 'nearest' or 'bicubic'"
                )
            raw_scale = a.get("scale")
            if (
                not isinstance(raw_scale, list)
                or len(raw_scale) != 2
                or any(type(value) is not int or value < 1 for value in raw_scale)
            ):
                raise ValueError(
                    "emit_mlir: upsample scale must be two positive integers"
                )
            scale = ", ".join(str(value) for value in raw_scale)
            self.lines.append(
                f'{indent}{r} = intent.upsample {x} '
                f'{{mode = "{mode}", scale = [{scale}]}} : '
                f'({self._ttn(op.operands[0], env)}) -> '
                f'{self._ttn(op.result, env)}')
        elif op.op == "contraction":
            x, y = (self.ssa[o] for o in op.operands)
            tx, ty = (self._ttn(name, env) for name in op.operands)
            ins, out = a["subscripts"].split("->")
            aL, bL = (s.strip() for s in ins.split(","))
            sub = f"{_strip(aL, fold)},{_strip(bL, fold)}->{_strip(out.strip(), fold)}"
            self.lines.append(
                f'{indent}{r} = intent.contraction {x}, {y} '
                f'{{subscripts = "{sub}"}} : ({tx}, {ty}) -> '
                f'{self._ttn(op.result, env)}')
        elif op.op == "reduce":
            # all/any become i1; numeric reductions preserve the operand dtype.
            # Both decisions were made before printing by _infer_op_dtypes.
            x = self.ssa[op.operands[0]]
            axes = ", ".join(
                str(
                    _axis_for_value(
                        op,
                        op.operands[0],
                        ax,
                        env,
                        fold,
                        self.raw_ranks,
                        context="reduce operand",
                    )
                )
                for ax in a["axes"]
            )
            self.lines.append(
                f'{indent}{r} = intent.reduce {x} {{fn = "{a["fn"]}", '
                f'axes = [{axes}], assoc = "{a["assoc"]}"}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "scan":
            x = self.ssa[op.operands[0]]
            axis = _axis_for_value(
                op,
                op.operands[0],
                a["axis"],
                env,
                fold,
                self.raw_ranks,
                context="scan operand",
            )
            self.lines.append(
                f'{indent}{r} = intent.scan {x} {{fn = "{a["fn"]}", '
                f'axis = {axis}}} : '
                f'({self._ttn(op.operands[0], env)}) -> '
                f'{self._ttn(op.result, env)}')
        elif op.op == "arg_reduce":
            # Index-of-extremum: an INTEGER (i32) result with `axis` collapsed.
            x = self.ssa[op.operands[0]]
            axis = _axis_for_value(
                op,
                op.operands[0],
                a["axis"],
                env,
                fold,
                self.raw_ranks,
                context="arg_reduce operand",
            )
            self.lines.append(
                f'{indent}{r} = intent.arg_reduce {x} {{fn = "{a["fn"]}", '
                f'axis = {axis} : i64}} : '
                f'({self._ttn(op.operands[0], env)}) -> '
                f'{self._ttn(op.result, env)}')
        elif op.op == "map_unary":
            fn = a.get("fn")
            if fn not in MAP_UNARY_FNS:
                raise ValueError(
                    f"emit_mlir: map_unary has no closed lowering for fn {fn!r}"
                )
            # Predicate fns (isnan/isinf/isfinite/logical_not) yield an i1; every
            # other fn is element-type-preserving.  The shared operation set is
            # also consumed by TypeCheck and the C++ scalar lowering.
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.map_unary {x} {{fn = "{fn}"}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "map_binary":
            axes = _op_broadcast_axes(op, self.broadcast_axes)
            contraction = _broadcast_mul_contraction_subscripts(op, env, axes)
            lhs_dtype = self.dtypes.get(op.operands[0])
            rhs_dtype = self.dtypes.get(op.operands[1])
            operand_dtype = _promote_dtype(lhs_dtype, rhs_dtype)
            expected_dtype = map_binary_result_dtype(
                a.get("fn"), lhs_dtype, rhs_dtype
            )
            if operand_dtype is None or expected_dtype is None:
                raise ValueError(
                    f"emit_mlir: map_binary {op.result!r} has no closed dtype "
                    "promotion"
                )
            if self.dtypes.get(op.result) != expected_dtype:
                raise ValueError(
                    f"emit_mlir: map_binary {op.result!r} result dtype "
                    f"{self.dtypes.get(op.result)!r} differs from "
                    f"the shared contract {expected_dtype!r}"
                )
            if self.dtypes[op.operands[0]] != self.dtypes[op.operands[1]]:
                # The outer-product contraction spelling has its own mixed-
                # precision accumulator rules.  A semantic map promotion is
                # instead made explicit on each operand below.
                contraction = None
            if contraction is not None:
                x, y = (self.ssa[name] for name in op.operands)
                tx, ty = (self._ttn(name, env) for name in op.operands)
                self.lines.append(
                    f'{indent}{r} = intent.contraction {x}, {y} '
                    f'{{subscripts = "{contraction}"}} : ({tx}, {ty}) -> '
                    f'{self._ttn(op.result, env)}'
                )
            else:
                x, tx = self._broadcast_operand(
                    op.operands[0],
                    env,
                    axes.get(op.operands[0], ()),
                    indent,
                    env[op.result],
                    element_dtype=operand_dtype,
                )
                y, ty = self._broadcast_operand(
                    op.operands[1],
                    env,
                    axes.get(op.operands[1], ()),
                    indent,
                    env[op.result],
                    element_dtype=operand_dtype,
                )
                self.lines.append(
                    f'{indent}{r} = intent.map_binary {x}, {y} '
                    f'{{fn = "{a["fn"]}"}} : ({tx}, {ty}) -> '
                    f'{self._ttn(op.result, env)}'
                )
        elif op.op == "select":
            axes = _op_broadcast_axes(op, self.broadcast_axes)
            c, tc = self._broadcast_operand(
                op.operands[0],
                env,
                axes.get(op.operands[0], ()),
                indent,
                env[op.result],
            )
            x, tx = self._broadcast_operand(
                op.operands[1],
                env,
                axes.get(op.operands[1], ()),
                indent,
                env[op.result],
            )
            y, ty = self._broadcast_operand(
                op.operands[2],
                env,
                axes.get(op.operands[2], ()),
                indent,
                env[op.result],
            )
            self.lines.append(
                f'{indent}{r} = intent.select {c}, {x}, {y} : '
                f'({tc}, {tx}, {ty}) -> {self._ttn(op.result, env)}')
        elif op.op == "gather":
            d, i = (self.ssa[o] for o in op.operands)
            td, ti = (self._ttn(o, env) for o in op.operands)
            axis = _axis_for_value(
                op,
                op.operands[0],
                a["axis"],
                env,
                fold,
                self.raw_ranks,
                context="gather data operand",
            )
            mode = a["mode"]
            # A bare IntentFunction has no Layer-B proof channel.  The caller
            # may pass only the exact projection derived from canonical J;
            # otherwise the target-specific authorization is never invented.
            candidate_refs = {ref for ref in (node_ref, op.node_id) if ref is not None}
            bounds_attr = (
                ", intent.index_bounds_proven = true"
                if candidate_refs & self.proven_index_bounds
                else ""
            )
            data_rank = len(env[op.operands[0]])
            index_rank = len(env[op.operands[1]])
            if axis < 0:
                axis += data_rank
            if mode == "along_axis" and index_rank == data_rank - 1:
                # JSON IntentIR permits the selected axis to be omitted from
                # index/result (one index per coordinate of the other axes).
                # Legalize to the stricter MLIR Intent op by inserting a unit
                # axis, gather at equal rank, then squeeze that unit axis.
                if index_rank == 0:
                    raise NotImplementedError(
                        "emit_mlir compressed along_axis gather does not yet "
                        "support a rank-0 index/result"
                    )
                expanded_index_shape = _insert_broadcast_axes(
                    env[op.operands[1]], (axis,)
                )
                assert expanded_index_shape is not None
                expanded_index, expanded_index_type = self._broadcast_operand(
                    op.operands[1], env, (axis,), indent
                )
                expanded_result_type = self._ttn_shape(
                    op.result, expanded_index_shape
                )
                self._n += 1
                gathered = f"%{op.result}_gather_expanded_{self._n}"
                self.lines.append(
                    f'{indent}{gathered} = intent.gather {d}, {expanded_index} '
                    f'{{axis = {axis} : i64, mode = "{mode}"{bounds_attr}}} : '
                    f'({td}, {expanded_index_type}) -> {expanded_result_type}')

                source_positions = [
                    dim for dim in range(data_rank) if dim != axis
                ]
                reassociation: list[list[int]] = []
                for index, position in enumerate(source_positions):
                    start = 0 if index == 0 else position
                    end = (
                        source_positions[index + 1]
                        if index + 1 < len(source_positions)
                        else data_rank
                    )
                    reassociation.append(list(range(start, end)))
                groups = "[" + ", ".join(
                    "[" + ", ".join(str(value) for value in group) + "]"
                    for group in reassociation
                ) + "]"
                self.lines.append(
                    f"{indent}{r} = tensor.collapse_shape {gathered} {groups} : "
                    f"{expanded_result_type} into {self._ttn(op.result, env)}"
                )
            else:
                if mode == "along_axis" and index_rank != data_rank:
                    raise NotImplementedError(
                        "emit_mlir along_axis gather requires index rank equal "
                        "to data rank or data rank minus one"
                    )
                self.lines.append(
                    f'{indent}{r} = intent.gather {d}, {i} '
                    f'{{axis = {axis} : i64, mode = "{mode}"{bounds_attr}}} : '
                    f'({td}, {ti}) -> {self._ttn(op.result, env)}')
        elif op.op in {"scatter", "atomic_scatter"}:
            axis = _axis_for_value(
                op,
                op.operands[0],
                a["axis"],
                env,
                fold,
                self.raw_ranks,
                context="scatter data operand",
            )
            atomic = a.get("atomic")
            ordering = a.get("ordering")
            memory_scope = a.get("memory_scope")
            scatter_reduce = a["reduce"] if op.op == "scatter" else "none"
            if op.op == "atomic_scatter" and atomic != "cas":
                raise ValueError(
                    "emit_mlir: atomic_scatter requires atomic='cas'"
                )
            if atomic == "cas":
                if (
                    len(op.operands) != 4
                    or not isinstance(op.result, list)
                    or len(op.result) != 2
                    or len(set(op.result)) != 2
                ):
                    raise ValueError(
                        "emit_mlir: atomic='cas' scatter requires operands "
                        "[data,index,compare,replacement] and results "
                        "[updated,observed]"
                    )
                if ordering not in ATOMIC_ORDERINGS:
                    raise ValueError(
                        "emit_mlir: atomic='cas' requires a closed ordering"
                    )
                if memory_scope not in ATOMIC_MEMORY_SCOPES:
                    raise ValueError(
                        "emit_mlir: atomic='cas' requires a closed memory_scope"
                    )
                if scatter_reduce != "none":
                    raise ValueError(
                        "emit_mlir: atomic='cas' requires reduce='none'"
                    )
                retry = a.get("cas_retry", "single_attempt")
                if retry not in ATOMIC_CAS_RETRIES:
                    raise ValueError(
                        "emit_mlir: atomic='cas' requires a closed cas_retry"
                    )
                updated, observed = op.result
                data, index, compare, replacement = op.operands
                result_ssa = self.fresh(updated)
                observed_ssa = self.fresh(observed)
                rendered_attrs = (
                    f'axis = {axis} : i64, mode = "{a["mode"]}", '
                    f'atomic = "cas", ordering = "{ordering}", '
                    f'memory_scope = "{memory_scope}", cas_retry = "{retry}"'
                )
                operand_types = ", ".join(
                    self._ttn(name, env) for name in op.operands
                )
                result_types = ", ".join(
                    self._ttn(name, env) for name in op.result
                )
                self.lines.append(
                    f'{indent}{result_ssa}, {observed_ssa} = '
                    f'intent.atomic_scatter {self.ssa[data]}, {self.ssa[index]}, '
                    f'{self.ssa[compare]}, {self.ssa[replacement]} '
                    f'{{{rendered_attrs}}} : ({operand_types}) -> '
                    f'({result_types})'
                )
                return
            if not isinstance(op.result, str):
                raise ValueError(
                    "emit_mlir: non-CAS scatter requires one string result"
                )
            d, i, s = (self.ssa[o] for o in op.operands)
            td, ti, ts = (self._ttn(o, env) for o in op.operands)
            if atomic is not None:
                if atomic not in ATOMIC_KINDS:
                    raise ValueError(
                        f"emit_mlir: scatter has unknown atomic kind {atomic!r}"
                    )
                if ordering not in ATOMIC_ORDERINGS:
                    raise ValueError(
                        "emit_mlir: atomic scatter requires a closed ordering"
                    )
                if memory_scope not in ATOMIC_MEMORY_SCOPES:
                    raise ValueError(
                        "emit_mlir: atomic scatter requires a closed memory_scope"
                    )
                if atomic == "add" and scatter_reduce != "add":
                    raise ValueError(
                        "emit_mlir: atomic='add' requires reduce='add'"
                    )
                if atomic == "cas" and scatter_reduce != "none":
                    raise ValueError(
                        "emit_mlir: atomic='cas' requires reduce='none'"
                    )
            atomic_attrs = ""
            if atomic is not None:
                atomic_attrs = (
                    f', atomic = "{atomic}", ordering = "{ordering}", '
                    f'memory_scope = "{memory_scope}"'
                )
            self.lines.append(
                f'{indent}{r} = intent.scatter {d}, {i}, {s} '
                f'{{axis = {axis} : i64, mode = "{a["mode"]}", '
                f'reduce = "{scatter_reduce}"{atomic_attrs}}} : '
                f'({td}, {ti}, {ts}) -> {self._ttn(op.result, env)}')
        elif op.op == "nonzero":
            # Dynamic-shape output (C8): [nnz, ndim] i64 index rows. The emitted
            # leading extent is the MAX upper bound (prod of cond shape).
            c = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.nonzero {c} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "masked_select":
            d, m = (self.ssa[o] for o in op.operands)
            td, tm = (self._ttn(o, env) for o in op.operands)
            self.lines.append(
                f'{indent}{r} = intent.masked_select {d}, {m} : '
                f'({td}, {tm}) -> {self._ttn(op.result, env)}')
        elif op.op == "masked_scatter":
            d, m, s = (self.ssa[o] for o in op.operands)
            td, tm, ts = (self._ttn(o, env) for o in op.operands)
            self.lines.append(
                f'{indent}{r} = intent.masked_scatter {d}, {m}, {s} : '
                f'({td}, {tm}, {ts}) -> {self._ttn(op.result, env)}')
        elif op.op == "transpose":
            operand = op.operands[0]
            x = self.ssa[operand]
            perm = _fold_permutation(a["perm"], fold, extents or {})
            rendered = ", ".join(str(value) for value in perm)
            self.lines.append(
                f'{indent}{r} = intent.transpose {x} '
                f'{{perm = [{rendered}]}} : '
                f'({self._ttn(operand, env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "concat":
            xs = ", ".join(self.ssa[o] for o in op.operands)
            txs = ", ".join(self._ttn(o, env) for o in op.operands)
            axis = _axis_for_value(
                op,
                op.operands[0],
                a["axis"],
                env,
                fold,
                self.raw_ranks,
                context="concat operand",
            )
            self.lines.append(
                f'{indent}{r} = intent.concat {xs} {{axis = {axis} : i64}} : '
                f'({txs}) -> {self._ttn(op.result, env)}')
        elif op.op == "pad":
            x = self.ssa[op.operands[0]]
            raw_lo = [_resolve_int(v, extents) for v in a["lo"]]
            raw_hi = [_resolve_int(v, extents) for v in a["hi"]]
            prefix = _fold_prefix_len(raw_lo, fold)
            lo = ", ".join(
                str(v) for v in raw_lo[prefix:]
            )
            hi = ", ".join(
                str(v) for v in raw_hi[prefix:]
            )
            self.lines.append(
                f'{indent}{r} = intent.pad {x} {{lo = [{lo}], hi = [{hi}], '
                f'value = {_mlir_f(a.get("value", 0.0))} : f64}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "tile":
            x = self.ssa[op.operands[0]]
            raw_reps = [int(v) for v in a["reps"]]
            prefix = _fold_prefix_len(raw_reps, fold)
            reps = ", ".join(str(v) for v in raw_reps[prefix:])
            self.lines.append(
                f'{indent}{r} = intent.tile {x} {{reps = [{reps}]}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "flip":
            x = self.ssa[op.operands[0]]
            axes = ", ".join(
                str(
                    _axis_for_value(
                        op,
                        op.operands[0],
                        value,
                        env,
                        fold,
                        self.raw_ranks,
                        context="flip operand",
                    )
                )
                for value in a["axes"]
            )
            self.lines.append(
                f'{indent}{r} = intent.flip {x} {{axes = [{axes}]}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "triu":
            x = self.ssa[op.operands[0]]
            self.lines.append(
                f'{indent}{r} = intent.triu {x} {{k = {int(a.get("k", 0))} : i64}} : '
                f'({self._ttn(op.operands[0], env)}) -> {self._ttn(op.result, env)}')
        elif op.op == "iota":
            # Integer column-index vector; print the explicit dtype, or the
            # declared result dtype when the optional attribute is omitted.
            # This is the same defaulting rule used by dtype inference and the
            # structured reference evaluator.
            dt = a.get("dtype")
            if dt is None:
                dt = self.dtypes.get(op.result) or _declared_op_dtype(op) or "i32"
            shape = env[op.result]
            if len(shape) != 1:
                raise ValueError("iota result must resolve to rank 1")
            n = int(shape[0])
            declared_n = _resolve_int(a["n"], extents)
            if declared_n != n:
                raise ValueError(
                    f"iota attrs.n={declared_n} differs from result extent {n}"
                )
            if self.dtypes[op.result] != _normalize_dtype(dt, "iota dtype"):
                raise ValueError(
                    f"iota result dtype {self.dtypes[op.result]} differs from {dt}"
                )
            self.lines.append(
                f'{indent}{r} = intent.iota : {_tt([n], _normalize_dtype(dt, "iota dtype"))}')
        elif op.op == "rand":
            # The dialect's native RNG is f32.  A J loop may require a lower or
            # wider carry dtype, so keep the f32 sample semantics and make the
            # required conversion explicit in Layer-A MLIR.
            result_type = self._ttn(op.result, env)
            result_dtype = self.dtypes[op.result]
            if result_dtype == "f32":
                self.lines.append(
                    f'{indent}{r} = intent.rand {{seed = {int(a["seed"])} : i64}} : '
                    f'{result_type}')
            else:
                self._n += 1
                native = f"%{op.result}_rand_f32_{self._n}"
                native_type = _tt(env[op.result], "f32")
                self.lines.append(
                    f'{indent}{native} = intent.rand '
                    f'{{seed = {int(a["seed"])} : i64}} : {native_type}'
                )
                self.lines.append(
                    f'{indent}{r} = intent.cast {native} '
                    f'{{to_dtype = "{result_dtype}"}} : '
                    f'({native_type}) -> {result_type}'
                )
        elif op.op == "const":
            # A literal value -> standard arith.constant (no intent semantics to
            # lower). A scalar splats dense over the (possibly rank-0) result
            # tensor; a list value (iota/arange materialized as an explicit
            # vector) prints as a dense elements array. An i1/int-dtyped const
            # (a boolean sentinel / mask threshold) carries its own element type.
            dt = self.dtypes.get(op.result)
            if dt is None:
                dt = _declared_op_dtype(op) or a.get("dtype", "f32")
            dt = _normalize_dtype(dt, "const dtype")
            authored_type = (
                op.type
                if isinstance(op.type, Mapping)
                else {"dtype": dt, "shape": list(env[op.result])}
            )
            try:
                literal = validate_const_literal(
                    a,
                    result_type=authored_type,
                    context="emit_mlir const",
                )
            except IntentOpContractError as exc:
                raise ValueError(str(exc)) from exc

            def format_i1(value: object) -> str:
                return "true" if bool(value) else "false"

            if dt == "i1":
                if literal.splat:
                    dense = f"dense<{format_i1(literal.values[0])}>"
                else:
                    nested = _mlir_dense_literal(
                        literal.values,
                        literal.shape,
                        format_i1,
                    )
                    dense = f"dense<{nested}>"
            elif dt in _INTEGER_DTYPES - {"i1"}:
                if literal.splat:
                    dense = f"dense<{literal.values[0]}>"
                else:
                    nested = _mlir_dense_literal(
                        literal.values,
                        literal.shape,
                        str,
                    )
                    dense = f"dense<{nested}>"
            elif literal.splat:
                dense = f"dense<{_mlir_f(literal.values[0], dt)}>"
            else:
                nested = _mlir_dense_literal(
                    literal.values,
                    literal.shape,
                    lambda value: _mlir_f(value, dt),
                )
                dense = f"dense<{nested}>"
            self.lines.append(
                f'{indent}{r} = arith.constant {dense} : {self._ttn(op.result, env)}')
        else:
            raise NotImplementedError(f"emit_mlir: unhandled op {op.op!r}")


def _tile(
    loop: IntentOp, extents: dict[str, int], schedule_tile: int | None
) -> int:
    if loop.trip is not None and loop.trip.is_dynamic:
        # Dynamic value loops have no semantic scan tile.  The MLIR ABI carries
        # the runtime upper bound and the induction value explicitly; returning
        # one keeps the shared shape helper total without inventing a C tile.
        if schedule_tile is not None:
            raise ValueError(
                "emit_mlir: dynamic value loops do not accept a scan schedule tile"
            )
        return 1
    if schedule_tile is not None:
        if isinstance(schedule_tile, bool) or not isinstance(schedule_tile, int):
            raise TypeError("loop schedule tile must be an integer")
        if schedule_tile <= 0:
            raise ValueError("loop schedule tile must be positive")
        return schedule_tile
    # Without C, use one full semantic-axis iteration.  This is a deterministic
    # semantics-preserving fallback, not a reconstruction of source tiling.
    if loop.trip is not None and loop.trip.axis in extents:
        return int(extents[loop.trip.axis])
    return 1


def _validate_mlir_representability(fn: IntentFunction) -> None:
    """Reject J values that have no representation in the Intent MLIR ABI.

    TypeCheck and the CPU reference intentionally understand named records,
    but the current Intent dialect has no record type and its structured loop
    verifier forbids tuple values crossing a region boundary.  Perform this
    capability check before shape/dtype inference so an accepted J artifact
    cannot run through several emitter phases and fail later at an unrelated
    ``NotImplementedError``.
    """

    for operation in walk_intent_ops(fn.body):
        if operation.op in {"record", "record_get"}:
            raise ValueError(
                "emit_mlir: named record operations are not representable in "
                "the current flat tensor/tuple-free Intent MLIR ABI; use a "
                "record-capable backend or lower the record at the J boundary"
            )
        if (
            isinstance(operation.type, Mapping)
            and operation.type.get("kind") == "record"
        ):
            raise ValueError(
                "emit_mlir: record-typed values are not representable in the "
                "current Intent MLIR ABI"
            )
        if operation.op == "loop" and any(
            state.type is not None and state.type.kind == "record"
            for state in operation.carry
        ):
            raise ValueError(
                "emit_mlir: record loop carries are not representable in the "
                "current Intent MLIR ABI"
            )


def _setup(
    fn: IntentFunction,
    dims: dict[str, int],
    fold: int,
    schedule_tile: int | None = None,
):
    """Shape inference shared by both emitters.

    Returns ``(loop, env, carries, yields, scan_specs, tile, ragged_types,
    raw_ranks)`` where scan_specs is a list of
    ``(slice_name, source_input_name, folded_axis)``.  ``raw_ranks`` is kept
    beside the folded shape environment so every authored axis is projected
    from the same pre-fold rank.
    """
    _validate_mlir_representability(fn)
    loop_position = next(
        index for index, operation in enumerate(fn.body) if operation.op == "loop"
    )
    loop = fn.body[loop_position]
    try:
        validate_ragged_region_boundaries(fn.body)
    except ValueError as exc:
        raise NotImplementedError(f"emit_mlir: {exc}") from exc
    if loop.trip is None:
        raise ValueError("emit_mlir: loop requires a closed trip binding")
    if loop.trip.is_dynamic and loop.scan:
        raise ValueError("emit_mlir: dynamic value loops cannot have scan inputs")
    extents = dict(dims)
    env: dict[str, list[object]] = {}
    ragged_types: dict[str, _RaggedValueType] = {}
    raw_ranks: dict[str, int] = {}
    for t in fn.inputs:
        full = _resolve(t.shape, extents)
        prefix = _fold_prefix_len(full, fold)
        # Folding leading batch dims is only valid when they are size 1 (a single
        # attention head). Refuse to silently miscompile a B>1 / H>1 instance.
        if prefix and any(d != 1 for d in full[:prefix]):
            raise ValueError(
                f"emit_mlir: cannot fold {fold} leading dims of {t.name} "
                f"(shape {full}); folded dims must be size 1")
        env[t.name] = _fold_shape(full, fold)
        _record_raw_rank(raw_ranks, t.name, len(full), "function input")
    # Materialize the dominating prefix in the shared inference environment.
    # This is required for dynamic loops whose upper_bound is produced by a
    # preceding Ragged query rather than passed as a direct function input.
    prefix = fn.body[:loop_position]
    if prefix:
        _infer_body(
            prefix,
            env,
            fold,
            extents,
            ragged_types=ragged_types,
            raw_ranks=raw_ranks,
        )
    tile = _tile(loop, extents, schedule_tile)
    scan_specs = []
    for s in loop.scan:
        source = s.source
        source_rank = _raw_rank_for_value(
            source,
            env,
            raw_ranks,
            context="loop scan source",
        )
        ax = _fold_axis(s.axis, source_rank, fold)
        slshape = list(env[source])
        slshape[ax] = tile
        env[s.name] = slshape
        _record_raw_rank(raw_ranks, s.name, source_rank, "loop scan binding")
        scan_specs.append((s.name, source, ax))
    if loop.trip.is_dynamic:
        upper_bound = loop.trip.upper_bound
        induction = loop.trip.index
        if not isinstance(upper_bound, str) or not isinstance(induction, str):
            raise ValueError(
                "emit_mlir: dynamic loop requires upper_bound and index bindings"
            )
        if upper_bound in env and env[upper_bound] != []:
            raise ValueError(
                "emit_mlir: dynamic loop upper_bound must be a rank-0 tensor"
            )
        env[induction] = []
        _record_raw_rank(raw_ranks, induction, 0, "dynamic loop induction")
    carries = [c.body_argument for c in loop.carry]
    yields = [c.next_value for c in loop.carry]
    for state in loop.carry:
        if state.type is not None:
            if state.type.kind != "tensor" or state.type.dtype is None:
                raise NotImplementedError(
                    "emit_mlir semantic-loop prototype requires a tensor loop "
                    "carry; named record carries need a record-aware ABI"
                )
            # Carry element types are part of the Layer-A value contract.  Keep
            # the same closed dtype validation used by the flat emitter, but do
            # not impose an execution-precision policy at this shape-inference
            # boundary.
            _normalize_dtype(state.type.dtype, "loop carry type")
            full = _resolve(state.type.shape, extents)
            env[state.body_argument] = _fold_shape(full, fold)
            _record_raw_rank(
                raw_ranks,
                state.body_argument,
                len(full),
                "loop carry body_argument",
            )
            _record_raw_rank(
                raw_ranks,
                state.result_value,
                len(full),
                "loop carry result_value",
            )
        elif isinstance(state.init_value, str) and state.init_value in raw_ranks:
            _record_raw_rank(
                raw_ranks,
                state.body_argument,
                raw_ranks[state.init_value],
                "loop carry initializer",
            )
    for _ in range(len(loop.body) + 3):
        _infer_body(
            loop.body,
            env,
            fold,
            extents,
            ragged_types=ragged_types,
            raw_ranks=raw_ranks,
        )
        moved = False
        for cname, yname in zip(carries, yields):
            if yname in env and env.get(cname) != env[yname]:
                env[cname] = list(env[yname])
                moved = True
        if not moved:
            break
    for state in loop.carry:
        if state.body_argument in env:
            env[state.result_value] = list(env[state.body_argument])
        if state.body_argument in raw_ranks:
            _record_raw_rank(
                raw_ranks,
                state.result_value,
                raw_ranks[state.body_argument],
                "loop result",
            )
    return loop, env, carries, yields, scan_specs, tile, ragged_types, raw_ranks


def emit_body_function(
    fn: IntentFunction,
    dims: dict[str, int],
    fold: int = 2,
    func_name: str = "flash_body_fwd",
    *,
    schedule_tile: int | None = None,
) -> str:
    """The loop body as an executable ``func.func`` (scan tiles + carries in,
    next carries out). This is what the host-driven trip loop launches."""
    loop, env, carries, yields, scan_specs, _, ragged_types, raw_ranks = _setup(
        fn, dims, fold, schedule_tile
    )
    dtypes = _loop_dtypes(fn, loop, ragged_types)
    p = _Printer(dtypes, ragged_types=ragged_types, raw_ranks=raw_ranks)

    referenced: set[str] = set()
    loop_position = next(
        index for index, operation in enumerate(fn.body) if operation is loop
    )
    for op in [*fn.body[:loop_position], *loop.body]:
        referenced.update(op.operands)
    # Outer inputs the body reads (Q, scale, ...): function inputs that appear in
    # the body directly (the scanned tensors enter via their slices, below).
    outer = [t.name for t in fn.inputs if t.name in referenced]

    args: list[str] = []
    for nm in outer:
        args.append(f"{p.arg(nm)}: {p._ttn(nm, env)}")
    if loop.trip is not None and loop.trip.is_dynamic:
        assert loop.trip.index is not None
        args.append(
            f"{p.arg(loop.trip.index)}: {p._ttn(loop.trip.index, env)}"
        )
    for sname, _src, _ax in scan_specs:
        args.append(f"{p.arg(sname)}: {p._ttn(sname, env)}")
    for c in carries:
        args.append(f"{p.arg(c)}: {p._ttn(c, env)}")
    rty = ", ".join(p._ttn(c, env) for c in carries)

    for op in fn.body[:loop_position]:
        p.emit_op(op, env, fold, extents=dict(dims))
    for op in loop.body:
        p.emit_op(op, env, fold, extents=dict(dims))
    yvals = ", ".join(p.ssa[y] for y in yields)
    ytys = ", ".join(p._ttn(y, env) for y in yields)
    symbol = _sanitize_mlir_symbol(func_name)

    head = (f"// Generated by intentir.frontend.emit_mlir from "
            f"{fn.name} (single-head, fold={fold}).\n"
            f"// One K-tile of the FlashAttention online-softmax recurrence.\n"
            f"module {{\n"
            f"  func.func @{symbol}({', '.join(args)}) -> ({rty}) {{")
    tail = (f"    return {yvals} : {ytys}\n  }}\n}}\n")
    return head + "\n" + "\n".join(p.lines) + "\n" + tail


def emit_loop_module(
    fn: IntentFunction,
    dims: dict[str, int],
    fold: int = 2,
    func_name: str = "flash_fwd",
    *,
    schedule_tile: int | None = None,
) -> str:
    """The whole function with the structured ``intent.loop``. Representability +
    lowers to ``scf.for`` via ``-convert-intent-to-linalg``."""
    loop, env, carries, yields, scan_specs, tile, ragged_types, raw_ranks = _setup(
        fn, dims, fold, schedule_tile
    )
    dtypes = _loop_dtypes(fn, loop, ragged_types)
    p = _Printer(dtypes, ragged_types=ragged_types, raw_ranks=raw_ranks)

    # Function signature: the IntentIR function inputs (folded, scalars rank-0).
    args = [f"{p.arg(t.name)}: {p._ttn(t.name, env)}" for t in fn.inputs]
    # Carry inits as function args too (so the module is self-contained and the
    # init *values* — NEG_INF/0/zeros — stay the caller's responsibility).
    init_names = [f"{c}_init" for c in carries]
    for c, init in zip(carries, init_names):
        env[init] = env[c]
        dtypes[init] = dtypes[c]
        args.append(f"{p.arg(init)}: {p._ttn(init, env)}")

    loop_position = next(
        index for index, operation in enumerate(fn.body) if operation is loop
    )
    # Emit dominating producers before the structured loop.  In particular,
    # `ragged_outer` may supply a dynamic upper bound while its descriptor is
    # produced outside the loop; dropping this prefix would make the printed
    # MLIR reference an SSA value that was never defined.
    for operation in fn.body[:loop_position]:
        p.emit_op(operation, env, fold, extents=dict(dims))
    prefix_lines = p.lines

    scan_ssa = ", ".join(p.ssa[s["from"]] for s in loop.scan)
    scan_tys = ", ".join(p._ttn(s["from"], env) for s in loop.scan)
    init_ssa = ", ".join(p.ssa[i] for i in init_names)
    init_tys = ", ".join(p._ttn(c, env) for c in carries)
    scan_axes = ", ".join(f"{axis} : i64" for _name, _source, axis in scan_specs)
    res_tys = ", ".join(p._ttn(c, env) for c in carries)
    scan_clause = (
        f"scan({scan_ssa} : {scan_tys})" if loop.scan else "scan()"
    )
    upper_bound_clause = ""
    if loop.trip.is_dynamic:
        assert loop.trip.upper_bound is not None
        upper_bound_clause = (
            f" upper_bound({p.ssa[loop.trip.upper_bound]} : "
            f"{p._ttn(loop.trip.upper_bound, env)})"
        )
    tile_clause = f", tile_hint = {tile} : i64" if not loop.trip.is_dynamic else ""

    # Body region block args: scan slices then carries.
    bargs = []
    for sname, _src, _ax in scan_specs:
        bargs.append(f"{p.arg(sname)}: {p._ttn(sname, env)}")
    if loop.trip.is_dynamic:
        assert loop.trip.index is not None
        bargs.append(
            f"{p.arg(loop.trip.index)}: {p._ttn(loop.trip.index, env)}"
        )
    for c in carries:
        # carry block arg shadows the init binding for the body
        bargs.append(f"{p.arg(c)}: {p._ttn(c, env)}")
    body_lines: list[str] = []
    saved, p.lines = p.lines, body_lines
    for op in loop.body:
        p.emit_op(
            op, env, fold, indent="      ", extents=dict(dims)
        )
    p.lines = saved
    yvals = ", ".join(p.ssa[y] for y in yields)
    ytys = ", ".join(p._ttn(y, env) for y in yields)

    out_res = next((o for o in fn.body if o.op != "loop"), None)
    loop_res = p.fresh("loop")
    symbol = _sanitize_mlir_symbol(func_name)

    lines = [
        f"// Generated by intentir.frontend.emit_mlir from {fn.name} "
        f"(single-head, fold={fold}).",
        "module {",
        f"  func.func @{symbol}({', '.join(args)}) -> ({res_tys}) {{",
        *prefix_lines,
        f"    {loop_res}:{len(carries)} = intent.loop "
        f"{scan_clause} init({init_ssa} : {init_tys}){upper_bound_clause} "
        f"{{scan_axes = [{scan_axes}]{tile_clause}}} "
        f"-> {res_tys} do {{",
        f"    ^bb0({', '.join(bargs)}):",
        *body_lines,
        f"      intent.yield {yvals} : {ytys}",
        "    }",
    ]
    rets = [f"{loop_res}#{i}" for i in range(len(carries))]
    lines.append(f"    return {', '.join(rets)} : {res_tys}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def emit_module(
    fn: IntentFunction,
    dims: dict[str, int],
    fold: int = 0,
    func_name: str | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
    value_dimension_equalities: Sequence[tuple[int, int]] = (),
    output_values: Mapping[str, str] | None = None,
    canonical_output_bindings: Sequence[Any] | None = None,
    inplace_outputs: Sequence[int] = (),
    paired_rotation: tuple[str, tuple[int, int, int, int], tuple[int, ...]]
    | None = None,
    loop_schedule_tile: int | None = None,
) -> str:
    """A flat (loop-free) IntentFunction as an executable ``func.func``.

    Inputs become function args (clean SSA names so the orchestrator keys them by
    JSON name), the atomic op-graph is emitted in order, and the declared outputs
    are returned. Covers softmax / layernorm / rmsnorm / silu / gelu / matmul /
    elementwise — every kernel whose body is a straight-line tensor graph.
    """
    # Keep the flat-graph entry point on the same capability boundary as the
    # structured loop emitters.  Without this call, record operations could
    # pass through setup-independent shape/dtype preparation and fail only
    # after partial MLIR emission with a less actionable error.
    _validate_mlir_representability(fn)
    extents = dict(dims)
    env: dict[str, list[object]] = {}
    ragged_types: dict[str, _RaggedValueType] = {}
    raw_ranks: dict[str, int] = {}
    for t in fn.inputs:
        full = _resolve(t.shape, extents)
        prefix = _fold_prefix_len(full, fold)
        if prefix and any(d != 1 for d in full[:prefix]):
            raise ValueError(
                f"emit_mlir: cannot fold {fold} leading dims of {t.name} "
                f"(shape {full}); folded dims must be size 1"
            )
        env[t.name] = _fold_shape(full, fold)
        _record_raw_rank(raw_ranks, t.name, len(full), "function input")
    # The accepted construction record owns output order, logical type and
    # return value.  ``IntentFunction.outputs`` remains in this low-level
    # printer because it is the materialized function ABI, but for canonical
    # callers it is only a fail-closed mirror of that shared record.
    if canonical_output_bindings is None:
        output_contracts = tuple(
            (tensor.name, tensor.dtype, tuple(tensor.shape), None)
            for tensor in fn.outputs
        )
    else:
        bindings = tuple(canonical_output_bindings)
        if len(bindings) != len(fn.outputs):
            raise ValueError(
                "emit_module: canonical output bindings must cover every "
                "materialized function output"
            )
        output_contracts: tuple[tuple[str, str, tuple[object, ...], str], ...] = ()
        checked_contracts: list[tuple[str, str, tuple[object, ...], str]] = []
        for index, (tensor, binding) in enumerate(zip(fn.outputs, bindings, strict=True)):
            output = getattr(binding, "output", None)
            output_type = getattr(binding, "output_type", None)
            value = getattr(binding, "value", None)
            abi_index = getattr(binding, "abi_index", None)
            dtype = getattr(output_type, "dtype", None)
            shape = getattr(output_type, "shape", None)
            if (
                not isinstance(output, str)
                or not isinstance(value, str)
                or isinstance(abi_index, bool)
                or abi_index != index
                or not isinstance(dtype, str)
                or not isinstance(shape, tuple)
                or tensor.name != output
                or tensor.dtype != dtype
                or tuple(tensor.shape) != shape
            ):
                raise ValueError(
                    "emit_module: materialized output ABI disagrees with its "
                    "canonical output binding"
                )
            checked_contracts.append((output, dtype, shape, value))
        output_contracts = tuple(checked_contracts)
    output_names = tuple(output for output, _dtype, _shape, _value in output_contracts)
    return_outputs = tuple(item.output for item in fn.returns)
    if return_outputs != output_names:
        raise ValueError(
            "emit_module: IntentFunction.returns must cover every declared "
            "output exactly once and in declaration order"
        )
    authored_return_values = {item.output: item.value for item in fn.returns}
    if len(authored_return_values) != len(fn.returns):
        raise ValueError("emit_module: duplicate logical return output")
    canonical_return_values = {
        output: value
        for output, _dtype, _shape, value in output_contracts
        if value is not None
    }
    if canonical_output_bindings is not None:
        if authored_return_values != canonical_return_values:
            raise ValueError(
                "emit_module: materialized returns disagree with canonical "
                "output bindings"
            )
        if output_values is None:
            return_values = canonical_return_values
        else:
            provided_return_values = dict(output_values)
            if provided_return_values != canonical_return_values:
                raise ValueError(
                    "emit_module: output_values must exactly match canonical "
                    "output bindings"
                )
            return_values = canonical_return_values
    elif output_values is None:
        return_values = authored_return_values
    else:
        provided_return_values = dict(output_values)
        if provided_return_values != authored_return_values:
            raise ValueError(
                "emit_module: output_values must exactly match the complete "
                "IntentFunction.returns binding"
            )
        # The canonical construction output record is the authority for the
        # emitted return mapping.  IntentFunction.returns is checked above so
        # a stale materialized function cannot silently diverge, but the
        # consumer must use the shared record after that proof rather than
        # rebuilding the mapping from the function wrapper.
        return_values = provided_return_values
    unknown_return_outputs = sorted(set(return_values) - set(output_names))
    if unknown_return_outputs:
        raise ValueError(
            "emit_module: logical returns reference undeclared outputs "
            f"{unknown_return_outputs}"
        )
    # A reduce whose result is a declared external output may intentionally
    # collapse its reduced axes (e.g. row_sum [M,N] -> [M]).  Seed only output
    # contracts; intermediate reductions retain the existing keepdims default.
    for output, _dtype, declared_shape, _value in output_contracts:
        full = _resolve(declared_shape, extents)
        prefix = _fold_prefix_len(full, fold)
        if prefix and any(d != 1 for d in full[:prefix]):
            raise ValueError(
                f"emit_mlir: cannot fold {fold} leading dims of output {output} "
                f"(shape {full}); folded dims must be size 1"
            )
        output_shape = _fold_shape(full, fold)
        env[output] = output_shape
        _record_raw_rank(raw_ranks, output, len(full), "function output")
        returned_value = return_values[output]
        previous_shape = env.get(returned_value)
        if previous_shape is not None and previous_shape != output_shape:
            raise ValueError(
                f"emit_module: return value {returned_value!r} has conflicting "
                "logical output shapes"
            )
        env[returned_value] = output_shape
        _record_raw_rank(raw_ranks, returned_value, len(full), "return value")
    _infer_body(
        fn.body,
        env,
        fold,
        extents,
        broadcast_axes=broadcast_axes,
        loop_schedule_tile=loop_schedule_tile,
        ragged_types=ragged_types,
        raw_ranks=raw_ranks,
    )

    dtypes = _flat_dtypes(fn, ragged_types)
    # Dtype inference can complete a descriptor that was authored without
    # redundant field dtypes.  Re-run shape inference with the completed map so
    # every consumer and the printer observe the same descriptor type.
    _infer_body(
        fn.body,
        env,
        fold,
        extents,
        broadcast_axes,
        loop_schedule_tile,
        ragged_types,
        dtypes,
        raw_ranks,
    )
    p = _Printer(
        dtypes,
        broadcast_axes,
        proven_index_bounds,
        loop_schedule_tile,
        ragged_types,
        raw_ranks,
    )
    args = [f"{p.arg(t.name)}: {p._ttn(t.name, env)}" for t in fn.inputs]
    for index, op in enumerate(fn.body):
        p.emit_op(
            op,
            env,
            fold,
            extents=extents,
            node_ref=f"body[{index}]",
        )

    outs: list[tuple[str, str]] = []
    for output, dtype, _declared_shape, _value in output_contracts:
        value = return_values[output]
        if value not in p.ssa:
            raise ValueError(
                f"emit_module: output {output!r} return value {value!r} was never produced"
            )
        if value not in env or env[value] != env[output]:
            raise ValueError(
                f"emit_module: output {output!r} shape {env[output]} does not "
                f"match return value {value!r} shape {env.get(value)}"
            )
        if value not in dtypes or dtypes[value] != dtype:
            raise ValueError(
                f"emit_module: output {output!r} dtype {dtype} does not "
                f"match return value {value!r} dtype {dtypes.get(value)}"
            )
        outs.append((output, value))
    ovals = ", ".join(p.ssa[value] for _output, value in outs)
    # Respect i1 (comparison) / integer (iota) result element types so the
    # function signature matches the value actually produced.
    otys = ", ".join(p._ttn(output, env) for output, _value in outs)
    name = _sanitize_mlir_symbol(fn.name if func_name is None else func_name)
    function_attrs = _function_attributes(
        fn,
        value_dimension_equalities=value_dimension_equalities,
        inplace_outputs=inplace_outputs,
        paired_rotation=paired_rotation,
    )

    head = (f"// Generated by intentir.frontend.emit_mlir from {fn.name} "
            f"(flat graph, fold={fold}).\n"
            f"module {{\n"
            f"  func.func @{name}({', '.join(args)}) -> ({otys})"
            f"{function_attrs} {{")
    tail = f"    return {ovals} : {otys}\n  }}\n}}\n"
    return head + "\n" + "\n".join(p.lines) + "\n" + tail


__all__ = ["emit_body_function", "emit_loop_module", "emit_module"]
