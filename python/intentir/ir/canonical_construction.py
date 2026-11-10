"""Shared Source/Harness construction authority.

This module contains only transport-neutral facts.  It deliberately does not
import a frontend, backend, Gate runner, or provider.  Source adapters and the
Lift ingress can therefore share the same facts, transfer occurrences, and
handle allocator without sharing their execution implementation.

The objects here are host-side authority records.  They are not additional J,
C, or pi fields and are never inserted into the proof artifact.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field, replace
import hashlib
from itertools import combinations
import json
import math
import operator as scalar_operator
import re
import struct
from typing import Any, Callable, Mapping, Sequence

from intentir.ir.intent_op_contract import _broadcast_projection, validate_intent_op_structure
from intentir.ir.source_projection_contracts import is_canonical_ordinal


_POINTER_RE = re.compile(r"^arg\[(\d+)\]:")
_HANDLE_PREFIXES = (
    "t", "v", "x", "d", "n", "a", "e", "q", "b", "s", "p", "c", "u",
)
# One stable declaration-table order is used by Source-owned construction and
# candidate-local extension.  Empty rows are intentional: an empty namespace
# is still owned by the same allocator and must not be silently re-created by
# a downstream repair helper.
_CANONICAL_NAMESPACE_ORDER = (
    "p", "c", "t", "v", "x", "d", "n", "a", "e", "q", "b", "s", "u",
)
# Public projections must import this order instead of spelling a namespace
# subset locally.  In particular, an empty candidate namespace is still owned
# by the same allocator and must not be re-created by an MCP or provider path.
CANONICAL_HANDLE_NAMESPACE_ORDER = _CANONICAL_NAMESPACE_ORDER
CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES = ("p", "c", "t", "x")
CANONICAL_CANDIDATE_OWNED_HANDLE_PREFIXES = tuple(
    prefix
    for prefix in _CANONICAL_NAMESPACE_ORDER
    if prefix not in CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES
)
# A computed i1 access guard has a closed canonical representation only for
# the access kinds whose downstream FunctionalBoundary consumer is implemented.
# Keep this capability in the transport-neutral construction authority so MCP
# preflight, canonical materialization, and FunctionalBoundary cannot drift
# into accepting different masked-access kinds.
COMPUTED_GUARD_SUPPORTED_ACCESS_KINDS = frozenset({"load", "store"})


def unique_exact_rank_insertion_projection(
    source_shape: Sequence[str | int],
    target_shape: Sequence[str | int],
) -> tuple[int, ...] | None:
    """Return the unique exact rank-insertion map for a typed guard.

    This is deliberately narrower than the general J broadcast contract:
    computed access guards may only gain missing target axes. A singleton
    source extent is not silently treated as a broadcast proof, and no
    runtime-valued extent is consulted. MCP preflight, canonical guard
    materialization, FunctionalBoundary, and Source projection all consume
    this one relation.
    """

    if len(source_shape) > len(target_shape):
        return None
    matches = [
        inserted_axes
        for inserted_axes in combinations(
            range(len(target_shape)),
            len(target_shape) - len(source_shape),
        )
        if all(
            source_extent == target_shape[target_axis]
            for source_extent, target_axis in zip(
                source_shape,
                (
                    axis
                    for axis in range(len(target_shape))
                    if axis not in inserted_axes
                ),
                strict=True,
            )
        )
    ]
    return matches[0] if len(matches) == 1 else None


def computed_guard_projection_supported(kind: Any) -> bool:
    """Return whether one access kind has a closed computed-guard projection."""

    return isinstance(kind, str) and kind in COMPUTED_GUARD_SUPPORTED_ACCESS_KINDS


CANONICAL_HANDLE_NAMESPACE_PATTERNS = {
    prefix: rf"^{prefix}[0-9]+$"
    for prefix in _CANONICAL_NAMESPACE_ORDER
}

# Proof-artifact wire ownership is declared here, alongside the construction
# records and boundary table.  Gate1/model and the provider schema import these
# declarations; neither consumer keeps a second root/J/access field list.
CANONICAL_ARTIFACT_ROOT_FIELDS = ("schema", "j", "c", "pi")
CANONICAL_J_WIRE_FIELDS = (
    "kernel_id",
    "intent",
    "node_ids",
    "boundary",
    "signature",
    "scopes",
    "storage",
    "dimension_constraints",
    "value_dimension_relations",
    "dimension_relations",
)
CANONICAL_C_WIRE_FIELDS = ("slots", "values")
CANONICAL_PI_WIRE_FIELDS = (
    "schema",
    "normalization_classes",
    "program_axes",
    "program_bindings",
    "iteration_bindings",
    "control_bindings",
    "materialization_bindings",
    "access_representation_bindings",
)
CANONICAL_PI_REQUIRED_FIELDS = ("schema", "program_axes", "program_bindings")
CANONICAL_BOUNDARY_ACCESS_WIRE_FIELDS = (
    "access_id",
    "kind",
    "tensor",
    "predicate",
    "fill_value",
    "node_ref",
    "index_expression",
    "semantic_access_authorization",
    "semantic_owner_authorization",
    "semantic_guard_authorization",
)
CANONICAL_BOUNDARY_ACCESS_REQUIRED_FIELDS = (
    "access_id",
    "kind",
    "tensor",
    "index_expression",
)
# These are the namespaces authored by a semantic J candidate.  ``p*/c*/t*/x*``
# are SourceFacts-owned declarations: a candidate may reference them but may
# neither allocate nor renumber them.  In particular, x* dimensions are never
# a candidate-local first-appearance namespace.
CANONICAL_AUTHORED_HANDLE_PREFIXES = (
    "v", "d", "n", "a", "e", "q", "b", "s", "u",
)
_ABI_HANDLE_RE = re.compile(r"(?:p|c)[0-9]+$")
_SOURCE_MEMORY_CALLS = frozenset(
    {
        "load",
        "store",
        "atomic_add",
        "atomic_and",
        "atomic_cas",
        "atomic_max",
        "atomic_min",
        "atomic_or",
        "atomic_xchg",
        "atomic_xor",
    }
)

# An owner receipt commits to an exact Source access and a typed scalar DAG.
# The hash is not an authorization by itself: Lift/Gate1 must also compare it
# with the independently issued Source ledger. No Source expression is public.
SOURCE_OWNER_TOKEN_PATTERN = r"so_[0-9a-f]{32}_[0-9a-f]{64}"
_OWNER_ARITY = {"input": 0, "const": 0, "neg": 1, "exp": 1, "add": 2, "sub": 2, "mul": 2, "div": 2}


def _source_owner_static_shape(argument: Mapping[str, Any]) -> list[int]:
    """Consume the declared shape, never specialize a symbolic contract.

    The current owner proof supports static rank zero/one only. A concrete
    Harness shape is a valuation, not authority to replace shape_contract.
    """
    shape = argument.get("shape_contract")
    if shape is None:
        # SourceFacts uses null for a statically shaped ABI carrier too.
        shape = argument.get("shape")
    if not isinstance(shape, list):
        raise ValueError("source_owner_shape_contract_unavailable")
    if any(isinstance(extent, str) for extent in shape):
        raise ValueError("source_owner_symbolic_domain_unproved")
    if any(type(extent) is not int or extent <= 0 for extent in shape):
        raise ValueError("source_owner_shape_contract_unavailable")
    if len(shape) > 1:
        raise ValueError("source_owner_multidimensional_domain_unproved")
    if argument.get("shape") != shape:
        raise ValueError("source_owner_shape_contract_mismatch")
    return list(shape)


def owner_float_bits(value: Any, dtype: str) -> str:
    if dtype not in {"f32", "f64"} or type(value) not in {int, float}:
        raise ValueError("source_owner_literal_unsupported")
    try:
        if not math.isfinite(value) or (type(value) is int and int(float(value)) != value):
            raise ValueError("source_owner_literal_unsupported")
        bits = struct.pack("!f" if dtype == "f32" else "!d", value)
    except (OverflowError, struct.error) as exc:
        raise ValueError("source_owner_literal_unsupported") from exc
    return bits.hex()


def owner_float_bits_allow_infinity(value: Any, dtype: str) -> str:
    """Encode an exact Source padding identity, including signed infinity.

    Ordinary owner constants remain finite-only.  Masked max reductions are
    the one closed owner path whose Source identity is ``-inf``; this helper
    admits only infinities (not NaN) in addition to the ordinary finite
    values, and still uses the declared dtype's exact IEEE encoding.
    """
    if dtype not in {"f32", "f64"} or type(value) not in {int, float}:
        raise ValueError("source_owner_literal_unsupported")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError("source_owner_literal_unsupported")
    try:
        bits = struct.pack("!f" if dtype == "f32" else "!d", value)
    except (OverflowError, struct.error) as exc:
        raise ValueError("source_owner_literal_unsupported") from exc
    return bits.hex()


def owner_reduction_identity_bits(value: Any, dtype: str, combiner: str) -> str:
    """Encode one explicitly declared reduction identity.

    ``ReductionBoundary.identity`` is a JSON scalar.  Finite sum/prod
    identities must remain numeric; max may use the canonical JSON spellings
    ``-inf``/``NEG_INF`` because JSON has no IEEE infinity literal.  No other
    symbolic or string value is interpreted here.
    """
    if combiner == "sum":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != 0.0:
            raise ValueError("source_owner_reduction_boundary_mismatch")
        expected = 0.0
    elif combiner == "prod":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != 1.0:
            raise ValueError("source_owner_reduction_boundary_mismatch")
        expected = 1.0
    elif combiner == "max":
        if isinstance(value, str):
            if value not in {"-inf", "NEG_INF"}:
                raise ValueError("source_owner_reduction_boundary_mismatch")
            expected = float("-inf")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or value != float("-inf"):
            raise ValueError("source_owner_reduction_boundary_mismatch")
        else:
            expected = float("-inf")
    else:
        raise ValueError("source_owner_reduction_boundary_mismatch")
    return owner_float_bits_allow_infinity(expected, dtype)


def source_float_literal(value: Any) -> int | float:
    """Parse the small, explicit Source fill grammar without evaluation."""
    if not isinstance(value, str):
        raise ValueError("source_owner_load_fill_unproved")
    try:
        node = ast.parse(value, mode="eval").body
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError("source_owner_load_fill_unproved") from exc
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Call)
        and isinstance(node.operand.func, ast.Name)
        and node.operand.func.id == "float"
        and not node.operand.keywords
        and len(node.operand.args) == 1
        and isinstance(node.operand.args[0], ast.Constant)
        and node.operand.args[0].value == "inf"
    ):
        return float("-inf")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and not node.keywords
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in {"inf", "-inf"}
    ):
        return float(node.args[0].value)
    raise ValueError("source_owner_load_fill_unproved")


def source_fill_json_literal(value: Any) -> int | float | str | None:
    """Project one proven Source fill into the canonical JSON spelling.

    This is the only Source-to-provider fill conversion. It deliberately
    shares ``source_float_literal`` with owner validation: booleans and
    arbitrary expressions are not accepted merely because JSON can encode
    them. JSON has no infinity number, so the two existing canonical string
    spellings ``-inf`` and ``inf`` represent the corresponding signed
    infinities.
    """

    try:
        parsed = source_float_literal(value)
    except (TypeError, ValueError, SyntaxError):
        return None
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return "-inf" if parsed == float("-inf") else "inf"
    if isinstance(parsed, float) and parsed.is_integer():
        return int(parsed)
    return parsed


def build_scalar_owner_dag(
    root: str,
    describe: Callable[[str], tuple[str, str, Any, Sequence[str]]],
) -> tuple[tuple[Any, ...], ...]:
    """Canonicalize an existing DAG, retaining operand order AND SSA sharing."""

    rows: list[tuple[Any, ...]] = []
    indices: dict[str, int] = {}
    active: set[str] = set()

    def visit(value: str) -> int:
        if value in active:
            raise ValueError("source_owner_dataflow_cycle")
        if value in indices:
            return indices[value]
        active.add(value)
        op, dtype, payload, operands = describe(value)
        if op not in _OWNER_ARITY or dtype not in {"f32", "f64"} or len(operands) != _OWNER_ARITY[op]:
            raise ValueError("source_owner_operation_unsupported")
        children = tuple(visit(operand) for operand in operands)
        if any(rows[index][1] != dtype for index in children):
            raise ValueError("source_owner_implicit_conversion_forbidden")
        active.remove(value)
        indices[value] = len(rows)
        rows.append((op, dtype, payload, children))
        return indices[value]

    visit(root)
    return tuple(rows)


def scalar_owner_digest(tensor: str, graph: Sequence[Sequence[Any]]) -> str:
    """Validate a closed scalar DAG before committing to its target and value."""

    if not isinstance(tensor, str) or re.fullmatch(r"t[0-9]+", tensor) is None or not graph:
        raise ValueError("source_owner_graph_invalid")
    for index, row in enumerate(graph):
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise ValueError("source_owner_graph_invalid")
        op, dtype, payload, children = row
        if (
            not isinstance(op, str) or op not in _OWNER_ARITY
            or dtype not in {"f32", "f64"}
            or not isinstance(children, (tuple, list))
            or len(children) != _OWNER_ARITY[op]
            or any(type(child) is not int or not 0 <= child < index for child in children)
            or any(graph[child][1] != dtype for child in children)
        ):
            raise ValueError("source_owner_graph_invalid")
        if op == "input":
            valid = isinstance(payload, str) and re.fullmatch(r"t[0-9]+", payload) is not None
        elif op == "const":
            valid = isinstance(payload, str) and re.fullmatch(
                r"[0-9a-f]{8}" if dtype == "f32" else r"[0-9a-f]{16}", payload
            ) is not None
        else:
            valid = payload is None
        if not valid:
            raise ValueError("source_owner_graph_invalid")
    if graph[-1][0] in {"input", "const"}:
        raise ValueError("source_owner_root_unsupported")
    # Rebuild by existing indices to enforce reachability and canonical order;
    # do not merge equal-but-distinct producers.
    rebuilt = build_scalar_owner_dag(
        str(len(graph) - 1),
        lambda value: (*graph[int(value)][:3], tuple(str(i) for i in graph[int(value)][3])),
    )
    payload = {"kind": "store", "tensor": tensor, "shape": [], "graph": graph}
    if json.dumps(rebuilt) != json.dumps(graph):
        raise ValueError("source_owner_graph_not_canonical")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_scalar_owner_binding(
    function: Mapping[str, Any],
    parameters: Sequence[Mapping[str, Any]],
    access: Mapping[str, Any],
) -> None:
    """Reject-only J check; Source ledger token equality is a separate obligation.

    Works before and after handle allocation. Only top-level scalar f32/f64
    arithmetic with exact ABI scalar leaves is supported. No load, broadcast,
    cast, region, reduction, padding or algebraic equivalence is inferred.
    """

    token = access.get("semantic_owner_authorization")
    if not isinstance(token, str) or re.fullmatch(SOURCE_OWNER_TOKEN_PATTERN, token) is None:
        raise ValueError("source_exact_owner_binding_unavailable")
    if not isinstance(access.get("node_ref"), str) or not access["node_ref"]:
        raise ValueError("source_owner_node_ref_missing")
    if access.get("kind") != "store":
        raise ValueError("source_owner_access_kind_unsupported")
    targets = [row for row in function.get("outputs", ()) if row.get("name") == access.get("tensor")]
    target_abi = [row for row in parameters if row.get("logical_ref") == access.get("tensor")]
    if (
        len(targets) != 1 or targets[0].get("shape") != []
        or targets[0].get("dtype") not in {"f32", "f64"}
        or len(target_abi) != 1 or target_abi[0].get("kind") not in {"tensor", "pointer"}
        or target_abi[0].get("mode") != "output" or target_abi[0].get("rank") != 0
        or target_abi[0].get("alias_of") is not None
        or target_abi[0].get("dtype") != targets[0]["dtype"]
    ):
        raise ValueError("source_owner_target_type_mismatch")
    try:
        index = ast.parse(access.get("index_expression"), mode="eval").body
    except (TypeError, SyntaxError) as exc:
        raise ValueError("source_owner_index_mismatch") from exc
    if not isinstance(index, (ast.List, ast.Tuple)) or index.elts:
        raise ValueError("source_owner_index_mismatch")
    inputs = {row["name"]: row for row in function.get("inputs", ())}
    scalar_inputs: dict[str, str] = {}
    for parameter in parameters:
        ref = parameter.get("logical_ref")
        row = inputs.get(ref)
        if parameter.get("kind") != "scalar" or row is None:
            continue
        if (
            parameter.get("alias_of") is not None or parameter.get("mode") != "value"
            or parameter.get("rank") != 0
        ):
            raise ValueError("source_owner_scalar_abi_invalid")
        if row.get("shape") == [] and row.get("dtype") in {"f32", "f64"}:
            if ref in scalar_inputs or parameter.get("dtype") != row["dtype"]:
                raise ValueError("source_owner_scalar_abi_invalid")
            scalar_inputs[ref] = row["dtype"]
    body = function.get("body", ())
    producers: dict[str, tuple[int, Mapping[str, Any]]] = {}
    nodes: dict[str, Mapping[str, Any]] = {}
    defined = set(inputs)
    for index, row in enumerate(body):
        if not isinstance(row, Mapping) or (row.get("id") is not None and not isinstance(row["id"], str)):
            raise ValueError("source_owner_graph_invalid")
        if row.get("id") in nodes or row.get("body") or row.get("positive") or row.get("negative"):
            raise ValueError("source_owner_graph_ambiguous")
        if row.get("id") is not None:
            nodes[row["id"]] = row
        result = row.get("result")
        results = [result] if isinstance(result, str) else result
        if isinstance(results, (tuple, list)):
            for value in results:
                if not isinstance(value, str) or value in defined:
                    raise ValueError("source_owner_graph_ambiguous")
                defined.add(value)
        if isinstance(result, str):
            producers[result] = (index, row)

    def describe(value: str) -> tuple[str, str, Any, Sequence[str]]:
        if value in scalar_inputs:
            return "input", scalar_inputs[value], value, ()
        if value not in producers:
            raise ValueError("source_owner_input_unsupported")
        index, row = producers[value]
        typ, attrs, operands = row.get("type"), row.get("attrs", {}), row.get("operands", [])
        if not isinstance(typ, Mapping) or set(typ) != {"dtype", "shape"} or typ["shape"] != []:
            raise ValueError("source_owner_type_unsupported")
        dtype = typ["dtype"]
        if not isinstance(attrs, Mapping) or not isinstance(operands, list):
            raise ValueError("source_owner_graph_invalid")
        if any(operand in producers and producers[operand][0] >= index for operand in operands):
            raise ValueError("source_owner_non_dominating_operand")
        if row.get("op") == "const" and set(attrs) == {"value"} and not operands:
            return "const", dtype, owner_float_bits(attrs["value"], dtype), ()
        fn = attrs.get("fn")
        if set(attrs) == {"fn"} and (
            (row.get("op") == "map_binary" and fn in {"add", "sub", "mul", "div"})
            or (row.get("op") == "map_unary" and fn == "neg")
        ):
            return fn, dtype, None, operands
        raise ValueError("source_owner_operation_unsupported")

    claimed = nodes.get(access["node_ref"])
    if claimed is None or not isinstance(claimed.get("result"), str):
        raise ValueError("source_owner_producer_mismatch_or_ambiguous")
    # Verify the authored claim first. A different matching node cannot repair
    # a false claim. Other roots are visited only to preserve the independent
    # uniqueness obligation after the claimed root has actually matched.
    matches: list[str | None] = []
    ordered_roots = (claimed, *(row for row in body if row is not claimed))
    for root_index, row in enumerate(ordered_roots):
        if root_index and matches != [access["node_ref"]]:
            raise ValueError("source_owner_producer_mismatch_or_ambiguous")
        result = row.get("result")
        if not isinstance(result, str):
            continue
        try:
            graph = build_scalar_owner_dag(result, describe)
            if graph[-1][1] != targets[0]["dtype"]:
                continue
            digest = scalar_owner_digest(access.get("tensor"), graph)
        except (ValueError, TypeError, KeyError, RecursionError):
            continue
        if digest == token.rsplit("_", 1)[1]:
            matches.append(row.get("id"))
    if matches != [access.get("node_ref")]:
        raise ValueError("source_owner_producer_mismatch_or_ambiguous")


def build_tensor_owner_dag(
    root: str,
    describe: Callable[[str], tuple[str, str, Sequence[int | str], Any, Sequence[str]]],
) -> tuple[tuple[Any, ...], ...]:
    """Exact typed DAG; never quotient distinct SSA nodes or reorder edges."""
    rows: list[tuple[Any, ...]] = []
    indices: dict[str, int] = {}
    active: set[str] = set()

    def visit(value: str) -> int:
        if value in active:
            raise ValueError("source_owner_dataflow_cycle")
        if value in indices:
            return indices[value]
        active.add(value)
        op, dtype, shape, payload, operands = describe(value)
        arity = {**_OWNER_ARITY, "load": 0, "reduce": 1, "cast": 1, "broadcast": 1, "reshape": 1, "transpose": 1, "tanh": 1, "rsqrt": 1, "rsqrt_ftz": 1}.get(op)
        if (
            arity is None or len(operands) != arity or dtype not in {"i32", "i64", "f32", "f64"}
            or any(not ((type(extent) is int and extent > 0) or (isinstance(extent, str) and re.fullmatch(r"x[0-9]+", extent))) for extent in shape)
        ):
            raise ValueError("source_owner_typed_operation_unsupported")
        if dtype in {"i32", "i64"} and op != "input":
            raise ValueError("source_owner_integer_operation_unsupported")
        children = tuple(visit(operand) for operand in operands)
        shape = tuple(shape)
        if op == "cast":
            source_dtype = rows[children[0]][1]
            if payload != owner_cast_kind(source_dtype, dtype):
                raise ValueError("source_owner_cast_semantics_unproved")
        elif any(rows[child][1] != dtype for child in children):
            raise ValueError("source_owner_implicit_conversion_forbidden")
        if op == "reduce":
            source_shape = rows[children[0]][2]
            if not isinstance(payload, (tuple, list)) or len(payload) != 2 or type(payload[0]) is not int or payload[1] not in {"sum", "prod", "max"} or not 0 <= payload[0] < len(source_shape) or shape != source_shape[:payload[0]] + source_shape[payload[0] + 1:]:
                raise ValueError("source_owner_reduction_contract_mismatch")
        elif op == "reshape":
            source_shape = rows[children[0]][2]
            # Static, row-major element order only. Symbolic product equality
            # and arbitrary compiler layout/reordering are not inferred here.
            if payload is not None or any(type(n) is not int for n in (*source_shape, *shape)) or math.prod(source_shape) != math.prod(shape):
                raise ValueError("source_owner_reshape_unproved")
        elif op == "transpose":
            source_shape = rows[children[0]][2]
            if not isinstance(payload, (tuple, list)) or any(type(axis) is not int for axis in payload) or sorted(payload) != list(range(len(source_shape))) or shape != tuple(source_shape[axis] for axis in payload):
                raise ValueError("source_owner_transpose_unproved")
        elif op == "broadcast":
            source_shape = rows[children[0]][2]
            if not isinstance(payload, (tuple, list)) or len(payload) != len(source_shape) or len(set(payload)) != len(payload) or any(type(axis) is not int or not 0 <= axis < len(shape) for axis in payload) or tuple(shape[axis] for axis in payload) != source_shape:
                raise ValueError("source_owner_broadcast_unproved")
        elif any(rows[child][2] != shape for child in children):
            raise ValueError("source_owner_implicit_broadcast_forbidden")
        if op in {"neg", "exp", "add", "sub", "mul", "div", "tanh", "rsqrt", "rsqrt_ftz"} and payload is not None:
            raise ValueError("source_owner_attributes_unsupported")
        if op == "rsqrt_ftz" and dtype != "f32":
            raise ValueError("source_owner_ftz_dtype_unsupported")
        if op == "const":
            if not isinstance(payload, str) or re.fullmatch(
                r"[0-9a-f]{8}" if dtype == "f32" else r"[0-9a-f]{16}", payload
            ) is None or not math.isfinite(struct.unpack("!f" if dtype == "f32" else "!d", bytes.fromhex(payload))[0]):
                raise ValueError("source_owner_literal_unsupported")
        active.remove(value)
        indices[value] = len(rows)
        rows.append((op, dtype, shape, payload, children))
        return indices[value]

    visit(root)
    return tuple(rows)


def owner_cast_kind(source: str, target: str) -> str:
    """Closed typed conversions; no bitcasts, unsigned or implicit casts."""
    if source in {"i32", "i64"} and target in {"f32", "f64"}:
        return "signed_to_float_rne"
    if (source, target) == ("f32", "f64"):
        return "float_extend"
    if (source, target) == ("f64", "f32"):
        return "float_truncate_rne"
    raise ValueError("source_owner_cast_semantics_unproved")


def owner_padding_obligations(
    graph: Sequence[Sequence[Any]], *, store: Mapping[str, Any] | None = None,
) -> dict[int, tuple[str, ...]]:
    """Prove fills on the complete DAG, not a preferred provenance path.

    The only folded arithmetic laws are exact neutral/zero laws plus the
    explicitly represented ``-inf`` max identity and ``exp(-inf) == 0``.
    No general host floating-point evaluation is used as a proof of target
    math. Unknown scalar values and unmasked loads cannot be assumed finite.
    """
    pending: list[set[str]] = []
    outside: list[str | None] = []
    domains: dict[str, tuple[Any, ...]] = {}
    obligations: dict[int, tuple[str, ...]] = {}
    for index, (op, dtype, shape, payload, children) in enumerate(graph):
        dependencies = set().union(*(pending[child] for child in children))
        bits: str | None = None
        if op == "load":
            if payload["predicate"] is not None:
                token = payload["occurrence"]
                dependencies.add(token)
                domains[token] = (tuple(payload["shape"]), tuple(payload["coordinates"]), payload["predicate"])
                bits = payload["fill"]
        elif op == "const":
            bits = payload
        elif op == "broadcast":
            bits = outside[children[0]]
        elif op in {"reshape", "transpose"}:
            # A uniform fill alone does not prove that a transformed mask
            # protects the same output coordinates. Keep the domain obligation
            # closed until a coordinate-transport proof exists.
            identity = (
                tuple(shape) == tuple(graph[children[0]][2]) if op == "reshape"
                else tuple(payload) == tuple(range(len(shape)))
            )
            if dependencies and not identity:
                raise ValueError("source_owner_padding_coordinate_transport_unproved")
            bits = outside[children[0]]
        elif op == "cast":
            source_dtype = graph[children[0]][1]
            source_bits = outside[children[0]]
            # Only exact finite neutral constants transport an identity.
            # In particular, do not host-evaluate an arbitrary i64->f32 cast.
            if source_dtype in {"f32", "f64"} and source_bits is not None:
                value = struct.unpack("!f" if source_dtype == "f32" else "!d", bytes.fromhex(source_bits))[0]
                if value in {-1.0, 0.0, 1.0}:
                    bits = owner_float_bits(value, dtype)
        elif op == "exp":
            source_bits = outside[children[0]]
            if source_bits is not None:
                value = struct.unpack(
                    "!f" if dtype == "f32" else "!d",
                    bytes.fromhex(source_bits),
                )[0]
                # The only masked-domain exp law admitted here is the exact
                # IEEE endpoint exp(-inf) == +0.  No finite approximation or
                # general host evaluation is used to discharge padding.
                if value == float("-inf"):
                    bits = owner_float_bits(0.0, dtype)
        elif op == "reduce":
            if dependencies:
                if len({json.dumps(domains[token], sort_keys=True) for token in dependencies}) != 1:
                    raise ValueError("source_owner_padding_domains_differ")
                identity = owner_float_bits_allow_infinity(
                    0.0
                    if payload[1] == "sum"
                    else 1.0
                    if payload[1] == "prod"
                    else float("-inf"),
                    dtype,
                )
                if outside[children[0]] != identity:
                    raise ValueError("source_owner_outside_identity_unproved")
                obligations[index] = tuple(sorted(dependencies))
            dependencies = set()
        elif children and all(outside[child] is not None for child in children):
            values = [struct.unpack("!f" if dtype == "f32" else "!d", bytes.fromhex(outside[child]))[0] for child in children]
            if op == "neg":
                bits = owner_float_bits(-values[0], dtype)
            elif op == "mul" and (0.0 in values or 1.0 in values):
                bits = owner_float_bits(values[0] * values[1], dtype)
            elif op in {"add", "sub"} and (values[0] == 0.0 or values[1] == 0.0):
                bits = owner_float_bits(values[0] + values[1] if op == "add" else values[0] - values[1], dtype)
            elif op == "div" and values[1] == 1.0:
                bits = outside[children[0]]
        pending.append(dependencies)
        outside.append(bits)
    if pending and pending[-1]:
        # The direct pointwise branch need not pass through a reduction if
        # its actual final store excludes exactly the same outside lanes.
        # Reductions above have already discharged their own identity laws;
        # a store mask cannot excuse an unproved reduction input identity.
        if store is None or store.get("predicate") is None:
            raise ValueError("source_owner_unclosed_padding_path")
        protected = (tuple(store["shape"]), tuple(store["coordinates"]), store["predicate"])
        if any(domains[token] != protected for token in pending[-1]):
            raise ValueError("source_owner_store_mask_domain_mismatch")
    return obligations


def tensor_owner_digest(store: Mapping[str, Any], graph: Sequence[Sequence[Any]]) -> str:
    if not graph or graph[-1][0] in {"input", "const", "load"}:
        raise ValueError("source_owner_root_unsupported")
    expected_shape = tuple(extent for coordinate, extent in zip(store["coordinates"], store["shape"], strict=True) if isinstance(coordinate, str))
    if tuple(graph[-1][2]) != expected_shape:
        raise ValueError("source_owner_store_value_shape_mismatch")
    for index, row in enumerate(graph):
        if len(row) != 5 or any(type(child) is not int or not 0 <= child < index for child in row[4]):
            raise ValueError("source_owner_graph_invalid")
    rebuilt = build_tensor_owner_dag(str(len(graph) - 1), lambda value: (
        *graph[int(value)][:4], tuple(str(child) for child in graph[int(value)][4])
    ))
    if json.dumps(rebuilt) != json.dumps(graph):
        raise ValueError("source_owner_graph_not_canonical")
    owner_padding_obligations(graph, store=store)
    return hashlib.sha256(json.dumps({"store": store, "graph": graph}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _owner_computed_guard_predicate(
    function: Mapping[str, Any], parameters: Sequence[Mapping[str, Any]],
    access: Mapping[str, Any], boundary: Mapping[str, Any],
    value_dimension_relations: Sequence[Mapping[str, Any]] | None,
    *, axis: int, extent: int | str, tensor_shape: Sequence[int | str],
) -> tuple[str, int, int | str]:
    """Prove one authored i1 less-than guard denotes this logical axis bound.

    A guard receipt proves Source occurrence, not J arithmetic. This check
    independently closes the exact J operands before the owner digest can
    consume the same predicate on either pre-allocation or canonical J.
    """
    if (
        access.get("kind") not in {"load", "store"}
        or not isinstance(access.get("semantic_guard_authorization"), str)
        or re.fullmatch(r"so_[0-9a-f]{32}", access["semantic_guard_authorization"]) is None
        or not isinstance(extent, str) or re.fullmatch(r"x[0-9]+", extent) is None
        or not isinstance(value_dimension_relations, Sequence)
    ):
        raise ValueError("source_owner_computed_guard_authorization_unproved")
    access_ref = access.get("access_id", access.get("access_label"))
    relations = [
        row for row in boundary.get("access_guard_relations", ())
        if isinstance(row, Mapping) and row.get("access_ref") == access_ref
    ]
    if len(relations) > 1:
        raise ValueError("source_owner_computed_guard_relation_ambiguous")
    authored_predicate = access.get("predicate")
    if not isinstance(authored_predicate, str):
        raise ValueError("source_owner_computed_guard_value_unproved")
    if not relations:
        raise ValueError("source_owner_computed_guard_relation_missing")
    relation = relations[0]
    value_ref = relation.get("value_ref")
    # TypeCheck and Functional Boundary consume this i1 only through the
    # access-guard relation. A duplicate ContractExpr predicate cannot close
    # downstream; reject it at the same owner boundary instead of signing a
    # producer that later fails the independent J legality check.
    if authored_predicate != "true":
        raise ValueError("source_owner_computed_guard_value_unproved")
    expected_axes = [index for index in range(len(tensor_shape)) if index != axis]
    if (
        relation.get("polarity") != "positive"
        or relation.get("axes") != expected_axes
    ):
        raise ValueError("source_owner_computed_guard_projection_unproved")
    # This bounded profile is a single logical column/row comparison. An
    # arbitrary tensor guard, reshape, fused operation or control region is
    # not alpha-equivalent to the Source owner's lt predicate.
    if len(tensor_shape) != 2 or axis not in {0, 1}:
        raise ValueError("source_owner_computed_guard_rank_unsupported")
    producers: dict[str, Mapping[str, Any]] = {}
    for row in function.get("body", ()):
        if not isinstance(row, Mapping) or row.get("body") or row.get("positive") or row.get("negative"):
            raise ValueError("source_owner_computed_guard_region_unsupported")
        result = row.get("result")
        if isinstance(result, str):
            if result in producers:
                raise ValueError("source_owner_computed_guard_producer_ambiguous")
            producers[result] = row
    guard = producers.get(value_ref)
    if guard is None:
        raise ValueError("source_owner_computed_guard_producer_missing")
    if guard.get("op") != "map_binary" or guard.get("attrs") != {"fn": "lt"}:
        raise ValueError("source_owner_computed_guard_comparison_operation_unproved")
    if guard.get("type") != {"dtype": "i1", "shape": [extent]}:
        raise ValueError("source_owner_computed_guard_comparison_type_unproved")
    if not isinstance(guard.get("operands"), list) or len(guard["operands"]) != 2:
        raise ValueError("source_owner_computed_guard_comparison_operands_unproved")
    coordinate_ref, limit_ref = guard["operands"]
    guard_broadcasts = [
        row for row in boundary.get("broadcasts", ())
        if isinstance(row, Mapping) and row.get("node_ref") == guard.get("id")
    ]
    seen_guard_operands: set[str] = set()
    for relation in guard_broadcasts:
        operand = relation.get("operand")
        if operand in seen_guard_operands:
            raise ValueError("source_owner_computed_guard_broadcast_ambiguous")
        if operand not in {coordinate_ref, limit_ref}:
            raise ValueError("source_owner_computed_guard_broadcast_operand_unproved")
        seen_guard_operands.add(operand)
        axes = relation.get("axes")
        if not isinstance(axes, list):
            raise ValueError("source_owner_computed_guard_broadcast_axis_type_unproved")
        if any(type(value) is not int for value in axes):
            raise ValueError("source_owner_computed_guard_broadcast_axis_type_unproved")
        if any(value not in {-1, 0} for value in axes):
            raise ValueError("source_owner_computed_guard_broadcast_axis_range_unproved")
        if len({0 if value == -1 else value for value in axes}) != len(axes):
            raise ValueError("source_owner_computed_guard_broadcast_axis_duplicate")
        expected_inserted = unique_exact_rank_insertion_projection(
            [] if operand == limit_ref else [extent],
            [extent],
        )
        normalized_axes = tuple(0 if value == -1 else value for value in axes)
        if expected_inserted is None or normalized_axes != expected_inserted:
            raise ValueError("source_owner_computed_guard_broadcast_projection_unproved")
        if operand == limit_ref:
            # Scalar J has an empty affine input map: [] and [0] are the
            # same exact expansion. The Source-bound scalar/type check below
            # still proves which value supplies the comparison limit.
            if expected_inserted != (0,):
                raise ValueError("source_owner_computed_guard_broadcast_projection_unproved")
        elif normalized_axes or expected_inserted:
            # A declared coordinate relation may only restate the identity.
            raise ValueError("source_owner_computed_guard_broadcast_coordinate_unproved")
    coordinate = producers.get(coordinate_ref)
    if coordinate is None or coordinate.get("op") != "iota":
        raise ValueError("source_owner_computed_guard_coordinate_producer_unproved")
    if coordinate.get("operands", []) != []:
        raise ValueError("source_owner_computed_guard_coordinate_operands_unproved")
    if coordinate.get("type") != {"dtype": "i32", "shape": [extent]}:
        raise ValueError("source_owner_computed_guard_coordinate_type_unproved")
    if coordinate.get("attrs") not in ({"n": extent}, {"n": extent, "dtype": "i32"}):
        raise ValueError("source_owner_computed_guard_coordinate_attrs_unproved")
    inputs = [row for row in function.get("inputs", ()) if row.get("name") == limit_ref]
    bindings = [row for row in parameters if row.get("logical_ref") == limit_ref]
    dimensions = [row for row in value_dimension_relations if row.get("value_ref") == limit_ref]
    if (
        len(inputs) != 1 or inputs[0].get("dtype") != "i32" or inputs[0].get("shape") != []
        or len(bindings) != 1 or bindings[0].get("kind") != "scalar"
        or bindings[0].get("mode") != "value" or bindings[0].get("dtype") != "i32"
        or bindings[0].get("alias_of") is not None
        or dimensions != [{"value_ref": limit_ref, "dimension_ref": extent, "relation": "eq"}]
    ):
        raise ValueError("source_owner_computed_guard_extent_unproved")
    return ("lt", axis, extent)


def _owner_j_memory_contract(
    function: Mapping[str, Any], parameters: Sequence[Mapping[str, Any]],
    boundary: Mapping[str, Any], access: Mapping[str, Any],
    value_dimension_relations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recognize direct rank-one/two domains and constant input/output slices."""
    tensor = access.get("tensor")
    kind = access.get("kind")
    rows = function.get("inputs" if kind == "load" else "outputs", ())
    targets = [row for row in rows if row.get("name") == tensor]
    abi = [row for row in parameters if row.get("logical_ref") == tensor]
    if len(targets) != 1 or len(abi) != 1:
        raise ValueError("source_owner_memory_abi_ambiguous")
    dtype, shape = targets[0].get("dtype"), targets[0].get("shape")
    if (
        dtype not in {"f32", "f64"} or not isinstance(shape, list) or len(shape) > 2
        or any(not ((type(n) is int and n > 0) or (isinstance(n, str) and re.fullmatch(r"x[0-9]+", n))) for n in shape)
        or abi[0].get("dtype") != dtype or abi[0].get("rank") != len(shape)
        or abi[0].get("kind") not in {"tensor", "pointer"} or abi[0].get("alias_of") is not None
        or abi[0].get("mode") != ("input" if kind == "load" else "output")
    ):
        raise ValueError("source_owner_memory_type_unsupported")
    def parse(text: Any) -> ast.expr:
        try:
            return ast.parse(text, mode="eval").body
        except (TypeError, SyntaxError) as exc:
            raise ValueError("source_owner_memory_expression_invalid") from exc

    expression = parse(access.get("index_expression"))
    if not isinstance(expression, (ast.List, ast.Tuple)):
        raise ValueError("source_owner_memory_index_not_ranked")
    if len(expression.elts) != len(shape):
        raise ValueError("source_owner_memory_index_rank_mismatch")
    coordinates: list[str | int] = []
    variable_axes: dict[str, tuple[int, int | str]] = {}
    for ordinal, (coordinate, extent) in enumerate(zip(expression.elts, shape, strict=True)):
        if kind in {"load", "store"} and isinstance(coordinate, ast.Name):
            if coordinate.id in variable_axes:
                raise ValueError("source_owner_memory_axis_reused")
            if re.fullmatch(r"x[0-9]+", coordinate.id):
                raise ValueError("source_owner_memory_extent_as_coordinate")
            variable_axes[coordinate.id] = (ordinal, extent)
            coordinates.append(f"axis{ordinal}")
        elif kind in {"load", "store"} and isinstance(coordinate, ast.Constant) and type(coordinate.value) is int and type(extent) is int and 0 <= coordinate.value < extent:
            coordinates.append(coordinate.value)
        else:
            raise ValueError("source_owner_memory_coordinate_expression_unsupported")

    def bounds(text: str, *, predicate: bool = False) -> set[tuple[str, str, int | str]]:
        tree = parse(text)
        clauses = tree.values if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And) else [tree]
        result: set[tuple[str, str, int | str]] = set()
        for clause in clauses:
            if not isinstance(clause, ast.Compare):
                raise ValueError(
                    "source_owner_predicate_expression_unsupported"
                    if predicate else "source_owner_domain_expression_unsupported"
                )
            for left, operator, right in zip([clause.left, *clause.comparators[:-1]], clause.ops, clause.comparators, strict=True):
                if isinstance(left, ast.Constant) and type(left.value) is int and left.value == 0 and isinstance(operator, ast.LtE) and isinstance(right, ast.Name):
                    result.add((right.id, "lower", 0))
                elif isinstance(left, ast.Name) and isinstance(operator, ast.Lt) and isinstance(right, ast.Constant) and type(right.value) is int:
                    result.add((left.id, "upper", right.value))
                elif isinstance(left, ast.Name) and isinstance(operator, ast.Lt) and isinstance(right, ast.Name) and re.fullmatch(r"x[0-9]+", right.id):
                    result.add((left.id, "upper", right.id))
                else:
                    raise ValueError(
                        "source_owner_predicate_expression_unsupported"
                        if predicate else "source_owner_domain_expression_unsupported"
                    )
        return result

    predicate = None
    if variable_axes:
        required = {(axis, bound, value) for axis, (_ordinal, extent) in variable_axes.items() for bound, value in (("lower", 0), ("upper", extent))}
        domains = boundary.get("domains", {})
        if not isinstance(domains, Mapping):
            raise ValueError("source_owner_domain_unproved")
        relevant = []
        for text in domains.values():
            if isinstance(text, str) and set(variable_axes) & {node.id for node in ast.walk(parse(text)) if isinstance(node, ast.Name)}:
                relevant.append(bounds(text))
        if not relevant:
            raise ValueError("source_owner_domain_coordinate_unbound")
        if set.union(*relevant) != required:
            raise ValueError("source_owner_domain_bounds_mismatch")
        if access.get("predicate") is not None:
            last = [(axis, ordinal, extent) for axis, (ordinal, extent) in variable_axes.items() if ordinal == len(shape) - 1]
            if len(last) != 1:
                raise ValueError("source_owner_predicate_unproved")
            axis, ordinal, extent = last[0]
            guard_relations = [
                row for row in boundary.get("access_guard_relations", ())
                if isinstance(row, Mapping)
                and row.get("access_ref") == access.get("access_id", access.get("access_label"))
            ]
            authored_predicate = access["predicate"]
            guard_results = {
                row.get("result") for row in function.get("body", ())
                if isinstance(row, Mapping) and isinstance(row.get("result"), str)
            }
            if guard_relations or authored_predicate in guard_results or authored_predicate == "true":
                predicate = _owner_computed_guard_predicate(
                    function, parameters, access, boundary, value_dimension_relations,
                    axis=ordinal, extent=extent, tensor_shape=shape,
                )
            else:
                observed = bounds(authored_predicate, predicate=True)
                if observed not in (required, {(axis, "upper", extent)}, {(axis, "lower", 0), (axis, "upper", extent)}):
                    raise ValueError("source_owner_predicate_unproved")
                predicate = ("lt", ordinal, extent)
    elif access.get("predicate") is not None:
        raise ValueError("source_owner_predicate_unproved")
    fill = None
    if predicate is not None and kind == "load":
        fill = owner_float_bits(access.get("fill_value"), dtype)
    elif access.get("fill_value") is not None:
        raise ValueError("source_owner_orphan_fill")
    occurrence = access.get("semantic_access_authorization")
    if not isinstance(occurrence, str) or re.fullmatch(r"so_[0-9a-f]{32}", occurrence) is None:
        raise ValueError("source_owner_load_occurrence_unproved")
    return dict(occurrence=occurrence, tensor=tensor, dtype=dtype, shape=shape,
                coordinates=coordinates, domain=shape, predicate=predicate, fill=fill)


def validated_j_computed_guard_bound(
    function: Mapping[str, Any], parameters: Sequence[Mapping[str, Any]],
    access: Mapping[str, Any], boundary: Mapping[str, Any],
    *, value_dimension_relations: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[int, str]:
    """Recheck an existing J guard's exact axis/extent, without issuing Source authority.

    The private Source ledger must still authenticate the access and guard
    occurrences at ingress and pre-Replay. This is only the shared J-content
    proof needed by consumers of a literal-true access predicate.
    """
    if access.get("kind") != "load" or access.get("predicate") != "true":
        raise ValueError("source_owner_computed_guard_padding_access_unproved")
    contract = _owner_j_memory_contract(
        function, parameters, boundary, access, value_dimension_relations,
    )
    predicate = contract["predicate"]
    if (
        not isinstance(predicate, tuple) or len(predicate) != 3
        or predicate[0] != "lt" or type(predicate[1]) is not int
        or not isinstance(predicate[2], str)
    ):
        raise ValueError("source_owner_computed_guard_padding_bound_unproved")
    return predicate[1], predicate[2]


def validate_source_owner_binding(
    function: Mapping[str, Any], parameters: Sequence[Mapping[str, Any]],
    access: Mapping[str, Any], boundary: Mapping[str, Any],
    *, value_dimension_relations: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Check candidate content against its claimed owner commitment.

    This is the shared reject-only content check, not Source issuance proof:
    it has no private Source ledger. Source-aware ingress and pre-Replay must
    separately match the receipt to their own current Source binding. Boundary
    projection and fallback reuse this check without gaining issuance power.
    """
    token = access.get("semantic_owner_authorization")
    if not isinstance(token, str) or re.fullmatch(SOURCE_OWNER_TOKEN_PATTERN, token) is None:
        raise ValueError("source_exact_owner_binding_unavailable")
    if access.get("semantic_access_authorization") != token.rsplit("_", 1)[0]:
        raise ValueError("source_owner_store_occurrence_mismatch")
    # The original, domain-free scalar ABI proof is disjoint from tensor DAG
    # commitments. Do not use it as a fallback for masked or indexed stores.
    if access.get("predicate") is None and access.get("fill_value") is None:
        try:
            _validate_scalar_owner_binding(function, parameters, access)
            return
        except (ValueError, TypeError, KeyError):
            pass
    if access.get("kind") != "store" or not isinstance(access.get("node_ref"), str):
        raise ValueError("source_owner_store_required")
    store = _owner_j_memory_contract(
        function, parameters, boundary, access, value_dimension_relations,
    )
    if not token.startswith(store["occurrence"] + "_"):
        raise ValueError("source_owner_store_occurrence_mismatch")
    load_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in boundary.get("accesses", ()):
        if row.get("kind") != "load":
            continue
        load_rows.setdefault(row.get("tensor"), []).append(row)
    loads: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
    producers: dict[str, tuple[int, Mapping[str, Any]]] = {}
    broadcast_views: dict[str, tuple[str, str, list[int | str], tuple[int, ...]]] = {}
    ids: set[str] = set()
    inputs = {row["name"]: row for row in function.get("inputs", ())}
    for index, row in enumerate(function.get("body", ())):
        node, result = row.get("id"), row.get("result")
        if row.get("body") or row.get("positive") or row.get("negative"):
            raise ValueError("source_owner_region_unsupported")
        if node is not None:
            if not isinstance(node, str) or node in ids:
                raise ValueError("source_owner_graph_ambiguous")
            ids.add(node)
        if isinstance(result, str):
            if result in producers or result in inputs:
                raise ValueError("source_owner_graph_ambiguous")
            producers[result] = (index, row)
        elif result is not None:
            raise ValueError("source_owner_multi_result_unsupported")
    def describe(value: str) -> tuple[str, str, Sequence[int | str], Any, Sequence[str]]:
        if value in broadcast_views:
            operand, dtype, shape, axes = broadcast_views[value]
            return "broadcast", dtype, shape, axes, (operand,)
        if value in load_rows:
            rows = load_rows[value]
            if len(rows) != 1 or rows[0].get("node_ref") is not None:
                raise ValueError("source_owner_load_operand_ambiguous")
            contract = _owner_j_memory_contract(
                function, parameters, boundary, rows[0], value_dimension_relations,
            )
            if contract["coordinates"] != [f"axis{axis}" for axis in range(len(contract["shape"]))]:
                raise ValueError("source_owner_scalar_load_requires_gather")
            loads[value] = (rows[0], contract)
            return "load", contract["dtype"], contract["shape"], contract, ()
        if value in inputs:
            parameter = [row for row in parameters if row.get("logical_ref") == value]
            typ = inputs[value]
            if len(parameter) != 1 or typ.get("shape") != [] or parameter[0].get("kind") != "scalar" or parameter[0].get("mode") != "value" or parameter[0].get("dtype") != typ.get("dtype") or parameter[0].get("alias_of") is not None:
                raise ValueError("source_owner_scalar_abi_invalid")
            return "input", typ["dtype"], (), value, ()
        index, row = producers[value]
        typ, attrs, operands = row.get("type"), row.get("attrs", {}), row.get("operands", [])
        if not isinstance(typ, Mapping) or set(typ) != {"dtype", "shape"} or not isinstance(operands, list):
            raise ValueError("source_owner_type_unsupported")
        if any(operand in producers and producers[operand][0] >= index for operand in operands):
            raise ValueError("source_owner_non_dominating_operand")
        op, payload = row.get("op"), None
        if op == "gather":
            # A physical scalar load may correspond to an explicit J take.
            # Prove the index and occurrence, not just the data tensor name.
            if (
                set(attrs) != {"axis", "mode"} or type(attrs.get("axis")) is not int
                or attrs["axis"] != 0 or attrs["mode"] != "take" or len(operands) != 2
                or typ.get("shape") != [] or operands[0] not in inputs
            ):
                raise ValueError("source_owner_scalar_load_gather_unproved")
            index_producer = producers.get(operands[1])
            if index_producer is None:
                raise ValueError("source_owner_scalar_load_index_producer_missing")
            index_row = index_producer[1]
            index_type, index_attrs = index_row.get("type", {}), index_row.get("attrs", {})
            coordinate = index_attrs.get("value")
            if (
                index_row.get("op") != "const" or index_row.get("operands", []) != []
                or set(index_attrs) != {"value"} or type(coordinate) is not int
                or set(index_type) != {"dtype", "shape"} or index_type["shape"] != []
                or index_type["dtype"] not in {"i32", "i64"}
                or not 0 <= coordinate < 2 ** (int(index_type["dtype"][1:]) - 1)
            ):
                raise ValueError("source_owner_scalar_load_index_contract_unproved")
            matching = []
            for load in load_rows.get(operands[0], ()):
                if load.get("node_ref") is not None:
                    continue
                contract = _owner_j_memory_contract(
                    function, parameters, boundary, load, value_dimension_relations,
                )
                if (
                    contract["coordinates"] == [coordinate]
                    and contract["dtype"] == typ["dtype"]
                    and contract["predicate"] is None and contract["fill"] is None
                ):
                    matching.append((load, contract))
            if len(matching) != 1:
                raise ValueError("source_owner_load_occurrence_ambiguous")
            loads[value] = matching[0]
            return "load", typ["dtype"], (), matching[0][1], ()
        if op == "const" and set(attrs) == {"value"} and not operands:
            payload = owner_float_bits(attrs["value"], typ["dtype"])
        elif op == "reshape" and set(attrs) == {"shape"} and attrs["shape"] == typ["shape"] and len(operands) == 1:
            payload = None
        elif op == "transpose" and set(attrs) == {"perm"} and isinstance(attrs["perm"], list) and len(operands) == 1:
            payload = tuple(attrs["perm"])
        elif op == "cast" and set(attrs) == {"to_dtype"} and attrs["to_dtype"] == typ["dtype"] and len(operands) == 1:
            source_type = inputs.get(operands[0])
            if source_type is None and operands[0] in producers:
                source_type = producers[operands[0]][1].get("type")
            if not isinstance(source_type, Mapping):
                raise ValueError("source_owner_cast_type_unproved")
            payload = owner_cast_kind(source_type.get("dtype"), typ["dtype"])
        elif op == "reduce" and set(attrs) in ({"fn", "axes", "assoc"}, {"fn", "axes", "assoc", "keepdims"}) and isinstance(attrs.get("axes"), list) and len(attrs["axes"]) == 1 and type(attrs["axes"][0]) is int and attrs.get("keepdims", False) is False and attrs.get("assoc") == "float_nondeterministic":
            payload = (attrs["axes"][0], attrs["fn"])
        elif set(attrs) == {"fn"} and ((op == "map_binary" and attrs["fn"] in {"add", "sub", "mul", "div"}) or (op == "map_unary" and attrs["fn"] in {"neg", "exp", "tanh", "rsqrt", "rsqrt_ftz"})):
            op = attrs["fn"]
            projected_operands = []
            for operand in operands:
                operand_type = inputs.get(operand)
                if operand_type is None and operand in producers:
                    operand_type = producers[operand][1].get("type")
                if not isinstance(operand_type, Mapping):
                    raise ValueError("source_owner_operand_type_unproved")
                if operand_type.get("shape") == typ["shape"]:
                    projected_operands.append(operand)
                    continue
                relations = [item for item in boundary.get("broadcasts", ()) if item.get("node_ref") == row.get("id") and item.get("operand") == operand]
                if row.get("op") != "map_binary" or operand_type.get("dtype") != typ["dtype"]:
                    raise ValueError("source_owner_broadcast_type_unproved")
                if not relations:
                    raise ValueError("source_owner_broadcast_relation_missing")
                if len(relations) != 1:
                    raise ValueError("source_owner_broadcast_relation_ambiguous")
                if not isinstance(relations[0].get("axes"), list):
                    raise ValueError("source_owner_broadcast_axis_type_unproved")
                declared_axes = relations[0]["axes"]
                if any(type(axis) is not int for axis in declared_axes):
                    raise ValueError("source_owner_broadcast_axis_type_unproved")
                target_rank = len(typ["shape"])
                normalized_axes = tuple(
                    axis if axis >= 0 else axis + target_rank
                    for axis in declared_axes
                )
                if any(not 0 <= axis < target_rank for axis in normalized_axes):
                    raise ValueError("source_owner_broadcast_axis_range_unproved")
                if len(set(normalized_axes)) != len(normalized_axes):
                    raise ValueError("source_owner_broadcast_axis_duplicate")
                # A proof-only view of the EXISTING broadcast relation. No J
                # operation is authored or inserted into the candidate.
                try:
                    projected, inserted = _broadcast_projection(
                        list(operand_type["shape"]), declared_axes,
                        typ["shape"],
                    )
                except (TypeError, ValueError, NotImplementedError) as exc:
                    raise ValueError("source_owner_broadcast_projection_unproved") from exc
                if projected is None or len(projected) != len(typ["shape"]):
                    raise ValueError("source_owner_broadcast_projection_unproved")
                declared_target_axes = tuple(sorted(normalized_axes))
                if operand_type["shape"] and declared_target_axes != inserted:
                    # Rank-bearing operands must name their inserted axes.
                    # A scalar has the same empty input map with either [] or
                    # declared target axes in the shared J contract.
                    raise ValueError("source_owner_broadcast_target_axes_mismatch")
                # The J relation names broadcast TARGET axes; the owner DAG
                # records where each SOURCE axis lands.  A singleton source
                # expansion is not represented by this exact DAG profile.
                axes = tuple(
                    axis for axis in range(len(typ["shape"]))
                    if axis not in inserted
                )
                if (
                    len(axes) != len(operand_type["shape"])
                    or [typ["shape"][axis] for axis in axes] != operand_type["shape"]
                ):
                    raise ValueError("source_owner_broadcast_shape_unproved")
                key = "@owner_broadcast:" + json.dumps([operand, typ["dtype"], typ["shape"], axes])
                if key in producers or key in inputs:
                    raise ValueError("source_owner_graph_ambiguous")
                broadcast_views[key] = (operand, typ["dtype"], typ["shape"], axes)
                projected_operands.append(key)
            operands = projected_operands
        else:
            if op == "const":
                raise ValueError("source_owner_const_contract_unproved")
            if op == "broadcast_in_dim":
                raise ValueError("source_owner_explicit_broadcast_unsupported")
            if op in {"map_binary", "map_unary"}:
                raise ValueError("source_owner_pointwise_contract_unproved")
            raise ValueError("source_owner_operation_unsupported")
        return op, typ["dtype"], typ["shape"], payload, operands

    claimed_roots = [
        (value, record) for value, record in producers.items()
        if record[1].get("id") == access["node_ref"]
    ]
    if not claimed_roots:
        raise ValueError("source_owner_declared_producer_missing")
    if len(claimed_roots) != 1:
        raise ValueError("source_owner_declared_producer_ambiguous")
    claimed_value = claimed_roots[0][0]
    ordered_roots = (
        claimed_roots[0],
        *((value, record) for value, record in producers.items() if value != claimed_value),
    )
    matches = []
    valid_matches = []
    claimed_failure: str | None = None
    for root_index, (value, (_index, row)) in enumerate(ordered_roots):
        # Relation checks must pass for the declared root before inspecting
        # competing roots. A duplicate content match still rejects even if
        # its own extra boundary relations are missing: no candidate repair
        # or permissive owner selection is performed here.
        if root_index and (matches != [access["node_ref"]] or valid_matches != matches):
            raise ValueError(claimed_failure or "source_owner_producer_mismatch_or_ambiguous")
        try:
            graph = build_tensor_owner_dag(value, describe)
            if graph[-1][1] != store["dtype"]:
                if root_index == 0:
                    claimed_failure = "source_owner_root_dtype_mismatch"
                continue
            if tensor_owner_digest(store, graph) != token.rsplit("_", 1)[1]:
                if root_index == 0:
                    claimed_failure = "source_owner_typed_dag_mismatch"
                continue
            matches.append(row.get("id"))
            # Reconstruct the same postorder solely to identify J reduction IDs.
            postorder: list[str] = []
            seen: set[str] = set()
            def order(current: str) -> None:
                if current in seen:
                    return
                seen.add(current)
                for operand in describe(current)[4]:
                    order(operand)
                postorder.append(current)
            order(value)
            for reduction_index, occurrences in owner_padding_obligations(graph, store=store).items():
                reduction = producers[postorder[reduction_index]][1]
                contracts = [item for item in boundary.get("reductions", ()) if item.get("node_ref") == reduction.get("id")]
                combiner = graph[reduction_index][3][1]
                if len(contracts) != 1 or contracts[0].get("tile_axis") != 0:
                    raise ValueError("source_owner_reduction_boundary_mismatch")
                identity_bits = owner_reduction_identity_bits(
                    contracts[0].get("identity"),
                    graph[reduction_index][1],
                    combiner,
                )
                if identity_bits != owner_reduction_identity_bits(
                    float("-inf") if combiner == "max" else 0.0 if combiner == "sum" else 1.0,
                    graph[reduction_index][1],
                    combiner,
                ):
                    raise ValueError("source_owner_reduction_boundary_mismatch")
                for occurrence in occurrences:
                    load = next(row for row, contract in loads.values() if contract["occurrence"] == occurrence)
                    load_ref = load.get("access_id", load.get("access_label"))
                    extensions = [item for item in boundary.get("consumer_extensions", ()) if item.get("consumer_ref") == reduction.get("id") and item.get("access_ref") == load_ref and item.get("operand_index") == 0]
                    if (
                        len(extensions) != 1
                        or extensions[0].get("tile_axis") != 0
                        or extensions[0].get("identity_law") != "two_sided"
                        or extensions[0].get("outside", {}).get("dtype") != graph[reduction_index][1]
                        or owner_reduction_identity_bits(
                            extensions[0]["outside"].get("value"),
                            graph[reduction_index][1],
                            combiner,
                        ) != identity_bits
                    ):
                        raise ValueError("source_owner_outside_relation_unproved")
            valid_matches.append(row.get("id"))
        except (ValueError, TypeError, KeyError, StopIteration, RecursionError) as exc:
            if root_index == 0:
                claimed_failure = (
                    exc.args[0]
                    if type(exc) is ValueError and len(exc.args) == 1
                    and isinstance(exc.args[0], str)
                    and re.fullmatch(r"source_owner_[a-z0-9_]{1,80}", exc.args[0])
                    else "source_owner_declared_root_unproved"
                )
            continue
    if matches != [access["node_ref"]] or valid_matches != matches:
        if not valid_matches and claimed_failure is not None:
            raise ValueError(claimed_failure)
        raise ValueError("source_owner_duplicate_producer_match")


@dataclass(frozen=True)
class SourceSemanticOccurrence:
    """One opaque Source semantic anchor eligible for compiler attestation.

    ``token`` is the only provider-visible identity.  The source span exists
    solely in the in-memory lexical ledger so the exact Source compilation can
    attest a counterpart through its debug location sidecar; it is never
    serialized.  An occurrence without a compiler counterpart is still a
    useful Source fact, but it must not authorize a provider-local owner or
    computed guard.
    """

    token: str
    role: str
    operation_family: str
    source_line: int | None = field(default=None, repr=False, compare=False)
    source_column: int | None = field(default=None, repr=False, compare=False)
    source_end_line: int | None = field(default=None, repr=False, compare=False)
    source_end_column: int | None = field(default=None, repr=False, compare=False)
    source_expression_key: str | None = field(default=None, repr=False, compare=False)
    binding_mode: str = field(default="direct", repr=False, compare=False)
    source_call_path: tuple[tuple[int, int], ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token, str)
            or re.fullmatch(r"so_[0-9a-f]{32}", self.token) is None
            or self.role not in {"access", "consumer", "predicate"}
            or self.operation_family not in {
                "map_binary", "map_unary", "select", "cast", "reduce",
                "iota", "reshape", "transpose", "concat", "flip", "sort",
                "memory_load", "memory_store", "memory_update",
            }
            or self.binding_mode not in {"direct", "exact_local_binding"}
        ):
            raise ValueError("invalid Source semantic occurrence")
        for value, name in (
            (self.source_line, "source line"),
            (self.source_column, "source column"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"invalid Source semantic occurrence {name}")
        if (self.source_line is None) != (self.source_column is None):
            raise ValueError("Source semantic occurrence span is incomplete")
        if (self.source_end_line is None) != (self.source_end_column is None):
            raise ValueError("Source semantic occurrence end span is incomplete")
        if self.source_end_line is not None:
            if (
                self.source_line is None
                or type(self.source_end_line) is not int
                or type(self.source_end_column) is not int
                or (self.source_end_line, self.source_end_column)
                <= (self.source_line, self.source_column)
            ):
                raise ValueError("Source semantic occurrence end span is invalid")
        if self.source_expression_key is not None and (
            not isinstance(self.source_expression_key, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.source_expression_key) is None
        ):
            raise ValueError("Source semantic occurrence expression key is invalid")
        if not isinstance(self.source_call_path, tuple) or any(
            not isinstance(frame, tuple) or len(frame) != 2
            or type(frame[0]) is not int or frame[0] <= 0
            or type(frame[1]) is not int or frame[1] < 0
            for frame in self.source_call_path
        ):
            raise ValueError("Source semantic occurrence call path is invalid")

    def to_json(self) -> dict[str, str]:
        return {
            "token": self.token,
            "role": self.role,
            "operation_family": self.operation_family,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SourceSemanticOccurrence":
        copied = _copy_mapping(value, "SourceSemanticOccurrence")
        if set(copied) != {"token", "role", "operation_family"}:
            raise ValueError("Source semantic occurrence wire has an incomplete shape")
        return cls(
            token=copied["token"],
            role=copied["role"],
            operation_family=copied["operation_family"],
        )


SOURCE_CONSTEXPR_UNKNOWN = object()


def source_constexpr_value(node: ast.AST, values: Mapping[str, Any]) -> Any:
    """Evaluate only closed scalar expressions over frozen constexpr facts."""
    unknown = SOURCE_CONSTEXPR_UNKNOWN
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) in {bool, int, float, str, type(None)} else unknown
    if isinstance(node, ast.Name):
        value = values.get(node.id, unknown)
        return value if type(value) in {bool, int, float, str, type(None)} else unknown
    try:
        if isinstance(node, ast.UnaryOp):
            value = source_constexpr_value(node.operand, values)
            op = {ast.Not: scalar_operator.not_, ast.USub: scalar_operator.neg, ast.UAdd: scalar_operator.pos, ast.Invert: scalar_operator.invert}.get(type(node.op))
            return op(value) if op is not None and value is not unknown else unknown
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = source_constexpr_value(item, values)
                if value is unknown:
                    return unknown
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.BinOp):
            left, right = source_constexpr_value(node.left, values), source_constexpr_value(node.right, values)
            if type(left) not in {bool, int, float} or type(right) not in {bool, int, float}:
                return unknown
            op = {ast.Add: scalar_operator.add, ast.Sub: scalar_operator.sub, ast.Mult: scalar_operator.mul, ast.FloorDiv: scalar_operator.floordiv, ast.Mod: scalar_operator.mod, ast.BitAnd: scalar_operator.and_, ast.BitOr: scalar_operator.or_, ast.BitXor: scalar_operator.xor}.get(type(node.op))
            return op(left, right) if op is not None else unknown
        if isinstance(node, ast.Compare):
            left = source_constexpr_value(node.left, values)
            for operation, item in zip(node.ops, node.comparators, strict=True):
                right = source_constexpr_value(item, values)
                op = {ast.Eq: scalar_operator.eq, ast.NotEq: scalar_operator.ne, ast.Lt: scalar_operator.lt, ast.LtE: scalar_operator.le, ast.Gt: scalar_operator.gt, ast.GtE: scalar_operator.ge}.get(type(operation))
                if left is unknown or right is unknown or op is None:
                    return unknown
                if not op(left, right):
                    return False
                left = right
            return True
    except (TypeError, ValueError, ArithmeticError):
        return unknown
    return unknown


def source_specialized_ast(source: str, facts: "SourceFacts | None") -> ast.Module:
    """Build a proof-only view of reachable, frozen Source specializations.

    Only declared constexpr arguments select branches. Each direct helper call
    has an independent view and an exact call-site path; Source code and its
    coordinates are not rewritten. Recursive/dynamic calls cannot be proved
    by this collector and retain the conservative unspecialized view.
    """
    tree = ast.parse(source)
    if facts is None:
        return tree
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    module_bindings: dict[str, int] = {}

    class ModuleBindings(ast.NodeVisitor):
        def bind(self, name: str) -> None:
            module_bindings[name] = module_bindings.get(name, 0) + 1

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                self.bind(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.bind(node.name)

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                self.bind(alias.asname or alias.name.split('.')[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                self.bind(alias.asname or alias.name)

    ModuleBindings().visit(tree)
    if module_bindings.get(facts.source_symbol) != 1 or '*' in module_bindings:
        return tree
    if facts.source_symbol not in functions:
        return tree
    frozen = {**facts.compiler_facts["bound_constexpr_values"], **facts.compiler_facts["schedule_values"]}
    entry_values = {
        name: frozen[opaque] for name, opaque in facts.source_to_opaque
        if opaque in facts.constexpr_parameters and opaque in frozen
    }
    projected: list[ast.FunctionDef] = []

    def project(name: str, values: Mapping[str, Any], path: tuple[tuple[int, int], ...], stack: tuple[str, ...]) -> None:
        if module_bindings.get(name) != 1:
            raise ValueError("source_helper_binding_shadowed")
        if name in stack:
            raise ValueError("source_recursive_helper_unproved")
        function = copy.deepcopy(functions[name])
        if function.args.vararg is not None or function.args.kwarg is not None:
            raise ValueError("source_helper_variadic_binding_unproved")
        writes = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))}
        writes.update(node.name for node in ast.walk(function) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not function)
        for node in ast.walk(function):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                writes.update(alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name) for alias in node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                writes.add(node.name)
        environment = {key: value for key, value in values.items() if key not in writes}

        class Select(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                return node

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
                return node

            def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
                selected = source_constexpr_value(node.test, environment)
                if selected is SOURCE_CONSTEXPR_UNKNOWN:
                    return self.generic_visit(node)
                result: list[ast.stmt] = []
                for statement in node.body if bool(selected) else node.orelse:
                    rewritten = self.visit(statement)
                    result.extend(rewritten if isinstance(rewritten, list) else [rewritten])
                return result

        body: list[ast.stmt] = []
        selector = Select()
        for statement in function.body:
            selected = selector.visit(statement)
            body.extend(selected if isinstance(selected, list) else [selected])
        function.body = body
        function._source_call_path = path
        projected.append(function)

        class Calls(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

            def visit_Call(self, node: ast.Call) -> None:
                self.generic_visit(node)
                if not isinstance(node.func, ast.Name) or node.func.id not in functions:
                    return
                callee_name = node.func.id
                parameters = (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
                if callee_name in writes or any(arg.arg == callee_name for arg in parameters):
                    raise ValueError("source_helper_binding_shadowed")
                callee = functions[callee_name]
                positional = (*callee.args.posonlyargs, *callee.args.args)
                if any(isinstance(arg, ast.Starred) for arg in node.args) or len(node.args) > len(positional) or any(kw.arg is None for kw in node.keywords):
                    raise ValueError("source_helper_call_binding_unproved")
                actuals = dict(zip((arg.arg for arg in positional), node.args))
                keyword_names = {arg.arg for arg in (*callee.args.args, *callee.args.kwonlyargs)}
                for keyword in node.keywords:
                    if keyword.arg in actuals or keyword.arg not in keyword_names:
                        raise ValueError("source_helper_call_binding_unproved")
                    actuals[keyword.arg] = keyword.value
                defaults = dict(zip((arg.arg for arg in positional[len(positional) - len(callee.args.defaults):]), callee.args.defaults))
                defaults.update((arg.arg, value) for arg, value in zip(callee.args.kwonlyargs, callee.args.kw_defaults, strict=True) if value is not None)
                if any(formal.arg not in actuals and formal.arg not in defaults for formal in (*positional, *callee.args.kwonlyargs)):
                    raise ValueError("source_helper_call_binding_unproved")
                bound = {}
                for formal in (*positional, *callee.args.kwonlyargs):
                    annotation = formal.annotation
                    if not isinstance(annotation, ast.Attribute) or annotation.attr != "constexpr":
                        continue
                    actual = actuals.get(formal.arg)
                    if actual is not None:
                        value = source_constexpr_value(actual, environment)
                    elif formal.arg in defaults:
                        value = source_constexpr_value(defaults[formal.arg], {})
                    else:
                        value = SOURCE_CONSTEXPR_UNKNOWN
                    if value is not SOURCE_CONSTEXPR_UNKNOWN:
                        bound[formal.arg] = value
                # The pinned Triton visitor evaluates keyword arguments first
                # and positional arguments last; the final visited child owns
                # the emitted tt.call location, just as for memory builtins.
                anchor = node.args[-1] if node.args else node.keywords[-1] if node.keywords else node.func
                project(callee_name, bound, (*path, (anchor.lineno, anchor.col_offset)), (*stack, name))

        visitor = Calls()
        for statement in body:
            visitor.visit(statement)

    try:
        project(facts.source_symbol, entry_values, (), ())
    except (ValueError, RecursionError):
        return tree
    tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef)] + projected
    return tree


def _source_semantic_operation_family(node: ast.AST) -> str | None:
    """Return a J family only for one explicit Source expression node.

    This is deliberately narrower than the provider's operation vocabulary.
    A missing or unsupported family suppresses authorization issuance; it must
    never be approximated from a TTIR opcode, Source order, tensor name, or
    candidate-local label.
    """

    if isinstance(node, ast.BinOp | ast.BoolOp | ast.Compare | ast.AugAssign):
        return "map_binary"
    if isinstance(node, ast.UnaryOp):
        return "map_unary"
    if isinstance(node, ast.IfExp):
        return "select"
    if not isinstance(node, ast.Call):
        return None
    leaf = (
        node.func.attr if isinstance(node.func, ast.Attribute)
        else node.func.id if isinstance(node.func, ast.Name)
        else None
    )
    return {
        "sigmoid": "map_unary", "exp": "map_unary", "log": "map_unary",
        "sqrt": "map_unary", "rsqrt": "map_unary", "abs": "map_unary",
        "sin": "map_unary", "cos": "map_unary", "tanh": "map_unary",
        "maximum": "map_binary", "minimum": "map_binary", "pow": "map_binary",
        "fma": "map_binary", "where": "select", "to": "cast",
        "cast": "cast", "astype": "cast", "float": "cast", "int": "cast",
        "sum": "reduce", "mean": "reduce", "max": "reduce", "min": "reduce",
        "prod": "reduce", "any": "reduce", "all": "reduce", "arange": "iota",
        "reshape": "reshape", "view": "reshape", "trans": "transpose",
        "transpose": "transpose", "join": "concat", "cat": "concat",
        "concat": "concat", "flip": "flip", "sort": "sort",
    }.get(leaf)


def _opaque_source_semantic_occurrence(
    node: ast.AST | None,
    *,
    role: str,
    scope: Sequence[str],
    source_identity: str,
    operation_family: str | None = None,
    binding_mode: str = "direct",
    call_path: tuple[tuple[int, int], ...] = (),
    expression_key: str | None = None,
) -> SourceSemanticOccurrence | None:
    """Create one non-textual Source anchor, or none when not provable."""

    if node is None:
        return None
    family = operation_family or _source_semantic_operation_family(node)
    line = getattr(node, "lineno", None)
    column = getattr(node, "col_offset", None)
    end_line = getattr(node, "end_lineno", None)
    end_column = getattr(node, "end_col_offset", None)
    if (
        family is None
        or not isinstance(line, int)
        or isinstance(line, bool)
        or line <= 0
        or not isinstance(column, int)
        or isinstance(column, bool)
        or column < 0
    ):
        return None
    # The preimage remains host-local.  A token is stable only for the mounted
    # Source/facts lineage and does not reveal source text, identifiers, spans,
    # or the lexical function name to the provider.
    material = json.dumps(
        {
            "source": source_identity,
            "role": role,
            "scope": list(scope),
            "family": family,
            "line": line,
            "column": column,
            "call_path": call_path,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return SourceSemanticOccurrence(
        token="so_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32],
        role=role,
        operation_family=family,
        source_line=line,
        source_column=column,
        source_end_line=end_line,
        source_end_column=end_column,
        source_expression_key=expression_key,
        binding_mode=binding_mode,
        source_call_path=call_path,
    )

# The boundary is part of the same construction transaction as operations,
# outputs, scopes, and dimensions.  Keep this declaration here (rather than
# re-declaring it in a consumer) so the raw boundary wire has one structural
# projection before any typed consumer interprets it.
CANONICAL_BOUNDARY_WIRE_FIELDS = (
    "domains",
    "accesses",
    "broadcasts",
    "aliases",
    "effects",
    "output_mappings",
    "reductions",
    "index_bounds",
    "updates",
    "result_relations",
    "realization_authorizations",
    "consumer_extensions",
    "access_guard_relations",
)


def canonical_artifact_envelope_error(value: Any) -> str | None:
    """Return the one transport-level error for a candidate artifact root.

    This deliberately establishes only the ownership boundary shared by every
    Lift transport.  It does not decode J/C/pi and it never repairs, moves, or
    supplies fields. Semantic validity remains the responsibility of the
    canonical decoder and CandidateCheck. Keeping the small guard here means
    a direct Responses reply and an MCP submission cannot disagree about what
    one candidate document is allowed to replace or enter for construction.
    """

    if not isinstance(value, Mapping):
        return "candidate root must be one JSON object"
    expected = set(CANONICAL_ARTIFACT_ROOT_FIELDS)
    actual = set(value)
    if actual != expected:
        expected_text = ", ".join(CANONICAL_ARTIFACT_ROOT_FIELDS)
        actual_text = ", ".join(sorted(str(key) for key in actual))
        return (
            "candidate root must contain exactly the canonical fields "
            f"{{{expected_text}}}; observed {{{actual_text}}}"
        )
    for field_name in ("j", "c", "pi"):
        if not isinstance(value.get(field_name), Mapping):
            return f"candidate root field {field_name!r} must be an object"
    return None


def require_canonical_artifact_envelope(value: Any) -> None:
    """Raise before a transport accepts a non-canonical candidate root."""

    error = canonical_artifact_envelope_error(value)
    if error is not None:
        raise ValueError(error)


# These are the actual fields emitted by the three shared construction
# records below. Keep this declaration next to the record builders so schema
# and MCP projections cannot substitute an unrelated IntentOp leaf (for
# example, map_binary) as the construction-record contract.
CANONICAL_CONSTRUCTION_RECORD_FIELDS: Mapping[str, tuple[str, ...]] = {
    "operation": (
        "path", "id", "op", "parent_id", "lexical_role", "depth",
        "region_scope_id", "is_final", "operands", "results", "type",
        "attrs", "wire",
    ),
    "output": (
        "output", "abi_index", "output_type", "return_value",
        "abi_parameter_index", "abi_parameter_name", "abi_parameter_mode",
        "mapping_index", "mapping_value", "authored_node_ref", "node_ref",
        "producer_record", "return_index",
    ),
    "scope": (
        "scope_id", "kind", "parent_scope", "definitions", "captures",
        "effects", "operation_paths", "operation_ids", "operation_results",
    ),
    "dimension": (
        "result", "expr",
    ),
    "value_dimension": (
        "value_ref", "dimension_ref", "relation",
    ),
    "boundary": CANONICAL_BOUNDARY_WIRE_FIELDS,
}

# These fields are added only after the canonical J type elaborator has closed
# every SSA value.  Keeping them separate from the raw-wire fields lets the
# ingress repair phase remain structural while ensuring the accepted bundle
# has one typed operation contract for all consumers.
CANONICAL_ELABORATED_OPERATION_RECORD_FIELDS = (
    *CANONICAL_CONSTRUCTION_RECORD_FIELDS["operation"],
    "operand_types", "result_types",
)


def _function_wire(function: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the one internal IntentFunction wire shape.

    ``IntentFunction.to_json()`` intentionally emits an ``intent_function``
    wrapper, while raw CandidateCheck payloads and a few ingress helpers carry
    the inner object directly.  Construction records must not silently read
    different levels of those two equivalent wires, so every shared builder
    normalizes the envelope before looking at inputs/body/outputs/returns.
    """

    if not isinstance(function, Mapping):
        return None
    nested = function.get("intent_function")
    if nested is not None:
        if set(function) != {"intent_function"} or not isinstance(nested, Mapping):
            return None
        return nested
    return function


def _operation_record_walk(
    body: Any,
    *,
    prefix: str = "body",
    parent_id: str | None = None,
    lexical_role: str = "function",
    depth: int = 0,
    region_scope_id: str | None = None,
) -> list[
    tuple[str, Mapping[str, Any], str | None, str, int, str | None, bool]
    ] | None:
    """Walk one authored operation wire in canonical lexical order.

    This is deliberately a structural walker.  It does not import or execute
    TypeCheck/Gate1 contracts, and it does not infer a missing field.  Keeping
    it here makes the operation *record* shared without making semantic proof
    part of the construction layer.
    """

    if not isinstance(body, list):
        return None
    rows: list[
        tuple[str, Mapping[str, Any], str | None, str, int, str | None, bool]
    ] = []
    for index, raw in enumerate(body):
        if not isinstance(raw, Mapping):
            return None
        path = f"{prefix}[{index}]"
        rows.append(
            (
                path,
                raw,
                parent_id,
                lexical_role,
                depth,
                region_scope_id,
                index == len(body) - 1,
            )
        )
        nested = raw.get("body", [])
        nested_scope = (
            raw.get("scope_id")
            if raw.get("op") == "loop" and isinstance(raw.get("scope_id"), str)
            else region_scope_id
        )
        nested_rows = _operation_record_walk(
            nested,
            prefix=f"{path}.body",
            parent_id=raw.get("id") if isinstance(raw.get("id"), str) else None,
            lexical_role="body",
            depth=depth + 1,
            region_scope_id=nested_scope,
        )
        if nested_rows is None:
            return None
        rows.extend(nested_rows)
        if raw.get("op") in {"effect_if", "value_if"}:
            for arm_name in ("positive", "negative"):
                arm = raw.get(arm_name)
                if arm is None:
                    continue
                if not isinstance(arm, Mapping):
                    return None
                arm_rows = _operation_record_walk(
                    arm.get("body", []),
                    prefix=f"{path}.{arm_name}.body",
                    parent_id=raw.get("id") if isinstance(raw.get("id"), str) else None,
                    lexical_role=arm_name,
                    depth=depth + 1,
                    region_scope_id=(
                        arm.get("scope_id")
                        if isinstance(arm.get("scope_id"), str)
                        else region_scope_id
                    ),
                )
                if arm_rows is None:
                    return None
                rows.extend(arm_rows)
    return rows


def _canonical_operation_wire(operation: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize equivalent optional-field spellings into one operation wire.

    Typed ``IntentOp`` serialization omits empty optional fields.  Normalize
    those spellings once here so raw construction records and typed operation
    views cannot disagree about an otherwise identical operation.  This does
    not infer a semantic field or repair an incomplete operation.
    """

    if not isinstance(operation, Mapping):
        return None
    result = dict(operation)
    body = result.get("body")
    if isinstance(body, list):
        normalized_body: list[dict[str, Any]] = []
        for child in body:
            normalized = _canonical_operation_wire(child)
            if normalized is None:
                return None
            normalized_body.append(normalized)
        result["body"] = normalized_body
    elif body is not None:
        return None
    for arm_name in ("positive", "negative"):
        arm = result.get(arm_name)
        if arm is None:
            continue
        if not isinstance(arm, Mapping):
            return None
        normalized_arm = dict(arm)
        arm_body = normalized_arm.get("body")
        if isinstance(arm_body, list):
            normalized_body = []
            for child in arm_body:
                normalized = _canonical_operation_wire(child)
                if normalized is None:
                    return None
                normalized_body.append(normalized)
            normalized_arm["body"] = normalized_body
        elif arm_body is not None:
            return None
        result[arm_name] = normalized_arm

    op_name = result.get("op")
    if not isinstance(op_name, str):
        return None
    for field_name in ("attrs", "operands", "body", "yield"):
        if field_name in result and result[field_name] in (None, [], {}):
            # Loop body/yield are required structural fields even when empty.
            if op_name == "loop" and field_name in {"body", "yield"}:
                continue
            result.pop(field_name, None)
    for field_name in ("type", "scope_id", "predicate", "positive", "negative"):
        if field_name in result and result[field_name] is None:
            result.pop(field_name, None)
    if op_name != "loop" and result.get("scan") == []:
        result.pop("scan", None)
    return result


def canonical_operation_records(
    function: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    """Build the transport-neutral operation construction table.

    The returned rows are the sole structural interpretation shared by Lift
    ingress, canonical J elaboration, Replay presentation, and TypeCheck's
    explicit type surface.  ``wire`` is the complete authored operation, so
    structured fields (loop carry/body/yield and conditional arms) cannot be
    silently lost when a consumer reads the compact columns.

    This helper only checks recordability: operation contracts remain owned by
    their existing semantic validators.  ``None`` means no safe shared table
    exists; callers must stay diagnostic/fail-closed and may not repair from a
    partial table.
    """

    function = _function_wire(function)
    if function is None:
        return None
    rows = _operation_record_walk(function.get("body"))
    if rows is None:
        return None
    raw_inputs = function.get("inputs")
    if not isinstance(raw_inputs, list):
        return None
    input_names = {
        item.get("name")
        for item in raw_inputs
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(input_names) != len(raw_inputs) or any(
        re.fullmatch(r"t[0-9]+", name) is None for name in input_names
    ):
        return None
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_results: set[str] = set()
    for (
        path,
        operation,
        parent_id,
        lexical_role,
        depth,
        region_scope_id,
        is_final,
    ) in rows:
        canonical_wire = _canonical_operation_wire(operation)
        if canonical_wire is None:
            return None
        operation_id = canonical_wire.get("id")
        operation_name = canonical_wire.get("op")
        if (
            not isinstance(operation_id, str)
            or not isinstance(operation_name, str)
            or re.fullmatch(r"n[0-9]+", operation_id) is None
            or operation_id in seen_ids
            or (
                parent_id is not None
                and parent_id not in seen_ids
            )
        ):
            return None
        raw_operands = canonical_wire.get("operands", [])
        if not isinstance(raw_operands, list) or any(
            not isinstance(item, str) for item in raw_operands
        ):
            return None
        raw_result = canonical_wire.get("result")
        if isinstance(raw_result, str):
            results = (raw_result,)
        elif isinstance(raw_result, list) and all(
            isinstance(item, str) for item in raw_result
        ):
            results = tuple(raw_result)
        elif raw_result is None:
            results = ()
        else:
            return None
        if any(re.fullmatch(r"v[0-9]+", item) is None for item in results):
            return None
        if (
            len(results) != len(set(results))
            or seen_results.intersection(results)
            or input_names.intersection(results)
        ):
            return None
        attrs = canonical_wire.get("attrs")
        if attrs is not None and not isinstance(attrs, Mapping):
            return None
        declared_type = canonical_wire.get("type")
        if declared_type is not None and not isinstance(declared_type, Mapping):
            return None
        seen_ids.add(operation_id)
        seen_results.update(results)
        records.append(
            {
                "path": path,
                "id": operation_id,
                "op": operation_name,
                "parent_id": parent_id,
                "lexical_role": lexical_role,
                "depth": depth,
                "region_scope_id": region_scope_id,
                "is_final": is_final,
                "operands": tuple(raw_operands),
                "results": results,
                "type": json.loads(_stable_json(declared_type)) if declared_type is not None else None,
                "attrs": json.loads(_stable_json(attrs)) if attrs is not None else None,
                "wire": json.loads(_stable_json(canonical_wire)),
            }
        )

    # Close the cross-field SSA relationships that the structural schema
    # cannot express by itself.  In particular, a loop's positional result
    # and yield arrays are witnesses for the carry roles, and every next_value
    # must be produced by a direct body operation.  Keeping this check on the
    # shared operation table prevents Gate1, Replay, and ingress repair from
    # each inventing a different loop interpretation.
    all_results = {
        result
        for record in records
        for result in record["results"]
    }
    occupied_local_bindings = set(input_names) | set(all_results)
    for record in records:
        if record["op"] != "loop":
            continue
        wire = record["wire"]
        carries = wire.get("carry")
        result_values = wire.get("result")
        yielded = wire.get("yield")
        if not isinstance(carries, list) or not isinstance(result_values, list):
            return None
        if not isinstance(yielded, list):
            return None
        expected_results = [
            state.get("result_value")
            for state in carries
            if isinstance(state, Mapping)
        ]
        expected_yields = [
            state.get("next_value")
            for state in carries
            if isinstance(state, Mapping)
        ]
        if (
            len(expected_results) != len(carries)
            or len(expected_yields) != len(carries)
            or result_values != expected_results
            or yielded != expected_yields
        ):
            return None
        role_names: list[str] = []
        local_bindings: list[str] = []
        extra_bindings: list[str] = []
        for state in carries:
            if not isinstance(state, Mapping):
                return None
            roles = [
                state.get("body_argument"),
                state.get("next_value"),
                state.get("result_value"),
            ]
            if any(not isinstance(name, str) for name in roles):
                return None
            role_names.extend(roles)
            local_bindings.append(roles[0])
            if roles[2] not in record["results"]:
                return None
        scan = wire.get("scan")
        if not isinstance(scan, list):
            return None
        for item in scan:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                return None
            local_bindings.append(item["name"])
            extra_bindings.append(item["name"])
        trip = wire.get("trip")
        if isinstance(trip, Mapping) and "index" in trip:
            if not isinstance(trip.get("index"), str):
                return None
            local_bindings.append(trip["index"])
            extra_bindings.append(trip["index"])
        if len(role_names) != len(set(role_names)):
            return None
        if len(local_bindings) != len(set(local_bindings)):
            return None
        if set(extra_bindings) & set(role_names):
            return None
        if set(local_bindings) & set(occupied_local_bindings):
            return None
        direct_body_path = record["path"] + ".body"
        direct_body_results = {
            result
            for child in records
            if isinstance(child.get("path"), str)
            and re.fullmatch(
                re.escape(direct_body_path) + r"\[\d+\]",
                child["path"],
            )
            for result in child["results"]
        }
        if any(
            state.get("next_value") not in direct_body_results
            for state in carries
        ):
            return None
        occupied_local_bindings.update(local_bindings)
    return tuple(records)


def canonical_operation_wire_rows(
    function: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...] | None:
    """Return canonical lexical paths and the authored operation mappings.

    This is the mutation-capable companion to ``canonical_operation_records``
    for the bounded ingress allocator. It uses the same traversal and
    envelope normalization, but preserves references to the private candidate
    copy so a caller can rename already-authored handles transactionally. It
    never creates an operation or returns rows for an unrecordable body.
    """

    function = _function_wire(function)
    if function is None:
        return None
    rows = _operation_record_walk(function.get("body"))
    if rows is None:
        return None
    return tuple((row[0], row[1]) for row in rows)


def semantic_operation_wire_rows(
    function: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...] | None:
    """Walk the provider semantic operation tree before canonical renaming.

    This is intentionally distinct from :func:`canonical_operation_records`:
    the host allocator needs lexical paths and authored operation mappings
    before ``n*``/``v*`` identities exist.  It validates only the region/body
    wire shape and never accepts, repairs, or interprets operation semantics.
    The strict canonical record builder remains the acceptance authority after
    the allocator has completed.
    """

    if not isinstance(function, Mapping):
        return None
    normalized = _function_wire(function)
    if normalized is None:
        return None
    rows = _operation_record_walk(normalized.get("body"))
    if rows is None:
        return None
    return tuple((row[0], row[1]) for row in rows)


def canonical_authored_handle_declarations(
    j: Mapping[str, Any],
) -> dict[str, list[str]] | None:
    """Collect candidate declarations plus the referenced Source x* view.

    This is the declaration half of :class:`CanonicalHandleAllocator`.  It is
    intentionally structural: it observes a complete authored J and never
    invents a missing producer, scope, relation, or loop field.  The returned
    rows are consumed by the allocator's dense-renaming pass; semantic
    validity remains owned by the strict J elaborator/CandidateCheck.

    ``x*`` is collected only to let callers reject a candidate that references
    a dimension outside the Source allocator; it is never candidate-owned or
    renumbered.  Keeping this walk beside the canonical operation-record walker is
    important.  A relift ingress must not maintain a second private traversal
    for ``v*/x*/d*/n*/e*/q*/b*/s*/u*`` and then silently disagree with Replay or
    Gate1 when a nested region is added.
    """

    if not isinstance(j, Mapping):
        return None
    intent = j.get("intent")
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, Mapping):
        return None
    function = _function_wire(function)
    if function is None:
        return None

    ordered: dict[str, list[str]] = {
        prefix: [] for prefix in (*CANONICAL_AUTHORED_HANDLE_PREFIXES, "x")
    }
    seen: dict[str, set[str]] = {
        prefix: set() for prefix in CANONICAL_AUTHORED_HANDLE_PREFIXES
    }
    duplicate: set[str] = set()

    def add(prefix: str, value: Any) -> None:
        pattern = CANONICAL_HANDLE_NAMESPACE_PATTERNS[prefix]
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            return
        if value in seen[prefix]:
            duplicate.add(value)
            return
        seen[prefix].add(value)
        ordered[prefix].append(value)

    def add_declared_handle(value: Any, prefixes: Sequence[str]) -> None:
        """Add a typed handle without treating an unrelated string as one."""

        if not isinstance(value, str):
            return
        for prefix in prefixes:
            if re.fullmatch(CANONICAL_HANDLE_NAMESPACE_PATTERNS[prefix], value):
                add(prefix, value)
                return

    scopes = j.get("scopes")
    if isinstance(scopes, list):
        for scope in scopes:
            if isinstance(scope, Mapping):
                add("q", scope.get("scope_id"))

    boundary = j.get("boundary")
    if isinstance(boundary, Mapping):
        domains = boundary.get("domains")
        if isinstance(domains, Mapping):
            for domain in domains:
                add("d", domain)
        for field_name, prefix, id_field in (
            ("accesses", "a", "access_id"),
            ("effects", "e", "effect_id"),
            ("realization_authorizations", "u", "auth_id"),
        ):
            values = boundary.get(field_name)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, Mapping):
                        add(prefix, item.get(id_field))

    storage = j.get("storage")
    if isinstance(storage, list):
        for item in storage:
            if isinstance(item, Mapping):
                add_declared_handle(item.get("storage_id"), ("b", "s"))

    operation_rows = semantic_operation_wire_rows(function)
    if operation_rows is None:
        return None
    # Operation identity is owned by the lexical operation table, not by the
    # provider's spelling of ``id``.  A frequent malformed candidate uses the
    # result handle (v*) as the operation id (often with the same text in
    # ``result``).  Counting only literal n* ids made the shared allocator see
    # an empty node namespace, so the canonical n* rewrite never ran and the
    # candidate reached Decode with two incompatible operation identities.
    #
    # The row order and unique authored ids are already explicit here.  For a
    # unique authored table, assigning the dense n* declaration sequence is a
    # structural rename, not semantic inference; the authoring-layout pass
    # below maps each old id to its exact lexical path and rewrites all known
    # node references transactionally.  Duplicate/missing ids remain
    # fail-closed and are not turned into a guessed graph.
    operation_ids = [
        operation.get("id")
        for _path, operation in operation_rows
    ]
    if (
        any(
            not isinstance(operation_id, str)
            or re.fullmatch(r"n[0-9]+", operation_id) is None
            for operation_id in operation_ids
        )
        or len(operation_ids) != len(set(operation_ids))
    ):
        return None
    # Preserve the authored lexical IDs here.  ``for_candidate_j`` derives
    # the dense allocator declaration from this table, and the ingress repair
    # then maps (for example) n3/n7 to n0/n1 while rewriting every node_ref.
    # Replacing the table with n0...nN at this point loses the old names and
    # makes the supposedly shared allocator unable to perform that repair.
    ordered["n"] = list(operation_ids)
    for _path, operation in operation_rows:
        result = operation.get("result")
        if isinstance(result, list):
            for value in result:
                add("v", value)
        else:
            add("v", result)
        if operation.get("op") == "loop":
            trip = operation.get("trip")
            if isinstance(trip, Mapping):
                add("v", trip.get("index"))
            scan = operation.get("scan")
            if isinstance(scan, list):
                for item in scan:
                    if isinstance(item, Mapping):
                        add("v", item.get("name"))
            carry = operation.get("carry")
            if isinstance(carry, list):
                for item in carry:
                    if isinstance(item, Mapping):
                        add("s", item.get("state_id"))
                        add("v", item.get("body_argument"))
                        add("v", item.get("next_value"))
                        add("v", item.get("result_value"))

    # Shape and relation dimensions share one first-appearance x* namespace.
    dimensions: list[str] = []
    dimension_seen: set[str] = set()

    def add_dimension(value: Any) -> None:
        if (
            not isinstance(value, str)
            or re.fullmatch(CANONICAL_HANDLE_NAMESPACE_PATTERNS["x"], value)
            is None
        ):
            return
        if value not in dimension_seen:
            dimension_seen.add(value)
            dimensions.append(value)

    for collection in ("inputs", "outputs"):
        values = function.get(collection)
        if isinstance(values, list):
            for tensor in values:
                if not isinstance(tensor, Mapping):
                    continue
                shape = tensor.get("shape")
                if isinstance(shape, list):
                    for dimension in shape:
                        add_dimension(dimension)
    signature = j.get("signature")
    parameters = signature.get("parameters") if isinstance(signature, Mapping) else None
    if isinstance(parameters, list):
        for parameter in parameters:
            if isinstance(parameter, Mapping):
                add_dimension(parameter.get("logical_ref"))
    relations = j.get("dimension_relations")
    if isinstance(relations, list):
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            add_dimension(relation.get("result"))
            expression = relation.get("expr")
            if isinstance(expression, Mapping):
                add_dimension(expression.get("arg"))
                add_dimension(expression.get("rhs"))
    value_relations = j.get("value_dimension_relations")
    if isinstance(value_relations, list):
        for relation in value_relations:
            if isinstance(relation, Mapping):
                add_dimension(relation.get("dimension_ref"))

    # Dimension handles can also occur in a typed operation shape or a
    # gather/iota extent.  They are still declarations of the same x*
    # namespace, not a second operation-local namespace.  Only exact x*
    # tokens are collected; no shape or operation is invented here.
    def collect_dimension_tokens(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                collect_dimension_tokens(item)
        elif isinstance(value, list):
            for item in value:
                collect_dimension_tokens(item)
        elif isinstance(value, str):
            for match in re.finditer(
                r"(?<![A-Za-z0-9_])x[0-9]+(?![A-Za-z0-9_])", value
            ):
                add_dimension(match.group(0))

    collect_dimension_tokens(function)
    collect_dimension_tokens(boundary)
    ordered["x"] = dimensions

    if duplicate:
        # Repeated declarations are not a safe permutation.  Leave the
        # candidate untouched so the strict checker reports the owning field.
        return None
    return ordered


def canonical_operation_objects(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, Any], ...] | None:
    """Materialize typed operation views from the shared records.

    Consumers that need ``IntentOp`` behavior must deserialize the exact
    ``wire`` column of the canonical operation record. They must not walk a
    second copy of ``IntentFunction.body`` and independently assign paths.
    The record remains authoritative for path, id, operands, results, type,
    attrs, and regions.
    """

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return None
    from intentir.ir.model import IntentOp, intent_op_regions

    record_by_path: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return None
        path = record.get("path")
        wire = record.get("wire")
        if (
            not isinstance(path, str)
            or path in record_by_path
            or not isinstance(wire, Mapping)
        ):
            return None
        record_by_path[path] = record

    # Deserialize only top-level canonical wires.  The nested objects yielded
    # from those wires are the same objects used for their parent, so every
    # consumer receives one typed operation tree rather than unrelated
    # per-record reconstructions.  This matters for region ownership and for
    # callers that compare a child operation with its enclosing loop/control
    # operation.
    top_level: list[tuple[str, Any]] = []
    for path in record_by_path:
        if not re.fullmatch(r"body\[\d+\]", path):
            continue
        try:
            operation = IntentOp.from_json(dict(record_by_path[path]["wire"]))
        except (TypeError, ValueError, KeyError):
            return None
        top_level.append((path, operation))
    if not top_level and record_by_path:
        return None

    result: list[tuple[str, Any]] = []

    def visit(path: str, operation: Any) -> None:
        result.append((path, operation))
        for suffix, region in intent_op_regions(operation):
            for index, child in enumerate(region):
                visit(f"{path}.{suffix}[{index}]", child)

    for path, operation in top_level:
        visit(path, operation)
    if {path for path, _operation in result} != set(record_by_path):
        return None
    for path, operation in result:
        wire = record_by_path[path].get("wire")
        if not isinstance(wire, Mapping):
            return None
        if operation._structural_payload() != dict(wire):
            return None
    return tuple(result)


def _coherent_operation_records(
    function: Mapping[str, Any],
    supplied: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...] | None:
    """Return records proven to belong to this exact function wire."""

    authored = canonical_operation_records(function)
    if authored is None:
        return None
    if supplied is None:
        return authored
    records = tuple(supplied)
    if len(records) != len(authored):
        return None
    fields = (
        "path", "id", "op", "parent_id", "lexical_role", "depth",
        "region_scope_id", "is_final", "operands", "results", "type",
        "attrs", "wire",
    )
    for candidate, expected in zip(records, authored, strict=True):
        if not isinstance(candidate, Mapping):
            return None
        if any(candidate.get(field) != expected.get(field) for field in fields):
            return None
    return tuple(dict(candidate) for candidate in records)


def _logical_scope_reference(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if re.fullmatch(r"(?:t|v)[0-9]+", value):
        return value
    return None


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    # One scalar stride carrier may be passed to several tensor pointer
    # formals in the same launch call.  The relation must be unique per
    # carrier/tensor/axis triple, not per carrier alone; each row below still
    # receives its own target-shape and instantiation checks.
    seen: set[tuple[str, str, int]] = set()
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _record_has_canonical_fields(
    record: Mapping[str, Any], record_name: str
) -> bool:
    required = CANONICAL_CONSTRUCTION_RECORD_FIELDS.get(record_name)
    return (
        required is not None
        and all(field in record for field in required)
    )


def canonical_scope_records(
    function: Mapping[str, Any],
    scopes: Sequence[Mapping[str, Any]],
    boundary: Mapping[str, Any],
    node_ids: Mapping[str, str],
    *,
    operation_records: Sequence[Mapping[str, Any]] | None = None,
    repairable_scope_fields: bool = False,
) -> tuple[dict[str, Any], ...] | None:
    """Allocate lexical definitions, captures, and effects once.

    This is the structural scope allocator shared by canonical J elaboration,
    ingress diagnostics, and Gate1's expected-scope check.  It deliberately
    does not prove dominance or evaluate predicates.  Those remain Gate1
    responsibilities; this function only establishes which authored region
    owns each operation result, local loop handle, external value use, and
    boundary effect.
    """

    function = _function_wire(function)
    if function is None or not isinstance(boundary, Mapping):
        return None
    if not isinstance(scopes, Sequence) or isinstance(scopes, (str, bytes)):
        return None
    records = _coherent_operation_records(function, operation_records)
    if records is None:
        return None
    declared = [scope for scope in scopes if isinstance(scope, Mapping)]
    if len(declared) != len(scopes):
        return None
    for scope in declared:
        # PortableScope is the persisted consumer-facing declaration, while
        # this function is the construction authority.  Strict consumers
        # require all three authored lists before deriving the projection;
        # bounded ingress repair may leave a list absent, but never malformed,
        # because the derived record will replace that absent/stale field.
        for field_name in ("definitions", "captures", "effects"):
            values = scope.get(field_name)
            if (
                values is not None
                and (
                    not isinstance(values, list)
                    or any(not isinstance(value, str) for value in values)
                    or len(values) != len(set(values))
                )
            ):
                return None
            if not repairable_scope_fields and not isinstance(values, list):
                return None
    by_scope = {
        scope.get("scope_id"): scope
        for scope in declared
        if isinstance(scope.get("scope_id"), str)
    }
    if len(by_scope) != len(declared):
        return None
    # This allocator is also used by deterministic ingress repair, before the
    # full typed archive is admitted.  Reject disconnected or cyclic scope
    # graphs here so no consumer can derive a partial lexical record.
    for scope_id, scope in by_scope.items():
        parent = scope.get("parent_scope")
        if parent is not None and (
            not isinstance(parent, str) or parent not in by_scope
        ):
            return None
    roots = [
        scope_id
        for scope_id, scope in by_scope.items()
        if scope.get("kind") == "function" and scope.get("parent_scope") is None
    ]
    if len(roots) != 1:
        return None
    root_scope = roots[0]
    for scope_id in by_scope:
        seen: set[str] = set()
        current: str | None = scope_id
        while current is not None:
            if current in seen:
                return None
            seen.add(current)
            current = by_scope[current].get("parent_scope")
        if root_scope not in seen:
            return None
    program_scopes = [
        scope_id
        for scope_id, scope in by_scope.items()
        if scope.get("kind") == "program_instance"
        and scope.get("parent_scope") == root_scope
    ]
    if len(program_scopes) > 1:
        return None
    execution_scope = program_scopes[0] if program_scopes else root_scope

    def owner_for(record: Mapping[str, Any]) -> str:
        region = record.get("region_scope_id")
        return region if isinstance(region, str) else execution_scope

    records_by_scope: dict[str, list[Mapping[str, Any]]] = {
        scope_id: [] for scope_id in by_scope
    }
    for record in records:
        owner = owner_for(record)
        if owner not in records_by_scope:
            return None
        records_by_scope[owner].append(record)

    loop_seeds: dict[str, Mapping[str, Any]] = {}
    branch_seeds: dict[str, Mapping[str, Any]] = {}
    for record in records:
        wire = record.get("wire")
        if not isinstance(wire, Mapping):
            return None
        if record.get("op") == "loop":
            scope_id = wire.get("scope_id")
            if isinstance(scope_id, str):
                if scope_id in loop_seeds:
                    return None
                loop_seeds[scope_id] = wire
        if record.get("op") in {"effect_if", "value_if"}:
            for polarity in ("positive", "negative"):
                arm = wire.get(polarity)
                if not isinstance(arm, Mapping):
                    continue
                scope_id = arm.get("scope_id")
                if isinstance(scope_id, str):
                    if scope_id in branch_seeds:
                        return None
                    branch_seeds[scope_id] = arm

    declared_loop_scope_ids = {
        scope_id
        for scope_id, scope in by_scope.items()
        if scope.get("kind") == "loop_region"
    }
    declared_branch_scope_ids = {
        scope_id
        for scope_id, scope in by_scope.items()
        if scope.get("kind") == "branch_region"
    }
    if set(loop_seeds) != declared_loop_scope_ids:
        return None
    if set(branch_seeds) != declared_branch_scope_ids:
        return None
    branch_parent_by_scope: dict[str, str] = {}
    for record in records:
        wire = record["wire"]
        owner = owner_for(record)
        if record.get("op") == "loop":
            loop_scope_id = wire.get("scope_id")
            if not isinstance(loop_scope_id, str):
                return None
            loop_scope = by_scope.get(loop_scope_id)
            if (
                not isinstance(loop_scope, Mapping)
                or loop_scope.get("kind") != "loop_region"
                or loop_scope.get("parent_scope") != owner
            ):
                return None
        if record.get("op") not in {"effect_if", "value_if"}:
            continue
        for polarity in ("positive", "negative"):
            arm = wire.get(polarity)
            if not isinstance(arm, Mapping):
                return None
            branch_scope_id = arm.get("scope_id")
            if not isinstance(branch_scope_id, str):
                return None
            branch_scope = by_scope.get(branch_scope_id)
            if (
                not isinstance(branch_scope, Mapping)
                or branch_scope.get("kind") != "branch_region"
                or branch_scope.get("parent_scope") != owner
            ):
                return None
            previous_parent = branch_parent_by_scope.get(branch_scope_id)
            if previous_parent is not None and previous_parent != owner:
                return None
            branch_parent_by_scope[branch_scope_id] = owner

    inputs = [
        item.get("name")
        for item in function.get("inputs", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    ]
    definitions: dict[str, list[str]] = {scope_id: [] for scope_id in by_scope}
    if root_scope in definitions:
        definitions[root_scope].extend(inputs)
    for scope_id, owned in records_by_scope.items():
        for record in owned:
            definitions[scope_id].extend(
                result for result in record.get("results", ())
                if isinstance(result, str)
            )
    for scope_id, wire in loop_seeds.items():
        if scope_id not in definitions:
            return None
        for scan in wire.get("scan", ()):
            if isinstance(scan, Mapping):
                definitions[scope_id].append(scan.get("name"))
        for carry in wire.get("carry", ()):
            if isinstance(carry, Mapping):
                definitions[scope_id].append(carry.get("body_argument"))
        trip = wire.get("trip")
        if isinstance(trip, Mapping) and isinstance(trip.get("index"), str):
            definitions[scope_id].append(trip.get("index"))
    definitions = {
        scope_id: _ordered_unique(values)
        for scope_id, values in definitions.items()
    }

    uses: dict[str, list[str]] = {scope_id: [] for scope_id in by_scope}

    def add_use(scope_id: str, value: Any) -> None:
        reference = _logical_scope_reference(value)
        if reference is not None:
            uses[scope_id].append(reference)

    for scope_id, owned in records_by_scope.items():
        for record in owned:
            wire = record["wire"]
            for value in wire.get("operands", ()):
                add_use(scope_id, value)
            if wire.get("op") in {"effect_if", "value_if"}:
                add_use(scope_id, wire.get("predicate"))
            if wire.get("op") == "loop":
                loop_scope = wire.get("scope_id")
                loop_use_scope = (
                    loop_scope
                    if isinstance(loop_scope, str) and loop_scope in uses
                    else scope_id
                )
                trip = wire.get("trip")
                if isinstance(trip, Mapping):
                    add_use(loop_use_scope, trip.get("upper_bound"))
                for scan in wire.get("scan", ()):
                    if isinstance(scan, Mapping):
                        # ``from`` is the canonical loop-scan input field.
                        # Keep scope derivation aligned with the operation
                        # contract and the structured SSA constructor; the
                        # legacy ``source`` spelling must not create a second
                        # visibility interpretation.
                        add_use(loop_use_scope, scan.get("from"))
                for carry in wire.get("carry", ()):
                    if isinstance(carry, Mapping):
                        add_use(loop_use_scope, carry.get("init_value"))

    for scope_id, arm in branch_seeds.items():
        add_use(scope_id, arm.get("yield_value"))

    captures: dict[str, list[str]] = {}
    for scope_id in by_scope:
        local = set(definitions[scope_id])
        captures[scope_id] = _ordered_unique(
            value for value in uses[scope_id] if value not in local
        )

    path_by_node = {
        node: path for node, path in node_ids.items()
        if isinstance(node, str) and isinstance(path, str)
    }
    record_by_path = {
        record["path"]: record for record in records
    }
    if len(path_by_node) != len(node_ids):
        return None
    expected_node_map = {
        record["id"]: record["path"]
        for record in records
        if isinstance(record.get("id"), str)
        and isinstance(record.get("path"), str)
    }
    # Check both directions.  Validating only authored node_ids entries would
    # allow a body operation to disappear from the node map and be interpreted
    # differently by Replay, Gate1, or an ingress repair pass.
    if expected_node_map != dict(path_by_node):
        return None
    for node, path in path_by_node.items():
        record = record_by_path.get(path)
        if record is None or record.get("id") != node:
            return None
    owner_by_path = {
        path: owner_for(record) for path, record in record_by_path.items()
    }
    declared_effect_owner: dict[str, str] = {}
    for scope_id, scope in by_scope.items():
        for effect_id in scope.get("effects", ()):
            if not isinstance(effect_id, str):
                continue
            previous = declared_effect_owner.get(effect_id)
            if previous is not None and previous != scope_id:
                return None
            declared_effect_owner[effect_id] = scope_id
    effect_ids: dict[str, list[str]] = {scope_id: [] for scope_id in by_scope}
    raw_effects = boundary.get("effects", [])
    if not isinstance(raw_effects, list):
        return None
    if repairable_scope_fields:
        # A node-less effect has no operation path from which ownership can be
        # derived.  Its authored scope membership is therefore the only
        # ownership witness (used for direct effects inside an effect_if arm).
        # Do not silently move such an effect to the execution scope when a
        # stale/partial scope wire omitted that witness.
        for effect in raw_effects:
            if not isinstance(effect, Mapping):
                return None
            effect_id = effect.get("effect_id")
            if effect.get("node_ref") is None and (
                not isinstance(effect_id, str)
                or effect_id not in declared_effect_owner
            ):
                return None
    seen_effects: set[str] = set()
    for effect in raw_effects:
        if not isinstance(effect, Mapping) or not isinstance(effect.get("effect_id"), str):
            return None
        effect_id = effect["effect_id"]
        if effect_id in seen_effects:
            return None
        seen_effects.add(effect_id)
        node_ref = effect.get("node_ref")
        path = path_by_node.get(node_ref) if isinstance(node_ref, str) else None
        if node_ref is not None and path is None:
            return None
        owner = (
            owner_by_path.get(path, execution_scope)
            if path is not None
            else declared_effect_owner.get(effect_id, execution_scope)
        )
        if owner not in effect_ids:
            return None
        effect_ids[owner].append(effect_id)
        if owner != execution_scope:
            add_use(owner, effect.get("target"))
    for scope_id in by_scope:
        local = set(definitions[scope_id])
        captures[scope_id] = _ordered_unique(
            value for value in captures[scope_id] + uses[scope_id]
            if value not in local
        )

    # A nested region may use a value outside its immediate parent. Carry that
    # capture upward until a lexical definition owns it; this keeps the parent
    # record complete without making a child definition visible in siblings or
    # the parent continuation.
    for _ in range(len(by_scope)):
        changed = False
        for scope_id, scope in by_scope.items():
            parent = scope.get("parent_scope")
            if not isinstance(parent, str) or parent not in captures:
                continue
            parent_defs = set(definitions[parent])
            before = len(captures[parent])
            captures[parent] = _ordered_unique(
                [
                    *captures[parent],
                    *(
                        value
                        for value in captures[scope_id]
                        if value not in parent_defs
                    ),
                ]
            )
            changed = changed or len(captures[parent]) != before
        if not changed:
            break

    # A construction record is not complete merely because every use was
    # copied into a captures list.  Prove that each captured t*/v* is defined
    # by an ancestor scope; otherwise a child can accidentally capture a
    # sibling-local or entirely unknown handle and Gate1/Replay will each
    # invent a different visibility interpretation.
    for scope_id, scope in by_scope.items():
        ancestor = scope.get("parent_scope")
        visible: set[str] = set()
        while isinstance(ancestor, str):
            parent = by_scope.get(ancestor)
            if parent is None:
                return None
            visible.update(definitions[ancestor])
            visible.update(captures[ancestor])
            ancestor = parent.get("parent_scope")
        if any(value not in visible for value in captures[scope_id]):
            return None

    result: list[dict[str, Any]] = []
    for scope in declared:
        scope_id = scope["scope_id"]
        owned = records_by_scope[scope_id]
        derived = {
            "scope_id": scope_id,
            "kind": scope.get("kind"),
            "parent_scope": scope.get("parent_scope"),
            "definitions": definitions[scope_id],
            "captures": captures[scope_id],
            "effects": effect_ids[scope_id],
            "operation_paths": [record["path"] for record in owned],
            "operation_ids": [record["id"] for record in owned],
            "operation_results": [
                value
                for record in owned
                for value in record.get("results", ())
            ],
        }
        # Definitions/captures are declarations of set membership; their
        # ordering is not semantic.  Effects are the Source/J transfer order,
        # so their order is part of the wire contract and must match exactly.
        if not repairable_scope_fields and (
            set(scope["definitions"]) != set(derived["definitions"])
            or set(scope["captures"]) != set(derived["captures"])
            or scope["effects"] != derived["effects"]
        ):
            return None
        result.append(derived)
    return tuple(result)


def _canonical_ssa_visibility(
    function: Mapping[str, Any],
) -> bool:
    """Prove lexical SSA visibility before a construction bundle is emitted.

    ``canonical_scope_records`` owns persisted definitions/captures/effects,
    but a set-like scope view alone cannot distinguish a use from a later
    sibling producer.  The host constructor must reject that forward use
    before CandidateCheck/Decode, and must also keep loop and conditional
    region bindings from leaking into a sibling or parent continuation.  This
    checker is deliberately structural: it neither infers an operation type
    nor repairs a missing producer; typed semantic validation remains at the
    existing CandidateCheck and elaboration boundaries.
    """

    normalized = _function_wire(function)
    if normalized is None:
        return False
    raw_inputs = normalized.get("inputs")
    body = normalized.get("body")
    if not isinstance(raw_inputs, list) or not isinstance(body, list):
        return False
    input_names = {
        item.get("name")
        for item in raw_inputs
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(input_names) != len(raw_inputs):
        return False

    def result_names(operation: Mapping[str, Any]) -> tuple[str, ...] | None:
        raw_result = operation.get("result")
        if raw_result is None:
            return ()
        if isinstance(raw_result, str):
            return (raw_result,)
        if isinstance(raw_result, list) and all(
            isinstance(item, str) for item in raw_result
        ):
            return tuple(raw_result)
        return None

    def visible_reference(value: Any, visible: set[str]) -> bool:
        return isinstance(value, str) and value in visible

    def visit(
        operations: Any,
        inherited: set[str],
    ) -> tuple[set[str], set[str]] | None:
        if not isinstance(operations, list):
            return None
        visible = set(inherited)
        direct_definitions: set[str] = set()
        for operation in operations:
            if not isinstance(operation, Mapping):
                return None
            op_name = operation.get("op")
            if not isinstance(op_name, str):
                return None
            operands = operation.get("operands", [])
            if not isinstance(operands, list) or any(
                not visible_reference(operand, visible) for operand in operands
            ):
                return None
            results = result_names(operation)
            if results is None or len(results) != len(set(results)):
                return None

            if op_name in {"effect_if", "value_if"}:
                if not visible_reference(operation.get("predicate"), visible):
                    return None
                arm_visibilities: dict[str, set[str]] = {}
                for arm_name in ("positive", "negative"):
                    arm = operation.get(arm_name)
                    if not isinstance(arm, Mapping):
                        return None
                    arm_result = visit(arm.get("body"), visible)
                    if arm_result is None:
                        return None
                    arm_visible, _arm_definitions = arm_result
                    arm_visibilities[arm_name] = arm_visible
                if op_name == "effect_if":
                    if results:
                        return None
                    continue
                if len(results) != 1 or results[0] in visible:
                    return None
                for arm_name in ("positive", "negative"):
                    arm = operation[arm_name]
                    if not visible_reference(
                        arm.get("yield_value"), arm_visibilities[arm_name]
                    ):
                        return None
                visible.add(results[0])
                direct_definitions.add(results[0])
                continue

            if op_name == "loop":
                carry = operation.get("carry")
                scan = operation.get("scan", [])
                trip = operation.get("trip")
                yielded = operation.get("yield")
                if (
                    not isinstance(carry, list)
                    or not isinstance(scan, list)
                    or not isinstance(trip, Mapping)
                    or not isinstance(yielded, list)
                    or any(not isinstance(item, Mapping) for item in carry)
                    or any(not isinstance(item, Mapping) for item in scan)
                    or len(carry) != len(results)
                    or len(carry) != len(yielded)
                ):
                    return None
                local_names: set[str] = set()
                for item in scan:
                    name = item.get("name")
                    source = item.get("from")
                    if (
                        not isinstance(name, str)
                        or name in visible
                        or name in local_names
                        or not visible_reference(source, visible)
                    ):
                        return None
                    local_names.add(name)
                for state in carry:
                    body_argument = state.get("body_argument")
                    init_value = state.get("init_value")
                    if (
                        not isinstance(body_argument, str)
                        or body_argument in visible
                        or body_argument in local_names
                        or (
                            isinstance(init_value, str)
                            and init_value not in {"-inf", "inf", "nan"}
                            and not visible_reference(init_value, visible)
                        )
                    ):
                        return None
                    local_names.add(body_argument)
                if "upper_bound" in trip:
                    if not visible_reference(trip.get("upper_bound"), visible):
                        return None
                    index = trip.get("index")
                    if (
                        not isinstance(index, str)
                        or index in visible
                        or index in local_names
                    ):
                        return None
                    local_names.add(index)
                if any(result in visible or result in local_names for result in results):
                    return None
                body_result = visit(operation.get("body"), visible | local_names)
                if body_result is None:
                    return None
                _body_visible, direct_body_definitions = body_result
                expected_results: list[str] = []
                expected_yields: list[str] = []
                carry_roles: set[str] = set()
                for state in carry:
                    body_argument = state.get("body_argument")
                    next_value = state.get("next_value")
                    result_value = state.get("result_value")
                    if (
                        not isinstance(body_argument, str)
                        or not isinstance(next_value, str)
                        or next_value not in direct_body_definitions
                        or not isinstance(result_value, str)
                        or len({body_argument, next_value, result_value}) != 3
                        or body_argument in carry_roles
                        or result_value in carry_roles
                        or next_value in carry_roles
                    ):
                        return None
                    carry_roles.update({body_argument, next_value, result_value})
                    expected_results.append(result_value)
                    expected_yields.append(next_value)
                if (
                    tuple(results) != tuple(expected_results)
                    or tuple(yielded) != tuple(expected_yields)
                    or len(expected_yields) != len(set(expected_yields))
                ):
                    return None
                visible.update(results)
                direct_definitions.update(results)
                continue

            if any(result in visible for result in results):
                return None
            visible.update(results)
            direct_definitions.update(results)
        return visible, direct_definitions

    return visit(body, input_names) is not None


def canonical_output_records(
    function: Mapping[str, Any],
    boundary: Mapping[str, Any],
    operation_records: Sequence[Mapping[str, Any]] | None,
    *,
    signature: Mapping[str, Any] | None = None,
    repairable_missing_returns: bool = False,
) -> tuple[dict[str, Any], ...] | None:
    """Build the one output/return/mapping/ABI record table.

    This is the raw-wire counterpart of ``JElaboration.output_bindings``.
    Lift ingress uses it before strict decoding; canonical J elaboration uses
    the same authored relationships after decoding.  If no operation table is
    supplied, it rebuilds that table from this exact function wire; a supplied
    table must match it byte-for-byte through ``_coherent_operation_records``.
    It accepts no inferred producer and never repairs a conflicting row.  In
    the bounded repair mode, a missing return row is tolerated only when the
    same output/value is already proven by an authored mapping; the caller may
    then materialize the return and must still run strict mode before committing
    the candidate.
    """

    function = _function_wire(function)
    if function is None or not isinstance(boundary, Mapping):
        return None
    if not isinstance(signature, Mapping):
        return None
    signature_parameters = signature.get("parameters")
    if not isinstance(signature_parameters, list):
        return None
    records = _coherent_operation_records(function, operation_records)
    if records is None:
        return None
    outputs = function.get("outputs")
    returns = function.get("returns")
    mappings = boundary.get("output_mappings", [])
    if not isinstance(outputs, list) or not isinstance(returns, list):
        if not (
            repairable_missing_returns
            and isinstance(outputs, list)
            and returns is None
        ):
            return None
        returns = []
    if not isinstance(mappings, list):
        return None

    output_names: list[str] = []
    output_types: dict[str, dict[str, Any]] = {}
    input_names = {
        item.get("name")
        for item in function.get("inputs", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if (
        not isinstance(function.get("inputs"), list)
        or len(input_names) != len(function["inputs"])
        or any(re.fullmatch(r"t[0-9]+", name) is None for name in input_names)
    ):
        return None
    for item in outputs:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("dtype"), str)
            or not isinstance(item.get("shape"), list)
        ):
            return None
        name = item["name"]
        if (
            re.fullmatch(r"t[0-9]+", name) is None
            or name in output_types
        ):
            return None
        output_names.append(name)
        output_types[name] = {
            "dtype": json.loads(_stable_json(item.get("dtype"))),
            "shape": json.loads(_stable_json(item.get("shape"))),
        }
    if not output_names:
        return None

    # The output record owns the physical ABI binding as well as the logical
    # output order.  Do not let a later consumer independently match an output
    # tensor to a p* row by position or by a second logical_ref scan.
    output_parameters: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for parameter_index, parameter in enumerate(signature_parameters):
        if not isinstance(parameter, Mapping):
            return None
        parameter_name = parameter.get("name")
        logical_ref = parameter.get("logical_ref")
        if not isinstance(logical_ref, str) or logical_ref not in output_types:
            continue
        if (
            not isinstance(parameter_name, str)
            or re.fullmatch(r"p[0-9]+", parameter_name) is None
            or not isinstance(logical_ref, str)
            or parameter.get("kind") not in {"tensor", "pointer"}
            or parameter.get("mode") not in {"output", "inout"}
            or logical_ref in output_parameters
        ):
            return None
        output_type = output_types[logical_ref]
        if (
            parameter.get("dtype") != output_type.get("dtype")
            or parameter.get("rank") != len(output_type.get("shape", ()))
        ):
            return None
        output_parameters[logical_ref] = (parameter_index, parameter)
    if set(output_parameters) != set(output_names):
        return None

    # The output record is the only owner of logical output order.  It must
    # agree with the output/inout subsequence of the physical signature; using
    # the declaration order from ``intent.outputs`` here would create a second
    # ABI interpretation for Gate2 and native Dataset projections.
    abi_output_names = [
        parameter.get("logical_ref")
        for parameter in signature_parameters
        if (
            isinstance(parameter, Mapping)
            and parameter.get("mode") in {"output", "inout"}
            and isinstance(parameter.get("logical_ref"), str)
            and parameter.get("logical_ref") in output_types
        )
    ]
    if tuple(abi_output_names) != tuple(output_names):
        return None

    mapping_by_output: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, item in enumerate(mappings):
        if not isinstance(item, Mapping):
            return None
        output = item.get("output")
        if not isinstance(output, str) or output not in output_types:
            return None
        if output in mapping_by_output:
            return None
        mapping_by_output[output] = (index, item)

    object_returns: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, item in enumerate(returns):
        if not isinstance(item, Mapping):
            return None
        output = item.get("output")
        value = item.get("value")
        if (
            not isinstance(output, str)
            or output not in output_types
            or not isinstance(value, str)
            or output in object_returns
        ):
            return None
        object_returns[output] = (index, item)
    authored_return_outputs = [item.get("output") for item in returns]
    if (
        authored_return_outputs != output_names
        and not repairable_missing_returns
    ):
        return None
    if not object_returns and not repairable_missing_returns:
        # Every declared output must have a function return.  A sparse
        # output_mapping is only an additional inout/store anchor; it cannot
        # replace the ABI return row in the canonical construction record.
        return None

    producers: dict[str, Mapping[str, Any]] = {}
    duplicate_producers: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            return None
        results = record.get("results")
        if not isinstance(results, (tuple, list)):
            return None
        for result in results:
            if not isinstance(result, str):
                return None
            if result in producers:
                duplicate_producers.add(result)
            else:
                producers[result] = record
    for result in duplicate_producers:
        producers.pop(result, None)

    operation_rows = records
    records: list[dict[str, Any]] = []
    for abi_index, output in enumerate(abi_output_names):
        parameter_index, parameter = output_parameters[output]
        candidates: list[str] = []
        authored_return = object_returns.get(output)
        if authored_return is not None:
            candidates.append(str(authored_return[1]["value"]))
        mapped = mapping_by_output.get(output)
        if mapped is not None:
            value = mapped[1].get("value")
            if not isinstance(value, str):
                return None
            candidates.append(value)
        if not candidates or len(set(candidates)) != 1:
            return None
        value = candidates[0]
        if re.fullmatch(r"(?:t|v)[0-9]+", value) is None:
            return None

        producer = producers.get(value)
        producer_is_proven = (
            producer is not None
            and (
                "contract_valid" not in producer
                or producer.get("contract_valid") is True
            )
        )
        authored_node_ref = (
            mapped[1].get("node_ref")
            if mapped is not None and isinstance(mapped[1].get("node_ref"), str)
            else None
        )
        producer_id = producer.get("id") if producer_is_proven else None
        if value.startswith("t"):
            # A t* return is a source input/inout binding.  A pure output
            # declaration is not a visible value producer and must not be
            # mistaken for one merely because it appears in outputs[].
            if value not in input_names:
                return None
        else:
            if not producer_is_proven or producer_id is None:
                return None
            if re.fullmatch(r"body\[\d+\]", str(producer.get("path"))) is None:
                # Values defined in a loop/conditional region do not escape
                # that region.  Only the top-level operation result can be a
                # function return value.
                return None
        if authored_node_ref is not None and authored_node_ref != producer_id:
            # ``output_mappings.node_ref`` is a sparse Layer-B store anchor,
            # not necessarily the Layer-A SSA producer.  Accept the distinct
            # node only when the shared boundary ledger proves one unique
            # output store/update access and matching write effect, and the
            # store operation consumes the returned value.  This keeps
            # output/return construction single-sourced without confusing
            # mutation with value production.
            accesses = boundary.get("accesses")
            effects = boundary.get("effects")
            if not isinstance(accesses, list) or not isinstance(effects, list):
                return None
            store_refs = {
                item.get("node_ref")
                for item in accesses
                if isinstance(item, Mapping)
                and item.get("kind") in {"store", "update"}
                and item.get("tensor") == output
                and isinstance(item.get("node_ref"), str)
                and any(
                    isinstance(effect, Mapping)
                    and effect.get("kind") == "write"
                    and effect.get("target") == output
                    and effect.get("access_ref") == item.get("access_id")
                    and effect.get("node_ref") == item.get("node_ref")
                    for effect in effects
                )
            }
            store_record = next(
                (
                    record for record in operation_rows
                    if record.get("id") == authored_node_ref
                    and record.get("op") in {"scatter", "masked_scatter", "store"}
                    and (
                        value in record.get("operands", ())
                        if isinstance(record.get("operands"), (tuple, list))
                        else False
                    )
                ),
                None,
            )
            if len(store_refs) != 1 or authored_node_ref not in store_refs:
                return None
            if store_record is None:
                return None
        records.append(
            {
                "output": output,
                "abi_index": abi_index,
                "output_type": output_types[output],
                "return_value": value,
                "abi_parameter_index": parameter_index,
                "abi_parameter_name": parameter["name"],
                "abi_parameter_mode": parameter["mode"],
                "mapping_index": mapped[0] if mapped is not None else None,
                "mapping_value": mapped[1].get("value") if mapped is not None else None,
                "authored_node_ref": authored_node_ref,
                "node_ref": producer_id,
                "producer_record": (
                    dict(producer) if producer_is_proven else None
                ),
                "return_index": (
                    authored_return[0] if authored_return is not None else None
                ),
            }
        )
    return tuple(records)


def canonical_value_type_records(
    function: Mapping[str, Any],
    operation_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Project explicit input/output/operation types from one op table."""

    function = _function_wire(function)
    if function is None:
        return None
    records = _coherent_operation_records(function, operation_records)
    if records is None:
        return None
    values: dict[str, dict[str, Any]] = {}
    for collection in ("inputs", "outputs"):
        raw_values = function.get(collection)
        if not isinstance(raw_values, list):
            return None
        collection_names: set[str] = set()
        for item in raw_values:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("dtype"), str)
                or not isinstance(item.get("shape"), list)
            ):
                return None
            name = item["name"]
            if re.fullmatch(r"t[0-9]+", name) is None or name in collection_names:
                return None
            collection_names.add(name)
            value_type = {
                "dtype": json.loads(_stable_json(item.get("dtype"))),
                "shape": json.loads(_stable_json(item.get("shape"))),
            }
            previous = values.get(name)
            if previous is not None and previous != value_type:
                return None
            values[name] = value_type
    for record in records:
        results = record.get("results")
        if not isinstance(results, (tuple, list)):
            return None
        typed_results: list[tuple[str, Mapping[str, Any]]] = []
        declared = record.get("type")
        if len(results) == 1 and isinstance(declared, Mapping):
            result = results[0]
            if not isinstance(result, str):
                return None
            typed_results.append((result, declared))
        elif record.get("op") == "loop":
            wire = record.get("wire")
            carries = wire.get("carry") if isinstance(wire, Mapping) else None
            if not isinstance(carries, list) or len(carries) != len(results):
                return None
            for index, state in enumerate(carries):
                if not isinstance(state, Mapping):
                    return None
                result = state.get("result_value")
                carry_type = state.get("type")
                if not isinstance(result, str) or not isinstance(carry_type, Mapping):
                    return None
                if result != results[index]:
                    return None
                typed_results.append((result, carry_type))
        for result, declared_type in typed_results:
            normalized = json.loads(_stable_json(dict(declared_type)))
            previous = values.get(result)
            if previous is not None and previous != normalized:
                return None
            values[result] = normalized
    return values


def _canonical_relation_records(
    rows: Any,
    *,
    record_name: str,
) -> tuple[dict[str, Any], ...] | None:
    """Copy one already-validated J relation table into the construction bundle.

    Dimension relations are validated by ``SemanticArchive`` before the normal
    construction pass.  This helper does not re-infer or evaluate them; it
    only gives all downstream consumers the same ordered, JSON-stable rows.
    A missing table is distinct from an empty table at the caller boundary,
    so malformed/non-array input remains fail-closed.
    """

    required = CANONICAL_CONSTRUCTION_RECORD_FIELDS[record_name]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        if set(row) != set(required):
            return None
        copied = json.loads(_stable_json(dict(row)))
        if not isinstance(copied, dict):
            return None
        if any(field not in copied for field in required):
            return None
        result.append(copied)
    return tuple(result)


def canonical_boundary_records(
    boundary: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    """Build the one structural boundary projection for construction.

    ``SemanticArchive`` remains responsible for the detailed boundary row
    contracts.  This helper does not infer or repair any row; it copies the
    already-decoded collections in their authored order and makes the exact
    collection set available to every later consumer.  A missing or unknown
    collection means that no shared construction bundle exists.
    """

    if not isinstance(boundary, Mapping):
        return None
    if set(boundary) != set(CANONICAL_BOUNDARY_WIRE_FIELDS):
        return None
    copied = json.loads(_stable_json(dict(boundary)))
    if not isinstance(copied, dict):
        return None
    if not all(field in copied for field in CANONICAL_BOUNDARY_WIRE_FIELDS):
        return None
    if not isinstance(copied.get("domains"), dict):
        return None
    if any(
        not isinstance(copied.get(field), list)
        for field in CANONICAL_BOUNDARY_WIRE_FIELDS
        if field != "domains"
    ):
        return None
    return (copied,)


@dataclass(frozen=True)
class CanonicalConstructionRecords:
    """Atomic operation/value/output/scope construction result.

    ``build`` returns only a complete bundle.  ``build_structural`` is the
    pre-output phase used by bounded ingress repair: it contains the same
    operation/value/scope records, but deliberately has no output records
    until the authored output/return/mapping relationships have been repaired
    or proven complete.  It is not a second construction authority; the
    complete ``build`` must still succeed before any repair is committed.
    """

    operation_records: tuple[Mapping[str, Any], ...]
    value_type_records: Mapping[str, Mapping[str, Any]]
    output_records: tuple[Mapping[str, Any], ...]
    scope_records: tuple[Mapping[str, Any], ...]
    dimension_records: tuple[Mapping[str, Any], ...] = ()
    value_dimension_records: tuple[Mapping[str, Any], ...] = ()
    boundary_records: tuple[Mapping[str, Any], ...] = ()

    @property
    def operation_by_path(self) -> dict[str, Mapping[str, Any]]:
        """Index the canonical operation rows without rebuilding their paths."""

        return {
            str(record["path"]): record
            for record in self.operation_records
            if isinstance(record, Mapping) and isinstance(record.get("path"), str)
        }

    @property
    def operation_by_id(self) -> dict[str, Mapping[str, Any]]:
        """Index operation rows by the same canonical ``n*`` identity."""

        return {
            str(record["id"]): record
            for record in self.operation_records
            if isinstance(record, Mapping) and isinstance(record.get("id"), str)
        }

    def operation_record_by_ref(
        self,
        node_ids: Mapping[str, str] | None = None,
    ) -> dict[str, Mapping[str, Any]] | None:
        """Return one raw operation-record table for paths and node ids.

        This is the raw-wire companion to ``typed_operation_by_ref``.  It is
        intended for structural projections such as Gate2 policy and Layer-C
        attachment checks; it does not permit a consumer to reinterpret the
        operation.  Unknown or disagreeing authored aliases fail closed.
        """

        by_path = self.operation_by_path
        by_id = self.operation_by_id
        result = dict(by_path)
        if node_ids is None:
            return result
        for node_id, path in node_ids.items():
            if not isinstance(node_id, str) or not isinstance(path, str):
                return None
            record = by_path.get(path)
            if record is None or node_id in result:
                return None
            if by_id.get(node_id) is not record:
                return None
            result[node_id] = record
        return result

    def authored_node_by_path(
        self,
        node_ids: Mapping[str, str],
    ) -> dict[str, str] | None:
        """Return the validated authored ``path -> n*`` projection."""

        if not isinstance(node_ids, Mapping):
            return None
        by_path = self.operation_by_path
        result: dict[str, str] = {}
        for node_id, path in node_ids.items():
            if not isinstance(node_id, str) or not isinstance(path, str):
                return None
            record = by_path.get(path)
            if record is None or record.get("id") != node_id:
                return None
            if path in result:
                return None
            result[path] = node_id
        return result

    @property
    def typed_operation_by_path(self) -> dict[str, Any] | None:
        """Index the one typed operation tree by canonical lexical path.

        This is the only typed path table that consumers should use.  In
        particular, callers must not rebuild it by walking ``intent.body``
        after the construction pass, because that would make path/region
        ownership a second interface contract.
        """

        rows = self.typed_operation_rows
        if rows is None:
            return None
        return dict(rows)

    @property
    def typed_operation_by_id(self) -> dict[str, Any] | None:
        """Index the same typed operation tree by canonical ``n*`` id."""

        rows = self.typed_operation_rows
        if rows is None:
            return None
        result: dict[str, Any] = {}
        for path, operation in rows:
            record = self.operation_by_path.get(path)
            operation_id = record.get("id") if record is not None else None
            if not isinstance(operation_id, str) or operation_id in result:
                return None
            result[operation_id] = operation
        return result

    def typed_operation_by_ref(
        self,
        node_ids: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Return one reference table for both paths and authored node ids.

        Boundary rows are allowed to refer to an operation by its stable
        ``n*`` id or by its canonical body path.  The mapping is constructed
        once here and rejects an unknown path instead of silently dropping the
        alias.  ``None`` is a fail-closed result, not an invitation to fall
        back to a raw-body walk.
        """

        # Materialize the typed tree exactly once.  Rebuilding it separately
        # for ``by_path`` and ``by_id`` creates equal-but-distinct IntentOp
        # objects; an identity comparison between those two views then rejects
        # every valid node map.  Path and n* indexes must be projections of one
        # typed construction result, not two deserializations.
        rows = self.typed_operation_rows
        if rows is None:
            return None
        by_path = dict(rows)
        by_id: dict[str, Any] = {}
        for path, operation in rows:
            record = self.operation_by_path.get(path)
            operation_id = record.get("id") if record is not None else None
            if not isinstance(operation_id, str) or operation_id in by_id:
                return None
            by_id[operation_id] = operation
        result = dict(by_path)
        if node_ids is None:
            return result
        for node_id, path in node_ids.items():
            if not isinstance(node_id, str) or not isinstance(path, str):
                return None
            operation = by_path.get(path)
            if operation is None or node_id in result:
                return None
            if by_id.get(node_id) is not operation:
                return None
            result[node_id] = operation
        return result

    @property
    def value_type_by_name(self) -> dict[str, Mapping[str, Any]]:
        """Return the closed SSA/tensor type table owned by this bundle."""

        return dict(self.value_type_records)

    @property
    def output_by_name(self) -> dict[str, Mapping[str, Any]]:
        """Index the output/return/ABI rows in the canonical bundle."""

        return {
            str(record["output"]): record
            for record in self.output_records
            if isinstance(record, Mapping) and isinstance(record.get("output"), str)
        }

    @property
    def output_names(self) -> tuple[str, ...]:
        """Logical outputs in the already-proven physical ABI order."""

        return tuple(
            str(record["output"])
            for record in self.output_records
            if isinstance(record, Mapping) and isinstance(record.get("output"), str)
        )

    @property
    def output_values(self) -> dict[str, str]:
        """Output-to-return SSA values from the same output records."""

        return {
            str(record["output"]): str(record["return_value"])
            for record in self.output_records
            if (
                isinstance(record, Mapping)
                and isinstance(record.get("output"), str)
                and isinstance(record.get("return_value"), str)
            )
        }

    @property
    def scope_by_id(self) -> dict[str, Mapping[str, Any]]:
        """Index the derived lexical ownership records."""

        return {
            str(record["scope_id"]): record
            for record in self.scope_records
            if isinstance(record, Mapping)
            and isinstance(record.get("scope_id"), str)
        }

    @property
    def dimension_by_result(self) -> dict[str, Mapping[str, Any]]:
        """Index the one ordered J dimension-relation construction table."""

        return {
            str(record["result"]): record
            for record in self.dimension_records
            if isinstance(record, Mapping)
            and isinstance(record.get("result"), str)
        }

    @property
    def boundary_record(self) -> Mapping[str, Any] | None:
        """Return the single construction-owned boundary wire projection."""

        if len(self.boundary_records) != 1:
            return None
        record = self.boundary_records[0]
        return record if isinstance(record, Mapping) else None

    def boundary_rows(self, name: str) -> tuple[Mapping[str, Any], ...] | None:
        """Return one boundary collection without rebuilding it from raw J."""

        record = self.boundary_record
        if record is None or name not in CANONICAL_BOUNDARY_WIRE_FIELDS:
            return None
        rows = record.get(name)
        if name == "domains" or not isinstance(rows, list):
            return None
        if any(not isinstance(item, Mapping) for item in rows):
            return None
        return tuple(rows)

    @property
    def boundary_domains(self) -> Mapping[str, Any] | None:
        """Return the construction-owned domain map without re-parsing J."""

        record = self.boundary_record
        domains = record.get("domains") if record is not None else None
        return domains if isinstance(domains, Mapping) else None

    @property
    def typed_operation_rows(self) -> tuple[tuple[str, Any], ...] | None:
        """Materialize typed operations from the stored canonical wire only."""

        return canonical_operation_objects(self.operation_records)

    def with_elaborated_value_types(
        self,
        value_types: Mapping[str, Any],
    ) -> "CanonicalConstructionRecords | None":
        """Close the raw construction table with one elaborated type map.

        The raw ingress pass can only project authored operation types.  A
        canonical J elaboration may prove additional SSA/loop types from a
        closed operation contract.  Merge those proofs into this same bundle,
        rejecting any disagreement instead of leaving raw and inferred type
        tables as competing authorities.
        """

        if not isinstance(value_types, Mapping):
            return None
        merged = {
            str(name): json.loads(_stable_json(dict(value_type)))
            for name, value_type in value_types.items()
            if isinstance(name, str) and isinstance(value_type, Mapping)
        }
        if len(merged) != len(value_types):
            return None
        for name, authored in self.value_type_records.items():
            inferred = merged.get(name)
            if inferred is not None and dict(authored) != inferred:
                return None
            merged[name] = json.loads(_stable_json(dict(authored)))
        typed_operations: list[dict[str, Any]] = []
        for record in self.operation_records:
            operands = record.get("operands")
            results = record.get("results")
            if not isinstance(operands, (tuple, list)) or not isinstance(
                results, (tuple, list)
            ):
                return None
            operand_types: list[dict[str, Any]] = []
            for operand in operands:
                if not isinstance(operand, str):
                    return None
                operand_type = merged.get(operand)
                if not isinstance(operand_type, Mapping):
                    return None
                # Preserve operand ordinal, including repeated operands;
                # keyed maps would erase that part of the operation contract.
                operand_types.append(
                    json.loads(_stable_json(dict(operand_type)))
                )
            result_types: list[dict[str, Any]] = []
            for result in results:
                if not isinstance(result, str):
                    return None
                result_type = merged.get(result)
                if not isinstance(result_type, Mapping):
                    return None
                result_types.append(
                    json.loads(_stable_json(dict(result_type)))
                )
            declared_type = record.get("type")
            if (
                isinstance(declared_type, Mapping)
                and len(results) == 1
                and dict(declared_type) != result_types[0]
            ):
                return None
            typed_record = dict(record)
            typed_record["operand_types"] = operand_types
            typed_record["result_types"] = result_types
            if any(
                field not in typed_record
                for field in CANONICAL_ELABORATED_OPERATION_RECORD_FIELDS
            ):
                return None
            typed_operations.append(typed_record)
        return replace(
            self,
            value_type_records=merged,
            operation_records=tuple(typed_operations),
        )

    @classmethod
    def build_structural(
        cls,
        function: Mapping[str, Any],
        boundary: Mapping[str, Any],
        scopes: Sequence[Mapping[str, Any]],
        node_ids: Mapping[str, str],
        *,
        operation_records: Sequence[Mapping[str, Any]] | None = None,
        repairable_scope_fields: bool = False,
        dimension_relations: Sequence[Mapping[str, Any]] | None = None,
        value_dimension_relations: Sequence[Mapping[str, Any]] | None = None,
    ) -> "CanonicalConstructionRecords | None":
        """Build the strict pre-output phase of the canonical bundle.

        This phase is strict about operation contracts, value types, node
        ownership, and lexical scopes.  It does not require the output table
        to be complete because deterministic ingress repair may need those
        already-proven structural facts to repair an output return or mapping.
        No caller may use this phase as a final artifact: ``build`` remains
        the commit gate.
        """

        records = _coherent_operation_records(function, operation_records)
        if records is None:
            return None
        if not _canonical_ssa_visibility(function):
            return None
        for record in records:
            if record.get("contract_valid") is False:
                return None
            wire = record.get("wire")
            if not isinstance(wire, Mapping):
                return None
            try:
                validate_intent_op_structure(
                    wire,
                    path=str(record.get("path", "body")),
                    recursive=True,
                )
            except (TypeError, ValueError):
                return None
        value_types = canonical_value_type_records(function, records)
        scope_records = canonical_scope_records(
            function,
            scopes,
            boundary,
            node_ids,
            operation_records=records,
            repairable_scope_fields=repairable_scope_fields,
        )
        if value_types is None or scope_records is None:
            return None
        if any(
            not _record_has_canonical_fields(record, "operation")
            for record in records
        ) or any(
            not _record_has_canonical_fields(record, "scope")
            for record in scope_records
        ):
            return None
        dimensions = _canonical_relation_records(
            () if dimension_relations is None else dimension_relations,
            record_name="dimension",
        )
        value_dimensions = _canonical_relation_records(
            ()
            if value_dimension_relations is None
            else value_dimension_relations,
            record_name="value_dimension",
        )
        boundary_records = canonical_boundary_records(boundary)
        if (
            dimensions is None
            or value_dimensions is None
            or boundary_records is None
        ):
            return None
        if any(
            not _record_has_canonical_fields(record, "boundary")
            for record in boundary_records
        ):
            return None
        return cls(
            records,
            value_types,
            (),
            scope_records,
            dimensions,
            value_dimensions,
            boundary_records,
        )

    @classmethod
    def build(
        cls,
        function: Mapping[str, Any],
        boundary: Mapping[str, Any],
        scopes: Sequence[Mapping[str, Any]],
        node_ids: Mapping[str, str],
        *,
        operation_records: Sequence[Mapping[str, Any]] | None = None,
        signature: Mapping[str, Any] | None = None,
        dimension_relations: Sequence[Mapping[str, Any]] | None = None,
        value_dimension_relations: Sequence[Mapping[str, Any]] | None = None,
    ) -> "CanonicalConstructionRecords | None":
        structural = cls.build_structural(
            function,
            boundary,
            scopes,
            node_ids,
            operation_records=operation_records,
            dimension_relations=dimension_relations,
            value_dimension_relations=value_dimension_relations,
        )
        if structural is None:
            return None
        outputs = canonical_output_records(
            function,
            boundary,
            structural.operation_records,
            signature=signature,
        )
        if outputs is None:
            return None
        if any(
            not _record_has_canonical_fields(record, "output")
            for record in outputs
        ):
            return None
        if any(
            not _record_has_canonical_fields(record, "operation")
            for record in structural.operation_records
        ) or any(
            not _record_has_canonical_fields(record, "scope")
            for record in structural.scope_records
        ):
            return None
        for output in outputs:
            value = output.get("return_value")
            actual_type = structural.value_type_records.get(value)
            expected_type = output.get("output_type")
            if not isinstance(expected_type, Mapping):
                return None
            # A multi-result operation may have no single ``operation.type``
            # wire even though the canonical J elaborator proves each result
            # from its operands.  Do not invent that type in this raw record;
            # when an explicit/loop-boundary type exists, however, compare it
            # here so output ABI and value typing cannot diverge.
            if not isinstance(actual_type, Mapping):
                continue
            if dict(actual_type) != dict(expected_type):
                return None
        return cls(
            structural.operation_records,
            structural.value_type_records,
            outputs,
            structural.scope_records,
            structural.dimension_records,
            structural.value_dimension_records,
            structural.boundary_records,
        )


def canonical_archive_records(
    archive: Any,
    *,
    structural: bool = False,
    repairable_scope_fields: bool = False,
) -> CanonicalConstructionRecords | None:
    """Build the one construction bundle for a semantic archive wire.

    This is the adapter boundary for consumers that receive a decoded J
    object.  It intentionally accepts either a mapping or an object exposing
    ``to_json`` and then delegates to the same operation/value/output/scope
    builders used by Lift ingress and canonical J elaboration.  Consumers
    must use the returned indexes/views instead of re-walking ``intent.body``
    or independently joining returns, mappings, and scopes.

    ``structural=True`` is only for pre-output diagnostics/repair.  It never
    constitutes an accepted artifact; callers that commit a candidate must
    call this function again in strict (complete) mode.
    """

    if isinstance(archive, Mapping):
        payload: Any = archive
    else:
        to_json = getattr(archive, "to_json", None)
        if not callable(to_json):
            return None
        payload = to_json()
    if not isinstance(payload, Mapping):
        return None
    function = payload.get("intent")
    boundary = payload.get("boundary")
    scopes = payload.get("scopes")
    node_ids = payload.get("node_ids")
    signature = payload.get("signature")
    value_dimension_relations = payload.get("value_dimension_relations")
    dimension_relations = payload.get("dimension_relations")
    if not isinstance(function, Mapping) or not isinstance(boundary, Mapping):
        return None
    if not isinstance(scopes, Sequence) or isinstance(scopes, (str, bytes)):
        return None
    if not isinstance(node_ids, Mapping) or not isinstance(signature, Mapping):
        return None
    if structural:
        return CanonicalConstructionRecords.build_structural(
            function,
            boundary,
            scopes,
            node_ids,
            repairable_scope_fields=repairable_scope_fields,
            dimension_relations=dimension_relations,
            value_dimension_relations=value_dimension_relations,
        )
    return CanonicalConstructionRecords.build(
        function,
        boundary,
        scopes,
        node_ids,
        signature=signature,
        dimension_relations=dimension_relations,
        value_dimension_relations=value_dimension_relations,
    )


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _copy_mapping(value: Mapping[str, Any], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{context} must be a string-keyed object")
    copied = json.loads(_stable_json(dict(value)))
    if not isinstance(copied, dict):
        raise TypeError(f"{context} must be a JSON object")
    return copied


def _view_json(view: Any) -> Mapping[str, Any]:
    if isinstance(view, Mapping):
        return view
    to_json = getattr(view, "to_json", None)
    if callable(to_json):
        value = to_json()
        if isinstance(value, Mapping):
            return value
    raise TypeError("compiler view must be a mapping or expose to_json()")


def _canonical_abi_name(name: Any, mapping: Mapping[str, str], context: str) -> str:
    """Map a source spelling to its one canonical ABI carrier.

    Facts may be built from an original Harness view or from its anonymous
    projection.  The latter already uses ``p*/c*``; the former must name every
    ABI item through ``source_to_opaque``.  Silently retaining an unknown
    source spelling would make equivalent views hash differently, so it is a
    closed construction error rather than a best-effort conversion.
    """

    if not isinstance(name, str):
        raise TypeError(f"{context} name must be a string")
    canonical = mapping.get(name, name)
    if _ABI_HANDLE_RE.fullmatch(canonical) is None:
        raise ValueError(
            f"{context} name {name!r} has no canonical p*/c* mapping"
        )
    return canonical


def project_host_launch_contract(contract: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    """Rename only ABI references in a host-issued launch contract."""
    copied = _copy_mapping(contract, "host launch contract")
    if not copied:
        return {}
    for collection, keys in (("strides", ("carrier", "tensor")), ("tiles", ("slot",))):
        for row in copied.get(collection, ()):
            for key in keys:
                row[key] = _canonical_abi_name(row[key], mapping, "host launch contract")
    return copied


def validate_host_launch_contract(
    contract: Mapping[str, Any], *, arguments: Sequence[Mapping[str, Any]],
    schedule_values: Mapping[str, Any], shape_coordinates: Mapping[str, int],
    tile_coordinates: Mapping[str, str], source_digest: str | None = None,
) -> None:
    """Validate the frozen declaration AND its current invocation separately.

    Valuation equality never creates a relation. The source adapter must first
    derive the relation from the exact launcher/helper AST; this consumer only
    checks its closed wire and instantiation. No candidate may issue it.
    """
    if not contract:
        return
    if set(contract) != {"source_digest", "strides", "tiles"}:
        raise ValueError("source_host_launch_contract_malformed")
    digest = contract["source_digest"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None or (source_digest is not None and digest != source_digest):
        raise ValueError("source_host_launch_digest_mismatch")
    if not isinstance(contract["strides"], list) or not isinstance(contract["tiles"], list):
        raise ValueError("source_host_launch_contract_malformed")
    by_name = {item["name"]: item for item in arguments}
    for argument in arguments:
        coordinate = argument.get("shape_coordinate")
        if coordinate is not None:
            if (
                not isinstance(coordinate, str) or coordinate not in shape_coordinates
                or argument.get("kind") != "scalar" or argument.get("dtype") not in {"i32", "i64"}
                or type(argument.get("value")) is not int
                or argument["value"] != shape_coordinates[coordinate]
                or not 0 < argument["value"] < 2 ** (int(argument["dtype"][1:]) - 1)
            ):
                raise ValueError("source_host_shape_scalar_mismatch")
        declared, shape = argument.get("shape_contract"), argument.get("shape")
        if declared is not None:
            if not isinstance(declared, list) or not isinstance(shape, list) or len(declared) != len(shape):
                raise ValueError("source_host_shape_contract_mismatch")
            for dimension, extent in zip(declared, shape, strict=True):
                expected = shape_coordinates.get(dimension) if isinstance(dimension, str) else dimension
                if type(expected) is not int or type(extent) is not int or extent <= 0 or expected != extent:
                    raise ValueError("source_host_shape_contract_mismatch")
    seen: set[str] = set()
    for row in contract["strides"]:
        if not isinstance(row, Mapping) or set(row) != {"carrier", "tensor", "axis"}:
            raise ValueError("source_host_stride_contract_malformed")
        carrier, tensor, axis = row["carrier"], row["tensor"], row["axis"]
        scalar, target = by_name.get(carrier), by_name.get(tensor)
        relation = (carrier, tensor, axis)
        if relation in seen or scalar is None or target is None or type(axis) is not int:
            raise ValueError("source_host_stride_binding_ambiguous")
        seen.add(relation)
        shape = target.get("shape")
        if scalar.get("kind") != "scalar" or scalar.get("dtype") not in {"i32", "i64"} or type(scalar.get("value")) is not int or target.get("kind") != "tensor" or not isinstance(shape, list) or not 0 <= axis < len(shape):
            raise ValueError("source_host_stride_type_mismatch")
        strides = target.get("strides")
        if strides is None:
            # CompilerView's absent strides denote a contiguous ABI carrier.
            strides = [math.prod(shape[index + 1:]) for index in range(len(shape))]
        if not isinstance(strides, list) or len(strides) != len(shape) or type(strides[axis]) is not int or scalar["value"] != strides[axis] or not 0 <= scalar["value"] < 2 ** (int(scalar["dtype"][1:]) - 1):
            raise ValueError("source_host_stride_instantiation_mismatch")
    seen.clear()
    for row in contract["tiles"]:
        if not isinstance(row, Mapping) or set(row) != {"slot", "coordinate", "law", "helper_digest"}:
            raise ValueError("source_host_tile_contract_malformed")
        slot, coordinate = row["slot"], row["coordinate"]
        if slot in seen or row["law"] != "next_power_of_two" or tile_coordinates.get(slot) != coordinate:
            raise ValueError("source_host_tile_binding_mismatch")
        seen.add(slot)
        helper_digest = row["helper_digest"]
        if not isinstance(helper_digest, str) or re.fullmatch(r"[0-9a-f]{64}", helper_digest) is None:
            raise ValueError("source_host_tile_helper_unbound")
        extent, tile = shape_coordinates.get(coordinate), schedule_values.get(slot)
        if type(extent) is not int or extent <= 0 or type(tile) is not int or tile != 1 << (extent - 1).bit_length():
            raise ValueError("source_host_tile_instantiation_mismatch")


def source_host_launch_contract_receipt(
    source_facts: Any,
    transfer_ledger: Any | None = None,
) -> dict[str, Any]:
    """Return the compact lineage receipt for one host launch contract.

    The receipt is deliberately not a second ABI or launch schema.  The
    canonical ``SourceFacts.compiler_facts`` object remains the sole owner of
    the actual contract.  This small, digest-only projection lets
    ``CanonicalConstruction``, the transfer ledger, and MCP prove that they
    are reading that same contract without copying physical argument names or
    asking the provider to reconstruct host facts.
    """

    if not isinstance(source_facts, SourceFacts):
        raise TypeError("host launch contract receipt requires SourceFacts")
    compiler_facts = source_facts.compiler_facts
    contract = compiler_facts.get("host_launch_contract")
    if not isinstance(contract, Mapping) or not contract:
        receipt: dict[str, Any] = {
            "protocol": "intentir.source-host-launch-contract-receipt.v1",
            "status": "unavailable",
            "reason": "source_host_launch_contract_unavailable",
            "source_facts_digest": source_facts.digest,
            "source_digest": source_facts.source_digest,
        }
        if transfer_ledger is not None:
            if not isinstance(transfer_ledger, SourceTransferLedger):
                raise TypeError("host launch receipt transfer ledger is invalid")
            if transfer_ledger.source_facts_digest != source_facts.digest:
                raise ValueError("host launch receipt differs from SourceFacts")
            receipt["transfer_ledger_digest"] = transfer_ledger.digest
            receipt["transfer_status"] = transfer_ledger.status
            receipt["occurrence_count"] = len(transfer_ledger.occurrences)
        return receipt

    validate_host_launch_contract(
        contract,
        arguments=source_facts.arguments,
        schedule_values=compiler_facts["schedule_values"],
        shape_coordinates=dict(source_facts.shape_coordinates),
        tile_coordinates=compiler_facts["schedule_tile_coordinates"],
        source_digest=source_facts.source_digest,
    )

    abi_payload = {
        "parameter_order": list(source_facts.parameter_order),
        "constexpr_parameters": list(source_facts.constexpr_parameters),
        "compiler_slots": list(source_facts.compiler_slots),
        "source_to_opaque": [list(row) for row in source_facts.source_to_opaque],
        "arguments": [dict(row) for row in source_facts.arguments],
    }
    launch_payload = {
        "shape_coordinates": [list(row) for row in source_facts.shape_coordinates],
        "schedule_values": dict(compiler_facts["schedule_values"]),
        "schedule_value_types": dict(compiler_facts["schedule_value_types"]),
        "schedule_tile_coordinates": dict(
            compiler_facts["schedule_tile_coordinates"]
        ),
        "grid": list(compiler_facts["grid"]),
        "grid_contract": list(compiler_facts["grid_contract"]),
        "grid_shape_coordinates": list(compiler_facts["grid_shape_coordinates"]),
        "host_launch_contract": dict(contract),
    }
    receipt: dict[str, Any] = {
        "protocol": "intentir.source-host-launch-contract-receipt.v1",
        "status": "pass",
        "source_facts_digest": source_facts.digest,
        "source_digest": source_facts.source_digest,
        "contract_digest": _digest(contract),
        "abi_digest": _digest(abi_payload),
        "launch_digest": _digest(launch_payload),
        "coverage": {
            "abi_order": True,
            "pointer": any(
                row.get("kind") == "tensor" for row in source_facts.arguments
            )
            and all(
                isinstance(row.get("name"), str)
                and row.get("kind") == "tensor"
                and row.get("mode") in {"input", "output", "inout"}
                for row in source_facts.arguments
                if row.get("kind") == "tensor"
            ),
            "stride": all(
                isinstance(row, Mapping)
                and isinstance(row.get("carrier"), str)
                and isinstance(row.get("tensor"), str)
                for row in contract["strides"]
            ),
            "shape": all(
                row.get("kind") != "tensor"
                or isinstance(row.get("shape"), list)
                for row in source_facts.arguments
            ),
            "constexpr": set(source_facts.constexpr_parameters).issubset(
                set(source_facts.parameter_order)
            ),
            "grid": all(
                isinstance(compiler_facts.get(field), list)
                for field in ("grid", "grid_contract", "grid_shape_coordinates")
            ),
            "logical_domain": bool(source_facts.shape_coordinates),
            "predicate_access_dependency": False,
        },
    }
    if transfer_ledger is not None:
        if not isinstance(transfer_ledger, SourceTransferLedger):
            raise TypeError("host launch receipt transfer ledger is invalid")
        if transfer_ledger.source_facts_digest != source_facts.digest:
            raise ValueError("host launch receipt differs from SourceFacts")
        receipt["transfer_ledger_digest"] = transfer_ledger.digest
        receipt["transfer_status"] = transfer_ledger.status
        receipt["occurrence_count"] = len(transfer_ledger.occurrences)
        access_dependency_closed = transfer_ledger.status == "pass"
        if access_dependency_closed:
            for occurrence, binding in zip(
                transfer_ledger.occurrences,
                transfer_ledger.bindings,
                strict=True,
            ):
                authorizations = binding.get("semantic_authorizations")
                access_view = (
                    authorizations.get("access_view")
                    if isinstance(authorizations, Mapping)
                    else None
                )
                guard = (
                    authorizations.get("computed_guard")
                    if isinstance(authorizations, Mapping)
                    else None
                )
                if not isinstance(access_view, Mapping) or access_view.get("status") != "pass":
                    access_dependency_closed = False
                    break
                if (
                    occurrence.masked
                    and occurrence.predicate is None
                    and (not isinstance(guard, Mapping) or guard.get("status") != "pass")
                ):
                    access_dependency_closed = False
                    break
        receipt["coverage"]["predicate_access_dependency"] = (
            access_dependency_closed
            and len(transfer_ledger.bindings) == len(transfer_ledger.occurrences)
        )
    return receipt


def _canonical_compiler_facts(
    raw: Mapping[str, Any],
    mapping: Mapping[str, str],
    *,
    arguments: Sequence[Mapping[str, Any]],
    parameter_order: Sequence[str],
    constexpr_parameters: Sequence[str],
    compiler_slots: Sequence[str],
) -> dict[str, Any]:
    """Project a compiler view once into the anonymous SourceFacts namespace.

    The source adapter is the only place that may know source spellings.  All
    consumers after Lift preparation must consume this copied, canonical view
    through ``SourceFacts`` instead of independently re-projecting the raw
    compiler view.  It is deliberately data-only: frontend compilation and
    execution remain owned by their respective adapters.
    """

    def canonical_object(field: str, *, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
        value = raw.get(field, default if default is not None else {})
        if not isinstance(value, Mapping):
            raise TypeError(f"compiler view {field} must be an object")
        result: dict[str, Any] = {}
        for name, item in value.items():
            canonical = _canonical_abi_name(name, mapping, f"compiler view {field}")
            if canonical in result:
                raise ValueError(f"compiler view {field} aliases one canonical carrier twice")
            result[canonical] = item
        return result

    schedule_values = canonical_object("schedule_values")
    schedule_types = canonical_object("schedule_value_types")
    bound_values = canonical_object("bound_constexpr_values")
    bound_types = canonical_object("bound_constexpr_value_types")
    bound_coordinates = canonical_object("bound_constexpr_shape_coordinates")
    tile_coordinates = canonical_object("schedule_tile_coordinates")
    if set(schedule_types) - set(schedule_values):
        raise ValueError("compiler schedule types must name declared schedule values")
    if any(
        re.fullmatch(r"c[0-9]+", name) is None
        for name in schedule_values
    ):
        raise ValueError("compiler schedule values must use canonical c* handles")
    if set(bound_types) - set(bound_values) or set(bound_coordinates) - set(bound_values):
        raise ValueError("compiler bound constexpr facts must name bound values")
    if set(tile_coordinates) - set(schedule_values):
        raise ValueError("compiler tile coordinates must name C schedule values")

    raw_roles = raw.get("compiler_roles", {})
    if not isinstance(raw_roles, Mapping):
        raise TypeError("compiler view compiler_roles must be an object")
    roles: dict[str, Any] = {}
    for name, value in raw_roles.items():
        canonical = _canonical_abi_name(
            name, mapping, "compiler view compiler_roles"
        )
        if canonical in roles:
            raise ValueError(
                "compiler view compiler_roles aliases one canonical slot twice"
            )
        roles[canonical] = value
    for source_or_canonical, canonical in zip(
        raw.get("compiler_slots", ()), compiler_slots, strict=True
    ):
        if canonical not in roles:
            # Original views do not encode a separate role map.  Their
            # compiler-slot spelling is the trusted public role; an already
            # anonymous view supplies the same value through compiler_roles.
            roles[canonical] = source_or_canonical
    for canonical in schedule_values:
        roles.setdefault(canonical, "constexpr_parameter")
    if set(compiler_slots) - set(schedule_values):
        raise ValueError(
            "compiler slots must be present in the canonical schedule-value ledger"
        )
    if set(roles) != set(schedule_values) or any(
        not isinstance(role, str) or not role
        for role in roles.values()
    ):
        raise ValueError(
            "compiler roles must cover exactly the canonical C schedule slots"
        )

    raw_aliases = raw.get("aliases", ())
    if not isinstance(raw_aliases, Sequence) or isinstance(raw_aliases, (str, bytes)):
        raise TypeError("compiler view aliases must be an ordered array")
    aliases: list[dict[str, Any]] = []
    for index, item in enumerate(raw_aliases):
        alias = _copy_mapping(item, f"compiler view aliases[{index}]")
        for alias_field in ("left", "right"):
            if alias_field in alias:
                alias[alias_field] = _canonical_abi_name(
                    alias[alias_field],
                    mapping,
                    f"compiler view aliases[{index}].{alias_field}",
                )
        aliases.append(alias)

    grid_contract = raw.get("grid_contract", ())
    if not isinstance(grid_contract, Sequence) or isinstance(grid_contract, (str, bytes)):
        raise TypeError("compiler view grid_contract must be an ordered array")
    replace_identifier = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    canonical_grid_contract = [
        replace_identifier.sub(
            lambda match: mapping.get(match.group(0), match.group(0)),
            item,
        )
        if isinstance(item, str) else item
        for item in grid_contract
    ]
    return {
        "schedule_values": schedule_values,
        "schedule_value_types": schedule_types,
        "bound_constexpr_values": bound_values,
        "bound_constexpr_value_types": bound_types,
        "bound_constexpr_shape_coordinates": bound_coordinates,
        "compiler_slots": list(compiler_slots),
        "compiler_roles": roles,
        "schedule_tile_coordinates": tile_coordinates,
        "host_launch_contract": project_host_launch_contract(raw.get("host_launch_contract", {}), mapping),
        "aliases": aliases,
        "arguments": [dict(item) for item in arguments],
        "shape_coordinates": _copy_mapping(
            raw.get("shape_coordinates", {}), "compiler view shape_coordinates"
        ),
        "grid": list(raw.get("grid", ())),
        "grid_contract": canonical_grid_contract,
        "grid_shape_coordinates": list(raw.get("grid_shape_coordinates", ())),
        "parameter_order": list(parameter_order),
        "constexpr_parameters": list(constexpr_parameters),
    }


def _bounded_source_expression(value: ast.AST | None) -> str | None:
    if value is None:
        return None
    try:
        rendered = ast.unparse(value)
    except Exception:
        return None
    return rendered if len(rendered) <= 512 else rendered[:512]


def _address_profile(polynomial: Any) -> tuple[int, int, tuple[int, ...]] | None:
    """Return the closed direct-address profile used by Source/J matching."""

    if not isinstance(polynomial, (list, tuple)):
        return None
    dynamic_terms = 0
    constants: list[int] = []
    for term in polynomial:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            return None
        monomial, coefficient = term
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            return None
        if not isinstance(monomial, (list, tuple)) or any(
            not isinstance(atom, str) for atom in monomial
        ):
            return None
        if not monomial:
            constants.append(coefficient)
            continue
        # Admit only atoms whose identity and integer type were closed by the
        # TTIR observer. Runtime scalar arguments are legitimate standalone
        # offsets; they need not be multiplied by a program/lane term. Unknown
        # region or SSA labels remain unavailable even when another atom in the
        # same monomial is recognized.
        if any(
            re.fullmatch(r"arg\[[0-9]+\]:.+", atom) is None
            and re.fullmatch(r"program_axis:[xyz]", atom) is None
            and not atom.startswith("lane:")
            and not atom.startswith(
                ("integer_relation:", "integer_memory_value:")
            )
            and re.fullmatch(r"region\[scf\.for\]\.arg\[0\]", atom) is None
            for atom in monomial
        ):
            return None
        dynamic_terms += 1
    return dynamic_terms, len(constants), tuple(
        sorted(item for item in constants if item != 0)
    )


AddressTerms = tuple[tuple[tuple[str, ...], int], ...]
AddressTypeLayer = tuple[AddressTerms, str, tuple[str, ...]]
AddressTypeProfile = tuple[AddressTypeLayer, ...]


def _canonical_address_terms(value: Any) -> AddressTerms | None:
    """Accept one closed integer-address polynomial, never a partial row."""

    if not isinstance(value, (list, tuple)):
        return None
    terms: list[tuple[tuple[str, ...], int]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return None
        atoms, coefficient = row
        if (
            not isinstance(atoms, (list, tuple))
            or any(not isinstance(atom, str) or not atom for atom in atoms)
            or isinstance(coefficient, bool)
            or not isinstance(coefficient, int)
            or coefficient == 0
        ):
            return None
        terms.append((tuple(atoms), coefficient))
    return tuple(sorted(terms))


def _combine_address_terms(layers: AddressTypeProfile) -> AddressTerms:
    """Return the exact sum of all typed addptr-layer offset polynomials."""

    combined: dict[tuple[str, ...], int] = {}
    for terms, _index_type, _chain in layers:
        for atoms, coefficient in terms:
            combined[atoms] = combined.get(atoms, 0) + coefficient
            if combined[atoms] == 0:
                del combined[atoms]
    return tuple(sorted(combined.items()))


def _coerce_address_type_profile(value: Any) -> AddressTypeProfile | None:
    """Parse one exact per-addptr typed-address proof profile.

    Every row is independently compiler-proven: the offset polynomial added by
    that ``tt.addptr``, its explicit integer/index type, and a bounded
    type-preserving SSA operation chain ending at the offset value. The host
    does not invent casts or merge type evidence from different rows.
    """

    if not isinstance(value, (list, tuple)):
        return None
    layers: list[AddressTypeLayer] = []
    for row in value:
        if isinstance(row, Mapping):
            if set(row) != {"address_terms", "index_type", "ssa_type_chain"}:
                return None
            raw_terms = row["address_terms"]
            if isinstance(raw_terms, list) and all(
                isinstance(term, Mapping)
                and set(term) == {"refs", "coefficient"}
                for term in raw_terms
            ):
                raw_terms = [
                    (term["refs"], term["coefficient"])
                    for term in raw_terms
                ]
            index_type = row["index_type"]
            raw_chain = row["ssa_type_chain"]
        elif isinstance(row, (list, tuple)) and len(row) == 3:
            raw_terms, index_type, raw_chain = row
        else:
            return None
        terms = _canonical_address_terms(raw_terms)
        if (
            not terms
            or not isinstance(index_type, str)
            or not re.fullmatch(r"(?:i[0-9]+|index)", index_type)
            or not isinstance(raw_chain, (list, tuple))
            or not raw_chain
            or any(not isinstance(item, str) or not item for item in raw_chain)
        ):
            return None
        layers.append((terms, index_type, tuple(raw_chain)))
    return tuple(layers)


def _address_type_profile_wire(profile: AddressTypeProfile) -> list[dict[str, Any]]:
    """Serialize host-owned typed-address proof without SSA value spellings."""

    return [
        {
            "address_terms": [
                {"refs": list(atoms), "coefficient": coefficient}
                for atoms, coefficient in terms
            ],
            "index_type": index_type,
            "ssa_type_chain": list(chain),
        }
        for terms, index_type, chain in profile
    ]


@dataclass(frozen=True)
class SourceFacts:
    """One canonical, privacy-safe Source/Harness fact set.

    ``arguments`` retain declaration order because that order is part of the
    ABI.  The actual Source text is never stored; only its digests and the
    selected symbol are retained.
    """

    frontend: str
    source_kind: str
    source_symbol: str
    parameter_order: tuple[str, ...]
    constexpr_parameters: tuple[str, ...] = ()
    source_to_opaque: tuple[tuple[str, str], ...] = ()
    arguments: tuple[Mapping[str, Any], ...] = ()
    compiler_slots: tuple[str, ...] = ()
    shape_coordinates: tuple[tuple[str, int], ...] = ()
    compiler_facts: Mapping[str, Any] = field(default_factory=dict)
    source_digest: str | None = None
    source_slice_digest: str | None = None

    def __post_init__(self) -> None:
        for name, label in (
            (self.frontend, "frontend"),
            (self.source_kind, "source_kind"),
            (self.source_symbol, "source_symbol"),
        ):
            if not isinstance(name, str) or not name:
                raise ValueError(f"SourceFacts.{label} must be non-empty")
        parameter_order = tuple(self.parameter_order)
        constexpr = tuple(self.constexpr_parameters)
        if (
            len(parameter_order) != len(set(parameter_order))
            or any(_ABI_HANDLE_RE.fullmatch(item) is None for item in parameter_order)
        ):
            raise ValueError(
                "SourceFacts parameter order must contain unique p*/c* handles"
            )
        if len(constexpr) != len(set(constexpr)) or not set(constexpr) <= set(parameter_order):
            raise ValueError("SourceFacts constexpr order must be a parameter subset")
        mapping = tuple(sorted(self.source_to_opaque, key=lambda item: item[0]))
        if len({source for source, _opaque in mapping}) != len(mapping):
            raise ValueError("SourceFacts source mapping has duplicate source names")
        if len({opaque for _source, opaque in mapping}) != len(mapping):
            raise ValueError("SourceFacts source mapping has duplicate opaque names")
        if any(
            not isinstance(source, str)
            or not isinstance(opaque, str)
            or not source.isidentifier()
            or _ABI_HANDLE_RE.fullmatch(opaque) is None
            for source, opaque in mapping
        ):
            raise ValueError("SourceFacts source mapping must use p*/c* ABI handles")
        slots = tuple(self.compiler_slots)
        if set(parameter_order) & set(slots):
            raise ValueError(
                "SourceFacts parameter and compiler ABI declarations must not overlap"
            )
        expected_abi = set(parameter_order) | set(slots)
        mapped_abi = {opaque for _source, opaque in mapping}
        if mapped_abi != expected_abi:
            raise ValueError(
                "SourceFacts source mapping must cover parameter and compiler ABI exactly"
            )
        parameter_p_order = tuple(
            name for name in parameter_order if name.startswith("p")
        )
        parameter_c_order = tuple(
            name for name in parameter_order if name.startswith("c")
        )
        mapped_p_order = tuple(
            sorted(
                (name for name in expected_abi if name.startswith("p")),
                key=lambda name: int(name[1:]),
            )
        )
        mapped_c_order = tuple(
            sorted(
                (name for name in expected_abi if name.startswith("c")),
                key=lambda name: int(name[1:]),
            )
        )
        if parameter_p_order != mapped_p_order:
            raise ValueError(
                "SourceFacts p* mapping must preserve runtime parameter order"
            )
        if (*parameter_c_order, *slots) != mapped_c_order:
            raise ValueError(
                "SourceFacts c* mapping must preserve Source-C then compiler-slot order"
            )
        if parameter_p_order != CanonicalHandleAllocator.canonical_names(
            "p", len(parameter_p_order)
        ):
            raise ValueError("SourceFacts p* handles must be dense and declaration-ordered")
        if mapped_c_order != CanonicalHandleAllocator.canonical_names(
            "c", len(mapped_c_order)
        ):
            raise ValueError("SourceFacts c* handles must be dense and declaration-ordered")
        runtime_parameter_order = tuple(
            name for name in parameter_order if name not in constexpr
        )
        arguments = tuple(
            _copy_mapping(item, "SourceFacts.arguments[]")
            for item in self.arguments
        )
        argument_names = tuple(
            item.get("name") for item in arguments if isinstance(item.get("name"), str)
        )
        if len(argument_names) != len(arguments) or len(argument_names) != len(set(argument_names)):
            raise ValueError("SourceFacts arguments must have unique names")
        if argument_names != runtime_parameter_order:
            raise ValueError(
                "SourceFacts runtime arguments must match parameter_order minus constexpr parameters"
            )
        if len(slots) != len(set(slots)) or any(
            not isinstance(name, str) or re.fullmatch(r"c[0-9]+", name) is None
            for name in slots
        ):
            raise ValueError("SourceFacts compiler slots must use unique c* handles")
        coordinates = tuple(self.shape_coordinates)
        if len({name for name, _value in coordinates}) != len(coordinates):
            raise ValueError("SourceFacts shape coordinates must be unique")
        if any(
            not isinstance(name, str)
            or re.fullmatch(r"h[0-9]+", name) is None
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in coordinates
        ):
            raise ValueError("SourceFacts shape coordinates are invalid")
        compiler_facts = _copy_mapping(
            self.compiler_facts, "SourceFacts.compiler_facts"
        )
        # Older closed SourceFacts predate Source-proven tile-coordinate facts.
        # Absence means no tile relation, never permission to infer one.
        compiler_facts.setdefault("schedule_tile_coordinates", {})
        compiler_facts.setdefault("host_launch_contract", {})
        expected_compiler_fact_keys = {
            "schedule_values", "schedule_value_types",
            "bound_constexpr_values", "bound_constexpr_value_types",
            "bound_constexpr_shape_coordinates", "compiler_slots",
            "compiler_roles", "aliases", "arguments", "shape_coordinates",
            "schedule_tile_coordinates", "grid", "grid_contract",
            "host_launch_contract",
            "grid_shape_coordinates",
            "parameter_order", "constexpr_parameters",
        }
        if set(compiler_facts) != expected_compiler_fact_keys:
            raise ValueError("SourceFacts compiler facts have an incomplete wire shape")
        if tuple(compiler_facts["parameter_order"]) != parameter_order:
            raise ValueError("SourceFacts compiler facts disagree on parameter order")
        if tuple(compiler_facts["constexpr_parameters"]) != constexpr:
            raise ValueError("SourceFacts compiler facts disagree on constexpr order")
        if tuple(compiler_facts["compiler_slots"]) != slots:
            raise ValueError("SourceFacts compiler facts disagree on compiler slots")
        schedule_values = compiler_facts["schedule_values"]
        schedule_types = compiler_facts["schedule_value_types"]
        bound_values = compiler_facts["bound_constexpr_values"]
        bound_types = compiler_facts["bound_constexpr_value_types"]
        bound_coordinates = compiler_facts["bound_constexpr_shape_coordinates"]
        compiler_roles = compiler_facts["compiler_roles"]
        tile_coordinates = compiler_facts["schedule_tile_coordinates"]
        if not all(
            isinstance(item, Mapping)
            for item in (
                schedule_values,
                schedule_types,
                bound_values,
                bound_types,
                bound_coordinates,
                compiler_roles,
                tile_coordinates,
            )
        ):
            raise ValueError("SourceFacts compiler facts have malformed ledgers")
        # Value-stable JSON copying sorts object keys, while this particular
        # map is an ABI ledger whose C declaration sequence is semantic.  Its
        # one valid order comes from the dense shared allocator, never from a
        # downstream view or lexical object ordering.
        if set(schedule_values) != set(mapped_c_order):
            raise ValueError(
                "SourceFacts compiler facts disagree on canonical C declaration order"
            )
        if set(schedule_types) != set(mapped_c_order):
            raise ValueError(
                "SourceFacts compiler facts require one dtype for every C value"
            )
        if set(slots) - set(schedule_values):
            raise ValueError(
                "SourceFacts compiler slots must be present in schedule values"
            )
        if set(compiler_roles) != set(mapped_c_order):
            raise ValueError(
                "SourceFacts compiler roles must cover exactly C schedule slots"
            )
        if set(tile_coordinates) - set(mapped_c_order) or any(
            not isinstance(coordinate, str)
            or coordinate not in dict(coordinates)
            for coordinate in tile_coordinates.values()
        ):
            raise ValueError(
                "SourceFacts tile coordinates must bind C slots to shape coordinates"
            )
        if {
            name
            for name, role in compiler_roles.items()
            if role in {"tile_extent", "reduction_tile_extent"}
        } != set(tile_coordinates):
            raise ValueError(
                "SourceFacts tile coordinates and closed C tile roles differ"
            )
        compiler_facts["schedule_values"] = {
            name: schedule_values[name] for name in mapped_c_order
        }
        compiler_facts["schedule_value_types"] = {
            name: schedule_types[name] for name in mapped_c_order
        }
        compiler_facts["compiler_roles"] = {
            name: compiler_roles[name] for name in mapped_c_order
        }
        compiler_facts["schedule_tile_coordinates"] = {
            name: tile_coordinates[name]
            for name in mapped_c_order
            if name in tile_coordinates
        }
        if set(bound_types) - set(bound_values) or set(bound_coordinates) - set(bound_values):
            raise ValueError(
                "SourceFacts bound constexpr ledgers are not closed"
            )
        if tuple(compiler_facts["arguments"]) != arguments:
            raise ValueError("SourceFacts compiler facts disagree on arguments")
        if compiler_facts["host_launch_contract"] and self.source_digest is None:
            raise ValueError("source_host_launch_digest_unavailable")
        validate_host_launch_contract(
            compiler_facts["host_launch_contract"], arguments=arguments,
            schedule_values=schedule_values, shape_coordinates=dict(coordinates),
            tile_coordinates=tile_coordinates, source_digest=self.source_digest,
        )
        if tuple(
            sorted(
                ((name, value) for name, value in compiler_facts["shape_coordinates"].items()),
                key=lambda item: item[0],
            )
        ) != coordinates:
            raise ValueError("SourceFacts compiler facts disagree on shape coordinates")
        object.__setattr__(self, "parameter_order", parameter_order)
        object.__setattr__(self, "constexpr_parameters", constexpr)
        object.__setattr__(self, "source_to_opaque", mapping)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "compiler_slots", slots)
        object.__setattr__(self, "shape_coordinates", coordinates)
        object.__setattr__(self, "compiler_facts", compiler_facts)

    @classmethod
    def from_compiler_view(
        cls,
        view: Any,
        *,
        frontend: str,
        source_kind: str,
        source_symbol: str,
        source_to_opaque: Mapping[str, str] | None = None,
        source_digest: str | None = None,
        source_slice_digest: str | None = None,
    ) -> "SourceFacts":
        raw = _view_json(view)
        raw_mapping = source_to_opaque or {}
        if not isinstance(raw_mapping, Mapping):
            raise TypeError("source_to_opaque must be a mapping")
        mapping: dict[str, str] = {}
        for source, opaque in raw_mapping.items():
            if (
                not isinstance(source, str)
                or not source.isidentifier()
                or not isinstance(opaque, str)
                or _ABI_HANDLE_RE.fullmatch(opaque) is None
            ):
                raise ValueError(
                    "source_to_opaque must map identifiers to p*/c* handles"
                )
            mapping[source] = opaque
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("source_to_opaque must be one-to-one")

        raw_arguments = raw.get("arguments", ())
        if not isinstance(raw_arguments, Sequence) or isinstance(raw_arguments, (str, bytes)):
            raise TypeError("compiler view arguments must be an ordered array")
        arguments: list[dict[str, Any]] = []
        for index, item in enumerate(raw_arguments):
            argument = _copy_mapping(item, f"compiler view argument {index}")
            argument["name"] = _canonical_abi_name(
                argument.get("name"), mapping, f"compiler view argument {index}"
            )
            arguments.append(argument)

        def ordered_names(field: str) -> tuple[str, ...]:
            raw_names = raw.get(field, ())
            if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
                raise TypeError(f"compiler view {field} must be an ordered array")
            return tuple(
                _canonical_abi_name(item, mapping, f"compiler view {field}")
                for item in raw_names
            )

        parameter_order = ordered_names("parameter_order")
        constexpr = ordered_names("constexpr_parameters")
        slots = ordered_names("compiler_slots")
        if any(not name.startswith("c") for name in slots):
            raise ValueError("compiler view compiler_slots must map to c* handles")
        mapped_abi = set(mapping.values())
        for name in (*parameter_order, *slots):
            if name not in mapped_abi:
                if _ABI_HANDLE_RE.fullmatch(name) is None:
                    raise ValueError(
                        "compiler view contains an unmapped ABI name"
                    )
                mapping[name] = name
                mapped_abi.add(name)
        raw_coordinates = raw.get("shape_coordinates", {})
        if not isinstance(raw_coordinates, Mapping):
            raise TypeError("compiler view shape_coordinates must be an object")
        coordinates = tuple(
            sorted(
                (
                    (name, value)
                    for name, value in raw_coordinates.items()
                    if isinstance(name, str)
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ),
                key=lambda item: item[0],
            )
        )
        if len(coordinates) != len(raw_coordinates):
            raise ValueError("compiler view shape_coordinates are invalid")
        compiler_facts = _canonical_compiler_facts(
            raw,
            mapping,
            arguments=arguments,
            parameter_order=parameter_order,
            constexpr_parameters=constexpr,
            compiler_slots=slots,
        )
        return cls(
            frontend=frontend,
            source_kind=source_kind,
            source_symbol=source_symbol,
            parameter_order=parameter_order,
            constexpr_parameters=constexpr,
            source_to_opaque=tuple(mapping.items()),
            arguments=tuple(arguments),
            compiler_slots=slots,
            shape_coordinates=coordinates,
            compiler_facts=compiler_facts,
            source_digest=source_digest,
            source_slice_digest=source_slice_digest,
        )

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SourceFacts":
        raw = _copy_mapping(value, "SourceFacts")
        return cls(
            frontend=raw["frontend"],
            source_kind=raw["source_kind"],
            source_symbol=raw["source_symbol"],
            parameter_order=tuple(raw["parameter_order"]),
            constexpr_parameters=tuple(raw.get("constexpr_parameters", ())),
            source_to_opaque=tuple(
                (row[0], row[1]) for row in raw.get("source_to_opaque", ())
            ),
            arguments=tuple(raw.get("arguments", ())),
            compiler_slots=tuple(raw.get("compiler_slots", ())),
            shape_coordinates=tuple(
                (row[0], row[1]) for row in raw.get("shape_coordinates", ())
            ),
            compiler_facts=raw["compiler_facts"],
            source_digest=raw.get("source_digest"),
            source_slice_digest=raw.get("source_slice_digest"),
        )

    @property
    def opaque_to_source(self) -> Mapping[str, str]:
        return {opaque: source for source, opaque in self.source_to_opaque}

    def argument_by_position(self, position: int | None) -> Mapping[str, Any] | None:
        if position is None or position < 0 or position >= len(self.arguments):
            return None
        return self.arguments[position]

    def logical_ref_for_position(self, position: int | None) -> str | None:
        argument = self.argument_by_position(position)
        value = argument.get("logical_ref") if argument is not None else None
        return value if isinstance(value, str) else None

    def rank_for_logical(self, logical_ref: str | None) -> int | None:
        if not isinstance(logical_ref, str):
            return None
        for argument in self.arguments:
            if argument.get("logical_ref") != logical_ref:
                continue
            shape = argument.get("shape")
            if isinstance(shape, list):
                return len(shape)
            rank = argument.get("rank")
            return rank if isinstance(rank, int) and not isinstance(rank, bool) else None
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "frontend": self.frontend,
            "source_kind": self.source_kind,
            "source_symbol": self.source_symbol,
            "parameter_order": list(self.parameter_order),
            "constexpr_parameters": list(self.constexpr_parameters),
            "source_to_opaque": [list(row) for row in self.source_to_opaque],
            "arguments": [dict(row) for row in self.arguments],
            "compiler_slots": list(self.compiler_slots),
            "shape_coordinates": [list(row) for row in self.shape_coordinates],
            "compiler_facts": dict(self.compiler_facts),
            "source_digest": self.source_digest,
            "source_slice_digest": self.source_slice_digest,
        }

    def compiler_view(self) -> dict[str, Any]:
        """Return the sole canonical compiler-side projection for consumers."""

        return _copy_mapping(self.compiler_facts, "SourceFacts.compiler_facts")

    @property
    def digest(self) -> str:
        return _digest(self.to_json())


def source_logical_scalar_parameters(
    facts: SourceFacts,
    source_text: str | None,
) -> tuple[str, ...]:
    """Return Source-proven scalar ABI carriers used as logical values.

    The ABI manifest tells us which parameters are scalars, but not whether a
    scalar is only used to form an address or is consumed by the logical
    computation.  This small Source-side binder makes that distinction once,
    before MCP/J construction.  It is deliberately conservative: a scalar is
    promoted only when a Python AST occurrence is in a value expression; a
    scalar seen only in pointer-update statements remains an ABI carrier.

    This is occurrence evidence, not a semantic operator inference.  CUDA/C++
    text and unparsable sources return no promotions and remain provider-owned
    under the existing fail-closed contract.
    """

    if not isinstance(facts, SourceFacts) or not isinstance(source_text, str):
        return ()
    scalar_handles = {
        item.get("name")
        for item in facts.arguments
        if isinstance(item, Mapping)
        and item.get("kind") == "scalar"
        and isinstance(item.get("name"), str)
    }
    # A bound non-C constexpr can retain a physical ``p*`` ABI carrier while
    # being absent from ``facts.arguments``.  It controls source specialization
    # rather than providing a run-time J value; promoting it merely because it
    # appears in an AST condition creates a dead rank-zero logical input.  The
    # argument manifest is the sole authority for values which may enter J.
    if not scalar_handles:
        return ()
    opaque_to_source = facts.opaque_to_source
    accepted_names = {
        name for handle in scalar_handles
        for name in (handle, opaque_to_source.get(handle))
        if isinstance(name, str)
    }
    pointer_names = {
        item.get("name")
        for item in facts.arguments
        if isinstance(item, Mapping)
        and item.get("kind") in {"tensor", "pointer"}
        and isinstance(item.get("name"), str)
    }
    pointer_names |= {
        opaque_to_source.get(name, name)
        for name in pointer_names
        if isinstance(name, str)
    }
    try:
        tree = ast.parse(source_text, mode="exec")
    except (SyntaxError, TypeError, ValueError):
        return ()
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == facts.source_symbol
    ]
    if len(functions) != 1:
        return ()
    function = functions[0]
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(function):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def names_in(node: ast.AST) -> set[str]:
        return {
            item.id for item in ast.walk(node)
            if isinstance(item, ast.Name)
        }

    def statement_root(node: ast.AST) -> ast.AST | None:
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, ast.stmt):
                return current
            current = parents.get(current)
        return None

    def is_pointer_update(node: ast.AST) -> bool:
        statement = statement_root(node)
        if isinstance(statement, ast.AugAssign):
            targets = names_in(statement.target)
            if targets & pointer_names:
                return True
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                names_in(statement.target)
                if isinstance(statement, ast.AnnAssign)
                else {
                    name for target in statement.targets for name in names_in(target)
                }
            )
            if targets & pointer_names:
                return True
        return False

    def is_memory_address_context(node: ast.AST) -> bool:
        """Return whether this occurrence contributes only to a memory address.

        A scalar inside the pointer operand of ``tl.load``/``tl.store`` (or
        block-pointer construction) is physically necessary but does not by
        itself establish a J rank-zero value.  The prior generic Call/BinOp
        walk promoted such address strides as logical scalars, leaving a dead
        ``t*`` input that strict TypeCheck correctly rejects.  Keep this
        source-side distinction occurrence-local: the same scalar is still
        promoted when it appears in a load value, predicate, result, or other
        non-address expression elsewhere.
        """

        current: ast.AST | None = node
        address_calls = {"load", "store", "make_block_ptr", "advance"}
        while current is not None:
            parent = parents.get(current)
            if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute):
                name = parent.func.attr
                if name in address_calls or name.startswith("atomic_"):
                    pointer_arguments: list[ast.AST] = []
                    if parent.args:
                        pointer_arguments.append(parent.args[0])
                    pointer_arguments.extend(
                        keyword.value
                        for keyword in parent.keywords
                        if keyword.arg in {"pointer", "ptr", "base"}
                    )
                    if any(
                        any(descendant is node for descendant in ast.walk(argument))
                        for argument in pointer_arguments
                    ):
                        return True
            current = parent
        return False

    def has_value_context(node: ast.AST) -> bool:
        if is_memory_address_context(node):
            return False
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, ast.Compare | ast.BinOp | ast.BoolOp | ast.UnaryOp):
                return True
            if isinstance(current, ast.Call):
                # A scalar receiver such as ``scale.to(dtype)`` is a value;
                # arange/grid arguments are shape or realization facts only.
                func = current.func
                if isinstance(func, ast.Attribute) and func.attr == "to":
                    return True
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr in {"arange", "program_id"}
                ):
                    return True
            if isinstance(current, ast.Subscript):
                return True
            current = parents.get(current)
        return False

    logical: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if node.id not in accepted_names or is_pointer_update(node):
            continue
        if has_value_context(node):
            canonical = next(
                (handle for handle in scalar_handles
                 if node.id in {handle, opaque_to_source.get(handle)}),
                None,
            )
            if canonical is not None:
                logical.add(canonical)
    parameter_order = tuple(
        name for name in facts.parameter_order
        if name in logical
    )
    return parameter_order

@dataclass(frozen=True)
class TransferOccurrence:
    """One lexical Source memory occurrence, independent of J numbering."""

    ordinal: int
    operation: str
    kind: str
    masked: bool
    predicate: str | None
    fill_value: str | None
    pointer_base: str | None
    pointer_position: int | None
    origin_positions: tuple[int, ...]
    context: tuple[str, ...]
    address_terms: tuple[tuple[tuple[str, ...], int], ...]
    address_profile: tuple[int, int, tuple[int, ...]] | None
    address_type_profile: AddressTypeProfile = ()
    logical_ref: str | None = None
    tensor_rank: int | None = None
    element_type: str | None = None
    semantic_consumer: SourceSemanticOccurrence | None = None
    semantic_predicate: SourceSemanticOccurrence | None = None
    semantic_access: SourceSemanticOccurrence | None = None

    def __post_init__(self) -> None:
        if (
            not is_canonical_ordinal(self.ordinal)
            or self.kind not in {"load", "store", "update"}
        ):
            raise ValueError("invalid Source transfer occurrence")
        if not isinstance(self.masked, bool):
            raise TypeError("Source transfer masked flag must be boolean")
        if self.element_type is not None and (
            not isinstance(self.element_type, str) or not self.element_type
        ):
            raise ValueError("invalid Source transfer element type")
        if self.semantic_consumer is not None:
            if not isinstance(self.semantic_consumer, SourceSemanticOccurrence):
                raise TypeError("Source transfer consumer occurrence is invalid")
            if self.semantic_consumer.role != "consumer":
                raise ValueError("Source transfer consumer occurrence has invalid role")
        if self.semantic_predicate is not None:
            if not isinstance(self.semantic_predicate, SourceSemanticOccurrence):
                raise TypeError("Source transfer predicate occurrence is invalid")
            if self.semantic_predicate.role != "predicate":
                raise ValueError("Source transfer predicate occurrence has invalid role")
        if self.semantic_access is not None:
            if not isinstance(self.semantic_access, SourceSemanticOccurrence):
                raise TypeError("Source transfer access occurrence is invalid")
            if self.semantic_access.role != "access":
                raise ValueError("Source transfer access occurrence has invalid role")
            if self.semantic_access.operation_family != f"memory_{self.kind}":
                raise ValueError("Source transfer access occurrence has invalid family")
        address_terms = _canonical_address_terms(self.address_terms)
        address_type_profile = _coerce_address_type_profile(
            self.address_type_profile
        )
        if address_terms is None or address_type_profile is None:
            raise ValueError("invalid Source transfer typed address profile")
        if address_terms != _combine_address_terms(address_type_profile):
            raise ValueError(
                "Source transfer typed-address layers differ from address terms"
            )
        object.__setattr__(self, "address_terms", address_terms)
        object.__setattr__(self, "address_type_profile", address_type_profile)
        if self.address_profile is not None:
            dynamic, constants, offsets = self.address_profile
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in (dynamic, constants)
            ) or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets):
                raise ValueError("invalid Source address profile")

    def to_json(self) -> dict[str, Any]:
        profile = self.address_profile
        return {
            "ordinal": self.ordinal,
            "operation": self.operation,
            "kind": self.kind,
            "masked": self.masked,
            "predicate": self.predicate,
            "fill_value": self.fill_value,
            "pointer_base": self.pointer_base,
            "pointer_position": self.pointer_position,
            "origin_positions": list(self.origin_positions),
            "context": list(self.context),
            "address_terms": [
                [list(atoms), coefficient]
                for atoms, coefficient in self.address_terms
            ],
            "address_profile": (
                {"dynamic_terms": profile[0], "constant_terms": profile[1], "constant_offsets": list(profile[2])}
                if profile is not None else None
            ),
            "logical_ref": self.logical_ref,
            "tensor_rank": self.tensor_rank,
            "element_type": self.element_type,
            "semantic_consumer": (
                self.semantic_consumer.to_json()
                if self.semantic_consumer is not None else None
            ),
            "semantic_predicate": (
                self.semantic_predicate.to_json()
                if self.semantic_predicate is not None else None
            ),
            "semantic_access": (
                self.semantic_access.to_json()
                if self.semantic_access is not None else None
            ),
            "address_type_profile": _address_type_profile_wire(
                self.address_type_profile
            ),
        }


@dataclass(frozen=True)
class SourceTransferLedger:
    """The sole Source occurrence ledger used by Source/J reconciliation."""

    source_facts_digest: str | None
    status: str
    reason: str
    occurrences: tuple[TransferOccurrence, ...] = ()
    bindings: tuple[Mapping[str, Any], ...] = ()
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"pass", "unavailable", "ambiguous"}:
            raise ValueError("invalid SourceTransferLedger status")
        if self.source_facts_digest is not None and not isinstance(
            self.source_facts_digest, str
        ):
            raise TypeError("SourceTransferLedger source_facts_digest must be a string")
        if not isinstance(self.reason, str):
            raise TypeError("SourceTransferLedger reason must be a string")
        occurrences = tuple(self.occurrences)
        for occurrence in occurrences:
            if (
                not isinstance(occurrence.operation, str)
                or not occurrence.operation
                or occurrence.predicate is not None
                and not isinstance(occurrence.predicate, str)
                or occurrence.fill_value is not None
                and not isinstance(occurrence.fill_value, str)
                or occurrence.pointer_base is not None
                and not isinstance(occurrence.pointer_base, str)
                or any(
                    isinstance(position, bool) or not isinstance(position, int)
                    for position in occurrence.origin_positions
                )
                or any(not isinstance(item, str) for item in occurrence.context)
            ):
                raise ValueError("Source transfer occurrence has invalid metadata")
            if self.status == "pass" and (
                occurrence.pointer_base is None
                or occurrence.element_type is None
                or occurrence.address_terms
                and not occurrence.address_type_profile
            ):
                raise ValueError(
                    "pass Source transfer ledger requires a closed typed address"
                )
            # A pass ledger is the closed Source occurrence authority.  A
            # masked load without an explicit Source ``other=`` value is not
            # an empty-fill fact: it is an unproved occurrence and must be
            # rejected at ledger construction, including JSON rehydration.
            # ``from_source_text`` deliberately uses ``ambiguous`` for its
            # syntax-only navigation projection, so it is not accidentally
            # promoted by this check.
            if (
                self.status == "pass"
                and occurrence.kind == "load"
                and occurrence.masked
                and occurrence.fill_value is None
            ):
                raise ValueError(
                    "pass Source transfer ledger requires masked-load fill_value"
                )
        if tuple(item.ordinal for item in occurrences) != tuple(range(len(occurrences))):
            raise ValueError("Source transfer occurrence ordinals must be contiguous")
        if len({item.ordinal for item in occurrences}) != len(occurrences):
            raise ValueError("Source transfer occurrence ordinals must be unique")
        if self.status == "pass" and occurrences and not self.bindings:
            raise ValueError(
                "pass Source transfer ledger requires occurrence bindings"
            )
        normalized_bindings: list[dict[str, Any]] = []
        for item in self.bindings:
            copied = _copy_mapping(item, "SourceTransferLedger.binding")
            if "address_profile" in copied:
                profile = copied["address_profile"]
                if isinstance(profile, Mapping):
                    profile = (
                        profile.get("dynamic_terms"),
                        profile.get("constant_terms"),
                        tuple(profile.get("constant_offsets", ())),
                    )
                elif isinstance(profile, list):
                    profile = (
                        profile[0],
                        profile[1],
                        tuple(profile[2]),
                    ) if len(profile) == 3 else None
                elif isinstance(profile, tuple) and len(profile) == 3:
                    profile = (profile[0], profile[1], tuple(profile[2]))
                copied["address_profile"] = profile
            if "source_address_type_profile" in copied:
                typed_profile = _coerce_address_type_profile(
                    copied["source_address_type_profile"]
                )
                copied["source_address_type_profile"] = typed_profile
            normalized_bindings.append(copied)
        object.__setattr__(self, "bindings", tuple(normalized_bindings))
        if self.bindings:
            if len(self.bindings) != len(occurrences):
                raise ValueError(
                    "Source transfer bindings must cover every occurrence exactly once"
                )
            seen_ordinals: set[int] = set()
            seen_accesses: set[str] = set()
            seen_effects: set[str] = set()
            for row in self.bindings:
                ordinal = row.get("ordinal")
                access_ref = row.get("access_ref")
                effect_id = row.get("effect_id")
                if (
                    not is_canonical_ordinal(ordinal)
                    or ordinal >= len(occurrences)
                    or ordinal in seen_ordinals
                    or not isinstance(access_ref, str)
                    or not access_ref
                    or re.fullmatch(r"a[0-9]+", access_ref) is None
                    or access_ref in seen_accesses
                    or not isinstance(effect_id, str)
                    or not effect_id
                    or re.fullmatch(r"e[0-9]+", effect_id) is None
                    or effect_id in seen_effects
                ):
                    raise ValueError(
                        "Source transfer bindings must have unique ordinal/access_ref/effect_id"
                    )
                occurrence = occurrences[ordinal]
                occurrence_profile = occurrence.address_profile
                expected_address_profile = (
                    {
                        "dynamic_terms": occurrence_profile[0],
                        "constant_terms": occurrence_profile[1],
                        "constant_offsets": list(occurrence_profile[2]),
                    }
                    if occurrence_profile is not None
                    else None
                )
                expected_index_expression = {
                    "address_terms": [
                        {"refs": list(atoms), "coefficient": coefficient}
                        for atoms, coefficient in occurrence.address_terms
                    ],
                    "address_profile": expected_address_profile,
                }
                if (
                    any(
                        key not in row
                        for key in (
                            "kind",
                            "tensor",
                            "masked",
                            "address_profile",
                            "node_ref",
                            "predicate",
                            "fill_value",
                            "source_other",
                            "source_predicate",
                            "source_pointer_base",
                            "source_element_type",
                            "source_address_type_profile",
                            "semantic_authorizations",
                            "index_expression",
                            "effect_order",
                            "pi_binding",
                        )
                    )
                    or row.get("kind") != occurrence.kind
                    or not isinstance(row.get("tensor"), str)
                    or re.fullmatch(r"t[0-9]+", row["tensor"]) is None
                    or row.get("masked") is not occurrence.masked
                    or row.get("source_other") != occurrence.fill_value
                    or row.get("source_predicate") != occurrence.predicate
                    or row.get("source_pointer_base") != occurrence.pointer_base
                    or row.get("source_element_type") != occurrence.element_type
                    or row.get("source_address_type_profile")
                    != occurrence.address_type_profile
                    or row.get("address_profile") != occurrence.address_profile
                    or (row.get("predicate") is not None) != occurrence.masked
                    or (
                        row.get("node_ref") is not None
                        and (
                            not isinstance(row.get("node_ref"), str)
                            or re.fullmatch(r"n[0-9]+", row["node_ref"]) is None
                        )
                    )
                    or (
                        occurrence.logical_ref is not None
                        and row.get("tensor") != occurrence.logical_ref
                    )
                    or row.get("index_expression") != expected_index_expression
                    or not isinstance(row.get("semantic_authorizations"), Mapping)
                    or (
                        "effect_order" in row
                        and row.get("effect_order") != ordinal
                    )
                    or (
                        "pi_binding" in row
                        and (
                            not isinstance(row.get("pi_binding"), Mapping)
                            or row["pi_binding"].get("status")
                            != "source_address_owner_bound"
                            or row["pi_binding"].get("tensor")
                            != occurrence.logical_ref
                            or row["pi_binding"].get("access_ref")
                            != access_ref
                            or row["pi_binding"].get("effect_id")
                            != effect_id
                            or row["pi_binding"].get("pointer_position")
                            != occurrence.pointer_position
                            or row["pi_binding"].get("element_type")
                            != occurrence.element_type
                            or row["pi_binding"].get("address_type_profile")
                            != _address_type_profile_wire(
                                occurrence.address_type_profile
                            )
                            or row["pi_binding"].get("address_profile")
                            != expected_address_profile
                        )
                    )
                ):
                    raise ValueError(
                        "Source transfer binding disagrees with its occurrence"
                    )
                authorizations = row["semantic_authorizations"]
                if set(authorizations) != {"access_view", "owner", "computed_guard"}:
                    raise ValueError("Source transfer authorizations are incomplete")
                access_view = authorizations["access_view"]
                owner = authorizations["owner"]
                guard = authorizations["computed_guard"]
                if not isinstance(access_view, Mapping) or not isinstance(owner, Mapping) or not isinstance(guard, Mapping):
                    raise ValueError("Source transfer authorization is malformed")
                semantic_access = occurrence.semantic_access
                if (
                    semantic_access is None
                    or set(access_view) != {"status", "token", "operation_family", "compiler_attestation"}
                    or access_view.get("status") != "pass"
                    or access_view.get("token") != semantic_access.token
                    or access_view.get("operation_family") != semantic_access.operation_family
                    or not isinstance(access_view.get("compiler_attestation"), str)
                    or re.fullmatch(r"sc_[0-9a-f]{32}", access_view["compiler_attestation"]) is None
                ):
                    raise ValueError("Source transfer access-view authorization is invalid")
                if owner.get("status") == "pass":
                    semantic = occurrence.semantic_consumer
                    token = owner.get("token")
                    outline = owner.get("authoring_outline")
                    outline_valid = True
                    if outline is not None:
                        try:
                            outline_valid = (
                                isinstance(outline, Mapping)
                                and set(outline) == {"store", "graph"}
                                and isinstance(outline["store"], Mapping)
                                and isinstance(outline["graph"], list)
                                and tensor_owner_digest(outline["store"], outline["graph"])
                                == token.rsplit("_", 1)[1]
                            )
                        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                            outline_valid = False
                    if (
                        occurrence.kind != "store" or semantic is None
                        or set(owner) != {"status", "token", "operation_family", "compiler_attestation"} | ({"authoring_outline"} if outline is not None else set())
                        or not outline_valid
                        or owner.get("operation_family") != semantic.operation_family
                        or not isinstance(token, str)
                        or re.fullmatch(SOURCE_OWNER_TOKEN_PATTERN, token) is None
                        or not token.startswith(semantic_access.token + "_")
                        or not isinstance(owner.get("compiler_attestation"), str)
                        or re.fullmatch(r"sc_[0-9a-f]{32}", owner["compiler_attestation"]) is None
                        or not isinstance(self.source_facts_digest, str)
                        or owner["compiler_attestation"] != "sc_" + hashlib.sha256(
                            (self.source_facts_digest + "|" + token).encode()
                        ).hexdigest()[:32]
                    ):
                        raise ValueError("source_exact_owner_binding_unavailable")
                elif (
                    set(owner) != {"status", "reason"}
                    or owner.get("status") != "unavailable"
                    or not isinstance(owner.get("reason"), str)
                    or not owner["reason"]
                ):
                    raise ValueError("Source transfer owner authorization is invalid")
                if guard.get("status") == "pass":
                    semantic = occurrence.semantic_predicate
                    if (
                        semantic is None
                        or set(guard) != {
                            "status", "token", "operation_family",
                            "compiler_attestation",
                        }
                        or guard.get("token") != semantic.token
                        or guard.get("operation_family") != semantic.operation_family
                        or not isinstance(guard.get("compiler_attestation"), str)
                        or re.fullmatch(r"sc_[0-9a-f]{32}", guard["compiler_attestation"])
                        is None
                    ):
                        raise ValueError("Source transfer guard authorization is invalid")
                elif (
                    set(guard) != {"status", "reason"}
                    or guard.get("status") != "unavailable"
                    or not isinstance(guard.get("reason"), str)
                    or not guard["reason"]
                ):
                    raise ValueError("Source transfer guard authorization is invalid")
                seen_ordinals.add(ordinal)
                seen_accesses.add(access_ref)
                seen_effects.add(effect_id)
            if tuple(
                row.get("ordinal") for row in normalized_bindings
            ) != tuple(range(len(occurrences))):
                raise ValueError(
                    "Source transfer bindings must preserve Source ordinal order"
                )
            if seen_ordinals != set(range(len(occurrences))):
                raise ValueError(
                    "Source transfer bindings must use contiguous occurrence ordinals"
                )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(_copy_mapping(item, "SourceTransferLedger.diagnostic") for item in self.diagnostics),
        )

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        source_facts: SourceFacts | None = None,
        diagnostics: Sequence[Mapping[str, Any]] = (),
    ) -> "SourceTransferLedger":
        return cls(
            source_facts.digest if source_facts is not None else None,
            "unavailable",
            reason,
            diagnostics=tuple(diagnostics),
        )

    @classmethod
    def from_observation(
        cls,
        observed: Any,
        *,
        source_facts: SourceFacts | None = None,
        logical_bindings: Mapping[str, Any] | None = None,
        lexical_ledger: "SourceTransferLedger | None" = None,
    ) -> "SourceTransferLedger":
        """Build the Source occurrence ledger from compiler sidecars.

        Compiler observations use frontend-local pointer spellings such as
        ``arg[0]`` and SSA predicates.  Those spellings are not a second
        semantic authority: each compiler memory event must instead carry a
        unique debug-location counterpart for the exact lexical Source memory
        call.  Source order, AST order, tensor/shape coincidence, and
        provider-local identity are deliberately not matching keys.
        """

        sidecar_names = (
            "diagnostic_memory_transfers",
            "diagnostic_memory_effect_origins",
            "diagnostic_memory_address_forms",
            "diagnostic_memory_source_locations",
            "diagnostic_memory_predicate_evidence",
            "diagnostic_memory_owner_evidence",
        )
        if not all(hasattr(observed, name) for name in sidecar_names):
            return cls.unavailable(
                "source_memory_occurrence_sidecars_incomplete",
                source_facts=source_facts,
                diagnostics=(
                    {
                        "rule_code": "source_transfer_ledger_unavailable",
                        "path": "<source-transfer-ledger>",
                        "observed_refs_or_types": [
                            "one or more required observation sidecars is absent"
                        ],
                    },
                ),
            )
        transfers = tuple(getattr(observed, sidecar_names[0]) or ())
        origins = tuple(getattr(observed, sidecar_names[1]) or ())
        forms = tuple(getattr(observed, sidecar_names[2]) or ())
        source_locations = tuple(getattr(observed, sidecar_names[3]) or ())
        predicate_evidence = tuple(getattr(observed, sidecar_names[4]) or ())
        owner_evidence = tuple(getattr(observed, sidecar_names[5]) or ())
        if (
            len(transfers) != len(origins)
            or len(origins) != len(forms)
            or len(forms) != len(source_locations)
            or len(source_locations) != len(predicate_evidence)
            or len(predicate_evidence) != len(owner_evidence)
        ):
            return cls.unavailable(
                "source_memory_occurrence_sidecars_incomplete",
                source_facts=source_facts,
                diagnostics=(
                    {
                        "rule_code": "source_transfer_ledger_unavailable",
                        "path": "<source-transfer-ledger>",
                        "observed_refs_or_types": [
                                "transfer sidecars disagree "
                                f"(transfers={len(transfers)} origins={len(origins)} "
                                f"forms={len(forms)} locations={len(source_locations)})"
                        ],
                    },
                ),
            )
        if not isinstance(lexical_ledger, SourceTransferLedger):
            return cls.unavailable(
                "source_memory_occurrence_lexical_ledger_unavailable",
                source_facts=source_facts,
                diagnostics=(
                    {
                        "rule_code": "source_transfer_ledger_unavailable",
                        "path": "<source-transfer-ledger>",
                        "observed_refs_or_types": [
                            "expected:exact lexical Source occurrence ledger"
                        ],
                    },
                ),
            )

        # An observed kernel with no memory operations is a valid empty ledger
        # only when the lexical Source ledger independently proves that it
        # also has no memory occurrence.  Do not turn an absent/ambiguous
        # lexical ledger, or a Source containing an unobserved store/update,
        # into an empty ``pass`` ledger merely because the compiler sidecar is
        # empty.
        if not transfers:
            observed_memory_operations: set[str] = set()
            raw_op_multiset = getattr(observed, "op_multiset", ())
            if isinstance(raw_op_multiset, (tuple, list)):
                for item in raw_op_multiset:
                    operation = item[0] if isinstance(item, (tuple, list)) and item else None
                    if isinstance(operation, str) and (
                        operation == "tt.load"
                        or operation == "tt.store"
                        or operation.startswith("tt.load.")
                        or operation.startswith("tt.store.")
                        or operation.startswith("tt.atomic")
                    ):
                        observed_memory_operations.add(operation)
            if lexical_ledger.status == "pass" and not lexical_ledger.occurrences:
                if observed_memory_operations:
                    if any(
                        operation.startswith("tt.atomic")
                        for operation in observed_memory_operations
                    ):
                        return cls.unavailable(
                            "source_atomic_update_observation_unavailable",
                            source_facts=source_facts,
                            diagnostics=(
                                {
                                    "rule_code": "source_atomic_update_authorization_unavailable",
                                    "path": "<source-ttir-observation>.op_multiset",
                                    "observed_refs_or_types": [
                                        "compiler atomic operation has no typed Source occurrence"
                                    ],
                                },
                            ),
                        )
                    return cls.unavailable(
                        "source_memory_occurrence_lexical_ledger_unavailable",
                        source_facts=source_facts,
                        diagnostics=(
                            {
                                "rule_code": "source_transfer_ledger_unavailable",
                                "path": "<source-ttir-observation>.op_multiset",
                                "observed_refs_or_types": [
                                    "compiler memory operation has no lexical Source occurrence"
                                ],
                            },
                        ),
                    )
                return cls(
                    source_facts.digest if source_facts is not None else None,
                    "pass",
                    "lexical_source_memory_occurrences",
                )
            if any(
                occurrence.kind == "update"
                for occurrence in lexical_ledger.occurrences
            ):
                return cls.unavailable(
                    "source_atomic_update_observation_unavailable",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_atomic_update_authorization_unavailable",
                            "path": "<source-transfer-ledger>.occurrences",
                            "observed_refs_or_types": [
                                "expected:typed atomic pointer/value/compare/result observation"
                            ],
                        },
                    ),
                )
            return cls.unavailable(
                "source_memory_occurrence_observation_unavailable",
                source_facts=source_facts,
                diagnostics=(
                    {
                        "rule_code": "source_transfer_ledger_unavailable",
                        "path": "<source-transfer-ledger>",
                        "observed_refs_or_types": [
                            "Source memory occurrence exists but compiler memory sidecar is empty"
                        ],
                    },
                ),
            )

        # The current TTIR observation emits typed producer sidecars only for
        # ordinary load/store operations.  An atomic Source occurrence must
        # never be paired with a normal ``store`` row merely because the
        # compiler lowered it to a write-like operation: CAS compare,
        # replacement value, observed result, ordering and duplicate policy
        # are separate semantic roles.  Until one observation record carries
        # all of those roles, reject the update explicitly and keep the
        # Source/J ledger closed rather than guessing an operand position.
        if any(
            occurrence.kind == "update"
            for occurrence in lexical_ledger.occurrences
        ):
            return cls.unavailable(
                "source_atomic_update_observation_unavailable",
                source_facts=source_facts,
                diagnostics=(
                    {
                        "rule_code": "source_atomic_update_authorization_unavailable",
                        "path": "<source-transfer-ledger>.occurrences",
                        "observed_refs_or_types": [
                            "expected:typed atomic pointer/value/compare/result observation"
                        ],
                    },
                ),
            )

        def access_location_key(
            *,
            kind: Any,
            masked: Any,
            line: Any,
            column: Any,
            call_path: Any = (),
        ) -> tuple[str, bool, int, int, tuple[tuple[int, int], ...]] | None:
            if (
                kind not in {"load", "store", "update"}
                or not isinstance(masked, bool)
                or isinstance(line, bool)
                or not isinstance(line, int)
                or line <= 0
                or isinstance(column, bool)
                or not isinstance(column, int)
                or column < 0
                or not isinstance(call_path, tuple)
                or any(not isinstance(frame, tuple) or len(frame) != 2 or type(frame[0]) is not int or frame[0] <= 0 or type(frame[1]) is not int or frame[1] < 0 for frame in call_path)
            ):
                return None
            return (kind, masked, line, column, call_path)

        source_to_compiler: dict[tuple[str, bool, int, int, tuple[tuple[int, int], ...]], int] = {}
        for compiler_index, location in enumerate(source_locations):
            if not isinstance(location, tuple) or len(location) != 5:
                return cls.unavailable(
                    "source_compiler_access_location_unavailable",
                    source_facts=source_facts,
                )
            key = access_location_key(
                kind=location[0],
                masked=location[1],
                line=location[2],
                column=location[3],
                call_path=location[4],
            )
            if key is None or key in source_to_compiler:
                return cls.unavailable(
                    "source_compiler_access_counterpart_ambiguous",
                    source_facts=source_facts,
                )
            source_to_compiler[key] = compiler_index

        def valid_predicate_evidence(
            value: Any,
        ) -> tuple[str, bool, int, int, str | None, int | None, int | None, bool, tuple[tuple[int, int], ...] | None, int | None, tuple[tuple[int, int], ...]] | None:
            if not isinstance(value, tuple) or len(value) != 11:
                return None
            kind, masked, line, column, operation, predicate_line, predicate_column, is_i1, predicate_path, producer_id, access_path = value
            key = access_location_key(
                kind=kind,
                masked=masked,
                line=line,
                column=column,
            )
            if key is None or not isinstance(is_i1, bool):
                return None
            if access_location_key(
                kind=kind,
                masked=masked,
                line=line,
                column=column,
                call_path=access_path,
            ) is None:
                return None
            if producer_id is not None and (type(producer_id) is not int or producer_id < 0):
                return None
            if predicate_line is None and predicate_column is None:
                if predicate_path is not None:
                    return None
            elif access_location_key(kind=kind, masked=masked, line=predicate_line, column=predicate_column, call_path=predicate_path) is None:
                return None
            if masked:
                if operation is None:
                    if (
                        predicate_line is not None
                        or predicate_column is not None
                        or is_i1
                        or producer_id is not None
                    ):
                        return None
                elif not isinstance(operation, str) or not operation or producer_id is None:
                    return None
                elif predicate_line is None and predicate_column is None:
                    # A direct predicate producer can survive compiler
                    # lowering without a standalone debug location.  Its
                    # only admissible consumer is the exact matched memory
                    # access that carried this sidecar; Source authorization
                    # must additionally prove a lexical def-use edge.
                    pass
                elif (
                    isinstance(predicate_line, bool)
                    or not isinstance(predicate_line, int)
                    or predicate_line <= 0
                    or isinstance(predicate_column, bool)
                    or not isinstance(predicate_column, int)
                    or predicate_column < 0
                ):
                    return None
            elif any(item is not None for item in (operation, predicate_line, predicate_column, producer_id)):
                return None
            return (
                kind, masked, line, column, operation, predicate_line,
                predicate_column, is_i1, predicate_path, producer_id,
                access_path,
            )

        normalized_predicate_evidence = tuple(
            valid_predicate_evidence(item) for item in predicate_evidence
        )
        if any(item is None for item in normalized_predicate_evidence):
            return cls.unavailable(
                "source_compiler_predicate_evidence_unavailable",
                source_facts=source_facts,
            )

        def valid_owner_evidence(
            value: Any,
        ) -> tuple[str, bool, int, int, str | None, int | None, int | None, bool, tuple[tuple[int, int], ...]] | None:
            if not isinstance(value, tuple) or len(value) != 9:
                return None
            kind, masked, line, column, operation, owner_line, owner_column, direct_edge, access_path = value
            key = access_location_key(
                kind=kind,
                masked=masked,
                line=line,
                column=column,
            )
            if key is None or not isinstance(direct_edge, bool):
                return None
            if access_location_key(
                kind=kind,
                masked=masked,
                line=line,
                column=column,
                call_path=access_path,
            ) is None:
                return None
            if operation is None:
                if direct_edge or owner_line is not None or owner_column is not None:
                    return None
            elif not isinstance(operation, str) or not operation:
                return None
            elif owner_line is None and owner_column is None:
                if not direct_edge:
                    return None
            elif (
                isinstance(owner_line, bool)
                or not isinstance(owner_line, int)
                or owner_line <= 0
                or isinstance(owner_column, bool)
                or not isinstance(owner_column, int)
                or owner_column < 0
            ):
                return None
            return (
                kind, masked, line, column, operation, owner_line,
                owner_column, direct_edge,
                access_path,
            )

        normalized_owner_evidence = tuple(
            valid_owner_evidence(item) for item in owner_evidence
        )
        if any(item is None for item in normalized_owner_evidence):
            return cls.unavailable(
                "source_compiler_owner_evidence_unavailable",
                source_facts=source_facts,
            )

        # The six compiler sidecars are emitted from one parsed TTIR walk, but
        # equal lengths alone do not prove that row ``n`` in each sidecar is
        # still the same memory operation.  Source/J admission later consumes
        # the exact row selected by the Source debug-location key; accepting a
        # reordered predicate or owner row here would therefore attach a
        # valid-looking producer to the wrong Source occurrence.  Recheck the
        # common occurrence identity before constructing any ledger binding.
        for compiler_index, location in enumerate(source_locations):
            transfer = transfers[compiler_index]
            predicate = normalized_predicate_evidence[compiler_index]
            owner = normalized_owner_evidence[compiler_index]
            assert predicate is not None and owner is not None
            location_kind, location_masked, location_line, location_column, call_path = location
            transfer_kind = getattr(transfer, "kind", None)
            transfer_masked = getattr(transfer, "masked", None)
            transfer_call_path = getattr(transfer, "source_call_path", None)
            if (
                transfer_kind != location_kind
                or transfer_masked is not location_masked
                or transfer_call_path != call_path
                or predicate[:4]
                != (location_kind, location_masked, location_line, location_column)
                or predicate[10] != call_path
                or owner[:4]
                != (location_kind, location_masked, location_line, location_column)
                or owner[8] != call_path
            ):
                return cls.unavailable(
                    "source_memory_occurrence_sidecars_incomplete",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_transfer_ledger_unavailable",
                            "path": f"<source-ttir-observation>.sidecars[{compiler_index}]",
                            "observed_refs_or_types": [
                                "expected:transfer/predicate/owner rows aligned to one compiler memory location"
                            ],
                        },
                    ),
                )

        lexical_by_compiler: dict[int, TransferOccurrence] = {}
        for lexical in lexical_ledger.occurrences:
            semantic_access = lexical.semantic_access
            if (
                semantic_access is None
                or semantic_access.source_line is None
                or semantic_access.source_column is None
            ):
                return cls.unavailable(
                    "source_semantic_access_location_unavailable",
                    source_facts=source_facts,
                )
            key = access_location_key(
                kind=lexical.kind,
                masked=lexical.masked,
                line=semantic_access.source_line,
                column=semantic_access.source_column,
                call_path=semantic_access.source_call_path,
            )
            compiler_index = source_to_compiler.get(key) if key is not None else None
            if compiler_index is None or compiler_index in lexical_by_compiler:
                return cls.unavailable(
                    "source_compiler_access_counterpart_ambiguous",
                    source_facts=source_facts,
                )
            lexical_by_compiler[compiler_index] = lexical
        if len(lexical_by_compiler) != len(transfers):
            return cls.unavailable(
                "source_compiler_access_counterpart_unavailable",
                source_facts=source_facts,
            )
        occurrences: list[TransferOccurrence] = []
        # The target ledger's ordinal remains the lexical Source occurrence
        # identifier, but every row below was selected by its unique Source /
        # compiler coordinate evidence rather than by either list position.
        for lexical in lexical_ledger.occurrences:
            semantic_access = lexical.semantic_access
            assert semantic_access is not None  # checked while constructing the map
            key = access_location_key(
                kind=lexical.kind,
                masked=lexical.masked,
                line=semantic_access.source_line,
                column=semantic_access.source_column,
                call_path=semantic_access.source_call_path,
            )
            assert key is not None
            compiler_index = source_to_compiler[key]
            transfer = transfers[compiler_index]
            origin = origins[compiler_index]
            form = forms[compiler_index]
            ordinal = lexical.ordinal
            operation, origin_positions, context = origin
            if not isinstance(form, (tuple, list)) or len(form) != 7:
                return cls.unavailable(
                    "source_memory_occurrence_typed_address_incomplete",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_access_typed_address_unproved",
                            "path": f"<source-transfer-ledger>[{ordinal}].index_expression",
                            "observed_refs_or_types": ["typed_address_fields=missing"],
                        },
                    ),
                )
            (
                form_operation,
                pointer_base,
                polynomial,
                form_context,
                element_type,
                raw_address_type_profile,
                form_call_path,
            ) = form
            if (
                operation != form_operation
                or tuple(context) != tuple(form_context)
                or not isinstance(form_call_path, tuple)
                or any(
                    not isinstance(frame, tuple)
                    or len(frame) != 2
                    or type(frame[0]) is not int
                    or frame[0] <= 0
                    or type(frame[1]) is not int
                    or frame[1] < 0
                    for frame in form_call_path
                )
                or form_call_path != semantic_access.source_call_path
            ):
                return cls.unavailable(
                    "source_memory_occurrence_sidecars_disagree",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_transfer_ledger_disagreement",
                            "path": f"<source-transfer-ledger>[{ordinal}]",
                            "observed_refs_or_types": [str(operation), str(form_operation)],
                        },
                    ),
                )
            kind = getattr(transfer, "kind", None)
            masked = getattr(transfer, "masked", None)
            predicate = getattr(transfer, "predicate", None)
            fill = getattr(transfer, "other", None)
            if kind not in {"load", "store", "update"} or not isinstance(masked, bool):
                return cls.unavailable("source_memory_occurrence_invalid", source_facts=source_facts)
            pointer_position = None
            if isinstance(pointer_base, str):
                match = _POINTER_RE.match(pointer_base)
                if match is not None:
                    pointer_position = int(match.group(1))
            logical_ref = None
            if (
                source_facts is not None
                and pointer_position is not None
                and logical_bindings is not None
            ):
                argument = source_facts.argument_by_position(pointer_position)
                argument_name = (
                    argument.get("name")
                    if isinstance(argument, Mapping)
                    else None
                )
                binding = (
                    logical_bindings.get(argument_name)
                    if isinstance(argument_name, str)
                    else None
                )
                if isinstance(binding, (tuple, list)) and binding:
                    candidate = binding[0]
                    if isinstance(candidate, str) and re.fullmatch(
                        r"t[0-9]+", candidate
                    ):
                        logical_ref = candidate
            if logical_ref is None and source_facts is not None:
                logical_ref = source_facts.logical_ref_for_position(pointer_position)
            # The lexical Source expression is the provider-facing
            # semantic predicate/fill.  The compiler predicate is kept only
            # when no exact lexical expression exists; it may be an opaque
            # SSA name and will not be published as J semantics.  ``lexical``
            # was selected above by the exact Source/compiler access location,
            # never by ordinal or a semantic guess.
            if isinstance(lexical.predicate, str):
                predicate = lexical.predicate
            if isinstance(lexical.fill_value, str):
                fill = lexical.fill_value
            # ``other=`` is part of the Source occurrence contract for every
            # masked load. The exact lexical Source occurrence and compiler
            # observation are both legitimate evidence, so reject only after
            # the lexical value has had a chance to close a TTIR-side omission.
            # No default or inferred fill is introduced here.
            if kind == "load" and masked and not isinstance(fill, str):
                return cls.unavailable(
                    "source_masked_load_fill_missing",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_access_fill_presence_unproved",
                            "path": f"<source-transfer-ledger>[{ordinal}].other",
                            "observed_refs_or_types": [
                                "masked Source load has no explicit other= fill"
                            ],
                        },
                    ),
                )
            if masked and not isinstance(predicate, str):
                return cls.unavailable(
                    "source_masked_access_predicate_unavailable",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_access_predicate_unproved",
                            "path": f"<source-transfer-ledger>[{ordinal}].predicate",
                            "observed_refs_or_types": [
                                "masked Source access has no closed predicate"
                            ],
                        },
                    ),
                )
            if logical_ref is None:
                return cls.unavailable(
                    "source_memory_occurrence_owner_unavailable",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_transfer_tensor_owner_unavailable",
                            "path": f"<source-transfer-ledger>[{ordinal}].tensor",
                            "observed_refs_or_types": [
                                f"pointer_position={pointer_position}",
                                "expected:unique SourceFacts/allocator logical tensor owner",
                            ],
                        },
                    ),
                )
            address_profile = _address_profile(polynomial)
            address_terms = _canonical_address_terms(polynomial)
            address_type_profile = _coerce_address_type_profile(
                raw_address_type_profile
            )
            address_failure = (
                "address_relation_unavailable"
                if address_profile is None or address_terms is None
                else "element_type_unavailable"
                if not isinstance(element_type, str) or not element_type
                else "address_type_profile_unavailable"
                if address_terms and not address_type_profile
                else "address_type_profile_disagrees"
                if address_type_profile is None
                or address_terms != _combine_address_terms(address_type_profile)
                else None
            )
            if address_failure is not None:
                return cls.unavailable(
                    "source_memory_occurrence_address_unavailable",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_access_index_expression_unproved",
                            "path": f"<source-transfer-ledger>[{ordinal}].index_expression",
                            "observed_refs_or_types": [
                                f"reason={address_failure}",
                                f"offset_term_count={len(address_terms) if address_terms is not None else -1}",
                            ],
                        },
                    ),
                )
            occurrences.append(
                TransferOccurrence(
                    ordinal=ordinal,
                    operation=str(operation),
                    kind=kind,
                    masked=masked,
                    predicate=predicate if isinstance(predicate, str) else None,
                    fill_value=fill if isinstance(fill, str) else None,
                    pointer_base=pointer_base if isinstance(pointer_base, str) else None,
                    pointer_position=pointer_position,
                    origin_positions=tuple(
                        item for item in origin_positions
                        if isinstance(item, int) and not isinstance(item, bool)
                    ),
                    context=tuple(item for item in context if isinstance(item, str)),
                    address_terms=address_terms,
                    address_profile=address_profile,
                    address_type_profile=address_type_profile,
                    logical_ref=logical_ref,
                    tensor_rank=(
                        source_facts.rank_for_logical(logical_ref)
                        if source_facts is not None else None
                    ),
                    element_type=element_type,
                    semantic_consumer=(
                        lexical.semantic_consumer
                    ),
                    semantic_predicate=(
                        lexical.semantic_predicate
                    ),
                    semantic_access=lexical.semantic_access,
                )
            )
        def compiler_family_matches(operation: str, family: str) -> bool:
            if family == "map_binary":
                return operation.startswith("arith.") or operation.startswith("math.")
            if family == "map_unary":
                return operation.startswith("arith.") or operation.startswith("math.")
            if family == "select":
                return operation in {"arith.select", "tt.select"}
            if family == "cast":
                return operation.startswith("arith.") or operation in {
                    "tt.fp_to_fp", "tt.bitcast"
                }
            if family == "reduce":
                return operation in {"tt.reduce", "tt.scan"}
            if family == "iota":
                return operation == "tt.make_range"
            return operation in {
                f"tt.{family}",
                f"tensor.{family}",
            }

        def owner_parameter_positions(names: Any) -> tuple[int, ...] | None:
            if source_facts is None or not isinstance(names, tuple) or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
                return None
            aliases = dict(source_facts.source_to_opaque)
            by_name = {row["name"]: index for index, row in enumerate(source_facts.arguments)}
            mapped = tuple(by_name.get(aliases.get(name, name)) for name in names)
            if any(position is None for position in mapped) or len(set(mapped)) != len(mapped):
                return None
            return mapped

        def tensor_owner_authorization(occurrence: TransferOccurrence) -> dict[str, Any] | None:
            if source_facts is None or logical_bindings is None or occurrence.semantic_access is None or occurrence.semantic_consumer is None or occurrence.kind != "store":
                return None
            # These rules are local proof failures, not exception messages or
            # provider text. Only this closed vocabulary may leave the issuer.
            safe_failures = frozenset({
                "source_owner_shape_contract_unavailable",
                "source_owner_symbolic_domain_unproved",
                "source_owner_multidimensional_domain_unproved",
                "source_owner_shape_contract_mismatch",
                "source_owner_memory_abi_unavailable",
                "source_owner_memory_contract_unproved",
                "source_owner_load_coordinates_unproved",
                "source_owner_load_domain_unproved",
                "source_owner_load_fill_unproved",
                "source_owner_load_fill_mismatch",
                "source_owner_store_coordinate_unproved",
                "source_owner_load_occurrence_ambiguous",
                "source_owner_tile_domain_ambiguous",
                "source_owner_scalar_abi_invalid",
                "source_owner_padding_domains_differ",
                "source_owner_outside_identity_unproved",
                "source_owner_unclosed_padding_path",
                "source_owner_store_mask_domain_mismatch",
                "source_owner_store_value_shape_mismatch",
                "source_owner_host_launch_contract_unavailable",
                "source_owner_program_domain_unproved",
                "source_owner_row_stride_unproved",
                "source_owner_address_overflow_unproved",
                "source_owner_tile_capacity_unproved",
                "source_owner_reshape_unproved",
                "source_owner_transpose_unproved",
                "source_owner_transform_domain_unproved",
                "source_owner_reduction_axis_unproved",
                "source_owner_padding_coordinate_transport_unproved",
            })

            def unavailable(error: Exception) -> dict[str, Any]:
                rule = error.args[0] if len(error.args) == 1 else None
                return {
                    "status": "unavailable",
                    "reason": rule if isinstance(rule, str) and rule in safe_failures
                    else "source_exact_owner_binding_unavailable",
                }

            launch = source_facts.compiler_facts.get("host_launch_contract", {})
            dimensions = {name: f"x{index}" for index, (name, _value) in enumerate(source_facts.shape_coordinates)}

            def source_shape(argument: Mapping[str, Any]) -> list[int | str]:
                declared = argument.get("shape_contract")
                if isinstance(declared, list) and any(
                    isinstance(extent, str) for extent in declared
                ):
                    if not launch:
                        raise ValueError("source_owner_host_launch_contract_unavailable")
                    if any(not ((type(n) is int and n > 0) or (isinstance(n, str) and n in dimensions)) for n in declared):
                        raise ValueError("source_owner_symbolic_domain_unproved")
                    if len(declared) > 2:
                        raise ValueError("source_owner_multidimensional_domain_unproved")
                    return [dimensions[n] if isinstance(n, str) else n for n in declared]
                return _source_owner_static_shape(argument)

            target = source_facts.argument_by_position(occurrence.pointer_position)
            try:
                if target is None:
                    raise ValueError("source_owner_memory_abi_unavailable")
                source_shape(target)
            except ValueError as error:
                return unavailable(error)
            access = occurrence.semantic_access
            proofs = [row for row in getattr(observed, "diagnostic_tensor_owner_graphs", ()) if isinstance(row, Mapping) and (row.get("line"), row.get("column"), row.get("call_path")) == (access.source_line, access.source_column, access.source_call_path)]
            if len(proofs) != 1:
                return None
            proof = proofs[0]
            names, types = proof.get("parameters"), proof.get("parameter_types")
            parameter_positions = owner_parameter_positions(names)
            if (
                parameter_positions is None or not isinstance(types, tuple)
                or len(types) != len(names)
                or proof.get("family") != occurrence.semantic_consumer.operation_family
            ):
                return None
            dtype_aliases = {"float32": "f32", "float64": "f64"}

            def source_position(position: Any) -> int:
                if type(position) is not int or not 0 <= position < len(parameter_positions):
                    raise ValueError("source_owner_memory_abi_unavailable")
                return parameter_positions[position]

            def memory_contract(evidence: Mapping[str, Any], selected: TransferOccurrence) -> dict[str, Any]:
                argument = source_facts.argument_by_position(selected.pointer_position)
                if argument is None or selected.semantic_access is None:
                    raise ValueError("source_owner_memory_abi_unavailable")
                dtype = dtype_aliases.get(argument.get("dtype"), argument.get("dtype"))
                shape = source_shape(argument)
                binding = logical_bindings.get(argument["name"])
                if (
                    source_position(evidence.get("pointer_position")) != selected.pointer_position
                    or evidence.get("kind") != selected.kind or evidence.get("dtype") != dtype
                    or bool(evidence.get("predicate") is not None) != selected.masked
                    or argument.get("kind") not in {"tensor", "pointer"} or argument.get("alias_of") is not None
                    or argument.get("mode") != ("input" if selected.kind == "load" else "output")
                    or not isinstance(shape, list) or len(shape) > 2
                    or not isinstance(binding, (tuple, list)) or len(binding) != 2 or binding[0] != selected.logical_ref
                ):
                    raise ValueError("source_owner_memory_contract_unproved")
                coordinate, physical = evidence["coordinate"], tuple(evidence["physical_shape"])
                row = evidence.get("row")
                if row is not None:
                    if len(shape) != 2 or not isinstance(row, (tuple, list)) or len(row) != 3:
                        raise ValueError("source_owner_program_domain_unproved")
                    axis, stride_position, stride_mode = row
                    grid_coordinates = source_facts.compiler_facts["grid_shape_coordinates"]
                    grid = source_facts.compiler_facts["grid"]
                    declared = argument.get("shape_contract")
                    if type(axis) is not int or axis != 0 or len(grid) != 1 or len(grid_coordinates) != 1 or type(grid[0]) is not int or not 0 < grid[0] < 2 ** 31 or not isinstance(declared, list) or grid_coordinates[axis] != declared[0] or grid[0] != dict(source_facts.shape_coordinates).get(declared[0]):
                        raise ValueError("source_owner_program_domain_unproved")
                    stride = source_facts.argument_by_position(source_position(stride_position))
                    if (
                        stride is None
                        or stride.get("dtype") != "i64"
                        or stride_mode not in {"direct_i64", "direct_i32", "signed_i32_extended"}
                        or types[stride_position]
                        != ("i64" if stride_mode == "direct_i64" else "i32")
                    ):
                        raise ValueError("source_owner_row_stride_unproved")
                    launch_strides = [
                        item
                        for item in launch.get("strides", ())
                        if isinstance(item, Mapping)
                    ]
                    direct_stride_rows = [
                        item
                        for item in launch_strides
                        if item
                        == {
                            "carrier": stride["name"],
                            "tensor": argument["name"],
                            "axis": 0,
                        }
                    ]
                    shared_stride_rows = []
                    for item in launch_strides:
                        if (
                            item.get("carrier") != stride["name"]
                            or item.get("axis") != 0
                            or item.get("tensor") == argument["name"]
                        ):
                            continue
                        launch_tensor = next(
                            (
                                candidate
                                for candidate in source_facts.arguments
                                if candidate.get("name") == item.get("tensor")
                            ),
                            None,
                        )
                        # A shared stride is a symbolic ABI relation only when
                        # the launch witness and the selected Source tensor
                        # declare exactly the same shape coordinates.  Numeric
                        # shape equality is not a substitute for this row.
                        if (
                            isinstance(launch_tensor, Mapping)
                            and launch_tensor.get("kind") == "tensor"
                            and isinstance(argument.get("shape_contract"), list)
                            and argument.get("shape_contract")
                            == launch_tensor.get("shape_contract")
                        ):
                            shared_stride_rows.append(item)
                    if len(direct_stride_rows) + len(shared_stride_rows) != 1:
                        raise ValueError("source_owner_row_stride_unproved")
                    tiles = [
                        item
                        for item in launch.get("tiles", ())
                        if isinstance(item, Mapping)
                        and item.get("coordinate") == declared[1]
                    ] if isinstance(declared[1], str) else []
                    # Some valid Triton launchers pass the constexpr tile
                    # directly through the compiler schedule and therefore
                    # expose no additional host-side tile relation.  The
                    # SourceFacts compiler ledger still contains the exact
                    # role, dtype, coordinate, and frozen value.  Consume
                    # that existing Source-owned relation only when it is
                    # unique and closed; never turn the symbolic shape name
                    # into a numeric bound merely because the host receipt
                    # lacks a tile row.
                    if not tiles and isinstance(declared[1], str):
                        compiler_facts = source_facts.compiler_facts
                        schedule_values = compiler_facts.get("schedule_values")
                        schedule_types = compiler_facts.get("schedule_value_types")
                        compiler_roles = compiler_facts.get("compiler_roles")
                        tile_coordinates = compiler_facts.get(
                            "schedule_tile_coordinates"
                        )
                        if all(
                            isinstance(value, Mapping)
                            for value in (
                                schedule_values,
                                schedule_types,
                                compiler_roles,
                                tile_coordinates,
                            )
                        ):
                            schedule_tiles = [
                                {
                                    "slot": slot,
                                    "coordinate": coordinate,
                                }
                                for slot, coordinate in tile_coordinates.items()
                                if coordinate == declared[1]
                                and compiler_roles.get(slot)
                                in {"tile_extent", "reduction_tile_extent"}
                                and schedule_types.get(slot) == "i32"
                                and type(schedule_values.get(slot)) is int
                                and 0 < schedule_values.get(slot) < 2**31
                            ]
                            if len(schedule_tiles) == 1:
                                tiles = schedule_tiles
                    capacity = (
                        source_facts.compiler_facts["schedule_values"][tiles[0]["slot"]]
                        if len(tiles) == 1
                        else declared[1]
                    )
                    # The raw row multiplication is i64 over a nonnegative
                    # i32 program id. The supported ABI is contiguous, so its
                    # row stride is the column extent and its column stride is
                    # one. Never infer a symbolic layout from explicit strides
                    # that happen to equal today's contiguous valuation.
                    strides = argument.get("strides")
                    if strides is not None:
                        raise ValueError("source_owner_row_stride_unproved")
                    stride_bound = capacity
                    if type(stride_bound) is not int or not 0 <= stride_bound < 2 ** 31:
                        raise ValueError("source_owner_address_overflow_unproved")
                    # The i64 carrier may be represented by a signed i32 ABI
                    # only when the same frozen launch contract makes the
                    # contiguous row stride equal to this bounded tile extent.
                    # A concrete invocation value cannot establish the bound.
                    if stride_mode == "signed_i32_extended" and stride_bound < 1:
                        raise ValueError("source_owner_row_stride_unproved")
                    if physical:
                        if len(physical) != 1 or coordinate != "axis0" or not selected.masked or len(tiles) != 1 or physical[0] != capacity:
                            raise ValueError("source_owner_tile_capacity_unproved")
                        predicate = evidence["predicate"]
                        bound = (
                            predicate[2]
                            if isinstance(predicate, (tuple, list))
                            and len(predicate) == 3
                            and tuple(predicate[:2]) == ("lt", 0)
                            else None
                        )
                        extent = None
                        if (
                            isinstance(bound, (tuple, list))
                            and len(bound) == 2
                            and bound[0] == "parameter"
                        ):
                            extent = source_facts.argument_by_position(
                                source_position(bound[1])
                            )
                            if (
                                extent is None
                                or extent.get("dtype") != "i32"
                                or extent.get("shape_coordinate") != declared[1]
                            ):
                                raise ValueError(
                                    "source_owner_load_domain_unproved"
                                )
                        elif type(bound) is int:
                            # A constexpr Source predicate may be folded to a
                            # literal by Triton before TTIR observation.  A
                            # literal alone is not a symbolic-domain proof:
                            # recover it only through the exact SourceFacts
                            # constexpr ledger, including its canonical type,
                            # constexpr identity, frozen value, and declared
                            # shape coordinate.  This does not use a Harness
                            # value to manufacture a domain relation.
                            compiler_facts = source_facts.compiler_facts
                            bound_values = compiler_facts.get(
                                "bound_constexpr_values"
                            )
                            bound_types = compiler_facts.get(
                                "bound_constexpr_value_types"
                            )
                            bound_coordinates = compiler_facts.get(
                                "bound_constexpr_shape_coordinates"
                            )
                            constexpr_parameters = set(
                                compiler_facts.get("constexpr_parameters", ())
                            )
                            matches = [
                                name
                                for name, value in bound_values.items()
                                if name in constexpr_parameters
                                and type(value) is int
                                and value == bound
                                and bound_types.get(name) == "i32"
                                and bound_coordinates.get(name) == declared[1]
                            ] if all(
                                isinstance(value, Mapping)
                                for value in (
                                    bound_values,
                                    bound_types,
                                    bound_coordinates,
                                )
                            ) else []
                            if len(matches) != 1:
                                raise ValueError(
                                    "source_owner_load_domain_unproved"
                                )
                        else:
                            raise ValueError("source_owner_load_domain_unproved")
                        if selected.kind == "load":
                            try:
                                lexical_fill = source_float_literal(selected.fill_value)
                            except (ValueError, TypeError, SyntaxError) as error:
                                raise ValueError("source_owner_load_fill_unproved") from error
                            if owner_float_bits_allow_infinity(lexical_fill, dtype) != evidence["fill"]:
                                raise ValueError("source_owner_load_fill_mismatch")
                        coordinates = ["axis0", "axis1"]
                        predicate = ("lt", 1, shape[1])
                    else:
                        if selected.masked or type(coordinate) is not int or type(shape[1]) is not int or not 0 <= coordinate < shape[1]:
                            raise ValueError("source_owner_store_coordinate_unproved")
                        coordinates, predicate = ["axis0", coordinate], None
                    return dict(occurrence=selected.semantic_access.token, tensor=selected.logical_ref,
                                dtype=dtype, shape=shape, coordinates=coordinates, domain=shape,
                                predicate=predicate, fill=evidence["fill"])
                if len(shape) > 1:
                    raise ValueError("source_owner_program_domain_unproved")
                if selected.kind == "load":
                    if (
                        not physical and type(coordinate) is int and not selected.masked
                        and evidence["fill"] is None
                        and ((not shape and coordinate == 0) or (len(shape) == 1 and 0 <= coordinate < shape[0]))
                    ):
                        coordinates = [coordinate] if shape else []
                    elif len(shape) != 1 or coordinate != "axis0" or len(physical) != 1:
                        raise ValueError("source_owner_load_coordinates_unproved")
                    elif selected.masked:
                        if evidence["predicate"] != ("lt", 0, shape[0]) or physical[0] < shape[0]:
                            raise ValueError("source_owner_load_domain_unproved")
                        try:
                            lexical_fill = source_float_literal(selected.fill_value)
                        except (ValueError, TypeError, SyntaxError) as exc:
                            raise ValueError("source_owner_load_fill_unproved") from exc
                        if owner_float_bits_allow_infinity(lexical_fill, dtype) != evidence["fill"]:
                            raise ValueError("source_owner_load_fill_mismatch")
                        coordinates = ["axis0"]
                    elif physical != tuple(shape):
                        raise ValueError("source_owner_load_domain_unproved")
                    else:
                        coordinates = ["axis0"]
                else:
                    if physical:
                        if len(shape) != 1 or coordinate != "axis0" or len(physical) != 1:
                            raise ValueError("source_owner_store_coordinate_unproved")
                        if selected.masked:
                            if evidence["predicate"] != ("lt", 0, shape[0]) or physical[0] < shape[0]:
                                raise ValueError("source_owner_store_mask_domain_mismatch")
                        elif physical != tuple(shape):
                            raise ValueError("source_owner_store_mask_domain_mismatch")
                        coordinates = ["axis0"]
                    else:
                        if selected.masked or type(coordinate) is not int or (not shape and coordinate != 0) or (shape and not 0 <= coordinate < shape[0]):
                            raise ValueError("source_owner_store_coordinate_unproved")
                        coordinates = [coordinate] if shape else []
                return dict(occurrence=selected.semantic_access.token, tensor=selected.logical_ref,
                            dtype=dtype, shape=shape, coordinates=coordinates, domain=shape,
                            predicate=evidence["predicate"], fill=evidence["fill"])

            try:
                store = memory_contract(proof, occurrence)
                graph = proof["graph"]
                load_contracts: dict[int, dict[str, Any]] = {}
                physical_to_logical: dict[tuple[int, ...], tuple[int | str, ...]] = {}
                if proof["physical_shape"]:
                    physical_to_logical[tuple(proof["physical_shape"])] = tuple(store["shape"])
                for index, (op, _dtype, physical, payload, _children) in enumerate(graph):
                    if op != "load":
                        continue
                    selected = [item for item in occurrences if item.kind == "load" and item.semantic_access is not None and (item.semantic_access.source_line, item.semantic_access.source_column, item.semantic_access.source_call_path) == (payload["line"], payload["column"], payload["call_path"])]
                    if len(selected) != 1:
                        raise ValueError("source_owner_load_occurrence_ambiguous")
                    contract = memory_contract(payload, selected[0])
                    load_contracts[index] = contract
                    logical = tuple(contract["shape"])
                    if physical and physical_to_logical.setdefault(tuple(physical), logical) != logical:
                        raise ValueError("source_owner_tile_domain_ambiguous")
                translated: list[tuple[Any, ...]] = []
                translated_indices: dict[int, int] = {}
                broadcast_indices: dict[tuple[int, tuple[int | str, ...]], int] = {}

                def broadcast(child: int, shape: tuple[int | str, ...]) -> int:
                    before = translated[child][2]
                    if before == shape:
                        return child
                    # Lift an existing per-program scalar across its lane
                    # domain, or an invariant scalar across program instances.
                    # Axis identities come from the proved load/address domain.
                    axes = () if not before else (0,) if len(before) == 1 and len(shape) == 2 and before[0] == shape[0] else None
                    if axes is None:
                        raise ValueError("source_owner_tile_domain_ambiguous")
                    key = child, shape
                    if key not in broadcast_indices:
                        broadcast_indices[key] = len(translated)
                        translated.append(("broadcast", translated[child][1], shape, axes, (child,)))
                    return broadcast_indices[key]

                for index, (op, dtype, shape, payload, children) in enumerate(graph):
                    children = tuple(translated_indices[child] for child in children)
                    physical = tuple(shape)
                    shape = physical_to_logical.get(physical, physical)
                    if op == "load":
                        payload = load_contracts[index]
                        shape = tuple(extent for coord, extent in zip(payload["coordinates"], payload["shape"], strict=True) if isinstance(coord, str))
                    elif op == "input":
                        if type(payload) is not int or not 0 <= payload < len(names):
                            raise ValueError("source_owner_scalar_abi_invalid")
                        argument = source_facts.arguments[source_position(payload)]
                        binding = logical_bindings.get(argument["name"])
                        if argument.get("kind") != "scalar" or argument.get("mode") != "value" or argument.get("alias_of") is not None or types[payload] != dtype or dtype_aliases.get(argument.get("dtype"), argument.get("dtype")) != dtype or not isinstance(binding, (tuple, list)) or len(binding) != 2 or binding[1] not in {"scalar", "shape_scalar"}:
                            raise ValueError("source_owner_scalar_abi_invalid")
                        payload = binding[0]
                    elif op == "reduce":
                        input_shape = translated[children[0]][2]
                        physical_input = tuple(graph[graph[index][4][0]][2])
                        if tuple(input_shape) == physical_input:
                            axis = payload[0]
                        elif len(physical_input) == 1 and len(input_shape) == 2 and payload[0] == 0 and physical_to_logical.get(physical_input) == tuple(input_shape):
                            # Existing positively proved row projection adds
                            # the program axis; only its local lane axis moves.
                            axis = 1
                        else:
                            raise ValueError("source_owner_reduction_axis_unproved")
                        shape = input_shape[:axis] + input_shape[axis + 1:]
                        payload = (axis, payload[1])
                    elif op == "broadcast":
                        child_shape = translated[children[0]][2]
                        payload = () if not child_shape else (0,) if len(child_shape) == 1 and len(shape) == 2 and child_shape[0] == shape[0] else None
                    elif op in {"reshape", "transpose"}:
                        # Do not reuse a tile shape coincidence to infer a
                        # logical row/column permutation or reshape.
                        original_child_shape = tuple(graph[graph[index][4][0]][2])
                        if tuple(translated[children[0]][2]) != original_child_shape or tuple(shape) != physical:
                            raise ValueError("source_owner_transform_domain_unproved")
                    elif op in {"add", "sub", "mul", "div", "neg", "exp", "tanh", "rsqrt", "rsqrt_ftz", "cast"}:
                        if not physical:
                            varying = {translated[child][2] for child in children if translated[child][2]}
                            if len(varying) > 1:
                                raise ValueError("source_owner_tile_domain_ambiguous")
                            shape = next(iter(varying), ())
                        if op in {"add", "sub", "mul", "div"}:
                            children = tuple(broadcast(child, shape) for child in children)
                    translated_indices[index] = len(translated)
                    translated.append((op, dtype, shape, payload, children))
                if not load_contracts or translated[-1][1] != store["dtype"]:
                    return None
                ordered = build_tensor_owner_dag(str(len(translated) - 1), lambda value: (
                    *translated[int(value)][:4], tuple(str(child) for child in translated[int(value)][4])
                ))
                if len(ordered) != len(translated):
                    raise ValueError("source_owner_graph_not_canonical")
                translated = list(ordered)
                digest = tensor_owner_digest(store, translated)
            except (ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
                return unavailable(error)
            # This commits one occurrence and its proven store/producer slice,
            # not the complete candidate, a run identity, or Gate acceptance.
            # Equal Source facts may legitimately recreate the same token in
            # independent runs; fresh lifecycle ownership is enforced outside
            # this content commitment. Consumers must recheck their actual J.
            token = access.token + "_" + digest
            attestation = hashlib.sha256((source_facts.digest + "|" + token).encode()).hexdigest()[:32]
            # This is the already-proven logical projection, not the raw
            # compiler graph. It is guidance only: no candidate operation is
            # created from it, and every submitted J is rechecked independently.
            outline = {"store": store, "graph": [list(row) for row in translated]}
            return dict(status="pass", token=token, operation_family=proof["family"], compiler_attestation="sc_" + attestation, authoring_outline=outline)

        def owner_authorization(
            occurrence: TransferOccurrence,
        ) -> dict[str, Any]:
            tensor_authorization = tensor_owner_authorization(occurrence)
            if tensor_authorization is not None and tensor_authorization.get("status") == "pass":
                return tensor_authorization
            unavailable = tensor_authorization or {
                "status": "unavailable",
                "reason": (
                    "source_atomic_update_authorization_unavailable"
                    if occurrence.kind == "update"
                    else "source_semantic_consumer_unavailable"
                    if occurrence.semantic_consumer is None
                    else "source_exact_owner_binding_unavailable"
                ),
            }
            semantic, access = occurrence.semantic_consumer, occurrence.semantic_access
            if (
                occurrence.kind != "store" or occurrence.masked or semantic is None or access is None
                or source_facts is None or logical_bindings is None
            ):
                return unavailable
            proofs = [
                row for row in getattr(observed, "diagnostic_scalar_owner_graphs", ())
                if isinstance(row, Mapping)
                and (row.get("line"), row.get("column"), row.get("call_path"), row.get("masked"))
                == (access.source_line, access.source_column, access.source_call_path, occurrence.masked)
            ]
            if len(proofs) != 1:
                return unavailable
            proof = proofs[0]
            names, types = proof.get("parameters"), proof.get("parameter_types")
            parameter_positions = owner_parameter_positions(names)
            graph = proof.get("graph")
            if (
                parameter_positions is None or not isinstance(types, tuple)
                or len(types) != len(names)
                or proof.get("family") != semantic.operation_family
                or type(proof.get("pointer_position")) is not int
                or not 0 <= proof["pointer_position"] < len(parameter_positions)
                or parameter_positions[proof["pointer_position"]] != occurrence.pointer_position
                or not isinstance(graph, tuple) or not graph
            ):
                return unavailable
            target = source_facts.argument_by_position(occurrence.pointer_position)
            if target is None or target.get("mode") != "output":
                return unavailable
            try:
                if _source_owner_static_shape(target) != []:
                    return unavailable
            except ValueError:
                return unavailable
            dtype_aliases = {"float32": "f32", "float64": "f64"}
            translated: list[tuple[Any, ...]] = []
            try:
                for op, dtype, payload, children in graph:
                    if op == "input":
                        if type(payload) is not int or not 0 <= payload < len(names):
                            return unavailable
                        argument = source_facts.arguments[parameter_positions[payload]]
                        binding = logical_bindings.get(argument["name"])
                        argument_dtype = dtype_aliases.get(argument.get("dtype"), argument.get("dtype"))
                        if (
                            argument.get("kind") != "scalar" or argument.get("mode") != "value"
                            or argument.get("alias_of") is not None
                            or types[payload] != dtype or argument_dtype != dtype
                            or not isinstance(binding, (tuple, list)) or len(binding) != 2
                            or binding[1] != "scalar"
                        ):
                            return unavailable
                        payload = binding[0]
                    translated.append((op, dtype, payload, children))
                target_dtype = dtype_aliases.get(target.get("dtype"), target.get("dtype"))
                if target_dtype != translated[-1][1]:
                    return unavailable
                digest = scalar_owner_digest(occurrence.logical_ref, translated)
            except (ValueError, TypeError, KeyError, IndexError, RecursionError):
                return unavailable
            token = access.token + "_" + digest
            attestation = hashlib.sha256((source_facts.digest + "|" + token).encode()).hexdigest()[:32]
            return {
                "status": "pass", "token": token,
                "operation_family": semantic.operation_family,
                "compiler_attestation": "sc_" + attestation,
            }

        def access_view_authorization(
            occurrence: TransferOccurrence,
            compiler_index: int,
        ) -> dict[str, Any]:
            semantic = occurrence.semantic_access
            if semantic is None:
                raise ValueError("exact Source access anchor is unavailable")
            transfer = transfers[compiler_index]
            if transfer.kind != occurrence.kind or transfer.masked is not occurrence.masked:
                raise ValueError("exact Source access/compiler binding is unavailable")
            attestation = hashlib.sha256(
                (
                    f"{source_facts.digest if source_facts is not None else ''}|"
                    f"{semantic.token}|{semantic.operation_family}"
                ).encode("utf-8")
            ).hexdigest()[:32]
            return {
                "status": "pass",
                "token": semantic.token,
                "operation_family": semantic.operation_family,
                "compiler_attestation": "sc_" + attestation,
            }

        def guard_authorization(
            occurrence: TransferOccurrence,
            compiler_index: int,
        ) -> dict[str, Any]:
            semantic = occurrence.semantic_predicate
            evidence = normalized_predicate_evidence[compiler_index]
            assert evidence is not None
            (
                evidence_kind, evidence_masked, access_line, access_column,
                predicate_operation, predicate_line, predicate_column,
                predicate_is_i1, predicate_call_path, predicate_producer_id,
                access_call_path,
            ) = evidence
            access_semantic = occurrence.semantic_access
            if not occurrence.masked:
                return {
                    "status": "unavailable",
                    "reason": "source_computed_guard_not_applicable",
                }
            if (
                semantic is None
                or access_semantic is None
                or semantic.source_line is None
                or semantic.source_column is None
                or access_semantic.source_line is None
                or access_semantic.source_column is None
            ):
                return {
                    "status": "unavailable",
                    "reason": "source_predicate_access_anchor_unavailable",
                }
            has_predicate_location = (
                predicate_line is not None or predicate_column is not None
            )
            predicate_location_matches = (
                has_predicate_location
                and predicate_call_path == semantic.source_call_path
                and (
                    (predicate_line, predicate_column) == (
                        semantic.source_line, semantic.source_column,
                    )
                    or (
                        semantic.source_end_line is not None
                        and (semantic.source_line, semantic.source_column)
                        <= (predicate_line, predicate_column)
                        < (semantic.source_end_line, semantic.source_end_column)
                    )
                )
            )
            if (
                not predicate_location_matches
                and has_predicate_location
                and semantic.source_expression_key is not None
                and predicate_producer_id is not None
                and predicate_call_path == semantic.source_call_path
            ):
                # A compiler can CSE two identical, stable Source masks. The
                # second access then uses the first mask's SSA producer and
                # inherits its debug location. Require both a unique lexical
                # expression match and the identical compiler producer id;
                # a shared op name or nearby location is never sufficient.
                for peer in occurrences:
                    peer_semantic = peer.semantic_predicate
                    peer_access = peer.semantic_access
                    if (
                        peer is occurrence
                        or peer_semantic is None
                        or peer_access is None
                        or peer.context != occurrence.context
                        or peer_semantic.source_call_path != semantic.source_call_path
                        or peer_semantic.source_expression_key != semantic.source_expression_key
                        or peer_semantic.operation_family != semantic.operation_family
                        or peer_semantic.source_end_line is None
                        or not (
                            (peer_semantic.source_line, peer_semantic.source_column)
                            <= (predicate_line, predicate_column)
                            < (peer_semantic.source_end_line, peer_semantic.source_end_column)
                        )
                    ):
                        continue
                    peer_index = source_to_compiler.get(access_location_key(
                        kind=peer.kind,
                        masked=peer.masked,
                        line=peer_access.source_line,
                        column=peer_access.source_column,
                        call_path=peer_access.source_call_path,
                    ))
                    peer_evidence = (
                        normalized_predicate_evidence[peer_index]
                        if peer_index is not None else None
                    )
                    if (
                        peer_evidence is not None
                        and peer_evidence[4] == predicate_operation
                        and peer_evidence[7] is True
                        and peer_evidence[8] == predicate_call_path
                        and peer_evidence[9] == predicate_producer_id
                    ):
                        predicate_location_matches = True
                        break
            # Triton may put an operation's debug location on its last Source
            # operand rather than the enclosing expression's opening token.
            # A location inside the exact lexical predicate span is eligible
            # only with the same helper call path and the direct i1 SSA edge
            # of the already matched memory access. Without a standalone
            # location, the strict straight-line Source def-use binding below
            # remains mandatory. Neither route searches by operation order.
            direct_access_predicate_edge = (
                not has_predicate_location
                and semantic.binding_mode in {"direct", "exact_local_binding"}
            )
            if (
                evidence_kind != occurrence.kind
                or evidence_masked is not True
                or (access_line, access_column) != (
                    access_semantic.source_line, access_semantic.source_column
                )
                or access_call_path != access_semantic.source_call_path
                or not (predicate_location_matches or direct_access_predicate_edge)
                or not isinstance(predicate_operation, str)
                or not predicate_is_i1
                or not compiler_family_matches(
                    predicate_operation, semantic.operation_family
                )
            ):
                return {
                    "status": "unavailable",
                    "reason": "source_compiler_predicate_access_projection_unavailable",
                }
            attestation = hashlib.sha256(
                (
                    f"{source_facts.digest if source_facts is not None else ''}|"
                    f"{semantic.token}|{occurrence.semantic_access.token}|"
                    f"{predicate_operation}"
                ).encode("utf-8")
            ).hexdigest()[:32]
            return {
                "status": "pass",
                "token": semantic.token,
                "operation_family": semantic.operation_family,
                "compiler_attestation": "sc_" + attestation,
            }

        bindings: list[dict[str, Any]] = []
        for occurrence in occurrences:
            profile = occurrence.address_profile
            if profile is None or occurrence.logical_ref is None:
                return cls.unavailable(
                    "source_memory_occurrence_binding_incomplete",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_transfer_binding_incomplete",
                            "path": f"<source-transfer-ledger>[{occurrence.ordinal}]",
                            "observed_refs_or_types": [
                                "owner, index expression and address-owner binding are required"
                            ],
                        },
                    ),
                )
            index_expression = {
                "address_terms": [
                    {"refs": list(atoms), "coefficient": coefficient}
                    for atoms, coefficient in occurrence.address_terms
                ],
                "address_profile": {
                    "dynamic_terms": profile[0],
                    "constant_terms": profile[1],
                    "constant_offsets": list(profile[2]),
                },
            }
            access_ref = f"a{occurrence.ordinal}"
            effect_id = f"e{occurrence.ordinal}"
            bindings.append(
                {
                    "ordinal": occurrence.ordinal,
                    "access_ref": access_ref,
                    "effect_id": effect_id,
                    "effect_order": occurrence.ordinal,
                    "kind": occurrence.kind,
                    "tensor": occurrence.logical_ref,
                    "masked": occurrence.masked,
                    "node_ref": None,
                    "predicate": occurrence.predicate,
                    "fill_value": occurrence.fill_value,
                    "index_expression": index_expression,
                    "address_profile": profile,
                    "source_other": occurrence.fill_value,
                    "source_predicate": occurrence.predicate,
                    "source_pointer_base": occurrence.pointer_base,
                    "source_element_type": occurrence.element_type,
                    "source_address_type_profile": (
                        occurrence.address_type_profile
                    ),
                    "semantic_authorizations": {
                        "access_view": access_view_authorization(
                            occurrence,
                            source_to_compiler[
                                access_location_key(
                                    kind=occurrence.kind,
                                    masked=occurrence.masked,
                                    line=occurrence.semantic_access.source_line,
                                    column=occurrence.semantic_access.source_column,
                                    call_path=occurrence.semantic_access.source_call_path,
                                )
                            ],
                        ),
                        "owner": owner_authorization(occurrence),
                        "computed_guard": guard_authorization(
                            occurrence,
                            source_to_compiler[
                                access_location_key(
                                    kind=occurrence.kind,
                                    masked=occurrence.masked,
                                    line=occurrence.semantic_access.source_line,
                                    column=occurrence.semantic_access.source_column,
                                    call_path=occurrence.semantic_access.source_call_path,
                                )
                            ],
                        ),
                    },
                    "pi_binding": {
                        "status": "source_address_owner_bound",
                        "access_ref": access_ref,
                        "effect_id": effect_id,
                        "tensor": occurrence.logical_ref,
                        "pointer_position": occurrence.pointer_position,
                        "element_type": occurrence.element_type,
                        "address_type_profile": _address_type_profile_wire(
                            occurrence.address_type_profile
                        ),
                        "address_profile": index_expression["address_profile"],
                    },
                }
            )
        return cls(
            source_facts.digest if source_facts is not None else None,
            "pass",
            "lexical_source_memory_occurrences",
            tuple(occurrences),
            tuple(bindings),
        )

    @classmethod
    def from_source_text(
        cls,
        source: str,
        *,
        source_kind: str,
        source_facts: SourceFacts | None = None,
    ) -> "SourceTransferLedger":
        """Create the MCP navigation projection from exact source text.

        This records memory calls in the declared constexpr specialization,
        literal mask/fill spellings, and exact helper invocation paths. It is
        still only a lexical view and is not accepted by Source
        reconciliation: no pointer form, typed address, or J access binding is
        available until the selected frontend compiler produces its observation.
        Keeping it in the same ledger type means the MCP reads a projection of
        the host construction authority instead of recreating occurrence order
        from several independent buckets.
        """

        if not isinstance(source, str) or not source.strip():
            return cls.unavailable("source_text_absent", source_facts=source_facts)
        rows: list[TransferOccurrence] = []
        if source_kind == "cuda":
            for line_number, raw_line in enumerate(source.splitlines(), start=1):
                # This is only a Source-navigation projection; TTIR remains
                # the typed occurrence authority.  Still, preserve every
                # lexical call.  Keeping only the first call on a line loses
                # an occurrence before SourceReconcile can compare it.
                matches = re.finditer(
                    r"\b(load|store|atomic_[A-Za-z0-9_]+|atomic[A-Za-z0-9_]*)\b",
                    raw_line,
                )
                for match in matches:
                    token = match.group(1)
                    rows_kind = (
                        "load"
                        if token == "load"
                        else "store"
                        if token == "store"
                        else "update"
                    )
                    rows.append(
                        TransferOccurrence(
                            ordinal=len(rows),
                            operation=token,
                            kind=rows_kind,
                            masked="mask" in raw_line.casefold(),
                            predicate=None,
                            fill_value=None,
                            pointer_base=None,
                            pointer_position=None,
                            origin_positions=(line_number,),
                            context=(),
                            address_terms=(),
                            address_profile=None,
                            address_type_profile=(),
                            logical_ref=None,
                            tensor_rank=None,
                            element_type=None,
                        )
                    )
            return cls(
                source_facts.digest if source_facts is not None else None,
                # A non-empty lexical projection has occurrence order, but
                # no typed pointer/ABI/access/effect binding yet.  It is
                # navigation input for MCP only, not a closed Source ledger
                # that SourceReconcile, fill validation, or pi binding may use.
                "ambiguous" if rows else "pass",
                (
                    "source_native_memory_navigation_untyped"
                    if rows
                    else "source_native_memory_navigation"
                ),
                tuple(rows),
            )
        if source_kind != "python":
            return cls.unavailable(
                "source_text_memory_ledger_unsupported_kind",
                source_facts=source_facts,
            )
        try:
            tree = source_specialized_ast(source, source_facts)
        except SyntaxError:
            return cls.unavailable("source_text_ast_unavailable", source_facts=source_facts)

        parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        parent_fields = {
            id(child): (parent, field_name)
            for parent in ast.walk(tree)
            for field_name, value in ast.iter_fields(parent)
            for child in (
                value
                if isinstance(value, list)
                else (value,)
            )
            if isinstance(child, ast.AST)
        }
        source_identity = (
            source_facts.digest
            if isinstance(source_facts, SourceFacts)
            else hashlib.sha256(source.encode("utf-8")).hexdigest()
        )

        control_regions = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
            ast.Match,
            *(
                (ast.TryStar,)
                if hasattr(ast, "TryStar")
                else ()
            ),
        )

        def enclosing_function(node: ast.AST) -> ast.AST | None:
            """Return the lexical function scope owning ``node``."""

            current: ast.AST | None = node
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return current
                current = parents.get(id(current))
            return None

        def scope_store_counts(function: ast.AST) -> dict[str, int]:
            """Count every lexical write in one function, excluding children.

            This is intentionally broader than the accepted direct-definition
            profile below.  Any alternate assignment, including one in a
            branch or loop, turns the name ambiguous rather than allowing a
            Source-order guess.
            """

            counts: dict[str, int] = {}

            class _StoreVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    if node is function:
                        self.generic_visit(node)

                def visit_AsyncFunctionDef(
                    self,
                    node: ast.AsyncFunctionDef,
                ) -> None:
                    if node is function:
                        self.generic_visit(node)

                def visit_Name(self, node: ast.Name) -> None:
                    if isinstance(node.ctx, ast.Store):
                        counts[node.id] = counts.get(node.id, 0) + 1

            _StoreVisitor().visit(function)
            return counts

        # A mask spelling may be a local name even though its unique semantic
        # producer is the RHS of a preceding assignment.  Establish that
        # lexical def-use edge only in the narrow straight-line profile below.
        # It is not a Source-order correspondence between independent nodes:
        # the producer remains tied to the load solely through the Python
        # binding and later requires an exact compiler debug-location match.
        direct_definitions: dict[int, dict[str, tuple[ast.AST, int]]] = {}
        store_counts_by_scope: dict[int, dict[str, int]] = {}
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            scope_id = id(function)
            store_counts_by_scope[scope_id] = scope_store_counts(function)
            definitions: dict[str, tuple[ast.AST, int]] = {}
            for statement_index, statement in enumerate(function.body):
                target: ast.Name | None = None
                value: ast.AST | None = None
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    target = statement.targets[0]
                    value = statement.value
                elif isinstance(statement, ast.AnnAssign) and isinstance(
                    statement.target,
                    ast.Name,
                ) and statement.value is not None:
                    target = statement.target
                    value = statement.value
                if (
                    target is None
                    or value is None
                    or _source_semantic_operation_family(value) is None
                    or target.id in definitions
                ):
                    continue
                definitions[target.id] = (value, statement_index)
            direct_definitions[scope_id] = definitions

        def direct_statement_index(
            node: ast.AST,
            function: ast.AST,
            *,
            allow_guarded_if: bool = False,
        ) -> int | None:
            """Return a direct function-body statement index, or reject.

            Only a mask use under one direct ``if`` may consume a unique
            earlier function-body definition. All other region crossings
            remain ineligible for this lexical proof.
            """

            current = node
            guarded_if_count = 0
            while current is not function:
                parent_field = parent_fields.get(id(current))
                if parent_field is None:
                    return None
                parent, field_name = parent_field
                if isinstance(parent, control_regions):
                    if (
                        not allow_guarded_if
                        or not isinstance(parent, ast.If)
                        or field_name not in {"body", "orelse"}
                        or guarded_if_count != 0
                    ):
                        return None
                    guarded_if_count += 1
                if parent is function:
                    if field_name != "body" or not isinstance(current, ast.stmt):
                        return None
                    try:
                        return list(function.body).index(current)
                    except ValueError:
                        return None
                current = parent
            return None

        def resolved_mask_semantic_anchor(
            mask: ast.AST | None,
        ) -> tuple[ast.AST | None, str]:
            """Resolve one unambiguous local mask binding, otherwise fail closed."""

            if not isinstance(mask, ast.Name):
                return mask, "direct"
            function = enclosing_function(mask)
            if function is None:
                return None, "direct"
            scope_id = id(function)
            if store_counts_by_scope.get(scope_id, {}).get(mask.id) != 1:
                return None, "direct"
            definition = direct_definitions.get(scope_id, {}).get(mask.id)
            use_index = direct_statement_index(
                mask, function, allow_guarded_if=True,
            )
            if definition is None or use_index is None:
                return None, "direct"
            value, definition_index = definition
            if definition_index >= use_index:
                return None, "direct"
            return value, "exact_local_binding"

        def stable_predicate_expression_key(
            mask: ast.AST | None,
        ) -> str | None:
            """Identify repeated pure masks with stable straight-line inputs.

            This key is host-local. It cannot authorize a guard by itself:
            the compiler must show both accesses use the same predicate SSA.
            """

            def pure(node: ast.AST) -> bool:
                if isinstance(node, ast.Name):
                    return isinstance(node.ctx, ast.Load)
                if isinstance(node, ast.Constant):
                    return type(node.value) in {int, float, bool, type(None)}
                if isinstance(node, ast.BinOp):
                    return isinstance(node.op, ast.BitAnd) and pure(node.left) and pure(node.right)
                if isinstance(node, ast.UnaryOp):
                    return isinstance(node.op, ast.Invert) and pure(node.operand)
                if isinstance(node, ast.Compare):
                    return (
                        len(node.ops) == 1
                        and isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq))
                        and pure(node.left)
                        and len(node.comparators) == 1
                        and pure(node.comparators[0])
                    )
                if isinstance(node, ast.Subscript):
                    return pure(node.value) and pure(node.slice)
                if isinstance(node, ast.Tuple):
                    return all(pure(item) for item in node.elts)
                if isinstance(node, ast.Slice):
                    return all(item is None or pure(item) for item in (node.lower, node.upper, node.step))
                return False

            if mask is None or not pure(mask):
                return None
            function = enclosing_function(mask)
            if function is None:
                return None
            use_index = direct_statement_index(mask, function)
            if use_index is None:
                return None
            arguments = {
                arg.arg for arg in (
                    *function.args.posonlyargs, *function.args.args,
                    *function.args.kwonlyargs,
                )
            }
            definitions = direct_definitions.get(id(function), {})
            stores = store_counts_by_scope.get(id(function), {})
            names = {
                item.id for item in ast.walk(mask)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
            }
            for name in names:
                if name in arguments:
                    if stores.get(name, 0) != 0:
                        return None
                else:
                    definition = definitions.get(name)
                    if (
                        stores.get(name, 0) != 1
                        or definition is None
                        or definition[1] >= use_index
                    ):
                        return None
            return hashlib.sha256(ast.dump(mask, include_attributes=False).encode()).hexdigest()

        def direct_load_consumer_anchor(
            memory_call: ast.Call,
        ) -> tuple[ast.AST | None, str]:
            """Return one uniquely bound straight-line load consumer.

            A boundary load may name a Layer-A owner only when Python itself
            proves a single lexical def-use edge: a direct function-body
            assignment of the load to one local, followed by exactly one
            direct function-body semantic RHS that reads that local exactly
            once.  Any extra read, alternate store, control region, nested
            function, or non-semantic use leaves the owner unavailable.

            This establishes only the Source lexical edge.  The caller still
            requires an exact compiler debug-location/family counterpart, so
            neither Source order nor an AST/candidate label can select an
            owner.
            """

            parent = parents.get(id(memory_call))
            if not isinstance(parent, ast.Assign):
                return None, "direct"
            if (
                len(parent.targets) != 1
                or not isinstance(parent.targets[0], ast.Name)
                or parent.value is not memory_call
            ):
                return None, "direct"
            function = enclosing_function(memory_call)
            if function is None or enclosing_function(parent) is not function:
                return None, "direct"
            load_statement_index = direct_statement_index(memory_call, function)
            local = parent.targets[0].id
            if (
                load_statement_index is None
                or store_counts_by_scope.get(id(function), {}).get(local) != 1
            ):
                return None, "direct"

            uses: list[ast.Name] = []

            class _UseVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    if node is function:
                        self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                    if node is function:
                        self.generic_visit(node)

                def visit_Name(self, node: ast.Name) -> None:
                    if node.id == local and isinstance(node.ctx, ast.Load):
                        uses.append(node)

            _UseVisitor().visit(function)
            if len(uses) != 1:
                return None, "direct"
            use = uses[0]
            if enclosing_function(use) is not function:
                return None, "direct"
            use_statement_index = direct_statement_index(use, function)
            if use_statement_index is None or use_statement_index <= load_statement_index:
                return None, "direct"
            statement = function.body[use_statement_index]
            semantic_value: ast.AST | None = None
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and _source_semantic_operation_family(statement.value) is not None
            ):
                semantic_value = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
                and _source_semantic_operation_family(statement.value) is not None
            ):
                semantic_value = statement.value
            if semantic_value is None:
                return None, "direct"
            semantic_uses = [
                node
                for node in ast.walk(semantic_value)
                if isinstance(node, ast.Name)
                and node.id == local
                and isinstance(node.ctx, ast.Load)
            ]
            if semantic_uses != [use]:
                return None, "direct"
            return semantic_value, "exact_local_binding"

        def memory_argument(memory_call: ast.Call, name: str, position: int | None) -> ast.AST | None:
            keywords = [item.value for item in memory_call.keywords if item.arg == name]
            positional = memory_call.args[position] if position is not None and position < len(memory_call.args) else None
            if (
                len(keywords) > 1
                or (keywords and positional is not None)
                or any(isinstance(item, ast.Starred) for item in memory_call.args)
                or any(item.arg is None for item in memory_call.keywords)
            ):
                raise ValueError("source_memory_call_binding_unproved")
            return keywords[0] if keywords else positional

        def resolved_store_value_anchor(
            memory_call: ast.Call,
            *,
            kind: str,
        ) -> tuple[ast.AST | None, str]:
            """Return a unique direct semantic producer for one store value.

            Frozen constexpr branches have already been selected, preserving
            original locations. Find the last dominating definition in this
            straight-line body; any intervening unresolved control-flow write
            invalidates it. The actual typed compiler DAG remains mandatory.

            Atomic updates deliberately do not use this ordinary-store
            resolver.  In particular, ``atomic_cas`` has distinct compare and
            replacement operands, so treating positional operand one as a
            store value would create a false Source producer claim.  Until an
            update-specific typed proof records all atomic roles, return no
            producer for every update and keep authoring fail-closed.
            """

            if kind != "store":
                return None, "direct"

            value = memory_argument(memory_call, "value", 1)
            if value is None:
                return None, "direct"
            if _source_semantic_operation_family(value) is not None:
                return value, "direct"
            if not isinstance(value, ast.Name):
                return None, "direct"
            function = enclosing_function(value)
            if function is None:
                return None, "direct"
            use_index = direct_statement_index(value, function)
            if use_index is None:
                return None, "direct"
            semantic_value: ast.AST | None = None
            defined = value.id in {
                arg.arg for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
            }
            for statement in function.body[:use_index]:
                if any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(statement)):
                    return None, "direct"
                writes = [node for node in ast.walk(statement) if isinstance(node, ast.Name) and node.id == value.id and isinstance(node.ctx, (ast.Store, ast.Del))]
                if not writes:
                    continue
                rhs: ast.AST | None = None
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == value.id:
                    rhs = statement.value
                elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.target.id == value.id:
                    rhs = statement.value
                elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name) and statement.target.id == value.id and defined:
                    rhs = statement
                defined = rhs is not None and len(writes) == 1
                semantic_value = rhs if defined and _source_semantic_operation_family(rhs) is not None else None
            if semantic_value is None:
                return None, "direct"
            return semantic_value, "exact_local_binding"

        class _Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: list[str] = []
                self.call_paths: list[tuple[tuple[int, int], ...]] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.scope.append(node.name)
                self.call_paths.append(getattr(node, "_source_call_path", ()))
                self.generic_visit(node)
                self.call_paths.pop()
                self.scope.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.scope.append(node.name)
                self.call_paths.append(())
                self.generic_visit(node)
                self.call_paths.pop()
                self.scope.pop()

            @staticmethod
            def _compiler_access_anchor(
                memory_call: ast.Call,
                kind: str,
            ) -> ast.AST | None:
                """Return the exact Source child owning Triton's memory location.

                The pinned visitor evaluates keyword arguments before positional
                arguments and emits at the final child's location. With the
                usual keyword mask this is the load pointer or store value;
                positional masks and keyword-only calls retain their own exact
                emitter location instead of borrowing that conventional span.
                """

                if kind not in {"load", "store", "update"}:
                    return None
                if memory_call.args:
                    return memory_call.args[-1]
                if memory_call.keywords:
                    return memory_call.keywords[-1]
                return None

            def visit_Call(self, node: ast.Call) -> None:
                leaf = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else None
                )
                # Python evaluates the callable and its arguments before the
                # enclosing call takes effect.  Visit children first so a
                # nested Source load/store receives the earlier ledger
                # ordinal instead of being recorded after its enclosing call.
                self.generic_visit(node)
                if leaf in _SOURCE_MEMORY_CALLS:
                    kind = (
                        "load"
                        if leaf == "load"
                        else "store"
                        if leaf == "store"
                        else "update"
                    )
                    mask = memory_argument(node, "mask", 1 if kind == "load" else 2 if kind == "store" else None)
                    if isinstance(mask, ast.Constant) and (mask.value is None or mask.value is True):
                        mask = None
                    fill = memory_argument(node, "other", 2) if kind == "load" and mask is not None else None
                    predicate_node, predicate_binding_mode = (
                        resolved_mask_semantic_anchor(mask)
                    )
                    predicate_expression_key = stable_predicate_expression_key(mask)
                    consumer_node, consumer_binding_mode = (
                        direct_load_consumer_anchor(node)
                        if kind == "load"
                        else resolved_store_value_anchor(node, kind=kind)
                    )
                    rows.append(
                        TransferOccurrence(
                            ordinal=len(rows),
                            operation=str(leaf),
                            kind=kind,
                            masked=mask is not None,
                            predicate=_bounded_source_expression(mask),
                            fill_value=_bounded_source_expression(fill),
                            pointer_base=None,
                            pointer_position=None,
                            origin_positions=(
                                node.lineno
                                if isinstance(getattr(node, "lineno", None), int)
                                else 0,
                            ),
                            context=tuple(self.scope),
                            address_terms=(),
                            address_profile=None,
                            address_type_profile=(),
                            logical_ref=None,
                            tensor_rank=None,
                            element_type=None,
                            semantic_consumer=_opaque_source_semantic_occurrence(
                                consumer_node,
                                role="consumer",
                                scope=self.scope,
                                source_identity=source_identity,
                                binding_mode=consumer_binding_mode,
                                call_path=self.call_paths[-1] if self.call_paths else (),
                            ),
                            semantic_predicate=_opaque_source_semantic_occurrence(
                                predicate_node,
                                role="predicate",
                                scope=self.scope,
                                source_identity=source_identity,
                                binding_mode=predicate_binding_mode,
                                call_path=self.call_paths[-1] if self.call_paths else (),
                                expression_key=predicate_expression_key,
                            ) if mask is not None else None,
                            semantic_access=_opaque_source_semantic_occurrence(
                                self._compiler_access_anchor(node, kind),
                                role="access",
                                scope=self.scope,
                                source_identity=source_identity,
                                operation_family=f"memory_{kind}",
                                call_path=self.call_paths[-1] if self.call_paths else (),
                            ),
                        )
                    )

        try:
            _Visitor().visit(tree)
        except ValueError:
            return cls.unavailable("source_memory_call_binding_unproved", source_facts=source_facts)
        return cls(
            source_facts.digest if source_facts is not None else None,
            # The AST projection preserves lexical occurrence order for the
            # mounted MCP brief, but it deliberately lacks compiled address
            # and ABI bindings.  Keep it distinct from a closed empty ledger.
            "ambiguous" if rows else "pass",
            (
                "source_syntax_memory_navigation_untyped"
                if rows
                else "source_syntax_memory_navigation"
            ),
            tuple(rows),
        )

    def with_bindings(
        self,
        bindings: Sequence[Mapping[str, Any]],
        *,
        status: str | None = None,
        reason: str | None = None,
        diagnostics: Sequence[Mapping[str, Any]] | None = None,
    ) -> "SourceTransferLedger":
        return replace(
            self,
            status=self.status if status is None else status,
            reason=self.reason if reason is None else reason,
            bindings=tuple(bindings),
            diagnostics=(
                self.diagnostics
                if diagnostics is None
                else tuple(diagnostics)
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source_facts_digest": self.source_facts_digest,
            "status": self.status,
            "reason": self.reason,
            "occurrences": [item.to_json() for item in self.occurrences],
            "bindings": [dict(item) for item in self.bindings],
            "diagnostics": [dict(item) for item in self.diagnostics],
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SourceTransferLedger":
        raw = _copy_mapping(value, "SourceTransferLedger")
        if set(raw) != {
            "source_facts_digest",
            "status",
            "reason",
            "occurrences",
            "bindings",
            "diagnostics",
        }:
            raise ValueError("SourceTransferLedger wire has an incomplete shape")
        if (
            not isinstance(raw["occurrences"], list)
            or not isinstance(raw["bindings"], list)
            or not isinstance(raw["diagnostics"], list)
        ):
            raise TypeError("SourceTransferLedger wire collections must be arrays")
        rows: list[TransferOccurrence] = []
        for item in raw["occurrences"]:
            if not isinstance(item, Mapping):
                raise TypeError("SourceTransferLedger occurrence must be an object")
            profile = item.get("address_profile")
            parsed_profile = None
            if isinstance(profile, Mapping):
                parsed_profile = (
                    profile.get("dynamic_terms"),
                    profile.get("constant_terms"),
                    tuple(profile.get("constant_offsets", ())),
                )
            rows.append(
                TransferOccurrence(
                    ordinal=item["ordinal"],
                    operation=item["operation"],
                    kind=item["kind"],
                    masked=item["masked"],
                    predicate=item.get("predicate"),
                    fill_value=item.get("fill_value"),
                    pointer_base=item.get("pointer_base"),
                    pointer_position=item.get("pointer_position"),
                    origin_positions=tuple(item.get("origin_positions", ())),
                    context=tuple(item.get("context", ())),
                    address_terms=item.get("address_terms"),
                    address_profile=parsed_profile,
                    address_type_profile=item.get("address_type_profile"),
                    logical_ref=item.get("logical_ref"),
                    tensor_rank=item.get("tensor_rank"),
                    element_type=item.get("element_type"),
                    semantic_consumer=(
                        SourceSemanticOccurrence.from_json(item["semantic_consumer"])
                        if isinstance(item.get("semantic_consumer"), Mapping)
                        else None
                    ),
                    semantic_predicate=(
                        SourceSemanticOccurrence.from_json(item["semantic_predicate"])
                        if isinstance(item.get("semantic_predicate"), Mapping)
                        else None
                    ),
                    semantic_access=(
                        SourceSemanticOccurrence.from_json(item["semantic_access"])
                        if isinstance(item.get("semantic_access"), Mapping)
                        else None
                    ),
                )
            )
        return cls(
            raw["source_facts_digest"],
            raw["status"],
            raw["reason"],
            tuple(rows),
            tuple(raw["bindings"]),
            tuple(raw["diagnostics"]),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_json())


def coerce_source_transfer_ledger(
    value: SourceTransferLedger | Mapping[str, Any] | None,
    *,
    source_facts: SourceFacts | None = None,
) -> SourceTransferLedger:
    """Normalize every ledger ingress through one wire conversion.

    Source-ledger consumers receive either the immutable dataclass or a
    transport projection.  Keeping this conversion here prevents the effect
    reorderer, fill checker, and pi binder from disagreeing about whether a
    nested ``source_transfer_ledger`` object is authoritative.  A malformed
    or absent projection is an unavailable ledger, never an empty one.
    """

    def validate_lineage(
        ledger: SourceTransferLedger,
    ) -> SourceTransferLedger:
        # A transport ledger is only usable for the Source facts that created
        # it.  Without this check a structurally valid {occurrences,
        # bindings} projection from another kernel could be consumed by the
        # effect, fill, or pi path and silently create cross-kernel ownership.
        if source_facts is not None:
            if ledger.source_facts_digest != source_facts.digest:
                return SourceTransferLedger.unavailable(
                    "source_transfer_ledger_source_facts_disagree",
                    source_facts=source_facts,
                    diagnostics=(
                        {
                            "rule_code": "source_transfer_ledger_source_facts_disagree",
                            "path": "<source-transfer-ledger>.source_facts_digest",
                            "observed_refs_or_types": [
                                f"ledger={ledger.source_facts_digest!r}",
                                f"expected={source_facts.digest!r}",
                            ],
                        },
                    ),
                )
        return ledger

    if isinstance(value, SourceTransferLedger):
        return validate_lineage(value)
    if isinstance(value, Mapping):
        nested = value.get("source_transfer_ledger")
        # The reorderer exposes ``source_access_matches`` only as an
        # ingress-local projection.  It is never a ledger by itself.  Do not
        # let that envelope fall through to ``from_json``: because
        # ``occurrences`` and ``bindings`` have defaults there, the projection
        # could otherwise be silently accepted as an empty pass ledger.
        if "source_access_matches" in value and not isinstance(nested, Mapping):
            return SourceTransferLedger.unavailable(
                "source_transfer_ledger_wire_invalid",
                source_facts=source_facts,
            )
        # ``reorder_boundary_effects_from_source_ledger`` may retain a
        # diagnostic match projection after discovering an order barrier.  Its
        # nested ledger is useful for explaining the failure, but it is not a
        # usable downstream authority while the envelope is ambiguous or
        # unavailable.  Do not let the nested ``pass`` status bypass the
        # outer transaction status.
        outer_status = value.get("status")
        if outer_status in {"ambiguous", "unavailable"}:
            return SourceTransferLedger.unavailable(
                f"source_transfer_ledger_projection_{outer_status}",
                source_facts=source_facts,
            )
        if outer_status is not None and outer_status != "pass":
            return SourceTransferLedger.unavailable(
                "source_transfer_ledger_wire_invalid",
                source_facts=source_facts,
            )
        raw = nested if isinstance(nested, Mapping) else value
        # A canonical transport must carry both sides of the occurrence
        # ledger.  Missing either side is an invalid wire shape, not a valid
        # empty ledger.  This keeps every downstream consumer on the same
        # occurrence/binding authority and preserves fail-closed behavior.
        if not all(key in raw for key in ("status", "occurrences", "bindings")):
            return SourceTransferLedger.unavailable(
                "source_transfer_ledger_wire_invalid",
                source_facts=source_facts,
            )
        try:
            return validate_lineage(SourceTransferLedger.from_json(raw))
        except (TypeError, ValueError, KeyError):
            return SourceTransferLedger.unavailable(
                "source_transfer_ledger_wire_invalid",
                source_facts=source_facts,
            )
    return SourceTransferLedger.unavailable(
        "canonical_source_transfer_ledger_required",
        source_facts=source_facts,
    )


@dataclass(frozen=True)
class CanonicalHandleAllocator:
    """Single declaration-order allocator shared by ABI and J repair."""

    declarations: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # ``logical_bindings`` is the closed Source-to-J subset of the ``t*``
    # namespace.  It is intentionally separate from ``declarations`` because
    # a runtime scalar used only as a value is not a logical tensor until
    # Source semantics prove that use.  Shape-coordinate scalars, however,
    # are already proved by the Harness ABI and therefore belong here.
    # Rows are ``(p* ABI handle, t* handle, role)`` in ABI order.  Compiler
    # slots are intentionally excluded: a C-only carrier can never acquire a
    # J logical owner through this allocator.
    logical_bindings: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        rows = tuple(self.declarations)
        seen_names: set[str] = set()
        seen_prefixes: set[str] = set()
        normalized: list[tuple[str, tuple[str, ...]]] = []
        for prefix, values in rows:
            if prefix not in _HANDLE_PREFIXES:
                raise ValueError(f"unsupported canonical handle namespace {prefix!r}")
            if prefix in seen_prefixes:
                raise ValueError(f"canonical handle namespace {prefix!r} is repeated")
            seen_prefixes.add(prefix)
            ordered = tuple(values)
            if any(
                not isinstance(value, str) or re.fullmatch(rf"{prefix}[0-9]+", value) is None
                for value in ordered
            ) or len(ordered) != len(set(ordered)):
                raise ValueError(f"invalid {prefix}* declarations")
            if ordered != self.canonical_names(prefix, len(ordered)):
                raise ValueError(
                    f"{prefix}* declarations must be gap-free and declaration-ordered"
                )
            if seen_names & set(ordered):
                raise ValueError("canonical handle namespaces overlap")
            seen_names.update(ordered)
            normalized.append((prefix, ordered))
        object.__setattr__(self, "declarations", tuple(normalized))
        bindings = tuple(tuple(row) for row in self.logical_bindings)
        seen_abi: set[str] = set()
        seen_logical: set[str] = set()
        allowed_roles = {"tensor", "shape_scalar", "scalar"}
        declared_tensor_order = self.declared("t")
        declared_tensors = set(declared_tensor_order)
        for row in bindings:
            if (
                not isinstance(row, (tuple, list))
                or len(row) != 3
                or not all(isinstance(item, str) for item in row)
            ):
                raise ValueError("invalid canonical logical binding row")
            abi_name, logical_name, role = row
            if (
                re.fullmatch(r"p[0-9]+", abi_name) is None
                or re.fullmatch(r"t[0-9]+", logical_name) is None
                or role not in allowed_roles
                or abi_name in seen_abi
                or logical_name in seen_logical
                or logical_name not in declared_tensors
            ):
                raise ValueError("canonical logical bindings are not unique or declared")
            seen_abi.add(abi_name)
            seen_logical.add(logical_name)
        if bindings and (
            set(seen_logical) != declared_tensors
            or tuple(row[1] for row in bindings) != declared_tensor_order
        ):
            raise ValueError(
                "canonical logical bindings must cover t* declarations in order"
            )
        object.__setattr__(self, "logical_bindings", bindings)

    @classmethod
    def from_source_facts(
        cls,
        facts: SourceFacts,
        *,
        logical_scalar_parameters: Sequence[str] = (),
    ) -> "CanonicalHandleAllocator":
        p_names = tuple(
            sorted(
                (
                    opaque for _source, opaque in facts.source_to_opaque
                    if opaque.startswith("p")
                ),
                key=lambda value: int(value[1:]),
            )
        )
        c_names = tuple(
            sorted(
                (
                    opaque for _source, opaque in facts.source_to_opaque
                    if opaque.startswith("c")
                ),
                key=lambda value: int(value[1:]),
            )
        )
        tensor_inputs = [
            item.get("name")
            for item in facts.arguments
            if item.get("kind") in {"tensor", "pointer"}
            and item.get("mode") in {"input", "inout"}
            and isinstance(item.get("name"), str)
        ]
        tensor_outputs = [
            item.get("name")
            for item in facts.arguments
            if item.get("kind") in {"tensor", "pointer"}
            and item.get("mode") == "output"
            and isinstance(item.get("name"), str)
            and item.get("name") not in tensor_inputs
        ]
        shape_scalar_parameters = [
            item.get("name")
            for item in facts.arguments
            if item.get("kind") == "scalar"
            and isinstance(item.get("shape_coordinate"), str)
            and isinstance(item.get("name"), str)
        ]
        logical_scalar_set = {
            name for name in logical_scalar_parameters
            if isinstance(name, str) and name.startswith("p")
        }
        # Source scalar bindings are an input-side logical declaration.  Keep
        # the declaration order in the physical Source ABI, then append
        # output-only tensors.  Shape scalars remain logical rank-0 values
        # even when they are not used by a data expression; a scalar promoted
        # by the Source binder uses the distinct ``scalar`` role.
        logical_inputs = [
            name for name in facts.parameter_order
            if name in set(tensor_inputs)
            or name in set(shape_scalar_parameters)
            or name in logical_scalar_set
        ]
        tensor_parameters = (*logical_inputs, *tensor_outputs)
        if len(tensor_parameters) != len(set(tensor_parameters)):
            raise ValueError("SourceFacts tensor ABI declarations are ambiguous")
        logical_bindings = tuple(
            (
                name,
                cls.canonical_names("t", len(tensor_parameters))[index],
                "shape_scalar"
                if name in shape_scalar_parameters and name not in logical_scalar_set
                else "scalar"
                if name in logical_scalar_set
                else "tensor",
            )
            for index, name in enumerate(tensor_parameters)
        )
        source_owned = {
            "p": p_names,
            "c": c_names,
            "t": cls.canonical_names("t", len(tensor_parameters)),
            "x": cls.canonical_names("x", len(facts.shape_coordinates)),
        }
        return cls(
            tuple(
                (prefix, source_owned.get(prefix, ()))
                for prefix in _CANONICAL_NAMESPACE_ORDER
            ),
            logical_bindings=logical_bindings,
        )

    @classmethod
    def from_declarations(
        cls,
        declarations: Mapping[str, Sequence[str]],
        *,
        logical_bindings: Sequence[Sequence[str]] = (),
    ) -> "CanonicalHandleAllocator":
        unknown = set(declarations) - set(_HANDLE_PREFIXES)
        if unknown:
            raise ValueError(
                "unsupported canonical handle namespace(s): "
                + ", ".join(sorted(unknown))
            )
        ordered_declarations = tuple(
            (prefix, tuple(declarations.get(prefix, ())))
            for prefix in _CANONICAL_NAMESPACE_ORDER
            if prefix in declarations
        )
        return cls(
            ordered_declarations,
            logical_bindings=tuple(tuple(row) for row in logical_bindings),
        )

    def logical_binding_map(self) -> Mapping[str, tuple[str, str]]:
        """Return the closed ABI-to-logical bindings without re-inference."""

        return {
            abi_name: (logical_name, role)
            for abi_name, logical_name, role in self.logical_bindings
        }

    @classmethod
    def abi_mapping(
        cls,
        runtime_parameters: Sequence[str],
        compiler_slots: Sequence[str],
    ) -> Mapping[str, str]:
        """Allocate one declaration-order p*/c* ABI map.

        Source adapters decide which source parameters are runtime carriers and
        which are compiler-only slots.  The naming and collision rules live
        here, so Dataset, Triton and native paths cannot accidentally create
        different opaque orders for the same declaration order.
        """

        runtime = tuple(runtime_parameters)
        slots = tuple(compiler_slots)
        if any(not isinstance(name, str) or not name.isidentifier() for name in (*runtime, *slots)):
            raise ValueError("ABI names must be identifiers")
        if len(runtime) != len(set(runtime)) or len(slots) != len(set(slots)):
            raise ValueError("ABI names must be unique within their declaration class")
        if set(runtime) & set(slots):
            raise ValueError("runtime ABI parameters and compiler slots overlap")
        p_names = cls.allocate("p", len(runtime))
        c_names = cls.allocate("c", len(slots), used=p_names)
        return {
            **dict(zip(runtime, p_names, strict=True)),
            **dict(zip(slots, c_names, strict=True)),
        }

    def declared(self, prefix: str) -> tuple[str, ...]:
        for current, values in self.declarations:
            if current == prefix:
                return values
        return ()

    def owns_namespace(self, prefix: str) -> bool:
        """Return whether this allocator owns ``prefix``, including empty rows."""

        return any(current == prefix for current, _values in self.declarations)

    @staticmethod
    def canonical_names(prefix: str, count: int) -> tuple[str, ...]:
        if prefix not in _HANDLE_PREFIXES or count < 0:
            raise ValueError("invalid canonical handle sequence")
        return tuple(f"{prefix}{index}" for index in range(count))

    def rename_map(self, prefix: str) -> Mapping[str, str]:
        return {
            old: f"{prefix}{index}"
            for index, old in enumerate(self.declared(prefix))
        }

    def all_rename_maps(self) -> Mapping[str, Mapping[str, str]]:
        return {prefix: self.rename_map(prefix) for prefix, _values in self.declarations}

    def with_declarations(
        self, declarations: Mapping[str, Sequence[str]]
    ) -> "CanonicalHandleAllocator":
        merged = {prefix: values for prefix, values in self.declarations}
        for prefix, values in declarations.items():
            normalized = tuple(values)
            existing = merged.get(prefix)
            # Source-owned rows (p*/c*/t*/x*) are never replaced.  Candidate
            # namespaces are present as empty rows in the Source allocator so
            # the declaration-table order stays stable; their first complete
            # candidate declaration is the one allowed extension of that same
            # allocator.  A second non-empty declaration must agree exactly.
            if existing is not None and existing and existing != normalized:
                raise ValueError(
                    f"canonical {prefix}* declarations differ from owning allocator"
                )
            merged[prefix] = normalized
        return type(self).from_declarations(
            merged,
            logical_bindings=self.logical_bindings,
        )

    def for_candidate_j(
        self,
        j: Mapping[str, Any],
    ) -> "CanonicalHandleAllocator":
        """Extend this Source-owned allocator with one candidate J declaration table.

        SourceFacts remains authoritative for ``p*/c*/t*/x*``.  The candidate
        contributes only the declaration counts/order of its semantic
        namespaces; those rows are immediately converted to dense canonical
        names in this same allocator instance.  If the candidate is too
        incomplete to expose a structural table, return the Source-owned
        allocator unchanged so strict CandidateCheck remains the authority.
        A conflicting Source-owned count is likewise left for fail-closed
        validation instead of being guessed here.
        """

        declarations = canonical_authored_handle_declarations(j)
        if declarations is None:
            return self
        dynamic: dict[str, tuple[str, ...]] = {}
        for prefix, values in declarations.items():
            if prefix in CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES:
                continue
            dynamic[prefix] = self.canonical_names(prefix, len(values))
        if not dynamic:
            return self
        try:
            return self.with_declarations(dynamic)
        except ValueError:
            return self

    @staticmethod
    def allocate(prefix: str, count: int, *, used: Sequence[str] = ()) -> tuple[str, ...]:
        if prefix not in _HANDLE_PREFIXES or count < 0:
            raise ValueError("invalid canonical allocation request")
        used_set = set(used)
        result: list[str] = []
        index = 0
        while len(result) < count:
            value = f"{prefix}{index}"
            index += 1
            if value in used_set:
                continue
            result.append(value)
            used_set.add(value)
        return tuple(result)


@dataclass(frozen=True)
class CanonicalConstruction:
    """The one host-side input bundle for canonical J/C/pi construction."""

    source_facts: SourceFacts
    transfer_ledger: SourceTransferLedger
    allocator: CanonicalHandleAllocator
    logical_scalar_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_facts, SourceFacts):
            raise TypeError("CanonicalConstruction requires SourceFacts")
        if not isinstance(self.transfer_ledger, SourceTransferLedger):
            raise TypeError("CanonicalConstruction requires SourceTransferLedger")
        if not isinstance(self.allocator, CanonicalHandleAllocator):
            raise TypeError("CanonicalConstruction requires CanonicalHandleAllocator")
        ledger_digest = self.transfer_ledger.source_facts_digest
        if ledger_digest is not None and ledger_digest != self.source_facts.digest:
            raise ValueError(
                "CanonicalConstruction transfer ledger differs from SourceFacts"
            )
        logical_scalars = tuple(self.logical_scalar_parameters)
        if len(logical_scalars) != len(set(logical_scalars)) or any(
            not isinstance(name, str) or not name.startswith("p")
            for name in logical_scalars
        ):
            raise ValueError("CanonicalConstruction logical scalar declarations are invalid")
        expected = CanonicalHandleAllocator.from_source_facts(
            self.source_facts,
            logical_scalar_parameters=logical_scalars,
        )
        if (
            self.allocator.declarations != expected.declarations
            or self.allocator.logical_bindings != expected.logical_bindings
        ):
            raise ValueError(
                "CanonicalConstruction allocator differs from SourceFacts"
            )
        source_parameter_order = tuple(
            name
            for name in self.source_facts.parameter_order
            if name.startswith("p")
        )
        if source_parameter_order != expected.declared("p"):
            raise ValueError(
                "SourceFacts parameter order differs from canonical p* allocator"
            )
        source_c_parameter_order = tuple(
            name
            for name in self.source_facts.parameter_order
            if name.startswith("c")
        )
        canonical_c_order = (
            *source_c_parameter_order,
            *self.source_facts.compiler_slots,
        )
        if canonical_c_order != expected.declared("c"):
            raise ValueError(
                "SourceFacts source-C/compiler-slot order differs from canonical c* allocator"
            )

    @classmethod
    def from_compiler_view(
        cls,
        view: Any,
        *,
        frontend: str,
        source_kind: str,
        source_symbol: str,
        source_to_opaque: Mapping[str, str] | None = None,
        source_digest: str | None = None,
        source_slice_digest: str | None = None,
    ) -> "CanonicalConstruction":
        facts = SourceFacts.from_compiler_view(
            view,
            frontend=frontend,
            source_kind=source_kind,
            source_symbol=source_symbol,
            source_to_opaque=source_to_opaque,
            source_digest=source_digest,
            source_slice_digest=source_slice_digest,
        )
        return cls(
            source_facts=facts,
            transfer_ledger=SourceTransferLedger.unavailable(
                "ttir_observation_not_run",
                source_facts=facts,
            ),
            allocator=CanonicalHandleAllocator.from_source_facts(facts),
        )

    @classmethod
    def from_source_facts(
        cls,
        facts: SourceFacts,
        *,
        source_text: str | None = None,
    ) -> "CanonicalConstruction":
        if not isinstance(facts, SourceFacts):
            raise TypeError("CanonicalConstruction requires SourceFacts")
        ledger = (
            SourceTransferLedger.from_source_text(
                source_text,
                source_kind=facts.source_kind,
                source_facts=facts,
            )
            if source_text is not None
            else SourceTransferLedger.unavailable(
                "source_text_not_mounted", source_facts=facts
            )
        )
        logical_scalars = source_logical_scalar_parameters(facts, source_text)
        return cls(
            source_facts=facts,
            transfer_ledger=ledger,
            allocator=CanonicalHandleAllocator.from_source_facts(
                facts,
                logical_scalar_parameters=logical_scalars,
            ),
            logical_scalar_parameters=logical_scalars,
        )

    def with_observation(self, observed: Any) -> "CanonicalConstruction":
        return replace(
            self,
            transfer_ledger=SourceTransferLedger.from_observation(
                observed,
                source_facts=self.source_facts,
                logical_bindings=self.allocator.logical_binding_map(),
                lexical_ledger=self.transfer_ledger,
            ),
        )

    def _json_payload(self) -> dict[str, Any]:
        """Return the construction payload before its identity receipt.

        The mounted MCP bundle and the host request must refer to one exact
        SourceFacts/ledger/allocator snapshot.  Keep the digest outside the
        hashed payload so serializing the receipt cannot recurse into itself.
        """

        return {
            "source_facts": self.source_facts.to_json(),
            "source_facts_digest": self.source_facts.digest,
            "transfer_ledger": self.transfer_ledger.to_json(),
            "transfer_ledger_digest": self.transfer_ledger.digest,
            "logical_scalar_parameters": list(self.logical_scalar_parameters),
            "allocator": {
                prefix: list(values)
                for prefix, values in self.allocator.declarations
            }
            | {
                "logical_bindings": [list(row) for row in self.allocator.logical_bindings]
            },
        }

    def to_json(self) -> dict[str, Any]:
        payload = self._json_payload()
        payload["authority_digest"] = _digest(payload)
        return payload

    @property
    def authority_digest(self) -> str:
        return _digest(self._json_payload())


__all__ = [
    "CANONICAL_ARTIFACT_ROOT_FIELDS",
    "canonical_artifact_envelope_error",
    "require_canonical_artifact_envelope",
    "CANONICAL_C_WIRE_FIELDS",
    "CANONICAL_PI_REQUIRED_FIELDS",
    "CANONICAL_PI_WIRE_FIELDS",
    "CANONICAL_BOUNDARY_ACCESS_REQUIRED_FIELDS",
    "CANONICAL_BOUNDARY_ACCESS_WIRE_FIELDS",
    "CANONICAL_BOUNDARY_WIRE_FIELDS",
    "CANONICAL_CONSTRUCTION_RECORD_FIELDS",
    "CANONICAL_ELABORATED_OPERATION_RECORD_FIELDS",
    "CANONICAL_CANDIDATE_OWNED_HANDLE_PREFIXES",
    "CANONICAL_HANDLE_NAMESPACE_ORDER",
    "CANONICAL_HANDLE_NAMESPACE_PATTERNS",
    "CANONICAL_J_WIRE_FIELDS",
    "CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES",
    "CanonicalConstruction",
    "CanonicalConstructionRecords",
    "CanonicalHandleAllocator",
    "coerce_source_transfer_ledger",
    "CANONICAL_AUTHORED_HANDLE_PREFIXES",
    "SourceFacts",
    "SourceSemanticOccurrence",
    "SourceTransferLedger",
    "TransferOccurrence",
    "source_host_launch_contract_receipt",
    "is_canonical_ordinal",
    "source_fill_json_literal",
    "unique_exact_rank_insertion_projection",
    "source_logical_scalar_parameters",
    "canonical_archive_records",
    "canonical_authored_handle_declarations",
    "canonical_boundary_records",
    "canonical_operation_objects",
    "canonical_operation_records",
    "canonical_operation_wire_rows",
    "semantic_operation_wire_rows",
    "validated_j_computed_guard_bound",
    "canonical_output_records",
    "canonical_value_type_records",
    "canonical_scope_records",
]
