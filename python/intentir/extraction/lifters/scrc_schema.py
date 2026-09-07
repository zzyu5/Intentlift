"""Responses schema for backend-neutral J plus Source C/pi Lift evidence.

The local :class:`Gate1Artifact` parser remains authoritative.  This schema is
the wire contract given to the model so closed Gate1 objects do not drift into
invented field names.  The model, rather than a construction engine, produces
the complete realization witness.  The project parser performs final
cross-field and opaque-reference validation.
"""

from __future__ import annotations

import copy
from typing import Any

from intentir.ir.dimension_relation import DIMENSION_RELATION_CONST_OPS
from intentir.ir.intent_op_contract import LOGICAL_DTYPES, intent_op_json_schema
from intentir.verify.gate1.model import (
    BOUNDARY_ACCESS_REQUIRED_FIELDS,
    BOUNDARY_WIRE_FIELDS,
    GATE1_ARTIFACT_MODEL_FIELDS,
    SEMANTIC_ARCHIVE_WIRE_FIELDS,
    Gate1Artifact,
    render_boundary_access_wire_grammar,
    render_gate1_artifact_envelope_grammar,
)
from intentir.verify.gate1.pi import (
    ACCESS_REPRESENTATION_BINDING_RELATIONS,
    CONTROL_BINDING_RELATIONS,
    ITERATION_BINDING_RELATIONS,
    MATERIALIZATION_BINDING_RELATIONS,
    NORMALIZATION_WITNESS_SCHEMA,
    PROGRAM_BINDING_RELATIONS,
)


def _closed_object(
    properties: dict[str, Any],
    required: tuple[str, ...] = (),
    *,
    description: str | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }
    if description is not None:
        schema["description"] = description
    return schema


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_STRING = {"type": "string", "minLength": 1}
_LOGICAL_DTYPE = {"type": "string", "enum": sorted(LOGICAL_DTYPES)}
_TENSOR_HANDLE = {"type": "string", "pattern": "^t[0-9]+$"}
_NODE_HANDLE = {"type": "string", "pattern": "^n[0-9]+$"}
_C_HANDLE = {"type": "string", "pattern": "^c[0-9]+$"}
_C_SUBJECT_HANDLE = {"type": "string", "pattern": "^[tndaequb][0-9]+$"}
_PARAMETER_HANDLE = {"type": "string", "pattern": "^p[0-9]+$"}
_LOGICAL_DIMENSION_STRING = {
    "type": "string",
    "minLength": 1,
    "not": {
        "pattern": r"(^|[^A-Za-z0-9_])p[0-9]+($|[^A-Za-z0-9_])",
    },
}
_PORTABLE_PREDICATE_STRING = {
    "type": "string",
    "minLength": 1,
    "description": (
        "One boolean ContractExpr v1 expression: symbols/literals, arithmetic, "
        "comparisons, and/or/not, and only ceil_div/cdiv/min/max calls. No "
        "assignment, prose, range notation, frontend calls, attributes, or "
        "program/grid/tile/lane coordinate construction."
    ),
}
_DIMENSION_CONSTRAINT_STRING = {
    "type": "string",
    "minLength": 1,
    "description": (
        "One boolean affine ContractExpr over declared x* dimensions. The "
        "v4 checker accepts integer literals, +/-, constant multiplication, "
        "comparisons, and conjunction. It rejects unknown references, "
        "non-affine arithmetic, disjunction, and negation. This J-owned "
        "legal-domain relation is never filled from a Harness valuation. "
        "When Source exposes logical and rounded extents separately, declare "
        "both x* handles and their relation instead of substituting the "
        "current rounded integer."
    ),
}
_STRING_ARRAY = _array(_STRING)
_INT_OR_STRING = {
    "anyOf": [{"type": "integer"}, _LOGICAL_DIMENSION_STRING]
}


def render_semantic_archive_wire_grammar() -> str:
    """Render global archive invariants already enforced by the strict parser."""

    return (
        "J scopes are an exact derived dominance archive, not a copy of pi "
        "regions. Straight-line J uses one parentless function scope. Its "
        "definitions are exactly J input handles plus results actually produced "
        "by operations in that scope; a declared logical output is not a "
        "definition unless an operation really produces that same handle. "
        "captures and effects are likewise exact, with no convenience extras. "
        "dimension_constraints explicitly declares the closed affine legal "
        "domain of symbolic x* dimensions. Shape-dependent operations such "
        "as pad must state symbolic lo/hi expressions such as x1-x0 in Layer A; "
        "both dimensions remain declared when Source exposes both. Fixed code "
        "only parses and proves the authored relation and never reads a "
        "Harness dimension to repair it. "
        "When one Source scalar is both logical rank-0 data and a shape extent, "
        "keep the t* value and x* dimension distinct and explicitly declare "
        "their equality in value_dimension_relations; fixed code never creates it. "
        "The logical tensor ABI is ordered: declare every fixed input/inout in "
        "intent_function.inputs before declaring output-only tensors in "
        "intent_function.outputs. Optional scalar/constexpr parameters proven to "
        "be logical rank-0 values are inserted after all fixed inputs and before "
        "outputs, in physical ABI order. Allocate one gap-free final t* sequence "
        "from that order; an inout output repeats its existing input t* and does "
        "not consume a new handle. The value-free ABI lock carries only the stable "
        "physical output order, never a baseline output t* that could be shifted "
        "by a later scalar decision. Do not place a later input after an "
        "output-only tensor merely because the physical parameter order differs. "
        "The C slots array is also ordered: preserve the exact declaration order "
        "supplied by layer_c_declarations.slot_order and never sort c* names by "
        "their numeric suffix. "
        "Boundary domain coordinates k* are one global logical namespace, not "
        "fresh per-domain rank positions. Reuse a k* only for the same logical "
        "coordinate with the same extent in every domain that uses it. If two "
        "independent domains have different extents, give them different k* "
        "handles and update every associated access, predicate, and domain "
        "expression together; never repair the conflict with a concrete shape. "
        "Known IntentOp contracts are closed: a contraction attrs object contains "
        "only its canonical subscripts field. An implementation accumulation "
        "choice is represented, when justified, by a typed boundary "
        "realization_authorization, never by an accumulation_dtype op attribute. "
        "The closed map_unary spelling for boolean complement is logical_not; "
        "the bare function name not has no lowering. Axis_ref fields in C and "
        "pi use only an integer ordinal or an x* dimension handle; k* names are "
        "ContractExpr coordinates and are not valid axis_ref values. "
        "Every semantic loop has exactly one corresponding loop_region scope. "
        "Every effect_if has two explicit positive/negative arms bound to "
        "distinct sibling branch_region scopes. Either arm body may be empty; "
        "the conditional has no result/yield, branch-local SSA never escapes, "
        "and scope.effects plus matching EffectSpec/access_ref is the sole "
        "effect membership authority. "
        "Every value_if has one explicit result/type and two explicit "
        "positive/negative {scope_id,body,yield_value} arms bound to distinct "
        "sibling branch_region scopes. Either pure body may be empty; both "
        "yields match the result type, and only the enclosing result escapes. "
        "value_if is neither select nor effect_if and is never reconstructed "
        "by fixed code. "
        "The storage array may be empty: ordinary SSA temporaries do not have "
        "semantic storage or lifetime objects. Add StorageObject only when J "
        "semantics requires storage identity, such as an alias/destination or "
        "loop-carried state. write_once requires exactly one matching boundary "
        "write event; producing an SSA value is not such an event. Every loop "
        "state requires one storage object binding its state, body, next and "
        "result values plus a non-literal init value, if any, across iterations. "
        "Storage event references are kind-specific: scope_entry/scope_exit "
        "use q* scopes, node uses a stable n* node, access uses a* access, "
        "effect uses e* effect, and state uses the exact state_id declared by a "
        "loop carry. A tensor t* or ordinary SSA value v* is never a storage "
        "event reference. A realization authorization is a J anchor with a "
        "gap-free declaration-order id u0,u1,...; r* is not an authorization id. "
        "Native-atomic authorization is reserved for an actual atomic scatter "
        "node and names that stable scatter n*, never an ordinary map node, "
        "access, or effect. Aliases relate two distinct handles; an inout "
        "handle is not a self-alias."
    )


def _tensor_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "name": _TENSOR_HANDLE,
            "dtype": _LOGICAL_DTYPE,
            "shape": _array(_INT_OR_STRING),
        },
        ("name", "dtype", "shape"),
        description=(
            "Logical tensor shape handles are assigned x0,x1,... at first "
            "appearance across the complete inputs-then-outputs ABI and then "
            "reused. Do not swap or gap x* names, and do not replace an "
            "uncontracted rank extent with a hidden concrete value such as 1. "
            "A rank-1 tensor remains rank 1 even when its concrete shape is [1]; "
            "it is not a rank-0 logical scalar."
        ),
    )


def _value_dimension_relation_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "value_ref": _TENSOR_HANDLE,
            "dimension_ref": {
                "type": "string",
                "pattern": r"^x[0-9]+$",
            },
            "relation": {"type": "string", "enum": ["eq"]},
        },
        ("value_ref", "dimension_ref", "relation"),
        description=(
            "LLM-authored equality between one integral rank-0 logical J input "
            "t* and one declared symbolic shape dimension x*. Fixed code checks "
            "the equality but never infers or inserts this relation."
        ),
    )


def _dimension_relation_schema() -> dict[str, Any]:
    dimension = {"type": "string", "pattern": r"^x[0-9]+$"}
    identity = _closed_object(
        {
            "op": {"type": "string", "enum": ["identity"]},
            "arg": dimension,
        },
        ("op", "arg"),
    )
    const_expr = _closed_object(
        {
            "op": {
                "type": "string",
                "enum": sorted(DIMENSION_RELATION_CONST_OPS),
            },
            "arg": dimension,
            "value": {"type": "integer", "minimum": 1},
        },
        ("op", "arg", "value"),
    )
    product_expr = _closed_object(
        {
            "op": {"type": "string", "enum": ["mul_dims"]},
            "arg": dimension,
            "rhs": dimension,
        },
        ("op", "arg", "rhs"),
    )
    return _closed_object(
        {
            "result": dimension,
            "expr": {"oneOf": [identity, const_expr, product_expr]},
        },
        ("result", "expr"),
        description=(
            "LLM-authored closed definition of one declared logical dimension "
            "from another. Fixed code validates and consumes it but never infers "
            "the relation from Harness values, grid, C, pi, or compiler output."
        ),
    )


def _intent_schema() -> dict[str, Any]:
    logical_return = _closed_object(
        {"output": _STRING, "value": _STRING},
        ("output", "value"),
        description=(
            "Layer-A value-to-result binding in the function ABI. When the "
            "returns array is present, it is complete: it contains exactly "
            "one entry for every declared output, in the same order as "
            "IntentFunction.outputs, with no omissions or duplicates. "
            "Destination aliasing and partial-write behavior belong only to "
            "the canonical J boundary. If the same output also has a boundary "
            "output_mapping, its value must be exactly identical; the mapping "
            "does not override or replace this ABI entry."
        ),
    )
    function = _closed_object(
        {
            "name": {"type": "string", "enum": ["scrc_j0"]},
            "inputs": _array(_tensor_schema()),
            "outputs": _array(_tensor_schema()),
            "returns": _array(logical_return),
            "body": _array({"$ref": "#/$defs/intent_op"}),
        },
        ("name", "inputs", "outputs", "returns", "body"),
        description=(
            "inputs is the complete logical input ABI and precedes all "
            "output-only tensors in outputs. Assign t* handles by first "
            "appearance across inputs then outputs; an inout output repeats "
            "its input handle and does not allocate a new t*. Prefer a complete "
            "returns array for every output; it is mandatory even when empty. Its output "
            "handles must follow the declared output ABI order exactly. A "
            "boundary output_mapping is a separate sparse store/post-state "
            "anchor; it may coexist for the same output only with the exact "
            "same value, never an alternative value or a missing returns ABI "
            "entry."
        ),
    )
    return _closed_object(
        {"intent_function": function}, ("intent_function",)
    )


def _boundary_schema() -> dict[str, Any]:
    access = _closed_object(
        {
            "access_id": _STRING,
            "kind": {
                "type": "string",
                "enum": ["load", "store", "update"],
            },
            "tensor": _STRING,
            "predicate": _nullable(_PORTABLE_PREDICATE_STRING),
            "fill_value": {},
            "node_ref": _nullable(_STRING),
            "index_expression": _STRING,
        },
        BOUNDARY_ACCESS_REQUIRED_FIELDS,
        description=(
            "Portable logical access in ContractExpr v1. Direct dense surface "
            "forms k0/[k0] and k1,k0/[k1,k0] are deterministically elaborated "
            "to the same structured ordered axes. General non-direct affine "
            "relations remain typed expressions but can be unsupported by a "
            "particular RefEval capability profile. "
            "A node_ref makes the access consumer-local: the relation is not "
            "implicitly inherited by sibling uses of the same tensor. "
            "fill_value is legal only on a load with a non-null predicate. An "
            "unconditional access uses the unique canonical spelling with both "
            "predicate and fill_value absent/null; consumers reject rather than "
            "silently erase an orphan fill_value. For a mode=inout logical "
            "tensor, the authored boundary must contain a matching pre-state "
            "load and post-state store with read/write effects; an output mapping "
            "alone is not a store."
        ),
    )
    access["allOf"] = [
        {
            "if": {
                "required": ["fill_value"],
                "properties": {
                    "fill_value": {"not": {"type": "null"}},
                },
            },
            "then": {
                "required": ["kind", "predicate"],
                "properties": {
                    "kind": {"const": "load"},
                    "predicate": {"type": "string", "minLength": 1},
                },
            },
        }
    ]
    broadcast = _closed_object(
        {
            "node_ref": _STRING,
            "operand": _STRING,
            "axes": _array(_INT_OR_STRING),
        },
        ("node_ref", "operand", "axes"),
        description=(
            "Implicit operand-broadcast annotation for one map_binary or "
            "select node only. node_ref must name that node and operand must "
            "be one of its operands. Never emit a boundary.broadcasts entry "
            "for an explicit broadcast_in_dim node, reshape, transpose, or "
            "any other operation: the explicit operation's own attributes "
            "already carry that transformation."
        ),
    )
    access_guard_relation = _closed_object(
        {
            "access_ref": _STRING,
            "value_ref": _STRING,
            "axes": _array({"type": "integer", "minimum": 0}),
            "polarity": {
                "type": "string",
                "enum": ["positive", "negative"],
            },
        },
        ("access_ref", "value_ref", "axes", "polarity"),
        description=(
            "LLM-authored J relation from one declared boundary access to one "
            "declared Layer-A guard value. axes contains only JSON integer "
            "access-axis ordinals and polarity is closed. Fixed code never "
            "infers or inserts this relation."
        ),
    )
    alias = _closed_object(
        {
            "left": _STRING,
            "right": _STRING,
            "kind": {
                "type": "string",
                "enum": ["must_alias", "may_alias", "no_alias"],
            },
        },
        ("left", "right", "kind"),
        description=(
            "One source-established relation between two distinct J handles. "
            "Do not emit a self-alias such as left=right; inout identity is "
            "represented by reusing the tensor handle in inputs and outputs."
        ),
    )
    effect = _closed_object(
        {
            "effect_id": _STRING,
            "kind": {
                "type": "string",
                "enum": [
                    "read",
                    "write",
                    "read_write",
                    "update",
                ],
            },
            "target": _STRING,
            "node_ref": _nullable(_STRING),
            "access_ref": _nullable(_STRING),
        },
        ("effect_id", "kind", "target", "access_ref"),
    )
    output_mapping = _closed_object(
        {
            "output": _STRING,
            "value": _STRING,
            "node_ref": _nullable(_STRING),
        },
        ("output", "value"),
        description=(
            "One optional sparse authored canonical output binding, normally "
            "used to anchor an inout post-state or a boundary store producer. "
            "When IntentFunction.returns is present, returns remains a complete "
            "output-ABI list; if this output is also listed there, both entries "
            "must use exactly the same value. Never use output_mappings to "
            "replace a missing returns entry or to override it with a different "
            "value. When an inout output returns the same logical handle and no "
            "boundary store/post-state anchor is needed, leave "
            "output_mappings empty."
        ),
    )
    reduction = _closed_object(
        {
            "node_ref": _STRING,
            "tile_axis": _INT_OR_STRING,
            "identity": {"not": {"type": "null"}},
        },
        ("node_ref", "tile_axis", "identity"),
        description=(
            "Reduce or inclusive-scan algebra and frontend-local tile-axis "
            "metadata. consumer_extensions is the unique authority for "
            "physical outside-domain contributions."
        ),
    )
    typed_outside = _closed_object(
        {
            "dtype": _LOGICAL_DTYPE,
            "value": {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "integer"},
                    {"type": "number"},
                    {"type": "string"},
                ]
            },
        },
        ("dtype", "value"),
    )
    consumer_extension = _closed_object(
        {
            "consumer_ref": _NODE_HANDLE,
            "operand_index": {"type": "integer", "minimum": 0},
            "access_ref": {"type": "string", "pattern": "^a[0-9]+$"},
            "tile_axis": {"type": "integer", "minimum": 0},
            "outside": typed_outside,
            "identity_law": {
                "type": "string",
                "enum": ["two_sided"],
            },
        },
        (
            "consumer_ref",
            "operand_index",
            "access_ref",
            "tile_axis",
            "outside",
            "identity_law",
        ),
        description=(
            "One LLM-authored Layer-B relation from a predicate-bearing load "
            "access to one reduce/scan operand. The referenced BoundaryAccess "
            "continues to own the predicate; no predicate is promoted to "
            "Layer-A SSA. The relation supports only two-sided sum/max/min "
            "identities."
        ),
    )
    index_bounds = _closed_object(
        {
            "node_ref": _STRING,
            "data": _STRING,
            "indices": _STRING,
            "axis": {"type": "integer", "minimum": 0},
        },
        ("node_ref", "data", "indices", "axis"),
        description=(
            "Typed precondition 0 <= indices < dim(data, axis), exactly bound "
            "to one gather/scatter node. Never use a proof string or proven flag."
        ),
    )
    update = _closed_object(
        {
            "node_ref": _STRING,
            "base": _STRING,
            "indices": _STRING,
            "values": _STRING,
            "axis": {"type": "integer", "minimum": 0},
            "combiner": {
                "type": "string", "enum": ["overwrite", "add"],
            },
            "duplicate_policy": {
                "type": "string", "enum": ["forbid", "combine", "one_of"],
            },
            "order": {
                "type": "string", "enum": ["input_order", "unordered"],
            },
        },
        (
            "node_ref", "base", "indices", "values", "axis", "combiner",
            "duplicate_policy", "order",
        ),
        description=(
            "Only describes one actual J op=scatter with mode=along_axis. "
            "node_ref, base, indices, values, axis, and combiner must match "
            "that scatter exactly, and the scatter must have one matching "
            "IndexBoundsSpec. combiner=overwrite corresponds to scatter "
            "reduce=none and combiner=add to reduce=add. Do not use this "
            "entry for update_slice, a loop-carried update, a static affine "
            "inout store, or an ordinary store; those remain represented by "
            "their J operation plus the matching access/effect/output mapping. "
            "Atomic/CAS/thread ownership are realization choices and must not "
            "appear here."
        ),
    )
    result_relation = _closed_object(
        {
            "output": _STRING,
            "kind": {
                "type": "string",
                "enum": ["exact", "tolerance", "one_of_updates"],
            },
            "update_node_ref": _STRING,
        },
        ("output", "kind"),
        description=(
            "Comparison class owned by J. Numeric tolerance values are forbidden "
            "and come only from external Gate2 policy Q. one_of_updates requires "
            "one non-empty update_node_ref. exact/tolerance must omit the "
            "update_node_ref field entirely; empty string or null is invalid."
        ),
    )
    authorization = _closed_object(
        {
            "auth_id": {"type": "string", "pattern": "^u[0-9]+$"},
            "kind": {
                "type": "string",
                "enum": [
                    "signed_widen",
                    "unsigned_widen",
                    "numeric_cast",
                    "accumulation_dtype",
                    "reassociation",
                    "native_atomic",
                    "privatized_update",
                ],
            },
            "subject_ref": _STRING,
            "source_dtype": _STRING,
            "target_dtype": _STRING,
        },
        ("auth_id", "kind", "subject_ref"),
        description=(
            "J/B-owned authorization for one sensitive representation choice. "
            "Ids are declaration-ordered u0,u1,... and never r*. The subject "
            "must be an existing J carrier. For accumulation_dtype, do not add "
            "an accumulation_dtype field to contraction attrs; if a typed pair "
            "is emitted, source_dtype describes the subject J carrier/result "
            "dtype rather than a physical input operand dtype. native_atomic "
            "must name the stable n* of an actual atomic scatter, not an access, "
            "effect, or ordinary map node. Compact pi cannot cite, extend, or "
            "redefine this semantic contract."
        ),
    )
    return _closed_object(
        {
            "domains": {
                "type": "object",
                "propertyNames": {"pattern": "^d[0-9]+$"},
                "additionalProperties": _PORTABLE_PREDICATE_STRING,
                "description": (
                    "Map each d* handle directly to one ContractExpr string. "
                    "Never emit an axes/members/domain object here. k* axes "
                    "are global logical coordinates: reuse one only for the "
                    "same logical axis and extent; distinct extents require "
                    "distinct k* names in all matching accesses and predicates."
                ),
            },
            "accesses": _array(access),
            "broadcasts": _array(broadcast),
            "aliases": _array(alias),
            "effects": _array(effect),
            "output_mappings": _array(output_mapping),
            "reductions": _array(reduction),
            "index_bounds": _array(index_bounds),
            "updates": _array(update),
            "result_relations": _array(result_relation),
            "realization_authorizations": _array(authorization),
            "consumer_extensions": _array(consumer_extension),
            "access_guard_relations": _array(access_guard_relation),
        },
        BOUNDARY_WIRE_FIELDS,
        description=render_boundary_access_wire_grammar(),
    )


def _signature_schema() -> dict[str, Any]:
    parameter = _closed_object(
        {
            "name": _PARAMETER_HANDLE,
            "kind": {
                "type": "string",
                "enum": ["tensor", "pointer", "scalar", "opaque", "constexpr"],
            },
            "mode": {
                "type": "string",
                "enum": ["input", "output", "inout", "value"],
            },
            # A tensor/pointer or a parameter with logical_ref is part of the
            # logical J ABI and therefore uses the closed dtype set.  An
            # unbound scalar/constexpr/opaque carrier is source/compiler
            # plumbing; its public spelling is intentionally allowed to remain
            # a non-empty string (for example ``str`` in the trusted compiler
            # view).  The conditional below keeps this wire schema aligned
            # with SignatureParameter.__post_init__.
            "dtype": _nullable(_STRING),
            "rank": _nullable({"type": "integer", "minimum": 0}),
            "alias_of": _nullable(_STRING),
            "logical_ref": _nullable(
                {
                    "anyOf": [
                        _TENSOR_HANDLE,
                        {"type": "string", "pattern": r"^x[0-9]+$"},
                    ]
                }
            ),
        },
        ("name", "kind", "mode"),
        description=(
            "One J-owned logical callable handle. logical_ref links a "
            "pointer/tensor/scalar/semantic-constexpr handle only to a "
            "declared J input/output tensor or a symbolic J dimension; it "
            "never names a v* SSA value. Logical scalar data and constexpr "
            "values used as runtime-like logical data use a dtype-matching "
            "rank-0 J input and must "
            "contribute to an output dependency or explicit boundary/shape "
            "semantics. A runtime scalar used only for representation/address "
            "realization is kind=scalar, mode=value, rank=0, has public dtype "
            "but no logical_ref/J input; it remains an explicit ABI carrier and "
            "compact pi never assigns it semantic ownership. "
            "Integral "
            "extent-only handles instead bind a symbolic dimension. Every trusted "
            "compiler-view tensor argument, including one unused by the frozen "
            "specialization, keeps a logical_ref whose J tensor dtype, rank and "
            "shape exactly match that view. The inactive tensor remains a typed "
            "ABI declaration but has no invented operation, access or effect. "
            "Implementation-selection constexprs are C slots and never appear "
            "in this array. A constexpr used anywhere in the active graph as "
            "logical data, extent, shape, index, mask, predicate, or result binds "
            "t*/x* even when constant-folded. Only a Harness-fixed constexpr whose "
            "Source uses are exclusively compile-time graph discrimination may be "
            "a specialization carrier with kind=constexpr, mode=value, rank=0, "
            "public dtype and no logical_ref; its concrete value stays only in "
            "Harness, while Lift authors the complete active J graph. Lift, not "
            "fixed code, owns that semantic classification. The carrier is neither "
            "Layer-A data nor a C/backend choice. An opaque parameter is "
            "representation-only frontend plumbing such as cute.CopyAtom; it "
            "has dtype but no logical_ref and contributes no J semantics. A tensor "
            "parameter keeps its trusted rank even when a hidden concrete extent "
            "is one: a rank-1 tensor with shape [1] is not a rank-0 logical scalar."
        ),
    )
    parameter["allOf"] = [
        {
            "if": {
                "anyOf": [
                    {
                        "required": ["kind"],
                        "properties": {
                            "kind": {"enum": ["tensor", "pointer"]},
                        },
                    },
                    {
                        "required": ["logical_ref"],
                        "properties": {
                            "logical_ref": {
                                "type": "string",
                                "pattern": r"^(?:t|x)[0-9]+$",
                            },
                        },
                    },
                ],
            },
            "then": {
                "required": ["dtype"],
                "properties": {"dtype": _LOGICAL_DTYPE},
            },
        }
    ]
    return _closed_object(
        {"parameters": _array(parameter)},
        ("parameters",),
    )


def _c_schema() -> dict[str, Any]:
    slot = _closed_object(
        {
            "name": _C_HANDLE,
            "kind": {
                "type": "string",
                "enum": ["constexpr", "compiler"],
                "description": (
                    "constexpr is a callable frontend compile-time parameter; "
                    "compiler is an external compiler-invocation option and never "
                    "a callable or kernel-body carrier."
                ),
            },
            "role": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_]*$",
                "description": (
                    "Required for every C slot. For kind=compiler it is the "
                    "trusted compiler role and must be copied exactly. For "
                    "kind=constexpr it is a concise source-derived scheduling "
                    "label, or constexpr when no such label is justified; it "
                    "must remain lower_snake_case scheduling metadata rather "
                    "than prose, a Source identifier, an operator/predicate/"
                    "strategy/API label, or a semantic explanation. When a slot "
                    "is used as pi.iteration_bindings.tile_slot, its role must "
                    "be exactly tile_extent or reduction_tile_extent."
                ),
            },
            "value_type": _STRING,
            "required": {"type": "boolean"},
            "subject_ref": {
                **_C_SUBJECT_HANDLE,
                "description": (
                    "Optional source-neutral attachment to one declared J "
                    "tensor/node/domain/access/effect/scope/authorization/storage "
                    "subject. Effect, scope, authorization, and storage subjects "
                    "are axis-free. It is not a transformation permission or "
                    "source binding. Compiler-kind slots must omit subject_ref "
                    "and axis_ref."
                ),
            },
            "axis_ref": {
                "anyOf": [
                    {"type": "integer", "minimum": 0},
                    {"type": "string", "pattern": "^x[0-9]+$"},
                ],
                "description": (
                    "Optional logical axis in AxisRefs(J,subject_ref), "
                    "encoded only as a non-negative integer ordinal or an x* "
                    "dimension handle. It requires subject_ref and never uses "
                    "a k* ContractExpr coordinate or a physical axis name."
                ),
            },
        },
        ("name", "kind", "role"),
    )
    slot["allOf"] = [
        {
            "if": {
                "properties": {"kind": {"const": "compiler"}},
                "required": ["kind"],
            },
            "then": {
                "not": {
                    "anyOf": [
                        {"required": ["subject_ref"]},
                        {"required": ["axis_ref"]},
                    ]
                }
            },
        }
    ]
    return _closed_object(
        {
            "slots": {
                **_array(slot),
                "description": (
                    "Ordered value-free declarations. Preserve the exact order "
                    "of the supplied layer_c_declarations.slot_order; the list "
                    "is not sorted by the numeric suffix of c* handles. "
                    "CandidateCheck and BindSource use this same order."
                ),
            },
            "values": {
                "type": "object",
                "propertyNames": {"pattern": "^c[0-9]+$"},
                "maxProperties": 0,
                "additionalProperties": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "integer"},
                        {"type": "number"},
                        {"type": "string"},
                    ]
                },
                "description": (
                    "Fresh model-authored Lift output must be exactly empty. "
                    "Trusted local BindSource injects the real, complete, typed "
                    "bindings only after semantic authorship and before the "
                    "Gate1 artifact is persisted. A choice without an "
                    "implementation consumer or actual binding must be omitted "
                    "from slots rather than left for Replay to trim or synthesize."
                ),
            },
        },
        ("slots", "values"),
        description=(
            "Fresh Lift authors only the value-free C_source declarations and "
            "an empty values object. Trusted local BindSource produces the "
            "complete persisted Gate1 C_source for this exact anonymous "
            "frontend/Harness compilation. Neither form is a portable target-"
            "lowering plugin. CUDA/RISC-V/other C_target plugins are selected "
            "only after canonical J is frozen and never enter this Lift wire object."
        ),
    )


def _pi_schema() -> dict[str, Any]:
    """Active, certification-only pi subwire offered to Lift.

    Every accepted field has a closed typed meaning and binds directly to J/C.
    The full post-BindSource realization graph is intentionally absent: new Lift
    candidates do not serialize regions, instances, routes, carriers, source
    nodes, or target recipes.
    """

    program_axis = _closed_object(
        {
            "domain_ref": {
                "type": "string",
                "pattern": "^d[0-9]+$",
                "description": "Declared J boundary domain directly observed on this physical launch axis; the current profile keeps the binding hard.",
            },
            "axis": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
                "description": "Observed Triton launch axis; it does not enter Replay.",
            },
        },
        ("domain_ref", "axis"),
    )
    axis_ref = {
        "oneOf": [
            {"type": "integer", "minimum": 0},
            {"type": "string", "pattern": "^x[0-9]+$"},
        ],
        "description": (
            "Logical subject-axis reference: integer ordinal or x* dimension; "
            "never a k* boundary coordinate."
        ),
    }
    program_logical_axis = _closed_object(
        {
            "domain_ref": {"type": "string", "pattern": "^d[0-9]+$"},
            "axis_ref": axis_ref,
        },
        ("domain_ref", "axis_ref"),
    )
    program_binding = _closed_object(
        {
            "axis": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
                "description": (
                    "Observed physical launch axis for this one closed program "
                    "coordinate; it is part of the same mapping as relation."
                ),
            },
            "relation": {
                "type": "string",
                "enum": list(PROGRAM_BINDING_RELATIONS),
            },
            "logical_axes": {
                "type": "array",
                "minItems": 1,
                "items": program_logical_axis,
            },
        },
        ("axis", "relation", "logical_axes"),
        description=(
            "One observed physical program coordinate, including its closed "
            "launch-axis identity and its relation over named J domain axes. "
            "No Source/Replay node identity or coordinate formula is authored."
        ),
    )
    iteration_logical_axis = _closed_object(
        {
            "subject_ref": {
                "type": "string",
                "pattern": "^(?:d|n)[0-9]+$",
            },
            "axis_ref": axis_ref,
        },
        ("subject_ref", "axis_ref"),
    )
    iteration_binding = _closed_object(
        {
            "relation": {
                "type": "string",
                "enum": list(ITERATION_BINDING_RELATIONS),
            },
            "logical_axes": {
                "type": "array",
                "minItems": 1,
                "items": iteration_logical_axis,
            },
            "tile_slot": {"type": "string", "pattern": "^c[0-9]+$"},
            "mode_slot": _nullable(
                {"type": "string", "pattern": "^c[0-9]+$"}
            ),
        },
        ("relation", "logical_axes", "tile_slot", "mode_slot"),
        description=(
            "Closed equivalence between serial/program tiled realization and "
            "the named J logical axes under one explicitly attached C tile. "
            "logical_axes.subject_ref is only a declared J domain d* or node "
            "n* (a tiled_domain uses d*; a tiled_recurrence uses an authored "
            "loop n*), never a tensor t* or SSA value v*. The referenced "
            "tile_slot must have role tile_extent or reduction_tile_extent and "
            "must be attached to the same logical axis."
        ),
    )

    def anchor_binding(
        *, relation_values: tuple[str, ...], field: str, pattern: str, description: str
    ) -> dict[str, Any]:
        return _closed_object(
            {
                "relation": {"type": "string", "enum": list(relation_values)},
                field: {"type": "string", "pattern": pattern},
            },
            ("relation", field),
            description=description,
        )

    control_binding = anchor_binding(
        relation_values=CONTROL_BINDING_RELATIONS,
        field="subject_ref",
        pattern="^(?:d|n|e)[0-9]+$",
        description="Closed terminal-control representation bound to J control/domain/effect.",
    )
    materialization_binding = anchor_binding(
        relation_values=MATERIALIZATION_BINDING_RELATIONS,
        field="value_ref",
        pattern="^(?:t|n|v)[0-9]+$",
        description="Closed scratch/recompute representation bound to one J value.",
    )
    access_binding = anchor_binding(
        relation_values=ACCESS_REPRESENTATION_BINDING_RELATIONS,
        field="access_ref",
        pattern="^a[0-9]+$",
        description=(
            "Closed pointer/address representation bound to one J access; tensor, "
            "logical index, predicate and effect remain hard."
        ),
    )
    return _closed_object(
        {
            "schema": {"const": NORMALIZATION_WITNESS_SCHEMA},
            "program_axes": _array(program_axis),
            "program_bindings": _array(program_binding),
            "iteration_bindings": _array(iteration_binding),
            "control_bindings": _array(control_binding),
            "materialization_bindings": _array(materialization_binding),
            "access_representation_bindings": _array(access_binding),
        },
        ("schema", "program_axes"),
        description=(
            "Source-realization witness pi_source for Gate1 only. It never "
            "enters Replay or production Lower and is not C_target. "
            "Author only closed relation kinds directly bound to existing J/C "
            "handles. Rows contain no source/replay node, target recipe, free-text "
            "equivalence or allowed diff. Unknown relations remain hard."
        ),
    )


def _portable_scope_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "scope_id": {"type": "string", "pattern": r"^q[0-9]+$"},
            "kind": {
                "type": "string",
                "enum": [
                    "function", "program_instance", "loop_region",
                    "branch_region",
                ],
            },
            "parent_scope": _nullable(
                {"type": "string", "pattern": r"^q[0-9]+$"}
            ),
            "captures": _STRING_ARRAY,
            "definitions": _STRING_ARRAY,
            "effects": _STRING_ARRAY,
        },
        (
            "scope_id", "kind", "parent_scope", "captures", "definitions",
            "effects",
        ),
        description=(
            "Exact portable dominance/capture/effect scope derived from J. "
            "definitions/captures/effects must contain neither missing nor extra "
            "handles. Contains no source AST, helper name or frontend spelling."
        ),
    )


def _storage_schema() -> dict[str, Any]:
    event = _closed_object(
        {
            "kind": {
                "type": "string",
                "enum": [
                    "scope_entry", "scope_exit", "node", "access", "state",
                    "effect",
                ],
            },
            "ref": _STRING,
        },
        ("kind", "ref"),
        description=(
            "Storage lifetime event reference with a kind-specific namespace: "
            "scope_entry/scope_exit -> q* scope, node -> stable n* node, "
            "access -> a* access, effect -> e* effect, and state -> the exact "
            "state_id of a semantic loop carry. Do not use tensor t* or SSA "
            "value v* as an event ref."
        ),
    )
    return _closed_object(
        {
            "storage_id": {
                "type": "string",
                "pattern": r"^(?:b|s)[0-9]+$",
            },
            "members": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": r"^(?:t|v|s)[0-9]+$",
                },
                "minItems": 1,
                "uniqueItems": True,
            },
            "scope_id": {"type": "string", "pattern": r"^q[0-9]+$"},
            "mutability": {
                "type": "string",
                "enum": ["immutable", "write_once", "mutable"],
            },
            "init_event": event,
            "last_use_event": event,
            "escapes": {"type": "boolean"},
            "crosses_iteration": {"type": "boolean"},
        },
        (
            "storage_id", "members", "scope_id", "mutability", "init_event",
            "last_use_event", "escapes", "crosses_iteration",
        ),
        description=(
            "Optional semantic storage identity/lifetime only; ordinary SSA "
            "temporaries require no entry. Physical allocation, register/shared/"
            "global placement and double buffering belong to C."
        ),
    )


def _location_map_schema() -> dict[str, Any]:
    span = _closed_object(
        {
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "start_column": _nullable({"type": "integer", "minimum": 0}),
            "end_column": _nullable({"type": "integer", "minimum": 0}),
        },
        ("start_line", "end_line"),
    )
    entry = _closed_object(
        {
            "span": span,
            "node_refs": _STRING_ARRAY,
            "access_refs": _STRING_ARRAY,
        },
        ("span", "node_refs", "access_refs"),
    )
    return _closed_object(
        {
            "source_name": {
                "type": "string", "enum": ["anonymous_source"]
            },
            "entries": _array(entry),
        },
        ("source_name", "entries"),
    )


def scrc_lift_core_json_schema() -> dict[str, Any]:
    """Return the complete descriptive source-aware response schema.

    This is the authoritative local schema used by CandidateCheck and the
    mounted schema-inspection tools. The provider can use the machine-only
    projection returned by :func:`scrc_lift_core_model_json_schema` when the
    duplicated prose would needlessly enlarge a bounded Lift request.
    """

    semantic_archive = _closed_object(
        {
            "kernel_id": {"type": "string", "enum": ["scrc_k0"]},
            "intent": _intent_schema(),
            "node_ids": {
                "type": "object",
                "propertyNames": {"pattern": "^n[0-9]+$"},
                "additionalProperties": {
                    "type": "string",
                    "pattern": (
                        "^body\\[[0-9]+\\]"
                        "((\\.body|\\.positive\\.body|\\.negative\\.body)"
                        "\\[[0-9]+\\])*$"
                    ),
                },
                "description": (
                    "Map n0,n1,... to structural paths such as body[0] or "
                    "body[1].positive.body[0], "
                    "not to SSA/result handles such as v0."
                ),
            },
            "boundary": _boundary_schema(),
            "signature": _signature_schema(),
            "scopes": {
                "type": "array",
                "minItems": 1,
                "items": _portable_scope_schema(),
            },
            "storage": _array(_storage_schema()),
            "dimension_constraints": _array(
                _DIMENSION_CONSTRAINT_STRING
            ),
            "value_dimension_relations": _array(
                _value_dimension_relation_schema()
            ),
            "dimension_relations": _array(_dimension_relation_schema()),
        },
        SEMANTIC_ARCHIVE_WIRE_FIELDS,
        description=render_semantic_archive_wire_grammar(),
    )
    schema = _closed_object(
        {
            "schema": {
                "type": "string",
                "enum": [Gate1Artifact.SCHEMA],
            },
            "j": semantic_archive,
            "c": _c_schema(),
            "pi": _pi_schema(),
        },
        GATE1_ARTIFACT_MODEL_FIELDS,
        description=render_gate1_artifact_envelope_grammar(),
    )
    # ``intent_op_json_schema`` is embedded below the artifact root, while its
    # recursive operation schema uses document-root ``#/$defs/...`` references.
    # Keep both definitions at this document root.  In particular, a nested
    # const literal must resolve against the same root as the operation body;
    # leaving ``const_literal`` only inside ``$defs.intent_op`` makes a valid
    # candidate fail with PointerToNowhere before CandidateCheck can diagnose
    # its actual contents.
    intent_op_schema = intent_op_json_schema()
    schema["$defs"] = {
        "const_literal": copy.deepcopy(
            intent_op_schema["$defs"]["const_literal"]
        ),
        "intent_op": intent_op_schema,
    }
    return copy.deepcopy(schema)


def _without_descriptions(value: Any) -> Any:
    """Remove schema prose while preserving every machine constraint."""

    if isinstance(value, dict):
        return {
            key: _without_descriptions(item)
            for key, item in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    return value


def scrc_lift_core_model_json_schema() -> dict[str, Any]:
    """Return the compact machine-only schema used on the provider wire."""

    return _without_descriptions(scrc_lift_core_json_schema())


def scrc_lift_consistency_only_text_format() -> dict[str, Any]:
    """Return the narrow provider schema for Source-to-J consistency only.

    This is a wire-side authoring restriction, not an acceptance or repair
    path.  The local CandidateCheck/Decode/TypeCheck path still receives and
    validates the complete core schema.  Consistency-only has no Gate1
    normalization or trusted C valuation, so carrying schemas for non-empty
    pi relations and C values only expands the provider search space.
    """

    result = scrc_lift_core_text_format(compact=True)
    schema = result["schema"]
    assert isinstance(schema, dict)

    c = schema["properties"]["c"]
    assert isinstance(c, dict)
    c_values = c["properties"]["values"]
    assert isinstance(c_values, dict)
    c_values.clear()
    c_values.update(
        {
            "type": "object",
            "maxProperties": 0,
            "additionalProperties": False,
        }
    )

    pi = schema["properties"]["pi"]
    assert isinstance(pi, dict)
    pi_properties = pi["properties"]
    assert isinstance(pi_properties, dict)
    for name in tuple(pi_properties):
        if name == "schema":
            continue
        value = pi_properties[name]
        if not isinstance(value, dict) or value.get("type") != "array":
            raise ValueError(
                f"consistency-only pi field {name!r} is not an array"
            )
        if name != "program_axes":
            del pi_properties[name]
            continue
        value.pop("items", None)
        value["maxItems"] = 0
    return result


def scrc_lift_core_text_format(*, compact: bool = False) -> dict[str, Any]:
    """Return the Responses format for Lift's source-aware LLM substage."""

    return {
        "type": "json_schema",
        "name": "scrc_lift_core",
        # Operation-specific semantic attrs remain open during the IntentOp
        # robustness audit. B is the one closed top-level boundary; C and pi
        # are separate closed objects.
        "strict": False,
        "schema": (
            scrc_lift_core_model_json_schema()
            if compact
            else scrc_lift_core_json_schema()
        ),
    }


__all__ = [
    "render_boundary_access_wire_grammar",
    "render_gate1_artifact_envelope_grammar",
    "render_semantic_archive_wire_grammar",
    "scrc_lift_core_json_schema",
    "scrc_lift_core_model_json_schema",
    "scrc_lift_consistency_only_text_format",
    "scrc_lift_core_text_format",
]
