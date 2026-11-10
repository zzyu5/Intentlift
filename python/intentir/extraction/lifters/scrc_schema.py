"""Responses schema for backend-neutral J plus Source C/pi Lift evidence.

The local :class:`Gate1Artifact` parser remains authoritative.  This schema is
the wire contract from which the provider semantic surface and host canonical
constructor are both projected.  The provider authors only unresolved J
semantics and an optional J-proven pi witness; the constructor produces the
complete realization artifact before final cross-field and opaque-reference
validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from intentir.ir.dimension_relation import DIMENSION_RELATION_CONST_OPS
from intentir.ir.canonical_construction import (
    CANONICAL_ARTIFACT_ROOT_FIELDS,
    CANONICAL_BOUNDARY_ACCESS_REQUIRED_FIELDS,
    CANONICAL_BOUNDARY_WIRE_FIELDS,
    CANONICAL_CANDIDATE_OWNED_HANDLE_PREFIXES,
    CANONICAL_C_WIRE_FIELDS,
    CANONICAL_CONSTRUCTION_RECORD_FIELDS,
    CANONICAL_HANDLE_NAMESPACE_ORDER,
    CANONICAL_HANDLE_NAMESPACE_PATTERNS,
    CANONICAL_J_WIRE_FIELDS,
    CANONICAL_PI_REQUIRED_FIELDS,
    CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES,
    SOURCE_OWNER_TOKEN_PATTERN,
    is_canonical_ordinal,
    source_host_launch_contract_receipt,
    source_fill_json_literal,
    tensor_owner_digest,
    unique_exact_rank_insertion_projection,
)
from intentir.ir.intent_op_contract import (
    CANONICAL_DTYPES,
    CANONICAL_J_OPS,
    STRUCTURED_INTENT_OPS,
    intent_op_json_schema,
)
from intentir.ir.source_projection_contracts import (
    PROVIDER_DIMENSION_EXTENT_RULE,
    build_source_semantic_obligations,
    owner_broadcast_requirements,
    provider_dimension_alias_map,
    provider_shape_from_canonical,
    validate_source_semantic_obligations,
)
from intentir.extraction.lifters.source_store_partition import (
    validate_source_store_partition_assembly,
)
from intentir.verify.gate1.model import (
    Gate1Artifact,
    render_boundary_access_wire_grammar,
    render_gate1_artifact_envelope_grammar,
)
from intentir.verify.gate1.pi import (
    ACCESS_REPRESENTATION_BINDING_RELATIONS,
    CONTROL_BINDING_RELATIONS,
    ITERATION_BINDING_RELATIONS,
    MATERIALIZATION_BINDING_RELATIONS,
    NORMALIZATION_CLASS_SCOPES,
    NORMALIZATION_WITNESS_SCHEMA,
    PROGRAM_BINDING_RELATIONS,
)


def unique_access_guard_projection_axes(
    predicate_shape: Sequence[str | int],
    target_shape: Sequence[str | int],
) -> tuple[int, ...] | None:
    """Compatibility name for the canonical exact guard projection."""

    return unique_exact_rank_insertion_projection(predicate_shape, target_shape)


# This is the only provider-authored projection of a candidate.  The complete
# artifact remains schema/j/c/pi, but the host constructor owns every omitted
# mechanical sibling.  Keep this declaration beside the canonical schema
# projection so MCP, direct Responses authoring, and semantic ingress cannot
# drift into separate root/J field lists.
CANONICAL_SEMANTIC_CANDIDATE_ROOT_FIELDS = ("j", "pi")
CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS = (
    "intent",
    "boundary",
    "storage",
    "dimension_constraints",
    "value_dimension_relations",
    "dimension_relations",
)


def canonical_semantic_candidate_contract() -> dict[str, Any]:
    """Return the one schema-derived provider authoring surface.

    The full artifact schema deliberately remains broader: the host constructs
    its Source-owned mechanical fields before CandidateCheck sees it.  Every
    provider-facing ingress must use this projection rather than infer a field
    allowance from a particular Source seed.
    """

    profile = canonical_wire_profile()
    value = profile.get("semantic_candidate")
    if not isinstance(value, dict):
        raise AssertionError("canonical wire profile lacks semantic candidate contract")
    root_fields = value.get("root_fields")
    j_fields = value.get("j_fields")
    required_paths = value.get("required_j_paths")
    template = value.get("template")
    j_field_types = value.get("j_field_types")
    boundary_field_types = value.get("boundary_field_types")
    mcp_write = profile.get("semantic_mcp_write")
    if (
        root_fields != list(CANONICAL_SEMANTIC_CANDIDATE_ROOT_FIELDS)
        or j_fields != list(CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS)
        or required_paths != ["j.intent.intent_function.body", "j.boundary"]
        or not isinstance(template, Mapping)
        or not isinstance(j_field_types, dict)
        or not isinstance(boundary_field_types, dict)
        or not isinstance(value.get("output_selection"), dict)
        or not isinstance(value.get("operation_namespace"), dict)
        or not isinstance(mcp_write, Mapping)
    ):
        raise AssertionError("canonical semantic candidate contract drifted")
    template_j = template.get("j")
    template_pi = template.get("pi")
    if (
        set(template) != set(CANONICAL_SEMANTIC_CANDIDATE_ROOT_FIELDS)
        or not isinstance(template_j, Mapping)
        or not isinstance(template_pi, Mapping)
        or set(template_j) != set(CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS)
        or template_pi
        or not isinstance(template_j.get("intent"), Mapping)
        or not isinstance(template_j["intent"].get("intent_function"), Mapping)
        or not isinstance(template_j["intent"]["intent_function"].get("body"), list)
    ):
        raise AssertionError("canonical semantic candidate template is malformed")
    for field in CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS:
        expected = j_field_types.get(field)
        actual = template_j.get(field)
        if expected == "object" and not isinstance(actual, Mapping):
            raise AssertionError("canonical semantic candidate template object drifted")
        if expected == "array" and not isinstance(actual, list):
            raise AssertionError("canonical semantic candidate template array drifted")

    writes = mcp_write.get("writes")
    steps = mcp_write.get("steps")
    mcp_tool_arguments = mcp_write.get("tool_arguments")
    mcp_descriptions = mcp_write.get("descriptions")
    required_mcp_steps = mcp_write.get("required_steps")
    if (
        mcp_write.get("initial_template") != template
        or not isinstance(writes, Mapping)
        or not isinstance(steps, list)
        or not steps
        or any(not isinstance(step, str) or not step for step in steps)
        or len(steps) != len(set(steps))
        or steps[0] != "begin_candidate"
        or steps[-1] != "finalize_candidate"
        or not isinstance(mcp_tool_arguments, Mapping)
        or not isinstance(mcp_descriptions, Mapping)
        or not isinstance(required_mcp_steps, list)
        or not required_mcp_steps
        or any(
            not isinstance(step, str) or step not in steps
            for step in required_mcp_steps
        )
        or len(required_mcp_steps) != len(set(required_mcp_steps))
        or any(
            steps.index(left) >= steps.index(right)
            for left, right in zip(required_mcp_steps, required_mcp_steps[1:])
        )
        or required_mcp_steps[0] != steps[0]
        or required_mcp_steps[-1] != steps[-1]
        or set(mcp_tool_arguments) != set(steps)
        or set(mcp_descriptions) != set(steps)
        or any(
            not isinstance(description, str) or not description
            for description in mcp_descriptions.values()
        )
        or writes.get("operation") != "j.intent.intent_function.body[]"
        or writes.get("access") != "j.boundary.accesses[]"
        or writes.get("output_mapping") != "j.boundary.output_mappings[]"
        or writes.get("j_field")
        != [
            field
            for field in CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS
            if field not in {"intent", "boundary"}
        ]
        or writes.get("boundary_field")
        != [
            field
            for field in boundary_field_types
            if field not in {"accesses", "output_mappings", "effects", "index_bounds"}
        ]
    ):
        raise AssertionError("canonical MCP write contract drifted")
    return copy.deepcopy(value)


def canonical_semantic_candidate_template() -> dict[str, Any]:
    """Return the sole provider semantic seed derived from the wire profile.

    This is deliberately a semantic surface, not a partial final artifact.
    The host still owns Source ABI/C rows and every mechanical J projection.
    Keeping the seed beside the profile makes MCP source-brief, direct Lift,
    and the ingress checker agree on omitted collection defaults.
    """

    contract = canonical_semantic_candidate_contract()
    template = contract["template"]
    if not isinstance(template, dict):
        raise AssertionError("canonical semantic candidate template is malformed")
    return copy.deepcopy(template)


def canonical_semantic_operation_namespace() -> dict[str, Any]:
    """Return the one schema-derived rule for names on the J op surface.

    This compact classification is intentionally separate from exact operation
    leaves.  It prevents a Source/API spelling from becoming a J ``op`` while
    keeping first-turn context small; callers use ``schema_lookup`` for a
    particular closed operation contract.
    """

    contract = canonical_semantic_candidate_contract()
    value = contract.get("operation_namespace")
    if not isinstance(value, dict):
        raise AssertionError("canonical semantic operation namespace is missing")
    return copy.deepcopy(value)


def render_canonical_semantic_candidate_skeleton() -> str:
    """Render the sole semantic candidate layout without parallel prose."""

    return json.dumps(
        canonical_semantic_candidate_template(),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def render_canonical_semantic_candidate_layout_summary() -> str:
    """Render a bounded diagnostic view of the sole semantic skeleton."""

    contract = canonical_semantic_candidate_contract()
    root_fields = contract["root_fields"]
    j_fields = contract["j_fields"]
    summary = (
        "root="
        + ",".join(root_fields)
        + ";j={"
        + ",".join(j_fields)
        + "};pi=sibling"
    )
    if len(summary) > 128:
        raise AssertionError("canonical semantic layout diagnostic exceeds its wire bound")
    return summary


def render_canonical_semantic_operation_namespace() -> str:
    """Render the compact schema-derived J operation naming rule."""

    return json.dumps(
        canonical_semantic_operation_namespace(),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def render_canonical_semantic_operation_namespace_summary() -> str:
    """Render the closed operation rule within a diagnostic field bound."""

    rule = canonical_semantic_operation_namespace()
    binary = rule["map_binary"]["examples"]
    unary = rule["map_unary"]["examples"]
    not_operations = rule["not_operations"]
    summary = (
        "J catalog;map_binary.fn:"
        + ",".join(binary)
        + ";map_unary.fn:"
        + ",".join(unary)
        + ";load/store->boundary.accesses.kind;"
        + "/".join(item for item in not_operations if item in {"program_id", "arange"})
        + " not J ops"
    )
    if len(summary) > 128:
        raise AssertionError("canonical operation diagnostic exceeds its wire bound")
    return summary


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
_LOGICAL_DTYPE = {"type": "string", "enum": sorted(CANONICAL_DTYPES)}
_TENSOR_HANDLE = {"type": "string", "pattern": "^t[0-9]+$"}

# Provider semantic submissions use local labels before the host allocates the
# canonical J namespaces.  Keep this policy next to the schema projection so
# Responses, MCP, and the ingress diagnostics cannot advertise different
# spellings.  ``t*`` remains the one canonical logical handle that is visible
# to the provider; all other canonical handles are host-owned at this stage.
_PROVIDER_LOCAL_LABEL_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.:-]*$"
_PROVIDER_FORBIDDEN_CANONICAL_PREFIXES = (
    "n",
    "v",
    "x",
    "p",
    "c",
    "a",
    "e",
)


def _provider_schema_projection(value: Any) -> Any:
    """Project canonical handle patterns onto the semantic provider wire.

    The completed artifact schema must continue to use dense canonical handle
    namespaces. Showing those patterns verbatim in a pre-construction provider
    contract is contradictory: the provider is required to use local labels,
    while the host owns canonical allocation.  This projection changes only
    the advertised pre-construction pattern; the final schema is untouched.
    """

    if isinstance(value, list):
        return [_provider_schema_projection(item) for item in value]
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    projected = {
        key: _provider_schema_projection(item) for key, item in value.items()
    }
    pattern = value.get("pattern")
    if isinstance(pattern, str):
        # The completed operation schema also contains combined value-handle
        # patterns such as (t|v). Leaving those unchanged would advertise
        # canonical v*/x* spellings on the semantic provider wire even though
        # ingress requires local labels and dim* aliases.
        provider_local_patterns = {
            *(
                rf"^{prefix}[0-9]+$"
                for prefix in _PROVIDER_FORBIDDEN_CANONICAL_PREFIXES
            ),
            r"^(?:t|v)[0-9]+$",
        }
        if pattern in provider_local_patterns:
            projected["pattern"] = _PROVIDER_LOCAL_LABEL_PATTERN
            projected["provider_local_label"] = True
    property_names = value.get("propertyNames")
    if isinstance(property_names, Mapping):
        projected["propertyNames"] = _provider_schema_projection(property_names)
    return projected


def _provider_row_schema_projection(
    value: Mapping[str, Any],
    *,
    host_fields: set[str],
) -> dict[str, Any]:
    """Remove host-owned row fields before exposing an item schema.

    The final boundary row schema requires ``access_id``/``effect_id``.  That
    requirement is correct after construction but contradictory on the
    semantic provider wire.  Merely listing those names in metadata still
    invites a model to emit them, so remove the properties and nested required
    entries from the provider projection itself.
    """

    projected = _provider_schema_projection(value)
    if not isinstance(projected, dict):
        raise TypeError("provider row schema projection must remain an object")
    properties = projected.get("properties")
    if isinstance(properties, dict):
        for field in host_fields:
            properties.pop(field, None)
        projected["properties"] = properties
    required = projected.get("required")
    if isinstance(required, list):
        projected["required"] = [
            field for field in required if field not in host_fields
        ]
    for branch_name in ("allOf", "anyOf", "oneOf"):
        branches = projected.get(branch_name)
        if isinstance(branches, list):
            projected[branch_name] = [
                _provider_row_schema_projection(branch, host_fields=host_fields)
                if isinstance(branch, Mapping)
                else branch
                for branch in branches
            ]
    return projected


def _provider_access_reference_projection(value: Any) -> Any:
    """Rewrite final ``a*`` reference slots onto the semantic ``accessN`` wire.

    ``access_ref`` is not a generic local SSA label: it identifies one
    provider-authored boundary access and is later allocated as canonical
    ``a*`` together with its derived ``e*`` effect.  Keeping this narrow
    projection separate from generic handle projection prevents the provider
    contract from accepting arbitrary labels here while the ingress requires a
    stable access occurrence key.
    """

    if isinstance(value, list):
        return [_provider_access_reference_projection(item) for item in value]
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    projected: dict[str, Any] = {}
    for key, item in value.items():
        if key == "access_ref" and isinstance(item, Mapping):
            projected[key] = {
                "type": "string",
                "pattern": r"^access[0-9]+$",
                "provider_local_access_label": True,
            }
        else:
            projected[key] = _provider_access_reference_projection(item)
    return projected
_NODE_HANDLE = {"type": "string", "pattern": "^n[0-9]+$"}
_C_HANDLE = {"type": "string", "pattern": "^c[0-9]+$"}
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
        "their equality in value_dimension_relations. Deterministic ingress may "
        "materialize this row only when the Harness shape-coordinate binding and "
        "the authored x* shape provide unique proof; ordinary Source scalar use "
        "still remains provider-authored and fail-closed. "
        "The logical tensor ABI is ordered: declare every fixed input/inout in "
        "intent_function.inputs before declaring output-only tensors in "
        "intent_function.outputs. Optional scalar/constexpr parameters proven to "
        "be logical rank-0 values are inserted after all fixed inputs and before "
        "outputs, in physical ABI order. Allocate one gap-free final t* sequence "
        "from that order; an inout output repeats its existing input t* and does "
        "not consume a new handle. The physical output order is authoritative; "
        "never use a baseline output t* that could be shifted by a later scalar "
        "decision. Do not place a later input after an "
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
        "the bare function name not has no lowering. Axis_ref fields in pi use "
        "only an integer ordinal or an x* dimension handle; k* names are "
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
    schema = _closed_object(
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
    return schema


def _value_dimension_relation_schema() -> dict[str, Any]:
    schema = _closed_object(
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
            "Equality between one integral rank-0 logical J input t* and one "
            "declared symbolic shape dimension x*. A uniquely matching Harness "
            "shape-coordinate row may be materialized by deterministic ingress; "
            "all other rows remain LLM-authored and fixed code never guesses them."
        ),
    )
    return schema


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
            "semantic_access_authorization": _nullable(_STRING),
            "semantic_owner_authorization": _nullable(_STRING),
            "semantic_guard_authorization": _nullable(_STRING),
        },
        CANONICAL_BOUNDARY_ACCESS_REQUIRED_FIELDS,
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
            "axes": _array({"type": "integer"}),
        },
        ("node_ref", "operand", "axes"),
        description=(
            "Implicit operand-broadcast annotation for one map_binary or "
            "select node only. axes contains JSON integer TARGET-axis "
            "positions, not dimension labels or the "
            "source-axis positions in a Source owner outline; scalar-to-tensor "
            "owner broadcasts still require a relation row, while [] and "
            "explicit target axes share the scalar empty input map. node_ref "
            "must name that node and operand must be one of its operands. "
            "Never emit a boundary.broadcasts entry "
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
            "owns the predicate. Existing typed fill-to-operand dataflow must "
            "prove the two-sided identity on every branch; a declaration alone "
            "grants no authority."
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
            "and come only from external Gate2 policy Q. When logical updates "
            "exist, every declared logical output needs exactly one relation. A "
            "direct output of an overwrite + one_of scatter must use "
            "one_of_updates and name that exact scatter; deterministic or combine "
            "updates, and all other outputs, must not use one_of_updates. "
            "one_of_updates requires one non-empty update_node_ref. "
            "exact/tolerance must omit the update_node_ref field entirely; empty "
            "string or null is invalid."
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
        CANONICAL_BOUNDARY_WIRE_FIELDS,
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
        },
        ("name", "kind", "role"),
    )
    return _closed_object(
        {
            "slots": {
                **_array(slot),
                "description": (
                    "Ordered declarations for this selected frontend. Preserve the exact order "
                    "of the supplied layer_c_declarations.slot_order; the list "
                    "is not sorted by the numeric suffix of c* handles. "
                    "CandidateCheck, Decode, and the frontend compiler use this same order."
                ),
            },
            "values": {
                "type": "object",
                "propertyNames": {"pattern": "^c[0-9]+$"},
                "additionalProperties": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "integer"},
                        {"type": "number"},
                        {"type": "string"},
                    ]
                },
                "description": (
                    "C values are the exact scalar bindings for this selected "
                    "frontend/Harness compilation. They are independent of J "
                    "semantics, must use the declared c* slots, and must not be "
                    "reconstructed from a target backend or moved into J/pi."
                ),
            },
        },
        CANONICAL_C_WIRE_FIELDS,
        description=(
            "C_source is the selected frontend's complete source realization "
            "interface: declaration order, kind, type, role, and "
            "scalar values. It is not J and is not a portable target-lowering "
            "plugin; CUDA/RISC-V/other C_target choices remain downstream of "
            "canonical J."
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
            "the named J logical axes under one C tile slot. "
            "logical_axes.subject_ref is only a declared J domain d* or node "
            "n* (a tiled_domain uses d*; a tiled_recurrence uses an authored "
            "loop n*), never a tensor t* or SSA value v*. The referenced "
            "tile_slot must have role tile_extent or reduction_tile_extent. "
            "The logical-axis ownership is declared only by this pi row, never "
            "duplicated on the C slot."
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
    access_binding = _closed_object(
        {
            "relation": {
                "type": "string",
                "enum": list(ACCESS_REPRESENTATION_BINDING_RELATIONS),
            },
            "access_ref": {"type": "string", "pattern": "^a[0-9]+$"},
            "carrier_ref": {"type": "string", "pattern": "^p[0-9]+$"},
            "axis_ref": {"type": "integer", "minimum": 0},
        },
        ("relation", "access_ref"),
        description=(
            "Closed pointer/address representation bound to one J access; tensor, "
            "logical index, predicate and effect remain hard. For address_recomputed "
            "and pointer_advanced, carrier_ref names the null-logical-ref scalar "
            "p* ABI carrier and axis_ref names its non-negative tensor axis."
        ),
    )
    access_binding["allOf"] = [
        {
            "if": {
                "properties": {
                    "relation": {
                        "enum": [
                            "address_recomputed",
                            "pointer_advanced",
                        ]
                    }
                },
                "required": ["relation"],
            },
            "then": {"required": ["carrier_ref", "axis_ref"]},
        }
    ]
    normalization_class = _closed_object(
        {
            "name": {
                "type": "string",
                "enum": sorted(NORMALIZATION_CLASS_SCOPES),
            },
            "scope": {
                "type": "string",
                "enum": sorted(set(NORMALIZATION_CLASS_SCOPES.values())),
            },
        },
        ("name", "scope"),
        description=(
            "Lift-authored class-level Gate1 normalization declaration. The "
            "class selects one closed representation family; it does not add "
            "J semantics, Source facts, Replay instructions, or per-occurrence "
            "exceptions. Gate1 validates the declaration and remains fail-closed."
        ),
    )
    normalization_class["allOf"] = [
        {
            "if": {
                "properties": {"name": {"const": name}},
                "required": ["name"],
            },
            "then": {"properties": {"scope": {"const": scope}}},
        }
        for name, scope in NORMALIZATION_CLASS_SCOPES.items()
    ]
    schema = _closed_object(
        {
            "schema": {"const": NORMALIZATION_WITNESS_SCHEMA},
            "normalization_classes": _array(normalization_class),
            "program_axes": _array(program_axis),
            "program_bindings": _array(program_binding),
            "iteration_bindings": _array(iteration_binding),
            "control_bindings": _array(control_binding),
            "materialization_bindings": _array(materialization_binding),
            "access_representation_bindings": _array(access_binding),
        },
        CANONICAL_PI_REQUIRED_FIELDS,
        description=(
            "Source-realization witness pi_source for Gate1 only. It never "
            "enters Replay or production Lower and is not C_target. "
            "Lift authors any applicable normalization class from Source evidence "
            "and J proofs; Gate1 only validates and applies its fixed matcher. "
            "Author only closed relation kinds directly bound to existing J/C "
            "handles. Rows contain no source/replay node, target recipe, free-text "
            "equivalence or allowed diff. Unknown relations remain hard."
        ),
    )
    # The two program-coordinate arrays are alternative wire encodings of one
    # relation. Empty/empty is valid when Source cannot prove a launch mapping;
    # otherwise exactly one family may be populated. Keep this structural XOR
    # in the shared provider/CandidateCheck schema so the model does not spend
    # a relift merely learning a post-hoc TypeCheck rule.
    schema["allOf"] = [
        {
            "not": {
                "properties": {
                    "program_axes": {"minItems": 1},
                    "program_bindings": {"minItems": 1},
                },
                "required": ["program_axes", "program_bindings"],
            }
        }
    ]
    return schema


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
            "The canonical construction pass derives scope identity, kind, parent, "
            "definitions, captures and effects from the operation graph and Source "
            "transfer ledger. The final wire contains neither missing nor extra "
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
        CANONICAL_J_WIRE_FIELDS,
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
        CANONICAL_ARTIFACT_ROOT_FIELDS,
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
    ordinary_operations, structured_operations = _operation_catalog(
        intent_op_schema
    )
    schema["$defs"] = {
        "const_literal": copy.deepcopy(
            intent_op_schema["$defs"]["const_literal"]
        ),
        "intent_op": intent_op_schema,
    }
    schema["x-intentir-first-turn-wire"] = {
        "boundary_rows": list(_FIRST_TURN_BOUNDARY_ROWS),
        "operation_names": list(_FIRST_TURN_OPERATION_NAMES),
        "operation_contract_names": list(CANONICAL_OPERATION_CONTRACT_NAMES),
    }
    schema["x-intentir-operation-catalog"] = {
        "ordinary": ordinary_operations,
        "structured": structured_operations,
    }
    # Publish the exact compact operation projection inside the mounted
    # artifact schema.  MCP/schema lookup, the provider-facing contract, and
    # CandidateCheck's schema preflight can therefore compare against one
    # schema-owned projection instead of independently rebuilding the leaf
    # contract.
    schema["x-intentir-operation-contracts"] = _compact_operation_contracts(
        intent_op_schema,
        tuple(sorted(CANONICAL_J_OPS)),
    )
    # Construction-record ownership is wire metadata, not a second provider
    # contract.  The standalone MCP context reader consumes this exact object
    # from the mounted schema, while the in-process profile below adds only
    # schema-derived row shapes.  Keeping the semantic ownership wording here
    # prevents the two authoring surfaces from drifting independently.
    schema["x-intentir-construction-records"] = {
        "operation": {
            "authority": "canonical_construction",
            "wire_contract": "intent_op_contract",
            "result_namespace": "host-allocated v* after semantic ingress",
            "operand_namespace": "canonical J t* or v* after host allocation",
            "fields": list(CANONICAL_CONSTRUCTION_RECORD_FIELDS["operation"]),
        },
        "output": {
            "order": "outputs -> returns -> ABI; one output record owns all rows",
            "fields": list(CANONICAL_CONSTRUCTION_RECORD_FIELDS["output"]),
        },
        "scope": {
            "lexical_record": (
                "every declared lexical scope: definitions/captures/effects "
                "are derived together from operation ownership and the "
                "boundary effect ledger; function scope owns its inputs"
            ),
            "structured_regions": (
                "program_instance, loop_region and branch_region retain their "
                "authored lexical ownership; node-backed effects derive from "
                "operation paths, while node-less effects require one existing "
                "scope owner; ambiguous ownership remains fail-closed"
            ),
            "fields": list(CANONICAL_CONSTRUCTION_RECORD_FIELDS["scope"]),
        },
        "dimension": {
            "authority": (
                "ordered J dimension definitions validated by the canonical "
                "SemanticArchive and copied without re-inference"
            ),
            "fields": list(CANONICAL_CONSTRUCTION_RECORD_FIELDS["dimension"]),
        },
        "value_dimension": {
            "authority": (
                "ordered rank-0 value/dimension equalities validated by the "
                "canonical SemanticArchive"
            ),
            "fields": list(
                CANONICAL_CONSTRUCTION_RECORD_FIELDS["value_dimension"]
            ),
        },
        "boundary": {
            "authority": (
                "one ordered construction projection of every boundary "
                "collection; consumers must not rebuild accesses, effects, "
                "broadcasts, bounds, or consumer extensions from raw J"
            ),
            "fields": list(CANONICAL_CONSTRUCTION_RECORD_FIELDS["boundary"]),
        },
    }
    schema["x-intentir-schema-sections"] = {
        name: {"path": list(path)}
        for name, path in CANONICAL_SCHEMA_SECTION_PATHS.items()
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


def scrc_lift_core_text_format(*, compact: bool = False) -> dict[str, Any]:
    """Return the Responses format for Lift's source-aware LLM substage."""

    return {
        "type": "json_schema",
        "name": "scrc_lift_core",
        # Operation-specific semantic attrs are closed by the shared IntentOp
        # schema embedded in the recursive artifact schema. B is the one
        # closed top-level boundary; C and pi are separate closed objects.
        # The active Responses provider advertises strict JSON Schema support.
        # This is a transport constraint, not a semantic repair: a response
        # that still cannot satisfy the recursive canonical shape remains
        # reject-only at the shared ingress and CandidateCheck boundary.
        "strict": True,
        "schema": (
            scrc_lift_core_model_json_schema()
            if compact
            else scrc_lift_core_json_schema()
        ),
    }


def scrc_semantic_candidate_text_format(
    construction: Any,
) -> dict[str, Any]:
    """Return the JSON-object output mode for one Lift submission.

    The completed artifact uses conditional, tuple, uniqueness, and open-attrs
    constraints that are not a portable provider output schema. The canonical
    semantic contract remains mounted in the prompt and the complete candidate
    remains locally checked; this transport setting does not relax
    CandidateCheck. ``construction`` is validated here so callers cannot enter
    the JSON path without the same Source-owned wire authority.
    """

    bundle = canonical_source_wire_bundle(construction)
    profile = bundle.get("profile")
    requirements = bundle.get("semantic_construction_requirements")
    compact_wire = (
        profile.get("semantic_provider_wire_compact")
        if isinstance(profile, Mapping)
        else None
    )
    if not isinstance(requirements, Mapping) or not isinstance(compact_wire, Mapping):
        raise TypeError("semantic candidate format lacks canonical source contract")
    return {"type": "json_object"}


def _schema_object_shape(value: Any, path: str) -> dict[str, Any]:
    """Return the small, machine-derived shape used by all Lift surfaces."""

    if not isinstance(value, dict) or value.get("type") != "object":
        raise TypeError(f"canonical wire section {path} must be an object")
    properties = value.get("properties")
    required = value.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise TypeError(f"canonical wire section {path} lacks object metadata")
    return {
        "required": list(required),
        "fields": list(properties),
    }


def _schema_array_item(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "array":
        raise TypeError(f"canonical wire section {path} must be an array")
    item = value.get("items")
    if not isinstance(item, dict):
        raise TypeError(f"canonical wire section {path} lacks array items")
    return item


def _canonical_empty_for_schema(value: Any, path: str) -> Any:
    """Build an empty collection from the authoritative schema shape."""

    if not isinstance(value, dict):
        raise TypeError(f"canonical wire section {path} is malformed")
    if value.get("type") == "array":
        return []
    if value.get("type") == "object":
        return {}
    raise TypeError(
        f"canonical wire section {path} is not an empty collection: "
        f"{value.get('type')!r}"
    )


# First-turn routing metadata, not a second schema.  The selected leaves are
# still projected from the authoritative properties below; less common rows
# remain available through one bounded schema lookup.
_FIRST_TURN_BOUNDARY_ROWS = (
    "accesses",
    # Responses has no schema_lookup action. Owner authoring requires both
    # rows on its first submission, including the integer `axes` leaf; a
    # prompt-only description left that leaf outside the mounted wire.
    "broadcasts",
    "access_guard_relations",
    "effects",
    "index_bounds",
    "output_mappings",
    # consumer_extensions.tile_axis is checked against the matching
    # reduce/scan boundary contract. Direct Responses therefore needs the
    # canonical reduction row schema in the same first-turn projection; an
    # isolated consumer-extension schema cannot express its join authority.
    "reductions",
    "updates",
    "result_relations",
    # Direct Responses has no schema_lookup action.  A guarded load feeding a
    # reduce/scan therefore needs this exact row schema in the first request;
    # the short semantic_required route alone cannot convey operand_index,
    # tile_axis, the typed outside value, and the closed identity law.
    "consumer_extensions",
)
_FIRST_TURN_OPERATION_NAMES = (
    "const",
    "map_unary",
    "map_binary",
    "cast",
    "iota",
    "reduce",
    "gather",
    "scatter",
    "tile",
)

# One mounted index for every source-neutral schema lookup section.  These are
# lookup aliases only; every path points into this same artifact schema.  The
# standalone schema lookup script and the MCP environment consume this index
# instead of carrying separate copies of the J/C/pi topology.
CANONICAL_SCHEMA_SECTION_PATHS = {
    "tensor": ("properties", "j", "properties", "intent", "properties", "intent_function", "properties", "inputs", "items"),
    "parameter": ("properties", "j", "properties", "signature", "properties", "parameters", "items"),
    "scope": ("properties", "j", "properties", "scopes", "items"),
    "storage": ("properties", "j", "properties", "storage", "items"),
    "domain": ("properties", "j", "properties", "boundary", "properties", "domains"),
    "access": ("properties", "j", "properties", "boundary", "properties", "accesses", "items"),
    "broadcast": ("properties", "j", "properties", "boundary", "properties", "broadcasts", "items"),
    "alias": ("properties", "j", "properties", "boundary", "properties", "aliases", "items"),
    "effect": ("properties", "j", "properties", "boundary", "properties", "effects", "items"),
    "output_mapping": ("properties", "j", "properties", "boundary", "properties", "output_mappings", "items"),
    "reduction": ("properties", "j", "properties", "boundary", "properties", "reductions", "items"),
    "index_bound": ("properties", "j", "properties", "boundary", "properties", "index_bounds", "items"),
    "update": ("properties", "j", "properties", "boundary", "properties", "updates", "items"),
    "result_relation": ("properties", "j", "properties", "boundary", "properties", "result_relations", "items"),
    "consumer_extension": ("properties", "j", "properties", "boundary", "properties", "consumer_extensions", "items"),
    "access_guard_relation": ("properties", "j", "properties", "boundary", "properties", "access_guard_relations", "items"),
    "value_dimension_relation": ("properties", "j", "properties", "value_dimension_relations", "items"),
    "dimension_relation": ("properties", "j", "properties", "dimension_relations", "items"),
    "authorization": ("properties", "j", "properties", "boundary", "properties", "realization_authorizations", "items"),
    "c_slot": ("properties", "c", "properties", "slots", "items"),
    "pi_program_axis": ("properties", "pi", "properties", "program_axes", "items"),
    "pi_program_binding": ("properties", "pi", "properties", "program_bindings", "items"),
    "pi_iteration_binding": ("properties", "pi", "properties", "iteration_bindings", "items"),
    "pi_control_binding": ("properties", "pi", "properties", "control_bindings", "items"),
    "pi_materialization_binding": ("properties", "pi", "properties", "materialization_bindings", "items"),
    "pi_access_representation_binding": ("properties", "pi", "properties", "access_representation_bindings", "items"),
}
# The profile is the operation-contract authority used by schema lookup and
# CandidateCheck.  First-turn prompt exposure remains bounded by
# ``_FIRST_TURN_OPERATION_NAMES``; the complete profile must nevertheless be
# projected from the same closed IntentOp catalog so an omitted leaf cannot
# become a second, silently different contract.
CANONICAL_OPERATION_CONTRACT_NAMES = tuple(sorted(CANONICAL_J_OPS))


def _operation_catalog(
    operation_schema: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Split ordinary/structured names from the IntentOp schema variants."""

    ordinary: list[str] = []
    structured: list[str] = []
    for variant in operation_schema.get("anyOf", []):
        names = (
            variant.get("properties", {})
            .get("op", {})
            .get("enum", [])
        )
        target = structured if "allOf" not in variant else ordinary
        for name in names:
            if name not in target:
                target.append(name)
    return ordinary, structured


def _compact_schema_node(
    value: Any,
    *,
    depth: int = 0,
    definitions: Mapping[str, Any] | None = None,
    _resolving_refs: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Keep high-signal machine constraints for a bounded MCP hint.

    This is a projection of the authoritative schema, not a second operation
    schema.  Recursive branches are capped so exact enum/union/closed-object
    constraints remain visible without copying the full recursive IntentOp
    definition into every first-turn packet.  Local ``$ref`` leaves are kept
    visible and, when their definition is available, include one compact
    resolved view.  Dropping a reference used to turn
    ``const.attrs.value`` into ``{}`` on the provider wire even though the
    ingress validator still enforced the recursive ``const_literal`` contract.
    That made the canonical graph contract internally inconsistent: an Agent
    could not learn the accepted value domain from MCP, then received a
    deterministic rejection at the write boundary.  A recursive reference is
    intentionally left as a reference after one expansion rather than being
    guessed or truncated into a permissive value.
    """

    if not isinstance(value, dict):
        return {}
    reference = value.get("$ref")
    if isinstance(reference, str):
        result: dict[str, Any] = {"$ref": reference}
        if reference.startswith("#/$defs/") and isinstance(definitions, Mapping):
            name = reference[len("#/$defs/") :]
            target = definitions.get(name)
            if isinstance(target, Mapping):
                if name in _resolving_refs:
                    result["recursive"] = True
                else:
                    result["resolved"] = _compact_schema_node(
                        target,
                        depth=depth,
                        definitions=definitions,
                        _resolving_refs=(*_resolving_refs, name),
                    )
        return result
    result: dict[str, Any] = {}
    for key in (
        "type",
        "pattern",
        "const",
        "enum",
        "required",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
    ):
        if key in value:
            result[key] = copy.deepcopy(value[key])
    if value.get("additionalProperties") is False:
        result["closed"] = True
    if depth < 3:
        properties = value.get("properties")
        if isinstance(properties, dict):
            result["properties"] = {
                key: _compact_schema_node(
                    item,
                    depth=depth + 1,
                    definitions=definitions,
                    _resolving_refs=_resolving_refs,
                )
                for key, item in properties.items()
            }
        items = value.get("items")
        if isinstance(items, dict):
            result["items"] = _compact_schema_node(
                items,
                depth=depth + 1,
                definitions=definitions,
                _resolving_refs=_resolving_refs,
            )
        elif items is False:
            result["items"] = False
        for key in ("oneOf", "anyOf", "allOf"):
            branches = value.get(key)
            if isinstance(branches, list):
                result[key] = [
                    _compact_schema_node(
                        item,
                        depth=depth + 1,
                        definitions=definitions,
                        _resolving_refs=_resolving_refs,
                    )
                    for item in branches
                ]
        negated = value.get("not")
        if isinstance(negated, dict):
            result["not"] = _compact_schema_node(
                negated,
                depth=depth + 1,
                definitions=definitions,
                _resolving_refs=_resolving_refs,
            )
    return result


def _compact_operation_contracts(
    operation_schema: dict[str, Any],
    names: tuple[str, ...] = CANONICAL_OPERATION_CONTRACT_NAMES,
) -> dict[str, Any]:
    """Project compact IntentOp leaves without inventing a second schema.

    The complete operation catalog remains in the mounted schema and is
    served by ``schema_lookup``.  The compact provider-facing profile uses
    the schema-declared bounded contract set so this profile and the
    standalone MCP context reader cannot drift into different hard-coded
    operation lists.
    """

    contracts: dict[str, Any] = {}
    definitions = operation_schema.get("$defs")
    if not isinstance(definitions, Mapping):
        definitions = {}
    else:
        definitions = dict(definitions)
    # ``intent_op_json_schema`` owns ``const_literal`` locally, while its
    # recursive structured branches refer to the enclosing document's
    # ``#/$defs/intent_op``.  The mounted artifact adds that outer definition
    # after obtaining this operation schema.  Make the compact projection
    # resolve the same recursive name without inventing a second operation
    # catalog; recursive occurrences are stopped by ``_resolving_refs``.
    definitions.setdefault("intent_op", operation_schema)
    common_fields: list[str] = []
    common_required: list[str] = []
    common_properties: dict[str, Any] = {}
    # Ordinary IntentOp variants carry the shared operation envelope.  Keep
    # this lookup schema-derived as well; otherwise a leaf-only projection can
    # accidentally teach the provider an attrs-only object.
    for variant in operation_schema.get("anyOf", []):
        if "allOf" not in variant:
            continue
        properties = variant.get("properties", {})
        required = variant.get("required", [])
        if isinstance(properties, dict):
            common_fields = list(properties)
            common_properties = dict(properties)
        if isinstance(required, list):
            common_required = list(required)
        break
    for name in names:
        branch: dict[str, Any] | None = None
        for variant in operation_schema.get("anyOf", []):
            op_enum = (
                variant.get("properties", {})
                .get("op", {})
                .get("enum", [])
            )
            if name not in op_enum:
                continue
            if name in {"loop", "effect_if", "value_if"}:
                branch = variant
            else:
                for condition in variant.get("allOf", []):
                    if (
                        condition.get("if", {})
                        .get("properties", {})
                        .get("op", {})
                        .get("const")
                        == name
                    ):
                        branch = condition.get("then")
                        break
            break
        if not isinstance(branch, dict):
            continue
        properties = branch.get("properties", {})
        # Structured operations are separate anyOf branches.  Their property
        # sets are already complete and deliberately exclude the ordinary
        # operation envelope fields (operands/attrs/type).  Merging the
        # ordinary fields here teaches the provider a wire that
        # validate_intent_op_structure and CandidateCheck reject.  Only the
        # ordinary conditional branches inherit common_fields/common_required.
        structured_branch = name in STRUCTURED_INTENT_OPS
        if structured_branch:
            fields = list(properties) if isinstance(properties, dict) else []
            required = list(branch.get("required", []))
            wire_properties = (
                dict(properties) if isinstance(properties, dict) else {}
            )
        else:
            fields = list(common_fields)
            if isinstance(properties, dict):
                fields.extend(
                    field for field in properties
                    if field not in fields
                )
            required = list(common_required)
            required.extend(
                field for field in branch.get("required", [])
                if field not in required
            )
            # The completed canonical construction bundle has one typed
            # operation record for every ordinary provider-authored result.
            # ``type`` is optional in the broad artifact schema because old
            # canonical/structured forms may carry a host-elaborated type
            # view, but it is not optional on the compact semantic authoring
            # surface: omitting it makes the Source-bound candidate impossible
            # to elaborate and defers an otherwise local error to
            # CandidateCheck.  Keep this rule in the shared compact projection
            # so direct Responses and MCP cannot teach different wires.
            if "type" in fields and "type" not in required:
                required.append("type")
            wire_properties = dict(common_properties)
            if isinstance(properties, dict):
                wire_properties.update(properties)
        # ``CanonicalConstructionRecords.operation_records`` is a host-side
        # elaboration table.  Its path/parent/depth metadata is deliberately
        # not copied here.  Providers author the exact IntentOp row instead,
        # with its leaf types projected from this same schema.
        wire_properties["op"] = {"const": name}
        item: dict[str, Any] = {
            "fields": fields,
            "required": required,
            "wire_schema": {
                "type": "object",
                "required": required,
                "closed": True,
                "properties": {
                    field: _compact_schema_node(
                        value,
                        definitions=definitions,
                    )
                    for field, value in wire_properties.items()
                    if field in fields
                },
            },
        }
        if not structured_branch and "type" in fields:
            # ``IntentOp.type`` is intentionally broad in the final artifact
            # schema because record/ragged variants share the common envelope.
            # A compact Lift authoring entry nevertheless needs the concrete
            # tensor form used by ordinary map/cast/iota operations; otherwise
            # a provider can satisfy the shallow ``object`` shape with a bare
            # dtype string and only discover the canonical-record requirement
            # at finalization.  This metadata is a projection of the existing
            # value-type contract, not an alternate operation wire.
            item["type_requirement"] = {
                "ordinary_tensor": {
                    "required": ["dtype", "shape"],
                    "dtype": copy.deepcopy(_LOGICAL_DTYPE),
                    "shape": _array(copy.deepcopy(_INT_OR_STRING)),
                },
                "alternative": "record_or_ragged value-type object when required by that operation",
            }
        attrs = properties.get("attrs")
        if isinstance(attrs, dict):
            item["attrs_required"] = list(attrs.get("required", []))
            item["attrs"] = _compact_schema_node(
                attrs,
                definitions=definitions,
            )
        for field in ("predicate", "result", "type", "trip", "yield"):
            if field in properties:
                item[field] = _compact_schema_node(
                    properties[field],
                    definitions=definitions,
                )
        contracts[name] = item
    return contracts


def canonical_artifact_skeleton(
    *,
    c_slots: Any = (),
    c_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the one empty artifact envelope accepted by the wire schema.

    This constructor is deliberately structural.  It allocates no J meaning,
    Source operation, SSA value, target E_C extension attachment, or pi relation.  The required
    fields and every empty collection are read from :func:`scrc_lift_core_json_schema`
    so an MCP seed, an example, and the provider contract cannot grow a second
    hand-written envelope grammar.
    """

    schema = scrc_lift_core_json_schema()
    root_properties = schema["properties"]
    j_schema = root_properties["j"]
    j_properties = j_schema["properties"]
    boundary_schema = j_properties["boundary"]
    boundary_properties = boundary_schema["properties"]
    function_schema = j_properties["intent"]["properties"]["intent_function"]
    function_properties = function_schema["properties"]
    scope_item = _schema_array_item(j_properties["scopes"], "j.scopes")
    scope_properties = scope_item["properties"]
    pi_schema = root_properties["pi"]
    pi_properties = pi_schema["properties"]

    function: dict[str, Any] = {}
    for name in function_schema["required"]:
        if name == "name":
            function[name] = function_properties[name].get(
                "enum", ["scrc_j0"]
            )[0]
        elif name in {"inputs", "outputs", "returns", "body"}:
            function[name] = []
        else:
            function[name] = _canonical_empty_for_schema(
                function_properties[name], f"j.intent.intent_function.{name}"
            )

    boundary: dict[str, Any] = {}
    for name in boundary_schema["required"]:
        boundary[name] = _canonical_empty_for_schema(
            boundary_properties[name], f"j.boundary.{name}"
        )

    scope: dict[str, Any] = {}
    for name in scope_item["required"]:
        if name == "scope_id":
            scope[name] = "q0"
        elif name == "kind":
            scope[name] = scope_properties[name].get("enum", ["function"])[0]
        elif name == "parent_scope":
            scope[name] = None
        elif scope_properties[name].get("type") == "array":
            scope[name] = []
        else:
            raise TypeError(f"no canonical empty value for j.scopes[].{name}")

    j: dict[str, Any] = {}
    for name in j_schema["required"]:
        if name == "kernel_id":
            j[name] = j_properties[name].get("enum", ["scrc_k0"])[0]
        elif name == "intent":
            j[name] = {"intent_function": function}
        elif name == "node_ids":
            j[name] = {}
        elif name == "boundary":
            j[name] = boundary
        elif name == "signature":
            signature_schema = j_properties[name]
            j[name] = {
                field: _canonical_empty_for_schema(
                    signature_schema["properties"][field],
                    f"j.signature.{field}",
                )
                for field in signature_schema["required"]
            }
        elif name == "scopes":
            j[name] = [scope]
        else:
            j[name] = _canonical_empty_for_schema(
                j_properties[name], f"j.{name}"
            )

    c_schema = root_properties["c"]
    c = {}
    for name in c_schema["required"]:
        if name == "slots":
            c[name] = copy.deepcopy(list(c_slots))
        elif name == "values":
            c[name] = copy.deepcopy(c_values if c_values is not None else {})
        else:
            c[name] = _canonical_empty_for_schema(
                c_schema["properties"][name], f"c.{name}"
            )
    pi: dict[str, Any] = {}
    for name, property_schema in pi_properties.items():
        if name == "schema":
            pi[name] = property_schema["const"]
        else:
            pi[name] = _canonical_empty_for_schema(
                property_schema, f"pi.{name}"
            )

    artifact = {
        "schema": root_properties["schema"].get("enum", [None])[0],
        "j": j,
        "c": c,
        "pi": pi,
    }
    if set(artifact) != set(schema["required"]):
        raise AssertionError("canonical artifact skeleton root drifted from schema")
    return artifact


def canonical_source_wire_seed(construction: Any) -> dict[str, Any]:
    """Materialize the one Source/Harness-bound authoring seed.

    The empty :func:`canonical_artifact_skeleton` is the schema seed used by
    mounted workspaces.  Responses Lift has no local MCP resource reader, so
    it needs the same mechanical seed in its user payload.  This constructor
    is intentionally limited to facts already closed by ``SourceFacts`` and
    ``CanonicalHandleAllocator``: it never creates an operation, boundary
    access, scope, dimension relation, output mapping, or target E_C extension
    attachment.  The provider may contribute only normalization classes; the
    host later derives every other pi row from the Source transfer ledger,
    canonical J, and C ownership.
    """

    source_facts = getattr(construction, "source_facts", None)
    allocator = getattr(construction, "allocator", None)
    if source_facts is None or allocator is None:
        raise TypeError("canonical source wire seed requires CanonicalConstruction")
    declarations = {
        prefix: tuple(allocator.declared(prefix))
        for prefix in CANONICAL_HANDLE_NAMESPACE_ORDER
    }
    p_handles = declarations["p"]
    c_handles = declarations["c"]
    logical_bindings = allocator.logical_binding_map()
    facts_arguments = {
        item.get("name"): dict(item)
        for item in source_facts.arguments
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    facts = source_facts.compiler_facts
    if not isinstance(facts, Mapping):
        raise ValueError("SourceFacts compiler facts are unavailable")
    bound_types = facts.get("bound_constexpr_value_types", {})
    bound_coordinates = facts.get("bound_constexpr_shape_coordinates", {})
    schedule_values = facts.get("schedule_values", {})
    schedule_types = facts.get("schedule_value_types", {})
    compiler_slots = facts.get("compiler_slots", ())
    compiler_roles = facts.get("compiler_roles", {})
    if not all(isinstance(item, Mapping) for item in (bound_types, bound_coordinates,
                                                       schedule_values, schedule_types,
                                                       compiler_roles)):
        raise ValueError("SourceFacts ABI/C ledgers are malformed")
    if not isinstance(compiler_slots, (list, tuple)):
        raise ValueError("SourceFacts compiler slots are malformed")

    x_handles = tuple(allocator.declared("x"))
    if len(x_handles) != len(source_facts.shape_coordinates):
        raise ValueError("canonical x* allocator differs from Source shape coordinates")
    dimension_by_coordinate = {
        coordinate: x_handles[index]
        for index, (coordinate, _value) in enumerate(source_facts.shape_coordinates)
    }

    def symbolic_shape(row: Mapping[str, Any]) -> list[int | str]:
        shape_contract = row.get("shape_contract")
        if shape_contract is None:
            # A tensor with no symbolic Source coordinate still has a closed
            # shape when SourceFacts carries its concrete valuation.  The
            # workload adapter intentionally permits this representation for
            # static ABI carriers (for example [1] or [3]); requiring a
            # separate shape_contract here would turn a closed literal into a
            # false missing-contract error.  Keep the literal extents instead
            # of inventing x* dimensions: x* is reserved for Source shape
            # coordinates already owned by the canonical allocator.
            raw_shape = row.get("shape")
            if (
                not isinstance(raw_shape, list)
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item < 0
                    for item in raw_shape
                )
            ):
                raise ValueError("Source tensor declaration lacks closed shape")
            shape_contract = raw_shape
        elif not isinstance(shape_contract, list):
            raise ValueError("Source tensor declaration lacks shape_contract")
        result: list[int | str] = []
        for coordinate in shape_contract:
            if (
                isinstance(coordinate, int)
                and not isinstance(coordinate, bool)
                and coordinate >= 0
            ):
                # A static extent is already a closed Source/Harness fact.  It
                # is not a missing x* relation and must remain literal rather
                # than being invented as a symbolic dimension for MCP/J.
                result.append(coordinate)
                continue
            if (
                not isinstance(coordinate, str)
                or coordinate not in dimension_by_coordinate
            ):
                raise ValueError("Source tensor shape contract is not closed")
            result.append(dimension_by_coordinate[coordinate])
        return result

    tensor_inputs: list[dict[str, Any]] = []
    tensor_outputs: list[dict[str, Any]] = []
    shape_scalar_inputs: list[dict[str, Any]] = []
    logical_scalar_inputs: list[dict[str, Any]] = []
    for parameter in source_facts.arguments:
        if not isinstance(parameter, Mapping):
            continue
        name = parameter.get("name")
        binding = logical_bindings.get(name)
        kind = parameter.get("kind")
        mode = parameter.get("mode")
        if not isinstance(name, str) or binding is None:
            continue
        logical, role = binding
        dtype = parameter.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError(f"Source logical binding {name!r} lacks dtype")
        if role == "shape_scalar":
            shape_scalar_inputs.append(
                {"name": logical, "dtype": dtype, "shape": []}
            )
            continue
        if role == "scalar":
            logical_scalar_inputs.append(
                {"name": logical, "dtype": dtype, "shape": []}
            )
            continue
        if kind not in {"tensor", "pointer"}:
            continue
        row = {"name": logical, "dtype": dtype, "shape": symbolic_shape(parameter)}
        if mode in {"input", "inout"}:
            tensor_inputs.append(dict(row))
        if mode in {"output", "inout"}:
            tensor_outputs.append(dict(row))
    # Bound non-C constexprs are absent from SourceFacts.arguments but remain
    # physical p* signature rows.  If the Source binder promoted one to a
    # logical scalar, materialize the same rank-zero t* input here rather than
    # letting the MCP projection invent a second scalar path.
    for name in p_handles:
        if name in facts_arguments:
            continue
        binding = logical_bindings.get(name)
        if binding is None or binding[1] not in {"scalar", "shape_scalar"}:
            continue
        dtype = bound_types.get(name)
        if not isinstance(dtype, str):
            raise ValueError(f"Source logical binding {name!r} lacks bound dtype")
        row = {"name": binding[0], "dtype": dtype, "shape": []}
        if binding[1] == "shape_scalar":
            shape_scalar_inputs.append(row)
        else:
            logical_scalar_inputs.append(row)

    input_rows = [*tensor_inputs, *shape_scalar_inputs, *logical_scalar_inputs]
    input_order = {
        name: index for index, name in enumerate(declarations["t"])
    }
    tensor_inputs = sorted(
        input_rows,
        key=lambda row: input_order.get(row.get("name"), len(input_order)),
    )

    signature: list[dict[str, Any]] = []
    for name in p_handles:
        row = facts_arguments.get(name)
        binding = logical_bindings.get(name)
        if row is not None:
            kind = row.get("kind")
            mode = row.get("mode", "value")
            dtype = row.get("dtype")
            shape = row.get("shape")
            rank = len(shape) if isinstance(shape, list) else 0
        else:
            kind = "constexpr" if name in source_facts.constexpr_parameters else "scalar"
            mode = "value"
            dtype = bound_types.get(name)
            rank = 0
        logical_ref = binding[0] if binding is not None else None
        if logical_ref is None:
            coordinate = bound_coordinates.get(name)
            if coordinate is not None:
                if not isinstance(coordinate, str):
                    raise ValueError(
                        "bound constexpr shape coordinate is malformed"
                    )
                logical_ref = dimension_by_coordinate.get(coordinate)
                if logical_ref is None:
                    raise ValueError(
                        "bound constexpr shape coordinate lacks a canonical dimension"
                    )
        item: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "mode": mode,
            "rank": rank,
            "alias_of": None,
            # A bound constexpr tied to a source shape coordinate owns this
            # exact signature-level x* relation.  It is neither a provider
            # operation nor a rank-zero J tensor input.
            "logical_ref": logical_ref,
        }
        if isinstance(dtype, str):
            item["dtype"] = dtype
        signature.append(item)

    c_slots: list[dict[str, Any]] = []
    for name in c_handles:
        kind = "compiler" if name in compiler_slots else "constexpr"
        role = compiler_roles.get(name, "constexpr_parameter")
        value_type = schedule_types.get(name)
        if not isinstance(role, str) or not isinstance(value_type, str):
            raise ValueError(f"canonical C slot {name!r} lacks closed role/type")
        c_slots.append(
            {
                "name": name,
                "kind": kind,
                "role": role,
                "value_type": value_type,
                "required": True,
            }
        )

    seed = canonical_artifact_skeleton(
        c_slots=c_slots,
        c_values=dict(schedule_values),
    )
    function = seed["j"]["intent"]["intent_function"]
    function["inputs"] = tensor_inputs
    function["outputs"] = tensor_outputs
    seed["j"]["signature"]["parameters"] = signature
    value_relations: list[dict[str, Any]] = []
    for parameter, (logical, role) in logical_bindings.items():
        if role not in {"shape_scalar", "scalar"}:
            continue
        argument = facts_arguments.get(parameter, {})
        coordinate = argument.get("shape_coordinate")
        if not isinstance(coordinate, str):
            coordinate = bound_coordinates.get(parameter)
        if coordinate is None:
            continue
        dimension = dimension_by_coordinate.get(coordinate)
        if dimension is None:
            raise ValueError("shape scalar binding lacks a unique dimension")
        value_relations.append(
            {"value_ref": logical, "dimension_ref": dimension, "relation": "eq"}
        )
    seed["j"]["value_dimension_relations"] = value_relations
    return seed


def _source_output_construction(
    seed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project Source-owned output ownership without inventing a producer.

    A provider selects the semantic SSA producer, but it must not separately
    invent the output ABI order, return row, and boundary mapping ownership.
    This auxiliary context is not part of the candidate artifact root.
    """

    j = seed.get("j") if isinstance(seed, Mapping) else None
    function = (
        j.get("intent", {}).get("intent_function")
        if isinstance(j, Mapping) and isinstance(j.get("intent"), Mapping)
        else None
    )
    signature = j.get("signature") if isinstance(j, Mapping) else None
    outputs = function.get("outputs") if isinstance(function, Mapping) else None
    parameters = signature.get("parameters") if isinstance(signature, Mapping) else None
    if not isinstance(outputs, list) or not isinstance(parameters, list):
        raise ValueError("canonical output construction lacks Source output rows")
    output_names: list[str] = []
    output_types: dict[str, tuple[str, list[Any]]] = {}
    for output in outputs:
        if not isinstance(output, Mapping) or not isinstance(output.get("name"), str):
            raise ValueError("canonical output construction has malformed output")
        name = output["name"]
        dtype = output.get("dtype")
        shape = output.get("shape")
        if (
            name in output_types
            or not isinstance(dtype, str)
            or not isinstance(shape, list)
        ):
            raise ValueError("canonical output construction has no closed logical type")
        output_names.append(name)
        output_types[name] = (dtype, shape)

    # This is the same Source ABI output/inout subsequence consumed later by
    # canonical_output_records().  Do not build a lossy ``logical_ref -> p*``
    # dictionary here: a duplicate owner or reordered owner is a preparation
    # failure, not a provider choice that may be silently overwritten.
    abi_rows: list[tuple[int, Mapping[str, Any]]] = []
    for parameter_index, parameter in enumerate(parameters):
        if not isinstance(parameter, Mapping):
            raise ValueError("canonical output construction has malformed ABI row")
        if parameter.get("mode") in {"output", "inout"}:
            abi_rows.append((parameter_index, parameter))
    if len(abi_rows) != len(output_names):
        raise ValueError("canonical output construction ABI arity disagrees with outputs")

    abi_by_output: dict[str, tuple[int, str]] = {}
    for output, (parameter_index, parameter) in zip(
        output_names, abi_rows, strict=True
    ):
        logical_ref = parameter.get("logical_ref")
        carrier = parameter.get("name")
        dtype, shape = output_types[output]
        if (
            logical_ref != output
            or not isinstance(carrier, str)
            or parameter.get("kind") not in {"tensor", "pointer"}
            or parameter.get("dtype") != dtype
            or parameter.get("rank") != len(shape)
            or output in abi_by_output
        ):
            raise ValueError("canonical output construction ABI owner is ambiguous")
        abi_by_output[output] = (parameter_index, carrier)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(output_names):
        parameter_index, _carrier = abi_by_output[name]
        dtype, shape = output_types[name]
        rows.append(
            {
                "output": name,
                "abi_index": index,
                "signature_index": parameter_index,
                "output_type": {"dtype": dtype, "shape": copy.deepcopy(shape)},
                # This is a provider-facing semantic selection, not the final
                # canonical return wire.  The provider must not spell a v*
                # here: host allocation has not happened yet.  The one
                # selected local result (or direct Source t* input) is later
                # rewritten, then materialized into output_mapping, return,
                # ABI order, and producer attachment by one constructor.
                "selection": {
                    "fields": ["value"],
                    "value": "one local result label or direct Source t* input",
                },
                "mapping_rule": (
                    "submit exactly one {value} selection in Source output "
                    "order; the host supplies output name, canonical v* value, "
                    "return row, and any Source-proven store attachment"
                ),
            }
        )
    return rows


def _provider_output_construction(
    output_construction: Sequence[Mapping[str, Any]],
    provider_dimension_aliases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Translate Source-owned output shapes onto the provider dim* surface.

    The canonical seed necessarily uses allocated ``x*`` dimension handles,
    while a semantic Lift submission must use the pre-allocation ``dim*``
    aliases.  Output construction is displayed in both the direct Responses
    entry and MCP transfer-ledger resource, so leaving its canonical type
    unchanged creates a contradictory authoring instruction: the provider is
    told to use ``dim*`` but shown ``x*`` as the output type.  Convert this
    explanatory provider projection once and fail closed if it references a
    canonical dimension outside the declared alias set.
    """

    try:
        aliases = provider_dimension_alias_map(list(provider_dimension_aliases))
    except (TypeError, ValueError) as exc:
        raise ValueError("provider output construction lacks dense dimension aliases") from exc

    projection: list[dict[str, Any]] = []
    for raw_row in output_construction:
        if not isinstance(raw_row, Mapping):
            raise TypeError("output construction has a malformed row")
        row = copy.deepcopy(dict(raw_row))
        output_type = row.get("output_type")
        shape = output_type.get("shape") if isinstance(output_type, Mapping) else None
        if not isinstance(shape, list):
            raise ValueError("output construction has no closed logical type")
        rewritten_shape = provider_shape_from_canonical(shape, aliases)
        row["output_type"] = {
            **dict(output_type),
            "shape": rewritten_shape,
        }
        projection.append(row)
    return projection


def _semantic_access_wire_slot(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the sole provider-facing wire row for one Source access.

    ``access_id`` is a host allocation and must never share a provider wire
    with its Source-order identity.  The only provider identity is therefore
    ``access_label`` together with its exact ordinal.  Every consumer of a
    semantic transfer occurrence (brief, ledger, MCP admission and semantic
    CandidateCheck) projects this object instead of inventing a second list of
    fields or a competing ``id`` spelling.
    """

    ordinal = row.get("ordinal")
    label = row.get("access_label")
    kind = row.get("kind")
    tensor = row.get("tensor")
    index = row.get("index")
    dimensions = index.get("dimensions") if isinstance(index, Mapping) else None
    if (
        not is_canonical_ordinal(ordinal)
        or label != f"access{ordinal}"
        or kind not in {"load", "store", "update"}
        or not isinstance(dimensions, list)
        or any(not isinstance(item, str) or not item for item in dimensions)
    ):
        raise ValueError("canonical semantic transfer access row is malformed")
    effect_kind = {"load": "read", "store": "write", "update": "update"}[kind]
    required = [
        "access_label",
        "ordinal",
        "kind",
        "tensor",
        "index_expression",
        "effect_kind",
        "effect_target",
        "effect_node_ref",
    ]
    if row.get("masked") is True:
        required.append("predicate")
        if kind == "load":
            required.append("fill_value")
    # A masked Source occurrence owns the fact that a predicate is required.
    # Its lexical predicate can nevertheless be unavailable on the semantic
    # surface (for example, Source ``mask`` is a local AST value whose J label
    # is deliberately provider-chosen).  Do not misstate that unresolved
    # provider-local label as Source ``null``: doing so forces a J access that
    # cannot possibly close against the observed masked occurrence.  Publish
    # one explicit provider-owned placeholder instead.  The host retains the
    # Source fact (masked=true); the provider merely supplies the prior local
    # boolean label it selected while authoring the operation graph.
    provider_owned_fields: list[str] = []
    # The transfer ledger proves the accessed logical tensor and its rank, but
    # it does not turn a tensor extent into a coordinate expression.  An index
    # is therefore provider-authored J semantics.  It remains constrained by
    # the exact Source occurrence, typed J graph, Source-order reconciliation,
    # and functional-boundary validation; the host must never manufacture it
    # from shape dimensions.
    provider_owned_expression_fields = ["index_expression"]
    provider_owned_relations: dict[str, dict[str, str]] = {}
    authorizations = row.get("semantic_authorizations")
    owner = authorizations.get("owner") if isinstance(authorizations, Mapping) else None
    # ``update`` is a physical write/update occurrence just like ``store``.
    # The Source authoring preflight and canonical access/effect construction
    # both require an exact owner for it; omitting the node/effect relation
    # here would leave no provider-wire slot in which that already-proven
    # obligation could be declared.  Keep the same narrow condition for both
    # write kinds rather than letting a later checker reject a row that the
    # projection made impossible to author.
    exact_owner = (
        kind in {"store", "update"}
        and isinstance(owner, Mapping)
        and isinstance(owner.get("operation_family"), str)
    )
    guard_authorization = (
        authorizations.get("computed_guard")
        if isinstance(authorizations, Mapping) else None
    )
    access_view_authorization = (
        authorizations.get("access_view") if isinstance(authorizations, Mapping) else None
    )
    guard_token = (
        guard_authorization.get("token")
        if isinstance(guard_authorization, Mapping) else None
    )
    access_view_token = (
        access_view_authorization.get("token")
        if isinstance(access_view_authorization, Mapping) else None
    )
    if (kind == "load" or exact_owner) and isinstance(access_view_token, str):
        # Multiple exact physical Source loads may read one logical input with
        # different views. This receipt identifies the source occurrence only;
        # it never authorizes a consumer node or a provider-authored view.
        required.append("semantic_access_authorization")
    if isinstance(guard_token, str):
        # This is a ledger-issued receipt, not a provider-owned value.  When
        # Source/compiler evidence authorizes a computed guard for this exact
        # access, the candidate must carry that exact opaque receipt before a
        # later boundary relation can consume the local i1 value.
        required.append("semantic_guard_authorization")
    predicate = row.get("predicate")
    if row.get("masked") is True and predicate is None:
        predicate = "<provider_authored_predicate>"
        provider_owned_fields.append("predicate")
    if exact_owner:
        required.append("node_ref")
        provider_owned_fields.extend(("node_ref", "effect_node_ref"))
        provider_owned_relations["effect_node_ref"] = {
            "must_equal": "node_ref", "reference": "provider_local_operation_id",
        }
    # ``tensor`` can still be unknown while the compiler-side occurrence
    # binder is completing.  That is a normal fail-closed Source preparation
    # state, not a malformed wire.  It remains explicit so no consumer
    # invents a parallel placeholder or treats an unfinished ledger as a
    # provider invitation to guess.
    template: dict[str, Any] = {
        "access_label": label,
        "ordinal": ordinal,
        "kind": kind,
        "tensor": tensor,
        "index_expression": "<provider_authored_index_expression>",
        "predicate": predicate,
        "fill_value": row.get("fill_value"),
        # Effect semantics are provider-authored in the same Source-occurrence
        # transaction as the access. The host may allocate e* identity, but it
        # must never infer a missing kind/target/node owner from the access.
        "effect_kind": effect_kind,
        "effect_target": tensor,
        "effect_node_ref": "<provider_local_consumer_operation_id>" if exact_owner else None,
        **({
            "node_ref": "<provider_local_consumer_operation_id>",
        } if exact_owner else {}),
        **(
            {"semantic_access_authorization": access_view_token}
            if "semantic_access_authorization" in required else {}
        ),
        **(
            {"semantic_guard_authorization": guard_token}
            if isinstance(guard_token, str) else {}
        ),
    }
    return {
        "ordinal": ordinal,
        "access_label": label,
        "provider_fields": list(template),
        "provider_owned_fields": provider_owned_fields,
        "provider_owned_expression_fields": provider_owned_expression_fields,
        "provider_owned_relations": provider_owned_relations,
        "required_fields": required,
        "wire_template": template,
    }


def _provider_source_fill_literal(value: object) -> int | float | str | None:
    """Project one closed Source ``other=`` scalar onto JSON J spelling.

    Transfer occurrences retain a lexical Source expression so the ledger can
    prove its provenance.  A provider access row, however, is JSON typed: a
    lexical ``"0"`` must be published as numeric ``0`` rather than a string
    that the later Source-fill matcher correctly rejects.  Keep non-finite
    spellings and non-literal expressions unchanged/reject-only; this helper
    never evaluates an expression or fabricates a fill.
    """

    return source_fill_json_literal(value)


def _logical_transfer_projection(construction: Any) -> dict[str, Any]:
    """Project closed transfer facts onto the provider semantic namespace.

    The transfer ledger itself is a host-private Source/ABI proof and may
    contain physical pointer or compiler spellings.  A Lift owner must instead
    see the one semantic projection derived from that same ledger, allocator,
    and dimension order.  This intentionally does *not* turn an unbound ABI
    address term into a J value: only a proven logical ``t*`` or a ``dim*``
    alias is published.  Any remaining computation is provider-authored local
    ``r*`` SSA and remains subject to the normal semantic graph checks.
    """

    allocator = getattr(construction, "allocator", None)
    source_facts = getattr(construction, "source_facts", None)
    ledger = getattr(construction, "transfer_ledger", None)
    if allocator is None or source_facts is None or ledger is None:
        raise TypeError("semantic transfer projection requires canonical construction")
    logical_bindings = allocator.logical_binding_map()
    if not isinstance(logical_bindings, Mapping):
        raise TypeError("canonical allocator logical bindings are unavailable")
    declared_dimensions = tuple(allocator.declared("x"))
    if any(re.fullmatch(r"x[0-9]+", item) is None for item in declared_dimensions):
        raise ValueError("canonical allocator dimensions are malformed")
    provider_dimension_for_canonical = provider_dimension_alias_map(
        [
            {"ordinal": index, "local": f"dim{index}"}
            for index, _handle in enumerate(declared_dimensions)
        ],
        canonical_dimensions=declared_dimensions,
    )
    dimensions = list(provider_dimension_for_canonical.values())
    dimension_for_token = {
        coordinate: dimensions[index]
        for index, (coordinate, _extent) in enumerate(
            getattr(source_facts, "shape_coordinates", ())
        )
        if isinstance(coordinate, str) and index < len(dimensions)
    }

    # The provider index wire has one component per accessed logical tensor,
    # not one component for every dimension declared by the whole kernel.  A
    # previous projection reused ``dimensions`` below for every occurrence.
    # That made a rank-1 transfer in a rank-2 kernel appear as a two-component
    # J index.  MCP then correctly required the provider to echo that bad
    # template, while the Source-ledger matcher correctly rejected it.  Derive
    # the components from the one Source-owned tensor declaration instead.
    # This is a projection correction only: it neither infers an address nor
    # relaxes the later occurrence/access/effect binding proof.
    seed = canonical_source_wire_seed(construction)
    seed_j = seed.get("j") if isinstance(seed, Mapping) else None
    seed_intent = seed_j.get("intent") if isinstance(seed_j, Mapping) else None
    seed_function = (
        seed_intent.get("intent_function") if isinstance(seed_intent, Mapping) else None
    )
    if not isinstance(seed_function, Mapping):
        raise ValueError("canonical Source seed lacks logical tensor declarations")
    tensor_dimensions: dict[str, tuple[str, ...]] = {}
    for field in ("inputs", "outputs"):
        declarations = seed_function.get(field)
        if not isinstance(declarations, list):
            raise ValueError("canonical Source tensor declarations are malformed")
        for declaration in declarations:
            name = declaration.get("name") if isinstance(declaration, Mapping) else None
            shape = declaration.get("shape") if isinstance(declaration, Mapping) else None
            if not isinstance(name, str) or not isinstance(shape, list):
                raise ValueError("canonical Source tensor declaration is malformed")
            try:
                provider_shape = tuple(
                    provider_dimension_for_canonical[extent]
                    if isinstance(extent, str)
                    else str(extent)
                    for extent in shape
                )
            except KeyError as exc:
                raise ValueError(
                    "canonical Source tensor shape references an undeclared dimension"
                ) from exc
            # Empty shape is valid for a physical rank-zero tensor. The exact
            # Source pointer binding below distinguishes it from an ABI scalar.
            if any(not item for item in provider_shape):
                raise ValueError("canonical Source tensor shape is malformed")
            previous = tensor_dimensions.get(name)
            if previous is not None and previous != provider_shape:
                raise ValueError("canonical Source tensor declarations disagree on shape")
            tensor_dimensions[name] = provider_shape

    # Source lexical expressions can use either opaque ABI handles or their
    # original source spelling.  Both maps are internal inputs to this one
    # projection; only the logical output names survive it.
    source_to_opaque = getattr(source_facts, "source_to_opaque", ())
    opaque_for_source = {
        source: opaque
        for source, opaque in source_to_opaque
        if isinstance(source, str) and isinstance(opaque, str)
    }
    logical_for_token: dict[str, str] = {
        abi: logical
        for abi, (logical, _role) in logical_bindings.items()
        if isinstance(abi, str)
        and isinstance(logical, str)
        and re.fullmatch(r"t[0-9]+", logical) is not None
    }
    for source, opaque in opaque_for_source.items():
        logical = logical_for_token.get(opaque)
        if logical is not None:
            logical_for_token[source] = logical

    semantic_token_map = {**logical_for_token, **dimension_for_token}
    allowed_expression_names = {
        *semantic_token_map.values(),
        "and",
        "or",
        "not",
        "min",
        "max",
        "ceil_div",
        "cdiv",
        "true",
        "false",
    }

    def semantic_expression(value: str | None) -> str | None:
        """Translate only expressions whose symbols are fully logical."""

        if not isinstance(value, str) or not value.strip():
            return None

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            return semantic_token_map.get(token, token)

        projected = re.sub(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])", replace, value)
        # A raw ABI/compiler token is never a semantic fallback.  Do not
        # publish a partially translated expression and invite its reuse.
        if re.search(r"(?<![A-Za-z0-9_])[pc][0-9]+(?![A-Za-z0-9_])", projected):
            return None
        names = set(
            re.findall(
                r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])",
                projected,
            )
        )
        if not names <= allowed_expression_names:
            return None
        return projected

    binding_by_ordinal = {
        row.get("ordinal"): row
        for row in getattr(ledger, "bindings", ())
        if isinstance(row, Mapping) and is_canonical_ordinal(row.get("ordinal"))
    }

    # Keep one compact receipt beside the semantic transfer rows.  It binds
    # this projection to the exact host launch/ABI contract already frozen in
    # SourceFacts, while keeping physical ABI names and launch expressions out
    # of the provider authoring surface.
    host_launch_receipt = source_host_launch_contract_receipt(
        source_facts,
        ledger,
    )

    occurrences: list[dict[str, Any]] = []
    for occurrence in ledger.occurrences:
        binding = binding_by_ordinal.get(occurrence.ordinal)
        logical_tensor = occurrence.logical_ref
        if not isinstance(logical_tensor, str) or re.fullmatch(r"t[0-9]+", logical_tensor) is None:
            # An unbound occurrence cannot be authored safely.  Publish the
            # ordinal/kind as a fail-closed obligation, never its pointer.
            logical_tensor = None
        if isinstance(logical_tensor, str):
            occurrence_dimensions = tensor_dimensions.get(logical_tensor)
            if occurrence_dimensions is None:
                raise ValueError("Source transfer occurrence lacks a closed logical tensor shape")
            if not occurrence_dimensions:
                argument = source_facts.argument_by_position(occurrence.pointer_position)
                if (
                    argument is None or argument.get("kind") not in {"tensor", "pointer"}
                    or argument.get("shape") != []
                ):
                    raise ValueError("rank-zero access requires an exact Source tensor pointer")
        else:
            # Before a Source observation binds a logical owner, this row is
            # explicitly blocked and cannot be submitted.  Retain the broad
            # dimension vocabulary only for that diagnostic projection; once
            # an owner exists, the branch above is mandatory and publishes
            # its exact rank-specific wire.
            occurrence_dimensions = tuple(dimensions)
        address_terms: list[dict[str, Any]] = []
        for atoms, coefficient in occurrence.address_terms:
            translated = [semantic_token_map.get(atom, atom) for atom in atoms]
            if any(
                atom not in semantic_token_map.values()
                for atom in translated
            ):
                continue
            address_terms.append({"refs": translated, "coefficient": coefficient})
        predicate = semantic_expression(occurrence.predicate)
        occurrence_inputs = {
            logical_tensor,
            *(
                ref
                for term in address_terms
                for ref in term["refs"]
                if isinstance(ref, str) and re.fullmatch(r"t[0-9]+", ref)
            ),
            *(
                re.findall(r"(?<![A-Za-z0-9_])t[0-9]+(?![A-Za-z0-9_])", predicate)
                if isinstance(predicate, str)
                else ()
            ),
        }
        source_fill = occurrence.fill_value if occurrence.kind == "load" and occurrence.masked else None
        provider_fill = _provider_source_fill_literal(source_fill)
        raw_authorizations = (
            binding.get("semantic_authorizations")
            if isinstance(binding, Mapping) else None
        )
        owner_authorization = (
            raw_authorizations.get("owner")
            if isinstance(raw_authorizations, Mapping) else None
        )
        guard_authorization = (
            raw_authorizations.get("computed_guard")
            if isinstance(raw_authorizations, Mapping) else None
        )
        access_view_authorization = (
            raw_authorizations.get("access_view")
            if isinstance(raw_authorizations, Mapping) else None
        )

        def public_authorization(value: object, *, owner: bool = False) -> dict[str, Any] | None:
            if (
                not isinstance(value, Mapping)
                or value.get("status") != "pass"
                or not isinstance(value.get("token"), str)
                or re.fullmatch(
                    SOURCE_OWNER_TOKEN_PATTERN if owner else r"so_[0-9a-f]{32}", value["token"]
                ) is None
                or not isinstance(value.get("operation_family"), str)
            ):
                return None
            public = {
                "token": value["token"],
                "operation_family": value["operation_family"],
            }
            outline = value.get("authoring_outline") if owner else None
            if outline is not None:
                # Only a Source-issued logical graph whose complete store
                # commitment matches the token may enter the authoring wire.
                if (
                    not isinstance(outline, Mapping)
                    or set(outline) != {"store", "graph"}
                    or not isinstance(outline["store"], Mapping)
                    or not isinstance(outline["graph"], list)
                    or tensor_owner_digest(outline["store"], outline["graph"])
                    != value["token"].rsplit("_", 1)[1]
                ):
                    raise ValueError("Source owner authoring outline is not committed")
                public["authoring_outline"] = copy.deepcopy(outline)
            if owner:
                # The host verifies the authored node_ref against its private
                # receipt. The provider must not round-trip an owner token.
                public.pop("token")
            return public

        row: dict[str, Any] = {
            "ordinal": occurrence.ordinal,
            "access_label": f"access{occurrence.ordinal}",
            "kind": occurrence.kind,
            "tensor": logical_tensor,
            "masked": occurrence.masked,
            "fill_value": provider_fill,
            "predicate": predicate,
            "effect_order": (
                binding.get("effect_order", occurrence.ordinal)
                if isinstance(binding, Mapping)
                else occurrence.ordinal
            ),
            "index": {
                "dimensions": list(occurrence_dimensions),
                "logical_inputs": sorted(
                    item for item in occurrence_inputs if isinstance(item, str)
                ),
                "address_terms": address_terms,
                "local_result_policy": "use a previously defined r* when a computed index or predicate is required",
            },
            "pi_binding": (
                {
                    "status": "source_address_owner_bound",
                    "access_label": f"access{occurrence.ordinal}",
                    "tensor": logical_tensor,
                    "address_profile": {
                        "dynamic_terms": occurrence.address_profile[0],
                        "constant_terms": occurrence.address_profile[1],
                        "constant_offsets": list(occurrence.address_profile[2]),
                    },
                }
                if occurrence.address_profile is not None and logical_tensor is not None
                else None
            ),
            "semantic_authorizations": {
                "access_view": public_authorization(access_view_authorization),
                "owner": public_authorization(owner_authorization, owner=True),
                "computed_guard": public_authorization(guard_authorization),
            },
        }
        # A missing translated predicate is a visible unresolved semantic
        # obligation, not a hidden physical expression.  A provider-local i1
        # is legal only when the exact Source/compiler ledger issued the
        # corresponding guard receipt.  Without either representation, this
        # row is deliberately blocked before Lift instead of inviting a
        # rejected local predicate that FunctionalBoundary cannot consume.
        if occurrence.masked and row["predicate"] is None:
            if isinstance(row["semantic_authorizations"].get("computed_guard"), Mapping):
                row["predicate_requirement"] = (
                    "author a domain predicate checked by the exact producer "
                    "proof, or a prior local i1 with the issued guard receipt"
                )
            else:
                row["predicate_requirement"] = (
                    "no closed logical predicate or ledger-authorized local "
                    "guard is available; do not submit this occurrence"
                )
                row["authoring_blocker"] = (
                    "source_computed_guard_authorization_unavailable"
                )
        if source_fill is not None and provider_fill is None:
            row["fill_requirement"] = (
                "Source fill is not a closed JSON scalar; do not submit this occurrence"
            )
        if logical_tensor is None:
            row["tensor_requirement"] = "no logical tensor binding was proven; do not submit this occurrence"
        if not isinstance(binding, Mapping):
            row["binding_requirement"] = "host source transfer binding is missing"
        # This is the one semantic-transfer authority for provider access
        # identity and row grammar.  Requirements, MCP admission and the
        # semantic CandidateCheck must project these fields; none may rebuild
        # a second label/ordinal/template table from the occurrence.
        row.update(_semantic_access_wire_slot(row))
        occurrences.append(row)
    projection = {
        "status": ledger.status,
        "host_launch_contract": host_launch_receipt,
        "occurrences": occurrences,
        "access_label_pattern": r"^access[0-9]+$",
        "semantic_reference_policy": {
            "tensor": "t* only when listed",
            "dimension": "dim* only when listed",
            "local_value": "prior r* only",
        },
    }
    # Keep the private/public authorization boundary enforced at the source of
    # the projection, not only in one consumer.  Access/guard receipts may
    # expose their deliberately public token/family pair; owner tokens,
    # status, and compiler attestations never may.  The host retains the
    # complete receipt in ``ledger.bindings`` for preflight and admission.
    for occurrence_row in occurrences:
        authorizations = occurrence_row.get("semantic_authorizations")
        if (
            not isinstance(authorizations, Mapping)
            or set(authorizations) != {"access_view", "owner", "computed_guard"}
        ):
            raise ValueError("semantic transfer authorization projection is malformed")
        for name, authorization in authorizations.items():
            if authorization is None:
                continue
            if not isinstance(authorization, Mapping):
                raise ValueError("semantic transfer authorization projection is malformed")
            if "status" in authorization or "compiler_attestation" in authorization:
                raise ValueError("semantic transfer projection leaked private receipt fields")
            if name == "owner":
                if "token" in authorization or set(authorization) - {
                    "operation_family", "authoring_outline"
                }:
                    raise ValueError("semantic transfer projection leaked Source owner token")
            elif set(authorization) != {"token", "operation_family"}:
                raise ValueError("semantic transfer authorization public shape is malformed")
    if re.search(
        r"(?<![A-Za-z0-9_])[pc][0-9]+(?![A-Za-z0-9_])",
        json.dumps(projection, sort_keys=True, ensure_ascii=False),
    ):
        raise ValueError("semantic transfer projection leaked a physical handle")
    return projection


def _semantic_construction_requirements(
    construction: Any,
    output_construction: Sequence[Mapping[str, Any]],
    *,
    transfer_projection: Mapping[str, Any] | None = None,
    provider_dimension_aliases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the one targeted semantic ingress contract for a Source.

    The provider-facing surface used to describe operation, access, and output
    obligations in three different places: the transfer projection, the
    output rows, and prompt prose.  A provider could therefore submit a valid
    looking body while omitting the boundary accesses needed by the canonical
  constructor.  This compact contract is derived from the same construction
  authority and contains each occurrence/output obligation together with the
  minimal already-proven logical facts needed to author that row. It is the
  only provider-facing copy of those facts; host-only ledger/ABI fields remain
  outside the contract.

    It is a contract, not a repair plan: the host derives canonical ids,
    scopes, returns, C rows, and Source-proven pi rows after one complete
    semantic submission. Access/effect meaning remains provider-authored; the
    host only allocates their canonical identities. Unproved Source facts
    remain explicit blockers instead of being guessed here.
    """

    ledger = getattr(construction, "transfer_ledger", None)
    if ledger is None:
        raise TypeError("semantic construction requirements need a transfer ledger")
    dimension_aliases = provider_dimension_alias_map(
        list(provider_dimension_aliases),
        canonical_dimensions=tuple(construction.allocator.declared("x")),
    )
    projection = (
        dict(transfer_projection)
        if isinstance(transfer_projection, Mapping)
        else _logical_transfer_projection(construction)
    )
    occurrences = projection.get("occurrences")
    if not isinstance(occurrences, list):
        raise TypeError("semantic transfer projection lacks occurrences")

    access_rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for row in occurrences:
        if not isinstance(row, Mapping):
            raise TypeError("semantic transfer projection has a malformed occurrence")
        wire_slot = {
            key: copy.deepcopy(row.get(key))
            for key in (
                "ordinal",
                "access_label",
                "provider_fields",
                "provider_owned_fields",
                "provider_owned_expression_fields",
                "provider_owned_relations",
                "required_fields",
                "wire_template",
            )
        }
        if (
            not is_canonical_ordinal(wire_slot["ordinal"])
            or wire_slot["access_label"] != f"access{wire_slot['ordinal']}"
            or not isinstance(wire_slot["provider_fields"], list)
            or not isinstance(wire_slot["provider_owned_fields"], list)
            or wire_slot["provider_owned_expression_fields"] != [
                "index_expression"
            ]
            or "index_expression" in wire_slot["provider_owned_fields"]
            or not isinstance(wire_slot["provider_owned_relations"], Mapping)
            or not isinstance(wire_slot["required_fields"], list)
            or not isinstance(wire_slot["wire_template"], Mapping)
            or wire_slot["wire_template"].get("ordinal") != wire_slot["ordinal"]
            or wire_slot["wire_template"].get("access_label")
            != wire_slot["access_label"]
            or "id" in wire_slot["wire_template"]
            or set(wire_slot["provider_owned_relations"])
            - {"effect_node_ref"}
            or (
                wire_slot["provider_owned_relations"].get("effect_node_ref")
                == {
                    "must_equal": "node_ref",
                    "reference": "provider_local_operation_id",
                }
            )
            != (
                "node_ref" in wire_slot["provider_owned_fields"]
                and "effect_node_ref" in wire_slot["provider_owned_fields"]
            )
            or ("node_ref" in wire_slot["required_fields"])
            != ("node_ref" in wire_slot["provider_owned_fields"])
        ):
            raise ValueError("semantic construction requirements lack canonical access wire")
        item: dict[str, Any] = {
            **wire_slot,
            # This is the only provider-facing copy of the already-proven
            # semantic occurrence facts.  Keep it beside the obligation so a
            # provider does not have to reconcile a second transfer projection
            # with this checklist.  It contains no physical ABI/compiler name.
            "source_semantics": {
                "kind": row.get("kind"),
                "tensor": row.get("tensor"),
                "masked": row.get("masked"),
                "predicate": row.get("predicate"),
                "fill_value": row.get("fill_value"),
                "index": copy.deepcopy(row.get("index")),
                "effect_order": row.get("effect_order"),
                "effect": {
                    "kind": wire_slot["wire_template"].get("effect_kind"),
                    "target": wire_slot["wire_template"].get("effect_target"),
                    "node_ref": wire_slot["wire_template"].get("effect_node_ref"),
                },
                "provider_owned_relations": copy.deepcopy(
                    wire_slot["provider_owned_relations"]
                ),
                "semantic_authorizations": copy.deepcopy(
                    row.get("semantic_authorizations")
                ),
                "predicate_requirement": row.get("predicate_requirement"),
                "pi_binding": copy.deepcopy(row.get("pi_binding")),
            },
        }
        authorizations = row.get("semantic_authorizations")
        owner = authorizations.get("owner") if isinstance(authorizations, Mapping) else None
        outline = owner.get("authoring_outline") if isinstance(owner, Mapping) else None
        if isinstance(outline, Mapping):
            item["source_semantics"]["owner_broadcast_requirements"] = (
                owner_broadcast_requirements(outline)
            )
        index = row.get("index")
        missing: list[str] = []
        if not isinstance(row.get("tensor"), str):
            missing.append("tensor")
        if row.get("masked") is True:
            if row.get("predicate") is None:
                missing.append("predicate")
            if row.get("kind") == "load" and row.get("fill_value") is None:
                missing.append("fill_value")
        # An empty address profile is still a valid Source fact only when the
        # provider can author the typed logical index from the listed inputs.
        # Keep that obligation visible rather than manufacturing an index.
        if not isinstance(index, Mapping) or not index.get("dimensions"):
            missing.append("index_expression")
        if missing:
            blocked.append(
                {
                    "ordinal": row.get("ordinal"),
                    "access_label": row.get("access_label"),
                    "missing_fields": missing,
                }
            )
        # Keep the Source facts explanatory, but make the executable provider
        # row itself a single canonical wire template.  In particular, there
        # is no ``id`` alias for the same Source occurrence.
        access_rows.append(item)

    if getattr(ledger, "status", None) != "pass":
        blocked.insert(
            0,
            {
                "scope": "source_transfer_ledger",
                "missing_fields": ["closed_source_transfer_ledger"],
            },
        )

    def source_store_requirements(output_name: Any) -> list[dict[str, Any]] | None:
        """Group only already-authorized physical writes for one output.

        This is a compact navigation aid for the provider.  It is deliberately
        not a partition proof and never chooses a J producer or a local label.
        If any matching physical write lacks a closed Source owner outline or
        a closed coordinate list, omit the grouping rather than projecting a
        partial relation that could be mistaken for complete coverage.
        """

        if not isinstance(output_name, str):
            return None
        selected: list[dict[str, Any]] = []
        for access_row in access_rows:
            if not isinstance(access_row, Mapping):
                continue
            source = access_row.get("source_semantics")
            if not isinstance(source, Mapping):
                continue
            if (
                source.get("kind") not in {"store", "update"}
                or source.get("tensor") != output_name
            ):
                continue
            authorizations = source.get("semantic_authorizations")
            owner = (
                authorizations.get("owner")
                if isinstance(authorizations, Mapping)
                else None
            )
            outline = (
                owner.get("authoring_outline")
                if isinstance(owner, Mapping)
                else None
            )
            store = outline.get("store") if isinstance(outline, Mapping) else None
            coordinates = store.get("coordinates") if isinstance(store, Mapping) else None
            family = owner.get("operation_family") if isinstance(owner, Mapping) else None
            ordinal = access_row.get("ordinal")
            if (
                not is_canonical_ordinal(ordinal)
                or not isinstance(family, str)
                or not family
                or not isinstance(coordinates, list)
                or any(
                    not (
                        isinstance(item, str)
                        and (
                            re.fullmatch(r"axis[0-9]+", item) is not None
                            or re.fullmatch(r"[0-9]+", item) is not None
                        )
                    )
                    and not (isinstance(item, int) and not isinstance(item, bool))
                    for item in coordinates
                )
            ):
                return None
            selected.append(
                {
                    "ordinal": ordinal,
                    "kind": source.get("kind"),
                    "operation_family": family,
                    "coordinates": copy.deepcopy(coordinates),
                }
            )
        return selected or None

    def source_store_partition_assembly(
        output_name: Any,
        output_type: Any,
        store_rows: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Project only a statically closed physical-store assembly.

        A locator is not enough to make a provider-visible logical result.  An
        assembly is exposed only when the existing Source owner outlines and
        store coordinates prove one complete, ordered, unit-width partition of
        the declared logical output.  No J producer label is allocated here;
        the provider must still author the canonical ``concat`` operation and
        the later functional-boundary proof remains authoritative.
        """

        if (
            not isinstance(output_name, str)
            or not isinstance(output_type, Mapping)
            or not isinstance(output_type.get("dtype"), str)
            or not isinstance(output_type.get("shape"), list)
            or not isinstance(store_rows, list)
            or len(store_rows) < 2
        ):
            return None
        output_shape = copy.deepcopy(output_type["shape"])
        output_dtype = output_type["dtype"]
        rank = len(output_shape)
        if rank < 1 or any(
            not (
                isinstance(extent, str)
                and bool(extent)
                or isinstance(extent, int)
                and not isinstance(extent, bool)
                and extent >= 0
            )
            for extent in output_shape
        ):
            return None

        access_by_ordinal = {
            row.get("ordinal"): row
            for row in access_rows
            if isinstance(row, Mapping) and is_canonical_ordinal(row.get("ordinal"))
        }
        owner_rows: list[tuple[dict[str, Any], list[Any], dict[str, Any]]] = []
        for store_row in store_rows:
            ordinal = store_row.get("ordinal")
            access_row = access_by_ordinal.get(ordinal)
            source = access_row.get("source_semantics") if isinstance(access_row, Mapping) else None
            authorizations = (
                source.get("semantic_authorizations")
                if isinstance(source, Mapping)
                else None
            )
            owner = (
                authorizations.get("owner")
                if isinstance(authorizations, Mapping)
                else None
            )
            outline = (
                owner.get("authoring_outline")
                if isinstance(owner, Mapping)
                else None
            )
            graph = outline.get("graph") if isinstance(outline, Mapping) else None
            root = graph[-1] if isinstance(graph, list) and graph else None
            coordinates = store_row.get("coordinates")
            if (
                not isinstance(source, Mapping)
                or source.get("kind") != "store"
                or source.get("tensor") != output_name
                or source.get("masked") is True
                or source.get("predicate") is not None
                or not isinstance(owner, Mapping)
                or not isinstance(outline, Mapping)
                or not isinstance(graph, list)
                or not isinstance(root, (list, tuple))
                or len(root) != 5
                or not isinstance(root[1], str)
                or not isinstance(root[2], list)
                or not isinstance(coordinates, list)
                or len(coordinates) != rank
            ):
                return None
            try:
                root_shape = provider_shape_from_canonical(
                    root[2], dimension_aliases
                )
            except ValueError:
                return None
            if any(
                not (
                    isinstance(extent, str)
                    and bool(extent)
                    or isinstance(extent, int)
                    and not isinstance(extent, bool)
                    and extent >= 0
                )
                for extent in root_shape
            ):
                return None
            owner_rows.append(
                (
                    dict(store_row),
                    copy.deepcopy(coordinates),
                    {
                        "dtype": root[1],
                        "shape": root_shape,
                    },
                )
            )

        if len({row[0].get("ordinal") for row in owner_rows}) != len(owner_rows):
            return None
        candidates: list[tuple[int, list[tuple[dict[str, Any], list[Any], dict[str, Any]]]]] = []
        for axis in range(rank):
            extent = output_shape[axis]
            if type(extent) is not int or extent != len(owner_rows):
                continue
            axis_coordinates = [row[1][axis] for row in owner_rows]
            if (
                any(type(value) is not int for value in axis_coordinates)
                or sorted(axis_coordinates) != list(range(extent))
            ):
                continue
            if any(
                any(
                    row[1][dimension] != owner_rows[0][1][dimension]
                    for row in owner_rows
                )
                for dimension in range(rank)
                if dimension != axis
            ):
                continue
            expected_slice_shape = [
                value for dimension, value in enumerate(output_shape) if dimension != axis
            ]
            if any(
                row[2]["dtype"] != output_dtype
                or row[2]["shape"] != expected_slice_shape
                for row in owner_rows
            ):
                continue
            candidates.append((axis, owner_rows))
        if len(candidates) != 1:
            return None
        axis, rows = candidates[0]
        ordered = sorted(rows, key=lambda row: row[1][axis])
        slice_type = {
            "dtype": output_dtype,
            "shape": [
                value for dimension, value in enumerate(output_shape) if dimension != axis
            ],
        }
        assembly = {
            "canonical_op": "concat",
            "axis": axis,
            "operand_store_ordinals": [row[0]["ordinal"] for row in ordered],
            "operand_coordinates": [row[1] for row in ordered],
            "operand_type": slice_type,
            "result_type": {
                "dtype": output_dtype,
                "shape": output_shape,
            },
        }
        validate_source_store_partition_assembly(
            {"assembly": assembly},
            access_rows=access_rows,
            output_name=output_name,
            output_type=output_type,
            expected_stores=[row[0] for row in ordered],
            dimension_aliases=dimension_aliases,
        )
        return assembly

    outputs: list[dict[str, Any]] = []
    for row in output_construction:
        if not isinstance(row, Mapping):
            raise TypeError("output construction has a malformed row")
        output_row = {
            "output": row.get("output"),
            "output_type": copy.deepcopy(row.get("output_type")),
            "required_fields": ["value"],
            # Output ordinal/name/type are Source/host facts.  The one
            # provider wire is deliberately just the selected semantic
            # value, so an MCP receipt can show the exact legal shape
            # rather than inviting the provider to echo these descriptors.
            "wire_template": {"value": "<local_result_or_source_t>"},
        }
        store_rows = source_store_requirements(row.get("output"))
        if store_rows is not None:
            # These are Source facts for navigation only.  They do not prove
            # complete partition coverage and do not authorize a producer.
            output_row["source_store_requirements"] = store_rows
            if len(store_rows) > 1:
                # Multiple Source-authorized physical writes may be slices of
                # one logical output.  Expose that already-observed relation
                # explicitly so graph authoring does not mistake a slice root
                # for the final output value.  This is a requirement locator,
                # not a partition proof and does not choose scatter/update
                # semantics; the existing canonical/functional checks remain
                # responsible for proving the complete binding later.
                output_row["source_store_partition_requirement"] = {
                    "output": row.get("output"),
                    "store_ordinals": [
                        item["ordinal"] for item in store_rows
                    ],
                    "store_coordinates": [
                        copy.deepcopy(item["coordinates"])
                        for item in store_rows
                    ],
                    "relation": (
                        "physical_store_slices_require_complete_logical_output_binding"
                    ),
                }
                assembly = source_store_partition_assembly(
                    row.get("output"), row.get("output_type"), store_rows
                )
                if assembly is not None:
                    output_row["source_store_partition_requirement"]["assembly"] = (
                        assembly
                    )
        outputs.append(output_row)

    return {
        "authority": "SourceFacts + SourceTransferLedger + CanonicalHandleAllocator",
        "source_transfer": {
            "status": getattr(ledger, "status", None),
            "reason": getattr(ledger, "reason", None),
            "closed_required": True,
        },
        "submission": {
            "root": ["j", "pi"],
            "construction_sections": {
                "operation": "j.intent.intent_function.body",
                "scope": "derived from the same complete operation body",
                "boundary": (
                    "provider-authored access/effect occurrence rows + "
                    "j.boundary.output_mappings"
                ),
                "output": "output rows + j.boundary.output_mappings",
            },
            "required": [
                "j.intent.intent_function.body",
                "j.boundary.accesses",
                "j.boundary.output_mappings",
                "pi",
            ],
            "atomic": True,
            "one_complete_candidate": True,
        },
        "semantic_required": {
            "operation_body": {
                "required": True,
                "rule": "complete producer-before-consumer operation rows; structured body/yield fields stay together",
            },
            "scope": {
                "required": True,
                "rule": "the complete operation body is the sole input to host lexical scope construction; keep nested loop/branch bodies and yields in the same submission",
            },
            "boundary": {
                "required": True,
                "rows": ["accesses", "output_mappings"],
                "rule": (
                    "submit access and effect semantics in Source order plus "
                    "outputs in ABI order; host allocates ids and attaches "
                    "owner receipts only after checking the declared node_ref"
                ),
            },
            "accesses": {
                "count": len(access_rows),
                "rows": access_rows,
                "order": "Source ordinal; access_label is local and exact",
            },
            "outputs": {
                "count": len(outputs),
                "rows": outputs,
                "order": "Source output ABI order",
            },
            "consumer_extensions": {
                "required_when": (
                    "one predicate-bearing load access contributes an operand "
                    "to a reduce or inclusive scan"
                ),
                "rule": (
                    "author one consumer_extensions row for every distinct direct "
                    "consumer operand edge; consumer_ref is the provider-local op* "
                    "consumer, access_ref is the exact provider-local accessN row, "
                    "tile_axis exactly equals the non-negative integer tile_axis "
                    "in that same consumer_ref's one boundary.reductions row, and "
                    "outside is the typed two-sided identity for that consumer. "
                    "Do not merge edges, infer an extension for another operation "
                    "class, or ask the host to create one"
                ),
            },
        },
        "host_derived": [
            "schema",
            "c",
            "signature",
            "canonical n*/v*/x*/a*/e*/q* handles",
            "scope records from the complete operation graph",
            "effects from the complete access rows and Source ledger",
            "index_bounds from the final closed gather/scatter operation graph",
            "returns and output attachment rows",
            "Source-proven pi bindings",
        ],
        "unproved_source_fields": blocked,
        "fail_closed": (
            "For provider-owned fields in unproved_source_fields, derive the "
            "value from the mounted anonymous Source/compiler evidence; never "
            "guess or omit a required row. Host-owned unresolved fields remain "
            "omitted and are rejected by the local constructor."
        ),
    }


def _semantic_candidate_profile(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project the provider semantic surface from the complete artifact schema.

    The constructor owns all omitted ABI/C/node/scope rows, but the provider
    must not be able to choose a different JSON container for an optional
    semantic collection.  This projection intentionally checks only direct
    container kinds; nested operation and boundary semantics remain subject to
    the same final fail-closed CandidateCheck after host construction.
    """

    root = schema.get("properties")
    if not isinstance(root, Mapping):
        raise TypeError("canonical artifact schema has no root properties")
    j_schema = root.get("j")
    if not isinstance(j_schema, Mapping):
        raise TypeError("canonical artifact schema has no J schema")
    j_properties = j_schema.get("properties")
    if not isinstance(j_properties, Mapping):
        raise TypeError("canonical artifact J schema has no properties")
    boundary_schema = j_properties.get("boundary")
    if not isinstance(boundary_schema, Mapping):
        raise TypeError("canonical artifact schema has no boundary schema")
    boundary_properties = boundary_schema.get("properties")
    if not isinstance(boundary_properties, Mapping):
        raise TypeError("canonical boundary schema has no properties")
    intent_schema = j_properties.get("intent")
    if not isinstance(intent_schema, Mapping):
        raise TypeError("canonical artifact schema has no intent schema")
    intent_properties = intent_schema.get("properties")
    function_schema = (
        intent_properties.get("intent_function")
        if isinstance(intent_properties, Mapping)
        else None
    )
    function_properties = (
        function_schema.get("properties")
        if isinstance(function_schema, Mapping)
        else None
    )
    body_schema = (
        function_properties.get("body")
        if isinstance(function_properties, Mapping)
        else None
    )
    if not isinstance(body_schema, Mapping) or body_schema.get("type") != "array":
        raise TypeError("canonical artifact schema has no operation body array")

    def direct_type(value: Any, path: str) -> str:
        if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
            raise TypeError(f"canonical semantic field {path} lacks direct type")
        return value["type"]

    j_field_types = {
        name: direct_type(j_properties.get(name), f"j.{name}")
        for name in CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS
    }
    boundary_field_types = {
        str(name): direct_type(value, f"j.boundary.{name}")
        for name, value in boundary_properties.items()
    }
    # A provider always starts from one complete *semantic* skeleton.  It is
    # not the final artifact (the host still owns schema/C/ABI/canonical
    # handles), but it deliberately contains every semantic J collection and
    # pi as a sibling.  Leaving the optional arrays out here forced prompts and
    # MCP views to describe their nesting independently, which in turn let a
    # provider flatten J and append pi as a second JSON fragment.
    template_j: dict[str, Any] = {}
    for name in CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS:
        field_type = j_field_types[name]
        if name == "intent":
            template_j[name] = {"intent_function": {"body": []}}
        elif field_type == "object":
            template_j[name] = {}
        elif field_type == "array":
            template_j[name] = []
        else:
            raise TypeError(
                f"canonical semantic field j.{name} cannot form a JSON skeleton"
            )
    pi_schema = root.get("pi")
    if not isinstance(pi_schema, Mapping) or pi_schema.get("type") != "object":
        raise TypeError("canonical artifact schema has no object pi field")
    template = {"j": template_j, "pi": {}}
    return {
        "root_fields": list(CANONICAL_SEMANTIC_CANDIDATE_ROOT_FIELDS),
        "j_fields": list(CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS),
        "required_j_paths": ["j.intent.intent_function.body", "j.boundary"],
        "host_owned_root_fields": ["schema", "c"],
        "host_owned_j_fields": [
            name
            for name in CANONICAL_J_WIRE_FIELDS
            if name not in CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS
        ],
        "forbidden_j_handle_prefixes": ["n", "v", "x", "p", "c", "a", "e"],
        "provider_handle_policy": {
            "allowed_source_handles": ["t*"],
            "provider_local_labels": ["op*", "r*", "dim*", "access*"],
            "forbidden_canonical_handles": [
                "n*",
                "v*",
                "x*",
                "p*",
                "c*",
                "a*",
                "e*",
            ],
            "host_derived_fields": [
                "node_ref",
                "output_producer_reference",
                "scope_id",
                "access_id",
                "effect_id",
            ],
        },
        # The final artifact's output_mapping row has output and node_ref
        # fields because it is a canonical construction record.  Neither is
        # provider-authored: at semantic ingress the only output contribution
        # is one value choice per Source-owned output, in Source ABI order.
        # Keeping this projection here prevents the full final-row schema from
        # becoming a second, contradictory authoring grammar.
        "output_selection": {
            "container": "array",
            "arity": "exactly_one_per_source_output_in_abi_order",
            "item_fields": ["value"],
            "value": "local_result_label_or_direct_source_t_handle",
        },
        "j_field_types": j_field_types,
        "boundary_field_types": boundary_field_types,
        "template": template,
    }


def _semantic_mcp_write_protocol(
    semantic_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the sole MCP semantic-write surface from the candidate schema.

    The MCP author is responsible for J meaning, but it must not have to
    reconstruct a complete artifact-shaped JSON document or independently
    maintain C/pi mechanical rows.  This projection describes the bounded
    writes which build one semantic candidate in the environment.  The final
    object still crosses the existing source-free semantic preflight and the
    one canonical Source/Harness constructor.

    Keep this next to ``_semantic_candidate_profile``: MCP tools, the mounted
    source brief, and CandidateCheck then share the same root/J/boundary field
    authority rather than spelling a fourth hand-maintained field list.
    """

    template = semantic_candidate.get("template")
    j_field_types = semantic_candidate.get("j_field_types")
    boundary_field_types = semantic_candidate.get("boundary_field_types")
    if (
        not isinstance(template, Mapping)
        or not isinstance(j_field_types, Mapping)
        or not isinstance(boundary_field_types, Mapping)
    ):
        raise TypeError("canonical semantic candidate lacks an MCP write projection")

    boundary_fields = tuple(
        field
        for field in boundary_field_types
        # Accesses and outputs have Source-order-specific append operations;
        # effects/index bounds are constructor-owned projections.
        if field not in {"accesses", "output_mappings", "effects", "index_bounds"}
    )
    j_fields = tuple(
        field
        for field in CANONICAL_SEMANTIC_CANDIDATE_J_FIELDS
        if field not in {"intent", "boundary"}
    )
    if (
        not boundary_fields
        or any(not isinstance(field, str) for field in (*boundary_fields, *j_fields))
    ):
        raise TypeError("canonical semantic MCP write fields are malformed")
    return {
        "authority": "canonical_semantic_candidate_schema",
        "initial_template": copy.deepcopy(dict(template)),
        "steps": [
            "begin_candidate",
            "declare_operation_graph",
            "declare_accesses",
            "declare_output_mappings",
            "declare_boundary_fields",
            "declare_j_fields",
            "declare_normalization_classes",
            "finalize_candidate",
        ],
        # A complete semantic submission always declares the graph, Source
        # access rows, Source-output selections, and one boundary-field
        # object.  The latter may be `{}` when no extra semantic boundary
        # relation is needed, but it is a required lifecycle edge because a
        # map/select with a non-trailing broadcast needs an explicit
        # `boundary.broadcasts` proof.  Keeping that declaration optional
        # made the phase machine hide the only legal way to submit such a
        # proof, then deferred the failure to Decode.  Other declared
        # semantic collections remain optional and still obey ``steps``
        # order.  The environment consumes this exact list to reject an early
        # finalization before any constructor or CandidateCheck work begins.
        "required_steps": [
            "begin_candidate",
            "declare_operation_graph",
            "declare_accesses",
            "declare_output_mappings",
            "declare_boundary_fields",
            "finalize_candidate",
        ],
        # Tool argument names and the direct JSON container kind are part of
        # the same authority.  MCP transport code must project these rows; it
        # must not maintain its own begin/declare/finalize parameter table.
        "tool_arguments": {
            "begin_candidate": None,
            "declare_operation_graph": {
                "field": "body_text",
                "json_container": "array",
                # Responses receives this declaration as JSON text. Keep the
                # non-empty-array invariant in the shared write contract so
                # the provider-facing function schema cannot generate `[]`.
                "pattern": r"^\s*\[\s*\{",
            },
            "declare_accesses": {
                "field": "accesses_text",
                "json_container": "array",
            },
            "declare_output_mappings": {
                "field": "mappings_text",
                "json_container": "array",
            },
            "declare_boundary_fields": {
                "field": "fields_text",
                "json_container": "object",
            },
            "declare_j_fields": {
                "field": "fields_text",
                "json_container": "object",
            },
            "declare_normalization_classes": {
                "field": "classes_text",
                "json_container": "array",
            },
            "finalize_candidate": None,
        },
        # Tool wording is part of the same semantic write authority as names
        # and argument rows.  The MCP server projects this map directly; it
        # must not carry a second handwritten description table.
        "descriptions": {
            "begin_candidate": (
                "Start one fresh schema-derived semantic J candidate."
            ),
            "declare_operation_graph": (
                "Each call atomically REPLACES the whole body; it never "
                "appends. Every call must therefore contain the complete, "
                "non-empty producer-order graph—not one row or a patch—with "
                "one J row for every begin receipt source_operation_requirement "
                "and every listed Source owner-DAG operation. Read exact "
                "operation leaves from schema_lookup. Each row has operands "
                "as an array (use [] for zero-input ops), canonical op/attrs, "
                "and fresh local id/result labels matching "
                "[A-Za-z_][A-Za-z0-9_.:-]*; do not use host ids, reserved "
                "numeric n*/v*/x*/p*/c*/a*/e* labels, TTIR spellings, or "
                "invent operations, attrs, predicates, or relations. Preserve "
                "Source dependency order and owner-DAG operations. Every "
                "result needs its exact exposed dtype/shape. Make each "
                "typed_output_requirement function-visible as an exact typed "
                "result or Source input; mapping cannot create/promote it. "
                "Each masked access needs an existing canonical-i1 result "
                "with exactly one authorized rank-insertion match; host does "
                "not create predicates or choose labels. Preserve all "
                "source_store_requirements and coordinates. On any graph "
                "error, rebuild and resubmit the entire graph from current "
                "Source facts; the host never completes a partial graph."
            ),
            "declare_accesses": (
                "After the required post-graph transfer-ledger reread, set the "
                "complete Source-order access array from its access rows; never "
                "submit a prefix or empty array. "
                "Copy every non-provider-owned field from the exposed wire "
                "template exactly. Supply only the exposed logical index, "
                "predicate/fill and provider-local owner relations. When "
                "the Source row exposes index_expression_requirement, copy "
                "that exact expression into index_expression; otherwise use "
                "a real provider-local expression, never a placeholder or null. Preserve "
                "domain, rank, mask and effect semantics. Do not emit host ids "
                "or invent coordinates, owners or predicates."
            ),
            "declare_output_mappings": (
                "Set the complete Source-output-order value-selection array. "
                "Each row is exactly `{value}` and must reference an exposed "
                "Source input or declared local result; output/node_ref are host-owned."
            ),
            "declare_boundary_fields": (
                "Set the exposed boundary object, or `{}` when none applies. "
                "Submit only exact exposed broadcasts, domains or result relations; "
                "effects and index bounds are host-derived. Never submit a partial "
                "map or invent padding/identity/update relations."
            ),
            "declare_j_fields": (
                "Set only the exposed schema-derived semantic J fields outside "
                "intent and boundary."
            ),
            "declare_normalization_classes": (
                "Set only the optional exposed normalization classes; other pi "
                "rows are host-derived."
            ),
            "finalize_candidate": (
                "Finalize once through the fail-closed ingress and CandidateCheck "
                "preflight."
            ),
        },
        "writes": {
            "operation": "j.intent.intent_function.body[]",
            "access": "j.boundary.accesses[]",
            "output_mapping": "j.boundary.output_mappings[]",
            "boundary_field": list(boundary_fields),
            "j_field": list(j_fields),
            "normalization_class": "pi.normalization_classes[]",
        },
        "host_derived": [
            "schema",
            "c",
            "signature",
            "canonical_handles",
            "scopes",
            "effects",
            "returns",
            "output_attachments",
            "pi_bindings",
        ],
        "finalize": (
            "serialize one semantic candidate, then run the existing semantic "
            "preflight and canonical Source/Harness construction transaction"
        ),
    }


def canonical_semantic_mcp_write_protocol() -> dict[str, Any]:
    """Return the schema-derived MCP write protocol used by every Lift owner."""

    profile = canonical_wire_profile()
    value = profile.get("semantic_mcp_write")
    if not isinstance(value, Mapping):
        raise AssertionError("canonical wire profile lacks MCP write protocol")
    return copy.deepcopy(dict(value))


def _provider_operation_projection(
    contract: Mapping[str, Any],
    *,
    include_attrs_schema: bool,
) -> dict[str, Any]:
    """Project one schema-owned operation onto the provider wire.

    The complete per-operation leaf is still available through the bounded
    ``schema_lookup(op:NAME)`` path.  The first-turn wire only needs the
    common operation shape and the attribute field/required lists; carrying
    every operation leaf in the initial MCP response made ``source/brief``
    exceed its hard result bound before an agent could submit a candidate.
    """

    fields = contract.get("fields")
    required = contract.get("required")
    if not isinstance(fields, list) or not isinstance(required, list):
        raise TypeError("canonical semantic operation fields are malformed")
    attrs = contract.get("attrs")
    attrs_fields: list[str] = []
    attrs_required: list[str] = []
    if isinstance(attrs, Mapping):
        properties = attrs.get("properties")
        if isinstance(properties, Mapping):
            attrs_fields = list(properties)
        required_attrs = attrs.get("required")
        if isinstance(required_attrs, list):
            attrs_required = list(required_attrs)
    item: dict[str, Any] = {
        "fields": [field for field in fields if field != "scope_id"],
        "required": [
            field for field in required if field not in {"id", "scope_id"}
        ],
        "attrs_fields": attrs_fields,
        "attrs_required": attrs_required,
    }
    # Preserve the schema-derived value-type requirement on every provider
    # projection.  In particular, the compact source brief is the first
    # authoring surface a provider sees; dropping this metadata there taught
    # the permissive final-artifact ``type`` envelope instead of the complete
    # value-type object required by canonical construction.  The same helper
    # also backs exact operation lookup, so this remains one projection rather
    # than a compact-wire-only rule.
    type_requirement = contract.get("type_requirement")
    if isinstance(type_requirement, Mapping):
        item["type_requirement"] = copy.deepcopy(dict(type_requirement))
    if include_attrs_schema:
        item.update(
            {
                "host_allocated_fields": [
                    field for field in ("scope_id",) if field in fields
                ],
                "provider_local_fields": [
                    field
                    for field in ("id", "result", "operands")
                    if field in fields
                ],
                "canonical_handle_policy": {
                    "forbidden_prefixes": list(
                        _PROVIDER_FORBIDDEN_CANONICAL_PREFIXES
                    ),
                    "id": "optional provider-local operation label",
                    "result": "provider-local result label; host allocates v*",
                    "operands": (
                        "Source t* or prior provider-local result label"
                    ),
                },
                "result_reference": (
                    "local result label; host allocates canonical v*"
                ),
                "operand_reference": (
                    "Source t* or earlier local result label"
                ),
            }
        )
        if isinstance(attrs, Mapping):
            item["attrs"] = _provider_schema_projection(attrs)
    return item


def _semantic_provider_wire(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the one compact provider submission wire from the profile.

    The artifact schema describes a post-construction J, where operation,
    scope, access and effect handles are mandatory.  A Lift submission exists
    *before* that construction, so exposing those final required fields to the
    provider creates a contradictory second wire.  This projection keeps the
    schema-derived semantic fields but marks allocator-owned fields as omitted
    or optional.  It is descriptive only: strict final validation still uses
    the unmodified artifact schema and CandidateCheck.
    """

    semantic = profile.get("semantic_candidate")
    operation_contracts = profile.get("operation_contracts")
    boundary_rows = profile.get("boundary_rows")
    first_turn = profile.get("first_turn")
    rows = profile.get("rows")
    pi = profile.get("pi")
    if not all(
        isinstance(value, Mapping)
        for value in (
            semantic,
            operation_contracts,
            boundary_rows,
            first_turn,
            rows,
            pi,
        )
    ):
        raise TypeError("canonical semantic provider wire has malformed profile")
    output_selection = semantic.get("output_selection")
    candidate_template = semantic.get("template")
    operation_namespace = semantic.get("operation_namespace")
    operation_names = first_turn.get("operation_names")
    boundary_names = first_turn.get("boundary_rows")
    if (
        not isinstance(output_selection, Mapping)
        or not isinstance(candidate_template, Mapping)
        or not isinstance(operation_namespace, Mapping)
        or not isinstance(operation_names, list)
        or not isinstance(boundary_names, list)
    ):
        raise TypeError("canonical semantic provider wire is incomplete")

    operations: dict[str, dict[str, Any]] = {}
    for name in operation_names:
        contract = operation_contracts.get(name)
        if not isinstance(name, str) or not isinstance(contract, Mapping):
            raise TypeError("canonical semantic operation contract is missing")
        # The brief uses the bounded first-turn form.  The full form is
        # generated below into the separate exact op lookup projection.
        # Both authoring transports must receive the closed attribute leaf on
        # their first turn.  A field/required-name summary is insufficient for
        # direct Responses: it has no schema_lookup action and otherwise has
        # to guess enum/value shapes before CandidateCheck.  This remains a
        # projection of the one canonical schema, not a permissive provider
        # grammar or a host-side repair path.
        operations[name] = _provider_operation_projection(
            contract,
            include_attrs_schema=True,
        )

    operation_policy = {
        "provider_local_fields": ["id", "result", "operands"],
        "host_allocated_fields": ["scope_id"],
        "operation_namespace": copy.deepcopy(dict(operation_namespace)),
        "canonical_handle_policy": {
            "forbidden_prefixes": list(_PROVIDER_FORBIDDEN_CANONICAL_PREFIXES),
            "id": "optional provider-local operation label",
            "result": "provider-local result label; host allocates v*",
            "operands": "Source t* or prior provider-local result label",
        },
        "result_reference": "local result label; host allocates canonical v*",
        "operand_reference": "Source t* or earlier local result label",
        "exact_leaf_lookup": "schema_lookup items=[\"op:NAME\"]",
    }

    boundary: dict[str, Any] = {}
    for name in boundary_names:
        row = boundary_rows.get(name)
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise TypeError("canonical semantic boundary row is missing")
        if name == "output_mappings":
            boundary[name] = {
                "semantic_submission": copy.deepcopy(dict(output_selection))
            }
            continue
        if name in {"effects", "index_bounds"}:
            boundary[name] = {
                "host_derived": True,
                "submission": "omit",
            }
            continue
        fields = row.get("fields")
        required = row.get("required")
        if not isinstance(fields, list) or not isinstance(required, list):
            raise TypeError("canonical semantic boundary fields are malformed")
        host_fields = [
            field for field in ("access_id", "effect_id", "semantic_owner_authorization")
            if field in fields
        ]
        provider_fields = [field for field in fields if field not in host_fields]
        provider_required = [field for field in required if field not in host_fields]
        if name == "accesses":
            # Access identity is needed only to join semantic relations before
            # construction.  It is deliberately a local occurrence label,
            # not the final a* identity allocated by the host.
            # One provider identity only: Source-order ``access_label`` plus
            # its exact ordinal.  ``access_id`` is host-owned and generic
            # ``id`` is deliberately not an alias on this wire.
            provider_fields = ["access_label", "ordinal", *provider_fields]
            provider_required = ["access_label", "ordinal", *provider_required]
        boundary[name] = {
            "fields": provider_fields,
            "required": provider_required,
            "host_allocated_fields": host_fields,
            "provider_local_reference_fields": [
                field
                for field in (
                    "node_ref",
                    "consumer_ref",
                    "update_node_ref",
                )
                if field in fields
            ],
            "canonical_reference_policy": {
                "node_ref": (
                    "provider-local operation label; host rewrites to canonical n*"
                ),
                "consumer_ref": (
                    "provider-local operation label; host rewrites to canonical n*"
                ),
                "update_node_ref": (
                    "provider-local operation label; host rewrites to canonical n*"
                ),
                "access_id": "omit; host allocates canonical a*",
                "effect_id": "omit; host allocates canonical e*",
                "access_ref": (
                    "provider-local accessN; host rewrites to canonical a*"
                ),
            },
        }
        row_schema = row.get("schema")
        if isinstance(row_schema, Mapping):
            provider_row_schema = _provider_row_schema_projection(
                row_schema,
                host_fields=set(host_fields),
            )
            provider_row_schema = _provider_access_reference_projection(
                provider_row_schema
            )
            if name == "accesses":
                properties = provider_row_schema.get("properties")
                if not isinstance(properties, dict):
                    raise TypeError("semantic access row lacks properties")
                properties.pop("id", None)
                properties["access_label"] = {
                    "type": "string",
                    "pattern": r"^access[0-9]+$",
                    "provider_local_access_label": True,
                }
                properties["ordinal"] = {
                    "type": "integer",
                    "minimum": 0,
                    "provider_source_ordinal": True,
                }
                required_fields = provider_row_schema.get("required")
                if not isinstance(required_fields, list):
                    raise TypeError("semantic access row lacks required fields")
                provider_row_schema["required"] = [
                    "access_label",
                    "ordinal",
                    *[field for field in required_fields if field != "access_label"],
                ]
            boundary[name]["schema"] = provider_row_schema

    pi_schema = pi.get("schema")
    if not isinstance(pi_schema, Mapping) or not isinstance(
        pi_schema.get("properties"), Mapping
    ):
        raise TypeError("canonical semantic pi wire is malformed")
    pi_properties = pi_schema["properties"]
    pi_rows = ("normalization_classes",)
    host_pi_rows = (
        "program_axes",
        "program_bindings",
        "iteration_bindings",
        "access_representation_bindings",
        "control_bindings",
        "materialization_bindings",
    )
    if any(name not in pi_properties for name in (*pi_rows, *host_pi_rows)):
        raise TypeError("canonical semantic pi rows are incomplete")
    domain = rows.get("domain")
    if not isinstance(domain, Mapping):
        raise TypeError("canonical semantic domain wire is malformed")
    return {
        "candidate_template": copy.deepcopy(dict(candidate_template)),
        "operation_namespace": copy.deepcopy(dict(operation_namespace)),
        "provider_local_label_pattern": _PROVIDER_LOCAL_LABEL_PATTERN,
        "forbidden_canonical_handle_prefixes": list(
            _PROVIDER_FORBIDDEN_CANONICAL_PREFIXES
        ),
        "host_derived_fields": [
            "node_ref",
            "output_producer_reference",
            "scope_id",
            "access_id",
            "effect_id",
        ],
        "dimension_reference": {
            "provider_form": "dim0, dim1, ...",
            "canonical_form": "x0, x1, ...",
            "host_rewrites": True,
            "operation_shape_extent_rule": PROVIDER_DIMENSION_EXTENT_RULE,
        },
        "host_derived_pi_fields": list(host_pi_rows),
        "operations": operations,
        "operation_policy": operation_policy,
        "boundary": {
            "domains": copy.deepcopy(dict(domain)),
            "rows": boundary,
        },
        "pi": {
            "provider_fields": {
                name: _provider_schema_projection(pi_properties[name])
                for name in pi_rows
            },
            "host_derived_fields": list(host_pi_rows),
        },
    }


def _compact_semantic_provider_wire(
    wire: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the shared low-density provider contract.

    The full semantic wire is useful to the mounted schema lookup, but the
    candidate skeleton, operation namespace, and canonical-reference prose
    are repetitive in the first direct Responses turn.  Boundary row schemas
    are different: direct Responses has no schema_lookup action, so omitting
    their closed leaves forces it to guess nested access/effect/output shapes.
    Keep those schemas while removing only duplicated layout and
    canonical-reference prose.  This remains one generated authority for MCP
    and direct Responses.
    """

    compact = copy.deepcopy(dict(wire))
    compact.pop("candidate_template", None)
    compact.pop("operation_namespace", None)
    operation_policy = compact.get("operation_policy")
    if isinstance(operation_policy, dict):
        operation_policy.pop("operation_namespace", None)
    boundary = compact.get("boundary")
    rows = boundary.get("rows") if isinstance(boundary, dict) else None
    if isinstance(rows, dict):
        for row in rows.values():
            if not isinstance(row, dict):
                continue
            row.pop("canonical_reference_policy", None)
    return compact
def _semantic_operation_namespace(
    operation_contracts: Mapping[str, Any],
    boundary_rows: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the complete J-op naming classification from closed schema rows."""

    if set(operation_contracts) != set(CANONICAL_J_OPS):
        raise TypeError("canonical operation namespace has an incomplete catalog")

    def attr_functions(operation: str) -> list[str]:
        contract = operation_contracts.get(operation)
        attrs = contract.get("attrs") if isinstance(contract, Mapping) else None
        properties = attrs.get("properties") if isinstance(attrs, Mapping) else None
        fn = properties.get("fn") if isinstance(properties, Mapping) else None
        values = fn.get("enum") if isinstance(fn, Mapping) else None
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise TypeError(f"canonical {operation}.attrs.fn is not closed")
        return list(values)

    accesses = boundary_rows.get("accesses")
    access_schema = accesses.get("schema") if isinstance(accesses, Mapping) else None
    properties = access_schema.get("properties") if isinstance(access_schema, Mapping) else None
    kind = properties.get("kind") if isinstance(properties, Mapping) else None
    access_kinds = kind.get("enum") if isinstance(kind, Mapping) else None
    if (
        not isinstance(access_kinds, list)
        or not all(isinstance(item, str) for item in access_kinds)
        or not {"load", "store"}.issubset(access_kinds)
    ):
        raise TypeError("canonical boundary access kinds lack load/store")

    binary = attr_functions("map_binary")
    unary = attr_functions("map_unary")
    for name, values, examples in (
        ("map_binary", binary, ("add", "mul", "lt")),
        ("map_unary", unary, ("sigmoid",)),
    ):
        if not set(examples).issubset(values):
            raise TypeError(f"canonical {name}.attrs.fn lacks required examples")
    return {
        "operation_catalog": sorted(CANONICAL_J_OPS),
        "map_binary": {
            "op": "map_binary",
            "attrs_field": "fn",
            "examples": ["add", "mul", "lt"],
        },
        "map_unary": {
            "op": "map_unary",
            "attrs_field": "fn",
            "examples": ["sigmoid"],
        },
        "not_operations": ["program_id", "arange", "load", "store"],
        "boundary_access_kinds": [
            item for item in access_kinds if item in {"load", "store"}
        ],
    }


def canonical_wire_profile() -> dict[str, Any]:
    """Return the compact canonical wire authority for all authoring surfaces."""

    schema = scrc_lift_core_json_schema()
    schema_sha256 = hashlib.sha256(
        json.dumps(
            schema,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    root = schema["properties"]
    j = root["j"]["properties"]
    function = j["intent"]["properties"]["intent_function"]
    boundary = j["boundary"]["properties"]
    signature = j["signature"]["properties"]
    pi = root["pi"]["properties"]

    parameter_item = _schema_array_item(
        signature["parameters"], "j.signature.parameters"
    )
    input_item = _schema_array_item(
        function["properties"]["inputs"], "j.intent.intent_function.inputs"
    )
    output_item = _schema_array_item(
        function["properties"]["outputs"], "j.intent.intent_function.outputs"
    )
    return_item = _schema_array_item(
        function["properties"]["returns"], "j.intent.intent_function.returns"
    )
    scope_item = _schema_array_item(j["scopes"], "j.scopes")
    c_slot_item = _schema_array_item(root["c"]["properties"]["slots"], "c.slots")
    reduction_item = _schema_array_item(
        boundary["reductions"], "j.boundary.reductions"
    )
    result_item = _schema_array_item(
        boundary["result_relations"], "j.boundary.result_relations"
    )
    dimension_relation_item = _schema_array_item(
        j["dimension_relations"], "j.dimension_relations"
    )
    domain_schema = boundary["domains"]
    domain_value = domain_schema.get("additionalProperties", {})
    boundary_rows = {
        name: {
            **_schema_object_shape(
                _schema_array_item(property_schema, f"j.boundary.{name}"),
                f"j.boundary.{name}[]",
            ),
            # Field names alone taught providers several legacy row shapes
            # (notably list-valued index bounds and ``stored`` result kinds).
            # This compact node remains a direct projection of the one
            # canonical schema, so it exposes the closed leaf types/enums
            # without creating a parallel authoring grammar.
            "schema": _compact_schema_node(
                _schema_array_item(property_schema, f"j.boundary.{name}")
            ),
        }
        for name, property_schema in boundary.items()
        if isinstance(property_schema, dict)
        and property_schema.get("type") == "array"
    }
    node_ids = j["node_ids"]
    node_id_values = node_ids.get("additionalProperties", {})
    first_turn = schema.get("x-intentir-first-turn-wire", {})
    contract_names = (
        tuple(first_turn.get("operation_contract_names", ()))
        if isinstance(first_turn, Mapping)
        else ()
    )
    if not contract_names:
        contract_names = (
            tuple(first_turn.get("operation_names", ()))
            if isinstance(first_turn, Mapping)
            else ()
        )
    if not contract_names or not all(
        isinstance(name, str) for name in contract_names
    ):
        raise TypeError("canonical schema has no valid operation contract list")
    operation_contracts = schema.get("x-intentir-operation-contracts")
    if not isinstance(operation_contracts, Mapping):
        raise TypeError("canonical schema has no operation contract projection")
    operation_contracts = copy.deepcopy(dict(operation_contracts))
    if set(operation_contracts) != set(contract_names):
        raise TypeError("canonical operation contract projection is incomplete")
    first_turn_names = first_turn.get("operation_names", [])
    if (
        not isinstance(first_turn_names, list)
        or any(name not in operation_contracts for name in first_turn_names)
    ):
        raise TypeError("canonical first-turn operation routing is stale")
    first_turn_rows = first_turn.get("boundary_rows", [])
    if (
        not isinstance(first_turn_rows, list)
        or any(name not in CANONICAL_BOUNDARY_WIRE_FIELDS for name in first_turn_rows)
    ):
        raise TypeError("canonical first-turn boundary routing is stale")
    construction_records = schema.get("x-intentir-construction-records")
    if not isinstance(construction_records, Mapping):
        raise TypeError("canonical schema has no construction-record profile")
    construction_records = copy.deepcopy(dict(construction_records))
    operation_record = construction_records.get("operation")
    output_record = construction_records.get("output")
    scope_record = construction_records.get("scope")
    if not all(
        isinstance(item, dict)
        for item in (operation_record, output_record, scope_record)
    ):
        raise TypeError("canonical construction-record profile is malformed")
    for record_name, fields in CANONICAL_CONSTRUCTION_RECORD_FIELDS.items():
        record = construction_records.get(record_name)
        if not isinstance(record, dict) or record.get("fields") != list(fields):
            raise TypeError(
                f"canonical {record_name} construction-record fields drifted"
            )
    output_record.update(
        {
            "input_tensor": _schema_object_shape(
                input_item, "j.intent.intent_function.inputs[]"
            ),
            "output_tensor": _schema_object_shape(
                output_item, "j.intent.intent_function.outputs[]"
            ),
            "return": _schema_object_shape(
                return_item, "j.intent.intent_function.returns[]"
            ),
            "mapping": boundary_rows.get("output_mappings", {}),
        }
    )
    scope_record["scope"] = _schema_object_shape(scope_item, "j.scopes[]")
    schema_sections = schema.get("x-intentir-schema-sections")
    if not isinstance(schema_sections, Mapping):
        raise TypeError("canonical schema has no schema-section index")
    profile = {
        "protocol": "intentir.scrc.canonical-wire",
        # The mounted MCP/schema lookup validates this receipt against the
        # exact artifact.schema.json bytes.  A stale generated bundle therefore
        # fails before it can become provider-facing context.
        "schema_sha256": schema_sha256,
        "schema": root["schema"].get("enum", [None])[0],
        "pi_schema": pi["schema"].get("const"),
        "root": _schema_object_shape(schema, "root"),
        "j": _schema_object_shape(root["j"], "root.j"),
        "node_ids": {
            "key_pattern": node_ids.get("propertyNames", {}).get("pattern"),
            "value_pattern": node_id_values.get("pattern"),
        },
        "intent_function": _schema_object_shape(
            function, "root.j.intent.intent_function"
        ),
        "return": {
            **_schema_object_shape(
                return_item, "j.intent.intent_function.returns[]"
            ),
            "schema": _compact_schema_node(return_item),
        },
        "boundary": {
            "required": list(root["j"]["properties"]["boundary"]["required"]),
            "collections": {
                name: {
                    "container": property_schema.get("type"),
                    "required": property_schema.get("required", []),
                }
                for name, property_schema in boundary.items()
            },
        },
        "signature": {
            "required": list(parameter_item.get("required", [])),
            "parameter": _schema_object_shape(
                parameter_item, "j.signature.parameters[]"
            ),
            "observation_fields_forbidden": [
                name for name in ("shape", "value") if name not in parameter_item.get("properties", {})
            ],
        },
        "c": _schema_object_shape(root["c"], "root.c"),
        "c_slot": _schema_object_shape(c_slot_item, "c.slots[]"),
        "pi": {
            **_schema_object_shape(root["pi"], "root.pi"),
            # pi is a proof witness rather than a free-form explanation.
            # Preserve its schema-owned XOR and each relation-row shape in the
            # bounded source framework so providers cannot fall back to the
            # older program/access binding wire.
            # π logical_axes is one nesting level deeper than the ordinary
            # first-turn row leaves.  Start this focused projection two levels
            # earlier so its schema-owned axis-object fields remain visible;
            # do not expand the entire artifact schema to obtain them.
            "schema": _compact_schema_node(root["pi"], depth=-2),
        },
        "scope": {
            **_schema_object_shape(scope_item, "j.scopes[]"),
            "schema": _compact_schema_node(scope_item),
        },
        "namespace_order": list(CANONICAL_HANDLE_NAMESPACE_ORDER),
        "source_owned_namespaces": list(CANONICAL_SOURCE_OWNED_HANDLE_PREFIXES),
        "candidate_owned_namespaces": list(CANONICAL_CANDIDATE_OWNED_HANDLE_PREFIXES),
        "namespace_patterns": dict(CANONICAL_HANDLE_NAMESPACE_PATTERNS),
        "semantic_candidate": _semantic_candidate_profile(schema),
        "normalization_classes": _compact_schema_node(
            pi["normalization_classes"].get("items", {})
        ),
        "operation_contracts": operation_contracts,
        "rows": {
            "domain": {
                "key_pattern": domain_schema.get("propertyNames", {}).get("pattern"),
                "value_type": domain_value.get("type"),
            },
            "reduction": _schema_object_shape(
                reduction_item, "j.boundary.reductions[]"
            ),
            "result_relation": _schema_object_shape(
                result_item, "j.boundary.result_relations[]"
            ),
            "dimension_relation": _schema_object_shape(
                dimension_relation_item, "j.dimension_relations[]"
            ),
            "dimension_relation_schema": _compact_schema_node(
                dimension_relation_item
            ),
            "value_dimension_relation_schema": _compact_schema_node(
                _schema_array_item(
                    j["value_dimension_relations"],
                    "j.value_dimension_relations[]",
                )
            ),
        },
        "boundary_rows": boundary_rows,
        "schema_sections": copy.deepcopy(dict(schema_sections)),
        "construction_records": construction_records,
        "first_turn": {
            "boundary_rows": list(first_turn.get("boundary_rows", [])),
            "operation_names": list(first_turn.get("operation_names", [])),
        },
    }
    profile["semantic_operation_contracts"] = {
        name: _provider_operation_projection(
            profile["operation_contracts"][name],
            include_attrs_schema=True,
        )
        for name in profile["first_turn"]["operation_names"]
    }
    semantic_candidate = profile["semantic_candidate"]
    if not isinstance(semantic_candidate, dict):
        raise TypeError("canonical semantic candidate profile is malformed")
    semantic_candidate["operation_namespace"] = _semantic_operation_namespace(
        profile["operation_contracts"],
        profile["boundary_rows"],
    )
    profile["semantic_mcp_write"] = _semantic_mcp_write_protocol(
        semantic_candidate
    )
    profile["semantic_provider_wire"] = _semantic_provider_wire(profile)
    profile["semantic_provider_wire_compact"] = _compact_semantic_provider_wire(
        profile["semantic_provider_wire"]
    )
    return profile


def canonical_wire_bundle() -> dict[str, Any]:
    """Return the single mounted authority for every provider-facing wire.

    The profile and empty artifact are deliberately obtained from the two
    canonical constructors above in one place.  Host-side prompt renderers,
    MCP context lookup, and CandidateCheck therefore consume the same schema
    projection instead of maintaining independent field lists.
    """

    profile = canonical_wire_profile()
    artifact_seed = canonical_artifact_skeleton()
    if set(artifact_seed) != set(profile["root"]["fields"]):
        raise AssertionError("canonical wire seed root drifted from profile")
    return {
        "protocol": "intentir.scrc.canonical-wire-bundle",
        "profile": profile,
        "artifact_seed": artifact_seed,
        "root_layout": profile["root"]["fields"],
        "namespace_order": profile["namespace_order"],
        "schema_sections": copy.deepcopy(profile["schema_sections"]),
        "construction_records": copy.deepcopy(profile["construction_records"]),
    }


def _provider_semantic_operand_inputs(
    construction: Any,
    seed: Mapping[str, Any],
    provider_dimension_aliases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project Source argument positions onto legal provider ``t*`` operands.

    Anonymous Source still contains its positional ABI parameters, whereas a
    semantic J operation may consume only the matching logical ``t*`` input.
    This small projection is derived from the same SourceFacts/allocator seed
    used by construction.  It deliberately publishes the argument ordinal,
    not a physical ``p*`` name, and cannot grant an address or compiler value
    a J-operand role.
    """

    source_facts = getattr(construction, "source_facts", None)
    allocator = getattr(construction, "allocator", None)
    if source_facts is None or allocator is None:
        raise TypeError("semantic operand projection requires canonical construction")
    logical_bindings = allocator.logical_binding_map()
    if not isinstance(logical_bindings, Mapping):
        raise TypeError("canonical allocator logical bindings are unavailable")
    j = seed.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    inputs = function.get("inputs") if isinstance(function, Mapping) else None
    if not isinstance(inputs, list):
        raise ValueError("canonical Source seed lacks logical inputs")
    aliases_by_canonical = provider_dimension_alias_map(
        list(provider_dimension_aliases),
        canonical_dimensions=allocator.declared("x"),
    )
    value_types: dict[str, dict[str, Any]] = {}
    for row in inputs:
        name = row.get("name") if isinstance(row, Mapping) else None
        dtype = row.get("dtype") if isinstance(row, Mapping) else None
        shape = row.get("shape") if isinstance(row, Mapping) else None
        if (
            not isinstance(name, str)
            or re.fullmatch(r"t[0-9]+", name) is None
            or not isinstance(dtype, str)
            or not isinstance(shape, list)
        ):
            raise ValueError("canonical Source logical input is malformed")
        provider_shape: list[int | str] = []
        for extent in shape:
            if isinstance(extent, int) and not isinstance(extent, bool) and extent >= 0:
                provider_shape.append(extent)
            elif isinstance(extent, str) and extent in aliases_by_canonical:
                provider_shape.append(aliases_by_canonical[extent])
            else:
                raise ValueError("canonical Source logical input shape is malformed")
        value_types[name] = {"dtype": dtype, "shape": provider_shape}

    projection: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    arguments = getattr(source_facts, "arguments", ())
    if not isinstance(arguments, (tuple, list)):
        raise TypeError("SourceFacts arguments are unavailable")
    for ordinal, argument in enumerate(arguments):
        name = argument.get("name") if isinstance(argument, Mapping) else None
        binding = logical_bindings.get(name) if isinstance(name, str) else None
        if (
            not isinstance(binding, tuple)
            or len(binding) != 2
            or not all(isinstance(item, str) for item in binding)
        ):
            continue
        provider_ref, role = binding
        if (
            role not in {"tensor", "scalar", "shape_scalar"}
            or provider_ref not in value_types
            or provider_ref in seen_refs
        ):
            continue
        projection.append(
            {
                "source_argument_ordinal": ordinal,
                "provider_ref": provider_ref,
                "role": role,
                "value_type": copy.deepcopy(value_types[provider_ref]),
            }
        )
        seen_refs.add(provider_ref)
    return projection


def _provider_program_domain_requirements(
    construction: Any,
    seed: Mapping[str, Any],
    provider_dimension_aliases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expose only Source-proven domain coverage to the semantic author.

    The production program-domain projector remains the sole owner of
    grid/shape/ledger reasoning.  This wrapper runs that projector over the
    same canonical seed with no provider domains, then publishes only the
    anonymous ``dim*`` coverage obligation—not a Source expression, runtime
    value, compiler spelling, or candidate row.
    """

    from .compiler_observation import (
        _source_domain_extent_symbols,
        materialize_source_program_domains,
    )

    packet = copy.deepcopy(dict(seed))
    j = packet.get("j")
    pi = packet.get("pi")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    if not isinstance(boundary, dict) or not isinstance(pi, dict):
        raise ValueError("canonical Source seed lacks a domain projection surface")
    boundary["domains"] = {}
    for field in ("program_axes", "program_bindings", "iteration_bindings"):
        pi[field] = []
    projection = materialize_source_program_domains(
        packet,
        source_facts=construction.source_facts,
        ledger=construction.transfer_ledger,
    )
    if not isinstance(projection, Mapping) or projection.get("status") != "pass":
        return {
            "status": "unavailable",
            "coverage": "no_provider_domain_submission_until_source_projection_is_closed",
        }
    artifact = projection.get("artifact")
    projected_j = artifact.get("j") if isinstance(artifact, Mapping) else None
    projected_boundary = (
        projected_j.get("boundary") if isinstance(projected_j, Mapping) else None
    )
    if not isinstance(projected_boundary, Mapping):
        raise ValueError("Source domain projection lost the boundary")
    canonical_dimensions = sorted(
        set(_source_domain_extent_symbols(projected_boundary).values())
    )
    alias_by_canonical = provider_dimension_alias_map(
        list(provider_dimension_aliases)
    )
    required_dimensions = [
        alias_by_canonical[dimension]
        for dimension in canonical_dimensions
        if dimension in alias_by_canonical
    ]
    if len(required_dimensions) != len(canonical_dimensions):
        raise ValueError("Source domain projection has no provider dimension alias")
    # The owner checker consumes logical J coordinates, not dimension extents.
    # Publish only coordinate labels already proved by the same Source-domain
    # projector. Do not infer one from dimension order or a runtime extent.
    domain_coordinates: dict[str, str] = {}
    projected_domains = projected_boundary.get("domains")
    if not isinstance(projected_domains, Mapping):
        raise ValueError("Source domain projection lost the domain map")
    for domain_ref, dimension in _source_domain_extent_symbols(projected_boundary).items():
        expression = projected_domains.get(domain_ref)
        match = re.fullmatch(
            r"0 <= (i[0-9]+) and \1 < (x[0-9]+)", expression
        ) if isinstance(expression, str) else None
        if match is None or match.group(2) != dimension:
            raise ValueError("Source domain projection has no exact coordinate label")
        alias = alias_by_canonical.get(dimension)
        if alias is None or alias in domain_coordinates:
            raise ValueError("Source domain projection has ambiguous coordinate label")
        domain_coordinates[alias] = match.group(1)
    if set(domain_coordinates) != set(required_dimensions):
        raise ValueError("Source domain projection lacks a required coordinate")
    return {
        "status": "pass",
        "required_dimensions": required_dimensions,
        "domain_coordinates": domain_coordinates,
        "coverage": "exactly_one_zero_based_rectangular_domain_per_required_dimension",
        # This is an atomic provenance choice, not permission for the host to
        # repair a provider map. With no provider domain rows the production
        # projector can materialize the whole Source-proven launch domain.
        # Once the provider supplies even one row, it owns the complete map
        # and the projector may only reuse exact rows already present.
        "submission_policy": (
            "omit_all_domains_for_atomic_source_projection_or_submit_complete_exact_provider_map"
        ),
        "provider_domain_ownership": (
            "nonempty_provider_map_is_provider_authored; host_reuses_exact_match_only"
        ),
    }


def _attach_source_proven_index_requirements(
    requirements: dict[str, Any], domain_requirements: Mapping[str, Any],
    provider_dimension_aliases: Sequence[Mapping[str, Any]],
) -> None:
    """Expose only coordinate expressions already committed by owner outlines."""

    if domain_requirements.get("status") != "pass":
        return
    coordinates = domain_requirements.get("domain_coordinates")
    accesses = requirements["semantic_required"]["accesses"]["rows"]
    if not isinstance(coordinates, Mapping) or not isinstance(accesses, list):
        return
    aliases = provider_dimension_alias_map(list(provider_dimension_aliases))
    by_token: dict[str, dict[str, Any]] = {}
    for row in accesses:
        auth = row.get("source_semantics", {}).get("semantic_authorizations", {})
        view = auth.get("access_view") if isinstance(auth, Mapping) else None
        token = view.get("token") if isinstance(view, Mapping) else None
        if isinstance(token, str):
            if token in by_token:
                raise ValueError("Source access view token is ambiguous")
            by_token[token] = row

    def expression(shape: Any, axes: Any) -> str | None:
        if not isinstance(shape, list) or not isinstance(axes, list) or len(shape) != len(axes):
            return None
        parts: list[str] = []
        for ordinal, (extent, axis) in enumerate(zip(shape, axes, strict=True)):
            if axis == f"axis{ordinal}" and isinstance(extent, str):
                coordinate = coordinates.get(aliases.get(extent))
                if not isinstance(coordinate, str) or re.fullmatch(r"i[0-9]+", coordinate) is None:
                    return None
                parts.append(coordinate)
            elif type(axis) is int and type(extent) is int and 0 <= axis < extent:
                parts.append(str(axis))
            else:
                return None
        return "[" + ", ".join(parts) + "]"

    def attach(row: dict[str, Any], shape: Any, axes: Any) -> None:
        index = expression(shape, axes)
        if index is None:
            return
        previous = row.get("index_expression_requirement")
        if previous is not None and previous != index:
            raise ValueError("Source owner outlines disagree on an access index")
        row["index_expression_requirement"] = index

    for row in accesses:
        auth = row.get("source_semantics", {}).get("semantic_authorizations", {})
        owner = auth.get("owner") if isinstance(auth, Mapping) else None
        outline = owner.get("authoring_outline") if isinstance(owner, Mapping) else None
        if not isinstance(outline, Mapping):
            continue
        store = outline.get("store")
        if isinstance(store, Mapping):
            attach(row, store.get("shape"), store.get("coordinates"))
        for graph_row in outline.get("graph", ()):
            if not isinstance(graph_row, (list, tuple)) or len(graph_row) != 5 or graph_row[0] != "load":
                continue
            load = graph_row[3]
            target = by_token.get(load.get("occurrence")) if isinstance(load, Mapping) else None
            if target is not None:
                attach(target, load.get("shape"), load.get("coordinates"))


def canonical_source_wire_bundle(construction: Any) -> dict[str, Any]:
    """Project the same wire authority for one real Source/Harness ingress.

    A mounted MCP worker reads this exact Source-bound bundle verbatim.  The
    direct Responses backend cannot perform that local read, so it receives
    the same schema/profile projection plus the same Source-bound mechanical
    seed.  Only closed ABI/C/tensor facts are populated; the semantic
    construction requirements are derived from the same ledger, while
    operation meaning and the optional J-proven pi witness remain
    provider-authored.
    """

    profile = canonical_wire_profile()
    seed = canonical_source_wire_seed(construction)
    if set(seed) != set(profile["root"]["fields"]):
        raise AssertionError("Source canonical seed root differs from wire profile")
    allocator = getattr(construction, "allocator", None)
    declared_dimensions = (
        allocator.declared("x")
        if allocator is not None and hasattr(allocator, "declared")
        else ()
    )
    if not isinstance(declared_dimensions, (tuple, list)) or any(
        not isinstance(item, str) or re.fullmatch(r"x[0-9]+", item) is None
        for item in declared_dimensions
    ):
        raise ValueError("Source construction lacks canonical dimension declarations")
    provider_dimension_aliases = [
        {
            "local": f"dim{index}",
            "ordinal": index,
            "source_role": "Source-declared logical dimension",
        }
        for index, _canonical in enumerate(declared_dimensions)
    ]
    provider_dimension_alias_map(
        provider_dimension_aliases,
        canonical_dimensions=declared_dimensions,
    )
    transfer_projection = _logical_transfer_projection(construction)
    # Keep the Source-owned output order/type facts, but publish their shape
    # through the same provider ``dim*`` aliases used by operation result
    # types and access expressions.  Canonical ``x*`` handles remain solely
    # in the host seed and are restored by the one ingress rewrite.
    output_construction = _provider_output_construction(
        _source_output_construction(seed),
        provider_dimension_aliases,
    )
    semantic_construction_requirements = _semantic_construction_requirements(
        construction,
        output_construction,
        transfer_projection=transfer_projection,
        provider_dimension_aliases=provider_dimension_aliases,
    )
    semantic_operand_inputs = _provider_semantic_operand_inputs(
        construction,
        seed,
        provider_dimension_aliases,
    )
    program_domain_requirements = _provider_program_domain_requirements(
        construction,
        seed,
        provider_dimension_aliases,
    )
    _attach_source_proven_index_requirements(
        semantic_construction_requirements,
        program_domain_requirements,
        provider_dimension_aliases,
    )
    semantic_required = semantic_construction_requirements.get(
        "semantic_required"
    )
    obligation_accesses = (
        semantic_required.get("accesses")
        if isinstance(semantic_required, Mapping)
        else None
    )
    obligation_outputs = (
        semantic_required.get("outputs")
        if isinstance(semantic_required, Mapping)
        else None
    )
    obligation_access_rows = (
        obligation_accesses.get("rows")
        if isinstance(obligation_accesses, Mapping)
        else None
    )
    obligation_output_rows = (
        obligation_outputs.get("rows")
        if isinstance(obligation_outputs, Mapping)
        else None
    )
    if not isinstance(obligation_access_rows, list) or not isinstance(
        obligation_output_rows, list
    ):
        raise ValueError("Source semantic obligation inputs are malformed")
    # This table is the only identity/state contract for Source-derived
    # semantic work.  Existing access/output rows remain the detailed Source
    # facts; MCP, CandidateCheck, and Relift refer back to them through stable
    # obligation ids instead of rebuilding a second rule-to-locator mapping.
    semantic_construction_requirements["obligations"] = (
        build_source_semantic_obligations(
            source_facts_digest=construction.source_facts.digest,
            access_rows=obligation_access_rows,
            output_rows=obligation_output_rows,
            program_domain_requirements=program_domain_requirements,
        )
    )
    validate_source_semantic_obligations(
        semantic_construction_requirements["obligations"],
        construction.source_facts.digest,
    )
    # The exact index expression is already proved from the Source owner
    # outline and declared program-domain coordinates. Carry that same value
    # on the canonical transfer row so MCP readers and the mounted writer
    # consume one projection instead of leaving it stranded in the parallel
    # requirements table.
    access_requirements = semantic_construction_requirements.get(
        "semantic_required", {}
    ).get("accesses", {}).get("rows", ())
    projected_accesses = transfer_projection.get("occurrences", ())
    if (
        not isinstance(access_requirements, list)
        or not isinstance(projected_accesses, list)
        or len(access_requirements) != len(projected_accesses)
    ):
        raise ValueError("Source access index projection is malformed")
    for ordinal, (requirement, projected) in enumerate(
        zip(access_requirements, projected_accesses, strict=True)
    ):
        if (
            not isinstance(requirement, Mapping)
            or not isinstance(projected, dict)
            or requirement.get("ordinal") != ordinal
            or projected.get("ordinal") != ordinal
        ):
            raise ValueError("Source access index projection ordinal is malformed")
        index_requirement = requirement.get("index_expression_requirement")
        if index_requirement is not None:
            if not isinstance(index_requirement, str) or not index_requirement:
                raise ValueError("Source access index expression requirement is malformed")
            projected["index_expression_requirement"] = index_requirement
    return {
        "protocol": "intentir.scrc.canonical-source-wire-bundle",
        "profile": profile,
        "artifact_seed": seed,
        "root_layout": profile["root"]["fields"],
        "namespace_order": profile["namespace_order"],
        "source_owned_namespaces": profile["source_owned_namespaces"],
        "candidate_owned_namespaces": profile["candidate_owned_namespaces"],
        "construction_records": copy.deepcopy(profile["construction_records"]),
        "construction_digest": construction.authority_digest,
        "source_facts_digest": construction.source_facts.digest,
        "source_slice_digest": construction.source_facts.source_slice_digest,
        "source_transfer_ledger": construction.transfer_ledger.to_json(),
        "host_launch_contract": source_host_launch_contract_receipt(
            construction.source_facts,
            construction.transfer_ledger,
        ),
        "semantic_transfer_projection": transfer_projection,
        "semantic_construction_requirements": semantic_construction_requirements,
        # The only provider-facing Source argument -> J operand bridge.  The
        # direct Responses transport mounts it here; MCP derives its entry
        # aliases from exactly this same projection.  Physical ABI names never
        # cross this authoring boundary.
        "semantic_operand_inputs": semantic_operand_inputs,
        "program_domain_requirements": program_domain_requirements,
        "output_construction": output_construction,
        # Provider submissions use dim* aliases.  The host maps these aliases
        # to the dense canonical x* declarations in one ingress transaction;
        # x* never appears on the provider submission surface.
        "provider_dimension_aliases": provider_dimension_aliases,
        # A π address row is admissible only when the Source ledger has
        # already proved both its p* owner and logical tensor axis.  The
        # current ledger deliberately carries no guessed pair, so the seed is
        # empty rather than inviting a provider to fabricate an incomplete
        # address_recomputed relation.
        "pi_access_binding_seed": [],
    }


def render_canonical_wire_contract() -> str:
    """Render the one compact semantic submission contract for Lift.

    The strict artifact schema is deliberately not pasted into a provider turn:
    its mandatory canonical handles and sibling rows exist only after host
    construction.  This renderer, MCP source-brief, and direct Responses use
    the same schema-derived semantic projection.
    """

    profile = canonical_wire_bundle()["profile"]
    projection = profile.get("semantic_provider_wire")
    if not isinstance(projection, Mapping):
        raise TypeError("canonical provider contract lacks semantic wire")
    return (
        "CANONICAL PROVIDER WIRE (schema-derived semantic submission surface)\n"
        + json.dumps(
            projection,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        "The candidate_template and operation_namespace above are the sole "
        "authoring layout and J-name authority. This is the provider submission "
        "surface, not the completed artifact. "
        "The operation rows are a bounded first-turn summary; use one exact "
        "schema_lookup(op:NAME) leaf when an attribute contract is missing. "
        "Fields marked host_allocated_fields or host_derived must be omitted. "
        "In particular, semantic_owner_authorization belongs only to the "
        "completed host artifact: submit node_ref, never an owner token or null. "
        "Provider submissions may use only Source t* handles and provider-local "
        "labels. Never submit canonical n*, v*, x*, p*, c*, a*, or e* handles; use "
        "op*/r*/dim*/access* local labels and let the host rewrite them. Canonical "
        "node_ref/output-producer references, scope_id, access_id, and effect_id "
        "are host-owned. A required access node_ref instead names a provider-local "
        "operation id and is rewritten by that same allocator. The host alone "
        "derives schema/C, ABI, "
        "canonical n*/v*/a*/e*/q* handles, "
        "scope membership, returns, output attachments, typed relations, and "
        "Source-proven π bindings; the strict completed root remains "
        "{schema,j,c,pi}. Provider π input is limited to J-proven "
        "normalization_classes. For every authored gather/scatter, use its "
        "exact schema leaf with data and indices operands plus an integer axis; "
        "do not author index_bounds, because the host derives exactly one row "
        "only from that closed final operation. Every authored scatter also "
        "requires one exact provider-authored boundary.updates row: it must "
        "use that scatter's local operation id and exactly match its base, "
        "indices, values, axis, reduce-derived combiner, duplicate policy, "
        "and order. The host only rewrites the local operation id to n*; it "
        "does not infer or fill update semantics."
    )


def render_canonical_preflight() -> str:
    """Render the compact preflight shipped to every Lift workspace."""

    profile = canonical_wire_bundle()["profile"]
    reduce_contract = profile["operation_contracts"].get("reduce", {})
    reduce_attrs = reduce_contract.get("attrs", {})
    reduce_attr_fields = list(reduce_attrs.get("properties", {}))
    reduce_attr_required = reduce_contract.get("attrs_required", [])
    normalization_item = profile["normalization_classes"]
    normalization_fields = list(normalization_item.get("properties", {}))
    skeleton = render_canonical_semantic_candidate_skeleton()
    operation_namespace = render_canonical_semantic_operation_namespace()
    return (
        "# IntentIR candidate preflight\n\n"
        "The mounted `schema/artifact.schema.json` and CandidateCheck are the "
        "only admission authorities. This checklist is derived from that "
        "same canonical wire profile and never constructs semantic fields.\n\n"
        "- Start every submission from this exact semantic skeleton (not a "
        "completed artifact, and not two concatenated objects): `"
        + skeleton
        + "`.\n"
        "- Use this exact schema-derived J operation namespace: `"
        + operation_namespace
        + "`.\n"
        f"- Domains are d* -> `{profile['rows']['domain']['value_type']}` "
        "ContractExpr strings; never emit a dense-domain object.\n"
        f"- Dimension relation rows have exactly `{profile['rows']['dimension_relation']['fields']}`; "
        "provider expr.arg/rhs use local dim* labels and the host maps them to "
        "declared x* handles. Constants use only the schema-declared closed "
        "op/value form. Never submit canonical x*, p*, or t* in this row.\n"
        f"- A reduction observation row has only "
        f"`{profile['rows']['reduction']['fields']}`; reduce attrs are the "
        f"schema-derived closed fields `{reduce_attr_fields}` with required "
        f"`{reduce_attr_required}`.\n"
        f"- A result relation has `{profile['rows']['result_relation']['fields']}` "
        "and uses `output`, never `result`.\n"
        f"- A signature row allows only `{profile['signature']['parameter']['fields']}`; "
        "never copy ABI observation `shape` or `value` into it.\n"
        f"- A normalization-class row allows only `{normalization_fields}`; "
        "use only the schema-derived name/scope pairs.\n"
        "- Provider π may contain only normalization_classes. The host derives "
        "program_axes, program_bindings, iteration_bindings, access and "
        "materialization bindings from SourceFacts, the transfer ledger, J, "
        "and C ownership; never submit those rows.\n"
        "- Provider semantic input uses only Source t* plus local op*/r*/dim*/access* "
        "labels. Do not emit canonical n*/v*/x*/p*/c*/a*/e* or canonical node "
        "references. A required access `node_ref` is the one exception: it must "
        "name a provider-local operation `id`, which the host maps to n*. Do not "
        "emit scope_id, access_id, or effect_id. The host maps dim* to Source-declared x* "
        "and allocates all canonical namespaces. C slots omit subject_ref/axis_ref.\n"
        "- MCP construction uses only begin_candidate, schema-derived "
        "declare_* calls, and finalize_candidate. Free candidate JSON and "
        "JSON Patch/patch/patches are invalid; host construction creates "
        "`{schema,j,c,pi}`.\n"
        "- CandidateCheck remains fail-closed; TypeCheck, BindSource, Replay, "
        "Gate1 and Gate2 remain external authorities.\n"
    )


def render_canonical_operation_reference() -> str:
    """Render the provider operation lookup from the canonical wire profile."""

    artifact_schema = scrc_lift_core_json_schema()
    operation_schema = artifact_schema["$defs"]["intent_op"]
    ordinary, structured = _operation_catalog(operation_schema)
    provider_operations = canonical_wire_bundle()["profile"][
        "semantic_operation_contracts"
    ]
    contracts = json.dumps(
        provider_operations,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    )
    lines = [
        "# IntentIR operation reference (schema-derived)",
        "The mounted canonical wire profile and CandidateCheck are authoritative. "
        "This file is the provider semantic projection; query schema_lookup "
        "with exactly one op:NAME for an omitted operation leaf.",
        "",
        "ordinary operations: " + ", ".join(ordinary),
        "structured operations: " + ", ".join(structured),
        "",
        "Every ordinary operation uses provider-local id/result labels, Source "
        "t* or prior local results as operands, and schema-derived op/attrs/type. "
        "Source load/store are boundary access kinds; API functions such as "
        "sigmoid are map_unary attrs.fn, not operation names. "
        "Never emit canonical n*/v*/x*/p*/c*/a*/e* handles; the host allocates them. "
        "Do not copy Source/API names into op or attrs.",
        "",
        "Selected operation contracts (schema-derived; query the mounted "
        "schema for any omitted leaf):",
        contracts,
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "canonical_artifact_skeleton",
    "canonical_source_wire_bundle",
    "canonical_source_wire_seed",
    "canonical_semantic_candidate_contract",
    "canonical_semantic_candidate_template",
    "canonical_semantic_mcp_write_protocol",
    "canonical_semantic_operation_namespace",
    "canonical_wire_bundle",
    "canonical_wire_profile",
    "CANONICAL_SCHEMA_SECTION_PATHS",
    "render_canonical_operation_reference",
    "render_canonical_preflight",
    "render_canonical_semantic_candidate_skeleton",
    "render_canonical_semantic_candidate_layout_summary",
    "render_canonical_semantic_operation_namespace",
    "render_canonical_semantic_operation_namespace_summary",
    "render_canonical_wire_contract",
    "render_boundary_access_wire_grammar",
    "render_gate1_artifact_envelope_grammar",
    "render_semantic_archive_wire_grammar",
    "scrc_lift_core_json_schema",
    "scrc_lift_core_model_json_schema",
    "scrc_lift_core_text_format",
    "scrc_semantic_candidate_text_format",
    "unique_access_guard_projection_axes",
]
