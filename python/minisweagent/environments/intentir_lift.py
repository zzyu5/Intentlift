"""Fail-closed filesystem sandbox for one IntentIR Lift invocation."""

from __future__ import annotations

import copy
import json
import hashlib
import os
import re
import site
import stat
import shutil
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel

from intentir.ir.canonical_construction import (
    computed_guard_projection_supported,
    is_canonical_ordinal,
    semantic_operation_wire_rows,
)
from intentir.ir.source_projection_contracts import (
    PROVIDER_DIMENSION_EXTENT_RULE,
    is_provider_dimension_extent,
    provider_dimension_alias_map,
    provider_shape_from_canonical,
    semantic_obligation_diagnostic,
    validate_source_semantic_obligations,
)
from intentir.ir.intent_op_contract import (
    STRUCTURED_INTENT_OPS,
    _broadcast_projection,
    is_closed_boolean_dtype,
    operation_attr_contract,
    operation_operands_match_arity,
)
from minisweagent.exceptions import (
    CandidateRejected,
    LiftPreparationFailed,
    Submitted,
)
from intentir.extraction.lifters.scrc_schema import (
    canonical_semantic_mcp_write_protocol,
    unique_access_guard_projection_axes,
)
from intentir.extraction.lifters.source_store_partition import (
    validate_source_store_partition_assembly,
)
from minisweagent.mcp_lift import (
    IntentIRLiftMCPServer,
    MCP_CONTEXT_ARGUMENTS,
    MCP_CONTEXT_VIEWS,
    MCP_FIXED_SUMMARY_VIEWS,
    MCP_SEMANTIC_WRITE_TOOL_NAMES,
    MCP_SCHEMA_LOOKUP_ITEM_CHARS,
    MCP_SELECTOR_CHARS,
    MAX_MCP_SCHEMA_LOOKUP_ITEMS,
    MAX_MCP_PATH_CHARS,
    MAX_MCP_QUERY_CHARS,
    MCPRequestError,
    mcp_resource_uris_for_views,
    mcp_resource_count_upper,
)
INTENTIR_LIFT_SUBMIT = "INTENTIR_LIFT_SUBMIT"
CANDIDATE_CHECK_PROTOCOL = "intentir.scrc.candidate-check.v1"
MULTI_ACTION_PROTOCOL = "intentir.agent-multi-action.v1"
MCP_SEMANTIC_FINALIZE_PROTOCOL = "intentir.mcp-semantic-finalize"
_LIFT_READ_ONLY_CONTEXT_COMMANDS = MCP_CONTEXT_VIEWS
_SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS = frozenset(
    {
        # The first brief remains intentionally small. A fresh owner may read
        # one exact fact family only when the source brief cannot settle its
        # current construction decision; this avoids re-embedding Source,
        # ABI, TTIR and transfer facts on every Lift.
        "source",
        "source_obligations",
        "semantic_transfer",
        "harness",
        "harness_abi",
        "source_ttir",
        "source_ttir_outline",
        "source_ttir_def_use",
    }
)
_SEMANTIC_ENTRY_CONTEXT_VIEWS = (
    "source_brief",
    *_SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS,
)
_SOURCE_OBLIGATION_SECTIONS = frozenset(
    {
        "scopes",
        "program_axes",
        "index_construction",
        "mask_uses",
        "fills",
        "casts",
        "memory_accesses",
        "control",
        "helper_calls",
        "expression_facts",
    }
)
MAX_CANDIDATE_BYTES = 4 * 1024 * 1024
MAX_ACTION_OUTPUT_BYTES = 1024 * 1024
MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_WORKSPACE_ENTRIES = 2048
MAX_WORKSPACE_DEPTH = 64
MAX_CANDIDATE_CHECK_FIELD_CHARS = 4096
MAX_CANDIDATE_CHECK_OBSERVED_ITEMS = 20
MAX_CANDIDATE_CHECK_RESULT_BYTES = 128 * 1024
MAX_STARTUP_CONTEXT_BYTES = 64 * 1024
_SANDBOX_START_MARKER = b"__INTENTIR_LIFT_SANDBOX_STARTED__\n"
# Graph correction uses the owner's ordinary step/stagnation budget.  The old
# six-failure cap was based on a retired 16-turn budget and terminated changed,
# provider-correctable graph declarations before the current 32-turn owner
# could reach accesses, outputs, or finalize.  Identical resubmissions still
# fail immediately; no graph failure becomes a candidate or repair base.
# Access declarations have the same bounded pre-candidate lifecycle as an
# operation graph.  Without this boundary a provider can keep resubmitting an
# empty or prefix array after the begin receipt has already supplied every
# Source-order slot, exhausting the physical-call cap without ever reaching
# finalize.  This is an authoring-preparation limit only: it neither accepts a
# partial access array nor changes CandidateCheck.
_MAX_ACCESS_DECLARATION_FAILURES = 4
_PROVIDER_CORRECTABLE_ACCESS_DECLARATION_DIAGNOSTICS = frozenset(
    {
        "accesses_must_be_json_array",
        "access_declaration_must_cover_each_source_occurrence",
        "access_declaration_rows_must_be_json_objects",
        "access_declaration_must_follow_source_ordinal",
        "access_declaration_is_missing_required_source_slot_field",
        "access_declaration_must_not_use_id_alias",
        "access_declaration_provider_predicate_must_be_local_label",
        "access_declaration_provider_predicate_must_select_compatible_boolean_result",
        "access_declaration_provider_index_expression_must_be_local_expression",
        "access_declaration_provider_node_ref_must_be_local_label",
        "access_declaration_provider_node_ref_must_select_direct_consumer",
        "access_declaration_effect_node_owner_must_match_access_owner",
        "access_declaration_must_match_source_wire_template",
        "access_declaration_must_match_source_index_requirement",
    }
)


def _safe_access_host_failure_rule(exc: BaseException) -> str:
    """Classify access-host failures without retaining declaration text."""

    if isinstance(exc, json.JSONDecodeError):
        return "access_declaration_invalid_json"
    detail = str(exc)
    if detail.startswith("duplicate JSON key "):
        return "access_declaration_duplicate_json_key"
    if detail.startswith("invalid JSON constant:"):
        return "access_declaration_non_json_constant"
    if detail in {
        "canonical_source_access_wire_template_is_malformed",
        "canonical MCP declaration argument is unavailable",
    }:
        return detail
    if isinstance(exc, KeyError):
        return "access_declaration_host_key_error"
    if isinstance(exc, TypeError):
        return "access_declaration_host_type_error"
    if isinstance(exc, ValueError):
        return "access_declaration_host_value_error"
    return "access_declaration_host_validation_unclassified"


def _safe_access_host_origin(exc: BaseException) -> str | None:
    """Return only a repository-local frame identity, never a full path."""

    traceback_node = exc.__traceback__
    while traceback_node is not None and traceback_node.tb_next is not None:
        traceback_node = traceback_node.tb_next
    if traceback_node is None:
        return None
    frame = traceback_node.tb_frame
    filename = Path(frame.f_code.co_filename).name
    function = frame.f_code.co_name
    if (
        filename != "intentir_lift.py"
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function) is None
    ):
        return None
    return f"{filename}:{function}:{traceback_node.tb_lineno}"


# Output selection is provider-owned and can be corrected only by choosing a
# different already-authored, function-visible value.  Bound that preparation
# loop exactly like graph/access authoring: a bad mapping never becomes a
# candidate or repair base, and repeated writes must not consume the global
# owner step limit.
_MAX_OUTPUT_DECLARATION_FAILURES = 4
_PROVIDER_CORRECTABLE_OUTPUT_DECLARATION_DIAGNOSTICS = frozenset(
    {
        "output_mappings_must_be_json_array",
        "output_declaration_must_cover_each_source_output",
        "output_declaration_rows_must_be_json_objects",
        "output_declaration_is_missing_required_source_slot_field",
        "output_declaration_must_contain_only_value",
        "output_declaration_requires_declared_operation_graph",
        "output_declaration_value_must_reference_source_input_or_local_result",
        "output_declaration_value_must_be_function_visible",
        "output_declaration_value_type_must_match_source_output",
    }
)
# Boundary collections are the last semantic write before finalization.  A
# missing graph-required broadcast used to consume the one CandidateCheck
# attempt, while an unchanged empty object could still consume all remaining
# turns.  Keep this pre-candidate correction finite and receipt-only.
_MAX_BOUNDARY_FIELDS_DECLARATION_FAILURES = 4
# A dynamic MCP phase exposes only one legal lifecycle edge, but an
# application-level error still returns a syntactically valid tool result. It
# is therefore not a ``FormatError``, and changing irrelevant request fields
# also keeps it from tripping exact action/output stagnation. Bound that
# pre-candidate loop by the phase edge and a digest of the safe error rule.
# Two occurrences of the same rule prove that another request at the same edge
# cannot reveal new construction facts; four distinct rules are the absolute
# per-edge preparation ceiling. Neither boundary creates or accepts a draft.
_MAX_REPEATED_MCP_PHASE_RULE_FAILURES = 2
_MAX_MCP_PHASE_APPLICATION_FAILURES = 4
# A graph declaration which never crosses the phase boundary cannot make
# progress merely because its JSON bytes change. Keep a bounded correction
# window and stop with a safe no-progress receipt before the 32-step owner
# limit is consumed.
_MAX_OPERATION_GRAPH_DECLARATION_FAILURES = 8
# Two occurrences of one graph rule at the same phase are enough to prove that
# changing provider JSON has not crossed the phase boundary.  Stop before a
# third model turn can turn a deterministic authoring rejection into an
# unrelated provider/SSE timeout; this is not the Agent step limit.
_MAX_SEMANTIC_NO_PROGRESS_RULE_FAILURES = 2
_PROVIDER_CORRECTABLE_OPERATION_GRAPH_DIAGNOSTICS = frozenset(
    {
        "operation_graph_must_be_valid_json",
        "operation_graph_must_not_contain_duplicate_json_keys",
        "operation_graph_must_use_finite_json_numbers",
        "operation_graph_must_be_nonempty_json_array",
        "operation_graph_rows_must_be_json_objects",
        "operation_graph_lexical_wire_is_malformed",
        "operation_row_must_include_canonical_op_field",
        "operation_must_use_only_canonical_leaf_fields",
        "operation_must_include_required_leaf_fields",
        "operation_op_must_be_canonical_j_name",
        "operation_id_must_be_provider_local_label",
        "operation_result_must_be_provider_local_label",
        "operation_operands_must_be_json_array",
        "operation_operands_must_match_canonical_leaf_arity",
        "operation_operand_must_reference_source_tensor_or_prior_local_result",
        "operation_result_must_be_new_provider_local_label",
        "operation_attrs_must_be_json_object",
        "operation_attrs_must_use_canonical_leaf_fields",
        "operation_attrs_must_include_required_leaf_fields",
        "operation_attr_value_must_match_canonical_leaf_contract",
        "operation_result_must_have_complete_value_type",
        "operation_type_shape_must_use_valid_provider_extent",
        "operation_type_shape_must_use_provider_dimension_alias",
        "operation_result_type_shape_must_match_operands",
        "operation_iota_result_type_must_match_attrs",
        "operation_nontrailing_broadcast_requires_local_id",
        "operation_graph_must_expose_source_output_assembly",
        "operation_graph_must_expose_each_source_output_type",
        "operation_graph_must_expose_compatible_access_predicate_type",
        "operation_graph_must_expose_access_consumer_id",
        "operation_graph_must_cover_source_operation_families",
        "operation_graph_must_cover_source_operation_attributes",
    }
)


def _operation_attr_value_matches_schema(value: Any, schema: Any) -> bool:
    """Check the bounded JSON-Schema leaf forms used by operation attrs.

    This is intentionally narrower than full artifact-schema validation.  It
    rejects only a value which directly contradicts the one canonical leaf
    contract already used by schema lookup and TypeCheck; cross-operation
    semantics remain with canonical construction and CandidateCheck.
    """

    if not isinstance(schema, Mapping):
        return False

    def strict_equal(left: Any, right: Any) -> bool:
        # Python considers True == 1 and False == 0. JSON Schema does not;
        # keep enum/const checks type-injective at the MCP boundary.
        if isinstance(left, bool) or isinstance(right, bool):
            return isinstance(left, bool) and isinstance(right, bool) and left == right
        if isinstance(left, (int, float)) and not isinstance(left, bool):
            if isinstance(right, (int, float)) and not isinstance(right, bool):
                return left == right
            return False
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            return False
        if isinstance(left, list) or isinstance(right, list):
            return (
                isinstance(left, list)
                and isinstance(right, list)
                and len(left) == len(right)
                and all(strict_equal(a, b) for a, b in zip(left, right))
            )
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            return (
                isinstance(left, Mapping)
                and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(strict_equal(left[key], right[key]) for key in left)
            )
        return type(left) is type(right) and left == right

    if "not" in schema and isinstance(schema.get("not"), Mapping):
        if _operation_attr_value_matches_schema(value, schema["not"]):
            return False
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and any(
        not _operation_attr_value_matches_schema(value, item)
        for item in all_of
    ):
        return False
    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and sum(
        _operation_attr_value_matches_schema(value, item) for item in one_of
    ) != 1:
        return False
    if "const" in schema:
        return strict_equal(value, schema["const"])
    enum = schema.get("enum")
    if isinstance(enum, list):
        return any(strict_equal(value, item) for item in enum)
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        return bool(alternatives) and any(
            _operation_attr_value_matches_schema(value, item)
            for item in alternatives
        )
    reference = schema.get("$ref")
    if reference == "#/$defs/const_literal":
        if isinstance(value, bool) or (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            return True
        if isinstance(value, str):
            return value in {"-inf", "inf", "nan"}
        return isinstance(value, list) and all(
            _operation_attr_value_matches_schema(
                item,
                {"$ref": "#/$defs/const_literal"},
            )
            for item in value
        )
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        return any(
            _operation_attr_value_matches_schema(value, {"type": item})
            for item in expected_type
            if isinstance(item, str)
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
        if valid and isinstance(schema.get("minimum"), (int, float)):
            valid = value >= schema["minimum"]
        if valid and isinstance(schema.get("maximum"), (int, float)):
            valid = value <= schema["maximum"]
        if valid and isinstance(schema.get("exclusiveMinimum"), (int, float)):
            valid = value > schema["exclusiveMinimum"]
        if valid and isinstance(schema.get("exclusiveMaximum"), (int, float)):
            valid = value < schema["exclusiveMaximum"]
        return valid
    if expected_type == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if valid and isinstance(schema.get("minimum"), (int, float)):
            valid = value >= schema["minimum"]
        if valid and isinstance(schema.get("maximum"), (int, float)):
            valid = value <= schema["maximum"]
        if valid and isinstance(schema.get("exclusiveMinimum"), (int, float)):
            valid = value > schema["exclusiveMinimum"]
        if valid and isinstance(schema.get("exclusiveMaximum"), (int, float)):
            valid = value < schema["exclusiveMaximum"]
        return valid
    if expected_type == "string":
        if not isinstance(value, str):
            return False
        minimum = schema.get("minLength")
        if isinstance(minimum, int) and len(value) < minimum:
            return False
        maximum = schema.get("maxLength")
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    return False
            except re.error:
                return False
        return True
    if expected_type == "array":
        if not isinstance(value, list):
            return False
        minimum = schema.get("minItems")
        if isinstance(minimum, int) and len(value) < minimum:
            return False
        maximum = schema.get("maxItems")
        if isinstance(maximum, int) and len(value) > maximum:
            return False
        if schema.get("uniqueItems") is True and any(
            strict_equal(left, right)
            for index, left in enumerate(value)
            for right in value[index + 1 :]
        ):
            return False
        item_schema = schema.get("items")
        if item_schema is False:
            return not value
        return (
            not isinstance(item_schema, Mapping)
            or all(_operation_attr_value_matches_schema(item, item_schema) for item in value)
        )
    if expected_type == "object":
        if not isinstance(value, Mapping):
            return False
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            required = schema.get("required", ())
            if isinstance(required, list) and any(name not in value for name in required):
                return False
            if schema.get("additionalProperties") is False and set(value) - set(properties):
                return False
            if any(
                name in properties
                and not _operation_attr_value_matches_schema(value[name], properties[name])
                for name in value
            ):
                return False
        return True
    # A schema leaf outside this bounded canonical projection is not safe to
    # infer at ingress. Reject it rather than silently turning an unknown leaf
    # into an accepted provider declaration.
    return False


def _validate_mounted_source_store_projection(
    access_rows: Sequence[Any],
    output_rows: Sequence[Any],
    output_construction: Sequence[Any],
    dimension_aliases: Mapping[str, str],
) -> None:
    """Keep begin-candidate slots bound to the mounted Source access rows.

    The context reader performs the same check before exposing ``source-brief``.
    Repeat it at the MCP write boundary because the environment consumes the
    mounted bundle directly when constructing ``required_slots``.  This is a
    consistency check only: it never creates an owner, producer, coordinate,
    partition, or logical output binding.
    """

    if len(output_rows) != len(output_construction):
        raise ValueError("canonical Source output projection cardinality is inconsistent")
    for output_ordinal, (required_row, output_row) in enumerate(
        zip(output_rows, output_construction, strict=True)
    ):
        if not isinstance(required_row, Mapping) or not isinstance(output_row, Mapping):
            raise ValueError("canonical Source output store projection is malformed")
        output_name = output_row.get("output")
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("canonical Source output store projection lacks output")
        stores = required_row.get("source_store_requirements")
        partition = required_row.get("source_store_partition_requirement")
        if stores is None:
            if partition is not None:
                raise ValueError(
                    "canonical Source partition locator lacks store requirements"
                )
            continue
        if not isinstance(stores, list) or not stores:
            raise ValueError(
                f"canonical Source store requirements are malformed at output {output_ordinal}"
            )
        expected: list[dict[str, Any]] = []
        for access_row in access_rows:
            if not isinstance(access_row, Mapping):
                raise ValueError("canonical Source access requirements are malformed")
            access_ordinal = access_row.get("ordinal")
            if not is_canonical_ordinal(access_ordinal):
                raise ValueError("canonical Source access ordinal is malformed")
            source = access_row.get("source_semantics")
            if not isinstance(source, Mapping):
                continue
            if source.get("kind") not in {"store", "update"}:
                continue
            if source.get("tensor") != output_name:
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
            if not isinstance(family, str) or not family or not isinstance(coordinates, list):
                continue
            expected.append(
                {
                    "ordinal": access_ordinal,
                    "kind": source.get("kind"),
                    "operation_family": family,
                    "coordinates": copy.deepcopy(coordinates),
                }
            )
        if stores != expected:
            raise ValueError(
                f"canonical Source store requirements are not access-bound at output {output_ordinal}"
            )
        if len(stores) > 1:
            if (
                not isinstance(partition, Mapping)
                or set(partition)
                not in (
                    {"output", "store_ordinals", "store_coordinates", "relation"},
                    {
                        "output",
                        "store_ordinals",
                        "store_coordinates",
                        "relation",
                        "assembly",
                    },
                )
                or partition.get("output") != output_name
                or partition.get("relation")
                != "physical_store_slices_require_complete_logical_output_binding"
                or partition.get("store_ordinals")
                != [item["ordinal"] for item in expected]
                or partition.get("store_coordinates")
                != [item["coordinates"] for item in expected]
            ):
                raise ValueError(
                    f"canonical Source partition locator is not store-bound at output {output_ordinal}"
                )
            validate_source_store_partition_assembly(
                partition,
                access_rows=access_rows,
                output_name=output_name,
                output_type=output_row.get("output_type"),
                expected_stores=expected,
                dimension_aliases=dimension_aliases,
            )
        elif partition is not None:
            raise ValueError(
                f"canonical Source partition locator is unnecessary at output {output_ordinal}"
            )


def _validate_mounted_source_output_projection(
    bundle: Mapping[str, Any],
    output_construction: Sequence[Any],
) -> None:
    """Bind provider-facing output types to the canonical Source seed."""

    seed = bundle.get("artifact_seed")
    j = seed.get("j") if isinstance(seed, Mapping) else None
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    seed_outputs = function.get("outputs") if isinstance(function, Mapping) else None
    dimension_aliases = provider_dimension_alias_map(
        bundle.get("provider_dimension_aliases")
    )
    if not isinstance(seed_outputs, list) or len(seed_outputs) != len(output_construction):
        raise ValueError("canonical Source output projection cardinality is inconsistent")
    for output_ordinal, (output_row, seed_row) in enumerate(
        zip(output_construction, seed_outputs, strict=True)
    ):
        if not isinstance(output_row, Mapping) or not isinstance(seed_row, Mapping):
            raise ValueError("canonical Source output declaration is malformed")
        seed_name = seed_row.get("name")
        seed_dtype = seed_row.get("dtype")
        seed_shape = seed_row.get("shape")
        if (
            not isinstance(seed_name, str)
            or not isinstance(seed_dtype, str)
            or not isinstance(seed_shape, list)
        ):
            raise ValueError("canonical Source seed output declaration is malformed")
        try:
            provider_shape = provider_shape_from_canonical(
                seed_shape, dimension_aliases
            )
        except ValueError as exc:
            raise ValueError("canonical Source seed output dimension is malformed") from exc
        if (
            output_row.get("output") != seed_name
            or output_row.get("output_type")
            != {"dtype": seed_dtype, "shape": provider_shape}
        ):
            raise ValueError(
                f"canonical Source output construction is not seed-bound at output {output_ordinal}"
            )


def _reject_non_json_constant(value: str) -> None:
    """Reject JavaScript-only numeric constants at the artifact boundary."""

    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Keep candidate wire parsing injective instead of folding duplicate keys."""

    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _parse_candidate_object(text: str) -> dict[str, Any]:
    """Validate one semantic candidate document without inspecting its fields.

    This is deliberately only a wire-level check. CandidateCheck remains the
    authority for schema and semantic validity. Keeping this check before an
    atomic replace prevents a malformed or concatenated model response from
    replacing the current attempt's last complete candidate bytes.
    """

    value = json.loads(
        text,
        parse_constant=_reject_non_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(value, dict):
        raise TypeError("candidate.json must contain one JSON object")
    return value


def _json_container_kind(value: Any) -> str:
    """Return the JSON primitive/container kind used by the mounted schema."""

    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _expected_result_shape_for_known_operands(
    operation: Mapping[str, Any],
    operand_shapes: Sequence[tuple[str | int, ...]],
) -> tuple[str | int, ...] | None:
    """Return the narrow shape relation enforced at graph ingress.

    ``None`` means that the ingress check deliberately has no unique proof and
    must defer the relation to the canonical constructor/boundary authority.
    An empty tuple is a real scalar shape, so it must not be used as the
    unconstrained sentinel.  Keeping this relation in one helper is important:
    the rejection predicate and its safe MCP feedback must identify exactly the
    same scalar/vector case instead of disagreeing about whether a diagnostic
    exists.
    """

    operation_name = operation.get("op")
    if operation_name in {"cast", "map_unary"} and len(operand_shapes) == 1:
        return operand_shapes[0]
    if operation_name != "map_binary" or len(operand_shapes) != 2:
        return None

    maximum_rank = max(len(shape) for shape in operand_shapes)
    maximal = {shape for shape in operand_shapes if len(shape) == maximum_rank}
    if len(maximal) != 1:
        return None
    expected = next(iter(maximal))
    # Only the already-proven scalar-to-tensor broadcast is accepted by this
    # narrow ingress rule.  Non-trailing or otherwise ambiguous broadcasting
    # remains the responsibility of the boundary proof and is not inferred
    # here.
    if all(shape == expected or not shape for shape in operand_shapes):
        return expected
    return None
def _exception_candidate_check_diagnostics(
    exc: BaseException,
) -> list[dict[str, Any]] | None:
    """Project only a checked, bounded diagnostic receipt from a callback.

    Constructor errors may know an exact rule/path, but their message can also
    contain provider or Source-derived text.  The environment must never turn
    that message into relift context.  It accepts only the closed diagnostic
    tuple exported by the canonical ingress boundary.
    """

    raw = getattr(exc, "candidate_check_diagnostics", None)
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 20:
        return None
    projected: list[dict[str, Any]] = []
    for item in raw:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"rule_code", "path", "observed_refs_or_types"}
            or not isinstance(item.get("rule_code"), str)
            or not item["rule_code"]
            or len(item["rule_code"]) > 128
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or len(item["path"]) > MAX_CANDIDATE_CHECK_FIELD_CHARS
            or not isinstance(item.get("observed_refs_or_types"), (list, tuple))
            or len(item["observed_refs_or_types"]) > MAX_CANDIDATE_CHECK_OBSERVED_ITEMS
            or any(
                not isinstance(value, str)
                or not value
                or len(value) > 128
                for value in item["observed_refs_or_types"]
            )
        ):
            return None
        projected.append(
            {
                "rule_code": item["rule_code"],
                "path": item["path"],
                "observed_refs_or_types": list(item["observed_refs_or_types"]),
            }
        )
    return projected


def _remove_path_iterative(path: Path) -> None:
    """Remove one exact path tree without Python recursion."""

    try:
        root_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(root_stat.st_mode) or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    stack: list[tuple[Path, bool]] = [(path, False)]
    while stack:
        current, visited = stack.pop()
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(current_stat.st_mode) or current.is_symlink():
            current.unlink(missing_ok=True)
            continue
        os.chmod(current, stat.S_IRWXU)
        if visited:
            current.rmdir()
            continue
        stack.append((current, True))
        with os.scandir(current) as iterator:
            for entry in iterator:
                child = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append((child, False))
                else:
                    child.unlink(missing_ok=True)


@dataclass
class IntentIRLiftWorkspace:
    """Private host directories projected into exactly one agent sandbox."""

    root: Path
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path

    @classmethod
    def create(cls, files: Mapping[str, str | bytes]) -> "IntentIRLiftWorkspace":
        root = Path(tempfile.mkdtemp(prefix="intentir-lift-"))
        workspace = cls(
            root=root,
            input_dir=root / "input",
            output_dir=root / "output",
            scratch_dir=root / "work",
            tmp_dir=root / "tmp",
        )
        workspace.input_dir.mkdir()
        workspace.output_dir.mkdir()
        workspace.scratch_dir.mkdir()
        workspace.tmp_dir.mkdir()
        for relative, content in files.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                workspace.cleanup()
                raise ValueError(f"unsafe Lift input path: {relative!r}")
            destination = workspace.input_dir / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                destination.write_bytes(content)
            else:
                destination.write_text(content, encoding="utf-8")
        return workspace

    @property
    def candidate_path(self) -> Path:
        return self.output_dir / "candidate.json"

    def read_candidate(self) -> str:
        entries = tuple(self.output_dir.iterdir())
        if entries != (self.candidate_path,):
            names = sorted(path.name for path in entries)
            raise ValueError(
                "Lift output must contain exactly candidate.json; "
                f"found {names}"
            )
        path = self.candidate_path
        if path.is_symlink() or not path.is_file():
            raise ValueError("candidate.json must be a regular file")
        size = path.stat().st_size
        if not 1 <= size <= MAX_CANDIDATE_BYTES:
            raise ValueError(
                f"candidate.json size must be in [1, {MAX_CANDIDATE_BYTES}]"
            )
        raw = path.read_bytes()
        if len(raw) != size:
            raise ValueError("candidate.json changed while being read")
        text = raw.decode("utf-8")
        _parse_candidate_object(text)
        return text

    def cleanup(self) -> None:
        _remove_path_iterative(self.root)


class IntentIRLiftEnvironmentConfig(BaseModel):
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path | None = None
    timeout: float | None = None
    candidate_check_limit: int = 0
    candidate_submission_check_limit: int = 0
    executable: str = os.getenv("MSWEA_BUBBLEWRAP_EXECUTABLE", "bwrap")
    env: dict[str, str] = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/sbin:/usr/bin:/bin",
    }


@dataclass(frozen=True)
class _SemanticAccessPredicateChoices:
    """Existing typed local values eligible for one Source access predicate."""

    compatible_labels: frozenset[str]
    boolean_result_count: int
    incompatible_boolean_shape_count: int


class IntentIRLiftEnvironment:
    """Execute agent shell actions with only /input, /output, and /work visible."""

    def __init__(
        self,
        *,
        config_class: type[IntentIRLiftEnvironmentConfig] = IntentIRLiftEnvironmentConfig,
        candidate_checker: Callable[[str], Mapping[str, Any]] | None = None,
        semantic_candidate_checker: Callable[[str], Mapping[str, Any]] | None = None,
        candidate_constructor: Callable[
            [str], tuple[str, tuple[Mapping[str, Any], ...]]
        ] | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        self._semantic_candidate_skeleton, self._semantic_operation_namespace = (
            self._mounted_semantic_authoring_contract()
        )
        self._semantic_write_contract = self._mounted_semantic_write_contract()
        # The generic schema seed deliberately has no Source-specific rows.
        # Read the already-mounted construction requirements once so the MCP
        # lifecycle can expose the exact *slots* it must complete without
        # publishing a second Source/ABI/C projection or inventing an
        # environment-local schema.  The full semantic content of a slot
        # remains available through the bounded transfer-ledger resource.
        self._semantic_construction_identity = (
            self._mounted_semantic_construction_identity()
        )
        self._semantic_provider_dimension_aliases = (
            self._mounted_semantic_provider_dimension_aliases()
        )
        self._semantic_construction_slots = (
            self._mounted_semantic_construction_slots()
        )
        self._semantic_source_tensor_handles = (
            self._mounted_semantic_source_tensor_handles()
        )
        self._semantic_source_tensor_value_types = (
            self._mounted_semantic_source_tensor_value_types()
        )
        self._semantic_source_input_tensor_handles = (
            self._mounted_semantic_input_tensor_handles()
        )
        self._semantic_source_output_tensor_value_types = (
            self._mounted_semantic_output_tensor_value_types()
        )
        if tuple(self._semantic_write_contract["steps"]) != MCP_SEMANTIC_WRITE_TOOL_NAMES:
            raise ValueError(
                "mounted MCP semantic write steps drifted from server tool surface"
            )
        if self._semantic_write_contract["protocol"] != canonical_semantic_mcp_write_protocol():
            raise ValueError(
                "mounted MCP semantic write authority drifted from canonical schema"
            )
        if self.config.candidate_check_limit < 0:
            raise ValueError("candidate_check_limit must be non-negative")
        if self.config.candidate_submission_check_limit < 0:
            raise ValueError(
                "candidate_submission_check_limit must be non-negative"
            )
        self._candidate_checker = candidate_checker
        self._semantic_candidate_checker = semantic_candidate_checker
        # This callback is the one host-owned Source/Harness construction
        # transaction reached by MCP finalize.  It is deliberately not a
        # provider patch hook: it receives one complete semantic surface and
        # either returns the complete canonical artifact or rejects it.
        self._candidate_constructor = candidate_constructor
        self._candidate_check_calls = 0
        self._candidate_check_evaluations = 0
        self._candidate_check_calls_by_trigger = {
            "explicit": 0,
            "submission": 0,
        }
        self._candidate_check_evaluations_by_trigger = {
            "explicit": 0,
            "submission": 0,
        }
        self._candidate_check_shared_evaluations = 0
        self._candidate_check_submission_reserved_evaluations = 0
        self._candidate_check_audit: list[dict[str, Any]] = []
        self._candidate_constructor_audit: list[dict[str, Any]] = []
        self._candidate_check_passed_bytes: bytes | None = None
        self._candidate_finalize_audit: list[dict[str, Any]] = []
        self._candidate_submission_attempted = False
        # The MCP writer owns only this semantic J/pi buffer.  It is initialized
        # from the schema-derived empty semantic template. C, canonical handles,
        # effects and the remaining pi proof are never mutable MCP state.
        self._semantic_construction_buffer: dict[str, Any] | None = None
        self._semantic_construction_audit: list[dict[str, Any]] = []
        self._semantic_construction_completed_steps: set[str] = set()
        # This is intentionally receipt-only telemetry.  It contains a
        # normalized declaration digest and small structural facts, never the
        # provider declaration or a Source/candidate fragment.
        self._semantic_operation_graph_failures: list[dict[str, Any]] = []
        self._semantic_access_declaration_failures: list[dict[str, Any]] = []
        self._semantic_output_declaration_failures: list[dict[str, Any]] = []
        self._semantic_boundary_field_failures: list[dict[str, Any]] = []
        self._multi_action_rejections = 0
        self._structured_context_queries: list[str] = []
        self._structured_schema_queries: list[list[str]] = []
        self._mcp_requests: list[dict[str, Any]] = []
        self._mcp_phase_application_failures: list[dict[str, Any]] = []
        self._preparation_failure_receipts: list[dict[str, Any]] = []
        self._semantic_entry_read = False
        self._semantic_entry_source_read = False
        self._semantic_entry_transfer_ledger_read = False
        # A pre-construction read establishes the initial Source authority.
        # Access authoring additionally requires one refresh after the graph
        # has been accepted, bound to this same in-memory transaction.
        self._semantic_transfer_read_generation = 0
        self._semantic_transfer_expected_start = 1
        self._semantic_operation_graph_transfer_generation: int | None = None
        self._semantic_access_transfer_generation: int | None = None
        self._semantic_transaction_identity: str | None = None
        self._semantic_entry_supplemental_read: str | None = None
        self._semantic_entry_schema_lookup_calls = 0
        # Filled only from the mounted source/brief semantic spine.  A fresh
        # owner may request an exact operation leaf only when that compact
        # Source fact explicitly says the brief lacks it.
        self._semantic_entry_missing_operation_leaves: frozenset[str] = frozenset()
        # Counts only summarize the already-mounted Source semantic spine.
        # They let the begin receipt distinguish a complete calculation from
        # an index/mask-only setup graph without exposing Source text or
        # creating a second provider-authored operation template.
        self._semantic_entry_operation_fact_count = 0
        self._semantic_entry_non_boundary_operation_fact_count = 0
        self._semantic_entry_operation_requirements: tuple[
            dict[str, Any], ...
        ] = ()
        self._semantic_entry_required_operation_families: dict[str, int] = {}
        self._semantic_entry_required_operation_attributes: tuple[
            tuple[str, str, str, int], ...
        ] = ()
        # Once ingress has exposed begin_candidate, stale/static clients must
        # not spend the remaining Lift budget attempting reads or obsolete
        # tools. Two rejected attempts are enough to establish that the
        # construction phase was not entered; no partial buffer is retained.
        self._semantic_preconstruction_phase_failures = 0
        self._stage_action_calls = 0
        self._stage_action_failures = 0
        self._stage_action_elapsed_ms = 0.0
        # ``/input`` contains the complete host construction seed solely for
        # the bounded MCP resource implementation.  An owner shell must not
        # bypass source-brief/read budgets by opening those files directly.
        # Host-issued MCP readers enter through the small scoped helper below.
        self._host_mcp_read_depth = 0
        self._mcp_server = IntentIRLiftMCPServer(
            self._mcp_dispatch,
            active_resource_uris=self._active_mcp_resource_uris,
        )
        self._validate_paths()
        self._startup_context, self._startup_context_receipt = (
            self._build_startup_context()
        )
        self._probe()

    def _mounted_semantic_authoring_contract(self) -> tuple[str, str]:
        """Read the host-published authoring contract; never recreate it here."""

        path = self.config.input_dir / "schema" / "canonical_wire_bundle.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical semantic authoring bundle is unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        profile = bundle.get("profile") if isinstance(bundle, Mapping) else None
        semantic = (
            profile.get("semantic_candidate")
            if isinstance(profile, Mapping)
            else None
        )
        template = semantic.get("template") if isinstance(semantic, Mapping) else None
        namespace = (
            semantic.get("operation_namespace")
            if isinstance(semantic, Mapping)
            else None
        )
        if not isinstance(template, Mapping) or not isinstance(namespace, Mapping):
            raise ValueError("canonical semantic authoring contract is malformed")
        root_fields = semantic.get("root_fields") if isinstance(semantic, Mapping) else None
        j_fields = semantic.get("j_fields") if isinstance(semantic, Mapping) else None
        j_field_types = semantic.get("j_field_types") if isinstance(semantic, Mapping) else None
        template_j = template.get("j")
        template_pi = template.get("pi")
        if (
            not isinstance(root_fields, list)
            or not isinstance(j_fields, list)
            or not isinstance(j_field_types, Mapping)
            or set(template) != set(root_fields)
            or not isinstance(template_j, Mapping)
            or not isinstance(template_pi, Mapping)
            or set(template_j) != set(j_fields)
            or template_pi
            or not isinstance(template_j.get("intent"), Mapping)
            or not isinstance(template_j["intent"].get("intent_function"), Mapping)
            or not isinstance(template_j["intent"]["intent_function"].get("body"), list)
            or any(
                j_field_types.get(field) == "object"
                and not isinstance(template_j.get(field), Mapping)
                for field in j_fields
            )
            or any(
                j_field_types.get(field) == "array"
                and not isinstance(template_j.get(field), list)
                for field in j_fields
            )
        ):
            raise ValueError("mounted semantic candidate skeleton drifted")
        return (
            json.dumps(template, ensure_ascii=False, separators=(",", ":")),
            json.dumps(namespace, ensure_ascii=False, separators=(",", ":")),
        )

    def _mounted_semantic_write_contract(self) -> dict[str, Any]:
        """Read the MCP write projection from the same mounted wire bundle.

        This is intentionally a reader, not another declaration of J fields.
        The schema module emits the profile; the environment only verifies the
        bounded destinations before accepting a provider declaration.
        """

        path = self.config.input_dir / "schema" / "canonical_wire_bundle.json"
        bundle = json.loads(path.read_text(encoding="utf-8"))
        profile = bundle.get("profile") if isinstance(bundle, Mapping) else None
        semantic = (
            profile.get("semantic_candidate")
            if isinstance(profile, Mapping)
            else None
        )
        write = (
            profile.get("semantic_mcp_write")
            if isinstance(profile, Mapping)
            else None
        )
        if not isinstance(semantic, Mapping) or not isinstance(write, Mapping):
            raise ValueError("canonical MCP semantic write contract is unavailable")
        writes = write.get("writes")
        steps = write.get("steps")
        required_steps = write.get("required_steps")
        initial_template = write.get("initial_template")
        tool_arguments = write.get("tool_arguments")
        boundary_types = semantic.get("boundary_field_types")
        j_types = semantic.get("j_field_types")
        if (
            not isinstance(writes, Mapping)
            or not isinstance(steps, list)
            or not steps
            or any(not isinstance(step, str) or not step for step in steps)
            or len(steps) != len(set(steps))
            or not isinstance(required_steps, list)
            or not required_steps
            or any(
                not isinstance(step, str) or step not in steps
                for step in required_steps
            )
            or len(required_steps) != len(set(required_steps))
            or initial_template != semantic.get("template")
            or not isinstance(tool_arguments, Mapping)
            or not isinstance(boundary_types, Mapping)
            or not isinstance(j_types, Mapping)
            or not isinstance(writes.get("boundary_field"), list)
            or not isinstance(writes.get("j_field"), list)
            or set(tool_arguments) != set(steps)
            or any(
                not isinstance(item, str)
                for item in (
                    *writes["boundary_field"],
                    *writes["j_field"],
                )
            )
            or any(
                row is not None
                and (
                    not isinstance(row, Mapping)
                    or set(row)
                    not in (
                        {"field", "json_container"},
                        {"field", "json_container", "pattern"},
                    )
                    or not isinstance(row.get("field"), str)
                    or row.get("json_container") not in {"array", "object"}
                    or "pattern" in row
                    and (
                        not isinstance(row.get("pattern"), str)
                        or row.get("json_container") != "array"
                        or not row.get("pattern")
                    )
                )
                for row in tool_arguments.values()
            )
        ):
            raise ValueError("canonical MCP semantic write fields are malformed")
        return {
            "boundary_fields": tuple(writes["boundary_field"]),
            "j_fields": tuple(writes["j_field"]),
            "boundary_field_types": dict(boundary_types),
            "j_field_types": dict(j_types),
            "tool_arguments": copy.deepcopy(dict(tool_arguments)),
            "steps": tuple(steps),
            "required_steps": tuple(required_steps),
            "protocol": copy.deepcopy(dict(write)),
        }

    @staticmethod
    def _stable_host_digest(value: Mapping[str, Any]) -> str:
        """Return an opaque digest for a mounted host-owned record."""

        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _mounted_semantic_construction_identity(self) -> dict[str, str]:
        """Load the identity shared by source/brief and transfer-ledger.

        The provider does not author any of these values.  The lifecycle uses
        them solely to reject a stale mounted brief or ledger before it can be
        associated with a fresh semantic transaction.
        """

        construction_path = self.config.input_dir / "canonical_construction.json"
        source_wire_path = (
            self.config.input_dir / "schema" / "canonical_source_wire_bundle.json"
        )
        if (
            not construction_path.is_file()
            or construction_path.is_symlink()
            or not source_wire_path.is_file()
            or source_wire_path.is_symlink()
        ):
            raise ValueError("canonical construction identity is unavailable")
        construction = json.loads(construction_path.read_text(encoding="utf-8"))
        source_wire = json.loads(source_wire_path.read_text(encoding="utf-8"))
        source_facts = (
            construction.get("source_facts")
            if isinstance(construction, Mapping)
            else None
        )
        transfer_ledger = (
            construction.get("transfer_ledger")
            if isinstance(construction, Mapping)
            else None
        )
        allocator = (
            construction.get("allocator")
            if isinstance(construction, Mapping)
            else None
        )
        construction_digest = (
            construction.get("authority_digest")
            if isinstance(construction, Mapping)
            else None
        )
        source_facts_digest = (
            self._stable_host_digest(source_facts)
            if isinstance(source_facts, Mapping)
            else None
        )
        if (
            not isinstance(construction_digest, str)
            or not construction_digest
            or not isinstance(source_facts_digest, str)
            or not source_facts_digest
            or not isinstance(transfer_ledger, Mapping)
            or transfer_ledger.get("source_facts_digest") != source_facts_digest
            or not isinstance(allocator, Mapping)
            or not isinstance(source_wire, Mapping)
            or source_wire.get("construction_digest") != construction_digest
            or source_wire.get("source_transfer_ledger") != transfer_ledger
            or source_wire.get("source_facts_digest")
            != construction.get("source_facts_digest")
        ):
            raise ValueError("canonical construction identity is malformed")
        if source_wire.get("source_slice_digest") != (
            source_facts.get("source_slice_digest")
            if isinstance(source_facts, Mapping)
            else None
        ):
            raise ValueError("canonical Source slice lineage is malformed")
        host_receipt = source_wire.get("host_launch_contract")
        transfer_projection = source_wire.get("semantic_transfer_projection")
        compiler_facts = (
            source_facts.get("compiler_facts")
            if isinstance(source_facts, Mapping)
            else None
        )
        contract = (
            compiler_facts.get("host_launch_contract")
            if isinstance(compiler_facts, Mapping)
            else None
        )
        if (
            not isinstance(host_receipt, Mapping)
            or not isinstance(transfer_projection, Mapping)
            or transfer_projection.get("host_launch_contract") != host_receipt
            or host_receipt.get("source_facts_digest") != source_facts_digest
        ):
            raise ValueError("canonical host launch receipt lineage is malformed")
        if host_receipt.get("transfer_ledger_digest") != self._stable_host_digest(transfer_ledger):
            raise ValueError("canonical host launch transfer lineage is malformed")
        if not isinstance(contract, Mapping) or not contract:
            if host_receipt.get("status") != "unavailable":
                raise ValueError("canonical host launch receipt falsely claims availability")
            if (
                isinstance(source_facts, Mapping)
                and source_facts.get("frontend") == "triton"
                and source_facts.get("source_kind") == "python"
            ):
                raise ValueError("canonical Triton host launch contract is unavailable")
        else:
            if host_receipt.get("status") != "pass":
                raise ValueError("canonical host launch receipt is not closed")
            if host_receipt.get("contract_digest") != self._stable_host_digest(contract):
                raise ValueError("canonical host launch contract digest is stale")
        payload = {
            key: value
            for key, value in construction.items()
            if key != "authority_digest"
        }
        if self._stable_host_digest(payload) != construction_digest:
            raise ValueError("canonical construction identity digest is stale")
        return {
            "construction_digest": construction_digest,
            "source_facts_digest": source_facts_digest,
            "transfer_ledger_digest": self._stable_host_digest(transfer_ledger),
            "allocator_snapshot_digest": self._stable_host_digest(allocator),
        }

    def _mounted_semantic_construction_slots(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Read Source-sized semantic completion slots from host authority.

        This is intentionally a narrow projection of
        ``semantic_construction_requirements`` rather than a second candidate
        grammar.  The schema/constructor remains responsible for the actual
        access and output row shapes.  The environment needs only enough
        immutable information to tell a fresh MCP owner which rows cannot be
        omitted before it spends a finalization attempt.
        """

        path = (
            self.config.input_dir
            / "schema"
            / "canonical_source_wire_bundle.json"
        )
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical Source construction requirements are unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        requirements = (
            bundle.get("semantic_construction_requirements")
            if isinstance(bundle, Mapping)
            else None
        )
        source_transfer = (
            bundle.get("source_transfer_ledger")
            if isinstance(bundle, Mapping)
            else None
        )
        transfer_projection = (
            bundle.get("semantic_transfer_projection")
            if isinstance(bundle, Mapping)
            else None
        )
        if not isinstance(source_transfer, Mapping):
            raise ValueError("canonical Source transfer ledger is unavailable")
        if source_transfer.get("status") != "pass":
            raise ValueError(
                "canonical Source transfer ledger is not closed: "
                f"{source_transfer.get('reason')}"
            )
        if (
            not isinstance(transfer_projection, Mapping)
            or transfer_projection.get("status") != "pass"
            or not isinstance(transfer_projection.get("occurrences"), list)
            or transfer_projection.get("host_launch_contract")
            != bundle.get("host_launch_contract")
        ):
            raise ValueError("canonical Source transfer projection is not closed")
        source_owner_outlines: dict[int, dict[str, Any]] = {}
        for occurrence in transfer_projection["occurrences"]:
            if not isinstance(occurrence, Mapping):
                raise ValueError("canonical Source transfer occurrence is malformed")
            ordinal = occurrence.get("ordinal")
            # ``semantic_transfer_projection`` is the public occurrence
            # projection emitted by ``_logical_transfer_projection``.  Its
            # Source semantic fields (including authorizations) live on the
            # occurrence row itself; ``source_semantics`` is the nested shape
            # used later by ``semantic_construction_requirements``.  Reading
            # the latter here silently discarded every public owner outline
            # and made a Source-authorized physical store look unavailable.
            # Keep this join on the canonical public row and let the private
            # ledger/outline checks below remain the authority.
            authorizations = occurrence.get("semantic_authorizations")
            owner = (
                authorizations.get("owner")
                if isinstance(authorizations, Mapping)
                else None
            )
            outline = owner.get("authoring_outline") if isinstance(owner, Mapping) else None
            if outline is None:
                continue
            if (
                not is_canonical_ordinal(ordinal)
                or not isinstance(outline, Mapping)
                or set(outline) != {"store", "graph"}
                or not isinstance(outline.get("store"), Mapping)
                or not isinstance(outline.get("graph"), list)
            ):
                raise ValueError("canonical Source owner outline is malformed")
            previous = source_owner_outlines.get(ordinal)
            if previous is not None and previous != dict(outline):
                raise ValueError("canonical Source owner outline is ambiguous")
            source_owner_outlines[ordinal] = copy.deepcopy(dict(outline))
        public_projection_text = json.dumps(
            transfer_projection,
            sort_keys=True,
            ensure_ascii=False,
        )
        # The public projection may contain a deliberately public computed
        # guard/access token, but never a host owner token or private receipt
        # fields.  The full private ledger remains in the mounted host record
        # and is consumed only by host-side admission.
        if (
            re.search(r"(?<![A-Za-z0-9_])so_[0-9a-f]{32}_[0-9a-f]{64}(?![A-Za-z0-9_])", public_projection_text)
            or any(
                isinstance(auth, Mapping)
                and (
                    "status" in auth
                    or "compiler_attestation" in auth
                    or (name == "owner" and "token" in auth)
                )
                for row in transfer_projection["occurrences"]
                if isinstance(row, Mapping)
                for name, auth in (
                    row.get("semantic_authorizations", {}).items()
                    if isinstance(row.get("semantic_authorizations"), Mapping)
                    else ()
                )
            )
        ):
            raise ValueError("canonical Source projection leaks a private authorization field")
        semantic_required = (
            requirements.get("semantic_required")
            if isinstance(requirements, Mapping)
            else None
        )
        accesses = (
            semantic_required.get("accesses")
            if isinstance(semantic_required, Mapping)
            else None
        )
        outputs = (
            semantic_required.get("outputs")
            if isinstance(semantic_required, Mapping)
            else None
        )
        access_rows = accesses.get("rows") if isinstance(accesses, Mapping) else None
        output_rows = outputs.get("rows") if isinstance(outputs, Mapping) else None
        obligation_rows = (
            requirements.get("obligations")
            if isinstance(requirements, Mapping)
            else None
        )
        projected_access_rows = (
            transfer_projection.get("occurrences")
            if isinstance(transfer_projection, Mapping)
            else None
        )
        if (
            not isinstance(access_rows, list)
            or not isinstance(output_rows, list)
            or not isinstance(projected_access_rows, list)
            or len(access_rows) != len(projected_access_rows)
            or any(not isinstance(row, Mapping) for row in access_rows)
            or any(not isinstance(row, Mapping) for row in projected_access_rows)
            or any(not isinstance(row, Mapping) for row in output_rows)
        ):
            raise ValueError("canonical Source semantic completion slots are malformed")
        if bundle.get("source_facts_digest") != self._semantic_construction_identity.get(
            "source_facts_digest"
        ):
            raise ValueError("canonical Source semantic obligation identity is stale")
        validated_obligations = validate_source_semantic_obligations(
            obligation_rows,
            self._semantic_construction_identity["source_facts_digest"],
        )

        private_occurrences = source_transfer.get("occurrences")
        if (
            not isinstance(private_occurrences, list)
            or len(private_occurrences) != len(projected_access_rows)
            or any(not isinstance(row, Mapping) for row in private_occurrences)
        ):
            raise ValueError("canonical Source private occurrence ledger is malformed")
        private_bindings = source_transfer.get("bindings")
        if (
            not isinstance(private_bindings, list)
            or len(private_bindings) != len(projected_access_rows)
            or any(not isinstance(row, Mapping) for row in private_bindings)
        ):
            raise ValueError("canonical Source private authorization ledger is malformed")

        def public_authorization_is_closed(
            value: object,
            *,
            owner: bool = False,
        ) -> bool:
            if value is None:
                return True
            if not isinstance(value, Mapping):
                return False
            if "status" in value or "compiler_attestation" in value:
                return False
            if owner:
                return (
                    "token" not in value
                    and set(value).issubset({"operation_family", "authoring_outline"})
                    and isinstance(value.get("operation_family"), str)
                )
            return (
                set(value) == {"token", "operation_family"}
                and isinstance(value.get("token"), str)
                and isinstance(value.get("operation_family"), str)
            )

        def public_authorization_matches_private(
            private: object,
            public: object,
            *,
            owner: bool = False,
        ) -> bool:
            if not isinstance(private, Mapping):
                return public is None
            status = private.get("status")
            if status == "unavailable":
                return public is None and set(private) == {"status", "reason"}
            if status != "pass" or not isinstance(public, Mapping):
                return False
            family = private.get("operation_family")
            token = private.get("token")
            if not isinstance(family, str) or not isinstance(token, str):
                return False
            if owner:
                expected = {"operation_family"}
                outline = private.get("authoring_outline")
                if outline is not None:
                    expected.add("authoring_outline")
                return (
                    set(public) == expected
                    and "token" not in public
                    and public.get("operation_family") == family
                    and (
                        outline is None
                        or public.get("authoring_outline") == outline
                    )
                )
            return (
                set(public) == {"token", "operation_family"}
                and public.get("token") == token
                and public.get("operation_family") == family
            )

        for ordinal, (
            private_row,
            private_binding,
            wire_row,
            requirement_row,
        ) in enumerate(
            zip(
                private_occurrences,
                private_bindings,
                projected_access_rows,
                access_rows,
                strict=True,
            )
        ):
            private_auth = private_binding.get("semantic_authorizations")
            public_auth = wire_row.get("semantic_authorizations")
            if (
                not is_canonical_ordinal(ordinal)
                or not is_canonical_ordinal(private_row.get("ordinal"))
                or not is_canonical_ordinal(private_binding.get("ordinal"))
                or not is_canonical_ordinal(wire_row.get("ordinal"))
                or not is_canonical_ordinal(requirement_row.get("ordinal"))
                or private_row.get("ordinal") != ordinal
                or private_binding.get("ordinal") != ordinal
                or wire_row.get("ordinal") != ordinal
                or requirement_row.get("ordinal") != ordinal
                or wire_row.get("access_label") != f"access{ordinal}"
                or requirement_row.get("access_label") != f"access{ordinal}"
                or wire_row.get("kind") != private_row.get("kind")
                or wire_row.get("masked") != private_row.get("masked")
                or wire_row.get("tensor") != private_row.get("logical_ref")
                or not isinstance(private_auth, Mapping)
                or not isinstance(public_auth, Mapping)
                or not public_authorization_is_closed(
                    public_auth.get("access_view")
                )
                or not public_authorization_is_closed(
                    public_auth.get("owner"),
                    owner=True,
                )
                or not public_authorization_is_closed(
                    public_auth.get("computed_guard")
                )
                or set(private_auth) != {"access_view", "owner", "computed_guard"}
                or not public_authorization_matches_private(
                    private_auth.get("access_view"),
                    public_auth.get("access_view"),
                )
                or not public_authorization_matches_private(
                    private_auth.get("owner"),
                    public_auth.get("owner"),
                    owner=True,
                )
                or not public_authorization_matches_private(
                    private_auth.get("computed_guard"),
                    public_auth.get("computed_guard"),
                )
            ):
                raise ValueError(
                    "canonical Source occurrence projection is not bound to its ledger"
                )

            requirement_source = requirement_row.get("source_semantics")
            wire_auth = wire_row.get("semantic_authorizations")
            requirement_auth = (
                requirement_source.get("semantic_authorizations")
                if isinstance(requirement_source, Mapping)
                else None
            )
            if (
                not isinstance(requirement_source, Mapping)
                or requirement_source.get("kind") != wire_row.get("kind")
                or requirement_source.get("tensor") != wire_row.get("tensor")
                or requirement_source.get("masked") != wire_row.get("masked")
                or requirement_source.get("predicate") != wire_row.get("predicate")
                or requirement_source.get("fill_value") != wire_row.get("fill_value")
                or requirement_source.get("index") != wire_row.get("index")
                or requirement_source.get("effect_order") != wire_row.get("effect_order")
                or requirement_source.get("pi_binding") != wire_row.get("pi_binding")
                or requirement_auth != wire_auth
                or requirement_row.get("required_fields") != wire_row.get("required_fields")
                or requirement_row.get("provider_fields") != wire_row.get("provider_fields")
                or requirement_row.get("provider_owned_fields")
                != wire_row.get("provider_owned_fields")
                or requirement_row.get("provider_owned_expression_fields")
                != wire_row.get("provider_owned_expression_fields")
                or requirement_row.get("provider_owned_relations")
                != wire_row.get("provider_owned_relations")
                or requirement_row.get("wire_template") != wire_row.get("wire_template")
                or requirement_row.get("index_expression_requirement")
                != wire_row.get("index_expression_requirement")
            ):
                raise ValueError(
                    "canonical Source requirements are not bound to transfer projection"
                )

        access_slots: list[dict[str, Any]] = []
        for ordinal, (requirement_row, wire_row) in enumerate(
            zip(access_rows, projected_access_rows, strict=True)
        ):
            label = wire_row.get("access_label")
            required = wire_row.get("required_fields")
            template = wire_row.get("wire_template")
            provider_owned_fields = wire_row.get("provider_owned_fields", [])
            provider_owned_expression_fields = wire_row.get(
                "provider_owned_expression_fields", []
            )
            provider_owned_relations = wire_row.get("provider_owned_relations")
            index_expression_requirement = wire_row.get(
                "index_expression_requirement"
            )
            semantic_authorizations = wire_row.get("semantic_authorizations")
            owner_authorization = (
                semantic_authorizations.get("owner")
                if isinstance(semantic_authorizations, Mapping)
                else None
            )
            owner_operation_family = (
                owner_authorization.get("operation_family")
                if isinstance(owner_authorization, Mapping)
                else None
            )
            if (
                requirement_row.get("ordinal") != ordinal
                or requirement_row.get("access_label") != label
                or wire_row.get("ordinal") != ordinal
                or label != f"access{ordinal}"
                or not isinstance(required, list)
                or any(not isinstance(field, str) or not field for field in required)
                or not isinstance(template, Mapping)
                or template.get("ordinal") != ordinal
                or template.get("access_label") != label
                or "id" in template
                or not isinstance(provider_owned_fields, list)
                or any(
                    not isinstance(field, str)
                    or field not in required
                    or field not in template
                    for field in provider_owned_fields
                )
                or not isinstance(provider_owned_expression_fields, list)
                or any(
                    not isinstance(field, str)
                    or field not in required
                    or field not in template
                    or field in provider_owned_fields
                    for field in provider_owned_expression_fields
                )
                or not isinstance(provider_owned_relations, Mapping)
                or (
                    index_expression_requirement is not None
                    and (
                        not isinstance(index_expression_requirement, str)
                        or not index_expression_requirement
                        or "index_expression" not in provider_owned_expression_fields
                    )
                )
                or (
                    "node_ref" in provider_owned_fields
                    and (
                        wire_row.get("kind") not in {"store", "update"}
                        or not isinstance(owner_operation_family, str)
                        or not owner_operation_family
                    )
                )
            ):
                raise ValueError("canonical Source access completion slots are malformed")
            access_slots.append(
                {
                    "ordinal": ordinal,
                    "access_label": label,
                    "required_fields": list(required),
                    "wire_template": copy.deepcopy(dict(template)),
                    "provider_owned_fields": list(provider_owned_fields),
                    # Expression-valued provider fields have a different
                    # contract from provider-local labels.  Preserve this
                    # distinction in the mounted completion slot; otherwise
                    # the access validator treats a legitimate local index
                    # expression as if it had to equal the Source placeholder.
                    "provider_owned_expression_fields": list(
                        provider_owned_expression_fields
                    ),
                    # Keep the already-validated relation contract attached
                    # to the same slot.  It is consumed by admission and by
                    # bounded repair feedback; no second hard-coded relation
                    # table is allowed to replace it.
                    "provider_owned_relations": copy.deepcopy(
                        dict(provider_owned_relations)
                    ),
                    "source_index_expression_requirement": index_expression_requirement,
                    # A Source-authorized store/update owner is the root of
                    # its issued owner DAG, not a load's direct tensor
                    # consumer. Preserve the already-bound operation family
                    # on the internal slot so graph preflight can offer only
                    # same-family local labels; final Source owner admission
                    # still proves the complete root/DAG relation.
                    "source_owner_operation_family": owner_operation_family,
                    "obligations": [
                        copy.deepcopy(obligation)
                        for obligation in validated_obligations
                        if obligation["source_locator"].get("ordinal") == ordinal
                    ],
                }
            )
            if not access_slots[-1]["obligations"]:
                raise ValueError(
                    "canonical Source access completion slot has no obligations"
                )

        output_slots: list[dict[str, Any]] = []
        output_construction = bundle.get("output_construction")
        if (
            not isinstance(output_construction, list)
            or len(output_construction) != len(output_rows)
            or any(not isinstance(row, Mapping) for row in output_construction)
        ):
            raise ValueError("canonical Source output construction is malformed")
        _validate_mounted_source_output_projection(bundle, output_construction)
        _validate_mounted_source_store_projection(
            access_rows,
            output_rows,
            output_construction,
            self._semantic_provider_dimension_aliases,
        )
        for ordinal, (row, source_output) in enumerate(
            zip(output_rows, output_construction, strict=True)
        ):
            required = row.get("required_fields")
            template = row.get("wire_template")
            if (
                not isinstance(row.get("output"), str)
                or row.get("output") != source_output.get("output")
                or row.get("output_type") != source_output.get("output_type")
                or not isinstance(required, list)
                or required != ["value"]
                or not isinstance(template, Mapping)
                or set(template) != {"value"}
            ):
                raise ValueError("canonical Source output completion slots are malformed")
            source_store_requirements = row.get("source_store_requirements")
            if source_store_requirements is not None:
                if (
                    not isinstance(source_store_requirements, list)
                    or not source_store_requirements
                    or any(
                        not isinstance(item, Mapping)
                        or set(item) != {"ordinal", "kind", "operation_family", "coordinates"}
                        or not is_canonical_ordinal(item.get("ordinal"))
                        or item.get("kind") not in {"store", "update"}
                        or not isinstance(item.get("operation_family"), str)
                        or not item.get("operation_family")
                        or not isinstance(item.get("coordinates"), list)
                        or any(
                            not (
                                isinstance(coordinate, int)
                                and not isinstance(coordinate, bool)
                            )
                            and not (
                                isinstance(coordinate, str)
                                and (
                                    re.fullmatch(r"axis[0-9]+", coordinate) is not None
                                    or re.fullmatch(r"[0-9]+", coordinate) is not None
                                )
                            )
                            for coordinate in item.get("coordinates", ())
                        )
                        for item in source_store_requirements
                    )
                    or len({item["ordinal"] for item in source_store_requirements})
                    != len(source_store_requirements)
                ):
                    raise ValueError("canonical Source output store requirements are malformed")
            partition_requirement = row.get(
                "source_store_partition_requirement"
            )
            if partition_requirement is not None:
                if (
                    not isinstance(partition_requirement, Mapping)
                        or set(partition_requirement)
                        not in (
                            {
                                "output",
                                "store_ordinals",
                                "store_coordinates",
                                "relation",
                            },
                            {
                                "output",
                                "store_ordinals",
                                "store_coordinates",
                                "relation",
                                "assembly",
                            },
                        )
                    or partition_requirement.get("output") != row.get("output")
                    or partition_requirement.get("relation")
                    != "physical_store_slices_require_complete_logical_output_binding"
                    or not isinstance(
                        partition_requirement.get("store_ordinals"), list
                    )
                    or not isinstance(
                        partition_requirement.get("store_coordinates"), list
                    )
                    or len(partition_requirement["store_ordinals"])
                    != len(partition_requirement["store_coordinates"])
                    or not partition_requirement["store_ordinals"]
                    or any(
                        type(item) is not int
                        for item in partition_requirement["store_ordinals"]
                    )
                    or any(
                        not isinstance(item, list)
                        for item in partition_requirement["store_coordinates"]
                    )
                ):
                    raise ValueError(
                        "canonical Source output partition requirement is malformed"
                    )
            source_owner_authoring_outlines: list[dict[str, Any]] = []
            if source_store_requirements is not None:
                for requirement in source_store_requirements:
                    ordinal_ref = requirement["ordinal"]
                    outline = source_owner_outlines.get(ordinal_ref)
                    if outline is None:
                        raise ValueError(
                            "canonical Source output owner outline is unavailable"
                        )
                    source_owner_authoring_outlines.append(
                        {
                            "ordinal": ordinal_ref,
                            "outline": copy.deepcopy(outline),
                        }
                    )
            # Output names/types remain in the transfer-ledger view.  An ABI
            # ordinal plus the schema-owned required selection is all that is
            # needed to prevent an omitted mapping while preserving a compact
            # first construction response.
            output_slot = {
                "ordinal": ordinal,
                "required_fields": ["value"],
                "wire_template": copy.deepcopy(dict(template)),
                "obligations": [
                    copy.deepcopy(obligation)
                    for obligation in validated_obligations
                    if obligation["source_locator"].get("output_ordinal")
                    == ordinal
                ],
            }
            if not output_slot["obligations"]:
                raise ValueError(
                    "canonical Source output completion slot has no obligations"
                )
            if source_store_requirements is not None:
                output_slot["source_store_requirements"] = copy.deepcopy(
                    source_store_requirements
                )
            if partition_requirement is not None:
                output_slot["source_store_partition_requirement"] = copy.deepcopy(
                    partition_requirement
                )
            if source_owner_authoring_outlines:
                output_slot["source_owner_authoring_outlines"] = (
                    source_owner_authoring_outlines
                )
            output_slots.append(output_slot)
        return {
            "accesses": tuple(access_slots),
            "outputs": tuple(output_slots),
            "obligations": tuple(validated_obligations),
        }

    def _mounted_semantic_source_tensor_declarations(
        self,
    ) -> dict[str, tuple[str, tuple[str | int, ...]]]:
        """Read the canonical Source tensor declarations once, consistently.

        ``intent_function.inputs`` and ``intent_function.outputs`` are two ABI
        collections, not two namespaces.  A legal inout therefore repeats its
        input ``t*`` in outputs.  Repetition within either collection is still
        ambiguous, and a cross-collection repetition is legal only when its
        complete declared type is identical.  Keeping this projection shared
        prevents graph-handle and graph-type ingress from disagreeing about a
        Source inout.
        """

        path = (
            self.config.input_dir
            / "schema"
            / "canonical_source_wire_bundle.json"
        )
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical Source tensor declarations are unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        seed = bundle.get("artifact_seed") if isinstance(bundle, Mapping) else None
        j = seed.get("j") if isinstance(seed, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        if not isinstance(function, Mapping):
            raise ValueError("canonical Source tensor declarations are malformed")
        result: dict[str, tuple[str, tuple[str | int, ...]]] = {}
        for field in ("inputs", "outputs"):
            rows = function.get(field)
            if not isinstance(rows, list):
                raise ValueError("canonical Source tensor declarations are malformed")
            collection_names: set[str] = set()
            for row in rows:
                name = row.get("name") if isinstance(row, Mapping) else None
                dtype = row.get("dtype") if isinstance(row, Mapping) else None
                shape = row.get("shape") if isinstance(row, Mapping) else None
                if (
                    not isinstance(name, str)
                    or re.fullmatch(r"t[0-9]+", name) is None
                    or not isinstance(dtype, str)
                    or not dtype
                    or not isinstance(shape, list)
                ):
                    raise ValueError("canonical Source tensor declarations are malformed")
                if name in collection_names:
                    raise ValueError("canonical Source tensor declarations are ambiguous")
                collection_names.add(name)
                try:
                    provider_shape = provider_shape_from_canonical(
                        shape, self._semantic_provider_dimension_aliases
                    )
                except ValueError as exc:
                    raise ValueError(
                        "canonical Source tensor declarations are malformed"
                    ) from exc
                declaration = (dtype, tuple(provider_shape))
                previous = result.get(name)
                if previous is not None and previous != declaration:
                    raise ValueError("canonical Source tensor declarations are ambiguous")
                result[name] = declaration
        return result

    def _mounted_semantic_provider_dimension_aliases(self) -> dict[str, str]:
        """Read the Source-issued canonical-to-provider dimension mapping."""

        construction_path = self.config.input_dir / "canonical_construction.json"
        source_wire_path = (
            self.config.input_dir / "schema" / "canonical_source_wire_bundle.json"
        )
        if (
            not construction_path.is_file()
            or construction_path.is_symlink()
            or not source_wire_path.is_file()
            or source_wire_path.is_symlink()
        ):
            raise ValueError("canonical Source dimension projection is unavailable")
        construction = json.loads(construction_path.read_text(encoding="utf-8"))
        source_wire = json.loads(source_wire_path.read_text(encoding="utf-8"))
        allocator = (
            construction.get("allocator")
            if isinstance(construction, Mapping)
            else None
        )
        canonical_dimensions = (
            allocator.get("x") if isinstance(allocator, Mapping) else None
        )
        aliases = (
            source_wire.get("provider_dimension_aliases")
            if isinstance(source_wire, Mapping)
            else None
        )
        return provider_dimension_alias_map(
            aliases,
            canonical_dimensions=canonical_dimensions,
        )

    def _mounted_semantic_source_tensor_handles(self) -> frozenset[str]:
        """Read the closed Source ``t*`` set used by graph ingress checks."""

        return frozenset(self._mounted_semantic_source_tensor_declarations())

    def _mounted_semantic_source_tensor_value_types(
        self,
    ) -> dict[str, tuple[str, tuple[str | int, ...]]]:
        """Read Source tensor types for bounded graph shape consistency.

        This is the same mounted seed used to validate source ``t*``
        references.  Its canonical ``x*`` dimensions are projected to the
        provider's pre-allocation ``dim*`` names solely for comparison with a
        provider-owned operation result type; no type is authored, repaired,
        or exposed through a second MCP resource.
        """

        try:
            result = self._mounted_semantic_source_tensor_declarations()
        except ValueError as exc:
            detail = str(exc)
            if detail == "canonical Source tensor declarations are unavailable":
                raise ValueError("canonical Source tensor types are unavailable") from exc
            if detail == "canonical Source tensor declarations are ambiguous":
                raise ValueError("canonical Source tensor types are ambiguous") from exc
            raise ValueError("canonical Source tensor types are malformed") from exc
        if frozenset(result) != self._semantic_source_tensor_handles:
            raise ValueError("canonical Source tensor type/handle projections disagree")
        return result

    def _mounted_semantic_input_tensor_handles(self) -> frozenset[str]:
        """Return only Source values legal as direct semantic output choices.

        A provider can return an existing logical input for an alias/inout
        output, but it cannot select an output declaration itself as though it
        were an SSA producer.  This is a narrow Source-seed projection, not a
        second output-record constructor.
        """

        path = self.config.input_dir / "schema" / "canonical_source_wire_bundle.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical Source input declarations are unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        seed = bundle.get("artifact_seed") if isinstance(bundle, Mapping) else None
        j = seed.get("j") if isinstance(seed, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        rows = function.get("inputs") if isinstance(function, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError("canonical Source input declarations are malformed")
        names = {
            row.get("name")
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("name"), str)
            and re.fullmatch(r"t[0-9]+", row["name"]) is not None
        }
        if len(names) != len(rows):
            raise ValueError("canonical Source input declarations are malformed")
        return frozenset(names)

    def _mounted_semantic_output_tensor_value_types(
        self,
    ) -> tuple[tuple[str, tuple[str | int, ...]], ...]:
        """Return Source output types in the sole ABI output order.

        The provider still chooses one existing semantic value for each output.
        This ordered type projection only lets the MCP write boundary reject a
        choice which the canonical output constructor must reject later; it
        does not choose, rename, or materialize an output value.
        """

        path = self.config.input_dir / "schema" / "canonical_source_wire_bundle.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical Source output declarations are unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        seed = bundle.get("artifact_seed") if isinstance(bundle, Mapping) else None
        j = seed.get("j") if isinstance(seed, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        rows = function.get("outputs") if isinstance(function, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError("canonical Source output declarations are malformed")
        result: list[tuple[str, tuple[str | int, ...]]] = []
        for row in rows:
            name = row.get("name") if isinstance(row, Mapping) else None
            value_type = (
                self._semantic_source_tensor_value_types.get(name)
                if isinstance(name, str)
                else None
            )
            if value_type is None:
                raise ValueError("canonical Source output declarations are malformed")
            result.append(value_type)
        if len(result) != len(self._semantic_construction_slots["outputs"]):
            raise ValueError("canonical Source output declarations are malformed")
        return tuple(result)

    def _semantic_slot_receipt(self) -> dict[str, Any]:
        """Return the bounded required-slot view for ``begin_candidate``."""

        typed_output_requirements: list[dict[str, Any]] = []
        source_owner_producer_requirements: list[dict[str, Any]] = []
        source_logical_output_assembly_requirements: list[dict[str, Any]] = []
        # The provider has already read the Source-order transfer ledger before
        # begin_candidate.  That ledger is the sole owner of complete
        # source_owner_authoring_outline graphs and of access/output wire
        # slots.  Repeating those rows here used to make one begin receipt
        # large enough for the Responses observation wrapper to elide its
        # middle, which hid the typed output obligations from the graph author.
        # Keep only graph-phase facts in this receipt; the private construction
        # slots remain unchanged and the later mandatory transfer reread still
        # supplies every access/output row.
        source_access_ordinals = {
            row.get("ordinal")
            for row in self._semantic_construction_slots["accesses"]
            if isinstance(row, Mapping) and is_canonical_ordinal(row.get("ordinal"))
        }
        for ordinal, (dtype, shape) in enumerate(
            self._semantic_source_output_tensor_value_types
        ):
            requirement = {
                "ordinal": ordinal,
                "dtype": dtype,
                "shape": list(shape),
            }
            partition_requirement = self._semantic_construction_slots[
                "outputs"
            ][ordinal].get("source_store_partition_requirement")
            if partition_requirement is not None:
                assembly = partition_requirement.get("assembly")
                if isinstance(assembly, Mapping):
                    source_logical_output_assembly_requirements.append(
                        {
                            "output_ordinal": ordinal,
                            "output": partition_requirement.get("output"),
                            "requirement": copy.deepcopy(assembly),
                        }
                    )
            source_owner_authoring_outlines = self._semantic_construction_slots[
                "outputs"
            ][ordinal].get("source_owner_authoring_outlines")
            if source_owner_authoring_outlines is not None:
                # Keep the graph-phase obligation adjacent to the typed output
                # requirement, but do not duplicate the owner DAG.  The DAG is
                # already exposed in ``source_owner_authoring_outlines``; this
                # compact index binds its Source output ordinal, owner ordinal,
                # and exact typed root to the operation-graph phase.  It is a
                # projection of existing Source owner evidence only: it never
                # allocates a J result, chooses a producer, or proves a
                # partition.
                for owner_row in source_owner_authoring_outlines:
                    if not isinstance(owner_row, Mapping):
                        raise ValueError(
                            "canonical Source owner authoring outline is malformed"
                        )
                    owner_ordinal = owner_row.get("ordinal")
                    outline = owner_row.get("outline")
                    graph = outline.get("graph") if isinstance(outline, Mapping) else None
                    if (
                        not is_canonical_ordinal(owner_ordinal)
                        or not isinstance(graph, list)
                        or not graph
                        or owner_ordinal not in source_access_ordinals
                    ):
                        raise ValueError(
                            "canonical Source owner producer outline is malformed"
                        )
                    root = graph[-1]
                    if (
                        not isinstance(root, (list, tuple))
                        or len(root) != 5
                        or not isinstance(root[0], str)
                        or not isinstance(root[1], str)
                        or not isinstance(root[2], list)
                        or any(
                            not (
                                isinstance(extent, str) and bool(extent)
                                or isinstance(extent, int)
                                and not isinstance(extent, bool)
                                and extent >= 0
                            )
                            for extent in root[2]
                        )
                        or not isinstance(root[4], list)
                    ):
                        raise ValueError(
                            "canonical Source owner producer root is malformed"
                        )
                    provider_root_shape = provider_shape_from_canonical(
                        root[2], self._semantic_provider_dimension_aliases
                    )
                    source_owner_producer_requirements.append(
                        {
                            "output_ordinal": ordinal,
                            "owner_ordinal": owner_ordinal,
                            "outline_field": "source_owner_authoring_outlines",
                            "producer_scope": (
                                "physical_store_slice"
                                if partition_requirement is not None
                                else "logical_output_value"
                            ),
                            "graph_node_count": len(graph),
                            # The complete DAG is already present in the
                            # transfer-ledger access row with this ordinal.
                            # This locator is Source evidence, not a guessed
                            # owner or a provider-submitted token.
                            "transfer_access_ordinal": owner_ordinal,
                            "root_operation": root[0],
                            "root_type": {
                                "dtype": root[1],
                                # Owner outlines are stored in canonical x*
                                # dimensions.  The provider graph is authored
                                # in dim* dimensions; exposing x* here made a
                                # valid slice root look incompatible with the
                                # graph ingress type contract.
                                "shape": provider_root_shape,
                            },
                        }
                    )
            typed_output_requirements.append(requirement)
        return {
            "operation_graph": {
                "source_operation_facts": self._semantic_entry_operation_fact_count,
                "non_boundary_source_operation_facts": (
                    self._semantic_entry_non_boundary_operation_fact_count
                ),
                "required_operation_families": dict(
                    self._semantic_entry_required_operation_families
                ),
                "required_operation_attributes": [
                    {
                        "op": operation,
                        "attribute": attribute,
                        "value": value,
                        "count": count,
                    }
                    for operation, attribute, value, count in (
                        self._semantic_entry_required_operation_attributes
                    )
                ],
                # This is the already-validated flat Source obligation view.
                # It is repeated once at the graph boundary so the provider
                # does not have to reconstruct or remember a 28-leaf Source
                # spine from an earlier brief.  The rows remain read-only
                # Source facts; they do not allocate provider labels or
                # authorize guessed operands/producers.
                "source_operation_requirements": copy.deepcopy(
                    list(self._semantic_entry_operation_requirements)
                ),
                "typed_output_requirements": typed_output_requirements,
                "source_owner_producer_requirements": (
                    source_owner_producer_requirements
                ),
                "source_logical_output_assembly_requirements": (
                    source_logical_output_assembly_requirements
                ),
                "obligations": [
                    copy.deepcopy(obligation)
                    for obligation in self._semantic_construction_slots[
                        "obligations"
                    ]
                    if obligation.get("kind")
                    in {
                        "operation_graph",
                        "source_owner_producer",
                        "physical_store_partition",
                    }
                ],
                "requirement": (
                    "represent the complete Source calculation spine and every "
                    "listed Source-authorized typed owner DAG from the matching "
                    "transfer-ledger access rows; index and mask setup alone "
                    "are not a complete operation graph"
                ),
            },
            "completion_contract": {
                "accesses": {
                    "count": len(self._semantic_construction_slots["accesses"]),
                    "source": "semantic_transfer_after_operation_graph",
                },
                "outputs": {
                    "count": len(self._semantic_construction_slots["outputs"]),
                    "source": "semantic_transfer_after_operation_graph",
                },
                "obligations": {
                    "count": len(
                        self._semantic_construction_slots["obligations"]
                    ),
                    "authority": "source_construction",
                    "resolution": "host_validator_receipt_only",
                    "mcp_pass_scope": (
                        "wire_or_relation_preflight_unless_final_admission_receipt_is_explicit"
                    ),
                },
            },
        }

    def _semantic_access_slot_diagnostic(
        self, accesses: Sequence[Any]
    ) -> tuple[str, dict[str, Any]] | None:
        """Reject an incomplete/wrong-order access declaration immediately.

        This is a slot-completion check only.  The canonical semantic
        CandidateCheck still owns row grammar, source fact compatibility, and
        all cross-field acceptance at ``finalize_candidate``.
        """

        expected = self._semantic_construction_slots["accesses"]
        if len(accesses) != len(expected):
            return (
                "access_declaration_must_cover_each_source_occurrence",
                {
                    "expected_row_count": len(expected),
                    "actual_row_count": len(accesses),
                },
            )
        predicate_choices: tuple[frozenset[str], ...] | None = None
        consumer_choices: tuple[frozenset[str], ...] | None = None
        for index, (slot, row) in enumerate(zip(expected, accesses, strict=True)):
            if not isinstance(row, Mapping):
                return (
                    "access_declaration_rows_must_be_json_objects",
                    {"row_index": index},
                )
            if row.get("access_label") != slot["access_label"]:
                return (
                    "access_declaration_must_follow_source_ordinal",
                    {
                        "row_index": index,
                        "expected_access_label": slot["access_label"],
                        "actual_access_label": row.get("access_label"),
                    },
                )
            if row.get("ordinal") != slot["ordinal"]:
                return (
                    "access_declaration_must_follow_source_ordinal",
                    {
                        "row_index": index,
                        "expected_ordinal": slot["ordinal"],
                        "actual_ordinal": row.get("ordinal"),
                    },
                )
            if "id" in row:
                return (
                    "access_declaration_must_not_use_id_alias",
                    {"row_index": index, "field": "id"},
                )
            missing_fields = [
                field for field in slot["required_fields"] if field not in row
            ]
            if missing_fields:
                return (
                    "access_declaration_is_missing_required_source_slot_field",
                    {
                        "row_index": index,
                        "required_fields": list(slot["required_fields"]),
                        # Field names and the row ordinal are structural
                        # receipts, not a retained rejected declaration.  The
                        # provider already owns the declaration and receives
                        # the canonical Source wire template separately.
                        "missing_fields": missing_fields,
                    },
                )
            template = slot.get("wire_template")
            if not isinstance(template, Mapping):
                return (
                    "canonical_source_access_wire_template_is_malformed",
                    {"row_index": index},
                )
            provider_owned = slot.get("provider_owned_fields", ())
            provider_owned_expressions = slot.get(
                "provider_owned_expression_fields", ()
            )
            if not isinstance(provider_owned, (list, tuple)) or any(
                not isinstance(field, str) for field in provider_owned
            ):
                return (
                    "canonical_source_access_wire_template_is_malformed",
                    {"row_index": index},
                )
            if not isinstance(provider_owned_expressions, (list, tuple)) or any(
                not isinstance(field, str)
                or field not in slot["required_fields"]
                or field not in template
                or field in provider_owned
                for field in provider_owned_expressions
            ):
                return (
                    "canonical_source_access_wire_template_is_malformed",
                    {"row_index": index},
                )
            # A transfer row's kind/tensor/index/predicate/fill is Source
            # authority, except an explicitly unresolved masked predicate:
            # Source owns its presence but the provider owns the local J label
            # selected for the boolean result.  Reject all other drift at the
            # MCP declaration that introduced it instead of waiting for the
            # ledger join in CandidateCheck.  Host-only absent fields (for
            # example a store's null fill) remain absent from the narrow
            # provider wire, hence the required-fields intersection.
            for field in (
                "kind",
                "tensor",
                "index_expression",
                "predicate",
                "fill_value",
                "node_ref",
                "effect_kind",
                "effect_target",
                "effect_node_ref",
            ):
                if field in provider_owned_expressions:
                    value = row.get(field)
                    if not isinstance(value, str) or not value or value.startswith(
                        "<"
                    ):
                        return (
                            "access_declaration_provider_index_expression_must_be_local_expression",
                            {"row_index": index, "field": field},
                        )
                    if (
                        field == "index_expression"
                        and isinstance(
                            slot.get("source_index_expression_requirement"), str
                        )
                        and value
                        != slot["source_index_expression_requirement"]
                    ):
                        return (
                            "access_declaration_must_match_source_index_requirement",
                            {"row_index": index, "field": field},
                        )
                    continue
                if field in provider_owned:
                    value = row.get(field)
                    if (
                        not isinstance(value, str)
                        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", value) is None
                        or value.startswith("<")
                    ):
                        return (
                            (
                                "access_declaration_provider_node_ref_must_be_local_label"
                                if field in {"node_ref", "effect_node_ref"}
                                else "access_declaration_provider_predicate_must_be_local_label"
                            ),
                            {
                                "row_index": index,
                                "field": field,
                                "expected": (
                                    "a provider-local operation id"
                                    if field in {"node_ref", "effect_node_ref"}
                                    else "a provider-local boolean result label"
                                ),
                            },
                        )
                    if field == "predicate":
                        if predicate_choices is None:
                            buffer = self._semantic_construction_buffer
                            j = buffer.get("j") if isinstance(buffer, Mapping) else None
                            intent = j.get("intent") if isinstance(j, Mapping) else None
                            function = (
                                intent.get("intent_function")
                                if isinstance(intent, Mapping)
                                else None
                            )
                            body = (
                                function.get("body")
                                if isinstance(function, Mapping)
                                else None
                            )
                            predicate_choices = (
                                self._semantic_access_predicate_choices(body)
                                if isinstance(body, list)
                                else None
                            )
                        if (
                            predicate_choices is None
                            or len(predicate_choices) != len(expected)
                        ):
                            return (
                                "canonical_source_access_wire_template_is_malformed",
                                {"row_index": index, "field": field},
                            )
                        if value not in predicate_choices[index].compatible_labels:
                            return (
                                "access_declaration_provider_predicate_must_select_compatible_boolean_result",
                                {"row_index": index, "field": field},
                            )
                    if field in {"node_ref", "effect_node_ref"}:
                        if consumer_choices is None:
                            buffer = self._semantic_construction_buffer
                            j = buffer.get("j") if isinstance(buffer, Mapping) else None
                            intent = j.get("intent") if isinstance(j, Mapping) else None
                            function = (
                                intent.get("intent_function")
                                if isinstance(intent, Mapping)
                                else None
                            )
                            body = (
                                function.get("body")
                                if isinstance(function, Mapping)
                                else None
                            )
                            consumer_choices = (
                                self._semantic_access_consumer_choices(body)
                                if isinstance(body, list)
                                else None
                            )
                        if (
                            consumer_choices is None
                            or len(consumer_choices) != len(expected)
                        ):
                            return (
                                "canonical_source_access_wire_template_is_malformed",
                                {"row_index": index, "field": field},
                            )
                        if value not in consumer_choices[index]:
                            return (
                                "access_declaration_provider_node_ref_must_select_direct_consumer",
                                {
                                    "row_index": index,
                                    "field": field,
                                    "provider_local_operation_labels": sorted(
                                        consumer_choices[index]
                                    )[:64],
                                },
                            )
                    continue
                if field in slot["required_fields"] and row.get(field) != template.get(field):
                    return (
                        "access_declaration_must_match_source_wire_template",
                        {
                            "row_index": index,
                            "field": field,
                            "category": "source_wire_value_mismatch",
                        },
                    )
            relation_error = self._semantic_access_provider_relation_error(
                slot, row
            )
            if relation_error is not None:
                return (
                    relation_error,
                    {
                        "row_index": index,
                        "field": "effect_node_ref"
                        if relation_error
                        == "access_declaration_effect_node_owner_must_match_access_owner"
                        else "provider_owned_relations",
                    },
                )
        return None

    @staticmethod
    def _semantic_access_provider_relation_error(
        slot: Mapping[str, Any], row: Mapping[str, Any]
    ) -> str | None:
        """Validate the Source-issued provider relation map for one access.

        ``provider_owned_relations`` is part of the mounted Source wire.  The
        method performs only the relation check already authorized there; it
        never allocates a node or infers an equality.  The current contract
        has one relation (``effect_node_ref`` must equal ``node_ref``), but
        consuming the map generically keeps future relation extensions from
        creating another parallel hard-coded validator.
        """

        relations = slot.get("provider_owned_relations")
        provider_owned = slot.get("provider_owned_fields")
        if not isinstance(relations, Mapping) or not isinstance(
            provider_owned, (list, tuple)
        ):
            return "canonical_source_access_wire_template_is_malformed"
        for field, relation in relations.items():
            if (
                not isinstance(field, str)
                or field not in provider_owned
                or not isinstance(relation, Mapping)
                or set(relation) != {"must_equal", "reference"}
                or not isinstance(relation.get("must_equal"), str)
                or relation.get("must_equal") not in provider_owned
                or not isinstance(relation.get("reference"), str)
                or not relation.get("reference")
            ):
                return "canonical_source_access_wire_template_is_malformed"
            reference = relation["must_equal"]
            if row.get(field) != row.get(reference):
                if field == "effect_node_ref" and reference == "node_ref":
                    return (
                        "access_declaration_effect_node_owner_must_match_access_owner"
                    )
                return "canonical_source_access_wire_template_is_malformed"
        return None

    def _semantic_access_failure_template(self, accesses: Sequence[Any]) -> dict[str, Any] | None:
        """Return a value-free shape for the first failed access row.

        The initial authoring brief already carries the exact Source wire row.
        A failed declaration must not echo either that row's literal values or
        the rejected declaration back into provider feedback.
        """

        def safe_template(slot: Mapping[str, Any]) -> dict[str, Any]:
            required = slot.get("required_fields")
            provider_owned = slot.get("provider_owned_fields", ())
            provider_owned_expressions = slot.get(
                "provider_owned_expression_fields", ()
            )
            if not isinstance(required, list):
                return {}
            owned = set(provider_owned) if isinstance(provider_owned, (list, tuple)) else set()
            expression_owned = (
                set(provider_owned_expressions)
                if isinstance(provider_owned_expressions, (list, tuple))
                else set()
            )
            result: dict[str, Any] = {}
            for field in required:
                if field in expression_owned:
                    result[field] = "<provider_authored_index_expression>"
                elif field == "access_label":
                    result[field] = slot.get("access_label")
                elif field == "ordinal":
                    result[field] = slot.get("ordinal")
                elif field in owned:
                    result[field] = "<provider_local_value>"
                else:
                    result[field] = "<copy_from_initial_source_wire>"
            source_index_requirement = slot.get(
                "source_index_expression_requirement"
            )
            if isinstance(source_index_requirement, str):
                result["index_expression_requirement"] = source_index_requirement
            return result

        expected = self._semantic_construction_slots["accesses"]
        for index, slot in enumerate(expected):
            row = accesses[index] if index < len(accesses) else None
            if (
                not isinstance(row, Mapping)
                or row.get("access_label") != slot["access_label"]
                or row.get("ordinal") != slot["ordinal"]
                or "id" in row
                or any(field not in row for field in slot["required_fields"])
            ):
                return safe_template(slot)
            template = slot.get("wire_template")
            if isinstance(template, Mapping):
                provider_owned = slot.get("provider_owned_fields", ())
                provider_owned_expressions = slot.get(
                    "provider_owned_expression_fields", ()
                )
                if not isinstance(provider_owned, (list, tuple)) or not isinstance(
                    provider_owned_expressions, (list, tuple)
                ):
                    return safe_template(slot)
                expression_owned = set(provider_owned_expressions)
                for field in (
                    "kind",
                    "tensor",
                    "index_expression",
                    "predicate",
                    "fill_value",
                    "node_ref",
                    "effect_kind",
                    "effect_target",
                    "effect_node_ref",
                ):
                    if field in expression_owned:
                        value = row.get(field)
                        if (
                            not isinstance(value, str)
                            or not value
                            or value.startswith("<")
                        ):
                            return safe_template(slot)
                    elif field in provider_owned:
                        value = row.get(field)
                        if (
                            not isinstance(value, str)
                            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", value) is None
                            or value.startswith("<")
                        ):
                            return safe_template(slot)
                    elif field in slot["required_fields"] and row.get(field) != template.get(field):
                        return safe_template(slot)
                if self._semantic_access_provider_relation_error(slot, row) is not None:
                    return safe_template(slot)
        return None

    def _semantic_access_failure_context(self) -> list[dict[str, Any]]:
        """Project compact safe templates for access-declaration repair.

        A failed access write must be rebuilt as one complete Source-ordered
        array, but repeating the full construction-slot records on every
        correction turn needlessly expands the model context.  Keep only the
        fixed row identity, required/provider-owned field names, and a
        value-free row template.  This projection never copies a rejected
        declaration or chooses a predicate, owner, coordinate, or local
        operation value.
        """

        result: list[dict[str, Any]] = []
        for slot in self._semantic_construction_slots["accesses"]:
            if not isinstance(slot, Mapping):
                continue
            required = slot.get("required_fields")
            provider_owned = slot.get("provider_owned_fields", ())
            provider_owned_expressions = slot.get(
                "provider_owned_expression_fields", ()
            )
            template = slot.get("wire_template")
            if not isinstance(required, list) or not isinstance(template, Mapping):
                continue
            owned = set(provider_owned) if isinstance(provider_owned, (list, tuple)) else set()
            expression_owned = (
                set(provider_owned_expressions)
                if isinstance(provider_owned_expressions, (list, tuple))
                else set()
            )
            safe_template: dict[str, Any] = {}
            for field in required:
                if field in expression_owned:
                    safe_template[field] = "<provider_authored_index_expression>"
                elif field in owned:
                    safe_template[field] = "<provider_local_value>"
                elif field in template:
                    safe_template[field] = copy.deepcopy(template[field])
                else:
                    safe_template[field] = "<copy_from_initial_source_wire>"
            source_index_requirement = slot.get(
                "source_index_expression_requirement"
            )
            if isinstance(source_index_requirement, str):
                safe_template["index_expression_requirement"] = (
                    source_index_requirement
                )
            result.append(
                {
                    "access_label": slot.get("access_label"),
                    "ordinal": slot.get("ordinal"),
                    "required_fields": list(required),
                    "provider_owned_fields": sorted(owned),
                    "provider_owned_expression_fields": sorted(expression_owned),
                    "provider_owned_relations": copy.deepcopy(
                        dict(slot.get("provider_owned_relations", {}))
                    )
                    if isinstance(slot.get("provider_owned_relations"), Mapping)
                    else {},
                    "wire_template": safe_template,
                }
            )
        return result

    def _semantic_output_value_types(
        self,
        function: Mapping[str, Any],
    ) -> tuple[set[str], dict[str, tuple[str, tuple[str | int, ...]]]] | None:
        """Project provider labels which can legally escape as outputs.

        All local results remain useful for distinguishing an unknown label
        from a region-local one.  The typed map is deliberately narrower: it
        contains direct Source inputs and top-level operation results whose
        complete type is already authored in this same graph.  This mirrors
        the visibility/type preconditions of ``canonical_output_records``
        without constructing an output mapping or selecting a value.
        """

        def valid_shape(value: Any) -> bool:
            return isinstance(value, list) and all(
                isinstance(extent, str) and bool(extent)
                or isinstance(extent, int)
                and not isinstance(extent, bool)
                and extent >= 0
                for extent in value
            )

        rows = semantic_operation_wire_rows(function)
        if rows is None:
            return None
        all_local_results: set[str] = set()
        visible_types = {
            name: value_type
            for name, value_type in self._semantic_source_tensor_value_types.items()
            if name in self._semantic_source_input_tensor_handles
        }
        for path, operation in rows:
            raw_result = operation.get("result")
            results = raw_result if isinstance(raw_result, list) else (raw_result,)
            all_local_results.update(
                result for result in results if isinstance(result, str)
            )
            if not isinstance(path, str) or re.fullmatch(
                r"body\[[0-9]+\]", path
            ) is None:
                continue
            operation_type = operation.get("type")
            if (
                isinstance(raw_result, str)
                and isinstance(operation_type, Mapping)
                and isinstance(operation_type.get("dtype"), str)
                and valid_shape(operation_type.get("shape"))
            ):
                visible_types[raw_result] = (
                    operation_type["dtype"],
                    tuple(operation_type["shape"]),
                )
                continue
            if operation.get("op") != "loop" or not isinstance(raw_result, list):
                continue
            carries = operation.get("carry")
            if not isinstance(carries, list) or len(carries) != len(raw_result):
                continue
            for result, carry in zip(raw_result, carries, strict=True):
                carry_type = carry.get("type") if isinstance(carry, Mapping) else None
                if (
                    isinstance(result, str)
                    and isinstance(carry_type, Mapping)
                    and isinstance(carry_type.get("dtype"), str)
                    and valid_shape(carry_type.get("shape"))
                ):
                    visible_types[result] = (
                        carry_type["dtype"],
                        tuple(carry_type["shape"]),
                    )
        return all_local_results, visible_types

    def _semantic_graph_missing_output_type(
        self,
        body: Sequence[Any],
    ) -> tuple[int, tuple[str, tuple[str | int, ...]]] | None:
        """Return the first Source output type absent from graph-visible values.

        This is an existence proof only.  Source inputs and top-level typed
        results are the same function-visible values accepted later by output
        selection.  The host neither chooses one of those values nor creates a
        producer; a missing type sends the complete graph back to its provider
        while graph authoring is still the active transaction phase.
        """

        projection = self._semantic_output_value_types({"body": list(body)})
        if projection is None:
            return None
        _local_results, visible_types = projection
        available_types = frozenset(visible_types.values())
        for ordinal, required_type in enumerate(
            self._semantic_source_output_tensor_value_types
        ):
            if required_type not in available_types:
                return ordinal, required_type
        return None

    def _semantic_graph_matching_output_type_count(
        self,
        body: Sequence[Any],
        required_type: tuple[str, tuple[str | int, ...]],
    ) -> int:
        """Count already-visible values with one exact Source output type.

        This is diagnostic projection only.  It never selects a producer,
        creates a result, or treats a matching type as proof of output
        ownership; the later output/Source checks remain authoritative.
        """

        projection = self._semantic_output_value_types({"body": list(body)})
        if projection is None:
            return 0
        _local_results, visible_types = projection
        return sum(
            value_type == required_type
            for value_type in visible_types.values()
        )

    def _semantic_output_assembly_requirements(self) -> tuple[Mapping[str, Any], ...]:
        """Return only Source-proven logical-output assembly requirements."""

        requirements: list[Mapping[str, Any]] = []
        for ordinal, slot in enumerate(self._semantic_construction_slots["outputs"]):
            if not isinstance(slot, Mapping):
                continue
            partition = slot.get("source_store_partition_requirement")
            assembly = partition.get("assembly") if isinstance(partition, Mapping) else None
            if not isinstance(assembly, Mapping):
                continue
            requirements.append(
                {
                    "output_ordinal": ordinal,
                    "output": partition.get("output"),
                    "requirement": assembly,
                }
            )
        return tuple(requirements)

    def _semantic_graph_missing_output_assembly(
        self,
        body: Sequence[Any],
    ) -> Mapping[str, Any] | None:
        """Require the exact typed concat that closes a proven store partition.

        This is deliberately narrower than accepting any same-shaped value:
        the assembly operation, axis, ordered operand count/types and result
        type all come from the already validated Source partition projection.
        Provider-local labels remain opaque and are never selected by the host.
        """

        requirements = self._semantic_output_assembly_requirements()
        if not requirements:
            return None
        rows = semantic_operation_wire_rows({"body": list(body)})
        projection = self._semantic_output_value_types({"body": list(body)})
        if rows is None or projection is None:
            return requirements[0]
        local_results, visible_types = projection
        for requirement in requirements:
            assembly = requirement.get("requirement")
            if not isinstance(assembly, Mapping):
                return requirement
            result_type = assembly.get("result_type")
            operand_type = assembly.get("operand_type")
            operand_count = assembly.get("operand_store_ordinals")
            axis = assembly.get("axis")
            if (
                not isinstance(result_type, Mapping)
                or not isinstance(operand_type, Mapping)
                or not isinstance(operand_count, list)
                or type(axis) is not int
            ):
                return requirement
            expected_result = (
                result_type.get("dtype"),
                tuple(result_type.get("shape", ())),
            )
            expected_operand = (
                operand_type.get("dtype"),
                tuple(operand_type.get("shape", ())),
            )
            found = False
            for path, operation in rows:
                if (
                    not isinstance(path, str)
                    or re.fullmatch(r"body\[[0-9]+\]", path) is None
                    or operation.get("op") != "concat"
                    or not isinstance(operation.get("attrs"), Mapping)
                    or operation["attrs"].get("axis") != axis
                    or not isinstance(operation.get("operands"), list)
                    or len(operation["operands"]) != len(operand_count)
                    or not all(
                        isinstance(operand, str)
                        and operand in local_results
                        and visible_types.get(operand) == expected_operand
                        for operand in operation["operands"]
                    )
                    or len(set(operation["operands"])) != len(operation["operands"])
                ):
                    continue
                result = operation.get("result")
                if (
                    not isinstance(result, str)
                    or visible_types.get(result) != expected_result
                ):
                    continue
                found = True
                break
            if not found:
                return requirement
        return None

    @staticmethod
    def _semantic_graph_operation_family_counts(
        rows: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> dict[str, int]:
        """Count canonical operation families from one shared wire walk.

        Graph ingress and its safe rejection diagnostic must observe the same
        lexical operation rows.  Keeping this count in one helper prevents a
        future nested-region change from making ingress reject one count while
        feedback reports another.  The helper is diagnostic/structural only:
        it neither maps Source facts nor creates a missing operation.
        """

        counts: dict[str, int] = {}
        for _path, operation in rows:
            name = operation.get("op")
            if isinstance(name, str):
                counts[name] = counts.get(name, 0) + 1
        return counts

    def _semantic_access_predicate_relations(
        self,
    ) -> tuple[Mapping[str, Any], ...] | None:
        """Project one typed predicate relation for every Source access slot.

        The MCP graph preflight and the canonical computed-guard adapter must
        consume the same relation: provider-local predicate dtype, the Source
        access target type, and the unique rank-insertion rule.  Keeping this
        relation at the slot boundary avoids a load-only shortcut here that
        would disagree with the later store/update guard validation.  This
        helper only exposes already-mounted types; it never selects a result,
        creates a predicate, or infers a target shape.
        """

        relations: list[Mapping[str, Any]] = []
        for ordinal, slot in enumerate(self._semantic_construction_slots["accesses"]):
            if not isinstance(slot, Mapping):
                return None
            provider_owned = slot.get("provider_owned_fields", ())
            template = slot.get("wire_template")
            if (
                not isinstance(provider_owned, (list, tuple))
                or not isinstance(template, Mapping)
            ):
                return None
            requires_predicate = "predicate" in provider_owned
            relation: dict[str, Any] = {
                "ordinal": ordinal,
                "requires_predicate": requires_predicate,
                "kind": template.get("kind"),
                "tensor": template.get("tensor"),
                "computed_guard": (
                    isinstance(template.get("semantic_guard_authorization"), str)
                    and bool(template.get("semantic_guard_authorization"))
                ),
            }
            if not requires_predicate:
                relations.append(relation)
                continue
            tensor = template.get("tensor")
            kind = template.get("kind")
            target_type = (
                self._semantic_source_tensor_value_types.get(tensor)
                if isinstance(tensor, str)
                else None
            )
            if target_type is None or kind not in {"load", "store", "update"}:
                return None
            if relation["computed_guard"] and not computed_guard_projection_supported(kind):
                # The Source ledger may identify a computed guard on an
                # unsupported access kind, but no downstream canonical
                # representation exists for it.  Keep the graph ingress
                # fail-closed instead of allowing a later phase to disagree.
                relation["shape_rule"] = "computed_guard_projection_unsupported_for_access_kind"
                relations.append(relation)
                continue
            relation["target_dtype"] = target_type[0]
            relation["target_shape"] = tuple(target_type[1])
            relation["shape_rule"] = (
                "i1_with_one_unique_exact_rank_insertion"
            )
            relations.append(relation)
        return tuple(relations)

    def _semantic_access_predicate_choices(
        self,
        body: Sequence[Any],
    ) -> tuple[_SemanticAccessPredicateChoices, ...] | None:
        """Return typed local predicate candidates per Source access slot.

        The graph owns result labels/types and the Source wire owns each
        access target.  This projection only asks whether an already authored
        boolean has the same unique guard projection later consumed by the
        canonical constructor. It also records a count-only distinction
        between a missing boolean result and booleans whose shapes do not
        satisfy that relation. It never selects a label or creates a guard.
        """

        projection = self._semantic_output_value_types({"body": list(body)})
        if projection is None:
            return None
        local_results, visible_types = projection
        relations = self._semantic_access_predicate_relations()
        if relations is None:
            return None
        choices: list[_SemanticAccessPredicateChoices] = []
        for relation in relations:
            if not relation.get("requires_predicate"):
                choices.append(_SemanticAccessPredicateChoices(frozenset(), 0, 0))
                continue
            if (
                relation.get("computed_guard")
                and not computed_guard_projection_supported(relation.get("kind"))
            ):
                choices.append(_SemanticAccessPredicateChoices(frozenset(), 0, 0))
                continue
            target_shape = relation.get("target_shape")
            if not isinstance(target_shape, tuple):
                return None
            compatible: set[str] = set()
            boolean_result_count = 0
            for label in local_results:
                predicate_type = visible_types.get(label)
                if (
                    predicate_type is None
                    or not is_closed_boolean_dtype(predicate_type[0])
                ):
                    continue
                boolean_result_count += 1
                if unique_access_guard_projection_axes(
                    predicate_type[1], target_shape
                ) is not None:
                    compatible.add(label)
            choices.append(
                _SemanticAccessPredicateChoices(
                    compatible_labels=frozenset(compatible),
                    boolean_result_count=boolean_result_count,
                    incompatible_boolean_shape_count=(
                        boolean_result_count - len(compatible)
                    ),
                )
            )
        return tuple(choices)

    def _semantic_access_consumer_choices(
        self,
        body: Sequence[Any],
    ) -> tuple[frozenset[str], ...] | None:
        """Return Source-compatible local refs for access owner slots.

        A load's ``node_ref`` is a direct consumer of its tensor. An exact
        Source-authorized store/update ``node_ref`` is instead the root owner
        operation, identified here only by the Source-issued operation family;
        the later owner admission checks its full typed DAG and occurrence.
        """

        rows = semantic_operation_wire_rows({"body": list(body)})
        if rows is None:
            return None
        choices: list[frozenset[str]] = []
        for slot in self._semantic_construction_slots["accesses"]:
            if not isinstance(slot, Mapping):
                return None
            provider_owned = slot.get("provider_owned_fields", ())
            template = slot.get("wire_template")
            if (
                not isinstance(provider_owned, (list, tuple))
                or not isinstance(template, Mapping)
            ):
                return None
            if "node_ref" not in provider_owned:
                choices.append(frozenset())
                continue
            tensor = template.get("tensor")
            compatible: set[str] = set()
            kind = template.get("kind")
            if kind == "load":
                if not isinstance(tensor, str):
                    return None
                for _path, operation in rows:
                    operation_id = operation.get("id")
                    operands = operation.get("operands")
                    result = operation.get("result")
                    if (
                        isinstance(operation_id, str)
                        and isinstance(operands, list)
                        and isinstance(result, str)
                        and operands.count(tensor) == 1
                    ):
                        compatible.add(operation_id)
            elif kind in {"store", "update"}:
                owner_family = slot.get("source_owner_operation_family")
                if not isinstance(owner_family, str) or not owner_family:
                    return None
                for _path, operation in rows:
                    operation_id = operation.get("id")
                    if (
                        isinstance(operation_id, str)
                        and operation.get("op") == owner_family
                    ):
                        compatible.add(operation_id)
            else:
                return None
            choices.append(frozenset(compatible))
        return tuple(choices)

    def _semantic_graph_missing_access_predicate_type(
        self,
        body: Sequence[Any],
    ) -> tuple[int, int, int, int] | None:
        """Return the first masked access without a compatible graph value.

        The receipt separates "no typed boolean result exists" from "typed
        booleans exist but none has the exact Source target projection". Both
        remain rejecting conditions; the distinction identifies the failed
        projection edge without exposing a rejected operation row or label.
        """

        choices = self._semantic_access_predicate_choices(body)
        relations = self._semantic_access_predicate_relations()
        if choices is None or relations is None:
            return None
        for relation, choice in zip(relations, choices, strict=True):
            if relation.get("requires_predicate") and not choice.compatible_labels:
                ordinal = relation.get("ordinal")
                if not is_canonical_ordinal(ordinal):
                    return None
                return (
                    ordinal,
                    len(choice.compatible_labels),
                    choice.boolean_result_count,
                    choice.incompatible_boolean_shape_count,
                )
        return None

    def _semantic_graph_missing_access_consumer_id(
        self,
        body: Sequence[Any],
    ) -> tuple[int, int] | None:
        """Return the first guarded load without a named direct consumer."""

        choices = self._semantic_access_consumer_choices(body)
        if choices is None:
            return None
        for ordinal, (slot, labels) in enumerate(
            zip(
                self._semantic_construction_slots["accesses"],
                choices,
                strict=True,
            )
        ):
            provider_owned = slot.get("provider_owned_fields", ())
            if "node_ref" in provider_owned and not labels:
                return ordinal, 0
        return None

    def _semantic_output_slot_diagnostic(self, mappings: Sequence[Any]) -> str | None:
        """Reject an incomplete output selection before finalization."""

        expected = self._semantic_construction_slots["outputs"]
        if len(mappings) != len(expected):
            return "output_declaration_must_cover_each_source_output"
        buffer = self._semantic_construction_buffer
        j = buffer.get("j") if isinstance(buffer, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        value_projection = (
            self._semantic_output_value_types(function)
            if isinstance(function, Mapping)
            else None
        )
        if value_projection is None:
            return "output_declaration_requires_declared_operation_graph"
        local_results, visible_types = value_projection
        for index, row in enumerate(mappings):
            if not isinstance(row, Mapping):
                return "output_declaration_rows_must_be_json_objects"
            # The semantic write projection is deliberately narrower than the
            # final construction record.  ``output`` and ``node_ref`` belong
            # to the host allocator; accepting them here creates a second
            # output wire and postpones a deterministic error until
            # CandidateCheck.
            if set(row) != {"value"}:
                return "output_declaration_must_contain_only_value"
            if "value" not in row:
                return "output_declaration_is_missing_required_source_slot_field"
            value = row.get("value")
            if not isinstance(value, str) or (
                value not in local_results
                and value not in self._semantic_source_input_tensor_handles
            ):
                return "output_declaration_value_must_reference_source_input_or_local_result"
            if value not in visible_types:
                return "output_declaration_value_must_be_function_visible"
            if visible_types[value] != self._semantic_source_output_tensor_value_types[index]:
                return "output_declaration_value_type_must_match_source_output"
        return None

    def _semantic_output_failure_template(self, mappings: Sequence[Any]) -> dict[str, Any] | None:
        """Return the sole legal provider row for the first bad output slot."""

        expected = self._semantic_construction_slots["outputs"]
        for index, slot in enumerate(expected):
            row = mappings[index] if index < len(mappings) else None
            if not isinstance(row, Mapping) or set(row) != {"value"}:
                template = slot.get("wire_template")
                return copy.deepcopy(dict(template)) if isinstance(template, Mapping) else None
        return None

    def _semantic_output_failure_diagnostic(
        self,
        mappings: Sequence[Any],
        detail: str,
    ) -> dict[str, Any] | None:
        """Return safe provider-local output choices for a bad value selection.

        A Source output tensor is a boundary declaration, not an SSA producer.
        When that distinction rejects an output mapping, the only useful safe
        correction hint is the local results already declared by this same
        provider graph.  Do not expose candidate bytes, Source text, canonical
        identities, or host-created output records.
        """

        if detail not in {
            "output_declaration_value_must_reference_source_input_or_local_result",
            "output_declaration_value_must_be_function_visible",
            "output_declaration_value_type_must_match_source_output",
        }:
            return None
        buffer = self._semantic_construction_buffer
        j = buffer.get("j") if isinstance(buffer, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        value_projection = (
            self._semantic_output_value_types(function)
            if isinstance(function, Mapping)
            else None
        )
        if value_projection is None:
            return None
        local_results, visible_types = value_projection
        row_index = next(
            (
                index
                for index, row in enumerate(mappings)
                if not isinstance(row, Mapping)
                or not isinstance(row.get("value"), str)
                or row.get("value") not in local_results
                and row.get("value") not in self._semantic_source_input_tensor_handles
                or row.get("value") not in visible_types
                or index >= len(self._semantic_source_output_tensor_value_types)
                or visible_types.get(row.get("value"))
                != self._semantic_source_output_tensor_value_types[index]
            ),
            None,
        )
        if row_index is None or row_index >= len(self._semantic_source_output_tensor_value_types):
            return None
        expected_type = self._semantic_source_output_tensor_value_types[row_index]
        results = sorted(
            value
            for value, value_type in visible_types.items()
            if value in local_results and value_type == expected_type
        )
        return {
            "row_index": row_index,
            "field": "value",
            "provider_local_result_labels": results[:64],
        }

    def _normalize_host_owned_output_ordinals(
        self,
        mappings: Sequence[Any],
    ) -> tuple[list[Any], bool]:
        """Remove only redundant ABI ordinals from semantic output selections.

        Semantic output rows are ordered by their array position and the host
        owns the final ABI ordinal.  A provider sometimes echoes that ordinal
        from the transfer ledger, which is mechanically redundant but used to
        force another model turn.  Repair it only when every ordinal exactly
        matches the already-fixed Source order; any other extra field remains
        a fail-closed semantic-wire error.
        """

        repaired: list[Any] = []
        changed = False
        for ordinal, row in enumerate(mappings):
            if (
                isinstance(row, Mapping)
                and set(row) == {"ordinal", "value"}
                and row.get("ordinal") == ordinal
            ):
                repaired.append({"value": copy.deepcopy(row.get("value"))})
                changed = True
            else:
                repaired.append(row)
        return repaired, changed

    def _semantic_operation_graph_diagnostic(
        self,
        body: Sequence[Any],
    ) -> str | None:
        """Validate the provider operation envelope at the MCP write boundary.

        The canonical schema has two distinct names in an operation row:
        ``op`` is the closed J operation name, while ``id`` and ``result``
        are provider-local labels consumed by the host allocator.  Keeping
        this check here prevents a provider-local ``op0`` or a host-owned
        ``v*`` from surviving several MCP calls before CandidateCheck reports
        the same deterministic contract error.
        """

        try:
            namespace = json.loads(self._semantic_operation_namespace)
        except (TypeError, ValueError, json.JSONDecodeError):
            return "canonical_operation_namespace_is_malformed"
        operation_catalog = namespace.get("operation_catalog")
        if not isinstance(operation_catalog, list) or any(
            not isinstance(name, str) for name in operation_catalog
        ):
            return "canonical_operation_namespace_is_malformed"
        catalog = frozenset(operation_catalog)
        canonical_prefixes = frozenset({"n", "v", "x", "p", "c", "a", "e"})
        local_label = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")

        def valid_local_label(value: Any) -> bool:
            if not isinstance(value, str) or not local_label.fullmatch(value):
                return False
            return not any(
                re.fullmatch(rf"{prefix}[0-9]+", value)
                for prefix in canonical_prefixes
            )

        def visit(value: Any) -> str | None:
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, Mapping):
                        return "operation_graph_rows_must_be_json_objects"
                    diagnostic = visit(item)
                    if diagnostic is not None:
                        return diagnostic
                return None
            if not isinstance(value, Mapping):
                return "operation_graph_rows_must_be_json_objects"
            if "op" not in value:
                return "operation_row_must_include_canonical_op_field"
            operation = value.get("op")
            if not isinstance(operation, str) or operation not in catalog:
                return "operation_op_must_be_canonical_j_name"
            if operation not in STRUCTURED_INTENT_OPS:
                allowed = {"id", "op", "result", "operands", "attrs", "type"}
                if set(value) - allowed:
                    return "operation_must_use_only_canonical_leaf_fields"
                if "attrs" not in value or "type" not in value:
                    return "operation_must_include_required_leaf_fields"
                attrs = value.get("attrs")
                if not isinstance(attrs, Mapping):
                    return "operation_attrs_must_be_json_object"
                try:
                    attr_contract = operation_attr_contract(str(operation))
                except (KeyError, TypeError, ValueError):
                    # The operation catalog has already been checked above.
                    # A contract lookup failure is therefore host-only state,
                    # not something safe or useful to disclose to a provider.
                    return "canonical_operation_namespace_is_malformed"
                allowed_attrs = attr_contract.get("allowed")
                required_attrs = attr_contract.get("required")
                if not isinstance(allowed_attrs, frozenset) or not isinstance(
                    required_attrs, frozenset
                ):
                    return "canonical_operation_namespace_is_malformed"
                if set(attrs) - allowed_attrs:
                    return "operation_attrs_must_use_canonical_leaf_fields"
                if required_attrs - set(attrs):
                    return "operation_attrs_must_include_required_leaf_fields"
                properties = attr_contract.get("properties")
                if not isinstance(properties, Mapping):
                    return "canonical_operation_namespace_is_malformed"
                if any(
                    not _operation_attr_value_matches_schema(
                        attr_value,
                        properties.get(attr_name),
                    )
                    for attr_name, attr_value in attrs.items()
                ):
                    return "operation_attr_value_must_match_canonical_leaf_contract"
                if "operands" in value and not isinstance(value.get("operands"), list):
                    return "operation_operands_must_be_json_array"
            if "id" in value and value.get("id") is not None and not valid_local_label(value.get("id")):
                return "operation_id_must_be_provider_local_label"
            result = value.get("result")
            result_values = result if isinstance(result, list) else [result]
            if result is not None and any(
                not valid_local_label(item) for item in result_values
            ):
                return "operation_result_must_be_provider_local_label"
            # Only structured operation regions contain operation rows.  Do
            # not recurse into ordinary ``attrs`` or ``type`` mappings: doing
            # that would mistake a valid leaf object for an operation which
            # lacks ``op`` and reproduce the historical live/replay split.
            if operation == "loop":
                child = value.get("body")
                if child is not None:
                    diagnostic = visit(child)
                    if diagnostic is not None:
                        return diagnostic
            elif operation in {"effect_if", "value_if"}:
                for arm_name in ("positive", "negative"):
                    arm = value.get(arm_name)
                    if not isinstance(arm, Mapping):
                        continue
                    child = arm.get("body")
                    if child is not None:
                        diagnostic = visit(child)
                        if diagnostic is not None:
                            return diagnostic
            return None

        envelope_diagnostic = visit(list(body))
        if envelope_diagnostic is not None:
            return envelope_diagnostic

        # This is deliberately an ingress-only def-use check.  It rejects a
        # reference which cannot possibly be resolved by the host allocator,
        # but does not infer a missing producer, rewrite an ``arg*`` spelling,
        # or attempt structured SSA validation.  The canonical constructor
        # remains responsible for complete loop/branch/scope semantics.
        rows = semantic_operation_wire_rows({"body": list(body)})
        if rows is None:
            return "operation_graph_lexical_wire_is_malformed"
        # Give an incomplete Source obligation graph precedence over errors in
        # one of its submitted rows. Otherwise a skeletal declaration can
        # repeatedly repair a local detail (for example, operand arity) while
        # omitting nearly the entire Source producer graph. This is diagnostic
        # ordering only: a graph must still pass every row-level rule below.
        operation_counts = self._semantic_graph_operation_family_counts(rows)
        if any(
            operation_counts.get(name, 0) < required
            for name, required in self._semantic_entry_required_operation_families.items()
        ):
            return "operation_graph_must_cover_source_operation_families"
        for operation, attribute, value, required in (
            self._semantic_entry_required_operation_attributes
        ):
            actual = sum(
                1
                for _path, row in rows
                if row.get("op") == operation
                and isinstance(row.get("attrs"), Mapping)
                and row["attrs"].get(attribute) == value
            )
            if actual < required:
                return "operation_graph_must_cover_source_operation_attributes"
        visible = set(self._semantic_source_tensor_handles)
        value_types = dict(self._semantic_source_tensor_value_types)

        def tensor_shape(operation: Mapping[str, Any]) -> tuple[str | int, ...] | None:
            operation_type = operation.get("type")
            if (
                not isinstance(operation_type, Mapping)
                or not isinstance(operation_type.get("dtype"), str)
                or not isinstance(operation_type.get("shape"), list)
            ):
                return None
            shape = operation_type["shape"]
            if any(
                not (
                    isinstance(extent, str)
                    and bool(extent)
                    or isinstance(extent, int)
                    and not isinstance(extent, bool)
                    and extent >= 0
                )
                for extent in shape
            ):
                return None
            return tuple(shape)

        def provider_dimension_extent(value: Any) -> Any:
            """Accept only literals or Source-issued aliases on provider wire."""

            if is_provider_dimension_extent(
                value, self._semantic_provider_dimension_aliases.values()
            ):
                return value
            return None

        for _path, operation in rows:
            operands = operation.get("operands", ())
            if not isinstance(operands, list):
                return "operation_operands_must_be_json_array"
            for operand in operands:
                if not isinstance(operand, str):
                    return "operation_operand_must_reference_source_tensor_or_prior_local_result"
                if re.fullmatch(r"t[0-9]+", operand) is not None:
                    if operand not in self._semantic_source_tensor_handles:
                        return (
                            "operation_operand_must_reference_source_tensor_"
                            "or_prior_local_result"
                        )
                elif operand not in visible:
                    return (
                        "operation_operand_must_reference_source_tensor_"
                        "or_prior_local_result"
                    )
            # Resolve each supplied operand before judging the operation's
            # total arity.  A dangling provider label is independently
            # correctable even when it also makes a zero-input operation
            # over-arity; returning its precise def-use receipt preserves the
            # same declaration classification in MCP and direct replay.
            attrs = operation.get("attrs")
            if not operation_operands_match_arity(
                str(operation.get("op")),
                operands,
                attrs if isinstance(attrs, Mapping) else None,
            ):
                return "operation_operands_must_match_canonical_leaf_arity"
            result = operation.get("result")
            result_values = result if isinstance(result, list) else (result,)
            for value in result_values:
                if value is None:
                    continue
                if not isinstance(value, str) or value in visible:
                    return "operation_result_must_be_new_provider_local_label"
                # Host canonicalization can allocate n*/v* identities, but
                # it cannot infer the provider-owned logical result type.
                # The canonical operation table consumes a value-type object,
                # not a bare dtype spelling: ordinary tensors use
                # {"dtype": "f32", "shape": ["dim0", ...]}.  Letting a
                # missing or string-only type through this write boundary used
                # to make an otherwise complete Source-ledger submission fail
                # later as canonical_construction_bundle_operation_incomplete.
                if operation.get("op") not in STRUCTURED_INTENT_OPS:
                    operation_type = operation.get("type")
                    tensor_type = (
                        isinstance(operation_type, Mapping)
                        and isinstance(operation_type.get("dtype"), str)
                        and bool(operation_type.get("dtype"))
                        and isinstance(operation_type.get("shape"), list)
                    )
                    structured_type = (
                        isinstance(operation_type, Mapping)
                        and isinstance(operation_type.get("kind"), str)
                        and operation_type.get("kind") in {"record", "ragged"}
                    )
                    if not tensor_type and not structured_type:
                        return "operation_result_must_have_complete_value_type"
                    if tensor_type and any(
                        not (
                            isinstance(extent, str)
                            and bool(extent)
                            or isinstance(extent, int)
                            and not isinstance(extent, bool)
                            and extent >= 0
                        )
                        for extent in operation_type["shape"]
                    ):
                        return "operation_type_shape_must_use_valid_provider_extent"
                    if tensor_type and any(
                        not is_provider_dimension_extent(
                            extent,
                            self._semantic_provider_dimension_aliases.values(),
                        )
                        for extent in operation_type["shape"]
                    ):
                        # Only Source-issued dimN aliases can represent
                        # symbolic extents on the provider wire. Reject an
                        # arbitrary value/argument label at its defining row,
                        # before it contaminates later operand-shape feedback.
                        return "operation_type_shape_must_use_provider_dimension_alias"
                    if tensor_type and operation.get("op") == "iota":
                        # Iota has no operands from which the generic
                        # rank-preservation check can recover its type.  Its
                        # semantic result is nevertheless fully fixed by its
                        # own public attrs: one dimension ``n`` and the
                        # declared integer dtype.  Enforce that local,
                        # schema-derived relation at graph declaration so a
                        # rank-two iota cannot spend the single finalization
                        # attempt only to fail Decoder type inference.
                        iota_extent = (
                            provider_dimension_extent(attrs.get("n"))
                            if isinstance(attrs, Mapping)
                            else None
                        )
                        iota_dtype = attrs.get("dtype") if isinstance(attrs, Mapping) else None
                        if iota_extent is None:
                            return "operation_type_shape_must_use_provider_dimension_alias"
                        if (
                            tensor_shape(operation) != (iota_extent,)
                            or (
                                isinstance(iota_dtype, str)
                                and operation_type.get("dtype") != iota_dtype
                            )
                        ):
                            return "operation_iota_result_type_must_match_attrs"
                    if tensor_type and operation.get("op") in {"map_binary", "select"}:
                        result_shape = tensor_shape(operation)
                        if result_shape is not None:
                            requires_boundary_broadcast = any(
                                isinstance(operand, str)
                                and operand in value_types
                                and self._unique_nontrailing_broadcast_axes(
                                    value_types[operand][1], result_shape
                                )
                                is not None
                                for operand in operands
                            )
                            if requires_boundary_broadcast and not isinstance(
                                operation.get("id"), str
                            ):
                                # The provider owns the boundary relation, but
                                # its node_ref must name the same local graph
                                # node.  Reject before commit when an implicit
                                # id would make that relation unwriteable; do
                                # not allocate or infer an id here.
                                return "operation_nontrailing_broadcast_requires_local_id"
                    known_shapes = [
                        value_types[operand][1]
                        for operand in operands
                        if isinstance(operand, str) and operand in value_types
                    ]
                    expected_shape = _expected_result_shape_for_known_operands(
                        operation,
                        known_shapes,
                    )
                    if (
                        expected_shape is not None
                        and tensor_shape(operation) != expected_shape
                    ):
                        return "operation_result_type_shape_must_match_operands"
                visible.add(value)
                if tensor_type:
                    shape = tensor_shape(operation)
                    dtype = operation_type.get("dtype")
                    if shape is not None and isinstance(dtype, str):
                        value_types[value] = (dtype, shape)
        # Whole-graph counts are checked before row details above; this stage
        # now reports type/predicate/output relations only for covered graphs.
        if self._semantic_graph_missing_output_type(body) is not None:
            return "operation_graph_must_expose_each_source_output_type"
        if self._semantic_graph_missing_access_predicate_type(body) is not None:
            return "operation_graph_must_expose_compatible_access_predicate_type"
        if self._semantic_graph_missing_access_consumer_id(body) is not None:
            return "operation_graph_must_expose_access_consumer_id"
        # Report the logical-output assembly only after the graph has at least
        # covered its Source operation families/attributes.  A one-row or
        # otherwise skeletal graph must first receive the complete-graph
        # contract; otherwise the assembly diagnostic hides the more basic
        # missing producer chain and encourages a concat-only resubmission.
        if self._semantic_graph_missing_output_assembly(body) is not None:
            return "operation_graph_must_expose_source_output_assembly"
        return None

    def _semantic_graph_requires_transfer_reread(self) -> bool:
        """Return whether access authoring lacks its graph-bound ledger read."""

        return (
            self._semantic_construction_buffer is not None
            and "declare_operation_graph" in self._semantic_construction_completed_steps
            and "declare_accesses" not in self._semantic_construction_completed_steps
            and self._semantic_operation_graph_transfer_generation is not None
            and self._semantic_transfer_read_generation
            <= self._semantic_operation_graph_transfer_generation
        )

    def _semantic_transfer_identity_is_current(
        self, result: Mapping[str, Any]
    ) -> bool:
        """Check a transfer receipt without retaining its Source-owned rows."""

        if result.get("returncode") != 0:
            return False
        try:
            payload = json.loads(str(result.get("output", "")))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return (
            isinstance(payload, Mapping)
            and payload.get("kind") == "canonical_semantic_transfer_ledger"
            and payload.get("construction_identity")
            == self._semantic_construction_identity
        )

    def _mcp_phase_surface(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the only MCP resource/tool choices valid in this phase.

        This is the one lifecycle authority used both to constrain the
        provider-facing Responses function schema and to reject a forged or
        stale action at dispatch.  The resource names here are view names;
        URI conversion remains owned by the MCP catalogue.
        """

        if self._semantic_construction_buffer is not None:
            next_step = self._semantic_next_required_action()
            if next_step is None:
                return (), ()
            if self._semantic_graph_requires_transfer_reread():
                # Do not let an operation graph displace the exact Source-order
                # occurrence table from the provider context.  This is a read
                # barrier, not a new construction step or a relaxation of the
                # complete-array access contract.
                return ("semantic_transfer",), ()
            return (), (next_step,)

        if not self._semantic_entry_read:
            return ("source_brief",), ()
        lookup_allowed = (
            bool(self._semantic_entry_missing_operation_leaves)
            and self._semantic_entry_schema_lookup_calls == 0
        )
        if not self._semantic_entry_transfer_ledger_read:
            # The transfer ledger is the source-order authority and must be
            # read before selecting a missing operation leaf. Keeping this edge
            # singular prevents a provider from looking up operations before it
            # has seen the transfer/output slots it must connect.
            return ("semantic_transfer",), ()
        if lookup_allowed:
            # Every Source-declared missing leaf is fetched atomically before
            # authoring; begin_candidate cannot bypass this prerequisite.
            return (), ("schema_lookup",)
        return (), ("begin_candidate",)

    def response_tools(self) -> list[dict[str, Any]]:
        """Project the current phase authority into one Responses tool schema."""

        from minisweagent.models.utils.actions_toolcall_response import (
            mcp_call_tool_response_api_for_surface,
        )

        views, tools = self._mcp_phase_surface()
        schema_lookup_items = (
            tuple(
                f"op:{leaf}"
                for leaf in sorted(self._semantic_entry_missing_operation_leaves)
            )
            if tools == ("schema_lookup",)
            else None
        )
        return [
            mcp_call_tool_response_api_for_surface(
                resource_uris=mcp_resource_uris_for_views(views),
                tool_names=tools,
                schema_lookup_items=schema_lookup_items,
            )
        ]

    def _mcp_action_is_in_phase_surface(self, action: Mapping[str, Any]) -> bool:
        """Return whether a parsed gateway action is currently exposed."""

        views, tools = self._mcp_phase_surface()
        if action.get("kind") == "resource":
            return action.get("view") in views
        if action.get("kind") == "tool":
            return action.get("name") in tools
        return False

    def _reject_mcp_action_outside_phase_surface(
        self,
        action: Mapping[str, Any],
    ) -> None:
        """Fail closed when a stale/static client bypasses the dynamic schema."""

        raise MCPRequestError("MCP action is not available in the current transaction phase")

    def _mcp_dispatch(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        """Dispatch one validated MCP resource/tool request to closed operations."""

        if not self._mcp_action_is_in_phase_surface(action):
            self._reject_mcp_action_outside_phase_surface(action)
        kind = action.get("kind")
        if kind == "resource":
            view = action.get("view")
            post_graph_transfer_reread = (
                view == "semantic_transfer"
                and self._semantic_graph_requires_transfer_reread()
            )
            if (
                self._semantic_construction_buffer is not None
                and not post_graph_transfer_reread
            ):
                self._raise_lift_preparation_failure(
                    detail="semantic_construction_forbids_resource_reads_after_begin",
                    attempted_step=f"resource:{view}",
                )
            if not post_graph_transfer_reread:
                if not self._semantic_entry_read and view != "source_brief":
                    raise MCPRequestError(
                        "read intentir://lift/source/brief before any other authoring resource"
                    )
                if view == "source_brief" and self._semantic_entry_read:
                    raise MCPRequestError(
                        "semantic-entry authoring permits one source/brief read"
                    )
                if self._semantic_entry_read and view not in _SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS:
                    raise MCPRequestError(
                        "semantic-entry authoring permits one exact supplemental fact read before construction"
                    )
                # The semantic transfer ledger is the sole Source-order
                # authority for access, fill, owner and output slots.
                if (
                    self._semantic_entry_read
                    and not self._semantic_entry_transfer_ledger_read
                    and view != "semantic_transfer"
                ):
                    raise MCPRequestError(
                        "read intentir://lift/source/transfer-ledger before any other supplemental fact"
                    )
                if (
                    view in _SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS
                    and self._semantic_entry_supplemental_read is not None
                    and not (
                        view == "semantic_transfer"
                        and self._semantic_transfer_expected_start > 1
                    )
                ):
                    raise MCPRequestError(
                        "semantic-entry authoring permits one total supplemental fact read"
                    )
            resource_action: dict[str, Any] = {}
            resource_action.update(
                {
                    key: value
                    for key, value in action.items()
                    if key not in {"kind", "view", "uri"}
                }
            )
            resource_action["view"] = view
            if (
                view == "semantic_transfer"
                and action.get("start", 1) != self._semantic_transfer_expected_start
            ):
                return self._structured_tool_error(
                    "semantic transfer pages must be read in Source order"
                )
            result = self._read_context_view(resource_action)
            if view == "source_brief" and result.get("returncode") == 0:
                try:
                    brief = json.loads(str(result.get("output", "")))
                    entry = brief.get("semantic_authoring_entry")
                    spine = entry.get("source_semantic_spine") if isinstance(entry, Mapping) else None
                    missing = spine.get("missing_operation_contracts") if isinstance(spine, Mapping) else None
                    operation_requirements = (
                        entry.get("source_operation_requirements")
                        if isinstance(entry, Mapping)
                        else None
                    )
                    required_families = (
                        spine.get("required_operation_families")
                        if isinstance(spine, Mapping)
                        else None
                    )
                    required_attributes = (
                        spine.get("required_operation_attributes")
                        if isinstance(spine, Mapping)
                        else None
                    )
                    dependency_edges = (
                        entry.get("source_operation_dependencies")
                        if isinstance(entry, Mapping)
                        else None
                    )
                    operation_positions = {
                        item.get("source_ref"): index
                        for index, item in enumerate(
                            operation_requirements
                            if isinstance(operation_requirements, list)
                            else []
                        )
                        if isinstance(item, Mapping)
                    }
                    result_positions = {
                        item.get("result"): index
                        for index, item in enumerate(
                            operation_requirements
                            if isinstance(operation_requirements, list)
                            else []
                        )
                        if isinstance(item, Mapping)
                        and isinstance(item.get("result"), str)
                        and re.fullmatch(r"sx[0-9]+_[0-9]+", item["result"])
                        is not None
                    }
                    expected_dependency_edges = {
                        (index, result_positions[operand])
                        for index, item in enumerate(
                            operation_requirements
                            if isinstance(operation_requirements, list)
                            else []
                        )
                        if isinstance(item, Mapping)
                        and isinstance(item.get("operands"), list)
                        for operand in item["operands"]
                        if isinstance(operand, str)
                        and operand in result_positions
                        and result_positions[operand] != index
                    }
                    actual_dependency_edges = (
                        [tuple(edge) for edge in dependency_edges]
                        if isinstance(dependency_edges, list)
                        and all(
                            isinstance(edge, list)
                            and len(edge) == 2
                            and all(
                                isinstance(value, int)
                                and not isinstance(value, bool)
                                for value in edge
                            )
                            for edge in dependency_edges
                        )
                        else []
                    )
                    valid_operation_dependencies = (
                        isinstance(operation_requirements, list)
                        and isinstance(dependency_edges, list)
                        and len(operation_positions) == len(operation_requirements)
                        and len(actual_dependency_edges) == len(dependency_edges)
                        and len(set(actual_dependency_edges)) == len(actual_dependency_edges)
                        and all(
                            0 <= producer < consumer < len(operation_requirements)
                            for consumer, producer in actual_dependency_edges
                        )
                        and set(actual_dependency_edges) == expected_dependency_edges
                    )
                    if (
                        not isinstance(spine, Mapping)
                        or spine.get("status") != "pass"
                    ):
                        reason = (
                            spine.get("reason")
                            if isinstance(spine, Mapping)
                            else None
                        )
                        detail = "semantic_source_operation_spine_unavailable"
                        if (
                            isinstance(reason, str)
                            and re.fullmatch(r"[a-z0-9_]+", reason) is not None
                        ):
                            detail += f":{reason}"
                        self._raise_lift_preparation_failure(
                            detail=detail,
                            attempted_step="read_source_brief",
                        )
                    if (
                        not isinstance(missing, list)
                        or not isinstance(operation_requirements, list)
                        or not isinstance(required_families, Mapping)
                        or not isinstance(required_attributes, list)
                        or not valid_operation_dependencies
                        or any(
                            not isinstance(item, Mapping)
                            or set(item) not in (
                                {
                                    "source_ref",
                                    "canonical_op",
                                    "operator",
                                    "operands",
                                },
                                {
                                    "source_ref",
                                    "canonical_op",
                                    "operator",
                                    "operands",
                                    "result",
                                },
                                {
                                    "source_ref",
                                    "canonical_op",
                                    "operator",
                                    "operands",
                                    "source_unit_count",
                                },
                                {
                                    "source_ref",
                                    "canonical_op",
                                    "operator",
                                    "operands",
                                    "result",
                                    "source_unit_count",
                                },
                                {
                                    "source_ref",
                                    "canonical_op",
                                    "canonical_attrs",
                                    "operands",
                                },
                                {
                                    "source_ref",
                                    "canonical_op",
                                    "canonical_attrs",
                                    "operands",
                                    "result",
                                },
                                {
                                    "source_ref",
                                    "canonical_op",
                                    "canonical_attrs",
                                    "operands",
                                    "source_unit_count",
                                },
                                {
                                    "source_ref",
                                    "canonical_op",
                                    "canonical_attrs",
                                    "operands",
                                    "result",
                                    "source_unit_count",
                                },
                            )
                            or not isinstance(item.get("source_ref"), str)
                            or re.fullmatch(r"s[0-9]+", item["source_ref"])
                            is None
                            or not isinstance(item.get("canonical_op"), str)
                            or not item["canonical_op"]
                            or not isinstance(item.get("operands"), list)
                            or any(
                                not isinstance(operand, str)
                                for operand in item["operands"]
                            )
                            or (
                                "result" in item
                                and (
                                    not isinstance(item["result"], str)
                                    or not item["result"]
                                    or len(item["result"]) > 128
                                )
                            )
                            or (
                                "canonical_attrs" in item
                                and (
                                    item["canonical_op"]
                                    not in {"map_binary", "map_unary", "reduce"}
                                    or not isinstance(item["canonical_attrs"], Mapping)
                                    or set(item["canonical_attrs"]) != {"fn"}
                                    or not isinstance(
                                        item["canonical_attrs"].get("fn"), str
                                    )
                                    or not item["canonical_attrs"].get("fn")
                                )
                            )
                            or (
                                "operator" in item
                                and (
                                    "canonical_attrs" in item
                                    or not isinstance(item["operator"], str)
                                    or not item["operator"]
                                )
                            )
                            or (
                                "operator" not in item
                                and "canonical_attrs" not in item
                            )
                            or (
                                "operator" in item
                                and item["canonical_op"]
                                not in {"cast", "iota"}
                            )
                            or (
                                "source_unit_count" in item
                                and (
                                    isinstance(item["source_unit_count"], bool)
                                    or not isinstance(item["source_unit_count"], int)
                                    or item["source_unit_count"] <= 1
                                )
                            )
                            for item in operation_requirements
                        )
                        or any(
                            not isinstance(name, str)
                            or not name
                            or isinstance(count, bool)
                            or not isinstance(count, int)
                            or count < 0
                            for name, count in required_families.items()
                        )
                        or any(
                            not isinstance(row, Mapping)
                            or set(row) != {"op", "attribute", "value", "count"}
                            or not all(
                                isinstance(row.get(name), str) and row.get(name)
                                for name in ("op", "attribute", "value")
                            )
                            or isinstance(row.get("count"), bool)
                            or not isinstance(row.get("count"), int)
                            or row["count"] <= 0
                            for row in required_attributes
                        )
                        or any(
                            not isinstance(item, str)
                            or re.fullmatch(r"[A-Za-z0-9_]+", item) is None
                            for item in missing
                        )
                        or entry.get("construction_identity")
                        != self._semantic_construction_identity
                    ):
                        raise ValueError("source brief semantic identity or spine is malformed")
                    dependencies_by_source_ref: dict[str, list[str]] = {
                        row["source_ref"]: [] for row in operation_requirements
                    }
                    for consumer, producer in actual_dependency_edges:
                        dependencies_by_source_ref[
                            operation_requirements[consumer]["source_ref"]
                        ].append(operation_requirements[producer]["source_ref"])
                    self._semantic_entry_missing_operation_leaves = frozenset(missing)
                    self._semantic_entry_operation_fact_count = len(
                        operation_requirements
                    )
                    self._semantic_entry_non_boundary_operation_fact_count = len(
                        operation_requirements
                    )
                    self._semantic_entry_operation_requirements = tuple(
                        {
                            **copy.deepcopy(dict(item)),
                            "source_dependencies": dependencies_by_source_ref[
                                item["source_ref"]
                            ],
                        }
                        for item in operation_requirements
                    )
                    self._semantic_entry_required_operation_families = {
                        str(name): int(count)
                        for name, count in required_families.items()
                    }
                    self._semantic_entry_required_operation_attributes = tuple(
                        (
                            str(row["op"]),
                            str(row["attribute"]),
                            str(row["value"]),
                            int(row["count"]),
                        )
                        for row in required_attributes
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    self._semantic_entry_read = False
                    return self._structured_tool_error(
                        "source brief lacks a valid semantic-spine lookup policy"
                    )
            if (
                view == "semantic_transfer"
                and result.get("returncode") == 0
                and not self._semantic_transfer_identity_is_current(result)
            ):
                return self._structured_tool_error(
                    "source transfer ledger identity differs from current construction"
                )
            if view == "semantic_transfer" and result.get("returncode") == 0:
                try:
                    transfer_payload = json.loads(str(result.get("output", "")))
                    access_page = transfer_payload.get("accesses")
                    output_page = transfer_payload.get("outputs")
                    if (
                        not isinstance(access_page, Mapping)
                        or not isinstance(output_page, Mapping)
                        or access_page.get("start") != self._semantic_transfer_expected_start
                        or output_page.get("start") != self._semantic_transfer_expected_start
                    ):
                        return self._structured_tool_error(
                            "semantic transfer page is not contiguous"
                        )
                    for page, slot_name in (
                        (access_page, "accesses"), (output_page, "outputs")
                    ):
                        rows = page.get("rows")
                        total = page.get("total")
                        start = self._semantic_transfer_expected_start
                        if (
                            not isinstance(rows, list)
                            or type(total) is not int
                            or total != len(self._semantic_construction_slots[slot_name])
                            or len(rows) > max(0, total - start + 1)
                            or page.get("count") != len(rows)
                            or page.get("next_start")
                            != (start + len(rows) if start + len(rows) <= total else None)
                            or (start <= total and not rows)
                        ):
                            return self._structured_tool_error(
                                "semantic transfer page cardinality is malformed"
                            )
                    next_starts = [
                        value
                        for value in (access_page.get("next_start"), output_page.get("next_start"))
                        if value is not None
                    ]
                    if next_starts:
                        next_start = max(next_starts)
                        if transfer_payload.get("next_start") != next_start:
                            return self._structured_tool_error(
                                "semantic transfer next_start is inconsistent"
                            )
                        self._semantic_transfer_expected_start = next_start
                    else:
                        if transfer_payload.get("next_start") is not None:
                            return self._structured_tool_error(
                                "semantic transfer next_start is inconsistent"
                            )
                        self._semantic_transfer_expected_start = 1
                        self._semantic_entry_transfer_ledger_read = True
                        self._semantic_transfer_read_generation += 1
                except (TypeError, ValueError, json.JSONDecodeError):
                    return self._structured_tool_error(
                        "semantic transfer page is malformed"
                    )
            if (
                view in _SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS
                and result.get("returncode") == 0
                and not post_graph_transfer_reread
            ):
                self._semantic_entry_supplemental_read = str(view)
                self._semantic_entry_source_read = view == "source"
                if view != "semantic_transfer":
                    self._semantic_entry_transfer_ledger_read = False
            if (
                post_graph_transfer_reread
                and result.get("returncode") == 0
            ):
                if (
                    self._semantic_transfer_read_generation
                    > (self._semantic_operation_graph_transfer_generation or 0)
                ):
                    self._semantic_access_transfer_generation = (
                        self._semantic_transfer_read_generation
                    )
                    self._semantic_construction_audit.append(
                        {
                            "step": "read_transfer_ledger_after_operation_graph",
                            "status": "pass",
                            "transaction": self._semantic_transaction_identity,
                            "transfer_read_generation": (
                                self._semantic_transfer_read_generation
                            ),
                        }
                    )
            return result
        if kind == "tool":
            name = action.get("name")
            arguments = action.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, Mapping):
                raise MCPRequestError("invalid MCP tool dispatch")
            if name == "schema_lookup":
                if self._semantic_construction_buffer is not None:
                    self._raise_lift_preparation_failure(
                        detail="semantic_construction_forbids_schema_lookup_after_begin",
                        attempted_step="schema_lookup",
                    )
                if not self._semantic_entry_read:
                    raise MCPRequestError(
                        "read intentir://lift/source/brief before schema lookup"
                    )
                if self._semantic_entry_schema_lookup_calls >= 1:
                    raise MCPRequestError(
                        "semantic-entry authoring permits one schema lookup"
                    )
                requested = arguments.get("items")
                requested_leaves = (
                    tuple(item[3:] for item in requested)
                    if isinstance(requested, list)
                    and 1 <= len(requested) <= MAX_MCP_SCHEMA_LOOKUP_ITEMS
                    and len(requested) == len(set(requested))
                    and all(
                        isinstance(item, str)
                        and re.fullmatch(r"op:[A-Za-z0-9_]+", item) is not None
                        for item in requested
                    )
                    else ()
                )
                expected_leaves = tuple(
                    sorted(self._semantic_entry_missing_operation_leaves)
                )
                if requested_leaves != expected_leaves:
                    raise MCPRequestError(
                        "schema_lookup must request exactly the Source-declared "
                        "missing operation contracts in canonical order"
                    )
                result = self._read_schema_bundle(dict(arguments))
                if result.get("returncode") == 0:
                    self._semantic_entry_schema_lookup_calls += 1
                return result
            if name in MCP_SEMANTIC_WRITE_TOOL_NAMES:
                return self._mcp_semantic_construction_write(
                    name, dict(arguments)
                )
            raise MCPRequestError(f"unsupported MCP tool: {name}")
        raise MCPRequestError("unsupported MCP dispatch kind")

    def _active_mcp_resource_uris(self) -> tuple[str, ...]:
        """Expose the fresh semantic-entry read surface."""

        return mcp_resource_uris_for_views(_SEMANTIC_ENTRY_CONTEXT_VIEWS)

    @staticmethod
    def _mcp_result(
        result: Mapping[str, Any], *, method: str, is_error: bool = False
    ) -> dict[str, Any]:
        """Wrap an environment result in a bounded MCP text result."""

        payload = dict(result)
        text = payload.get("output", "")
        if not isinstance(text, str):
            text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return {
            "mcp": "intentir-lift",
            "method": method,
            "isError": is_error or payload.get("returncode") not in {0, None},
            "content": [{"type": "text", "text": text}],
            "returncode": payload.get("returncode", 0),
            "exception_info": payload.get("exception_info", ""),
        }

    def _mcp_phase_failure_key(self) -> tuple[str, str]:
        """Return a receipt-only identity for the current lifecycle edge."""

        views, tools = self._mcp_phase_surface()
        if len(views) + len(tools) != 1:
            # Active Lift phases are singular. Retain a deterministic key for
            # malformed or future non-singular surfaces; this remains
            # telemetry, not action-admission authority.
            phase = ",".join(
                [
                    *(f"resource:{view}" for view in views),
                    *(f"tool:{tool}" for tool in tools),
                ]
            ) or "terminal"
        elif views:
            phase = f"resource:{views[0]}"
        else:
            phase = f"tool:{tools[0]}"
        state = {
            "mode": "semantic_entry",
            "phase": phase,
            "construction_started": self._semantic_construction_buffer is not None,
            "completed_steps": sorted(self._semantic_construction_completed_steps),
            "transfer_read_generation": self._semantic_transfer_read_generation,
        }
        phase_sha256 = hashlib.sha256(
            json.dumps(
                state,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return phase, phase_sha256

    @staticmethod
    def _mcp_application_error_text(response: Mapping[str, Any]) -> str:
        """Extract an error rule for hashing without retaining response bytes."""

        content = response.get("content")
        if isinstance(content, list) and len(content) == 1:
            item = content[0]
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                text = item["text"]
                try:
                    payload = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return text
                if isinstance(payload, Mapping):
                    diagnostic = payload.get("diagnostic")
                    if isinstance(diagnostic, str):
                        return diagnostic
                return text
        return str(response.get("returncode", "application_error"))

    def _record_mcp_phase_application_failure(
        self,
        *,
        phase: str,
        phase_sha256: str,
        category: str,
        rule_text: str,
        origin: str | None = None,
    ) -> None:
        """Fail closed on repeated errors at one candidate lifecycle edge.

        Only hashes and fixed state-machine labels enter evidence. Raw MCP
        output, provider arguments, Source content, and rejected declarations
        are deliberately excluded, so this receipt can never become a repair
        base or a semantic hint.
        """

        rule_sha256 = hashlib.sha256(
            f"{category}\0{rule_text}".encode("utf-8", errors="replace")
        ).hexdigest()
        phase_rows = [
            row
            for row in self._mcp_phase_application_failures
            if row.get("phase_sha256") == phase_sha256
        ]
        repeated_count = 1 + sum(
            row.get("category") == category
            and row.get("rule_sha256") == rule_sha256
            for row in phase_rows
        )
        row = {
            "phase": phase,
            "phase_sha256": phase_sha256,
            "category": category,
            "rule_sha256": rule_sha256,
            "attempt": len(phase_rows) + 1,
            "same_rule_count": repeated_count,
            "failure_class": "mcp_protocol_rejected",
            "progress_changed": False,
        }
        if isinstance(origin, str) and re.fullmatch(
            r"[A-Za-z0-9_]+\.py:[A-Za-z_][A-Za-z0-9_]*:[0-9]{1,6}",
            origin,
        ):
            row["origin"] = origin
        self._mcp_phase_application_failures.append(row)
        if repeated_count >= _MAX_REPEATED_MCP_PHASE_RULE_FAILURES:
            self._raise_lift_preparation_failure(
                detail="mcp_no_progress",
                attempted_step=phase,
                rule_code=category,
            )
        if len(phase_rows) + 1 >= _MAX_MCP_PHASE_APPLICATION_FAILURES:
            self._raise_lift_preparation_failure(
                detail="mcp_no_progress",
                attempted_step=phase,
                rule_code=category,
            )

    def _execute_mcp(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Execute an MCP request while retaining the existing sandbox audit."""

        started = time.monotonic()
        method = str(action.get("method", ""))
        params = action.get("params", {})
        phase, phase_sha256 = self._mcp_phase_failure_key()
        error_category: str | None = None
        error_origin: str | None = None
        try:
            if method in {"initialize", "resources/list", "tools/list"}:
                result = self._mcp_server.handle(method, params)
                response = {
                    "mcp": "intentir-lift",
                    "method": method,
                    "isError": False,
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                result, sort_keys=True, ensure_ascii=False
                            ),
                        }
                    ],
                    "returncode": 0,
                    "exception_info": "",
                }
            else:
                result = self._mcp_server.dispatch(method, params)
                response = self._mcp_result(result, method=method)
        except (CandidateRejected, LiftPreparationFailed, Submitted):
            # A passing MCP candidate submission is the environment's normal
            # agent-termination signal.  A rejected submission is also a
            # terminal signal for this logical Lift: its bounded receipt is
            # consumed by the host relift lifecycle, not returned as a normal
            # MCP observation for the same agent to reinterpret.
            raise
        except MCPRequestError as exc:
            _views, tools = self._mcp_phase_surface()
            if (
                self._semantic_construction_buffer is None
                and tools == ("begin_candidate",)
            ):
                self._semantic_preconstruction_phase_failures += 1
                if self._semantic_preconstruction_phase_failures >= 2:
                    self._raise_lift_preparation_failure(
                        detail="semantic_construction_not_started_after_begin_exposed",
                        attempted_step="begin_candidate",
                    )
            error_category = "mcp_request_rejected"
            response = {
                "mcp": "intentir-lift",
                "method": method,
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "fail",
                                "diagnostic": f"MCPRequestError: {exc}",
                            },
                            sort_keys=True,
                        ),
                    }
                ],
                "returncode": 1,
                "exception_info": "",
            }
        except Exception as exc:
            error_category = f"internal_{type(exc).__name__}"
            # Retain only a code location from this project's Python tree.
            # Exception text can contain rejected declarations and must not
            # enter the safe phase receipt or any subsequent fresh owner.
            project_python = Path(__file__).resolve().parents[2]
            frame = exc.__traceback__
            while frame is not None:
                code = frame.tb_frame.f_code
                location = Path(code.co_filename).resolve()
                if location.is_relative_to(project_python):
                    error_origin = (
                        f"{location.name}:{code.co_name}:{frame.tb_lineno}"
                    )
                frame = frame.tb_next
            response = {
                "mcp": "intentir-lift",
                "method": method,
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "fail",
                                "diagnostic": f"{type(exc).__name__}: {exc}",
                            },
                            sort_keys=True,
                        ),
                    }
                ],
                "returncode": 1,
                "exception_info": "",
            }
            # An exception raised by the host MCP implementation is not a
            # provider-correctable declaration error.  Returning it as a
            # normal tool error lets the model repeat the same lifecycle edge
            # until the bounded repeated-application guard fires, which hides
            # the real project defect behind
            # ``mcp_phase_repeated_application_error``.  Stop this fresh
            # transaction immediately.  The preparation receipt contains only
            # the closed phase/step state; it never includes the exception
            # text, provider arguments, Source facts, or candidate bytes.
            self._raise_lift_preparation_failure(
                detail="semantic_mcp_host_internal_error",
                attempted_step=phase,
                origin=error_origin,
            )
        if response["isError"] and (
            error_category is not None
            or phase.startswith("resource:")
            or self._semantic_construction_buffer is None
        ):
            self._record_mcp_phase_application_failure(
                phase=phase,
                phase_sha256=phase_sha256,
                category=error_category or "application_error",
                rule_text=self._mcp_application_error_text(response),
                origin=error_origin,
            )
        response["elapsed_ms"] = (time.monotonic() - started) * 1000.0
        self._mcp_requests.append(
            {
                "method": method,
                "status": "error" if response["isError"] else "pass",
                "elapsed_ms": response["elapsed_ms"],
            }
        )
        content = response.get("content")
        content_text = (
            content[0].get("text")
            if isinstance(content, list)
            and len(content) == 1
            and isinstance(content[0], Mapping)
            and isinstance(content[0].get("text"), str)
            else None
        )
        return {
            # The Responses observation template applies JSON escaping once.
            # Re-serializing the MCP envelope here needlessly escapes the
            # receipt a second time and can push a valid bounded Source brief
            # or begin receipt past that template's truncation threshold.
            "output": (
                content_text
                if content_text is not None
                else json.dumps(response, sort_keys=True, ensure_ascii=False)
            ),
            "returncode": response["returncode"],
            "exception_info": response["exception_info"],
        }

    def _build_startup_context(self) -> tuple[str, dict[str, Any]]:
        """Publish receipts and MCP entry points without embedding Source text."""

        receipts: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        startup_files = (
            "source.py",
            "harness_view.json",
            "canonical_construction.json",
            "feedback_frontier.json",
            "feedback.json",
            "schema/artifact.schema.json",
            "schema/canonical_skeleton.json",
            "schema/canonical_wire_bundle.json",
            "schema/canonical_source_wire_bundle.json",
            "knowledge/lift_contract.md",
            "knowledge/preflight.md",
        )
        for relative in startup_files:
            path = self.config.input_dir / relative
            if path.is_symlink():
                raise ValueError(f"Lift startup input must not be a symlink: {relative}")
            if not path.is_file():
                receipts[relative] = {"missing": True}
                continue
            raw = path.read_bytes()
            total_bytes += len(raw)
            receipts[relative] = {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        candidate_path = self.config.output_dir / "candidate.json"
        # The output directory is disposable owner state. A stale or complete
        # candidate from any prior path is never an authoring entry.
        try:
            candidate_path.unlink(missing_ok=True)
        except OSError as exc:
            raise ValueError(
                "candidate.json cannot be cleared before fresh semantic authoring"
            ) from exc
        candidate_present = False
        self._semantic_entry_read = False
        self._semantic_entry_source_read = False
        self._semantic_entry_transfer_ledger_read = False
        self._semantic_transfer_read_generation = 0
        self._semantic_transfer_expected_start = 1
        self._semantic_operation_graph_transfer_generation = None
        self._semantic_access_transfer_generation = None
        self._semantic_transaction_identity = None
        self._semantic_entry_supplemental_read = None
        self._semantic_entry_schema_lookup_calls = 0
        self._semantic_entry_missing_operation_leaves = frozenset()
        self._semantic_entry_operation_fact_count = 0
        self._semantic_entry_non_boundary_operation_fact_count = 0
        self._semantic_entry_required_operation_families = {}
        self._semantic_entry_required_operation_attributes = ()
        active_resources = [
            "intentir://lift/source/brief",
            "intentir://lift/source/transfer-ledger before construction",
            "an op:NAME schema lookup only when exposed",
            "the currently exposed construction action",
        ]
        payload: dict[str, Any] = {
            "mcp": "intentir-lift",
            "candidate_present": candidate_present,
            "authoring_mode": "semantic_entry",
            "authoring_resource": "intentir://lift/source/brief",
            "semantic_entry_required": True,
            "input_file_count": len(receipts),
            "input_total_bytes": total_bytes,
            "resources": active_resources,
            "authority": (
                "exact Source, Harness, compiler observation, feedback, and schema "
                "facts are available only through bounded MCP reads; no candidate "
                "is mounted, and MCP declarations have no canonical construction "
                "or acceptance authority"
            ),
        }
        receipt = {
            "mcp": payload["mcp"],
            "total_bytes": total_bytes,
            "candidate_present": candidate_present,
            "authoring_mode": "semantic_entry",
            "authoring_resource": payload["authoring_resource"],
            "semantic_entry_required": not candidate_present,
            "files": receipts,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False), receipt

    @staticmethod
    def _is_batched_read_only_command(command: Any) -> bool:
        if not isinstance(command, str) or any(
            token in command
            for token in (";", "|", "&", ">", "<", "\n", "\r", "`", "$(")
        ):
            return False
        try:
            arguments = shlex.split(command)
        except ValueError:
            return False
        if len(arguments) < 3 or arguments[0] != "python":
            return False
        if arguments[1] == "/input/tools/schema_query.py":
            return True
        return (
            arguments[1] == "/input/tools/context_query.py"
            and arguments[2] in _LIFT_READ_ONLY_CONTEXT_COMMANDS
        )

    def _semantic_write_sequence_error(self, names: Sequence[str]) -> str | None:
        """Return an exact protocol error for an invalid construction sequence.

        The canonical profile retains the complete write catalogue for
        extension compatibility, but active Lift construction has exactly one
        mandatory transaction chain.  Optional catalogue entries must not
        become alternate phase transitions merely because an older static
        client can name them.
        """

        required = tuple(self._semantic_write_contract["required_steps"])
        completed = set(self._semantic_construction_completed_steps)
        for name in names:
            if name not in self._semantic_write_contract["steps"]:
                return "semantic_construction_unknown_step"
            if name not in required:
                return "semantic_construction_optional_step_not_permitted"
            if name in completed:
                return "semantic_construction_duplicate_step"
            expected = next((step for step in required if step not in completed), None)
            if expected is None:
                return "semantic_construction_is_already_complete"
            if name != expected:
                return "semantic_construction_only_next_required_step_permitted_" + expected
            completed.add(name)
        return None

    def _semantic_next_required_action(self) -> str | None:
        """Return the sole remaining mandatory write in this transaction."""

        completed = self._semantic_construction_completed_steps
        for step in self._semantic_write_contract["required_steps"]:
            if step not in completed:
                return str(step)
        return None

    def _semantic_next_legal_action(self) -> str | None:
        """Return the next phase edge, including a mandatory read barrier.

        The write sequence alone names ``declare_accesses`` after an operation
        graph. That declaration is nevertheless unavailable until the graph
        has been followed by a fresh transfer-ledger read. Receipts must
        advertise the same phase edge as the dynamic MCP tool surface; otherwise
        a provider is directed to an action that the host immediately rejects.
        """

        if self._semantic_graph_requires_transfer_reread():
            return "read_transfer_ledger_after_operation_graph"
        return self._semantic_next_required_action()

    def _last_operation_graph_failure_receipt(self) -> dict[str, Any] | None:
        """Project the last graph failure without retaining declaration bytes.

        A terminal graph-preparation exit must preserve the same public
        row/type/attribute signal that the immediately preceding MCP receipt
        exposed.  The next owner is nevertheless fresh: this is bounded
        diagnostic metadata, not a graph, candidate, draft, or repair base.
        """

        if not self._semantic_operation_graph_failures:
            return None
        last = self._semantic_operation_graph_failures[-1]
        if not isinstance(last, Mapping):
            return None
        receipt: dict[str, Any] = {
            "failure_count": len(self._semantic_operation_graph_failures),
            "last_rule_code": str(last.get("detail", ""))[:128],
        }
        for key in (
            "attempt",
            "operation_rows",
            "row_index",
            "result_rank",
            "expected_rank",
            "output_ordinal",
            "access_ordinal",
            "function_visible_matching_value_count",
            "function_visible_boolean_result_count",
            "function_visible_compatible_boolean_result_count",
            "incompatible_boolean_shape_count",
            "direct_single_result_consumer_id_count",
            "required_operation_count",
            "actual_operation_count",
            "missing_operation_count",
            "minimum_operation_rows",
            "same_rule_count",
            "same_gap_count",
        ):
            value = last.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                receipt[key] = value
        for key in ("expected_shape", "actual_shape", "predicate_target_shape"):
            value = last.get(key)
            if (
                isinstance(value, list)
                and len(value) <= 16
                and all(
                    isinstance(extent, (str, int))
                    and not isinstance(extent, bool)
                    for extent in value
                )
            ):
                receipt[key] = list(value)
        aliases = last.get("provider_dimension_aliases")
        if (
            isinstance(aliases, list)
            and len(aliases) <= 32
            and all(
                isinstance(alias, str)
                and re.fullmatch(r"dim[0-9]+", alias) is not None
                for alias in aliases
            )
        ):
            receipt["provider_dimension_aliases"] = list(aliases)
        actual_extent = last.get("actual_extent")
        if type(actual_extent) is int and actual_extent >= 0:
            receipt["actual_extent"] = actual_extent
        elif isinstance(actual_extent, str) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.:-]{0,63}", actual_extent
        ):
            receipt["actual_extent"] = actual_extent
        expected_dtype = last.get("expected_dtype")
        if isinstance(expected_dtype, str) and expected_dtype:
            receipt["expected_dtype"] = expected_dtype[:64]
        for key in (
            "json_container",
            "declaration_sha256",
            "phase",
            "phase_digest",
            "failure_class",
            "operation",
            "row_path",
            "field",
            "attribute",
            "required_attribute_value",
            "graph_repair_rule",
            "expected_op_source",
            "label_rule",
            "failure_layer",
            "predicate_shape_rule",
            "predicate_projection_status",
            "output_type_visibility_rule",
            "type_repair_rule",
            "dimension_reference_rule",
        ):
            value = last.get(key)
            if isinstance(value, str) and value:
                receipt[key] = value[:256]
        if isinstance(last.get("complete_graph_required"), bool):
            receipt["complete_graph_required"] = last["complete_graph_required"]
        for key in ("missing_operation_families", "missing_operation_attributes"):
            value = last.get(key)
            if not isinstance(value, list) or len(value) > 16:
                continue
            expected_keys = (
                {
                    "operation",
                    "required_operation_count",
                    "actual_operation_count",
                    "missing_operation_count",
                }
                if key == "missing_operation_families"
                else {
                    "operation",
                    "attribute",
                    "required_attribute_value",
                    "required_operation_count",
                    "actual_operation_count",
                    "missing_operation_count",
                }
            )
            if not all(
                isinstance(item, Mapping)
                and set(item) == expected_keys
                and all(
                    isinstance(item.get(field), int)
                    and not isinstance(item.get(field), bool)
                    and item.get(field) >= 0
                    for field in (
                        "required_operation_count",
                        "actual_operation_count",
                        "missing_operation_count",
                    )
                )
                and all(
                    isinstance(item.get(field), str) and item.get(field)
                    for field in (
                        ("operation",)
                        if key == "missing_operation_families"
                        else ("operation", "attribute", "required_attribute_value")
                    )
                )
                for item in value
            ):
                continue
            receipt[key] = [dict(item) for item in value]
        for key in ("allowed_fields", "required_fields"):
            value = last.get(key)
            if isinstance(value, list) and all(
                isinstance(item, str) and item for item in value
            ):
                receipt[key] = list(value[:32])
        required_type = last.get("required_output_type")
        if isinstance(required_type, Mapping):
            dtype = required_type.get("dtype")
            shape = required_type.get("shape")
            if (
                isinstance(dtype, str)
                and isinstance(shape, list)
                and all(
                    isinstance(extent, (str, int)) and not isinstance(extent, bool)
                    for extent in shape
                )
            ):
                receipt["required_output_type"] = {
                    "dtype": dtype[:64],
                    "shape": list(shape[:16]),
                }
        return receipt

    def _raise_lift_preparation_failure(
        self,
        *,
        detail: str,
        attempted_step: str | None = None,
        rule_code: str | None = None,
        origin: str | None = None,
    ) -> None:
        """Terminate an unrecoverable pre-candidate construction transaction.

        After a semantic buffer starts, reopening Source ingress, starting a
        second buffer, or skipping a required lifecycle edge cannot repair the
        current transaction.  Continuing used to consume the whole model
        budget while no candidate could exist.  Keep the exit receipt bounded:
        it names only state-machine facts and never carries candidate bytes or
        Source content.
        """

        if detail == "semantic_mcp_host_internal_error":
            failure_class = "lift_mcp_host_internal_error"
        elif detail == "mcp_no_progress":
            failure_class = "lift_mcp_no_progress"
        elif detail in {
            "mcp_phase_repeated_application_error",
            "mcp_phase_application_error_limit_exhausted",
        }:
            failure_class = "lift_mcp_application_error"
        else:
            failure_class = "lift_authoring_preparation"
        receipt: dict[str, Any] = {
            "failure_class": failure_class,
            "diagnostic": detail,
            "completed_steps": sorted(self._semantic_construction_completed_steps),
        }
        if attempted_step is not None:
            receipt["attempted_step"] = attempted_step
        if isinstance(rule_code, str) and rule_code:
            receipt["rule_code"] = rule_code[:128]
        if isinstance(origin, str) and re.fullmatch(
            r"[A-Za-z0-9_.-]+:[A-Za-z_][A-Za-z0-9_]*:[0-9]{1,6}",
            origin,
        ):
            receipt["origin"] = origin[:160]
        if detail == "mcp_no_progress":
            _phase, phase_digest = self._mcp_phase_failure_key()
            receipt["state_digest"] = phase_digest
            receipt["progress_changed"] = False
        if attempted_step == "declare_operation_graph":
            graph_failure = self._last_operation_graph_failure_receipt()
            if graph_failure is not None:
                receipt["operation_graph_failure"] = graph_failure
        next_action = self._semantic_next_legal_action()
        if next_action is not None:
            receipt["next_legal_action"] = next_action
        self._preparation_failure_receipts.append(copy.deepcopy(receipt))
        raise LiftPreparationFailed(
            {
                "role": "exit",
                "content": "LiftPreparationFailed",
                "extra": {
                    "exit_status": "LiftPreparationFailed",
                    "submission": "",
                    "preparation_receipt": receipt,
                },
            }
        )

    def validate_actions(self, actions: list[dict[str, Any]]) -> str | None:
        """Admit independent reads or one ordered semantic construction batch."""

        def is_read_only_mcp_action(action: Any) -> bool:
            if not isinstance(action, Mapping):
                return False
            tool_name = action.get("tool_name")
            if tool_name != "mcp_call_tool":
                return False
            method = action.get("method")
            params = action.get("params")
            if method in {"resources/list", "tools/list"}:
                return isinstance(params, Mapping) and not params
            if method == "tools/call" and isinstance(params, Mapping):
                # schema_lookup is a bounded read-only MCP tool.  It may be
                # batched with resource reads; candidate writes/checks remain
                # single-action below.
                return (
                    set(params) == {"name", "arguments"}
                    and params.get("name") == "schema_lookup"
                    and isinstance(params.get("arguments"), Mapping)
                )
            if method != "resources/read" or not isinstance(params, Mapping):
                return False
            # The gateway is shared with write/check calls.  Permit only the
            # bounded resource URI read shape in a batch; tools/call is always
            # single-action so a candidate/check/patch cannot be coalesced.
            return (
                set(params) <= {
                    "uri", "query", "path", "start", "count", "context",
                    "op", "section",
                }
                and isinstance(params.get("uri"), str)
                and params["uri"].startswith("intentir://lift/")
            )

        def is_ordered_semantic_construction_batch() -> bool:
            """Allow one short, deterministic begin→declare→finalize batch.

            These writes are not independent. They are safe to coalesce only
            because the order is checked before any action executes and each
            step addresses the one in-memory semantic buffer. This keeps the
            agent to a few provider turns without reopening arbitrary patches
            or parallel candidate writes.
            """

            names: list[str] = []
            for action in actions:
                if not isinstance(action, Mapping) or action.get("tool_name") != "mcp_call_tool":
                    return False
                if action.get("method") != "tools/call":
                    return False
                params = action.get("params")
                if not isinstance(params, Mapping) or set(params) != {"name", "arguments"}:
                    return False
                name = params.get("name")
                if name not in MCP_SEMANTIC_WRITE_TOOL_NAMES or not isinstance(
                    params.get("arguments"), Mapping
                ):
                    return False
                names.append(str(name))
            if len(names) != len(set(names)):
                return False
            return self._semantic_write_sequence_error(names) is None

        if len(actions) <= 1:
            return None
        if len(actions) > 4:
            self._multi_action_rejections += 1
            return (
                f"{MULTI_ACTION_PROTOCOL}: at most four read-only or ordered "
                "semantic-construction calls are allowed; no action in this "
                "response was executed"
            )
        if all(
            (
                self._is_batched_read_only_command(action.get("command"))
                if isinstance(action, Mapping)
                and "command" in action
                else is_read_only_mcp_action(action)
            )
            for action in actions
        ):
            return None
        if is_ordered_semantic_construction_batch():
            return None
        self._multi_action_rejections += 1
        return (
            f"{MULTI_ACTION_PROTOCOL}: multiple calls are allowed only for "
            "independent read-only calls or one ordered semantic construction batch; "
            "no action in this response was executed"
        )

    @staticmethod
    def _structured_tool_error(diagnostic: str) -> dict[str, Any]:
        return {
            "output": json.dumps(
                {
                    "mcp": "intentir-lift",
                    "status": "fail",
                    "diagnostic": diagnostic,
                },
                sort_keys=True,
            )
            + "\n",
            "returncode": 2,
            "exception_info": "",
        }

    def _read_context_view(
        self, action: Mapping[str, Any]
    ) -> dict[str, Any]:
        allowed = MCP_CONTEXT_ARGUMENTS
        if not set(action).issubset(allowed):
            return self._structured_tool_error("unknown context-inspection argument")
        view = action.get("view")
        if view not in MCP_CONTEXT_VIEWS:
            return self._structured_tool_error("unsupported context-inspection view")

        def bounded_int(name: str, default: int, lower: int, upper: int) -> int:
            value = action.get(name, default)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
            return value

        def bounded_text(name: str, *, required: bool = False) -> str | None:
            value = action.get(name)
            if value is None and not required:
                return None
            max_chars = (
                MAX_MCP_QUERY_CHARS
                if name == "query"
                else MAX_MCP_PATH_CHARS
                if name == "path"
                else MCP_SELECTOR_CHARS
            )
            if (
                not isinstance(value, str)
                or (required and not value)
                or len(value) > max_chars
            ):
                raise ValueError(f"{name} must be a bounded string")
            return value

        try:
            start = bounded_int("start", 1, 1, 1_000_000)
            fixed_summary_views = MCP_FIXED_SUMMARY_VIEWS
            count_upper = (
                min(mcp_resource_count_upper(str(view)), 500)
                if view in fixed_summary_views
                else 80
                if view in {
                    "harness_abi", "source_obligations", "semantic_transfer",
                }
                else 160
                if view in {"source_ttir"}
                else 200
                if view in {"source_ttir_outline"}
                else 160
            )
            count = bounded_int(
                "count",
                24 if view == "harness_abi" else 80 if view in {"source_ttir", "source_ttir_outline"} else 32 if view in {"source_obligations", "semantic_transfer"} else 40,
                1,
                count_upper,
            )
            context = bounded_int("context", 3, 0, 16)
            query = bounded_text(
                "query",
                required=view
                in {
                    "source_find", "source_ttir_find", "source_ttir_def_use",
                    "feedback_category", "knowledge",
                },
            )
            path = bounded_text("path")
            section = bounded_text("section")
        except ValueError as exc:
            return self._structured_tool_error(str(exc))

        if view == "source_obligations":
            if section in {"", "overview"}:
                section = None
            elif section is not None and section not in _SOURCE_OBLIGATION_SECTIONS:
                return self._structured_tool_error(
                    "source_obligations section must be one of "
                    + ", ".join(sorted(_SOURCE_OBLIGATION_SECTIONS))
                )
        command = ["python", "/input/tools/context_query.py"]
        if view == "source":
            command += ["source", "--start", str(start), "--lines", str(count)]
        elif view == "source_ttir":
            command += ["source-ttir", "--start", str(start), "--lines", str(count)]
        elif view == "source_ttir_outline":
            command += ["source-ttir-outline", "--start", str(start), "--count", str(count)]
        elif view == "source_ttir_find":
            command += ["source-ttir-find", str(query), "--context", str(context)]
        elif view == "source_ttir_def_use":
            command += ["source-ttir-def-use", str(query), "--context", str(context)]
        elif view == "source_observer":
            command.append("source-observer")
            if path is not None:
                command += ["--path", path]
        elif view == "realization_ledger":
            command.append("realization-ledger")
        elif view == "source_outline":
            command += ["source-outline", "--start", str(start), "--count", str(count)]
        elif view == "source_find":
            command += ["source-find", str(query), "--context", str(context)]
        elif view == "source_brief":
            command.append("source-brief")
        elif view == "source_obligations":
            command.append("source-obligations")
            if section is not None:
                command += [
                    "--section",
                    section,
                    "--start",
                    str(start),
                    "--count",
                    str(count),
                ]
        elif view == "semantic_transfer":
            command += [
                "semantic-transfer",
                "--start",
                str(start),
                "--count",
                str(count),
            ]
        elif view == "feedback_frontier":
            command.append("feedback-frontier")
        elif view in {"harness", "feedback"}:
            command.append(view)
            if path is not None:
                command += ["--path", path]
        elif view == "feedback_category":
            command += ["feedback-category", str(query)]
        elif view == "harness_abi":
            command += [
                "harness-abi",
                "--start",
                str(start),
                "--count",
                str(count),
            ]
        elif view == "harness_c_contract":
            command.append("harness-c-contract")
        else:
            command += ["knowledge", str(query), "--context", str(context)]
        self._structured_context_queries.append(str(view))
        result = self._execute_host_mcp_reader(
            " ".join(shlex.quote(item) for item in command)
        )
        if view == "source_brief" and result.get("returncode") == 0:
            self._semantic_entry_read = True
        return result

    def _read_schema_bundle(
        self, action: Mapping[str, Any]
    ) -> dict[str, Any]:
        # schema_lookup and all context resources share the same mounted-wire
        # receipt validation.  Do this before dispatching schema_query.py so a
        # stale bundle cannot expose a schema leaf while j-framework would
        # reject the same workspace.
        receipt = self._execute_host_mcp_reader(
            "python /input/tools/context_query.py canonical-wire-check"
        )
        if receipt.get("returncode") != 0:
            return self._structured_tool_error(
                "canonical wire bundle validation failed before schema_lookup"
            )
        if set(action) - {"tool_name", "tool_call_id", "items"}:
            return self._structured_tool_error("unknown schema-inspection argument")
        items = action.get("items")
        if (
            not isinstance(items, list)
            or not 1 <= len(items) <= MAX_MCP_SCHEMA_LOOKUP_ITEMS
            or len(items) != len(set(items))
            or any(
                not isinstance(item, str)
                or len(item) > MCP_SCHEMA_LOOKUP_ITEM_CHARS
                or re.fullmatch(r"op:[A-Za-z0-9_]+", item) is None
                for item in items
            )
        ):
            return self._structured_tool_error(
                "schema items must use a bounded distinct semantic operation selector set"
            )
        self._structured_schema_queries.append(list(items))
        command = ["python", "/input/tools/schema_query.py", "bundle", *items]
        return self._execute_host_mcp_reader(
            " ".join(shlex.quote(item) for item in command)
        )

    def _execute_host_mcp_reader(self, command: str) -> dict[str, Any]:
        """Execute one host-selected bounded MCP reader.

        The command is constructed by the dispatcher from validated MCP
        arguments.  The depth flag is deliberately scoped rather than a
        provider-visible capability: ordinary owner shell actions cannot open
        `/input` and therefore cannot bypass the resource lifecycle.
        """

        self._host_mcp_read_depth += 1
        try:
            return self.execute({"command": command})
        finally:
            self._host_mcp_read_depth -= 1

    @staticmethod
    def _parse_semantic_declaration(text: str) -> Any:
        """Parse one MCP declaration without accepting non-JSON wire forms."""

        return json.loads(
            text,
            parse_constant=_reject_non_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )

    @staticmethod
    def _operation_graph_exception_diagnostic(exc: BaseException) -> str | None:
        """Classify safe, provider-correctable graph wire failures.

        JSON decoder wording and internal parser exceptions are deliberately
        not exposed verbatim.  The returned codes describe only the public
        declaration contract, and the caller keeps all other failures behind
        the generic schema-derived receipt.
        """

        if isinstance(exc, json.JSONDecodeError):
            return "operation_graph_must_be_valid_json"
        detail = str(exc)
        if detail.startswith("duplicate JSON key "):
            return "operation_graph_must_not_contain_duplicate_json_keys"
        if detail.startswith("invalid JSON constant:"):
            return "operation_graph_must_use_finite_json_numbers"
        if detail in _PROVIDER_CORRECTABLE_OPERATION_GRAPH_DIAGNOSTICS:
            return detail
        return None

    def _operation_graph_receipt_diagnostic(
        self,
        declaration_text: str | None,
        body: Any,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """Return non-sensitive structure for a rejected graph declaration.

        The stable error code is the semantic authority.  For an attribute
        envelope error, also name the declaration row and its public contract
        field set.  A leaf-value mismatch additionally carries that one
        schema-derived public leaf contract.  This avoids making a provider
        guess which operation in a large Source graph needs correction,
        without returning Source text, candidate bytes, or a canonical handle.
        """

        normalized: bytes
        container = "invalid"
        row_count: int | None = None
        if isinstance(body, list):
            container = "array"
            row_count = len(body)
            try:
                normalized = json.dumps(
                    body,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                normalized = b"<unserializable-json-array>"
        elif body is not None:
            container = _json_container_kind(body)
            normalized = f"<{container}>".encode("ascii")
        else:
            # Invalid JSON has no normalized value.  A digest of its exact
            # bytes is still safe to compare without writing those bytes to an
            # evidence file or provider receipt.
            normalized = (declaration_text or "").encode("utf-8", "replace")
        diagnostic_result: dict[str, Any] = {
            "json_container": container,
            "declaration_sha256": hashlib.sha256(normalized).hexdigest(),
        }
        if row_count is not None:
            diagnostic_result["operation_rows"] = row_count
        if detail == "operation_graph_must_be_nonempty_json_array":
            # This is a complete wire-level failure, not an optional section.
            # Expose the repair relation without synthesizing any Source
            # operation, producer, owner, or result for the provider.
            diagnostic_result.update(
                {
                    "minimum_operation_rows": 1,
                    "complete_graph_required": True,
                    "graph_repair_rule": (
                        "resubmit one complete non-empty producer-order graph "
                        "from the current Source facts; [] is not a placeholder"
                    ),
                }
            )

        def declaration_rows(value: Any) -> list[tuple[str, Mapping[str, Any]]]:
            """Return structural row paths without accepting the graph."""

            found: list[tuple[str, Mapping[str, Any]]] = []

            def visit_rows(rows: Any, prefix: str) -> None:
                if not isinstance(rows, list):
                    return
                for index, row in enumerate(rows):
                    if not isinstance(row, Mapping):
                        continue
                    path = f"{prefix}[{index}]"
                    found.append((path, row))
                    operation = row.get("op")
                    if operation == "loop":
                        visit_rows(row.get("body"), path + ".body")
                    elif operation in {"effect_if", "value_if"}:
                        for arm_name in ("positive", "negative"):
                            arm = row.get(arm_name)
                            if isinstance(arm, Mapping):
                                visit_rows(
                                    arm.get("body"), path + f".{arm_name}.body"
                                )

            visit_rows(value, "body")
            return found

        structural_rows = declaration_rows(body)
        try:
            namespace = json.loads(self._semantic_operation_namespace)
            operation_catalog = namespace.get("operation_catalog")
            diagnostic_catalog = (
                frozenset(operation_catalog)
                if isinstance(operation_catalog, list)
                and all(isinstance(item, str) for item in operation_catalog)
                else frozenset()
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            diagnostic_catalog = frozenset()
        local_label = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")

        def provider_local_label(value: Any) -> bool:
            return (
                isinstance(value, str)
                and local_label.fullmatch(value) is not None
                and re.fullmatch(r"[nvxpcae][0-9]+", value) is None
            )

        field_details = {
            "operation_row_must_include_canonical_op_field": "op",
            "operation_op_must_be_canonical_j_name": "op",
            "operation_must_use_only_canonical_leaf_fields": "row",
            "operation_must_include_required_leaf_fields": "row",
            "operation_id_must_be_provider_local_label": "id",
            "operation_result_must_be_provider_local_label": "result",
            "operation_operands_must_be_json_array": "operands",
            "operation_operands_must_match_canonical_leaf_arity": "operands",
            "operation_operand_must_reference_source_tensor_or_prior_local_result": "operands",
            "operation_result_must_be_new_provider_local_label": "result",
            "operation_attrs_must_be_json_object": "attrs",
            "operation_result_must_have_complete_value_type": "type",
            "operation_type_shape_must_use_valid_provider_extent": "type.shape",
            "operation_type_shape_must_use_provider_dimension_alias": "type.shape",
            "operation_iota_result_type_must_match_attrs": "type",
            "operation_nontrailing_broadcast_requires_local_id": "id",
        }
        target_field = field_details.get(str(detail))
        if detail == "operation_op_must_be_canonical_j_name":
            diagnostic_result["expected_op_source"] = "operation_namespace.operation_catalog"
        elif detail in {
            "operation_id_must_be_provider_local_label",
            "operation_result_must_be_provider_local_label",
        }:
            diagnostic_result["label_rule"] = (
                "fresh provider-local label; no host n*/v*/x*/p*/c*/a*/e* numeric id"
            )
        if target_field is not None and structural_rows:
            selected_path: str | None = None
            selected_row: Mapping[str, Any] | None = None
            visible_labels = set(self._semantic_source_tensor_handles)
            for path, row in structural_rows:
                operation = row.get("op")
                operands = row.get("operands")
                operation_type = row.get("type")
                raw_result = row.get("result")
                result_labels = (
                    raw_result if isinstance(raw_result, list) else [raw_result]
                )
                bad = (
                    detail == "operation_row_must_include_canonical_op_field"
                    and "op" not in row
                    or detail == "operation_op_must_be_canonical_j_name"
                    and (
                        not isinstance(operation, str)
                        or operation not in diagnostic_catalog
                    )
                    or detail == "operation_must_use_only_canonical_leaf_fields"
                    and isinstance(operation, str)
                    and operation not in STRUCTURED_INTENT_OPS
                    and bool(
                        set(row)
                        - {"id", "op", "result", "operands", "attrs", "type"}
                    )
                    or detail == "operation_must_include_required_leaf_fields"
                    and isinstance(operation, str)
                    and operation not in STRUCTURED_INTENT_OPS
                    and ("attrs" not in row or "type" not in row)
                    or detail == "operation_id_must_be_provider_local_label"
                    and row.get("id") is not None
                    and not provider_local_label(row.get("id"))
                    or detail == "operation_result_must_be_provider_local_label"
                    and raw_result is not None
                    and any(not provider_local_label(value) for value in result_labels)
                    or detail == "operation_operands_must_be_json_array"
                    and not isinstance(operands, list)
                    or detail == "operation_operands_must_match_canonical_leaf_arity"
                    and isinstance(operation, str)
                    and isinstance(operands, list)
                    and not operation_operands_match_arity(
                        operation,
                        operands,
                        row.get("attrs") if isinstance(row.get("attrs"), Mapping) else None,
                    )
                    or detail
                    == "operation_operand_must_reference_source_tensor_or_prior_local_result"
                    and isinstance(operands, list)
                    and any(
                        not isinstance(operand, str) or operand not in visible_labels
                        for operand in operands
                    )
                    or detail == "operation_result_must_be_new_provider_local_label"
                    and raw_result is not None
                    and any(value in visible_labels for value in result_labels)
                    or detail == "operation_attrs_must_be_json_object"
                    and not isinstance(row.get("attrs"), Mapping)
                    or detail == "operation_result_must_have_complete_value_type"
                    and "result" in row
                    and not isinstance(operation_type, Mapping)
                    or detail == "operation_type_shape_must_use_valid_provider_extent"
                    and isinstance(operation_type, Mapping)
                    and isinstance(operation_type.get("shape"), list)
                    and any(
                        not (
                            type(extent) is int and extent >= 0
                            or isinstance(extent, str) and bool(extent)
                        )
                        for extent in operation_type["shape"]
                    )
                    or detail == "operation_type_shape_must_use_provider_dimension_alias"
                    and isinstance(operation_type, Mapping)
                    and isinstance(operation_type.get("shape"), list)
                    and any(
                        not is_provider_dimension_extent(
                            extent,
                            self._semantic_provider_dimension_aliases.values(),
                        )
                        for extent in operation_type["shape"]
                    )
                    or detail == "operation_type_shape_must_use_provider_dimension_alias"
                    and operation == "iota"
                    and isinstance(row.get("attrs"), Mapping)
                    and not is_provider_dimension_extent(
                        row["attrs"].get("n"),
                        self._semantic_provider_dimension_aliases.values(),
                    )
                    or detail == "operation_iota_result_type_must_match_attrs"
                    and operation == "iota"
                    or detail == "operation_nontrailing_broadcast_requires_local_id"
                    and isinstance(operation, str)
                    and operation in {"map_binary", "select"}
                    and not isinstance(row.get("id"), str)
                )
                if bad:
                    selected_path, selected_row = path, row
                    break
                visible_labels.update(
                    value
                    for value in result_labels
                    if isinstance(value, str)
                )
            if selected_row is None:
                selected_path, selected_row = structural_rows[0]
            top_level = re.match(r"body\[([0-9]+)\]", selected_path or "")
            if top_level is not None:
                diagnostic_result["row_index"] = int(top_level.group(1))
            operation = selected_row.get("op")
            if isinstance(operation, str):
                diagnostic_result["operation"] = operation[:128]
            diagnostic_field = target_field
            if (
                detail == "operation_type_shape_must_use_provider_dimension_alias"
                and operation == "iota"
                and isinstance(selected_row.get("attrs"), Mapping)
                and not is_provider_dimension_extent(
                    selected_row["attrs"].get("n"),
                    self._semantic_provider_dimension_aliases.values(),
                )
            ):
                diagnostic_field = "attrs.n"
                extent = selected_row["attrs"].get("n")
                if type(extent) is int and extent >= 0:
                    diagnostic_result["actual_extent"] = extent
                elif isinstance(extent, str) and re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_.:-]{0,63}", extent
                ):
                    diagnostic_result["actual_extent"] = extent
            diagnostic_result["field"] = diagnostic_field
            if detail == "operation_type_shape_must_use_provider_dimension_alias":
                diagnostic_result["provider_dimension_aliases"] = list(
                    self._semantic_provider_dimension_aliases.values()
                )
                diagnostic_result["dimension_reference_rule"] = (
                    PROVIDER_DIMENSION_EXTENT_RULE
                )
                selected_type = selected_row.get("type")
                shape = (
                    selected_type.get("shape")
                    if isinstance(selected_type, Mapping)
                    else None
                )
                if isinstance(shape, list):
                    diagnostic_result["actual_shape"] = copy.deepcopy(shape[:16])
                if selected_path is not None:
                    diagnostic_result["row_path"] = selected_path
        if (
            detail
            in {
                "operation_attrs_must_use_canonical_leaf_fields",
                "operation_attrs_must_include_required_leaf_fields",
                "operation_attr_value_must_match_canonical_leaf_contract",
            }
            and isinstance(body, list)
        ):
            for row_path, row in structural_rows:
                if not isinstance(row.get("op"), str):
                    continue
                try:
                    contract = operation_attr_contract(row["op"])
                except (KeyError, TypeError, ValueError):
                    continue
                allowed = contract.get("allowed")
                required = contract.get("required")
                attrs = row.get("attrs")
                if not isinstance(allowed, frozenset) or not isinstance(
                    required, frozenset
                ):
                    continue
                has_extra = isinstance(attrs, Mapping) and bool(set(attrs) - allowed)
                missing_required = not isinstance(attrs, Mapping) or bool(
                    required - set(attrs)
                )
                properties = contract.get("properties")
                invalid_value_name = (
                    next(
                        (
                            name
                            for name, attr_value in attrs.items()
                            if isinstance(properties, Mapping)
                            and not _operation_attr_value_matches_schema(
                                attr_value,
                                properties.get(name),
                            )
                        ),
                        None,
                    )
                    if isinstance(attrs, Mapping)
                    else None
                )
                if (
                    detail == "operation_attrs_must_use_canonical_leaf_fields"
                    and has_extra
                    or detail
                    == "operation_attrs_must_include_required_leaf_fields"
                    and missing_required
                    or detail == "operation_attr_value_must_match_canonical_leaf_contract"
                    and invalid_value_name is not None
                ):
                    diagnostic_result["row_path"] = row_path
                    top_level = re.match(r"body\[([0-9]+)\]", row_path)
                    if top_level is not None:
                        diagnostic_result["row_index"] = int(top_level.group(1))
                    diagnostic_result["operation"] = row["op"]
                    diagnostic_result["field"] = (
                        f"attrs.{invalid_value_name}"
                        if invalid_value_name is not None
                        else "attrs"
                    )
                    diagnostic_result["allowed_fields"] = sorted(allowed)
                    diagnostic_result["required_fields"] = sorted(required)
                    if invalid_value_name is not None:
                        diagnostic_result["attribute"] = invalid_value_name
                        property_schema = properties.get(invalid_value_name)
                        if isinstance(property_schema, Mapping):
                            # ``operation_attr_contract`` is the same
                            # canonical leaf authority already exposed by a
                            # permitted schema lookup and consumed by
                            # TypeCheck.  It contains no Source-specific
                            # value, candidate fragment, or host handle.
                            diagnostic_result["attribute_contract"] = copy.deepcopy(
                                dict(property_schema)
                            )
                    break

        # A graph may contain several invalid leaf attributes.  Returning only
        # the first one causes a model to repair one row, resubmit the whole
        # graph, and discover the next row in a new turn.  That is especially
        # expensive for the bounded MCP lifecycle and does not provide new
        # semantic evidence.  Report every offending operation/attribute in a
        # compact schema-only list; never copy the rejected attribute value,
        # operands, result labels, or any other candidate content.
        if (
            detail == "operation_attr_value_must_match_canonical_leaf_contract"
            and isinstance(body, list)
        ):
            invalid_contracts: list[dict[str, Any]] = []
            for row_path, row in structural_rows:
                operation = row.get("op")
                attrs = row.get("attrs")
                if not isinstance(operation, str) or not isinstance(attrs, Mapping):
                    continue
                try:
                    contract = operation_attr_contract(operation)
                except (KeyError, TypeError, ValueError):
                    continue
                properties = contract.get("properties")
                if not isinstance(properties, Mapping):
                    continue
                for attribute, attribute_value in attrs.items():
                    property_schema = properties.get(attribute)
                    if not isinstance(attribute, str) or not isinstance(property_schema, Mapping):
                        continue
                    if _operation_attr_value_matches_schema(
                        attribute_value, property_schema
                    ):
                        continue
                    invalid_contracts.append(
                        {
                            "row_path": row_path,
                            "operation": operation,
                            "attribute": attribute,
                            "attribute_contract": copy.deepcopy(dict(property_schema)),
                        }
                    )
                    if len(invalid_contracts) >= 16:
                        break
                if len(invalid_contracts) >= 16:
                    break
            if invalid_contracts:
                diagnostic_result["invalid_attribute_contracts"] = invalid_contracts
        if (
            detail
            in {
                "operation_id_must_be_provider_local_label",
                "operation_result_must_be_provider_local_label",
            }
            and isinstance(body, list)
        ):
            invalid_label_locations: list[dict[str, Any]] = []
            for row_path, row in structural_rows:
                fields = ("id",) if detail == "operation_id_must_be_provider_local_label" else ("result",)
                for field in fields:
                    value = row.get(field)
                    values = value if isinstance(value, list) else [value]
                    if value is None:
                        continue
                    invalid = any(not provider_local_label(item) for item in values)
                    if not invalid:
                        continue
                    invalid_label_locations.append(
                        {
                            "row_path": row_path,
                            "field": field,
                            "label_rule": (
                                "provider-local identifier matching "
                                "[A-Za-z_][A-Za-z0-9_.:-]* and not exactly "
                                "[n|v|x|p|c|a|e][0-9]+"
                            ),
                        }
                    )
                    if len(invalid_label_locations) >= 32:
                        break
                if len(invalid_label_locations) >= 32:
                    break
            if invalid_label_locations:
                diagnostic_result["invalid_provider_label_locations"] = (
                    invalid_label_locations
                )
        if (
            detail == "operation_result_type_shape_must_match_operands"
            and isinstance(body, list)
        ):
            # This is the same deliberately narrow shape-preservation proof
            # used by graph ingress.  In particular, ``()`` is a scalar and
            # must be treated as a real operand shape: a vector-plus-scalar
            # rejection still needs an exact expected shape in its receipt.
            # Return only a safe row path/op and the shape relation, never a
            # provider label, operand value, Source fragment, or candidate row.
            value_shapes = {
                value: shape
                for value, (_dtype, shape) in self._semantic_source_tensor_value_types.items()
            }
            rows = semantic_operation_wire_rows({"body": body})
            if rows is not None:
                for path, row in rows:
                    operands = row.get("operands")
                    operation_type = row.get("type")
                    row_result = row.get("result")
                    shape = (
                        tuple(operation_type.get("shape"))
                        if isinstance(operation_type, Mapping)
                        and isinstance(operation_type.get("shape"), list)
                        else None
                    )
                    if isinstance(operands, list):
                        known = [
                            value_shapes[operand]
                            for operand in operands
                            if isinstance(operand, str) and operand in value_shapes
                        ]
                        expected = _expected_result_shape_for_known_operands(
                            row,
                            known,
                        )
                        if (
                            expected is not None
                            and shape is not None
                            and shape != expected
                        ):
                            diagnostic_result.update(
                                {
                                    "row_path": path,
                                    "operation": row.get("op"),
                                    "field": "type.shape",
                                    "actual_shape": list(shape),
                                    "expected_shape": list(expected),
                                    "shape_repair_rule": (
                                        "set this row's type.shape exactly to expected_shape; "
                                        "preserve its op and operands and resubmit the complete graph"
                                    ),
                                    "result_rank": len(shape),
                                    "expected_rank": len(expected),
                                }
                            )
                            match = re.match(r"body\[([0-9]+)\]", path)
                            if match is not None:
                                diagnostic_result["row_index"] = int(match.group(1))
                            break
                    if isinstance(row_result, str) and shape is not None:
                        value_shapes[row_result] = shape
        if detail == "operation_iota_result_type_must_match_attrs" and isinstance(
            body, list
        ):
            # Feed the same bounded type-relation shape used by other result
            # type diagnostics.  The operation-specific check remains
            # fail-closed, but its public repair surface is deliberately
            # operation-neutral so future type relations do not require a new
            # receipt field or a new Agent protocol branch.
            for row_path, row in structural_rows:
                if row.get("op") != "iota":
                    continue
                attrs = row.get("attrs")
                if not isinstance(attrs, Mapping):
                    continue
                extent = attrs.get("n")
                if not (
                    isinstance(extent, str)
                    and bool(extent)
                    or isinstance(extent, int)
                    and not isinstance(extent, bool)
                    and extent >= 0
                ):
                    continue
                diagnostic_result.update(
                    {
                        "row_path": row_path,
                        "operation": row.get("op"),
                        "field": "type",
                        "expected_shape": [extent],
                        "type_repair_rule": (
                            "set this row's declared type to the returned "
                            "operation type relation; preserve op, attrs, "
                            "operands and result"
                        ),
                    }
                )
                dtype = attrs.get("dtype")
                if isinstance(dtype, str) and dtype:
                    diagnostic_result["expected_dtype"] = dtype
                match = re.match(r"body\[([0-9]+)\]", row_path)
                if match is not None:
                    diagnostic_result["row_index"] = int(match.group(1))
                break
        if (
            detail == "operation_graph_must_expose_source_output_assembly"
            and isinstance(body, list)
        ):
            missing = self._semantic_graph_missing_output_assembly(body)
            if missing is not None:
                requirement = missing.get("requirement")
                if isinstance(requirement, Mapping):
                    diagnostic_result.update(
                        {
                            "output_ordinal": missing.get("output_ordinal"),
                            "field": "assembly",
                            "required_output_assembly": copy.deepcopy(dict(requirement)),
                            "assembly_rule": (
                                "one top-level canonical concat with the exact Source-proven "
                                "axis, ordered slice count, slice type and logical output type"
                            ),
                        }
                    )
        if (
            detail == "operation_graph_must_expose_each_source_output_type"
            and isinstance(body, list)
        ):
            missing = self._semantic_graph_missing_output_type(body)
            if missing is not None:
                output_ordinal, (dtype, shape) = missing
                required_type = (dtype, shape)
                diagnostic_result.update(
                    {
                        "output_ordinal": output_ordinal,
                        "field": "type",
                        "required_output_type": {
                            "dtype": dtype,
                            "shape": list(shape),
                        },
                        "function_visible_matching_value_count": (
                            self._semantic_graph_matching_output_type_count(
                                body,
                                required_type,
                            )
                        ),
                        # Output selection cannot create or type a producer.
                        # Keep this as a schema-level visibility rule rather
                        # than returning a label or a candidate fragment.
                        "output_type_visibility_rule": (
                            "function_visible_typed_result_or_source_input_with_exact_dtype_shape"
                        ),
                    }
                )
        if (
            detail == "operation_graph_must_cover_source_operation_families"
            and isinstance(body, list)
        ):
            rows = semantic_operation_wire_rows({"body": body})
            if rows is not None:
                counts = self._semantic_graph_operation_family_counts(rows)
                missing_families = [
                    {
                        "operation": name,
                        "required_operation_count": required,
                        "actual_operation_count": counts.get(name, 0),
                        "missing_operation_count": required - counts.get(name, 0),
                    }
                    for name, required in sorted(
                        self._semantic_entry_required_operation_families.items()
                    )
                    if counts.get(name, 0) < required
                ]
                if missing_families:
                    # Canonical Source-derived counters only; never expose
                    # rejected rows, operands, or provider-local labels.
                    diagnostic_result["missing_operation_families"] = (
                        missing_families[:16]
                    )
                missing_family = next(
                    (
                        (name, required, counts.get(name, 0))
                        for name, required in sorted(
                            self._semantic_entry_required_operation_families.items()
                        )
                        if counts.get(name, 0) < required
                    ),
                    None,
                )
                if missing_family is not None:
                    name, required, actual = missing_family
                    diagnostic_result.update(
                        {
                            "operation": name,
                            "field": "op",
                            "required_operation_count": required,
                            "actual_operation_count": actual,
                            "missing_operation_count": required - actual,
                        }
                    )
        if (
            detail == "operation_graph_must_cover_source_operation_attributes"
            and isinstance(body, list)
        ):
            rows = semantic_operation_wire_rows({"body": body})
            if rows is not None:
                for operation, attribute, value, required in (
                    self._semantic_entry_required_operation_attributes
                ):
                    actual = sum(
                        1
                        for _path, row in rows
                        if row.get("op") == operation
                        and isinstance(row.get("attrs"), Mapping)
                        and row["attrs"].get(attribute) == value
                    )
                    if actual >= required:
                        continue
                    diagnostic_result.update(
                        {
                            "operation": operation,
                            "field": f"attrs.{attribute}",
                            "attribute": attribute,
                            "required_attribute_value": value,
                            "required_operation_count": required,
                            "actual_operation_count": actual,
                        }
                    )
                    break
                if self._semantic_entry_required_operation_attributes:
                    attribute_gaps: list[dict[str, Any]] = []
                    for operation, attribute, value, required in (
                        self._semantic_entry_required_operation_attributes
                    ):
                        actual = sum(
                            1
                            for _path, row in rows
                            if row.get("op") == operation
                            and isinstance(row.get("attrs"), Mapping)
                            and row["attrs"].get(attribute) == value
                        )
                        if actual < required:
                            attribute_gaps.append(
                                {
                                    "operation": operation,
                                    "attribute": attribute,
                                    "required_attribute_value": value,
                                    "required_operation_count": required,
                                    "actual_operation_count": actual,
                                    "missing_operation_count": required - actual,
                                }
                            )
                    if attribute_gaps:
                        diagnostic_result["missing_operation_attributes"] = (
                            attribute_gaps[:16]
                        )
        if (
            detail
            == "operation_graph_must_expose_compatible_access_predicate_type"
            and isinstance(body, list)
        ):
            missing = self._semantic_graph_missing_access_predicate_type(body)
            if missing is not None:
                (
                    access_ordinal,
                    matching_count,
                    boolean_result_count,
                    incompatible_shape_count,
                ) = missing
                diagnostic_result.update(
                    {
                        "access_ordinal": access_ordinal,
                        "field": "predicate",
                        "function_visible_boolean_result_count": (
                            boolean_result_count
                        ),
                        "function_visible_compatible_boolean_result_count": (
                            matching_count
                        ),
                        "incompatible_boolean_shape_count": incompatible_shape_count,
                        "predicate_projection_status": (
                            "no_boolean_result"
                            if boolean_result_count == 0
                            else "no_exact_shape_projection"
                        ),
                        # This is a Source-owned type relation, not a
                        # provider-owned predicate. Counts distinguish a
                        # missing boolean result from a shape mismatch without
                        # exposing labels or rejected operation rows.
                        "predicate_shape_rule": (
                            "i1_with_one_unique_exact_rank_insertion"
                        ),
                    }
                )
                try:
                    relations = self._semantic_access_predicate_relations()
                    relation = (
                        relations[access_ordinal]
                        if relations is not None
                        and 0 <= access_ordinal < len(relations)
                        else None
                    )
                    target_shape = (
                        relation.get("target_shape")
                        if isinstance(relation, Mapping)
                        else None
                    )
                    if isinstance(target_shape, tuple):
                        diagnostic_result["predicate_target_shape"] = list(
                            target_shape
                        )
                except (IndexError, KeyError, TypeError):
                    # The public rule remains sufficient if a malformed
                    # mounted slot prevents this optional detail.
                    pass
        if (
            detail == "operation_graph_must_expose_access_consumer_id"
            and isinstance(body, list)
        ):
            missing = self._semantic_graph_missing_access_consumer_id(body)
            if missing is not None:
                access_ordinal, matching_count = missing
                diagnostic_result.update(
                    {
                        "access_ordinal": access_ordinal,
                        "field": "node_ref",
                        "direct_single_result_consumer_id_count": matching_count,
                    }
                )
        return diagnostic_result

    def _safe_operation_graph_receipt_diagnostic(
        self,
        declaration_text: str | None,
        body: Any,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """Keep diagnostic construction from becoming a second host failure.

        The detailed diagnostic is intentionally best effort.  If a malformed
        provider declaration reaches an unexpected branch in that diagnostic,
        return only a container kind and digest.  This preserves fail-closed
        graph acceptance without exposing declaration bytes or an internal
        traceback, and it keeps the original public rule in the surrounding
        receipt.
        """

        try:
            return self._operation_graph_receipt_diagnostic(
                declaration_text,
                body,
                detail,
            )
        except Exception:
            if isinstance(body, list):
                container = "array"
                try:
                    normalized = json.dumps(
                        body,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                except (TypeError, ValueError, UnicodeError):
                    normalized = b"<unserializable-json-array>"
            elif body is None:
                container = "invalid"
                normalized = (declaration_text or "").encode(
                    "utf-8", "replace"
                )
            else:
                container = _json_container_kind(body)
                normalized = f"<{container}>".encode("ascii")
            return {
                "json_container": container,
                "declaration_sha256": hashlib.sha256(normalized).hexdigest(),
                "diagnostic": "operation_graph_diagnostic_unavailable",
                "failure_layer": "host_diagnostic",
            }

    def _record_operation_graph_failure(
        self,
        *,
        detail: str,
        declaration_text: str | None,
        body: Any,
    ) -> None:
        """Bound a failed graph phase without retaining provider bytes."""

        diagnostic = self._safe_operation_graph_receipt_diagnostic(
            declaration_text,
            body,
            detail,
        )
        phase, phase_digest = self._mcp_phase_failure_key()
        same_rule_count = sum(
            row.get("detail") == detail
            and row.get("phase_digest") == phase_digest
            for row in self._semantic_operation_graph_failures
        ) + 1
        # A rule can remain the same while the Agent corrects one missing
        # Source-proven family and exposes another.  No-progress must compare
        # the bounded public gap, not merely the rule label; otherwise a
        # changing graph is terminated as if it were an unchanged repair.
        gap_fields = (
            "operation",
            "row_path",
            "row_index",
            "field",
            "attribute",
            "required_attribute_value",
            "required_operation_count",
            "actual_operation_count",
            "missing_operation_count",
            "missing_operation_families",
            "missing_operation_attributes",
            "expected_shape",
            "actual_shape",
            "actual_extent",
            "expected_rank",
            "result_rank",
            "expected_dtype",
            "type_repair_rule",
            "required_output_type",
            "function_visible_matching_value_count",
            "function_visible_boolean_result_count",
            "function_visible_compatible_boolean_result_count",
            "incompatible_boolean_shape_count",
            "predicate_target_shape",
            "predicate_shape_rule",
            "predicate_projection_status",
            "output_type_visibility_rule",
            "minimum_operation_rows",
            "operation_rows",
            "complete_graph_required",
            "graph_repair_rule",
        )
        safe_gap = {
            key: diagnostic[key]
            for key in gap_fields
            if key in diagnostic
        }
        gap_digest = hashlib.sha256(
            json.dumps(
                {
                    "detail": detail,
                    "phase_digest": phase_digest,
                    "gap": safe_gap,
                },
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        # An identical declaration is only a no-progress repeat when the
        # lifecycle edge and the bounded public gap are also unchanged.  The
        # same provider text can legitimately be observed after a fresh phase
        # transition or after the host exposes a different safe diagnostic;
        # declaration bytes alone are not a semantic progress predicate.
        repeated_unchanged = any(
            row.get("declaration_sha256") == diagnostic["declaration_sha256"]
            and row.get("phase_digest") == phase_digest
            and row.get("gap_digest") == gap_digest
            for row in self._semantic_operation_graph_failures
        )
        same_gap_count = sum(
            row.get("gap_digest") == gap_digest
            for row in self._semantic_operation_graph_failures
        ) + 1
        row = {
            "step": "declare_operation_graph",
            "status": "fail",
            "detail": detail,
            **diagnostic,
            "attempt": len(self._semantic_operation_graph_failures) + 1,
            "phase": phase,
            "phase_digest": phase_digest,
            "failure_class": "mcp_graph_rejected",
            "progress_changed": False,
            "same_rule_count": same_rule_count,
            "same_gap_count": same_gap_count,
            "gap_digest": gap_digest,
        }
        if repeated_unchanged:
            row["repeated_unchanged"] = True
        self._semantic_operation_graph_failures.append(dict(row))
        self._semantic_construction_audit.append(row)
        if repeated_unchanged or same_gap_count >= _MAX_SEMANTIC_NO_PROGRESS_RULE_FAILURES:
            self._raise_lift_preparation_failure(
                detail="mcp_no_progress",
                attempted_step="declare_operation_graph",
                rule_code=detail,
            )
        if len(self._semantic_operation_graph_failures) >= _MAX_OPERATION_GRAPH_DECLARATION_FAILURES:
            self._raise_lift_preparation_failure(
                detail="mcp_no_progress",
                attempted_step="declare_operation_graph",
                rule_code="operation_graph_failure_window_exhausted",
            )

    def _record_semantic_access_failure(
        self,
        *,
        detail: str,
        declaration_text: str | None,
        accesses: Any,
        diagnostic: Mapping[str, Any] | None,
    ) -> None:
        """Bound failed complete-access declarations without retaining bytes.

        The begin receipt already contains every Source-order access slot.  A
        failed write may receive a bounded first-row template and diagnostic,
        but it cannot turn into an unbounded stream of empty/prefix arrays.
        Keep only a digest and row-count telemetry; rejected declaration bytes
        must not become a candidate, diagnostic template, or repair base.
        """

        serialized = (
            declaration_text.encode("utf-8")
            if isinstance(declaration_text, str)
            else b""
        )
        declaration_sha256 = hashlib.sha256(serialized).hexdigest()
        repeated_unchanged = any(
            row.get("declaration_sha256") == declaration_sha256
            and row.get("detail") == detail
            for row in self._semantic_access_declaration_failures
        )
        safe_diagnostic: dict[str, Any] = {}
        if isinstance(diagnostic, Mapping):
            # Retain only Source-slot coordinates and field names.  In
            # particular, do not copy an actual provider value from a rejected
            # declaration into the audit/evidence lineage.
            for field in (
                "row_index",
                "expected_row_count",
                "actual_row_count",
                "expected",
                "category",
                "required_fields",
                "missing_fields",
                "field",
                "provider_local_operation_labels",
            ):
                value = diagnostic.get(field)
                if field in {
                    "required_fields",
                    "missing_fields",
                    "provider_local_operation_labels",
                }:
                    if isinstance(value, list) and all(
                        isinstance(item, str) for item in value
                    ):
                        safe_diagnostic[field] = list(value)
                elif field in {"row_index", "expected_row_count", "actual_row_count"}:
                    if isinstance(value, int) and not isinstance(value, bool):
                        safe_diagnostic[field] = value
                elif isinstance(value, str):
                    safe_diagnostic[field] = value
        row = {
            "step": "declare_accesses",
            "status": "fail",
            "detail": detail,
            "declaration_sha256": declaration_sha256,
            "row_count": len(accesses) if isinstance(accesses, list) else None,
            "attempt": len(self._semantic_access_declaration_failures) + 1,
        }
        if safe_diagnostic:
            row["diagnostic"] = safe_diagnostic
        if repeated_unchanged:
            row["repeated_unchanged"] = True
        self._semantic_access_declaration_failures.append(dict(row))
        self._semantic_construction_audit.append(row)
        if repeated_unchanged:
            self._raise_lift_preparation_failure(
                detail="mcp_no_progress",
                attempted_step="declare_accesses",
                rule_code=detail,
            )
        if len(self._semantic_access_declaration_failures) >= (
            _MAX_ACCESS_DECLARATION_FAILURES
        ):
            self._raise_lift_preparation_failure(
                detail="semantic_access_declaration_failure_limit_exhausted",
                attempted_step="declare_accesses",
            )

    def _record_semantic_output_failure(
        self,
        *,
        detail: str,
        declaration_text: str | None,
        mappings: Any,
        diagnostic: Mapping[str, Any] | None,
    ) -> None:
        """Bound failed output selections without retaining provider bytes.

        The row count, failing field, and labels already authored in the same
        provider graph are safe structural receipts.  The rejected mapping
        itself is represented only by a digest and can never become a repair
        base or a host-selected output value.
        """

        serialized = (
            declaration_text.encode("utf-8")
            if isinstance(declaration_text, str)
            else b""
        )
        declaration_sha256 = hashlib.sha256(serialized).hexdigest()
        repeated_unchanged = any(
            row.get("declaration_sha256") == declaration_sha256
            and row.get("detail") == detail
            for row in self._semantic_output_declaration_failures
        )
        safe_diagnostic: dict[str, Any] = {}
        if isinstance(diagnostic, Mapping):
            row_index = diagnostic.get("row_index")
            field = diagnostic.get("field")
            labels = diagnostic.get("provider_local_result_labels")
            if isinstance(row_index, int) and not isinstance(row_index, bool):
                safe_diagnostic["row_index"] = row_index
            if isinstance(field, str):
                safe_diagnostic["field"] = field
            if (
                isinstance(labels, list)
                and len(labels) <= 64
                and all(isinstance(label, str) for label in labels)
            ):
                safe_diagnostic["provider_local_result_labels"] = list(labels)
        row = {
            "step": "declare_output_mappings",
            "status": "fail",
            "detail": detail,
            "declaration_sha256": declaration_sha256,
            "row_count": len(mappings) if isinstance(mappings, list) else None,
            "attempt": len(self._semantic_output_declaration_failures) + 1,
        }
        if safe_diagnostic:
            row["diagnostic"] = safe_diagnostic
        if repeated_unchanged:
            row["repeated_unchanged"] = True
        self._semantic_output_declaration_failures.append(dict(row))
        self._semantic_construction_audit.append(row)
        if repeated_unchanged:
            self._raise_lift_preparation_failure(
                detail="mcp_no_progress",
                attempted_step="declare_output_mappings",
                rule_code=detail,
            )
        if len(self._semantic_output_declaration_failures) >= (
            _MAX_OUTPUT_DECLARATION_FAILURES
        ):
            self._raise_lift_preparation_failure(
                detail="semantic_output_declaration_failure_limit_exhausted",
                attempted_step="declare_output_mappings",
            )

    @staticmethod
    def _unique_nontrailing_broadcast_axes(
        source_shape: tuple[str | int, ...],
        result_shape: tuple[str | int, ...],
    ) -> tuple[int, ...] | None:
        """Return a uniquely required rank insertion, never a generated row.

        This invokes the canonical broadcast projection used by J typing.  It
        only recognizes a relation when an unannotated operand cannot align,
        the rank difference fixes the number of inserted axes, and exactly one
        explicit axis set can align that operand with the already declared
        result type.  Ambiguous/value-dependent cases remain Decoder-owned.
        """

        if len(source_shape) >= len(result_shape):
            return None
        # The canonical projection accepts an empty axes list for a lower-rank
        # operand and leaves ordinary trailing NumPy broadcasting to its
        # consumer.  Determine that ordinary case directly before asking for
        # a rank insertion; otherwise [dim0] -> [dim0, dim1] would be
        # mistaken for a trailing broadcast merely because the projection
        # deferred alignment.
        trailing_compatible = all(
            source_extent == result_extent
            or source_extent == 1
            or result_extent == 1
            for source_extent, result_extent in zip(
                reversed(source_shape), reversed(result_shape)
            )
        )
        if trailing_compatible:
            return None
        missing_rank = len(result_shape) - len(source_shape)
        matches: list[tuple[int, ...]] = []
        for axes in combinations(range(len(result_shape)), missing_rank):
            try:
                _projected, inserted = _broadcast_projection(
                    list(source_shape), axes, list(result_shape)
                )
            except (NotImplementedError, ValueError):
                continue
            if tuple(inserted) == axes:
                matches.append(axes)
        return matches[0] if len(matches) == 1 else None

    def _semantic_boundary_broadcast_diagnostic(
        self, fields: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]] | None:
        """Require graph-implied non-trailing broadcast rows before finalize.

        The provider remains the sole author of ``boundary.broadcasts``.  This
        check only compares the provider's already declared graph types using
        the canonical projection and returns the exact provider-local relation
        still required by that graph.  It neither creates a row nor changes a
        value/result type, and intentionally leaves ambiguous relations to the
        normal Decoder/CandidateCheck authority.
        """

        buffer = self._semantic_construction_buffer
        j = buffer.get("j") if isinstance(buffer, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        body = function.get("body") if isinstance(function, Mapping) else None
        rows = semantic_operation_wire_rows({"body": body}) if isinstance(body, list) else None
        if rows is None:
            return None
        value_shapes = {
            value: shape
            for value, (_dtype, shape) in self._semantic_source_tensor_value_types.items()
        }
        required: list[dict[str, Any]] = []
        for _path, operation in rows:
            operands = operation.get("operands")
            operation_type = operation.get("type")
            result = operation.get("result")
            shape = (
                tuple(operation_type.get("shape"))
                if isinstance(operation_type, Mapping)
                and isinstance(operation_type.get("shape"), list)
                else None
            )
            if (
                operation.get("op") in {"map_binary", "select"}
                and isinstance(operands, list)
                and shape is not None
            ):
                node_ref = operation.get("id")
                for operand in operands:
                    source_shape = value_shapes.get(operand)
                    if not isinstance(operand, str) or source_shape is None:
                        continue
                    axes = self._unique_nontrailing_broadcast_axes(
                        source_shape, shape
                    )
                    if axes is None:
                        continue
                    if not isinstance(node_ref, str) or not node_ref:
                        return (
                            "boundary_broadcast_requires_provider_operation_id",
                            {"operation": operation.get("op")},
                        )
                    required.append(
                        {
                            "node_ref": node_ref,
                            "operand": operand,
                            "axes": list(axes),
                        }
                    )
            if isinstance(result, str) and shape is not None:
                value_shapes[result] = shape
        if not required:
            return None
        broadcasts = fields.get("broadcasts", [])
        if not isinstance(broadcasts, list):
            return (
                "boundary_broadcasts_must_be_json_array",
                {"required_broadcasts": required},
            )
        declared = {
            (row.get("node_ref"), row.get("operand")): row.get("axes")
            for row in broadcasts
            if isinstance(row, Mapping)
        }
        missing = [
            row
            for row in required
            if declared.get((row["node_ref"], row["operand"])) != row["axes"]
        ]
        if missing:
            return (
                "boundary_broadcasts_must_cover_nontrailing_operations",
                {
                    "required_broadcasts": missing,
                    "declared_broadcast_count": len(broadcasts),
                },
            )
        return None

    def _record_semantic_boundary_fields_failure(
        self,
        *,
        detail: str,
        declaration_text: str | None,
        fields: Any,
    ) -> None:
        """Bound a repeated final semantic-field failure without retaining it."""

        serialized = (
            declaration_text.encode("utf-8")
            if isinstance(declaration_text, str)
            else b""
        )
        declaration_sha256 = hashlib.sha256(serialized).hexdigest()
        repeated_unchanged = any(
            row.get("declaration_sha256") == declaration_sha256
            and row.get("detail") == detail
            for row in self._semantic_boundary_field_failures
        )
        row = {
            "step": "declare_boundary_fields",
            "status": "fail",
            "detail": detail,
            "declaration_sha256": declaration_sha256,
            "field_names": (
                sorted(fields) if isinstance(fields, Mapping) else None
            ),
            "attempt": len(self._semantic_boundary_field_failures) + 1,
        }
        if repeated_unchanged:
            row["repeated_unchanged"] = True
        self._semantic_boundary_field_failures.append(dict(row))
        self._semantic_construction_audit.append(row)
        if repeated_unchanged:
            self._raise_lift_preparation_failure(
                detail="mcp_no_progress",
                attempted_step="declare_boundary_fields",
                rule_code=detail,
            )
        if len(self._semantic_boundary_field_failures) >= (
            _MAX_BOUNDARY_FIELDS_DECLARATION_FAILURES
        ):
            self._raise_lift_preparation_failure(
                detail="semantic_boundary_fields_failure_limit_exhausted",
                attempted_step="declare_boundary_fields",
            )

    def _semantic_write_receipt(
        self,
        *,
        status: str,
        step: str,
        detail: str,
        returncode: int,
        required_slots: Mapping[str, Any] | None = None,
        next_legal_action: str | None = None,
        canonical_row_template: Mapping[str, Any] | None = None,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "step": step,
            "detail": detail,
        }
        all_obligations = self._semantic_construction_slots.get(
            "obligations", ()
        )
        safe_diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}

        def step_obligations() -> list[dict[str, Any]]:
            kinds = {
                "declare_operation_graph": {"operation_graph"},
                "declare_accesses": {"access_binding", "access_index"},
                "declare_output_mappings": {"logical_output_binding"},
                "declare_boundary_fields": {
                    "source_owner_broadcast",
                    "source_computed_guard",
                },
            }.get(step, set())
            if not kinds:
                return []
            selected = [
                row
                for row in all_obligations
                if isinstance(row, Mapping) and row.get("kind") in kinds
            ]
            if status == "fail":
                ordinal = safe_diagnostic.get("row_index")
                if step == "declare_accesses" and is_canonical_ordinal(ordinal):
                    selected = [
                        row
                        for row in selected
                        if row.get("source_locator", {}).get("ordinal") == ordinal
                    ]
                elif step == "declare_output_mappings" and is_canonical_ordinal(
                    ordinal
                ):
                    selected = [
                        row
                        for row in selected
                        if row.get("source_locator", {}).get("output_ordinal")
                        == ordinal
                    ]
                elif step == "declare_boundary_fields":
                    if "broadcast" in detail:
                        selected = [
                            row
                            for row in selected
                            if row.get("kind") == "source_owner_broadcast"
                        ]
                    elif "guard" in detail or "predicate" in detail:
                        selected = [
                            row
                            for row in selected
                            if row.get("kind") == "source_computed_guard"
                        ]
                    else:
                        selected = []
            result_status = "rejected" if status == "fail" else "preflight_passed"
            return [
                semantic_obligation_diagnostic(
                    row,
                    self._semantic_construction_identity["source_facts_digest"],
                    status=result_status,
                )
                for row in selected
            ]

        obligation_results = step_obligations()
        if obligation_results:
            payload["obligation_results"] = obligation_results
            payload["obligation_pass_scope"] = (
                "declaration_preflight_only; final Source-owner admission remains pending"
                if status == "pass"
                else "the listed obligation was rejected by this declaration preflight"
            )
        if status == "fail":
            phase, phase_digest = self._mcp_phase_failure_key()
            failure_class = {
                "declare_operation_graph": "mcp_graph_rejected",
                "declare_accesses": "mcp_access_rejected",
                "declare_output_mappings": "mcp_output_rejected",
                "declare_boundary_fields": "mcp_boundary_rejected",
                "finalize_candidate": "mcp_finalize_rejected",
            }.get(step, "mcp_protocol_rejected")
            row_index = safe_diagnostic.get("row_index")
            row_path_value = safe_diagnostic.get("row_path")
            ordinal = (
                safe_diagnostic.get("access_ordinal")
                if step == "declare_operation_graph"
                else safe_diagnostic.get("output_ordinal")
                if step == "declare_output_mappings"
                else None
            )
            if isinstance(row_path_value, str) and re.fullmatch(
                r"body(?:\[\d+\])(?:\.(?:body|positive|negative))*(?:\[\d+\])?",
                row_path_value,
            ):
                rule_path = row_path_value
            elif isinstance(row_index, int) and not isinstance(row_index, bool):
                rule_path = (
                    f"body[{row_index}]"
                    if step == "declare_operation_graph"
                    else f"{step}[{row_index}]"
                )
            elif is_canonical_ordinal(ordinal):
                rule_path = (
                    f"accesses[{ordinal}]"
                    if step == "declare_operation_graph"
                    else f"output_mappings[{ordinal}]"
                )
            else:
                rule_path = step
            field = safe_diagnostic.get("field")
            if isinstance(field, str) and field:
                rule_path += f".{field[:128]}"

            expected: dict[str, Any] = {}
            actual: dict[str, Any] = {}
            expected_keys = (
                "expected_op_source",
                "label_rule",
                "allowed_fields",
                "required_fields",
                "attribute",
                "attribute_contract",
                "invalid_attribute_contracts",
                "invalid_provider_label_locations",
                "required_output_type",
                "required_output_assembly",
                "assembly_rule",
                "expected_shape",
                "shape_repair_rule",
                "provider_dimension_aliases",
                "dimension_reference_rule",
                "expected_dtype",
                "type_repair_rule",
                "required_attribute_value",
                "expected_rank",
                "required_operation_count",
                "missing_operation_families",
                "missing_operation_attributes",
                "expected_row_count",
                "minimum_operation_rows",
                "complete_graph_required",
                "graph_repair_rule",
                "expected",
                "category",
            )
            actual_keys = (
                "operation",
                "actual_operation_count",
                "missing_operation_count",
                "result_rank",
                "field",
                "missing_fields",
                "function_visible_matching_value_count",
                "function_visible_boolean_result_count",
                "function_visible_compatible_boolean_result_count",
                "incompatible_boolean_shape_count",
                "predicate_projection_status",
                "actual_row_count",
                "direct_single_result_consumer_id_count",
                "actual_shape",
                "actual_extent",
                "json_container",
                "operation_rows",
            )
            for key in expected_keys:
                value = safe_diagnostic.get(key)
                if isinstance(value, (str, int, float, bool, list, dict)):
                    expected[key] = copy.deepcopy(value)
            for key in actual_keys:
                value = safe_diagnostic.get(key)
                if isinstance(value, (str, int, float, bool, list, dict)):
                    actual[key] = copy.deepcopy(value)
            # The rejection code and path identify the failed slot, but not
            # every slot has a field-level binding rule in its rule name.
            # Attach the common operand reference contract at the field
            # boundary so each retry receives an actionable constraint
            # without echoing the rejected operand or graph.
            if (
                step == "declare_operation_graph"
                and safe_diagnostic.get("field") == "operands"
            ):
                expected["operand_reference_contract"] = (
                    "Each operand is a Source tensor t* or a provider-local result "
                    "declared by an earlier graph row; do not use host ABI/value "
                    "handles or forward references."
                )
            if not expected:
                expected = {"rule": detail[:128]}
            if not actual:
                actual = {"status": "rejected"}
            if step == "declare_operation_graph":
                repair_scope = (
                    "whole_graph"
                    if any(
                        marker in detail
                        for marker in (
                            "must_be_nonempty",
                            "must_cover",
                            "must_expose",
                            "lexical_wire",
                        )
                    )
                    else "single_operation"
                )
            elif step == "declare_accesses":
                repair_scope = (
                    "single_access"
                    if isinstance(row_index, int)
                    else "whole_accesses"
                )
            elif step == "declare_output_mappings":
                repair_scope = (
                    "single_output"
                    if isinstance(row_index, int)
                    else "whole_outputs"
                )
            elif step == "declare_boundary_fields":
                repair_scope = "single_boundary_field"
            else:
                repair_scope = "current_phase"
            payload.update(
                {
                    "failure_class": failure_class,
                    "rule_code": detail[:128],
                    "rule_path": rule_path,
                    "expected_relation": expected,
                    "actual_relation": actual,
                    "repair_scope": repair_scope,
                    "state_digest": phase_digest,
                    "next_legal_action": (
                        next_legal_action
                        if next_legal_action is not None
                        else self._semantic_next_legal_action()
                    ),
                    "progress_changed": False,
                    "phase": phase,
                }
            )
        if required_slots is not None:
            payload["required_slots"] = copy.deepcopy(dict(required_slots))
        if next_legal_action is not None:
            payload["next_legal_action"] = next_legal_action
        if canonical_row_template is not None:
            payload["canonical_row_template"] = copy.deepcopy(
                dict(canonical_row_template)
            )
        if diagnostic is not None:
            payload["diagnostic"] = copy.deepcopy(dict(diagnostic))
        return {
            "output": json.dumps(payload, sort_keys=True) + "\n",
            "returncode": returncode,
            "exception_info": "",
        }

    def _semantic_write_ready(self, step: str) -> dict[str, Any] | None:
        """Enforce the fresh semantic-entry read boundary for MCP writes."""

        if not self._semantic_entry_read:
            return self._semantic_write_receipt(
                status="fail",
                step=step,
                detail="read_source_brief_before_semantic_construction",
                returncode=1,
            )
        if not self._semantic_entry_transfer_ledger_read:
            return self._semantic_write_receipt(
                status="fail",
                step=step,
                detail="read_source_transfer_ledger_before_semantic_construction",
                returncode=1,
            )
        if step == "declare_accesses" and self._semantic_graph_requires_transfer_reread():
            return self._semantic_write_receipt(
                status="fail",
                step=step,
                detail="reread_source_transfer_ledger_after_operation_graph_before_accesses",
                returncode=1,
                next_legal_action="read_transfer_ledger_after_operation_graph",
            )
        return None

    def _semantic_construction_seed(self) -> dict[str, Any]:
        """Return the sole template available to this fresh MCP owner."""

        return _parse_candidate_object(self._semantic_candidate_skeleton)

    def _mcp_semantic_construction_write(
        self,
        name: str,
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Accumulate one bounded J semantic write and finalize once.

        The provider never writes a complete artifact here.  It may fill only
        source-visible J semantics and normalization classes in an isolated
        buffer. ``finalize_candidate`` hands that one object to the existing
        semantic preflight and canonical construction transaction, which is
        still the only path that creates C, canonical handles, effects and π
        bindings.
        """

        ready = self._semantic_write_ready(name)
        if ready is not None:
            return ready
        sequence_error = self._semantic_write_sequence_error((name,))
        if sequence_error is not None:
            if self._semantic_construction_buffer is not None and (
                sequence_error
                in {
                    "semantic_construction_duplicate_step",
                    "semantic_construction_optional_step_not_permitted",
                    "semantic_construction_is_already_complete",
                }
                or sequence_error.startswith(
                    "semantic_construction_only_next_required_step_permitted_"
                )
            ):
                self._raise_lift_preparation_failure(
                    detail=sequence_error,
                    attempted_step=name,
                )
            next_step = None
            template = None
            marker = "semantic_construction_required_step_missing_"
            if sequence_error.startswith(marker):
                next_step = sequence_error.removeprefix(marker)
                if next_step == "declare_accesses":
                    template = self._semantic_access_failure_template(())
                elif next_step == "declare_output_mappings":
                    template = self._semantic_output_failure_template(())
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail=sequence_error,
                returncode=1,
                next_legal_action=next_step,
                canonical_row_template=template,
            )
        if name == "begin_candidate":
            if self._semantic_construction_buffer is not None:
                return self._semantic_write_receipt(
                    status="fail",
                    step=name,
                    detail="semantic_construction_already_started",
                    returncode=1,
                )
            try:
                self._semantic_construction_buffer = copy.deepcopy(
                    self._semantic_construction_seed()
                )
            except Exception:
                return self._semantic_write_receipt(
                    status="fail",
                    step=name,
                    detail="semantic_construction_seed_unavailable",
                    returncode=1,
                )
            # This nonce is receipt-only state.  It identifies this fresh
            # in-memory transaction without asking the provider to echo it or
            # coupling separate fresh roots which happen to share a kernel.
            self._semantic_transaction_identity = "semantic-entry-" + os.urandom(16).hex()
            self._semantic_construction_audit.append(
                {
                    "step": name,
                    "status": "pass",
                    "mode": "semantic_entry",
                    "transaction": self._semantic_transaction_identity,
                }
            )
            self._semantic_construction_completed_steps.add(name)
            return self._semantic_write_receipt(
                status="pass",
                step=name,
                detail=(
                    "semantic_candidate_buffer_started; complete the listed "
                    "Source-order slots before finalize_candidate"
                ),
                returncode=0,
                required_slots=self._semantic_slot_receipt(),
            )

        buffer = self._semantic_construction_buffer
        if not isinstance(buffer, dict):
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail="begin_candidate_required",
                returncode=1,
            )
        j = buffer.get("j")
        pi = buffer.get("pi")
        if not isinstance(j, dict) or not isinstance(pi, dict):
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail="semantic_candidate_buffer_malformed",
                returncode=1,
            )
        boundary = j.get("boundary")
        intent = j.get("intent")
        function = intent.get("intent_function") if isinstance(intent, dict) else None
        if not isinstance(boundary, dict) or not isinstance(function, dict):
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail="canonical_semantic_template_malformed",
                returncode=1,
            )

        def declaration_text(step: str) -> str:
            row = self._semantic_write_contract["tool_arguments"].get(step)
            if (
                not isinstance(row, Mapping)
                or not isinstance(row.get("field"), str)
                or row["field"] not in action
            ):
                raise ValueError("canonical MCP declaration argument is unavailable")
            return str(action[row["field"]])

        graph_declaration_text: str | None = None
        access_declaration_text: str | None = None
        output_declaration_text: str | None = None
        boundary_declaration_text: str | None = None
        graph_body: Any = None
        try:
            deterministic_repair: str | None = None
            write_diagnostic: Mapping[str, Any] | None = None
            if name == "declare_operation_graph":
                graph_declaration_text = declaration_text(name)
                body = self._parse_semantic_declaration(graph_declaration_text)
                graph_body = body
                if not isinstance(body, list):
                    raise ValueError("operation_graph_must_be_nonempty_json_array")
                if not body:
                    raise ValueError("operation_graph_must_be_nonempty_json_array")
                if any(not isinstance(row, Mapping) for row in body):
                    raise ValueError("operation_graph_rows_must_be_json_objects")
                operation_diagnostic = self._semantic_operation_graph_diagnostic(body)
                if operation_diagnostic is not None:
                    raise ValueError(operation_diagnostic)
                function["body"] = copy.deepcopy(body)
            elif name == "declare_accesses":
                access_declaration_text = declaration_text(name)
                accesses = self._parse_semantic_declaration(access_declaration_text)
                if not isinstance(accesses, list) or any(
                    not isinstance(row, Mapping) for row in accesses
                ):
                    raise ValueError("accesses_must_be_json_array")
                slot_diagnostic = self._semantic_access_slot_diagnostic(accesses)
                if slot_diagnostic is not None:
                    detail, write_diagnostic = slot_diagnostic
                    raise ValueError(detail)
                boundary["accesses"] = copy.deepcopy(accesses)
            elif name == "declare_output_mappings":
                output_declaration_text = declaration_text(name)
                mappings = self._parse_semantic_declaration(output_declaration_text)
                if not isinstance(mappings, list) or any(
                    not isinstance(row, Mapping) for row in mappings
                ):
                    raise ValueError("output_mappings_must_be_json_array")
                mappings, ordinal_repaired = self._normalize_host_owned_output_ordinals(
                    mappings
                )
                if ordinal_repaired:
                    deterministic_repair = "output_ordinal_host_derived"
                slot_diagnostic = self._semantic_output_slot_diagnostic(mappings)
                if slot_diagnostic is not None:
                    write_diagnostic = self._semantic_output_failure_diagnostic(
                        mappings,
                        slot_diagnostic,
                    )
                    raise ValueError(slot_diagnostic)
                boundary["output_mappings"] = copy.deepcopy(mappings)
            elif name == "declare_boundary_fields":
                boundary_declaration_text = declaration_text(name)
                fields = self._parse_semantic_declaration(boundary_declaration_text)
                allowed = set(self._semantic_write_contract["boundary_fields"])
                types = self._semantic_write_contract["boundary_field_types"]
                if not isinstance(fields, Mapping) or not set(fields) <= allowed:
                    raise ValueError("boundary_fields_must_use_canonical_semantic_names")
                for field, value in fields.items():
                    expected = types.get(field)
                    actual = _json_container_kind(value)
                    if isinstance(expected, str) and actual != expected:
                        raise ValueError(f"boundary_field_{field}_has_wrong_container")
                    boundary[field] = copy.deepcopy(value)
                broadcast_diagnostic = self._semantic_boundary_broadcast_diagnostic(fields)
                if broadcast_diagnostic is not None:
                    detail, write_diagnostic = broadcast_diagnostic
                    raise ValueError(detail)
            elif name == "declare_j_fields":
                fields = self._parse_semantic_declaration(declaration_text(name))
                allowed = set(self._semantic_write_contract["j_fields"])
                types = self._semantic_write_contract["j_field_types"]
                if not isinstance(fields, Mapping) or not set(fields) <= allowed:
                    raise ValueError("j_fields_must_use_canonical_semantic_names")
                for field, value in fields.items():
                    expected = types.get(field)
                    actual = _json_container_kind(value)
                    if isinstance(expected, str) and actual != expected:
                        raise ValueError(f"j_field_{field}_has_wrong_container")
                    j[field] = copy.deepcopy(value)
            elif name == "declare_normalization_classes":
                classes = self._parse_semantic_declaration(declaration_text(name))
                if not isinstance(classes, list) or any(
                    not isinstance(row, Mapping) for row in classes
                ):
                    raise ValueError("normalization_classes_must_be_json_array")
                if classes:
                    pi["normalization_classes"] = copy.deepcopy(classes)
                else:
                    pi.pop("normalization_classes", None)
            elif name == "finalize_candidate":
                candidate_text = json.dumps(
                    buffer,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                self._semantic_construction_audit.append(
                    {"step": name, "status": "attempted"}
                )
                self._semantic_construction_completed_steps.add(name)
                return self._finalize_semantic_candidate(candidate_text)
            else:
                raise ValueError("unknown_semantic_construction_step")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            # Source-sized slot failures are intentionally actionable at the
            # declaration which caused them.  Keep all other parsing/schema
            # failures compact and generic so this MCP response never becomes
            # a second full validation authority.
            detail = str(exc)
            graph_diagnostic = (
                self._operation_graph_exception_diagnostic(exc)
                if name == "declare_operation_graph"
                else None
            )
            if graph_diagnostic is not None:
                detail = graph_diagnostic
                graph_receipt_diagnostic = self._safe_operation_graph_receipt_diagnostic(
                    graph_declaration_text,
                    graph_body,
                    detail,
                )
                self._record_operation_graph_failure(
                    detail=detail,
                    declaration_text=graph_declaration_text,
                    body=graph_body,
                )
                write_diagnostic = graph_receipt_diagnostic
            elif name == "declare_operation_graph":
                # A graph failure outside the public declaration taxonomy is
                # host state, not a provider repair task.  Do not disguise it
                # as a malformed provider graph and consume more model turns.
                self._raise_lift_preparation_failure(
                    detail="semantic_operation_graph_host_validation_failure",
                    attempted_step="declare_operation_graph",
                )
            elif name == "declare_accesses":
                if detail not in _PROVIDER_CORRECTABLE_ACCESS_DECLARATION_DIAGNOSTICS:
                    self._raise_lift_preparation_failure(
                        detail="semantic_access_declaration_host_validation_failure",
                        attempted_step="declare_accesses",
                        rule_code=_safe_access_host_failure_rule(exc),
                        origin=_safe_access_host_origin(exc),
                    )
                self._record_semantic_access_failure(
                    detail=detail,
                    declaration_text=access_declaration_text,
                    accesses=locals().get("accesses"),
                    diagnostic=(
                        write_diagnostic
                        if isinstance(write_diagnostic, Mapping)
                        else None
                    ),
                )
                if isinstance(write_diagnostic, Mapping):
                    write_diagnostic = {
                        key: value
                        for key, value in write_diagnostic.items()
                        if key
                        in {
                            "row_index",
                            "expected_row_count",
                            "actual_row_count",
                            "expected",
                            "category",
                            "required_fields",
                            "missing_fields",
                            "field",
                            "provider_local_operation_labels",
                        }
                    }
            elif name == "declare_output_mappings":
                if (
                    detail
                    not in _PROVIDER_CORRECTABLE_OUTPUT_DECLARATION_DIAGNOSTICS
                ):
                    self._raise_lift_preparation_failure(
                        detail="semantic_output_declaration_host_validation_failure",
                        attempted_step="declare_output_mappings",
                    )
                self._record_semantic_output_failure(
                    detail=detail,
                    declaration_text=output_declaration_text,
                    mappings=locals().get("mappings"),
                    diagnostic=(
                        write_diagnostic
                        if isinstance(write_diagnostic, Mapping)
                        else None
                    ),
                )
                if isinstance(write_diagnostic, Mapping):
                    write_diagnostic = {
                        key: value
                        for key, value in write_diagnostic.items()
                        if key
                        in {
                            "row_index",
                            "field",
                            "provider_local_result_labels",
                        }
                    }
            elif name == "declare_boundary_fields":
                if detail not in {
                    "boundary_fields_must_use_canonical_semantic_names",
                    "boundary_broadcasts_must_be_json_array",
                    "boundary_broadcasts_must_cover_nontrailing_operations",
                    "boundary_broadcast_requires_provider_operation_id",
                } and not detail.startswith("boundary_field_"):
                    self._raise_lift_preparation_failure(
                        detail="semantic_boundary_fields_host_validation_failure",
                        attempted_step="declare_boundary_fields",
                    )
                self._record_semantic_boundary_fields_failure(
                    detail=detail,
                    declaration_text=boundary_declaration_text,
                    fields=locals().get("fields"),
                )
            elif detail not in {
                # These are provider-correctable operation-graph errors.  Do
                # not collapse them to the generic schema message: doing so
                # makes a provider that used a host ``x*`` result label guess
                # at the JSON envelope instead of correcting that label.
                "operation_op_must_be_canonical_j_name",
                "operation_id_must_be_provider_local_label",
                "operation_result_must_be_provider_local_label",
                "operation_operand_must_reference_source_tensor_or_prior_local_result",
                "operation_result_must_be_new_provider_local_label",
                "operation_result_must_have_complete_value_type",
                "operation_type_shape_must_use_provider_dimension_alias",
                "operation_result_type_shape_must_match_operands",
                "operation_graph_lexical_wire_is_malformed",
                "access_declaration_must_cover_each_source_occurrence",
                "access_declaration_rows_must_be_json_objects",
                "access_declaration_must_follow_source_ordinal",
                "access_declaration_is_missing_required_source_slot_field",
                "access_declaration_must_not_use_id_alias",
                "access_declaration_provider_predicate_must_be_local_label",
                "access_declaration_provider_predicate_must_select_compatible_boolean_result",
                "access_declaration_provider_index_expression_must_be_local_expression",
                "access_declaration_provider_node_ref_must_be_local_label",
                "access_declaration_provider_node_ref_must_select_direct_consumer",
                "access_declaration_must_match_source_wire_template",
                "access_declaration_must_match_source_index_requirement",
                "output_declaration_must_cover_each_source_output",
                "output_declaration_rows_must_be_json_objects",
                "output_mappings_must_be_json_array",
                "output_declaration_is_missing_required_source_slot_field",
                "output_declaration_must_contain_only_value",
                "output_declaration_requires_declared_operation_graph",
                "output_declaration_value_must_reference_source_input_or_local_result",
                "output_declaration_value_must_be_function_visible",
                "output_declaration_value_type_must_match_source_output",
            }:
                detail = "invalid_schema_derived_semantic_declaration"
            access_template = (
                self._semantic_access_failure_template(
                    accesses if isinstance(locals().get("accesses"), list) else ()
                )
                if name == "declare_accesses"
                else None
            )
            output_template = (
                self._semantic_output_failure_template(
                    mappings if isinstance(locals().get("mappings"), list) else ()
                )
                if name == "declare_output_mappings"
                else None
            )
            access_slots = (
                {
                    "row_count": len(self._semantic_construction_slots["accesses"]),
                    "rows": self._semantic_access_failure_context(),
                    "rule": (
                        "resubmit one complete Source-ordinal access array; "
                        "copy fixed wire fields exactly, fill only provider-owned "
                        "local labels, and never use the rejected array as a base"
                    ),
                }
                if name == "declare_accesses"
                else None
            )
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail=detail,
                returncode=1,
                required_slots=access_slots,
                next_legal_action=name,
                canonical_row_template=access_template or output_template,
                diagnostic=locals().get("write_diagnostic"),
            )
        audit_row: dict[str, Any] = {"step": name, "status": "pass"}
        if deterministic_repair is not None:
            audit_row["deterministic_repair"] = deterministic_repair
        if name == "declare_operation_graph":
            self._semantic_operation_graph_transfer_generation = (
                self._semantic_transfer_read_generation
            )
            self._semantic_access_transfer_generation = None
            self._semantic_entry_transfer_ledger_read = False
            self._semantic_transfer_expected_start = 1
            audit_row["transfer_read_generation"] = (
                self._semantic_operation_graph_transfer_generation
            )
        if self._semantic_transaction_identity is not None:
            audit_row["transaction"] = self._semantic_transaction_identity
        self._semantic_construction_audit.append(audit_row)
        self._semantic_construction_completed_steps.add(name)
        next_step = self._semantic_next_legal_action()
        return self._semantic_write_receipt(
            status="pass",
            step=name,
            detail=(
                "semantic_required_slots_complete_finalize_now"
                if next_step == "finalize_candidate"
                else "semantic_declaration_recorded"
            ),
            returncode=0,
            next_legal_action=next_step,
        )

    def _finalize_semantic_candidate(self, candidate_text: str) -> dict[str, Any]:
        """Finalize the MCP-owned semantic buffer through the sole ingress."""

        def reject(result: Mapping[str, Any]) -> None:
            raise CandidateRejected(
                {
                    "role": "exit",
                    "content": "CandidateRejected",
                    "extra": {
                        "exit_status": "CandidateRejected",
                        "submission": "",
                        "diagnostics": dict(result),
                    },
                }
            )

        if self._candidate_submission_attempted:
            result = {
                "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                "status": "fail",
                "diagnostic": (
                    "one complete semantic candidate submission is permitted "
                    "per logical Lift; use the next fresh relift with its "
                    "structured guidance"
                ),
            }
            reject(result)
        self._candidate_submission_attempted = True
        if not isinstance(candidate_text, str) or not candidate_text:
            result = {
                "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                "status": "fail",
                "diagnostic": "finalize requires a non-empty semantic construction buffer",
            }
            reject(result)
        candidate_bytes = candidate_text.encode("utf-8")
        if not 1 <= len(candidate_bytes) <= MAX_CANDIDATE_BYTES:
            result = {
                "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                "status": "fail",
                "diagnostic": "candidate_text size is outside the allowed range",
            }
            reject(result)
        try:
            _parse_candidate_object(candidate_text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            result = {
                "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                "status": "fail",
                "diagnostic": (
                    "semantic construction buffer must serialize to one JSON candidate; "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
            reject(result)
        if self._semantic_candidate_checker is not None:
            try:
                surface = self._semantic_candidate_checker(candidate_text)
            except Exception as exc:
                diagnostics = _exception_candidate_check_diagnostics(exc)
                surface = {
                    "protocol": CANDIDATE_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": diagnostics
                    if diagnostics is not None
                    else [
                        {
                            "rule_code": "semantic_surface_check_failed",
                            "path": "<candidate>",
                            "observed_refs_or_types": [type(exc).__name__],
                        }
                    ],
                }
            if (
                not isinstance(surface, Mapping)
                or set(surface) != {"protocol", "status", "diagnostics"}
                or surface.get("protocol") != CANDIDATE_CHECK_PROTOCOL
                or surface.get("status") not in {"pass", "fail"}
                or not isinstance(surface.get("diagnostics"), list)
            ):
                raise RuntimeError("semantic candidate surface checker returned invalid result")
            if surface["status"] == "fail":
                # Semantic-surface admission happens before the candidate is
                # written and before host construction.  It is nevertheless
                # the only actionable reason for this fresh relift to end, so
                # retain the same bounded diagnostic shape as CandidateCheck.
                # Without this receipt the lifecycle incorrectly reduces a
                # precise semantic-wire rejection to "worker terminated
                # without a candidate" and the next fresh agent cannot fix
                # the actual field/handle violation.
                diagnostics = [
                    dict(item)
                    for item in surface["diagnostics"]
                    if isinstance(item, Mapping)
                ]
                self._candidate_check_audit.append(
                    {
                        "protocol": CANDIDATE_CHECK_PROTOCOL,
                        "status": "fail",
                        "trigger": "semantic_preflight",
                        "evaluated": True,
                        "budget_pool": "semantic_preflight",
                        "rule_codes": [
                            item.get("rule_code")
                            for item in diagnostics
                            if isinstance(item.get("rule_code"), str)
                        ],
                        "rule_paths": [
                            {
                                "rule_code": item["rule_code"],
                                "path": item["path"],
                            }
                            for item in diagnostics
                            if isinstance(item.get("rule_code"), str)
                            and isinstance(item.get("path"), str)
                        ],
                        "rule_diagnostics": diagnostics,
                    }
                )
                result = {
                    "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                    "status": "fail",
                    "diagnostics": surface["diagnostics"],
                    "authority": (
                        "schema-derived semantic surface preflight; no host "
                        "construction, repair, or acceptance occurred"
                    ),
                }
                reject(result)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".candidate-direct-submit-",
            dir=self.config.output_dir,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output_file:
                output_file.write(candidate_bytes)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary, self.config.output_dir / "candidate.json")
        finally:
            temporary.unlink(missing_ok=True)
        self._candidate_check_passed_bytes = None
        receipt = {
            "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
            "status": "written",
            "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "candidate_bytes": len(candidate_bytes),
            "authority": (
                "MCP-assembled semantic J bytes; finalize performs the required "
                "single-object JSON wire check before existing fail-closed "
                "preflight and canonical construction"
            ),
        }
        self._candidate_finalize_audit.append(dict(receipt))
        output = {
            "output": (
                INTENTIR_LIFT_SUBMIT
                + "\n"
                + json.dumps(receipt, sort_keys=True)
                + "\n"
            ),
            "returncode": 0,
            "exception_info": "",
        }
        self._check_submission(output)
        # _check_submission mutates the output with the bounded CandidateCheck
        # receipt before rejecting it.  End this logical Lift immediately so a
        # model cannot spend additional turns rereading the same J and trying
        # a second submission after the single-submit boundary.
        if output.get("returncode") != 0:
            reject(
                {
                    "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                    "status": "fail",
                    "diagnostic": "candidate submission failed host admission",
                    "output": output.get("output", ""),
                }
            )
        return output

    def stagnation_state(self) -> dict[str, Any]:
        """Return only the exact current candidate byte identity."""

        path = self.config.output_dir / "candidate.json"
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("candidate absent")
            raw = path.read_bytes()
        except (OSError, ValueError):
            return {"protocol": "intentir.lift-stagnation-state.v1", "candidate": None}
        return {
            "protocol": "intentir.lift-stagnation-state.v1",
            "candidate": {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            },
        }

    def lifecycle_progress_state(self) -> dict[str, Any]:
        """Return bounded MCP phase progress without Source-owned content.

        This is diagnostic state only.  It is intentionally limited to the
        current lifecycle edge, completed semantic write names, counters, and
        the next legal action.  It never exposes Source rows, J declarations,
        candidate bytes, provider output, or trajectory messages.
        """

        phase, phase_digest = self._mcp_phase_failure_key()
        completed = sorted(self._semantic_construction_completed_steps)
        last_completed = None
        if self._semantic_construction_audit:
            last = self._semantic_construction_audit[-1]
            if isinstance(last, Mapping) and isinstance(last.get("step"), str):
                last_completed = last["step"]
        last_failure: Mapping[str, Any] | None = None
        for failures in (
            self._semantic_boundary_field_failures,
            self._semantic_output_declaration_failures,
            self._semantic_access_declaration_failures,
            self._semantic_operation_graph_failures,
            self._mcp_phase_application_failures,
        ):
            if failures and isinstance(failures[-1], Mapping):
                last_failure = failures[-1]
                break
        return {
            "protocol": "intentir.lift-lifecycle-progress.v1",
            "phase": phase,
            "phase_digest": phase_digest,
            "next_legal_action": self._semantic_next_legal_action(),
            "last_completed_step": last_completed,
            "completed_step_count": len(completed),
            "transfer_read_generation": self._semantic_transfer_read_generation,
            "graph_failure_count": len(self._semantic_operation_graph_failures),
            "access_failure_count": len(self._semantic_access_declaration_failures),
            "output_failure_count": len(self._semantic_output_declaration_failures),
            "boundary_failure_count": len(self._semantic_boundary_field_failures),
            "mcp_phase_failure_count": len(self._mcp_phase_application_failures),
            "last_failure_class": (
                last_failure.get("failure_class")
                if last_failure is not None
                else None
            ),
            "last_rule_code": (
                last_failure.get("detail")
                or last_failure.get("category")
                if last_failure is not None
                else None
            ),
            "progress_changed": last_failure is None,
        }

    def material_progress_state(self) -> dict[str, Any]:
        """Return exact identities of model-writable artifact and work bytes."""

        files: list[dict[str, Any]] = []
        for label, root in (
            ("output", self.config.output_dir),
            ("work", self.config.scratch_dir),
        ):
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    files.append(
                        {"root": label, "path": relative, "kind": "symlink"}
                    )
                    continue
                if not path.is_file():
                    continue
                raw = path.read_bytes()
                files.append(
                    {
                        "root": label,
                        "path": relative,
                        "kind": "file",
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
        return {
            "protocol": "intentir.lift-material-progress.v1",
            "files": files,
        }

    def _execute_candidate_check(
        self,
        candidate_path_name: str,
        *,
        trigger: str = "explicit",
    ) -> dict[str, Any]:
        if trigger not in {"explicit", "submission"}:
            raise ValueError("unsupported CandidateCheck trigger")
        self._candidate_check_calls += 1
        self._candidate_check_calls_by_trigger[trigger] += 1
        candidate_bytes: bytes | None = None
        budget_pool: str | None = None
        checker_evaluated = False
        if candidate_path_name not in {
            "/work/candidate.json", "/output/candidate.json",
        }:
            result: Mapping[str, Any] = {
                "protocol": CANDIDATE_CHECK_PROTOCOL,
                "status": "fail",
                "diagnostics": [
                    {
                        "rule_code": "invalid_candidate_path",
                        "path": "<candidate>",
                        "observed_refs_or_types": [],
                    }
                ],
            }
        elif self._candidate_checker is None:
            result = {
                "protocol": CANDIDATE_CHECK_PROTOCOL,
                "status": "fail",
                "diagnostics": [
                    {
                        "rule_code": "candidate_check_unavailable",
                        "path": "<candidate>",
                        "observed_refs_or_types": [],
                    }
                ],
            }
        else:
            candidate_path = (
                self.config.scratch_dir / "candidate.json"
                if candidate_path_name.startswith("/work/")
                else self.config.output_dir / "candidate.json"
            )
            try:
                if candidate_path.is_symlink() or not candidate_path.is_file():
                    raise ValueError("candidate must be a regular file")
                size = candidate_path.stat().st_size
                if not 1 <= size <= MAX_CANDIDATE_BYTES:
                    raise ValueError("candidate size is outside the allowed range")
                candidate_bytes = candidate_path.read_bytes()
                if len(candidate_bytes) != size:
                    raise ValueError("candidate changed while being read")
                candidate_text = candidate_bytes.decode("utf-8")
            except Exception as exc:
                result = {
                    "protocol": CANDIDATE_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "candidate_check_failed",
                            "path": "<candidate>",
                            "observed_refs_or_types": [type(exc).__name__],
                        }
                    ],
                }
            else:
                if (
                    self.config.candidate_check_limit == 0
                    or self._candidate_check_shared_evaluations
                    < self.config.candidate_check_limit
                ):
                    self._candidate_check_shared_evaluations += 1
                    budget_pool = "shared"
                elif (
                    trigger == "submission"
                    and self._candidate_check_submission_reserved_evaluations
                    < self.config.candidate_submission_check_limit
                ):
                    self._candidate_check_submission_reserved_evaluations += 1
                    budget_pool = "submission_reserve"
                if budget_pool is None:
                    result = {
                        "protocol": CANDIDATE_CHECK_PROTOCOL,
                        "status": "fail",
                        "diagnostics": [
                            {
                                "rule_code": "check_budget_exhausted",
                                "path": "<candidate>",
                                "observed_refs_or_types": [],
                            }
                        ],
                    }
                else:
                    checker_evaluated = True
                    self._candidate_check_evaluations += 1
                    self._candidate_check_evaluations_by_trigger[trigger] += 1
                    try:
                        if self._candidate_constructor is not None:
                            constructed_text, construction_receipts = self._candidate_constructor(
                                candidate_text
                            )
                            if not isinstance(constructed_text, str) or not constructed_text:
                                raise TypeError(
                                    "canonical candidate constructor must return non-empty text"
                                )
                            if not isinstance(construction_receipts, tuple) or any(
                                not isinstance(item, Mapping) for item in construction_receipts
                            ):
                                raise TypeError(
                                    "canonical constructor receipts must be a tuple of mappings"
                                )
                            constructed_bytes = constructed_text.encode("utf-8")
                            if not 1 <= len(constructed_bytes) <= MAX_CANDIDATE_BYTES:
                                raise ValueError(
                                    "constructed candidate size is outside the allowed range"
                                )
                            before_sha256 = hashlib.sha256(
                                candidate_bytes
                            ).hexdigest()
                            after_sha256 = hashlib.sha256(constructed_bytes).hexdigest()
                            changed = constructed_bytes != candidate_bytes
                            if changed:
                                descriptor, temporary_name = tempfile.mkstemp(
                                    prefix=".candidate-construction-",
                                    dir=candidate_path.parent,
                                )
                                temporary = Path(temporary_name)
                                try:
                                    with os.fdopen(descriptor, "wb") as output:
                                        output.write(constructed_bytes)
                                        output.flush()
                                        os.fsync(output.fileno())
                                    os.replace(temporary, candidate_path)
                                finally:
                                    temporary.unlink(missing_ok=True)
                                if candidate_path.read_bytes() != constructed_bytes:
                                    raise RuntimeError(
                                        "candidate changed while canonical construction "
                                        "was being committed"
                                    )
                                candidate_bytes = constructed_bytes
                                candidate_text = constructed_text
                            self._candidate_constructor_audit.append(
                                {
                                    "status": "pass",
                                    "trigger": trigger,
                                    "changed": changed,
                                    "before_sha256": before_sha256,
                                    "after_sha256": after_sha256,
                                    "construction_receipt_count": len(construction_receipts),
                                    "rule_codes": [
                                        item.get("rule_code")
                                        for item in construction_receipts
                                        if isinstance(item.get("rule_code"), str)
                                    ],
                                }
                            )
                        result = self._candidate_checker(candidate_text)
                    except Exception as exc:
                        diagnostics = _exception_candidate_check_diagnostics(exc)
                        result = {
                            "protocol": CANDIDATE_CHECK_PROTOCOL,
                            "status": "fail",
                            "diagnostics": diagnostics
                            if diagnostics is not None
                            else [
                                {
                                    "rule_code": "candidate_check_failed",
                                    "path": "<candidate>",
                                    "observed_refs_or_types": [
                                        type(exc).__name__
                                    ],
                                }
                            ],
                        }
        protocol = result.get("protocol")
        status = result.get("status")
        diagnostics = result.get("diagnostics")
        if (
            set(result) != {"protocol", "status", "diagnostics"}
            or protocol != CANDIDATE_CHECK_PROTOCOL
            or status not in {"pass", "fail"}
            or not isinstance(diagnostics, list)
            or len(diagnostics) > 20
            or (status == "pass" and bool(diagnostics))
            or (status == "fail" and not diagnostics)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {
                    "rule_code",
                    "path",
                    "observed_refs_or_types",
                }
                or not isinstance(item.get("rule_code"), str)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("observed_refs_or_types"), list)
                or len(item.get("rule_code", "")) > MAX_CANDIDATE_CHECK_FIELD_CHARS
                or len(item.get("path", "")) > MAX_CANDIDATE_CHECK_FIELD_CHARS
                or len(item.get("observed_refs_or_types", []))
                > MAX_CANDIDATE_CHECK_OBSERVED_ITEMS
                or any(
                    not isinstance(observed, str)
                    or len(observed) > MAX_CANDIDATE_CHECK_FIELD_CHARS
                    for observed in item.get("observed_refs_or_types", [])
                )
                for item in diagnostics
            )
            or len(
                json.dumps(result, sort_keys=True, ensure_ascii=False).encode(
                    "utf-8"
                )
            )
            > MAX_CANDIDATE_CHECK_RESULT_BYTES
        ):
            raise RuntimeError("candidate checker returned an invalid closed result")
        self._candidate_check_audit.append(
            {
                "protocol": protocol,
                "status": status,
                "trigger": trigger,
                "budget_pool": budget_pool,
                "evaluated": checker_evaluated,
                "rule_codes": [
                    item.get("rule_code")
                    for item in diagnostics
                    if isinstance(item, Mapping)
                    and isinstance(item.get("rule_code"), str)
                ],
                "rule_paths": [
                    {
                        "rule_code": item.get("rule_code"),
                        "path": item.get("path"),
                    }
                    for item in diagnostics
                    if isinstance(item, Mapping)
                    and isinstance(item.get("rule_code"), str)
                    and isinstance(item.get("path"), str)
                ],
                "rule_diagnostics": [
                    {
                        "rule_code": item.get("rule_code"),
                        "path": item.get("path"),
                        "observed_refs_or_types": list(
                            item.get("observed_refs_or_types", [])
                        ),
                    }
                    for item in diagnostics
                    if isinstance(item, Mapping)
                    and isinstance(item.get("rule_code"), str)
                    and isinstance(item.get("path"), str)
                    and isinstance(item.get("observed_refs_or_types"), list)
                ],
            }
        )
        if status == "pass" and checker_evaluated and candidate_bytes is not None:
            self._candidate_check_passed_bytes = candidate_bytes
        elif (
            checker_evaluated
            and candidate_bytes == self._candidate_check_passed_bytes
        ):
            self._candidate_check_passed_bytes = None
        return {
            "output": json.dumps(result, sort_keys=True) + "\n",
            "returncode": 0 if status == "pass" else 1,
            "exception_info": "",
        }

    def _validate_paths(self) -> None:
        if self.config.tmp_dir is None:
            self.config.tmp_dir = self.config.scratch_dir.parent / "tmp"
        paths = (
            self.config.input_dir,
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        )
        resolved = tuple(path.resolve(strict=True) for path in paths)
        if len(set(resolved)) != len(resolved):
            raise ValueError("Lift input, output, and work directories must differ")
        if any(not path.is_dir() or path.is_symlink() for path in paths):
            raise ValueError("Lift sandbox paths must be real directories")
        (
            self.config.input_dir,
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        ) = resolved

    def _base_command(self) -> list[str]:
        command = [
            self.config.executable,
            "--unshare-all",
            "--unshare-user-try",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
        ]
        for system_path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            command.extend(["--ro-bind-try", system_path, system_path])
        for package_root in site.getsitepackages():
            intentir_package = (Path(package_root) / "intentir").resolve()
            if intentir_package.is_dir() and str(intentir_package).startswith(
                ("/usr/", "/lib/", "/lib64/")
            ):
                command.extend(
                    [
                        "--tmpfs",
                        str(intentir_package),
                        "--remount-ro",
                        str(intentir_package),
                    ]
                )
        command.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--ro-bind",
                str(self.config.input_dir),
                "/input",
                "--bind",
                str(self.config.output_dir),
                "/output",
                "--bind",
                str(self.config.scratch_dir),
                "/work",
                "--bind",
                str(self.config.tmp_dir),
                "/tmp",
                "--chdir",
                "/work",
                "--remount-ro",
                "/dev",
                "--remount-ro",
                "/",
            ]
        )
        for key, value in self.config.env.items():
            command.extend(["--setenv", key, value])
        return command

    def _probe(self) -> None:
        executable = shutil.which(self.config.executable)
        if executable is None:
            raise RuntimeError("IntentIR Lift requires bubblewrap (bwrap)")
        command = self._base_command() + [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            (
                "python -c 'import importlib.util,sys; "
                "sys.exit(importlib.util.find_spec(\"intentir\") is not None)'"
            ),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            raise RuntimeError(f"bubblewrap capability probe failed: {exc}") from exc
        if result.returncode != 0:
            diagnostic = result.stdout.strip()[:500]
            raise RuntimeError(
                "bubblewrap isolation probe failed"
                + (f": {diagnostic}" if diagnostic else "")
            )

    def _writable_usage(self) -> tuple[int, int, int]:
        entries = 0
        size = 0
        max_depth = 0
        stack = [
            (self.config.output_dir, 0),
            (self.config.scratch_dir, 0),
            (self.config.tmp_dir, 0),
        ]
        while stack:
            directory, depth = stack.pop()
            max_depth = max(max_depth, depth)
            if max_depth > MAX_WORKSPACE_DEPTH:
                return entries, size, max_depth
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > MAX_WORKSPACE_ENTRIES:
                        return entries, size, max_depth
                    item_stat = entry.stat(follow_symlinks=False)
                    size += max(item_stat.st_size, item_stat.st_blocks * 512)
                    if size > MAX_WORKSPACE_BYTES:
                        return entries, size, max_depth
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((Path(entry.path), depth + 1))
        return entries, size, max_depth

    def _reset_writable_dirs(self) -> None:
        for directory in (
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        ):
            os.chmod(directory, stat.S_IRWXU)
            for child in directory.iterdir():
                _remove_path_iterative(child)

    def reset_stage_execution_metrics(self) -> None:
        """Start a fresh per-agent-stage action timer without changing state."""

        self._stage_action_calls = 0
        self._stage_action_failures = 0
        self._stage_action_elapsed_ms = 0.0

    @property
    def stage_execution_metrics(self) -> dict[str, Any]:
        return {
            "elapsed_ms": self._stage_action_elapsed_ms,
            "count": self._stage_action_calls,
            "failures": self._stage_action_failures,
        }

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Time one agent action while preserving the closed executor path."""

        started = time.monotonic()
        self._stage_action_calls += 1
        try:
            return self._execute(action, cwd, timeout=timeout)
        except Exception:
            self._stage_action_failures += 1
            raise
        finally:
            self._stage_action_elapsed_ms += (
                time.monotonic() - started
            ) * 1000.0

    def _execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if cwd not in {"", "/work"}:
            raise ValueError("IntentIR Lift cwd is fixed at /work")
        command = action.get("command", "")
        if action.get("tool_name") == "mcp_call_tool":
            return self._execute_mcp(
                {
                    "method": action.get("method"),
                    "params": action.get("params"),
                }
            )
        if not isinstance(command, str):
            raise TypeError("Lift shell command must be a string")
        if "/input" in command and self._host_mcp_read_depth == 0:
            raise ValueError(
                "Lift owner input is available only through bounded MCP resources"
            )
        bounded_shell = (
            f"printf '{_SANDBOX_START_MARKER.decode()}'; "
            # Keep non-time resource protections, but do not impose a hidden
            # CPU-time deadline on an otherwise unbounded Lift action.
            "ulimit -f 8192 -n 64 -u 128 -v 524288; "
            "set -o pipefail; "
            f"/bin/bash --noprofile --norc -c \"$1\" 2>&1 | "
            f"/usr/bin/head -c {MAX_ACTION_OUTPUT_BYTES}"
        )
        invocation = self._base_command() + [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            bounded_shell,
            "intentir-lift-action",
            command,
        ]
        action_timeout = timeout if timeout is not None else self.config.timeout
        process: subprocess.Popen[bytes] | None = None
        limit_error = ""
        try:
            with tempfile.TemporaryFile() as output_file:
                process = subprocess.Popen(
                    invocation,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                )
                started = time.monotonic()
                while process.poll() is None:
                    if (
                        action_timeout is not None
                        and time.monotonic() - started >= action_timeout
                    ):
                        limit_error = (
                            f"action exceeded {action_timeout} second timeout"
                        )
                        process.kill()
                        break
                    try:
                        entries, size, depth = self._writable_usage()
                    except Exception as exc:
                        limit_error = (
                            "workspace became unreadable: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        process.kill()
                        break
                    if (
                        entries > MAX_WORKSPACE_ENTRIES
                        or size > MAX_WORKSPACE_BYTES
                        or depth > MAX_WORKSPACE_DEPTH
                    ):
                        limit_error = (
                            "workspace exceeded limit: "
                            f"entries={entries}/{MAX_WORKSPACE_ENTRIES}, "
                            f"bytes={size}/{MAX_WORKSPACE_BYTES}, "
                            f"depth={depth}/{MAX_WORKSPACE_DEPTH}"
                        )
                        process.kill()
                        break
                    time.sleep(0.05)
                returncode = process.wait(timeout=5)
                if not limit_error:
                    try:
                        entries, size, depth = self._writable_usage()
                    except Exception as exc:
                        limit_error = (
                            "workspace became unreadable: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        if (
                            entries > MAX_WORKSPACE_ENTRIES
                            or size > MAX_WORKSPACE_BYTES
                            or depth > MAX_WORKSPACE_DEPTH
                        ):
                            limit_error = (
                                "workspace exceeded limit: "
                                f"entries={entries}/{MAX_WORKSPACE_ENTRIES}, "
                                f"bytes={size}/{MAX_WORKSPACE_BYTES}, "
                                f"depth={depth}/{MAX_WORKSPACE_DEPTH}"
                            )
                output_file.seek(0)
                raw_output = output_file.read(
                    len(_SANDBOX_START_MARKER) + MAX_ACTION_OUTPUT_BYTES + 1
                )
            marker_at = raw_output.find(_SANDBOX_START_MARKER)
            if marker_at < 0:
                diagnostic = raw_output[:500].decode("utf-8", errors="replace")
                raise RuntimeError(
                    "bubblewrap action failed before sandbox start"
                    + (f": {diagnostic.strip()}" if diagnostic.strip() else "")
                )
            action_output = raw_output[marker_at + len(_SANDBOX_START_MARKER) :]
            output_text = action_output[:MAX_ACTION_OUTPUT_BYTES].decode(
                "utf-8", errors="replace"
            )
            output: dict[str, Any] = {
                "output": output_text,
                "returncode": returncode,
                "exception_info": limit_error,
            }
            if len(action_output) >= MAX_ACTION_OUTPUT_BYTES:
                output["returncode"] = returncode or 2
                output["exception_info"] = (
                    f"action output reached {MAX_ACTION_OUTPUT_BYTES} byte limit"
                )
            if limit_error:
                output["returncode"] = -1
                self._reset_writable_dirs()
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            raise RuntimeError(
                f"IntentIR Lift sandbox action failed: {type(exc).__name__}: {exc}"
            ) from exc
        return output

    def _check_submission(self, output: dict[str, Any]) -> None:
        """Commit only the internally emitted MCP finalize marker.

        Ordinary shell output no longer calls this method.  The marker is an
        implementation detail of ``finalize_candidate`` rather than an
        alternate agent submission protocol, so a model cannot write an
        arbitrary ``candidate.json`` and enter construction by printing it.
        """

        lines = str(output.get("output", "")).lstrip().splitlines()
        marker = INTENTIR_LIFT_SUBMIT
        if (
            not lines
            or lines[0].strip() != marker
            or output.get("returncode") != 0
        ):
            return
        if not self._semantic_entry_read:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected: read the semantic authoring entry through "
                "intentir://lift/source/brief before submitting the first candidate.\n"
            )
            return
        workspace = IntentIRLiftWorkspace(
            root=self.config.input_dir.parent,
            input_dir=self.config.input_dir,
            output_dir=self.config.output_dir,
            scratch_dir=self.config.scratch_dir,
            tmp_dir=self.config.tmp_dir,
        )
        try:
            candidate_text = workspace.read_candidate()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected by the local syntax check: "
                f"{type(exc).__name__}: {exc}\n"
            )
            return
        candidate_bytes = candidate_text.encode("utf-8")
        if self._candidate_checker is not None:
            if candidate_bytes != self._candidate_check_passed_bytes:
                preflight = self._execute_candidate_check(
                    "/output/candidate.json", trigger="submission"
                )
                if preflight["returncode"] != 0:
                    output["returncode"] = 2
                    output["output"] += (
                        "\nSubmission preflight rejected the current candidate. "
                        "Do not resubmit in this logical Lift; the next fresh "
                        "relift receives the bounded structured diagnostic.\n"
                        + str(preflight["output"])
                    )
                    return
        try:
            final_candidate_text = workspace.read_candidate()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected while confirming the checked bytes: "
                f"{type(exc).__name__}: {exc}\n"
            )
            return
        checked_bytes = self._candidate_check_passed_bytes
        if checked_bytes is None or final_candidate_text.encode("utf-8") != checked_bytes:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected because canonical candidate bytes changed "
                "after CandidateCheck. Start a new logical Lift rather than "
                "resubmitting this workspace.\n"
            )
            return
        raise Submitted(
            {
                "role": "exit",
                "content": "candidate.json submitted",
                "extra": {
                    "exit_status": "Submitted",
                    "submission": "candidate.json",
                },
            }
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "system": "Linux",
            "startup_context": self._startup_context,
            "semantic_candidate_skeleton": self._semantic_candidate_skeleton,
            "semantic_operation_namespace": self._semantic_operation_namespace,
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "sandbox": "intentir-bubblewrap-v1",
                        "network": "unshared",
                        "input": "read-only",
                        "output": "candidate-only",
                        "command_timeout_seconds": self.config.timeout,
                        "candidate_check_shared_limit": (
                            self.config.candidate_check_limit
                        ),
                        "candidate_check_submission_reserve": (
                            self.config.candidate_submission_check_limit
                        ),
                    },
                    "environment_type": (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    ),
                },
                "startup_context": dict(self._startup_context_receipt),
                "authoring": {
                    "mode": "semantic_entry",
                    "resource": "intentir://lift/source/brief",
                    "semantic_entry_required": True,
                    "semantic_entry_read": self._semantic_entry_read,
                    "semantic_entry_source_read": self._semantic_entry_source_read,
                    "semantic_entry_transfer_ledger_read": (
                        self._semantic_entry_transfer_ledger_read
                    ),
                    "semantic_transfer_read_generation": (
                        self._semantic_transfer_read_generation
                    ),
                    "operation_graph_transfer_generation": (
                        self._semantic_operation_graph_transfer_generation
                    ),
                    "access_phase_transfer_generation": (
                        self._semantic_access_transfer_generation
                    ),
                    "access_phase_transfer_reread_required": (
                        self._semantic_graph_requires_transfer_reread()
                    ),
                    "semantic_entry_supplemental_read": (
                        self._semantic_entry_supplemental_read
                    ),
                    "semantic_entry_schema_lookup_calls": (
                        self._semantic_entry_schema_lookup_calls
                    ),
                    "semantic_preconstruction_phase_failures": (
                        self._semantic_preconstruction_phase_failures
                    ),
                    "mcp_phase_application_failures": list(
                        self._mcp_phase_application_failures
                    ),
                    "preparation_failures": list(
                        self._preparation_failure_receipts
                    ),
                },
                "candidate_check": {
                    "calls": self._candidate_check_calls,
                    "attempts": self._candidate_check_calls,
                    "evaluations": self._candidate_check_evaluations,
                    "limit": (
                        0
                        if self.config.candidate_check_limit == 0
                        else self.config.candidate_check_limit
                        + self.config.candidate_submission_check_limit
                    ),
                    "time_governed": self.config.candidate_check_limit == 0,
                    "explicit": {
                        "calls": self._candidate_check_calls_by_trigger[
                            "explicit"
                        ],
                        "evaluations": (
                            self._candidate_check_evaluations_by_trigger[
                                "explicit"
                            ]
                        ),
                    },
                    "submission": {
                        "calls": self._candidate_check_calls_by_trigger[
                            "submission"
                        ],
                        "evaluations": (
                            self._candidate_check_evaluations_by_trigger[
                                "submission"
                            ]
                        ),
                    },
                    "shared_budget": {
                        "used": self._candidate_check_shared_evaluations,
                        "limit": self.config.candidate_check_limit,
                    },
                    "submission_reserve": {
                        "used": (
                            self._candidate_check_submission_reserved_evaluations
                        ),
                        "limit": self.config.candidate_submission_check_limit,
                    },
                    "results": list(self._candidate_check_audit),
                },
                "canonical_constructor": {
                    "authority": (
                        "one bounded deterministic Source/Harness construction before "
                        "CandidateCheck; construction telemetry is not part of its result"
                    ),
                    "calls": len(self._candidate_constructor_audit),
                    "results": list(self._candidate_constructor_audit),
                },
                "mcp_semantic_construction": {
                    "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                    "authority": (
                        "MCP fills only schema-derived J semantics and optional "
                        "normalization classes; finalize is the sole path to "
                        "canonical Source/Harness construction"
                    ),
                    "steps": list(self._semantic_construction_audit),
                    "finalizations": list(self._candidate_finalize_audit),
                },
                "multi_action_policy": {
                    "protocol": MULTI_ACTION_PROTOCOL,
                    "max_actions": 4,
                    "authority": "whole-response fail-before-execute",
                    "allowed": (
                        "independent read-only context/schema calls or one "
                        "ordered MCP semantic construction batch"
                    ),
                    "rejections": self._multi_action_rejections,
                },
                "structured_inspection": {
                    "protocol": "intentir.agent-structured-inspection.v1",
                    "authority": (
                        "bounded exact read-only views; no semantic classification, "
                        "candidate edit, or acceptance authority"
                    ),
                    "context_views": list(self._structured_context_queries),
                    "schema_bundles": [
                        list(items) for items in self._structured_schema_queries
                    ],
                },
                "mcp": {
                    "authority": (
                        "local bounded resources, the mechanical canonical-J "
                        "framework projection, and exact candidate tools only; "
                        "MCP does not invent unresolved semantics, repair, "
                        "select acceptance, or certify J/C/pi"
                    ),
                    "requests": list(self._mcp_requests),
                },
            }
        }


__all__ = [
    "CANDIDATE_CHECK_PROTOCOL",
    "MCP_SEMANTIC_FINALIZE_PROTOCOL",
    "INTENTIR_LIFT_SUBMIT",
    "MULTI_ACTION_PROTOCOL",
    "IntentIRLiftEnvironment",
    "IntentIRLiftEnvironmentConfig",
    "IntentIRLiftWorkspace",
    "MAX_ACTION_OUTPUT_BYTES",
    "MAX_CANDIDATE_BYTES",
    "MAX_STARTUP_CONTEXT_BYTES",
    "MAX_WORKSPACE_BYTES",
    "MAX_WORKSPACE_DEPTH",
    "MAX_WORKSPACE_ENTRIES",
]
