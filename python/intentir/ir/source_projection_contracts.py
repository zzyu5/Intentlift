"""Dependency-free primitive contracts for Source wire projections.

This module is also the low-level authority for Source-derived semantic
obligations.  Keeping obligation identity and validation here lets schema
construction, the mounted MCP reader, CandidateCheck, and Relift consume one
record instead of reconstructing the same Source relation independently.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

PROVIDER_DIMENSION_EXTENT_RULE = (
    "symbolic type.shape extents must be Source-issued dimN aliases; "
    "integer extents remain literal"
)


SEMANTIC_OBLIGATION_AUTHORITY = "source_construction"
SEMANTIC_OBLIGATION_STATUSES = frozenset(
    {"open", "preflight_passed", "satisfied", "rejected"}
)
SEMANTIC_OBLIGATION_STAGES = frozenset(
    {
        "operation_graph",
        "accesses",
        "boundary",
        "outputs",
        "candidate_admission",
        "gate1",
    }
)
_SEMANTIC_OBLIGATION_FIELDS = frozenset(
    {
        "obligation_id",
        "authority",
        "kind",
        "stage",
        "rule_code",
        "target_path",
        "target_field",
        "source_locator",
        "binding_contract",
        "status",
        "discharge_condition",
    }
)
_SEMANTIC_SOURCE_LOCATOR_FIELDS = frozenset(
    {
        "scope",
        "ordinal",
        "access_label",
        "owner_graph_node",
        "consumer_graph_node",
        "operand_position",
        "output_ordinal",
        "output",
        "dimension",
        "coordinate",
        "category",
    }
)
_SEMANTIC_OBLIGATION_FORBIDDEN_KEYS = frozenset(
    {
        "artifact",
        "candidate",
        "candidate_draft",
        "candidate_text",
        "draft",
        "operation_row",
        "provider_response",
        "raw_response",
        "raw_response_text",
        "repair_base",
        "response_text",
        "source_code",
        "source_program",
        "source_text",
        "trajectory",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _safe_obligation_value(value: Any, *, depth: int = 0) -> bool:
    """Whether a host-authored obligation value is bounded public metadata."""

    if depth > 6:
        return False
    if value is None or isinstance(value, (bool, str)):
        return not isinstance(value, str) or len(value) <= 1024
    if type(value) is int:
        return -(2**63) <= value < 2**63
    if isinstance(value, float):
        return False
    if isinstance(value, (list, tuple)):
        return len(value) <= 128 and all(
            _safe_obligation_value(item, depth=depth + 1) for item in value
        )
    if isinstance(value, Mapping):
        if len(value) > 64:
            return False
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key or len(raw_key) > 96:
                return False
            key = raw_key.lower()
            if key in _SEMANTIC_OBLIGATION_FORBIDDEN_KEYS or "token" in key:
                return False
            if not _safe_obligation_value(item, depth=depth + 1):
                return False
        return True
    return False


def canonical_semantic_obligation_id(
    source_facts_digest: str,
    *,
    authority: str,
    kind: str,
    source_locator: Mapping[str, Any],
    target_path: str,
) -> str:
    """Derive a stable, candidate-independent identity for one obligation."""

    if (
        not isinstance(source_facts_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_facts_digest) is None
        or not isinstance(authority, str)
        or not authority
        or not isinstance(kind, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{1,79}", kind) is None
        or not isinstance(target_path, str)
        or not target_path
        or not isinstance(source_locator, Mapping)
    ):
        raise ValueError("semantic obligation identity input is malformed")
    payload = {
        "source_facts_digest": source_facts_digest,
        "authority": authority,
        "kind": kind,
        "source_locator": dict(source_locator),
        "target_path": target_path,
    }
    return "obl_" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _source_obligation(
    source_facts_digest: str,
    *,
    kind: str,
    stage: str,
    rule_code: str,
    target_path: str,
    target_field: str,
    source_locator: Mapping[str, Any],
    binding_contract: Mapping[str, Any],
    discharge_validator: str,
    discharge_condition: str,
) -> dict[str, Any]:
    locator = copy.deepcopy(dict(source_locator))
    row = {
        "obligation_id": canonical_semantic_obligation_id(
            source_facts_digest,
            authority=SEMANTIC_OBLIGATION_AUTHORITY,
            kind=kind,
            source_locator=locator,
            target_path=target_path,
        ),
        "authority": SEMANTIC_OBLIGATION_AUTHORITY,
        "kind": kind,
        "stage": stage,
        "rule_code": rule_code,
        "target_path": target_path,
        "target_field": target_field,
        "source_locator": locator,
        "binding_contract": copy.deepcopy(dict(binding_contract)),
        "status": "open",
        "discharge_condition": {
            "validator": discharge_validator,
            "result": "pass",
            "condition": discharge_condition,
        },
    }
    validate_source_semantic_obligations([row], source_facts_digest)
    return row


def build_source_semantic_obligations(
    *,
    source_facts_digest: str,
    access_rows: Sequence[Mapping[str, Any]],
    output_rows: Sequence[Mapping[str, Any]],
    program_domain_requirements: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the single Source-derived obligation table for semantic ingress.

    The table describes required relations and their discharge authority.  It
    never allocates a provider-local operation, chooses a producer, or copies
    a private Source owner token.
    """

    obligations: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        obligations.append(_source_obligation(source_facts_digest, **kwargs))

    add(
        kind="operation_graph",
        stage="operation_graph",
        rule_code="semantic_operation_graph_required",
        target_path="j.intent.intent_function.body",
        target_field="body",
        source_locator={"scope": "source_semantic_operation_body"},
        binding_contract={
            "relation": "complete_producer_before_consumer_operation_graph",
            "operand_reference": "source_tensor_or_prior_provider_local_result",
        },
        discharge_validator="semantic_operation_graph_admission",
        discharge_condition=(
            "the complete typed operation graph passes canonical graph admission"
        ),
    )

    for access in access_rows:
        if not isinstance(access, Mapping):
            raise ValueError("semantic access obligation source is malformed")
        ordinal = access.get("ordinal")
        access_label = access.get("access_label")
        source = access.get("source_semantics")
        required_fields = access.get("required_fields")
        if (
            not is_canonical_ordinal(ordinal)
            or access_label != f"access{ordinal}"
            or not isinstance(source, Mapping)
            or not isinstance(required_fields, list)
        ):
            raise ValueError("semantic access obligation source is malformed")
        locator = {"ordinal": ordinal, "access_label": access_label}
        access_path = f"j.boundary.accesses[{ordinal}]"
        add(
            kind="access_binding",
            stage="accesses",
            rule_code="semantic_access_binding_required",
            target_path=access_path,
            target_field="access",
            source_locator=locator,
            binding_contract={
                "relation": "exact_source_occurrence_binding",
                "kind": source.get("kind"),
                "tensor": source.get("tensor"),
                "required_fields": copy.deepcopy(required_fields),
                "provider_owned_relations": copy.deepcopy(
                    source.get("provider_owned_relations", {})
                ),
            },
            discharge_validator="semantic_access_admission",
            discharge_condition=(
                "the Source-order access row and its provider-owned relations pass admission"
            ),
        )
        index_requirement = access.get("index_expression_requirement")
        if isinstance(index_requirement, str) and index_requirement:
            add(
                kind="access_index",
                stage="accesses",
                rule_code="source_access_index_relation_required",
                target_path=f"{access_path}.index_expression",
                target_field="index_expression",
                source_locator=locator,
                binding_contract={
                    "relation": "exact_source_index_expression",
                    "index_expression": index_requirement,
                },
                discharge_validator="semantic_access_admission",
                discharge_condition=(
                    "the submitted logical index equals the Source-proven coordinate expression"
                ),
            )
        authorizations = source.get("semantic_authorizations")
        owner = (
            authorizations.get("owner")
            if isinstance(authorizations, Mapping)
            else None
        )
        outline = owner.get("authoring_outline") if isinstance(owner, Mapping) else None
        graph = outline.get("graph") if isinstance(outline, Mapping) else None
        if isinstance(outline, Mapping) and isinstance(graph, list) and graph:
            add(
                kind="source_owner_producer",
                stage="candidate_admission",
                rule_code="source_owner_exact_producer_required",
                target_path=f"{access_path}.node_ref",
                target_field="node_ref",
                source_locator={
                    **locator,
                    "owner_graph_node": len(graph) - 1,
                },
                binding_contract={
                    "relation": "exact_source_owner_producer_dag",
                    "operation_family": owner.get("operation_family"),
                    "graph_node_count": len(graph),
                    "outline_locator": "source_owner_authoring_outline.graph",
                },
                discharge_validator="validate_source_owner_binding",
                discharge_condition=(
                    "the candidate node_ref is the unique typed result of the complete Source owner DAG"
                ),
            )
            broadcasts = source.get("owner_broadcast_requirements")
            if broadcasts is not None and not isinstance(broadcasts, list):
                raise ValueError("semantic owner broadcast obligations are malformed")
            for requirement in broadcasts or ():
                if not isinstance(requirement, Mapping):
                    raise ValueError("semantic owner broadcast obligation is malformed")
                consumer = requirement.get("consumer_graph_node")
                operand = requirement.get("operand_position")
                axes = requirement.get("target_axes")
                broadcast_locator = {
                    **locator,
                    "consumer_graph_node": consumer,
                    "operand_position": operand,
                }
                add(
                    kind="source_owner_broadcast",
                    stage="boundary",
                    rule_code="source_owner_broadcast_relation_missing",
                    target_path="j.boundary.broadcasts",
                    target_field="broadcasts",
                    source_locator=broadcast_locator,
                    binding_contract={
                        "relation": "exact_source_owner_broadcast_edge",
                        "target_axes": copy.deepcopy(axes),
                    },
                    discharge_validator="validate_source_owner_binding",
                    discharge_condition=(
                        "one boundary broadcast binds this exact owner consumer operand and target axis set"
                    ),
                )
        predicate_requirement = source.get("predicate_requirement")
        if source.get("masked") is True or predicate_requirement is not None:
            add(
                kind="source_computed_guard",
                stage="boundary",
                rule_code="source_owner_computed_guard_relation_missing",
                target_path="j.boundary.access_guard_relations",
                target_field="access_guard_relations",
                source_locator=locator,
                binding_contract={
                    "relation": "exact_source_computed_guard",
                    "predicate_requirement": predicate_requirement,
                    "polarity": "positive",
                    "result_dtype": "i1",
                },
                discharge_validator="validate_source_owner_binding",
                discharge_condition=(
                    "the access is bound to the unique Source-compatible typed guard and projection"
                ),
            )

    for output_ordinal, output in enumerate(output_rows):
        if not isinstance(output, Mapping):
            raise ValueError("semantic output obligation source is malformed")
        output_name = output.get("output")
        output_type = output.get("output_type")
        if not isinstance(output_name, str) or not isinstance(output_type, Mapping):
            raise ValueError("semantic output obligation source is malformed")
        locator = {"output_ordinal": output_ordinal, "output": output_name}
        add(
            kind="logical_output_binding",
            stage="outputs",
            rule_code="semantic_logical_output_binding_required",
            target_path=f"j.boundary.output_mappings[{output_ordinal}].node_ref",
            target_field="node_ref",
            source_locator=locator,
            binding_contract={
                "relation": "exact_source_abi_output_binding",
                "output_type": copy.deepcopy(dict(output_type)),
            },
            discharge_validator="semantic_output_admission",
            discharge_condition=(
                "the selected value is the unique typed logical result for this Source ABI output"
            ),
        )
        partition = output.get("source_store_partition_requirement")
        if partition is not None:
            if not isinstance(partition, Mapping):
                raise ValueError("semantic output partition obligation is malformed")
            add(
                kind="physical_store_partition",
                stage="candidate_admission",
                rule_code="source_output_partition_proof_required",
                target_path=f"j.boundary.output_mappings[{output_ordinal}].node_ref",
                target_field="node_ref",
                source_locator=locator,
                binding_contract={
                    "relation": partition.get("relation"),
                    "store_ordinals": copy.deepcopy(
                        partition.get("store_ordinals")
                    ),
                    "store_coordinates": copy.deepcopy(
                        partition.get("store_coordinates")
                    ),
                    **(
                        {"assembly": copy.deepcopy(partition["assembly"])}
                        if isinstance(partition.get("assembly"), Mapping)
                        else {}
                    ),
                },
                discharge_validator="validate_source_store_partition_assembly",
                discharge_condition=(
                    "all Source stores form one complete, ordered, nonoverlapping typed logical output partition"
                ),
            )

    if program_domain_requirements.get("status") == "pass":
        dimensions = program_domain_requirements.get("required_dimensions")
        coordinates = program_domain_requirements.get("domain_coordinates")
        if not isinstance(dimensions, list) or not isinstance(coordinates, Mapping):
            raise ValueError("semantic program-domain obligations are malformed")
        for dimension in dimensions:
            coordinate = coordinates.get(dimension)
            if not isinstance(dimension, str) or not isinstance(coordinate, str):
                raise ValueError("semantic program-domain obligation is malformed")
            add(
                kind="program_domain",
                stage="candidate_admission",
                rule_code="source_program_domain_projection_required",
                target_path="j.boundary.domains",
                target_field="domains",
                source_locator={
                    "dimension": dimension,
                    "coordinate": coordinate,
                },
                binding_contract={
                    "relation": "exact_zero_based_source_program_domain",
                    "coverage": program_domain_requirements.get("coverage"),
                    "submission_policy": program_domain_requirements.get(
                        "submission_policy"
                    ),
                },
                discharge_validator="materialize_source_program_domains",
                discharge_condition=(
                    "the complete provider domain map is omitted for atomic Source projection or exactly matches it"
                ),
            )

    return validate_source_semantic_obligations(obligations, source_facts_digest)


def validate_source_semantic_obligations(
    rows: Any,
    source_facts_digest: str,
) -> list[dict[str, Any]]:
    """Validate and copy the canonical Source obligation table fail-closed."""

    if (
        not isinstance(source_facts_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_facts_digest) is None
        or not isinstance(rows, list)
    ):
        raise ValueError("canonical semantic obligation table is malformed")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _SEMANTIC_OBLIGATION_FIELDS:
            raise ValueError("canonical semantic obligation row is malformed")
        obligation_id = row.get("obligation_id")
        authority = row.get("authority")
        kind = row.get("kind")
        stage = row.get("stage")
        rule_code = row.get("rule_code")
        target_path = row.get("target_path")
        target_field = row.get("target_field")
        locator = row.get("source_locator")
        binding = row.get("binding_contract")
        status = row.get("status")
        discharge = row.get("discharge_condition")
        if (
            authority != SEMANTIC_OBLIGATION_AUTHORITY
            or not isinstance(kind, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{1,79}", kind) is None
            or stage not in SEMANTIC_OBLIGATION_STAGES
            or not isinstance(rule_code, str)
            or re.fullmatch(r"[a-z][a-z0-9_.]{1,95}", rule_code) is None
            or not isinstance(target_path, str)
            or not 1 <= len(target_path) <= 256
            or not isinstance(target_field, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", target_field) is None
            or not isinstance(locator, Mapping)
            or not locator
            or set(locator) - _SEMANTIC_SOURCE_LOCATOR_FIELDS
            or not _safe_obligation_value(locator)
            or not isinstance(binding, Mapping)
            or not binding
            or not isinstance(binding.get("relation"), str)
            or not _safe_obligation_value(binding)
            or status not in SEMANTIC_OBLIGATION_STATUSES
            or not isinstance(discharge, Mapping)
            or set(discharge) != {"validator", "result", "condition"}
            or not isinstance(discharge.get("validator"), str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{1,95}", discharge["validator"])
            is None
            or discharge.get("result") != "pass"
            or not isinstance(discharge.get("condition"), str)
            or not 1 <= len(discharge["condition"]) <= 512
            or not _safe_obligation_value(discharge)
        ):
            raise ValueError("canonical semantic obligation row is malformed")
        expected_id = canonical_semantic_obligation_id(
            source_facts_digest,
            authority=authority,
            kind=kind,
            source_locator=locator,
            target_path=target_path,
        )
        if obligation_id != expected_id or obligation_id in seen:
            raise ValueError("canonical semantic obligation identity is malformed")
        seen.add(obligation_id)
        result.append(copy.deepcopy(dict(row)))
    return result
def source_semantic_obligation_index(
    rows: Any,
    source_facts_digest: str,
) -> dict[str, dict[str, Any]]:
    """Return the validated obligation table indexed by stable identity."""

    validated = validate_source_semantic_obligations(rows, source_facts_digest)
    return {row["obligation_id"]: row for row in validated}


def matching_source_semantic_obligations(
    rows: Any,
    source_facts_digest: str,
    *,
    kind: str | None = None,
    rule_code: str | None = None,
    source_locator: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select obligations by canonical fields without inferring a repair."""

    validated = validate_source_semantic_obligations(rows, source_facts_digest)
    locator = dict(source_locator or {})
    if set(locator) - _SEMANTIC_SOURCE_LOCATOR_FIELDS:
        raise ValueError("semantic obligation lookup locator is malformed")
    return [
        row
        for row in validated
        if (kind is None or row["kind"] == kind)
        and (rule_code is None or row["rule_code"] == rule_code)
        and all(row["source_locator"].get(key) == value for key, value in locator.items())
    ]


def semantic_obligation_diagnostic(
    row: Mapping[str, Any],
    source_facts_digest: str,
    *,
    status: str,
) -> dict[str, Any]:
    """Project one canonical obligation into the shared safe diagnostic shape."""

    validated = validate_source_semantic_obligations([dict(row)], source_facts_digest)[0]
    if status not in {"preflight_passed", "satisfied", "rejected"}:
        raise ValueError("semantic obligation diagnostic status is malformed")
    return {
        "obligation_id": validated["obligation_id"],
        "authority": validated["authority"],
        "kind": validated["kind"],
        "stage": validated["stage"],
        "rule_code": validated["rule_code"],
        "path": validated["target_path"],
        "target_field": validated["target_field"],
        "source_locator": copy.deepcopy(validated["source_locator"]),
        "binding_contract": copy.deepcopy(validated["binding_contract"]),
        "status": status,
        "discharge_condition": copy.deepcopy(validated["discharge_condition"]),
    }


def safe_semantic_obligation_guidance(value: Any) -> dict[str, Any] | None:
    """Keep only the bounded public fields of a host obligation result.

    This projection intentionally does not accept an obligation definition:
    only Source construction can issue one.  It lets downstream diagnostics
    preserve an already-issued identity without trusting arbitrary nested
    candidate or provider payloads.
    """

    if not isinstance(value, Mapping):
        return None
    obligation_id = value.get("obligation_id")
    authority = value.get("authority")
    kind = value.get("kind")
    stage = value.get("stage")
    target_field = value.get("target_field")
    locator = value.get("source_locator")
    binding = value.get("binding_contract")
    status = value.get("status")
    discharge = value.get("discharge_condition")
    if (
        not isinstance(obligation_id, str)
        or re.fullmatch(r"obl_[0-9a-f]{64}", obligation_id) is None
        or authority not in {SEMANTIC_OBLIGATION_AUTHORITY, "gate1_observer"}
        or not isinstance(kind, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{1,79}", kind) is None
        or stage not in SEMANTIC_OBLIGATION_STAGES
        or not isinstance(target_field, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", target_field) is None
        or not isinstance(locator, Mapping)
        or not locator
        or set(locator) - _SEMANTIC_SOURCE_LOCATOR_FIELDS
        or not _safe_obligation_value(locator)
        or not isinstance(binding, Mapping)
        or not binding
        or not _safe_obligation_value(binding)
        or status not in SEMANTIC_OBLIGATION_STATUSES
        or not isinstance(discharge, Mapping)
        or set(discharge) != {"validator", "result", "condition"}
        or not _safe_obligation_value(discharge)
    ):
        return None
    return {
        "obligation_id": obligation_id,
        "authority": authority,
        "kind": kind,
        "stage": stage,
        "target_field": target_field,
        "source_locator": copy.deepcopy(dict(locator)),
        "binding_contract": copy.deepcopy(dict(binding)),
        "status": status,
        "discharge_condition": copy.deepcopy(dict(discharge)),
    }


def gate1_observer_obligation_diagnostic(
    category: str,
    *,
    passed: bool,
) -> dict[str, Any]:
    """Return the same diagnostic envelope for one Gate1 observer category."""

    if re.fullmatch(r"[a-z][a-z0-9_]{1,63}", category) is None:
        raise ValueError("Gate1 observer obligation category is malformed")
    locator = {"category": category}
    target_path = f"observer_diff.categories[{category}]"
    identity_payload = {
        "authority": "gate1_observer",
        "kind": "gate1_structure_category",
        "source_locator": locator,
        "target_path": target_path,
    }
    result = {
        "obligation_id": "obl_"
        + hashlib.sha256(
            _canonical_json(identity_payload).encode("utf-8")
        ).hexdigest(),
        "authority": "gate1_observer",
        "kind": "gate1_structure_category",
        "stage": "gate1",
        "rule_code": f"gate1.{category}",
        "path": target_path,
        "target_field": "category",
        "source_locator": locator,
        "binding_contract": {
            "relation": "source_replay_algorithm_structure_category",
            "category": category,
        },
        "status": "satisfied" if passed else "rejected",
        "discharge_condition": {
            "validator": "Gate1ObserverComparison",
            "result": "pass",
            "condition": (
                "the normal Replay comparison passes this exact structural category"
            ),
        },
    }
    if safe_semantic_obligation_guidance(result) is None:
        raise ValueError("Gate1 observer obligation diagnostic is malformed")
    return result


def is_canonical_ordinal(value: Any) -> bool:
    """Return whether a Source-order ordinal is a strict non-negative int."""

    return type(value) is int and value >= 0


def provider_dimension_alias_map(
    rows: Any,
    *,
    canonical_dimensions: Any = None,
) -> dict[str, str]:
    """Validate and return the Source-owned ``xN`` -> provider ``dimN`` map.

    The ordinal relation is the authority.  Neither a provider label nor a
    tensor/value name can stand in for a symbolic extent, and malformed,
    sparse, reordered, or allocator-mismatched aliases fail closed.
    """

    if not isinstance(rows, list):
        raise ValueError("canonical Source dimension aliases are malformed")
    mapping: dict[str, str] = {}
    for ordinal, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or not is_canonical_ordinal(row.get("ordinal"))
            or row.get("ordinal") != ordinal
            or row.get("local") != f"dim{ordinal}"
        ):
            raise ValueError("canonical Source dimension aliases are not dense")
        mapping[f"x{ordinal}"] = f"dim{ordinal}"
    if canonical_dimensions is not None and (
        not isinstance(canonical_dimensions, (list, tuple))
        or list(canonical_dimensions) != [f"x{i}" for i in range(len(mapping))]
    ):
        raise ValueError("canonical Source dimension aliases differ from allocator")
    return mapping


def is_provider_dimension_extent(value: Any, aliases: Any) -> bool:
    """Whether a provider-side extent is literal or a declared Source dimN."""

    if type(value) is int:
        return value >= 0
    return isinstance(value, str) and value in aliases


def provider_shape_from_canonical(shape: Any, aliases: Mapping[str, str]) -> list[int | str]:
    """Project one canonical tensor shape through the validated Source aliases.

    This is the only xN -> dimN shape rewrite.  It rejects undeclared
    canonical dimensions instead of constructing a plausible provider alias.
    """

    if not isinstance(shape, list):
        raise ValueError("canonical tensor shape is malformed")
    projected: list[int | str] = []
    for extent in shape:
        if type(extent) is int and extent >= 0:
            projected.append(extent)
        elif isinstance(extent, str) and extent in aliases:
            projected.append(aliases[extent])
        else:
            raise ValueError("canonical tensor shape references an undeclared dimension")
    return projected


def owner_broadcast_requirements(outline: Any) -> list[dict[str, Any]]:
    """Project existing owner-DAG broadcast edges as provider locators.

    The returned graph indices are navigation facts only; they are not J
    identifiers or proof that a provider-authored boundary relation is valid.
    """

    if not isinstance(outline, Mapping):
        raise ValueError("Source owner broadcast outline is malformed")
    graph = outline.get("graph")
    if not isinstance(graph, list):
        raise ValueError("Source owner broadcast outline is malformed")
    result: list[dict[str, Any]] = []
    for consumer_index, consumer in enumerate(graph):
        if not isinstance(consumer, (list, tuple)) or len(consumer) != 5:
            raise ValueError("Source owner broadcast graph is malformed")
        children = consumer[4]
        if not isinstance(children, (list, tuple)):
            raise ValueError("Source owner broadcast children are malformed")
        for operand_index, child in enumerate(children):
            if type(child) is not int or not 0 <= child < consumer_index:
                raise ValueError("Source owner broadcast child is unbound")
            projection = graph[child]
            if not isinstance(projection, (list, tuple)) or len(projection) != 5:
                raise ValueError("Source owner broadcast producer is malformed")
            if projection[0] != "broadcast":
                continue
            shape, source_axes = projection[2], projection[3]
            if (
                not isinstance(shape, (list, tuple))
                or not isinstance(source_axes, (list, tuple))
                or any(type(axis) is not int for axis in source_axes)
                or len(set(source_axes)) != len(source_axes)
                or any(axis < 0 or axis >= len(shape) for axis in source_axes)
            ):
                raise ValueError("Source owner broadcast axes are malformed")
            result.append(
                {
                    "consumer_graph_node": consumer_index,
                    "operand_position": operand_index,
                    "target_axes": [
                        axis for axis in range(len(shape)) if axis not in source_axes
                    ],
                }
            )
    return result
