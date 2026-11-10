"""Bounded, source-aware context lookup for one IntentIR Lift worker.

The tool exposes only files already mounted in the worker sandbox.  It never
constructs or repairs J/C/pi.  Its purpose is to keep Source, Harness and
external diagnostics observable even when the model transport elides long
shell output.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Mapping, Sequence

try:
    from source_store_partition import validate_source_store_partition_assembly
    from source_projection_contracts import (
        is_canonical_ordinal as _is_canonical_ordinal,
        owner_broadcast_requirements,
        provider_dimension_alias_map,
        provider_shape_from_canonical,
        validate_source_semantic_obligations,
    )
except ImportError:
    from intentir.extraction.lifters.source_store_partition import (
        validate_source_store_partition_assembly,
    )
    from intentir.ir.source_projection_contracts import (
        is_canonical_ordinal as _is_canonical_ordinal,
        owner_broadcast_requirements,
        provider_dimension_alias_map,
        provider_shape_from_canonical,
        validate_source_semantic_obligations,
    )


MAX_OUTPUT_CHARS = 10_000
# The Lift observation wrapper may encode the MCP result once more before it
# reaches Responses.  Keep the first-turn projection compact enough that the
# complete canonical seed remains visible; exact Source/contract pages stay
# available through their named resources.
FIRST_TURN_BRIEF_CHARS = 4_800
QUERY_COMMAND = "python /input/tools/context_query.py"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_schema(input_root: Path) -> dict[str, Any]:
    """Load the exact schema mounted beside this Lift worker.

    The context tool is shipped as a standalone script, so it cannot import
    the host package.  It validates the host-generated wire bundle against the
    same ``schema/artifact.schema.json`` bytes that CandidateCheck receives.
    The script contains no second list of J, boundary, signature, or pi fields.
    """

    path = input_root / "schema" / "artifact.schema.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("mounted canonical artifact schema is unavailable")
    value = _load_json(path)
    if not isinstance(value, dict):
        raise TypeError("mounted artifact schema must be an object")
    return value


def _canonical_wire_seed(input_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the host-generated wire bundle and validate its mounted schema."""

    bundle_path = input_root / "schema" / "canonical_wire_bundle.json"
    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise FileNotFoundError("mounted canonical wire bundle is unavailable")
    bundle = _load_json(bundle_path)
    if not isinstance(bundle, Mapping):
        raise TypeError("mounted canonical wire bundle must be an object")
    profile = bundle.get("profile")
    artifact = bundle.get("artifact_seed")
    if not isinstance(profile, dict) or not isinstance(artifact, dict):
        raise TypeError("canonical wire bundle lacks profile or artifact seed")

    schema = _canonical_schema(input_root)
    schema_sha256 = hashlib.sha256(
        json.dumps(
            schema,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if profile.get("schema_sha256") != schema_sha256:
        raise ValueError(
            "canonical wire profile is generated from different artifact.schema.json bytes"
        )
    root_properties = schema.get("properties")
    required_root = schema.get("required")
    if not isinstance(root_properties, dict) or not isinstance(required_root, list):
        raise TypeError("canonical artifact schema has no root object metadata")
    skeleton_path = input_root / "schema" / "canonical_skeleton.json"
    if not skeleton_path.is_file() or skeleton_path.is_symlink():
        raise FileNotFoundError("mounted canonical skeleton is unavailable")
    skeleton = _load_json(skeleton_path)
    if artifact != skeleton:
        raise ValueError("canonical wire bundle seed differs from mounted skeleton")
    if set(artifact) != set(required_root):
        raise ValueError("canonical wire bundle seed has the wrong root keys")
    schema_value = root_properties.get("schema", {}).get("enum", [])
    if not schema_value or artifact.get("schema") != schema_value[0]:
        raise ValueError("canonical wire bundle has the wrong artifact schema")
    if profile.get("schema") != artifact.get("schema"):
        raise ValueError("canonical wire profile and artifact seed disagree")
    if profile.get("pi_schema") != artifact.get("pi", {}).get("schema"):
        raise ValueError("canonical wire profile and pi seed disagree")

    def schema_fields(path: str) -> list[str]:
        value: Any = schema
        for component in path.split("."):
            if not isinstance(value, Mapping):
                raise TypeError(f"canonical schema path is not an object: {path}")
            value = value.get(component)
        if not isinstance(value, Mapping):
            raise TypeError(f"canonical schema path is missing: {path}")
        properties = value.get("properties")
        if not isinstance(properties, Mapping):
            raise TypeError(f"canonical schema path has no properties: {path}")
        return [str(field) for field in properties]

    for profile_path, schema_path in (
        ("root", ""),
        ("j", "properties.j"),
        ("c", "properties.c"),
        ("pi", "properties.pi"),
    ):
        item = profile.get(profile_path)
        if not isinstance(item, Mapping):
            raise TypeError(f"canonical wire profile lacks {profile_path}")
        if profile_path == "root":
            fields = [str(field) for field in root_properties]
        else:
            fields = schema_fields(schema_path)
        profile_fields = item.get("fields")
        if (
            not isinstance(profile_fields, list)
            or not all(isinstance(field, str) for field in profile_fields)
            or len(profile_fields) != len(set(profile_fields))
            or profile_fields != fields
        ):
            raise ValueError(
                f"canonical wire profile {profile_path} fields/order drifted from schema"
            )
        required = (
            list(required_root)
            if profile_path == "root"
            else list(
                schema.get("properties", {})
                .get(profile_path, {})
                .get("required", [])
            )
        )
        if item.get("required") != required:
            raise ValueError(
                f"canonical wire profile {profile_path} required fields drifted"
            )

    if bundle.get("root_layout") != profile["root"]["fields"]:
        raise ValueError("canonical wire bundle root layout is stale")
    namespace_order = profile.get("namespace_order")
    source_owned = profile.get("source_owned_namespaces")
    candidate_owned = profile.get("candidate_owned_namespaces")
    if (
        not isinstance(namespace_order, list)
        or not namespace_order
        or len(namespace_order) != len(set(namespace_order))
        or not all(isinstance(item, str) and item for item in namespace_order)
        or not isinstance(source_owned, list)
        or not isinstance(candidate_owned, list)
        or set(source_owned) | set(candidate_owned) != set(namespace_order)
        or set(source_owned) & set(candidate_owned)
        or bundle.get("namespace_order") != namespace_order
    ):
        raise ValueError("canonical namespace ownership/order is not bundle-derived")
    if bundle.get("construction_records") != profile.get("construction_records"):
        raise ValueError("canonical construction records are not bundle-derived")
    operation_contracts = schema.get("x-intentir-operation-contracts")
    if (
        not isinstance(operation_contracts, Mapping)
        or profile.get("operation_contracts") != operation_contracts
        or bundle.get("profile", {}).get("operation_contracts") != operation_contracts
    ):
        raise ValueError("canonical operation contracts are not schema-derived")
    schema_sections = schema.get("x-intentir-schema-sections")
    if (
        not isinstance(schema_sections, Mapping)
        or profile.get("schema_sections") != schema_sections
        or bundle.get("schema_sections") != profile.get("schema_sections")
    ):
        raise ValueError("canonical schema-section index is not bundle-derived")
    return profile, artifact


def _validate_semantic_mcp_write_projection(
    semantic_candidate: Mapping[str, Any],
    mcp_write_protocol: Mapping[str, Any],
) -> None:
    """Reject a mounted semantic write wire that drifted from its J contract.

    This context tool deliberately cannot import the host package.  It instead
    verifies that the profile's source-brief projection is self-consistent
    before exposing it to the provider.  The host schema module remains the
    only place that declares the actual step names and destinations.
    """

    template = semantic_candidate.get("template")
    steps = mcp_write_protocol.get("steps")
    required_steps = mcp_write_protocol.get("required_steps")
    arguments = mcp_write_protocol.get("tool_arguments")
    descriptions = mcp_write_protocol.get("descriptions")
    writes = mcp_write_protocol.get("writes")
    if (
        not isinstance(template, Mapping)
        or mcp_write_protocol.get("initial_template") != template
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
        or not isinstance(arguments, Mapping)
        or set(arguments) != set(steps)
        or not isinstance(descriptions, Mapping)
        or set(descriptions) != set(steps)
        or any(not isinstance(text, str) or not text for text in descriptions.values())
        or not isinstance(writes, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, (str, list))
            for key, value in writes.items()
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
            for row in arguments.values()
        )
    ):
        raise ValueError("canonical MCP semantic write projection is malformed")


def _validate_source_store_projection(
    required_access_rows: list[Any],
    required_output_rows: list[Any],
    output_construction: list[Any],
    dimension_aliases: Mapping[str, str],
) -> None:
    """Validate the Source store locator against its same mounted projection.

    ``source_store_requirements`` and ``source_store_partition_requirement``
    are navigation projections, not proofs.  They nevertheless must remain
    exact projections of the Source-authorized access rows.  This reader has
    no host constructor available, so it joins the public requirement rows
    back to the public ``source_semantics``/owner outline already mounted in
    the same transaction.  It never invents a missing store, coordinate, or
    logical output binding; any incomplete join rejects the mounted bundle.
    """

    if len(required_output_rows) != len(output_construction):
        raise ValueError("canonical Source output projection cardinality is inconsistent")

    for output_ordinal, (required_row, output_row) in enumerate(
        zip(required_output_rows, output_construction, strict=True)
    ):
        if not isinstance(required_row, Mapping) or not isinstance(output_row, Mapping):
            raise ValueError("canonical Source output store projection is malformed")
        output_name = output_row.get("output")
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("canonical Source output store projection lacks output")

        raw_stores = required_row.get("source_store_requirements")
        raw_partition = required_row.get("source_store_partition_requirement")
        if raw_stores is None:
            if raw_partition is not None:
                raise ValueError(
                    "canonical Source partition locator lacks store requirements"
                )
            continue
        if (
            not isinstance(raw_stores, list)
            or not raw_stores
            or len({
                item.get("ordinal")
                for item in raw_stores
                if isinstance(item, Mapping)
            })
            != len(raw_stores)
        ):
            raise ValueError(
                f"canonical Source store requirements are malformed at output {output_ordinal}"
            )

        expected_stores: list[dict[str, Any]] = []
        for access_row in required_access_rows:
            if not isinstance(access_row, Mapping):
                raise ValueError("canonical Source access requirements are malformed")
            access_ordinal = access_row.get("ordinal")
            if not _is_canonical_ordinal(access_ordinal):
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
            if (
                not isinstance(family, str)
                or not family
                or not isinstance(coordinates, list)
            ):
                # An unclosed owner outline is not a physical-store locator.
                # It remains fail-closed at the owner/Source admission layer;
                # do not manufacture a partial expected row here.
                continue
            expected_stores.append(
                {
                    "ordinal": access_ordinal,
                    "kind": source.get("kind"),
                    "operation_family": family,
                    "coordinates": copy.deepcopy(coordinates),
                }
            )

        if raw_stores != expected_stores:
            raise ValueError(
                f"canonical Source store requirements are not access-bound at output {output_ordinal}"
            )

        if len(raw_stores) > 1:
            if (
                not isinstance(raw_partition, Mapping)
                or set(raw_partition)
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
                or raw_partition.get("output") != output_name
                or raw_partition.get("relation")
                != "physical_store_slices_require_complete_logical_output_binding"
                or raw_partition.get("store_ordinals")
                != [item["ordinal"] for item in expected_stores]
                or raw_partition.get("store_coordinates")
                != [item["coordinates"] for item in expected_stores]
            ):
                raise ValueError(
                    f"canonical Source partition locator is not store-bound at output {output_ordinal}"
                )
            validate_source_store_partition_assembly(
                raw_partition,
                access_rows=required_access_rows,
                output_name=output_name,
                output_type=output_row.get("output_type"),
                expected_stores=expected_stores,
                dimension_aliases=dimension_aliases,
            )
        elif raw_partition is not None:
            raise ValueError(
                f"canonical Source partition locator is unnecessary at output {output_ordinal}"
            )


def _canonical_wire_check(input_root: Path) -> None:
    """Validate the mounted canonical bundle without returning its payload.

    ``schema_lookup`` and the resource projections must reject the same stale
    mounted bundle.  Keeping this command as a thin call over
    ``_canonical_wire_seed`` gives the sandbox environment one validation
    entrypoint instead of a second hand-written receipt checker.
    """

    profile, artifact = _canonical_wire_seed(input_root)
    _emit_json(
        {
            "protocol": "intentir.canonical-wire-check",
            "status": "pass",
            "schema_sha256": profile["schema_sha256"],
            "root_layout": profile["root"]["fields"],
            "artifact_root": list(artifact),
            "authority": "mounted canonical schema, skeleton, and wire bundle",
        }
    )


def _mounted_source_wire_seed(
    input_root: Path,
    profile: Mapping[str, Any],
    construction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the host-built Source-bound mechanical artifact seed.

    MCP is a mounted consumer of the construction authority; it must not
    rebuild ABI, tensor, dimension, or C rows from ``harness_view.json``.
    The host writes this bundle from ``canonical_source_wire_seed`` beside the
    generic schema bundle.  A missing or stale bundle is a preparation error,
    not permission to fall back to a second constructor.
    """

    path = input_root / "schema" / "canonical_source_wire_bundle.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "mounted Source-bound canonical wire bundle is unavailable"
        )
    bundle = _load_json(path)
    if not isinstance(bundle, Mapping):
        raise TypeError("Source-bound canonical wire bundle must be an object")
    mounted_profile = bundle.get("profile")
    seed = bundle.get("artifact_seed")
    if mounted_profile != dict(profile):
        raise ValueError(
            "Source-bound canonical wire bundle uses a different wire profile"
        )
    if not isinstance(seed, dict):
        raise TypeError("Source-bound canonical wire bundle lacks artifact_seed")
    if set(seed) != set(profile["root"]["fields"]):
        raise ValueError("Source-bound canonical seed has the wrong root layout")
    if seed.get("schema") != profile.get("schema"):
        raise ValueError("Source-bound canonical seed has the wrong schema")
    if bundle.get("root_layout") != profile["root"]["fields"]:
        raise ValueError("Source-bound canonical bundle root layout is stale")
    if bundle.get("namespace_order") != profile.get("namespace_order"):
        raise ValueError("Source-bound canonical bundle namespace order is stale")
    if isinstance(construction, Mapping):
        if bundle.get("construction_digest") != construction.get(
            "authority_digest"
        ):
            raise ValueError(
                "Source-bound canonical seed differs from construction authority"
            )
        if bundle.get("source_transfer_ledger") != construction.get(
            "transfer_ledger"
        ):
            raise ValueError(
                "Source-bound canonical ledger differs from construction authority"
            )
        if bundle.get("source_facts_digest") != construction.get(
            "source_facts_digest"
        ):
            raise ValueError(
                "Source-bound SourceFacts digest differs from construction authority"
            )
        source_facts = construction.get("source_facts")
        expected_slice = (
            source_facts.get("source_slice_digest")
            if isinstance(source_facts, Mapping)
            else None
        )
        if bundle.get("source_slice_digest") != expected_slice:
            raise ValueError(
                "Source-bound Source slice digest differs from construction authority"
            )
        _validate_host_launch_contract_receipt(bundle, construction)
        _validate_source_seed_mechanical_rows(seed, construction)
    return seed


def _validate_host_launch_contract_receipt(
    bundle: Mapping[str, Any], construction: Mapping[str, Any]
) -> None:
    """Validate the mounted digest receipt without rebuilding the contract."""

    receipt = bundle.get("host_launch_contract")
    facts = construction.get("source_facts")
    ledger = construction.get("transfer_ledger")
    if not isinstance(receipt, Mapping) or not isinstance(facts, Mapping):
        raise ValueError("mounted host launch contract receipt is unavailable")
    compiler_facts = facts.get("compiler_facts")
    if not isinstance(compiler_facts, Mapping):
        raise ValueError("mounted SourceFacts compiler facts are unavailable")
    contract = compiler_facts.get("host_launch_contract")
    source_facts_digest = construction.get("source_facts_digest")
    if receipt.get("source_facts_digest") != source_facts_digest:
        raise ValueError("mounted host launch receipt has the wrong SourceFacts digest")
    if not isinstance(contract, Mapping) or not contract:
        if receipt.get("status") != "unavailable":
            raise ValueError("mounted host launch receipt falsely claims availability")
        if not isinstance(ledger, Mapping):
            raise ValueError("mounted construction lacks transfer ledger")
        ledger_digest = hashlib.sha256(
            json.dumps(
                dict(ledger),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if receipt.get("transfer_ledger_digest") != ledger_digest:
            raise ValueError("mounted unavailable host receipt transfer lineage disagrees")
        return
    if receipt.get("status") != "pass":
        raise ValueError("mounted host launch receipt is not closed")
    contract_digest = hashlib.sha256(
        json.dumps(
            dict(contract),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if receipt.get("contract_digest") != contract_digest:
        raise ValueError("mounted host launch receipt contract digest disagrees")
    if receipt.get("source_digest") != facts.get("source_digest"):
        raise ValueError("mounted host launch receipt source digest disagrees")
    if not isinstance(ledger, Mapping):
        raise ValueError("mounted construction lacks transfer ledger")
    ledger_digest = hashlib.sha256(
        json.dumps(
            dict(ledger),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if receipt.get("transfer_ledger_digest") != ledger_digest:
        raise ValueError("mounted host launch receipt transfer lineage disagrees")


def _validate_source_seed_mechanical_rows(
    seed: Mapping[str, Any], construction: Mapping[str, Any]
) -> None:
    """Reject a stale Source seed whose mechanical rows drifted after mounting.

    The host constructor is the only producer of these rows.  This helper is
    intentionally a validator, not a second constructor: it compares the
    mounted seed with the already-mounted allocator, SourceFacts and compiler
    ledger, and never fills or reorders a field.  Without this check a bundle
    could carry a valid authority digest while its copied J/C seed exposed a
    different p*/c*/t* projection to MCP.
    """

    allocator = construction.get("allocator")
    facts = construction.get("source_facts")
    if not isinstance(allocator, Mapping) or not isinstance(facts, Mapping):
        raise ValueError("mounted Source construction lacks allocator or facts")
    j = seed.get("j")
    c = seed.get("c")
    if not isinstance(j, Mapping) or not isinstance(c, Mapping):
        raise ValueError("mounted Source seed lacks J or C")
    intent = j.get("intent")
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    signature = j.get("signature")
    if not isinstance(function, Mapping) or not isinstance(signature, Mapping):
        raise ValueError("mounted Source seed lacks function or signature")

    def dense(prefix: str) -> list[str]:
        values = allocator.get(prefix)
        if not isinstance(values, list) or any(
            item != f"{prefix}{index}" for index, item in enumerate(values)
        ):
            raise ValueError(f"mounted allocator {prefix}* is not dense")
        return list(values)

    p_order = dense("p")
    c_order = dense("c")
    t_order = dense("t")
    x_order = dense("x")
    logical_bindings = allocator.get("logical_bindings")
    binding_by_p: dict[str, tuple[str, str]] = {}
    if not isinstance(logical_bindings, list):
        raise ValueError("mounted logical binding table is unavailable")
    for row in logical_bindings:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or not all(isinstance(item, str) for item in row)
            or row[0] in binding_by_p
        ):
            raise ValueError("mounted logical binding table is malformed")
        binding_by_p[row[0]] = (row[1], row[2])

    parameters = signature.get("parameters")
    slots = c.get("slots")
    values = c.get("values")
    if not isinstance(parameters, list) or not isinstance(slots, list) or not isinstance(values, Mapping):
        raise ValueError("mounted Source seed ABI rows are incomplete")
    if [row.get("name") for row in parameters if isinstance(row, Mapping)] != p_order:
        raise ValueError("mounted Source seed p* signature order differs from allocator")
    if [row.get("name") for row in slots if isinstance(row, Mapping)] != c_order:
        raise ValueError("mounted Source seed c* slot order differs from allocator")

    compiler_facts = facts.get("compiler_facts")
    if not isinstance(compiler_facts, Mapping):
        raise ValueError("mounted Source facts lack compiler ledger")
    schedule_values = compiler_facts.get("schedule_values")
    if not isinstance(schedule_values, Mapping) or dict(values) != dict(schedule_values):
        raise ValueError("mounted Source seed C values differ from compiler ledger")

    # A bound non-C constexpr which names a Source shape coordinate owns one
    # signature-level p* -> x* relation.  It is distinct from allocator
    # p* -> t* bindings: it neither introduces a logical tensor nor grants
    # the provider any ABI ownership.  Reconstruct the relation from the same
    # mounted SourceFacts coordinate order used by the canonical seed.
    bound_coordinates = compiler_facts.get("bound_constexpr_shape_coordinates", {})
    raw_coordinates = facts.get("shape_coordinates")
    if not isinstance(bound_coordinates, Mapping) or not isinstance(raw_coordinates, list):
        raise ValueError("mounted Source facts lack bound constexpr shape coordinates")
    if len(raw_coordinates) != len(x_order):
        raise ValueError("mounted Source coordinate allocator differs from facts")
    dimension_by_coordinate: dict[str, str] = {}
    for index, row in enumerate(raw_coordinates):
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or row[0] in dimension_by_coordinate
        ):
            raise ValueError("mounted Source shape coordinates are malformed")
        dimension_by_coordinate[row[0]] = x_order[index]
    bound_dimension_by_p: dict[str, str] = {}
    for name, coordinate in bound_coordinates.items():
        if not isinstance(name, str) or not isinstance(coordinate, str):
            raise ValueError("mounted bound constexpr shape coordinate is malformed")
        dimension = dimension_by_coordinate.get(coordinate)
        if dimension is None:
            raise ValueError("mounted bound constexpr shape coordinate is unknown")
        bound_dimension_by_p[name] = dimension

    for row in parameters:
        if not isinstance(row, Mapping):
            raise ValueError("mounted Source seed signature row is malformed")
        name = row.get("name")
        if not isinstance(name, str):
            raise ValueError("mounted Source seed signature row lacks p* name")
        binding = binding_by_p.get(name)
        expected_logical_ref = (
            binding[0]
            if binding is not None
            else bound_dimension_by_p.get(name)
        )
        if expected_logical_ref is not None and row.get("logical_ref") != expected_logical_ref:
            raise ValueError("mounted Source seed logical_ref differs from allocator")
        if expected_logical_ref is None and row.get("logical_ref") is not None:
            raise ValueError("mounted Source seed contains an unowned logical_ref")

    arguments = facts.get("arguments")
    if not isinstance(arguments, list):
        raise ValueError("mounted Source facts arguments are unavailable")
    argument_by_p = {
        item.get("name"): item
        for item in arguments
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    expected_inputs: list[str] = []
    expected_outputs: list[str] = []
    for p_name in p_order:
        binding = binding_by_p.get(p_name)
        argument = argument_by_p.get(p_name)
        if binding is None:
            continue
        logical, role = binding
        if role in {"shape_scalar", "scalar"}:
            expected_inputs.append(logical)
            continue
        if role != "tensor" or not isinstance(argument, Mapping):
            continue
        mode = argument.get("mode")
        if mode in {"input", "inout"}:
            expected_inputs.append(logical)
        if mode in {"output", "inout"}:
            expected_outputs.append(logical)
    inputs = function.get("inputs")
    outputs = function.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("mounted Source seed function ABI rows are incomplete")
    input_names = [row.get("name") for row in inputs if isinstance(row, Mapping)]
    output_names = [row.get("name") for row in outputs if isinstance(row, Mapping)]
    if input_names != expected_inputs or output_names != expected_outputs:
        raise ValueError("mounted Source seed tensor order differs from allocator/facts")
    if [*input_names, *[name for name in output_names if name not in input_names]] != t_order:
        raise ValueError("mounted Source seed t* declaration order is not canonical")



def _mounted_source_kind(input_root: Path) -> str:
    """Read source representation metadata from canonical construction."""

    construction = _mounted_canonical_construction(input_root)
    source_facts = (
        construction.get("source_facts")
        if isinstance(construction, Mapping)
        else None
    )
    source_kind = source_facts.get("source_kind") if isinstance(source_facts, Mapping) else None
    if isinstance(source_kind, str) and source_kind:
        return source_kind

    harness_path = input_root / "harness_view.json"
    if harness_path.is_file():
        value = _load_json(harness_path)
        if isinstance(value, dict):
            source_kind = value.get("source_kind")
            if isinstance(source_kind, str) and source_kind:
                return source_kind
            if value.get("frontend") == "cuda":
                return "cuda"
    return "python"


def _mounted_canonical_construction(input_root: Path) -> dict[str, Any] | None:
    """Read the one host-derived construction authority, if present.

    This standalone MCP helper intentionally consumes a mounted host record.
    It cannot import the repository package and must not reconstruct Source
    facts, transfer occurrence order, or handle allocation from local buckets.
    """

    path = input_root / "canonical_construction.json"
    if not path.is_file() or path.is_symlink():
        return None
    value = _load_json(path)
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("source_facts"), dict)
        or not isinstance(value.get("transfer_ledger"), dict)
        or not isinstance(value.get("allocator"), dict)
    ):
        return None
    authority_digest = value.get("authority_digest")
    if not isinstance(authority_digest, str) or not authority_digest:
        return None
    payload = {
        key: item for key, item in value.items() if key != "authority_digest"
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if authority_digest != expected_digest:
        return None
    source_facts = value["source_facts"]
    transfer_ledger = value["transfer_ledger"]
    source_facts_digest = hashlib.sha256(
        json.dumps(
            source_facts,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if value.get("source_facts_digest") != source_facts_digest:
        return None
    transfer_ledger_digest = hashlib.sha256(
        json.dumps(
            transfer_ledger,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if value.get("transfer_ledger_digest") != transfer_ledger_digest:
        return None
    if transfer_ledger.get("source_facts_digest") != source_facts_digest:
        return None
    return value


def _feedback_value(input_root: Path) -> dict[str, Any] | None:
    value = _load_json(input_root / "feedback.json")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("feedback.json must be null or an object")
    retired = sorted(
        key
        for key in value
        if str(key).startswith("previous_") or key == "repair_base"
    )
    if retired:
        raise ValueError(
            "feedback.json contains retired candidate transport field(s): "
            + ", ".join(retired)
        )
    return value


def _feedback_frontier(input_root: Path) -> None:
    """Print the host-authored bounded projection without reading the candidate."""

    path = input_root / "feedback_frontier.json"
    if not path.is_file():
        _emit_json(None)
        return
    value = _load_json(path)
    if value is not None and not isinstance(value, dict):
        raise TypeError("feedback_frontier.json must be null or an object")
    _emit_json(value)


def _observer_categories(feedback: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(feedback, dict):
        return []
    diagnostics = feedback.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    observer = diagnostics.get("observer_diff")
    if not isinstance(observer, dict):
        unresolved = diagnostics.get("unresolved_gate1")
        unresolved_diagnostics = (
            unresolved.get("diagnostics")
            if isinstance(unresolved, dict)
            else None
        )
        observer = (
            unresolved_diagnostics.get("observer_diff")
            if isinstance(unresolved_diagnostics, dict)
            else None
        )
    if not isinstance(observer, dict):
        return []
    categories = observer.get("categories")
    if not isinstance(categories, list):
        return []
    answer: list[dict[str, Any]] = []
    for category in categories:
        if not isinstance(category, dict):
            continue
        answer.append(
            {
                "category": category.get("category"),
                "difference_count": category.get("difference_count"),
            }
        )
    return answer


def _emit_json(value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    if len(text) > MAX_OUTPUT_CHARS:
        raise ValueError(
            f"query result is {len(text)} characters; use a narrower command"
        )
    print(text)


def _emit_compact_json(value: Any) -> None:
    """Emit one machine-readable compact ledger under the same hard byte bound."""

    text = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(text) > MAX_OUTPUT_CHARS:
        raise ValueError(
            f"query result is {len(text)} characters; use a narrower command"
        )
    print(text)


def _feedback(input_root: Path, path: str | None = None) -> None:
    value = _feedback_value(input_root)
    if path is not None:
        _emit_json(_resolve_json_path(value, path))
        return
    try:
        _emit_json(value)
    except ValueError:
        if not isinstance(value, dict):
            raise
        diagnostics = value.get("diagnostics")
        reduced = dict(value)
        if isinstance(diagnostics, dict):
            reduced_diagnostics = dict(diagnostics)
            reduced_diagnostics.pop("observer_diff", None)
            reduced["diagnostics"] = reduced_diagnostics
        reduced["observer_categories"] = _observer_categories(value)
        reduced["narrow_query"] = (
            f"{QUERY_COMMAND} feedback-category CATEGORY"
        )
        _emit_json(reduced)


def _feedback_category_value(input_root: Path, name: str) -> dict[str, Any]:
    """Return one exact closed Gate1 category as a bounded read-only value."""

    value = _feedback_value(input_root)
    diagnostics = value.get("diagnostics") if isinstance(value, dict) else None
    observer = diagnostics.get("observer_diff") if isinstance(diagnostics, dict) else None
    category_root = "diagnostics.observer_diff.categories"
    if not isinstance(observer, dict) and isinstance(diagnostics, dict):
        unresolved = diagnostics.get("unresolved_gate1")
        unresolved_diagnostics = (
            unresolved.get("diagnostics")
            if isinstance(unresolved, dict)
            else None
        )
        observer = (
            unresolved_diagnostics.get("observer_diff")
            if isinstance(unresolved_diagnostics, dict)
            else None
        )
        category_root = (
            "diagnostics.unresolved_gate1.diagnostics.observer_diff.categories"
        )
    categories = observer.get("categories") if isinstance(observer, dict) else None
    if not isinstance(categories, list):
        raise KeyError("feedback contains no observer-diff categories")
    for category_index, category in enumerate(categories):
        if isinstance(category, dict) and category.get("category") == name:
            result = dict(category)
            rendered = json.dumps(category, sort_keys=True)
            category_path = f"{category_root}.{category_index}"
            if name == "dataflow" and isinstance(
                result.get("effect_order_witnesses"), list
            ):
                result["effect_order_hint"] = (
                    "Compare the ordered Source and Replay event sequences, "
                    "then inspect the LLM-authored J scopes, loop/control, and effects. "
                    "Compact pi has no effect order. The "
                    "witness is diagnostic only and does not prescribe a repair."
                )
                result["effect_order_query"] = (
                    f"{QUERY_COMMAND} feedback --path {category_path}."
                    "effect_order_witnesses"
                )
            if (
                name == "dataflow"
                and "region[root]" in rendered
                and "region[scf.for:" in rendered
                and "region.member" in rendered
            ):
                result["diagnostic_hint"] = (
                    "The observer examples cross root/scf.for region membership. "
                    "Inspect exact Source definition scope and the authored J loop/control; "
                    "fixed code has not inferred which producer, if any, must move."
                )
                result["knowledge_query"] = (
                    f"{QUERY_COMMAND} knowledge 'lexical lifetime' --context 5"
                )
            if len(
                json.dumps(result, sort_keys=True, ensure_ascii=False)
            ) > MAX_OUTPUT_CHARS:
                node_witnesses = result.pop("node_witnesses", None)
                if isinstance(node_witnesses, list):
                    result["node_witness_count"] = len(node_witnesses)
                    result["node_witness_query"] = (
                        f"{QUERY_COMMAND} feedback --path {category_path}."
                        "node_witnesses"
                    )
                    result["bounded_view"] = (
                        "bulk node witnesses omitted so causal role/effect "
                        "witnesses remain visible"
                    )
            return result
    available = [
        item.get("category") for item in categories if isinstance(item, dict)
    ]
    raise KeyError(f"unknown feedback category {name!r}; choose {available}")


def _feedback_category(input_root: Path, name: str) -> None:
    _emit_json(_feedback_category_value(input_root, name))


def _source(input_root: Path, start: int, line_count: int) -> None:
    if start < 1 or line_count < 1:
        raise ValueError("source start and lines must be positive")
    lines = (input_root / "source.py").read_text(encoding="utf-8").splitlines()
    selected: list[str] = []
    used = 0
    for ordinal, line in enumerate(lines[start - 1 :], start=start):
        rendered = f"{ordinal:5d} | {line}"
        if selected and used + len(rendered) + 1 > MAX_OUTPUT_CHARS - 300:
            break
        selected.append(rendered)
        used += len(rendered) + 1
        if len(selected) >= line_count:
            break
    print("\n".join(selected))
    next_line = start + len(selected)
    if next_line <= len(lines):
        print(
            f"\n[source continues: run `{QUERY_COMMAND} source --start "
            f"{next_line} --lines {line_count}`]"
        )


def _source_ttir_observation(input_root: Path) -> dict[str, Any]:
    path = input_root / "source_ttir.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("source_ttir.json is unavailable for this Lift input")
    value = _load_json(path)
    if not isinstance(value, dict):
        raise TypeError("source_ttir.json must contain one object")
    return value


def _source_ttir_text(input_root: Path) -> str:
    value = _source_ttir_observation(input_root)
    if value.get("status") != "pass":
        diagnostic = value.get("diagnostic", "Source TTIR observation did not pass")
        raise ValueError(str(diagnostic))
    text = value.get("ttir")
    if not isinstance(text, str) or not text:
        raise ValueError("passing source_ttir.json contains no TTIR text")
    return text


def _source_ttir(input_root: Path, start: int, line_count: int) -> None:
    if start < 1 or line_count < 1:
        raise ValueError("Source TTIR start and lines must be positive")
    lines = _source_ttir_text(input_root).splitlines()
    selected: list[str] = []
    used = 0
    for ordinal, line in enumerate(lines[start - 1 :], start=start):
        rendered = f"{ordinal:5d} | {line}"
        if selected and used + len(rendered) + 1 > MAX_OUTPUT_CHARS - 300:
            break
        selected.append(rendered)
        used += len(rendered) + 1
        if len(selected) >= line_count:
            break
    print("\n".join(selected))
    next_line = start + len(selected)
    if next_line <= len(lines):
        print(
            f"\n[TTIR continues: run `{QUERY_COMMAND} source-ttir --start "
            f"{next_line} --lines {line_count}`]"
        )


def _source_ttir_find(
    input_root: Path, query: str, context: int, *, exact_token: bool = False
) -> None:
    if not query or len(query) > 256 or any(character.isspace() for character in query):
        raise ValueError("Source TTIR query must be one bounded non-whitespace token")
    if context < 0 or context > 20:
        raise ValueError("Source TTIR context must be in [0, 20]")
    if exact_token and not query.startswith("%"):
        raise ValueError("Source TTIR def-use query must name one %SSA value")
    lines = _source_ttir_text(input_root).splitlines()
    needle = query if exact_token else query.casefold()
    matches = [
        index
        for index, line in enumerate(lines)
        if (needle in line if exact_token else needle in line.casefold())
    ]
    if not matches:
        raise KeyError(f"no Source TTIR lines match {query!r}")
    ranges: list[tuple[int, int]] = []
    for index in matches:
        begin = max(0, index - context)
        end = min(len(lines), index + context + 1)
        if ranges and begin <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((begin, end))
    rendered_blocks: list[str] = []
    used = 0
    for begin, end in ranges:
        block = "\n".join(
            f"{ordinal + 1:5d} | {lines[ordinal]}"
            for ordinal in range(begin, end)
        )
        if rendered_blocks and used + len(block) + 2 > MAX_OUTPUT_CHARS - 200:
            break
        rendered_blocks.append(block)
        used += len(block) + 2
    print("\n\n".join(rendered_blocks))
    if len(rendered_blocks) < len(ranges):
        print("\n[more TTIR matches omitted; use a narrower token]")


def _source_ttir_outline(input_root: Path, start: int, count: int) -> None:
    if start < 1 or count < 1 or count > 200:
        raise ValueError(
            "Source TTIR outline start/count must be positive and count <= 200"
        )
    interesting = (
        "module ", "tt.func", "func.func", "^bb", "scf.for", "scf.if",
        "scf.while", "cf.br", "cf.cond_br", "tt.load", "tt.store",
        "tt.atomic", "tt.reduce", "tt.scan", "tt.dot", "scf.yield",
        "math.fma", "arith.extf", "arith.truncf", "tt.return", "func.return",
    )
    rows = [
        {"line": ordinal, "text": line.strip()}
        for ordinal, line in enumerate(
            _source_ttir_text(input_root).splitlines(), start=1
        )
        if any(token in line for token in interesting)
    ]
    selected = rows[start - 1 : start - 1 + count]
    _emit_json(
        {
            "protocol": "intentir.lift-source-ttir-outline.v1",
            "start": start,
            "returned": len(selected),
            "total": len(rows),
            "rows": selected,
            "next_start": (
                start + len(selected)
                if start - 1 + len(selected) < len(rows)
                else None
            ),
            "authority": (
                "syntax navigation over exact Source TTIR; no semantic "
                "classification or J/C/pi inference"
            ),
        }
    )


def _source_observer_value(input_root: Path) -> dict[str, Any]:
    value = _source_ttir_observation(input_root)
    ttir = value.get("ttir")
    summary = {
        key: item for key, item in value.items() if key not in {"ttir", "fingerprint"}
    }
    summary["ttir_lines"] = len(ttir.splitlines()) if isinstance(ttir, str) else 0
    fingerprint = value.get("fingerprint")
    summary["fingerprint_sections"] = (
        list(fingerprint) if isinstance(fingerprint, dict) else []
    )
    summary["queries"] = [
        f"{QUERY_COMMAND} source-ttir-outline --start 1 --count 80",
        f"{QUERY_COMMAND} source-ttir-find TOKEN --context 3",
        f"{QUERY_COMMAND} source-ttir-def-use %SSA --context 2",
        f"{QUERY_COMMAND} source-observer --path fingerprint.loops",
        f"{QUERY_COMMAND} source-observer --path fingerprint.structured_ifs",
        f"{QUERY_COMMAND} source-observer --path fingerprint.memory_effects",
        f"{QUERY_COMMAND} source-observer --path fingerprint.dataflow_edges",
    ]
    return summary


def _source_observer(input_root: Path, path: str | None) -> None:
    value = _source_ttir_observation(input_root)
    if path is not None:
        _emit_json(_resolve_json_path(value, path))
        return
    _emit_json(_source_observer_value(input_root))


def _source_semantic_spine(input_root: Path) -> dict[str, Any]:
    """Return compact, deterministic Source operation/control/return facts.

    This is deliberately not Source text, TTIR, or an alternate J artifact.
    It preserves only the small expression topology a fresh Lift owner needs
    to choose the actual J operation graph; tensor/access ownership remains
    in the transfer ledger.  Source names are replaced by deterministic
    argument/local aliases so the brief does not turn into a second source
    payload or invite copying frontend spellings into J.
    """

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "kind": "source_semantic_spine",
            "status": "unavailable",
            "reason": reason,
            "operations": [],
            "required_operation_families": {},
            "required_operation_attributes": [],
            "control": [],
            "returns": [],
            "missing_operation_contracts": [],
        }

    source_kind = _mounted_source_kind(input_root)
    if source_kind != "python":
        return unavailable("native_source_requires_explicit_operation_leaf")
    source = (input_root / "source.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="source.py")
    function_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not function_nodes:
        return unavailable("source_entry_function_unavailable")
    projection_path = input_root / "source_projection.json"
    entry_name: str | None = None
    if projection_path.is_file() and not projection_path.is_symlink():
        projection = _load_json(projection_path)
        value = projection.get("function") if isinstance(projection, Mapping) else None
        if isinstance(value, str):
            entry_name = value
    if not isinstance(entry_name, str) or not entry_name:
        return unavailable("source_entry_projection_unavailable")
    functions_by_name = {node.name: node for node in function_nodes}
    if len(functions_by_name) != len(function_nodes) or entry_name not in functions_by_name:
        return unavailable("source_entry_projection_ambiguous")
    helper_edges: dict[str, tuple[str, ...]] = {}
    for function in function_nodes:
        callees = tuple(
            child.func.id
            for child in ast.walk(function)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in functions_by_name
        )
        helper_edges[function.name] = callees
    reachable: set[str] = set()
    active: set[str] = set()
    recursive_helper_graph = False

    def visit_helper(name: str) -> None:
        nonlocal recursive_helper_graph
        if name in active:
            recursive_helper_graph = True
            return
        if name in reachable:
            return
        active.add(name)
        reachable.add(name)
        for callee in helper_edges.get(name, ()):
            visit_helper(callee)
        active.remove(name)

    visit_helper(entry_name)
    selected = [functions_by_name[entry_name]]
    selected.extend(
        node
        for node in function_nodes
        if node.name != entry_name and node.name in reachable
    )
    # A Source alias belongs to a lexical function scope.  Keeping the scope
    # on facts which bridge a helper invocation avoids treating helper ``arg0``
    # as the entry's first ABI argument (or colliding equally numbered locals
    # in two functions).  These are navigation facts only: the provider still
    # chooses every J result label and operation row.
    function_scopes = {
        function.name: ("entry" if function.name == entry_name else f"helper{ordinal}")
        for ordinal, function in enumerate(selected)
    }

    def scoped_ref(scope: str, value: str) -> str:
        return f"{scope}.{value}"

    # The source wire owns the sole anonymous Source argument -> logical J
    # operand bridge.  Reuse it verbatim enough to add this brief's lexical
    # navigation alias; do not reconstruct bindings from ABI spellings here.
    entry_argument_projection: list[dict[str, Any]] = []
    source_wire_path = input_root / "schema" / "canonical_source_wire_bundle.json"
    if source_wire_path.is_file() and not source_wire_path.is_symlink():
        source_wire = _load_json(source_wire_path)
        operand_inputs = (
            source_wire.get("semantic_operand_inputs")
            if isinstance(source_wire, Mapping)
            else None
        )
        if not isinstance(operand_inputs, list):
            raise ValueError("canonical Source wire lacks semantic operand inputs")
        for row in operand_inputs:
            ordinal = row.get("source_argument_ordinal") if isinstance(row, Mapping) else None
            provider_ref = row.get("provider_ref") if isinstance(row, Mapping) else None
            role = row.get("role") if isinstance(row, Mapping) else None
            value_type = row.get("value_type") if isinstance(row, Mapping) else None
            if (
                not _is_canonical_ordinal(ordinal)
                or not isinstance(provider_ref, str)
                or re.fullmatch(r"t[0-9]+", provider_ref) is None
                or role not in {"tensor", "scalar", "shape_scalar"}
                or not isinstance(value_type, Mapping)
            ):
                raise ValueError("canonical Source semantic operand input is malformed")
            entry_argument_projection.append(
                {
                    "source_alias": scoped_ref("entry", f"arg{ordinal}"),
                    "provider_ref": provider_ref,
                    "role": f"logical_{role}",
                    "value_type": copy.deepcopy(dict(value_type)),
                }
            )
    operations: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    returns: list[dict[str, Any]] = []
    helper_call_bindings: list[dict[str, Any]] = []
    operation_hints: set[str] = set()
    unresolved: set[str] = set()
    unclassified_calls: set[str] = set()
    if recursive_helper_graph:
        unclassified_calls.add("recursive_helper_call_graph")

    binary_names = {
        ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div",
        ast.FloorDiv: "floordiv", ast.Mod: "remainder", ast.Pow: "pow",
        ast.BitAnd: "and", ast.BitOr: "or", ast.BitXor: "xor",
        ast.LShift: "left_shift", ast.RShift: "right_shift",
    }
    compare_names = {
        ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge",
        ast.Eq: "eq", ast.NotEq: "ne",
    }
    boolean_names = {ast.And: "logical_and", ast.Or: "logical_or"}
    unary_names = {
        ast.USub: "neg", ast.UAdd: "identity", ast.Not: "logical_not",
        ast.Invert: "logical_not",
    }
    unary_calls = {"sigmoid", "exp", "log", "sqrt", "rsqrt", "abs", "sin", "cos", "tanh"}
    binary_calls = {"maximum", "minimum", "pow"}
    reduction_calls = {"sum", "mean", "max", "min", "prod", "any", "all"}
    cast_calls = {"to", "cast", "astype", "float", "int"}
    select_calls = {"where"}
    canonical_call_families = {
        "dot": "contraction",
        "full": "const",
        "zeros": "const",
        "reshape": "reshape",
        "view": "reshape",
        "trans": "transpose",
        "transpose": "transpose",
        "join": "concat",
        "cat": "concat",
        "concat": "concat",
        "flip": "flip",
        "sort": "sort",
    }
    compile_only_calls = {
        "multiple_of", "max_contiguous", "static_assert", "static_range",
        "assume", "debug_barrier",
    }
    boundary_calls = {"load", "store", "atomic_add", "atomic_and", "atomic_cas", "atomic_max", "atomic_min", "atomic_or", "atomic_xchg", "atomic_xor"}
    index_calls = {"program_id", "num_programs", "arange"}

    def required_operation_family(kind: object, operator: object) -> str | None:
        """Map only uniquely Source-proven expression leaves to canonical J.

        Boundary transfers are owned by the occurrence ledger and program_id
        is owned by the program-domain projection. Unknown/helper calls remain
        authoring obligations, but are not guessed into a canonical operation.
        """

        if kind in {"binary", "compare", "boolean"}:
            return "map_binary"
        if kind == "fused_multiply_add":
            return "map_binary"
        if kind == "binary_call":
            return "map_binary"
        if kind in {"unary", "unary_call"}:
            return "map_unary"
        if kind in {"select_call", "select_expression"}:
            return "select"
        if kind == "cast":
            return "cast"
        if kind == "reduction":
            return "reduce"
        if kind == "index_source" and operator == "arange":
            return "iota"
        if kind == "canonical_call" and isinstance(operator, str):
            return canonical_call_families.get(operator)
        return None

    for function_ordinal, function in enumerate(selected):
        scope = function_scopes[function.name]
        aliases: dict[str, str] = {
            argument.arg: f"arg{index}"
            for index, argument in enumerate(
                [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
            )
        }
        local_count = 0
        # Give each existing expression node one deterministic Source-only
        # value occurrence.  The previous compact view used labels such as
        # ``binary:mul`` and ``call:sum`` for nested expressions; repeated
        # expressions consequently had ambiguous operands and no matching
        # producer result.  These labels are occurrence identities, not J
        # ids, and are used only to preserve the AST's existing dataflow.
        source_expression_refs: dict[int, str] = {}
        source_expression_index = 0
        for expression in ast.walk(function):
            if isinstance(
                expression,
                (
                    ast.BinOp,
                    ast.BoolOp,
                    ast.Compare,
                    ast.UnaryOp,
                    ast.IfExp,
                    ast.Call,
                ),
            ):
                source_expression_refs[id(expression)] = (
                    f"sx{function_ordinal}_{source_expression_index}"
                )
                source_expression_index += 1

        def alias_for_name(name: str) -> str:
            nonlocal local_count
            if name not in aliases:
                aliases[name] = f"local{local_count}"
                local_count += 1
            return aliases[name]

        def value_ref(node: ast.AST | None) -> str:
            if node is None:
                return "none"
            expression_ref = source_expression_refs.get(id(node))
            if expression_ref is not None:
                return expression_ref
            if isinstance(node, ast.Name):
                return alias_for_name(node.id)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (bool, int, float, str)) or node.value is None:
                    return "literal:" + repr(node.value)[:40]
                return "literal"
            if isinstance(node, ast.Attribute):
                return "attribute:" + node.attr
            if isinstance(node, ast.Tuple):
                return "tuple[" + str(len(node.elts)) + "]"
            if isinstance(node, ast.List):
                return "list[" + str(len(node.elts)) + "]"
            if isinstance(node, ast.Call):
                return "call:" + call_leaf(node)
            if isinstance(node, ast.BinOp):
                return "binary:" + binary_names.get(type(node.op), "unknown")
            if isinstance(node, ast.Compare):
                return "compare:" + compare_names.get(type(node.ops[0]), "unknown") if node.ops else "compare"
            return type(node).__name__.lower()

        def call_leaf(node: ast.Call) -> str:
            if isinstance(node.func, ast.Attribute):
                return node.func.attr
            if isinstance(node.func, ast.Name):
                return node.func.id
            return "dynamic"

        # Separate pointer-address construction from logical J calculation.
        # Trace simple local definitions backwards from every boundary pointer
        # operand, then retain a name as address-only only when it has no use
        # outside that closed definition/pointer subgraph. A value also used by
        # a mask, store value, return, or ordinary calculation remains logical.
        assignment_candidates: dict[str, list[ast.AST]] = {}
        augmented_assignment_candidates: dict[str, list[ast.AST]] = {}
        ambiguous_assignment_names: set[str] = set()
        for candidate in ast.walk(function):
            if isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                targets = (
                    candidate.targets
                    if isinstance(candidate, ast.Assign)
                    else [candidate.target]
                )
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    assignment_candidates.setdefault(targets[0].id, []).append(
                        candidate.value
                    )
                else:
                    ambiguous_assignment_names.update(
                        item.id
                        for target in targets
                        for item in ast.walk(target)
                        if isinstance(item, ast.Name)
                    )
            elif isinstance(candidate, ast.AugAssign):
                if isinstance(candidate.target, ast.Name):
                    augmented_assignment_candidates.setdefault(
                        candidate.target.id, []
                    ).append(candidate.value)
                else:
                    ambiguous_assignment_names.update(
                        item.id
                        for item in ast.walk(candidate.target)
                        if isinstance(item, ast.Name)
                    )
        for name, values in augmented_assignment_candidates.items():
            if len(values) == 1 and name not in assignment_candidates:
                assignment_candidates[name] = values
            else:
                ambiguous_assignment_names.add(name)
        assignment_values = {
            name: values[0]
            for name, values in assignment_candidates.items()
            if len(values) == 1 and name not in ambiguous_assignment_names
        }
        ambiguous_assignment_names.update(
            name for name, values in assignment_candidates.items() if len(values) != 1
        )

        pointer_expression_node_ids: set[int] = set()
        pointer_seed_names: set[str] = set()
        for candidate in ast.walk(function):
            if not isinstance(candidate, ast.Call) or call_leaf(candidate) not in boundary_calls:
                continue
            pointer_values: list[ast.AST] = list(candidate.args[:1])
            pointer_values.extend(
                keyword.value
                for keyword in candidate.keywords
                if keyword.arg in {"pointer", "ptr", "base"}
            )
            for pointer_value in pointer_values:
                for descendant in ast.walk(pointer_value):
                    pointer_expression_node_ids.add(id(descendant))
                    if isinstance(descendant, ast.Name) and isinstance(
                        descendant.ctx, ast.Load
                    ):
                        pointer_seed_names.add(descendant.id)
                    if (
                        isinstance(descendant, ast.Call)
                        and isinstance(descendant.func, ast.Name)
                        and descendant.func.id in function_scopes
                    ):
                        # A helper returning a pointer crosses lexical scopes.
                        # The compact per-function def/use proof below cannot
                        # establish that its body is address-only, so do not
                        # silently classify helper arithmetic as logical J.
                        unclassified_calls.add(descendant.func.id)

        address_dependency_names = set(pointer_seed_names)
        pending = list(pointer_seed_names)
        while pending:
            name = pending.pop()
            definition = assignment_values.get(name)
            if definition is None:
                continue
            for descendant in ast.walk(definition):
                if not isinstance(descendant, ast.Name) or not isinstance(
                    descendant.ctx, ast.Load
                ):
                    continue
                if descendant.id not in address_dependency_names:
                    address_dependency_names.add(descendant.id)
                    pending.append(descendant.id)

        address_definition_node_ids: set[int] = set()
        for name in address_dependency_names:
            definition = assignment_values.get(name)
            if definition is not None:
                address_definition_node_ids.update(id(item) for item in ast.walk(definition))
        externally_used_names = {
            candidate.id
            for candidate in ast.walk(function)
            if isinstance(candidate, ast.Name)
            and isinstance(candidate.ctx, ast.Load)
            and id(candidate) not in pointer_expression_node_ids
            and id(candidate) not in address_definition_node_ids
        }
        logical_dependency_names = address_dependency_names & (
            externally_used_names | ambiguous_assignment_names
        )
        pending = list(logical_dependency_names)
        while pending:
            name = pending.pop()
            definition = assignment_values.get(name)
            if definition is None:
                continue
            for descendant in ast.walk(definition):
                if (
                    isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                    and descendant.id in address_dependency_names
                    and descendant.id not in logical_dependency_names
                ):
                    logical_dependency_names.add(descendant.id)
                    pending.append(descendant.id)
        address_only_names = address_dependency_names - logical_dependency_names

        def call_semantic_operands(node: ast.Call, operator: str) -> list[str]:
            """Project the actual value receiver for Source cast calls.

            Triton spells casts as ``value.to(dtype)``.  The dtype argument is
            an attribute, not the J value operand; preserving only ``args``
            therefore erased the Source dependency on logical inputs such as
            a scalar multiplier.  Keep this source-only fact in the compact
            spine so an author can bind the existing provider ``t*`` value
            without exposing Source text or fabricating an operation.
            """

            if operator in cast_calls and isinstance(node.func, ast.Attribute):
                return [value_ref(node.func.value)]
            return [value_ref(item) for item in node.args]

        def record_operation(node: ast.AST, result: str | None = None) -> None:
            kind: str | None = None
            operator: str | None = None
            operands: list[str] = []
            keyword_names: list[str] = []
            if isinstance(node, ast.BinOp):
                kind, operator = "binary", binary_names.get(type(node.op), "unknown")
                operands = [value_ref(node.left), value_ref(node.right)]
                operation_hints.add("map_binary")
            elif isinstance(node, ast.BoolOp):
                kind, operator = "boolean", boolean_names.get(
                    type(node.op), "unknown"
                )
                operands = [value_ref(item) for item in node.values]
                operation_hints.add("map_binary")
            elif isinstance(node, ast.Compare):
                kind = "compare"
                operator = compare_names.get(type(node.ops[0]), "unknown") if node.ops else "unknown"
                operands = [value_ref(node.left), *[value_ref(item) for item in node.comparators]]
                operation_hints.add("map_binary")
            elif isinstance(node, ast.UnaryOp):
                kind, operator = "unary", unary_names.get(type(node.op), "unknown")
                operands = [value_ref(node.operand)]
                operation_hints.add("map_unary")
            elif isinstance(node, ast.IfExp):
                kind, operator = "select_expression", "where"
                operands = [
                    value_ref(node.test),
                    value_ref(node.body),
                    value_ref(node.orelse),
                ]
                unresolved.add("select")
            elif isinstance(node, ast.Call):
                operator = call_leaf(node)
                operands = call_semantic_operands(node, operator)
                keyword_names = sorted(
                    item.arg for item in node.keywords if isinstance(item.arg, str)
                )
                if operator in boundary_calls:
                    kind = "boundary_transfer"
                elif operator in index_calls:
                    kind = "index_source"
                    if operator == "arange":
                        unresolved.add("iota")
                elif operator in unary_calls:
                    kind = "unary_call"
                    operation_hints.add("map_unary")
                elif operator in binary_calls:
                    kind = "binary_call"
                    operation_hints.add("map_binary")
                elif operator in reduction_calls:
                    # A reduction has distinct operand, axis, and
                    # associativity semantics.  It must not be collapsed to
                    # a generic unary hint merely because it is nested under
                    # a binary AST expression.
                    kind = "reduction"
                    unresolved.add("reduce")
                elif operator in cast_calls:
                    kind = "cast"
                    operation_hints.add("cast")
                elif operator in select_calls:
                    kind = "select_call"
                    unresolved.add("select")
                elif operator == "fma":
                    kind = "fused_multiply_add"
                    operation_hints.add("map_binary")
                elif operator in canonical_call_families:
                    kind = "canonical_call"
                    unresolved.add(canonical_call_families[operator])
                elif operator in compile_only_calls:
                    kind = "compiler_hint"
                elif operator in function_scopes:
                    kind = "helper_call"
                else:
                    kind = "unclassified_call"
                    unclassified_calls.add(operator)
            else:
                return
            if operator == "unknown":
                unclassified_calls.add("unknown_operator")
            item: dict[str, Any] = {
                "kind": kind,
                "operator": operator,
                "operands": operands,
            }
            if isinstance(node, ast.Compare):
                item["comparison_operators"] = [
                    compare_names.get(type(comparison), "unknown")
                    for comparison in node.ops
                ]
                if "unknown" in item["comparison_operators"]:
                    unclassified_calls.add("unknown_operator")
            # Array order is the deterministic Source traversal order and the
            # enclosing spine already declares ``entry_scope``. Repeating an
            # ordinal and ``scope=entry`` on every operation made large real
            # kernels exceed the bounded first-turn brief before authoring
            # could start. Preserve only non-default scope, authored result,
            # and non-empty keyword facts; no operation or dependency is
            # dropped by this wire compaction.
            if scope != "entry":
                item["scope"] = scope
            if result is not None:
                item["result"] = result
            if keyword_names:
                item["keyword_names"] = keyword_names
            # Keep nested Source calls visible without expanding into Source
            # text or inventing a second J SSA.  This retains semantic leaves
            # such as sigmoid under a helper return and load/to under a fused
            # assignment, which the old outer-expression-only summary hid.
            nested: list[dict[str, Any]] = []
            for child in ast.walk(node):
                if child is node or not isinstance(
                    child,
                    (
                        ast.BinOp,
                        ast.BoolOp,
                        ast.Compare,
                        ast.UnaryOp,
                        ast.IfExp,
                        ast.Call,
                    ),
                ):
                    continue
                if id(child) in pointer_expression_node_ids:
                    continue
                nested_keywords: list[str] = []
                if isinstance(child, ast.BinOp):
                    child_kind = "binary"
                    leaf = binary_names.get(type(child.op), "unknown")
                    child_operands = [value_ref(child.left), value_ref(child.right)]
                elif isinstance(child, ast.BoolOp):
                    child_kind = "boolean"
                    leaf = boolean_names.get(type(child.op), "unknown")
                    child_operands = [value_ref(item) for item in child.values]
                elif isinstance(child, ast.Compare):
                    child_kind = "compare"
                    leaf = (
                        compare_names.get(type(child.ops[0]), "unknown")
                        if child.ops else "unknown"
                    )
                    child_operands = [
                        value_ref(child.left),
                        *[value_ref(item) for item in child.comparators],
                    ]
                elif isinstance(child, ast.UnaryOp):
                    child_kind = "unary"
                    leaf = unary_names.get(type(child.op), "unknown")
                    child_operands = [value_ref(child.operand)]
                elif isinstance(child, ast.IfExp):
                    child_kind = "select_expression"
                    leaf = "where"
                    child_operands = [
                        value_ref(child.test),
                        value_ref(child.body),
                        value_ref(child.orelse),
                    ]
                else:
                    leaf = call_leaf(child)
                    child_kind = (
                        "boundary_transfer" if leaf in boundary_calls
                        else "index_source" if leaf in index_calls
                        else "unary_call" if leaf in unary_calls
                        else "binary_call" if leaf in binary_calls
                        else "reduction" if leaf in reduction_calls
                        else "cast" if leaf in cast_calls
                        else "select_call" if leaf in select_calls
                        else "fused_multiply_add" if leaf == "fma"
                        else "canonical_call" if leaf in canonical_call_families
                        else "compiler_hint" if leaf in compile_only_calls
                        else "helper_call" if leaf in function_scopes
                        else "unclassified_call"
                    )
                    child_operands = call_semantic_operands(child, leaf)
                    nested_keywords = sorted(
                        keyword.arg
                        for keyword in child.keywords
                        if isinstance(keyword.arg, str)
                    )
                if child_kind == "unary_call":
                    operation_hints.add("map_unary")
                elif child_kind in {"binary", "boolean", "compare"}:
                    operation_hints.add("map_binary")
                elif child_kind == "binary_call":
                    operation_hints.add("map_binary")
                elif child_kind == "cast":
                    operation_hints.add("cast")
                elif child_kind in {"select_call", "select_expression"}:
                    unresolved.add("select")
                elif child_kind == "fused_multiply_add":
                    operation_hints.add("map_binary")
                elif child_kind == "canonical_call":
                    unresolved.add(canonical_call_families[leaf])
                elif child_kind == "unclassified_call":
                    unclassified_calls.add(leaf)
                elif child_kind == "index_source":
                    if leaf == "arange":
                        unresolved.add("iota")
                elif child_kind == "reduction":
                    unresolved.add("reduce")
                nested_item: dict[str, Any] = {
                    "kind": child_kind,
                    "operator": leaf,
                    "operands": child_operands,
                }
                if leaf == "unknown":
                    unclassified_calls.add("unknown_operator")
                if isinstance(child, ast.Compare):
                    nested_item["comparison_operators"] = [
                        compare_names.get(type(comparison), "unknown")
                        for comparison in child.ops
                    ]
                    if "unknown" in nested_item["comparison_operators"]:
                        unclassified_calls.add("unknown_operator")
                if nested_keywords:
                    nested_item["keyword_names"] = nested_keywords
                child_ref = source_expression_refs.get(id(child))
                if child_ref is not None:
                    nested_item["result"] = child_ref
                nested.append(nested_item)
            if nested:
                item["nested_operations"] = nested
            # Direct calls to a mounted helper have an otherwise invisible
            # value bridge: helper ``argN`` is not an ABI/J operand, but its
            # caller expression is.  Retain that exact AST binding so a
            # provider can inline the helper's Source semantics into its one
            # entry graph without guessing that ``arg0`` means ``t0``.
            for child in ast.walk(node):
                if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Name):
                    continue
                callee_scope = function_scopes.get(child.func.id)
                if callee_scope is None or callee_scope == "entry":
                    continue
                bindings = [
                    {
                        "callee_alias": scoped_ref(callee_scope, f"arg{index}"),
                        "caller_alias": scoped_ref(scope, value_ref(argument)),
                    }
                    for index, argument in enumerate(child.args)
                ]
                helper_call_bindings.append(
                    {
                        "caller_scope": scope,
                        "callee_scope": callee_scope,
                        "callee": child.func.id,
                        "argument_bindings": bindings,
                        "result_context": (
                            scoped_ref(scope, result) if isinstance(result, str) else None
                        ),
                    }
                )
            operations.append(item)

        def target_alias(target: ast.AST) -> str | None:
            if isinstance(target, ast.Name):
                return alias_for_name(target.id)
            if isinstance(target, (ast.Tuple, ast.List)):
                return "tuple_target[" + str(len(target.elts)) + "]"
            return None

        def walk_statements(statements: Sequence[ast.stmt]) -> None:
            for statement in statements:
                if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    # Pointer/scalar in-place updates feed the Source transfer
                    # ledger; they are address construction, not a logical J
                    # map operation.  Publishing them as map_binary rewrote
                    # entry arguments and made the compact spine actively
                    # misleading for Triton kernels.
                    if isinstance(statement, ast.AugAssign):
                        if (
                            isinstance(statement.target, ast.Name)
                            and statement.target.id in address_only_names
                        ):
                            continue
                        # ``x op= y`` is one real binary operation whose result
                        # replaces x.  Reify only this AST-local equivalence so
                        # ordinary logical in-place updates are not omitted.
                        augmented = ast.copy_location(
                            ast.BinOp(
                                left=statement.target,
                                op=statement.op,
                                right=statement.value,
                            ),
                            statement,
                        )
                        record_operation(
                            augmented,
                            target_alias(statement.target),
                        )
                        continue
                    raw_target = (
                        statement.targets[0]
                        if isinstance(statement, ast.Assign) and statement.targets
                        else statement.target
                    )
                    if (
                        isinstance(raw_target, ast.Name)
                        and raw_target.id in address_only_names
                    ):
                        continue
                    target = (
                        target_alias(statement.targets[0]) if isinstance(statement, ast.Assign) and statement.targets
                        else target_alias(statement.target)
                    )
                    value = statement.value
                    if value is not None:
                        record_operation(value, target)
                    continue
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    record_operation(statement.value)
                    continue
                if isinstance(statement, ast.Return):
                    # Helper returns carry real operation semantics (for
                    # example SwiGLU's sigmoid/multiply), while the return
                    # binding itself remains a compact control fact.
                    if statement.value is not None:
                        record_operation(statement.value, f"return{len(returns)}")
                    returns.append({"ordinal": len(returns), "scope": scope, "value": value_ref(statement.value)})
                    continue
                if isinstance(statement, ast.If):
                    control.append({"ordinal": len(control), "scope": scope, "kind": "if", "condition": value_ref(statement.test), "has_else": bool(statement.orelse)})
                    walk_statements(statement.body)
                    walk_statements(statement.orelse)
                    continue
                if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                    control.append({"ordinal": len(control), "scope": scope, "kind": "for" if isinstance(statement, (ast.For, ast.AsyncFor)) else "while", "control_input": value_ref(statement.iter if isinstance(statement, (ast.For, ast.AsyncFor)) else statement.test)})
                    walk_statements(statement.body)
                    walk_statements(statement.orelse)

        walk_statements(function.body)
    scope_invocations: dict[str, int] = {"entry": 1}
    pending_bindings = list(helper_call_bindings)
    while pending_bindings:
        progressed = False
        remaining: list[dict[str, Any]] = []
        for binding in pending_bindings:
            caller_scope = binding.get("caller_scope")
            callee_scope = binding.get("callee_scope")
            caller_count = scope_invocations.get(caller_scope)
            if (
                isinstance(caller_scope, str)
                and isinstance(callee_scope, str)
                and isinstance(caller_count, int)
            ):
                scope_invocations[callee_scope] = (
                    scope_invocations.get(callee_scope, 0) + caller_count
                )
                progressed = True
            else:
                remaining.append(binding)
        if not progressed:
            unclassified_calls.add("unresolved_helper_call_graph")
            break
        pending_bindings = remaining
    # The hierarchical ``operations`` view is useful for Source navigation,
    # but it is too easy for a provider to mistake a nested leaf for part of
    # its parent.  Add a deterministic, Source-owned flat view.  These refs
    # are fact identities only; they are never provider/J labels and never
    # create an operation absent from the AST.
    source_fact_ordinal = 0
    operation_requirements: list[dict[str, Any]] = []
    for operation in operations:
        facts = [operation]
        nested = operation.get("nested_operations")
        if isinstance(nested, list):
            facts.extend(item for item in nested if isinstance(item, Mapping))
        for fact in facts:
            source_ref = f"s{source_fact_ordinal}"
            source_fact_ordinal += 1
            fact["source_ref"] = source_ref
            family = required_operation_family(
                fact.get("kind"), fact.get("operator")
            )
            if family is None:
                continue
            operands = fact.get("operands")
            row: dict[str, Any] = {
                "source_ref": source_ref,
                "source_order": len(operation_requirements),
                # ``canonical_op`` is the canonical family identity.  Do not
                # publish the same value a second time under ``family``: the
                # provider-facing projection must have one authoritative
                # spelling while the host-side counters remain separate.
                "canonical_op": family,
                "operands": copy.deepcopy(operands)
                if isinstance(operands, list)
                else [],
            }
            # Preserve the existing Source assignment target when the AST
            # fact has one.  This is a read-only Source fact that lets the
            # provider connect a top-level producer to the already-projected
            # Source value flow; it is not a J id/result and is never copied
            # into the submitted graph verbatim.
            result = fact.get("result")
            if isinstance(result, str) and result:
                row["result"] = result
            if family in {"map_binary", "map_unary", "reduce"}:
                row["canonical_attrs"] = {"fn": fact.get("operator")}
            else:
                # These leaves have no schema-derived ``fn`` attribute, so
                # retain their Source operator explicitly.
                row["operator"] = fact.get("operator")
            units = (
                2
                if fact.get("kind") == "fused_multiply_add"
                else max(1, len(operands) - 1)
                if fact.get("kind") in {"compare", "boolean"}
                and isinstance(operands, list)
                else 1
            )
            if units > 1:
                row["source_unit_count"] = units
            operation_requirements.append(row)
    # The flat AST walk above is Source-navigation order, not necessarily
    # producer order: an outer expression is recorded before its nested
    # operands.  Derive exact producer edges only from the unique expression
    # occurrence refs already present in each row's operands/results, then
    # expose a stable topological order.  These refs remain Source fact
    # locators; they are never provider/J labels.  Operands without a
    # canonical operation producer (for example, a transfer-owned load) stay
    # external and are resolved through the existing transfer ledger.
    source_result_refs = {
        row["result"]: row["source_ref"]
        for row in operation_requirements
        if isinstance(row.get("result"), str)
        and re.fullmatch(r"sx[0-9]+_[0-9]+", row["result"]) is not None
    }
    operation_dependencies: dict[str, set[str]] = {}
    operation_rows_by_ref = {
        row["source_ref"]: row for row in operation_requirements
    }
    for row in operation_requirements:
        source_ref = row["source_ref"]
        operands = row.get("operands")
        dependencies = sorted(
            {
                source_result_refs[operand]
                for operand in operands
                if isinstance(operand, str)
                and operand in source_result_refs
                and source_result_refs[operand] != source_ref
            }
        ) if isinstance(operands, list) else []
        row["source_dependencies"] = dependencies
        operation_dependencies[source_ref] = set(dependencies)
    ordered_operation_requirements: list[dict[str, Any]] = []
    remaining_operation_refs = set(operation_rows_by_ref)
    while remaining_operation_refs:
        ready = sorted(
            (
                source_ref
                for source_ref in remaining_operation_refs
                if operation_dependencies[source_ref].isdisjoint(
                    remaining_operation_refs
                )
            ),
            key=lambda source_ref: operation_rows_by_ref[source_ref]["source_order"],
        )
        if not ready:
            unclassified_calls.add("cyclic_source_operation_requirements")
            break
        source_ref = ready[0]
        ordered_operation_requirements.append(operation_rows_by_ref[source_ref])
        remaining_operation_refs.remove(source_ref)
    if not remaining_operation_refs:
        operation_requirements = ordered_operation_requirements
    required_operation_families: dict[str, int] = {}
    required_operation_attributes: dict[tuple[str, str, str], int] = {}
    for operation in operations:
        scope = operation.get("scope", "entry")
        operation_weight = scope_invocations.get(scope, 0)
        if operation_weight <= 0:
            continue
        facts = [operation]
        nested = operation.get("nested_operations")
        if isinstance(nested, list):
            facts.extend(item for item in nested if isinstance(item, Mapping))
        for fact in facts:
            family = required_operation_family(fact.get("kind"), fact.get("operator"))
            if family is not None:
                operands = fact.get("operands")
                units = (
                    2
                    if fact.get("kind") == "fused_multiply_add"
                    else
                    max(1, len(operands) - 1)
                    if fact.get("kind") in {"compare", "boolean"}
                    and isinstance(operands, list)
                    else 1
                )
                units *= operation_weight
                required_operation_families[family] = (
                    required_operation_families.get(family, 0) + units
                )
                operator = fact.get("operator")
                attribute_values = (
                    fact.get("comparison_operators")
                    if fact.get("kind") == "compare"
                    else [operator]
                )
                attribute_family = (
                    family
                    if family in {"map_binary", "map_unary", "reduce"}
                    and fact.get("kind") != "fused_multiply_add"
                    else None
                )
                if attribute_family is not None and isinstance(
                    attribute_values, list
                ):
                    for raw_attribute_value in attribute_values:
                        attribute_value = (
                            {"maximum": "max", "minimum": "min"}.get(
                                raw_attribute_value, raw_attribute_value
                            )
                            if isinstance(raw_attribute_value, str)
                            else None
                        )
                        if not isinstance(attribute_value, str):
                            continue
                        key = (attribute_family, "fn", attribute_value)
                        increment = (
                            operation_weight
                            if fact.get("kind") == "compare"
                            else units
                        )
                        required_operation_attributes[key] = (
                            required_operation_attributes.get(key, 0) + increment
                        )
                elif fact.get("kind") == "fused_multiply_add":
                    for value in ("mul", "add"):
                        key = ("map_binary", "fn", value)
                        required_operation_attributes[key] = (
                            required_operation_attributes.get(key, 0)
                            + operation_weight
                        )
    required_contracts = set(required_operation_families) | operation_hints | unresolved
    return {
        "kind": "source_semantic_spine",
        "status": "unavailable" if unclassified_calls else "pass",
        "reason": "source_operation_family_unavailable" if unclassified_calls else None,
        "unclassified_calls": sorted(unclassified_calls),
        "entry_scope": "entry",
        "operation_order": "Source order",
        "entry_argument_projection": entry_argument_projection,
        "helper_call_bindings": helper_call_bindings,
        "alias_convention": (
            "helper aliases navigate; J inputs use entry_argument_projection.provider_ref"
        ),
        "operations": operations,
        "operation_requirements": operation_requirements,
        "required_operation_families": {
            name: required_operation_families[name]
            for name in sorted(required_operation_families)
        },
        "required_operation_attributes": [
            {
                "op": operation,
                "attribute": attribute,
                "value": value,
                "count": count,
            }
            for (operation, attribute, value), count in sorted(
                required_operation_attributes.items()
            )
        ],
        "control": control,
        "returns": returns,
        # Include every canonical family actually required by this Source in
        # the first brief. A kernel that needs both iota and reduce must not be
        # forced through a one-leaf lookup bottleneck before it can make its
        # first complete graph declaration.
        "basic_operation_contracts": sorted(required_contracts),
        "missing_operation_contracts": [],
        "authority": "Source AST facts only; not text, TTIR, or J",
    }


def _source_brief(input_root: Path) -> None:
    """Expose the selected frontend's Source and compiler/Harness evidence."""

    # The brief is the authoring entry, not a Source preview.  Source content
    # is available only through the one bounded supplemental MCP read so a
    # fresh provider starts with J construction facts rather than an arbitrary
    # slice of program text plus a second, competing task description.
    source_kind = _mounted_source_kind(input_root)
    ttir_path = input_root / "source_ttir.json"
    if (
        ttir_path.is_file()
        and not ttir_path.is_symlink()
        and source_kind != "cuda"
    ):
        compiler_realization: dict[str, Any] = {
            "selected_frontend": "triton_ttir",
            "source_kind": source_kind,
            "model_visible": True,
            "lift_acceptance_authority": False,
            "observer_available": True,
            "observer_query": f"{QUERY_COMMAND} source-observer",
            "authority": (
                "Source-TTIR and compiler observer facts are mounted for the "
                "Triton/TTIR branch; query the exact observer separately. They "
                "are compiler evidence and do not replace J semantics or select "
                "C/pi fields."
            ),
        }
    elif source_kind == "cuda":
        compiler_realization = {
            "selected_frontend": "cuda_cpp_native",
            "source_kind": source_kind,
            "model_visible": True,
            "lift_acceptance_authority": False,
            "authority": (
                "CUDA C++ source is a native source projection. Use its CUDA "
                "compiler/Harness view and never parse it as Python or substitute "
                "Triton TTIR evidence."
            ),
        }
    else:
        compiler_realization = {
            "selected_frontend": "python_or_tilelang_native_projection",
            "source_kind": source_kind,
            "model_visible": True,
            "lift_acceptance_authority": False,
            "authority": (
                "Use the selected frontend's mounted Source projection and full "
                "compiler/Harness view. A Triton TTIR query is available only when "
                "source_ttir.json is mounted."
            ),
        }
    framework = _j_framework_value(input_root)
    semantic_spine = _source_semantic_spine(input_root)
    source_operation_requirements: list[dict[str, Any]] | None = None
    source_operation_dependencies: list[list[int]] | None = None
    if isinstance(semantic_spine, Mapping):
        # ``operation_requirements`` is the compact provider-facing flat
        # projection.  Keep the richer hierarchical AST walk host-side; the
        # provider must not reconstruct it from nested summaries.
        semantic_spine = dict(semantic_spine)
        raw_operation_requirements = semantic_spine.get("operation_requirements")
        if isinstance(raw_operation_requirements, list):
            source_operation_requirements = copy.deepcopy(raw_operation_requirements)
            requirement_positions = {
                row.get("source_ref"): index
                for index, row in enumerate(source_operation_requirements)
                if isinstance(row, Mapping)
            }
            source_operation_dependencies = []
            for index, row in enumerate(source_operation_requirements):
                if not isinstance(row, Mapping):
                    raise ValueError("canonical Source operation requirement is malformed")
                for dependency in row.get("source_dependencies", ()):
                    producer_index = requirement_positions.get(dependency)
                    if producer_index is None or producer_index >= index:
                        raise ValueError(
                            "canonical Source operation dependency order is malformed"
                        )
                    source_operation_dependencies.append([index, producer_index])
                row.pop("source_order", None)
                row.pop("source_dependencies", None)
        semantic_spine.pop("operations", None)
        semantic_spine.pop("operation_requirements", None)
    semantic_contract = framework.get("semantic_candidate_contract")
    mcp_write_protocol = framework.get("mcp_write_protocol")
    transfer_projection = framework.get("semantic_transfer_projection")
    construction_identity = framework.get("construction_identity")
    host_launch_receipt = framework.get("host_launch_contract")
    if not isinstance(transfer_projection, Mapping):
        raise ValueError("canonical framework lacks semantic transfer projection")
    transfer_occurrences = transfer_projection.get("occurrences")
    if not isinstance(transfer_occurrences, list) or re.search(
        r"(?<![A-Za-z0-9_])[pc][0-9]+(?![A-Za-z0-9_])",
        json.dumps(transfer_projection, sort_keys=True, ensure_ascii=False),
    ):
        raise ValueError("canonical framework semantic transfer projection is invalid")
    if not isinstance(semantic_contract, Mapping):
        raise ValueError("canonical framework lacks semantic candidate contract")
    if not isinstance(mcp_write_protocol, Mapping):
        raise ValueError("canonical framework lacks MCP semantic write protocol")
    if (
        not isinstance(construction_identity, Mapping)
        or set(construction_identity)
        != {
            "construction_digest",
            "source_facts_digest",
            "transfer_ledger_digest",
            "allocator_snapshot_digest",
        }
        or any(not isinstance(value, str) or not value for value in construction_identity.values())
    ):
        raise ValueError("canonical framework lacks construction identity")
    if (
        not isinstance(host_launch_receipt, Mapping)
        or host_launch_receipt.get("source_facts_digest")
        != construction_identity.get("source_facts_digest")
        or transfer_projection.get("host_launch_contract")
        != host_launch_receipt
    ):
        raise ValueError("canonical framework host launch receipt is not transfer-bound")
    semantic_template = semantic_contract.get("template")
    operation_namespace = semantic_contract.get("operation_namespace")
    if not isinstance(semantic_template, Mapping):
        raise ValueError("canonical framework lacks semantic candidate template")
    if not isinstance(operation_namespace, Mapping):
        raise ValueError("canonical framework lacks semantic operation namespace")
    compact_wire = framework.get("compact_provider_wire")
    compact_operations = (
        compact_wire.get("operations") if isinstance(compact_wire, Mapping) else None
    )
    basic_leaves = semantic_spine.get("basic_operation_contracts")
    if (
        not isinstance(compact_operations, Mapping)
        or not isinstance(basic_leaves, list)
        or any(not isinstance(name, str) for name in basic_leaves)
        or any(name not in compact_operations for name in basic_leaves)
    ):
        raise ValueError("canonical framework lacks compact basic operation leaves")
    if source_operation_requirements is None:
        raise ValueError("canonical Source operation requirements are unavailable")
    _validate_semantic_mcp_write_projection(semantic_contract, mcp_write_protocol)
    # The first authoring entry is intentionally semantic-only.  Publishing a
    # complete artifact-shaped seed here made providers copy host-owned schema,
    # C, ABI, node, and scope fields back into a candidate.  The constructor
    # still retains that mechanical seed privately; this projection contains
    # only the schema-derived authoring surface and Source facts needed to fill
    # it.
    def brief_construction_summary(requirements: Any) -> dict[str, Any]:
        """Project only first-turn arity and shape from the full ledger.

        Exact occurrence/output facts stay in ``semantic-transfer``.  This
        prevents a second full transfer ledger from entering every fresh
        provider context while retaining a schema-derived requirement for
        complete Source-order rows.
        """

        if not isinstance(requirements, Mapping):
            raise ValueError("canonical framework lacks semantic construction requirements")
        transfer = requirements.get("source_transfer")
        submission = requirements.get("submission")
        semantic_required = requirements.get("semantic_required")
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
        if not all(
            isinstance(value, Mapping)
            for value in (transfer, submission, semantic_required, accesses, outputs)
        ):
            raise ValueError("canonical semantic construction requirements are malformed")
        access_count = accesses.get("count")
        output_count = outputs.get("count")
        obligations = requirements.get("obligations")
        if (
            isinstance(access_count, bool)
            or not isinstance(access_count, int)
            or access_count < 0
            or isinstance(output_count, bool)
            or not isinstance(output_count, int)
            or output_count < 0
            or not isinstance(obligations, list)
            or not obligations
        ):
            raise ValueError("canonical semantic construction row counts are malformed")
        for obligation in obligations:
            stage = obligation.get("stage") if isinstance(obligation, Mapping) else None
            if not isinstance(stage, str):
                raise ValueError("canonical semantic obligation stage is malformed")
        typed_outputs: list[dict[str, Any]] = []
        output_rows = outputs.get("rows")
        if not isinstance(output_rows, list) or len(output_rows) != output_count:
            raise ValueError("canonical semantic output rows are malformed")
        for ordinal, row in enumerate(output_rows):
            output_type = row.get("output_type") if isinstance(row, Mapping) else None
            dtype = output_type.get("dtype") if isinstance(output_type, Mapping) else None
            shape = output_type.get("shape") if isinstance(output_type, Mapping) else None
            if (
                not isinstance(dtype, str)
                or not dtype
                or not isinstance(shape, list)
                or len(shape) > 16
                or any(
                    not (
                        isinstance(extent, str) and bool(extent)
                        or isinstance(extent, int)
                        and not isinstance(extent, bool)
                        and extent >= 0
                    )
                    for extent in shape
                )
            ):
                raise ValueError("canonical semantic output type is malformed")
            typed_outputs.append(
                {
                    "ordinal": ordinal,
                    # Keep the Source brief's typed-output row identical to
                    # the ``begin_candidate`` receipt.  Both are projections
                    # of the same Source ABI obligation, and the graph
                    # ingress/feedback contract consumes the flat
                    # ``dtype``/``shape`` pair.  A nested ``output_type``
                    # spelling here made the Agent translate the same fact
                    # between two wire shapes before it could author a typed
                    # producer; no producer or output binding is inferred by
                    # this normalization.
                    "dtype": dtype[:64],
                    "shape": copy.deepcopy(shape),
                }
            )
            source_store_requirements = row.get("source_store_requirements")
            if source_store_requirements is not None:
                if (
                    not isinstance(source_store_requirements, list)
                    or not source_store_requirements
                    or any(
                        not isinstance(item, Mapping)
                        or set(item)
                        != {"ordinal", "kind", "operation_family", "coordinates"}
                        or not _is_canonical_ordinal(item.get("ordinal"))
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
                    raise ValueError(
                        "canonical semantic output store requirements are malformed"
                    )
                typed_outputs[-1]["source_store_requirements"] = copy.deepcopy(
                    source_store_requirements
                )
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
                        not _is_canonical_ordinal(item)
                        for item in partition_requirement["store_ordinals"]
                    )
                    or any(
                        not isinstance(item, list)
                        for item in partition_requirement["store_coordinates"]
                    )
                ):
                    raise ValueError(
                        "canonical semantic output partition requirement is malformed"
                    )
                assembly = partition_requirement.get("assembly")
                if assembly is not None:
                    if (
                        not isinstance(assembly, Mapping)
                        or set(assembly)
                        != {
                            "canonical_op",
                            "axis",
                            "operand_store_ordinals",
                            "operand_coordinates",
                            "operand_type",
                            "result_type",
                        }
                        or assembly.get("canonical_op") != "concat"
                        or type(assembly.get("axis")) is not int
                        or assembly.get("result_type")
                        != {"dtype": dtype, "shape": copy.deepcopy(shape)}
                        or not isinstance(assembly.get("operand_type"), Mapping)
                        or assembly["operand_type"].get("dtype") != dtype
                        or not isinstance(assembly.get("operand_type", {}).get("shape"), list)
                        or not isinstance(assembly.get("operand_store_ordinals"), list)
                        or not isinstance(assembly.get("operand_coordinates"), list)
                        or len(assembly["operand_store_ordinals"])
                        != len(assembly["operand_coordinates"])
                        or len(assembly["operand_store_ordinals"])
                        != len(source_store_requirements or ())
                    ):
                        raise ValueError(
                            "canonical semantic output assembly requirement is malformed"
                        )
                typed_outputs[-1]["source_store_partition_requirement"] = (
                    copy.deepcopy(dict(partition_requirement))
                )
        return {
            "authority": requirements.get("authority"),
            "source_transfer": {
                "status": transfer.get("status"),
                "closed_required": transfer.get("closed_required"),
                "exact_resource": "intentir://lift/source/transfer-ledger",
            },
            "submission": {
                "required": copy.deepcopy(submission.get("required")),
                "atomic": submission.get("atomic"),
                "one_complete_candidate": submission.get("one_complete_candidate"),
            },
            "semantic_required": {
                "operation_body": {"required": True},
                "accesses": {"count": access_count, "order": "Source ordinal"},
                "outputs": {
                    "count": output_count,
                    "order": "Source output ABI order",
                    "typed_output_requirements": typed_outputs,
                },
            },
            "obligations": {
                "count": len(obligations),
                "resource": "semantic-transfer",
                "resolution": "host_validator_only",
            },
        }

    def brief_write_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
        """Project only the executable MCP shape into the first brief.

        Tool descriptions, host-derived rows, and the duplicate initial
        template are transport metadata, not first-turn semantic facts.  The
        template is already carried once in this brief, and exact tool/op
        details remain available through the MCP tool surface and one bounded
        schema lookup.
        """

        steps = protocol.get("steps")
        required_steps = protocol.get("required_steps")
        arguments = protocol.get("tool_arguments")
        if (
            not isinstance(steps, list)
            or not isinstance(required_steps, list)
            or not isinstance(arguments, Mapping)
            or set(arguments) != set(steps)
        ):
            raise ValueError("canonical MCP semantic write protocol is malformed")
        return {
            "authority": protocol.get("authority"),
            "steps": copy.deepcopy(steps),
            "required_steps": copy.deepcopy(required_steps),
        }

    framework_projection: dict[str, Any] = {
        # The transfer-ledger resource repeats this opaque record.  The MCP
        # lifecycle compares the two before opening construction, then binds
        # a graph-after-ledger reread to the same mounted transaction.
        "construction_identity": copy.deepcopy(dict(construction_identity)),
        # The exact host launch receipt is revalidated and carried by the
        # mandatory transfer-ledger read. Repeating it in the first brief
        # consumes authoring context without adding an Agent-owned choice.
        "semantic_candidate": copy.deepcopy(dict(semantic_template)),
        "operation_namespace": copy.deepcopy(dict(operation_namespace)),
        "mcp_write_protocol": brief_write_protocol(mcp_write_protocol),
        "semantic_construction_requirements": brief_construction_summary(
            framework.get("semantic_construction_requirements")
        ),
        "program_domain_requirements": copy.deepcopy(
            framework.get("program_domain_requirements", {})
        ),
        "provider_dimension_aliases": copy.deepcopy(
            framework.get("provider_dimension_aliases", [])
        ),
        "source_operation_requirements": source_operation_requirements,
        # Encode expression producer edges once as [consumer_index,
        # producer_index] pairs. This is the same Source-derived relation as
        # the per-row begin receipt, without repeating a long field name on
        # every operation row in the bounded first-turn brief.
        "source_operation_dependencies": source_operation_dependencies,
        "source_transfer_counts": {
            "occurrences": len(transfer_occurrences),
        },
        "operation_contract_lookup": framework.get("operation_contract_lookup"),
        "source_semantic_spine": semantic_spine,
        # The spine tells a fresh owner which ordinary leaves occur.  Include
        # their compact canonical forms here so it normally spends no turn on
        # schema. If the brief budget moves them out, every required leaf is
        # fetched atomically by the one bounded lookup.
        "basic_operation_leaves": {
            name: copy.deepcopy(compact_operations[name]) for name in basic_leaves
        },
    }
    brief: dict[str, Any] = {
        "protocol": "intentir.lift-source-brief.v4",
        "source": {
            "resource": "intentir://lift/source/text",
            "on_demand": True,
        },
        "required_construction_fact": {
            "resource": "intentir://lift/source/transfer-ledger",
            "must_read_before_write": True,
        },
        "semantic_authoring_entry": framework_projection,
        "compiler_realization": {
            key: compiler_realization[key]
            for key in ("selected_frontend", "source_kind", "observer_available")
            if key in compiler_realization
        },
    }

    # A no-candidate authoring-preparation failure has no repair base and a
    # fresh semantic-entry owner does not read feedback/frontier.  Carry only
    # the host-sanitized graph receipt into its first required resource so the
    # next legal lifecycle can correct one public wire rule without seeing any
    # rejected declaration, candidate, Source text, or provider trajectory.
    frontier_path = input_root / "feedback_frontier.json"
    if frontier_path.is_file() and not frontier_path.is_symlink():
        frontier = _load_json(frontier_path)
        preparation = (
            frontier.get("preparation_receipt")
            if isinstance(frontier, Mapping)
            else None
        )
        if isinstance(preparation, Mapping):
            prior: dict[str, Any] = {}
            for key in (
                "failure_class",
                "diagnostic",
                "attempted_step",
                "next_legal_action",
            ):
                value = preparation.get(key)
                if isinstance(value, str) and value:
                    prior[key] = value[:256]
            graph = preparation.get("operation_graph_failure")
            if isinstance(graph, Mapping):
                # ``feedback_frontier.json`` is produced from the host's
                # already-sanitized preparation receipt.  Copy that bounded
                # diagnostic object as one opaque relation envelope instead
                # of maintaining a second field whitelist here.  The host
                # projector is the single authority for dropping candidate,
                # Source, declaration and operation payloads; preserving the
                # envelope keeps future safe relation fields synchronized.
                try:
                    safe_graph = copy.deepcopy(dict(graph))
                    json.dumps(
                        safe_graph,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                except (TypeError, ValueError, UnicodeError):
                    safe_graph = {}
                if safe_graph:
                    prior["operation_graph_failure"] = safe_graph
            if prior:
                brief["prior_authoring_feedback"] = prior

    def brief_size() -> int:
        return len(
            json.dumps(
                brief,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    if brief_size() > FIRST_TURN_BRIEF_CHARS:
        # The seed and compact leaf projection are the minimum wire authority.
        # This fallback removes only frontend prose; it never removes ABI rows,
        # occurrence counts, or schema-derived field names.
        brief["compiler_realization"] = {
            key: brief["compiler_realization"][key]
            for key in ("selected_frontend", "source_kind")
            if key in brief.get("compiler_realization", {})
        }
    if brief_size() > FIRST_TURN_BRIEF_CHARS:
        # Compiler realization is not an authoring prerequisite: the active
        # first-turn surface exposes only source brief followed by the transfer
        # ledger.  Drop this auxiliary receipt before failing the one resource
        # the lifecycle requires.  Source text itself is likewise not exposed
        # by the semantic-entry phase, so retaining its informational locator
        # cannot justify making source_brief unreadable.
        brief.pop("compiler_realization", None)
    if brief_size() > FIRST_TURN_BRIEF_CHARS:
        brief.pop("source", None)
    if brief_size() > FIRST_TURN_BRIEF_CHARS:
        # Compact leaf projections are a convenience, not an independent
        # semantic authority.  Preserve every leaf by promoting it to the one
        # bounded schema lookup that follows the transfer ledger; this is
        # strictly safer than truncating the brief or returning an error that
        # prevents construction from starting.
        entry = brief.get("semantic_authoring_entry")
        if not isinstance(entry, Mapping):
            raise ValueError("source brief semantic entry is malformed")
        spine = entry.get("source_semantic_spine")
        leaves = entry.get("basic_operation_leaves")
        if not isinstance(spine, Mapping) or not isinstance(leaves, Mapping):
            raise ValueError("source brief compact operation projection is malformed")
        basic = spine.get("basic_operation_contracts")
        missing = spine.get("missing_operation_contracts")
        if (
            not isinstance(basic, list)
            or not isinstance(missing, list)
            or any(not isinstance(name, str) for name in [*basic, *missing])
        ):
            raise ValueError("source brief operation lookup policy is malformed")
        spine = dict(spine)
        spine["basic_operation_contracts"] = []
        spine["missing_operation_contracts"] = sorted(set([*basic, *missing]))
        entry = dict(entry)
        entry["source_semantic_spine"] = spine
        entry.pop("basic_operation_leaves", None)
        brief["semantic_authoring_entry"] = entry
    if brief_size() > MAX_OUTPUT_CHARS:
        # ``source_store_requirements`` is a Source-authorized navigation aid,
        # not a required first-turn contract.  The complete rows remain in the
        # mounted canonical Source bundle and are revalidated into the
        # begin_candidate typed-output receipt.  Drop only this duplicated
        # first-turn copy when the bounded brief would otherwise be rejected;
        # retain output dtype/shape and every operation/access obligation.
        entry = brief.get("semantic_authoring_entry")
        requirements = (
            entry.get("semantic_construction_requirements")
            if isinstance(entry, Mapping)
            else None
        )
        semantic_required = (
            requirements.get("semantic_required")
            if isinstance(requirements, Mapping)
            else None
        )
        outputs = (
            semantic_required.get("outputs")
            if isinstance(semantic_required, Mapping)
            else None
        )
        typed_output_requirements = (
            outputs.get("typed_output_requirements")
            if isinstance(outputs, Mapping)
            else None
        )
        if (
            not isinstance(entry, Mapping)
            or not isinstance(requirements, Mapping)
            or not isinstance(semantic_required, Mapping)
            or not isinstance(outputs, Mapping)
            or not isinstance(typed_output_requirements, list)
        ):
            raise ValueError("source brief semantic construction summary is malformed")
        compact_requirements = copy.deepcopy(dict(requirements))
        compact_required = dict(compact_requirements["semantic_required"])
        compact_outputs = dict(compact_required["outputs"])
        compact_typed_outputs: list[dict[str, Any]] = []
        for row in typed_output_requirements:
            if not isinstance(row, Mapping):
                raise ValueError("source brief typed output requirements are malformed")
            compact_row = dict(row)
            compact_row.pop("source_store_requirements", None)
            # The complete partition locator is revalidated and carried by
            # begin_candidate.  Keeping this duplicate in the first-turn
            # brief can push a real kernel over the fixed MCP output budget;
            # remove only the duplicate, never the mounted Source bundle or
            # the begin receipt that the construction phase consumes.
            # The complete partition proof is revalidated from the mounted
            # bundle and carried by begin_candidate.required_slots.  It is a
            # duplicate on the first-turn brief; remove the whole copy so a
            # large real kernel cannot lose its operation spine to prompt
            # budget pressure.  This does not remove the requirement from the
            # MCP transaction or weaken its graph check.
            compact_row.pop("source_store_partition_requirement", None)
            compact_typed_outputs.append(compact_row)
        compact_outputs["typed_output_requirements"] = compact_typed_outputs
        compact_required["outputs"] = compact_outputs
        compact_requirements["semantic_required"] = compact_required
        compact_entry = dict(entry)
        compact_entry["semantic_construction_requirements"] = compact_requirements
        brief["semantic_authoring_entry"] = compact_entry
    if brief_size() > MAX_OUTPUT_CHARS:
        # These descriptions explain the projection but do not carry a
        # semantic obligation.  Keep the structured Source argument map,
        # operation rows, family counts and attributes; remove only prose
        # labels when a large real kernel still exceeds the fixed MCP result
        # budget after the optional store-navigation copy was deferred.
        entry = brief.get("semantic_authoring_entry")
        spine = entry.get("source_semantic_spine") if isinstance(entry, Mapping) else None
        if not isinstance(entry, Mapping) or not isinstance(spine, Mapping):
            raise ValueError("source brief semantic spine is malformed")
        compact_entry = dict(entry)
        compact_spine = dict(spine)
        for field in ("entry_scope", "operation_order", "alias_convention", "authority"):
            compact_spine.pop(field, None)
        compact_entry["source_semantic_spine"] = compact_spine
        brief["semantic_authoring_entry"] = compact_entry
    if brief_size() > MAX_OUTPUT_CHARS:
        # A fresh retry may also carry a bounded preparation frontier.  Keep
        # only the rule-level fields needed to choose the next legal action;
        # the full Source facts and current typed receipts remain authoritative
        # elsewhere.  Never let historical diagnostic prose make the current
        # semantic entry unreadable.
        prior = brief.get("prior_authoring_feedback")
        if isinstance(prior, Mapping):
            compact_prior: dict[str, Any] = {}
            for field in (
                "failure_class",
                "attempted_step",
                "next_legal_action",
            ):
                value = prior.get(field)
                if isinstance(value, str) and value:
                    compact_prior[field] = value[:96]
            graph = prior.get("operation_graph_failure")
            if isinstance(graph, Mapping):
                # The mounted frontier is already a bounded, host-sanitized
                # relation envelope.  Retain it as a whole during brief
                # compaction so adding one safe relation (shape, dtype,
                # predicate, or future type contract) cannot silently split
                # the direct and MCP feedback paths.
                try:
                    compact_graph = copy.deepcopy(dict(graph))
                    json.dumps(
                        compact_graph,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                except (TypeError, ValueError, UnicodeError):
                    compact_graph = {}
                if compact_graph:
                    compact_prior["operation_graph_failure"] = compact_graph
            if compact_prior:
                brief["prior_authoring_feedback"] = compact_prior
            else:
                brief.pop("prior_authoring_feedback", None)
    if brief_size() > MAX_OUTPUT_CHARS:
        # Prior feedback is diagnostic convenience only.  If a large source
        # projection still leaves no room after the bounded reduction above,
        # start the fresh owner from current Source facts rather than rejecting
        # the semantic entry or exposing an unbounded history payload.
        brief.pop("prior_authoring_feedback", None)
    if brief_size() > MAX_OUTPUT_CHARS:
        raise ValueError("source brief exceeds the bounded semantic entry budget")
    _emit_compact_json(brief)


def _semantic_transfer(input_root: Path, start: int, count: int) -> None:
    """Expose one bounded page of canonical Source-order transfer obligations."""

    if start < 1 or count < 1 or count > 80:
        raise ValueError("semantic transfer start/count must be positive and count <= 80")
    framework = _j_framework_value(input_root)
    transfer_projection = framework.get("semantic_transfer_projection")
    requirements = framework.get("semantic_construction_requirements")
    construction_identity = framework.get("construction_identity")
    projection_occurrences = (
        transfer_projection.get("occurrences")
        if isinstance(transfer_projection, Mapping)
        else None
    )
    host_launch_receipt = (
        transfer_projection.get("host_launch_contract")
        if isinstance(transfer_projection, Mapping)
        else None
    )
    if not isinstance(requirements, Mapping) or not isinstance(projection_occurrences, list):
        raise ValueError("canonical framework lacks semantic construction requirements")
    if not isinstance(host_launch_receipt, Mapping):
        raise ValueError("canonical semantic transfer lacks host launch receipt")
    if (
        not isinstance(construction_identity, Mapping)
        or set(construction_identity)
        != {
            "construction_digest",
            "source_facts_digest",
            "transfer_ledger_digest",
            "allocator_snapshot_digest",
        }
        or any(not isinstance(value, str) or not value for value in construction_identity.values())
    ):
        raise ValueError("canonical semantic transfer identity is malformed")
    obligation_rows = validate_source_semantic_obligations(
        requirements.get("obligations"),
        construction_identity["source_facts_digest"],
    )
    semantic_required = requirements.get("semantic_required")
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
    if (
        not isinstance(access_rows, list)
        or not isinstance(output_rows, list)
        or len(access_rows) != len(projection_occurrences)
    ):
        raise ValueError("canonical semantic transfer rows are malformed")

    source_transfer = requirements.get("source_transfer")
    if not isinstance(source_transfer, Mapping):
        raise ValueError("canonical semantic transfer status is malformed")
    transfer_status = source_transfer.get("status")
    transfer_reason = source_transfer.get("reason")

    def compact_access(
        source_row: Mapping[str, Any],
        wire_row: Mapping[str, Any],
        *,
        blocked: bool = False,
    ) -> dict[str, Any]:
        required = wire_row.get("required_fields")
        source = source_row.get("source_semantics")
        template = wire_row.get("wire_template")
        provider_fields = wire_row.get("provider_fields")
        provider_owned = wire_row.get("provider_owned_fields")
        provider_owned_expressions = wire_row.get(
            "provider_owned_expression_fields"
        )
        provider_relations = wire_row.get("provider_owned_relations")
        source_authorizations = source.get("semantic_authorizations")
        source_owner = (
            source_authorizations.get("owner")
            if isinstance(source_authorizations, Mapping)
            else None
        )
        source_owner_outline = (
            source_owner.get("authoring_outline")
            if isinstance(source_owner, Mapping)
            else None
        )
        if source_owner_outline is not None and (
            not isinstance(source_owner_outline, Mapping)
            or set(source_owner_outline) != {"store", "graph"}
            or not isinstance(source_owner_outline.get("store"), Mapping)
            or not isinstance(source_owner_outline.get("graph"), list)
        ):
            raise ValueError("canonical Source owner outline is malformed")
        broadcast_requirements = (
            owner_broadcast_requirements(source_owner_outline)
            if isinstance(source_owner_outline, Mapping)
            else None
        )
        if (
            broadcast_requirements is not None
            and source.get("owner_broadcast_requirements")
            != broadcast_requirements
        ):
            raise ValueError("canonical Source owner broadcast projection is inconsistent")
        if (
            not isinstance(required, list)
            or not isinstance(source, Mapping)
            or not isinstance(template, Mapping)
            or not isinstance(provider_fields, list)
            or not isinstance(provider_owned, list)
            or any(
                not isinstance(field, str)
                or field not in required
                or field not in template
                for field in provider_owned
            )
            or not isinstance(provider_owned_expressions, list)
            or any(
                not isinstance(field, str)
                or field not in required
                or field not in template
                or field in provider_owned
                for field in provider_owned_expressions
            )
            or not isinstance(provider_relations, Mapping)
            or wire_row.get("ordinal") != source_row.get("ordinal")
            or wire_row.get("access_label") != source_row.get("access_label")
            or source_row.get("index_expression_requirement")
            != wire_row.get("index_expression_requirement")
            or (
                wire_row.get("index_expression_requirement") is not None
                and (
                    not isinstance(
                        wire_row.get("index_expression_requirement"), str
                    )
                    or not wire_row.get("index_expression_requirement")
                    or "index_expression" not in provider_owned_expressions
                )
            )
            or template.get("access_label") != wire_row.get("access_label")
            or template.get("ordinal") != wire_row.get("ordinal")
            or "id" in template
        ):
            raise ValueError("canonical semantic access row is malformed")
        resolved: dict[str, Any] = {}
        unresolved: list[str] = []
        if not blocked:
            for field in required:
                if field in provider_owned or field in provider_owned_expressions:
                    # The ledger proves that this field is required, and the
                    # provider wire describes its legal namespace, but only
                    # the semantic author may choose its actual expression or
                    # local label.  Do not present a placeholder as a Source
                    # value to be copied.
                    unresolved.append(field)
                    continue
                value = template.get(field)
                if value is None:
                    unresolved.append(field)
                else:
                    resolved[field] = copy.deepcopy(value)
            # Effect order and the source-side address-owner π slot are already
            # deterministic host facts. They are returned as resolved state,
            # never as declaration fields for the provider to author.
            for field in ("effect_order", "pi_binding"):
                value = source.get(field)
                if value is not None:
                    resolved[field] = copy.deepcopy(value)
        else:
            unresolved.append("closed_source_transfer_ledger")
        result = {
            "ordinal": wire_row.get("ordinal"),
            "access_label": wire_row.get("access_label"),
            "provider_fields": list(provider_fields),
            "provider_owned_fields": list(provider_owned),
            "provider_owned_expression_fields": list(provider_owned_expressions),
            "provider_owned_relations": copy.deepcopy(dict(provider_relations)),
            "required_fields": list(required),
            "wire_template": copy.deepcopy(dict(template)),
            "resolved_fields": resolved,
            "unresolved_fields": unresolved,
            # Provider-owned expressions and local labels are intentionally
            # unresolved at this read. They are authoring obligations, not a
            # failed host Source binding (the blocked-ledger branch below is).
            "next_legal_action": "declare_accesses",
        }
        access_obligations = [
            copy.deepcopy(row)
            for row in obligation_rows
            if row["source_locator"].get("ordinal") == wire_row.get("ordinal")
        ]
        if not access_obligations:
            raise ValueError("canonical semantic access has no obligation contract")
        result["obligations"] = access_obligations
        predicate_requirement = wire_row.get("predicate_requirement")
        if isinstance(predicate_requirement, str):
            result["predicate_requirement"] = predicate_requirement
        index_requirement = wire_row.get("index_expression_requirement")
        if isinstance(index_requirement, str):
            result["index_expression_requirement"] = index_requirement
        if source_owner_outline is not None:
            # This is an existing Source-issued committed outline. It contains
            # only the logical typed DAG and store relation; owner tokens and
            # host handles are never copied into the provider surface.
            result["source_owner_authoring_outline"] = copy.deepcopy(
                dict(source_owner_outline)
            )
            result["owner_broadcast_requirements"] = copy.deepcopy(
                broadcast_requirements
            )
        return result

    if transfer_status != "pass":
        # An ambiguous Source ledger is a host preparation failure, never an
        # authoring prompt.  Keeping this branch explicit prevents a fresh
        # provider from treating missing tensor/predicate ownership as an
        # invitation to guess.
        _emit_compact_json(
            {
                "kind": "canonical_semantic_transfer_ledger",
                "status": transfer_status,
                "reason": transfer_reason,
                "construction_identity": copy.deepcopy(dict(construction_identity)),
                "host_launch_contract": copy.deepcopy(dict(host_launch_receipt)),
                "occurrences": [
                    compact_access(source_row, wire_row, blocked=True)
                    for source_row, wire_row in zip(
                        access_rows[start - 1 : start - 1 + count],
                        projection_occurrences[start - 1 : start - 1 + count],
                        strict=True,
                    )
                    if isinstance(source_row, Mapping) and isinstance(wire_row, Mapping)
                ],
                "unresolved_fields": ["closed_source_transfer_ledger"],
                "next_legal_action": "host_source_binding_required",
            }
        )
        return
    first = start - 1

    def page(rows: list[Any], total: int) -> dict[str, Any]:
        next_start = start + len(rows)
        return {
            "start": start,
            "count": len(rows),
            "total": total,
            "next_start": next_start if next_start <= total else None,
            "rows": rows,
        }

    # ``count`` is a requested upper bound, not a promise that the complete
    # page fits the MCP output limit. In particular, owner outlines can make
    # a nine-occurrence ledger larger than the fixed 10k character ceiling.
    # Choose the largest prefix that fits, preserving every row and its order.
    for page_count in range(count, 0, -1):
        access_page = [
            compact_access(source_row, wire_row)
            for source_row, wire_row in zip(
                access_rows[first : first + page_count],
                projection_occurrences[first : first + page_count],
                strict=True,
            )
            if isinstance(source_row, Mapping) and isinstance(wire_row, Mapping)
        ]
        output_page = [
            {
                "ordinal": first + offset,
                "output": row.get("output"),
                "output_type": copy.deepcopy(row.get("output_type")),
                "required_fields": ["value"],
                "wire_template": copy.deepcopy(row.get("wire_template")),
                "obligations": [
                    copy.deepcopy(obligation)
                    for obligation in obligation_rows
                    if obligation["source_locator"].get("output_ordinal")
                    == first + offset
                ],
            }
            for offset, row in enumerate(output_rows[first : first + page_count])
            if isinstance(row, Mapping)
        ]
        payload = {
            "kind": "canonical_semantic_transfer_ledger",
            "status": "pass",
            "construction_identity": copy.deepcopy(dict(construction_identity)),
            "host_launch_contract": copy.deepcopy(dict(host_launch_receipt)),
            "accesses": page(access_page, len(access_rows)),
            "outputs": page(output_page, len(output_rows)),
            "next_legal_action": "begin_candidate_then_declare_accesses",
        }
        next_starts = [
            item["next_start"]
            for item in (payload["accesses"], payload["outputs"])
            if item["next_start"] is not None
        ]
        # Access and output rows are two independently sized Source-order
        # projections sharing one page cursor.  Once one side is exhausted,
        # its next_start is None; the live side must still be allowed to
        # continue.  Use the greatest live cursor so no occurrence page is
        # dropped and let the exhausted side return an explicitly empty page.
        payload["next_start"] = max(next_starts) if next_starts else None
        if payload["next_start"] is not None:
            payload["next_legal_action"] = "read_next_transfer_page"
        if len(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))) <= MAX_OUTPUT_CHARS:
            _emit_compact_json(payload)
            return
    raise ValueError("one semantic transfer row exceeds the MCP output limit")


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


def _bounded_unparse(node: ast.AST, limit: int = 160) -> str:
    """Render one exact AST expression for navigation, never for inference."""

    try:
        value = ast.unparse(node)
    except Exception:
        value = ast.dump(node, annotate_fields=False, include_attributes=False)
    return value[:limit]


def _source_obligations_value(
    input_root: Path,
    section: str | None,
    start: int,
    count: int,
) -> dict[str, Any]:
    """Project bounded Source syntax facts for the MCP Lift owner.

    The labels are navigation buckets, not semantic classifications.  Every
    expression, call, keyword and line is copied from the mounted Source AST
    (or native text fallback); this helper never chooses a J operation, C slot,
    access/effect, predicate, carrier, or pi relation.
    """

    if start < 1 or count < 1 or count > 80:
        raise ValueError("source-obligations start/count must be positive and count <= 80")
    source_kind = _mounted_source_kind(input_root)
    source_path = input_root / "source.py"
    source_text = source_path.read_text(encoding="utf-8")
    buckets: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
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
        )
    }

    if source_kind == "cuda":
        memory_pattern = re.compile(
            r"\b(?:load|store|atomic_[A-Za-z0-9_]+|atomic[A-Za-z0-9_]+)\b"
        )
        control_pattern = re.compile(r"\b(?:for|while|if|else|switch|return)\b")
        for line_number, raw_line in enumerate(source_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if memory_pattern.search(line):
                buckets["memory_accesses"].append(
                    {"line": line_number, "text": line[:240]}
                )
            if control_pattern.search(line):
                buckets["control"].append(
                    {"line": line_number, "text": line[:240]}
                )
            if "mask" in line.casefold():
                buckets["mask_uses"].append(
                    {"line": line_number, "text": line[:240]}
                )
            if "other" in line.casefold() or "fill" in line.casefold():
                buckets["fills"].append(
                    {"line": line_number, "text": line[:240]}
                )
        authority = (
            "bounded literal native Source line facts; CUDA/C++ text is not parsed "
            "as Python and no J/C/pi field is inferred"
        )
    else:
        try:
            tree = ast.parse(source_text)
        except SyntaxError as exc:
            return {
                "protocol": "intentir.lift-source-obligations",
                "source_kind": source_kind,
                "status": "unavailable",
                "diagnostic": f"Source AST unavailable: {exc}",
                "authority": "read-only navigation; no J/C/pi inference",
            }

        class ObservationVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_stack: list[str] = []

            def _scope(self) -> str | None:
                return self.function_stack[-1] if self.function_stack else None

            def _record(
                self,
                bucket: str,
                node: ast.AST,
                **fields: Any,
            ) -> None:
                line = getattr(node, "lineno", None)
                if not isinstance(line, int):
                    return
                item: dict[str, Any] = {"line": line}
                scope = self._scope()
                if scope is not None:
                    item["scope"] = scope
                item.update(fields)
                buckets[bucket].append(item)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._record(
                    "scopes",
                    node,
                    kind="function",
                    name=node.name,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    parameters=[
                        argument.arg for argument in node.args.args
                    ],
                )
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._record(
                    "scopes",
                    node,
                    kind="async_function",
                    name=node.name,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    parameters=[
                        argument.arg for argument in node.args.args
                    ],
                )
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_For(self, node: ast.For) -> None:
                self._record(
                    "control",
                    node,
                    kind="for",
                    target=_bounded_unparse(node.target),
                    iterator=_bounded_unparse(node.iter),
                )
                self.generic_visit(node)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                self._record(
                    "control",
                    node,
                    kind="async_for",
                    target=_bounded_unparse(node.target),
                    iterator=_bounded_unparse(node.iter),
                )
                self.generic_visit(node)

            def visit_While(self, node: ast.While) -> None:
                self._record(
                    "control",
                    node,
                    kind="while",
                    condition=_bounded_unparse(node.test),
                )
                self.generic_visit(node)

            def visit_If(self, node: ast.If) -> None:
                self._record(
                    "control",
                    node,
                    kind="if",
                    condition=_bounded_unparse(node.test),
                    has_else=bool(node.orelse),
                )
                self.generic_visit(node)

            def visit_Return(self, node: ast.Return) -> None:
                self._record(
                    "control",
                    node,
                    kind="return",
                    value=(
                        _bounded_unparse(node.value)
                        if node.value is not None
                        else None
                    ),
                )
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                targets = [_bounded_unparse(target) for target in node.targets]
                expression = _bounded_unparse(node.value)
                self._record(
                    "expression_facts",
                    node,
                    kind="assignment",
                    targets=targets,
                    expression=expression,
                )
                lowered = expression.casefold()
                if any(
                    token in lowered
                    for token in (
                        "program_id(",
                        "arange(",
                        "make_block_ptr(",
                        "offsets",
                    )
                ):
                    self._record(
                        "program_axes",
                        node,
                        targets=targets,
                        expression=expression,
                    )
                if any(
                    token in lowered
                    for token in (
                        "ptr",
                        "offset",
                        "stride",
                        "program_id(",
                        "arange(",
                    )
                ):
                    self._record(
                        "index_construction",
                        node,
                        targets=targets,
                        expression=expression,
                    )
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                if node.value is not None:
                    target = _bounded_unparse(node.target)
                    expression = _bounded_unparse(node.value)
                    self._record(
                        "expression_facts",
                        node,
                        kind="annotated_assignment",
                        targets=[target],
                        expression=expression,
                    )
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                function = (
                    _bounded_unparse(node.func)
                    if isinstance(node.func, ast.AST)
                    else "<call>"
                )
                leaf = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else None
                )
                call = _bounded_unparse(node)
                keywords = {
                    item.arg: _bounded_unparse(item.value)
                    for item in node.keywords
                    if item.arg is not None
                }
                if leaf in _SOURCE_MEMORY_CALLS:
                    self._record(
                        "memory_accesses",
                        node,
                        function=function,
                        call=call,
                        mask=keywords.get("mask"),
                        other=keywords.get("other"),
                    )
                if "mask" in keywords:
                    self._record(
                        "mask_uses",
                        node,
                        function=function,
                        mask=keywords["mask"],
                    )
                if "other" in keywords:
                    self._record(
                        "fills",
                        node,
                        function=function,
                        other=keywords["other"],
                    )
                if leaf in {"to", "cast", "astype"} or ".to(" in call:
                    self._record(
                        "casts",
                        node,
                        function=function,
                        call=call,
                    )
                if leaf in {"program_id", "arange", "make_block_ptr"}:
                    self._record(
                        "program_axes",
                        node,
                        function=function,
                        call=call,
                    )
                if leaf not in _SOURCE_MEMORY_CALLS and leaf not in {
                    "program_id",
                    "arange",
                    "load",
                    "store",
                }:
                    self._record(
                        "helper_calls",
                        node,
                        function=function,
                        call=call,
                    )
                self.generic_visit(node)

        ObservationVisitor().visit(tree)
        authority = (
            "bounded literal Python Source AST facts; bucket names are navigation "
            "only and no J/C/pi field is inferred"
        )

    for values in buckets.values():
        values.sort(key=lambda item: (int(item.get("line", 0)), json.dumps(item, sort_keys=True)))
    if section is not None:
        selected = buckets.get(section)
        if selected is None:
            raise ValueError(f"unknown source-obligations section: {section}")
        page = selected[start - 1 : start - 1 + count]
        return {
            "protocol": "intentir.lift-source-obligations",
            "source_kind": source_kind,
            "section": section,
            "start": start,
            "total": len(selected),
            "items": page,
            "next_start": (
                start + len(page)
                if start - 1 + len(page) < len(selected)
                else None
            ),
            "authority": authority,
        }
    return {
        "protocol": "intentir.lift-source-obligations",
        "source_kind": source_kind,
        "status": "pass",
        "counts": {name: len(values) for name, values in buckets.items()},
        "sections": {
            name: values[: min(count, 3)] for name, values in buckets.items()
        },
        "section_queries": {
            name: f"{QUERY_COMMAND} source-obligations --section {name}"
            for name in buckets
        },
        "authority": authority,
    }


def _source_obligations(
    input_root: Path,
    section: str | None,
    start: int,
    count: int,
) -> None:
    _emit_compact_json(_source_obligations_value(input_root, section, start, count))

def _source_find(input_root: Path, query: str, context: int) -> None:
    """Find literal Source text with bounded line-numbered context.

    This is deliberately a text lookup rather than a parser or semantic
    extractor.  Lift still decides what the matched Source means and authors
    every J/C/pi field.
    """

    if not query:
        raise ValueError("source-find query must be non-empty")
    if context < 0 or context > 20:
        raise ValueError("source-find context must be in [0, 20]")
    lines = (input_root / "source.py").read_text(encoding="utf-8").splitlines()
    needle = query.casefold()
    matches = [
        index for index, line in enumerate(lines) if needle in line.casefold()
    ]
    if not matches:
        raise KeyError(f"no Source lines match {query!r}")

    ranges: list[tuple[int, int]] = []
    for index in matches:
        begin = max(0, index - context)
        end = min(len(lines), index + context + 1)
        if ranges and begin <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((begin, end))

    blocks: list[str] = []
    used = 0
    omitted = False
    match_set = set(matches)
    for begin, end in ranges:
        rendered = [f"--- source.py:{begin + 1} ---"]
        rendered.extend(
            f"{'>' if ordinal in match_set else ' '} {ordinal + 1:5d} | "
            f"{lines[ordinal]}"
            for ordinal in range(begin, end)
        )
        block = "\n".join(rendered)
        remaining = MAX_OUTPUT_CHARS - 300 - used
        if len(block) + 1 > remaining:
            if remaining > 80:
                blocks.append(block[: remaining - 40] + "\n[context truncated]")
            omitted = True
            break
        blocks.append(block)
        used += len(block) + 1
    print("\n".join(blocks))
    if omitted:
        print("\n[more matches omitted; use a narrower literal query]")


def _native_source_outline_value(
    input_root: Path, start: int, count: int, source_kind: str
) -> dict[str, Any]:
    """Return bounded line-based navigation for non-Python source projections."""

    if start < 1 or count < 1 or count > 48:
        raise ValueError("native source-outline start/count must be positive and count <= 48")
    lines = (input_root / "source.py").read_text(encoding="utf-8").splitlines()
    items: list[dict[str, Any]] = []
    depth = 0
    memory_pattern = re.compile(
        r"\b(?:load|store|atomic_[A-Za-z0-9_]+|atomic[A-Za-z0-9_]+)\b"
    )
    function_pattern = re.compile(
        r"\b(?:__global__|__device__|__host__)?\s*[A-Za-z_][A-Za-z0-9_<>:*&\s]*"
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)"
    )
    control_pattern = re.compile(r"\b(?:for|while|if|else|switch|return)\b")
    for ordinal, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        kind = "native_line"
        if function_pattern.search(stripped) and "(" in stripped:
            kind = "native_function_or_call"
        elif control_pattern.search(stripped):
            kind = "native_control"
        memory_calls = [
            [ordinal, match.group(0)] for match in memory_pattern.finditer(stripped)
        ]
        items.append(
            {
                "kind": kind,
                "line": ordinal,
                "end_line": ordinal,
                "depth": depth,
                "header": stripped[:200],
                "ordered_memory_calls": memory_calls,
                "contains_return": "return" in stripped,
                "loop_target_names": [],
                "lexically_preexisting_writes": [],
            }
        )
        depth = max(0, depth + stripped.count("{") - stripped.count("}"))
    page = items[start - 1 : start - 1 + count]
    next_start = start + len(page) if start - 1 + len(page) < len(items) else None
    return {
        "protocol": "intentir.lift-source-outline.native.v1",
        "source_kind": source_kind,
        "authority": (
            "bounded line-based native Source navigation; C++/native text is "
            "never parsed as Python and no J/C/pi field is inferred"
        ),
        "total": len(items),
        "start": start,
        "items": [
            {"ordinal": start + index, **item}
            for index, item in enumerate(page)
        ],
        "next_start": next_start,
        "next_command": (
            f"{QUERY_COMMAND} source-outline --start {next_start} --count {count}"
            if next_start is not None
            else None
        ),
    }


def _source_outline_value(input_root: Path, start: int, count: int) -> dict[str, Any]:
    """Return syntax-only control, write, and memory-call navigation facts.

    The extra records deliberately stop before semantic classification.  A name
    written in a loop after an earlier lexical binding is only a Source
    navigation fact: Lift still decides whether it is a compiler-visible loop
    carrier and authors every J/C/pi relation.  Likewise, memory-call order is
    literal AST order, not an inferred J effect or pi order.
    """

    if start < 1 or count < 1 or count > 48:
        raise ValueError("source-outline start must be positive and count in [1, 48]")
    source_kind = _mounted_source_kind(input_root)
    if source_kind != "python":
        return _native_source_outline_value(input_root, start, count, source_kind)
    source = (input_root / "source.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename="source.py")
    items: list[dict[str, Any]] = []

    def bound_names(node: ast.AST) -> list[tuple[str, int, int]]:
        found: dict[str, tuple[int, int]] = {}

        class BoundNameVisitor(ast.NodeVisitor):
            def visit_Name(self, item: ast.Name) -> None:
                if isinstance(item.ctx, ast.Store):
                    found.setdefault(
                        item.id,
                        (getattr(item, "lineno", 0), getattr(item, "col_offset", 0)),
                    )

            def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, item: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, item: ast.Lambda) -> None:
                return

        BoundNameVisitor().visit(node)
        return [
            (name, line, column)
            for name, (line, column) in sorted(
                found.items(), key=lambda item: (item[1][0], item[1][1], item[0])
            )
        ]

    def function_bindings_before(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        control: ast.AST,
    ) -> dict[str, tuple[int, int]]:
        bindings: dict[str, tuple[int, int]] = {}
        arguments = [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if function.args.vararg is not None:
            arguments.append(function.args.vararg)
        if function.args.kwarg is not None:
            arguments.append(function.args.kwarg)
        for argument in arguments:
            bindings.setdefault(
                argument.arg,
                (
                    getattr(argument, "lineno", function.lineno),
                    getattr(argument, "col_offset", function.col_offset),
                ),
            )
        control_line = getattr(control, "lineno", 0)
        # A loop can be nested below an if/with/try rather than appearing as a
        # direct function-body statement.  Use literal source order for the
        # checklist and filter every discovered binding by its own location;
        # otherwise walking the enclosing statement can accidentally count a
        # write from inside the loop as a pre-existing definition.
        for statement in function.body:
            for name, line, column in bound_names(statement):
                if line < control_line:
                    bindings.setdefault(name, (line, column))
        return bindings

    def target_names(node: ast.AST) -> set[str]:
        return {name for name, _line, _column in bound_names(node)}

    def contains_return(node: ast.AST) -> bool:
        found = False

        class ReturnVisitor(ast.NodeVisitor):
            def visit_Return(self, item: ast.Return) -> None:
                nonlocal found
                found = True

            def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, item: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, item: ast.Lambda) -> None:
                return

        for child in getattr(node, "body", []):
            ReturnVisitor().visit(child)
        for child in getattr(node, "orelse", []):
            ReturnVisitor().visit(child)
        return found

    def ordered_memory_calls(node: ast.AST) -> list[list[int | str]]:
        calls: list[tuple[int, int, str]] = []
        memory_names = {
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

        class MemoryCallVisitor(ast.NodeVisitor):
            def visit_Call(self, item: ast.Call) -> None:
                function = item.func
                leaf = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else function.id
                    if isinstance(function, ast.Name)
                    else None
                )
                if leaf in memory_names:
                    calls.append(
                        (
                            getattr(item, "lineno", 0),
                            getattr(item, "col_offset", 0),
                            ast.unparse(function)[:80],
                        )
                    )
                self.generic_visit(item)

            def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, item: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, item: ast.Lambda) -> None:
                return

        for child in getattr(node, "body", []):
            MemoryCallVisitor().visit(child)
        for child in getattr(node, "orelse", []):
            MemoryCallVisitor().visit(child)
        return [[line, call] for line, _column, call in sorted(calls)]

    class OutlineVisitor(ast.NodeVisitor):
        depth = 0
        functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

        def record(
            self,
            node: ast.AST,
            kind: str,
            extra: dict[str, Any] | None = None,
        ) -> None:
            line = getattr(node, "lineno", None)
            if not isinstance(line, int):
                return
            header = lines[line - 1].strip() if line <= len(lines) else ""
            record = {
                "kind": kind,
                "line": line,
                "end_line": getattr(node, "end_lineno", line),
                "depth": self.depth,
                "header": header[:200],
            }
            if extra:
                record.update(extra)
            items.append(record)

        def nested(self, node: ast.AST, kind: str) -> None:
            self.record(node, kind)
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.record(
                node,
                "function",
                {"ordered_memory_calls": ordered_memory_calls(node)},
            )
            self.functions.append(node)
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1
            self.functions.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.record(
                node,
                "async_function",
                {"ordered_memory_calls": ordered_memory_calls(node)},
            )
            self.functions.append(node)
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1
            self.functions.pop()

        def loop_extra(self, node: ast.For | ast.AsyncFor | ast.While) -> dict[str, Any]:
            function = self.functions[-1] if self.functions else None
            prior = (
                function_bindings_before(function, node)
                if function is not None
                else {}
            )
            body_writes: dict[str, tuple[int, int]] = {}
            for child in node.body:
                for name, line, column in bound_names(child):
                    body_writes.setdefault(name, (line, column))
            induction_names = (
                target_names(node.target)
                if isinstance(node, (ast.For, ast.AsyncFor))
                else set()
            )
            carried = [
                {
                    "name": name,
                    "definition_line": definition[0],
                    "first_loop_write_line": body_writes[name][0],
                }
                for name, definition in sorted(
                    prior.items(), key=lambda item: (item[1][0], item[1][1], item[0])
                )
                if name in body_writes and name not in induction_names
            ]
            return {
                "loop_target_names": sorted(induction_names),
                "lexically_preexisting_writes": carried,
                "contains_return": contains_return(node),
            }

        def visit_For(self, node: ast.For) -> None:
            self.record(node, "for", self.loop_extra(node))
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.record(node, "async_for", self.loop_extra(node))
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_While(self, node: ast.While) -> None:
            self.record(node, "while", self.loop_extra(node))
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_If(self, node: ast.If) -> None:
            has_return = contains_return(node)
            self.record(
                node,
                "if",
                {
                    "contains_return": has_return,
                    **(
                        {"ordered_memory_calls": ordered_memory_calls(node)}
                        if has_return
                        else {}
                    ),
                },
            )
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_Try(self, node: ast.Try) -> None:
            self.nested(node, "try")

        def visit_With(self, node: ast.With) -> None:
            self.nested(node, "with")

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            self.nested(node, "async_with")

        def visit_Match(self, node: ast.Match) -> None:
            self.nested(node, "match")

        def visit_Return(self, node: ast.Return) -> None:
            self.record(node, "return")

        def visit_Break(self, node: ast.Break) -> None:
            self.record(node, "break")

        def visit_Continue(self, node: ast.Continue) -> None:
            self.record(node, "continue")

    OutlineVisitor().visit(tree)
    page = items[start - 1 : start - 1 + count]
    next_start = start + len(page) if start - 1 + len(page) < len(items) else None
    return {
        "protocol": "intentir.lift-source-outline.v2",
        "authority": (
            "syntax-only Source navigation; names, return presence, and memory-call "
            "order are literal AST facts, not carrier/effect classification or "
            "frontend tuple order; Lift still authors all J/C/pi fields"
        ),
        "total": len(items),
        "start": start,
        "items": [
            {"ordinal": start + index, **item}
            for index, item in enumerate(page)
        ],
        "next_start": next_start,
        "next_command": (
            f"{QUERY_COMMAND} source-outline --start {next_start} "
            f"--count {count}"
            if next_start is not None
            else None
        ),
    }


def _source_outline(input_root: Path, start: int, count: int) -> None:
    _emit_json(_source_outline_value(input_root, start, count))


def _resolve_json_path(value: Any, path: str | None) -> Any:
    if not path:
        return value
    current = value
    for component in path.split("."):
        if isinstance(current, dict):
            if component not in current:
                raise KeyError(
                    f"unknown JSON path component {component!r}; "
                    f"choose {list(current)}; use dot-separated paths such as "
                    "j.signature.parameters.0"
                )
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            current = current[int(component)]
        else:
            raise KeyError(
                f"cannot descend through {component!r}; paths use dot-separated "
                "keys and decimal list components, not slash or bracket syntax"
            )
    return current


def _harness(input_root: Path, path: str | None) -> None:
    value = _resolve_json_path(
        _load_json(input_root / "harness_view.json"), path
    )
    try:
        _emit_json(value)
    except ValueError:
        if path is not None or not isinstance(value, dict):
            raise
        _emit_json(
            {
                "keys": list(value),
                "query": f"{QUERY_COMMAND} harness --path KEY[.SUBKEY]",
            }
        )


def _canonical_abi_parts(input_root: Path) -> dict[str, Any]:
    """Project ABI/C facts from the mounted canonical construction only.

    The MCP process must not reconstruct ABI rows from ``harness_view.json``.
    That file is a frontend input, while the mounted Source bundle is the
    already validated and anonymized construction authority.  A missing or
    inconsistent bundle is a preparation failure rather than permission to
    fall back to another ABI constructor.
    """

    construction = _mounted_canonical_construction(input_root)
    if construction is None:
        raise ValueError("canonical construction authority is not mounted")
    profile, _generic_seed = _canonical_wire_seed(input_root)
    seed = _mounted_source_wire_seed(input_root, profile, construction)
    source_facts = construction.get("source_facts")
    if not isinstance(source_facts, Mapping):
        raise TypeError("canonical construction has no SourceFacts")
    compiler_facts = source_facts.get("compiler_facts")
    if not isinstance(compiler_facts, Mapping):
        raise TypeError("canonical SourceFacts has no compiler facts")

    parameter_order = source_facts.get("parameter_order")
    constexpr_parameters = source_facts.get("constexpr_parameters")
    arguments = source_facts.get("arguments")
    compiler_slots = source_facts.get("compiler_slots")
    if not all(
        isinstance(value, list)
        for value in (parameter_order, constexpr_parameters, arguments, compiler_slots)
    ):
        raise TypeError("canonical SourceFacts ABI declarations are malformed")

    bound_values = compiler_facts.get("bound_constexpr_values", {})
    bound_value_types = compiler_facts.get("bound_constexpr_value_types", {})
    schedule_values = compiler_facts.get("schedule_values", {})
    schedule_types = compiler_facts.get("schedule_value_types", {})
    compiler_roles = compiler_facts.get("compiler_roles", {})
    if not all(
        isinstance(value, Mapping)
        for value in (
            bound_values,
            bound_value_types,
            schedule_values,
            schedule_types,
            compiler_roles,
        )
    ):
        raise TypeError("canonical SourceFacts ABI ledgers are malformed")

    c_root = seed.get("c")
    c_slots = c_root.get("slots") if isinstance(c_root, Mapping) else None
    if not isinstance(c_slots, list):
        raise TypeError("canonical Source seed has no C slot declarations")
    raw_slots: list[dict[str, Any]] = []
    for row in c_slots:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise TypeError("canonical Source seed contains an invalid C slot")
        raw_slots.append(
            {
                "slot": row["name"],
                **{
                    key: row[key]
                    for key in ("kind", "role", "value_type", "required")
                    if key in row
                },
            }
        )
    expected_c = [
        name
        for name in parameter_order
        if isinstance(name, str) and name.startswith("c")
    ] + list(compiler_slots)
    if [row["slot"] for row in raw_slots] != expected_c:
        raise ValueError("canonical Source seed C declaration order differs from SourceFacts")
    seed_values = c_root.get("values") if isinstance(c_root, Mapping) else None
    if seed_values != dict(schedule_values):
        raise ValueError("canonical Source seed C values differ from SourceFacts")
    j_root = seed.get("j")
    signature = j_root.get("signature") if isinstance(j_root, Mapping) else None
    seed_parameters = signature.get("parameters") if isinstance(signature, Mapping) else None
    if not isinstance(seed_parameters, list):
        raise TypeError("canonical Source seed has no signature parameters")
    seed_p = [
        row.get("name") for row in seed_parameters if isinstance(row, Mapping)
    ]
    expected_p = [name for name in parameter_order if name.startswith("p")]
    if seed_p != expected_p:
        raise ValueError("canonical Source seed parameter order differs from SourceFacts")
    bound_parameters = [
        name for name in constexpr_parameters if name in bound_values
    ]
    if set(bound_values) != set(bound_parameters):
        raise ValueError(
            "canonical SourceFacts bound constexpr values are outside constexpr_parameters"
        )

    semantic = {
        "parameter_order": list(parameter_order),
        "constexpr_parameters": list(constexpr_parameters),
        "shape_coordinates": compiler_facts.get("shape_coordinates", {}),
        "bound_constexpr_shape_coordinates": compiler_facts.get(
            "bound_constexpr_shape_coordinates", {}
        ),
        "aliases": compiler_facts.get("aliases", []),
    }
    return {
        "semantic": semantic,
        "realization": compiler_facts,
        "parameter_order": list(parameter_order),
        "arguments": [dict(item) for item in arguments],
        "bound_parameters": bound_parameters,
        "constexpr_parameters": list(constexpr_parameters),
        "bound_values": dict(bound_values),
        "bound_value_types": dict(bound_value_types),
        "schedule_values": dict(schedule_values),
        "schedule_types": dict(schedule_types),
        "compiler_slots": list(compiler_slots),
        "compiler_roles": dict(compiler_roles),
        "raw_slots": raw_slots,
    }


def _harness_abi_value(
    input_root: Path, start: int = 1, count: int = 24
) -> dict[str, Any]:
    """Project the exact Harness-owned J/C classification into one small ledger.

    The anonymous compiler view already decides which public handles are p* and
    c*, which constexprs are bound non-C carriers, and which shape coordinates
    are parametric.  This helper only joins those existing fields by handle.  It
    never chooses a logical_ref, creates a dimension, or classifies Source use.
    """

    if start < 1 or count < 1 or count > 80:
        raise ValueError("harness-abi start/count must be positive and count <= 80")
    parts = _canonical_abi_parts(input_root)
    semantic = parts["semantic"]
    parameter_order = parts["parameter_order"]
    arguments = parts["arguments"]
    bound_parameters = parts["bound_parameters"]
    bound_values = parts["bound_values"]
    bound_value_types = parts["bound_value_types"]
    schedule_values = parts["schedule_values"]
    raw_slots = parts["raw_slots"]
    if not isinstance(parameter_order, list) or not all(
        isinstance(item, str) for item in parameter_order
    ):
        raise TypeError("Harness parameter_order must be an array of strings")
    if not isinstance(bound_parameters, list) or not all(isinstance(item, str) for item in bound_parameters):
        raise TypeError("Harness bound constexpr declarations must be strings")
    slots_by_name = {item["slot"]: item for item in raw_slots}
    arguments_by_name = {
        item.get("name"): item
        for item in arguments
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    entries: list[dict[str, Any]] = []
    argument_fields = (
        "kind",
        "mode",
        "dtype",
        "rank",
        "shape",
        "shape_contract",
        "shape_coordinate",
        "value",
    )
    visible_order = list(parameter_order) + [
        item["slot"] for item in raw_slots if item["slot"] not in parameter_order
    ]
    for name in visible_order:
        entry: dict[str, Any] = {
            "name": name,
            "artifact_owner": "J.signature" if name.startswith("p") else "C",
        }
        argument = arguments_by_name.get(name)
        if argument is not None:
            entry["classification"] = (
                "runtime_or_representation_parameter"
                if name.startswith("p")
                else "pure_c_argument"
            )
            entry["declared"] = {
                field: argument[field]
                for field in argument_fields
                if field in argument
            }
        elif name in bound_parameters:
            value = bound_values.get(name)
            dtype = bound_value_types.get(name)
            if not isinstance(dtype, str) or not dtype:
                if isinstance(value, bool):
                    dtype = "i1"
                elif isinstance(value, int):
                    dtype = "i32" if -(2**31) <= value < 2**31 else "i64"
                elif isinstance(value, float):
                    dtype = "f64"
                elif isinstance(value, str):
                    dtype = "str"
                else:
                    dtype = None
            declared: dict[str, Any] = {
                "kind": "constexpr",
                "mode": "value",
                "dtype": dtype,
                "rank": 0,
            }
            bound_coordinates = semantic.get(
                "bound_constexpr_shape_coordinates", {}
            )
            if isinstance(bound_coordinates, Mapping) and name in bound_coordinates:
                declared["shape_coordinate"] = bound_coordinates[name]
            entry.update(
                {
                    "classification": "bound_non_c_constexpr_carrier",
                    "bound_valuation": value,
                    "declared": declared,
                }
            )
        elif name in slots_by_name:
            declaration = slots_by_name[name]
            entry.update(
                {
                    "classification": (
                        "compiler_option_c_slot"
                        if declaration.get("kind") == "compiler"
                        else "pure_c_constexpr_slot"
                    ),
                    "c_declaration": dict(declaration),
                    "c_value": schedule_values.get(name),
                }
            )
        else:
            entry["classification"] = "unresolved_in_exact_harness_view"
        entries.append(entry)
    selected = entries[start - 1 : start - 1 + count]
    next_start = start + len(selected)
    return {
        "protocol": "intentir.lift-harness-abi-view.v1",
        "parameters": selected,
        "parameter_start": start,
        "parameter_total": len(entries),
        "parameter_next_command": (
            f"{QUERY_COMMAND} harness-abi --start {next_start} --count {count}"
            if next_start <= len(entries)
            else None
        ),
        "bound_non_c_constexprs": list(bound_parameters),
        "bound_non_c_constexpr_values": dict(bound_values),
        "pure_c_values": dict(schedule_values),
        "compiler_c_slots": list(parts["compiler_slots"]),
        "layer_c_declarations": [dict(item) for item in raw_slots],
        "shape_coordinates": semantic.get("shape_coordinates", []),
        "bound_constexpr_shape_coordinates": semantic.get(
            "bound_constexpr_shape_coordinates", {}
        ),
        "aliases": semantic.get("aliases", []),
        "agent_owned_not_inferred": [
            "p* logical_ref and t*/x* ownership",
            "logical operators, control, accesses, effects, and outputs",
            "each typed pi relation and every J/C handle bound by it",
        ],
        "authority": (
            "Exact read-only selected-frontend Harness projection. Runtime "
            "shape/scalar values, C values, compiler roles and ABI order are "
            "copied from the mounted canonical SourceFacts construction; the query "
            "never chooses J/C/pi fields."
        ),
    }


def _compact_provider_wire_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-derived compact provider wire.

    The host profile now materializes this projection once.  The standalone
    context tool only reads it; it must not repeat the projection rules used
    by direct Responses authoring.
    """

    compact = profile.get("semantic_provider_wire_compact")
    if not isinstance(compact, Mapping):
        raise ValueError("canonical profile lacks compact semantic provider wire")
    return copy.deepcopy(dict(compact))


def _semantic_construction_identity(
    construction: Mapping[str, Any],
) -> dict[str, str]:
    """Return the opaque identity shared by brief and transfer-ledger reads.

    These values bind the small provider-facing views to one already-mounted
    Source/Harness construction.  They are digests of host-owned records, not
    a second Source projection and not a provider declaration field.
    """

    authority_digest = construction.get("authority_digest")
    source_facts = construction.get("source_facts")
    transfer_ledger = construction.get("transfer_ledger")
    allocator = construction.get("allocator")
    ledger_source_facts_digest = (
        transfer_ledger.get("source_facts_digest")
        if isinstance(transfer_ledger, Mapping)
        else None
    )

    def digest(value: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    source_facts_digest = (
        digest(source_facts) if isinstance(source_facts, Mapping) else None
    )
    declared_source_facts_digest = construction.get("source_facts_digest")
    declared_transfer_ledger_digest = construction.get("transfer_ledger_digest")
    transfer_ledger_digest = (
        digest(transfer_ledger) if isinstance(transfer_ledger, Mapping) else None
    )
    if (
        not isinstance(authority_digest, str)
        or not authority_digest
        or not isinstance(source_facts_digest, str)
        or not source_facts_digest
        or declared_source_facts_digest != source_facts_digest
        or ledger_source_facts_digest != source_facts_digest
        or not isinstance(transfer_ledger, Mapping)
        or declared_transfer_ledger_digest != transfer_ledger_digest
        or not isinstance(allocator, Mapping)
    ):
        raise ValueError("canonical construction identity is malformed")

    return {
        "construction_digest": authority_digest,
        "source_facts_digest": source_facts_digest,
        "transfer_ledger_digest": digest(transfer_ledger),
        "allocator_snapshot_digest": digest(allocator),
    }


def _j_framework_from_source_seed(
    input_root: Path,
    profile: Mapping[str, Any],
    seed: Mapping[str, Any],
    construction: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the host-built Source seed without reconstructing its fields."""

    j = seed.get("j")
    c = seed.get("c")
    if not isinstance(j, Mapping) or not isinstance(c, Mapping):
        raise TypeError("Source-bound canonical seed lacks J or C")
    intent = j.get("intent")
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    signature = j.get("signature")
    inputs = function.get("inputs") if isinstance(function, Mapping) else None
    outputs = function.get("outputs") if isinstance(function, Mapping) else None
    parameters = signature.get("parameters") if isinstance(signature, Mapping) else None
    slots = c.get("slots")
    values = c.get("values")
    if not all(
        isinstance(item, list)
        for item in (inputs, outputs, parameters, slots)
    ) or not isinstance(values, Mapping):
        raise TypeError("Source-bound canonical seed has incomplete ABI rows")

    allocator = construction.get("allocator")
    source_facts = construction.get("source_facts")
    if not isinstance(allocator, Mapping) or not isinstance(source_facts, Mapping):
        raise TypeError("mounted canonical construction authority is incomplete")
    source_bundle = _load_json(
        input_root / "schema" / "canonical_source_wire_bundle.json"
    )
    provider_dimension_aliases = source_bundle.get("provider_dimension_aliases")
    dimension_aliases = provider_dimension_alias_map(
        provider_dimension_aliases,
        canonical_dimensions=allocator.get("x"),
    )
    host_launch_receipt = source_bundle.get("host_launch_contract")
    transfer_projection = source_bundle.get("semantic_transfer_projection")
    if (
        not isinstance(host_launch_receipt, Mapping)
        or not isinstance(transfer_projection, Mapping)
        or transfer_projection.get("host_launch_contract")
        != host_launch_receipt
    ):
        raise ValueError("canonical host launch receipt is not transfer-bound")
    namespace_order = profile.get("namespace_order")
    source_owned = profile.get("source_owned_namespaces")
    candidate_owned = profile.get("candidate_owned_namespaces")
    if (
        not isinstance(namespace_order, list)
        or not isinstance(source_owned, list)
        or not isinstance(candidate_owned, list)
        or set(source_owned) | set(candidate_owned) != set(namespace_order)
        or set(source_owned) & set(candidate_owned)
    ):
        raise ValueError("canonical namespace ownership is malformed")
    if [key for key in allocator if key != "logical_bindings"] != namespace_order:
        raise ValueError("mounted allocator order differs from wire authority")
    for prefix in namespace_order:
        declared = allocator.get(prefix)
        if not isinstance(declared, list) or any(
            not isinstance(item, str) for item in declared
        ):
            raise ValueError(f"mounted allocator {prefix}* declarations are malformed")
        if declared != [f"{prefix}{index}" for index in range(len(declared))]:
            raise ValueError(f"mounted allocator {prefix}* declarations are not dense")

    source_bundle = _load_json(
        input_root / "schema" / "canonical_source_wire_bundle.json"
    )
    semantic_transfer_projection = source_bundle.get("semantic_transfer_projection")
    if (
        not isinstance(semantic_transfer_projection, Mapping)
        or not isinstance(semantic_transfer_projection.get("occurrences"), list)
        or re.search(
            r"(?<![A-Za-z0-9_])[pc][0-9]+(?![A-Za-z0-9_])",
            json.dumps(semantic_transfer_projection, sort_keys=True, ensure_ascii=False),
        )
    ):
        raise ValueError("canonical Source wire bundle lacks semantic transfer projection")
    semantic_construction_requirements = source_bundle.get(
        "semantic_construction_requirements"
    )
    if not isinstance(semantic_construction_requirements, Mapping):
        raise ValueError(
            "canonical Source wire bundle lacks semantic construction requirements"
        )
    source_facts_digest = source_bundle.get("source_facts_digest")
    semantic_obligations = semantic_construction_requirements.get("obligations")
    try:
        validate_source_semantic_obligations(
            semantic_obligations, source_facts_digest
        )
    except ValueError as exc:
        raise ValueError(
            "canonical Source wire bundle lacks a closed semantic obligation contract"
        ) from exc
    output_construction = source_bundle.get("output_construction")
    if not isinstance(output_construction, list) or any(
        not isinstance(row, Mapping) for row in output_construction
    ):
        raise ValueError("canonical Source wire bundle lacks output construction")
    if any("abi_carrier" in row for row in output_construction):
        raise ValueError("canonical Source output projection leaked ABI carrier")
    if len(output_construction) != len(outputs):
        raise ValueError("canonical Source output projection cardinality is inconsistent")
    for output_ordinal, (output_row, seed_row) in enumerate(
        zip(output_construction, outputs, strict=True)
    ):
        if not isinstance(seed_row, Mapping):
            raise ValueError("canonical Source seed output declaration is malformed")
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
                f"canonical Source output construction is not seed-bound at ordinal {output_ordinal}"
            )
    submission = semantic_construction_requirements.get("submission")
    semantic_required = semantic_construction_requirements.get("semantic_required")
    construction_sections = (
        submission.get("construction_sections")
        if isinstance(submission, Mapping)
        else None
    )
    if (
        not isinstance(submission, Mapping)
        or submission.get("root") != ["j", "pi"]
        or not isinstance(construction_sections, Mapping)
        or set(construction_sections)
        != {"operation", "scope", "boundary", "output"}
        or submission.get("required") != [
            "j.intent.intent_function.body",
            "j.boundary.accesses",
            "j.boundary.output_mappings",
            "pi",
        ]
        or submission.get("atomic") is not True
        or not isinstance(semantic_required, Mapping)
    ):
        raise ValueError("canonical semantic construction requirements are malformed")

    # Validate the public occurrence projection against the private ledger
    # before any brief, framework, or transfer page is exposed.  The ledger is
    # authoritative but its receipts must never be returned to the provider;
    # this function only performs an internal join and emits the sanitized
    # projection below.
    projected_occurrences = semantic_transfer_projection.get("occurrences")
    private_ledger = source_bundle.get("source_transfer_ledger")
    private_occurrences = (
        private_ledger.get("occurrences")
        if isinstance(private_ledger, Mapping)
        else None
    )
    private_bindings = (
        private_ledger.get("bindings")
        if isinstance(private_ledger, Mapping)
        else None
    )
    required_accesses = (
        semantic_required.get("accesses")
        if isinstance(semantic_required, Mapping)
        else None
    )
    required_access_rows = (
        required_accesses.get("rows")
        if isinstance(required_accesses, Mapping)
        else None
    )
    required_outputs = (
        semantic_required.get("outputs")
        if isinstance(semantic_required, Mapping)
        else None
    )
    required_output_rows = (
        required_outputs.get("rows")
        if isinstance(required_outputs, Mapping)
        else None
    )
    if (
        not isinstance(projected_occurrences, list)
        or not isinstance(private_occurrences, list)
        or not isinstance(private_bindings, list)
        or not isinstance(required_access_rows, list)
        or len(projected_occurrences) != len(private_occurrences)
        or len(projected_occurrences) != len(private_bindings)
        or len(projected_occurrences) != len(required_access_rows)
        or not isinstance(required_output_rows, list)
        or len(required_output_rows) != len(output_construction)
    ):
        raise ValueError("canonical Source projection cardinality is inconsistent")

    _validate_source_store_projection(
        required_access_rows,
        required_output_rows,
        output_construction,
        dimension_aliases,
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
                and (outline is None or public.get("authoring_outline") == outline)
            )
        return (
            set(public) == {"token", "operation_family"}
            and public.get("token") == token
            and public.get("operation_family") == family
        )

    for ordinal, (private_row, private_binding, wire_row, requirement_row) in enumerate(
        zip(
            private_occurrences,
            private_bindings,
            projected_occurrences,
            required_access_rows,
            strict=True,
        )
    ):
        private_auth = (
            private_binding.get("semantic_authorizations")
            if isinstance(private_binding, Mapping)
            else None
        )
        public_auth = (
            wire_row.get("semantic_authorizations")
            if isinstance(wire_row, Mapping)
            else None
        )
        if (
            not isinstance(private_row, Mapping)
            or not isinstance(private_binding, Mapping)
            or not isinstance(wire_row, Mapping)
            or not isinstance(requirement_row, Mapping)
            or not _is_canonical_ordinal(private_row.get("ordinal"))
            or not _is_canonical_ordinal(private_binding.get("ordinal"))
            or not _is_canonical_ordinal(wire_row.get("ordinal"))
            or not _is_canonical_ordinal(requirement_row.get("ordinal"))
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
            or set(private_auth) != {"access_view", "owner", "computed_guard"}
            or not isinstance(public_auth, Mapping)
            or re.search(
                r"(?<![A-Za-z0-9_])so_[0-9a-f]{32}_[0-9a-f]{64}(?![A-Za-z0-9_])",
                json.dumps(wire_row, sort_keys=True, ensure_ascii=False),
            )
            or any(
                isinstance(value, Mapping)
                and (
                    "status" in value
                    or "compiler_attestation" in value
                    or (name == "owner" and "token" in value)
                )
                for name, value in public_auth.items()
            )
            or not all(
                public_authorization_matches_private(
                    private_auth.get(name),
                    public_auth.get(name),
                    owner=name == "owner",
                )
                for name in ("access_view", "owner", "computed_guard")
            )
        ):
            raise ValueError("canonical Source authorization projection is inconsistent")
        source_semantics = requirement_row.get("source_semantics")
        owner_outline = (
            public_auth.get("owner", {}).get("authoring_outline")
            if isinstance(public_auth.get("owner"), Mapping)
            else None
        )
        expected_broadcasts = (
            owner_broadcast_requirements(owner_outline)
            if isinstance(owner_outline, Mapping)
            else None
        )
        if (
            not isinstance(source_semantics, Mapping)
            or source_semantics.get("kind") != wire_row.get("kind")
            or source_semantics.get("tensor") != wire_row.get("tensor")
            or source_semantics.get("masked") != wire_row.get("masked")
            or source_semantics.get("predicate") != wire_row.get("predicate")
            or source_semantics.get("fill_value") != wire_row.get("fill_value")
            or source_semantics.get("index") != wire_row.get("index")
            or source_semantics.get("effect_order") != wire_row.get("effect_order")
            or source_semantics.get("predicate_requirement")
            != wire_row.get("predicate_requirement")
            or source_semantics.get("pi_binding") != wire_row.get("pi_binding")
            or source_semantics.get("semantic_authorizations") != public_auth
            or (
                expected_broadcasts is not None
                and source_semantics.get("owner_broadcast_requirements")
                != expected_broadcasts
            )
            or (
                expected_broadcasts is None
                and "owner_broadcast_requirements" in source_semantics
            )
            or any(
                requirement_row.get(field) != wire_row.get(field)
                for field in (
                    "provider_fields",
                    "provider_owned_fields",
                    "provider_owned_expression_fields",
                    "provider_owned_relations",
                    "required_fields",
                    "wire_template",
                )
            )
        ):
            raise ValueError("canonical Source requirements are not projection-bound")

    for ordinal, (required_row, output_row) in enumerate(
        zip(required_output_rows, output_construction, strict=True)
    ):
        if (
            not isinstance(required_row, Mapping)
            or not isinstance(output_row, Mapping)
            or required_row.get("output") != output_row.get("output")
            or required_row.get("output_type") != output_row.get("output_type")
            or required_row.get("required_fields") != ["value"]
            or required_row.get("wire_template") != {"value": "<local_result_or_source_t>"}
        ):
            raise ValueError(
                f"canonical Source output projection is inconsistent at ordinal {ordinal}"
            )
    canonical_dimensions = allocator.get("x")
    try:
        provider_dimension_alias_map(
            provider_dimension_aliases,
            canonical_dimensions=canonical_dimensions,
        )
    except ValueError as exc:
        raise ValueError("canonical Source dimension aliases are not allocator-bound") from exc
    # TTIR is an optional frontend observation.  The canonical J seed is built
    # from SourceFacts, the transfer ledger, and the allocator, so a workload
    # whose compiler observation is unavailable must still be able to read its
    # first-authoring seed.  Keep a malformed mounted observation fail-closed,
    # but do not make absence of the optional TTIR sidecar an implicit
    # dependency of the generic J framework.
    source_observation: Mapping[str, Any] | None = None
    source_ttir_path = input_root / "source_ttir.json"
    if source_ttir_path.is_file() or source_ttir_path.is_symlink():
        source_observation = _source_ttir_observation(input_root)
    source_runtime_parameter_uses = (
        source_observation.get("source_runtime_parameter_uses", [])
        if source_observation is not None
        else []
    )
    if not isinstance(source_runtime_parameter_uses, list) or any(
        not isinstance(item, str) for item in source_runtime_parameter_uses
    ):
        source_runtime_parameter_uses = []

    semantic_candidate = profile.get("semantic_candidate")
    mcp_write_protocol = profile.get("semantic_mcp_write")
    if not isinstance(semantic_candidate, Mapping):
        raise ValueError("canonical wire profile lacks semantic candidate contract")
    root_fields = semantic_candidate.get("root_fields")
    j_fields = semantic_candidate.get("j_fields")
    semantic_template = semantic_candidate.get("template")
    operation_namespace = semantic_candidate.get("operation_namespace")
    j_field_types = semantic_candidate.get("j_field_types")
    boundary_field_types = semantic_candidate.get("boundary_field_types")
    template_j = semantic_template.get("j") if isinstance(semantic_template, Mapping) else None
    template_pi = semantic_template.get("pi") if isinstance(semantic_template, Mapping) else None
    if (
        not isinstance(root_fields, list)
        or not isinstance(j_fields, list)
        or not isinstance(semantic_template, Mapping)
        or not isinstance(j_field_types, Mapping)
        or not isinstance(boundary_field_types, Mapping)
        or set(semantic_template) != set(root_fields)
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
        or not isinstance(operation_namespace, Mapping)
        or not isinstance(mcp_write_protocol, Mapping)
    ):
        raise ValueError("canonical semantic candidate contract is malformed")
    _validate_semantic_mcp_write_projection(semantic_candidate, mcp_write_protocol)

    construction_identity = _semantic_construction_identity(construction)

    return {
        "kind": "canonical_j_framework",
        "status": "pass",
        "template_only": True,
        "submit_forbidden": True,
        "schema": profile["schema"],
        "root_layout": profile["root"]["fields"],
        "j_layout": profile["j"]["fields"],
        "namespace_order": list(namespace_order),
        "source_owned_namespaces": list(source_owned),
        "candidate_owned_namespaces": list(candidate_owned),
        "allocator": {
            prefix: list(allocator[prefix]) for prefix in namespace_order
        },
        "compact_provider_wire": _compact_provider_wire_contract(profile),
        "output_construction": copy.deepcopy(output_construction),
        "semantic_construction_requirements": copy.deepcopy(
            dict(semantic_construction_requirements)
        ),
        "program_domain_requirements": copy.deepcopy(
            source_bundle.get("program_domain_requirements", {})
        ),
        "provider_dimension_aliases": copy.deepcopy(provider_dimension_aliases),
        "semantic_candidate_contract": copy.deepcopy(dict(semantic_candidate)),
        "mcp_write_protocol": copy.deepcopy(dict(mcp_write_protocol)),
        "fixed_names": {
            "kernel_id": j.get("kernel_id"),
            "intent_function": function.get("name") if isinstance(function, Mapping) else None,
            "tensor_order": "inputs (including true inout) then output-only",
        },
        "canonical_wire_seed": copy.deepcopy(dict(seed)),
        "operation_contract_names": sorted(profile["operation_contracts"]),
        "operation_contract_lookup": "schema_lookup items=[\"op:NAME\"]",
        "construction_records": copy.deepcopy(profile["construction_records"]),
        "mechanical": {
            "harness_tensor_observations": {
                "inputs": copy.deepcopy(inputs),
                "outputs": copy.deepcopy(outputs),
            },
            "j_signature_parameters": copy.deepcopy(parameters),
            "c_slots": copy.deepcopy(slots),
            "c_values": copy.deepcopy(dict(values)),
        },
        "semantic_transfer_projection": copy.deepcopy(
            dict(semantic_transfer_projection)
        ),
        "host_launch_contract": copy.deepcopy(dict(host_launch_receipt)),
        "construction_identity": construction_identity,
        "source_transfer_counts": {
            "occurrences": len(semantic_transfer_projection.get("occurrences", ())),
        },
        "source_runtime_parameter_uses": source_runtime_parameter_uses,
        "source_transfer_queries": {
            "source_obligations": f"{QUERY_COMMAND} source-obligations"
        },
        "authority": (
            "Host canonical Source/Harness construction seed. MCP only exposes "
            "this projection; it does not reconstruct J/C/pi facts or acceptance."
        ),
    }


def _j_framework_value(input_root: Path) -> dict[str, Any]:
    """Build the compact mechanical seed for one canonical J artifact.

    This is the only Source-facing constructor in the Lift MCP surface.  It
    projects closed Harness facts and literal Source occurrence navigation into
    a small framework; it deliberately does not infer operators, SSA values,
    predicates, scopes, loops, or pi relations.  The returned object is a
    framework, not a candidate and must never be submitted as ``candidate.json``.
    """

    construction = _mounted_canonical_construction(input_root)
    if construction is None:
        raise ValueError("canonical construction authority is not mounted")
    wire_profile, _generic_seed = _canonical_wire_seed(input_root)
    source_seed = _mounted_source_wire_seed(
        input_root,
        wire_profile,
        construction,
    )
    return _j_framework_from_source_seed(
        input_root,
        wire_profile,
        source_seed,
        construction,
    )

def _harness_abi(
    input_root: Path, start: int = 1, count: int = 24
) -> None:
    _emit_json(_harness_abi_value(input_root, start, count))


def _harness_c_contract_value(input_root: Path) -> dict[str, Any]:
    """Project only the trusted Harness-owned C declaration/value ledger.

    This view is deliberately narrower than ``harness-abi``.  It does not
    expose p* parameters, logical tensor names, J references, or inferred
    attachments. It projects only the C declarations already closed by the
    mounted canonical Source construction.
    """

    parts = _canonical_abi_parts(input_root)
    schedule_values = parts["schedule_values"]
    raw_slots = parts["raw_slots"]
    if not isinstance(schedule_values, dict) or not isinstance(raw_slots, list):
        raise TypeError("Harness C declarations are malformed")

    allowed_fields = (
        "slot",
        "kind",
        "role",
        "value_type",
        "required",
    )
    slots: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_slots):
        if not isinstance(raw, dict) or not isinstance(raw.get("slot"), str):
            raise TypeError(f"Harness C declaration {index} is malformed")
        slot = raw["slot"]
        if "subject_ref" in raw or "axis_ref" in raw:
            raise ValueError(
                f"Harness C slot {slot!r} illegally carries J ownership; "
                "use a target E_C extension or pi binding"
            )
        row = {
            field: raw[field]
            for field in allowed_fields
            if field in raw
        }
        if "kind" not in row or "value_type" not in row or "required" not in row:
            raise TypeError(f"Harness C declaration {slot!r} is incomplete")
        if slot not in schedule_values:
            raise TypeError(f"Harness C declaration {slot!r} has no trusted value")
        row["trusted_value"] = schedule_values[slot]
        slots.append(row)

    return {
        "slot_order": [row["slot"] for row in slots],
        "slots": slots,
        "authority": (
            "Exact selected-frontend canonical C declaration/value projection. "
            "Rows and order are copied from the mounted Source construction; this "
            "query never creates J fields, p* parameters, subject/axis "
            "attachments, or backend recipes."
        ),
    }


def _harness_c_contract(input_root: Path) -> None:
    _emit_json(_harness_c_contract_value(input_root))


def _realization_ledger(input_root: Path) -> None:
    """Expose raw Triton/TTIR realization facts without assigning ownership."""

    observation = _source_ttir_observation(input_root)
    text = _source_ttir_text(input_root)
    lines = text.splitlines()
    program_pattern = re.compile(
        r'^\s*(?P<result>%[A-Za-z0-9_.$-]+)\s*=\s*'
        r'(?:(?:"tt\.get_program_id")|tt\.get_program_id)\s+'
        r'(?P<axis>[xyz])\b'
    )
    operation_pattern = re.compile(
        r'^\s*(?:%[^=]+?\s*=\s*)?(?:"(?P<quoted>[A-Za-z0-9_.-]+)"|'
        r'(?P<plain>[A-Za-z][A-Za-z0-9_.-]*))\b'
    )
    programs: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        match = program_pattern.search(line)
        if match is None:
            continue
        result = match.group("result")
        token = re.compile(
            rf'(?<![A-Za-z0-9_.$-]){re.escape(result)}(?![A-Za-z0-9_.$-])'
        )
        users: list[dict[str, Any]] = []
        for user_line_number, user_line in enumerate(lines, start=1):
            if user_line_number == line_number or token.search(user_line) is None:
                continue
            operation = operation_pattern.search(user_line)
            if operation is None:
                continue
            users.append(
                {
                    "line": user_line_number,
                    "op": operation.group("quoted") or operation.group("plain"),
                }
            )
        programs.append(
            {
                "physical_axis": "xyz".index(match.group("axis")),
                "result": result,
                "line": line_number,
                "direct_users": users[:16],
                "direct_user_count": len(users),
            }
        )
    fingerprint = observation.get("fingerprint")
    fingerprint = fingerprint if isinstance(fingerprint, dict) else {}
    construction = _mounted_canonical_construction(input_root)
    if construction is None:
        raise ValueError("canonical construction authority is not mounted")
    source_facts = construction.get("source_facts")
    compiler_facts = (
        source_facts.get("compiler_facts")
        if isinstance(source_facts, Mapping)
        else None
    )
    if not isinstance(compiler_facts, Mapping):
        raise TypeError("canonical SourceFacts has no compiler facts")
    memory_counts = {
        "load": len(re.findall(r'(?m)^\s*(?:%[^=]+?=\s*)?(?:"tt\.load"|tt\.load)\b', text)),
        "store": len(re.findall(r'(?m)^\s*(?:"tt\.store"|tt\.store)\b', text)),
        "atomic": len(re.findall(r'(?m)^\s*(?:%[^=]+?=\s*)?(?:"tt\.atomic[^"]*"|tt\.atomic[A-Za-z0-9_.-]*)\b', text)),
        "addptr": len(re.findall(r'(?m)^\s*%[^=]+?=\s*(?:"tt\.addptr"|tt\.addptr)\b', text)),
    }
    _emit_json(
        {
            "protocol": "intentir.lift-realization-ledger",
            "program_coordinates": programs,
            "loops": fingerprint.get("loops", []),
            "terminal_forks": fingerprint.get("terminal_forks", []),
            "memory_operation_counts": memory_counts,
            "harness_grid": {
                key: compiler_facts.get(key)
                for key in ("grid", "grid_contract", "grid_shape_coordinates")
                if key in compiler_facts
            },
            "authority": (
                "Read-only raw Triton Source/TTIR facts plus canonical compiler facts. This ledger does "
                "not classify semantics, choose J/C/pi ownership, or author fields."
            ),
        }
    )


def _knowledge(input_root: Path, query: str, context: int) -> None:
    if not query:
        raise ValueError("knowledge query must be non-empty")
    needle = query.casefold()
    blocks: list[str] = []
    for relative in (
        "knowledge/lift_contract.md",
        "knowledge/preflight.md",
    ):
        lines = (input_root / relative).read_text(encoding="utf-8").splitlines()
        ranges: list[tuple[int, int]] = []
        for index, line in enumerate(lines):
            if needle in line.casefold():
                ranges.append(
                    (
                        max(0, index - context),
                        min(len(lines), index + context + 1),
                    )
                )
        merged: list[tuple[int, int]] = []
        for begin, end in ranges:
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((begin, end))
        for begin, end in merged:
            rendered = [f"--- {relative}:{begin + 1} ---"]
            rendered.extend(
                f"{ordinal + 1:5d} | {lines[ordinal]}"
                for ordinal in range(begin, end)
            )
            block = "\n".join(rendered)
            if sum(len(item) + 1 for item in blocks) + len(block) > MAX_OUTPUT_CHARS:
                print("\n".join(blocks))
                print("\n[more matches omitted; use a narrower query]")
                return
            blocks.append(block)
    if not blocks:
        raise KeyError(f"no contract/preflight lines match {query!r}")
    print("\n".join(blocks))


def main() -> int:
    global QUERY_COMMAND
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/input"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    feedback = subparsers.add_parser("feedback")
    feedback.add_argument("--path")
    subparsers.add_parser("feedback-frontier")
    category = subparsers.add_parser("feedback-category")
    category.add_argument("name")
    source = subparsers.add_parser("source")
    source.add_argument("--start", type=int, default=1)
    source.add_argument("--lines", type=int, default=160)
    source_ttir = subparsers.add_parser("source-ttir")
    source_ttir.add_argument("--start", type=int, default=1)
    source_ttir.add_argument("--lines", type=int, default=160)
    source_ttir_find = subparsers.add_parser("source-ttir-find")
    source_ttir_find.add_argument("query")
    source_ttir_find.add_argument("--context", type=int, default=3)
    source_ttir_def_use = subparsers.add_parser("source-ttir-def-use")
    source_ttir_def_use.add_argument("value")
    source_ttir_def_use.add_argument("--context", type=int, default=2)
    source_ttir_outline = subparsers.add_parser("source-ttir-outline")
    source_ttir_outline.add_argument("--start", type=int, default=1)
    source_ttir_outline.add_argument("--count", type=int, default=80)
    source_observer = subparsers.add_parser("source-observer")
    source_observer.add_argument("--path")
    subparsers.add_parser("realization-ledger")
    source_find = subparsers.add_parser("source-find")
    source_find.add_argument("query")
    source_find.add_argument("--context", type=int, default=3)
    source_outline = subparsers.add_parser("source-outline")
    source_outline.add_argument("--start", type=int, default=1)
    source_outline.add_argument("--count", type=int, default=32)
    subparsers.add_parser("source-brief")
    semantic_transfer = subparsers.add_parser("semantic-transfer")
    semantic_transfer.add_argument("--start", type=int, default=1)
    semantic_transfer.add_argument("--count", type=int, default=32)
    subparsers.add_parser("canonical-wire-check")
    source_obligations = subparsers.add_parser("source-obligations")
    source_obligations.add_argument("--section")
    source_obligations.add_argument("--start", type=int, default=1)
    source_obligations.add_argument("--count", type=int, default=32)
    harness = subparsers.add_parser("harness")
    harness.add_argument("--path")
    harness_abi = subparsers.add_parser("harness-abi")
    harness_abi.add_argument("--start", type=int, default=1)
    harness_abi.add_argument("--count", type=int, default=24)
    subparsers.add_parser("harness-c-contract")
    knowledge = subparsers.add_parser("knowledge")
    knowledge.add_argument("query")
    knowledge.add_argument("--context", type=int, default=3)
    args = parser.parse_args()
    if args.input_root != Path("/input"):
        QUERY_COMMAND = " ".join(
            (
                "python",
                shlex.quote(sys.argv[0]),
                "--input-root",
                shlex.quote(str(args.input_root)),
            )
        )

    if args.command == "feedback":
        _feedback(args.input_root, args.path)
    elif args.command == "feedback-frontier":
        _feedback_frontier(args.input_root)
    elif args.command == "feedback-category":
        _feedback_category(args.input_root, args.name)
    elif args.command == "source":
        _source(args.input_root, args.start, args.lines)
    elif args.command == "source-ttir":
        _source_ttir(args.input_root, args.start, args.lines)
    elif args.command == "source-ttir-find":
        _source_ttir_find(args.input_root, args.query, args.context)
    elif args.command == "source-ttir-def-use":
        _source_ttir_find(
            args.input_root, args.value, args.context, exact_token=True
        )
    elif args.command == "source-ttir-outline":
        _source_ttir_outline(args.input_root, args.start, args.count)
    elif args.command == "source-observer":
        _source_observer(args.input_root, args.path)
    elif args.command == "source-find":
        _source_find(args.input_root, args.query, args.context)
    elif args.command == "source-outline":
        _source_outline(args.input_root, args.start, args.count)
    elif args.command == "source-brief":
        _source_brief(args.input_root)
    elif args.command == "semantic-transfer":
        _semantic_transfer(args.input_root, args.start, args.count)
    elif args.command == "canonical-wire-check":
        _canonical_wire_check(args.input_root)
    elif args.command == "source-obligations":
        _source_obligations(
            args.input_root,
            args.section,
            args.start,
            args.count,
        )
    elif args.command == "harness":
        _harness(args.input_root, args.path)
    elif args.command == "harness-abi":
        _harness_abi(args.input_root, args.start, args.count)
    elif args.command == "harness-c-contract":
        _harness_c_contract(args.input_root)
    elif args.command == "realization-ledger":
        _realization_ledger(args.input_root)
    elif args.command == "knowledge":
        _knowledge(args.input_root, args.query, args.context)
    else:
        raise AssertionError(f"unhandled context query command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
