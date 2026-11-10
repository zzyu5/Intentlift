"""Read-only direct-Agent snapshot projected from the MCP source authority.

The explicit direct-Agent route has no resource-read round trip.  It therefore
receives one snapshot of the same *provider-visible* Source/Harness facts that
the MCP semantic-entry resources expose.  This is an input projection only:
the provider still authors the semantic ``j``/``pi`` surface and the canonical
host constructor remains the sole producer of artifact schema, C, ABI,
handles, scopes, effects, and private owner receipts.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Mapping

from intentir.ir.canonical_construction import SOURCE_OWNER_TOKEN_PATTERN


DIRECT_AGENT_SNAPSHOT_VIEWS = (
    "source_brief",
    "semantic_transfer",
    "source",
    "source_ttir",
    "harness_abi",
    "harness_c_contract",
)


def _required_value(bundle: Mapping[str, Any], name: str) -> Any:
    value = bundle.get(name)
    if value is None:
        raise TypeError(f"canonical direct-Agent context lacks {name}")
    return value


def build_direct_agent_mcp_snapshot(
    *,
    source_wire: Mapping[str, Any],
    anonymous_source: str,
    compiler_view: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the direct counterpart to the MCP semantic-entry read surface.

    Every structured fact here is copied from ``canonical_source_wire_bundle``
    or the already-authoritative compiler view.  In particular, this function
    does not infer J operations, author a candidate field, rebuild an ABI, or
    expose the private transfer ledger.  The shape deliberately follows MCP
    resource names, so a direct author reads the same kinds of Source facts
    rather than a separately maintained prompt-only schema.
    """

    if not isinstance(source_wire, Mapping):
        raise TypeError("direct-Agent context requires canonical source wire")
    if not isinstance(anonymous_source, str) or not anonymous_source:
        raise ValueError("direct-Agent context requires anonymous Source text")
    artifact_seed = _required_value(source_wire, "artifact_seed")
    semantic_requirements = _required_value(
        source_wire, "semantic_construction_requirements"
    )
    transfer_projection = _required_value(
        source_wire, "semantic_transfer_projection"
    )
    output_construction = _required_value(source_wire, "output_construction")
    operand_inputs = _required_value(source_wire, "semantic_operand_inputs")
    domain_requirements = _required_value(
        source_wire, "program_domain_requirements"
    )
    aliases = _required_value(source_wire, "provider_dimension_aliases")
    construction_records = _required_value(source_wire, "construction_records")
    host_launch_contract = _required_value(source_wire, "host_launch_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (
            artifact_seed,
            semantic_requirements,
            transfer_projection,
            domain_requirements,
            construction_records,
            host_launch_contract,
        )
    ) or not isinstance(output_construction, list) or not isinstance(
        operand_inputs, list
    ) or not isinstance(aliases, list):
        raise TypeError("canonical direct-Agent context has an invalid source-wire shape")

    # ``semantic_transfer_projection`` is the provider-visible, sanitized
    # projection.  ``source_transfer_ledger`` may carry host-private receipts
    # and is intentionally excluded.  Check the assembled value as a final
    # boundary invariant before it leaves the host.
    views: dict[str, Any] = {
        "source_brief": {
            "artifact_seed": copy.deepcopy(dict(artifact_seed)),
            "semantic_operand_inputs": copy.deepcopy(operand_inputs),
            "program_domain_requirements": copy.deepcopy(dict(domain_requirements)),
            "output_construction": copy.deepcopy(output_construction),
            "provider_dimension_aliases": copy.deepcopy(aliases),
            "construction_records": copy.deepcopy(dict(construction_records)),
            "host_launch_contract": copy.deepcopy(dict(host_launch_contract)),
        },
        "semantic_transfer": {
            # The exact semantic construction requirements are already the
            # direct turn's ``semantic_authoring_entry``.  Keep this resource
            # focused on the MCP-only, ordered occurrence projection instead
            # of transmitting that same authority a second time.
            "semantic_construction_requirements_authority": (
                "semantic_authoring_entry.semantic_construction_requirements"
            ),
            "semantic_transfer_projection": copy.deepcopy(dict(transfer_projection)),
        },
        "source": {"text": anonymous_source},
        "harness_abi": {
            "artifact_seed": copy.deepcopy(dict(artifact_seed)),
            "semantic_operand_inputs": copy.deepcopy(operand_inputs),
            "program_domain_requirements": copy.deepcopy(dict(domain_requirements)),
        },
        "harness_c_contract": {
            "c": copy.deepcopy(artifact_seed.get("c")),
            "host_launch_contract": copy.deepcopy(dict(host_launch_contract)),
            "output_construction": copy.deepcopy(output_construction),
        },
    }
    if compiler_view is not None:
        if not isinstance(compiler_view, Mapping):
            raise TypeError("direct-Agent compiler view must be a mapping")
        views["source_ttir"] = copy.deepcopy(dict(compiler_view))
    else:
        # Preserve the resource's availability semantics without pretending
        # that an absent compiler observation is an empty observation.
        views["source_ttir"] = {"status": "unavailable"}

    snapshot = {
        "authority": "canonical_mcp_resource_view_snapshot",
        "read_only": True,
        "resource_views": views,
        "provider_may_author": ["j", "pi"],
        "host_constructs": [
            "schema",
            "c",
            "abi",
            "canonical_handles",
            "scopes",
            "effects",
            "owner_receipts",
        ],
    }
    try:
        encoded = json.dumps(
            snapshot, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TypeError("direct-Agent MCP snapshot is not JSON-safe") from exc
    if re.search(SOURCE_OWNER_TOKEN_PATTERN, encoded) is not None:
        raise ValueError("direct-Agent MCP snapshot leaked a private owner receipt")
    return snapshot


__all__ = ["DIRECT_AGENT_SNAPSHOT_VIEWS", "build_direct_agent_mcp_snapshot"]
