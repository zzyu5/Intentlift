"""Bounded MCP resources and tools for one source-aware IntentIR Lift owner.

The Responses provider cannot attach a local stdio MCP server directly.  This
module therefore keeps the standard MCP resource/tool request shape at the
agent boundary and lets the isolated Lift environment execute it locally. MCP
fills only a schema-derived semantic J buffer; the existing Source/Harness
constructor derives mechanical J/C/pi fields at finalize time, and acceptance
remains fail-closed in CandidateCheck/Gate1.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import sys
from typing import Any, Callable, Mapping, Sequence

from intentir.extraction.lifters.scrc_schema import (
    canonical_semantic_mcp_write_protocol,
)
from intentir.ir.intent_op_contract import CANONICAL_J_OPS


MCP_PROTOCOL_VERSION = "2025-06-18"
MAX_MCP_QUERY_CHARS = 512
MAX_MCP_PATH_CHARS = 512
MCP_SELECTOR_CHARS = 64
MCP_SCHEMA_LOOKUP_ITEM_CHARS = 128
# A fresh semantic entry gets exactly one schema-lookup *call*.  That call
# may carry the small, Source-declared set of otherwise unavailable operation
# leaves; treating this as a small arbitrary item limit made a source with
# several independent operation families choose which required contract to
# omit. The canonical operation vocabulary itself is the finite upper bound.
MAX_MCP_SCHEMA_LOOKUP_ITEMS = len(CANONICAL_J_OPS)
MAX_MCP_DECLARATION_CHARS = 64 * 1024
MAX_MCP_RESULT_BYTES = 1024 * 1024
# The MCP server listing and the provider-facing Responses function schema
# must project the same argument objects.  The dispatcher still applies the
# URI-dependent limits below, but field ownership and shape have one source.
MCP_RESOURCE_READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "uri": {"type": "string", "minLength": 1, "maxLength": MAX_MCP_PATH_CHARS},
        "query": {"type": "string", "maxLength": MAX_MCP_QUERY_CHARS},
        "path": {"type": "string", "maxLength": MAX_MCP_PATH_CHARS},
        "start": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
        "count": {"type": "integer", "minimum": 1, "maximum": 500},
        "context": {"type": "integer", "minimum": 0, "maximum": 16},
        "section": {"type": "string", "maxLength": MCP_SELECTOR_CHARS},
    },
    "required": ["uri"],
}

def _semantic_write_tool_schemas() -> dict[str, dict[str, Any]]:
    """Project MCP write arguments from the canonical semantic schema.

    A declaration is transported as one bounded JSON text value.  Keeping the
    payload opaque to the Responses function schema avoids requiring provider
    support for the artifact's deeply nested conditional schema; the local
    MCP environment parses it immediately and only the canonical constructor
    accepts the assembled semantic candidate.
    """

    protocol = canonical_semantic_mcp_write_protocol()
    writes = protocol.get("writes") if isinstance(protocol, Mapping) else None
    steps = protocol.get("steps") if isinstance(protocol, Mapping) else None
    argument_rows = (
        protocol.get("tool_arguments") if isinstance(protocol, Mapping) else None
    )
    if (
        not isinstance(writes, Mapping)
        or not isinstance(steps, list)
        or not isinstance(argument_rows, Mapping)
        or not all(isinstance(step, str) for step in steps)
        or set(argument_rows) != set(steps)
    ):
        raise TypeError("canonical MCP write argument projection is malformed")
    json_text = {
        "type": "string",
        "minLength": 1,
        "maxLength": MAX_MCP_DECLARATION_CHARS,
    }
    no_arguments = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }
    result: dict[str, dict[str, Any]] = {}
    for step in steps:
        row = argument_rows[step]
        if row is None:
            result[step] = copy.deepcopy(no_arguments)
            continue
        if (
            not isinstance(row, Mapping)
            or set(row) not in (
                {"field", "json_container"},
                {"field", "json_container", "pattern"},
            )
            or not isinstance(row.get("field"), str)
            or not isinstance(row.get("json_container"), str)
            or row["json_container"] not in {"array", "object"}
            or "pattern" in row
            and (
                not isinstance(row.get("pattern"), str)
                or row["json_container"] != "array"
                or not row["pattern"]
            )
        ):
            raise TypeError("canonical MCP write argument row is malformed")
        field = row["field"]
        field_schema = copy.deepcopy(json_text)
        if isinstance(row.get("pattern"), str):
            field_schema["pattern"] = row["pattern"]
        result[step] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {field: field_schema},
            "required": [field],
        }
    return result


MCP_TOOL_ARGUMENT_SCHEMAS: dict[str, dict[str, Any]] = {
    "schema_lookup": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_MCP_SCHEMA_LOOKUP_ITEMS,
                "items": {
                    "type": "string",
                    "maxLength": MCP_SCHEMA_LOOKUP_ITEM_CHARS,
                },
            }
        },
        "required": ["items"],
    },
    **_semantic_write_tool_schemas(),
}
MCP_SEMANTIC_WRITE_TOOL_NAMES = tuple(
    name for name in MCP_TOOL_ARGUMENT_SCHEMAS if name != "schema_lookup"
)

_SCHEMA_LOOKUP_TOOL_DESCRIPTION = (
    "Read a bounded semantic-authoring schema leaf; the host constructs "
    "mechanical artifact fields."
)


def _semantic_write_tool_descriptions() -> dict[str, str]:
    """Project write tool text from the canonical semantic authority."""

    protocol = canonical_semantic_mcp_write_protocol()
    steps = protocol.get("steps") if isinstance(protocol, Mapping) else None
    descriptions = protocol.get("descriptions") if isinstance(protocol, Mapping) else None
    if (
        not isinstance(steps, list)
        or tuple(steps) != MCP_SEMANTIC_WRITE_TOOL_NAMES
        or not isinstance(descriptions, Mapping)
        or set(descriptions) != set(steps)
        or any(not isinstance(item, str) or not item for item in descriptions.values())
    ):
        raise TypeError("canonical MCP write descriptions are malformed")
    return {step: str(descriptions[step]) for step in steps}


# Both the MCP tools/list response and the provider-facing Responses gateway
# project this exact catalogue.  Keeping descriptions beside the shared input
# schemas prevents the direct Responses path from exposing a bare argument
# name while the stdio MCP path carries the actual construction contract.
MCP_TOOL_DESCRIPTIONS: dict[str, str] = {
    "schema_lookup": _SCHEMA_LOOKUP_TOOL_DESCRIPTION,
    **_semantic_write_tool_descriptions(),
}
if set(MCP_TOOL_DESCRIPTIONS) != set(MCP_TOOL_ARGUMENT_SCHEMAS):
    raise TypeError("canonical MCP tool descriptions do not match tool schemas")


class MCPRequestError(ValueError):
    """A malformed or unavailable MCP request from the Lift owner."""


@dataclass(frozen=True)
class MCPResource:
    uri: str
    name: str
    description: str
    view: str
    query_required: bool = False


_RESOURCES: tuple[MCPResource, ...] = (
    MCPResource(
        "intentir://lift/source/brief",
        "Source brief",
        "Compact Source overview plus the canonical semantic authoring entry; host mechanics are not provider-authored.",
        "source_brief",
    ),
    MCPResource(
        "intentir://lift/source/obligations",
        "Source obligations",
        "Bounded Source transfer/control/mask/fill ledger; page a named section only when needed.",
        "source_obligations",
    ),
    MCPResource(
        "intentir://lift/source/transfer-ledger",
        "Semantic transfer ledger",
        "Bounded Source-order semantic access/output ledger; follow next_start until null before construction. count 1..80 is a page upper bound.",
        "semantic_transfer",
    ),
    MCPResource(
        "intentir://lift/source/text",
        "Source text",
        "Paged exact selected-frontend Source text.",
        "source",
    ),
    MCPResource(
        "intentir://lift/source/outline",
        "Source outline",
        "Paged syntax-only Source outline for control-flow navigation.",
        "source_outline",
    ),
    MCPResource(
        "intentir://lift/source/search",
        "Source search",
        "Literal bounded Source search; query is required.",
        "source_find",
        query_required=True,
    ),
    MCPResource(
        "intentir://lift/compiler/ttir",
        "Source TTIR",
        "Paged exact Triton/TTIR compiler observation when this frontend exposes it.",
        "source_ttir",
    ),
    MCPResource(
        "intentir://lift/compiler/ttir-outline",
        "Source TTIR outline",
        "Bounded TTIR outline for navigation only.",
        "source_ttir_outline",
    ),
    MCPResource(
        "intentir://lift/compiler/ttir-search",
        "Source TTIR search",
        "Literal bounded TTIR search; query is required.",
        "source_ttir_find",
        query_required=True,
    ),
    MCPResource(
        "intentir://lift/compiler/ttir-def-use",
        "Source TTIR def-use",
        "Bounded compiler def-use observation; query is required.",
        "source_ttir_def_use",
        query_required=True,
    ),
    MCPResource(
        "intentir://lift/compiler/observer",
        "Compiler observer",
        "One bounded exact selected-frontend compiler observer path.",
        "source_observer",
    ),
    MCPResource(
        "intentir://lift/compiler/realization-ledger",
        "Realization ledger",
        "Read-only compiler facts that may ground an existing J/C anchor in pi.",
        "realization_ledger",
    ),
    MCPResource(
        "intentir://lift/harness/abi",
        "Harness ABI",
        "Compact trusted p*/c*/shape-coordinate and selected-frontend C ledger.",
        "harness_abi",
    ),
    MCPResource(
        "intentir://lift/harness/c-contract",
        "Harness C contract",
        "Exact selected-frontend C slot order, declarations, and trusted values; no J or p* projection.",
        "harness_c_contract",
    ),
    MCPResource(
        "intentir://lift/harness/view",
        "Harness view",
        "Bounded exact selected-frontend Harness path.",
        "harness",
    ),
    MCPResource(
        "intentir://lift/feedback/frontier",
        "Feedback frontier",
        "Bounded rule-level diagnostic frontier for this fresh semantic entry.",
        "feedback_frontier",
    ),
    MCPResource(
        "intentir://lift/feedback/detail",
        "Feedback detail",
        "One bounded exact closed diagnostic path.",
        "feedback",
    ),
    MCPResource(
        "intentir://lift/feedback/category",
        "Feedback category",
        "One named Gate diagnostic category; query is required.",
        "feedback_category",
        query_required=True,
    ),
    MCPResource(
        "intentir://lift/knowledge/search",
        "Contract search",
        "Bounded source-neutral contract/preflight lookup; query is required.",
        "knowledge",
        query_required=True,
    ),
)

MCP_RESOURCE_URIS = tuple(resource.uri for resource in _RESOURCES)
MCP_TOOL_NAMES = tuple(MCP_TOOL_ARGUMENT_SCHEMAS)

_RESOURCE_BY_URI = {resource.uri: resource for resource in _RESOURCES}
MCP_RESOURCE_ARGUMENTS = frozenset(
    {"uri", "query", "path", "start", "count", "context", "section"}
)
MCP_CONTEXT_ARGUMENTS = frozenset(
    (MCP_RESOURCE_ARGUMENTS - {"uri"}) | {"view"}
)
MCP_RESOURCE_VIEWS = frozenset(resource.view for resource in _RESOURCES)
# These views are summary-shaped and therefore share the public 1..500 page
# bound.  The mounted context reader may apply a narrower view-specific bound;
# it must not invent a wider protocol shape.
MCP_FIXED_SUMMARY_VIEWS = frozenset(
    {
        "source_brief",
        "feedback_frontier",
        "harness",
        "harness_c_contract",
        "source",
        "source_outline",
    }
)
MCP_CONTEXT_VIEWS = MCP_RESOURCE_VIEWS


def mcp_resource_uris_for_views(views: Sequence[str]) -> tuple[str, ...]:
    """Project one validated resource-view allowlist onto MCP URIs.

    The environment owns the lifecycle's allowed fact families; this catalogue
    owns URI-to-view association.  Deriving ``resources/list`` from that same
    view set prevents an advertised resource from drifting away from the
    dispatcher admission boundary.
    """

    if isinstance(views, (str, bytes)):
        raise TypeError("MCP resource views must be a sequence")
    selected = tuple(views)
    if (
        any(
            not isinstance(view, str) or view not in MCP_RESOURCE_VIEWS
            for view in selected
        )
        or len(selected) != len(set(selected))
    ):
        raise ValueError("MCP resource view allowlist is invalid")
    return tuple(resource.uri for resource in _RESOURCES if resource.view in selected)


def mcp_resource_count_upper(view: str) -> int:
    """Return the single public page upper bound for one MCP resource view."""

    if view == "semantic_transfer":
        return 80
    return 500 if view in MCP_FIXED_SUMMARY_VIEWS else 200


def mcp_resource_count_upper_for_uri(uri: str) -> int:
    """Return the same public page bound from a canonical resource URI."""

    resource = _RESOURCE_BY_URI.get(uri)
    if resource is None:
        raise ValueError("MCP resource URI is unavailable")
    return mcp_resource_count_upper(resource.view)


# Private aliases remain source-compatible for local callers, but all new
# protocol validation must use the exported declarations above.
_RESOURCE_ARGUMENTS = MCP_RESOURCE_ARGUMENTS


class IntentIRLiftMCPServer:
    """Route a small MCP surface to an isolated environment callback.

    ``dispatch`` receives an internal, already allowlisted action and returns a
    normal environment execution result.  The server has no provider or
    acceptance authority of its own. Its one constructor-like resource is the
    read-only mechanical J framework; it does not author unresolved semantic
    fields or repair/accept candidates.
    """

    def __init__(
        self,
        dispatch: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        *,
        active_resource_uris: Callable[[], Sequence[str]] | None = None,
    ):
        self._dispatch = dispatch
        self._active_resource_uris = active_resource_uris

    @staticmethod
    def server_info() -> dict[str, str]:
        return {"name": "intentir-lift", "version": "stable"}

    def resources(self) -> list[dict[str, str]]:
        """List only resources admitted by this server owner, when supplied.

        The global catalogue is intentionally retained for multi-frontend and
        extension owners.  A semantic Lift worker must not discover inactive
        compiler/Harness/candidate views and then receive a contradictory
        dispatcher rejection.
        """

        active = (
            frozenset(self._active_resource_uris())
            if self._active_resource_uris is not None
            else None
        )
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": "application/json",
            }
            for resource in _RESOURCES
            if active is None or resource.uri in active
        ]

    @staticmethod
    def tools() -> list[dict[str, Any]]:
        schema_tool = [
            {
                "name": "schema_lookup",
                "description": MCP_TOOL_DESCRIPTIONS["schema_lookup"],
                "inputSchema": copy.deepcopy(
                    MCP_TOOL_ARGUMENT_SCHEMAS["schema_lookup"]
                ),
            }
        ]
        return [
            *schema_tool,
            *[
                {
                    "name": name,
                    "description": MCP_TOOL_DESCRIPTIONS[name],
                    "inputSchema": copy.deepcopy(MCP_TOOL_ARGUMENT_SCHEMAS[name]),
                }
                for name in MCP_SEMANTIC_WRITE_TOOL_NAMES
            ],
        ]

    @staticmethod
    def _resource_action(params: Mapping[str, Any]) -> dict[str, Any]:
        if set(params) - _RESOURCE_ARGUMENTS:
            raise MCPRequestError("unknown resources/read argument")
        uri = params.get("uri")
        if not isinstance(uri, str) or uri not in _RESOURCE_BY_URI:
            raise MCPRequestError("resource URI is unavailable")
        for name in ("query", "path"):
            value = params.get(name)
            if value is not None and (
                not isinstance(value, str)
                or len(value) > (
                    MAX_MCP_QUERY_CHARS if name == "query" else MAX_MCP_PATH_CHARS
                )
            ):
                raise MCPRequestError(f"{name} must be a bounded string")
        resource = _RESOURCE_BY_URI[uri]
        # Fixed summary resources do not page their payload; accepting the
        # bounded optional count keeps the public resource wire contract
        # consistent with the mounted context reader. Variable views retain
        # their view-specific limits below.
        count_upper = mcp_resource_count_upper(resource.view)
        for name, lower, upper in (
            ("start", 1, 1_000_000),
            ("count", 1, count_upper),
            ("context", 0, 16),
        ):
            value = params.get(name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not lower <= value <= upper
            ):
                raise MCPRequestError(f"{name} must be in [{lower}, {upper}]")
        for name in ("section",):
            value = params.get(name)
            if value is not None and (
                not isinstance(value, str) or len(value) > 64
            ):
                raise MCPRequestError(f"{name} must be a bounded string")
        query = params.get("query")
        if resource.query_required and (
            not isinstance(query, str)
            or not query
            or len(query) > MAX_MCP_QUERY_CHARS
        ):
            raise MCPRequestError(f"resource {uri} requires a bounded query")
        return {
            "kind": "resource",
            "uri": uri,
            "view": resource.view,
            **{
                key: value
                for key, value in params.items()
                if key != "uri" and value is not None
            },
        }

    @staticmethod
    def _tool_action(params: Mapping[str, Any]) -> dict[str, Any]:
        if set(params) != {"name", "arguments"}:
            raise MCPRequestError("tools/call requires exactly name and arguments")
        name = params.get("name")
        arguments = params.get("arguments")
        if name not in MCP_TOOL_NAMES or not isinstance(arguments, Mapping):
            raise MCPRequestError("MCP tool name or arguments are invalid")
        arguments = dict(arguments)
        if name == "schema_lookup":
            items = arguments.get("items")
            if (
                set(arguments) != {"items"}
                or not isinstance(items, list)
                or not 1 <= len(items) <= MAX_MCP_SCHEMA_LOOKUP_ITEMS
                or len(items) != len(set(items))
                or any(
                    not isinstance(item, str)
                    or len(item) > MCP_SCHEMA_LOOKUP_ITEM_CHARS
                    for item in items
                )
            ):
                raise MCPRequestError(
                    "schema_lookup requires one bounded, distinct op:NAME selector set"
                )
        elif name in MCP_SEMANTIC_WRITE_TOOL_NAMES:
            schema = MCP_TOOL_ARGUMENT_SCHEMAS[str(name)]
            required = schema.get("required")
            if required is None:
                required = []
            if not isinstance(required, list):
                raise RuntimeError("canonical MCP write schema is malformed")
            if not required:
                if arguments:
                    raise MCPRequestError(f"{name} requires exactly no arguments")
                return {"kind": "tool", "name": str(name), "arguments": arguments}
            if len(required) != 1 or not isinstance(required[0], str):
                raise RuntimeError("canonical MCP declaration schema is malformed")
            field = required[0]
            value = arguments.get(field)
            if (
                set(arguments) != {field}
                or not isinstance(value, str)
                or not value
                or len(value) > MAX_MCP_DECLARATION_CHARS
            ):
                raise MCPRequestError(f"{name} requires exactly one bounded {field}")
        return {"kind": "tool", "name": str(name), "arguments": arguments}

    def dispatch(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(method, str) or not isinstance(params, Mapping):
            raise MCPRequestError("MCP method and params must be objects")
        if method == "resources/read":
            return self._dispatch(self._resource_action(params))
        if method == "tools/call":
            return self._dispatch(self._tool_action(params))
        raise MCPRequestError("unsupported MCP dispatch method")

    @staticmethod
    def _text_result(result: Mapping[str, Any]) -> tuple[str, bool]:
        text = result.get("output", "")
        if not isinstance(text, str):
            text = json.dumps(dict(result), sort_keys=True, ensure_ascii=False)
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_MCP_RESULT_BYTES:
            text = (
                encoded[:MAX_MCP_RESULT_BYTES]
                .decode("utf-8", errors="ignore")
                + "\n[mcp result truncated at byte limit]"
            )
        return text, result.get("returncode") not in {0, None}

    def handle(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(method, str) or not isinstance(params, Mapping):
            raise MCPRequestError("MCP method and params must be objects")
        if method == "initialize":
            return {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": self.server_info(),
                "capabilities": {"resources": {}, "tools": {}},
            }
        if method == "resources/list":
            return {"resources": self.resources()}
        if method == "tools/list":
            return {"tools": self.tools()}
        if method == "resources/read":
            text, is_error = self._text_result(self.dispatch(method, params))
            return {
                "contents": [
                    {
                        "uri": str(params.get("uri", "")),
                        "mimeType": "application/json",
                        "text": text,
                    }
                ],
                "isError": is_error,
            }
        if method == "tools/call":
            text, is_error = self._text_result(self.dispatch(method, params))
            return {"content": [{"type": "text", "text": text}], "isError": is_error}
        raise MCPRequestError("unsupported MCP method")

    def handle_jsonrpc(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Handle one newline-delimited JSON-RPC MCP message.

        Notifications intentionally return ``None``.  The transport is useful
        for a host-side MCP client, while the active Responses path calls the
        same ``handle`` method through the in-process bridge.
        """

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str) or not isinstance(params, Mapping):
            error = {
                "code": -32600,
                "message": "invalid JSON-RPC MCP request",
            }
        else:
            try:
                result = self.handle(method, params)
            except MCPRequestError as exc:
                error = {"code": -32602, "message": str(exc)}
            except Exception as exc:  # pragma: no cover - transport boundary
                error = {"code": -32000, "message": f"{type(exc).__name__}: {exc}"}
            else:
                if request_id is None:
                    return None
                return {"jsonrpc": "2.0", "id": request_id, "result": result}
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "error": error}


def serve_stdio(server: IntentIRLiftMCPServer) -> None:
    """Serve newline-delimited MCP JSON-RPC on stdin/stdout.

    This deliberately accepts a preconstructed server so callers must provide
    the already-authorized bounded dispatcher; the module never discovers host
    paths or creates an unguarded candidate writer.
    """

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise MCPRequestError("JSON-RPC request must be an object")
            response = server.handle_jsonrpc(request)
        except Exception as exc:  # pragma: no cover - transport boundary
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
        if response is not None:
            sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
            sys.stdout.flush()


def validate_mcp_gateway_call(
    value: Mapping[str, Any], *, first_read_only: bool = False
) -> None:
    """Validate one provider-facing gateway payload with server rules.

    Responses parsing and the isolated MCP environment must reject the same
    method-specific shape.  This helper deliberately delegates to the exact
    resource/tool argument validators used by :class:`IntentIRLiftMCPServer`;
    it does not execute a resource read, write a candidate, or accept an
    artifact.
    """

    if (
        not isinstance(value, Mapping)
        or set(value) != {"method", "params"}
        or value.get("method") not in {"resources/read", "tools/call"}
        or not isinstance(value.get("params"), Mapping)
    ):
        raise MCPRequestError(
            "mcp_call_tool requires exactly method and object params; method "
            "must be resources/read or tools/call"
        )
    method = value["method"]
    if first_read_only and method != "resources/read":
        raise MCPRequestError("the first Lift turn permits resources/read only")
    if method == "resources/read":
        IntentIRLiftMCPServer._resource_action(dict(value["params"]))
    else:
        IntentIRLiftMCPServer._tool_action(dict(value["params"]))


__all__ = [
    "IntentIRLiftMCPServer",
    "MCP_PROTOCOL_VERSION",
    "MCPRequestError",
    "MCP_RESOURCE_READ_SCHEMA",
    "MCP_SELECTOR_CHARS",
    "MCP_SCHEMA_LOOKUP_ITEM_CHARS",
    "MCP_RESOURCE_ARGUMENTS",
    "MCP_CONTEXT_ARGUMENTS",
    "MCP_RESOURCE_VIEWS",
    "MCP_FIXED_SUMMARY_VIEWS",
    "MCP_CONTEXT_VIEWS",
    "mcp_resource_uris_for_views",
    "mcp_resource_count_upper",
    "mcp_resource_count_upper_for_uri",
    "MCP_RESOURCE_URIS",
    "MCP_TOOL_ARGUMENT_SCHEMAS",
    "MCP_TOOL_DESCRIPTIONS",
    "MCP_TOOL_NAMES",
    "MCP_SEMANTIC_WRITE_TOOL_NAMES",
    "validate_mcp_gateway_call",
    "serve_stdio",
]
