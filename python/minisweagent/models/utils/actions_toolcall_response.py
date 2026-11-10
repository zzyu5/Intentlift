"""Parse actions & format observations for OpenAI Responses API toolcalls"""

import copy
import json
import time
from collections.abc import Sequence

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.mcp_lift import (
    MCP_RESOURCE_READ_SCHEMA,
    MCP_TOOL_ARGUMENT_SCHEMAS,
    MCP_TOOL_DESCRIPTIONS,
    MCPRequestError,
    mcp_resource_count_upper_for_uri,
    validate_mcp_gateway_call,
)


MAX_BASH_TOOL_CHARS = 8192


_MCP_RESOURCE_READ_PARAMS = MCP_RESOURCE_READ_SCHEMA


def _mcp_tool_call_branch(name: str, arguments: dict) -> dict:
    """Bind one MCP tool name to its only valid argument object.

    A single unconstrained ``arguments.oneOf`` lets a provider combine the
    right tool name with the wrong argument shape.  The local MCP dispatcher
    rejects that combination, but discovering it only after a tool call wastes
    a model turn and makes the provider-facing contract differ from the
    server contract.  Keep the correlation in the function schema itself.
    """

    description = MCP_TOOL_DESCRIPTIONS.get(name)
    name_schema = {"type": "string", "enum": [name]}
    argument_schema = copy.deepcopy(arguments)
    if isinstance(description, str) and description:
        name_schema["description"] = description
        argument_schema["description"] = description
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": name_schema,
            "arguments": argument_schema,
        },
        "required": ["name", "arguments"],
    }


_MCP_GATEWAY_TOOL_PARAMS = {
    "type": "object",
    "oneOf": [
        _mcp_tool_call_branch(name, arguments)
        for name, arguments in MCP_TOOL_ARGUMENT_SCHEMAS.items()
    ],
}


_MCP_GATEWAY_PARAMS = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["resources/read"],
                },
                "params": _MCP_RESOURCE_READ_PARAMS,
            },
            "required": ["method", "params"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["tools/call"],
                },
                "params": _MCP_GATEWAY_TOOL_PARAMS,
            },
            "required": ["method", "params"],
        },
    ],
}


def mcp_call_tool_response_api_for_surface(
    *,
    resource_uris: Sequence[str] = (),
    tool_names: Sequence[str] = (),
    schema_lookup_items: Sequence[str] | None = None,
) -> dict:
    """Build the exact MCP gateway schema for one Lift transaction phase.

    The server has always validated the transaction after a call.  A static
    Responses schema nevertheless advertised every MCP resource and every
    construction write on every turn, so a model could select an action that
    was already impossible.  This projection is intentionally a restriction
    of the canonical gateway grammar: it changes neither argument shape nor
    acceptance rules, and exposes only actions the current phase can execute.
    """

    resources = tuple(resource_uris)
    names = tuple(tool_names)
    lookup_items = (
        tuple(schema_lookup_items) if schema_lookup_items is not None else None
    )
    if (
        not resources
        and not names
        or len(resources) != len(set(resources))
        or len(names) != len(set(names))
        or any(not isinstance(uri, str) or not uri for uri in resources)
        or any(name not in MCP_TOOL_ARGUMENT_SCHEMAS for name in names)
        or lookup_items is not None
        and (
            names != ("schema_lookup",)
            or not lookup_items
            or len(lookup_items) != len(set(lookup_items))
            or any(
                not isinstance(item, str)
                or not item.startswith("op:")
                or len(item) <= len("op:")
                for item in lookup_items
            )
        )
    ):
        raise ValueError("MCP phase surface must contain unique known actions")

    branches: list[dict] = []
    if resources:
        resource_params = copy.deepcopy(_MCP_RESOURCE_READ_PARAMS)
        properties = resource_params.get("properties")
        if not isinstance(properties, dict) or not isinstance(
            properties.get("uri"), dict
        ):
            raise RuntimeError("canonical MCP resource schema is malformed")
        uri_schema = dict(properties["uri"])
        uri_schema["enum"] = list(resources)
        properties["uri"] = uri_schema
        count_schema = properties.get("count")
        if isinstance(count_schema, dict):
            count_schema = dict(count_schema)
            count_schema["maximum"] = min(
                mcp_resource_count_upper_for_uri(uri) for uri in resources
            )
            properties["count"] = count_schema
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "method": {"type": "string", "enum": ["resources/read"]},
                    "params": resource_params,
                },
                "required": ["method", "params"],
            }
        )
    if names:
        tool_branches = []
        for name in names:
            arguments = copy.deepcopy(MCP_TOOL_ARGUMENT_SCHEMAS[name])
            if name == "schema_lookup" and lookup_items is not None:
                properties = arguments.get("properties")
                if not isinstance(properties, dict) or not isinstance(
                    properties.get("items"), dict
                ):
                    raise RuntimeError("canonical schema lookup arguments are malformed")
                item_array = dict(properties["items"])
                item_schema = item_array.get("items")
                if not isinstance(item_schema, dict):
                    raise RuntimeError("canonical schema lookup item schema is malformed")
                exact_item_schema = dict(item_schema)
                exact_item_schema["enum"] = list(lookup_items)
                item_array["minItems"] = len(lookup_items)
                item_array["maxItems"] = len(lookup_items)
                item_array["items"] = exact_item_schema
                item_array["prefixItems"] = [
                    {**dict(item_schema), "const": item}
                    for item in lookup_items
                ]
                properties["items"] = item_array
            tool_branches.append(_mcp_tool_call_branch(name, arguments))
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "method": {"type": "string", "enum": ["tools/call"]},
                    "params": {
                        "type": "object",
                        "oneOf": [
                            *tool_branches,
                        ],
                    },
                },
                "required": ["method", "params"],
            }
        )
    description = (
        "Use the single IntentIR Lift MCP gateway. This turn exposes only "
        "the current transaction phase actions; use exactly one listed "
        "resource or construction call."
    )
    if names:
        description += " Current construction contract: " + " ".join(
            MCP_TOOL_DESCRIPTIONS[name] for name in names
        )
    return {
        "type": "function",
        "name": "mcp_call_tool",
        "description": description,
        "parameters": {
            "type": "object",
            "oneOf": branches,
        },
    }

MCP_CALL_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "mcp_call_tool",
    "description": (
        "Use the single IntentIR Lift MCP gateway. Set method to resources/read "
        "for one bounded resource or tools/call for one server tool, and pass "
        "the exact MCP params object. For schema_lookup, arguments is exactly "
        "{items:[...]}; semantic J construction is only begin_candidate, the "
        "schema-derived declare_* calls, then finalize_candidate. The gateway "
        "reads bounded facts and constructs the one candidate buffer; it never "
        "invents unresolved semantics or accepts a candidate."
    ),
    "parameters": {
        **_MCP_GATEWAY_PARAMS,
    },
}

MCP_FIRST_READ_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "mcp_call_tool",
    "description": (
        "Required first Lift action: read exactly one bounded IntentIR Lift "
        "MCP resource. Use method=resources/read; this first action is read-only."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "method": {"type": "string", "enum": ["resources/read"]},
            "params": _MCP_RESOURCE_READ_PARAMS,
        },
        "required": ["method", "params"],
    },
}

# OpenRouter/OpenAI Responses API uses a flat structure (no nested "function" key)
BASH_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "bash",
    "description": "Execute a bash command",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "maxLength": MAX_BASH_TOOL_CHARS,
                "description": "The bash command to execute",
            }
        },
        "required": ["command"],
    },
}

APPLY_REPLAY_PATCH_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "apply_replay_patch",
    "description": (
        "Atomically replace one exact text span in the existing Replay source. "
        "The caller supplies both old and new text; the tool never infers code."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {
                "type": "string",
                "enum": ["/output/replay.source"],
                "description": "The existing Replay source to patch",
            },
            "old_text": {
                "type": "string",
                "description": "A non-empty exact span that must occur once",
            },
            "new_text": {
                "type": "string",
                "description": "The exact replacement text authored by the model",
            },
            "submit": {
                "type": "boolean",
                "description": (
                    "If true, run the existing Replay submission preflight "
                    "immediately after the exact replacement"
                ),
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
}

SUBMIT_REPLAY_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "submit_replay",
    "description": (
        "Atomically write the complete Replay source authored in this call and "
        "run the existing Replay submission preflight. The tool never edits code."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_text": {
                "type": "string",
                "description": "The complete exact replay.source text",
            },
        },
        "required": ["source_text"],
    },
}


def _format_error_message(error_text: str) -> dict:
    """Create a FormatError message in Responses API format."""
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": error_text}],
        "extra": {"interrupt_type": "FormatError"},
    }


def _get(obj, key):
    """Read ``key`` from an object or dict (Responses API responses come as either)."""
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def finish_reason_from_responses_api(response) -> str | None:
    """Map a Responses API response to a ``finish_reason``-like string for ``format_error_template``.

    The Responses API reports a ``max_tokens`` truncation as ``status="incomplete"`` with
    ``incomplete_details.reason="max_output_tokens"``; map that to ``"length"`` so the same
    ``finish_reason``-based templates work as for chat completions. Otherwise returns the raw status.
    """
    status = _get(response, "status")
    if status != "incomplete":
        return status
    return "length" if _get(_get(response, "incomplete_details"), "reason") == "max_output_tokens" else status


def parse_toolcall_actions_response(
    output: list,
    *,
    format_error_template: str,
    template_kwargs: dict | None = None,
    max_tool_calls: int = 1,
    enable_bash_tool: bool = True,
    enable_lift_mcp_tool: bool = False,
    mcp_first_read_only: bool = False,
    enable_replay_patch_tool: bool = False,
    enable_submit_replay_tool: bool = False,
) -> list[dict]:
    """Parse tool calls from a Responses API response output.

    Filters for function_call items and parses them.
    Response API format has name/arguments at top level with call_id:
    {"type": "function_call", "call_id": "...", "name": "bash", "arguments": "..."}

    ``template_kwargs`` are extra variables exposed to ``format_error_template`` (e.g.
    ``{"finish_reason": ...}``), matching ``parse_toolcall_actions``.
    """
    template_kwargs = template_kwargs or {}
    if (
        not isinstance(max_tool_calls, int)
        or isinstance(max_tool_calls, bool)
        or max_tool_calls < 1
    ):
        raise ValueError("max_tool_calls must be a positive integer")
    tool_calls = []
    for item in output:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type == "function_call":
            tool_calls.append(
                item.model_dump() if hasattr(item, "model_dump") else dict(item) if not isinstance(item, dict) else item
            )
    if not tool_calls:
        error_text = Template(format_error_template, undefined=StrictUndefined).render(
            error="No tool calls found in the response. Every response MUST include at least one tool call.",
            actions=[],
            has_tool_calls=False,
            **template_kwargs,
        )
        raise FormatError(_format_error_message(error_text))
    if len(tool_calls) > max_tool_calls:
        error_text = Template(format_error_template, undefined=StrictUndefined).render(
            error=(
                "IntentIR permits at most "
                f"{max_tool_calls} tool call(s) per model turn; "
                f"received {len(tool_calls)}."
            ),
            actions=[],
            has_tool_calls=True,
            **template_kwargs,
        )
        raise FormatError(_format_error_message(error_text))
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.get("arguments", "{}"))
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        tool_name = tool_call.get("name")
        if tool_name == "bash" and enable_bash_tool:
            if (
                not isinstance(args, dict)
                or "command" not in args
                or not isinstance(args.get("command"), str)
                or len(args.get("command", "")) > MAX_BASH_TOOL_CHARS
            ):
                error_msg += "Missing or non-string 'command' argument in bash tool call."
        elif tool_name == "mcp_call_tool" and enable_lift_mcp_tool:
            try:
                validate_mcp_gateway_call(
                    args,
                    first_read_only=mcp_first_read_only,
                )
            except (MCPRequestError, TypeError, ValueError) as exc:
                error_msg += str(exc)
        elif tool_name == "apply_replay_patch" and enable_replay_patch_tool:
            if (
                not isinstance(args, dict)
                or not {"path", "old_text", "new_text"}.issubset(args)
                or not set(args).issubset(
                    {"path", "old_text", "new_text", "submit"}
                )
                or not all(
                    isinstance(args.get(key), str)
                    for key in ("path", "old_text", "new_text")
                )
                or (
                    "submit" in args and not isinstance(args.get("submit"), bool)
                )
            ):
                error_msg += (
                    "apply_replay_patch requires string path, old_text, new_text, "
                    "and an optional boolean submit."
                )
        elif tool_name == "submit_replay" and enable_submit_replay_tool:
            if (
                not isinstance(args, dict)
                or set(args) != {"source_text"}
                or not isinstance(args.get("source_text"), str)
            ):
                error_msg += "submit_replay requires exactly source_text."
        else:
            error_msg += f"Unknown tool '{tool_name}'."
        if error_msg:
            error_text = Template(format_error_template, undefined=StrictUndefined).render(
                error=error_msg.strip(), actions=[], has_tool_calls=True, **template_kwargs
            )
            raise FormatError(_format_error_message(error_text))
        tool_call_id = tool_call.get("call_id") or tool_call.get("id")
        if tool_name == "bash":
            actions.append({"command": args["command"], "tool_call_id": tool_call_id})
        elif tool_name == "mcp_call_tool":
            actions.append(
                {
                    "tool_name": "mcp_call_tool",
                    "method": args["method"],
                    "params": dict(args["params"]),
                    "tool_call_id": tool_call_id,
                }
            )
        elif tool_name == "apply_replay_patch":
            actions.append(
                {
                    "tool_name": "apply_replay_patch",
                    "path": args["path"],
                    "old_text": args["old_text"],
                    "new_text": args["new_text"],
                    "submit": args.get("submit", False),
                    "tool_call_id": tool_call_id,
                }
            )
        else:
            actions.append(
                {
                    "tool_name": "submit_replay",
                    "source_text": args["source_text"],
                    "tool_call_id": tool_call_id,
                }
            )
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into function_call_output messages for Responses API."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg: dict = {
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["type"] = "function_call_output"
            msg["call_id"] = action["tool_call_id"]
            msg["output"] = content
        else:  # human issued commands
            msg["type"] = "message"
            msg["role"] = "user"
            msg["content"] = [{"type": "input_text", "text": content}]
        results.append(msg)
    return results
