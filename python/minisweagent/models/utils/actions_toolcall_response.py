"""Parse actions & format observations for OpenAI Responses API toolcalls"""

import json
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError


MAX_BASH_TOOL_CHARS = 8192
MAX_EXACT_PATCH_TOOL_CHARS = 8192
MAX_CANDIDATE_TOOL_CHARS = 256 * 1024
MAX_WRITE_TEXT_TOOL_CHARS = 64 * 1024

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

APPLY_PATCH_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "apply_patch",
    "description": (
        "Atomically replace one exact text span in the existing candidate file. "
        "The caller supplies both old and new text; the tool never infers content."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {
                "type": "string",
                "enum": ["/output/candidate.json"],
                "description": "The existing Lift candidate to patch",
            },
            "old_text": {
                "type": "string",
                "maxLength": MAX_EXACT_PATCH_TOOL_CHARS,
                "description": "A non-empty exact span that must occur once",
            },
            "new_text": {
                "type": "string",
                "maxLength": MAX_EXACT_PATCH_TOOL_CHARS,
                "description": "The exact replacement text authored by the model",
            },
            "submit": {
                "type": "boolean",
                "description": (
                    "If true, run the existing submission preflight immediately "
                    "after applying the exact replacement"
                ),
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
}

SUBMIT_CANDIDATE_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "submit_candidate",
    "description": (
        "Atomically write the complete candidate text authored in this call and "
        "run the existing submission preflight. The tool never edits fields."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_text": {
                "type": "string",
                "maxLength": MAX_CANDIDATE_TOOL_CHARS,
                "description": "The complete exact candidate.json text",
            },
        },
        "required": ["candidate_text"],
    },
}

WRITE_TEXT_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "write_text",
    "description": (
        "Atomically overwrite or append exact model-authored text to one fixed "
        "Lift workspace file. The tool never interprets the text."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {
                "type": "string",
                "enum": [
                    "/output/candidate.json",
                    "/work/semantic_ledger.md",
                    "/work/construction_plan.md",
                    "/work/build.py",
                ],
            },
            "content": {
                "type": "string",
                "maxLength": MAX_WRITE_TEXT_TOOL_CHARS,
                "description": "The exact text bytes authored by the model",
            },
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
            },
            "submit": {
                "type": "boolean",
                "description": (
                    "Only for /output/candidate.json: run existing submission "
                    "preflight after the write"
                ),
            },
        },
        "required": ["path", "content", "mode"],
    },
}

SAVE_SEMANTIC_LEDGER_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "save_semantic_ledger",
    "description": (
        "Save one compact, proof-irrelevant semantic working ledger authored by "
        "the Lift owner. This is not J/C/pi and has no acceptance authority."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 8192,
                "description": "Compact source-grounded working notes",
            }
        },
        "required": ["content"],
    },
}

SAVE_CONSTRUCTION_PLAN_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "save_construction_plan",
    "description": (
        "Save one compact, proof-irrelevant construction checklist authored by "
        "the same Lift owner. It records the owner's intended J/C/pi topology "
        "and obligations but is not an artifact and has no acceptance authority."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 8192,
                "description": "Compact model-authored construction checklist",
            }
        },
        "required": ["content"],
    },
}

CHECK_SUBMIT_CANDIDATE_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "check_and_submit_candidate",
    "description": (
        "Explicitly run the existing CandidateCheck preflight on the current "
        "exact candidate bytes and submit only on PASS. It never edits the file."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    },
}

INSPECT_LIFT_CONTEXT_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "inspect_lift_context",
    "description": (
        "Read one bounded exact Source/value-free-Harness/feedback-category/"
        "current-candidate or source-neutral knowledge view without guessing filesystem "
        "paths. The tool is read-only and never classifies or repairs J/C/pi."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "view": {
                "type": "string",
                "enum": [
                    "source",
                    "source_outline",
                    "source_find",
                    "source_brief",
                    "harness",
                    "harness_abi",
                    "feedback",
                    "feedback_category",
                    "candidate",
                    "candidate_find",
                    "candidate_nodes",
                    "candidate_node",
                    "candidate_abi",
                    "candidate_control",
                    "candidate_diff",
                    "knowledge",
                ],
            },
            "query": {"type": "string", "maxLength": 512},
            "path": {"type": "string", "maxLength": 512},
            "start": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "count": {"type": "integer", "minimum": 1, "maximum": 160},
            "context": {"type": "integer", "minimum": 0, "maximum": 16},
            "op": {"type": "string", "maxLength": 64},
            "section": {
                "type": "string",
                "enum": ["overview", "outputs", "boundary", "pi"],
            },
        },
        "required": ["view"],
    },
}

INSPECT_LIFT_SCHEMA_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "inspect_lift_schema",
    "description": (
        "Read a compact exact bundle from the closed artifact schema and public "
        "operation catalog. Items are top, catalog, op:NAME, or section:NAME."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "string",
                    "pattern": "^(?:top|catalog|op:[A-Za-z0-9_]+|section:[A-Za-z0-9_.]+)$",
                },
            }
        },
        "required": ["items"],
    },
}

CHECK_CANDIDATE_TOOL_RESPONSE_API = {
    "type": "function",
    "name": "check_candidate",
    "description": (
        "Run the shared diagnostic-only CandidateCheck on the current exact "
        "candidate. It never edits or submits bytes and has no certification authority."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
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
    enable_apply_patch_tool: bool = False,
    enable_submit_candidate_tool: bool = False,
    enable_write_text_tool: bool = False,
    enable_semantic_ledger_tool: bool = False,
    enable_construction_plan_tool: bool = False,
    enable_check_submit_candidate_tool: bool = False,
    enable_lift_context_tool: bool = False,
    enable_lift_schema_tool: bool = False,
    enable_check_candidate_tool: bool = False,
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
                f"{max_tool_calls} bash tool call(s) per model turn; "
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
        if tool_name == "bash":
            if (
                not isinstance(args, dict)
                or "command" not in args
                or not isinstance(args.get("command"), str)
                or len(args.get("command", "")) > MAX_BASH_TOOL_CHARS
            ):
                error_msg += "Missing or non-string 'command' argument in bash tool call."
        elif tool_name == "apply_patch" and enable_apply_patch_tool:
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
                or len(args.get("old_text", "")) > MAX_EXACT_PATCH_TOOL_CHARS
                or len(args.get("new_text", "")) > MAX_EXACT_PATCH_TOOL_CHARS
                or (
                    "submit" in args and not isinstance(args.get("submit"), bool)
                )
            ):
                error_msg += (
                    "apply_patch requires string path, old_text, new_text, and "
                    "an optional boolean submit."
                )
        elif tool_name == "submit_candidate" and enable_submit_candidate_tool:
            if (
                not isinstance(args, dict)
                or set(args) != {"candidate_text"}
                or not isinstance(args.get("candidate_text"), str)
                or len(args.get("candidate_text", "")) > MAX_CANDIDATE_TOOL_CHARS
            ):
                error_msg += "submit_candidate requires exactly candidate_text."
        elif tool_name == "write_text" and enable_write_text_tool:
            if (
                not isinstance(args, dict)
                or not {"path", "content", "mode"}.issubset(args)
                or not set(args).issubset({"path", "content", "mode", "submit"})
                or not isinstance(args.get("path"), str)
                or not isinstance(args.get("content"), str)
                or len(args.get("content", "")) > MAX_WRITE_TEXT_TOOL_CHARS
                or args.get("mode") not in {"overwrite", "append"}
                or (
                    "submit" in args and not isinstance(args.get("submit"), bool)
                )
            ):
                error_msg += (
                    "write_text requires string path/content, mode overwrite|append, "
                    "and an optional boolean submit."
                )
        elif tool_name == "save_semantic_ledger" and enable_semantic_ledger_tool:
            if (
                not isinstance(args, dict)
                or set(args) != {"content"}
                or not isinstance(args.get("content"), str)
                or not 1 <= len(args.get("content", "")) <= 8192
            ):
                error_msg += (
                    "save_semantic_ledger requires exactly 1..8192 characters "
                    "of content."
                )
        elif (
            tool_name == "save_construction_plan"
            and enable_construction_plan_tool
        ):
            if (
                not isinstance(args, dict)
                or set(args) != {"content"}
                or not isinstance(args.get("content"), str)
                or not 1 <= len(args.get("content", "")) <= 8192
            ):
                error_msg += (
                    "save_construction_plan requires exactly 1..8192 characters "
                    "of content."
                )
        elif (
            tool_name == "check_and_submit_candidate"
            and enable_check_submit_candidate_tool
        ):
            if not isinstance(args, dict) or args:
                error_msg += "check_and_submit_candidate takes no arguments."
        elif tool_name == "inspect_lift_context" and enable_lift_context_tool:
            allowed = {
                "view", "query", "path", "start", "count", "context", "op", "section"
            }
            if (
                not isinstance(args, dict)
                or not set(args).issubset(allowed)
                or args.get("view") not in {
                    "source", "source_outline", "source_find", "source_brief",
                    "harness", "harness_abi",
                    "feedback", "feedback_category", "candidate", "candidate_find",
                    "candidate_nodes", "candidate_node", "candidate_abi", "candidate_control",
                    "candidate_diff", "knowledge",
                }
                or any(
                    key in args and not isinstance(args[key], str)
                    for key in ("query", "path", "op", "section")
                )
                or any(
                    key in args
                    and (
                        not isinstance(args[key], int)
                        or isinstance(args[key], bool)
                    )
                    for key in ("start", "count", "context")
                )
            ):
                error_msg += "inspect_lift_context has invalid closed arguments."
        elif tool_name == "inspect_lift_schema" and enable_lift_schema_tool:
            items = args.get("items") if isinstance(args, dict) else None
            if (
                not isinstance(args, dict)
                or set(args) != {"items"}
                or not isinstance(items, list)
                or not 1 <= len(items) <= 12
                or any(not isinstance(item, str) for item in items)
            ):
                error_msg += "inspect_lift_schema requires 1..12 string items."
        elif tool_name == "check_candidate" and enable_check_candidate_tool:
            if not isinstance(args, dict) or args:
                error_msg += "check_candidate takes no arguments."
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
        elif tool_name == "apply_patch":
            actions.append(
                {
                    "tool_name": "apply_patch",
                    "path": args["path"],
                    "old_text": args["old_text"],
                    "new_text": args["new_text"],
                    "submit": args.get("submit", False),
                    "tool_call_id": tool_call_id,
                }
            )
        elif tool_name == "submit_candidate":
            actions.append(
                {
                    "tool_name": "submit_candidate",
                    "candidate_text": args["candidate_text"],
                    "tool_call_id": tool_call_id,
                }
            )
        elif tool_name == "write_text":
            actions.append(
                {
                    "tool_name": "write_text",
                    "path": args["path"],
                    "content": args["content"],
                    "mode": args["mode"],
                    "submit": args.get("submit", False),
                    "tool_call_id": tool_call_id,
                }
            )
        elif tool_name == "save_semantic_ledger":
            actions.append(
                {
                    "tool_name": "save_semantic_ledger",
                    "content": args["content"],
                    "tool_call_id": tool_call_id,
                }
            )
        elif tool_name == "save_construction_plan":
            actions.append(
                {
                    "tool_name": "save_construction_plan",
                    "content": args["content"],
                    "tool_call_id": tool_call_id,
                }
            )
        elif tool_name == "check_and_submit_candidate":
            actions.append(
                {
                    "tool_name": "check_and_submit_candidate",
                    "tool_call_id": tool_call_id,
                }
            )
        elif tool_name == "inspect_lift_context":
            actions.append(
                {
                    "tool_name": "inspect_lift_context",
                    **args,
                    "tool_call_id": tool_call_id,
                }
            )
        elif tool_name == "inspect_lift_schema":
            actions.append(
                {
                    "tool_name": "inspect_lift_schema",
                    "items": list(args["items"]),
                    "tool_call_id": tool_call_id,
                }
            )
        elif tool_name == "check_candidate":
            actions.append(
                {
                    "tool_name": "check_candidate",
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
