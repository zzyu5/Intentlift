import copy
import hashlib
import json
import logging
import re
import signal
import time
import threading
from contextlib import contextmanager
from collections.abc import Callable, Mapping

import litellm
import httpx

from minisweagent.exceptions import (
    FormatError,
    LimitsExceeded,
    MiniSWERequestCancelled,
)
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall_response import (
    APPLY_REPLAY_PATCH_TOOL_RESPONSE_API,
    BASH_TOOL_RESPONSE_API,
    MCP_CALL_TOOL_RESPONSE_API,
    MCP_FIRST_READ_TOOL_RESPONSE_API,
    SUBMIT_REPLAY_TOOL_RESPONSE_API,
    finish_reason_from_responses_api,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_response_model")


_SENSITIVE_PROVIDER_ERROR_KEY = re.compile(
    r"(?i)(?:api[_ -]?key|access[_ -]?token|auth(?:orization)?|bearer|cookie|credential|password|secret|token)"
)
_MAX_PROVIDER_ERROR_TEXT = 8192
_MAX_PROVIDER_ERROR_ITEMS = 32
_SAFE_PROVIDER_ERROR_FIELDS = frozenset(
    {
        "code",
        "message",
        "detail",
        "reason",
        "param",
        "request_id",
        "status",
        "status_code",
    }
)


def _redact_provider_error_text(value: str) -> str:
    """Keep provider diagnostics while removing credential-shaped material."""

    text = value
    text = re.sub(
        r"(?i)(bearer\s+)[^\s,;]+",
        r"\1<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)((?:authorization|api[_ -]?key|token|secret|password)\s*[:=]\s*)[^\s,;]+",
        r"\1<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)(https?://[^\s?]+)\?[^\s]+",
        r"\1?<redacted>",
        text,
    )
    text = re.sub(r"(?i)https?://[^\s\"'<>]+", "<redacted-url>", text)
    text = re.sub(
        r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]{1,5})?(?![0-9])",
        "<redacted-address>",
        text,
    )
    text = re.sub(
        r"(?<![A-Za-z0-9_])/(?:[^\s\"']+/){1,}[^\s\"']*",
        "<redacted-path>",
        text,
    )
    return text[:_MAX_PROVIDER_ERROR_TEXT]


def _redact_provider_error_value(value, *, key: str | None = None):
    """Preserve external error structure while recursively removing secrets."""

    if key is not None and _SENSITIVE_PROVIDER_ERROR_KEY.search(key):
        return "<redacted>"
    if isinstance(value, str):
        return _redact_provider_error_text(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_provider_error_value(item_value, key=str(item_key))
            for item_key, item_value in list(value.items())[:_MAX_PROVIDER_ERROR_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _redact_provider_error_value(item)
            for item in value[:_MAX_PROVIDER_ERROR_ITEMS]
        ]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return _redact_provider_error_text(str(value))


def _safe_provider_error_block(value):
    """Keep a bounded provider error object after recursive redaction."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return _redact_provider_error_value(value)
    if isinstance(value, str):
        return _redact_provider_error_text(value)
    return _redact_provider_error_value(value)


def _safe_sse_error_envelope(event) -> dict[str, object]:
    """Retain a redacted SSE error only; never retain the full Responses body."""

    get = ResponsesSSEFailure._get
    result: dict[str, object] = {}
    event_type = get(event, "type")
    if isinstance(event_type, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", event_type):
        result["event_type"] = event_type
    if isinstance(event, Mapping):
        top_level = {
            key: value
            for key, value in event.items()
            if key in _SAFE_PROVIDER_ERROR_FIELDS
        }
    else:
        top_level = {
            key: get(event, key)
            for key in ("code", "message", "detail", "reason", "param", "request_id")
            if get(event, key) is not None
        }
    safe_top_level = _safe_provider_error_block(top_level)
    if safe_top_level:
        result["event"] = safe_top_level
    event_error = _safe_provider_error_block(get(event, "error"))
    if event_error:
        result["error"] = event_error
    response = get(event, "response")
    response_error = _safe_provider_error_block(get(response, "error"))
    if response_error:
        result["response_error"] = response_error
    response_status = get(response, "status")
    if isinstance(response_status, str) and re.fullmatch(
        r"[A-Za-z0-9_.:-]{1,96}", response_status
    ):
        result["response_status"] = response_status
    return result


def _safe_stream_exception_envelope(exc: BaseException) -> dict[str, object]:
    """Project an SDK/relay read failure without retaining a response body."""

    result: dict[str, object] = {
        "exception_type": type(exc).__name__[:96],
        "message": _redact_provider_error_text(str(exc)),
    }
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        result["http_status"] = status_code
    return result


def _stream_exception_failure_class(exc: BaseException) -> str:
    """Classify an iterator/read failure before it can become a candidate."""

    name = type(exc).__name__.casefold()
    text = str(exc).casefold()
    if "no_more_retry" in text or "no more retry" in text:
        return "provider_overloaded"
    if "ratelimit" in name or "rate limit" in text:
        return "provider_rate_limit"
    if any(
        marker in name or marker in text
        for marker in (
            "quota",
            "usage limit",
            "billing hard limit",
            "insufficient quota",
        )
    ):
        return "provider_usage_limit"
    if any(
        marker in name or marker in text
        for marker in ("overload", "high demand", "server overloaded")
    ):
        return "provider_overloaded"

    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        if status_code in {401, 403}:
            return "provider_authentication_error"
        if status_code == 408:
            return "provider_timeout"
        if status_code == 429:
            return "provider_rate_limit"
        if status_code == 402:
            return "provider_usage_limit"
        if status_code >= 500:
            return "provider_server_error"
        if status_code >= 400:
            return "provider_protocol_error"
    if "timeout" in name or "timed out" in text or "request_timeout" in text:
        return "provider_timeout"
    if "authentication" in name or "invalid api key" in text:
        return "provider_authentication_error"
    if "response.completed" in text or "stream disconnected" in text:
        return "provider_stream_incomplete"
    if any(marker in name for marker in ("internalserver", "serviceunavailable")):
        return "provider_server_error"
    if any(marker in name for marker in ("connection", "transport", "network")):
        return "provider_transport_error"
    return "provider_transport_error"


class ResponsesSSEFailure(RuntimeError):
    """A terminal SSE event reduced to a closed, response-free failure class."""

    _CODE_CLASSES = {
        "authentication_error": "provider_authentication_error",
        "invalid_api_key": "provider_authentication_error",
        "insufficient_quota": "provider_usage_limit",
        "usage_limit_reached": "provider_usage_limit",
        "rate_limit_exceeded": "provider_rate_limit",
        "rate_limit_error": "provider_rate_limit",
        "request_timeout": "provider_timeout",
        "overloaded": "provider_overloaded",
        "high_demand": "provider_overloaded",
        "server_overloaded": "provider_overloaded",
        "server_error": "provider_server_error",
        "internal_server_error": "provider_server_error",
        "server_is_overloaded": "provider_overloaded",
        "invalid_request_error": "provider_protocol_error",
        "unsupported_parameter": "provider_protocol_error",
    }
    _INCOMPLETE_REASON_CLASSES = {
        "content_filter": "provider_content_filter",
        "max_output_tokens": "provider_output_limit",
    }

    @staticmethod
    def _get(value, key):
        if isinstance(value, Mapping):
            return value.get(key)
        return getattr(value, key, None)

    @classmethod
    def _classify_error(cls, error) -> str | None:
        for field in ("code", "type"):
            value = cls._get(error, field)
            if isinstance(value, str):
                failure_class = cls._CODE_CLASSES.get(value.strip().casefold())
                if failure_class is not None:
                    return failure_class
        return None

    def __init__(
        self,
        event=None,
        *,
        failure_class: str | None = None,
        failure_origin: str | None = None,
        safe_error_envelope: Mapping[str, object] | None = None,
    ) -> None:
        event_type = self._get(event, "type")
        response = self._get(event, "response")
        error_candidates = (
            self._get(event, "error"),
            self._get(response, "error"),
        )
        classified = next(
            (
                candidate_class
                for error in error_candidates
                if (candidate_class := self._classify_error(error)) is not None
            ),
            None,
        )
        if classified is None and event_type == "response.incomplete":
            details = self._get(response, "incomplete_details")
            reason = self._get(details, "reason")
            if isinstance(reason, str):
                classified = self._INCOMPLETE_REASON_CLASSES.get(
                    reason.strip().casefold()
                )
        if failure_class is not None:
            classified = failure_class
        elif classified is None and event_type == "response.failed":
            classified = "provider_response_failed"
        elif classified is None and event_type == "response.incomplete":
            classified = "provider_response_incomplete"
        self.failure_class = classified or "provider_sse_error_event"
        self.failure_origin = failure_origin or {
            "response.failed": "sse_response_failed",
            "response.incomplete": "sse_response_incomplete",
            "error": "sse_error_event",
        }.get(event_type, "sse_terminal_event")
        self.safe_error_envelope = _safe_sse_error_envelope(event)
        if isinstance(safe_error_envelope, Mapping):
            self.safe_error_envelope.update(
                _safe_provider_error_block(safe_error_envelope)
            )
        super().__init__(self.failure_class)


class LitellmResponseModelConfig(LitellmModelConfig):
    max_tool_calls_per_response: int = 1
    enable_bash_tool: bool = True
    enable_lift_mcp_tool: bool = False
    require_first_lift_mcp: bool = False
    routine_reasoning_effort: str | None = None
    routine_timeout_seconds: float | None = None
    semantic_initial_turns: int = 1
    enable_replay_patch_tool: bool = False
    enable_submit_replay_tool: bool = False
    compact_tool_history: bool = False
    compact_tool_arguments_over_chars: int = 4096
    compact_tool_outputs_keep_recent: int = 4
    compact_workspace_mirrors: bool = False
    compact_response_history_keep_recent: int = 0
    use_previous_response_id: bool = False
    # Transport recovery is deliberately separate from logical Lift calls.
    # The active workflow may disable retries for expensive Lift turns while
    # other callers retain a bounded transport-only retry.
    transient_retry_attempts: int = 0
    # A Responses SSE stream may be retried twice after its first failed
    # physical request.  This is separate from generic transport retries so
    # the lifecycle can allow exactly three consecutive SSE attempts without
    # retrying authentication, quota, overload, or semantic failures.
    sse_retry_attempts: int = 0
    # Zero retains the generic model default. Replay sets both explicitly so a
    # changing source cannot turn into an unbounded provider-call episode.
    physical_provider_call_limit: int = 0
    transport_retry_limit: int = 0


class LitellmResponseModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        if not 1 <= self.config.max_tool_calls_per_response <= 4:
            raise ValueError("max_tool_calls_per_response must be in [1, 4]")
        if self.config.compact_tool_arguments_over_chars < 256:
            raise ValueError("compact_tool_arguments_over_chars must be at least 256")
        if self.config.compact_tool_outputs_keep_recent < 0:
            raise ValueError("compact_tool_outputs_keep_recent must be non-negative")
        if self.config.compact_response_history_keep_recent < 0:
            raise ValueError(
                "compact_response_history_keep_recent must be non-negative"
            )
        if self.config.require_first_lift_mcp and not self.config.enable_lift_mcp_tool:
            raise ValueError(
                "require_first_lift_mcp needs enable_lift_mcp_tool"
            )
        if self.config.routine_reasoning_effort not in {
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        }:
            raise ValueError("unsupported routine_reasoning_effort")
        if (
            self.config.routine_timeout_seconds is not None
            and self.config.routine_timeout_seconds <= 0
        ):
            raise ValueError("routine_timeout_seconds must be positive when set")
        if not 0 <= self.config.semantic_initial_turns <= 4:
            raise ValueError("semantic_initial_turns must be in [0, 4]")
        if not 0 <= self.config.transient_retry_attempts <= 1:
            raise ValueError("transient_retry_attempts must be in [0, 1]")
        if not 0 <= self.config.sse_retry_attempts <= 2:
            raise ValueError("sse_retry_attempts must be in [0, 2]")
        if self.config.physical_provider_call_limit < 0:
            raise ValueError("physical_provider_call_limit must be non-negative")
        if self.config.transport_retry_limit < 0:
            raise ValueError("transport_retry_limit must be non-negative")
        self._physical_provider_calls = 0
        self._transport_retries = 0
        self._history_compaction_packets = 0
        self._history_compacted_items = 0
        self._history_compacted_chars = 0
        self._workspace_mirror_compacted_blocks = 0
        self._workspace_mirror_compacted_chars = 0
        self._response_history_compacted_packets = 0
        self._response_history_compacted_items = 0
        self._response_history_compacted_chars = 0
        self._response_history_chain_identities: list[tuple[str, str]] = []
        self._response_history_chain_sha256 = ""
        self._response_history_chain_chars = 0
        self._input_tokens = 0
        self._cached_input_tokens = 0
        self._cache_write_input_tokens = 0
        self._provider_attempts: list[dict] = []
        self._continuation_requests = 0
        self._continuation_fallbacks = 0
        self._continuation_input_items = 0
        self._continuation_input_chars = 0
        self._stateless_input_items = 0
        self._stateless_input_chars = 0
        self._continuation_disabled = False
        self._continuation_disable_reason: str | None = None
        self._cancel_requested: Callable[[], int | None] | None = None

    def set_cancel_requested(
        self, callback: Callable[[], int | None] | None
    ) -> None:
        """Install a proof-irrelevant candidate-wave cancellation hook."""

        if callback is not None and not callable(callback):
            raise TypeError("cancel callback must be callable")
        self._cancel_requested = callback

    def _raise_if_cancelled(self) -> None:
        callback = self._cancel_requested
        winner = callback() if callback is not None else None
        if winner is not None:
            raise MiniSWERequestCancelled(winner)

    @property
    def physical_provider_calls(self) -> int:
        return self._physical_provider_calls

    @property
    def transport_retries(self) -> int:
        return self._transport_retries

    def _reserve_physical_provider_call(self, *, transport_retry: bool) -> None:
        """Reserve one actual request before it reaches the provider.

        The limit is intentionally checked at the physical attempt boundary,
        not merely before a logical model query.  This also caps a retry or a
        continuation fallback that happens inside a single query.
        """

        if (
            self.config.physical_provider_call_limit
            and self._physical_provider_calls
            >= self.config.physical_provider_call_limit
        ):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "PhysicalProviderCallLimitExceeded",
                    "extra": {
                        "exit_status": "PhysicalProviderCallLimitExceeded",
                        "submission": "",
                        "physical_provider_calls": self._physical_provider_calls,
                        "physical_provider_call_limit": (
                            self.config.physical_provider_call_limit
                        ),
                    },
                }
            )
        if (
            transport_retry
            and self.config.transport_retry_limit
            and self._transport_retries >= self.config.transport_retry_limit
        ):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "TransportRetryLimitExceeded",
                    "extra": {
                        "exit_status": "TransportRetryLimitExceeded",
                        "submission": "",
                        "transport_retries": self._transport_retries,
                        "transport_retry_limit": self.config.transport_retry_limit,
                    },
                }
            )
        self._physical_provider_calls += 1
        if transport_retry:
            self._transport_retries += 1

    @property
    def history_compaction(self) -> dict[str, int | bool]:
        return {
            "enabled": self.config.compact_tool_history,
            "packets": self._history_compaction_packets,
            "items": self._history_compacted_items,
            "chars": self._history_compacted_chars,
            "workspace_mirror_blocks": self._workspace_mirror_compacted_blocks,
            "workspace_mirror_chars": self._workspace_mirror_compacted_chars,
            "response_history_packets": self._response_history_compacted_packets,
            "response_history_items": self._response_history_compacted_items,
            "response_history_chars": self._response_history_compacted_chars,
        }

    @property
    def prompt_cache_telemetry(self) -> dict[str, int | str | bool]:
        key = self.config.model_kwargs.get("prompt_cache_key")
        return {
            "policy": (
                "stable-append-only-prefix"
                if not self.config.compact_tool_history
                and not self.config.compact_workspace_mirrors
                and self.config.compact_response_history_keep_recent == 0
                else "history-transforming"
            ),
            "key_present": isinstance(key, str) and bool(key),
            "input_tokens": self._input_tokens,
            "cached_input_tokens": self._cached_input_tokens,
            "cache_write_input_tokens": self._cache_write_input_tokens,
        }

    @property
    def response_continuation_telemetry(self) -> dict[str, int | str | bool | None]:
        return {
            "enabled": self.config.use_previous_response_id,
            "disabled": self._continuation_disabled,
            "disable_reason": self._continuation_disable_reason,
            "requests": self._continuation_requests,
            "fallbacks": self._continuation_fallbacks,
            "continuation_input_items": self._continuation_input_items,
            "continuation_input_chars": self._continuation_input_chars,
            "stateless_input_items": self._stateless_input_items,
            "stateless_input_chars": self._stateless_input_chars,
        }

    @staticmethod
    def _usage_counter(value, key: str) -> int:
        observed = value.get(key) if isinstance(value, dict) else getattr(value, key, 0)
        return (
            observed
            if isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed >= 0
            else 0
        )

    @staticmethod
    def _response_to_json(value) -> dict:
        """Normalize a completed Responses object without serializer warnings.

        Some relays put a plain usage mapping into a typed Responses model.
        Calling ``model_dump`` with its default warning policy then emits a
        Pydantic warning on every Lift turn even though the response is usable.
        Suppress only that serialization warning and keep the normalized JSON
        mapping; this does not alter provider failure or completion handling.
        """

        if isinstance(value, dict):
            return copy.deepcopy(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(mode="json", warnings=False)
            except TypeError:
                dumped = model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped
        try:
            dumped = dict(value)
        except (TypeError, ValueError):
            return {"value": repr(value)}
        return dumped if isinstance(dumped, dict) else {"value": repr(value)}

    @staticmethod
    def _compaction_receipt(*, kind: str, text: str) -> str:
        return json.dumps(
            {
                "protocol": "intentir.agent-history-compaction.v1",
                "kind": kind,
                "status": "compacted",
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "chars": len(text),
                "authority": (
                    "transport-only; rerun the tool or inspect the persistent "
                    "workspace for exact content"
                ),
            },
            sort_keys=True,
        )

    def _compact_prepared_history(self, messages: list[dict]) -> list[dict]:
        if not self.config.compact_tool_history:
            return messages
        compacted = [dict(message) for message in messages]
        output_indices = [
            index
            for index, message in enumerate(compacted)
            if message.get("type") == "function_call_output"
        ]
        keep = self.config.compact_tool_outputs_keep_recent
        stale_outputs = set(output_indices[:-keep] if keep else output_indices)
        packet_items = 0
        packet_chars = 0
        for index, message in enumerate(compacted):
            item_type = message.get("type")
            if item_type == "function_call":
                arguments = message.get("arguments")
                if (
                    isinstance(arguments, str)
                    and len(arguments)
                    > self.config.compact_tool_arguments_over_chars
                ):
                    receipt = self._compaction_receipt(
                        kind="tool_arguments", text=arguments
                    )
                    try:
                        parsed = json.loads(arguments)
                    except (TypeError, ValueError):
                        parsed = {}
                    if message.get("name") == "apply_replay_patch":
                        replacement = {
                            "path": (
                                parsed.get("path", "/output/candidate.json")
                                if isinstance(parsed, dict)
                                else "/output/candidate.json"
                            ),
                            "old_text": receipt,
                            "new_text": "",
                            "submit": False,
                        }
                    elif message.get("name") == "submit_replay":
                        replacement = {"source_text": receipt}
                    elif message.get("name") == "mcp_call_tool":
                        method = (
                            parsed.get("method", "resources/read")
                            if isinstance(parsed, dict)
                            else "resources/read"
                        )
                        params = (
                            parsed.get("params", {})
                            if isinstance(parsed, dict)
                            else {}
                        )
                        if not isinstance(params, dict):
                            params = {}
                        replacement = {"method": method, "params": params}
                    else:
                        replacement = {"command": f"# {receipt}"}
                    message["arguments"] = json.dumps(replacement, sort_keys=True)
                    packet_items += 1
                    packet_chars += max(
                        0, len(arguments) - len(message["arguments"])
                    )
            elif item_type == "function_call_output" and index in stale_outputs:
                output = message.get("output")
                if isinstance(output, str):
                    receipt = self._compaction_receipt(
                        kind="tool_output", text=output
                    )
                    message["output"] = receipt
                    packet_items += 1
                    packet_chars += max(0, len(output) - len(receipt))
        if packet_items:
            self._history_compaction_packets += 1
            self._history_compacted_items += packet_items
            self._history_compacted_chars += packet_chars
        return compacted

    @staticmethod
    def _replace_workspace_mirror_blocks(text: str) -> tuple[str, int, int]:
        """Replace re-readable exact workspace mirrors with deterministic receipts.

        The local trajectory is never changed.  Marked bytes already have an exact
        filesystem authority, so later stateless provider packets do not need to
        resend them after the model has received them once.
        """

        opening = "<persistent-workspace-mirror>"
        closing = "</persistent-workspace-mirror>"
        cursor = 0
        parts: list[str] = []
        blocks = 0
        removed_chars = 0
        while True:
            start = text.find(opening, cursor)
            if start < 0:
                parts.append(text[cursor:])
                break
            end = text.find(closing, start + len(opening))
            if end < 0:
                parts.append(text[cursor:])
                break
            exact_start = start + len(opening)
            exact_text = text[exact_start:end]
            receipt = json.dumps(
                {
                    "protocol": "intentir.agent-workspace-mirror.v1",
                    "status": "compacted-after-first-use",
                    "sha256": hashlib.sha256(exact_text.encode("utf-8")).hexdigest(),
                    "chars": len(exact_text),
                    "authority": (
                        "transport-only; exact bytes remain in the persistent "
                        "allowlisted workspace and may be reread"
                    ),
                },
                sort_keys=True,
            )
            parts.extend((text[cursor:exact_start], "\n", receipt, "\n"))
            cursor = end
            blocks += 1
            removed_chars += max(0, len(exact_text) - len(receipt) - 2)
        return "".join(parts), blocks, removed_chars

    def _compact_workspace_mirror_messages(
        self, messages: list[dict]
    ) -> list[dict]:
        if not self.config.compact_workspace_mirrors:
            return messages
        compacted = copy.deepcopy(messages)
        later_response = False
        response_after: list[bool] = [False] * len(compacted)
        for index in range(len(compacted) - 1, -1, -1):
            response_after[index] = later_response
            if compacted[index].get("object") == "response":
                later_response = True
        packet_blocks = 0
        packet_chars = 0
        for index, message in enumerate(compacted):
            if not response_after[index] or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                replaced, blocks, chars = self._replace_workspace_mirror_blocks(
                    content
                )
                message["content"] = replaced
                packet_blocks += blocks
                packet_chars += chars
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "input_text":
                        continue
                    text = item.get("text")
                    if not isinstance(text, str):
                        continue
                    replaced, blocks, chars = self._replace_workspace_mirror_blocks(
                        text
                    )
                    item["text"] = replaced
                    packet_blocks += blocks
                    packet_chars += chars
        self._workspace_mirror_compacted_blocks += packet_blocks
        self._workspace_mirror_compacted_chars += packet_chars
        return compacted

    def _compact_old_response_groups(self, messages: list[dict]) -> list[dict]:
        keep = self.config.compact_response_history_keep_recent
        if keep == 0:
            return messages
        response_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("object") == "response"
        ]
        if len(response_indices) <= keep:
            return messages
        cutoff = response_indices[-keep]
        dropped: list[dict] = []
        retained: list[dict] = []
        receipt_insert_at = 0
        for index, message in enumerate(messages):
            is_old_runtime_item = index < cutoff and (
                message.get("object") == "response"
                or message.get("type") == "function_call_output"
            )
            if is_old_runtime_item:
                dropped.append(message)
                receipt_insert_at = len(retained)
            else:
                retained.append(message)
        if not dropped:
            return messages
        identities = [self._response_history_item_identity(item) for item in dropped]
        cached = self._response_history_chain_identities
        if identities[: len(cached)] != cached:
            # Conversation replacement is legal only at explicit continuation
            # boundaries. Rebuild the transport-only chain once in that rare case;
            # the ordinary append-only path serializes each old item exactly once.
            cached = []
            self._response_history_chain_identities = []
            self._response_history_chain_sha256 = ""
            self._response_history_chain_chars = 0
        for identity, item in zip(
            identities[len(cached) :],
            dropped[len(cached) :],
            strict=True,
        ):
            encoded = json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
            previous = (
                bytes.fromhex(self._response_history_chain_sha256)
                if self._response_history_chain_sha256
                else bytes(32)
            )
            self._response_history_chain_sha256 = hashlib.sha256(
                previous + len(encoded).to_bytes(8, "big") + encoded
            ).hexdigest()
            self._response_history_chain_chars += len(encoded)
            self._response_history_chain_identities.append(identity)
        removed_chars = (
            self._response_history_chain_chars + max(0, len(dropped) - 1) + 2
        )
        receipt_text = json.dumps(
            {
                "protocol": "intentir.agent-response-history.v2",
                "status": "compacted",
                "dropped_items": len(dropped),
                "chain_sha256": self._response_history_chain_sha256,
                "canonical_item_chars": self._response_history_chain_chars,
                "authority": (
                    "transport-only; full trajectory is retained locally and exact "
                    "working files remain in the persistent workspace; chain is "
                    "sha256(previous_chain || item_length || canonical_item)"
                ),
            },
            sort_keys=True,
        )
        retained.insert(
            receipt_insert_at,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": receipt_text}],
            },
        )
        self._response_history_compacted_packets += 1
        self._response_history_compacted_items += len(dropped)
        self._response_history_compacted_chars += max(
            0, removed_chars - len(receipt_text)
        )
        return retained

    @staticmethod
    def _response_history_item_identity(message: dict) -> tuple[str, str]:
        """Return the append-only local identity of one runtime history item."""

        if message.get("object") == "response":
            value = message.get("id")
            if isinstance(value, str) and value:
                return ("response", value)
        if message.get("type") == "function_call_output":
            value = message.get("call_id")
            if isinstance(value, str) and value:
                return ("function_call_output", value)
        encoded = json.dumps(
            message,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return ("canonical", hashlib.sha256(encoded).hexdigest())

    def _compact_stale_lift_continuations(
        self, messages: list[dict]
    ) -> list[dict]:
        """Keep only the latest exact Gate1 continuation in the API packet.

        The full trajectory remains local.  The newest continuation contains the
        current exact feedback and candidate (or their persistent paths), so older
        continuation envelopes are stale transport state rather than authority.
        """

        marker = "intentir.lift-owner-continuation.v1"
        continuation_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "user"
            and marker
            in json.dumps(message.get("content", ""), ensure_ascii=False, default=str)
        ]
        if len(continuation_indices) <= 1:
            return messages
        stale = set(continuation_indices[:-1])
        latest = continuation_indices[-1]
        dropped = [messages[index] for index in continuation_indices[:-1]]
        serialized = json.dumps(
            dropped,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        receipt_text = json.dumps(
            {
                "protocol": "intentir.lift-continuation-history.v1",
                "status": "stale-continuations-compacted",
                "dropped_items": len(dropped),
                "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                "chars": len(serialized),
                "authority": (
                    "transport-only; the full trajectory remains local and the "
                    "latest continuation is exact"
                ),
            },
            sort_keys=True,
        )
        retained: list[dict] = []
        for index, message in enumerate(messages):
            if index in stale:
                continue
            if index == latest:
                retained.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": receipt_text}],
                    }
                )
            retained.append(message)
        self._response_history_compacted_packets += 1
        self._response_history_compacted_items += len(dropped)
        self._response_history_compacted_chars += max(
            0, len(serialized) - len(receipt_text)
        )
        return retained

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Build one complete, stateless Responses input history.

        The MCP lifecycle depends on the ordered pair of every provider
        ``function_call`` and its following ``function_call_output``. A
        receipt or a recent-items window is not an equivalent history: it can
        leave the model with a tool result whose call is no longer present,
        or hide the graph rejection that must guide the next authoring turn.
        Keep this serializer mechanical. It flattens completed Responses
        envelopes, removes only local ``extra`` metadata, and normalizes text
        block direction. It never compacts, summarizes, or drops a prior item.
        """

        result: list[dict] = []
        for message in messages:
            items = (
                message.get("output", [])
                if message.get("object") == "response"
                else [message]
            )
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                normalized = {
                    key: copy.deepcopy(value)
                    for key, value in item.items()
                    if key != "extra"
                }
                result.append(self._normalize_responses_history_item(normalized))
        return result

    @staticmethod
    def _normalize_responses_history_item(item: dict) -> dict:
        """Normalize one local history item without changing tool semantics."""

        if item.get("type") != "message":
            return item
        role = item.get("role")
        content = item.get("content")
        target_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, str):
            item["content"] = [{"type": target_type, "text": content}]
            return item
        if not isinstance(content, list):
            return item
        normalized_content: list = []
        for block in content:
            if not isinstance(block, dict):
                normalized_content.append(copy.deepcopy(block))
                continue
            normalized_block = copy.deepcopy(block)
            if normalized_block.get("type") in {"input_text", "output_text"}:
                normalized_block["type"] = target_type
            normalized_content.append(normalized_block)
        item["content"] = normalized_content
        return item

    @staticmethod
    def _latest_response_id(messages: list[dict]) -> tuple[int, str | None]:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("object") != "response":
                continue
            response_id = message.get("id")
            return index, response_id if isinstance(response_id, str) and response_id else None
        return -1, None

    @staticmethod
    def _packet_shape(messages: list[dict]) -> tuple[int, int]:
        return (
            len(messages),
            len(
                json.dumps(
                    messages,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                )
            ),
        )

    @staticmethod
    def _is_previous_response_error(exc: Exception) -> bool:
        # A stateless replay of the exact local history is the authoritative
        # fallback. Once a continuation request itself receives a provider
        # BadRequest, retrying the same previous_response_id cannot add value
        # and only multiplies relay latency.
        if isinstance(exc, litellm.exceptions.BadRequestError):
            return True
        text = f"{type(exc).__name__}: {exc}".casefold()
        return (
            "previous_response_id" in text
            or "previous response id" in text
            or "no tool call found for function call output" in text
            or (
                "previous response" in text
                and any(token in text for token in ("not found", "unknown", "unsupported"))
            )
        )

    @staticmethod
    def _is_transient_provider_error(exc: BaseException) -> bool:
        """Retry only failures proven to occur inside an active SSE stream.

        HTTP responses and request-establishment/transport exceptions are
        lifecycle-terminal.  Stream iteration failures are wrapped at the SSE
        boundary with an explicit ``sse_*`` origin, which is the only evidence
        that permits the configured consecutive-stream retry budget.
        """

        # Provider-declared overload, rate, usage, and explicit no-more-retry
        # outcomes are lifecycle-terminal.  They must not enter the generic
        # LiteLLM retry path, even when the relay exposes them as a generic
        # 5xx/service-unavailable exception.
        failure_class = _stream_exception_failure_class(exc)
        failure_text = f"{type(exc).__name__}: {exc}".casefold()
        if failure_class in {
            "provider_overloaded",
            "provider_rate_limit",
            "provider_usage_limit",
        } or "no_more_retry" in failure_text:
            return False

        # Retry only stream failures that are genuinely transport/SSE-shaped.
        # Provider-declared authentication, quota, rate, overload, content
        # filtering, output limits, and invalid-request events were classified
        # above or below and remain terminal.  In particular, an SSE wrapper
        # must not make a semantic/provider rejection look retryable.
        if isinstance(exc, ResponsesSSEFailure):
            return (
                exc.failure_origin.startswith("sse_")
                and exc.failure_class
                in {
                    "provider_sse_error_event",
                    "provider_stream_incomplete",
                    "provider_response_failed",
                    "provider_response_incomplete",
                    "provider_timeout",
                    "provider_server_error",
                }
            )
        return False

    def _record_packet(self, messages: list[dict], *, continuation: bool) -> tuple[int, int]:
        items, chars = self._packet_shape(messages)
        if continuation:
            self._continuation_requests += 1
            self._continuation_input_items += items
            self._continuation_input_chars += chars
        else:
            self._stateless_input_items += items
            self._stateless_input_chars += chars
        return items, chars

    def _response_tools(
        self,
        *,
        response_count: int,
        response_tools: list[dict] | None = None,
    ) -> list[dict]:
        if response_tools is not None:
            if not response_tools or any(
                not isinstance(tool, dict) for tool in response_tools
            ):
                raise ValueError("dynamic Responses tool surface is malformed")
            return copy.deepcopy(response_tools)
        if self.config.require_first_lift_mcp and response_count == 0:
            # The first owner turn is deliberately read-only.  This makes the
            # MCP bridge an actual context boundary instead of an optional set
            # of tools that a provider can bypass by emitting a candidate from
            # the startup receipt alone.  Later turns receive the ordinary
            # MCP read/write surface and CandidateCheck remains authoritative.
            return [MCP_FIRST_READ_TOOL_RESPONSE_API]
        return [
            *(
                [BASH_TOOL_RESPONSE_API]
                if self.config.enable_bash_tool
                else []
            ),
            *([MCP_CALL_TOOL_RESPONSE_API] if self.config.enable_lift_mcp_tool else []),
            *(
                [APPLY_REPLAY_PATCH_TOOL_RESPONSE_API]
                if self.config.enable_replay_patch_tool
                else []
            ),
            *(
                [SUBMIT_REPLAY_TOOL_RESPONSE_API]
                if self.config.enable_submit_replay_tool
                else []
            ),
        ]

    @contextmanager
    def _hard_provider_deadline(self, timeout):
        """Enforce the configured turn deadline around streaming LiteLLM calls.

        LiteLLM's HTTP timeout does not reliably bound iteration over a relay's
        SSE stream.  Lift/Replay must still fail closed at the model-turn
        boundary, so the main-thread caller also owns a real wall-clock alarm.
        Worker-thread callers retain LiteLLM's native timeout because Python
        signals cannot safely interrupt a non-main thread.
        """

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or threading.current_thread() is not threading.main_thread()
        ):
            yield
            return
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        started = time.monotonic()

        def on_alarm(_signum, _frame):
            raise litellm.exceptions.Timeout(
                "provider response exceeded the configured turn deadline",
                model=self.config.model_name,
                llm_provider="responses",
            )

        signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            if previous_timer[0] > 0:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(0.001, previous_timer[0] - elapsed),
                    previous_timer[1],
                )
            else:
                signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    @staticmethod
    def _materialize_streaming_response(response):
        """Return the completed Responses object for a streaming LiteLLM call.

        A streaming Responses iterator is usable only after the provider emits
        its completed response envelope.  A closed function-call item is not a
        substitute for that envelope: accepting it would turn a truncated SSE
        transport into a valid Lift/Replay action.
        Duck typing is intentional: LiteLLM has changed the concrete iterator
        class between releases, while the completed-response contract is
        stable.
        """

        if not hasattr(response, "completed_response"):
            # ``_query`` always requests the Responses SSE path.  A relay may
            # nevertheless return a plain completed envelope, which is safe
            # only when it explicitly says ``status=completed`` and carries an
            # output list.  An arbitrary iterator or a function-call-looking
            # object is not a completion witness and must not reach the action
            # parser as if it were one.
            status = (
                response.get("status")
                if isinstance(response, dict)
                else getattr(response, "status", None)
            )
            output = (
                response.get("output")
                if isinstance(response, dict)
                else getattr(response, "output", None)
            )
            if status == "completed" and isinstance(output, (list, tuple)):
                return response
            raise RuntimeError(
                "Responses stream ended without a completed response object"
            )
        function_items: dict[str, dict] = {}
        anonymous_function_items: list[dict] = []
        completed_event_response = None
        saw_completed = False
        last_event_type: str | None = None
        try:
            iterator = iter(response)
        except Exception as exc:
            raise ResponsesSSEFailure(
                failure_class=_stream_exception_failure_class(exc),
                failure_origin="sse_read_error",
                safe_error_envelope={
                    **_safe_stream_exception_envelope(exc),
                    "terminal_event_received": False,
                },
            ) from exc
        while True:
            try:
                _event = next(iterator)
            except StopIteration:
                break
            except ResponsesSSEFailure:
                raise
            except Exception as exc:
                raise ResponsesSSEFailure(
                    failure_class=_stream_exception_failure_class(exc),
                    failure_origin="sse_read_error",
                    safe_error_envelope={
                        **_safe_stream_exception_envelope(exc),
                        **(
                            {"last_event_type": last_event_type}
                            if isinstance(last_event_type, str)
                            else {}
                        ),
                        "terminal_event_received": False,
                    },
                ) from exc
            event_type = (
                _event.get("type") if isinstance(_event, dict)
                else getattr(_event, "type", None)
            )
            if isinstance(event_type, str):
                last_event_type = event_type
            if event_type in {"error", "response.failed", "response.incomplete"}:
                raise ResponsesSSEFailure(_event)
            if event_type == "response.completed":
                saw_completed = True
                completed_event_response = (
                    _event.get("response") if isinstance(_event, dict)
                    else getattr(_event, "response", None)
                )
            item = (
                _event.get("item") if isinstance(_event, dict)
                else getattr(_event, "item", None)
            )
            item_type = (
                item.get("type") if isinstance(item, dict)
                else getattr(item, "type", None)
            )
            if item_type != "function_call":
                continue
            item_value = LitellmResponseModel._response_to_json(item)
            item_id = item_value.get("id")
            if isinstance(item_id, str) and item_id:
                function_items[item_id] = item_value
            else:
                anonymous_function_items.append(item_value)
        if not saw_completed:
            raise ResponsesSSEFailure(
                failure_class="provider_stream_incomplete",
                failure_origin="sse_completion_missing",
                safe_error_envelope={
                    "last_event_type": last_event_type,
                    "terminal_event_received": False,
                },
            )
        completed_event = getattr(response, "completed_response", None)
        completed_response = getattr(completed_event, "response", None)
        if completed_response is None:
            completed_response = completed_event_response
        if completed_response is None:
            raise ResponsesSSEFailure(
                failure_class="provider_protocol_error",
                failure_origin="sse_completed_envelope_invalid",
            )
        status = (
            completed_response.get("status")
            if isinstance(completed_response, dict)
            else getattr(completed_response, "status", None)
        )
        output = (
            completed_response.get("output")
            if isinstance(completed_response, dict)
            else getattr(completed_response, "output", None)
        )
        if status != "completed" or not isinstance(output, (list, tuple)):
            raise ResponsesSSEFailure(
                failure_class="provider_response_incomplete",
                failure_origin="sse_completed_response_incomplete",
            )
        output_items = list(output) if isinstance(output, (list, tuple)) else []
        output_ids = {
            item_id
            for item in output_items
            if isinstance(
                item_id := (
                    item.get("id")
                    if isinstance(item, dict)
                    else getattr(item, "id", None)
                ),
                str,
            )
            and item_id
        }
        recovered_items = [
            item
            for item_id, item in function_items.items()
            if item_id not in output_ids
        ]
        recovered_items.extend(anonymous_function_items)
        if recovered_items:
            if isinstance(completed_response, dict):
                completed_response = {
                    **completed_response,
                    "output": [*output_items, *recovered_items],
                }
            else:
                completed_response.output = [*output_items, *recovered_items]
        return completed_response

    def _query(
        self,
        messages: list[dict[str, str]],
        *,
        response_count: int = 0,
        semantic_turn: bool = False,
        response_tools: list[dict] | None = None,
        **kwargs,
    ):
        if self.config.routine_reasoning_effort is not None and not semantic_turn:
            kwargs["reasoning_effort"] = self.config.routine_reasoning_effort
        if self.config.routine_timeout_seconds is not None and not semantic_turn:
            configured_timeout = kwargs.get(
                "timeout", self.config.model_kwargs.get("timeout")
            )
            kwargs["timeout"] = (
                min(float(configured_timeout), self.config.routine_timeout_seconds)
                if isinstance(configured_timeout, (int, float))
                and not isinstance(configured_timeout, bool)
                and configured_timeout > 0
                else self.config.routine_timeout_seconds
            )
        try:
            model_kwargs = self.config.model_kwargs | kwargs
            # The active protocol is always a stateless full-history stream.
            # Do not allow a caller or inherited config to switch either
            # property for one turn.
            model_kwargs.pop("previous_response_id", None)
            model_kwargs["stream"] = True
            model_kwargs["store"] = False
            tools = self._response_tools(
                response_count=response_count,
                response_tools=response_tools,
            )
            if tools:
                # Lift and Replay are action protocols.  A completed response
                # without a tool call cannot advance either protocol and only
                # creates an avoidable format-error turn.
                model_kwargs.setdefault("tool_choice", "required")
            if model_kwargs.get("timeout") is None:
                model_kwargs["timeout"] = httpx.Timeout(None)
            def issue_response(request_kwargs):
                response = litellm.responses(
                    model=self.config.model_name,
                    input=messages,
                    tools=tools,
                    **request_kwargs,
                )
                return self._materialize_streaming_response(response)

            with self._hard_provider_deadline(model_kwargs.get("timeout")):
                # An invalid SSE/SDK response must fail closed. Reissuing the
                # turn as non-streaming here would hide a second physical
                # provider call from the Lift budget and could bypass the
                # completed-stream contract.
                return issue_response(model_kwargs)
        except litellm.exceptions.AuthenticationError as e:
            e.message += " Configure credentials through the IntentIR provider settings."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response_tools = kwargs.pop("response_tools", None)
        if response_tools is not None and (
            not isinstance(response_tools, list)
            or any(not isinstance(tool, dict) for tool in response_tools)
        ):
            raise ValueError("dynamic Responses tool surface is malformed")
        # The local history is the sole conversation authority. Even if a
        # legacy caller passes a continuation option, it must not reach the
        # provider or change the input packet shape.
        kwargs.pop("previous_response_id", None)
        previous_response_id = None
        prepared_messages = self._prepare_messages_for_api(messages)
        response_count = sum(
            message.get("object") == "response" for message in messages
        )
        last_response_index = max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("object") == "response"
            ),
            default=-1,
        )
        continuation_turn = any(
            message.get("role") == "user"
            and "intentir.lift-owner-continuation.v1"
            in json.dumps(message.get("content", ""), ensure_ascii=False)
            for message in messages[last_response_index + 1 :]
        )
        semantic_turn = (
            response_count < self.config.semantic_initial_turns
            or continuation_turn
        )
        abort_exceptions = list(self.abort_exceptions)
        if not self.config.retry_timeouts:
            abort_exceptions.append(litellm.exceptions.Timeout)
        for attempt in retry(
            logger=logger,
            abort_exceptions=abort_exceptions,
            max_attempts=max(
                self.config.retry_attempts
                + self.config.transient_retry_attempts,
                self.config.retry_attempts + self.config.sse_retry_attempts,
            ),
            # Never fall back to the historical "retry every exception"
            # policy.  In particular, a relay may surface an overload as a
            # generic 5xx or service-unavailable exception; the predicate
            # above must get the opportunity to classify it and stop here.
            retry_predicate=self._is_transient_provider_error,
        ):
            # This check deliberately sits outside the retry attempt context:
            # cancellation is a candidate-wave decision, never a transport
            # failure eligible for another provider call.
            self._raise_if_cancelled()
            with attempt:
                call_was_fallback = False
                self._reserve_physical_provider_call(
                    transport_retry=attempt.retry_state.attempt_number > 1
                )
                started = time.monotonic()
                packet_items, packet_chars = self._record_packet(
                    prepared_messages,
                    continuation=previous_response_id is not None,
                )
                try:
                    response = self._query(
                        prepared_messages,
                        response_count=response_count,
                        semantic_turn=semantic_turn,
                        response_tools=response_tools,
                        **(
                            {"previous_response_id": previous_response_id}
                            if previous_response_id is not None
                            else {}
                        ),
                        **kwargs,
                    )
                except Exception as exc:
                    if exc is None:
                        pass
                    else:
                        self._provider_attempts.append(
                            {
                                "physical_call": self._physical_provider_calls,
                                "attempt": attempt.retry_state.attempt_number,
                                "elapsed_ms": (time.monotonic() - started) * 1000.0,
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "reasoning_effort": (
                                    self.config.model_kwargs.get("reasoning_effort")
                                    if semantic_turn
                                    else self.config.routine_reasoning_effort
                                    or self.config.model_kwargs.get("reasoning_effort")
                                ),
                                "continuation": previous_response_id is not None,
                                "previous_response_id_present": (
                                    previous_response_id is not None
                                ),
                                "fallback": call_was_fallback,
                                "input_items": packet_items,
                                "input_chars": packet_chars,
                            }
                        )
                        raise exc
                if response is not None:
                    usage = (
                        response.get("usage")
                        if isinstance(response, dict)
                        else getattr(response, "usage", None)
                    )
                    input_details = (
                        usage.get("input_tokens_details")
                        if isinstance(usage, dict)
                        else getattr(usage, "input_tokens_details", None)
                    )
                    input_tokens = self._usage_counter(usage, "input_tokens")
                    cached_input_tokens = self._usage_counter(
                        input_details, "cached_tokens"
                    ) or self._usage_counter(usage, "cached_input_tokens")
                    cache_write_input_tokens = self._usage_counter(
                        input_details, "cache_write_tokens"
                    ) or self._usage_counter(usage, "cache_write_input_tokens")
                    self._input_tokens += input_tokens
                    self._cached_input_tokens += cached_input_tokens
                    self._cache_write_input_tokens += cache_write_input_tokens
                    self._provider_attempts.append(
                        {
                            "physical_call": self._physical_provider_calls,
                            "attempt": attempt.retry_state.attempt_number,
                            "elapsed_ms": (time.monotonic() - started) * 1000.0,
                            "status": "success",
                            "error_type": None,
                            "input_tokens": input_tokens,
                            "cached_input_tokens": cached_input_tokens,
                            "cache_write_input_tokens": cache_write_input_tokens,
                            "reasoning_effort": (
                                self.config.model_kwargs.get("reasoning_effort")
                                if semantic_turn
                                else self.config.routine_reasoning_effort
                                or self.config.model_kwargs.get("reasoning_effort")
                            ),
                            "continuation": previous_response_id is not None,
                            "previous_response_id_present": (
                                previous_response_id is not None
                            ),
                            "fallback": call_was_fallback,
                            "input_items": packet_items,
                            "input_chars": packet_chars,
                        }
                    )
        # Preserve the completed provider-attempt receipt above, but do not let
        # a late response start another tool/model step after a sibling won.
        self._raise_if_cancelled()
        cost_output = self._calculate_cost(response)
        if self.config.use_global_model_stats:
            GLOBAL_MODEL_STATS.add(cost_output["cost"])
        try:
            actions = self._parse_actions(
                response,
                response_count=response_count,
            )
        except FormatError as e:
            # hasattr guard: litellm.responses() returns a pydantic object, but tests
            # may inject a plain dict; dict(response) is the correct fallback in that case.
            # Inner try: if serialization itself fails, repr() guarantees the key is always set.
            try:
                e.messages[0]["extra"]["response"] = self._response_to_json(response)
            except Exception:
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = self._response_to_json(response)
        message["extra"] = {
            "actions": actions,
            **cost_output,
            "timestamp": time.time(),
            "provider_elapsed_ms": self._provider_attempts[-1]["elapsed_ms"],
        }
        return message

    def _parse_actions(self, response, *, response_count: int = 0) -> list[dict]:
        return parse_toolcall_actions_response(
            getattr(response, "output", []),
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": finish_reason_from_responses_api(response)},
            max_tool_calls=self.config.max_tool_calls_per_response,
            enable_bash_tool=self.config.enable_bash_tool,
            enable_lift_mcp_tool=self.config.enable_lift_mcp_tool,
            mcp_first_read_only=(
                self.config.require_first_lift_mcp
                and response_count == 0
            ),
            enable_replay_patch_tool=self.config.enable_replay_patch_tool,
            enable_submit_replay_tool=self.config.enable_submit_replay_tool,
        )

    def format_message(self, **kwargs) -> dict:
        """Create a Responses message with role-correct text blocks."""

        role = kwargs.get("role", "user")
        content = kwargs.get("content", "")
        extra = kwargs.get("extra")
        block_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, str):
            content = [{"type": block_type, "text": content}]
        elif isinstance(content, list):
            content = copy.deepcopy(content)
            for block in content:
                if isinstance(block, dict) and block.get("type") in {
                    "input_text",
                    "output_text",
                }:
                    block["type"] = block_type
        message = {"type": "message", "role": role, "content": content}
        if extra:
            message["extra"] = extra
        return message

    def serialize(self) -> dict:
        value = super().serialize()
        info = value.setdefault("info", {})
        info["agent_runtime"] = {
            "protocol": "intentir.agent-runtime.v1",
            "history_compaction": self.history_compaction,
            "prompt_cache": self.prompt_cache_telemetry,
            "response_continuation": self.response_continuation_telemetry,
            "provider_attempts": [dict(item) for item in self._provider_attempts],
        }
        return value

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
