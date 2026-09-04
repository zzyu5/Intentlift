"""Minimal HTTP client for the LLM provider (OpenAI ``responses`` wire API).

Dependency-free (stdlib ``urllib``) so extraction runs without extra packages.
The API key is read from :mod:`provider` and is NEVER logged, printed, or put in
an exception message — only ``Authorization`` header bytes that urllib sends.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .provider import ProviderConfig, get_provider


# Source-aware Lift and source-blind Replay each own their logical retry
# budget. The default Responses transport performs one physical attempt so a
# provider outage cannot silently consume several long model requests inside
# one logical invocation. Explicit callers may still supply a bounded policy.
DEFAULT_RESPONSES_MAX_RETRIES = 1
DEFAULT_RESPONSES_TIMEOUT_SECONDS = 600.0
XHIGH_RESPONSES_MAX_RETRIES = 1
XHIGH_RESPONSES_TIMEOUT_SECONDS = 3600.0
# A missing timeout keeps the historical finite default for callers such as
# Replay.  An explicit ``None`` is different: source-aware Lift uses it to
# request an unbounded urllib wait.
_DEFAULT_RESPONSES_TIMEOUT = object()


class ProviderError(RuntimeError):
    """Provider failure with secret-free request accounting evidence."""

    def __init__(
        self,
        message: str,
        *,
        request_sha256: str,
        attempts: int,
        elapsed_ms: float,
        failure_class: str = "provider_transport_error",
    ) -> None:
        super().__init__(message)
        self.request_sha256 = request_sha256
        self.attempts = attempts
        self.elapsed_ms = elapsed_ms
        if not isinstance(failure_class, str) or not failure_class:
            raise ValueError("provider failure_class must be non-empty")
        self.failure_class = failure_class


@dataclass
class LLMReply:
    text: str
    model: str
    raw: dict
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    raw_response: bytes = b""
    attempts: int = 1
    elapsed_ms: float = 0.0


def _content_text_chunks(value: Any) -> list[str]:
    """Extract text items without assuming one provider's response model."""

    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    chunks: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") not in (None, "output_text", "text"):
            continue
        text = item.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return chunks


def _extract_output_text(data: dict) -> str:
    """Pull assistant text from Responses and compatible response envelopes."""

    # Preferred: the SDK convenience field.
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") not in (None, "message"):
                continue
            chunks.extend(_content_text_chunks(item.get("content")))
    if chunks:
        return "".join(chunks)

    # A few OpenAI-compatible relays expose a Chat Completions-shaped envelope
    # even when the endpoint is selected as Responses.  Supporting that shape
    # here prevents a valid model answer from becoming a false empty-completion
    # error; the caller still applies the exact SCRC JSON contract afterwards.
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message")
            if isinstance(message, Mapping):
                chunks.extend(_content_text_chunks(message.get("content")))
            elif isinstance(choice.get("text"), str):
                chunks.append(choice["text"])
    return "".join(chunks)


def _empty_completion_diagnostic(data: Mapping[str, Any]) -> str:
    """Describe an empty model completion without retaining response content."""

    details: list[str] = ["empty completion"]
    status = data.get("status")
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    output = data.get("output")
    if isinstance(output, list):
        item_types = [
            item.get("type")
            for item in output
            if isinstance(item, Mapping) and isinstance(item.get("type"), str)
        ]
        if item_types:
            details.append("output_types=" + ",".join(item_types[:8]))
    choices = data.get("choices")
    if isinstance(choices, list):
        details.append(f"choices={len(choices)}")
    return "; ".join(details)


def _responses_payload(
    instructions: str,
    user_input: str,
    provider: ProviderConfig,
    effort: str | None,
    text_format: Mapping[str, Any] | None,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": provider.model,
        "instructions": instructions,
        "input": user_input,
        "reasoning": {"effort": effort or provider.reasoning_effort},
    }
    if text_format is not None:
        payload["text"] = {"format": dict(text_format)}
    if max_output_tokens is not None:
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        payload["max_output_tokens"] = max_output_tokens
    return payload


def responses_request_body(
    instructions: str,
    user_input: str,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    text_format: Mapping[str, Any] | None = None,
    max_output_tokens: int | None = None,
) -> bytes:
    """Return the exact secret-free JSON bytes sent to ``/responses``."""

    return json.dumps(
        _responses_payload(
            instructions,
            user_input,
            provider,
            effort,
            text_format,
            max_output_tokens,
        ),
        separators=(",", ":"),
    ).encode()


def responses_request_sha256(
    instructions: str,
    user_input: str,
    *,
    provider: ProviderConfig,
    effort: str | None = None,
    text_format: Mapping[str, Any] | None = None,
    max_output_tokens: int | None = None,
) -> str:
    return hashlib.sha256(
        responses_request_body(
            instructions,
            user_input,
            provider=provider,
            effort=effort,
            text_format=text_format,
            max_output_tokens=max_output_tokens,
        )
    ).hexdigest()


def _resolve_responses_request_policy(
    provider: ProviderConfig,
    effort: str | None,
    max_retries: int | None,
    timeout: float | None | object,
) -> tuple[int, float | None]:
    """Resolve one physical transport attempt for the default Lift/Replay path.

    ``xhigh``/``max`` responses can legitimately take longer than the ordinary
    request window, so their timeout remains larger, but neither effort silently
    duplicates an in-flight expensive request. Explicit caller values always
    win.
    """
    effective_effort = (effort or provider.reasoning_effort).strip().lower()
    is_extended_effort = effective_effort in {"xhigh", "max"}
    resolved_retries = max_retries
    if resolved_retries is None:
        resolved_retries = (
            XHIGH_RESPONSES_MAX_RETRIES
            if is_extended_effort
            else DEFAULT_RESPONSES_MAX_RETRIES
        )
    if timeout is _DEFAULT_RESPONSES_TIMEOUT:
        resolved_timeout = (
            XHIGH_RESPONSES_TIMEOUT_SECONDS
            if is_extended_effort
            else DEFAULT_RESPONSES_TIMEOUT_SECONDS
        )
    elif timeout is None:
        resolved_timeout = None
    else:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError(
                "timeout must be None or a finite positive number"
            )
        resolved_timeout = float(timeout)
    if resolved_retries < 1:
        raise ValueError("max_retries must be at least 1")
    return int(resolved_retries), resolved_timeout


def call_responses(
    instructions: str,
    user_input: str,
    *,
    provider: ProviderConfig | None = None,
    effort: str | None = None,
    text_format: Mapping[str, Any] | None = None,
    max_retries: int | None = None,
    timeout: float | None | object = _DEFAULT_RESPONSES_TIMEOUT,
    max_output_tokens: int | None = None,
) -> LLMReply:
    """One ``/responses`` round-trip. ``instructions`` is the system prompt."""
    p = provider or get_provider()
    max_retries, timeout = _resolve_responses_request_policy(
        p, effort, max_retries, timeout
    )
    url = p.base_url.rstrip("/") + "/responses"
    body = responses_request_body(
        instructions,
        user_input,
        provider=p,
        effort=effort,
        text_format=text_format,
        max_output_tokens=max_output_tokens,
    )
    request_sha256 = hashlib.sha256(body).hexdigest()
    headers = {
        "Authorization": f"Bearer {p.api_key}",
        "Content-Type": "application/json",
    }
    last = ""
    failure_class = "provider_transport_error"
    started = time.monotonic()
    attempts_made = 0
    for attempt in range(max_retries):
        attempts_made = attempt + 1
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw_response = r.read()
                data = json.loads(raw_response)
            if not isinstance(data, dict):
                last = "Responses payload is not a JSON object"
                failure_class = "provider_protocol_error"
                break
            text = _extract_output_text(data)
            if not text.strip():
                last = _empty_completion_diagnostic(data)
                failure_class = "provider_no_candidate"
            else:
                return LLMReply(
                    text=text,
                    model=p.model,
                    raw=data,
                    request_sha256=request_sha256,
                    response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    raw_response_sha256=hashlib.sha256(raw_response).hexdigest(),
                    raw_response=raw_response,
                    attempts=attempts_made,
                    elapsed_ms=(time.monotonic() - started) * 1000.0,
                )
        except json.JSONDecodeError:
            last = "Responses endpoint returned invalid JSON"
            failure_class = "provider_protocol_error"
            break
        except urllib.error.HTTPError as e:
            # Do not retain the provider body: relift diagnostics must remain
            # secret-free and must never echo a relay's prompt/error payload.
            last = f"HTTP {e.code}"
            failure_class = (
                "provider_authentication_error"
                if e.code in (401, 403)
                else "provider_http_error"
            )
            if e.code in (400, 401, 403, 404):
                break  # not retryable
        except (urllib.error.URLError, TimeoutError) as e:
            # Keep endpoint details and relay-provided text out of the closed
            # diagnostic.  The typed class below is sufficient for lifecycle
            # handling and avoids leaking transport metadata into evidence.
            last = type(e).__name__
            reason = getattr(e, "reason", None)
            is_timeout = (
                isinstance(e, TimeoutError)
                or isinstance(reason, TimeoutError)
                or "timeout" in type(reason).__name__.casefold()
            )
            failure_class = (
                "provider_timeout"
                if is_timeout
                else "provider_transport_error"
            )
        if attempt < max_retries - 1:
            time.sleep(2.0 * (attempt + 1))
    raise ProviderError(
        f"provider call failed after {attempts_made} tries: {last}",
        request_sha256=request_sha256,
        attempts=attempts_made,
        elapsed_ms=(time.monotonic() - started) * 1000.0,
        failure_class=failure_class,
    )
