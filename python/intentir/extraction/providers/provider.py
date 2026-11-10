"""LLM provider configuration loading.

Reads ``llm_providers.local.json`` (gitignored; holds API keys) from the repo
root and exposes a typed config. Actual HTTP calling lives in a client built on
top of this; kept dependency-free here so config can load without a network or
extra packages.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# provider.py -> extraction -> intentir -> python -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROVIDERS_PATH = REPO_ROOT / "llm_providers.local.json"
PROVIDERS_PATH_ENV = "INTENTIR_PROVIDER_CONFIG"
PROVIDER_NAME_ENV = "INTENTIR_PROVIDER_NAME"
PROVIDER_MODEL_ENV = "INTENTIR_PROVIDER_MODEL"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_MODEL_ENV = "OPENAI_MODEL"

# This is only the label for an explicitly environment-backed provider.  A
# provider file remains authoritative whenever one is supplied.
DEFAULT_ENV_PROVIDER_NAME = "openai"
DEFAULT_PROVIDER_NAME = "gpt"
# The environment-backed provider is used when a managed runner injects only
# OPENAI_BASE_URL and OPENAI_API_KEY.  Keep its fallback aligned with the
# repository's explicit gpt Responses route; callers may still select a
# different listed model through OPENAI_MODEL.
DEFAULT_ENV_PROVIDER_MODEL = "gpt-5.6-terra"


class ProviderConfigurationError(ValueError):
    """Closed provider-selection failure without exposing config contents."""

    failure_class = "provider_configuration_unavailable"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    wire_api: str = "responses"
    reasoning_effort: str = "high"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("provider name must not be empty")
        if not self.base_url.strip():
            raise ValueError(f"provider {self.name!r} base_url must not be empty")
        if not self.model.strip():
            raise ValueError(f"provider {self.name!r} model must not be empty")


def _environment_provider() -> ProviderConfig | None:
    """Build an in-process provider without persisting credential material.

    The normal source of provider configuration is the ignored JSON file.  A
    caller may instead inject an OpenAI-compatible endpoint and key through
    the process environment, which is useful for ephemeral runners and keeps
    the key out of the repository and all evidence artifacts.
    """
    base_url = os.environ.get(OPENAI_BASE_URL_ENV, "").strip()
    api_key = os.environ.get(OPENAI_API_KEY_ENV, "").strip()
    if not base_url or not api_key:
        return None
    name = (
        os.environ.get(PROVIDER_NAME_ENV)
        or os.environ.get("OPENAI_PROVIDER_NAME")
        or DEFAULT_ENV_PROVIDER_NAME
    ).strip()
    model = (
        os.environ.get(PROVIDER_MODEL_ENV)
        or os.environ.get(OPENAI_MODEL_ENV)
        or DEFAULT_ENV_PROVIDER_MODEL
    ).strip()
    if not name or not model:
        return None
    return ProviderConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
        wire_api="responses",
        reasoning_effort="high",
    )


def get_environment_provider() -> ProviderConfig:
    """Select only the injected transport, never a local-file namesake."""

    provider = _environment_provider()
    if provider is None:
        raise ProviderConfigurationError(
            "OpenAI environment transport is not configured"
        )
    return provider


def load_providers(path: os.PathLike | str | None = None) -> dict[str, ProviderConfig]:
    configured = os.environ.get(PROVIDERS_PATH_ENV)
    if path is not None:
        p = Path(path)
    elif configured:
        p = Path(configured).expanduser()
    else:
        p = DEFAULT_PROVIDERS_PATH
    if not p.exists():
        # An explicitly supplied path must remain authoritative and fail
        # closed.  Only the ordinary default lookup can use ephemeral process
        # credentials as a fallback.
        if path is None and not configured:
            env_provider = _environment_provider()
            if env_provider is not None:
                return {env_provider.name: env_provider}
        raise ProviderConfigurationError("LLM provider config is unavailable")
    try:
        raw = json.loads(p.read_text())
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ProviderConfigurationError("LLM provider config is malformed") from exc
    if not isinstance(raw, dict):
        raise ProviderConfigurationError("LLM provider config is malformed")
    out: dict[str, ProviderConfig] = {}
    for name, cfg in raw.items():
        try:
            out[name] = ProviderConfig(
                name=name,
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
                model=cfg.get("model", name),
                wire_api=cfg.get("wire_api", "responses"),
                reasoning_effort=cfg.get("reasoning_effort", "high"),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ProviderConfigurationError("LLM provider config is malformed") from exc
    # A repository-local provider file remains the default and is authoritative
    # when a caller supplied it explicitly.  For the ordinary default lookup,
    # also expose an environment-injected provider under its own name.  This
    # lets a runner explicitly select a healthy temporary/managed endpoint via
    # INTENTIR_PROVIDER_NAME without serializing its credentials into a local
    # JSON file or silently replacing the configured default transport.
    if path is None and not configured:
        env_provider = _environment_provider()
        if env_provider is not None:
            out.setdefault(env_provider.name, env_provider)
    return out


def get_provider(
    name: str | None = None, path: os.PathLike | str | None = None
) -> ProviderConfig:
    """Return one configured provider, defaulting to local ``gpt``.

    An omitted provider never inherits the first entry of a local credentials
    file, because file ordering cannot choose the active transport.
    """
    providers = load_providers(path)
    if not providers:
        raise ProviderConfigurationError("no providers configured")
    if name is None:
        name = (
            os.environ.get(PROVIDER_NAME_ENV, "").strip()
            or DEFAULT_PROVIDER_NAME
        )
    if name not in providers:
        raise ProviderConfigurationError("requested provider is unavailable")
    return providers[name]
