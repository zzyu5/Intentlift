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

# The active Gate commands historically select the provider as ``cpa``.  This
# name is only the process-environment fallback; a real provider file remains
# authoritative whenever one is supplied.
DEFAULT_ENV_PROVIDER_NAME = "cpa"
DEFAULT_ENV_PROVIDER_MODEL = "gpt-5.4"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    wire_api: str = "responses"
    reasoning_effort: str = "high"


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
        raise FileNotFoundError(f"LLM provider config not found: {p}")
    raw = json.loads(p.read_text())
    out: dict[str, ProviderConfig] = {}
    for name, cfg in raw.items():
        out[name] = ProviderConfig(
            name=name,
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg.get("model", name),
            wire_api=cfg.get("wire_api", "responses"),
            reasoning_effort=cfg.get("reasoning_effort", "high"),
        )
    return out


def get_provider(
    name: str | None = None, path: os.PathLike | str | None = None
) -> ProviderConfig:
    """Return one provider config; defaults to the first configured provider."""
    providers = load_providers(path)
    if not providers:
        raise ValueError("no providers configured")
    if name is None:
        name = next(iter(providers))
    if name not in providers:
        raise KeyError(f"provider {name!r} not in {list(providers)}")
    return providers[name]
