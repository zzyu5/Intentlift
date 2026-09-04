"""Compose a fresh mini-swe episode for one source-blind Replay generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from minisweagent import Environment, Model, package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.intentir_replay import (
    IntentIRReplayEnvironment,
    IntentIRReplayWorkspace,
)
from minisweagent.models.litellm_response_model import LitellmResponseModel
from minisweagent.utils.serialize import recursive_merge


_ALLOWED_OVERRIDE_KEYS = {
    "agent": {"step_limit", "wall_time_limit_seconds"},
    "environment": {
        "timeout",
        "replay_check_limit",
        "replay_submission_check_limit",
        "replay_compile_check_limit",
        "replay_source_inspect_limit",
    },
}
_OVERRIDE_BOUNDS = {
    ("agent", "step_limit"): (1, 64),
    ("agent", "wall_time_limit_seconds"): (1, 7200),
    ("environment", "timeout"): (1, 300),
    ("environment", "replay_check_limit"): (0, 32),
    ("environment", "replay_submission_check_limit"): (0, 4),
    ("environment", "replay_compile_check_limit"): (0, 16),
    ("environment", "replay_source_inspect_limit"): (0, 32),
}
@dataclass(frozen=True)
class IntentIRReplayRunResult:
    source_text: str | None
    trajectory: Mapping[str, Any]
    exit_status: str
    model_calls: int
    elapsed_ms: float
    physical_provider_calls: int | None = None
    transport_retries: int = 0

    def __post_init__(self) -> None:
        if self.physical_provider_calls is None:
            object.__setattr__(
                self, "physical_provider_calls", self.model_calls
            )


class IntentIRReplayRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        result: IntentIRReplayRunResult,
        cause: BaseException | None = None,
    ) -> None:
        self.result = result
        self.cause = cause
        super().__init__(message)


def load_intentir_replay_config() -> dict[str, Any]:
    path = Path(package_dir) / "config" / "intentir_replay.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("IntentIR Replay config must be a mapping")
    return value


def create_intentir_replay_model(
    *,
    model_name: str,
    api_base: str,
    api_key: str,
    reasoning_effort: str,
    timeout_seconds: float,
    prompt_cache_key: str | None = None,
) -> LitellmResponseModel:
    config = load_intentir_replay_config()["model"]
    model_kwargs = {
        **config.pop("model_kwargs", {}),
        "api_base": api_base,
        "api_key": api_key,
        "reasoning_effort": reasoning_effort,
        "timeout": timeout_seconds,
    }
    if prompt_cache_key:
        model_kwargs["prompt_cache_key"] = prompt_cache_key
    qualified_name = model_name if "/" in model_name else f"openai/{model_name}"
    return LitellmResponseModel(
        model_name=qualified_name,
        model_kwargs=model_kwargs,
        **config,
    )


def validated_intentir_replay_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply only bounded budget overrides to the fixed Replay config."""

    supplied = dict(overrides or {})
    unknown_sections = sorted(set(supplied) - set(_ALLOWED_OVERRIDE_KEYS))
    if unknown_sections:
        raise ValueError(
            "Replay worker overrides contain forbidden section(s): "
            + ", ".join(unknown_sections)
        )
    normalized: dict[str, dict[str, int]] = {}
    for section, raw_values in supplied.items():
        if not isinstance(raw_values, Mapping):
            raise TypeError(f"Replay worker override {section} must be a mapping")
        unknown_keys = sorted(
            set(raw_values) - _ALLOWED_OVERRIDE_KEYS[section]
        )
        if unknown_keys:
            raise ValueError(
                f"Replay worker override {section} has forbidden field(s): "
                + ", ".join(unknown_keys)
            )
        normalized[section] = {}
        for key, value in raw_values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"Replay worker override {section}.{key} must be an integer"
                )
            lower, upper = _OVERRIDE_BOUNDS[(section, key)]
            if not lower <= value <= upper:
                raise ValueError(
                    f"Replay worker override {section}.{key} must be in "
                    f"[{lower}, {upper}]"
                )
            normalized[section][key] = value
    config = recursive_merge(load_intentir_replay_config(), normalized)
    if config.get("agent", {}).get("output_path") is not None:
        raise ValueError("Replay agent output_path must remain disabled")
    return config


def _replay_startup_context(workspace: IntentIRReplayWorkspace) -> str:
    """Return only immutable paths and byte counts; the agent reads files itself."""

    packet_bytes = (workspace.input_dir / "packet.json").stat().st_size
    semantic_bytes = (
        workspace.input_dir / "semantic-program.intent"
    ).stat().st_size
    exact: dict[str, Any] = {
        "protocol": "intentir.replay-startup-context.v6",
        "authority": (
            "path and byte-count receipt only; immutable exact inputs remain in "
            "/input and no semantic or target program is embedded here"
        ),
        "packet_bytes": packet_bytes,
        "semantic_program_bytes": semantic_bytes,
        "paths": {
            "semantic_program": "/input/semantic-program.intent",
            "semantic_program_json": "/input/semantic-program.json",
            "packet": "/input/packet.json",
            "frontend": "/input/frontend.json",
            "contract": "/input/REPLAY_CONTRACT.md",
            "tools": "/input/TOOL.md",
            "replay": "/output/replay.source",
        },
    }
    if (workspace.input_dir / "c.json").is_file():
        exact["paths"]["c"] = "/input/c.json"
    return json.dumps(
        exact,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def run_intentir_replay(
    *,
    model: Model,
    workspace: IntentIRReplayWorkspace,
    config_overrides: Mapping[str, Any] | None = None,
    environment: Environment | None = None,
    replay_checker: Callable[[str], Mapping[str, Any]] | None = None,
    replay_compile_checker: Callable[[str], Mapping[str, Any]] | None = None,
) -> IntentIRReplayRunResult:
    """Run one new DefaultAgent; no messages or workspace survive this call."""

    config = validated_intentir_replay_config(config_overrides)
    if environment is not None and (
        replay_checker is not None or replay_compile_checker is not None
    ):
        raise ValueError(
            "Replay checkers must be installed by the supplied environment"
        )
    environment = environment or IntentIRReplayEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        replay_checker=replay_checker,
        replay_compile_checker=replay_compile_checker,
        **config.get("environment", {}),
    )
    agent = DefaultAgent(model, environment, **config["agent"])
    started = time.monotonic()
    caught: Exception | None = None
    try:
        terminal = agent.run(
            replay_startup_context=_replay_startup_context(workspace)
        )
    except Exception as exc:
        caught = exc
        terminal = {"exit_status": type(exc).__name__, "submission": ""}
    exit_status = str(terminal.get("exit_status", ""))
    source_text: str | None = None
    if exit_status == "Submitted":
        try:
            source_text = workspace.read_replay()
        except Exception as exc:
            caught = exc
            exit_status = "InvalidSubmission"
    elif caught is None:
        activate_fallback = getattr(
            environment, "activate_committed_fallback", None
        )
        if callable(activate_fallback):
            try:
                source_text = activate_fallback(exit_status)
            except Exception as exc:
                caught = exc
                exit_status = "InvalidSubmission"
            else:
                if source_text is not None:
                    exit_status = "SubmittedCheckpoint"
    elapsed_ms = (time.monotonic() - started) * 1000.0
    trajectory = agent.save(None)
    result = IntentIRReplayRunResult(
        source_text=source_text,
        trajectory=trajectory,
        exit_status=exit_status,
        model_calls=agent.n_calls,
        elapsed_ms=elapsed_ms,
        physical_provider_calls=getattr(
            model, "physical_provider_calls", agent.n_calls
        ),
        transport_retries=getattr(model, "transport_retries", 0),
    )
    if caught is not None:
        raise IntentIRReplayRunError(
            f"IntentIR Replay worker failed: {type(caught).__name__}: {caught}",
            result=result,
            cause=caught,
        ) from caught
    if exit_status not in {"Submitted", "SubmittedCheckpoint"} or source_text is None:
        raise IntentIRReplayRunError(
            f"IntentIR Replay worker terminated without source: {exit_status}",
            result=result,
        )
    return result


__all__ = [
    "IntentIRReplayRunError",
    "IntentIRReplayRunResult",
    "create_intentir_replay_model",
    "load_intentir_replay_config",
    "run_intentir_replay",
    "validated_intentir_replay_config",
]
