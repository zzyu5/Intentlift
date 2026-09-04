"""Compose the in-repository mini-swe components for one IntentIR Lift round."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from minisweagent import Environment, Model, package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.intentir_lift import (
    IntentIRLiftEnvironment,
    IntentIRLiftWorkspace,
)
from minisweagent.models.litellm_response_model import LitellmResponseModel
from minisweagent.utils.serialize import recursive_merge


@dataclass(frozen=True)
class IntentIRLiftRunResult:
    candidate_text: str | None
    trajectory: Mapping[str, Any]
    exit_status: str
    model_calls: int
    elapsed_ms: float
    physical_provider_calls: int | None = None
    transport_retries: int = 0
    stage_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    continued_from_messages: int = 0

    def __post_init__(self) -> None:
        if self.physical_provider_calls is None:
            object.__setattr__(
                self, "physical_provider_calls", self.model_calls
            )


class IntentIRLiftRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        result: IntentIRLiftRunResult,
        cause: BaseException | None = None,
    ) -> None:
        self.result = result
        self.cause = cause
        super().__init__(message)


@dataclass(frozen=True)
class _IntentIRLiftStageResult:
    output_text: str | None
    trajectory: Mapping[str, Any]
    exit_status: str
    model_calls: int
    elapsed_ms: float
    physical_provider_calls: int
    transport_retries: int
    stage_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    caught: Exception | None = None


def load_intentir_lift_config() -> dict[str, Any]:
    path = Path(package_dir) / "config" / "intentir_lift.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("IntentIR Lift config must be a mapping")
    return value


def _provider_model_turn_metrics(
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize provider-turn timings already recorded by the model."""

    info = trajectory.get("info", {})
    runtime = info.get("agent_runtime", {}) if isinstance(info, Mapping) else {}
    attempts = runtime.get("provider_attempts", []) if isinstance(runtime, Mapping) else []
    elapsed_ms = 0.0
    count = 0
    for attempt in attempts if isinstance(attempts, list) else []:
        if not isinstance(attempt, Mapping):
            continue
        value = attempt.get("elapsed_ms")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            elapsed_ms += max(0.0, float(value))
            count += 1
    return {"elapsed_ms": elapsed_ms, "count": count}


def create_intentir_lift_model(
    *,
    model_name: str,
    api_base: str,
    api_key: str,
    reasoning_effort: str,
    timeout_seconds: float | None,
    prompt_cache_key: str | None = None,
) -> LitellmResponseModel:
    """Configure the existing Responses model for an OpenAI-compatible CPA."""

    config = load_intentir_lift_config()["model"]
    model_kwargs = {
        **config.pop("model_kwargs", {}),
        "api_base": api_base,
        "api_key": api_key,
        "reasoning_effort": reasoning_effort,
        # Keep the config JSON-serializable.  The existing Responses model
        # converts an explicit None to an all-unbounded httpx Timeout only at
        # the provider call boundary, where LiteLLM's ``timeout or`` fallback
        # cannot reintroduce its finite default.
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


def _run_agent_stage(
    *,
    model: Model,
    environment: Environment,
    agent_config: Mapping[str, Any],
    output_reader: Callable[[], str],
    initial_messages: list[dict[str, Any]] | None = None,
) -> _IntentIRLiftStageResult:
    """Run one fresh or branch-continuation context and read its opaque output."""

    stage_started = time.monotonic()
    reset_execution_metrics = getattr(
        environment, "reset_stage_execution_metrics", None
    )
    if callable(reset_execution_metrics):
        reset_execution_metrics()
    physical_before = getattr(model, "physical_provider_calls", None)
    retries_before = getattr(model, "transport_retries", None)
    initialization_started = time.monotonic()
    agent = DefaultAgent(model, environment, **dict(agent_config))
    initialization_elapsed_ms = (time.monotonic() - initialization_started) * 1000.0
    agent_started = time.monotonic()
    caught: Exception | None = None
    try:
        terminal = (
            agent.run()
            if initial_messages is None
            else agent.run_from_messages(initial_messages)
        )
    except Exception as exc:
        caught = exc
        terminal = {"exit_status": type(exc).__name__, "submission": ""}
    agent_elapsed_ms = (time.monotonic() - agent_started) * 1000.0
    exit_status = str(terminal.get("exit_status", ""))
    output_text: str | None = None
    candidate_read_started = time.monotonic()
    if exit_status == "Submitted":
        try:
            output_text = output_reader()
        except Exception as exc:
            caught = exc
            exit_status = "InvalidSubmission"
    elif caught is None:
        activate_fallback = getattr(
            environment, "activate_committed_fallback", None
        )
        if callable(activate_fallback):
            try:
                output_text = activate_fallback(exit_status)
            except Exception as exc:
                caught = exc
                exit_status = "InvalidSubmission"
            else:
                if output_text is not None:
                    exit_status = "SubmittedCheckpoint"
    candidate_read_elapsed_ms = (
        time.monotonic() - candidate_read_started
    ) * 1000.0
    serialization_started = time.monotonic()
    trajectory = agent.save(None)
    serialization_elapsed_ms = (time.monotonic() - serialization_started) * 1000.0
    physical_after = getattr(model, "physical_provider_calls", None)
    retries_after = getattr(model, "transport_retries", None)
    physical_calls = (
        physical_after - physical_before
        if isinstance(physical_before, int)
        and isinstance(physical_after, int)
        and physical_after >= physical_before
        else agent.n_calls
    )
    transport_retries = (
        retries_after - retries_before
        if isinstance(retries_before, int)
        and isinstance(retries_after, int)
        and retries_after >= retries_before
        else 0
    )
    action_metrics = getattr(environment, "stage_execution_metrics", None)
    if callable(action_metrics):
        action_metrics = action_metrics()
    if not isinstance(action_metrics, Mapping):
        action_metrics = {"elapsed_ms": 0.0, "count": 0, "failures": 0}
    stage_metrics = {
        "agent_initialization": {
            "elapsed_ms": initialization_elapsed_ms,
        },
        "agent_loop": {
            "elapsed_ms": agent_elapsed_ms,
            "model_calls": agent.n_calls,
        },
        "provider_model_turn": _provider_model_turn_metrics(trajectory),
        "tool_action": dict(action_metrics),
        "candidate_read": {
            "elapsed_ms": candidate_read_elapsed_ms,
        },
        "trajectory_serialization": {
            "elapsed_ms": serialization_elapsed_ms,
        },
        "lift_stage_total": {
            "elapsed_ms": (time.monotonic() - stage_started) * 1000.0,
        },
    }
    return _IntentIRLiftStageResult(
        output_text=output_text,
        trajectory=trajectory,
        exit_status=exit_status,
        model_calls=agent.n_calls,
        elapsed_ms=stage_metrics["lift_stage_total"]["elapsed_ms"],
        physical_provider_calls=physical_calls,
        transport_retries=transport_retries,
        stage_metrics=stage_metrics,
        caught=caught,
    )


def _public_result(
    stage: _IntentIRLiftStageResult,
    *,
    continued_from_messages: int = 0,
) -> IntentIRLiftRunResult:
    return IntentIRLiftRunResult(
        candidate_text=stage.output_text,
        trajectory=stage.trajectory,
        exit_status=stage.exit_status,
        model_calls=stage.model_calls,
        elapsed_ms=stage.elapsed_ms,
        physical_provider_calls=stage.physical_provider_calls,
        transport_retries=stage.transport_retries,
        stage_metrics=stage.stage_metrics,
        continued_from_messages=continued_from_messages,
    )


def run_intentir_lift(
    *,
    model: Model,
    workspace: IntentIRLiftWorkspace,
    config_overrides: Mapping[str, Any] | None = None,
    environment: Environment | None = None,
    candidate_checker: Callable[[str], Mapping[str, Any]] | None = None,
    continuation_history: list[dict[str, Any]] | None = None,
    continuation_message: str | None = None,
) -> IntentIRLiftRunResult:
    """Run a fresh or branch-continuation agent and return one full candidate."""

    config = recursive_merge(
        load_intentir_lift_config(), dict(config_overrides or {})
    )
    if environment is not None and candidate_checker is not None:
        raise ValueError(
            "candidate_checker must be installed by the supplied environment"
        )
    environment = environment or IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        candidate_checker=candidate_checker,
        **config.get("environment", {}),
    )
    if (continuation_history is None) != (continuation_message is None):
        raise ValueError(
            "continuation_history and continuation_message must be supplied together"
        )
    initial_messages = None
    if continuation_history is not None and continuation_message is not None:
        initial_messages = [
            *continuation_history,
            model.format_message(role="user", content=continuation_message),
        ]
    stage = _run_agent_stage(
        model=model,
        environment=environment,
        agent_config=config["agent"],
        output_reader=workspace.read_candidate,
        initial_messages=initial_messages,
    )
    result = _public_result(
        stage,
        continued_from_messages=(
            len(continuation_history)
            if continuation_history is not None
            else 0
        ),
    )
    if stage.caught is not None:
        raise IntentIRLiftRunError(
            "IntentIR Lift worker failed: "
            f"{type(stage.caught).__name__}: {stage.caught}",
            result=result,
            cause=stage.caught,
        ) from stage.caught
    if stage.exit_status not in {"Submitted", "SubmittedCheckpoint"} or stage.output_text is None:
        raise IntentIRLiftRunError(
            "IntentIR Lift worker terminated without a candidate: "
            f"{stage.exit_status}",
            result=result,
        )
    return result


__all__ = [
    "create_intentir_lift_model",
    "IntentIRLiftRunError",
    "IntentIRLiftRunResult",
    "load_intentir_lift_config",
    "run_intentir_lift",
]
