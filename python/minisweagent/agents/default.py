"""Basic agent control loop retained for IntentIR Lift and Replay."""

import copy
import hashlib
import json
import logging
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, TimeExceeded
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    max_consecutive_stagnant_steps: int = 0
    """Exit after this many byte-identical no-progress actions (0 = disabled)."""
    material_progress_advisory_seconds: int = 0
    """After this many seconds without authored-byte progress, ask for construction."""
    material_progress_stall_seconds: int = 0
    """Exit after this many seconds without authored-byte progress (0 = disabled)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        if self.config.max_consecutive_stagnant_steps < 0:
            raise ValueError("max_consecutive_stagnant_steps must be non-negative")
        if self.config.material_progress_advisory_seconds < 0:
            raise ValueError("material_progress_advisory_seconds must be non-negative")
        if self.config.material_progress_stall_seconds < 0:
            raise ValueError("material_progress_stall_seconds must be non-negative")
        if (
            self.config.material_progress_stall_seconds
            and self.config.material_progress_advisory_seconds
            >= self.config.material_progress_stall_seconds
        ):
            raise ValueError(
                "material progress advisory must precede the stall timeout"
            )
        self._last_stagnation_fingerprint: str | None = None
        self._consecutive_stagnant_steps = 0
        self._stagnation_observations = 0
        self._stagnation_duplicate_observations = 0
        self._stagnation_terminated = False
        self._start_time = time.time()
        self._material_progress_started = time.monotonic()
        self._material_progress_last_state = self._material_progress_state()
        self._material_progress_events = 0
        self._material_progress_observations = 0
        self._material_progress_advisories = 0
        self._material_progress_advisory_sent = False
        self._material_progress_terminated = False

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        return self._run_loop()

    def run_from_messages(
        self,
        messages: list[dict],
        *,
        task: str = "",
        **kwargs,
    ) -> dict:
        """Continue one audited conversation without recreating its initial prompt."""

        if not messages:
            raise ValueError("continued agent messages must be non-empty")
        if any(not isinstance(message, dict) for message in messages):
            raise TypeError("continued agent messages must be dictionaries")
        if messages[0].get("role") != "system":
            raise ValueError("continued agent history must retain its system message")
        if any(message.get("role") == "exit" for message in messages):
            raise ValueError("continued agent messages must not contain exit records")
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = copy.deepcopy(messages)
        return self._run_loop()

    def _run_loop(self) -> dict:
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                if self.config.output_path is not None:
                    self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        actions = message.get("extra", {}).get("actions", [])
        validate_actions = getattr(self.env, "validate_actions", None)
        diagnostic = (
            validate_actions(actions)
            if len(actions) > 1 and callable(validate_actions)
            else None
        )
        if diagnostic is None:
            outputs = [self.env.execute(action) for action in actions]
        else:
            outputs = [
                {
                    "output": diagnostic + "\n",
                    "returncode": 2,
                    "exception_info": (
                        "the whole multi-action response was rejected before "
                        "execution"
                    ),
                }
                for _action in actions
            ]
        observations = self.add_messages(
            *self.model.format_observation_messages(
                message,
                outputs,
                self.get_template_vars(),
            )
        )
        self._observe_stagnation(actions, outputs)
        self._observe_material_progress()
        return observations

    @staticmethod
    def _state_digest(state: object) -> str:
        encoded = json.dumps(
            state,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _material_progress_state(self) -> str | None:
        reader = getattr(self.env, "material_progress_state", None)
        if not callable(reader):
            return None
        return self._state_digest(reader())

    def _observe_material_progress(self) -> None:
        """Govern orchestration by time since exact authored bytes last changed.

        The environment exposes only byte identities of the owner's writable
        artifact/work files.  This hook neither parses nor judges their semantics,
        and it does not cap total model/check calls while bytes keep changing.
        """

        advisory = self.config.material_progress_advisory_seconds
        stall = self.config.material_progress_stall_seconds
        if advisory == 0 and stall == 0:
            return
        current = self._material_progress_state()
        if current is None:
            return
        self._material_progress_observations += 1
        if current != self._material_progress_last_state:
            self._material_progress_last_state = current
            self._material_progress_started = time.monotonic()
            self._material_progress_events += 1
            self._material_progress_advisory_sent = False
            return
        elapsed = time.monotonic() - self._material_progress_started
        if stall and elapsed >= stall:
            self._material_progress_terminated = True
            self.add_messages(
                {
                    "role": "exit",
                    "content": "OrchestrationStall",
                    "extra": {
                        "exit_status": "OrchestrationStall",
                        "submission": "",
                        "reason": "material_progress_timeout",
                        "no_material_progress_seconds": elapsed,
                        "material_state_sha256": current,
                    },
                }
            )
            return
        if advisory and elapsed >= advisory and not self._material_progress_advisory_sent:
            self._material_progress_advisory_sent = True
            self._material_progress_advisories += 1
            directive = json.dumps(
                {
                    "protocol": "intentir.agent-material-progress.v1",
                    "status": "construction_due",
                    "no_material_progress_seconds": int(elapsed),
                    "material_state_sha256": current,
                    "directive": (
                        "Stop rediscovering unchanged inputs. Author or edit the "
                        "complete current artifact now, then run its closed checker."
                    ),
                    "authority": (
                        "orchestration-only exact-byte timer; no semantic field, "
                        "operator, relation, or repair is inferred"
                    ),
                },
                sort_keys=True,
            )
            self.add_messages(self.model.format_message(role="user", content=directive))

    @staticmethod
    def _stable_action(action: object) -> object:
        """Drop provider-allocated call ids while preserving exact requested work."""

        if isinstance(action, dict):
            return {
                key: DefaultAgent._stable_action(value)
                for key, value in sorted(action.items())
                if key != "tool_call_id"
            }
        if isinstance(action, list):
            return [DefaultAgent._stable_action(value) for value in action]
        return action

    def _observe_stagnation(self, actions: list[dict], outputs: list[dict]) -> None:
        """Stop only an exact, consecutive orchestration loop.

        The optional environment state contains only exact artifact/source byte
        digests.  Tool requests and closed outputs must also be byte-identical;
        this hook never interprets whether a semantic edit is good or bad.
        """

        limit = self.config.max_consecutive_stagnant_steps
        if limit == 0:
            return
        state_reader = getattr(self.env, "stagnation_state", None)
        state = state_reader() if callable(state_reader) else None
        payload = {
            "actions": self._stable_action(actions),
            "outputs": outputs,
            "environment_state": state,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()
        self._stagnation_observations += 1
        if fingerprint == self._last_stagnation_fingerprint:
            self._consecutive_stagnant_steps += 1
            self._stagnation_duplicate_observations += 1
        else:
            self._last_stagnation_fingerprint = fingerprint
            self._consecutive_stagnant_steps = 1
        if self._consecutive_stagnant_steps < limit:
            return
        self._stagnation_terminated = True
        self.add_messages(
            {
                "role": "exit",
                "content": "OrchestrationStall",
                "extra": {
                    "exit_status": "OrchestrationStall",
                    "submission": "",
                    "stagnation_fingerprint": fingerprint,
                    "consecutive_stagnant_steps": self._consecutive_stagnant_steps,
                },
            }
        )

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
                "agent_runtime": {
                    "stagnation": {
                        "protocol": "minisweagent.exact-stagnation.v1",
                        "authority": (
                            "exact consecutive action/output and artifact/source "
                            "byte-digest equality only; no semantic judgement"
                        ),
                        "limit": self.config.max_consecutive_stagnant_steps,
                        "observations": self._stagnation_observations,
                        "no_progress_duplicates": (
                            self._stagnation_duplicate_observations
                        ),
                        "no_progress_duplicate_ratio": (
                            self._stagnation_duplicate_observations
                            / self._stagnation_observations
                            if self._stagnation_observations
                            else 0.0
                        ),
                        "consecutive": self._consecutive_stagnant_steps,
                        "terminated": self._stagnation_terminated,
                        "last_fingerprint": self._last_stagnation_fingerprint,
                    },
                    "material_progress": {
                        "protocol": "intentir.agent-material-progress.v1",
                        "authority": (
                            "exact writable-byte identity and elapsed time only; "
                            "no semantic judgement or total call limit"
                        ),
                        "advisory_seconds": (
                            self.config.material_progress_advisory_seconds
                        ),
                        "stall_seconds": self.config.material_progress_stall_seconds,
                        "observations": self._material_progress_observations,
                        "byte_state_changes": self._material_progress_events,
                        "advisories": self._material_progress_advisories,
                        "terminated": self._material_progress_terminated,
                        "last_state_sha256": self._material_progress_last_state,
                    },
                },
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
