"""Isolated Codex CLI execution for IntentIR's proof-irrelevant agent runtime.

The CLI owns its model/tool loop, while this launcher owns the filesystem,
credential, timeout, and evidence boundaries.  It never interprets or edits an
IntentIR artifact or Replay source.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .provider import ProviderConfig


CODEX_CLI_AGENT_RUNTIME = "intentir.codex-cli.agent-runtime.v5"
CODEX_CLI_TOOL_BRIDGE_PROTOCOL = "intentir.codex-cli.tool-bridge.v3"
CODEX_CLI_TOOL_BRIDGE_ROOT_NAME = "intentir-agent-tools"
MAX_CODEX_STDOUT_BYTES = 32 * 1024 * 1024
MAX_CODEX_STDERR_BYTES = 1024 * 1024
MAX_CODEX_TOOL_REQUEST_BYTES = 4096
MAX_CODEX_TOOL_RESPONSE_BYTES = 1024 * 1024
MAX_CODEX_TOOL_ARGUMENTS = 4
MAX_CODEX_TOOL_ARGUMENT_CHARS = 256
_CODEX_HOME_PREFIX = "intentir-codex-home-"
_CODEX_TOOL_ROOT_PREFIX = "intentir-codex-tools-"


class CodexCLIError(RuntimeError):
    """Fail-closed runtime error without provider secrets or model payloads."""


class CodexCLICancelled(CodexCLIError):
    """One audited CLI invocation was stopped after a sibling Gate1 closure."""

    def __init__(
        self,
        winner_branch_id: int,
        *,
        elapsed_ms: float,
        stdout_sha256: str,
        stderr_sha256: str,
    ) -> None:
        if not isinstance(winner_branch_id, int) or winner_branch_id < 1:
            raise ValueError("winner_branch_id must be positive")
        self.winner_branch_id = winner_branch_id
        self.elapsed_ms = elapsed_ms
        self.stdout_sha256 = stdout_sha256
        self.stderr_sha256 = stderr_sha256
        self.physical_calls = 1
        super().__init__(
            f"candidate superseded by Gate1 winner branch {winner_branch_id}"
        )


class CodexCLISubmissionAccepted(CodexCLIError):
    """A closed host tool accepted exact bytes, so final model prose was skipped."""

    def __init__(
        self,
        *,
        elapsed_ms: float,
        thread_id: str,
    ) -> None:
        self.elapsed_ms = elapsed_ms
        self.thread_id = thread_id
        self.physical_calls = 1
        super().__init__("exact-byte submission accepted by the closed host tool")


class CodexCLIDiagnosticReady(CodexCLIError):
    """Stable exact bytes were rejected and a same-thread repair can start now."""

    def __init__(
        self,
        *,
        elapsed_ms: float,
        thread_id: str,
    ) -> None:
        self.elapsed_ms = elapsed_ms
        self.thread_id = thread_id
        self.physical_calls = 1
        super().__init__("exact-byte diagnostic is ready for same-thread repair")


def _codex_tool_client_source() -> str:
    """Return the fixed client for closed host diagnostics inside Codex's sandbox."""

    return '''from __future__ import annotations
import json
from pathlib import Path
import secrets
import sys
import time

PROTOCOL = "intentir.codex-cli.tool-bridge.v3"
ROOT = Path("/workspace/work/intentir-agent-tools")
REQUESTS = ROOT / "requests"
RESPONSES = ROOT / "responses"
LIMIT = 1024 * 1024
TIMEOUT = 120.0

if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
    print("usage: intentir_agent_tool.py TOOL [ARG ...]")
    raise SystemExit(0)
if len(sys.argv) < 2 or len(sys.argv) > 2 + 4 or not sys.argv[1] or len(sys.argv[1]) > 64:
    raise SystemExit("usage: intentir_agent_tool.py TOOL [ARG ...]")
arguments = sys.argv[2:]
if any(not item or len(item) > 256 for item in arguments):
    raise SystemExit("IntentIR tool arguments are empty or too long")
request_id = secrets.token_hex(16)
request = json.dumps(
    {
        "protocol": PROTOCOL,
        "request_id": request_id,
        "tool": sys.argv[1],
        "arguments": arguments,
    },
    sort_keys=True,
).encode() + b"\\n"
temporary = REQUESTS / ("." + request_id + ".tmp")
request_path = REQUESTS / (request_id + ".json")
response_path = RESPONSES / (request_id + ".json")
temporary.write_bytes(request)
temporary.replace(request_path)
deadline = time.monotonic() + TIMEOUT
while not response_path.exists():
    if time.monotonic() >= deadline:
        raise SystemExit("IntentIR tool response timed out")
    time.sleep(0.02)
if response_path.stat().st_size > LIMIT:
    raise SystemExit("IntentIR tool response exceeded its byte limit")
raw = response_path.read_bytes()
response_path.unlink(missing_ok=True)
envelope = json.loads(raw)
if (
    envelope.get("protocol") != PROTOCOL
    or envelope.get("request_id") != request_id
    or envelope.get("tool") != sys.argv[1]
):
    raise SystemExit("IntentIR tool response protocol mismatch")
result = envelope.get("result")
if not isinstance(result, dict):
    raise SystemExit("IntentIR tool response lacks a result object")
print(json.dumps(result, sort_keys=True, ensure_ascii=False))
raise SystemExit(0 if result.get("status") == "pass" else 1)
'''


@dataclass
class CodexCLIToolBridge:
    """Closed file-mailbox bridge to diagnostic-only host checks.

    Codex's inner workspace-write sandbox intentionally denies every socket while
    the outer process still needs provider network access.  A mailbox under the
    already isolated writable ``work/`` mount preserves both boundaries: the model
    can name one allowlisted tool, and the host reads no model-selected path.
    """

    root_path: Path
    handlers: Mapping[str, Callable[[], Mapping[str, Any]]]
    argument_handlers: Mapping[
        str, Callable[[tuple[str, ...]], Mapping[str, Any]]
    ] = field(default_factory=dict)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _receipts: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root_path = Path(self.root_path).resolve(strict=False)
        if self.root_path.name != CODEX_CLI_TOOL_BRIDGE_ROOT_NAME:
            raise ValueError("Codex tool bridge must use the fixed allowlisted name")
        parent = self.root_path.parent.resolve(strict=True)
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError("Codex tool bridge parent must be a real directory")
        normalized = dict(self.handlers)
        normalized_arguments = dict(self.argument_handlers)
        if not normalized and not normalized_arguments:
            raise ValueError("Codex tool bridge needs at least one closed handler")
        if any(
            not isinstance(name, str)
            or not name
            or len(name) > 64
            or not callable(handler)
            for name, handler in normalized.items()
        ):
            raise ValueError("Codex tool bridge needs closed named handlers")
        if any(
            not isinstance(name, str)
            or not name
            or len(name) > 64
            or not callable(handler)
            for name, handler in normalized_arguments.items()
        ):
            raise ValueError("Codex tool bridge needs closed argument handlers")
        if set(normalized) & set(normalized_arguments):
            raise ValueError("Codex tool bridge tool names must be unique")
        self.handlers = normalized
        self.argument_handlers = normalized_arguments

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Codex tool bridge is already running")
        if self.root_path.exists() or self.root_path.is_symlink():
            raise CodexCLIError("Codex tool bridge path is unexpectedly occupied")
        try:
            self.root_path.mkdir(mode=0o700)
            self.requests_path.mkdir(mode=0o700)
            self.responses_path.mkdir(mode=0o700)
        except Exception:
            shutil.rmtree(self.root_path, ignore_errors=True)
            raise
        self._thread = threading.Thread(
            target=self._serve,
            name="intentir-codex-tool-bridge",
            daemon=True,
        )
        self._thread.start()

    @property
    def requests_path(self) -> Path:
        return self.root_path / "requests"

    @property
    def responses_path(self) -> Path:
        return self.root_path / "responses"

    @staticmethod
    def _read_regular_file(path: Path, limit: int) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise CodexCLIError("Codex tool request must be a regular file")
            if metadata.st_size > limit:
                raise CodexCLIError("Codex tool request exceeded its byte limit")
            chunks: list[bytes] = []
            remaining = limit + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > limit:
                raise CodexCLIError("Codex tool request exceeded its byte limit")
            return raw
        finally:
            os.close(descriptor)

    def _write_response(self, request_id: str, encoded: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{request_id}-", dir=self.responses_path
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.responses_path / f"{request_id}.json")
        finally:
            temporary.unlink(missing_ok=True)

    def _serve_one(self, request_path: Path, request_id: str) -> None:
        started = time.monotonic()
        tool = "<invalid>"
        arguments: tuple[str, ...] = ()
        try:
            if re.fullmatch(r"[0-9a-f]{32}", request_id) is None:
                raise ValueError("request id is invalid")
            raw = self._read_regular_file(
                request_path, MAX_CODEX_TOOL_REQUEST_BYTES
            )
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("request root must be an object")
            if request.get("protocol") != CODEX_CLI_TOOL_BRIDGE_PROTOCOL:
                raise ValueError("request protocol mismatch")
            if request.get("request_id") != request_id:
                raise ValueError("request id mismatch")
            tool_value = request.get("tool")
            if not isinstance(tool_value, str) or tool_value not in (
                set(self.handlers) | set(self.argument_handlers)
            ):
                raise ValueError("requested tool is not allowlisted")
            tool = tool_value
            raw_arguments = request.get("arguments", [])
            if not isinstance(raw_arguments, list) or len(raw_arguments) > MAX_CODEX_TOOL_ARGUMENTS:
                raise ValueError("tool arguments exceed the closed limit")
            if any(
                not isinstance(item, str)
                or not item
                or len(item) > MAX_CODEX_TOOL_ARGUMENT_CHARS
                for item in raw_arguments
            ):
                raise ValueError("tool arguments must be bounded non-empty strings")
            arguments = tuple(raw_arguments)
            if tool in self.argument_handlers:
                result = self.argument_handlers[tool](arguments)
            elif arguments:
                raise ValueError("this tool does not accept arguments")
            else:
                result = self.handlers[tool]()
            if not isinstance(result, Mapping):
                raise TypeError("tool handler result must be an object")
            result_value = dict(result)
        except Exception as exc:
            result_value = {
                "protocol": CODEX_CLI_TOOL_BRIDGE_PROTOCOL,
                "status": "fail",
                "diagnostics": [
                    {
                        "rule_code": "tool_bridge_rejected",
                        "path": "<tool-bridge>",
                        "observed_refs_or_types": [type(exc).__name__],
                    }
                ],
            }
        envelope = {
            "protocol": CODEX_CLI_TOOL_BRIDGE_PROTOCOL,
            "request_id": request_id,
            "tool": tool,
            "result": result_value,
        }
        encoded = (
            json.dumps(
                envelope,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > MAX_CODEX_TOOL_RESPONSE_BYTES:
            result_value = {
                "protocol": CODEX_CLI_TOOL_BRIDGE_PROTOCOL,
                "status": "fail",
                "diagnostics": [
                    {
                        "rule_code": "tool_bridge_response_too_large",
                        "path": "<tool-bridge>",
                        "observed_refs_or_types": [],
                    }
                ],
            }
            envelope["result"] = result_value
            encoded = (
                json.dumps(
                    envelope,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        self._write_response(request_id, encoded)
        receipt = {
            "tool": tool,
            "argument_count": len(arguments),
            "arguments_sha256": hashlib.sha256(
                json.dumps(
                    list(arguments),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "status": result_value.get("status"),
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "result_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        with self._lock:
            self._receipts.append(receipt)

    def _serve(self) -> None:
        while not self._stop.is_set():
            handled = False
            for request in sorted(self.requests_path.glob("*.json")):
                handled = True
                request_id = request.stem
                processing = self.requests_path / f".{request_id}.processing"
                try:
                    os.replace(request, processing)
                except FileNotFoundError:
                    continue
                try:
                    self._serve_one(processing, request_id)
                finally:
                    processing.unlink(missing_ok=True)
            if not handled:
                self._stop.wait(0.02)

    @property
    def evidence(self) -> dict[str, Any]:
        with self._lock:
            receipts = [dict(item) for item in self._receipts]
        return {
            "protocol": CODEX_CLI_TOOL_BRIDGE_PROTOCOL,
            "transport": "isolated_workspace_file_mailbox",
            "calls": len(receipts),
            "receipts": receipts,
            "authority": (
                "closed allowlisted workspace callbacks; each handler response "
                "declares its own authority, and the mailbox transport performs no "
                "semantic inference, lowering, repair, Gate, or certification"
            ),
        }

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        if (
            self.root_path.name != CODEX_CLI_TOOL_BRIDGE_ROOT_NAME
            or self.root_path.parent == self.root_path
        ):
            raise RuntimeError("refusing to remove an unrecognized Codex bridge")
        shutil.rmtree(self.root_path, ignore_errors=True)

    def __enter__(self) -> "CodexCLIToolBridge":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class CodexCLITurn:
    thread_id: str
    elapsed_ms: float
    usage: Mapping[str, int]
    events: tuple[Mapping[str, Any], ...] = field(repr=False)
    stderr: str = field(default="", repr=False)
    model_segments: int = 1
    tool_actions: int = 0
    resumed: bool = False

    def to_evidence(self) -> dict[str, Any]:
        return {
            "protocol": CODEX_CLI_AGENT_RUNTIME,
            "thread_id": self.thread_id,
            "elapsed_ms": self.elapsed_ms,
            "usage": dict(self.usage),
            "model_segments": self.model_segments,
            "tool_actions": self.tool_actions,
            "resumed": self.resumed,
            "stderr": self.stderr,
            "events": [dict(item) for item in self.events],
        }


@dataclass
class CodexCLISession:
    """One persistent Codex thread over one externally isolated workspace."""

    provider: ProviderConfig
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path
    model: str
    effort: str
    codex_executable: str = "/usr/bin/codex"
    bwrap_executable: str = "bwrap"
    _thread_id: str | None = field(default=None, init=False)
    _session_home: Path = field(init=False, repr=False)
    _tool_root: Path = field(init=False, repr=False)
    _tool_client_path: Path = field(init=False, repr=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.provider.wire_api != "responses":
            raise ValueError("Codex CLI runtime requires a Responses provider")
        if not self.provider.api_key:
            raise ValueError("Codex CLI runtime requires a non-empty provider key")
        if not self.model:
            raise ValueError("Codex CLI runtime requires a model")
        if self.effort not in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
            "ultra",
        }:
            raise ValueError("unsupported Codex reasoning effort")
        paths = (
            self.input_dir,
            self.output_dir,
            self.scratch_dir,
            self.tmp_dir,
        )
        resolved = tuple(Path(path).resolve(strict=True) for path in paths)
        if len(set(resolved)) != len(resolved):
            raise ValueError("Codex input/output/work/tmp directories must differ")
        if any(not path.is_dir() or path.is_symlink() for path in resolved):
            raise ValueError("Codex workspace paths must be real directories")
        (
            self.input_dir,
            self.output_dir,
            self.scratch_dir,
            self.tmp_dir,
        ) = resolved
        if shutil.which(self.bwrap_executable) is None:
            raise CodexCLIError("Codex CLI runtime requires bubblewrap")
        if not Path(self.codex_executable).is_file():
            raise CodexCLIError("Codex CLI executable is unavailable")
        self._session_home = Path(
            tempfile.mkdtemp(prefix=_CODEX_HOME_PREFIX)
        ).resolve()
        self._tool_root = Path(
            tempfile.mkdtemp(prefix=_CODEX_TOOL_ROOT_PREFIX)
        ).resolve()
        self._tool_client_path = self._tool_root / "intentir_agent_tool.py"
        self._tool_client_path.write_text(
            _codex_tool_client_source(), encoding="utf-8"
        )
        self._tool_client_path.chmod(0o444)

    @property
    def thread_id(self) -> str | None:
        return self._thread_id

    @property
    def session_home(self) -> Path:
        return self._session_home

    @staticmethod
    def _toml_string(value: str) -> str:
        # JSON strings are valid TOML basic strings for the characters accepted
        # by provider/model configuration here.
        return json.dumps(value, ensure_ascii=False)

    def _codex_config_args(self, effort: str) -> list[str]:
        string = self._toml_string
        return [
            "-c",
            f"model_reasoning_effort={string(effort)}",
            "-c",
            'model_provider="IntentIR"',
            "-c",
            'model_providers.IntentIR.name="IntentIR"',
            "-c",
            f"model_providers.IntentIR.base_url={string(self.provider.base_url)}",
            "-c",
            f"model_providers.IntentIR.wire_api={string(self.provider.wire_api)}",
            "-c",
            "model_providers.IntentIR.supports_websockets=false",
            "-c",
            "model_providers.IntentIR.requires_openai_auth=false",
            "-c",
            'model_providers.IntentIR.env_key="OPENAI_API_KEY"',
            "-c",
            'shell_environment_policy.inherit="core"',
            "-c",
            'shell_environment_policy.exclude=["^OPENAI_API_KEY$","^CODEX_HOME$"]',
            "-c",
            "allow_login_shell=false",
            "-c",
            "sandbox_workspace_write.network_access=false",
            "-c",
            "sandbox_workspace_write.exclude_slash_tmp=true",
            "-c",
            "sandbox_workspace_write.exclude_tmpdir_env_var=true",
            "-c",
            "features.hooks=false",
            "-c",
            "features.multi_agent=false",
            "-c",
            "features.memories=false",
        ]

    def _sandbox_prefix(self) -> list[str]:
        """Build an outer sandbox that retains provider network for Codex core.

        The model's shell receives an additional Codex workspace-write sandbox
        with network disabled.  The outer namespace prevents host reads even if
        that inner sandbox or a model command is wrong.
        """

        return [
            self.bwrap_executable,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-ipc",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind-try",
            "/bin",
            "/bin",
            "--ro-bind-try",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--ro-bind",
            "/etc",
            "/etc",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/workspace",
            "--dir",
            "/workspace/input",
            "--dir",
            "/workspace/output",
            "--dir",
            "/workspace/work",
            "--dir",
            "/workspace/tmp",
            "--dir",
            "/workspace/.git",
            "--dir",
            "/workspace/.agents",
            "--dir",
            "/workspace/.codex",
            "--ro-bind",
            str(self._tool_client_path),
            "/workspace/.agents/intentir_agent_tool.py",
            "--remount-ro",
            "/workspace",
            "--ro-bind",
            str(self.input_dir),
            "/workspace/input",
            "--bind",
            str(self.output_dir),
            "/workspace/output",
            "--bind",
            str(self.scratch_dir),
            "/workspace/work",
            "--bind",
            str(self.tmp_dir),
            "/workspace/tmp",
            "--bind",
            str(self._session_home),
            "/codex-home",
            "--chdir",
            "/workspace",
        ]

    def _command(self, prompt: str, *, effort: str, resumed: bool) -> list[str]:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Codex prompt must be non-empty")
        if resumed and self._thread_id is None:
            raise ValueError("cannot resume Codex before the first turn")
        common = [
            self.codex_executable,
            "--strict-config",
            "-a",
            "never",
            "-s",
            "workspace-write",
            "exec",
        ]
        if resumed:
            invocation = [
                *common,
                "resume",
                "--ignore-user-config",
                "--ignore-rules",
                "--json",
                "--skip-git-repo-check",
                "-m",
                self.model,
                *self._codex_config_args(effort),
                str(self._thread_id),
                prompt,
            ]
        else:
            invocation = [
                *common,
                "--ignore-user-config",
                "--ignore-rules",
                "--json",
                "-C",
                "/workspace",
                "--skip-git-repo-check",
                "-m",
                self.model,
                *self._codex_config_args(effort),
                prompt,
            ]
        return [*self._sandbox_prefix(), *invocation]

    @staticmethod
    def _read_bounded(stream: Any, limit: int, label: str) -> bytes:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if size > limit:
            raise CodexCLIError(f"Codex {label} exceeded its byte limit")
        stream.seek(0)
        return stream.read()

    @staticmethod
    def _parse_events(raw: bytes) -> tuple[dict[str, Any], ...]:
        events: list[dict[str, Any]] = []
        for index, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CodexCLIError(
                    f"Codex JSONL event {index} is invalid"
                ) from exc
            if not isinstance(value, dict):
                raise CodexCLIError(f"Codex JSONL event {index} is not an object")
            events.append(value)
        if not events:
            raise CodexCLIError("Codex produced no JSONL events")
        return tuple(events)

    def _recover_partial_thread_identity(self, raw: bytes) -> str | None:
        """Remember a unique ``thread.started`` from an interrupted JSONL stream.

        A Codex turn can be killed by the outer wall-time limit after the CLI has
        created and persisted its thread but before it emits ``turn.completed``.
        Only the already-emitted thread identity is recovered here; partial turns,
        usage, model text, and workspace artifacts remain unaccepted until their
        independent host checks pass.
        """

        thread_ids: set[str] = set()
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # The process may have been killed halfway through its final JSONL
                # record.  Earlier complete records are still authoritative.
                continue
            if (
                isinstance(event, Mapping)
                and event.get("type") == "thread.started"
                and isinstance(event.get("thread_id"), str)
                and event["thread_id"]
            ):
                thread_ids.add(event["thread_id"])
        if not thread_ids:
            return None
        if len(thread_ids) != 1:
            raise CodexCLIError(
                "interrupted Codex turn has conflicting thread identities"
            )
        thread_id = next(iter(thread_ids))
        if self._thread_id is not None and thread_id != self._thread_id:
            raise CodexCLIError("interrupted Codex turn belongs to a different thread")
        self._thread_id = thread_id
        return thread_id

    def run(
        self,
        prompt: str,
        *,
        timeout_seconds: float | None,
        effort: str | None = None,
        resume: bool | None = None,
        stop_requested: Callable[[], int | None] | None = None,
        submission_accepted: Callable[[], bool] | None = None,
        diagnostic_ready: Callable[[], bool] | None = None,
        submission_grace_seconds: float = 2.0,
        diagnostic_grace_seconds: float = 2.0,
    ) -> CodexCLITurn:
        if self._closed:
            raise CodexCLIError("Codex session is closed")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("Codex turn timeout must be positive when set")
        if submission_grace_seconds < 0:
            raise ValueError("submission grace must be non-negative")
        if diagnostic_grace_seconds < 0:
            raise ValueError("diagnostic grace must be non-negative")
        selected_effort = effort or self.effort
        resumed = self._thread_id is not None if resume is None else bool(resume)
        command = self._command(prompt, effort=selected_effort, resumed=resumed)
        environment = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HOME": "/tmp",
            "CODEX_HOME": "/codex-home",
            "OPENAI_API_KEY": self.provider.api_key,
        }
        started = time.monotonic()
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
            limit_error: str | None = None
            cancelled_by: int | None = None
            accepted_submission = False
            accepted_at: float | None = None
            ready_diagnostic = False
            diagnostic_at: float | None = None
            while process.poll() is None:
                elapsed = time.monotonic() - started
                observed_winner = (
                    stop_requested() if stop_requested is not None else None
                )
                if observed_winner is not None:
                    if (
                        not isinstance(observed_winner, int)
                        or observed_winner < 1
                    ):
                        limit_error = "candidate stop callback returned an invalid winner"
                    else:
                        cancelled_by = observed_winner
                else:
                    if submission_accepted is not None:
                        observed_acceptance = submission_accepted()
                        if not isinstance(observed_acceptance, bool):
                            limit_error = (
                                "submission callback returned a non-boolean value"
                            )
                        elif observed_acceptance:
                            if accepted_at is None:
                                accepted_at = time.monotonic()
                            if (
                                time.monotonic() - accepted_at
                                >= submission_grace_seconds
                            ):
                                accepted_submission = True
                    if (
                        limit_error is None
                        and not accepted_submission
                        and diagnostic_ready is not None
                    ):
                        observed_diagnostic = diagnostic_ready()
                        if not isinstance(observed_diagnostic, bool):
                            limit_error = (
                                "diagnostic callback returned a non-boolean value"
                            )
                        elif observed_diagnostic:
                            if diagnostic_at is None:
                                diagnostic_at = time.monotonic()
                            if (
                                time.monotonic() - diagnostic_at
                                >= diagnostic_grace_seconds
                            ):
                                ready_diagnostic = True
                    if (
                        limit_error is None
                        and not accepted_submission
                        and not ready_diagnostic
                    ):
                        if (
                            timeout_seconds is not None
                            and elapsed >= timeout_seconds
                        ):
                            limit_error = "Codex turn exceeded its wall-time limit"
                        elif stdout.tell() > MAX_CODEX_STDOUT_BYTES:
                            limit_error = "Codex stdout exceeded its byte limit"
                        elif stderr.tell() > MAX_CODEX_STDERR_BYTES:
                            limit_error = "Codex stderr exceeded its byte limit"
                if (
                    limit_error is not None
                    or cancelled_by is not None
                    or accepted_submission
                    or ready_diagnostic
                ):
                    process.kill()
                    break
                time.sleep(0.05)
            returncode = process.wait(timeout=10)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            raw_stdout = self._read_bounded(
                stdout, MAX_CODEX_STDOUT_BYTES, "stdout"
            )
            raw_stderr = self._read_bounded(
                stderr, MAX_CODEX_STDERR_BYTES, "stderr"
            )
        secret = self.provider.api_key.encode("utf-8")
        if secret and (secret in raw_stdout or secret in raw_stderr):
            raise CodexCLIError("Codex tool boundary exposed the provider secret")
        if cancelled_by is not None:
            raise CodexCLICancelled(
                cancelled_by,
                elapsed_ms=elapsed_ms,
                stdout_sha256=hashlib.sha256(raw_stdout).hexdigest(),
                stderr_sha256=hashlib.sha256(raw_stderr).hexdigest(),
            )
        if accepted_submission:
            thread_id = self._recover_partial_thread_identity(raw_stdout)
            if thread_id is None:
                raise CodexCLIError(
                    "accepted Codex submission lacks a resumable thread identity"
                )
            raise CodexCLISubmissionAccepted(
                elapsed_ms=elapsed_ms,
                thread_id=thread_id,
            )
        if ready_diagnostic:
            thread_id = self._recover_partial_thread_identity(raw_stdout)
            if thread_id is None:
                raise CodexCLIError(
                    "diagnostic-ready Codex turn lacks a resumable thread identity"
                )
            raise CodexCLIDiagnosticReady(
                elapsed_ms=elapsed_ms,
                thread_id=thread_id,
            )
        if limit_error is not None:
            if limit_error == "Codex turn exceeded its wall-time limit":
                self._recover_partial_thread_identity(raw_stdout)
            raise CodexCLIError(limit_error)
        if returncode != 0:
            raise CodexCLIError(f"Codex CLI exited with status {returncode}")
        events = self._parse_events(raw_stdout)
        thread_ids = {
            str(event["thread_id"])
            for event in events
            if event.get("type") == "thread.started"
            and isinstance(event.get("thread_id"), str)
        }
        if len(thread_ids) != 1:
            raise CodexCLIError("Codex turn lacks one closed thread identity")
        thread_id = next(iter(thread_ids))
        if self._thread_id is not None and thread_id != self._thread_id:
            raise CodexCLIError("Codex resumed a different thread")
        completed = [
            event for event in events if event.get("type") == "turn.completed"
        ]
        if len(completed) != 1:
            raise CodexCLIError("Codex turn did not complete exactly once")
        usage_value = completed[0].get("usage")
        usage = {
            str(key): int(value)
            for key, value in (
                usage_value.items() if isinstance(usage_value, Mapping) else ()
            )
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        }
        model_segments = sum(
            1
            for event in events
            if event.get("type") == "item.completed"
            and isinstance(event.get("item"), Mapping)
            and event["item"].get("type") == "agent_message"
        )
        tool_actions = sum(
            1
            for event in events
            if event.get("type") == "item.completed"
            and isinstance(event.get("item"), Mapping)
            and event["item"].get("type")
            in {"command_execution", "file_change", "mcp_tool_call"}
        )
        self._thread_id = thread_id
        stderr_text = raw_stderr.decode("utf-8", errors="replace")[-8192:]
        return CodexCLITurn(
            thread_id=thread_id,
            elapsed_ms=elapsed_ms,
            usage=usage,
            events=events,
            stderr=stderr_text,
            model_segments=max(1, model_segments),
            tool_actions=tool_actions,
            resumed=resumed,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        home = self._session_home
        tool_root = self._tool_root
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if (
            home.parent != temporary_root
            or not home.name.startswith(_CODEX_HOME_PREFIX)
        ):
            raise RuntimeError("refusing to remove an unrecognized Codex home")
        shutil.rmtree(home)
        if (
            tool_root.parent != temporary_root
            or not tool_root.name.startswith(_CODEX_TOOL_ROOT_PREFIX)
        ):
            raise RuntimeError("refusing to remove an unrecognized Codex tool root")
        self._tool_client_path.chmod(0o600)
        shutil.rmtree(tool_root)

    def __enter__(self) -> "CodexCLISession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "CODEX_CLI_AGENT_RUNTIME",
    "CODEX_CLI_TOOL_BRIDGE_PROTOCOL",
    "CODEX_CLI_TOOL_BRIDGE_ROOT_NAME",
    "MAX_CODEX_STDERR_BYTES",
    "MAX_CODEX_STDOUT_BYTES",
    "CodexCLIError",
    "CodexCLICancelled",
    "CodexCLIDiagnosticReady",
    "CodexCLISubmissionAccepted",
    "CodexCLISession",
    "CodexCLIToolBridge",
    "CodexCLITurn",
]
