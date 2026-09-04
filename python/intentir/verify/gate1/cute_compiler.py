"""Isolated compile-only CuTe frontend for strict Gate1 observation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .compiler import RuntimeArgument


_CACHE_SCHEMA_VERSION = 1
_WORKER = Path(__file__).with_name("_cute_worker.py")


@dataclass(frozen=True)
class CuteKernelSource:
    """One CuTe device kernel and the host ``@cute.jit`` compile entry."""

    kind: Literal["module_file", "code"]
    device_function: str
    entry_function: str
    path: str | os.PathLike[str] | None = None
    code: str | None = None
    trusted: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"module_file", "code"}:
            raise ValueError("CuteKernelSource.kind must be module_file or code")
        for label, name in (
            ("device", self.device_function),
            ("entry", self.entry_function),
        ):
            if not name.isidentifier() or "__" in name:
                raise ValueError(f"invalid CuTe {label} function name: {name!r}")
        if self.kind == "module_file":
            if self.path is None or self.code is not None:
                raise ValueError("module_file source requires path and forbids code")
            object.__setattr__(self, "path", str(Path(self.path).expanduser()))
        elif not isinstance(self.code, str) or self.path is not None:
            raise ValueError("code source requires Python text and forbids path")

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        device_function: str,
        entry_function: str,
        *,
        trusted: bool = False,
    ) -> "CuteKernelSource":
        return cls(
            "module_file",
            device_function,
            entry_function,
            path=path,
            trusted=trusted,
        )

    @classmethod
    def from_code(
        cls,
        code: str,
        device_function: str,
        entry_function: str,
        *,
        trusted: bool = False,
    ) -> "CuteKernelSource":
        return cls(
            "code",
            device_function,
            entry_function,
            code=code,
            trusted=trusted,
        )

    def read_code(self) -> tuple[str, str]:
        if self.kind == "code":
            assert self.code is not None
            return self.code, "<gate1-cute-replay>"
        assert self.path is not None
        source_path = Path(self.path).resolve()
        try:
            return source_path.read_text(encoding="utf-8"), str(source_path)
        except OSError as exc:
            raise ValueError(f"cannot read CuTe module {source_path}: {exc}") from exc


@dataclass(frozen=True)
class CuteCompileCase:
    """Positional compile entry plus launch/schedule evidence for one CuTe case.

    Device-only representation arguments (for example a TMA CopyAtom) are
    constructed inside the source/replay ``@cute.jit`` entry.  The positional
    arguments here are only the semantic tensor/scalar values and an optional
    stream appended by the worker.
    """

    entry_arguments: Sequence[RuntimeArgument]
    grid: tuple[int, ...]
    block: tuple[int, ...]
    smem_bytes: int
    schedule: Mapping[str, int | float | bool | str] = field(default_factory=dict)
    append_stream: bool = True
    device: str = "cuda"

    def __post_init__(self) -> None:
        arguments = tuple(self.entry_arguments)
        if not arguments or not all(
            isinstance(argument, RuntimeArgument) for argument in arguments
        ):
            raise TypeError("entry_arguments must contain RuntimeArgument values")
        grid = tuple(int(value) for value in self.grid)
        block = tuple(int(value) for value in self.block)
        if len(grid) != 3 or len(block) != 3 or any(
            value <= 0 for value in (*grid, *block)
        ):
            raise ValueError("CuTe grid and block must each have three positive extents")
        if isinstance(self.smem_bytes, bool) or int(self.smem_bytes) < 0:
            raise ValueError("smem_bytes must be a non-negative integer")
        schedule = dict(self.schedule)
        for name, value in schedule.items():
            if not name.isidentifier() or "__" in name:
                raise ValueError(f"invalid CuTe schedule name: {name!r}")
            if not isinstance(value, (bool, int, float, str)):
                raise ValueError(f"CuTe schedule value {name!r} is not scalar")
        if self.device != "cuda" and not self.device.startswith("cuda:"):
            raise ValueError("CuTe Gate1 currently supports CUDA only")
        object.__setattr__(self, "entry_arguments", arguments)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "smem_bytes", int(self.smem_bytes))
        object.__setattr__(self, "schedule", schedule)

    def to_payload(self) -> dict[str, Any]:
        return {
            "entry_arguments": [argument.to_payload() for argument in self.entry_arguments],
            "grid": list(self.grid),
            "block": list(self.block),
            "smem_bytes": self.smem_bytes,
            "schedule": dict(self.schedule),
            "append_stream": self.append_stream,
            "device": self.device,
        }


@dataclass(frozen=True)
class CuteCompileResult:
    ok: bool
    mlir: str | None = None
    ptx: str | None = None
    cubin_sha256: str | None = None
    python_interface: Mapping[str, Any] = field(default_factory=dict)
    compiler: Mapping[str, Any] = field(default_factory=dict)
    target: Mapping[str, Any] = field(default_factory=dict)
    diagnostic: str = ""
    cache_key: str | None = None
    elapsed_ms: float = 0.0
    asm_keys: tuple[str, ...] = ()
    cache_hit: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CuteCompileResult":
        return cls(
            ok=bool(payload.get("ok", False)),
            mlir=payload.get("mlir"),
            ptx=payload.get("ptx"),
            cubin_sha256=payload.get("cubin_sha256"),
            python_interface=dict(payload.get("python_interface") or {}),
            compiler=dict(payload.get("compiler") or {}),
            target=dict(payload.get("target") or {}),
            diagnostic=str(payload.get("diagnostic", "")),
            cache_key=payload.get("cache_key"),
            elapsed_ms=float(payload.get("elapsed_ms", 0.0)),
            asm_keys=tuple(payload.get("asm_keys", ())),
            cache_hit=bool(payload.get("cache_hit", False)),
        )

    def to_payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["asm_keys"] = list(self.asm_keys)
        return result


class CuteCompilationError(RuntimeError):
    def __init__(self, result: CuteCompileResult):
        super().__init__(result.diagnostic or "CuTe compilation failed")
        self.result = result


class CuteCompiler:
    """Compile CuTe source in a fresh process and retain MLIR/PTX topology."""

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        timeout: float = 300.0,
        python_executable: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        arch: str = "sm_120a",
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not arch.startswith("sm_"):
            raise ValueError("CuTe arch must be an sm_* target")
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.timeout = float(timeout)
        self.python_executable = str(python_executable or sys.executable)
        self.extra_env = {str(key): str(value) for key, value in (env or {}).items()}
        self.arch = arch

    @staticmethod
    def _installed_version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _automatic_key(self, payload: Mapping[str, Any]) -> str:
        material = {
            "schema": _CACHE_SCHEMA_VERSION,
            "request": payload,
            "worker_sha256": hashlib.sha256(_WORKER.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": self._installed_version("torch"),
            "cutlass": self._installed_version("nvidia-cutlass-dsl"),
            "cutlass_cu13": self._installed_version("nvidia-cutlass-dsl-libs-cu13"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "arch": self.arch,
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def _read_cache(self, key: str) -> CuteCompileResult | None:
        if self.cache_dir is None:
            return None
        try:
            payload = json.loads(self._cache_path(key).read_text(encoding="utf-8"))
            if payload.get("cache_schema") != _CACHE_SCHEMA_VERSION:
                return None
            result = CuteCompileResult.from_payload(payload["result"])
            if not result.ok:
                return None
            return replace(result, cache_key=key, cache_hit=True)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, key: str, result: CuteCompileResult) -> None:
        if self.cache_dir is None or not result.ok:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self._cache_path(key)
        payload = {
            "cache_schema": _CACHE_SCHEMA_VERSION,
            "result": replace(result, cache_hit=False).to_payload(),
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.cache_dir
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def compile(
        self,
        source: CuteKernelSource,
        case: CuteCompileCase,
        *,
        cache_key: str | None = None,
    ) -> CuteCompileResult:
        if not isinstance(source, CuteKernelSource):
            raise TypeError("source must be CuteKernelSource")
        if not isinstance(case, CuteCompileCase):
            raise TypeError("case must be CuteCompileCase")
        code, filename = source.read_code()
        if not source.trusted:
            from .cute_codegen import validate_cute_module

            validate_cute_module(
                code,
                required_kernel=source.device_function,
                required_entry=source.entry_function,
            )
        request = {
            "schema": _CACHE_SCHEMA_VERSION,
            "source": {
                "code": code,
                "filename": filename,
                "device_function": source.device_function,
                "entry_function": source.entry_function,
                "trust_role": "trusted_source" if source.trusted else "untrusted_replay",
            },
            "case": case.to_payload(),
        }
        if cache_key is not None and (
            not isinstance(cache_key, str) or not cache_key.strip()
        ):
            raise ValueError("cache_key must be a non-empty string")
        automatic = self._automatic_key(request)
        resolved = (
            automatic
            if cache_key is None
            else hashlib.sha256((cache_key + "\0" + automatic).encode()).hexdigest()
        )
        cached = self._read_cache(resolved)
        if cached is not None:
            return cached

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="intentir-cute-gate1-") as temporary:
            work = Path(temporary)
            request_path = work / "request.json"
            result_path = work / "result.json"
            worker_request = request
            if source.kind == "code":
                # CuTe preprocesses @cute.jit functions through inspect, which
                # requires a real file.  The untrusted bytes were validated
                # above; materialize exactly those bytes only for the lifetime
                # of this isolated compile invocation.
                replay_path = work / "validated_replay.py"
                replay_path.write_text(code, encoding="utf-8")
                worker_request = {
                    **request,
                    "source": {
                        **request["source"],
                        "filename": str(replay_path),
                    },
                }
            request_path.write_text(
                json.dumps(worker_request, sort_keys=True), encoding="utf-8"
            )
            environment = dict(os.environ)
            environment.update(self.extra_env)
            environment.update(
                {
                    "CUTE_DSL_ARCH": self.arch,
                    "CUTE_DSL_KEEP_IR": "1",
                    "CUTE_DSL_KEEP_PTX": "1",
                    "CUTE_DSL_KEEP_CUBIN": "1",
                    "CUTE_DSL_NO_CACHE": "1",
                    "CUTE_DSL_DUMP_DIR": str(work / "cute-dump"),
                    "CUTE_DSL_CACHE_DIR": str(work / "cute-cache"),
                    "PYTHONHASHSEED": "0",
                }
            )
            command = [
                self.python_executable,
                str(_WORKER),
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ]
            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                diagnostic = f"CuTe compile worker timed out after {self.timeout:.1f}s"
                if exc.stderr:
                    diagnostic += "\n" + str(exc.stderr)
                return CuteCompileResult(
                    ok=False,
                    diagnostic=diagnostic,
                    cache_key=resolved,
                    elapsed_ms=(time.monotonic() - started) * 1000.0,
                )
            elapsed = (time.monotonic() - started) * 1000.0
            try:
                result = CuteCompileResult.from_payload(
                    json.loads(result_path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, TypeError) as exc:
                diagnostic = f"CuTe worker did not produce a valid result: {exc}"
                if process.stdout:
                    diagnostic += "\nworker stdout:\n" + process.stdout
                if process.stderr:
                    diagnostic += "\nworker stderr:\n" + process.stderr
                result = CuteCompileResult(ok=False, diagnostic=diagnostic)
            if process.returncode != 0:
                diagnostic = result.diagnostic or (
                    f"CuTe compile worker exited with status {process.returncode}"
                )
                if process.stdout:
                    diagnostic += "\nworker stdout:\n" + process.stdout
                if process.stderr:
                    diagnostic += "\nworker stderr:\n" + process.stderr
                result = replace(result, ok=False, diagnostic=diagnostic)

        result = replace(
            result,
            cache_key=resolved,
            elapsed_ms=result.elapsed_ms or elapsed,
            cache_hit=False,
        )
        self._write_cache(resolved, result)
        return result

    def compile_or_raise(
        self,
        source: CuteKernelSource,
        case: CuteCompileCase,
        *,
        cache_key: str | None = None,
    ) -> CuteCompileResult:
        result = self.compile(source, case, cache_key=cache_key)
        if not result.ok:
            raise CuteCompilationError(result)
        return result


__all__ = [
    "CuteCompilationError",
    "CuteCompileCase",
    "CuteCompileResult",
    "CuteCompiler",
    "CuteKernelSource",
]
