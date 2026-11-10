"""Isolated compile-only TileLang frontend for SCRC Gate1."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Literal, Mapping

from .cuda_compiler import FrontendParameter
from .tilelang_codegen import validate_tilelang_module


_WORKER = Path(__file__).with_name("_tilelang_worker.py")


@dataclass(frozen=True)
class TileLangCompileCase:
    parameters: tuple[FrontendParameter, ...]
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    schedule: Mapping[str, int | float | bool | str] = field(default_factory=dict)
    target: str = "cuda"

    def __post_init__(self) -> None:
        parameters = tuple(
            item if isinstance(item, FrontendParameter) else FrontendParameter(**item)
            for item in self.parameters
        )
        if not parameters:
            raise ValueError("TileLang compile case requires parameters")
        for label, dimensions in (("grid", self.grid), ("block", self.block)):
            dims = tuple(int(item) for item in dimensions)
            if len(dims) != 3 or any(item <= 0 for item in dims):
                raise ValueError(f"TileLang {label} must contain three positive dimensions")
            object.__setattr__(self, label, dims)
        if self.target != "cuda":
            raise ValueError("active TileLang Gate1 profile lowers to CUDA")
        schedule = dict(self.schedule)
        if any(
            not isinstance(key, str)
            or not key.isidentifier()
            or "__" in key
            for key in schedule
        ):
            raise ValueError("TileLang schedule keys must be safe identifiers")
        if any(
            not isinstance(value, (bool, int, float, str))
            or (isinstance(value, float) and not math.isfinite(value))
            for value in schedule.values()
        ):
            raise TypeError("TileLang schedule values must be finite JSON scalars")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "schedule", schedule)

    @property
    def interface(self) -> dict[str, Any]:
        return {"parameters": [item.to_json() for item in self.parameters]}

    def to_payload(self) -> dict[str, Any]:
        return {
            "parameters": [item.to_json() for item in self.parameters],
            "grid": list(self.grid),
            "block": list(self.block),
            "schedule": dict(self.schedule),
            "target": self.target,
        }


@dataclass(frozen=True)
class TileLangKernelSource:
    kind: Literal["factory_file", "code"]
    function: str
    path: str | os.PathLike[str] | None = None
    code: str | None = None
    factory_kwargs: Mapping[str, int | float | bool | str] = field(default_factory=dict)
    trusted: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"factory_file", "code"}:
            raise ValueError("unsupported TileLang source kind")
        if not self.function.isidentifier() or "__" in self.function:
            raise ValueError("invalid TileLang function name")
        if self.kind == "factory_file":
            if self.path is None or self.code is not None or not self.trusted:
                raise ValueError("TileLang factory_file is trusted and path-only")
            object.__setattr__(self, "path", str(Path(self.path).expanduser()))
        elif not isinstance(self.code, str) or self.path is not None:
            raise ValueError("TileLang code source requires only code")
        kwargs = dict(self.factory_kwargs)
        if any(not isinstance(key, str) or not key.isidentifier() for key in kwargs):
            raise ValueError("TileLang factory kwargs require identifier keys")
        if any(not isinstance(value, (bool, int, float, str)) for value in kwargs.values()):
            raise TypeError("TileLang factory kwargs must be JSON scalars")
        object.__setattr__(self, "factory_kwargs", kwargs)
        if not self.trusted:
            assert self.code is not None
            validate_tilelang_module(self.code, required_function=self.function)

    @classmethod
    def from_factory_file(
        cls,
        path: str | os.PathLike[str],
        factory: str,
        *,
        factory_kwargs: Mapping[str, int | float | bool | str] | None = None,
    ) -> "TileLangKernelSource":
        return cls(
            "factory_file",
            factory,
            path=path,
            factory_kwargs=factory_kwargs or {},
            trusted=True,
        )

    @classmethod
    def from_code(
        cls,
        code: str,
        function: str,
        *,
        trusted: bool = False,
    ) -> "TileLangKernelSource":
        return cls("code", function, code=code, trusted=trusted)

    @classmethod
    def from_replay(cls, code: str, function: str) -> "TileLangKernelSource":
        return cls.from_code(code, function, trusted=False)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "function": self.function,
            "path": str(Path(self.path).resolve()) if self.path is not None else None,
            "code": self.code,
            "factory_kwargs": dict(self.factory_kwargs),
        }

    def read_code(self) -> str:
        """Read the exact Python/TIR source bytes consumed by the frontend."""

        if self.code is not None:
            return self.code
        assert self.path is not None
        try:
            return Path(self.path).resolve().read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read TileLang source {self.path}: {exc}") from exc


@dataclass(frozen=True)
class TileLangCompileResult:
    ok: bool
    tir: str | None = None
    cuda: str | None = None
    ptx: str | None = None
    interface: Mapping[str, Any] = field(default_factory=dict)
    compiler: Mapping[str, Any] = field(default_factory=dict)
    target: Mapping[str, Any] = field(default_factory=dict)
    diagnostic: str = ""
    cache_key: str | None = None
    elapsed_ms: float = 0.0

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, cache_key: str) -> "TileLangCompileResult":
        return cls(
            bool(payload.get("ok", False)),
            tir=payload.get("tir"),
            cuda=payload.get("cuda"),
            ptx=payload.get("ptx"),
            interface=dict(payload.get("interface") or {}),
            compiler=dict(payload.get("compiler") or {}),
            target=dict(payload.get("target") or {}),
            diagnostic=str(payload.get("diagnostic") or ""),
            cache_key=cache_key,
            elapsed_ms=float(payload.get("elapsed_ms") or 0.0),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tir": self.tir,
            "cuda": self.cuda,
            "ptx": self.ptx,
            "interface": dict(self.interface),
            "compiler": dict(self.compiler),
            "target": dict(self.target),
            "diagnostic": self.diagnostic,
            "cache_key": self.cache_key,
            "elapsed_ms": self.elapsed_ms,
        }


class TileLangCompiler:
    def __init__(
        self,
        *,
        timeout: float = 300.0,
        python_executable: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("TileLang compiler timeout must be positive")
        self.timeout = float(timeout)
        self.python_executable = str(python_executable or sys.executable)
        self.extra_env = {str(key): str(value) for key, value in (env or {}).items()}

    def compile(
        self,
        source: TileLangKernelSource,
        case: TileLangCompileCase,
    ) -> TileLangCompileResult:
        request = {"source": source.to_payload(), "case": case.to_payload()}
        encoded = json.dumps(request, sort_keys=True, allow_nan=False)
        cache_key = hashlib.sha256(encoded.encode()).hexdigest()
        environment = os.environ.copy()
        environment.update(self.extra_env)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [self.python_executable, str(_WORKER)],
                input=encoded,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return TileLangCompileResult(
                False,
                diagnostic=f"TileLang compilation timed out after {self.timeout:g}s",
                cache_key=cache_key,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        if result.returncode != 0:
            return TileLangCompileResult(
                False,
                diagnostic=(result.stderr or result.stdout or "TileLang worker failed").strip(),
                cache_key=cache_key,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return TileLangCompileResult(
                False,
                diagnostic=f"TileLang worker emitted invalid JSON: {exc}; stderr={result.stderr.strip()}",
                cache_key=cache_key,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        return TileLangCompileResult.from_payload(payload, cache_key=cache_key)


__all__ = [
    "TileLangCompileCase",
    "TileLangCompileResult",
    "TileLangCompiler",
    "TileLangKernelSource",
]
