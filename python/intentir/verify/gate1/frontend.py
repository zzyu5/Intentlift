"""One registry for active SCRC frontend capabilities.

The registry describes proof-relevant frontend behavior; it does not change
``J``, ``C`` or ``pi``.  Callers can therefore select a public frontend without
forking the source-blind Replay packet or the SCRC lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compiler import CompileCase, KernelSource, TritonCompiler
from .cuda_compiler import CudaCompileCase, CudaCompiler, CudaKernelSource
from .cute_compiler import CuteCompileCase, CuteCompiler, CuteKernelSource
from .cute_orchestrate import CuteGate1Runner
from .model import Frontend
from .native_orchestrate import CudaGate1Runner, TileLangGate1Runner
from .orchestrate import Gate1Config, Gate1Runner
from .profile import ObserverProfile, observer_profile_for
from .replay import ReplayWriter
from .tilelang_compiler import (
    TileLangCompileCase,
    TileLangCompiler,
    TileLangKernelSource,
)


@dataclass(frozen=True)
class FrontendGate1Capability:
    frontend: Frontend
    replay_language: str
    source_extensions: tuple[str, ...]
    compiler_type: type[Any]
    source_type: type[Any]
    case_type: type[Any]
    runner_type: type[Any]
    profile: ObserverProfile

    def build_runner(
        self,
        *,
        compiler: Any,
        writer: ReplayWriter,
        masked_writer: ReplayWriter | None = None,
        config: Gate1Config | None = None,
    ) -> Any:
        if not isinstance(compiler, self.compiler_type):
            raise TypeError(
                f"{self.frontend.value} requires {self.compiler_type.__name__}"
            )
        runner_kwargs: dict[str, Any] = {
            "compiler": compiler,
            "writer": writer,
            "config": config,
        }
        if self.frontend in {Frontend.CUDA, Frontend.TILELANG}:
            runner_kwargs["masked_writer"] = masked_writer
        elif masked_writer is not None:
            raise ValueError("Triton Gate1 owns masked Replay on its normal writer")
        return self.runner_type(
            **runner_kwargs,
        )


def _capabilities() -> dict[Frontend, FrontendGate1Capability]:
    rows = (
        (Frontend.TRITON, "python", (".py",), TritonCompiler, KernelSource, CompileCase, Gate1Runner),
        (Frontend.CUTE, "python", (".py",), CuteCompiler, CuteKernelSource, CuteCompileCase, CuteGate1Runner),
        (Frontend.TILELANG, "python", (".py",), TileLangCompiler, TileLangKernelSource, TileLangCompileCase, TileLangGate1Runner),
        (Frontend.CUDA, "cuda-c++", (".cu",), CudaCompiler, CudaKernelSource, CudaCompileCase, CudaGate1Runner),
    )
    return {
        frontend: FrontendGate1Capability(
            frontend,
            language,
            extensions,
            compiler,
            source,
            case,
            runner,
            observer_profile_for(frontend),
        )
        for frontend, language, extensions, compiler, source, case, runner in rows
    }


_ACTIVE = _capabilities()


def frontend_gate1_capability(
    frontend: Frontend | str,
) -> FrontendGate1Capability:
    try:
        return _ACTIVE[Frontend(frontend)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"no active SCRC Gate1 frontend for {frontend!r}") from exc


def active_gate1_frontends() -> tuple[Frontend, ...]:
    return tuple(_ACTIVE)

__all__ = [
    "FrontendGate1Capability",
    "active_gate1_frontends",
    "frontend_gate1_capability",
]
