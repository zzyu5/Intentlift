"""Compile-only CUDA C++ frontend for SCRC Gate1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Literal, Mapping

from .cuda_codegen import validate_cuda_module


@dataclass(frozen=True)
class FrontendParameter:
    kind: Literal["pointer", "scalar", "opaque"]
    dtype: str
    rank: int = 0
    mode: Literal["input", "output", "inout", "value"] = "value"

    def __post_init__(self) -> None:
        if self.kind not in {"pointer", "scalar", "opaque"}:
            raise ValueError("unsupported frontend parameter kind")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("frontend parameter dtype must be non-empty")
        if not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("frontend parameter rank must be non-negative")
        if self.kind == "pointer" and self.mode not in {"input", "output", "inout"}:
            raise ValueError("pointer mode must describe an effect")
        if self.kind != "pointer" and self.mode != "value":
            raise ValueError("non-pointer parameters must use value mode")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CudaCompileCase:
    parameters: tuple[FrontendParameter, ...]
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    # Opaque C-slot valuation used by the Harness and PTX observer.  These
    # names are artifact handles, not CUDA preprocessor identifiers; launch
    # roles are consumed through ``grid``/``block`` and actual nvcc flags
    # through the separately allowlisted ``compiler_options``.
    schedule: Mapping[str, int | float | bool | str] = field(default_factory=dict)
    arch: str = "sm_80"
    compiler_options: tuple[str, ...] = ()
    include_dirs: tuple[str, ...] = ()
    # A Dataset CUDA source may expose a physical ABI (for example a
    # CUtensorMap slot) while Gate1 intentionally compares its logical J ABI.
    # This optional view is used only for trusted module-file sources; Replay
    # code is checked against ``parameters`` because it must realize the
    # source-blind logical contract.
    physical_parameters: tuple[FrontendParameter, ...] | None = None

    def __post_init__(self) -> None:
        parameters = tuple(
            item if isinstance(item, FrontendParameter) else FrontendParameter(**item)
            for item in self.parameters
        )
        if not parameters:
            raise ValueError("CUDA compile case requires at least one parameter")
        for label, dimensions in (("grid", self.grid), ("block", self.block)):
            dims = tuple(int(item) for item in dimensions)
            if len(dims) != 3 or any(item <= 0 for item in dims):
                raise ValueError(f"CUDA {label} must contain three positive dimensions")
            object.__setattr__(self, label, dims)
        schedule = dict(self.schedule)
        if any(
            not isinstance(key, str)
            or not key.isidentifier()
            or "__" in key
            for key in schedule
        ):
            raise ValueError("CUDA schedule keys must be safe identifiers")
        if any(
            not isinstance(value, (bool, int, float, str))
            or (isinstance(value, float) and not math.isfinite(value))
            for value in schedule.values()
        ):
            raise TypeError("CUDA schedule values must be finite JSON scalars")
        allowed = (
            "--fmad=",
            "--ftz=",
            "--maxrregcount=",
            "--prec-div=",
            "--prec-sqrt=",
            "--use_fast_math",
        )
        options = tuple(str(item) for item in self.compiler_options)
        if any(not item.startswith(allowed) for item in options):
            raise ValueError("CUDA compiler option is outside the Gate1 profile")
        arch = _normalize_arch(self.arch)
        include_dirs = tuple(
            str(Path(item).expanduser().resolve()) for item in self.include_dirs
        )
        physical_parameters = self.physical_parameters
        if physical_parameters is not None:
            physical_parameters = tuple(
                item
                if isinstance(item, FrontendParameter)
                else FrontendParameter(**item)
                for item in physical_parameters
            )
            if not physical_parameters:
                raise ValueError("CUDA physical ABI view cannot be empty")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "compiler_options", options)
        object.__setattr__(self, "include_dirs", include_dirs)
        object.__setattr__(self, "arch", arch)
        object.__setattr__(self, "physical_parameters", physical_parameters)

    @property
    def interface(self) -> dict[str, Any]:
        return {
            "parameters": [item.to_json() for item in self.parameters],
            "grid_rank": sum(int(item != 1) for item in self.grid) or 1,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "parameters": [item.to_json() for item in self.parameters],
            "grid": list(self.grid),
            "block": list(self.block),
            "schedule": dict(self.schedule),
            "arch": self.arch,
            "compiler_options": list(self.compiler_options),
            "include_dirs": list(self.include_dirs),
            "physical_parameters": (
                [item.to_json() for item in self.physical_parameters]
                if self.physical_parameters is not None
                else None
            ),
        }


@dataclass(frozen=True)
class CudaKernelSource:
    kind: Literal["module_file", "code"]
    function: str
    path: str | os.PathLike[str] | None = None
    code: str | None = None
    trusted: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"module_file", "code"}:
            raise ValueError("unsupported CUDA kernel source kind")
        if not self.function.isidentifier() or "__" in self.function:
            raise ValueError("invalid CUDA kernel function name")
        if self.kind == "module_file":
            if self.path is None or self.code is not None:
                raise ValueError("CUDA module_file requires only path")
            object.__setattr__(self, "path", str(Path(self.path).expanduser()))
        elif not isinstance(self.code, str) or self.path is not None:
            raise ValueError("CUDA code source requires only code")
        if not self.trusted:
            assert self.code is not None
            validate_cuda_module(self.code, required_kernel=self.function)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], function: str) -> "CudaKernelSource":
        return cls("module_file", function, path=path, trusted=True)

    @classmethod
    def from_code(
        cls,
        code: str,
        function: str,
        *,
        trusted: bool = False,
    ) -> "CudaKernelSource":
        return cls("code", function, code=code, trusted=trusted)

    @classmethod
    def from_replay(cls, code: str, function: str) -> "CudaKernelSource":
        return cls.from_code(code, function, trusted=False)

    def read_code(self) -> str:
        if self.code is not None:
            return self.code
        assert self.path is not None
        try:
            return Path(self.path).resolve().read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read CUDA source {self.path}: {exc}") from exc


@dataclass(frozen=True)
class CudaCompileResult:
    ok: bool
    ptx: str | None = None
    entry: str | None = None
    interface: Mapping[str, Any] = field(default_factory=dict)
    compiler: Mapping[str, Any] = field(default_factory=dict)
    target: Mapping[str, Any] = field(default_factory=dict)
    diagnostic: str = ""
    cache_key: str | None = None
    elapsed_ms: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ptx": self.ptx,
            "entry": self.entry,
            "interface": dict(self.interface),
            "compiler": dict(self.compiler),
            "target": dict(self.target),
            "diagnostic": self.diagnostic,
            "cache_key": self.cache_key,
            "elapsed_ms": self.elapsed_ms,
        }


def _normalize_arch(raw: str) -> str:
    value = str(raw or "").strip().lower()
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        raise ValueError("CUDA arch must contain a compute capability")
    # CUDA's architecture-specific targets (for example sm_90a/sm_120a) are
    # not aliases of their portable counterparts.  Instructions such as
    # setmaxnreg are rejected when the trailing ``a`` is silently discarded.
    suffix = "a" if value.endswith("a") else ""
    return f"sm_{digits}{suffix}"


def find_nvcc() -> str | None:
    direct = shutil.which("nvcc")
    if direct:
        return direct
    candidates: list[Path] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(key)
        if home:
            candidates.append(Path(home) / "bin" / "nvcc")
    candidates.extend((Path("/usr/local/cuda/bin/nvcc"), Path("/opt/cuda/bin/nvcc")))
    for pattern in ("/usr/local/cuda-*/bin/nvcc", "/opt/cuda-*/bin/nvcc"):
        candidates.extend(Path(item) for item in sorted(glob.glob(pattern), reverse=True))
    return next((str(item) for item in candidates if item.is_file()), None)


_CUDA_DTYPE_ALIASES = (
    ("unsigned long long", "u64"),
    ("long long", "i64"),
    ("unsigned int", "u32"),
    ("unsigned short", "u16"),
    ("unsigned char", "u8"),
    ("__nv_bfloat16", "bf16"),
    ("nv_bfloat16", "bf16"),
    ("bfloat16_t", "bf16"),
    ("__half", "f16"),
    ("half_t", "f16"),
    ("double", "f64"),
    ("float", "f32"),
    ("uint64_t", "u64"),
    ("int64_t", "i64"),
    ("uint32_t", "u32"),
    ("int32_t", "i32"),
    ("uint16_t", "u16"),
    ("int16_t", "i16"),
    ("uint8_t", "u8"),
    ("int8_t", "i8"),
    ("bool", "i1"),
    ("int", "i32"),
    ("short", "i16"),
    ("char", "i8"),
)


def _without_cuda_comments_and_literals(code: str) -> str:
    """Blank comments/literals while retaining delimiters and newlines."""

    result = list(code)
    index = 0
    state = "normal"
    while index < len(code):
        character = code[index]
        following = code[index + 1] if index + 1 < len(code) else ""
        if state == "normal":
            if character == "/" and following == "/":
                result[index] = result[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if character == "/" and following == "*":
                result[index] = result[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if character in {"\"", "'"}:
                result[index] = " "
                state = character
            index += 1
            continue
        if state == "line_comment":
            if character == "\n":
                state = "normal"
            elif character != "\r":
                result[index] = " "
            index += 1
            continue
        if state == "block_comment":
            if character == "*" and following == "/":
                result[index] = result[index + 1] = " "
                index += 2
                state = "normal"
                continue
            if character not in {"\n", "\r"}:
                result[index] = " "
            index += 1
            continue
        # String/character literal.  Keep line structure but hide all content.
        if character == "\\":
            result[index] = " "
            if index + 1 < len(code) and code[index + 1] not in {"\n", "\r"}:
                result[index + 1] = " "
                index += 2
            else:
                index += 1
            continue
        if character == state:
            result[index] = " "
            state = "normal"
        elif character not in {"\n", "\r"}:
            result[index] = " "
        index += 1
    return "".join(result)


def _matching_cuda_parenthesis(code: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(code)):
        character = code[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("CUDA kernel declaration has unbalanced parentheses")


def _split_cuda_parameters(parameters: str) -> tuple[str, ...]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(parameters):
        if character in "(<[{":
            depth += 1
        elif character in ")]>}":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(parameters[start:index].strip())
            start = index + 1
    tail = parameters[start:].strip()
    if tail:
        result.append(tail)
    if not result or any(not item for item in result):
        return ()
    return tuple(result)


def _cuda_kernel_parameter_declarations(code: str, function: str) -> tuple[str, ...]:
    inspected = _without_cuda_comments_and_literals(code)
    match = re.search(
        rf"\b__global__\b[^;{{}}]*?\b{re.escape(function)}\s*\(",
        inspected,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(
            f"CUDA source has no __global__ declaration for {function!r}"
        )
    opening = inspected.find("(", match.start())
    closing = _matching_cuda_parenthesis(inspected, opening)
    return _split_cuda_parameters(code[opening + 1 : closing])


def _cuda_parameter_name(declaration: str) -> str:
    declaration = re.sub(r"__attribute__\s*\(.*?\)", " ", declaration)
    match = re.search(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$",
        declaration.strip(),
    )
    if match is None:
        raise ValueError(f"cannot parse CUDA parameter declaration {declaration!r}")
    return match.group(1)


def _cuda_parameter_dtype(declaration: str, name: str) -> str:
    type_spelling = declaration[: declaration.rfind(name)].strip()
    if "CUtensorMap" in type_spelling:
        return "tensor_map"
    for spelling, dtype in _CUDA_DTYPE_ALIASES:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(spelling)}(?![A-Za-z0-9_])", type_spelling):
            return dtype
    normalized = re.sub(
        r"\b(?:const|volatile|restrict|__restrict__|static|register)\b",
        " ",
        type_spelling,
    )
    normalized = re.sub(r"[*&]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _observe_cuda_interface(code: str, function: str) -> dict[str, Any]:
    """Observe the physical CUDA declaration emitted by this source.

    CUDA C++ declarations do not carry J rank, symbolic shape, or effect mode.
    Those facts remain in the Harness/J interface.  This observer therefore
    records only declaration-order physical ABI facts and pointer pointee
    constness, rather than copying semantic metadata into the compiler result.
    """

    parameters: list[dict[str, Any]] = []
    for position, declaration in enumerate(
        _cuda_kernel_parameter_declarations(code, function)
    ):
        name = _cuda_parameter_name(declaration)
        if "CUtensorMap" in declaration:
            kind = "opaque"
        elif "*" in declaration:
            kind = "pointer"
        elif "&" in declaration:
            raise ValueError(
                f"CUDA parameter {name!r} uses a reference outside the native ABI"
            )
        else:
            kind = "scalar"
        row: dict[str, Any] = {
            "position": position,
            "name": name,
            "kind": kind,
            "dtype": _cuda_parameter_dtype(declaration, name),
            "declaration": " ".join(declaration.split()),
        }
        if kind == "pointer":
            star = declaration.find("*")
            row["pointee_const"] = bool(
                re.search(r"\bconst\b", declaration[:star])
            )
        parameters.append(row)
    return {"parameters": parameters}


def _cuda_dtype_key(value: str) -> str:
    normalized = str(value).strip().lower().removeprefix("torch.")
    aliases = dict(_CUDA_DTYPE_ALIASES)
    return aliases.get(normalized, normalized)


def _validate_cuda_interface(
    observed: Mapping[str, Any],
    expected: tuple[FrontendParameter, ...],
) -> None:
    raw_parameters = observed.get("parameters")
    if not isinstance(raw_parameters, list):
        raise ValueError("CUDA compiler did not report a physical parameter interface")
    if len(raw_parameters) != len(expected):
        raise ValueError("CUDA physical parameter count differs from the compile case")
    expected_kind = {
        "pointer": "pointer",
        "scalar": "scalar",
        "opaque": "opaque",
    }
    for index, (parameter, raw) in enumerate(zip(expected, raw_parameters, strict=True)):
        if not isinstance(raw, Mapping):
            raise ValueError(f"CUDA physical parameter {index} is not an object")
        if raw.get("kind") != expected_kind[parameter.kind]:
            raise ValueError(
                f"CUDA physical parameter {index} kind differs from the accepted source ABI"
            )
        if _cuda_dtype_key(str(raw.get("dtype"))) != _cuda_dtype_key(parameter.dtype):
            raise ValueError(
                f"CUDA physical parameter {index} dtype differs from the accepted source ABI"
            )
        if parameter.kind == "pointer" and not isinstance(
            raw.get("pointee_const"), bool
        ):
            raise ValueError(
                f"CUDA physical pointer parameter {index} lacks pointee constness"
            )


def _expected_cuda_parameters(
    source: CudaKernelSource,
    case: CudaCompileCase,
) -> tuple[FrontendParameter, ...]:
    if source.kind == "module_file" and case.physical_parameters is not None:
        return case.physical_parameters
    return case.parameters


class CudaCompiler:
    """Compile a trusted Source or validated Replay to PTX without launching it."""

    def __init__(self, *, nvcc: str | None = None, timeout: float = 180.0) -> None:
        if timeout <= 0:
            raise ValueError("CUDA compiler timeout must be positive")
        self.nvcc = str(nvcc) if nvcc is not None else find_nvcc()
        self.timeout = float(timeout)

    def _version(self) -> str:
        if self.nvcc is None:
            return "unavailable"
        try:
            result = subprocess.run(
                [self.nvcc, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            lines = (result.stdout or result.stderr).strip().splitlines()
            return lines[-1] if lines else "nvcc"
        except Exception:
            return "nvcc"

    def compile(self, source: CudaKernelSource, case: CudaCompileCase) -> CudaCompileResult:
        started = time.perf_counter()
        if self.nvcc is None:
            return CudaCompileResult(
                False,
                diagnostic="nvcc is unavailable",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        try:
            code = source.read_code()
        except Exception as exc:
            return CudaCompileResult(
                False,
                diagnostic=f"source read failed: {type(exc).__name__}: {exc}",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        interface: Mapping[str, Any] = {}
        try:
            interface = _observe_cuda_interface(code, source.function)
            _validate_cuda_interface(
                interface,
                _expected_cuda_parameters(source, case),
            )
        except Exception as exc:
            return CudaCompileResult(
                False,
                entry=source.function,
                interface=interface,
                diagnostic=(
                    f"CUDA physical ABI observation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                cache_key=None,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        key_payload = {
            "source": code,
            "entry": source.function,
            "case": case.to_payload(),
            "nvcc": self._version(),
        }
        cache_key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory(prefix="intentir-cuda-gate1-") as raw_dir:
            directory = Path(raw_dir)
            source_path = directory / "kernel.cu"
            ptx_path = directory / "kernel.ptx"
            source_path.write_text(code, encoding="utf-8")
            command = [
                self.nvcc,
                "--ptx",
                "--std=c++17",
                "-O3",
                "-lineinfo",
                f"-arch={case.arch}",
                # Never materialize opaque schedule handles as -D macros.
                # A TU-global ``-Dc0=...`` also rewrites vendor headers and
                # makes the compiler consume an artifact name as source code.
                *case.compiler_options,
                *(
                    item
                    for directory in case.include_dirs
                    for item in ("-I", directory)
                ),
                str(source_path),
                "-o",
                str(ptx_path),
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                    cwd=directory,
                )
            except subprocess.TimeoutExpired:
                return CudaCompileResult(
                    False,
                    entry=source.function,
                    diagnostic=f"CUDA compilation timed out after {self.timeout:g}s",
                    cache_key=cache_key,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            if result.returncode != 0 or not ptx_path.is_file():
                diagnostic = (result.stderr or result.stdout or "nvcc failed").strip()
                return CudaCompileResult(
                    False,
                    entry=source.function,
                    compiler={"name": "nvcc", "version": self._version()},
                    target={"arch": case.arch},
                    diagnostic=diagnostic,
                    cache_key=cache_key,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            return CudaCompileResult(
                True,
                ptx=ptx_path.read_text(encoding="utf-8"),
                entry=source.function,
                interface=interface,
                compiler={"name": "nvcc", "version": self._version()},
                target={"arch": case.arch},
                cache_key=cache_key,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )


__all__ = [
    "CudaCompileCase",
    "CudaCompileResult",
    "CudaCompiler",
    "CudaKernelSource",
    "FrontendParameter",
    "find_nvcc",
]
