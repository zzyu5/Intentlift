"""Private subprocess entry point for :mod:`intentir.verify.gate1.compiler`.

This file is executed by path rather than imported by the parent.  Keeping all
Torch/Triton imports here makes the public compiler module import-light and,
more importantly, guarantees that model-generated Python is never imported in
the long-lived orchestration process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from intentir.verify.triton_compat import (
    TRITON_DOT_COMPAT_PROTOCOL as _TRITON_DOT_COMPAT_PROTOCOL,
    normalize_deprecated_triton_dot as _normalize_deprecated_triton_dot,
)

_REQUEST_SCHEMA = 2


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        public = {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if public:
            return public
    return str(value)


def _package_coordinates(path: Path) -> tuple[list[tuple[str, Path]], Path] | None:
    """Recover Python package coordinates without executing package code."""

    packages: list[tuple[str, Path]] = []
    current = path.parent
    while (current / "__init__.py").is_file():
        packages.append((current.name, current))
        current = current.parent
    if not packages:
        return None
    packages.reverse()
    return packages, current


def _namespace_package_chain(packages: list[tuple[str, Path]]) -> str:
    """Install package shells after an optional native extension import fails."""

    top = packages[0][0]
    for name in list(sys.modules):
        if name == top or name.startswith(top + "."):
            sys.modules.pop(name, None)
    components: list[str] = []
    for component, directory in packages:
        components.append(component)
        qualified = ".".join(components)
        module = ModuleType(qualified)
        module.__file__ = str(directory / "__init__.py")
        module.__package__ = qualified
        module.__path__ = [str(directory)]  # type: ignore[attr-defined]
        spec = importlib.util.spec_from_loader(
            qualified, loader=None, is_package=True
        )
        if spec is not None:
            spec.submodule_search_locations = [str(directory)]
        module.__spec__ = spec
        sys.modules[qualified] = module
    return ".".join(components)


def _load_module(
    code: str,
    directory: Path,
    *,
    filename: str | None = None,
    trusted_source: bool = False,
    source_identity_code: str | None = None,
) -> ModuleType:
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    path: Path
    package_name: str | None = None
    if trusted_source and filename and filename != "<gate1-replay>":
        candidate = Path(filename).resolve()
        identity = source_identity_code if source_identity_code is not None else code
        if candidate.is_file() and candidate.read_text(encoding="utf-8") == identity:
            path = candidate
            coordinates = _package_coordinates(path)
            if coordinates is not None:
                packages, package_root = coordinates
                if str(package_root) not in sys.path:
                    sys.path.insert(0, str(package_root))
                root_name = packages[0][0]
                try:
                    importlib.import_module(root_name)
                    package_name = ".".join(item[0] for item in packages)
                except Exception:
                    package_name = _namespace_package_chain(packages)
        else:
            raise ValueError("trusted source filename no longer matches request bytes")
    else:
        path = directory / f"gate1_kernel_{digest}.py"
        path.write_text(code, encoding="utf-8")
    if source_identity_code is not None and source_identity_code != code:
        path = directory / f"gate1_kernel_{digest}.py"
        path.write_text(code, encoding="utf-8")
    leaf = f"{path.stem}_gate1_{digest}_{os.getpid()}"
    module_name = f"{package_name}.{leaf}" if package_name else f"_{leaf}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create import specification for replay module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _torch_dtype(torch: Any, name: str) -> Any:
    normalized = name.removeprefix("torch.").lower()
    aliases = {
        "bool": torch.bool,
        "uint8": torch.uint8,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "int8": torch.int8,
        "int16": torch.int16,
        "short": torch.int16,
        "int32": torch.int32,
        "int": torch.int32,
        "int64": torch.int64,
        "index": torch.int64,
        "long": torch.int64,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported tensor dtype {name!r}")
    return aliases[normalized]


def _scalar_value(spec: Mapping[str, Any]) -> int | float | bool:
    value = spec["value"]
    dtype = spec.get("dtype")
    if dtype is None:
        return value
    normalized = str(dtype).removeprefix("torch.").lower()
    if normalized == "bool":
        return bool(value)
    if normalized.startswith(("int", "uint")) or normalized in {"long", "short"}:
        return int(value)
    if normalized.startswith(("float", "bfloat")) or normalized in {"half", "double"}:
        return float(value)
    raise ValueError(f"unsupported scalar dtype {dtype!r}")


def _runtime_value(torch: Any, spec: Mapping[str, Any], device: str) -> Any:
    kind = spec.get("kind")
    if kind == "none":
        return None
    if kind == "scalar":
        return _scalar_value(spec)
    if kind != "tensor":
        raise ValueError(f"unknown runtime argument kind {kind!r}")
    dtype = _torch_dtype(torch, str(spec["dtype"]))
    shape = tuple(int(dim) for dim in spec["shape"])
    strides = spec.get("strides")
    if strides is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return torch.empty_strided(
        shape,
        tuple(int(stride) for stride in strides),
        dtype=dtype,
        device=device,
    )


def _jit_metadata_owner(kernel: Any) -> Any:
    """Find the wrapped Triton JITFunction that owns ABI metadata.

    Real libraries commonly expose a callable kernel wrapper (for example
    FlagGems ``LibEntry``) while keeping ``constexprs`` and
    ``do_not_specialize`` on an inner JITFunction.  Follow only the conventional
    ``fn`` wrapper chain and fail closed on cycles or unrelated objects.
    """

    current = kernel
    visited: set[int] = set()
    while True:
        marker = id(current)
        if marker in visited:
            raise TypeError("cyclic Triton kernel wrapper chain")
        visited.add(marker)
        if all(
            hasattr(current, attribute)
            for attribute in ("arg_names", "constexprs", "do_not_specialize")
        ):
            return current
        nested = getattr(current, "fn", None)
        if nested is None:
            raise TypeError(
                "Triton kernel wrapper does not expose JIT ABI metadata"
            )
        current = nested


def _compiled_kernel(value: Any) -> Any:
    """Extract one CompiledKernel from a wrapper-specific warmup result."""

    if hasattr(value, "asm"):
        return value
    if isinstance(value, (tuple, list)):
        matches = [item for item in value if hasattr(item, "asm")]
        if len(matches) == 1:
            return matches[0]
    raise TypeError(
        "Triton warmup did not return one compiled kernel artifact"
    )


def _interface(kernel: Any) -> dict[str, Any]:
    metadata = _jit_metadata_owner(kernel)
    arg_names = list(metadata.arg_names)
    outer_arg_names = getattr(kernel, "arg_names", None)
    if outer_arg_names is not None and list(outer_arg_names) != arg_names:
        raise TypeError(
            "Triton wrapper argument order differs from inner JIT metadata"
        )
    constexpr_indices = sorted(int(index) for index in metadata.constexprs)
    do_not_specialize_raw = list(metadata.do_not_specialize)

    def _name(item: Any) -> str:
        if isinstance(item, int) and 0 <= item < len(arg_names):
            return arg_names[item]
        return str(item)

    return {
        "arg_names": arg_names,
        "constexprs": [arg_names[index] for index in constexpr_indices],
        "constexpr_indices": constexpr_indices,
        "do_not_specialize": [_name(item) for item in do_not_specialize_raw],
        "do_not_specialize_raw": _jsonable(do_not_specialize_raw),
    }


def _target_metadata(torch: Any, triton: Any, device: str) -> dict[str, Any]:
    target: dict[str, Any] = {}
    try:
        current = triton.runtime.driver.active.get_current_target()
        target["triton"] = _jsonable(current)
        for attribute in ("backend", "arch", "warp_size"):
            if hasattr(current, attribute):
                target[attribute] = _jsonable(getattr(current, attribute))
    except Exception as exc:
        target["triton_query_error"] = f"{type(exc).__name__}: {exc}"

    try:
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        target["cuda"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
        }
    except Exception as exc:
        target["cuda_query_error"] = f"{type(exc).__name__}: {exc}"
    return target


def _compiler_metadata(torch: Any, triton: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "triton": str(triton.__version__),
        "gate1_worker_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "frontend_compat_protocol": _TRITON_DOT_COMPAT_PROTOCOL,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TRITON_KERNEL_DUMP",
                "TRITON_ALWAYS_COMPILE",
                "TRITON_STORE_BINARY_ONLY",
            )
        },
    }


def _compile(request: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Triton Gate1 requires a CUDA target")

    source = request["source"]
    case = request["case"]
    device = str(case.get("device", "cuda"))
    source_code = str(source["code"])
    code, frontend_compat = _normalize_deprecated_triton_dot(source_code)
    function_name = str(source["function"])

    with torch.cuda.device(device), tempfile.TemporaryDirectory(
        prefix="intentir-gate1-module-"
    ) as temporary:
        module = _load_module(
            code,
            Path(temporary),
            filename=(
                str(source.get("filename"))
                if source.get("filename") is not None
                else None
            ),
            trusted_source=source.get("trust_role") == "trusted_source",
            source_identity_code=source_code,
        )
        kernel = getattr(module, function_name, None)
        if kernel is None:
            raise AttributeError(f"Triton function {function_name!r} is missing")
        if not hasattr(kernel, "warmup") or not hasattr(kernel, "arg_names"):
            raise TypeError(f"{function_name!r} is not a Triton JIT function")

        interface = _interface(kernel)
        declared = set(interface["arg_names"])
        argument_specs = dict(case["arguments"])
        schedule = dict(case.get("schedule", {}))
        unknown_runtime = sorted(set(argument_specs) - declared)
        if unknown_runtime:
            raise ValueError(f"runtime arguments not in Python interface: {unknown_runtime}")
        missing = sorted(declared - set(argument_specs) - set(schedule))
        if missing:
            raise ValueError(f"missing Python interface arguments: {missing}")

        values = {
            name: _runtime_value(torch, spec, device)
            for name, spec in argument_specs.items()
        }
        values.update(schedule)
        grid = tuple(int(dim) for dim in case["grid"])

        # JITFunction.warmup invokes the complete frontend/compiler pipeline but
        # its run() path returns before grid canonicalization and kernel launch.
        # A library symbol may be wrapped in tuning/dispatch layers.  Gate1 C is
        # already frozen, so compile the metadata-owning JITFunction directly;
        # invoking an outer autotuner would both run policy and conflict with C.
        compilable = _jit_metadata_owner(kernel)
        warmup_result = compilable.warmup(grid=grid, **values)
        if warmup_result is None:
            raise RuntimeError("Triton warmup returned no compiled kernel")
        compiled = _compiled_kernel(warmup_result)
        asm = compiled.asm
        asm_keys = sorted(str(key) for key in asm.keys())
        ttir = str(asm["ttir"]) if "ttir" in asm else None
        ttgir = str(asm["ttgir"]) if "ttgir" in asm else None
        if not ttir or not ttgir:
            raise RuntimeError(
                f"compiled artifact lacks TTIR/TTGIR (asm keys: {asm_keys})"
            )

        target = _target_metadata(torch, triton, device)
        metadata_target = getattr(getattr(compiled, "metadata", None), "target", None)
        if metadata_target is not None:
            target["compiled"] = _jsonable(metadata_target)
        compiler_metadata = _compiler_metadata(torch, triton)
        compiler_metadata["frontend_compat"] = frontend_compat
        return {
            "ok": True,
            "ttir": ttir,
            "ttgir": ttgir,
            "python_interface": interface,
            "compiler": compiler_metadata,
            "target": target,
            "diagnostic": "",
            "asm_keys": asm_keys,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    result: dict[str, Any]
    request: Mapping[str, Any] = {}
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if request.get("schema") != _REQUEST_SCHEMA:
            raise ValueError(f"unsupported request schema {request.get('schema')!r}")
        result = _compile(request)
    except BaseException as exc:  # worker must report compiler diagnostics verbatim
        compiler: dict[str, Any] = {}
        target: dict[str, Any] = {}
        try:
            import torch
            import triton

            compiler = _compiler_metadata(torch, triton)
            device = str(request.get("case", {}).get("device", "cuda"))
            target = _target_metadata(torch, triton, device)
        except BaseException:
            # The original traceback is the actionable diagnostic.  Metadata
            # collection is best-effort and must never obscure that failure.
            pass
        result = {
            "ok": False,
            "ttir": None,
            "ttgir": None,
            "python_interface": {},
            "compiler": compiler,
            "target": target,
            "diagnostic": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "asm_keys": [],
        }
    result["elapsed_ms"] = (time.monotonic() - started) * 1000.0
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
