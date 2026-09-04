"""Private compile-only worker for the CuTe Gate1 frontend.

The parent process performs the replay policy check and passes only a JSON
request.  This worker is the first process that imports ``cutlass`` or executes
the Python module.  ``cute.compile`` builds the host launcher and device kernel
but does not launch it; the returned MLIR and PTX are the Level-2 observation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
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


_REQUEST_SCHEMA = 1
_OBSERVER_PROTOCOL = "cute-4.4-mlir-ptx-topology-v1"


def _function_interface(code: str, name: str) -> dict[str, Any]:
    tree = ast.parse(code)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"CuTe function {name!r} must resolve exactly once")
    node = matches[0]
    if node.args.vararg is not None or node.args.kwarg is not None:
        raise ValueError("CuTe Gate1 functions may not use variadic arguments")
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return {
        "function_name": name,
        "arg_names": [argument.arg for argument in arguments],
        "annotations": [
            ast.unparse(argument.annotation) if argument.annotation is not None else None
            for argument in arguments
        ],
    }


def _load_module(code: str, filename: str, directory: Path) -> ModuleType:
    digest = hashlib.sha256(code.encode()).hexdigest()[:16]
    module = ModuleType(f"_intentir_cute_gate1_{digest}_{os.getpid()}")
    module.__file__ = filename
    module.__package__ = None
    sys.modules[module.__name__] = module
    try:
        executable = compile(code, filename, "exec")
        exec(executable, module.__dict__)
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def _cute_dtype(cutlass: Any, name: str) -> Any:
    aliases = {
        "float16": "Float16",
        "half": "Float16",
        "bfloat16": "BFloat16",
        "float32": "Float32",
        "float": "Float32",
        "float64": "Float64",
        "int8": "Int8",
        "int16": "Int16",
        "int32": "Int32",
        "int64": "Int64",
        "uint8": "Uint8",
        "uint16": "Uint16",
        "uint32": "Uint32",
        "uint64": "Uint64",
        "bool": "Boolean",
    }
    normalized = name.removeprefix("torch.").lower()
    attribute = aliases.get(normalized)
    if attribute is None or not hasattr(cutlass, attribute):
        raise ValueError(f"unsupported CuTe tensor dtype {name!r}")
    return getattr(cutlass, attribute)


def _scalar_value(spec: Mapping[str, Any]) -> int | float | bool:
    value = spec["value"]
    dtype = str(spec.get("dtype") or "").removeprefix("torch.").lower()
    if dtype == "bool":
        return bool(value)
    if dtype.startswith(("int", "uint")):
        return int(value)
    if dtype.startswith(("float", "bfloat")) or dtype in {"half", "double"}:
        return float(value)
    return value


def _runtime_value(cutlass: Any, cute: Any, spec: Mapping[str, Any]) -> Any:
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor

    kind = spec.get("kind")
    if kind == "none":
        return None
    if kind == "scalar":
        return _scalar_value(spec)
    if kind != "tensor":
        raise ValueError(f"unknown CuTe entry argument kind {kind!r}")
    dtype = _cute_dtype(cutlass, str(spec["dtype"]))
    shape = tuple(int(dimension) for dimension in spec["shape"])
    strides = spec.get("strides")
    alignment = int(spec.get("assumed_align") or 16)
    if strides is None:
        return make_fake_compact_tensor(dtype, shape, assumed_align=alignment)
    return make_fake_tensor(
        dtype,
        shape,
        tuple(int(stride) for stride in strides),
        assumed_align=alignment,
    )


def _compiler_metadata(torch: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "nvidia_cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "nvidia_cutlass_dsl_libs_base": importlib.metadata.version(
            "nvidia-cutlass-dsl-libs-base"
        ),
        "nvidia_cutlass_dsl_libs_cu13": importlib.metadata.version(
            "nvidia-cutlass-dsl-libs-cu13"
        ),
        "gate1_worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "observer_protocol": _OBSERVER_PROTOCOL,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUTE_DSL_ARCH",
                "CUTE_DSL_KEEP_IR",
                "CUTE_DSL_KEEP_PTX",
                "CUTE_DSL_NO_CACHE",
            )
        },
    }


def _target_metadata(torch: Any, device: str) -> dict[str, Any]:
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "cuda": {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
        },
        "arch": os.environ.get("CUTE_DSL_ARCH"),
    }


def _compile(request: Mapping[str, Any]) -> dict[str, Any]:
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import make_fake_stream

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CuTe Gate1 requires a CUDA target")

    source = request["source"]
    case = request["case"]
    code = str(source["code"])
    filename = str(source.get("filename") or "<gate1-cute-replay>")
    device_function = str(source["device_function"])
    entry_function = str(source["entry_function"])
    device_interface = _function_interface(code, device_function)
    entry_interface = _function_interface(code, entry_function)

    device = str(case.get("device", "cuda"))
    with torch.cuda.device(device), tempfile.TemporaryDirectory(
        prefix="intentir-cute-gate1-module-"
    ) as temporary:
        module = _load_module(code, filename, Path(temporary))
        kernel = getattr(module, device_function, None)
        entry = getattr(module, entry_function, None)
        if kernel is None or entry is None:
            raise AttributeError("CuTe device or entry function is missing")
        if not callable(kernel) or not callable(entry):
            raise TypeError("CuTe device and entry symbols must be callable")

        values = [
            _runtime_value(cutlass, cute, argument)
            for argument in case["entry_arguments"]
        ]
        if bool(case.get("append_stream", True)):
            values.append(make_fake_stream())
        if len(values) != len(entry_interface["arg_names"]):
            raise ValueError(
                "CuTe compile entry arity differs from the frozen positional harness"
            )

        compiled = cute.compile(entry, *values)
        mlir = compiled.__mlir__
        ptx = compiled.__ptx__
        cubin = compiled.__cubin__
        if not isinstance(mlir, str) or not mlir.strip():
            raise RuntimeError("CuTe compile artifact lacks MLIR")
        if not isinstance(ptx, str) or not ptx.strip():
            raise RuntimeError("CuTe compile artifact lacks PTX")
        cubin_digest = (
            hashlib.sha256(bytes(cubin)).hexdigest()
            if isinstance(cubin, (bytes, bytearray))
            else None
        )
        return {
            "ok": True,
            "mlir": mlir,
            "ptx": ptx,
            "cubin_sha256": cubin_digest,
            "python_interface": {
                "device": device_interface,
                "entry": entry_interface,
            },
            "compiler": _compiler_metadata(torch),
            "target": _target_metadata(torch, device),
            "diagnostic": "",
            "asm_keys": ["mlir", "ptx", "cubin"],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    request: Mapping[str, Any] = {}
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if request.get("schema") != _REQUEST_SCHEMA:
            raise ValueError(f"unsupported request schema {request.get('schema')!r}")
        result = _compile(request)
    except BaseException as exc:  # preserve the compiler's actionable traceback
        compiler: dict[str, Any] = {}
        target: dict[str, Any] = {}
        try:
            import torch

            compiler = _compiler_metadata(torch)
            target = _target_metadata(
                torch, str(request.get("case", {}).get("device", "cuda"))
            )
        except BaseException:
            pass
        result = {
            "ok": False,
            "mlir": None,
            "ptx": None,
            "cubin_sha256": None,
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
