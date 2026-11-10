"""Extraction and fail-closed policy for model-generated CuTe replay modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import NoReturn

from .codegen import CodeExtractionError, extract_python_code


class UnsafeCuteReplayCodeError(ValueError):
    """The module is outside the reviewed public CuTe Python subset."""


@dataclass(frozen=True)
class CuteModuleValidation:
    kernel_name: str
    entry_name: str
    tree: ast.Module


_ALLOWED_IMPORTS = {
    "cutlass": None,
    "cutlass.cute": "cute",
    "cutlass.pipeline": None,
}
_ALLOWED_FROM_IMPORTS = {
    "cutlass": {
        "Boolean",
        "BFloat16",
        "Float16",
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64",
    },
    "cutlass.cute.nvgpu": {"cpasync"},
    "cutlass.pipeline.sm90": {"PipelineTmaAsync", "make_pipeline_state"},
}
_SAFE_NAME_CALLS = {
    "bool",
    "float",
    "int",
    "range",
    "make_pipeline_state",
    *_ALLOWED_FROM_IMPORTS["cutlass"],
}
_FORBIDDEN_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Match,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.Yield,
    ast.YieldFrom,
)
_FORBIDDEN_CALL_NAMES = {
    "breakpoint",
    "compile",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "hasattr",
    "input",
    "locals",
    "open",
    "print",
    "setattr",
    "vars",
    "__import__",
}
_FORBIDDEN_ATTRIBUTES = {
    "chmod",
    "connect",
    "dump",
    "dumps",
    "exec",
    "load",
    "loads",
    "mkdir",
    "open",
    "popen",
    "read",
    "read_bytes",
    "read_text",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "send",
    "socket",
    "system",
    "unlink",
    "write",
    "write_bytes",
    "write_text",
}
_MISSING = object()


def _fail(node: ast.AST | None, message: str) -> NoReturn:
    if node is not None and getattr(node, "lineno", None) is not None:
        message = f"line {node.lineno}: {message}"
    raise UnsafeCuteReplayCodeError(message)


def _identifier(value: str, node: ast.AST, what: str) -> None:
    if "__" in value:
        _fail(node, f"dunder {what} {value!r} is forbidden")


def _validate_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            expected_alias = _ALLOWED_IMPORTS.get(alias.name, _MISSING)
            if expected_alias is _MISSING or alias.asname != expected_alias:
                _fail(node, f"unsupported CuTe replay import {alias.name!r}")
        return
    if node.level != 0 or node.module not in _ALLOWED_FROM_IMPORTS:
        _fail(node, "unsupported CuTe replay from-import")
    allowed = _ALLOWED_FROM_IMPORTS[node.module]
    for alias in node.names:
        if alias.name not in allowed or alias.asname is not None:
            _fail(node, f"unsupported import {node.module}.{alias.name}")


def _decorator(node: ast.AST, expected: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "cute"
        and node.attr == expected
    )


def _literal_constant(node: ast.AST) -> bool:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return False
    if isinstance(value, (bool, int, float, str, type(None))):
        return True
    if isinstance(value, (tuple, list)):
        return all(isinstance(item, (bool, int, float, str)) for item in value)
    return False


def _validate_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    decorator: str,
    required_kernel: str,
) -> None:
    if isinstance(node, ast.AsyncFunctionDef):
        _fail(node, "async CuTe replay functions are forbidden")
    if len(node.decorator_list) != 1 or not _decorator(
        node.decorator_list[0], decorator
    ):
        _fail(node, f"function must be decorated only with @cute.{decorator}")
    if node.args.vararg is not None or node.args.kwarg is not None:
        _fail(node, "variadic CuTe replay functions are forbidden")
    if getattr(node, "type_params", ()):
        _fail(node, "generic CuTe replay functions are forbidden")
    # Defaults execute while loading the module, before CuTe compiles its body.
    for default in (*node.args.defaults, *node.args.kw_defaults):
        if default is not None and not _literal_constant(default):
            _fail(default, "CuTe replay defaults must be literals")
    for argument in [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]:
        _identifier(argument.arg, argument, "argument")
        _validate_annotation(argument.annotation)
    _validate_annotation(node.returns)
    for item in ast.walk(ast.Module(body=node.body, type_ignores=[])):
        if isinstance(item, _FORBIDDEN_NODES):
            _fail(item, f"{type(item).__name__} is forbidden in CuTe replay code")
        if isinstance(item, ast.Name):
            _identifier(item.id, item, "name")
        if isinstance(item, ast.Attribute):
            _identifier(item.attr, item, "attribute")
            if item.attr in _FORBIDDEN_ATTRIBUTES:
                _fail(item, f"host-side attribute call {item.attr!r} is forbidden")
        if isinstance(item, ast.Call):
            target = item.func
            if isinstance(target, ast.Name):
                if target.id in _FORBIDDEN_CALL_NAMES:
                    _fail(target, f"host-side call {target.id!r} is forbidden")
                if target.id not in _SAFE_NAME_CALLS | {required_kernel}:
                    _fail(target, f"unqualified call {target.id!r} is not public CuTe")
            elif not isinstance(target, ast.Attribute):
                _fail(target, "dynamic CuTe replay call target is forbidden")


def _validate_annotation(node: ast.AST | None) -> None:
    """Permit public type names only, never executable annotation expressions."""

    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return
    numeric = _ALLOWED_FROM_IMPORTS["cutlass"]
    if isinstance(node, ast.Name) and node.id in numeric | {"bool", "int", "float"}:
        return
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if (
            node.value.id == "cutlass" and node.attr in numeric | {"Constexpr"}
        ) or (
            node.value.id == "cute"
            and node.attr in {"Tensor", "Shape", "Stride", "Layout", "IntTuple"}
        ):
            return
    _fail(node, "CuTe replay annotations must be approved public type names")


def validate_cute_module(
    code: str,
    *,
    required_kernel: str,
    required_entry: str,
) -> CuteModuleValidation:
    """Validate imports, constants and exactly one kernel/entry pair."""

    if not isinstance(code, str) or not code.strip():
        raise UnsafeCuteReplayCodeError("CuTe replay module is empty")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCuteReplayCodeError(f"invalid Python: {exc}") from exc
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _validate_import(node)
            continue
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
                or not node.targets[0].id.isupper()
                or not _literal_constant(node.value)
            ):
                _fail(node, "top-level CuTe values must be uppercase literals")
            _identifier(node.targets[0].id, node, "constant")
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _identifier(node.name, node, "function")
            if node.name in functions:
                _fail(node, f"duplicate function {node.name!r}")
            functions[node.name] = node
            continue
        _fail(node, "CuTe replay top level permits only imports, constants and functions")
    if set(functions) != {required_kernel, required_entry}:
        raise UnsafeCuteReplayCodeError(
            "CuTe replay must define exactly the required kernel and compile entry"
        )
    _validate_function(
        functions[required_kernel],
        decorator="kernel",
        required_kernel=required_kernel,
    )
    _validate_function(
        functions[required_entry],
        decorator="jit",
        required_kernel=required_kernel,
    )
    entry_calls_kernel = any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == required_kernel
        for item in ast.walk(functions[required_entry])
    )
    if not entry_calls_kernel:
        _fail(functions[required_entry], "CuTe compile entry never invokes the kernel")
    return CuteModuleValidation(required_kernel, required_entry, tree)


def extract_cute_module(
    response: str,
    *,
    required_kernel: str,
    required_entry: str,
) -> str:
    code = extract_python_code(response)
    validate_cute_module(
        code,
        required_kernel=required_kernel,
        required_entry=required_entry,
    )
    return code


__all__ = [
    "CodeExtractionError",
    "CuteModuleValidation",
    "UnsafeCuteReplayCodeError",
    "extract_cute_module",
    "validate_cute_module",
]
