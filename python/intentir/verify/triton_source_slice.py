"""Deterministic fixed-C source projection for real Triton kernel modules.

The projection is Harness-side plumbing.  It selects one top-level kernel and
the module declarations required to load it, while replacing source-side
autotune/heuristic selection with the concrete schedule already bound by the
Harness.  It never invents an operator, predicate, or implementation strategy.
"""

from __future__ import annotations

import ast
import builtins
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TRITON_SOURCE_SLICE_PROTOCOL = "intentir.triton.fixed-c-source-slice.v3"
_VARARG_ARRAY_ANNOTATION = "VAR_ARGS_ARRAY"


class TritonSourceSliceError(ValueError):
    """The requested real-source projection cannot be formed unambiguously."""


def _annotation_is_constexpr(annotation: ast.expr | None) -> bool:
    return annotation is not None and ast.unparse(annotation).endswith(
        "constexpr"
    )


def _decorator_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    try:
        return ast.unparse(target)
    except ValueError:
        return "<unknown>"


def _is_jit_decorator(node: ast.expr) -> bool:
    # ``@triton.jit(...)`` is a decorator factory.  The source slice cannot
    # prove which callable it returns without executing the frontend, so keep
    # the accepted form to the direct decorator object only.
    return not isinstance(node, ast.Call) and _decorator_name(node) in {
        "triton.jit", "jit"
    }


def _is_schedule_decorator(node: ast.expr) -> bool:
    return _decorator_name(node) in {
        "triton.autotune",
        "triton.heuristics",
        "autotune",
        "heuristics",
    }


class _BoundNameCollector(ast.NodeVisitor):
    """Collect names bound by one module statement, without entering scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(
            alias.asname or alias.name.split(".")[0] for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)


def _bound_names(node: ast.AST) -> set[str]:
    collector = _BoundNameCollector()
    collector.visit(node)
    return collector.names


class _LoadNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def _load_names(node: ast.AST) -> set[str]:
    collector = _LoadNameCollector()
    collector.visit(node)
    return collector.names


def _function_dependencies(function: ast.FunctionDef) -> set[str]:
    local_names = {
        function.name,
        *(
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        ),
    }
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node is not function:
                local_names.add(node.name)
    retained_decorators = [
        decorator
        for decorator in function.decorator_list
        if not _is_schedule_decorator(decorator)
    ]
    projection = copy.deepcopy(function)
    projection.decorator_list = retained_decorators
    return _load_names(projection) - local_names - set(dir(builtins))


def _statement_dependencies(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.FunctionDef):
        return _function_dependencies(statement)
    return _load_names(statement) - _bound_names(statement) - set(dir(builtins))


_DYNAMIC_REFLECTION_NAMES = frozenset(
    {
        "eval",
        "exec",
        "getattr",
        "globals",
        "__import__",
        "locals",
        "setattr",
        "vars",
    }
)
_DYNAMIC_REFLECTION_ATTRIBUTES = frozenset(
    {
        "__bases__",
        "__class__",
        "__dict__",
        "__getattr__",
        "__getattribute__",
        "__mro__",
        "__subclasses__",
        "__setattr__",
        "__delattr__",
    }
)
_STATIC_TRITON_LIBDEVICE_NAMES = frozenset({"pow", "rsqrt"})


def _static_attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        prefix = _static_attribute_path(node.value)
        return (*prefix, node.attr) if prefix is not None else None
    return None


def _direct_callable_names(statements: Iterable[ast.stmt]) -> set[str]:
    """Return names whose call target is statically declared by the slice."""

    names = set(dir(builtins)) - _DYNAMIC_REFLECTION_NAMES
    for statement in statements:
        # Imports in real Triton modules are often guarded by a version or
        # device check and may therefore live inside ``if``/``try`` nodes.
        # Walk the selected statements, but collect only declarations that
        # introduce call targets.  In particular, do not treat arbitrary
        # assignment targets as callable merely because they are closed names.
        for node in ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Import):
                names.update(
                    alias.asname or alias.name.split(".")[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in node.names)
    return names


def _normalize_triton_compatibility_aliases(
    statements: Iterable[ast.stmt],
) -> tuple[tuple[ast.stmt, ...], tuple[str, ...]]:
    """Resolve the known xFormers libdevice compatibility shim statically.

    xFormers uses ``libdevice_find("rsqrt")`` (and, in older kernels,
    ``libdevice_find("pow")``) so one source tree can run against several
    Triton releases.  The resolver itself performs reflective module lookup;
    retaining it in a fixed Source projection would make the projection depend
    on the ambient Python installation.  Its first supported candidate is the
    direct Triton libdevice module, so this exact literal form can be replaced
    without changing the selected kernel operation.

    Arbitrary resolver calls, non-literal names, and unknown libdevice names
    remain for the normal fail-closed validator to reject.
    """

    original = tuple(statements)
    resolver_names: set[str] = set()
    resolver_imports: set[int] = set()
    for index, statement in enumerate(original):
        if not isinstance(statement, ast.ImportFrom):
            continue
        if statement.module != "xformers.triton.importing":
            continue
        for alias in statement.names:
            if alias.name == "libdevice_find":
                resolver_names.add(alias.asname or alias.name)
                resolver_imports.add(index)
    if not resolver_names:
        return original, ()

    normalized: list[ast.stmt] = []
    labels: list[str] = []
    for statement in original:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in resolver_names
                and len(value.args) == 1
                and not value.keywords
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)
                and value.args[0].value in _STATIC_TRITON_LIBDEVICE_NAMES
            ):
                libdevice_name = value.args[0].value
                replacement = ast.ImportFrom(
                    module="triton.language.extra.libdevice",
                    names=[
                        ast.alias(
                            name=libdevice_name,
                            asname=(
                                target.id
                                if target.id != libdevice_name
                                else None
                            ),
                        )
                    ],
                    level=0,
                )
                ast.copy_location(replacement, statement)
                normalized.append(replacement)
                labels.append(
                    "compat:xformers.triton.importing:"
                    f"{target.id}->{libdevice_name}"
                )
                continue
        normalized.append(statement)

    remaining_resolver_loads = {
        node.id
        for statement in normalized
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in resolver_names
    }
    if not remaining_resolver_loads:
        normalized = [
            statement
            for index, statement in enumerate(normalized)
            if index not in resolver_imports
        ]
    return tuple(normalized), tuple(labels)


class _StaticDependencySurfaceValidator(ast.NodeVisitor):
    """Reject dependency edges that cannot be closed from the selected AST.

    The validator deliberately accepts ordinary Triton attribute calls such as
    ``tl.load`` and ``x.to``.  It rejects reflection, dynamic imports, callable
    values obtained through a subscript/factory, and direct calls whose name is
    not a top-level function/import or a builtin.  This keeps registry lookup,
    function factories, and local callable aliases from silently pulling an
    unselected sibling into the source projection.
    """

    def __init__(self, direct_callable_names: set[str]) -> None:
        self.direct_callable_names = direct_callable_names

    def _error(self, node: ast.AST, detail: str) -> None:
        line = getattr(node, "lineno", "?")
        raise TritonSourceSliceError(
            f"selected-kernel dependency is not statically closed at line "
            f"{line}: {detail}"
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._validate_decorators(node)
        for field, value in ast.iter_fields(node):
            if field == "decorator_list":
                continue
            if isinstance(value, ast.AST):
                self.visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._validate_decorators(node)
        for field, value in ast.iter_fields(node):
            if field == "decorator_list":
                continue
            if isinstance(value, ast.AST):
                self.visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)

    def _validate_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            if _is_schedule_decorator(decorator) or _is_jit_decorator(decorator):
                continue
            self._error(
                decorator,
                "decorator must be a direct Triton JIT or known schedule decorator",
            )

    def _is_closed_call_target(self, node: ast.Call) -> bool:
        """Return whether a call used as an attribute receiver is closed.

        Triton commonly chains a value-producing intrinsic with a method, for
        example ``tl.load(ptr).to(tl.float32)`` or
        ``tl.zeros(shape).to(tl.float32)``.  The outer method call has an
        ``ast.Call`` receiver, but that is not the same shape as an indirect
        callable such as ``factory()(x)`` or ``registry()[name](x)``.  Keep
        the distinction structural: the inner call must still target a
        directly declared name or a static attribute path.
        """

        if isinstance(node.func, ast.Name):
            return node.func.id in self.direct_callable_names
        if not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr in _DYNAMIC_REFLECTION_ATTRIBUTES:
            return False
        return self._is_closed_value_expression(node.func.value)

    def _is_closed_value_expression(self, node: ast.AST) -> bool:
        """Prove that an expression can produce a value without an indirect call.

        Triton values are routinely transformed before a method call, for
        example ``(p * q).to(dtype)``, ``(offs + 1).to(tl.float32)``, and
        ``x[mask, None].to(dtype)``.  Checking only Name/Attribute/Call
        receivers rejects those real kernels.  This grammar follows value
        expressions recursively while keeping callable targets stricter: a
        Subscript or a factory can produce a value, but neither can become an
        outer callable target.
        """

        if isinstance(node, (ast.Name, ast.Constant)):
            return True
        if isinstance(node, ast.Attribute):
            if node.attr in _DYNAMIC_REFLECTION_ATTRIBUTES:
                return False
            return self._is_closed_value_expression(node.value)
        if isinstance(node, ast.Call):
            return self._is_closed_call_target(node)
        if isinstance(node, ast.Subscript):
            return self._is_closed_value_expression(node.value) and self._is_closed_value_expression(
                node.slice
            )
        if isinstance(node, ast.Slice):
            return all(
                item is None or self._is_closed_value_expression(item)
                for item in (node.lower, node.upper, node.step)
            )
        if isinstance(node, ast.Starred):
            return self._is_closed_value_expression(node.value)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return all(self._is_closed_value_expression(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(
                (key is None or self._is_closed_value_expression(key))
                and self._is_closed_value_expression(value)
                for key, value in zip(node.keys, node.values, strict=True)
            )
        if isinstance(node, ast.BinOp):
            return self._is_closed_value_expression(node.left) and self._is_closed_value_expression(
                node.right
            )
        if isinstance(node, ast.UnaryOp):
            return self._is_closed_value_expression(node.operand)
        if isinstance(node, ast.BoolOp):
            return all(self._is_closed_value_expression(item) for item in node.values)
        if isinstance(node, ast.Compare):
            return self._is_closed_value_expression(node.left) and all(
                self._is_closed_value_expression(item) for item in node.comparators
            )
        if isinstance(node, ast.IfExp):
            return (
                self._is_closed_value_expression(node.test)
                and self._is_closed_value_expression(node.body)
                and self._is_closed_value_expression(node.orelse)
            )
        return False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "importlib" or alias.name.startswith("importlib."):
                self._error(node, "dynamic importlib dependency is forbidden")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "importlib" or (node.module or "").startswith("importlib."):
            self._error(node, "dynamic importlib dependency is forbidden")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _DYNAMIC_REFLECTION_ATTRIBUTES:
            self._error(node, f"reflective attribute {node.attr!r} is forbidden")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        path = _static_attribute_path(node.func)
        if path is not None and (
            path[-1] in _DYNAMIC_REFLECTION_NAMES
            or path[:1] == ("importlib",)
        ):
            self._error(node, f"dynamic callable {'.'.join(path)!r} is forbidden")
        if isinstance(node.func, ast.Name):
            if node.func.id not in self.direct_callable_names:
                self._error(
                    node,
                    f"callable name {node.func.id!r} is not a selected function, "
                    "import, or builtin",
                )
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in _DYNAMIC_REFLECTION_ATTRIBUTES:
                self._error(node, f"reflective callable attribute {node.func.attr!r} is forbidden")
            receiver = node.func.value
            if not self._is_closed_value_expression(receiver):
                self._error(
                    node,
                    "attribute call has a dynamic receiver; callable factories and "
                    "registry lookups are forbidden",
                )
        else:
            self._error(
                node,
                "call target is not a statically named function or attribute; "
                "indirect callable/registry dependency is forbidden",
            )
        self.generic_visit(node)


def _validate_static_dependency_surface(statements: Iterable[ast.stmt]) -> None:
    selected = tuple(statements)
    validator = _StaticDependencySurfaceValidator(_direct_callable_names(selected))
    for statement in selected:
        validator.visit(statement)


@dataclass(frozen=True)
class TritonSourceSlice:
    code: str
    function: str
    parameters: tuple[str, ...]
    constexpr_parameters: tuple[str, ...]
    included_declarations: tuple[str, ...]
    included_functions: tuple[str, ...]
    removed_decorators: tuple[str, ...]
    inlined_imports: tuple[str, ...] = ()
    protocol: str = TRITON_SOURCE_SLICE_PROTOCOL
    vararg_arity: int | None = None

    def validate_abi(
        self,
        *,
        parameter_order: Iterable[str],
        constexpr_parameters: Iterable[str],
    ) -> None:
        expected_order = tuple(parameter_order)
        expected_constexpr = tuple(constexpr_parameters)
        if expected_order != self.parameters:
            raise TritonSourceSliceError(
                "Harness parameter order differs from the selected source ABI: "
                f"{expected_order!r} != {self.parameters!r}"
            )
        if expected_constexpr != self.constexpr_parameters:
            raise TritonSourceSliceError(
                "Harness constexpr set/order differs from the selected source ABI: "
                f"{expected_constexpr!r} != {self.constexpr_parameters!r}"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "function": self.function,
            "parameters": list(self.parameters),
            "constexpr_parameters": list(self.constexpr_parameters),
            "included_declarations": list(self.included_declarations),
            "included_functions": list(self.included_functions),
            "removed_decorators": list(self.removed_decorators),
            "inlined_imports": list(self.inlined_imports),
            "vararg_arity": self.vararg_arity,
        }


def build_triton_source_slice(
    source: str,
    *,
    source_symbol: str,
) -> TritonSourceSlice:
    """Select one real kernel and its mechanically required module closure."""

    if not source_symbol.isidentifier():
        raise TritonSourceSliceError("source symbol must be a Python identifier")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise TritonSourceSliceError(f"source cannot be parsed: {exc}") from exc
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == source_symbol
    ]
    if len(matches) != 1:
        raise TritonSourceSliceError(
            f"source symbol {source_symbol!r} must resolve exactly once at module "
            f"top level; found {len(matches)}"
        )
    target = matches[0]
    if not any(_is_jit_decorator(item) for item in target.decorator_list):
        raise TritonSourceSliceError("selected source function is not @triton.jit")
    unexpected = [
        _decorator_name(item)
        for item in target.decorator_list
        if not _is_jit_decorator(item) and not _is_schedule_decorator(item)
    ]
    if unexpected:
        raise TritonSourceSliceError(
            "selected kernel has unsupported non-schedule decorators: "
            + ", ".join(unexpected)
        )

    bindings: dict[str, list[int]] = {}
    for index, statement in enumerate(tree.body):
        for name in _bound_names(statement):
            bindings.setdefault(name, []).append(index)
    selected: set[int] = {tree.body.index(target)}
    pending = list(_function_dependencies(target))
    unresolved: set[str] = set()
    while pending:
        name = pending.pop()
        owners = bindings.get(name, [])
        if not owners:
            unresolved.add(name)
            continue
        if len(owners) != 1:
            raise TritonSourceSliceError(
                f"module dependency {name!r} is bound by multiple top-level statements"
            )
        owner = owners[0]
        if owner in selected:
            continue
        selected.add(owner)
        pending.extend(_statement_dependencies(tree.body[owner]))
    # Names supplied by Triton's compile-time language may legitimately have no
    # Python declaration (for example function-local type properties).  Every
    # module-global call/helper/import, however, must have resolved above.
    called_names = {
        call.func.id
        for call in ast.walk(target)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    unresolved_calls = sorted(unresolved & called_names)
    if unresolved_calls:
        raise TritonSourceSliceError(
            "selected kernel has unresolved called globals: "
            + ", ".join(unresolved_calls)
        )

    selected_statements = tuple(tree.body[index] for index in sorted(selected))
    selected_statements, normalized_aliases = _normalize_triton_compatibility_aliases(
        selected_statements
    )
    _validate_static_dependency_surface(selected_statements)

    projected_body: list[ast.stmt] = []
    removed: list[str] = []
    included_declarations: list[str] = []
    included_functions: list[str] = []
    for statement in selected_statements:
        item = copy.deepcopy(statement)
        names = sorted(_bound_names(item))
        included_declarations.extend(names)
        if isinstance(item, ast.FunctionDef):
            included_functions.append(item.name)
            if item.name == source_symbol:
                retained: list[ast.expr] = []
                for decorator in item.decorator_list:
                    if _is_schedule_decorator(decorator):
                        removed.append(_decorator_name(decorator))
                    else:
                        retained.append(decorator)
                item.decorator_list = retained
        projected_body.append(item)
    projected = ast.Module(body=projected_body, type_ignores=[])
    ast.fix_missing_locations(projected)
    code = ast.unparse(projected) + "\n"
    parameters = tuple(
        argument.arg
        for argument in (
            *target.args.posonlyargs,
            *target.args.args,
            *target.args.kwonlyargs,
        )
    )
    constexprs = tuple(
        argument.arg
        for argument in (
            *target.args.posonlyargs,
            *target.args.args,
            *target.args.kwonlyargs,
        )
        if _annotation_is_constexpr(argument.annotation)
    )
    return TritonSourceSlice(
        code,
        source_symbol,
        parameters,
        constexprs,
        tuple(dict.fromkeys(included_declarations)),
        tuple(included_functions),
        tuple(removed),
        tuple(normalized_aliases),
    )


def _is_vararg_array_annotation(annotation: ast.expr | None) -> bool:
    return (
        isinstance(annotation, ast.Constant)
        and annotation.value == _VARARG_ARRAY_ANNOTATION
    )


def _function_parameter_nodes(function: ast.FunctionDef) -> tuple[ast.arg, ...]:
    parameters: list[ast.arg] = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    if function.args.vararg is not None:
        parameters.append(function.args.vararg)
    if function.args.kwarg is not None:
        parameters.append(function.args.kwarg)
    return tuple(parameters)


class _LexicalVarargMarkerCollector(ast.NodeVisitor):
    """Collect local markers without entering a nested lexical scope."""

    def __init__(self) -> None:
        self.markers: list[ast.AnnAssign] = []

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _is_vararg_array_annotation(node.annotation):
            self.markers.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _function_vararg_markers(function: ast.FunctionDef) -> tuple[ast.AST, ...]:
    parameter_markers = [
        argument
        for argument in _function_parameter_nodes(function)
        if _is_vararg_array_annotation(argument.annotation)
    ]
    collector = _LexicalVarargMarkerCollector()
    for statement in function.body:
        collector.visit(statement)
    return (*parameter_markers, *collector.markers)


def _all_vararg_markers(node: ast.AST) -> tuple[ast.AST, ...]:
    return tuple(
        item
        for item in ast.walk(node)
        if (
            isinstance(item, ast.arg)
            and _is_vararg_array_annotation(item.annotation)
        )
        or (
            isinstance(item, ast.AnnAssign)
            and _is_vararg_array_annotation(item.annotation)
        )
    )


def _selected_vararg_entry(
    tree: ast.Module,
    *,
    function_name: str,
) -> tuple[ast.FunctionDef, bool]:
    targets = [
        function
        for function in tree.body
        if isinstance(function, ast.FunctionDef)
        and function.name == function_name
    ]
    if len(targets) != 1:
        raise TritonSourceSliceError(
            "selected vararg entry must resolve to one top-level function"
        )
    target = targets[0]
    lexical_markers = _function_vararg_markers(target)
    lexical_marker_ids = {id(item) for item in lexical_markers}
    foreign_markers = [
        item
        for item in _all_vararg_markers(tree)
        if id(item) not in lexical_marker_ids
    ]
    if foreign_markers:
        lines = sorted(
            {
                str(getattr(item, "lineno", "?"))
                for item in foreign_markers
            }
        )
        raise TritonSourceSliceError(
            "VAR_ARGS_ARRAY markers outside the selected top-level entry "
            "lexical scope are unsupported (line(s): "
            + ", ".join(lines)
            + ")"
        )
    return target, bool(lexical_markers)


class _VarargLoopSubstituter(ast.NodeTransformer):
    def __init__(
        self,
        *,
        loop_variable: str,
        inline_variables: set[str],
        index: int,
    ) -> None:
        self.loop_variable = loop_variable
        self.inline_variables = inline_variables
        self.index = index

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.inline_variables
            and isinstance(node.slice, ast.Name)
            and node.slice.id == self.loop_variable
        ):
            return ast.copy_location(
                ast.Name(
                    id=f"{node.value.id}{self.index}",
                    ctx=copy.deepcopy(node.ctx),
                ),
                node,
            )
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.loop_variable:
            return ast.copy_location(ast.Constant(self.index), node)
        return node

    def _preserve_nested_scope(self, node: ast.AST) -> ast.AST:
        forbidden = {self.loop_variable, *self.inline_variables}
        captured = sorted(
            {
                item.id
                for item in ast.walk(node)
                if isinstance(item, ast.Name) and item.id in forbidden
            }
        )
        if captured:
            raise TritonSourceSliceError(
                "VAR_ARGS_ARRAY expansion cannot rewrite across a nested "
                f"lexical scope; referenced names: {captured}"
            )
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._preserve_nested_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._preserve_nested_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._preserve_nested_scope(node)

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return self._preserve_nested_scope(node)

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self._preserve_nested_scope(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return self._preserve_nested_scope(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return self._preserve_nested_scope(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return self._preserve_nested_scope(node)


class _ClosedVarargExpander(ast.NodeTransformer):
    def __init__(self, arity: int) -> None:
        self.arity = arity
        self.inline_variables: set[str] = set()
        self.marker_count = 0
        self._entry: ast.FunctionDef | None = None

    def expand_entry(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if self._entry is not None:
            raise RuntimeError("VAR_ARGS_ARRAY expander entry is already set")
        self._entry = node
        result = self.visit(node)
        assert isinstance(result, ast.FunctionDef)
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node is not self._entry:
            return node
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return node

    def visit_arguments(self, node: ast.arguments) -> ast.AST:
        unsupported = [
            argument.arg
            for argument in (*node.posonlyargs, *node.kwonlyargs)
            if _is_vararg_array_annotation(argument.annotation)
        ]
        if node.vararg is not None and _is_vararg_array_annotation(
            node.vararg.annotation
        ):
            unsupported.append(node.vararg.arg)
        if node.kwarg is not None and _is_vararg_array_annotation(
            node.kwarg.annotation
        ):
            unsupported.append(node.kwarg.arg)
        if unsupported:
            raise TritonSourceSliceError(
                "VAR_ARGS_ARRAY is supported only on ordinary positional "
                f"parameters; found {sorted(unsupported)}"
            )
        positional = [*node.posonlyargs, *node.args]
        defaulted = positional[-len(node.defaults):] if node.defaults else []
        defaulted_markers = sorted(
            argument.arg
            for argument in defaulted
            if _is_vararg_array_annotation(argument.annotation)
        )
        if defaulted_markers:
            raise TritonSourceSliceError(
                "VAR_ARGS_ARRAY positional parameters with defaults are "
                "unsupported because expansion cannot preserve their ABI "
                f"unambiguously: {defaulted_markers}"
            )
        expanded: list[ast.arg] = []
        for argument in node.args:
            if not _is_vararg_array_annotation(argument.annotation):
                expanded.append(self.visit(argument))
                continue
            self.marker_count += 1
            self.inline_variables.add(argument.arg)
            expanded.extend(
                ast.copy_location(ast.arg(arg=f"{argument.arg}{index}"), argument)
                for index in range(self.arity)
            )
        node.args = expanded
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if not _is_vararg_array_annotation(node.annotation):
            return self.generic_visit(node)
        if (
            node.value is not None
            or node.simple != 1
            or not isinstance(node.target, ast.Name)
        ):
            raise TritonSourceSliceError(
                "local VAR_ARGS_ARRAY declarations must be bare annotated names"
            )
        self.marker_count += 1
        self.inline_variables.add(node.target.id)
        return None

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
        iterator = node.iter
        exact = (
            isinstance(node.target, ast.Name)
            and isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
            and len(iterator.args) == 1
            and not iterator.keywords
            and isinstance(iterator.args[0], ast.Call)
            and isinstance(iterator.args[0].func, ast.Name)
            and iterator.args[0].func.id == "len"
            and len(iterator.args[0].args) == 1
            and not iterator.args[0].keywords
            and isinstance(iterator.args[0].args[0], ast.Name)
            and iterator.args[0].args[0].id in self.inline_variables
            and not node.orelse
        )
        if not exact:
            return self.generic_visit(node)
        assert isinstance(node.target, ast.Name)
        expanded: list[ast.stmt] = []
        for index in range(self.arity):
            substituter = _VarargLoopSubstituter(
                loop_variable=node.target.id,
                inline_variables=self.inline_variables,
                index=index,
            )
            for statement in node.body:
                cloned = substituter.visit(copy.deepcopy(statement))
                transformed = self.visit(cloned)
                if transformed is None:
                    continue
                if isinstance(transformed, list):
                    expanded.extend(transformed)
                else:
                    expanded.append(transformed)
        return expanded


def expand_triton_varargs(
    source_slice: TritonSourceSlice,
    *,
    arity: int,
) -> TritonSourceSlice:
    """Apply the closed ``VAR_ARGS_ARRAY`` specialization declared by Source.

    The transformation handles only exact ``range(len(array))`` loops and
    constant-arity positional/local arrays.  Any marker or array reference that
    survives expansion is rejected instead of being interpreted dynamically.
    """

    if isinstance(arity, bool) or not isinstance(arity, int) or not 1 <= arity <= 64:
        raise TritonSourceSliceError("VAR_ARGS_ARRAY arity must be in [1, 64]")
    tree = ast.parse(source_slice.code)
    target, has_marker = _selected_vararg_entry(
        tree, function_name=source_slice.function
    )
    if not has_marker:
        raise TritonSourceSliceError(
            "vararg arity was supplied but Source has no VAR_ARGS_ARRAY marker"
        )
    expander = _ClosedVarargExpander(arity)
    expander.expand_entry(target)
    if expander.marker_count == 0:
        raise TritonSourceSliceError("VAR_ARGS_ARRAY expansion made no progress")
    residual_markers = [
        node
        for node in ast.walk(target)
        if (
            isinstance(node, ast.arg)
            and _is_vararg_array_annotation(node.annotation)
        )
        or (
            isinstance(node, ast.AnnAssign)
            and _is_vararg_array_annotation(node.annotation)
        )
    ]
    residual_names = sorted(
        {
            node.id
            for node in ast.walk(target)
            if isinstance(node, ast.Name)
            and node.id in expander.inline_variables
        }
    )
    if residual_markers or residual_names:
        raise TritonSourceSliceError(
            "VAR_ARGS_ARRAY expansion left dynamic marker/reference(s): "
            + ", ".join(residual_names or ["annotation"])
        )
    ast.fix_missing_locations(tree)
    parameters = tuple(
        argument.arg
        for argument in (
            *target.args.posonlyargs,
            *target.args.args,
            *target.args.kwonlyargs,
        )
    )
    constexprs = tuple(
        argument.arg
        for argument in (
            *target.args.posonlyargs,
            *target.args.args,
            *target.args.kwonlyargs,
        )
        if _annotation_is_constexpr(argument.annotation)
    )
    return TritonSourceSlice(
        ast.unparse(tree) + "\n",
        source_slice.function,
        parameters,
        constexprs,
        source_slice.included_declarations,
        source_slice.included_functions,
        source_slice.removed_decorators,
        source_slice.inlined_imports,
        vararg_arity=arity,
    )


def requires_triton_vararg_expansion(source_slice: TritonSourceSlice) -> bool:
    """Return whether the selected entry lexical scope carries the marker."""

    tree = ast.parse(source_slice.code)
    _target, has_marker = _selected_vararg_entry(
        tree, function_name=source_slice.function
    )
    return has_marker


def _module_source_path(
    module: str,
    search_roots: Iterable[str | Path],
) -> Path | None:
    """Resolve one imported helper using the declared external layout.

    The workload manifest names the upstream logical module.  The checked-in
    external source archive is intentionally a source-only mirror and keeps a
    few helpers in their repository-oriented directories, so a literal
    ``module.replace('.', '/')`` lookup is insufficient.  Keep those layout
    aliases explicit and small: this is source provenance, not a recursive
    basename search.  A direct logical path wins; multiple aliases in one
    root are rejected so an ambiguous helper can never be silently selected.
    """

    relative = Path(*module.split(".")).with_suffix(".py")
    aliases: tuple[Path, ...] = ()
    if module.startswith("xformers.triton."):
        # The external xFormers source mirror stores the import-support
        # modules under ``support/triton`` while preserving their upstream
        # logical import names.
        tail = Path(*module.split(".")[2:]).with_suffix(".py")
        aliases = (Path("xformers") / "support" / "triton" / tail,)
    for root in search_roots:
        root_path = Path(root).expanduser().resolve()
        direct = root_path / relative
        if direct.is_file():
            return direct
        candidates = [root_path / item for item in aliases if (root_path / item).is_file()]
        if len(candidates) > 1:
            raise TritonSourceSliceError(
                f"external helper module {module!r} has ambiguous source layout: "
                + ", ".join(str(item) for item in candidates)
            )
        if candidates:
            return candidates[0]
    return None


def _is_static_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_static_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None
            and _is_static_literal(key)
            and _is_static_literal(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Invert, ast.Not)
    ):
        return _is_static_literal(node.operand)
    return False


def _plain_helper_statement_dependencies(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.FunctionDef):
        if statement.decorator_list:
            raise TritonSourceSliceError(
                f"plain imported helper {statement.name!r} must not be decorated"
            )
        forbidden = next(
            (
                node
                for node in ast.walk(statement)
                if isinstance(
                    node,
                    (
                        ast.AsyncFunctionDef,
                        ast.Await,
                        ast.Global,
                        ast.Import,
                        ast.ImportFrom,
                        ast.Nonlocal,
                        ast.Yield,
                        ast.YieldFrom,
                    ),
                )
            ),
            None,
        )
        if forbidden is not None:
            raise TritonSourceSliceError(
                f"plain imported helper {statement.name!r} contains unsupported "
                f"{type(forbidden).__name__}"
            )
        return _function_dependencies(statement)
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        if isinstance(statement, ast.ImportFrom):
            if statement.level != 0 or statement.module is None:
                raise TritonSourceSliceError(
                    "plain imported helper dependencies require absolute imports"
                )
            modules = (statement.module,)
        else:
            modules = tuple(alias.name for alias in statement.names)
        unsupported = sorted(
            {
                module
                for module in modules
                if module.split(".", 1)[0] not in sys.stdlib_module_names
                and module.split(".", 1)[0] != "triton"
            }
        )
        if unsupported:
            raise TritonSourceSliceError(
                "plain imported helper closure permits only standard-library "
                f"or Triton imports; found {unsupported}"
            )
        return set()
    value: ast.expr | None = None
    if isinstance(statement, ast.Assign):
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        value = statement.value
    if value is not None and _is_static_literal(value):
        return _statement_dependencies(statement)
    names = sorted(_bound_names(statement))
    label = names[0] if names else type(statement).__name__
    raise TritonSourceSliceError(
        f"plain imported helper dependency {label!r} is not a closed import, "
        "undecorated function, or static literal"
    )


def _build_plain_helper_closure(
    source: str,
    *,
    source_symbol: str,
) -> list[ast.stmt]:
    """Project one undecorated Python helper and its closed declarations."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise TritonSourceSliceError(
            f"plain imported helper source cannot be parsed: {exc}"
        ) from exc
    bindings: dict[str, list[int]] = {}
    for index, statement in enumerate(tree.body):
        for name in _bound_names(statement):
            bindings.setdefault(name, []).append(index)
    owners = bindings.get(source_symbol, [])
    if len(owners) != 1:
        raise TritonSourceSliceError(
            f"plain imported helper {source_symbol!r} must have one top-level "
            f"binding; found {len(owners)}"
        )
    target_index = owners[0]
    target = tree.body[target_index]
    if not isinstance(target, ast.FunctionDef) or target.decorator_list:
        raise TritonSourceSliceError(
            f"plain imported helper {source_symbol!r} must be one undecorated "
            "top-level function"
        )

    selected = {target_index}
    pending = list(_plain_helper_statement_dependencies(target))
    while pending:
        name = pending.pop()
        dependency_owners = bindings.get(name, [])
        if len(dependency_owners) != 1:
            raise TritonSourceSliceError(
                f"plain imported helper dependency {name!r} must have one "
                f"top-level binding; found {len(dependency_owners)}"
            )
        owner = dependency_owners[0]
        if owner in selected:
            continue
        selected.add(owner)
        pending.extend(
            _plain_helper_statement_dependencies(tree.body[owner])
        )
    _validate_static_dependency_surface(
        tree.body[target_index] for target_index in sorted(selected)
    )
    return [
        copy.deepcopy(statement)
        for index, statement in enumerate(tree.body)
        if index in selected
    ]


def inline_imported_triton_helpers(
    source_slice: TritonSourceSlice,
    *,
    search_roots: Iterable[str | Path],
) -> TritonSourceSlice:
    """Inline closed imported helpers without importing a package.

    Real source trees can have a Python package ``__init__`` that requires an
    optional native extension even though the selected Triton kernel only needs
    a small helper module.  This routine resolves actually referenced imported
    ``@triton.jit`` helpers or undecorated Python helpers from explicit source
    roots and forms their ordinary AST closure.  Plain-helper dependencies are
    restricted to standard-library/Triton imports, other undecorated functions,
    and static literals; ambiguous, conflicting, decorated, dynamically
    initialized, or unresolved declarations remain fail-closed.
    """

    roots = tuple(Path(item).expanduser().resolve() for item in search_roots)
    tree = ast.parse(source_slice.code)
    retained_loads: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef):
            retained_loads.update(_function_dependencies(statement))
    changed = True
    while changed:
        changed = False
        for statement in tree.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            if not (_bound_names(statement) & retained_loads):
                continue
            expanded = retained_loads | _load_names(statement)
            if expanded != retained_loads:
                retained_loads = expanded
                changed = True

    replacements: dict[int, list[ast.stmt]] = {}
    inlined_labels: list[str] = []
    for index, statement in enumerate(tree.body):
        if not isinstance(statement, ast.ImportFrom) or statement.level != 0:
            continue
        module = statement.module or ""
        if module == "triton" or module.startswith("triton."):
            continue
        requested = [
            alias
            for alias in statement.names
            if (alias.asname or alias.name) in retained_loads
        ]
        if not requested:
            continue
        module_path = _module_source_path(module, roots)
        if module_path is None:
            raise TritonSourceSliceError(
                f"cannot resolve imported Triton helper module {module!r}"
            )
        module_source = module_path.read_text(encoding="utf-8")
        replacement: list[ast.stmt] = []
        for alias in requested:
            if alias.asname is not None and alias.asname != alias.name:
                raise TritonSourceSliceError(
                    "aliased imported Triton helpers are not supported"
                )
            module_tree = ast.parse(module_source)
            definitions = [
                node
                for node in module_tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == alias.name
            ]
            if len(definitions) != 1:
                raise TritonSourceSliceError(
                    f"imported helper {module}:{alias.name} must resolve to one "
                    "top-level function"
                )
            definition = definitions[0]
            if any(_is_jit_decorator(item) for item in definition.decorator_list):
                helper = build_triton_source_slice(
                    module_source, source_symbol=alias.name
                )
                replacement.extend(copy.deepcopy(ast.parse(helper.code).body))
            else:
                replacement.extend(
                    _build_plain_helper_closure(
                        module_source, source_symbol=alias.name
                    )
                )
            inlined_labels.append(f"{module}:{alias.name}")
        replacements[index] = replacement

    if not replacements:
        return source_slice

    expanded: list[ast.stmt] = []
    for index, statement in enumerate(tree.body):
        expanded.extend(replacements.get(index, [statement]))
    deduplicated: list[ast.stmt] = []
    seen_structures: set[str] = set()
    named_declarations: dict[str, str] = {}
    for statement in expanded:
        structure = ast.dump(statement, include_attributes=False)
        for name in _bound_names(statement):
            previous = named_declarations.get(name)
            if previous is not None and previous != structure:
                raise TritonSourceSliceError(
                    f"inlined declaration {name!r} has conflicting definitions"
                )
            named_declarations[name] = structure
        if structure in seen_structures:
            continue
        seen_structures.add(structure)
        deduplicated.append(statement)
    _validate_static_dependency_surface(deduplicated)
    projected = ast.Module(body=deduplicated, type_ignores=[])
    ast.fix_missing_locations(projected)
    code = ast.unparse(projected) + "\n"
    declarations: list[str] = []
    functions: list[str] = []
    for statement in deduplicated:
        declarations.extend(sorted(_bound_names(statement)))
        if isinstance(statement, ast.FunctionDef):
            functions.append(statement.name)
    return TritonSourceSlice(
        code,
        source_slice.function,
        source_slice.parameters,
        source_slice.constexpr_parameters,
        tuple(dict.fromkeys(declarations)),
        tuple(functions),
        source_slice.removed_decorators,
        tuple(dict.fromkeys((*source_slice.inlined_imports, *inlined_labels))),
        vararg_arity=source_slice.vararg_arity,
    )


__all__ = [
    "TRITON_SOURCE_SLICE_PROTOCOL",
    "TritonSourceSlice",
    "TritonSourceSliceError",
    "build_triton_source_slice",
    "expand_triton_varargs",
    "inline_imported_triton_helpers",
    "requires_triton_vararg_expansion",
]
