"""Safe extraction of Triton replay modules from model responses.

Gate1 deliberately compiles code produced by an untrusted model.  Compiling a
``@triton.jit`` function executes the surrounding Python module and executes
parts of the function through Triton's Python frontend, so a Markdown fence is
not a security boundary.  This module therefore accepts a deliberately small
Python subset before replay code is handed to the isolated compiler process.

The accepted subset is sufficient for ordinary, self-contained Triton kernels:

* imports are limited to ``triton``, ``triton.language as tl`` and a fixed
  global pure ``libdevice`` namespace policy;
* every top-level definition is decorated only with ``@triton.jit``;
* function calls are limited to public ``tl`` functions, one reviewed pure
  top-level Triton utility, globally approved pure ``libdevice`` functions,
  ``range`` and simple scalar constructors (plus Triton's tensor ``.to``
  conversion);
* Python facilities that can perform host I/O, reflection, dynamic execution,
  non-terminating loops, or hidden imports are rejected.

This is a policy boundary, not a general-purpose Python sandbox.  Resource
limits and process isolation are still enforced by :mod:`compiler`.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import NoReturn

__all__ = [
    "CodeExtractionError",
    "ModuleValidation",
    "UnsafeReplayCodeError",
    "extract_python_code",
    "extract_python_module",
    "TRITON_REPLAY_ABI_CONTRACT",
    "TRITON_REPLAY_CONTROL_CONTRACT",
    "TRITON_REPLAY_STRUCTURE_CONTRACT",
    "TRITON_NAMESPACE_CALLS",
    "TRITON_REPLAY_API_CONTRACT",
    "validate_replay_module",
    "validate_triton_module",
]


class CodeExtractionError(ValueError):
    """The model response does not contain one unambiguous Python module."""


class UnsafeReplayCodeError(ValueError):
    """The replay module is outside Gate1's safe Triton-Python subset."""


@dataclass(frozen=True)
class ModuleValidation:
    """Result of validating one replay module."""

    function_names: tuple[str, ...]
    tree: ast.Module


_FENCE_RE = re.compile(
    r"```[ \t]*(?P<label>[A-Za-z0-9_+.-]*)[ \t]*\r?\n"
    r"(?P<body>.*?)```",
    re.DOTALL,
)
_PYTHON_FENCE_LABELS = {"", "py", "python", "python3"}

# These are the only non-Triton call targets admitted in a kernel body.  They
# are constructors/iteration helpers with no I/O or reflection capability.
_SAFE_BUILTIN_CALLS = {"bool", "float", "int", "range"}

# A Triton module is already required for the ``@triton.jit`` decorator, but
# its top-level namespace contains runtime/compiler machinery that must not be
# reachable from a model-authored kernel body.  Keep this list exact and
# source-independent.  ``tl.cdiv`` remains the in-kernel spelling for ceiling
# division; this set is intentionally not a general ``triton.*`` allowlist.
TRITON_NAMESPACE_CALLS = frozenset({"next_power_of_2"})
TRITON_REPLAY_API_CONTRACT = (
    "For a Triton Replay module, the kernel body may call only the reviewed "
    "pure top-level utility `triton.next_power_of_2(value)` with one positional "
    "argument and no keywords. Use `tl.cdiv` for ceiling division inside the "
    "kernel. No other `triton.*` call or namespace attribute is admitted by "
    "the Replay ingress policy; `triton.jit` remains the required decorator. "
    "When its result supplies `tl.arange` or another constexpr-only argument, "
    "call it directly in that constexpr context or bind it as `name: tl.constexpr`; "
    "an unannotated local assignment is not a constexpr carrier in Triton.\n"
)
TRITON_REPLAY_ABI_CONTRACT = (
    "For Triton Replay ABI use, a J.signature scalar whose `logical_ref` is absent "
    "is an ABI-only address or representation carrier: keep it in the exact "
    "signature and use it only in a target pointer/address expression that reaches "
    "a load, store, atomic transfer, pointer base/stride, or pointer-advance "
    "operation. When `frontend.json` contains `representation_contract`, its "
    "`address_bindings` are the only ownership authority for such carriers. Each "
    "row binds one anonymous `carrier` to one J `access_ref`, its anonymous tensor, "
    "and one tensor `axis` with `role=\"stride\"`; use that carrier for the "
    "corresponding access address, including every access row when one carrier is "
    "shared. Do not infer ownership from parameter position or contiguous-layout "
    "convention. Every listed carrier must reach the address slice, and every "
    "null-logical-ref scalar not listed there must remain unread. A carrier-derived "
    "local must remain address-only; do not reuse it as a semantic value or control "
    "input. Never use an unowned stride or address scalar as a J shape, extent, grid "
    "cardinality, logical coordinate, predicate, fill/value operand, reducer input, "
    "loop bound, or schedule value. The ingress checker rejects every carrier use "
    "outside that narrow address slice. A J shape handle without a scalar logical "
    "binding is not an implicit scalar argument; realize it from the direct "
    "launch/program-id mapping when the target requires it. When the contract also "
    "contains `program_bindings`, each row is the closed mapping from one anonymous "
    "J domain to its physical launch axis; use exactly one `tl.program_id(axis)` for "
    "each distinct listed axis, do not assign an independent axis to an unlisted "
    "reduction/lane domain, and never use `tl.num_programs` to walk that domain. A "
    "J scalar may enter semantic expressions only through its declared `logical_ref`, "
    "and an unbound J constexpr carrier remains unread in the body.\n"
)
TRITON_REPLAY_CONTROL_CONTRACT = (
    "For Triton Replay control flow, the ingress policy admits only `for` loops "
    "over `range`, `tl.range`, or `tl.static_range`; it rejects Python `while` "
    "loops. Never emit `while`, use `tl.num_programs` to serially walk a logical "
    "J domain, or use a loop to compensate for a missing launch mapping. You may "
    "use `tl.num_programs` only as launch geometry metadata to decode a direct "
    "`tl.program_id` mapping; it is not a semantic extent and must not replace a "
    "J-authored scalar. Realize ordinary finite J domains with `tl.program_id` "
    "plus decoded lane/tile coordinates and their authored predicates. Use a "
    "target loop only for a J-authored loop or a closed C `reduction_tile_extent` "
    "contraction/reduction exception, and keep that loop as an admitted `for` "
    "form.\n"
)
TRITON_REPLAY_STRUCTURE_CONTRACT = (
    "For Triton Replay structural fidelity, preserve every J-authored boundary "
    "access and effect as a distinct occurrence, including repeated accesses to "
    "the same tensor and index; do not hoist or reuse an earlier load across an "
    "intervening authored effect, and do not replace several authored transfers "
    "with one physical transfer. If an access has no J predicate, emit no target "
    "mask or other arguments for that access; a domain bound or a physical tile does "
    "not create a transfer predicate or fill. Preserve each J reduction's own "
    "reshape/reduce boundary, axis, combiner, and downstream dataflow. A domain "
    "used only as a reduction/contraction axis is not a launch axis and must not "
    "become an independent program_id; realize it with the permitted lane or "
    "target-loop structure instead. A tensor-shaped J map is a vector operation: "
    "keep its logical lane dimension as a target tensor and do not scalarize it "
    "into a Python/Triton loop or one scalar transfer per lane. For every J "
    "reduce, emit one target local reduction with the authored combiner and "
    "reduction axis (for example, one `tl.sum` for a sum reduce), preserving the "
    "collapsed-versus-kept result shape and the reduction's downstream use. Do "
    "not replace a J reduce with a serial accumulator, a loop over its dimension, "
    "or a duplicated sequence of scalar loads/stores. A `for range(...)` is "
    "allowed only for a J-authored loop or the explicitly closed C-tiled "
    "contraction exception; a J shape/extent or reduction dimension alone never "
    "authorizes such a loop. In the common shape `[d0,...,dk] -> "
    "[d0,...,d(k-1)]` case, form the `k` lane values with a target vector (such "
    "as `tl.arange`), keep all maps and transfers over that vector, and perform "
    "one local reduction over that lane for each distinct J reduce occurrence.\n"
    "Do not fold or delete a nonzero J constant merely because it is an "
    "algebraic identity for one input (for example, a `-inf` seed feeding a "
    "J-authored max chain); the only identity fold permitted by this contract "
    "is the exact all-zero add case stated elsewhere. Every boundary load is "
    "an authored occurrence, not a cacheable tensor value: if J has repeated "
    "loads of the same tensor or address, emit a separate target `tl.load` for "
    "each occurrence and place it in the declared effect order, including after "
    "intervening stores. Never common-subexpression-eliminate, hoist, or reuse a "
    "prior load across another authored boundary effect.\n"
    "A repeated tensor name does not identify one boundary value: use the "
    "distinct J `access_ref` occurrences in `boundary.accesses` as the transfer "
    "identity. If two access rows read the same tensor, create and use two "
    "separate loads at their respective rows; a local variable may not stand in "
    "for both access refs, even when their addresses are equal.\n"
    "When one J reduction axis is represented by one-dimensional target lanes "
    "and the J reduction has no separate target-axis requirement, use the public "
    "default reduction form (for example `tl.sum(value)` without an explicit "
    "`axis=`) so the target retains its flatten/reshape-to-local-reduce boundary. "
    "Do not replace that authored default reduction with `axis=0` when doing so "
    "removes the target's required reshape boundary.\n"
    "For a vector transfer whose address has a scalar base offset plus a lane "
    "vector, form the complete vector index first (for example, "
    "`lane_offset = scalar_offset + lane`) and then perform one pointer-plus-"
    "vector address operation in `tl.load` or `tl.store`. Do not spell the "
    "transfer as `pointer + scalar_offset + lane`: Triton lowers that chained "
    "form to two pointer advances and it breaks the one-authored-transfer "
    "boundary.\n"
)

# Public ``tl`` calls are generally dataflow-building operations.  The handful
# below are intentionally excluded because they accept executable/opaque code
# or dynamically named symbols and would defeat structural review.
_FORBIDDEN_TL_CALLS = {
    "inline_asm_elementwise",
    "extern_elementwise",
}

# Security policy is global and kernel-independent.  It is deliberately not
# selected from J, pi, a dataset identity, or a per-operation catalog.
GLOBAL_LIBDEVICE_CALLS = frozenset(
    {
        "atan", "atan2", "cos", "erf", "exp", "finitef", "isinf", "isnan", "log",
        "pow", "rsqrt", "sin", "sqrt", "tan", "tanh",
    }
)
# Triton 3.6 exposes some reviewed math only through the public libdevice
# namespace.  Accepting these spellings under ``tl`` merely defers a
# deterministic API error to compilation and wastes a formal Gate1 round.
TRITON_UNAVAILABLE_CALL_GUIDANCE = {
    "atan": "import libdevice and call `libdevice.atan`",
    "atan2": "import libdevice and call `libdevice.atan2`",
    "isnan": "import libdevice and call `libdevice.isnan`",
    "pow": "use the public `x ** y` operator",
    "tan": "import libdevice and call `libdevice.tan`",
    "tanh": "import libdevice and call `libdevice.tanh`",
}

# A conservative statement/expression allowlist.  Context and operator marker
# nodes are included because ``ast.walk`` yields them independently.
_ALLOWED_BODY_NODES = (
    ast.Expr,
    ast.Constant,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.For,
    ast.If,
    ast.Return,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Starred,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.MatMult,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)


def _fail(node: ast.AST | None, message: str) -> NoReturn:
    if node is not None and getattr(node, "lineno", None) is not None:
        message = f"line {node.lineno}: {message}"
    raise UnsafeReplayCodeError(message)


def extract_python_code(response: str) -> str:
    """Extract one Python module from a raw or Markdown-fenced model response.

    A response with code fences must contain exactly one Python (or unlabeled)
    fence.  Refusing ambiguous multi-block answers prevents a caller from
    accidentally compiling a different block than the one it audited.
    """

    if not isinstance(response, str) or not response.strip():
        raise CodeExtractionError("model response is empty")

    matches = list(_FENCE_RE.finditer(response))
    if matches:
        candidates = [
            match.group("body")
            for match in matches
            if match.group("label").lower() in _PYTHON_FENCE_LABELS
        ]
        if not candidates:
            raise CodeExtractionError("model response has no Python code fence")
        if len(candidates) != 1:
            raise CodeExtractionError(
                f"model response has {len(candidates)} Python code fences; expected one"
            )
        code = candidates[0]
    else:
        # Raw mode is useful for providers that return the code payload without
        # presentation Markdown.  Any prose will naturally fail Python parsing.
        code = response

    code = code.lstrip("\ufeff").strip()
    if not code:
        raise CodeExtractionError("extracted Python module is empty")
    return code + "\n"


def extract_python_module(
    response: str,
    *,
    required_function: str | None = None,
) -> str:
    """Extract and validate a model-generated Triton module."""

    code = extract_python_code(response)
    validate_triton_module(
        code,
        required_function=required_function,
    )
    return code


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_attr(node: ast.AST, root: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and _is_name(node.value, root)
    )


def _check_identifier(identifier: str, node: ast.AST, what: str) -> None:
    if "__" in identifier:
        _fail(node, f"dunder {what} {identifier!r} is forbidden")


def _validate_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "triton" and alias.asname is None:
                continue
            if alias.name == "triton.language" and alias.asname == "tl":
                continue
            _fail(
                node,
                "imports are limited to `import triton` and "
                "`import triton.language as tl`",
            )
        return

    if (
        node.module == "triton"
        and node.level == 0
        and len(node.names) == 1
        and node.names[0].name == "language"
        and node.names[0].asname == "tl"
    ):
        return
    if (
        node.module == "triton.language.extra.cuda"
        and node.level == 0
        and len(node.names) == 1
        and node.names[0].name == "libdevice"
        and node.names[0].asname is None
    ):
        return
    _fail(
        node,
        "from-imports are limited to `from triton import language as tl` "
        "and `from triton.language.extra.cuda import libdevice`",
    )


def _validate_annotation(node: ast.AST | None) -> None:
    if node is None:
        return
    if not _is_attr(node, "tl", "constexpr"):
        _fail(node, "only `tl.constexpr` parameter annotations are allowed")


def _validate_jit_decorator(node: ast.AST, arg_names: set[str]) -> None:
    if _is_attr(node, "triton", "jit"):
        return
    if not (
        isinstance(node, ast.Call) and _is_attr(node.func, "triton", "jit")
    ):
        _fail(node, "every function must be decorated only with `@triton.jit`")
    if node.args:
        _fail(node, "`@triton.jit` positional arguments are forbidden")
    seen: set[str] = set()
    for keyword in node.keywords:
        if keyword.arg != "do_not_specialize":
            _fail(keyword, "only `do_not_specialize` is allowed in `@triton.jit`")
        if keyword.arg in seen:
            _fail(keyword, "duplicate `do_not_specialize` argument")
        seen.add(keyword.arg)
        try:
            value = ast.literal_eval(keyword.value)
        except (TypeError, ValueError, SyntaxError):
            _fail(keyword.value, "`do_not_specialize` must be a literal list/tuple")
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, (str, int)) for item in value
        ):
            _fail(keyword.value, "invalid `do_not_specialize` value")
        for item in value:
            if isinstance(item, str) and item not in arg_names:
                _fail(keyword.value, f"unknown `do_not_specialize` argument {item!r}")
            if isinstance(item, int) and not 0 <= item < len(arg_names):
                _fail(keyword.value, f"invalid `do_not_specialize` index {item}")


def _stored_names(nodes: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for statement in nodes:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
    return names


def _validate_call(
    node: ast.Call,
    jit_functions: set[str],
    imported_namespaces: set[str],
    allowed_libdevice_calls: frozenset[str],
) -> None:
    target = node.func
    if isinstance(target, ast.Name):
        if target.id not in _SAFE_BUILTIN_CALLS | jit_functions:
            _fail(node, f"call to Python name {target.id!r} is forbidden")
        return

    if isinstance(target, ast.Attribute):
        _check_identifier(target.attr, target, "attribute")
        if _is_name(target.value, "tl"):
            if target.attr in TRITON_UNAVAILABLE_CALL_GUIDANCE:
                _fail(
                    node,
                    f"`tl.{target.attr}` is unavailable in the public frontend; "
                    f"{TRITON_UNAVAILABLE_CALL_GUIDANCE[target.attr]}",
                )
            if target.attr.startswith("_") or target.attr in _FORBIDDEN_TL_CALLS:
                _fail(node, f"call to `tl.{target.attr}` is forbidden")
            return
        # ``tl.math`` is a public Triton namespace.  It is semantically no
        # different from other public ``tl`` operators and must remain
        # available to a source-blind replay when the captured source uses it
        # (for example ``tl.math.erf``).  This extends only the target API
        # surface; it supplies no Source or pi information to Replay.
        if (
            isinstance(target.value, ast.Attribute)
            and _is_name(target.value.value, "tl")
            and target.value.attr == "math"
        ):
            if target.attr.startswith("_"):
                _fail(node, f"call to `tl.math.{target.attr}` is forbidden")
            return
        if _is_name(target.value, "libdevice"):
            if (
                "libdevice" not in imported_namespaces
                or target.attr not in allowed_libdevice_calls
            ):
                _fail(node, f"call to `libdevice.{target.attr}` is forbidden")
            return
        if _is_name(target.value, "triton"):
            if target.attr not in TRITON_NAMESPACE_CALLS:
                _fail(node, f"call to `triton.{target.attr}` is forbidden")
            if len(node.args) != 1 or node.keywords:
                _fail(
                    node,
                    "`triton.next_power_of_2` requires one positional argument "
                    "and no keywords",
                )
            return
        # Triton tensors expose ``.to(dtype)`` as a frontend conversion.  No
        # other arbitrary Python method dispatch is permitted.
        if target.attr == "to" and len(node.args) <= 1 and not node.keywords:
            return
    _fail(
        node,
        "calls must target a public `tl` function, allowlisted Triton "
        "namespace function, range/scalar constructor, or `.to`",
    )


def _validate_loop(node: ast.For) -> None:
    iterator = node.iter
    if not isinstance(iterator, ast.Call):
        _fail(iterator, "for-loops must iterate over `range`, `tl.range`, or `tl.static_range`")
    target = iterator.func
    valid = _is_name(target, "range") or (
        isinstance(target, ast.Attribute)
        and _is_name(target.value, "tl")
        and target.attr in {"range", "static_range"}
    )
    if not valid:
        _fail(iterator, "for-loops must iterate over `range`, `tl.range`, or `tl.static_range`")
    if node.orelse:
        _fail(node, "for-else is not allowed in replay kernels")


def _validate_function(
    node: ast.FunctionDef,
    *,
    jit_functions: set[str],
    imported_namespaces: set[str],
    allowed_libdevice_calls: frozenset[str],
) -> None:
    _check_identifier(node.name, node, "function name")
    args = node.args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg:
        _fail(node, "positional-only, variadic, and keyword-only parameters are forbidden")
    if args.defaults or args.kw_defaults:
        _fail(node, "function parameter defaults are forbidden")
    if node.returns is not None or node.type_comment is not None:
        _fail(node, "return and type-comment annotations are forbidden")

    arg_names = {arg.arg for arg in args.args}
    for arg in args.args:
        _check_identifier(arg.arg, arg, "parameter")
        _validate_annotation(arg.annotation)
    if len(node.decorator_list) != 1:
        _fail(node, "every function must have exactly one `@triton.jit` decorator")
    _validate_jit_decorator(node.decorator_list[0], arg_names)

    local_names = arg_names | _stored_names(node.body)
    allowed_load_names = (
        local_names
        | _SAFE_BUILTIN_CALLS
        | jit_functions
        | {"tl"}
        | imported_namespaces
    )

    for statement in node.body:
        for child in ast.walk(statement):
            if isinstance(child, ast.FunctionDef):
                _fail(child, "nested functions are forbidden")
            if not isinstance(child, _ALLOWED_BODY_NODES):
                _fail(child, f"Python construct {type(child).__name__} is forbidden")
            if isinstance(child, ast.Name):
                _check_identifier(child.id, child, "name")
                if isinstance(child.ctx, ast.Load) and child.id not in allowed_load_names:
                    _fail(child, f"global/free name {child.id!r} is forbidden")
            elif isinstance(child, ast.Attribute):
                _check_identifier(child.attr, child, "attribute")
                if _is_name(child.value, "tl"):
                    if child.attr.startswith("_"):
                        _fail(child, f"private Triton attribute {child.attr!r} is forbidden")
                elif (
                    isinstance(child.value, ast.Attribute)
                    and _is_name(child.value.value, "tl")
                    and child.value.attr == "math"
                ):
                    if child.attr.startswith("_"):
                        _fail(child, f"private Triton math attribute {child.attr!r} is forbidden")
                elif _is_name(child.value, "libdevice"):
                    if (
                        "libdevice" not in imported_namespaces
                        or child.attr not in allowed_libdevice_calls
                    ):
                        _fail(
                            child,
                            f"attribute access `libdevice.{child.attr}` is forbidden",
                        )
                elif _is_name(child.value, "triton"):
                    if child.attr not in TRITON_NAMESPACE_CALLS:
                        _fail(
                            child,
                            f"attribute access `triton.{child.attr}` is forbidden",
                        )
                elif child.attr != "to":
                    _fail(child, f"attribute access {child.attr!r} is forbidden")
            elif isinstance(child, ast.Call):
                _validate_call(
                    child,
                    jit_functions,
                    imported_namespaces,
                    allowed_libdevice_calls,
                )
                if any(keyword.arg is None for keyword in child.keywords):
                    _fail(child, "dynamic **keyword expansion is forbidden")
            elif isinstance(child, ast.For):
                _validate_loop(child)
            elif isinstance(child, ast.Constant):
                if child.value is not None and not isinstance(
                    child.value, (bool, int, float, str)
                ):
                    _fail(child, f"literal type {type(child.value).__name__} is forbidden")
                if isinstance(child.value, str) and "__" in child.value:
                    _fail(child, "dunder strings are forbidden")


def _validate_jit_call_graph(
    functions: list[ast.FunctionDef], names: set[str]
) -> None:
    """Reject recursive sibling-JIT calls while allowing pure helper DAGs."""

    graph: dict[str, set[str]] = {}
    nodes = {node.name: node for node in functions}
    for function in functions:
        graph[function.name] = {
            child.func.id
            for child in ast.walk(function)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in names
        }

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            _fail(nodes[name], "recursive Triton helper calls are forbidden")
        visiting.add(name)
        for target in graph[name]:
            visit(target)
        visiting.remove(name)
        visited.add(name)

    for name in graph:
        visit(name)


def validate_triton_module(
    code: str,
    *,
    required_function: str | None = None,
) -> ModuleValidation:
    """Validate ``code`` against Gate1's strict Triton replay policy.

    The function raises :class:`UnsafeReplayCodeError` for both syntax errors
    and policy violations.  Returning the parsed tree lets callers avoid a
    second parse when they need function metadata.
    """

    reviewed_libdevice_calls = GLOBAL_LIBDEVICE_CALLS
    if not isinstance(code, str) or not code.strip():
        raise UnsafeReplayCodeError("replay module is empty")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        where = f"line {exc.lineno}: " if exc.lineno is not None else ""
        raise UnsafeReplayCodeError(f"{where}invalid Python syntax: {exc.msg}") from exc

    function_nodes: list[ast.FunctionDef] = []
    saw_triton = False
    saw_tl = False
    imported_namespaces: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if "__" in node.value.value:
                _fail(node, "dunder strings are forbidden")
            continue  # module docstring
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _validate_import(node)
            if isinstance(node, ast.Import):
                saw_triton |= any(alias.name == "triton" for alias in node.names)
                if any(alias.name == "triton" for alias in node.names):
                    imported_namespaces.add("triton")
                saw_tl |= any(alias.name == "triton.language" for alias in node.names)
            else:
                if node.module == "triton":
                    saw_tl = True
                elif node.module == "triton.language.extra.cuda":
                    imported_namespaces.add("libdevice")
            continue
        if isinstance(node, ast.FunctionDef):
            if any(existing.name == node.name for existing in function_nodes):
                _fail(node, f"duplicate function definition {node.name!r}")
            function_nodes.append(node)
            continue
        _fail(node, "top level may contain only Triton imports and `@triton.jit` functions")

    if not saw_triton:
        raise UnsafeReplayCodeError("module must contain `import triton`")
    if not saw_tl:
        raise UnsafeReplayCodeError("module must import `triton.language as tl`")
    if not function_nodes:
        raise UnsafeReplayCodeError("module contains no `@triton.jit` function")
    function_names = {node.name for node in function_nodes}
    for node in function_nodes:
        _validate_function(
            node,
            jit_functions=function_names,
            imported_namespaces=imported_namespaces,
            allowed_libdevice_calls=reviewed_libdevice_calls,
        )
    _validate_jit_call_graph(function_nodes, function_names)
    functions = tuple(node.name for node in function_nodes)
    if required_function is not None and required_function not in function_names:
        raise UnsafeReplayCodeError(
            f"required Triton function {required_function!r} is missing"
        )
    return ModuleValidation(functions, tree)


# Naming used by the Gate1 design document and early callers.
validate_replay_module = validate_triton_module
