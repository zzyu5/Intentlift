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


_CODE_EXTRACTION_RULE_CODES = frozenset(
    {
        "replay_response_empty",
        "replay_python_fence_missing",
        "replay_python_fence_ambiguous",
        "replay_extracted_module_empty",
    }
)

_UNSAFE_REPLAY_RULE_CODES = frozenset(
    {
        "replay_module_empty",
        "replay_python_syntax_invalid",
        "replay_dunder_identifier_forbidden",
        "replay_import_forbidden",
        "replay_annotation_forbidden",
        "replay_jit_decorator_invalid",
        "replay_jit_positional_arguments_forbidden",
        "replay_jit_keyword_forbidden",
        "replay_jit_keyword_duplicate",
        "replay_jit_specialization_literal_required",
        "replay_jit_specialization_value_invalid",
        "replay_jit_specialization_argument_unknown",
        "replay_jit_specialization_index_invalid",
        "replay_python_call_forbidden",
        "replay_triton_api_unavailable",
        "replay_cat_arity_invalid",
        "replay_cat_keyword_invalid",
        "replay_cat_keyword_duplicate",
        "replay_cat_reorder_required",
        "replay_tl_call_forbidden",
        "replay_tl_math_call_forbidden",
        "replay_libdevice_call_forbidden",
        "replay_triton_namespace_call_forbidden",
        "replay_triton_utility_signature_invalid",
        "replay_call_target_forbidden",
        "replay_loop_iterator_invalid",
        "replay_for_else_forbidden",
        "replay_function_parameters_invalid",
        "replay_function_defaults_forbidden",
        "replay_function_annotations_forbidden",
        "replay_function_decorator_count_invalid",
        "replay_nested_function_forbidden",
        "replay_python_construct_forbidden",
        "replay_global_name_forbidden",
        "replay_private_tl_attribute_forbidden",
        "replay_private_tl_math_attribute_forbidden",
        "replay_libdevice_attribute_forbidden",
        "replay_triton_attribute_forbidden",
        "replay_attribute_forbidden",
        "replay_dynamic_keyword_forbidden",
        "replay_literal_type_forbidden",
        "replay_dunder_string_forbidden",
        "replay_recursive_helper_forbidden",
        "replay_duplicate_function_forbidden",
        "replay_top_level_statement_forbidden",
        "replay_triton_import_missing",
        "replay_tl_import_missing",
        "replay_jit_function_missing",
        "replay_required_function_missing",
    }
)


class _StructuredReplayPolicyError(ValueError):
    """One closed, source-free Replay ingress classification."""

    _allowed_rule_codes: frozenset[str] = frozenset()

    def __init__(
        self,
        message: str,
        *,
        rule_code: str,
        path: str,
        line: int | None = None,
    ) -> None:
        if rule_code not in self._allowed_rule_codes:
            raise ValueError("Replay policy exception has an unknown rule code")
        if not isinstance(path, str) or not path.startswith("<") or not path.endswith(">"):
            raise ValueError("Replay policy exception path must be a closed marker")
        if line is not None and (
            not isinstance(line, int) or isinstance(line, bool) or line < 1
        ):
            raise ValueError("Replay policy exception line must be a positive integer")
        super().__init__(message)
        self.rule_code = rule_code
        self.path = path
        self.line = line
        self.exception_type = type(self).__name__


class CodeExtractionError(_StructuredReplayPolicyError):
    """The model response does not contain one unambiguous Python module."""

    _allowed_rule_codes = _CODE_EXTRACTION_RULE_CODES


class UnsafeReplayCodeError(_StructuredReplayPolicyError):
    """The replay module is outside Gate1's safe Triton-Python subset."""

    _allowed_rule_codes = _UNSAFE_REPLAY_RULE_CODES


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
    "an unannotated local assignment is not a constexpr carrier in Triton. "
    "On the active public Triton frontend, `tl.tanh` is not admitted: for "
    "J `map_unary(fn=tanh)`, import `libdevice` from "
    "`triton.language.extra.cuda` and call `libdevice.tanh(value.to(tl.float32))`.\n"
    "J `map_unary(fn=rsqrt_ftz)` requires an f32 `libdevice.rsqrt(value)` "
    "with a fresh compiled FTZ receipt; ordinary rsqrt is not equivalent.\n"
    "The installed public Triton cat signature is tl.cat(input, other, "
    "can_reorder=False): it concatenates exactly two tensors and has no dim "
    "or axis keyword. The current implementation also requires "
    "can_reorder=True, which permits element reordering. Use cat only when "
    "the J dataflow is order-insensitive; for ordered components such as "
    "weighted columns, compute each component directly instead of packing "
    "them with cat. For three or more order-insensitive tensors, nest "
    "two-input calls; do not use the PyTorch-style cat(inputs..., dim=...) "
    "spelling.\n"
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
    "shared. The `axis` is the logical tensor axis whose coordinate is multiplied "
    "by the carrier; it is not the vector lane axis merely because the access is "
    "a vector transfer. For a rank-2 row access with logical index `[row, lane]` "
    "and a row-stride carrier bound to axis 0, use the canonical two-stage "
    "address realization: first materialize `row_ptr = base + row * carrier`, "
    "then form the vector transfer from `row_ptr + lane`. Keep that row pointer "
    "as an explicit local; do not collapse it into `base + row * carrier + lane`, "
    "and never use `lane * carrier` or `extent * carrier`. A contiguous lane "
    "contributes its unit index directly unless another explicit J/representation "
    "binding says otherwise. Do not infer ownership from parameter position or "
    "contiguous-layout convention. For every bound address axis, form the logical "
    "coordinate in the exact integer dtype of that axis's bound carrier before "
    "multiplication or pointer addition. In particular, when a program-id row "
    "coordinate is multiplied by an i64 row-stride carrier, explicitly widen the "
    "program id first (for example `tl.program_id(0).to(tl.int64)`); do not let "
    "an i32 program id silently choose the pointer-add index type. Reuse that "
    "typed row coordinate for every access bound to the same axis. Do not propagate "
    "that row-carrier dtype to an unbound contiguous lane: a J/iota `tl.arange` "
    "lane remains in its authored integer dtype (for example i32), and the final "
    "row-pointer-plus-lane address uses that lane directly. Every listed carrier must reach the address slice, and every "
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
    "logical collapsed-versus-kept relation and the reduction's downstream use. "
    "When the retained dimensions are scalar program coordinates, a logical "
    "kept singleton is one scalar target carrier: do not reshape that scalar to "
    "`(1,)`; retain it through scalar consumers and let a later vector consumer "
    "broadcast it. Do "
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
    "When an output mapping's producer is already available at its declared "
    "effect position, emit that store at the earliest legal point in the J "
    "effect order before constructing later independent consumers; do not defer "
    "an earlier scalar/statistics store to a final epilogue merely to group "
    "stores.\n"
    "When one J reduction axis is represented by one-dimensional target lanes "
    "and the J reduction has no separate target-axis requirement, use the public "
    "default reduction form (for example `tl.sum(value)` without an explicit "
    "`axis=`) so the target retains its flatten/reshape-to-local-reduce boundary. "
    "Do not replace that authored default reduction with `axis=0` when doing so "
    "removes the target's required reshape boundary.\n"
    "A `consumer_extension` is a separate J-owned consumer edge, not an alias "
    "for a load's `fill_value`. For every declared extension, preserve the "
    "declared access predicate and make the extension's typed outside identity "
    "reach that exact consumer operand along the closed provenance path. Apply "
    "a predicate/select only when the closed extension ledger marks "
    "`requires_post_consumer_identity=true` (for example, after a subtraction "
    "that changes the fill). A closed producer mapping may discharge the "
    "extension without a select: in particular, `exp(-inf)` is the typed zero "
    "outside identity, so a masked `other=-inf` followed by `exp` must feed the "
    "reduction directly. Do not add a redundant `where/select` merely because "
    "the load fill literal differs from the consumer identity. The load's "
    "`other=` handles the access-local fill, while the extension records the "
    "proven consumer mapping. When one access has multiple "
    "consumer_extensions, retain one independently observable consumer edge per "
    "row; do not merge the consumers merely because they share one physical "
    "load.\n"
    "For a vector transfer whose address has a scalar base offset plus a lane "
    "vector, form the complete vector index first (for example, "
    "`lane_offset = scalar_offset + lane`) and then perform one pointer-plus-"
    "vector address operation in `tl.load` or `tl.store`. Do not spell the "
    "transfer as `pointer + scalar_offset + lane`: Triton lowers that chained "
    "form to two pointer advances and it breaks the one-authored-transfer "
    "boundary.\n"
    "Before returning a Triton module, perform a shape audit for every "
    "intermediate: preserve the exact rank and extents of each typed J value "
    "after its physical tile projection, introduce singleton dimensions only "
    "when the consuming J broadcast/consumer requires them, and never create a "
    "larger Cartesian product merely to make an axis number fit. A `tl.reshape` "
    "is legal only when its input and output element counts are exactly equal. "
    "A reduction axis must already exist on the value being reduced; do not "
    "prepend artificial dimensions before `tl.max`/`tl.sum` or reduce a newly "
    "invented axis. For a tiled contraction, keep the two operand tiles in their "
    "declared contracted layout, form the product over the two visible output "
    "axes and one contracted lane axis, and reduce only that contracted lane; "
    "for a score tile, reduce the actual row/column tile axis and restore a "
    "singleton only at the declared kept-dimension consumer. Tie each physical "
    "tile range to the C slot attached to its own J domain (`tile_extent` versus "
    "`reduction_tile_extent`), use every attached slot in that role, and never "
    "substitute one domain's tile for another domain's extent. For a J "
    "contraction with two or more visible tile axes, preserve the operand tile "
    "layout and contraction-axis orientation instead of scalarizing it into "
    "elementwise multiply plus a reduction. Use the public target matrix/tensor "
    "contraction form (for Triton, `tl.dot`) when the operand tiles are "
    "compatible, and use `tl.trans` only when the declared J subscripts require "
    "the operand axis reversal. Keep all non-contracted output axes as tensor "
    "dimensions; do not collapse a rank-4/rank-3 J value to a scalar or one-"
    "dimensional lane merely because one program coordinate is scalar. The "
    "reduction belongs only to the authored contraction/reduce axis, not to an "
    "invented feature or query axis.\n"
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
    "isinf": "import libdevice and call `libdevice.isinf`",
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


def _fail(
    node: ast.AST | None,
    message: str,
    *,
    rule_code: str,
    path: str,
) -> NoReturn:
    line = getattr(node, "lineno", None) if node is not None else None
    if isinstance(line, int):
        message = f"line {line}: {message}"
    else:
        line = None
    raise UnsafeReplayCodeError(
        message,
        rule_code=rule_code,
        path=path,
        line=line,
    )


def extract_python_code(response: str) -> str:
    """Extract one Python module from a raw or Markdown-fenced model response.

    A response with code fences must contain exactly one Python (or unlabeled)
    fence.  Refusing ambiguous multi-block answers prevents a caller from
    accidentally compiling a different block than the one it audited.
    """

    if not isinstance(response, str) or not response.strip():
        raise CodeExtractionError(
            "model response is empty",
            rule_code="replay_response_empty",
            path="<response>",
        )

    matches = list(_FENCE_RE.finditer(response))
    if matches:
        candidates = [
            match.group("body")
            for match in matches
            if match.group("label").lower() in _PYTHON_FENCE_LABELS
        ]
        if not candidates:
            raise CodeExtractionError(
                "model response has no Python code fence",
                rule_code="replay_python_fence_missing",
                path="<response-fence>",
            )
        if len(candidates) != 1:
            raise CodeExtractionError(
                f"model response has {len(candidates)} Python code fences; expected one",
                rule_code="replay_python_fence_ambiguous",
                path="<response-fence>",
            )
        code = candidates[0]
    else:
        # Raw mode is useful for providers that return the code payload without
        # presentation Markdown.  Any prose will naturally fail Python parsing.
        code = response

    code = code.lstrip("\ufeff").strip()
    if not code:
        raise CodeExtractionError(
            "extracted Python module is empty",
            rule_code="replay_extracted_module_empty",
            path="<module>",
        )
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
        _fail(
            node,
            f"dunder {what} {identifier!r} is forbidden",
            rule_code="replay_dunder_identifier_forbidden",
            path="<identifier>",
        )


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
                rule_code="replay_import_forbidden",
                path="<import>",
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
        rule_code="replay_import_forbidden",
        path="<import>",
    )


def _validate_annotation(node: ast.AST | None) -> None:
    if node is None:
        return
    if not _is_attr(node, "tl", "constexpr"):
        _fail(
            node,
            "only `tl.constexpr` parameter annotations are allowed",
            rule_code="replay_annotation_forbidden",
            path="<parameter-annotation>",
        )


def _validate_jit_decorator(node: ast.AST, arg_names: set[str]) -> None:
    if _is_attr(node, "triton", "jit"):
        return
    if not (
        isinstance(node, ast.Call) and _is_attr(node.func, "triton", "jit")
    ):
        _fail(
            node,
            "every function must be decorated only with `@triton.jit`",
            rule_code="replay_jit_decorator_invalid",
            path="<decorator>",
        )
    if node.args:
        _fail(
            node,
            "`@triton.jit` positional arguments are forbidden",
            rule_code="replay_jit_positional_arguments_forbidden",
            path="<decorator>",
        )
    seen: set[str] = set()
    for keyword in node.keywords:
        if keyword.arg != "do_not_specialize":
            _fail(
                keyword,
                "only `do_not_specialize` is allowed in `@triton.jit`",
                rule_code="replay_jit_keyword_forbidden",
                path="<decorator-keyword>",
            )
        if keyword.arg in seen:
            _fail(
                keyword,
                "duplicate `do_not_specialize` argument",
                rule_code="replay_jit_keyword_duplicate",
                path="<decorator-keyword>",
            )
        seen.add(keyword.arg)
        try:
            value = ast.literal_eval(keyword.value)
        except (TypeError, ValueError, SyntaxError):
            _fail(
                keyword.value,
                "`do_not_specialize` must be a literal list/tuple",
                rule_code="replay_jit_specialization_literal_required",
                path="<decorator-keyword>",
            )
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, (str, int)) for item in value
        ):
            _fail(
                keyword.value,
                "invalid `do_not_specialize` value",
                rule_code="replay_jit_specialization_value_invalid",
                path="<decorator-keyword>",
            )
        for item in value:
            if isinstance(item, str) and item not in arg_names:
                _fail(
                    keyword.value,
                    f"unknown `do_not_specialize` argument {item!r}",
                    rule_code="replay_jit_specialization_argument_unknown",
                    path="<decorator-keyword>",
                )
            if isinstance(item, int) and not 0 <= item < len(arg_names):
                _fail(
                    keyword.value,
                    f"invalid `do_not_specialize` index {item}",
                    rule_code="replay_jit_specialization_index_invalid",
                    path="<decorator-keyword>",
                )


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
            _fail(
                node,
                f"call to Python name {target.id!r} is forbidden",
                rule_code="replay_python_call_forbidden",
                path="<call>",
            )
        return

    if isinstance(target, ast.Attribute):
        _check_identifier(target.attr, target, "attribute")
        if _is_name(target.value, "tl"):
            if target.attr in TRITON_UNAVAILABLE_CALL_GUIDANCE:
                _fail(
                    node,
                    f"`tl.{target.attr}` is unavailable in the public frontend; "
                    f"{TRITON_UNAVAILABLE_CALL_GUIDANCE[target.attr]}",
                    rule_code="replay_triton_api_unavailable",
                    path="<call>",
                )
            if target.attr == "cat":
                if len(node.args) not in {2, 3}:
                    _fail(
                        node,
                        "tl.cat requires two tensor operands and an optional "
                        "positional can_reorder flag",
                        rule_code="replay_cat_arity_invalid",
                        path="<call>",
                    )
                keyword_names = [keyword.arg for keyword in node.keywords]
                if any(name != "can_reorder" for name in keyword_names):
                    _fail(
                        node,
                        "tl.cat accepts only the optional can_reorder keyword; "
                        "dim/axis are not public Triton arguments",
                        rule_code="replay_cat_keyword_invalid",
                        path="<call>",
                    )
                if len(keyword_names) != len(set(keyword_names)):
                    _fail(
                        node,
                        "tl.cat has duplicate keyword arguments",
                        rule_code="replay_cat_keyword_duplicate",
                        path="<call>",
                    )
                if len(node.args) == 3:
                    reorder_node = node.args[2]
                    if not isinstance(reorder_node, ast.Constant) or reorder_node.value is not True:
                        _fail(
                            node,
                            "tl.cat requires literal can_reorder=True on this "
                            "Triton frontend",
                            rule_code="replay_cat_reorder_required",
                            path="<call>",
                        )
                elif not node.keywords:
                    _fail(
                        node,
                        "tl.cat requires literal can_reorder=True on this "
                        "Triton frontend",
                        rule_code="replay_cat_reorder_required",
                        path="<call>",
                    )
                else:
                    reorder_keyword = next(
                        keyword
                        for keyword in node.keywords
                        if keyword.arg == "can_reorder"
                    )
                    if (
                        not isinstance(reorder_keyword.value, ast.Constant)
                        or reorder_keyword.value.value is not True
                    ):
                        _fail(
                            node,
                            "tl.cat requires literal can_reorder=True on this "
                            "Triton frontend",
                            rule_code="replay_cat_reorder_required",
                            path="<call>",
                        )
            if target.attr.startswith("_") or target.attr in _FORBIDDEN_TL_CALLS:
                _fail(
                    node,
                    f"call to `tl.{target.attr}` is forbidden",
                    rule_code="replay_tl_call_forbidden",
                    path="<call>",
                )
            return
        # ``tl.math`` is a public Triton namespace.  Its available operations
        # remain available to a source-blind replay (for example
        # ``tl.math.erf``), but it shares the known unavailable public names
        # with top-level ``tl``.  This is target-API validation only; it
        # supplies no Source or pi information to Replay.
        if (
            isinstance(target.value, ast.Attribute)
            and _is_name(target.value.value, "tl")
            and target.value.attr == "math"
        ):
            if target.attr in TRITON_UNAVAILABLE_CALL_GUIDANCE:
                _fail(
                    node,
                    f"`tl.math.{target.attr}` is unavailable in the public "
                    "frontend; "
                    f"{TRITON_UNAVAILABLE_CALL_GUIDANCE[target.attr]}",
                    rule_code="replay_triton_api_unavailable",
                    path="<call>",
                )
            if target.attr.startswith("_"):
                _fail(
                    node,
                    f"call to `tl.math.{target.attr}` is forbidden",
                    rule_code="replay_tl_math_call_forbidden",
                    path="<call>",
                )
            return
        if _is_name(target.value, "libdevice"):
            if (
                "libdevice" not in imported_namespaces
                or target.attr not in allowed_libdevice_calls
            ):
                _fail(
                    node,
                    f"call to `libdevice.{target.attr}` is forbidden",
                    rule_code="replay_libdevice_call_forbidden",
                    path="<call>",
                )
            return
        if _is_name(target.value, "triton"):
            if target.attr not in TRITON_NAMESPACE_CALLS:
                _fail(
                    node,
                    f"call to `triton.{target.attr}` is forbidden",
                    rule_code="replay_triton_namespace_call_forbidden",
                    path="<call>",
                )
            if len(node.args) != 1 or node.keywords:
                _fail(
                    node,
                    "`triton.next_power_of_2` requires one positional argument "
                    "and no keywords",
                    rule_code="replay_triton_utility_signature_invalid",
                    path="<call>",
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
        rule_code="replay_call_target_forbidden",
        path="<call>",
    )


def _validate_loop(node: ast.For) -> None:
    iterator = node.iter
    if not isinstance(iterator, ast.Call):
        _fail(
            iterator,
            "for-loops must iterate over `range`, `tl.range`, or `tl.static_range`",
            rule_code="replay_loop_iterator_invalid",
            path="<loop>",
        )
    target = iterator.func
    valid = _is_name(target, "range") or (
        isinstance(target, ast.Attribute)
        and _is_name(target.value, "tl")
        and target.attr in {"range", "static_range"}
    )
    if not valid:
        _fail(
            iterator,
            "for-loops must iterate over `range`, `tl.range`, or `tl.static_range`",
            rule_code="replay_loop_iterator_invalid",
            path="<loop>",
        )
    if node.orelse:
        _fail(
            node,
            "for-else is not allowed in replay kernels",
            rule_code="replay_for_else_forbidden",
            path="<loop>",
        )


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
        _fail(
            node,
            "positional-only, variadic, and keyword-only parameters are forbidden",
            rule_code="replay_function_parameters_invalid",
            path="<function-signature>",
        )
    if args.defaults or args.kw_defaults:
        _fail(
            node,
            "function parameter defaults are forbidden",
            rule_code="replay_function_defaults_forbidden",
            path="<function-signature>",
        )
    if node.returns is not None or node.type_comment is not None:
        _fail(
            node,
            "return and type-comment annotations are forbidden",
            rule_code="replay_function_annotations_forbidden",
            path="<function-signature>",
        )

    arg_names = {arg.arg for arg in args.args}
    for arg in args.args:
        _check_identifier(arg.arg, arg, "parameter")
        _validate_annotation(arg.annotation)
    if len(node.decorator_list) != 1:
        _fail(
            node,
            "every function must have exactly one `@triton.jit` decorator",
            rule_code="replay_function_decorator_count_invalid",
            path="<decorator>",
        )
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
                _fail(
                    child,
                    "nested functions are forbidden",
                    rule_code="replay_nested_function_forbidden",
                    path="<function-body>",
                )
            if not isinstance(child, _ALLOWED_BODY_NODES):
                _fail(
                    child,
                    f"Python construct {type(child).__name__} is forbidden",
                    rule_code="replay_python_construct_forbidden",
                    path="<function-body>",
                )
            if isinstance(child, ast.Name):
                _check_identifier(child.id, child, "name")
                if isinstance(child.ctx, ast.Load) and child.id not in allowed_load_names:
                    _fail(
                        child,
                        f"global/free name {child.id!r} is forbidden",
                        rule_code="replay_global_name_forbidden",
                        path="<name>",
                    )
            elif isinstance(child, ast.Attribute):
                _check_identifier(child.attr, child, "attribute")
                if _is_name(child.value, "tl"):
                    if child.attr.startswith("_"):
                        _fail(
                            child,
                            f"private Triton attribute {child.attr!r} is forbidden",
                            rule_code="replay_private_tl_attribute_forbidden",
                            path="<attribute>",
                        )
                elif (
                    isinstance(child.value, ast.Attribute)
                    and _is_name(child.value.value, "tl")
                    and child.value.attr == "math"
                ):
                    if child.attr.startswith("_"):
                        _fail(
                            child,
                            f"private Triton math attribute {child.attr!r} is forbidden",
                            rule_code="replay_private_tl_math_attribute_forbidden",
                            path="<attribute>",
                        )
                elif _is_name(child.value, "libdevice"):
                    if (
                        "libdevice" not in imported_namespaces
                        or child.attr not in allowed_libdevice_calls
                    ):
                        _fail(
                            child,
                            f"attribute access `libdevice.{child.attr}` is forbidden",
                            rule_code="replay_libdevice_attribute_forbidden",
                            path="<attribute>",
                        )
                elif _is_name(child.value, "triton"):
                    if child.attr not in TRITON_NAMESPACE_CALLS:
                        _fail(
                            child,
                            f"attribute access `triton.{child.attr}` is forbidden",
                            rule_code="replay_triton_attribute_forbidden",
                            path="<attribute>",
                        )
                elif child.attr != "to":
                    _fail(
                        child,
                        f"attribute access {child.attr!r} is forbidden",
                        rule_code="replay_attribute_forbidden",
                        path="<attribute>",
                    )
            elif isinstance(child, ast.Call):
                _validate_call(
                    child,
                    jit_functions,
                    imported_namespaces,
                    allowed_libdevice_calls,
                )
                if any(keyword.arg is None for keyword in child.keywords):
                    _fail(
                        child,
                        "dynamic **keyword expansion is forbidden",
                        rule_code="replay_dynamic_keyword_forbidden",
                        path="<call>",
                    )
            elif isinstance(child, ast.For):
                _validate_loop(child)
            elif isinstance(child, ast.Constant):
                if child.value is not None and not isinstance(
                    child.value, (bool, int, float, str)
                ):
                    _fail(
                        child,
                        f"literal type {type(child.value).__name__} is forbidden",
                        rule_code="replay_literal_type_forbidden",
                        path="<literal>",
                    )
                if isinstance(child.value, str) and "__" in child.value:
                    _fail(
                        child,
                        "dunder strings are forbidden",
                        rule_code="replay_dunder_string_forbidden",
                        path="<literal>",
                    )


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
            _fail(
                nodes[name],
                "recursive Triton helper calls are forbidden",
                rule_code="replay_recursive_helper_forbidden",
                path="<function-call-graph>",
            )
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
        raise UnsafeReplayCodeError(
            "replay module is empty",
            rule_code="replay_module_empty",
            path="<module>",
        )
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        where = f"line {exc.lineno}: " if exc.lineno is not None else ""
        raise UnsafeReplayCodeError(
            f"{where}invalid Python syntax: {exc.msg}",
            rule_code="replay_python_syntax_invalid",
            path="<module-syntax>",
            line=exc.lineno,
        ) from exc

    function_nodes: list[ast.FunctionDef] = []
    saw_triton = False
    saw_tl = False
    imported_namespaces: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if "__" in node.value.value:
                _fail(
                    node,
                    "dunder strings are forbidden",
                    rule_code="replay_dunder_string_forbidden",
                    path="<module-docstring>",
                )
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
                _fail(
                    node,
                    f"duplicate function definition {node.name!r}",
                    rule_code="replay_duplicate_function_forbidden",
                    path="<module-function>",
                )
            function_nodes.append(node)
            continue
        _fail(
            node,
            "top level may contain only Triton imports and `@triton.jit` functions",
            rule_code="replay_top_level_statement_forbidden",
            path="<module-top-level>",
        )

    if not saw_triton:
        raise UnsafeReplayCodeError(
            "module must contain `import triton`",
            rule_code="replay_triton_import_missing",
            path="<module-imports>",
        )
    if not saw_tl:
        raise UnsafeReplayCodeError(
            "module must import `triton.language as tl`",
            rule_code="replay_tl_import_missing",
            path="<module-imports>",
        )
    if not function_nodes:
        raise UnsafeReplayCodeError(
            "module contains no `@triton.jit` function",
            rule_code="replay_jit_function_missing",
            path="<module-function>",
        )
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
            f"required Triton function {required_function!r} is missing",
            rule_code="replay_required_function_missing",
            path="<module-entry>",
        )
    return ModuleValidation(functions, tree)


# Naming used by the Gate1 design document and early callers.
validate_replay_module = validate_triton_module
