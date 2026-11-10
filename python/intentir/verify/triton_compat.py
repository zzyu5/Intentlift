"""Source-neutral compatibility rewrites for pinned Triton frontends."""

from __future__ import annotations

import ast
import hashlib
from typing import Any


TRITON_DOT_COMPAT_PROTOCOL = "triton_static_dot_transpose_v1"


def normalize_deprecated_triton_dot(code: str) -> tuple[str, dict[str, Any]]:
    """Normalize removed static ``trans_a/trans_b`` keywords to ``tl.trans``."""

    tree = ast.parse(code)
    language_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton.language" and alias.asname:
                    language_aliases.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module == "triton":
            for alias in node.names:
                if alias.name == "language":
                    language_aliases.add(alias.asname or alias.name)

    # An imported name is not enough: a parameter, assignment, nested import
    # or declaration may shadow it. Only the uniquely bound language module
    # can issue a compatibility rewrite.
    language_aliases = {
        name for name in language_aliases
        if sum(
            isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == name
            or isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del))
            or isinstance(node, ast.arg) and node.arg == name
            or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name
            for node in ast.walk(tree)
        ) == 1
        and not any(
            isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))
            and any(isinstance(part, ast.Name) and part.id == name for part in ast.walk(node))
            for node in ast.walk(tree)
        )
    }

    class Normalizer(ast.NodeTransformer):
        def __init__(self) -> None:
            self.count = 0

        def visit_Call(self, node: ast.Call) -> ast.AST:
            node = self.generic_visit(node)
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "dot"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in language_aliases
            ):
                return node
            transpose: dict[str, bool] = {}
            retained: list[ast.keyword] = []
            for keyword in node.keywords:
                if keyword.arg not in {"trans_a", "trans_b"}:
                    retained.append(keyword)
                    continue
                if not isinstance(keyword.value, ast.Constant) or not isinstance(
                    keyword.value.value, bool
                ):
                    raise ValueError(
                        "deprecated Triton dot transpose must be a static boolean"
                    )
                transpose[keyword.arg] = keyword.value.value
            if not transpose:
                return node
            if len(node.args) < 2:
                raise ValueError("deprecated Triton dot call lacks two operands")
            alias = node.func.value.id

            def transposed(value: ast.expr) -> ast.Call:
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=alias, ctx=ast.Load()),
                        attr="trans",
                        ctx=ast.Load(),
                    ),
                    args=[value],
                    keywords=[],
                )

            if transpose.get("trans_a", False):
                node.args[0] = transposed(node.args[0])
            if transpose.get("trans_b", False):
                node.args[1] = transposed(node.args[1])
            node.keywords = retained
            self.count += 1
            return node

    normalizer = Normalizer()
    normalized_tree = normalizer.visit(tree)
    if normalizer.count:
        ast.fix_missing_locations(normalized_tree)
        normalized = ast.unparse(normalized_tree) + "\n"
    else:
        normalized = code
    return normalized, {
        "protocol": TRITON_DOT_COMPAT_PROTOCOL,
        "rewrite_count": normalizer.count,
        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
    }


__all__ = ["TRITON_DOT_COMPAT_PROTOCOL", "normalize_deprecated_triton_dot"]
