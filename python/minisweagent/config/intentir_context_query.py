"""Bounded, source-aware context lookup for one IntentIR Lift worker.

The tool exposes only files already mounted in the worker sandbox.  It never
constructs or repairs J/C/pi.  Its purpose is to keep Source, Harness and
external diagnostics observable even when the model transport elides long
shell output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any


MAX_OUTPUT_CHARS = 10_000
QUERY_COMMAND = "python /input/tools/context_query.py"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _feedback_without_candidate(input_root: Path) -> dict[str, Any] | None:
    value = _load_json(input_root / "feedback.json")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("feedback.json must be null or an object")
    return {
        str(key): item
        for key, item in value.items()
        if key not in {"previous_artifact", "previous_candidate"}
    }


def _feedback_frontier(input_root: Path) -> None:
    """Print the host-authored bounded projection without reading the candidate."""

    path = input_root / "feedback_frontier.json"
    if not path.is_file():
        _emit_json(None)
        return
    value = _load_json(path)
    if value is not None and not isinstance(value, dict):
        raise TypeError("feedback_frontier.json must be null or an object")
    _emit_json(value)


def _previous_candidate(input_root: Path) -> dict[str, Any] | None:
    value = _load_json(input_root / "feedback.json")
    if not isinstance(value, dict):
        return None
    candidate = value.get("previous_candidate")
    candidate_field = "previous_candidate"
    if candidate is None:
        candidate = value.get("previous_artifact")
        candidate_field = "previous_artifact"
    if isinstance(candidate, str):
        candidate = json.loads(candidate)
    if candidate is None:
        return None
    if not isinstance(candidate, dict):
        raise TypeError(f"feedback {candidate_field} must be an object")
    c = candidate.get("c")
    if isinstance(c, dict) and "values" in c and c.get("values") != {}:
        raise ValueError(f"feedback {candidate_field} must be value-free")
    return candidate


def _write_previous(input_root: Path, output_root: Path) -> bool:
    candidate = _previous_candidate(input_root)
    if candidate is None:
        return False
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "candidate.json"
    if target.is_symlink():
        raise ValueError("existing candidate.json must not be a symlink")
    if target.exists():
        if not target.is_file():
            raise ValueError("existing candidate.json must be a regular file")
        current = _load_json(target)
        if not isinstance(current, dict):
            raise TypeError("existing candidate.json must contain one object")
        return False
    target.write_text(
        json.dumps(
            candidate,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def _observer_categories(feedback: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(feedback, dict):
        return []
    diagnostics = feedback.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    observer = diagnostics.get("observer_diff")
    if not isinstance(observer, dict):
        unresolved = diagnostics.get("unresolved_gate1")
        unresolved_diagnostics = (
            unresolved.get("diagnostics")
            if isinstance(unresolved, dict)
            else None
        )
        observer = (
            unresolved_diagnostics.get("observer_diff")
            if isinstance(unresolved_diagnostics, dict)
            else None
        )
    if not isinstance(observer, dict):
        return []
    categories = observer.get("categories")
    if not isinstance(categories, list):
        return []
    answer: list[dict[str, Any]] = []
    for category in categories:
        if not isinstance(category, dict):
            continue
        answer.append(
            {
                "category": category.get("category"),
                "difference_count": category.get("difference_count"),
            }
        )
    return answer


def _emit_json(value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    if len(text) > MAX_OUTPUT_CHARS:
        raise ValueError(
            f"query result is {len(text)} characters; use a narrower command"
        )
    print(text)


def _emit_compact_json(value: Any) -> None:
    """Emit one machine-readable compact ledger under the same hard byte bound."""

    text = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(text) > MAX_OUTPUT_CHARS:
        raise ValueError(
            f"query result is {len(text)} characters; use a narrower command"
        )
    print(text)


def _bootstrap(input_root: Path, output_root: Path) -> None:
    feedback = _feedback_without_candidate(input_root)
    candidate_already_present = (output_root / "candidate.json").is_file()
    copied = _write_previous(input_root, output_root)
    files = {}
    for relative in (
        "source.py",
        "harness_view.json",
        "feedback_frontier.json",
        "feedback.json",
    ):
        path = input_root / relative
        if not path.is_file():
            files[relative] = {"missing": True}
            continue
        files[relative] = {
            "bytes": path.stat().st_size,
            "read_command": {
                "source.py": f"{QUERY_COMMAND} source --start 1 --lines 160",
                "harness_view.json": f"{QUERY_COMMAND} harness",
                "feedback_frontier.json": f"{QUERY_COMMAND} feedback-frontier",
                "feedback.json": f"{QUERY_COMMAND} feedback",
            }[relative],
        }
    overview = {
        "candidate_copied_to_output": copied,
        "candidate_already_present": candidate_already_present,
        "files": files,
        "feedback_kind": (
            feedback.get("failure_kind")
            if isinstance(feedback, dict)
            else None
        ),
        "feedback_round": (
            feedback.get("relift_round")
            if isinstance(feedback, dict)
            else None
        ),
        "observer_categories": _observer_categories(feedback),
        "feedback_diagnostic_keys": (
            list(feedback.get("diagnostics", {}))
            if isinstance(feedback, dict)
            and isinstance(feedback.get("diagnostics"), dict)
            else []
        ),
        "next": [
            f"Run `{QUERY_COMMAND} feedback-frontier` before repairing a prior candidate.",
            f"Use `{QUERY_COMMAND} feedback --path diagnostics.KEY` for one exact diagnostic section.",
            "Read Source and Harness separately so neither can be transport-elided.",
            f"Use `{QUERY_COMMAND} harness-abi` for the exact compact p*/c*/shape-coordinate ledger.",
            f"Use `{QUERY_COMMAND} source-find TEXT --context 3` for literal Source lookup.",
            f"Use `{QUERY_COMMAND} source-outline --start 1 --count 32` to locate syntax-level control before paging a long Source.",
            f"After drafting, run `{QUERY_COMMAND} candidate-abi` for a read-only ABI diff.",
            f"Run `{QUERY_COMMAND} candidate-control` to audit authored control, time, effects, and outputs before CandidateCheck.",
            f"Inspect a draft with `{QUERY_COMMAND} candidate --path j.PATH` or `candidate-find HANDLE`.",
            f"Page an authored graph with `{QUERY_COMMAND} candidate-nodes --start 1 --count 40`.",
            "When Gate1 reports dataflow, run `python /input/tools/context_query.py feedback-category dataflow`; it preserves effect-order witnesses even when bulk node witnesses are paged out.",
            "When that dataflow view contains local_frontier, inspect its non-recursive immediate profiles before bulk recursive-hash witnesses; it is diagnostic-only and never prescribes J/C/pi.",
            f"Use `{QUERY_COMMAND} knowledge QUERY` and schema_query for focused lookup.",
        ],
    }
    _emit_json(overview)


def _feedback(input_root: Path, path: str | None = None) -> None:
    value = _feedback_without_candidate(input_root)
    if path is not None:
        _emit_json(_resolve_json_path(value, path))
        return
    try:
        _emit_json(value)
    except ValueError:
        if not isinstance(value, dict):
            raise
        diagnostics = value.get("diagnostics")
        reduced = dict(value)
        if isinstance(diagnostics, dict):
            reduced_diagnostics = dict(diagnostics)
            reduced_diagnostics.pop("observer_diff", None)
            reduced["diagnostics"] = reduced_diagnostics
        reduced["observer_categories"] = _observer_categories(value)
        reduced["narrow_query"] = (
            f"{QUERY_COMMAND} feedback-category CATEGORY"
        )
        _emit_json(reduced)


def _feedback_category_value(input_root: Path, name: str) -> dict[str, Any]:
    """Return one exact closed Gate1 category as a bounded read-only value."""

    value = _feedback_without_candidate(input_root)
    diagnostics = value.get("diagnostics") if isinstance(value, dict) else None
    observer = diagnostics.get("observer_diff") if isinstance(diagnostics, dict) else None
    category_root = "diagnostics.observer_diff.categories"
    if not isinstance(observer, dict) and isinstance(diagnostics, dict):
        unresolved = diagnostics.get("unresolved_gate1")
        unresolved_diagnostics = (
            unresolved.get("diagnostics")
            if isinstance(unresolved, dict)
            else None
        )
        observer = (
            unresolved_diagnostics.get("observer_diff")
            if isinstance(unresolved_diagnostics, dict)
            else None
        )
        category_root = (
            "diagnostics.unresolved_gate1.diagnostics.observer_diff.categories"
        )
    categories = observer.get("categories") if isinstance(observer, dict) else None
    if not isinstance(categories, list):
        raise KeyError("feedback contains no observer-diff categories")
    for category_index, category in enumerate(categories):
        if isinstance(category, dict) and category.get("category") == name:
            result = dict(category)
            rendered = json.dumps(category, sort_keys=True)
            category_path = f"{category_root}.{category_index}"
            if name == "dataflow" and isinstance(
                result.get("effect_order_witnesses"), list
            ):
                result["effect_order_hint"] = (
                    "Compare the ordered Source and Replay event sequences, "
                    "then inspect the LLM-authored J scopes, loop/control, and effects. "
                    "Compact pi has no effect order. The "
                    "witness is diagnostic only and does not prescribe a repair."
                )
                result["effect_order_query"] = (
                    f"{QUERY_COMMAND} feedback --path {category_path}."
                    "effect_order_witnesses"
                )
            if (
                name == "dataflow"
                and "region[root]" in rendered
                and "region[scf.for:" in rendered
                and "region.member" in rendered
            ):
                result["diagnostic_hint"] = (
                    "The observer examples cross root/scf.for region membership. "
                    "Inspect exact Source definition scope and the authored J loop/control; "
                    "fixed code has not inferred which producer, if any, must move."
                )
                result["knowledge_query"] = (
                    f"{QUERY_COMMAND} knowledge 'lexical lifetime' --context 5"
                )
            if len(
                json.dumps(result, sort_keys=True, ensure_ascii=False)
            ) > MAX_OUTPUT_CHARS:
                node_witnesses = result.pop("node_witnesses", None)
                if isinstance(node_witnesses, list):
                    result["node_witness_count"] = len(node_witnesses)
                    result["node_witness_query"] = (
                        f"{QUERY_COMMAND} feedback --path {category_path}."
                        "node_witnesses"
                    )
                    result["bounded_view"] = (
                        "bulk node witnesses omitted so causal role/effect "
                        "witnesses remain visible"
                    )
            return result
    available = [
        item.get("category") for item in categories if isinstance(item, dict)
    ]
    raise KeyError(f"unknown feedback category {name!r}; choose {available}")


def _feedback_category(input_root: Path, name: str) -> None:
    _emit_json(_feedback_category_value(input_root, name))


def _source(input_root: Path, start: int, line_count: int) -> None:
    if start < 1 or line_count < 1:
        raise ValueError("source start and lines must be positive")
    lines = (input_root / "source.py").read_text(encoding="utf-8").splitlines()
    selected: list[str] = []
    used = 0
    for ordinal, line in enumerate(lines[start - 1 :], start=start):
        rendered = f"{ordinal:5d} | {line}"
        if selected and used + len(rendered) + 1 > MAX_OUTPUT_CHARS - 300:
            break
        selected.append(rendered)
        used += len(rendered) + 1
        if len(selected) >= line_count:
            break
    print("\n".join(selected))
    next_line = start + len(selected)
    if next_line <= len(lines):
        print(
            f"\n[source continues: run `{QUERY_COMMAND} source --start "
            f"{next_line} --lines {line_count}`]"
        )


def _source_brief(input_root: Path) -> None:
    """Expose anonymous Source syntax plus the value-free Lift contract."""

    complete_source_outline = _source_outline_value(input_root, 1, 48)
    source_outline = {
        **complete_source_outline,
        "items": complete_source_outline["items"][:16],
        "next_start": (
            17 if len(complete_source_outline["items"]) > 16 else None
        ),
        "next_command": (
            f"{QUERY_COMMAND} source-outline --start 17 --count 16"
            if len(complete_source_outline["items"]) > 16
            else None
        ),
    }
    _emit_compact_json(
        {
            "protocol": "intentir.lift-source-brief.v3",
            "source": source_outline,
            "harness_abi": _compact_harness_abi_value(input_root),
            "compiler_realization": {
                "model_visible": False,
                "lift_acceptance_authority": False,
                "authority": (
                    "Source TTIR, SSA/dataflow realization, launch/grid, schedule, "
                    "target/compiler metadata and actual C values are host-private"
                ),
            },
            "authority": (
                "anonymous Source syntax plus value-free ABI/C declarations only; "
                "Lift authors every J/C/pi field"
            ),
        }
    )

def _source_find(input_root: Path, query: str, context: int) -> None:
    """Find literal Source text with bounded line-numbered context.

    This is deliberately a text lookup rather than a parser or semantic
    extractor.  Lift still decides what the matched Source means and authors
    every J/C/pi field.
    """

    if not query:
        raise ValueError("source-find query must be non-empty")
    if context < 0 or context > 20:
        raise ValueError("source-find context must be in [0, 20]")
    lines = (input_root / "source.py").read_text(encoding="utf-8").splitlines()
    needle = query.casefold()
    matches = [
        index for index, line in enumerate(lines) if needle in line.casefold()
    ]
    if not matches:
        raise KeyError(f"no Source lines match {query!r}")

    ranges: list[tuple[int, int]] = []
    for index in matches:
        begin = max(0, index - context)
        end = min(len(lines), index + context + 1)
        if ranges and begin <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((begin, end))

    blocks: list[str] = []
    used = 0
    omitted = False
    match_set = set(matches)
    for begin, end in ranges:
        rendered = [f"--- source.py:{begin + 1} ---"]
        rendered.extend(
            f"{'>' if ordinal in match_set else ' '} {ordinal + 1:5d} | "
            f"{lines[ordinal]}"
            for ordinal in range(begin, end)
        )
        block = "\n".join(rendered)
        remaining = MAX_OUTPUT_CHARS - 300 - used
        if len(block) + 1 > remaining:
            if remaining > 80:
                blocks.append(block[: remaining - 40] + "\n[context truncated]")
            omitted = True
            break
        blocks.append(block)
        used += len(block) + 1
    print("\n".join(blocks))
    if omitted:
        print("\n[more matches omitted; use a narrower literal query]")


def _source_outline_value(input_root: Path, start: int, count: int) -> dict[str, Any]:
    """Return syntax-only control, write, and memory-call navigation facts.

    The extra records deliberately stop before semantic classification.  A name
    written in a loop after an earlier lexical binding is only a Source
    navigation fact: Lift still decides whether it is a compiler-visible loop
    carrier and authors every J/C/pi relation.  Likewise, memory-call order is
    literal AST order, not an inferred J effect or pi order.
    """

    if start < 1 or count < 1 or count > 48:
        raise ValueError("source-outline start must be positive and count in [1, 48]")
    source = (input_root / "source.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename="source.py")
    items: list[dict[str, Any]] = []

    def bound_names(node: ast.AST) -> list[tuple[str, int, int]]:
        found: dict[str, tuple[int, int]] = {}

        class BoundNameVisitor(ast.NodeVisitor):
            def visit_Name(self, item: ast.Name) -> None:
                if isinstance(item.ctx, ast.Store):
                    found.setdefault(
                        item.id,
                        (getattr(item, "lineno", 0), getattr(item, "col_offset", 0)),
                    )

            def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, item: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, item: ast.Lambda) -> None:
                return

        BoundNameVisitor().visit(node)
        return [
            (name, line, column)
            for name, (line, column) in sorted(
                found.items(), key=lambda item: (item[1][0], item[1][1], item[0])
            )
        ]

    def function_bindings_before(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        control: ast.AST,
    ) -> dict[str, tuple[int, int]]:
        bindings: dict[str, tuple[int, int]] = {}
        arguments = [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if function.args.vararg is not None:
            arguments.append(function.args.vararg)
        if function.args.kwarg is not None:
            arguments.append(function.args.kwarg)
        for argument in arguments:
            bindings.setdefault(
                argument.arg,
                (
                    getattr(argument, "lineno", function.lineno),
                    getattr(argument, "col_offset", function.col_offset),
                ),
            )
        control_line = getattr(control, "lineno", 0)
        # A loop can be nested below an if/with/try rather than appearing as a
        # direct function-body statement.  Use literal source order for the
        # checklist and filter every discovered binding by its own location;
        # otherwise walking the enclosing statement can accidentally count a
        # write from inside the loop as a pre-existing definition.
        for statement in function.body:
            for name, line, column in bound_names(statement):
                if line < control_line:
                    bindings.setdefault(name, (line, column))
        return bindings

    def target_names(node: ast.AST) -> set[str]:
        return {name for name, _line, _column in bound_names(node)}

    def contains_return(node: ast.AST) -> bool:
        found = False

        class ReturnVisitor(ast.NodeVisitor):
            def visit_Return(self, item: ast.Return) -> None:
                nonlocal found
                found = True

            def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, item: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, item: ast.Lambda) -> None:
                return

        for child in getattr(node, "body", []):
            ReturnVisitor().visit(child)
        for child in getattr(node, "orelse", []):
            ReturnVisitor().visit(child)
        return found

    def ordered_memory_calls(node: ast.AST) -> list[list[int | str]]:
        calls: list[tuple[int, int, str]] = []
        memory_names = {
            "load",
            "store",
            "atomic_add",
            "atomic_and",
            "atomic_cas",
            "atomic_max",
            "atomic_min",
            "atomic_or",
            "atomic_xchg",
            "atomic_xor",
        }

        class MemoryCallVisitor(ast.NodeVisitor):
            def visit_Call(self, item: ast.Call) -> None:
                function = item.func
                leaf = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else function.id
                    if isinstance(function, ast.Name)
                    else None
                )
                if leaf in memory_names:
                    calls.append(
                        (
                            getattr(item, "lineno", 0),
                            getattr(item, "col_offset", 0),
                            ast.unparse(function)[:80],
                        )
                    )
                self.generic_visit(item)

            def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, item: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, item: ast.Lambda) -> None:
                return

        for child in getattr(node, "body", []):
            MemoryCallVisitor().visit(child)
        for child in getattr(node, "orelse", []):
            MemoryCallVisitor().visit(child)
        return [[line, call] for line, _column, call in sorted(calls)]

    class OutlineVisitor(ast.NodeVisitor):
        depth = 0
        functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

        def record(
            self,
            node: ast.AST,
            kind: str,
            extra: dict[str, Any] | None = None,
        ) -> None:
            line = getattr(node, "lineno", None)
            if not isinstance(line, int):
                return
            header = lines[line - 1].strip() if line <= len(lines) else ""
            record = {
                "kind": kind,
                "line": line,
                "end_line": getattr(node, "end_lineno", line),
                "depth": self.depth,
                "header": header[:200],
            }
            if extra:
                record.update(extra)
            items.append(record)

        def nested(self, node: ast.AST, kind: str) -> None:
            self.record(node, kind)
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.record(
                node,
                "function",
                {"ordered_memory_calls": ordered_memory_calls(node)},
            )
            self.functions.append(node)
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1
            self.functions.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.record(
                node,
                "async_function",
                {"ordered_memory_calls": ordered_memory_calls(node)},
            )
            self.functions.append(node)
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1
            self.functions.pop()

        def loop_extra(self, node: ast.For | ast.AsyncFor | ast.While) -> dict[str, Any]:
            function = self.functions[-1] if self.functions else None
            prior = (
                function_bindings_before(function, node)
                if function is not None
                else {}
            )
            body_writes: dict[str, tuple[int, int]] = {}
            for child in node.body:
                for name, line, column in bound_names(child):
                    body_writes.setdefault(name, (line, column))
            induction_names = (
                target_names(node.target)
                if isinstance(node, (ast.For, ast.AsyncFor))
                else set()
            )
            carried = [
                {
                    "name": name,
                    "definition_line": definition[0],
                    "first_loop_write_line": body_writes[name][0],
                }
                for name, definition in sorted(
                    prior.items(), key=lambda item: (item[1][0], item[1][1], item[0])
                )
                if name in body_writes and name not in induction_names
            ]
            return {
                "loop_target_names": sorted(induction_names),
                "lexically_preexisting_writes": carried,
                "contains_return": contains_return(node),
            }

        def visit_For(self, node: ast.For) -> None:
            self.record(node, "for", self.loop_extra(node))
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.record(node, "async_for", self.loop_extra(node))
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_While(self, node: ast.While) -> None:
            self.record(node, "while", self.loop_extra(node))
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_If(self, node: ast.If) -> None:
            has_return = contains_return(node)
            self.record(
                node,
                "if",
                {
                    "contains_return": has_return,
                    **(
                        {"ordered_memory_calls": ordered_memory_calls(node)}
                        if has_return
                        else {}
                    ),
                },
            )
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_Try(self, node: ast.Try) -> None:
            self.nested(node, "try")

        def visit_With(self, node: ast.With) -> None:
            self.nested(node, "with")

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            self.nested(node, "async_with")

        def visit_Match(self, node: ast.Match) -> None:
            self.nested(node, "match")

        def visit_Return(self, node: ast.Return) -> None:
            self.record(node, "return")

        def visit_Break(self, node: ast.Break) -> None:
            self.record(node, "break")

        def visit_Continue(self, node: ast.Continue) -> None:
            self.record(node, "continue")

    OutlineVisitor().visit(tree)
    page = items[start - 1 : start - 1 + count]
    next_start = start + len(page) if start - 1 + len(page) < len(items) else None
    return {
        "protocol": "intentir.lift-source-outline.v2",
        "authority": (
            "syntax-only Source navigation; names, return presence, and memory-call "
            "order are literal AST facts, not carrier/effect classification or "
            "frontend tuple order; Lift still authors all J/C/pi fields"
        ),
        "total": len(items),
        "start": start,
        "items": [
            {"ordinal": start + index, **item}
            for index, item in enumerate(page)
        ],
        "next_start": next_start,
        "next_command": (
            f"{QUERY_COMMAND} source-outline --start {next_start} "
            f"--count {count}"
            if next_start is not None
            else None
        ),
    }


def _source_outline(input_root: Path, start: int, count: int) -> None:
    _emit_json(_source_outline_value(input_root, start, count))


def _resolve_json_path(value: Any, path: str | None) -> Any:
    if not path:
        return value
    current = value
    for component in path.split("."):
        if isinstance(current, dict):
            if component not in current:
                raise KeyError(
                    f"unknown JSON path component {component!r}; "
                    f"choose {list(current)}"
                )
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            current = current[int(component)]
        else:
            raise KeyError(f"cannot descend through {component!r}")
    return current


def _harness(input_root: Path, path: str | None) -> None:
    value = _resolve_json_path(
        _load_json(input_root / "harness_view.json"), path
    )
    try:
        _emit_json(value)
    except ValueError:
        if path is not None or not isinstance(value, dict):
            raise
        _emit_json(
            {
                "keys": list(value),
                "query": f"{QUERY_COMMAND} harness --path KEY[.SUBKEY]",
            }
        )


def _harness_abi_value(
    input_root: Path, start: int = 1, count: int = 24
) -> dict[str, Any]:
    """Project the exact Harness-owned J/C classification into one small ledger.

    The anonymous compiler view already decides which public handles are p* and
    c*, which constexprs are bound non-C carriers, and which shape coordinates
    are parametric.  This helper only joins those existing fields by handle.  It
    never chooses a logical_ref, creates a dimension, or classifies Source use.
    """

    if start < 1 or count < 1 or count > 80:
        raise ValueError("harness-abi start/count must be positive and count <= 80")
    view = _load_json(input_root / "harness_view.json")
    if not isinstance(view, dict):
        raise TypeError("harness_view.json must be an object")
    semantic = view.get("semantic_lift_view", {})
    declarations = view.get("layer_c_declarations", {})
    if not isinstance(semantic, dict) or not isinstance(declarations, dict):
        raise TypeError("Harness view requires semantic and Layer-C declarations")
    parameter_order = semantic.get("parameter_order", [])
    arguments = semantic.get("arguments", [])
    bound_parameters = semantic.get("bound_constexpr_parameters", [])
    raw_slots = declarations.get("slots", [])
    if not isinstance(parameter_order, list) or not all(
        isinstance(item, str) for item in parameter_order
    ):
        raise TypeError("Harness parameter_order must be an array of strings")
    if not isinstance(arguments, list):
        raise TypeError("Harness arguments must be an array")
    if not isinstance(bound_parameters, list) or not all(
        isinstance(item, str) for item in bound_parameters
    ):
        raise TypeError("Harness bound constexpr declarations must be strings")
    if not isinstance(raw_slots, list) or not all(
        isinstance(item, dict) and isinstance(item.get("slot"), str)
        for item in raw_slots
    ):
        raise TypeError("Harness Layer-C slot declarations are malformed")
    slots_by_name = {item["slot"]: item for item in raw_slots}
    arguments_by_name = {
        item.get("name"): item
        for item in arguments
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    entries: list[dict[str, Any]] = []
    argument_fields = (
        "kind",
        "mode",
        "dtype",
        "rank",
        "shape_contract",
        "shape_coordinate",
    )
    visible_order = list(parameter_order) + [
        item["slot"] for item in raw_slots if item["slot"] not in parameter_order
    ]
    for name in visible_order:
        entry: dict[str, Any] = {
            "name": name,
            "artifact_owner": "J.signature" if name.startswith("p") else "C",
        }
        argument = arguments_by_name.get(name)
        if argument is not None:
            entry["classification"] = (
                "runtime_or_representation_parameter"
                if name.startswith("p")
                else "pure_c_argument"
            )
            entry["declared"] = {
                field: argument[field]
                for field in argument_fields
                if field in argument
            }
        elif name in bound_parameters:
            entry.update(
                {
                    "classification": "bound_non_c_constexpr_carrier",
                }
            )
        elif name in slots_by_name:
            declaration = slots_by_name[name]
            entry.update(
                {
                    "classification": (
                        "compiler_option_c_slot"
                        if declaration.get("kind") == "compiler"
                        else "pure_c_constexpr_slot"
                    ),
                    "c_declaration": dict(declaration),
                }
            )
        else:
            entry["classification"] = "unresolved_in_exact_harness_view"
        entries.append(entry)
    selected = entries[start - 1 : start - 1 + count]
    next_start = start + len(selected)
    return {
        "protocol": "intentir.lift-harness-abi-view.v1",
        "parameters": selected,
        "parameter_start": start,
        "parameter_total": len(entries),
        "parameter_next_command": (
            f"{QUERY_COMMAND} harness-abi --start {next_start} --count {count}"
            if next_start <= len(entries)
            else None
        ),
        "bound_non_c_constexprs": list(bound_parameters),
        "layer_c_declarations": [dict(item) for item in raw_slots],
        "shape_coordinates": semantic.get("shape_coordinates", []),
        "bound_constexpr_shape_coordinates": semantic.get(
            "bound_constexpr_shape_coordinates", {}
        ),
        "aliases": semantic.get("aliases", []),
        "agent_owned_not_inferred": [
            "p* logical_ref and t*/x* ownership",
            "logical operators, control, accesses, effects, and outputs",
            "each typed pi relation and every J/C handle bound by it",
        ],
        "authority": (
            "Value-free read-only projection of harness_view.json. No actual "
            "shape/scalar/C value, grid, launch, schedule, target or compiler "
            "metadata is present."
        ),
    }


def _compact_harness_abi_value(input_root: Path) -> dict[str, Any]:
    """Return the complete public ABI ledger without verbose per-row wrappers.

    Every value is copied from ``harness_view.json`` through
    :func:`_harness_abi_value`.  The projection deliberately omits concrete tensor
    shapes when a parametric ``shape_contract`` is present and records only rank;
    it never chooses a J tensor/dimension, ``logical_ref``, C attachment, or pi
    relation.  The ordinary paged ``harness-abi`` query remains the exact verbose
    source for any row that needs inspection.
    """

    verbose = _harness_abi_value(input_root, 1, 80)
    rows: list[list[Any]] = []
    for entry in verbose["parameters"]:
        name = entry["name"]
        classification = entry.get("classification")
        declared = entry.get("declared")
        if isinstance(declared, dict):
            kind = declared.get("kind")
            if kind == "tensor":
                row = [
                    name,
                    "tensor",
                    declared.get("mode"),
                    declared.get("dtype"),
                    declared.get("rank"),
                    declared.get("shape_contract"),
                ]
            elif kind == "scalar":
                row = [
                    name,
                    "scalar",
                    declared.get("mode"),
                    declared.get("dtype"),
                ]
                if declared.get("shape_coordinate") is not None:
                    row.append(declared["shape_coordinate"])
            else:
                row = [name, kind, declared.get("mode"), declared.get("dtype")]
                while row and row[-1] is None:
                    row.pop()
        elif classification == "bound_non_c_constexpr_carrier":
            row = [name, "bound_declaration"]
        elif isinstance(entry.get("c_declaration"), dict):
            declaration = entry["c_declaration"]
            row = [
                name,
                "c_declaration",
                declaration.get("kind"),
                declaration.get("value_type"),
                declaration.get("required"),
                declaration.get("role"),
            ]
        else:
            row = [name, classification or "unresolved"]
        rows.append(row)
    return {
        "protocol": "intentir.lift-harness-abi-compact.v1",
        "row_grammar": {
            "ownership": "p*=J.signature; c*=C",
            "tensor": [
                "name", "tensor", "mode", "dtype", "rank", "shape_contract",
            ],
            "scalar": ["name", "scalar", "mode", "dtype", "shape_coordinate?"],
            "bound": ["name", "bound_declaration"],
            "c_declaration": [
                "name", "c_declaration", "kind", "value_type", "required", "role?",
            ],
        },
        "parameter_rows": rows,
        "parameter_total": verbose["parameter_total"],
        "parameter_next_command": None,
        "shape_coordinates": verbose["shape_coordinates"],
        "bound_constexpr_shape_coordinates": verbose[
            "bound_constexpr_shape_coordinates"
        ],
        "aliases": verbose["aliases"],
        "agent_owned_not_inferred": verbose["agent_owned_not_inferred"],
        "exact_row_query": f"{QUERY_COMMAND} harness-abi --start START --count COUNT",
        "authority": (
            "Value-free public ABI/C declaration projection. Tensor values/shapes, "
            "runtime scalars, C values, grid, schedule and target metadata are absent."
        ),
    }


def _harness_abi(
    input_root: Path, start: int = 1, count: int = 24
) -> None:
    _emit_json(_harness_abi_value(input_root, start, count))


def _candidate(output_root: Path, path: str | None) -> None:
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("/output/candidate.json does not exist")
    value = _resolve_json_path(_load_json(candidate_path), path)
    try:
        _emit_json(value)
    except ValueError:
        if path is not None or not isinstance(value, dict):
            raise
        _emit_json(
            {
                "keys": list(value),
                "query": f"{QUERY_COMMAND} candidate --path KEY[.SUBKEY]",
            }
        )


def _candidate_find(output_root: Path, needle: str) -> None:
    """List bounded exact-value occurrences in the current LLM draft."""

    if not needle:
        raise ValueError("candidate-find requires a non-empty exact value")
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("/output/candidate.json does not exist")
    value = _load_json(candidate_path)
    paths: list[str] = []

    def visit(item: Any, path: str) -> None:
        if len(paths) >= 256:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}.{index}" if path else str(index))
        elif item == needle:
            paths.append(path)

    visit(value, "")
    _emit_json(
        {
            "exact_value": needle,
            "paths": paths,
            "match_count": len(paths),
            "truncated": len(paths) >= 256,
            "authority": "read-only; candidate.json was not edited",
        }
    )


def _bounded_diff_value(value: Any) -> Any:
    if value is _MISSING:
        return {"missing": True}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and len(value) <= 240:
        return value
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "type": type(value).__name__,
        "json_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


_MISSING = object()


def _candidate_diff(input_root: Path, output_root: Path) -> None:
    """Show a bounded mechanical diff from the supplied parent to the draft."""

    previous = _previous_candidate(input_root)
    candidate_path = output_root / "candidate.json"
    if previous is None or not candidate_path.is_file():
        _emit_json(
            {
                "protocol": "intentir.lift-candidate-diff.v1",
                "previous_candidate_found": previous is not None,
                "candidate_found": candidate_path.is_file(),
                "authority": "read-only; no artifact field was inferred or edited",
            }
        )
        return
    current = _load_json(candidate_path)
    differences: list[dict[str, Any]] = []
    total = 0
    limit = 48

    def record(path: str, before: Any, after: Any) -> None:
        nonlocal total
        total += 1
        if len(differences) < limit:
            differences.append(
                {
                    "path": path or "<root>",
                    "before": _bounded_diff_value(before),
                    "after": _bounded_diff_value(after),
                }
            )

    def visit(before: Any, after: Any, path: str) -> None:
        if type(before) is not type(after):
            record(path, before, after)
            return
        if isinstance(before, dict):
            for key in sorted(set(before) | set(after)):
                child = f"{path}.{key}" if path else str(key)
                if key not in before:
                    record(child, _MISSING, after[key])
                elif key not in after:
                    record(child, before[key], _MISSING)
                else:
                    visit(before[key], after[key], child)
            return
        if isinstance(before, list):
            common = min(len(before), len(after))
            for index in range(common):
                visit(before[index], after[index], f"{path}.{index}")
            for index in range(common, max(len(before), len(after))):
                record(
                    f"{path}.{index}",
                    before[index] if index < len(before) else _MISSING,
                    after[index] if index < len(after) else _MISSING,
                )
            return
        if before != after:
            record(path, before, after)

    visit(previous, current, "")
    component_changed = {
        component: previous.get(component) != current.get(component)
        for component in ("j", "c", "pi")
        if isinstance(previous, dict) and isinstance(current, dict)
    }
    _emit_json(
        {
            "protocol": "intentir.lift-candidate-diff.v1",
            "previous_candidate_found": True,
            "candidate_found": True,
            "artifact_changed": total > 0,
            "component_changed": component_changed,
            "difference_count": total,
            "differences": differences,
            "differences_omitted": max(0, total - len(differences)),
            "authority": (
                "read-only mechanical diff of two LLM-authored candidates; "
                "no semantic repair or acceptance decision was made"
            ),
        }
    )


def _iter_candidate_nodes(
    value: Any, path: str = ""
) -> list[tuple[str, dict[str, Any]]]:
    """Return authored node objects in their JSON traversal order."""

    nodes: list[tuple[str, dict[str, Any]]] = []

    def visit(item: Any, current_path: str) -> None:
        if isinstance(item, dict):
            if isinstance(item.get("id"), str) and isinstance(
                item.get("op"), str
            ):
                nodes.append((current_path, item))
            for key, child in item.items():
                visit(
                    child,
                    f"{current_path}.{key}" if current_path else str(key),
                )
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(
                    child,
                    f"{current_path}.{index}" if current_path else str(index),
                )

    visit(value, path)
    return nodes


def _node_summary(path: str, node: dict[str, Any]) -> dict[str, Any]:
    """Project only fields already authored on a node into a compact view."""

    summary: dict[str, Any] = {
        "path": path,
        "id": node.get("id"),
        "op": node.get("op"),
    }
    for field in ("operands", "result", "results", "type", "attrs"):
        if field in node:
            summary[field] = node[field]
    if node.get("op") in {"effect_if", "value_if"}:
        summary["predicate"] = node.get("predicate")
        summary["arms"] = {
            arm_name: {
                "scope_id": arm.get("scope_id"),
                "body_item_count": len(arm.get("body", []))
                if isinstance(arm.get("body"), list)
                else None,
                **(
                    {"yield_value": arm.get("yield_value")}
                    if "yield_value" in arm
                    else {}
                ),
            }
            for arm_name in ("positive", "negative")
            if isinstance((arm := node.get(arm_name)), dict)
        }
    for field in (
        "body",
        "then_body",
        "else_body",
        "positive_body",
        "negative_body",
        "carry",
        "results",
    ):
        value = node.get(field)
        if isinstance(value, list):
            summary[f"{field}_item_count"] = len(value)
    summary["authored_keys"] = list(node)
    return summary


def _candidate_nodes(
    output_root: Path,
    start: int,
    count: int,
    op: str | None = None,
) -> None:
    """Page a compact read-only view of the current authored J nodes."""

    if start < 1:
        raise ValueError("candidate-nodes start must be positive")
    if count < 1 or count > 100:
        raise ValueError("candidate-nodes count must be in [1, 100]")
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("/output/candidate.json does not exist")
    candidate = _load_json(candidate_path)
    nodes = _iter_candidate_nodes(candidate)
    if op is not None:
        nodes = [item for item in nodes if item[1].get("op") == op]
    selected = nodes[start - 1 : start - 1 + count]
    summaries: list[dict[str, Any]] = []
    for path, node in selected:
        summary = _node_summary(path, node)
        trial = summaries + [summary]
        if len(
            json.dumps(trial, sort_keys=True, ensure_ascii=False)
        ) > MAX_OUTPUT_CHARS - 1_500:
            if not summaries:
                summaries.append(
                    {
                        "path": path,
                        "id": node.get("id"),
                        "op": node.get("op"),
                        "authored_keys": list(node),
                        "inspect_command": (
                            f"{QUERY_COMMAND} candidate-node {node.get('id')}"
                        ),
                        "detail_omitted": "single compact node exceeded page budget",
                    }
                )
            break
        summaries = trial
    next_start = start + len(summaries)
    _emit_json(
        {
            "protocol": "intentir.lift-candidate-node-query.v1",
            "filter_op": op,
            "start": start,
            "returned": len(summaries),
            "total": len(nodes),
            "nodes": summaries,
            "next_command": (
                f"{QUERY_COMMAND} candidate-nodes --start {next_start} "
                f"--count {count}" + (f" --op {op}" if op is not None else "")
                if next_start <= len(nodes)
                else None
            ),
            "authority": "read-only; candidate.json was not edited or repaired",
        }
    )


def _candidate_node(output_root: Path, node_id: str) -> None:
    """Inspect one exact authored node without inferring its meaning."""

    if not node_id:
        raise ValueError("candidate-node requires a non-empty node id")
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("/output/candidate.json does not exist")
    matches = [
        item
        for item in _iter_candidate_nodes(_load_json(candidate_path))
        if item[1].get("id") == node_id
    ]
    if not matches:
        raise KeyError(f"candidate contains no authored node {node_id!r}")
    if len(matches) != 1:
        raise ValueError(
            f"candidate contains {len(matches)} authored nodes named {node_id!r}"
        )
    path, node = matches[0]
    result = {
        "protocol": "intentir.lift-candidate-node-query.v1",
        "path": path,
        "node": node,
        "authority": "read-only; candidate.json was not edited or repaired",
    }
    try:
        _emit_json(result)
    except ValueError:
        _emit_json(
            {
                "protocol": result["protocol"],
                "summary": _node_summary(path, node),
                "nested_nodes": [
                    _node_summary(child_path, child)
                    for child_path, child in _iter_candidate_nodes(node, path)
                    if child is not node
                ][:48],
                "exact_path_command": (
                    f"{QUERY_COMMAND} candidate --path {path}"
                ),
                "authority": result["authority"],
            }
        )


_CONTROL_NODE_OPS = frozenset(
    {
        "loop",
        "effect_if",
        "value_if",
        "reduce",
        "scan",
        "contraction",
    }
)
_CONTROL_BOUNDARY_FIELDS = (
    "accesses",
    "effects",
    "output_mappings",
    "reductions",
    "updates",
    "consumer_extensions",
    "access_guard_relations",
    "result_relations",
)
_CONTROL_PI_FIELDS = (
    "program_axes",
    "program_bindings",
    "iteration_bindings",
    "control_bindings",
    "materialization_bindings",
    "access_representation_bindings",
)


def _candidate_operand_access_records(
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Join only exact candidate node operands with their authored accesses."""

    j = candidate.get("j")
    boundary = j.get("boundary") if isinstance(j, dict) else None
    accesses = boundary.get("accesses", []) if isinstance(boundary, dict) else []
    authored = [item for item in accesses if isinstance(item, dict)]
    records: list[dict[str, Any]] = []
    for path, node in _iter_candidate_nodes(candidate):
        node_ref = node.get("id")
        operands = node.get("operands", [])
        if not isinstance(node_ref, str) or not isinstance(operands, list):
            continue
        attached = [item for item in authored if item.get("node_ref") == node_ref]
        operand_accesses = []
        for ordinal, operand in enumerate(operands):
            matches = [
                {
                    key: access.get(key)
                    for key in (
                        "access_id", "kind", "tensor", "index_expression",
                        "predicate", "fill_value",
                    )
                    if key in access
                }
                for access in attached
                if access.get("kind") == "load" and access.get("tensor") == operand
            ]
            operand_accesses.append(
                {
                    "ordinal": ordinal,
                    "operand": operand,
                    "authored_load_accesses": matches,
                }
            )
        stores = [
            {
                key: access.get(key)
                for key in (
                    "access_id", "kind", "tensor", "index_expression", "predicate",
                )
                if key in access
            }
            for access in attached
            if access.get("kind") in {"store", "update"}
        ]
        records.append(
            {
                "path": path,
                "node_ref": node_ref,
                "op": node.get("op"),
                "result": node.get("result"),
                "operand_accesses": operand_accesses,
                "authored_store_or_update_accesses": stores,
            }
        )
    return records


def _candidate_control_records(
    candidate: dict[str, Any], section: str
) -> list[dict[str, Any]]:
    j = candidate.get("j")
    boundary = j.get("boundary") if isinstance(j, dict) else None
    pi = candidate.get("pi")
    records: list[dict[str, Any]] = []
    if section == "operands":
        return _candidate_operand_access_records(candidate)
    if section == "boundary":
        owner = boundary if isinstance(boundary, dict) else {}
        fields = _CONTROL_BOUNDARY_FIELDS
        prefix = "j.boundary"
    elif section == "pi":
        owner = pi if isinstance(pi, dict) else {}
        fields = _CONTROL_PI_FIELDS
        prefix = "pi"
    else:
        raise ValueError(
            "candidate-control section must be overview, outputs, operands, "
            "boundary, or pi"
        )
    for field in fields:
        value = owner.get(field, [])
        if isinstance(value, list):
            records.extend(
                {"path": f"{prefix}.{field}.{index}", "value": item}
                for index, item in enumerate(value)
            )
        elif isinstance(value, dict):
            records.extend(
                {"path": f"{prefix}.{field}.{key}", "value": item}
                for key, item in value.items()
            )
        elif value is not None:
            records.append({"path": f"{prefix}.{field}", "value": value})
    return records


def _candidate_control_value(
    output_root: Path,
    section: str = "overview",
    start: int = 1,
    count: int = 20,
) -> dict[str, Any]:
    """Expose authored control/time/effect/output facts without repairing them."""

    if start < 1 or count < 1 or count > 80:
        raise ValueError("candidate-control start/count must be positive and count <= 80")
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        return {
            "protocol": "intentir.lift-candidate-control-view.v2",
            "candidate_found": False,
            "authority": "read-only; no artifact field was inferred or edited",
        }
    candidate = _load_json(candidate_path)
    if not isinstance(candidate, dict):
        raise TypeError("candidate.json must contain one root object")
    j = candidate.get("j")
    intent = j.get("intent") if isinstance(j, dict) else None
    function = (
        intent.get("intent_function") if isinstance(intent, dict) else None
    )
    boundary = j.get("boundary") if isinstance(j, dict) else None
    pi = candidate.get("pi")
    if not isinstance(function, dict):
        function = {}
    if not isinstance(boundary, dict):
        boundary = {}
    if not isinstance(pi, dict):
        pi = {}
    authority = (
        "Read-only projection of the current LLM-authored candidate. Counts and "
        "pages preserve authored multiplicity/order; no missing control, effect, "
        "output, operand/access edge, carrier, route, or relation was inferred "
        "or repaired."
    )
    if section == "overview":
        control_nodes = [
            _node_summary(path, node)
            for path, node in _iter_candidate_nodes(candidate)
            if node.get("op") in _CONTROL_NODE_OPS
        ]
        return {
            "protocol": "intentir.lift-candidate-control-view.v2",
            "candidate_found": True,
            "section": section,
            "intent_outputs": function.get("outputs", []),
            "intent_returns": function.get("returns", []),
            "scope_count": len(j.get("scopes", []))
            if isinstance(j, dict) and isinstance(j.get("scopes", []), list)
            else 0,
            "control_reduction_nodes": control_nodes[:20],
            "control_reduction_nodes_omitted": max(0, len(control_nodes) - 20),
            "boundary_counts": {
                field: len(boundary.get(field, []))
                if isinstance(boundary.get(field, []), (list, dict))
                else 0
                for field in _CONTROL_BOUNDARY_FIELDS
            },
            "pi_counts": {
                field: len(pi.get(field, []))
                if isinstance(pi.get(field, []), (list, dict))
                else 0
                for field in _CONTROL_PI_FIELDS
            },
            "next_queries": [
                f"{QUERY_COMMAND} candidate-control --section outputs",
                f"{QUERY_COMMAND} candidate-control --section operands --start 1 --count 20",
                f"{QUERY_COMMAND} candidate-control --section boundary --start 1 --count 20",
                f"{QUERY_COMMAND} candidate-control --section pi --start 1 --count 20",
            ],
            "authority": authority,
        }
    if section == "outputs":
        output_names = {
            item.get("name")
            for item in function.get("outputs", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        effects = boundary.get("effects", [])
        write_effects = [
            item
            for item in effects
            if isinstance(item, dict)
            and (
                item.get("target") in output_names
                or item.get("kind") in {"write", "update"}
            )
        ] if isinstance(effects, list) else []
        return {
            "protocol": "intentir.lift-candidate-control-view.v2",
            "candidate_found": True,
            "section": section,
            "intent_outputs": function.get("outputs", []),
            "intent_returns": function.get("returns", []),
            "output_mappings": boundary.get("output_mappings", []),
            "write_or_update_effects": write_effects,
            "access_guard_relations": boundary.get("access_guard_relations", []),
            "pi": {field: pi.get(field, []) for field in _CONTROL_PI_FIELDS},
            "authority": authority,
        }
    records = _candidate_control_records(candidate, section)
    selected: list[dict[str, Any]] = []
    for record in records[start - 1 : start - 1 + count]:
        trial = selected + [record]
        if len(json.dumps(trial, sort_keys=True, ensure_ascii=False)) > MAX_OUTPUT_CHARS - 1_500:
            break
        selected = trial
    next_start = start + len(selected)
    return {
        "protocol": "intentir.lift-candidate-control-view.v2",
        "candidate_found": True,
        "section": section,
        "start": start,
        "returned": len(selected),
        "total": len(records),
        "items": selected,
        "next_command": (
            f"{QUERY_COMMAND} candidate-control --section {section} "
            f"--start {next_start} --count {count}"
            if next_start <= len(records)
            else None
        ),
        "authority": authority,
    }


def _candidate_control(
    output_root: Path, section: str, start: int, count: int
) -> None:
    _emit_json(_candidate_control_value(output_root, section, start, count))


def _candidate_attachments(output_root: Path) -> None:
    """List only already-authored anchors available for C/pi binding."""

    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError("/output/candidate.json does not exist")
    candidate = _load_json(candidate_path)
    j = candidate.get("j") if isinstance(candidate, dict) else None
    j = j if isinstance(j, dict) else {}
    boundary = j.get("boundary")
    boundary = boundary if isinstance(boundary, dict) else {}
    c = candidate.get("c") if isinstance(candidate, dict) else None
    c = c if isinstance(c, dict) else {}
    pi = candidate.get("pi") if isinstance(candidate, dict) else None
    pi = pi if isinstance(pi, dict) else {}
    structured_nodes = [
        {
            "node_ref": node.get("id"),
            "op": node.get("op"),
            "trip": node.get("trip"),
        }
        for _path, node in _iter_candidate_nodes(candidate)
        if node.get("op") in {"loop", "effect_if", "value_if", "reduce", "scan"}
    ]
    accesses = [
        {
            key: item.get(key)
            for key in (
                "access_id", "kind", "tensor", "node_ref", "index_expression",
                "predicate", "fill_value",
            )
            if key in item
        }
        for item in boundary.get("accesses", [])
        if isinstance(item, dict)
    ]
    _emit_json(
        {
            "protocol": "intentir.lift-candidate-attachments",
            "domains": boundary.get("domains", {}),
            "structured_nodes": structured_nodes,
            "accesses": accesses,
            "c_slots": c.get("slots", []),
            "pi": {field: pi.get(field, []) for field in _CONTROL_PI_FIELDS},
            "authority": (
                "Read-only projection of the current LLM-authored candidate. It "
                "does not derive AxisRefs, add attachments, choose relations, or "
                "repair any field."
            ),
        }
    )


def _candidate_abi(input_root: Path, output_root: Path) -> None:
    """Compare one draft with the Harness-owned mechanical ABI, read-only.

    This intentionally does not propose logical refs, tensor declarations,
    symbolic dimensions, C subjects/axes, or any other semantic field.  It
    merely makes the same already-authored public plumbing visible without
    forcing an agent to manually diff two large JSON documents.
    """

    view = _load_json(input_root / "harness_view.json")
    if not isinstance(view, dict):
        raise TypeError("harness_view.json must be an object")
    semantic = view.get("semantic_lift_view", {})
    declarations = view.get("layer_c_declarations", {})
    if not isinstance(semantic, dict) or not isinstance(declarations, dict):
        raise TypeError("Harness view requires semantic and Layer-C declarations")
    raw_c_declarations = declarations.get("slots", [])
    if not isinstance(raw_c_declarations, list) or not all(
        isinstance(item, dict) and isinstance(item.get("slot"), str)
        for item in raw_c_declarations
    ):
        raise TypeError("Harness Layer-C declarations are malformed")
    candidate_path = output_root / "candidate.json"
    if not candidate_path.is_file():
        _emit_json(
            {
                "protocol": "intentir.lift-candidate-abi-query.v1",
                "candidate_found": False,
                "next": (
                    "Write a complete LLM-authored /output/candidate.json, "
                    f"then rerun `{QUERY_COMMAND} candidate-abi`."
                ),
            }
        )
        return
    candidate = _load_json(candidate_path)
    if not isinstance(candidate, dict):
        raise TypeError("candidate.json must contain one root object")

    issues: list[dict[str, Any]] = []

    def issue(
        code: str,
        path: str,
        expected: Any,
        actual: Any,
    ) -> None:
        issues.append(
            {
                "code": code,
                "path": path,
                "expected": expected,
                "actual": actual,
            }
        )

    parameter_order = semantic.get("parameter_order", [])
    if not isinstance(parameter_order, list) or not all(
        isinstance(name, str) for name in parameter_order
    ):
        raise TypeError("Harness parameter_order must be an array of strings")
    expected_p = [name for name in parameter_order if name.startswith("p")]
    expected_c = [
        item["slot"]
        for item in raw_c_declarations
        if item.get("kind") == "constexpr"
    ]

    j = candidate.get("j")
    signature = j.get("signature") if isinstance(j, dict) else None
    parameters = (
        signature.get("parameters") if isinstance(signature, dict) else None
    )
    if not isinstance(parameters, list):
        parameters = []
        issue(
            "signature_parameters_missing",
            "j.signature.parameters",
            expected_p,
            None,
        )
    actual_parameters = {
        item.get("name"): item
        for item in parameters
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    actual_p = [
        item.get("name") if isinstance(item, dict) else None
        for item in parameters
    ]
    if actual_p != expected_p:
        issue(
            "parameter_order_mismatch",
            "j.signature.parameters[*].name",
            expected_p,
            actual_p,
        )

    arguments = semantic.get("arguments", [])
    if not isinstance(arguments, list):
        raise TypeError("Harness arguments must be an array")
    arguments_by_name = {
        item.get("name"): item
        for item in arguments
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    bound_constexprs = semantic.get("bound_constexpr_parameters", [])
    if not isinstance(bound_constexprs, list):
        raise TypeError("Harness bound constexpr parameters must be an array")

    for name in expected_p:
        actual = actual_parameters.get(name)
        if actual is None:
            issue(
                (
                    "bound_non_c_constexpr_missing_from_j"
                    if name in bound_constexprs
                    else "parameter_missing"
                ),
                f"j.signature.parameters[{name}]",
                (
                    "bound non-C constexpr carrier in J.signature"
                    if name in bound_constexprs
                    else "one declaration in Harness order"
                ),
                None,
            )
            continue
        argument = arguments_by_name.get(name)
        if argument is None:
            if name not in bound_constexprs:
                issue(
                    "parameter_unclassified",
                    f"j.signature.parameters[{name}]",
                    "Harness argument or bound constexpr",
                    actual,
                )
                continue
            expected_fields = {
                "kind": ["constexpr"],
                "mode": "value",
            }
        else:
            source_kind = argument.get("kind")
            allowed_kinds = {
                "tensor": ["pointer", "tensor"],
                "scalar": ["scalar"],
                "opaque": ["opaque"],
                "none": ["opaque"],
            }.get(source_kind)
            if allowed_kinds is None:
                issue(
                    "argument_kind_unknown",
                    f"harness.arguments[{name}].kind",
                    ["tensor", "scalar", "opaque", "none"],
                    source_kind,
                )
                continue
            expected_fields = {
                "kind": allowed_kinds,
                "dtype": argument.get("dtype"),
            }
            if "mode" in argument:
                expected_fields["mode"] = argument.get("mode")
            if source_kind == "tensor":
                expected_fields["rank"] = argument.get("rank")
            elif source_kind == "scalar":
                expected_fields["rank"] = 0
            else:
                expected_fields["rank"] = None
                expected_fields["logical_ref"] = None
        for field, expected in expected_fields.items():
            observed = actual.get(field)
            matches = (
                observed in expected
                if isinstance(expected, list)
                else observed == expected
            )
            if not matches:
                issue(
                    "parameter_field_mismatch",
                    f"j.signature.parameters[{name}].{field}",
                    expected,
                    observed,
                )

    c = candidate.get("c")
    slots = c.get("slots") if isinstance(c, dict) else None
    values = c.get("values") if isinstance(c, dict) else None
    if not isinstance(slots, list):
        slots = []
        issue("c_slots_missing", "c.slots", "array", None)
    actual_slots = [item for item in slots if isinstance(item, dict)]
    actual_constexpr_c = [
        item.get("name")
        for item in actual_slots
        if item.get("kind") == "constexpr"
    ]
    if actual_constexpr_c != expected_c:
        issue(
            "constexpr_c_order_mismatch",
            "c.slots[kind=constexpr].name",
            expected_c,
            actual_constexpr_c,
        )
    expected_compiler = [
        item["slot"]
        for item in raw_c_declarations
        if item.get("kind") == "compiler"
    ]
    actual_compiler = [
        item.get("name")
        for item in actual_slots
        if item.get("kind") == "compiler"
    ]
    if actual_compiler != expected_compiler:
        issue(
            "compiler_c_order_mismatch",
            "c.slots[kind=compiler].name",
            expected_compiler,
            actual_compiler,
        )
    if values != {}:
        issue("c_values_mismatch", "c.values", {}, values)
    slots_by_name = {
        item.get("name"): item
        for item in actual_slots
        if isinstance(item.get("name"), str)
    }
    for declaration in raw_c_declarations:
        name = declaration["slot"]
        slot = slots_by_name.get(name)
        expected = {
            key: declaration.get(key)
            for key in ("kind", "value_type", "required")
        }
        if declaration.get("kind") == "compiler":
            expected["role"] = declaration.get("role")
        observed = (
            {key: slot.get(key) for key in expected}
            if slot is not None else None
        )
        if observed != expected:
            issue(
                "c_declaration_mismatch",
                f"c.slots[{name}]",
                expected,
                observed,
            )

    expected_constexprs = semantic.get("constexpr_parameters", [])
    if not isinstance(expected_constexprs, list):
        raise TypeError("Harness constexpr_parameters must be an array")
    actual_constexprs = [
        item.get("name")
        for item in parameters
        if isinstance(item, dict) and item.get("kind") == "constexpr"
    ] + actual_constexpr_c
    if set(actual_constexprs) != set(expected_constexprs):
        issue(
            "constexpr_membership_mismatch",
            "J constexpr parameters union C constexpr slots",
            sorted(expected_constexprs),
            sorted(
                name for name in actual_constexprs if isinstance(name, str)
            ),
        )

    expected_aliases = semantic.get("aliases", [])
    boundary = j.get("boundary") if isinstance(j, dict) else None
    actual_aliases = (
        boundary.get("aliases") if isinstance(boundary, dict) else None
    )
    if actual_aliases != expected_aliases:
        issue(
            "alias_mismatch",
            "j.boundary.aliases",
            expected_aliases,
            actual_aliases,
        )

    issue_limit = 48
    _emit_json(
        {
            "protocol": "intentir.lift-candidate-abi-query.v1",
            "candidate_found": True,
            "matches_public_mechanical_abi": not issues,
            "issue_count": len(issues),
            "issues": issues[:issue_limit],
            "issues_omitted": max(0, len(issues) - issue_limit),
            "counts": {
                "expected_j_parameters": len(expected_p),
                "actual_j_parameters": len(parameters),
                "expected_constexpr_c_slots": len(expected_c),
                "expected_compiler_c_slots": len(expected_compiler),
            },
            "agent_owned_not_inferred": [
                "parameter logical_ref and logical tensor/dimension ownership",
                "bound-constexpr semantic dtype/rank/logical_ref",
                "C slot subject_ref and axis_ref",
                "all J graph, boundary, scope, storage, and pi content",
            ],
            "authority": (
                "Read-only diagnostic only; validator, CandidateCheck and the "
                "external Runner remain authoritative and no field was edited."
            ),
        }
    )


def _knowledge(input_root: Path, query: str, context: int) -> None:
    if not query:
        raise ValueError("knowledge query must be non-empty")
    needle = query.casefold()
    blocks: list[str] = []
    for relative in (
        "knowledge/lift_contract.md",
        "knowledge/preflight.md",
    ):
        lines = (input_root / relative).read_text(encoding="utf-8").splitlines()
        ranges: list[tuple[int, int]] = []
        for index, line in enumerate(lines):
            if needle in line.casefold():
                ranges.append(
                    (
                        max(0, index - context),
                        min(len(lines), index + context + 1),
                    )
                )
        merged: list[tuple[int, int]] = []
        for begin, end in ranges:
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((begin, end))
        for begin, end in merged:
            rendered = [f"--- {relative}:{begin + 1} ---"]
            rendered.extend(
                f"{ordinal + 1:5d} | {lines[ordinal]}"
                for ordinal in range(begin, end)
            )
            block = "\n".join(rendered)
            if sum(len(item) + 1 for item in blocks) + len(block) > MAX_OUTPUT_CHARS:
                print("\n".join(blocks))
                print("\n[more matches omitted; use a narrower query]")
                return
            blocks.append(block)
    if not blocks:
        raise KeyError(f"no contract/preflight lines match {query!r}")
    print("\n".join(blocks))


def main() -> int:
    global QUERY_COMMAND
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/output"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap")
    feedback = subparsers.add_parser("feedback")
    feedback.add_argument("--path")
    subparsers.add_parser("feedback-frontier")
    category = subparsers.add_parser("feedback-category")
    category.add_argument("name")
    source = subparsers.add_parser("source")
    source.add_argument("--start", type=int, default=1)
    source.add_argument("--lines", type=int, default=160)
    source_find = subparsers.add_parser("source-find")
    source_find.add_argument("query")
    source_find.add_argument("--context", type=int, default=3)
    source_outline = subparsers.add_parser("source-outline")
    source_outline.add_argument("--start", type=int, default=1)
    source_outline.add_argument("--count", type=int, default=32)
    subparsers.add_parser("source-brief")
    harness = subparsers.add_parser("harness")
    harness.add_argument("--path")
    harness_abi = subparsers.add_parser("harness-abi")
    harness_abi.add_argument("--start", type=int, default=1)
    harness_abi.add_argument("--count", type=int, default=24)
    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--path")
    candidate_find = subparsers.add_parser("candidate-find")
    candidate_find.add_argument("value")
    subparsers.add_parser("candidate-diff")
    candidate_nodes = subparsers.add_parser("candidate-nodes")
    candidate_nodes.add_argument("--start", type=int, default=1)
    candidate_nodes.add_argument("--count", type=int, default=40)
    candidate_nodes.add_argument("--op")
    candidate_node = subparsers.add_parser("candidate-node")
    candidate_node.add_argument("node_id")
    subparsers.add_parser("candidate-abi")
    subparsers.add_parser("candidate-attachments")
    candidate_control = subparsers.add_parser("candidate-control")
    candidate_control.add_argument(
        "--section",
        choices=("overview", "outputs", "operands", "boundary", "pi"),
        default="overview",
    )
    candidate_control.add_argument("--start", type=int, default=1)
    candidate_control.add_argument("--count", type=int, default=20)
    knowledge = subparsers.add_parser("knowledge")
    knowledge.add_argument("query")
    knowledge.add_argument("--context", type=int, default=3)
    subparsers.add_parser("copy-previous")
    args = parser.parse_args()
    if args.input_root != Path("/input") or args.output_root != Path("/output"):
        QUERY_COMMAND = " ".join(
            (
                "python",
                shlex.quote(sys.argv[0]),
                "--input-root",
                shlex.quote(str(args.input_root)),
                "--output-root",
                shlex.quote(str(args.output_root)),
            )
        )

    if args.command == "bootstrap":
        _bootstrap(args.input_root, args.output_root)
    elif args.command == "feedback":
        _feedback(args.input_root, args.path)
    elif args.command == "feedback-frontier":
        _feedback_frontier(args.input_root)
    elif args.command == "feedback-category":
        _feedback_category(args.input_root, args.name)
    elif args.command == "source":
        _source(args.input_root, args.start, args.lines)
    elif args.command == "source-find":
        _source_find(args.input_root, args.query, args.context)
    elif args.command == "source-outline":
        _source_outline(args.input_root, args.start, args.count)
    elif args.command == "source-brief":
        _source_brief(args.input_root)
    elif args.command == "harness":
        _harness(args.input_root, args.path)
    elif args.command == "harness-abi":
        _harness_abi(args.input_root, args.start, args.count)
    elif args.command == "candidate":
        _candidate(args.output_root, args.path)
    elif args.command == "candidate-find":
        _candidate_find(args.output_root, args.value)
    elif args.command == "candidate-diff":
        _candidate_diff(args.input_root, args.output_root)
    elif args.command == "candidate-nodes":
        _candidate_nodes(args.output_root, args.start, args.count, args.op)
    elif args.command == "candidate-node":
        _candidate_node(args.output_root, args.node_id)
    elif args.command == "candidate-abi":
        _candidate_abi(args.input_root, args.output_root)
    elif args.command == "candidate-attachments":
        _candidate_attachments(args.output_root)
    elif args.command == "candidate-control":
        _candidate_control(
            args.output_root, args.section, args.start, args.count
        )
    elif args.command == "knowledge":
        _knowledge(args.input_root, args.query, args.context)
    elif not _write_previous(args.input_root, args.output_root):
        raise ValueError("feedback contains no previous candidate or artifact")
    else:
        print("copied previous candidate to /output/candidate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
