"""TileLang Level-2 observer: structural TIR with diagnostic PTX evidence."""

from __future__ import annotations

import ast
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

from intentir.ir.canonical_construction import CanonicalHandleAllocator

from .ptx import (
    PTXFingerprint,
    compare_ptx_fingerprints,
    fingerprint_ptx,
)


TILELANG_FINGERPRINT_SCHEMA = "tilelang-tir-semantic-structure-ptx-diagnostic-v6"
_T_OP_RE = re.compile(r"\bT\.([A-Za-z_][A-Za-z0-9_]*)")
_SCHEDULE_ITERATORS = {
    "parallel",
    "serial",
    "vectorized",
    "unroll",
    "pipelined",
    "thread_binding",
}
_SCHEDULE_ITERATOR_PATHS = {f"t.{name}" for name in _SCHEDULE_ITERATORS}
_ZERO_ARG_SCALAR_TYPES = re.compile(
    r"\bT\.(?:bool|float16|float32|float64|int8|int16|int32|int64|uint8|uint16|uint32|uint64)\(\s*\)"
)


class TileLangObservationError(ValueError):
    """TileLang artifacts cannot support the strong observer."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def normalize_tilelang_tir_ops(tir: str) -> tuple[str, ...]:
    """Retain public TileLang/TIR operation topology, not source identifiers."""

    if not isinstance(tir, str) or not tir.strip():
        raise TileLangObservationError("TileLang TIR is empty")
    # Use the same proof-irrelevant let/identifier normalization as the
    # structural fingerprint so the operation inventory cannot reintroduce an
    # ANF distinction that the structure deliberately quotients.
    normalized_tir = ast.unparse(_normalized_tilelang_tree(tir))
    operations: list[str] = []
    for raw in normalized_tir.splitlines():
        line = raw.split("#", 1)[0]
        for match in _T_OP_RE.findall(line):
            # Loop spelling and zero-argument scalar declarations are
            # realization/compiler plumbing.  The AST structure below retains
            # their bounds and uses, while the topology inventory records one
            # frontend-neutral loop for every actual Python/TIR ``for``.
            if match.lower() in _SCHEDULE_ITERATORS:
                continue
            if match in {
                "bool",
                "float16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            } and _ZERO_ARG_SCALAR_TYPES.search(line):
                continue
            operations.append(f"T.{match}")
        stripped = line.strip()
        if stripped.startswith("for "):
            operations.append("control.for")
        elif stripped.startswith("if "):
            operations.append("control.if")
        elif stripped.startswith("else:"):
            operations.append("control.else")
    if "T.prim_func" not in operations:
        raise TileLangObservationError("TileLang artifact lacks a T.prim_func")
    if "T.Kernel" not in operations and "T.launch_thread" not in operations:
        raise TileLangObservationError("TileLang artifact lacks device launch topology")
    return tuple(operations)


def _call_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_path(node.value)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


class _TIRAlphaNormalizer(ast.NodeTransformer):
    """Remove authored identifiers while retaining TIR data/control structure."""

    def __init__(self) -> None:
        self.scopes: list[dict[str, str]] = [{}]
        self.next_parameter = 0
        self.next_local = 0

    def _bind(self, name: str, *, parameter: bool = False) -> str:
        scope = self.scopes[-1]
        existing = scope.get(name)
        if existing is not None:
            return existing
        if parameter:
            replacement = CanonicalHandleAllocator.canonical_names(
                "p", self.next_parameter + 1
            )[-1]
            self.next_parameter += 1
        else:
            replacement = CanonicalHandleAllocator.canonical_names(
                "v", self.next_local + 1
            )[-1]
            self.next_local += 1
        scope[name] = replacement
        return replacement

    def _lookup(self, name: str) -> str | None:
        for scope in reversed(self.scopes):
            replacement = scope.get(name)
            if replacement is not None:
                return replacement
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "tir_entry"
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.returns = self.visit(node.returns) if node.returns is not None else None
        self.scopes.append({})
        try:
            node.args = self.visit(node.args)  # type: ignore[assignment]
            node.body = [self.visit(item) for item in node.body]
        finally:
            self.scopes.pop()
        node.type_comment = None
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._bind(node.arg, parameter=True)
        node.annotation = (
            self.visit(node.annotation) if node.annotation is not None else None
        )
        node.type_comment = None
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "T":
            return node
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            node.id = self._bind(node.id)
        else:
            replacement = self._lookup(node.id)
            if replacement is not None:
                node.id = replacement
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        path = _call_path(node.func)
        if path == "T.func_attr" and node.args and isinstance(node.args[0], ast.Dict):
            attributes = node.args[0]
            for index, key in enumerate(attributes.keys):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "global_symbol"
                    and index < len(attributes.values)
                ):
                    attributes.values[index] = ast.Constant("<entry>")
        elif path == "T.block" and node.args:
            if isinstance(node.args[0], ast.Constant) and isinstance(
                node.args[0].value, str
            ):
                node.args[0] = ast.Constant("<block>")
        return node


class _SingleUsePureLetNormalizer(ast.NodeTransformer):
    """Quotient proof-irrelevant A-normal-form choices inside one basic block.

    TileLang/TIR printers may retain a one-use scalar buffer read as an annotated
    local or inline that same expression at its sole consumer.  Neither J nor pi
    assigns ownership to this local spelling.  We remove only consecutive,
    call-free, single-use assignments before their immediate consumer, so no
    control boundary or intervening effect can be crossed.
    """

    @staticmethod
    def _binding(statement: ast.stmt) -> tuple[str, ast.expr] | None:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            name, value = statement.target.id, statement.value
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name, value = statement.targets[0].id, statement.value
        else:
            return None
        if any(
            isinstance(node, (ast.Call, ast.Await, ast.NamedExpr))
            for node in ast.walk(value)
        ):
            return None
        return name, value

    @staticmethod
    def _load_count(node: ast.AST, name: str) -> int:
        return sum(
            1
            for item in ast.walk(node)
            if isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Load)
            and item.id == name
        )

    @staticmethod
    def _has_rebind(node: ast.AST, name: str) -> bool:
        return any(
            isinstance(item, ast.Name)
            and isinstance(item.ctx, (ast.Store, ast.Del))
            and item.id == name
            for item in ast.walk(node)
        )

    @staticmethod
    def _substitute(node: ast.AST, name: str, value: ast.expr) -> ast.AST:
        class Substitution(ast.NodeTransformer):
            def visit_Name(self, item: ast.Name) -> ast.AST:
                if isinstance(item.ctx, ast.Load) and item.id == name:
                    return ast.copy_location(copy.deepcopy(value), item)
                return item

        return Substitution().visit(node)

    def _statement_list(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        visited = [self.visit(statement) for statement in statements]
        result: list[ast.stmt] = []
        index = 0
        while index < len(visited):
            run_end = index
            bindings: list[tuple[ast.stmt, str, ast.expr]] = []
            while run_end < len(visited):
                binding = self._binding(visited[run_end])
                if binding is None:
                    break
                bindings.append((visited[run_end], binding[0], binding[1]))
                run_end += 1
            if not bindings or run_end >= len(visited):
                result.append(visited[index])
                index += 1
                continue

            consumer = visited[run_end]
            kept: list[ast.stmt] = []
            for binding_index, (statement, name, value) in enumerate(bindings):
                suffix: list[ast.AST] = [
                    item[0] for item in bindings[binding_index + 1 :]
                ] + [consumer]
                uses = sum(self._load_count(item, name) for item in suffix)
                rebound = any(self._has_rebind(item, name) for item in suffix)
                if uses != 1 or rebound:
                    kept.append(statement)
                    continue
                for later_index in range(binding_index + 1, len(bindings)):
                    later_statement, later_name, later_value = bindings[later_index]
                    later_statement = self._substitute(later_statement, name, value)
                    refreshed = self._binding(later_statement)
                    if refreshed is not None:
                        later_name, later_value = refreshed
                    bindings[later_index] = (
                        later_statement,
                        later_name,
                        later_value,
                    )
                consumer = self._substitute(consumer, name, value)  # type: ignore[assignment]
            result.extend(kept)
            result.append(consumer)
            index = run_end + 1
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._statement_list(node.body)
        return node

    def visit_With(self, node: ast.With) -> ast.AST:
        node.items = [self.visit(item) for item in node.items]  # type: ignore[list-item]
        node.body = self._statement_list(node.body)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        node.target = self.visit(node.target)  # type: ignore[assignment]
        node.iter = self.visit(node.iter)  # type: ignore[assignment]
        node.body = self._statement_list(node.body)
        node.orelse = self._statement_list(node.orelse)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        node.test = self.visit(node.test)  # type: ignore[assignment]
        node.body = self._statement_list(node.body)
        node.orelse = self._statement_list(node.orelse)
        return node


class _TIRSemanticNormalizer(_TIRAlphaNormalizer):
    """Erase only loop-schedule spellings, retaining bounds and loop bodies."""

    def visit_For(self, node: ast.For) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        iterator = node.iter
        if isinstance(iterator, ast.Call):
            path = _call_path(iterator.func)
            if (
                path is not None
                and path.lower() in _SCHEDULE_ITERATOR_PATHS
            ):
                iterator.func = ast.Name(id="range", ctx=ast.Load())
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = super().visit_Call(node)  # type: ignore[assignment]
        path = _call_path(node.func)
        if path in {"T.reads", "T.writes"}:
            node.args = sorted(
                node.args,
                key=lambda item: ast.dump(
                    item, annotate_fields=True, include_attributes=False
                ),
            )
        return node


def _normalized_tilelang_tree(tir: str) -> ast.AST:
    try:
        tree = ast.parse(tir)
    except SyntaxError as exc:
        raise TileLangObservationError(f"TileLang TIR is not parseable: {exc}") from exc
    tree = _SingleUsePureLetNormalizer().visit(tree)
    tree = _TIRSemanticNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def normalize_tilelang_tir_structure(tir: str) -> str:
    """Build an identifier-free structural AST for the public TIR program.

    Unlike the operation inventory, this retains constants, loop bounds,
    subscripts, operators, buffer declarations, calls and control nesting.  It
    deliberately ignores comments, source locations, entry names and block
    labels.
    """

    if not isinstance(tir, str) or not tir.strip():
        raise TileLangObservationError("TileLang TIR is empty")
    normalized = _normalized_tilelang_tree(tir)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def _semantic_tilelang_projection(
    tir: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Project public TIR semantics without consulting generated PTX.

    TileLang's frontend boundary is TIR.  The projection therefore derives its
    numerical vocabulary, global memory effect kinds, and program-coordinate
    vocabulary directly from the normalized TIR AST.  PTX stays attached to the
    fingerprint as lower-level diagnostic evidence only.

    The numerical inventory is intentionally a set of broad families.  It does
    not count instructions or temporary materializations, which are realization
    choices rather than J-owned semantics.
    """

    normalized = _normalized_tilelang_tree(tir)
    global_buffers: set[str] = set()
    for node in ast.walk(normalized):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and _call_path(node.value.func) == "T.match_buffer"
        ):
            global_buffers.add(node.targets[0].id)

    families: set[str] = set()
    memory_kinds: set[str] = set()
    program_coordinates: set[str] = set()
    binary_families = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mul",
        ast.Div: "div",
    }
    unary_calls = {
        "T.exp": "exp",
        "T.exp2": "exp",
        "T.floor": "floor",
        "T.log": "log",
        "T.log2": "log",
        "T.max": "max",
        "T.min": "min",
        "T.rsqrt": "rsqrt",
        "T.sqrt": "sqrt",
        "T.sin": "sin",
    }

    def references_global_buffer(value: ast.AST) -> bool:
        return any(
            isinstance(item, ast.Name) and item.id in global_buffers
            for item in ast.walk(value)
        )

    for node in ast.walk(normalized):
        if isinstance(node, ast.BinOp):
            family = binary_families.get(type(node.op))
            if family is not None:
                families.add(family)
        elif isinstance(node, ast.IfExp):
            families.add("select")
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id not in global_buffers:
                continue
            if isinstance(node.ctx, ast.Load):
                memory_kinds.add("load")
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                memory_kinds.add("store")
        elif not isinstance(node, ast.Call):
            continue

        if not isinstance(node, ast.Call):
            continue
        path = _call_path(node.func)
        family = unary_calls.get(str(path))
        if family is not None:
            families.add(family)
        elif path in {"T.Select", "T.if_then_else"}:
            families.add("select")
        elif path in {"T.atomic_add", "T.atomic_max", "T.atomic_min"}:
            memory_kinds.add("atomic")
        elif path == "T.copy" and len(node.args) >= 2:
            if references_global_buffer(node.args[0]):
                memory_kinds.add("load")
            if references_global_buffer(node.args[1]):
                memory_kinds.add("store")
        elif path == "T.reduce" and len(node.args) >= 3:
            reducer = node.args[2]
            if isinstance(reducer, ast.Constant) and isinstance(reducer.value, str):
                if reducer.value == "sum":
                    families.add("add")
                elif reducer.value in {"max", "min"}:
                    families.add(reducer.value)
        elif path == "T.launch_thread" and node.args:
            coordinate = node.args[0]
            if (
                isinstance(coordinate, ast.Constant)
                and isinstance(coordinate.value, str)
                and re.fullmatch(r"blockIdx\.[xyz]", coordinate.value)
            ):
                program_coordinates.add(coordinate.value)

    return (
        tuple(sorted(families)),
        tuple(sorted(memory_kinds)),
        tuple(sorted(program_coordinates)),
    )


@dataclass(frozen=True)
class TileLangFingerprint:
    schema_version: str
    tir_ops: tuple[str, ...]
    tir_op_multiset: tuple[tuple[str, int], ...]
    tir_structure: str
    interface: tuple[tuple[str, str], ...]
    semantic_value_families: tuple[str, ...]
    semantic_memory_effect_kinds: tuple[str, ...]
    program_coordinates: tuple[str, ...]
    ptx: PTXFingerprint
    schedule_view: tuple[tuple[str, str], ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tir_ops": list(self.tir_ops),
            "tir_op_multiset": [list(item) for item in self.tir_op_multiset],
            "tir_structure": self.tir_structure,
            "interface": [list(item) for item in self.interface],
            "semantic_value_families": list(self.semantic_value_families),
            "semantic_memory_effect_kinds": list(self.semantic_memory_effect_kinds),
            "program_coordinates": list(self.program_coordinates),
            "ptx": self.ptx.to_dict(),
            "schedule_view": [list(item) for item in self.schedule_view],
            "digest": self.digest,
        }


def fingerprint_tilelang(
    tir: str,
    ptx: str,
    *,
    interface: Mapping[str, Any] | None = None,
    target: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> TileLangFingerprint:
    tir_ops = normalize_tilelang_tir_ops(tir)
    tir_structure = normalize_tilelang_tir_structure(tir)
    interface_view = tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
            for key, value in (interface or {}).items()
        )
    )
    schedule_view = tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
            for key, value in (schedule or {}).items()
        )
    )
    semantic_families, semantic_memory, program_coordinates = (
        _semantic_tilelang_projection(tir)
    )
    ptx_fingerprint = fingerprint_ptx(
        ptx,
        interface=interface,
        target=target,
        schedule=schedule,
    )
    payload = {
        "schema": TILELANG_FINGERPRINT_SCHEMA,
        "tir_ops": tir_ops,
        "tir_structure": tir_structure,
        "interface": interface_view,
        "semantic_value_families": semantic_families,
        "semantic_memory_effect_kinds": semantic_memory,
        "program_coordinates": program_coordinates,
        "ptx": ptx_fingerprint.digest,
    }
    return TileLangFingerprint(
        TILELANG_FINGERPRINT_SCHEMA,
        tir_ops,
        tuple(sorted(Counter(tir_ops).items())),
        tir_structure,
        interface_view,
        semantic_families,
        semantic_memory,
        program_coordinates,
        ptx_fingerprint,
        schedule_view,
        _digest(payload),
    )


@dataclass(frozen=True)
class TileLangDifference:
    category: str
    source: Any
    replay: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "source": self.source,
            "replay": self.replay,
            "message": self.message,
        }


@dataclass(frozen=True)
class TileLangComparison:
    hard_pass: bool
    hard_passes: Mapping[str, bool]
    diagnostic_passes: Mapping[str, bool]
    differences: tuple[TileLangDifference, ...]
    source_digest: str
    replay_digest: str

    @property
    def primary_difference_category(self) -> str | None:
        return next(
            (
                item.category
                for item in self.differences
                if item.category in self.hard_passes
                and not self.hard_passes[item.category]
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_pass": self.hard_pass,
            "hard_passes": dict(self.hard_passes),
            "diagnostic_passes": dict(self.diagnostic_passes),
            "primary_difference_category": self.primary_difference_category,
            "differences": [item.to_dict() for item in self.differences],
            "source_digest": self.source_digest,
            "replay_digest": self.replay_digest,
        }


def compare_tilelang_fingerprints(
    source: TileLangFingerprint,
    replay: TileLangFingerprint,
) -> TileLangComparison:
    ptx = compare_ptx_fingerprints(source.ptx, replay.ptx)
    base_fields = (
        ("schema", source.schema_version, replay.schema_version),
        ("interface", source.interface, replay.interface),
        ("tir_structure", source.tir_structure, replay.tir_structure),
        ("tir_topology", source.tir_ops, replay.tir_ops),
        ("tir_operations", source.tir_op_multiset, replay.tir_op_multiset),
    )
    passes = {category: expected == actual for category, expected, actual in base_fields}
    # PTX is below TileLang's frontend/compiler boundary.  Keep only its schema,
    # callable ABI and selected target as a hard safety envelope; exact backend
    # instruction spelling, multiplicity and topology remain auditable but
    # cannot veto an equal structural TIR program.
    ptx_hard_fields = {"schema", "interface", "target"}
    passes.update(
        {
            f"ptx.{key}": value
            for key, value in ptx.hard_passes.items()
            if key in ptx_hard_fields
        }
    )
    diagnostic_passes = {
        f"ptx.{key}": value
        for key, value in {**ptx.hard_passes, **ptx.diagnostic_passes}.items()
    }
    differences = [
        TileLangDifference(
            category,
            expected,
            actual,
            f"TileLang {category} differs between Source and Replay",
        )
        for category, expected, actual in base_fields
        if expected != actual
    ]
    differences.extend(
        TileLangDifference(
            f"ptx.{item.category}",
            item.source,
            item.replay,
            item.message,
        )
        for item in ptx.differences
    )
    return TileLangComparison(
        all(passes.values()),
        passes,
        diagnostic_passes,
        tuple(differences),
        source.digest,
        replay.digest,
    )


class TileLangComparator:
    def compare(
        self,
        source: TileLangFingerprint,
        replay: TileLangFingerprint,
    ) -> TileLangComparison:
        return compare_tilelang_fingerprints(source, replay)


__all__ = [
    "TILELANG_FINGERPRINT_SCHEMA",
    "TileLangComparator",
    "TileLangComparison",
    "TileLangDifference",
    "TileLangFingerprint",
    "TileLangObservationError",
    "compare_tilelang_fingerprints",
    "fingerprint_tilelang",
    "normalize_tilelang_tir_ops",
    "normalize_tilelang_tir_structure",
]
