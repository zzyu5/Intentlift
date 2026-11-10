"""Strict normalized MLIR/PTX topology observer for the CuTe frontend."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping


CUTE_FINGERPRINT_SCHEMA = "cute-4.4-mlir-ptx-topology-v1"
_SSA_RE = re.compile(r"%[A-Za-z0-9_.$-]+(?:#[0-9]+)?")
_SYMBOL_RE = re.compile(r"@[A-Za-z_.$][A-Za-z0-9_.$-]*")
_TYPE_ALIAS_RE = re.compile(r"![A-Za-z_][A-Za-z0-9_.$-]*(?=\s|=|,|\)|>|$)")
_OP_RE = re.compile(r'(?P<quoted>"[A-Za-z_][A-Za-z0-9_.-]*")|(?P<plain>[A-Za-z_][A-Za-z0-9_.-]*)')
_PTX_OPCODE_RE = re.compile(
    r"^(?:@!?%[A-Za-z0-9_.$-]+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]+)"
)


class CuteObservationError(ValueError):
    """The compiler artifact cannot support strict CuTe Gate1 comparison."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _balanced_end(text: str, start: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise CuteObservationError("unterminated MLIR location annotation")


def _strip_locations(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise CuteObservationError("CuTe MLIR is empty")
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#loc")
    )
    output: list[str] = []
    index = 0
    while index < len(text):
        if text.startswith("loc(", index):
            boundary = index == 0 or not (
                text[index - 1].isalnum() or text[index - 1] in "_.$"
            )
            if boundary:
                index = _balanced_end(text, index + 3) + 1
                continue
        output.append(text[index])
        index += 1
    return "".join(output)


def _leading_space(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _definition_tokens(line: str) -> set[str]:
    stripped = line.lstrip()
    definitions: set[str] = set()
    if stripped.startswith("%") and "=" in stripped:
        lhs = stripped.split("=", 1)[0]
        definitions.update(_SSA_RE.findall(lhs))
    if any(
        marker in stripped
        for marker in ("cuda.kernel @", "func.func @", "gpu.func @", "llvm.func @")
    ):
        opening = stripped.find("(")
        closing = stripped.find(")", opening + 1)
        if opening >= 0 and closing >= 0:
            definitions.update(_SSA_RE.findall(stripped[opening + 1 : closing]))
    if stripped.startswith("^") and "(" in stripped:
        definitions.update(_SSA_RE.findall(stripped.split("(", 1)[1].split(")", 1)[0]))
    match = re.search(r"\bscf\.for\s+(%[A-Za-z0-9_.$-]+)\s*=", stripped)
    if match:
        definitions.add(match.group(1))
    return definitions


def _alpha_ssa(lines: list[str]) -> list[str]:
    scopes: list[tuple[int, dict[str, str]]] = [(-1, {})]
    counter = 0

    def lookup(base: str) -> str | None:
        for _indent, values in reversed(scopes):
            if base in values:
                return values[base]
        return None

    normalized: list[str] = []
    for raw in lines:
        indent = _leading_space(raw)
        while len(scopes) > 1 and indent < scopes[-1][0]:
            scopes.pop()
        if indent > scopes[-1][0]:
            scopes.append((indent, {}))
        definitions = _definition_tokens(raw)
        mapping = scopes[-1][1]
        for token in sorted(definitions, key=raw.find):
            base = token.split("#", 1)[0]
            if base not in mapping:
                mapping[base] = f"%v{counter}"
                counter += 1

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            base, separator, suffix = token.partition("#")
            canonical = lookup(base)
            if canonical is None:
                nonlocal counter
                scopes[0][1][base] = f"%v{counter}"
                canonical = scopes[0][1][base]
                counter += 1
            return canonical + (separator + suffix if separator else "")

        normalized.append(_SSA_RE.sub(replace, raw))
    return normalized


def normalize_cute_mlir(mlir: str) -> str:
    """Remove presentation-only names while retaining exact dataflow text."""

    stripped = _strip_locations(mlir)
    raw_lines = [line.rstrip() for line in stripped.splitlines() if line.strip()]
    lines = _alpha_ssa(raw_lines)
    symbols: dict[str, str] = {}
    aliases: dict[str, str] = {}

    def symbol(match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in symbols:
            symbols[value] = f"@s{len(symbols)}"
        return symbols[value]

    def alias(match: re.Match[str]) -> str:
        value = match.group(0)
        if value.startswith(("!cute", "!llvm", "!gpu", "!cuda")):
            return value
        if value not in aliases:
            aliases[value] = f"!t{len(aliases)}"
        return aliases[value]

    result: list[str] = []
    for line in lines:
        line = _SYMBOL_RE.sub(symbol, line)
        line = _TYPE_ALIAS_RE.sub(alias, line)
        line = re.sub(r"[ \t]+", " ", line.strip())
        result.append(line)
    if not any("cuda.kernel" in line or "gpu.func" in line for line in result):
        raise CuteObservationError("CuTe MLIR contains no device kernel")
    return "\n".join(result) + "\n"


def _mlir_ops(normalized: str) -> tuple[str, ...]:
    operations: list[str] = []
    for line in normalized.splitlines():
        value = line
        if value.startswith(("!", "#", "module", "}")):
            continue
        if "=" in value and value.startswith("%"):
            value = value.split("=", 1)[1].lstrip()
        match = _OP_RE.match(value)
        if match is None:
            continue
        operation = (match.group("quoted") or match.group("plain")).strip('"')
        if "." in operation or operation in {"cuda", "gpu"}:
            operations.append(operation)
    if not operations:
        raise CuteObservationError("CuTe MLIR operation sequence is empty")
    return tuple(operations)


def _ptx_opcodes(ptx: str) -> tuple[str, ...]:
    if not isinstance(ptx, str) or not ptx.strip():
        raise CuteObservationError("CuTe PTX is empty")
    operations: list[str] = []
    for raw in ptx.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith((".", "{", "}", "$")) or line.endswith(":"):
            continue
        match = _PTX_OPCODE_RE.match(line)
        if match is not None:
            operations.append(match.group("opcode"))
    if not operations:
        raise CuteObservationError("CuTe PTX instruction sequence is empty")
    return tuple(operations)


@dataclass(frozen=True)
class CuteFingerprint:
    schema_version: str
    mlir_sha256: str
    mlir_ops: tuple[str, ...]
    mlir_op_multiset: tuple[tuple[str, int], ...]
    ptx_opcode_sha256: str
    ptx_opcodes: tuple[str, ...]
    async_operations: tuple[str, ...]
    python_interface: tuple[tuple[str, str], ...]
    schedule_view: tuple[tuple[str, str], ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mlir_sha256": self.mlir_sha256,
            "mlir_ops": list(self.mlir_ops),
            "mlir_op_multiset": [list(item) for item in self.mlir_op_multiset],
            "ptx_opcode_sha256": self.ptx_opcode_sha256,
            "ptx_opcodes": list(self.ptx_opcodes),
            "async_operations": list(self.async_operations),
            "python_interface": [list(item) for item in self.python_interface],
            "schedule_view": [list(item) for item in self.schedule_view],
            "digest": self.digest,
        }


def fingerprint_cute(
    mlir: str,
    ptx: str,
    *,
    python_interface: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> CuteFingerprint:
    normalized = normalize_cute_mlir(mlir)
    mlir_ops = _mlir_ops(normalized)
    ptx_opcodes = _ptx_opcodes(ptx)
    async_operations = tuple(
        operation
        for operation in ptx_opcodes
        if operation.startswith(("cp.async", "mbarrier", "fence", "barrier", "bar."))
    )
    if not async_operations:
        raise CuteObservationError(
            "CuTe strong observer requires explicit PTX async/barrier evidence"
        )
    interface_view = tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
            for key, value in (python_interface or {}).items()
        )
    )
    schedule_view = tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
            for key, value in (schedule or {}).items()
        )
    )
    op_multiset = tuple(sorted(Counter(mlir_ops).items()))
    payload = {
        "schema": CUTE_FINGERPRINT_SCHEMA,
        "mlir": hashlib.sha256(normalized.encode()).hexdigest(),
        "mlir_ops": mlir_ops,
        "ptx_opcodes": ptx_opcodes,
        "async": async_operations,
        "interface": interface_view,
        "schedule": schedule_view,
    }
    return CuteFingerprint(
        CUTE_FINGERPRINT_SCHEMA,
        payload["mlir"],
        mlir_ops,
        op_multiset,
        _digest(ptx_opcodes),
        ptx_opcodes,
        async_operations,
        interface_view,
        schedule_view,
        _digest(payload),
    )


@dataclass(frozen=True)
class CuteDifference:
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
class CuteComparison:
    hard_pass: bool
    differences: tuple[CuteDifference, ...]
    source_digest: str
    replay_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_pass": self.hard_pass,
            "differences": [item.to_dict() for item in self.differences],
            "source_digest": self.source_digest,
            "replay_digest": self.replay_digest,
        }


def compare_cute_fingerprints(
    source: CuteFingerprint, replay: CuteFingerprint
) -> CuteComparison:
    fields = (
        ("schema", source.schema_version, replay.schema_version),
        ("interface", source.python_interface, replay.python_interface),
        ("mlir_topology", source.mlir_sha256, replay.mlir_sha256),
        ("mlir_operations", source.mlir_ops, replay.mlir_ops),
        ("ptx_topology", source.ptx_opcode_sha256, replay.ptx_opcode_sha256),
        ("async_operations", source.async_operations, replay.async_operations),
    )
    differences = tuple(
        CuteDifference(
            category,
            expected,
            actual,
            f"CuTe {category} differs between source and replay",
        )
        for category, expected, actual in fields
        if expected != actual
    )
    return CuteComparison(
        not differences,
        differences,
        source.digest,
        replay.digest,
    )


class CuteComparator:
    def compare(
        self, source: CuteFingerprint, replay: CuteFingerprint
    ) -> CuteComparison:
        return compare_cute_fingerprints(source, replay)


__all__ = [
    "CUTE_FINGERPRINT_SCHEMA",
    "CuteComparator",
    "CuteComparison",
    "CuteDifference",
    "CuteFingerprint",
    "CuteObservationError",
    "compare_cute_fingerprints",
    "fingerprint_cute",
    "normalize_cute_mlir",
]
