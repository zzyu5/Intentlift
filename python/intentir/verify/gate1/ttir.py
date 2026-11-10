"""Deterministic semantic fingerprints for Triton 3.6 TTIR.

Gate1 compares two compilations produced by the *same* pinned Triton frontend.
It therefore does not need a general MLIR parser, but it must avoid comparing
unstable presentation details such as SSA names, debug locations, or the text
order of independent pure operations.  This module implements that deliberately
small boundary:

* parse the public ``tt.func`` runtime ABI and infer pointer access modes;
* build an SSA producer/consumer graph, including typed ``tt.reduce``/
  ``tt.scan`` regions and ``scf.for`` loop-carried values;
* canonicalize commutative operands and bounded comparison rewrites;
* record reductions, masked memory effects and fills, dots, loops, and effects;
* emit an immutable, JSON-serializable fingerprint.

The parser is intentionally strict.  An unresolved SSA value is a parse error,
not an invitation to silently produce a weaker fingerprint.  It targets the
custom assembly emitted by Triton 3.6 TTIR and also accepts quoted generic op
names (for example ``"tt.reduce"``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from intentir.ir.canonical_construction import build_scalar_owner_dag, build_tensor_owner_dag, owner_cast_kind, owner_float_bits
from .compiler import EXTERNAL_MATH_FAILURE_RULES, validated_external_math_symbols

__all__ = [
    "DotFingerprint",
    "InterfaceArgument",
    "LoopFingerprint",
    "MemoryEffect",
    "ReductionFingerprint",
    "RuntimeInterface",
    "StructuredIfArmFingerprint",
    "StructuredIfFingerprint",
    "TerminalForkArmFingerprint",
    "TerminalForkFingerprint",
    "TTIRFingerprint",
    "TTIR_FINGERPRINT_SCHEMA",
    "TTIRParseError",
    "TTIRParser",
    "diagnostic_source_locations",
    "extract_ttir_fingerprint",
    "fingerprint_ttir",
    "parse_ttir",
]


TTIR_FINGERPRINT_SCHEMA = "triton-ttir-3.6-gate1-v13"
_SCHEMA_VERSION = TTIR_FINGERPRINT_SCHEMA
_SSA_RE = re.compile(r"%[A-Za-z0-9_.$-]+(?:#[0-9]+)?")
_LOC_START_RE = re.compile(r"loc\s*\(")
_LOCATION_DEFINITION_RE = re.compile(
    r"(?m)^\s*(?P<name>#loc[A-Za-z0-9_.$-]*)\s*=\s*(?P<value>loc\s*\(.*)$"
)
_LOCATION_REFERENCE_RE = re.compile(r"#loc[A-Za-z0-9_.$-]*")
_LOCATION_COORDINATE_RE = re.compile(
    r'(?P<file>"(?:\\.|[^"\\])*")\s*:\s*(?P<line>[0-9]+)\s*:\s*(?P<column>[0-9]+)'
)
_SourceCallPath = tuple[tuple[int, int], ...]
_StatementSourceLocation = tuple[str, int, int, _SourceCallPath]
_OP_RE = re.compile(
    r'(?P<quoted>"[A-Za-z_][A-Za-z0-9_.-]*")|'
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_.-]*)"
)
_COMMUTATIVE_OPS = {
    "arith.addf",
    "arith.addi",
    "arith.andi",
    "arith.maxf",
    "arith.maxnumf",
    "arith.maximumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minf",
    "arith.minnumf",
    "arith.minimumf",
    "arith.minsi",
    "arith.minui",
    "arith.mulf",
    "arith.muli",
    "arith.ori",
    "arith.xori",
}
_REGION_OWNERS = {"tt.reduce", "tt.scan", "scf.for", "scf.if"}
_REGION_TERMINATORS = {"tt.reduce.return", "tt.scan.return", "scf.yield"}
_EXPECTED_REGION_TERMINATOR = {
    "tt.reduce": "tt.reduce.return",
    "tt.scan": "tt.scan.return",
    "scf.for": "scf.yield",
    "scf.if": "scf.yield",
}
_TERMINATORS = {"tt.return", *_REGION_TERMINATORS}
_KNOWN_PURE_OPS = {
    # Pinned pure Arith operations observed in the corpus or emitted by the
    # supported bounded normalizations.  Dialect prefixes are intentionally
    # not trusted: a future ``arith.*`` spelling is unsupported until it is
    # classified here.
    "arith.addf",
    "arith.addi",
    "arith.andi",
    "arith.bitcast",
    "arith.cmpf",
    "arith.cmpi",
    "arith.constant",
    "arith.divf",
    "arith.divsi",
    "arith.divui",
    "arith.extf",
    "arith.extsi",
    "arith.extui",
    "arith.floordivsi",
    "arith.fptosi",
    "arith.fptoui",
    "arith.index_cast",
    "arith.index_castui",
    "arith.maximumf",
    "arith.maxf",
    "arith.maxnumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minimumf",
    "arith.minf",
    "arith.minnumf",
    "arith.minsi",
    "arith.minui",
    "arith.mulf",
    "arith.muli",
    "arith.negf",
    "arith.ori",
    "arith.remsi",
    "arith.remui",
    "arith.select",
    "arith.shli",
    "arith.shrsi",
    "arith.shrui",
    "arith.sitofp",
    "arith.subf",
    "arith.subi",
    "arith.truncf",
    "arith.trunci",
    "arith.uitofp",
    "arith.xori",
    # Pinned pure Math operations.  Keep this explicit for the same reason.
    "math.absf",
    "math.absi",
    "math.acos",
    "math.asin",
    "math.atan",
    "math.atan2",
    "math.cbrt",
    "math.ceil",
    "math.copysign",
    "math.cos",
    "math.erf",
    "math.exp",
    "math.exp2",
    "math.expm1",
    "math.floor",
    "math.fma",
    "math.log",
    "math.log10",
    "math.log1p",
    "math.log2",
    "math.powf",
    "math.round",
    "math.roundeven",
    "math.rsqrt",
    "math.sin",
    "math.sqrt",
    "math.tan",
    "math.tanh",
    "math.trunc",
    # Structured owners are pure only when every operation in their nested
    # region is accepted by this same closed check.
    "scf.for",
    "scf.if",
    "tt.reduce",
    "tt.scan",
    "tt.return",
    *_REGION_TERMINATORS,
    # Triton value/address constructors.  Loads, stores, atomics,
    # descriptors, barriers, assertions, and other observable operations are
    # deliberately absent.
    "tt.addptr",
    "tt.advance",
    "tt.bitcast",
    "tt.broadcast",
    "tt.cat",
    "tt.clamp",
    "tt.dot",
    "tt.expand_dims",
    "tt.fp_to_fp",
    "tt.gather",
    "tt.get_num_programs",
    "tt.get_program_id",
    "tt.join",
    "tt.make_range",
    "tt.make_tensor_descriptor",
    "tt.precise_divf",
    "tt.precise_sqrt",
    "tt.reshape",
    "tt.split",
    "tt.splat",
    "tt.trans",
}
_KNOWN_EFFECT_OPS = {
    "gpu.barrier",
    "gpu.wait",
    "tt.assert",
    "tt.async_wait",
    "tt.atomic_cas",
    "tt.atomic_rmw",
    "tt.barrier",
    "tt.debug_barrier",
    "tt.descriptor_gather",
    "tt.descriptor_load",
    "tt.descriptor_reduce",
    "tt.descriptor_scatter",
    "tt.descriptor_store",
    "tt.device_assert",
    "tt.load",
    "tt.print",
    "tt.store",
}
_CMP_FLIPS = {
    "sgt": "slt",
    "sge": "sle",
    "ugt": "ult",
    "uge": "ule",
    "ogt": "olt",
    "oge": "ole",
}
_SYMMETRIC_CMP = {"eq", "ne", "oeq", "one", "ueq", "une"}


class TTIRParseError(ValueError):
    """The TTIR is outside the supported deterministic Gate1 subset."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*([,(){}\[\]<>])\s*", r"\1", text)
    text = re.sub(r"\s*:\s*", ":", text)
    text = re.sub(r"\s*=\s*", "=", text)
    return text


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    """Split while respecting MLIR's nested delimiters and quoted strings."""

    pieces: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == delimiter and not stack:
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def _balanced_end(text: str, start: int, opening: str, closing: str) -> int:
    if start >= len(text) or text[start] != opening:
        raise TTIRParseError(f"expected {opening!r} at offset {start}")
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise TTIRParseError(f"unterminated {opening!r} starting at offset {start}")


def _strip_locations(text: str) -> str:
    """Remove location definitions and balanced ``loc(...)`` annotations."""

    if not isinstance(text, str) or not text.strip():
        raise TTIRParseError("TTIR text is empty")
    # Keep physical lines aligned with the input so parser-owned operations
    # can carry their own provenance through SSA renaming and call expansion.
    text = "\n".join(
        "" if line.lstrip().startswith("#loc") else line
        for line in text.splitlines()
    )
    output: list[str] = []
    index = 0
    while index < len(text):
        match = _LOC_START_RE.match(text, index)
        boundary = index == 0 or not (text[index - 1].isalnum() or text[index - 1] in "_.$")
        if match and boundary:
            opening = match.end() - 1
            end = _balanced_end(text, opening, "(", ")") + 1
            output.append("\n" * text[index:end].count("\n"))
            index = end
            continue
        output.append(text[index])
        index += 1
    return "".join(output)


def _diagnostic_source_location_records(
    text: str,
    *,
    statement_locations: dict[int, _StatementSourceLocation] | None = None,
    source_file: str | None = None,
) -> tuple[tuple[str, tuple[str, ...], int, int], ...]:
    """Return narrow raw-TTIR location records for provenance only.

    The observer parser intentionally consumes TTIR with locations stripped,
    while Source authorization needs the compiler's original debug coordinate.
    Keep the bridge deliberately local: a record is emitted only for a direct
    operation statement with exactly one resolvable coordinate.  In
    particular, this does not turn a nearby location definition, a nested
    region close, or an operation sequence position into an identity.
    """

    if not isinstance(text, str) or not text.strip():
        raise TTIRParseError("TTIR text is empty")

    definitions: dict[str, str] = {}
    for match in _LOCATION_DEFINITION_RE.finditer(text):
        start = match.start("value")
        opening = text.find("(", start)
        if opening < 0:
            continue
        try:
            end = _balanced_end(text, opening, "(", ")")
        except TTIRParseError:
            continue
        definitions[match.group("name")] = text[start : end + 1]

    def location_tree(location: str, seen: frozenset[str] = frozenset()) -> tuple[str, int, int, tuple[tuple[str, int, int], ...]] | None:
        current = location.strip()
        if current in seen:
            return None
        seen = seen | {current}
        if _LOCATION_REFERENCE_RE.fullmatch(current):
            definition = definitions.get(current)
            return location_tree(definition, seen) if definition is not None else None
        if current.startswith("loc(") and _balanced_end(current, 3, "(", ")") == len(current) - 1:
            return location_tree(current[4:-1], seen)
        match = _LOCATION_COORDINATE_RE.fullmatch(current)
        if match is not None:
            return json.loads(match["file"]), int(match["line"]), int(match["column"]), ()
        if current.startswith("callsite(") and current.endswith(")"):
            payload = current[9:-1]
            depth = 0
            quoted = False
            escaped = False
            for index, character in enumerate(payload):
                if quoted:
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        quoted = False
                    continue
                if character == '"':
                    quoted = True
                elif character in "([<":
                    depth += 1
                elif character in ")]>":
                    depth -= 1
                elif depth == 0 and payload.startswith(" at ", index):
                    callee = location_tree(payload[:index], seen)
                    caller = location_tree(payload[index + 4:], seen)
                    if callee is None or caller is None:
                        return None
                    return (*callee[:3], (*caller[3], caller[:3], *callee[3]))
            return None
        named = re.fullmatch(r'"(?:\\.|[^"\\])*"\s*\((.*)\)', current, re.S)
        return location_tree(named[1], seen) if named is not None else None

    def coordinates(location: str) -> tuple[int, int, _SourceCallPath] | None:
        try:
            resolved = location_tree(location)
        except (ValueError, TTIRParseError, RecursionError):
            return None
        if resolved is None:
            return None
        filename, line, column, frames = resolved
        if source_file is not None and (filename != source_file or any(frame[0] != source_file for frame in frames)):
            return None
        return line, column, tuple((frame[1], frame[2]) for frame in frames)

    records: list[tuple[str, tuple[str, ...], int, int]] = []
    active_name: str | None = None
    active_results: tuple[str, ...] = ()
    active_text = ""
    recorded = False
    active_line = 0
    # A direct operation can span lines, and its location may be followed by
    # a type or attribute suffix.  The former implementation required
    # ``loc(...)`` to be the final character and then chose the last token
    # before it as the operation.  That confuses ordinary typed TTIR with
    # location definitions/attributes and silently drops valid provenance.
    # Track only the operation head of the current direct statement instead.
    for physical_line, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.lstrip()
        if stripped.startswith("#loc"):
            continue
        head = _operation_head(raw_line)
        if head is not None:
            active_results, active_name, _tail, _quoted = head
            active_line = physical_line
            active_text = raw_line
            recorded = False
        elif stripped.startswith(("}", "^", "tt.func", "module")):
            active_name = None
            continue
        elif active_name is not None and not recorded:
            active_text = (active_text + "\n" + raw_line)[-16384:]
        else:
            continue
        if recorded or active_name is None:
            continue
        candidates: list[tuple[int, int, _SourceCallPath]] = []
        incomplete = False
        for location_match in _LOC_START_RE.finditer(active_text):
            opening = active_text.find(
                "(", location_match.start(), location_match.end()
            )
            if opening < 0:
                incomplete = True
                continue
            try:
                end = _balanced_end(active_text, opening, "(", ")")
            except TTIRParseError:
                incomplete = True
                continue
            coordinate = coordinates(active_text[location_match.start() : end + 1])
            if coordinate is not None:
                candidates.append(coordinate)
        if incomplete or len(set(candidates)) != 1:
            continue
        line, column, call_path = candidates[0]
        if line <= 0 or column < 0:
            continue
        records.append((active_name, active_results, line, column))
        if statement_locations is not None:
            statement_locations[active_line] = (active_name, line, column, call_path)
        recorded = True
    return tuple(records)


def diagnostic_source_locations(text: str) -> tuple[tuple[str, int, int], ...]:
    """Return compiler-attested operation locations for Source provenance.

    Gate1 deliberately excludes debug locations from its equality and digest:
    locations are presentation metadata, not replay membership.  Source-aware
    construction nevertheless needs a *reject-only* bridge from an exact
    Source occurrence to an observed compiler operation.  This helper keeps
    that bridge ephemeral and conservative.  It returns a row only when an
    operation has exactly one resolvable file/line/column coordinate; missing,
    fused, callsite-only, or multiply located operations are intentionally
    absent rather than guessed.

    The result is never serialized by :class:`TTIRFingerprint`, exposed to a
    provider, or used to accept Gate1 membership.  Its consumers must bind it
    to an independently issued opaque Source authorization token.
    """

    return tuple(
        (operation, line, column)
        for operation, _results, line, column in _diagnostic_source_location_records(
            text
        )
    )


def _canonical_attrs(text: str) -> tuple[tuple[str, str], ...]:
    text = text.strip()
    if not text:
        return ()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    attrs: list[tuple[str, str]] = []
    for item in _split_top_level(text):
        if "=" in item:
            name, value = item.split("=", 1)
            attrs.append((_normalize_space(name), _normalize_space(value)))
        else:
            attrs.append((_normalize_space(item), ""))
    return tuple(sorted(attrs))


def _validate_attribute_dictionary(
    text: str, *, context: str
) -> tuple[tuple[str, str], ...]:
    """Parse one pinned bare-name attr dictionary and reject duplicates."""

    attrs = _canonical_attrs(text)
    seen: set[str] = set()
    for name, _value in attrs:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name) is None:
            raise TTIRParseError(
                f"{context} has malformed attribute name {name!r}"
            )
        if name in seen:
            raise TTIRParseError(
                f"{context} repeats attribute {name!r}"
            )
        seen.add(name)
    return attrs


_SCF_FOR_INTEGER_DTYPE_RE = r"(?:i|u)(?:1|8|16|32|64)"
_SCF_FOR_INTEGER_LITERAL_RE = re.compile(r"[+-]?[0-9]+")
_SCF_FOR_DENSE_INTEGER_RE = re.compile(
    r"dense<(?P<values>.*)>:tensor<"
    r"(?P<shape>[1-9][0-9]*(?:x[1-9][0-9]*)*)x"
    rf"(?P<dtype>{_SCF_FOR_INTEGER_DTYPE_RE})>"
)


def _flatten_scf_for_integer_list(text: str) -> list[str] | None:
    """Flatten a dense integer literal without accepting arbitrary syntax."""

    stripped = text.strip()
    if _SCF_FOR_INTEGER_LITERAL_RE.fullmatch(stripped) is not None:
        return [stripped]
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    items = _split_top_level(stripped[1:-1])
    if not items:
        return None
    flattened: list[str] = []
    for item in items:
        nested = _flatten_scf_for_integer_list(item)
        if nested is None:
            return None
        flattened.extend(nested)
    return flattened


def _validate_scf_for_closing_attributes(
    attrs: Sequence[tuple[str, str]], *, context: str
) -> tuple[tuple[str, str], ...]:
    """Validate the small set of Triton compiler hints emitted after ``scf.for``.

    These attributes are compiler metadata rather than J-visible computation.
    Keep the parser strict, but do not reject real Triton divisibility hints.
    """

    validated: list[tuple[str, str]] = []
    for name, value in attrs:
        if name == "tt.num_stages":
            if re.fullmatch(r"[1-9][0-9]*:i32", value) is None:
                raise TTIRParseError(
                    f"{context} has invalid tt.num_stages value {value!r}"
                )
            validated.append((name, value))
            continue

        if re.fullmatch(r"tt\.divisibility_arg[0-9]+", name) is None:
            raise TTIRParseError(
                f"{context} has unsupported closing attribute {name!r}"
            )
        match = _SCF_FOR_DENSE_INTEGER_RE.fullmatch(value)
        if match is None:
            raise TTIRParseError(
                f"{context} has invalid {name} value {value!r}"
            )
        values = _flatten_scf_for_integer_list(match.group("values"))
        if values is None:
            raise TTIRParseError(
                f"{context} has non-integer dense values for {name!r}"
            )
        shape = tuple(int(item) for item in match.group("shape").split("x"))
        element_count = 1
        for extent in shape:
            element_count *= extent
        # A scalar dense literal is an MLIR splat; a bracketed literal must
        # provide exactly one integer for every static tensor element.
        if len(values) != 1 and len(values) != element_count:
            raise TTIRParseError(
                f"{context} dense values for {name!r} do not match tensor shape"
            )
        validated.append((name, value))
    return tuple(validated)


def _pointer_dtype(type_text: str) -> str | None:
    match = re.search(r"!tt\.ptr<([^,>]+)", type_text)
    return _normalize_space(match.group(1)) if match else None


@dataclass(frozen=True)
class InterfaceArgument:
    """One positional runtime argument; SSA spelling is deliberately omitted."""

    position: int
    type: str
    kind: str
    dtype: str | None
    rank: int | None
    access: str
    attributes: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "type": self.type,
            "kind": self.kind,
            "dtype": self.dtype,
            "rank": self.rank,
            "access": self.access,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class RuntimeInterface:
    arguments: tuple[InterfaceArgument, ...]
    results: tuple[str, ...] = ()
    python: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arguments": [argument.to_dict() for argument in self.arguments],
            "results": list(self.results),
            "python": {key: json.loads(value) for key, value in self.python},
        }


@dataclass(frozen=True)
class ReductionFingerprint:
    axis: int
    combiners: tuple[str, ...]
    input_count: int
    result_count: int

    @property
    def combiner(self) -> str | None:
        return self.combiners[0] if len(self.combiners) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "combiners": list(self.combiners),
            "input_count": self.input_count,
            "result_count": self.result_count,
        }


@dataclass(frozen=True)
class MemoryEffect:
    kind: str
    masked: bool
    predicate: str | None = None
    other: str | None = None
    attributes: str = ""
    # Compiler-local provenance for the Source-aware diagnostic transfer
    # sidecar. It is excluded from equality and the public fingerprint so
    # debug/call-expansion details cannot affect Gate1.
    source_call_path: _SourceCallPath = field(
        default=(), repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "masked": self.masked,
            "predicate": self.predicate,
            "other": self.other,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class DotFingerprint:
    operation: str
    operand_count: int
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "operand_count": self.operand_count,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class LoopFingerprint:
    lower: str
    upper: str
    step: str
    init_values: tuple[str, ...]
    yielded_values: tuple[str, ...]
    body_effects: tuple[tuple[str, int], ...]
    block_argument_count: int
    induction_use_count: int
    carried_argument_use_counts: tuple[int, ...]
    captured_values: tuple[str, ...]
    body_op_multiset: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "step": self.step,
            "init_values": list(self.init_values),
            "yielded_values": list(self.yielded_values),
            "body_effects": dict(self.body_effects),
            "block_argument_count": self.block_argument_count,
            "induction_use_count": self.induction_use_count,
            "carried_argument_use_counts": list(
                self.carried_argument_use_counts
            ),
            "captured_values": list(self.captured_values),
            "body_op_multiset": dict(self.body_op_multiset),
        }


@dataclass(frozen=True)
class StructuredIfArmFingerprint:
    """One polarity-preserving arm of a continuing effect-only ``scf.if``."""

    role: str
    members: tuple[str, ...]
    effect_order: tuple[str, ...]
    structured_regions: tuple[str, ...]
    yielded_value: str | None = None
    yield_type: str | None = None
    terminal: str = "scf.yield:void"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "members": list(self.members),
            "effect_order": list(self.effect_order),
            "structured_regions": list(self.structured_regions),
            "yielded_value": self.yielded_value,
            "yield_type": self.yield_type,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class StructuredIfFingerprint:
    """Hard evidence for one scalar, result-free conditional continuation."""

    ordinal: int
    predicate: str
    region_context: tuple[str, ...]
    enclosing_terminal_arm: str | None
    result_type: str | None
    positive_arm: StructuredIfArmFingerprint
    negative_arm: StructuredIfArmFingerprint
    continuation: str = "parent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "predicate": self.predicate,
            "region_context": list(self.region_context),
            "enclosing_terminal_arm": self.enclosing_terminal_arm,
            "result_type": self.result_type,
            "positive_arm": self.positive_arm.to_dict(),
            "negative_arm": self.negative_arm.to_dict(),
            "continuation": self.continuation,
        }


@dataclass(frozen=True)
class TerminalForkArmFingerprint:
    """One role-preserving arm of the closed terminal binary-fork profile."""

    role: str
    members: tuple[str, ...]
    effect_order: tuple[str, ...]
    structured_regions: tuple[str, ...]
    terminal: str = "tt.return:void"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "members": list(self.members),
            "effect_order": list(self.effect_order),
            "structured_regions": list(self.structured_regions),
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class TerminalForkFingerprint:
    """Hard evidence for one non-merging scalar conditional terminal fork."""

    predicate: str
    region_context: tuple[str, ...]
    true_arm: TerminalForkArmFingerprint
    false_arm: TerminalForkArmFingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "region_context": list(self.region_context),
            "true_arm": self.true_arm.to_dict(),
            "false_arm": self.false_arm.to_dict(),
        }


@dataclass(frozen=True)
class TTIRFingerprint:
    """Immutable Gate1 fingerprint.  All sequence fields are canonically sorted."""

    interface: RuntimeInterface
    op_multiset: tuple[tuple[str, int], ...]
    dataflow_edges: tuple[tuple[str, str, str], ...]
    reductions: tuple[ReductionFingerprint, ...]
    memory_effects: tuple[MemoryEffect, ...]
    mask_predicates: tuple[str, ...]
    effect_multiset: tuple[tuple[str, int], ...]
    dots: tuple[DotFingerprint, ...]
    loops: tuple[LoopFingerprint, ...]
    structured_ifs: tuple[StructuredIfFingerprint, ...] = ()
    terminal_forks: tuple[TerminalForkFingerprint, ...] = ()
    schedule_view: tuple[tuple[str, str], ...] = ()
    schema_version: str = _SCHEMA_VERSION
    # Diagnostic-only normalized node descriptions.  They are deliberately
    # excluded from equality, ``to_dict`` and ``digest``: Gate1 membership is
    # still owned solely by the versioned fields above.  SCRC may use this
    # ephemeral sidecar to explain a failed edge without exposing SSA names.
    diagnostic_node_signatures: tuple[tuple[str, str], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Diagnostic-only, non-recursive node profiles.  Recursive semantic node
    # hashes are required by the hard dataflow observer, but one upstream
    # mutation necessarily re-hashes every downstream consumer.  These local
    # profiles retain only the immediate producer signatures, operand roles,
    # lexical region roles.  SCRC can therefore
    # report a bounded divergence frontier without weakening or changing the
    # versioned membership relation above.
    diagnostic_node_profiles: tuple[
        tuple[
            str,
            str,
            tuple[tuple[str, str], ...],
            tuple[str, ...],
        ],
        ...,
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Diagnostic-only expansion of representation carriers contracted by the
    # v12 hard observer.  It exposes only the carrier result contract and its
    # normalized anonymous origins, never the discarded arithmetic formula.
    # This lets source-aware Lift repair an authored route whose dependency set
    # was incomplete without turning Gate1 feedback into a hidden Source trace.
    diagnostic_representation_profiles: tuple[
        tuple[str, str, tuple[str, ...]], ...
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Diagnostic-only closed classification of compiler-visible ``scf.for``
    # bounds.  Each row is ``(block_arity, basis, extent_handle, tile)``.
    # It is intentionally excluded from equality, ``to_dict`` and ``digest``;
    # source-aware Lift and source-blind Replay preflight may only reject an
    # authored induction basis that contradicts the compiled loop.
    diagnostic_iteration_bases: tuple[
        tuple[int, str, str, int], ...
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Diagnostic-only expansion of each synthetic ``effect.trace`` node.
    # Keeping the ordered event spellings beside the opaque trace digest lets
    # SCRC explain *which* lexical effect sequence diverged.  Like node
    # signatures, this sidecar is excluded from membership, serialization and
    # the digest; it can diagnose a rejection but can never accept a candidate.
    diagnostic_effect_traces: tuple[
        tuple[str, str, tuple[str, ...]], ...
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Diagnostic-only lexical memory transfers.  Unlike ``memory_effects``,
    # which is canonically sorted for Gate1 membership, this sidecar retains
    # the one-to-one order shared by the provenance and address sidecars
    # below.  It is source-aware ingress evidence only: it is not serialized,
    # compared, or included in the Gate1 digest.
    diagnostic_memory_transfers: tuple[MemoryEffect, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Diagnostic-only lexical memory events with the function-argument
    # positions that may supply each pointer operand.  Replay's source-blind
    # compile preflight uses this sidecar to check ordering already authored in
    # pi.  It is deliberately excluded from equality, serialization and the
    # digest, so it cannot change Gate1 membership.
    diagnostic_memory_effect_origins: tuple[
        tuple[str, tuple[int, ...], tuple[str, ...]], ...
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Diagnostic-only normalized pointer-address forms for source-aware Lift.
    # Each row is ``(effect_kind, pointer_base, polynomial, context,
    # element_type, typed_offset_layers)`` where ``polynomial`` contains only
    # anonymous argument/program/lane atoms. Every typed layer retains one
    # nonzero ``tt.addptr`` offset polynomial, its explicit integer/index type,
    # and the bounded SSA type-propagation chain ending at that offset. A
    # direct zero-offset base pointer has no layers. The
    # forms are deliberately excluded from equality, serialization, and the
    # digest: they can reject an impossible source-bound axis claim, but they
    # can never weaken or alter Gate1 membership.
    diagnostic_memory_address_forms: tuple[
        tuple[
            str,
            str,
            tuple[tuple[tuple[str, ...], int], ...],
            tuple[str, ...],
            str | None,
            tuple[
                tuple[
                    tuple[tuple[tuple[str, ...], int], ...],
                    str | None,
                    tuple[str, ...],
                ],
                ...,
            ],
                _SourceCallPath,
        ],
        ...,
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Compiler-attested Source locations for the narrow provenance bridge used
    # by Source-aware Lift.  As with the other diagnostic sidecars this is
    # excluded from equality, serialization and digest: a debug-location
    # change must never alter Gate1 membership.  Ledger construction consumes
    # it only to reject a requested semantic-owner/guard authorization when a
    # unique Source-to-compiler counterpart is unavailable.
    diagnostic_source_locations: tuple[tuple[str, int, int], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # One compiler-visible source coordinate and call path per memory operation,
    # retained with its transfer kind and mask bit from the same parsed node.
    # This is a reject-only provenance sidecar: Source construction may bind a
    # lexical memory call only when its own exact coordinate has one compatible
    # row here.  It is excluded from equality, serialization, and digest just
    # like the other diagnostic sidecars.
    diagnostic_memory_source_locations: tuple[
        tuple[str, bool, int, int, _SourceCallPath], ...
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Exact compiler-local predicate use for each direct memory operation.
    # Rows are ``(kind, masked, access_line, access_column, predicate_op,
    # predicate_line, predicate_column, predicate_is_i1, predicate_call_path,
    # predicate_producer_id, access_call_path)``. The final two fields are
    # compiler-local identities
    # used only to prove that two memory accesses consume the same SSA value.
    # The producer path must match the exact Source invocation. A producer may
    # be an exact direct SSA operand of the matched memory operation while
    # lacking its own compiler debug coordinate; in that case predicate_op
    # and predicate_is_i1 remain present but both predicate coordinates are
    # null.  This is still not a wildcard: consumers must bind it through the
    # same exact memory access and an independently proven Source def-use
    # edge.  This remains an internal reject-only sidecar and never changes
    # Gate1 membership.
    diagnostic_memory_predicate_evidence: tuple[
        tuple[str, bool, int, int, str | None, int | None, int | None, bool, _SourceCallPath | None, int | None, _SourceCallPath], ...
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Exact compiler-local value edge for each direct memory operation.  Rows
    # are ``(kind, masked, access_line, access_column, owner_op, owner_line,
    # owner_column, direct_edge, access_call_path)``.  For a load the owner is its sole direct
    # SSA user; for a store it is the direct producer of the stored value.  A
    # standalone owner location can be absent after lowering, but a direct
    # edge still belongs only to the already exact matched memory access.
    # This reject-only sidecar is excluded from equality, serialization, and
    # digest. The final access_call_path prevents two call-expanded memory
    # events with the same line/column from being interchanged.
    diagnostic_memory_owner_evidence: tuple[
        tuple[str, bool, int, int, str | None, int | None, int | None, bool, _SourceCallPath], ...
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    # Host-only, pre-normalization typed scalar producer DAGs. No raw Source,
    # TTIR, coordinates, ABI names or DAGs enter serialization/Gate1 digest.
    diagnostic_scalar_owner_graphs: tuple[Mapping[str, Any], ...] = field(
        default=(), repr=False, compare=False,
    )
    diagnostic_tensor_owner_graphs: tuple[Mapping[str, Any], ...] = field(
        default=(), repr=False, compare=False,
    )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": self.interface.to_dict(),
            "op_multiset": dict(self.op_multiset),
            "dataflow_edges": [list(edge) for edge in self.dataflow_edges],
            "reductions": [item.to_dict() for item in self.reductions],
            "memory_effects": [item.to_dict() for item in self.memory_effects],
            "mask_predicates": list(self.mask_predicates),
            "effect_multiset": dict(self.effect_multiset),
            "dots": [item.to_dict() for item in self.dots],
            "loops": [item.to_dict() for item in self.loops],
            "structured_ifs": [
                item.to_dict() for item in self.structured_ifs
            ],
            "terminal_forks": [
                item.to_dict() for item in self.terminal_forks
            ],
            "schedule_view": {key: json.loads(value) for key, value in self.schedule_view},
        }


@dataclass
class _Operation:
    node_id: int
    name: str
    result_names: tuple[str, ...]
    operands: list[str]
    local_signature: str
    context: tuple[int, ...]
    result_types: tuple[str, ...] = ()
    axis: int | None = None
    reverse: bool | None = None
    cmp_predicate: str | None = None
    cmp_flipped: bool = False
    header_operand_count: int = 0
    region_results: list[str] = field(default_factory=list)
    region_block_args: list[str] = field(default_factory=list)
    constant_value: float | None = None
    callee: str | None = None
    call_operand_types: tuple[str, ...] = ()
    call_result_types: tuple[str, ...] = ()
    terminator_types: tuple[str, ...] = ()
    fork_id: int | None = None
    fork_arm: str | None = None
    structured_if_id: int | None = None
    structured_if_arm: str | None = None
    structured_if_result_type: str | None = None
    # Host-only evidence bound while parsing this exact statement. Never
    # included in local signatures, graph identities or Gate1 digests.
    source_location: tuple[str, int, int] | None = field(default=None, repr=False, compare=False)
    source_call_path: _SourceCallPath = field(default=(), repr=False, compare=False)


@dataclass
class _TerminalFork:
    fork_id: int
    predicate: str
    context: tuple[int, ...]
    sequence_index: int


@dataclass(frozen=True)
class _FunctionParts:
    name: str
    visibility: str
    arguments: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]
    results: tuple[str, ...]
    body: str
    attributes: tuple[tuple[str, str], ...] = ()
    source_locations: Mapping[int, _StatementSourceLocation] = field(default_factory=dict, repr=False, compare=False)


def _owner_runtime_parameters(
    function: _FunctionParts, interface: Mapping[str, Any] | None,
    *, ttir: str,
) -> tuple[str, ...] | None:
    """Consume named compiler specialization slots, never infer lost scalars."""
    if not isinstance(interface, Mapping):
        return None
    names, constexpr = interface.get("arg_names"), interface.get("constexpr_indices")
    if (
        not isinstance(names, list) or any(not isinstance(name, str) for name in names)
        or len(set(names)) != len(names) or not isinstance(constexpr, list)
        or any(type(index) is not int or not 0 <= index < len(names) for index in constexpr)
        or len(set(constexpr)) != len(constexpr)
        or interface.get("constexprs") != [names[index] for index in constexpr]
    ):
        return None
    runtime = interface.get("compiled_arg_names")
    specialized = interface.get("specialized_arg_names")
    if (
        interface.get("compiled_ttir_sha256") != hashlib.sha256(ttir.encode("utf-8")).hexdigest()
        or not isinstance(runtime, list) or not isinstance(specialized, list)
        or any(not isinstance(name, str) for name in (*runtime, *specialized))
        or len(set(runtime)) != len(runtime) or len(set(specialized)) != len(specialized)
        or set(runtime) & set(specialized) or set(runtime) | set(specialized) != set(names)
        or runtime != [name for name in names if name not in specialized]
        or not {names[index] for index in constexpr}.issubset(specialized)
        or len(runtime) != len(function.arguments)
    ):
        return None
    return tuple(runtime)


def _diagnostic_iteration_bases(
    function: _FunctionParts,
    operations: list[_Operation],
    result_to_node: Mapping[str, int],
    leaf_labels: Mapping[str, str],
) -> tuple[tuple[int, str, str, int], ...]:
    """Classify only the three closed pi induction bases from exact TTIR SSA.

    The classifier recognizes direct zero-to-extent tile steps and the exact
    ceil-div forms used by ``rounded_extent_step`` and ``tile_ordinal``.  It
    deliberately returns no row for an ambiguous or different expression.
    """

    nodes = {operation.node_id: operation for operation in operations}
    arguments: dict[str, str] = {}
    for name, _type, _attrs in function.arguments:
        label = leaf_labels.get(name)
        match = (
            re.fullmatch(r"arg\[([0-9]+)\]:.*", label)
            if isinstance(label, str)
            else None
        )
        if match is not None:
            # Publish the pre-specialization physical ABI handle, never the
            # compiler's compacted argN ordinal. Ambiguous scalar aliases are
            # deliberately labelled ``unmapped*`` upstream and yield no row.
            arguments[name] = f"p{match.group(1)}"
    passthrough = {
        "arith.extsi", "arith.extui", "arith.index_cast", "arith.index_castui",
        "arith.trunci",
    }

    def producer(value: str) -> _Operation | None:
        node_id = result_to_node.get(value)
        return nodes.get(node_id) if node_id is not None else None

    def integer_constant(value: str, active: frozenset[str] = frozenset()) -> int | None:
        if value in active:
            return None
        operation = producer(value)
        if operation is None:
            return None
        if operation.name in passthrough and len(operation.operands) == 1:
            return integer_constant(operation.operands[0], active | {value})
        constant = operation.constant_value
        if (
            operation.name != "arith.constant"
            or constant is None
            or not float(constant).is_integer()
        ):
            return None
        return int(constant)

    def argument_index(
        value: str, active: frozenset[str] = frozenset()
    ) -> str | None:
        if value in arguments:
            return arguments[value]
        if value in active:
            return None
        operation = producer(value)
        if operation is None or operation.name not in passthrough:
            return None
        if len(operation.operands) != 1:
            return None
        return argument_index(operation.operands[0], active | {value})

    def ceil_div_extent(
        value: str, tile: int, active: frozenset[str] = frozenset()
    ) -> str | None:
        if value in active:
            return None
        operation = producer(value)
        if operation is None:
            return None
        if operation.name in {"arith.ceildivsi", "arith.ceildivui"}:
            if len(operation.operands) != 2:
                return None
            if integer_constant(operation.operands[1]) != tile:
                return None
            return argument_index(operation.operands[0])
        if operation.name not in {"arith.divsi", "arith.divui", "arith.floordivsi"}:
            return None
        if len(operation.operands) != 2 or integer_constant(operation.operands[1]) != tile:
            return None
        numerator = producer(operation.operands[0])
        if numerator is None:
            return None
        if numerator.name == "arith.addi" and len(numerator.operands) == 2:
            for extent, offset in (
                (numerator.operands[0], numerator.operands[1]),
                (numerator.operands[1], numerator.operands[0]),
            ):
                if integer_constant(offset) == tile - 1:
                    index = argument_index(extent)
                    if index is not None:
                        return index
        if numerator.name == "arith.subi" and len(numerator.operands) == 2:
            if integer_constant(numerator.operands[1]) != 1:
                return None
            summed = producer(numerator.operands[0])
            if summed is not None and summed.name == "arith.addi" and len(summed.operands) == 2:
                for extent, offset in (
                    (summed.operands[0], summed.operands[1]),
                    (summed.operands[1], summed.operands[0]),
                ):
                    if integer_constant(offset) == tile:
                        index = argument_index(extent)
                        if index is not None:
                            return index
        return None

    def rounded_extent(value: str, tile: int) -> str | None:
        operation = producer(value)
        if operation is None or operation.name != "arith.muli":
            return None
        if len(operation.operands) != 2:
            return None
        for quotient, factor in (
            (operation.operands[0], operation.operands[1]),
            (operation.operands[1], operation.operands[0]),
        ):
            if integer_constant(factor) == tile:
                index = ceil_div_extent(quotient, tile)
                if index is not None:
                    return index
        return None

    result: list[tuple[int, str, str, int]] = []
    for loop in operations:
        if loop.name != "scf.for" or loop.header_operand_count < 3:
            continue
        lower, upper, step = loop.operands[:3]
        if integer_constant(lower) != 0:
            continue
        step_value = integer_constant(step)
        if step_value is None or step_value <= 0:
            continue
        extent = argument_index(upper)
        if extent is not None:
            result.append(
                (len(loop.region_block_args), "extent_step", extent, step_value)
            )
            continue
        if step_value > 1:
            extent = rounded_extent(upper, step_value)
            if extent is not None:
                result.append(
                    (
                        len(loop.region_block_args),
                        "rounded_extent_step",
                        extent,
                        step_value,
                    )
                )
                continue
        if step_value == 1:
            upper_operation = producer(upper)
            if upper_operation is not None and upper_operation.name in {
                "arith.ceildivsi", "arith.ceildivui", "arith.divsi",
                "arith.divui", "arith.floordivsi",
            } and len(upper_operation.operands) == 2:
                tile = integer_constant(upper_operation.operands[1])
                if tile is not None and tile > 1:
                    extent = ceil_div_extent(upper, tile)
                    if extent is not None:
                        result.append(
                            (
                                len(loop.region_block_args),
                                "tile_ordinal",
                                extent,
                                tile,
                            )
                        )
    return tuple(sorted(result))


_FUNCTION_RE = re.compile(
    r"^\s*tt\.func\s+(?:(?P<visibility>public|private)\s+)?@"
    r'(?P<name>"(?:\\.|[^"\\])*"|[A-Za-z0-9_.$-]+)\s*\(',
    flags=re.MULTILINE,
)
_SYMBOL_RE = r'"(?:\\.|[^"\\])*"|[A-Za-z0-9_.$-]+'


def _symbol_value(token: str) -> str:
    if not token.startswith('"'):
        return token
    try:
        value = json.loads(token)
    except json.JSONDecodeError as error:
        raise TTIRParseError(f"invalid quoted MLIR symbol {token!r}") from error
    if not isinstance(value, str):
        raise TTIRParseError(f"invalid quoted MLIR symbol {token!r}")
    return value


def _parse_function_definition(
    text: str, match: re.Match[str],
    statement_locations: Mapping[int, _StatementSourceLocation] | None = None,
) -> _FunctionParts:
    args_open = match.end() - 1
    args_close = _balanced_end(text, args_open, "(", ")")
    args_text = text[args_open + 1 : args_close]

    arguments: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
    for item in _split_top_level(args_text):
        argument_match = re.match(
            r"(?P<name>%[A-Za-z0-9_.$-]+)\s*:\s*(?P<rest>.*)",
            item,
            re.S,
        )
        if argument_match is None:
            raise TTIRParseError(f"cannot parse function argument: {item!r}")
        rest = argument_match.group("rest").strip()
        attr_start = _find_top_level_char(rest, "{")
        if attr_start is None:
            type_text, attrs = rest, ()
        else:
            attr_end = _balanced_end(rest, attr_start, "{", "}")
            type_text = rest[:attr_start]
            attrs = _canonical_attrs(rest[attr_start : attr_end + 1])
        arguments.append(
            (
                argument_match.group("name"),
                _validate_closed_type(
                    type_text,
                    context=f"tt.func argument {argument_match.group('name')}",
                ),
                attrs,
            )
        )

    # The first brace after the argument list can be the function attribute
    # dictionary.  ``_next_body_brace`` skips it and returns the actual body.
    body_open = _next_body_brace(text, args_close + 1)
    between = text[args_close + 1 : body_open]
    attribute_marker = re.search(r"\battributes\s*\{", between)
    function_attributes: tuple[tuple[str, str], ...] = ()
    if attribute_marker is not None:
        attribute_open = between.find("{", attribute_marker.start())
        attribute_close = _balanced_end(
            between, attribute_open, "{", "}"
        )
        if between[attribute_close + 1 :].strip():
            raise TTIRParseError(
                "unexpected syntax after tt.func attribute dictionary"
            )
        function_attributes = _canonical_attrs(
            between[attribute_open : attribute_close + 1]
        )
        between = between[: attribute_marker.start()]
    body_close = _balanced_end(text, body_open, "{", "}")

    result_text = between.strip()
    if result_text.startswith("->"):
        result_text = result_text[2:].strip()
        if result_text.startswith("(") and result_text.endswith(")"):
            results = tuple(
                _validate_closed_type(
                    item,
                    context=f"tt.func @{_symbol_value(match.group('name'))} result",
                )
                for item in _split_top_level(result_text[1:-1])
            )
        else:
            results = (
                _validate_closed_type(
                    result_text,
                    context=f"tt.func @{_symbol_value(match.group('name'))} result",
                ),
            )
    else:
        results = ()
    return _FunctionParts(
        name=_symbol_value(match.group("name")),
        visibility=match.group("visibility") or "private",
        arguments=tuple(arguments),
        results=results,
        body=text[body_open + 1 : body_close],
        attributes=function_attributes,
        source_locations={
            line - text.count("\n", 0, body_open + 1): location
            for line, location in (statement_locations or {}).items()
            if text.count("\n", 0, body_open + 1) < line
            <= text.count("\n", 0, body_close) + 1
        },
    )


def _parse_functions(
    text: str,
    statement_locations: Mapping[int, _StatementSourceLocation] | None = None,
) -> dict[str, _FunctionParts]:
    functions: dict[str, _FunctionParts] = {}
    for match in _FUNCTION_RE.finditer(text):
        function = _parse_function_definition(text, match, statement_locations)
        if function.name in functions:
            raise TTIRParseError(f"duplicate tt.func symbol @{function.name}")
        functions[function.name] = function
    public = [
        function
        for function in functions.values()
        if function.visibility == "public"
    ]
    if len(public) != 1:
        raise TTIRParseError(
            f"expected exactly one public tt.func, found {len(public)}"
        )
    return functions


def _parse_function(text: str) -> _FunctionParts:
    functions = _parse_functions(text)
    return next(
        function for function in functions.values() if function.visibility == "public"
    )


def _validate_function_metadata_boundary(function: _FunctionParts) -> None:
    """Validate the closed function-metadata boundary for TTIR v7."""

    if function.visibility != "public":
        for position, (_, _, attributes) in enumerate(function.arguments):
            if attributes:
                raise TTIRParseError(
                    f"reachable private tt.func @{function.name} argument "
                    f"{position} has unsupported attributes"
                )
    seen: set[str] = set()
    for name, value in function.attributes:
        if name in seen:
            raise TTIRParseError(
                f"reachable tt.func @{function.name} repeats "
                f"attribute {name!r}"
            )
        seen.add(name)
        if name != "noinline" or value not in {"false", "true"}:
            raise TTIRParseError(
                f"reachable tt.func @{function.name} has unsupported "
                f"function attribute {name!r}={value!r}"
            )


def _find_top_level_char(text: str, wanted: str) -> int | None:
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "<": ">"}
    quote: str | None = None
    for index, char in enumerate(text):
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == wanted and not stack:
            return index
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
    return None


def _next_body_brace(text: str, start: int) -> int:
    cursor = start
    while cursor < len(text):
        if text.startswith("attributes", cursor):
            attr_open = text.find("{", cursor)
            cursor = _balanced_end(text, attr_open, "{", "}") + 1
            continue
        if text[cursor] == "{":
            return cursor
        cursor += 1
    raise TTIRParseError("public tt.func has no body")


def _parse_results(lhs: str) -> tuple[str, ...]:
    results: list[str] = []
    for item in _split_top_level(lhs):
        item = item.strip()
        group = re.fullmatch(r"(?P<base>%[A-Za-z0-9_.$-]+):(?P<count>[0-9]+)", item)
        if group:
            base = group.group("base")
            count = int(group.group("count"))
            results.extend(f"{base}#{index}" for index in range(count))
        elif _SSA_RE.fullmatch(item):
            results.append(item)
        else:
            raise TTIRParseError(f"cannot parse operation result: {item!r}")
    return tuple(results)


def _operation_head(
    line: str,
) -> tuple[tuple[str, ...], str, str, bool] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(("^", "}", ")", "module", "tt.func")):
        return None
    lhs: tuple[str, ...] = ()
    tail = stripped
    assignment = re.match(r"(?P<lhs>%[^=]+?)\s*=\s*(?P<tail>.*)$", stripped)
    if assignment:
        lhs = _parse_results(assignment.group("lhs"))
        tail = assignment.group("tail").strip()
    match = _OP_RE.match(tail)
    if match is None:
        return None
    token = match.group("quoted") or match.group("plain")
    assert token is not None
    name = token.strip('"')
    if "." not in name:
        return None
    return lhs, name, tail[match.end() :].strip(), token.startswith('"')


def _validate_quoted_generic_shell(
    name: str,
    tail: str,
    *,
    result_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate the closed, region-free MLIR generic-operation shell.

    Gate1 validates syntax before applying its closed operation/effect
    classification.  Merely recognizing ``"dialect.op"`` is insufficient:
    accepting arbitrary trailing prose would make malformed TTIR
    indistinguishable from a well-formed but observer-unsupported operation.

    Region owners are parsed by their dedicated structured parsers, so this
    helper intentionally accepts only the region-free spelling
    ``(%operands) [<{properties}>] [{attributes}] : inputs -> results``.
    """

    stripped = tail.strip()
    if not stripped.startswith("("):
        raise TTIRParseError(
            f'quoted generic operation "{name}" is missing its operand list'
        )
    operands_close = _balanced_end(stripped, 0, "(", ")")
    operand_items = _split_top_level(stripped[1:operands_close])
    for index, item in enumerate(operand_items):
        if _SSA_RE.fullmatch(item) is None:
            raise TTIRParseError(
                f'quoted generic operation "{name}" operand {index} is not '
                f"one SSA value: {item!r}"
            )

    remainder = stripped[operands_close + 1 :].strip()
    colon = _find_top_level_token(remainder, ":", last=True)
    if colon is None:
        raise TTIRParseError(
            f'quoted generic operation "{name}" is missing its explicit '
            "function type"
        )
    prefix = remainder[:colon].strip()
    signature = remainder[colon + 1 :].strip()

    # Properties and attributes are both optional, but their order and
    # delimiters are part of the pinned generic syntax.  No unparsed token is
    # allowed between the operand list and the type contract.
    if prefix.startswith("<"):
        properties_close = _balanced_end(prefix, 0, "<", ">")
        properties = prefix[: properties_close + 1]
        if not (properties.startswith("<{") and properties.endswith("}>")):
            raise TTIRParseError(
                f'quoted generic operation "{name}" has malformed properties'
            )
        _validate_attribute_dictionary(
            properties[1:-1],
            context=f'quoted generic operation "{name}" properties',
        )
        prefix = prefix[properties_close + 1 :].strip()
    if prefix.startswith("{"):
        attributes_close = _balanced_end(prefix, 0, "{", "}")
        _validate_attribute_dictionary(
            prefix[: attributes_close + 1],
            context=f'quoted generic operation "{name}" attributes',
        )
        prefix = prefix[attributes_close + 1 :].strip()
    if prefix:
        raise TTIRParseError(
            f'quoted generic operation "{name}" has unsupported or malformed '
            f"trailing syntax before its type: {prefix!r}"
        )

    input_types, result_types = _functional_types(
        signature, context=f'quoted generic operation "{name}"'
    )
    if len(input_types) != len(operand_items):
        raise TTIRParseError(
            f'quoted generic operation "{name}" operand/type arity mismatch'
        )
    if len(result_types) != result_count:
        raise TTIRParseError(
            f'quoted generic operation "{name}" result/type arity mismatch'
        )
    return tuple(operand_items), input_types, result_types


def _parse_call_contract(
    tail: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Parse a Triton ``tt.call`` target and its explicit function type.

    Triton 3.6 normally emits custom assembly, ``tt.call @callee(...) :
    (...) -> (...)``.  The quoted generic spelling is accepted only when it
    carries the same explicit ``callee = @...`` attribute.  Gate1 does not
    infer missing call contracts from a callee: that would turn malformed or
    version-skewed TTIR into weaker evidence.
    """

    custom = re.match(rf"@(?P<name>{_SYMBOL_RE})\s*\(", tail)
    if custom is not None:
        callee = _symbol_value(custom.group("name"))
        args_open = custom.end() - 1
    else:
        generic = re.search(rf"\bcallee\s*=\s*@(?P<name>{_SYMBOL_RE})", tail)
        if generic is None:
            raise TTIRParseError("tt.call is missing a statically known callee")
        callee = _symbol_value(generic.group("name"))
        stripped = tail.lstrip()
        if not stripped.startswith("("):
            raise TTIRParseError("generic tt.call is missing its operand list")
        args_open = len(tail) - len(stripped)

    args_close = _balanced_end(tail, args_open, "(", ")")
    signature_colon = tail.find(":", args_close + 1)
    if signature_colon < 0:
        raise TTIRParseError(f"tt.call @{callee} is missing its explicit type")
    signature = tail[signature_colon + 1 :].strip()
    if not signature.startswith("("):
        raise TTIRParseError(f"tt.call @{callee} has an invalid operand type list")
    input_close = _balanced_end(signature, 0, "(", ")")
    operand_types = tuple(
        _normalize_space(item)
        for item in _split_top_level(signature[1:input_close])
    )
    remainder = signature[input_close + 1 :].strip()
    if not remainder.startswith("->"):
        raise TTIRParseError(f"tt.call @{callee} is missing its result type")
    result_text = remainder[2:].strip()
    if result_text == "()":
        result_types: tuple[str, ...] = ()
    elif result_text.startswith("("):
        result_close = _balanced_end(result_text, 0, "(", ")")
        if result_text[result_close + 1 :].strip():
            raise TTIRParseError(f"tt.call @{callee} has trailing type syntax")
        result_types = tuple(
            _normalize_space(item)
            for item in _split_top_level(result_text[1:result_close])
        )
    else:
        result_types = (_normalize_space(result_text),)
    return callee, operand_types, result_types


def _find_top_level_token(text: str, token: str, *, last: bool = False) -> int | None:
    """Find a token outside MLIR delimiters and quoted strings."""

    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    found: int | None = None
    index = 0
    while index <= len(text) - len(token):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        if char in pairs:
            stack.append(pairs[char])
            index += 1
            continue
        if stack and char == stack[-1]:
            stack.pop()
            index += 1
            continue
        if not stack and text.startswith(token, index):
            found = index
            if not last:
                return found
            index += len(token)
            continue
        index += 1
    return found


def _validate_closed_type(type_text: str, *, context: str) -> str:
    """Validate the encodingless, alias-free type boundary observed by Gate1.

    TTIR v7 deliberately does not interpret module-level type/encoding aliases.
    Accepting only their use-site spelling would let two different alias
    definitions hard-pass, so every proof-relevant type must be self-contained.
    Pinned TTIR is pre-layout and therefore needs no ranked-tensor encoding.
    """

    normalized = _normalize_space(type_text)
    if not normalized:
        raise TTIRParseError(f"{context} has an empty type")
    if re.search(r"#[A-Za-z_][A-Za-z0-9_.$-]*", normalized) is not None:
        raise TTIRParseError(
            f"{context} uses a module-level type/encoding alias; tensor "
            "encodings are unsupported and Gate1 v7 requires self-contained "
            "encodingless TTIR types"
        )
    if re.search(r"\s", normalized) is not None:
        raise TTIRParseError(
            f"{context} has unsupported trailing or compound type syntax: "
            f"{normalized!r}"
        )
    cursor = 0
    while True:
        marker = normalized.find("tensor<", cursor)
        if marker < 0:
            break
        opening = marker + len("tensor")
        closing = _balanced_end(normalized, opening, "<", ">")
        pieces = _split_top_level(normalized[opening + 1 : closing])
        if len(pieces) != 1:
            raise TTIRParseError(
                f"{context} uses a ranked-tensor encoding; Gate1 v7 "
                "requires encodingless TTIR types"
            )
        cursor = closing + 1
    return normalized


def _declared_types(text: str) -> tuple[str, ...]:
    text = text.strip()
    if not text or text == "()":
        return ()
    if text.startswith("("):
        closing = _balanced_end(text, 0, "(", ")")
        if text[closing + 1 :].strip():
            raise TTIRParseError(f"trailing syntax after type list {text!r}")
        text = text[1:closing]
    return tuple(
        _validate_closed_type(item, context="explicit operation type")
        for item in _split_top_level(text)
    )


def _functional_types(
    text: str, *, context: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse one explicit ``inputs -> results`` MLIR type contract."""

    signature = text.strip()
    arrow = _find_top_level_token(signature, "->")
    if arrow is None:
        raise TTIRParseError(f"{context} is missing its explicit function type")
    if _find_top_level_token(signature[arrow + 2 :], "->") is not None:
        raise TTIRParseError(f"{context} has more than one result arrow")
    input_text = signature[:arrow].strip()
    if not input_text.startswith("("):
        raise TTIRParseError(
            f"{context} function type inputs must use an explicit tuple"
        )
    input_close = _balanced_end(input_text, 0, "(", ")")
    if input_text[input_close + 1 :].strip():
        raise TTIRParseError(
            f"{context} has trailing syntax after its input type tuple"
        )
    inputs = _declared_types(input_text)
    results = _declared_types(signature[arrow + 2 :])
    return inputs, results


def _ranked_tensor_contract(
    type_text: str | None,
) -> tuple[tuple[str, ...], str, tuple[str, ...]] | None:
    """Return ``(shape, element, encoding)`` for a ranked tensor type.

    This deliberately recognizes only the closed ranked-tensor spelling used
    by pinned Triton TTIR.  It is not a general MLIR type parser.
    """

    if type_text is None:
        return None
    normalized = _normalize_space(type_text)
    if not normalized.startswith("tensor<"):
        return None
    opening = normalized.find("<")
    try:
        closing = _balanced_end(normalized, opening, "<", ">")
    except TTIRParseError:
        return None
    if normalized[closing + 1 :].strip():
        return None
    pieces = _split_top_level(normalized[opening + 1 : closing])
    if not pieces:
        return None
    if len(pieces) != 1:
        # Gate1 v7 does not parse module-level encoding aliases or prove their
        # denotation.  Comparing only alias spellings would let two different
        # layouts hard-pass, so the pinned scan subset is explicitly
        # encodingless.
        raise TTIRParseError(
            "tt.scan tensor encodings are unsupported by Gate1 v7; inputs "
            "and results must be encodingless"
        )
    shaped = _split_top_level(pieces[0], delimiter="x")
    if not shaped:
        return None
    dimensions = tuple(_normalize_space(item) for item in shaped[:-1])
    if any(re.fullmatch(r"(?:\?|[0-9]+)", item) is None for item in dimensions):
        return None
    element = _normalize_space(shaped[-1])
    if not element:
        return None
    return dimensions, element, ()


def _parse_scan_header(
    tail: str,
) -> tuple[list[str], int, bool]:
    """Parse Triton 3.6's quoted-generic ``tt.scan`` opening line.

    ``tt.scan`` has no custom assembly format in Triton 3.6.  The pinned
    compiler emits the operand list, a closed property dictionary containing
    exactly ``axis`` and ``reverse``, and then the single combine region.
    """

    stripped = tail.strip()
    if not stripped.startswith("("):
        raise TTIRParseError("generic tt.scan is missing its operand list")
    operands_close = _balanced_end(stripped, 0, "(", ")")
    operand_items = _split_top_level(stripped[1:operands_close])
    if not operand_items:
        raise TTIRParseError("tt.scan must have at least one input")
    operands: list[str] = []
    for index, item in enumerate(operand_items):
        if _SSA_RE.fullmatch(item) is None:
            raise TTIRParseError(
                f"tt.scan operand {index} is not one SSA value: {item!r}"
            )
        operands.append(item)

    remainder = stripped[operands_close + 1 :].strip()
    if not remainder.startswith("<{"):
        raise TTIRParseError(
            "tt.scan requires the Triton 3.6 <{axis, reverse}> property dictionary"
        )
    properties_close = _balanced_end(remainder, 0, "<", ">")
    properties_text = remainder[1:properties_close]
    properties = _canonical_attrs(properties_text)
    if len(properties) != 2 or {name for name, _ in properties} != {
        "axis",
        "reverse",
    }:
        raise TTIRParseError(
            "tt.scan properties must contain exactly one axis and one reverse"
        )
    property_map = dict(properties)
    axis_match = re.fullmatch(r"(-?[0-9]+):i32", property_map["axis"])
    if axis_match is None:
        raise TTIRParseError("tt.scan axis must be an explicit i32 integer")
    reverse_text = property_map["reverse"]
    if reverse_text not in {"false", "true"}:
        raise TTIRParseError("tt.scan reverse must be an explicit boolean")

    region_open = _normalize_space(remainder[properties_close + 1 :])
    if region_open != "({":
        raise TTIRParseError(
            "tt.scan must have exactly one quoted-generic combine region"
        )
    return operands, int(axis_match.group(1)), reverse_text == "true"


def _parse_reduce_header(tail: str) -> tuple[list[str], int]:
    """Parse Triton 3.6's quoted-generic ``tt.reduce`` opening line."""

    stripped = tail.strip()
    if not stripped.startswith("("):
        raise TTIRParseError("generic tt.reduce is missing its operand list")
    operands_close = _balanced_end(stripped, 0, "(", ")")
    operand_items = _split_top_level(stripped[1:operands_close])
    if not operand_items:
        raise TTIRParseError("tt.reduce must have at least one input")
    operands: list[str] = []
    for index, item in enumerate(operand_items):
        if _SSA_RE.fullmatch(item) is None:
            raise TTIRParseError(
                f"tt.reduce operand {index} is not one SSA value: {item!r}"
            )
        operands.append(item)

    remainder = stripped[operands_close + 1 :].strip()
    if not remainder.startswith("<{"):
        raise TTIRParseError(
            "tt.reduce requires the Triton 3.6 <{axis}> property dictionary"
        )
    properties_close = _balanced_end(remainder, 0, "<", ">")
    properties = _validate_attribute_dictionary(
        remainder[1:properties_close],
        context="tt.reduce properties",
    )
    if len(properties) != 1 or properties[0][0] != "axis":
        raise TTIRParseError("tt.reduce properties must contain exactly one axis")
    axis_match = re.fullmatch(r"(-?[0-9]+):i32", properties[0][1])
    if axis_match is None:
        raise TTIRParseError("tt.reduce axis must be an explicit i32 integer")
    if _normalize_space(remainder[properties_close + 1 :]) != "({":
        raise TTIRParseError(
            "tt.reduce must have exactly one quoted-generic combine region"
        )
    return operands, int(axis_match.group(1))


def _parse_typed_terminator(
    name: str,
    tail: str,
    *,
    allow_empty: bool,
) -> tuple[list[str], tuple[str, ...]]:
    """Parse one exact pinned custom terminator shell."""

    stripped = tail.strip()
    if not stripped:
        if allow_empty:
            return [], ()
        raise TTIRParseError(f"{name} requires at least one returned SSA value")
    if "{" in stripped or "}" in stripped:
        raise TTIRParseError(
            f"{name} attributes are unsupported by Gate1's pinned terminator shell"
        )
    colon = _find_top_level_token(stripped, ":", last=True)
    if colon is None:
        raise TTIRParseError(f"{name} is missing its explicit result types")
    operand_text = stripped[:colon].strip()
    type_text = stripped[colon + 1 :].strip()
    if not type_text:
        raise TTIRParseError(f"{name} is missing its explicit result types")
    if type_text.startswith("("):
        raise TTIRParseError(
            f"{name} custom result types must be an unparenthesized list"
        )

    if (
        not operand_text
        or operand_text.startswith(",")
        or operand_text.endswith(",")
        or re.search(r",\s*,", operand_text) is not None
    ):
        raise TTIRParseError(
            f"{name} requires a non-empty comma-separated operand list"
        )
    operand_items = _split_top_level(operand_text)
    if not operand_items or any(not item.strip() for item in operand_items):
        raise TTIRParseError(
            f"{name} requires a non-empty comma-separated operand list"
        )
    operands: list[str] = []
    for index, item in enumerate(operand_items):
        item = item.strip()
        if _SSA_RE.fullmatch(item) is None:
            raise TTIRParseError(
                f"{name} operand {index} is not one SSA value: {item!r}"
            )
        operands.append(item)

    declared = _declared_types(type_text)
    if len(declared) != len(operands):
        raise TTIRParseError(f"{name} SSA/type arity mismatch")
    return operands, declared


def _parse_scan_return(
    tail: str,
) -> tuple[list[str], tuple[str, ...]]:
    """Parse the exact Triton 3.6 custom ``tt.scan.return`` spelling."""

    return _parse_typed_terminator(
        "tt.scan.return", tail, allow_empty=False
    )


def _trailing_type_signature(tail: str) -> str | None:
    colon = _find_top_level_token(tail, ":", last=True)
    if colon is None:
        return None
    signature = tail[colon + 1 :].strip()
    return signature or None


def _loaded_value_type(pointer_type: str | None) -> str | None:
    if pointer_type is None:
        return None
    pointer_type = _normalize_space(pointer_type)
    if pointer_type.startswith("!tt.ptr<"):
        opening = pointer_type.find("<")
        closing = _balanced_end(pointer_type, opening, "<", ">")
        if pointer_type[closing + 1 :].strip():
            return None
        parts = _split_top_level(pointer_type[opening + 1 : closing])
        return _normalize_space(parts[0]) if parts else None
    marker = pointer_type.find("!tt.ptr<")
    if marker < 0 or not pointer_type.startswith("tensor<"):
        return None
    opening = pointer_type.find("<", marker)
    closing = _balanced_end(pointer_type, opening, "<", ">")
    parts = _split_top_level(pointer_type[opening + 1 : closing])
    if not parts:
        return None
    return _normalize_space(
        pointer_type[:marker] + parts[0] + pointer_type[closing + 1 :]
    )


def _comparison_result_type(operand_type: str | None) -> str | None:
    if operand_type is None:
        return None
    operand_type = _normalize_space(operand_type)
    if not operand_type.startswith("tensor<"):
        return "i1"
    body = operand_type[len("tensor<") : -1] if operand_type.endswith(">") else ""
    if not body:
        return None
    pieces = _split_top_level(body)
    shape_and_element = pieces[0]
    separator = shape_and_element.rfind("x")
    if separator < 0:
        return None
    pieces[0] = shape_and_element[: separator + 1] + "i1"
    return _normalize_space(f"tensor<{','.join(pieces)}>")


def _is_i1_predicate_type(type_text: str | None) -> bool:
    """Return whether one parser-validated SSA type is a boolean predicate.

    Triton's vector masks are ranked ``tensor<...xi1>`` values.  A token-boundary
    regular expression is not sufficient here: the ``x`` immediately before
    the element type is part of the MLIR tensor grammar, but is also an
    alphanumeric character.  Reuse the closed tensor-type parser so this
    accepts only scalar ``i1`` or an encodingless ranked tensor whose actual
    element type is exactly ``i1``.  Unknown or malformed types remain false.
    """

    if not isinstance(type_text, str):
        return False
    normalized = _normalize_space(type_text)
    if normalized == "i1":
        return True
    try:
        contract = _ranked_tensor_contract(normalized)
    except TTIRParseError:
        return False
    return contract is not None and contract[1] == "i1"


def _operation_result_types(
    name: str,
    tail: str,
    result_count: int,
    operand_types: Sequence[str | None],
    *,
    call_result_types: tuple[str, ...] = (),
) -> tuple[str | None, ...]:
    """Infer only result types exposed explicitly by Triton custom assembly.

    This is intentionally not a general MLIR type checker.  Unknown result
    forms remain ``None`` and are rejected only if they cross a ``tt.call`` or
    ``tt.return`` boundary where Gate1 must prove the actual SSA type.
    """

    if result_count == 0:
        return ()
    if name == "tt.call":
        if len(call_result_types) != result_count:
            raise TTIRParseError("tt.call result SSA/type arity mismatch")
        return tuple(call_result_types)
    if name == "scf.for":
        arrow = _find_top_level_token(tail, "->", last=True)
        if arrow is None:
            return (None,) * result_count
        result_text = tail[arrow + 2 :].lstrip()
        if not result_text.startswith("("):
            return (None,) * result_count
        closing = _balanced_end(result_text, 0, "(", ")")
        inferred = _declared_types(result_text[: closing + 1])
        return (
            tuple(inferred)
            if len(inferred) == result_count
            else (None,) * result_count
        )
    if name == "tt.addptr":
        inferred = operand_types[0] if operand_types else None
        return (inferred,) if result_count == 1 else (None,) * result_count
    if name == "tt.load" or name.startswith("tt.atomic"):
        inferred = _loaded_value_type(operand_types[0] if operand_types else None)
        return (inferred,) if result_count == 1 else (None,) * result_count
    if name in {"arith.cmpi", "arith.cmpf"}:
        inferred = _comparison_result_type(
            operand_types[0] if operand_types else None
        )
        return (inferred,) if result_count == 1 else (None,) * result_count
    if name == "arith.select":
        inferred = operand_types[1] if len(operand_types) >= 3 else None
        if (
            inferred is None
            or operand_types[2] != inferred
            or result_count != 1
        ):
            return (None,) * result_count
        signature = _trailing_type_signature(tail)
        if signature is None:
            return (None,)
        declared = _declared_types(signature)
        expected = (operand_types[0], inferred)
        if None in expected or declared != expected:
            return (None,)
        return (inferred,)

    signature = _trailing_type_signature(tail)
    if signature is None:
        return (None,) * result_count
    arrow = _find_top_level_token(signature, "->", last=True)
    if arrow is not None:
        inferred = _declared_types(signature[arrow + 2 :])
    else:
        conversion = _find_top_level_token(signature, " to ", last=True)
        inferred = (
            _declared_types(signature[conversion + 4 :])
            if conversion is not None
            else _declared_types(signature)
        )
    if len(inferred) != result_count:
        return (None,) * result_count
    return tuple(inferred)


def _scalar_constant(tail: str) -> float | None:
    """Return a simple scalar or dense-splat arith.constant value."""

    value = tail.split(":", 1)[0].strip()
    if value.startswith("dense<"):
        value = value[len("dense<") :]
        if value.endswith(">"):
            value = value[:-1]
        if any(token in value for token in ("[", "]", ",", "0x")):
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _authorized_extern_rsqrt(operation: _Operation, symbols: frozenset[str]) -> str | None:
    """Exact pure extern ABI, with library authority from this compile only."""
    if operation.name != "tt.extern_elementwise" or len(operation.operands) != 1 or len(operation.result_names) != 1:
        return None
    match = re.fullmatch(
        r"tt\.extern_elementwise %\{([^{}]*)\}:\(([^,(){}]+)\)->([^,(){}]+)",
        operation.local_signature,
    )
    if match is None or match[2] != match[3]:
        return None
    attrs = _split_top_level(match[1])
    expected = {'libname=""', 'libpath=""', 'pure=true'}
    symbol_attrs = [item for item in attrs if item.startswith('symbol=')]
    if len(attrs) != 4 or len(symbol_attrs) != 1 or set(attrs) - set(symbol_attrs) != expected:
        return None
    symbol = symbol_attrs[0].removeprefix('symbol=')
    types = {'"__nv_rsqrtf"': "f32", '"__nv_rsqrt"': "f64"}
    dtype = types.get(symbol)
    if dtype is None or symbol[1:-1] not in symbols:
        return None
    if re.fullmatch(r"(?:" + dtype + r"|tensor<(?:[1-9][0-9]*x)+" + dtype + r">)", match[2]) is None:
        return None
    return match[2]


def _bounded_normalize_operations(
    operations: list[_Operation],
    result_to_node: dict[str, int],
    terminal_forks: list[_TerminalFork],
    external_math_symbols: frozenset[str] = frozenset(),
    external_math_ftz: bool = False,
) -> tuple[list[_Operation], dict[str, int], list[_TerminalFork]]:
    """Apply the registered bounded rewrites after lexical SSA resolution.

    Gate1 deliberately keeps these false-reject normalizations narrow: neutral
    ``x + 0``, ``1 / sqrt(x)`` versus ``rsqrt(x)``, Triton's exact
    libdevice-rsqrt spelling versus ``math.rsqrt``, and a direct typed FMA versus
    its direct typed multiply-add/subtract expansion.  Applying them to the
    parsed graph is essential: MLIR SSA spellings are region-scoped, so a
    textual whole-function substitution would otherwise depend on whether
    sibling or nested regions happened to reuse the same local name.
    """

    node_by_id = {operation.node_id: operation for operation in operations}
    original_uses: Counter[str] = Counter()
    for operation in operations:
        original_uses.update(operation.operands)
        original_uses.update(operation.region_results)

    def direct_constant(name: str) -> float | None:
        node_id = result_to_node.get(name)
        if node_id is None:
            return None
        producer = node_by_id[node_id]
        if producer.name != "arith.constant":
            return None
        return producer.constant_value

    substitutions: dict[str, str] = {}
    removed_nodes: set[int] = set()
    normalization_constant_nodes: set[int] = set()

    # Rule N0: Triton may spell the same closed unary math operation either as a
    # typed ``tt.extern_elementwise`` libdevice call or as a ``math.*`` op.
    # Canonicalize only the explicitly named one-input/one-output symbols whose
    # input and result types agree; every unknown extern remains hard evidence.
    extern_unary_ops = {
        "__nv_rsqrtf": ("math.rsqrt", "f32"),
        "__nv_rsqrt": ("math.rsqrt", "f64"),
        "__nv_exp2f": ("math.exp2", "f32"),
        "__nv_exp2": ("math.exp2", "f64"),
        "__nv_expf": ("math.exp", "f32"),
        "__nv_exp": ("math.exp", "f64"),
    }
    for operation in operations:
        if (
            operation.name != "tt.extern_elementwise"
            or len(operation.result_names) != 1
            or len(operation.operands) != 1
        ):
            continue
        symbol_match = re.search(
            r'\bsymbol\s*=\s*"(?P<symbol>[^"]+)"',
            operation.local_signature,
        )
        signature_match = re.search(
            r":\((?P<input>[^,(){}]+)\)->(?P<result>[^,(){}]+)$",
            operation.local_signature,
        )
        if symbol_match is None or signature_match is None:
            continue
        canonical = extern_unary_ops.get(symbol_match.group("symbol"))
        if symbol_match.group("symbol") in {"__nv_rsqrtf", "__nv_rsqrt"} and _authorized_extern_rsqrt(operation, external_math_symbols) is None:
            continue
        input_type = _normalize_space(signature_match.group("input"))
        result_type = _normalize_space(signature_match.group("result"))
        element_match = re.fullmatch(
            r"(?:tensor<[^>]*x)?(?P<dtype>f32|f64)>?",
            input_type,
        )
        if (
            canonical is None
            or input_type != result_type
            or element_match is None
            or element_match.group("dtype") != canonical[1]
        ):
            continue
        # The reviewed f32 libdevice branch with nvvm-reflect-ftz=1 is a
        # distinct semantic operation. Never normalize it to ordinary rsqrt.
        is_ftz_rsqrt = (
            symbol_match.group("symbol") == "__nv_rsqrtf" and external_math_ftz
        )
        operation.name = canonical[0]
        operation.local_signature = _normalize_space(
            f"{operation.name} %"
            + (" {rsqrt_ftz=true}" if is_ftz_rsqrt else "")
            + f" : {result_type}"
        )
        operation.header_operand_count = 1

    # Rule N1: scalar/tensor floating or integer neutral addition.
    for operation in operations:
        if (
            operation.name not in {"arith.addf", "arith.addi"}
            or len(operation.result_names) != 1
            or len(operation.operands) != 2
        ):
            continue
        left, right = operation.operands
        neutral: str | None = None
        replacement: str | None = None
        if direct_constant(left) == 0.0:
            neutral, replacement = left, right
        elif direct_constant(right) == 0.0:
            neutral, replacement = right, left
        if neutral is None or replacement is None:
            continue
        substitutions[operation.result_names[0]] = replacement
        removed_nodes.add(operation.node_id)
        normalization_constant_nodes.add(result_to_node[neutral])

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in substitutions and name not in seen:
            seen.add(name)
            name = substitutions[name]
        return name

    for operation in operations:
        operation.operands = [resolve(name) for name in operation.operands]
        operation.region_results = [
            resolve(name) for name in operation.region_results
        ]
    for terminal_fork in terminal_forks:
        terminal_fork.predicate = resolve(terminal_fork.predicate)

    # Rule N2: direct 1 / sqrt(x) is represented as the same semantic node as
    # direct rsqrt(x).  This stays deliberately narrower than algebraic
    # simplification and does not chase arbitrary producer chains.
    for operation in operations:
        if (
            operation.node_id in removed_nodes
            or operation.name != "arith.divf"
            or len(operation.result_names) != 1
            or len(operation.operands) != 2
        ):
            continue
        numerator, denominator = operation.operands
        sqrt_node_id = result_to_node.get(denominator)
        if sqrt_node_id is None or direct_constant(numerator) != 1.0:
            continue
        sqrt_operation = node_by_id[sqrt_node_id]
        if (
            sqrt_operation.name != "math.sqrt"
            or len(sqrt_operation.result_names) != 1
            or len(sqrt_operation.operands) != 1
        ):
            continue
        operation.name = "math.rsqrt"
        operation.operands = [sqrt_operation.operands[0]]
        result_type = operation.local_signature.rsplit(":", 1)[-1]
        operation.local_signature = _normalize_space(
            f"math.rsqrt % : {result_type}"
        )
        operation.header_operand_count = 1
        normalization_constant_nodes.add(result_to_node[numerator])
        if original_uses[denominator] == 1:
            removed_nodes.add(sqrt_node_id)

    # Rule N3: one direct typed ``fma(a,b,c)`` has the same observer-level value
    # tree as ``a*b+c``.  A direct ``c = 0-d``/``neg(d)`` is represented as
    # ``a*b-d``.  Operand identities and their already-compiled types remain
    # untouched; no cast motion, reassociation, contraction of a larger tree, or
    # shared producer is hidden by this rewrite.  Floating rounding remains the
    # responsibility of the predeclared Gate2 comparison relation/Q.
    expanded_operations: list[_Operation] = []
    next_node_id = max((item.node_id for item in operations), default=-1) + 1
    for operation in operations:
        if (
            operation.node_id in removed_nodes
            or operation.name != "math.fma"
            or len(operation.result_names) != 1
            or len(operation.operands) != 3
        ):
            expanded_operations.append(operation)
            continue
        left, right, addend = operation.operands
        result_type = _normalize_space(operation.local_signature.rsplit(":", 1)[-1])
        multiply_result = f"{operation.result_names[0]}$fma_mul"
        while multiply_result in result_to_node:
            multiply_result += "$"
        multiply = _Operation(
            node_id=next_node_id,
            name="arith.mulf",
            result_names=(multiply_result,),
            operands=[left, right],
            local_signature=_normalize_space(
                f"arith.mulf %, % : {result_type}"
            ),
            context=operation.context,
            result_types=(result_type,),
            header_operand_count=2,
            fork_id=operation.fork_id,
            fork_arm=operation.fork_arm,
            structured_if_id=operation.structured_if_id,
            structured_if_arm=operation.structured_if_arm,
        )
        next_node_id += 1
        result_to_node[multiply_result] = multiply.node_id
        node_by_id[multiply.node_id] = multiply

        addend_node_id = result_to_node.get(addend)
        addend_operation = (
            node_by_id.get(addend_node_id) if addend_node_id is not None else None
        )
        subtract_operand: str | None = None
        if (
            addend_operation is not None
            and addend_operation.name == "arith.subf"
            and len(addend_operation.operands) == 2
            and direct_constant(addend_operation.operands[0]) == 0.0
        ):
            subtract_operand = addend_operation.operands[1]
            normalization_constant_nodes.add(
                result_to_node[addend_operation.operands[0]]
            )
        elif (
            addend_operation is not None
            and addend_operation.name == "arith.negf"
            and len(addend_operation.operands) == 1
        ):
            subtract_operand = addend_operation.operands[0]
        if subtract_operand is not None:
            operation.name = "arith.subf"
            operation.operands = [multiply_result, subtract_operand]
            if original_uses[addend] == 1 and addend_node_id is not None:
                removed_nodes.add(addend_node_id)
        else:
            operation.name = "arith.addf"
            operation.operands = [multiply_result, addend]
        operation.local_signature = _normalize_space(
            f"{operation.name} %, % : {result_type}"
        )
        operation.header_operand_count = 2
        expanded_operations.extend((multiply, operation))
    operations = expanded_operations

    remaining_uses: Counter[str] = Counter()
    for operation in operations:
        if operation.node_id in removed_nodes:
            continue
        remaining_uses.update(operation.operands)
        remaining_uses.update(operation.region_results)
    for node_id in normalization_constant_nodes:
        operation = node_by_id[node_id]
        if all(remaining_uses[result] == 0 for result in operation.result_names):
            removed_nodes.add(node_id)

    normalized = [
        operation
        for operation in operations
        if operation.node_id not in removed_nodes
    ]
    normalized_result_to_node = {
        result: operation.node_id
        for operation in normalized
        for result in operation.result_names
    }
    return normalized, normalized_result_to_node, terminal_forks


_PREDICATE_VIEW_OPS = {
    "arith.cmpi",
    "arith.cmpf",
    "arith.andi",
    "arith.ori",
    "arith.xori",
    "arith.select",
    "tt.broadcast",
    "tt.expand_dims",
    "tt.reshape",
    "tt.trans",
}

_DEFAULT_ZERO_LOWER_BOUND_CLASS = "default_zero_lower_bound"
_CLOSED_CONSUMER_IDENTITY_CLASS = "closed_consumer_identity"


def _zero_lower_bound_class_limit(
    anchor_exemptions: frozenset[str],
) -> int | None:
    """Read one bounded class cardinality from the unary Gate1 profile."""

    values = {
        int(item.split(":", 1)[1])
        for item in anchor_exemptions
        if item.startswith(_DEFAULT_ZERO_LOWER_BOUND_CLASS + ":")
        and item.split(":", 1)[1].isdigit()
    }
    if len(values) != 1:
        return None
    value = next(iter(values))
    return value if value > 0 else None


def _zero_lower_bound_mask_profile(
    operations: Sequence[_Operation],
    result_to_node: Mapping[str, int],
    anchor_exemptions: frozenset[str],
) -> tuple[frozenset[int], dict[str, str]]:
    """Find only compiler mask scaffolding authorized by one J access class.

    The profile recognizes the closed shape ``and(zero <= coordinate,
    surviving_predicate)`` at a load/store mask.  It never rewrites a bare
    predicate, an upper bound, a control predicate, a value, or a predicate
    with a shared zero constant.  The caller supplies the maximum number of
    J-authorized predicated access occurrences; an observation with more such
    shapes is not normalized.
    """

    limit = _zero_lower_bound_class_limit(anchor_exemptions)
    if limit is None:
        return frozenset(), {}
    node_by_id = {operation.node_id: operation for operation in operations}
    uses: dict[str, list[tuple[_Operation, int]]] = defaultdict(list)
    for operation in operations:
        for index, operand in enumerate(operation.operands):
            uses.setdefault(operand, []).append((operation, index))

    def is_mask_operand(operation: _Operation, index: int) -> bool:
        return (
            operation.name == "tt.load"
            and index == 1
            and operation.header_operand_count >= 2
        ) or (
            operation.name == "tt.store"
            and index == 2
            and operation.header_operand_count >= 3
        )

    def direct_zero_constant(name: str) -> int | None:
        node_id = result_to_node.get(name)
        if node_id is None:
            return None
        producer = node_by_id[node_id]
        return (
            node_id
            if producer.name == "arith.constant"
            and producer.constant_value is not None
            and float(producer.constant_value) == 0.0
            else None
        )

    def is_lane_coordinate(
        name: str, active: frozenset[str] = frozenset()
    ) -> bool:
        """Accept only a range lane and its type/rank-preserving wrappers."""

        if name in active:
            return False
        node_id = result_to_node.get(name)
        if node_id is None:
            return False
        producer = node_by_id[node_id]
        if producer.name == "tt.make_range":
            return True
        if producer.name not in {
            "arith.extsi",
            "arith.extui",
            "arith.index_cast",
            "arith.index_castui",
            "arith.trunci",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.reshape",
            "tt.splat",
            "tt.trans",
        } or len(producer.operands) != 1:
            return False
        return is_lane_coordinate(producer.operands[0], active | {name})

    def lower_bound_comparison(name: str) -> tuple[int, str] | None:
        node_id = result_to_node.get(name)
        if node_id is None:
            return None
        producer = node_by_id[node_id]
        if producer.name != "arith.cmpi" or len(producer.operands) != 2:
            return None
        operands = list(producer.operands)
        if producer.cmp_flipped:
            operands.reverse()
        if producer.cmp_predicate not in {"sle", "ule"}:
            return None
        zero_node = direct_zero_constant(operands[0])
        if (
            zero_node is None
            or direct_zero_constant(operands[1]) is not None
            or not is_lane_coordinate(operands[1])
        ):
            return None
        return node_id, operands[1]

    nodes: set[int] = set()
    aliases: dict[str, str] = {}
    matched = 0
    for operation in operations:
        if operation.name != "arith.andi" or len(operation.operands) != 2:
            continue
        root = operation.result_names[0] if len(operation.result_names) == 1 else None
        if root is None:
            continue
        root_uses = uses.get(root, [])
        if not root_uses or not all(is_mask_operand(*item) for item in root_uses):
            continue
        candidates = [
            (index, lower_bound_comparison(operand))
            for index, operand in enumerate(operation.operands)
        ]
        lower = [(index, value) for index, value in candidates if value is not None]
        if len(lower) != 1:
            continue
        lower_index, (comparison_node, survivor) = lower[0]
        survivor_node = result_to_node.get(survivor)
        if survivor_node is None or node_by_id[survivor_node].name not in _PREDICATE_VIEW_OPS:
            continue
        comparison = operation.operands[lower_index]
        comparison_uses = uses.get(comparison, [])
        if len(comparison_uses) != 1 or comparison_uses[0][0] is not operation:
            continue
        comparison_operation = node_by_id[comparison_node]
        comparison_operands = list(comparison_operation.operands)
        if comparison_operation.cmp_flipped:
            comparison_operands.reverse()
        zero_value = comparison_operands[0]
        zero_node = direct_zero_constant(zero_value)
        if zero_node is None:
            continue
        zero_uses = uses.get(zero_value, [])
        if len(zero_uses) != 1 or zero_uses[0][0] is not comparison_operation:
            continue
        nodes.update({operation.node_id, comparison_node, zero_node})
        aliases[root] = survivor
        matched += 1

    if matched == 0 or matched > limit:
        return frozenset(), {}
    return frozenset(nodes), aliases


def _closed_consumer_identity_class_limit(
    anchor_exemptions: frozenset[str],
) -> int | None:
    """Read the bounded cardinality of the π-authorized identity class."""

    values = {
        int(item.split(":", 1)[1])
        for item in anchor_exemptions
        if item.startswith(_CLOSED_CONSUMER_IDENTITY_CLASS + ":")
        and item.split(":", 1)[1].isdigit()
    }
    if len(values) != 1:
        return None
    value = next(iter(values))
    return value if value > 0 else None


def _closed_consumer_identity_nodes(
    operations: Sequence[_Operation],
    result_to_node: Mapping[str, int],
    anchor_exemptions: frozenset[str],
) -> frozenset[int]:
    """Find only optional value ``select`` nodes covered by π's class.

    The matcher is intentionally source-blind and syntactic.  A candidate must
    be an authored-mask ``arith.select`` with one of the two closed
    ``-inf -> 0`` spellings: a zero false arm after ``exp``, or a negative-
    infinity false arm immediately before ``exp``.  Its result must reach a
    reduction/value consumer through only transparent wrappers.  The J/π
    certificate proves the corresponding closed producer mapping; this
    function only identifies the bounded compiler materialization in one
    observation.  In particular, a zero false arm before ``exp`` is *not*
    accepted because ``exp(0)`` is not the outside identity.  Unknown selects
    stay hard.
    """

    limit = _closed_consumer_identity_class_limit(anchor_exemptions)
    if limit is None:
        return frozenset()
    node_by_id = {operation.node_id: operation for operation in operations}
    uses: dict[str, list[_Operation]] = defaultdict(list)
    for operation in operations:
        for operand in [*operation.operands, *operation.region_results]:
            uses.setdefault(operand, []).append(operation)

    def producer(value: str) -> _Operation | None:
        node_id = result_to_node.get(value)
        return node_by_id.get(node_id) if node_id is not None else None

    def reaches_exp(
        value: str,
        active: frozenset[str] = frozenset(),
    ) -> bool:
        if value in active:
            return False
        operation = producer(value)
        if operation is None:
            return False
        if operation.name in {"math.exp", "math.exp2", "math.expm1"}:
            return True
        if operation.name not in {
            "arith.cast", "arith.extf", "arith.truncf", "tt.broadcast",
            "tt.expand_dims", "tt.reshape", "tt.trans", "arith.select",
            "arith.subf", "arith.addf", "arith.mulf",
        }:
            return False
        return any(
            reaches_exp(operand, active | {value})
            for operand in operation.operands
        )

    def reaches_sub(
        value: str,
        active: frozenset[str] = frozenset(),
    ) -> bool:
        """Recognize the closed pre-exp producer boundary only."""

        if value in active:
            return False
        operation = producer(value)
        if operation is None:
            return False
        if operation.name == "arith.subf":
            return True
        if operation.name not in {
            "arith.cast", "arith.extf", "arith.truncf", "tt.broadcast",
            "tt.expand_dims", "tt.reshape", "tt.trans",
        }:
            return False
        return any(
            reaches_sub(operand, active | {value})
            for operand in operation.operands
        )

    transparent_shape_wrappers = {
        "tt.broadcast",
        "tt.expand_dims",
        "tt.reshape",
        "tt.trans",
    }

    def reaches_reduction(
        value: str,
        active: frozenset[str] = frozenset(),
    ) -> bool:
        """Follow only one-input shape views from a value to a consumer.

        A select may feed both the reduction and the final pointwise value.
        The identity quotient is valid for the whole SSA value in that case,
        but the matcher must not treat an arbitrary arithmetic user as proof of
        a reduction path.  Following uses, rather than only the direct user,
        handles Triton's common ``select -> reshape -> tt.reduce`` spelling
        while keeping the accepted wrapper set closed.
        """

        if value in active:
            return False
        for user in uses.get(value, ()):
            if user.name in {"tt.reduce", "tt.scan"}:
                return True
            if user.name in {"math.exp", "math.exp2", "math.expm1"}:
                if len(user.result_names) != 1:
                    continue
                if reaches_reduction(
                    user.result_names[0], active | {value}
                ):
                    return True
            if (
                user.name not in transparent_shape_wrappers
                or len(user.operands) != 1
                or len(user.result_names) != 1
            ):
                continue
            if reaches_reduction(
                user.result_names[0], active | {value}
            ):
                return True
        return False

    def reaches_exp_after(
        value: str,
        active: frozenset[str] = frozenset(),
    ) -> bool:
        """Recognize a select result consumed by ``exp`` through views."""

        if value in active:
            return False
        for user in uses.get(value, ()):
            if user.name in {"math.exp", "math.exp2", "math.expm1"}:
                return True
            if (
                user.name in transparent_shape_wrappers
                and len(user.operands) == 1
                and len(user.result_names) == 1
                and reaches_exp_after(user.result_names[0], active | {value})
            ):
                return True
        return False

    def constant_kind(operation: _Operation) -> str | None:
        if operation.name != "arith.constant":
            return None
        if operation.constant_value is not None:
            if float(operation.constant_value) == 0.0:
                return "zero"
            if operation.constant_value == float("-inf"):
                return "neg_inf"
        # MLIR commonly prints an f32 -inf splat as its IEEE hex payload;
        # _scalar_constant intentionally keeps such dense literals opaque.
        if re.search(r"0xFF800000", operation.local_signature, re.IGNORECASE):
            return "neg_inf"
        return None

    matched: list[int] = []
    for operation in operations:
        if operation.name != "arith.select" or len(operation.operands) != 3:
            continue
        condition, true_value, false_value = operation.operands
        false_operation = producer(false_value)
        if false_operation is None:
            continue
        false_kind = constant_kind(false_operation)
        if false_kind not in {"zero", "neg_inf"}:
            continue
        condition_users = uses.get(condition, [])
        if not any(
            (
                user.name == "tt.load"
                and user.header_operand_count >= 2
                and len(user.operands) > 1
                and user.operands[1] == condition
            )
            or (
                user.name == "tt.store"
                and user.header_operand_count >= 3
                and len(user.operands) > 2
                and user.operands[2] == condition
            )
            for user in condition_users
        ):
            continue
        result_names = operation.result_names
        if len(result_names) != 1:
            continue
        if not reaches_reduction(result_names[0]):
            continue
        post_exp_form = false_kind == "zero" and reaches_exp(true_value)
        pre_exp_form = (
            false_kind == "neg_inf"
            and reaches_sub(true_value)
            and reaches_exp_after(result_names[0])
        )
        if not (post_exp_form or pre_exp_form):
            continue
        matched.append(operation.node_id)
        if len(matched) > limit:
            return frozenset()
    return frozenset(matched)

_REPRESENTATION_ANCHOR_OPS = {
    # These stages have an owner in the closed realization topology.  Their
    # existence remains hard evidence even though the pure formula between
    # them is quotiented below.
    "tt.get_program_id",
    "tt.get_num_programs",
    "tt.addptr",
}


def _is_identity_reshape(operation: _Operation) -> bool:
    """Recognize only a type-preserving pure reshape.

    Triton may insert ``tt.reshape`` around a value while lowering an
    otherwise unchanged reduction.  A reshape is transparent here only when
    its explicit input and output types are byte-for-byte identical; rank
    changes, shape changes, and unknown signatures remain hard.
    """

    if operation.name != "tt.reshape":
        return False
    signature = _trailing_type_signature(operation.local_signature)
    if signature is None:
        return False
    arrow = _find_top_level_token(signature, "->", last=True)
    if arrow is None:
        return False
    left = _normalize_space(signature[:arrow])
    right = _normalize_space(signature[arrow + 2 :])
    return bool(left) and left == right


def _representation_only_nodes(
    operations: Sequence[_Operation],
    result_to_node: Mapping[str, int],
    terminal_forks: Sequence[_TerminalFork],
    *,
    anchor_exemptions: frozenset[str] = frozenset(),
    identity_nodes: frozenset[int] = frozenset(),
) -> frozenset[int]:
    """Find pure compiler scaffolding with no independent artifact owner.

    The observer keeps three hard slices intact:

    * values contributing to public returns or stored/updated values;
    * the boolean structure and exact constants of observable predicates;
    * effects, structured regions, transfers, and explicit address/program
      anchors.

    Remaining known-pure operations are representation-only when they merely
    spell a program schedule, address offset, rank lift, or loop bound.  Their
    concrete arithmetic tree is not present in J/C/pi and therefore cannot be
    a source-blind hidden answer.  The fingerprint contracts that tree to a
    typed carrier-origin edge; it does not erase the surviving predicate,
    pointer, transfer, region, or effect that consumes it.
    """

    node_by_id = {operation.node_id: operation for operation in operations}
    value_nodes: set[int] = set()
    predicate_nodes: set[int] = set()

    def mark_value(name: str, active: frozenset[int] = frozenset()) -> None:
        node_id = result_to_node.get(name)
        if node_id is None or node_id in value_nodes or node_id in active:
            return
        operation = node_by_id[node_id]

        if node_id in identity_nodes:
            # A π-authorized post-consumer identity materialization is a
            # representation detail.  Preserve the producer value as the
            # semantic operand and keep the select/zero literal out of the
            # hard value slice.
            if len(operation.operands) >= 2:
                mark_value(operation.operands[1], active | {node_id})
            return

        if _is_identity_reshape(operation):
            # Keep the value boundary transparent: the reshape itself may be
            # omitted from the fingerprint, but its producer still participates
            # in the hard value dataflow.
            for operand in operation.operands:
                mark_value(operand, active | {node_id})
            return
        value_nodes.add(node_id)

        # A load is the semantic value boundary.  Its pointer, mask, and fill
        # are observed by the address/predicate/effect slices instead of being
        # folded into the data-value computation.
        if operation.name == "tt.load":
            return
        if operation.name in _REPRESENTATION_ANCHOR_OPS:
            return
        if operation.name == "scf.for":
            try:
                slot = operation.result_names.index(name)
            except ValueError as error:  # pragma: no cover - map is co-built
                raise TTIRParseError(
                    f"loop result {name!r} is absent from its producer"
                ) from error
            names = [
                operation.operands[3 + slot],
                operation.region_results[slot],
            ]
        elif operation.name == "scf.if":
            names = list(operation.region_results)
        else:
            names = [*operation.operands, *operation.region_results]
        for operand in names:
            mark_value(operand, active | {node_id})

    def mark_predicate(name: str, active: frozenset[int] = frozenset()) -> None:
        node_id = result_to_node.get(name)
        if node_id is None or node_id in predicate_nodes or node_id in active:
            return
        operation = node_by_id[node_id]
        if operation.name == "arith.constant":
            predicate_nodes.add(node_id)
            return
        if operation.name not in _PREDICATE_VIEW_OPS:
            # Numeric index/address construction below the comparison is a
            # representation carrier.  The comparison itself, its polarity,
            # exact direct constants, and boolean composition remain hard.
            return
        predicate_nodes.add(node_id)
        for operand in [*operation.operands, *operation.region_results]:
            producer_id = result_to_node.get(operand)
            if producer_id is None:
                continue
            producer = node_by_id[producer_id]
            if (
                producer.name == "arith.constant"
                or producer.name in _PREDICATE_VIEW_OPS
            ):
                mark_predicate(operand, active | {node_id})

    for operation in operations:
        if operation.name == "tt.store" and len(operation.operands) >= 2:
            mark_value(operation.operands[1])
        elif operation.name == "tt.return":
            for operand in operation.operands:
                mark_value(operand)
        elif operation.name.startswith("tt.atomic"):
            # Pointer operand zero belongs to the address slice.  All remaining
            # authored update operands are observable values or predicates and
            # stay conservative hard evidence.
            for operand in operation.operands[1:]:
                mark_value(operand)

        if operation.name == "tt.load" and operation.header_operand_count >= 2:
            mark_predicate(operation.operands[1])
        elif (
            operation.name == "tt.store"
            and operation.header_operand_count >= 3
        ):
            mark_predicate(operation.operands[2])
        elif operation.name == "scf.if" and operation.operands:
            mark_predicate(operation.operands[0])
        elif (
            operation.name in {"tt.assert", "tt.device_assert"}
            and operation.operands
        ):
            mark_predicate(operation.operands[0])
    for terminal_fork in terminal_forks:
        mark_predicate(terminal_fork.predicate)

    zero_bound_nodes, _zero_bound_aliases = _zero_lower_bound_mask_profile(
        operations, result_to_node, anchor_exemptions
    )
    # A class-level lower-bound quotient is valid only for pure predicate
    # scaffolding.  If one of its nodes also contributes to a stored/returned
    # value, keep the entire class hard rather than partially erasing it.
    if zero_bound_nodes and zero_bound_nodes.isdisjoint(value_nodes):
        predicate_nodes.difference_update(zero_bound_nodes)
    else:
        zero_bound_nodes = frozenset()

    representation: set[int] = set()
    for operation in operations:
        if operation.node_id in value_nodes or operation.node_id in predicate_nodes:
            continue
        if (
            operation.name in _REPRESENTATION_ANCHOR_OPS
            and operation.name not in anchor_exemptions
        ):
            continue
        if operation.name in _REGION_OWNERS or operation.name in _TERMINATORS:
            continue
        if _operation_has_effect(operation):
            continue
        # Unknown operations never become presentation noise merely because
        # they happen not to reach the current output slice.
        if operation.name not in _KNOWN_PURE_OPS:
            continue
        representation.add(operation.node_id)
    representation.update(zero_bound_nodes)
    representation.update(identity_nodes)
    return frozenset(representation)


def _cmp_info(name: str, tail: str) -> tuple[str | None, bool]:
    if name not in {"arith.cmpi", "arith.cmpf"}:
        return None, False
    match = re.match(r"(?P<predicate>[A-Za-z]+)\s*,", tail)
    if match is None:
        return None, False
    predicate = match.group("predicate").lower()
    return _CMP_FLIPS.get(predicate, predicate), predicate in _CMP_FLIPS


def _local_signature(name: str, tail: str) -> str:
    erased = _SSA_RE.sub("%", tail)
    cache_modifiers = {
        "tt.load": "ca|cg|cv",
        "tt.store": "wb|cg|cs|wt",
    }.get(name)
    if cache_modifiers is not None:
        # Triton cache modifiers are target cache-policy hints.  They do not
        # change the memory effect, predicate, fill, address/dataflow topology,
        # or J denotation, and the closed J/C/pi artifact does not carry a
        # frontend cache-policy spelling.  Keeping them in the hard fingerprint
        # would therefore make Replay closure mathematically underdetermined.
        # Normalize only Triton 3.6's closed cache-hint values.  Closed eviction
        # policies are likewise presentation choices: pi owns the existence and
        # order of distinct transfer instances, while the public frontend may
        # need an eviction spelling merely to keep those instances from being
        # CSE'd.  The spelling itself is absent from J/C/pi and cannot be hard
        # evidence.  Unknown policies, masks, fills, types, volatility, and all
        # other operation attributes remain observed.
        erased = re.sub(
            rf"\bcacheModifier\s*=\s*(?:{cache_modifiers})(?=\s|:|,|$)",
            "",
            erased,
        )
        erased = re.sub(
            r"\bevictionPolicy\s*=\s*(?:evict_first|evict_last)"
            r"(?=\s|:|,|$)",
            "",
            erased,
        )
    predicate, _ = _cmp_info(name, tail)
    if predicate is not None:
        erased = re.sub(r"^[A-Za-z]+(?=\s*,)", predicate, erased)
    # Region opening punctuation does not describe the operation itself.
    erased = re.sub(r"\(\{\s*$", "", erased).strip()
    return _normalize_space(f"{name} {erased}")


def _parse_for_header(
    tail: str,
) -> tuple[list[str], list[str], str, tuple[str, ...]]:
    match = re.match(
        r"(?P<iv>%[A-Za-z0-9_.$-]+)\s*=\s*(?P<lower>%[A-Za-z0-9_.$-]+)"
        r"\s+to\s+(?P<upper>%[A-Za-z0-9_.$-]+)\s+step\s+"
        r"(?P<step>%[A-Za-z0-9_.$-]+)(?P<rest>.*)",
        tail,
    )
    if match is None:
        raise TTIRParseError(f"cannot parse scf.for header: {tail!r}")
    operands = [match.group("lower"), match.group("upper"), match.group("step")]
    block_args = [match.group("iv")]
    rest = match.group("rest").strip()
    if not rest.endswith("{"):
        raise TTIRParseError("scf.for header must end with exactly one body opener")
    rest = rest[:-1].strip()
    colon = _find_top_level_token(rest, ":", last=True)
    if colon is None:
        raise TTIRParseError("scf.for header is missing its induction type")
    prefix = rest[:colon].strip()
    induction_type = _validate_closed_type(
        rest[colon + 1 :], context="scf.for induction type"
    )
    result_types: tuple[str, ...] = ()
    if prefix:
        marker = re.match(r"iter_args\s*\(", prefix)
        if marker is None:
            raise TTIRParseError(
                f"scf.for has unsupported trailing header syntax {prefix!r}"
            )
        opening = prefix.find("(", marker.start())
        closing = _balanced_end(rest, opening, "(", ")")
        items = _split_top_level(prefix[opening + 1 : closing])
        if not items:
            raise TTIRParseError("scf.for iter_args must not be empty")
        for item in items:
            pair = re.fullmatch(
                r"(?P<arg>%[A-Za-z0-9_.$-]+)\s*=\s*(?P<init>%[A-Za-z0-9_.$-]+)",
                item,
            )
            if pair is None:
                raise TTIRParseError(f"cannot parse scf.for iter_arg: {item!r}")
            block_args.append(pair.group("arg"))
            operands.append(pair.group("init"))
        suffix = prefix[closing + 1 :].strip()
        if not suffix.startswith("->"):
            raise TTIRParseError("scf.for iter_args require explicit result types")
        result_text = suffix[2:].strip()
        if not result_text.startswith("("):
            raise TTIRParseError("scf.for result types must use an explicit tuple")
        result_close = _balanced_end(result_text, 0, "(", ")")
        if result_text[result_close + 1 :].strip():
            raise TTIRParseError("scf.for has trailing syntax after result types")
        result_types = _declared_types(result_text[: result_close + 1])
    return operands, block_args, induction_type, result_types


def _parse_closed_if_header(
    tail: str, *, result_count: int
) -> tuple[str, str | None]:
    """Parse one closed continuing ``scf.if`` header.

    Gate1 v10 admits either the v9 resultless shell or exactly one explicitly
    typed SSA result.  The latter must use the pinned custom spelling
    ``%predicate -> (type) {``; all other result arities and header extensions
    remain outside the observer profile.
    """

    stripped = tail.strip()
    if result_count == 0:
        match = re.fullmatch(
            r"(?P<predicate>%[A-Za-z0-9_.$-]+)\s*\{",
            stripped,
        )
        if match is None:
            raise TTIRParseError(
                "resultless scf.if must use exactly one scalar predicate"
            )
        return match.group("predicate"), None
    if result_count != 1:
        raise TTIRParseError(
            "value-producing scf.if requires exactly one SSA result"
        )
    match = re.match(
        r"(?P<predicate>%[A-Za-z0-9_.$-]+)\s*->\s*",
        stripped,
    )
    if match is None:
        raise TTIRParseError(
            "value-producing scf.if requires an explicit one-result type"
        )
    result_shell = stripped[match.end() :].strip()
    if not result_shell.startswith("("):
        raise TTIRParseError(
            "value-producing scf.if result type must use an explicit tuple"
        )
    result_close = _balanced_end(result_shell, 0, "(", ")")
    if result_shell[result_close + 1 :].strip() != "{":
        raise TTIRParseError(
            "value-producing scf.if header has unsupported trailing syntax"
        )
    result_types = _declared_types(result_shell[: result_close + 1])
    if len(result_types) != 1:
        raise TTIRParseError(
            "value-producing scf.if requires exactly one result type"
        )
    return match.group("predicate"), result_types[0]


_BLOCK_HEADER_RE = re.compile(
    r"\s*\^(?P<label>[A-Za-z0-9_.$-]+)"
    r"(?:\s*\((?P<args>.*)\))?\s*:\s*(?://.*)?$"
)
_TERMINAL_COND_BR_RE = re.compile(
    r"cf\.cond_br\s+(?P<predicate>%[A-Za-z0-9_.$-]+)\s*,\s*"
    r"\^(?P<true>[A-Za-z0-9_.$-]+)\s*,\s*"
    r"\^(?P<false>[A-Za-z0-9_.$-]+)\s*$"
)


def _parse_block_header(line: str) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    match = _BLOCK_HEADER_RE.fullmatch(line)
    if match is None:
        return None
    arguments: list[tuple[str, str]] = []
    for item in _split_top_level(match.group("args") or ""):
        argument = re.fullmatch(
            r"(?P<name>%[A-Za-z0-9_.$-]+)\s*:\s*(?P<type>.+)",
            item,
        )
        if argument is None:
            raise TTIRParseError(f"cannot parse region block argument: {item!r}")
        arguments.append(
            (
                argument.group("name"),
                _validate_closed_type(
                    argument.group("type"),
                    context=f"region block argument {argument.group('name')}",
                ),
            )
        )
    return match.group("label"), tuple(arguments)


def _operation_has_effect(operation: _Operation) -> bool:
    """Classify one operation under Gate1's closed effect contract.

    Returning ``False`` is a positive purity claim, not the absence of a match
    in an effect denylist.  Unknown operations are observer-unsupported at
    every nesting level because otherwise reordering two future effectful ops
    could hard-pass before Gate1 learns their semantics.
    """

    if operation.name in {
        "tt.extern_elementwise",
        "tt.elementwise_inline_asm",
    }:
        pure = re.search(r"\bpure=(true|false)\b", operation.local_signature)
        # Missing/unknown conditional-purity metadata is outside the closed
        # observer boundary and is conservatively effectful.
        return pure is None or pure.group(1) != "true"
    if operation.name in _KNOWN_EFFECT_OPS:
        return True
    if operation.name in _KNOWN_PURE_OPS:
        return False
    raise TTIRParseError(
        f"unclassified operation {operation.name} is outside Gate1's closed "
        "operation/effect profile"
    )


def _effect_occurrence_identities(
    operations: list[_Operation],
    terminal_forks: list[_TerminalFork],
) -> dict[int, str]:
    """Assign a canonical identity to every effectful operation occurrence.

    Content hashes cannot identify effect *events*: two otherwise identical
    loads separated by a store denote different observations, and consumers
    must retain which observation they use.  Identities are therefore derived
    from the closed effect trace after reachable calls have been expanded and
    bounded normalization has run.  Each path consists only of structured
    region kinds, terminal-fork roles, and ordinals within an effect trace.

    Raw node ids are used only to traverse the already-expanded lexical order;
    they never enter the resulting identity.  Pure operations receive no
    occurrence suffix, so identical pure nodes remain content-equivalent.
    """

    region_ids = {
        operation.node_id
        for operation in operations
        if operation.name in _REGION_OWNERS
    }
    structured_if_ids = {
        operation.node_id
        for operation in operations
        if operation.name == "scf.if"
    }
    terminal_fork_by_id = {
        terminal_fork.fork_id: terminal_fork
        for terminal_fork in terminal_forks
    }
    if len(terminal_fork_by_id) != len(terminal_forks):
        raise TTIRParseError("expanded terminal-fork identities are not unique")

    direct_sequence: dict[int | None, list[_Operation]] = {}
    direct_terminal_forks: dict[int | None, list[_TerminalFork]] = {}
    direct_fork_arms: dict[tuple[int, str], list[_Operation]] = {}
    direct_if_arms: dict[tuple[int, str], list[_Operation]] = {}

    for terminal_fork in terminal_forks:
        parent = (
            terminal_fork.context[-1]
            if terminal_fork.context
            else None
        )
        if parent is not None and parent not in region_ids:
            raise TTIRParseError(
                "terminal fork has a non-structured caller context"
            )
        direct_terminal_forks.setdefault(parent, []).append(terminal_fork)

    for operation in operations:
        parent = operation.context[-1] if operation.context else None
        if parent is not None and parent not in region_ids:
            raise TTIRParseError(
                f"operation {operation.name} has a non-region parent"
            )
        terminal_fork = (
            terminal_fork_by_id.get(operation.fork_id)
            if operation.fork_id is not None
            else None
        )
        if operation.fork_id is not None and terminal_fork is None:
            raise TTIRParseError(
                "operation refers to an unknown expanded terminal fork"
            )
        if operation.fork_arm not in {None, "true", "false"}:
            raise TTIRParseError("operation has an invalid terminal-fork role")
        is_direct_fork_member = (
            terminal_fork is not None
            and operation.context == terminal_fork.context
        )
        if is_direct_fork_member:
            assert operation.fork_arm is not None
            direct_fork_arms.setdefault(
                (terminal_fork.fork_id, operation.fork_arm), []
            ).append(operation)
        elif parent in structured_if_ids:
            if (
                operation.structured_if_id != parent
                or operation.structured_if_arm
                not in {"positive", "negative"}
            ):
                raise TTIRParseError(
                    "direct scf.if member lost its polarity during expansion"
                )
            direct_if_arms.setdefault(
                (parent, operation.structured_if_arm), []
            ).append(operation)
        else:
            direct_sequence.setdefault(parent, []).append(operation)

    region_effect_cache: dict[int, bool] = {}
    fork_effect_cache: dict[int, bool] = {}

    def region_has_effect(region_id: int) -> bool:
        cached = region_effect_cache.get(region_id)
        if cached is not None:
            return cached
        cached = any(
            operation.name not in _TERMINATORS
            and _operation_has_effect(operation)
            for operation in operations
            if region_id in operation.context
        )
        region_effect_cache[region_id] = cached
        return cached

    def terminal_fork_has_effect(terminal_fork: _TerminalFork) -> bool:
        cached = fork_effect_cache.get(terminal_fork.fork_id)
        if cached is not None:
            return cached
        cached = any(
            operation.name not in _TERMINATORS
            and _operation_has_effect(operation)
            for operation in operations
            if operation.fork_id == terminal_fork.fork_id
        )
        fork_effect_cache[terminal_fork.fork_id] = cached
        return cached

    identities: dict[int, str] = {}

    def visit_items(
        items: list[tuple[int, int, _Operation | _TerminalFork]],
        scope_path: tuple[str, ...],
    ) -> None:
        effect_ordinal = 0
        for _, _, item in sorted(items, key=lambda entry: (entry[0], entry[1])):
            if isinstance(item, _TerminalFork):
                if not terminal_fork_has_effect(item):
                    continue
                event_path = (
                    *scope_path,
                    f"effect[{effect_ordinal}]:control.conditional",
                )
                effect_ordinal += 1
                for role in ("true", "false"):
                    arm_items = [
                        (operation.node_id, 1, operation)
                        for operation in direct_fork_arms.get(
                            (item.fork_id, role), ()
                        )
                    ]
                    visit_items(
                        arm_items,
                        (*event_path, f"arm[{role}]"),
                    )
                continue

            if item.node_id in region_ids:
                if not region_has_effect(item.node_id):
                    continue
                if item.node_id in structured_if_ids:
                    event_path = (
                        *scope_path,
                        f"effect[{effect_ordinal}]:control.scf.if",
                    )
                    effect_ordinal += 1
                    for role in ("positive", "negative"):
                        arm_items = [
                            (operation.node_id, 1, operation)
                            for operation in direct_if_arms.get(
                                (item.node_id, role), ()
                            )
                        ]
                        visit_items(
                            arm_items,
                            (*event_path, f"arm[{role}]"),
                        )
                    continue
                event_path = (
                    *scope_path,
                    f"effect[{effect_ordinal}]:region[{item.name}]",
                )
                effect_ordinal += 1
                visit_scope(item.node_id, event_path)
                continue

            if item.name in _TERMINATORS or not _operation_has_effect(item):
                continue
            event_path = (
                *scope_path,
                f"effect[{effect_ordinal}]:{item.name}",
            )
            effect_ordinal += 1
            identities[item.node_id] = f"event[{'/'.join(event_path)}]"

    def visit_scope(parent: int | None, scope_path: tuple[str, ...]) -> None:
        items: list[tuple[int, int, _Operation | _TerminalFork]] = [
            (operation.node_id, 1, operation)
            for operation in direct_sequence.get(parent, ())
        ]
        items.extend(
            (terminal_fork.sequence_index, 0, terminal_fork)
            for terminal_fork in direct_terminal_forks.get(parent, ())
        )
        visit_items(items, scope_path)

    visit_scope(None, ("root",))

    expected = {
        operation.node_id
        for operation in operations
        if operation.name not in _TERMINATORS
        and _operation_has_effect(operation)
    }
    if identities.keys() != expected:
        raise TTIRParseError(
            "closed effect trace did not assign exactly one identity to every "
            "effectful operation occurrence"
        )
    return identities


def _canonical_python_interface(
    metadata: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if not metadata:
        return ()
    # Only callable-signature facts participate in the hard interface. Triton's
    # do_not_specialize metadata controls JIT specialization policy, not the
    # source/replay semantics tested by SCRC.
    metadata = {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "arg_names",
            "constexprs",
            "constexpr_indices",
            "function_name",
            "launch_grid",
            "grid",
            "grid_contract",
        }
    }
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if (
            key == "constexprs"
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ):
            normalized[str(key)] = sorted(value, key=str)
        elif isinstance(value, tuple):
            normalized[str(key)] = list(value)
        else:
            normalized[str(key)] = value
    return tuple((key, _stable_json(value)) for key, value in sorted(normalized.items()))


def _canonical_schedule(
    schedule: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if not schedule:
        return ()
    return tuple((str(key), _stable_json(value)) for key, value in sorted(schedule.items()))


class TTIRParser:
    """Parse one Triton 3.6 TTIR module into a Gate1 fingerprint."""

    def parse(
        self,
        text: str,
        *,
        python_interface: Mapping[str, Any] | None = None,
        schedule: Mapping[str, Any] | None = None,
        representation_anchor_exemptions: frozenset[str] = frozenset(),
        runtime_values: Mapping[str, int | bool] | None = None,
        external_math: Mapping[str, Any] | None = None,
        external_math_source: str | None = None,
    ) -> TTIRFingerprint:
        external_math_symbols = validated_external_math_symbols(
            text if external_math_source is None else external_math_source, external_math,
        )
        statement_locations: dict[int, _StatementSourceLocation] = {}
        source_location_records = _diagnostic_source_location_records(
            text, statement_locations=statement_locations,
            source_file=python_interface.get("compiled_source_file") if isinstance(python_interface, Mapping) else None,
        )
        cleaned = _strip_locations(text)
        # Module type aliases are definitions, while proof-relevant use sites
        # carry only the alias spelling.  Ignoring the definition would let a
        # source/replay pair keep ``!alias`` unchanged while changing its
        # denotation.  Gate1 v7 does not elaborate MLIR aliases, so reject the
        # entire closed profile before parsing any interface or operation type.
        if re.search(
            r"(?m)^\s*![A-Za-z_][A-Za-z0-9_.$-]*\s*=\s*type(?:\s|$)",
            cleaned,
        ) is not None:
            raise TTIRParseError(
                "module-level type aliases are unsupported by Gate1 v7; "
                "proof-relevant types must be self-contained"
            )
        functions = _parse_functions(cleaned, statement_locations)
        function = next(
            item for item in functions.values() if item.visibility == "public"
        )
        operations, result_to_node, leaf_labels, terminal_forks = (
            self._expand_reachable_calls(function, functions)
        )
        if external_math_symbols and external_math_source is not None and external_math_source != text:
            # Only the host's established Gate1 normalization may rebind a
            # receipt. Its raw extern ABI/attribute multiset must be intact;
            # adding/changing a call cannot borrow a different compilation.
            originals = _parse_functions(_strip_locations(external_math_source))
            original_entry = next(item for item in originals.values() if item.visibility == "public")
            original_operations, _results, _leaves, _forks = self._expand_reachable_calls(original_entry, originals)
            def extern_signatures(items: Sequence[_Operation]) -> Counter[str]:
                return Counter(item.local_signature for item in items if item.name == "tt.extern_elementwise")
            if extern_signatures(operations) != extern_signatures(original_operations):
                external_math_symbols = frozenset()
        for operation in operations:
            if operation.name != "tt.extern_elementwise" or re.search(
                r'\bsymbol="__nv_rsqrtf?"', operation.local_signature,
            ) is None:
                continue
            if _authorized_extern_rsqrt(operation, external_math_symbols) is None:
                rule = external_math.get("rule_code") if isinstance(external_math, Mapping) else None
                raise TTIRParseError(
                    rule if isinstance(rule, str) and rule in EXTERNAL_MATH_FAILURE_RULES
                    else "source_extern_rsqrt_authorization_unavailable"
                )
        owner_graphs = ()
        tensor_owner_graphs = ()
        if len(functions) == 1:
            owner_graphs = self._scalar_owner_graphs(text, function, python_interface)
        tensor_owner_graphs = self._tensor_owner_graphs(
            text, function, python_interface, external_math_symbols,
            external_math_ftz=bool(external_math_symbols) and external_math.get("ftz") is True,
            expanded=(operations, result_to_node, terminal_forks),
        )
        operations, result_to_node, terminal_forks = (
            _bounded_normalize_operations(
                operations, result_to_node, terminal_forks, external_math_symbols,
                external_math_ftz=external_math_symbols != frozenset() and external_math.get("ftz") is True,
            )
        )
        return self._fingerprint(
            function,
            operations,
            result_to_node,
            leaf_labels,
            terminal_forks,
            python_interface=python_interface,
            schedule=schedule,
            representation_anchor_exemptions=representation_anchor_exemptions,
            runtime_values=runtime_values,
            source_locations=tuple(
                (operation, line, column)
                for operation, _results, line, column in source_location_records
            ),
            scalar_owner_graphs=owner_graphs,
            tensor_owner_graphs=tensor_owner_graphs,
        )

    def _tensor_owner_graphs(
        self, text: str, function: _FunctionParts,
        python_interface: Mapping[str, Any] | None,
        external_math_symbols: frozenset[str] = frozenset(),
        external_math_ftz: bool = False,
        expanded: tuple[list[_Operation], dict[str, int], list[_TerminalFork]] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Private typed edges for lane vectors and direct program-row addresses.

        No logical shape, padding law or Source occurrence is invented here.
        Runtime bounds and row/stride edges require a frozen host-launch
        contract at the Source ledger before they can authorize a receipt.
        """
        names = _owner_runtime_parameters(function, python_interface, ttir=text)
        if names is None:
            return ()
        if expanded is None:
            raw, result_to_node, _leaves, fork = self._parse_operations(function)
        else:
            raw, result_to_node, forks = expanded
            fork = forks or None
        nodes = {op.node_id: op for op in raw}
        # Calls have already been checked for exact argument/result types and
        # independently renamed SSA values. Do not reject unrelated stores
        # merely because another dependency slice contains structured control.
        # Each memory event and producer below must still be outside a region;
        # only the explicitly checked reduction combiner may enter one.
        if fork is not None:
            return ()
        arguments = {name: (index, dtype) for index, (name, dtype, _attrs) in enumerate(function.arguments)}
        arithmetic = {"arith.addf": "add", "arith.subf": "sub", "arith.mulf": "mul", "arith.divf": "div", "arith.negf": "neg"}

        def producer(value: str) -> _Operation:
            op = nodes.get(result_to_node.get(value))
            if op is None or len(op.result_names) != 1 or len(op.result_types) != 1:
                raise ValueError("source_owner_producer_unsupported")
            return op

        def value_type(value: str) -> str:
            typ = arguments[value][1] if value in arguments else producer(value).result_types[0]
            if not isinstance(typ, str):
                raise ValueError("source_owner_type_unsupported")
            return typ

        def typed(typ: str) -> tuple[str, tuple[int, ...]]:
            tensor = _ranked_tensor_contract(typ)
            if tensor is None:
                return typ, ()
            shape, dtype, _encoding = tensor
            if any(not dimension.isdigit() or int(dimension) <= 0 for dimension in shape):
                raise ValueError("source_owner_dynamic_tile_unsupported")
            return dtype, tuple(int(dimension) for dimension in shape)

        def bare(op: _Operation, suffix: str) -> None:
            if op.local_signature != _normalize_space(op.name + " " + ", ".join("%" for _ in op.operands) + " : " + suffix):
                raise ValueError("source_owner_attributes_unsupported")

        def constant(value: str) -> tuple[str, tuple[int, ...], str]:
            op = producer(value)
            dtype, shape = typed(value_type(value))
            if op.name == "tt.splat" and len(op.operands) == 1:
                bare(op, value_type(op.operands[0]) + " -> " + value_type(value))
                source_dtype, source_shape, bits = constant(op.operands[0])
                if source_shape or source_dtype != dtype:
                    raise ValueError("source_owner_splat_type_mismatch")
                return dtype, shape, bits
            if op.name != "arith.constant" or op.operands or dtype not in {"f32", "f64"}:
                raise ValueError("source_owner_literal_unsupported")
            literal = r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
            match = re.fullmatch(r"arith\.constant " + ("dense<" + literal + ">" if shape else literal) + ":" + re.escape(value_type(value)), op.local_signature)
            if match is not None:
                number = float(match[1])
                if Decimal(match[1]) != Decimal.from_float(number):
                    raise ValueError("source_owner_literal_inexact")
                return dtype, shape, owner_float_bits(number, dtype)
            hex_match = re.fullmatch(
                r"arith\.constant dense<0x([0-9A-Fa-f]+)>:" + re.escape(value_type(value)),
                op.local_signature,
            )
            if hex_match is None:
                raise ValueError("source_owner_literal_unsupported")
            expected_digits = 8 if dtype == "f32" else 16
            bits = hex_match[1].lower()
            if len(bits) != expected_digits:
                raise ValueError("source_owner_literal_unsupported")
            return dtype, shape, bits

        def integer(value: str) -> int:
            op = producer(value)
            dtype, _shape = typed(value_type(value))
            if dtype not in {"i32", "i64"}:
                raise ValueError("source_owner_coordinate_type_unsupported")
            if op.name == "tt.splat" and len(op.operands) == 1:
                bare(op, value_type(op.operands[0]) + " -> " + value_type(value))
                if typed(value_type(op.operands[0])) != (dtype, ()):
                    raise ValueError("source_owner_coordinate_type_mismatch")
                return integer(op.operands[0])
            match = re.fullmatch(r"arith\.constant (?:dense<(-?[0-9]+)>|(-?[0-9]+)):" + re.escape(value_type(value)), op.local_signature)
            if op.name != "arith.constant" or op.operands or match is None:
                raise ValueError("source_owner_coordinate_not_constant")
            number = int(match[1] if match[1] is not None else match[2])
            if not -(2 ** (int(dtype[1:]) - 1)) <= number < 2 ** (int(dtype[1:]) - 1):
                raise ValueError("source_owner_coordinate_overflow")
            return number

        def lane(value: str) -> int:
            op = producer(value)
            dtype, shape = typed(value_type(value))
            if op.name != "tt.make_range" or op.operands or dtype != "i32" or len(shape) != 1 or shape[0] >= 2 ** 31:
                raise ValueError("source_owner_coordinate_not_full_range")
            attributes = (f"{{end = {shape[0]} : i32, start = 0 : i32}}", f"{{start = 0 : i32, end = {shape[0]} : i32}}")
            if op.local_signature not in {_normalize_space("tt.make_range " + attrs + " : " + value_type(value)) for attrs in attributes}:
                raise ValueError("source_owner_coordinate_not_full_range")
            return shape[0]

        def row_offset(value: str) -> tuple[int, int, str]:
            op = producer(value)
            if op.name != "arith.muli" or len(op.operands) != 2 or value_type(value) not in {"i32", "i64"}:
                raise ValueError("source_owner_row_offset_unproved")
            bare(op, value_type(value))
            matches = []
            for pid, stride in (op.operands, tuple(reversed(op.operands))):
                stride_argument = stride
                if stride in arguments:
                    argument_dtype = arguments[stride][1]
                    if argument_dtype == "i64":
                        stride_mode = "direct_i64"
                    elif argument_dtype == "i32" and value_type(value) == "i32":
                        stride_mode = "direct_i32"
                    else:
                        continue
                else:
                    narrowed = producer(stride)
                    if narrowed.name != "arith.extsi" or len(narrowed.operands) != 1 or value_type(stride) != "i64" or value_type(narrowed.operands[0]) != "i32":
                        continue
                    bare(narrowed, "i32 to i64")
                    stride_argument = narrowed.operands[0]
                    stride_mode = "signed_i32_extended"
                if stride_argument not in arguments or arguments[stride_argument][1] != ("i32" if stride_mode in {"direct_i32", "signed_i32_extended"} else "i64"):
                    continue
                extended = producer(pid)
                if stride_mode in {"direct_i64", "signed_i32_extended"}:
                    if (
                        extended.name != "arith.extsi"
                        or len(extended.operands) != 1
                        or value_type(pid) != "i64"
                        or value_type(extended.operands[0]) != "i32"
                    ):
                        continue
                    bare(extended, "i32 to i64")
                    program = producer(extended.operands[0])
                else:
                    # Triton may keep both program-id and stride arithmetic in
                    # i32 when the host ABI value is source-level i64.  This
                    # is a distinct compiler ABI mode, not an i64 valuation
                    # inferred from the current launch.  Preserve the mode so
                    # the Source owner can require the host launch contract's
                    # independent stride/range proof below.
                    if value_type(pid) != "i32":
                        continue
                    program = producer(pid)
                    if value_type(value) != "i32":
                        continue
                    stride_mode = "direct_i32"
                match = re.fullmatch(r"tt\.get_program_id ([xyz]):i32", program.local_signature)
                if program.name == "tt.get_program_id" and not program.operands and match is not None:
                    matches.append(("xyz".index(match[1]), arguments[stride_argument][0], stride_mode))
            if len(matches) != 1:
                raise ValueError("source_owner_row_stride_unproved")
            return matches[0]

        def pointer(value: str) -> tuple[int, str | int, str | None, tuple[int, int, str] | None]:
            if value in arguments:
                position, typ = arguments[value]
                if _pointer_dtype(typ) not in {"f32", "f64"}:
                    raise ValueError("source_owner_pointer_type_unsupported")
                return position, 0, None, None
            op = producer(value)
            if op.name == "tt.splat" and len(op.operands) == 1:
                bare(op, value_type(op.operands[0]) + " -> " + value_type(value))
                position, offset, coordinate, row = pointer(op.operands[0])
                if offset != 0 or coordinate is not None:
                    raise ValueError("source_owner_pointer_unsupported")
                return position, 0, None, row
            if op.name != "tt.addptr" or len(op.operands) != 2:
                raise ValueError("source_owner_pointer_unsupported")
            bare(op, ", ".join(value_type(operand) for operand in op.operands))
            position, base, coordinate, row = pointer(op.operands[0])
            if base != 0 or coordinate is not None:
                raise ValueError("source_owner_pointer_offset_unsupported")
            if row is None:
                try:
                    return position, 0, None, row_offset(op.operands[1])
                except ValueError:
                    pass
            try:
                lane(op.operands[1])
                return position, "axis0", op.operands[1], row
            except ValueError:
                return position, integer(op.operands[1]), None, row

        def mask_contract(mask_value: str, lane_value: str | None, shape: tuple[int, ...]) -> tuple[str, int, Any]:
            if lane_value is None:
                raise ValueError("source_owner_scalar_mask_unproved")
            mask = producer(mask_value)
            if mask.name != "arith.cmpi" or mask.cmp_predicate not in {"slt", "ult"} or mask.cmp_flipped or len(mask.operands) != 2 or mask.operands[0] != lane_value or value_type(mask.operands[0]) != value_type(mask.operands[1]):
                raise ValueError("source_owner_mask_unproved")
            expected = "arith.cmpi " + mask.cmp_predicate + ", %, % : " + value_type(lane_value)
            if mask.local_signature != _normalize_space(expected) or typed(value_type(mask_value)) != ("i1", shape):
                raise ValueError("source_owner_mask_type_mismatch")
            try:
                bound: Any = integer(mask.operands[1])
            except ValueError:
                splat = producer(mask.operands[1])
                if splat.name != "tt.splat" or len(splat.operands) != 1 or splat.operands[0] not in arguments or arguments[splat.operands[0]][1] != "i32":
                    raise ValueError("source_owner_mask_bound_unproved")
                bare(splat, "i32 -> " + value_type(mask.operands[1]))
                bound = ("parameter", arguments[splat.operands[0]][0])
            if len(shape) != 1 or (isinstance(bound, int) and not 0 < bound <= shape[0]):
                raise ValueError("source_owner_mask_domain_unproved")
            return ("lt", 0, bound)

        def memory(op: _Operation) -> dict[str, Any]:
            kind = "load" if op.name == "tt.load" else "store"
            location = op.source_location
            if location is None or location[0] != op.name:
                raise ValueError("source_owner_memory_location_ambiguous")
            if op.context or len(op.operands) not in ({1, 3} if kind == "load" else {2, 3}):
                raise ValueError("source_owner_memory_form_unsupported")
            bare(op, value_type(op.operands[0]))
            transferred = op.result_names[0] if kind == "load" else op.operands[1]
            if _loaded_value_type(value_type(op.operands[0])) != value_type(transferred):
                raise ValueError("source_owner_memory_element_type_mismatch")
            position, coordinate, lane_value, row = pointer(op.operands[0])
            dtype = _pointer_dtype(function.arguments[position][1])
            predicate, fill = None, None
            if kind == "load":
                load_dtype, physical_shape = typed(value_type(op.result_names[0]))
                coordinate_shape_matches = (
                    physical_shape == (lane(lane_value),) if lane_value is not None
                    else physical_shape == () and type(coordinate) is int
                )
                if dtype != load_dtype or not coordinate_shape_matches:
                    raise ValueError("source_owner_load_coordinate_shape_mismatch")
                if len(op.operands) == 3:
                    predicate = mask_contract(op.operands[1], lane_value, physical_shape)
                    fill_dtype, fill_shape, fill = constant(op.operands[2])
                    if (fill_dtype, fill_shape) != (dtype, physical_shape):
                        raise ValueError("source_owner_fill_type_mismatch")
            else:
                physical_shape = (lane(lane_value),) if lane_value is not None else ()
                if typed(value_type(op.operands[1])) != (dtype, physical_shape):
                    raise ValueError("source_owner_store_slice_unsupported")
                if len(op.operands) == 3:
                    predicate = mask_contract(op.operands[2], lane_value, physical_shape)
            # Keep the exact compiler producer identity in the private
            # observation only.  It is never serialized into Gate1/J or sent
            # to a provider; Source-aware admission uses it to ensure that a
            # store authorization is tied to the same TTIR store/value edge
            # that produced the owner DAG.
            stored_value_producer_id = (
                result_to_node.get(op.operands[1])
                if kind == "store" and len(op.operands) >= 2
                else None
            )
            stored_value_producer = (
                nodes.get(stored_value_producer_id)
                if stored_value_producer_id is not None
                else None
            )
            return dict(line=location[1], column=location[2], call_path=op.source_call_path, kind=kind,
                        pointer_position=position, coordinate=coordinate, dtype=dtype,
                        row=row,
                        physical_shape=physical_shape, predicate=predicate, fill=fill,
                        memory_node_id=op.node_id,
                        stored_value_producer_id=stored_value_producer_id,
                        stored_value_producer_operation=(
                            stored_value_producer.name
                            if stored_value_producer is not None else None
                        ),
                        stored_value_type=(
                            value_type(op.operands[1])
                            if kind == "store" and len(op.operands) >= 2
                            else None
                        ))

        def describe(value: str) -> tuple[str, str, Sequence[int], Any, Sequence[str]]:
            if value in arguments:
                position, dtype = arguments[value]
                return "input", dtype, (), position, ()
            op = producer(value)
            dtype, shape = typed(value_type(value))
            if op.context:
                raise ValueError("source_owner_region_escape")
            if op.name == "tt.load":
                return "load", dtype, shape, memory(op), ()
            if op.name == "arith.constant":
                constant_dtype, constant_shape, bits = constant(value)
                return "const", constant_dtype, constant_shape, bits, ()
            if op.name == "tt.splat" and len(op.operands) == 1:
                bare(op, value_type(op.operands[0]) + " -> " + value_type(value))
                if typed(value_type(op.operands[0])) != (dtype, ()):
                    raise ValueError("source_owner_splat_type_mismatch")
                return "broadcast", dtype, shape, (), tuple(op.operands)
            if op.name in {"tt.reshape", "tt.trans"} and len(op.operands) == 1:
                source_type = value_type(op.operands[0])
                source_dtype, source_shape = typed(source_type)
                source_contract = _ranked_tensor_contract(source_type)
                target_contract = _ranked_tensor_contract(value_type(value))
                if source_dtype != dtype or source_contract is None or target_contract is None or source_contract[2] or target_contract[2]:
                    raise ValueError("source_owner_transform_layout_unproved")
                suffix = source_type + " -> " + value_type(value)
                if op.name == "tt.reshape":
                    allowed = {
                        _normalize_space("tt.reshape % : " + suffix),
                        _normalize_space("tt.reshape % {allow_reorder = false} : " + suffix),
                    }
                    if op.local_signature not in allowed:
                        raise ValueError("source_owner_reshape_order_unproved")
                    return "reshape", dtype, shape, None, tuple(op.operands)
                match = re.search(r"order=array<i32:([0-9, ]+)>", op.local_signature)
                if match is None:
                    raise ValueError("source_owner_transpose_order_unproved")
                order = tuple(int(axis.strip()) for axis in match[1].split(","))
                expected = _normalize_space("tt.trans % {order = array<i32: " + ", ".join(str(axis) for axis in order) + ">} : " + suffix)
                if op.local_signature != expected or sorted(order) != list(range(len(source_shape))):
                    raise ValueError("source_owner_transpose_order_unproved")
                return "transpose", dtype, shape, order, tuple(op.operands)
            if op.name in {"arith.sitofp", "arith.extf", "arith.truncf"} and len(op.operands) == 1:
                source_type = value_type(op.operands[0])
                bare(op, source_type + " to " + value_type(value))
                source_dtype, source_shape = typed(source_type)
                kind = owner_cast_kind(source_dtype, dtype)
                expected_op = {"signed_to_float_rne": "arith.sitofp", "float_extend": "arith.extf", "float_truncate_rne": "arith.truncf"}[kind]
                if op.name != expected_op or source_shape != shape:
                    raise ValueError("source_owner_cast_semantics_unproved")
                return "cast", dtype, shape, kind, tuple(op.operands)
            if op.name in {"math.exp", "math.rsqrt", "math.tanh"} and len(op.operands) == 1:
                bare(op, value_type(value))
                if value_type(op.operands[0]) != value_type(value):
                    raise ValueError("source_owner_math_type_mismatch")
                return op.name.split(".")[1], dtype, shape, None, tuple(op.operands)
            if op.name == "tt.extern_elementwise":
                external_type = _authorized_extern_rsqrt(op, external_math_symbols)
                if external_type != value_type(value) or len(op.operands) != 1 or value_type(op.operands[0]) != external_type:
                    raise ValueError("source_owner_extern_rsqrt_unproved")
                return (
                    "rsqrt_ftz" if dtype == "f32" and external_math_ftz else "rsqrt"
                ), dtype, shape, None, tuple(op.operands)
            if op.name in arithmetic:
                bare(op, value_type(value))
                if any(value_type(operand) != value_type(value) for operand in op.operands):
                    raise ValueError("source_owner_arithmetic_type_mismatch")
                return arithmetic[op.name], dtype, shape, None, tuple(op.operands)
            if op.name == "tt.reduce":
                if type(op.axis) is not int or len(op.operands) != 1 or len(op.region_block_args) != 2 or len(op.region_results) != 1:
                    raise ValueError("source_owner_reduction_unsupported")
                input_dtype, input_shape = typed(value_type(op.operands[0]))
                if not 0 <= op.axis < len(input_shape) or shape != input_shape[:op.axis] + input_shape[op.axis + 1:]:
                    raise ValueError("source_owner_reduction_axis_unproved")
                combiner = producer(op.region_results[0])
                nested = [child for child in raw if op.node_id in child.context]
                if input_dtype != dtype or combiner.name not in {"arith.addf", "arith.mulf", "arith.maxnumf"} or tuple(combiner.operands) != tuple(op.region_block_args) or combiner.context != (op.node_id,) or combiner.result_types != (dtype,) or len(nested) != 2 or sum(child.name == "tt.reduce.return" for child in nested) != 1:
                    raise ValueError("source_owner_combiner_dataflow_mismatch")
                bare(combiner, dtype)
                # Exact generic region header, including axis and both types.
                expected = _normalize_space("tt.reduce axis=" + str(op.axis) + ":i32 :(" + value_type(op.operands[0]) + ")->(" + value_type(value) + ")")
                if op.local_signature != expected:
                    raise ValueError("source_owner_reduction_attributes_unsupported")
                reduction = {
                    "arith.addf": "sum",
                    "arith.mulf": "prod",
                    "arith.maxnumf": "max",
                }[combiner.name]
                return "reduce", dtype, shape, (op.axis, reduction), tuple(op.operands)
            raise ValueError("source_owner_operation_unsupported")

        proofs: list[Mapping[str, Any]] = []
        for op in raw:
            if op.name != "tt.store":
                continue
            try:
                store = memory(op)
                graph = build_tensor_owner_dag(op.operands[1], describe)
                if not any(row[0] == "load" for row in graph):
                    continue
                root_op = graph[-1][0]
                family = root_op if root_op in {"reduce", "cast", "reshape", "transpose"} else "map_unary" if root_op in {"neg", "exp", "tanh", "rsqrt", "rsqrt_ftz"} else "map_binary"
                proofs.append(dict(store, parameters=tuple(names),
                                   parameter_types=tuple(dtype for _name, dtype, _attrs in function.arguments),
                                   graph=graph, family=family))
            except (ValueError, TypeError, KeyError, RecursionError, InvalidOperation):
                continue
        return tuple(proofs)

    def _scalar_owner_graphs(
        self, text: str, function: _FunctionParts,
        python_interface: Mapping[str, Any] | None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Prove direct store edges on the raw parser graph, before rewrites."""

        names = _owner_runtime_parameters(function, python_interface, ttir=text)
        if names is None:
            return ()  # No ABI-position inference without the compiler receipt.
        raw, results, _leaves, fork = self._parse_operations(function)
        if fork is not None or any(op.context or op.name.startswith(("scf.", "cf.")) for op in raw):
            return ()
        arguments = {name: (index, dtype) for index, (name, dtype, _attrs) in enumerate(function.arguments)}
        nodes = {op.node_id: op for op in raw}
        arithmetic = {"arith.addf": "add", "arith.subf": "sub", "arith.mulf": "mul", "arith.divf": "div", "arith.negf": "neg"}

        def describe(value: str) -> tuple[str, str, Any, Sequence[str]]:
            if value in arguments:
                index, dtype = arguments[value]
                return "input", dtype, index, ()
            op = nodes.get(results.get(value))
            if op is None or len(op.result_names) != 1 or len(op.result_types) != 1:
                raise ValueError("source_owner_producer_unsupported")
            dtype = op.result_types[0]
            if op.name == "arith.constant" and not op.operands:
                literal = re.fullmatch(
                    r"arith\.constant ([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?):(f32|f64)",
                    op.local_signature,
                )
                if literal is None or literal[2] != dtype or op.constant_value is None:
                    raise ValueError("source_owner_literal_unsupported")
                # The observer's float cache is insufficient for exact f32
                # decimal rounding. Admit only literals exactly represented
                # there; do not authorize a double-rounded value.
                try:
                    if Decimal(literal[1]) != Decimal.from_float(op.constant_value):
                        raise ValueError("source_owner_literal_inexact")
                except InvalidOperation as exc:
                    raise ValueError("source_owner_literal_unsupported") from exc
                return "const", dtype, owner_float_bits(op.constant_value, dtype), ()
            if op.name not in arithmetic:
                raise ValueError("source_owner_operation_unsupported")
            # Exact bare op syntax excludes fastmath and all unmodelled attrs.
            expected = op.name + " " + ", ".join("%" for _ in op.operands) + " : " + dtype
            if op.local_signature != _normalize_space(expected):
                raise ValueError("source_owner_attributes_unsupported")
            return arithmetic[op.name], dtype, None, tuple(op.operands)

        proofs: list[Mapping[str, Any]] = []
        for op in raw:
            if op.name != "tt.store" or len(op.operands) not in {2, 3}:
                continue
            # A scalar output at its ABI base, with no inferred coordinates.
            pointer = arguments.get(op.operands[0])
            if pointer is None or _pointer_dtype(pointer[1]) not in {"f32", "f64"}:
                continue
            location = op.source_location
            if location is None or location[0] != op.name:
                continue
            try:
                graph = build_scalar_owner_dag(op.operands[1], describe)
            except (ValueError, TypeError, KeyError, RecursionError):
                continue
            if graph[-1][0] not in {"add", "sub", "mul", "div", "neg"} or graph[-1][1] != _pointer_dtype(pointer[1]):
                continue
            stored_value_producer_id = results.get(op.operands[1])
            stored_value_producer = (
                nodes.get(stored_value_producer_id)
                if stored_value_producer_id is not None
                else None
            )
            if (
                not isinstance(stored_value_producer_id, int)
                or stored_value_producer is None
                or not stored_value_producer.result_types
                or len(stored_value_producer.result_types) != 1
                or stored_value_producer.result_types[0] != graph[-1][1]
            ):
                # The scalar DAG is not enough by itself: the direct TTIR
                # value edge must also be known before this proof can be
                # consumed by Source-aware admission.
                continue
            _operation_name, line, column = location
            proofs.append({
                "line": line, "column": column, "masked": len(op.operands) == 3,
                "call_path": op.source_call_path,
                "kind": "store", "pointer_position": pointer[0],
                "coordinate": 0, "physical_shape": (),
                "predicate": None, "fill": None,
                "dtype": _pointer_dtype(pointer[1]),
                "memory_node_id": op.node_id,
                "stored_value_producer_id": stored_value_producer_id,
                "stored_value_producer_operation": stored_value_producer.name,
                "stored_value_type": stored_value_producer.result_types[0],
                "parameters": tuple(names),
                "parameter_types": tuple(dtype for _name, dtype, _attrs in function.arguments),
                "family": "map_unary" if graph[-1][0] == "neg" else "map_binary",
                "graph": graph,
            })
        return tuple(proofs)

    def _parse_operations(
        self, function: _FunctionParts
    ) -> tuple[
        list[_Operation],
        dict[str, int],
        dict[str, str],
        _TerminalFork | None,
    ]:
        operations: list[_Operation] = []
        result_to_node: dict[str, int] = {}
        leaf_labels = {
            name: f"arg[{index}]:{type_text}"
            for index, (name, type_text, _) in enumerate(function.arguments)
        }
        value_types: dict[str, str | None] = {
            name: type_text for name, type_text, _ in function.arguments
        }
        value_fork_arms: dict[str, str | None] = {
            name: None for name, _, _ in function.arguments
        }
        region_stack: list[tuple[int, int]] = []
        region_block_counts: dict[int, int] = {}
        scan_return_types: dict[int, tuple[str, ...]] = {}
        region_terminator_types: dict[int, tuple[str, ...]] = {}
        structured_if_arms: dict[int, str] = {}
        structured_if_seen_else: set[int] = set()
        structured_if_terminated_arms: set[tuple[int, str]] = set()
        structured_if_terminator_types: dict[
            tuple[int, str], tuple[str, ...]
        ] = {}
        # One lexical binding map for the function and one for every active
        # operation region.  MLIR SSA names need only be unique within their
        # region; sibling scf.for/tt.reduce/tt.scan bodies may legally reuse
        # spellings such as ``%n_offset``.  Internal qualified keys keep the
        # existing fingerprint graph globally addressable without exposing SSA
        # spelling in the final fingerprint.
        scope_bindings: list[dict[str, str]] = [
            {name: name for name, _, _ in function.arguments}
        ]
        has_cf_operation = re.search(
            r'(?m)^\s*(?:"cf\.[A-Za-z0-9_.-]+"|cf\.[A-Za-z0-9_.-]+)\b',
            function.body,
        ) is not None
        if has_cf_operation and function.results:
            raise TTIRParseError(
                "closed terminal binary fork requires a void tt.func"
            )
        if has_cf_operation and re.search(
            r'(?m)^\s*(?:%[^=]+?\s*=\s*)?(?:"tt\.call"|tt\.call)\b',
            function.body,
        ) is not None:
            raise TTIRParseError(
                "a terminal-fork tt.func cannot contain tt.call; CFG helper "
                "interprocedural expansion is unsupported"
            )
        terminal_fork: _TerminalFork | None = None
        fork_targets: dict[str, str] = {}
        active_fork_arm: str | None = None
        seen_fork_arms: set[str] = set()
        terminated_fork_arms: set[str] = set()
        entry_terminated = False

        def qualified_name(name: str, owner_node: int | None) -> str:
            if owner_node is None:
                return name
            arm_suffix = ""
            if (
                owner_node < len(operations)
                and operations[owner_node].name == "scf.if"
            ):
                role = structured_if_arms.get(owner_node)
                if role in {"positive", "negative"}:
                    arm_suffix = f"$a{role}"
            if "#" in name:
                base, suffix = name.rsplit("#", 1)
                return f"{base}$r{owner_node}{arm_suffix}#{suffix}"
            return f"{name}$r{owner_node}{arm_suffix}"

        def define(
            name: str,
            *,
            line_number: int,
            bindings: dict[str, str] | None = None,
            owner_node: int | None = None,
        ) -> str:
            selected = scope_bindings[-1] if bindings is None else bindings
            if name in selected:
                raise TTIRParseError(
                    f"line {line_number}: duplicate SSA definition {name}"
                )
            if owner_node is None and bindings is None and region_stack:
                owner_node = region_stack[-1][0]
            qualified = qualified_name(name, owner_node)
            selected[name] = qualified
            value_fork_arms[qualified] = active_fork_arm
            return qualified

        def resolve(name: str) -> str:
            for bindings in reversed(scope_bindings):
                qualified = bindings.get(name)
                if qualified is not None:
                    return qualified
            # Preserve the original spelling so the existing final unresolved
            # check emits a deterministic parse error with useful context.
            return name

        def enclosing_structured_if() -> tuple[int, str] | None:
            for owner_id, _ in reversed(region_stack):
                if operations[owner_id].name == "scf.if":
                    role = structured_if_arms.get(owner_id)
                    if role not in {"positive", "negative"}:
                        raise TTIRParseError(
                            "active scf.if has no well-formed arm polarity"
                        )
                    return owner_id, role
            return None

        def close_scan_region(
            owner_id: int, closing_text: str, *, line_number: int
        ) -> None:
            operation = operations[owner_id]
            if not closing_text.startswith("})"):
                raise TTIRParseError(
                    f"line {line_number}: tt.scan must close its quoted-generic region"
                )
            suffix = closing_text[2:].strip()
            if not suffix.startswith(":"):
                raise TTIRParseError(
                    f"line {line_number}: tt.scan is missing its explicit function type"
                )
            declared_inputs, declared_results = _functional_types(
                suffix[1:], context="tt.scan"
            )
            input_count = len(operation.operands)
            if input_count < 1:
                raise TTIRParseError("tt.scan must have at least one input")
            if len(operation.result_names) != input_count:
                raise TTIRParseError(
                    "tt.scan must have the same number of SSA results as inputs"
                )
            if len(declared_inputs) != input_count:
                raise TTIRParseError(
                    "tt.scan explicit input type arity does not match its operands"
                )
            if len(declared_results) != input_count:
                raise TTIRParseError(
                    "tt.scan explicit result type arity does not match its inputs"
                )

            actual_inputs = tuple(value_types.get(name) for name in operation.operands)
            if any(type_text is None for type_text in actual_inputs):
                raise TTIRParseError("tt.scan input SSA type cannot be established")
            normalized_actual_inputs = tuple(
                _normalize_space(type_text)
                for type_text in actual_inputs
                if type_text is not None
            )
            if declared_inputs != normalized_actual_inputs:
                raise TTIRParseError(
                    "tt.scan explicit input types do not match actual SSA types"
                )
            if declared_results != declared_inputs:
                raise TTIRParseError(
                    "tt.scan result type must match the corresponding input type"
                )

            tensor_contracts = tuple(
                _ranked_tensor_contract(type_text)
                for type_text in declared_inputs
            )
            if any(contract is None for contract in tensor_contracts):
                raise TTIRParseError("tt.scan inputs and results must be ranked tensors")
            checked_contracts = tuple(
                contract for contract in tensor_contracts if contract is not None
            )
            first_shape, _, first_encoding = checked_contracts[0]
            if any(
                shape != first_shape or encoding != first_encoding
                for shape, _, encoding in checked_contracts[1:]
            ):
                raise TTIRParseError(
                    "tt.scan inputs and results must have one common shape and encoding"
                )
            if operation.axis is None:
                raise TTIRParseError("tt.scan is missing its axis property")
            if not 0 <= operation.axis < len(first_shape):
                raise TTIRParseError(
                    f"tt.scan axis {operation.axis} is outside rank {len(first_shape)}"
                )
            if operation.reverse is None:
                raise TTIRParseError("tt.scan is missing its reverse property")

            if region_block_counts.get(owner_id, 0) != 1:
                raise TTIRParseError("tt.scan must contain exactly one combine block")
            expected_block_types = tuple(
                [contract[1] for contract in checked_contracts] * 2
            )
            actual_block_types = tuple(
                value_types.get(name) for name in operation.region_block_args
            )
            if len(actual_block_types) != 2 * input_count:
                raise TTIRParseError(
                    "tt.scan combine block must take exactly 2N ordered arguments"
                )
            if actual_block_types != expected_block_types:
                raise TTIRParseError(
                    "tt.scan combine block argument types must be ordered lhs[0:N], "
                    "rhs[0:N]"
                )

            declared_returns = scan_return_types.get(owner_id)
            if declared_returns is None:
                raise TTIRParseError(
                    "tt.scan combine block must end with exactly one tt.scan.return"
                )
            if len(operation.region_results) != input_count:
                raise TTIRParseError(
                    "tt.scan.return must return exactly one value per scan input"
                )
            actual_return_types = tuple(
                value_types.get(name) for name in operation.region_results
            )
            expected_return_types = tuple(
                contract[1] for contract in checked_contracts
            )
            if any(type_text is None for type_text in actual_return_types):
                raise TTIRParseError(
                    "tt.scan.return operand SSA type cannot be established"
                )
            if declared_returns != expected_return_types:
                raise TTIRParseError(
                    "tt.scan.return explicit types do not match scan element types"
                )
            if actual_return_types != expected_return_types:
                raise TTIRParseError(
                    "tt.scan.return actual SSA types do not match scan element types"
                )

            for result, type_text in zip(
                operation.result_names, declared_results, strict=True
            ):
                value_types[result] = type_text
            operation.local_signature = _normalize_space(
                "tt.scan "
                f"axis={operation.axis}:i32 "
                f"reverse={'true' if operation.reverse else 'false'} "
                f":({','.join(declared_inputs)})->"
                f"({','.join(declared_results)})"
            )

        def close_reduce_region(
            owner_id: int, closing_text: str, *, line_number: int
        ) -> None:
            operation = operations[owner_id]
            if not closing_text.startswith("})"):
                raise TTIRParseError(
                    f"line {line_number}: tt.reduce must close its "
                    "quoted-generic region"
                )
            suffix = closing_text[2:].strip()
            if not suffix.startswith(":"):
                raise TTIRParseError(
                    f"line {line_number}: tt.reduce is missing its explicit "
                    "function type"
                )
            declared_inputs, declared_results = _functional_types(
                suffix[1:], context="tt.reduce"
            )
            input_count = len(operation.operands)
            if input_count < 1 or len(operation.result_names) != input_count:
                raise TTIRParseError(
                    "tt.reduce must have one SSA result per input"
                )
            if len(declared_inputs) != input_count:
                raise TTIRParseError(
                    "tt.reduce explicit input type arity does not match its operands"
                )
            if len(declared_results) != input_count:
                raise TTIRParseError(
                    "tt.reduce explicit result type arity does not match its inputs"
                )
            actual_inputs = tuple(value_types.get(name) for name in operation.operands)
            if any(type_text is None for type_text in actual_inputs):
                raise TTIRParseError("tt.reduce input SSA type cannot be established")
            if declared_inputs != tuple(actual_inputs):
                raise TTIRParseError(
                    "tt.reduce explicit input types do not match actual SSA types"
                )
            tensor_contracts = tuple(
                _ranked_tensor_contract(type_text) for type_text in declared_inputs
            )
            if any(contract is None for contract in tensor_contracts):
                raise TTIRParseError("tt.reduce inputs must be ranked tensors")
            contracts = tuple(
                contract for contract in tensor_contracts if contract is not None
            )
            first_shape, _, first_encoding = contracts[0]
            if any(
                shape != first_shape or encoding != first_encoding
                for shape, _, encoding in contracts[1:]
            ):
                raise TTIRParseError(
                    "tt.reduce inputs must have one common shape and encoding"
                )
            if operation.axis is None or not 0 <= operation.axis < len(first_shape):
                raise TTIRParseError(
                    f"tt.reduce axis {operation.axis} is outside rank {len(first_shape)}"
                )
            element_types = tuple(contract[1] for contract in contracts)
            expected_result_types = tuple(
                (
                    contract[1]
                    if len(contract[0]) == 1
                    else "tensor<"
                    + "x".join(
                        [
                            *(
                                extent
                                for index, extent in enumerate(contract[0])
                                if index != operation.axis
                            ),
                            contract[1],
                        ]
                    )
                    + ">"
                )
                for contract in contracts
            )
            if declared_results != expected_result_types:
                raise TTIRParseError(
                    "tt.reduce result types must remove exactly its reduction axis"
                )
            if region_block_counts.get(owner_id, 0) != 1:
                raise TTIRParseError("tt.reduce must contain exactly one combine block")
            block_types = tuple(
                value_types.get(name) for name in operation.region_block_args
            )
            if block_types != tuple([*element_types, *element_types]):
                raise TTIRParseError(
                    "tt.reduce combine block arguments must be ordered "
                    "lhs[0:N], rhs[0:N]"
                )
            terminator_types = region_terminator_types.get(owner_id)
            if terminator_types is None:
                raise TTIRParseError(
                    "tt.reduce combine block must end with exactly one "
                    "tt.reduce.return"
                )
            if len(operation.region_results) != input_count:
                raise TTIRParseError(
                    "tt.reduce.return must return exactly one value per input"
                )
            if terminator_types != element_types:
                raise TTIRParseError(
                    "tt.reduce.return explicit types do not match reduce "
                    "element types"
                )
            actual_return_types = tuple(
                value_types.get(name) for name in operation.region_results
            )
            if actual_return_types != element_types:
                raise TTIRParseError(
                    "tt.reduce.return actual SSA types do not match reduce "
                    "element types"
                )
            for result, type_text in zip(
                operation.result_names, declared_results, strict=True
            ):
                value_types[result] = type_text
            operation.local_signature = _normalize_space(
                "tt.reduce "
                f"axis={operation.axis}:i32 "
                f":({','.join(declared_inputs)})->"
                f"({','.join(declared_results)})"
            )
            operation.result_types = declared_results

        def close_for_region(
            owner_id: int, closing_text: str, *, line_number: int
        ) -> None:
            closing = closing_text.strip()
            if not closing.startswith("}"):
                raise TTIRParseError(
                    f"line {line_number}: scf.for has malformed closing syntax"
                )
            operation = operations[owner_id]
            suffix = closing[1:].strip()
            if suffix:
                if not suffix.startswith("{"):
                    raise TTIRParseError(
                        f"line {line_number}: scf.for has malformed closing syntax"
                    )
                attr_close = _balanced_end(suffix, 0, "{", "}")
                if suffix[attr_close + 1 :].strip():
                    raise TTIRParseError(
                        f"line {line_number}: scf.for has trailing closing syntax"
                    )
                close_attrs = _validate_attribute_dictionary(
                    suffix[: attr_close + 1],
                    context="scf.for closing attributes",
                )
                validated_close_attrs = _validate_scf_for_closing_attributes(
                    close_attrs,
                    context="scf.for closing attributes",
                )
                # ``tt.divisibility_arg*`` is a Triton compiler hint.  It is
                # deliberately validated above so malformed metadata cannot
                # pass as TTIR, but it must not become a hard J-vs-Source
                # semantic fingerprint.  Keep the pre-existing num_stages
                # behavior for scheduling metadata compatibility.
                semantic_close_attrs = tuple(
                    (name, value)
                    for name, value in validated_close_attrs
                    if name == "tt.num_stages"
                )
                if semantic_close_attrs:
                    operation.local_signature = _normalize_space(
                        operation.local_signature
                        + " close_attrs={"
                        + ",".join(
                            f"{name}={value}"
                            for name, value in semantic_close_attrs
                        )
                        + "}"
                    )
            if region_block_counts.get(owner_id, 0) != 1:
                raise TTIRParseError("scf.for must contain exactly one body block")
            init_count = max(0, len(operation.operands) - 3)
            if owner_id not in region_terminator_types and init_count:
                raise TTIRParseError(
                    "scf.for body must end with exactly one scf.yield"
                )
            init_types = tuple(
                value_types.get(name) for name in operation.operands[3:]
            )
            if any(type_text is None for type_text in init_types):
                raise TTIRParseError(
                    "scf.for iter_arg input SSA type cannot be established"
                )
            if len(operation.region_block_args) != 1 + len(init_types):
                raise TTIRParseError(
                    "scf.for block arguments must contain one induction value "
                    "and one value per iter_arg"
                )
            block_types = tuple(
                value_types.get(name) for name in operation.region_block_args[1:]
            )
            result_types = tuple(
                value_types.get(name) for name in operation.result_names
            )
            yield_types = region_terminator_types.get(owner_id, ())
            actual_yield_types = tuple(
                value_types.get(name) for name in operation.region_results
            )
            if not (
                block_types == init_types
                and result_types == init_types
                and yield_types == init_types
                and actual_yield_types == init_types
                and len(operation.region_results) == len(init_types)
            ):
                raise TTIRParseError(
                    "scf.for iter_args, results, block arguments, and yield "
                    "types must close positionally"
                )

        def close_if_region(
            owner_id: int, closing_text: str, *, line_number: int
        ) -> None:
            if closing_text.strip() != "}":
                raise TTIRParseError(
                    f"line {line_number}: scf.if has malformed closing syntax"
                )
            operation = operations[owner_id]
            if structured_if_arms.get(owner_id) not in {
                "positive",
                "negative",
            }:
                raise TTIRParseError("scf.if closed without an active arm")
            if not operation.result_names:
                if operation.region_results:
                    raise TTIRParseError(
                        "resultless scf.if must not yield SSA values"
                    )
                return
            if len(operation.result_names) != 1:
                raise TTIRParseError(
                    "value-producing scf.if requires exactly one SSA result"
                )
            if owner_id not in structured_if_seen_else:
                raise TTIRParseError(
                    "value-producing scf.if requires explicit positive and "
                    "negative arms"
                )
            expected_keys = {
                (owner_id, "positive"),
                (owner_id, "negative"),
            }
            if not expected_keys.issubset(structured_if_terminated_arms):
                raise TTIRParseError(
                    "each value-producing scf.if arm must end in exactly one "
                    "scf.yield"
                )
            if len(operation.region_results) != 2:
                raise TTIRParseError(
                    "value-producing scf.if must have one yielded value per arm"
                )
            result_type = operation.structured_if_result_type
            if result_type is None:
                raise TTIRParseError(
                    "value-producing scf.if lost its explicit result type"
                )
            for role, yielded in zip(
                ("positive", "negative"),
                operation.region_results,
                strict=True,
            ):
                declared = structured_if_terminator_types[(owner_id, role)]
                if declared != (result_type,):
                    raise TTIRParseError(
                        f"scf.if {role} yield type must match its result type"
                    )
                if value_types.get(yielded) != result_type:
                    raise TTIRParseError(
                        f"scf.if {role} yielded SSA type must match its result type"
                    )
            value_types[operation.result_names[0]] = result_type

        def close_region(
            owner_id: int, closing_text: str, *, line_number: int
        ) -> None:
            owner = operations[owner_id]
            if owner.name == "tt.scan":
                close_scan_region(
                    owner_id, closing_text, line_number=line_number
                )
            elif owner.name == "tt.reduce":
                close_reduce_region(
                    owner_id, closing_text, line_number=line_number
                )
            elif owner.name == "scf.for":
                close_for_region(
                    owner_id, closing_text, line_number=line_number
                )
            elif owner.name == "scf.if":
                close_if_region(
                    owner_id, closing_text, line_number=line_number
                )
            else:  # pragma: no cover - region owner set is closed above
                raise TTIRParseError(f"unsupported region owner {owner.name}")

        for line_number, line in enumerate(function.body.splitlines(), 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if not stripped or stripped.startswith("//"):
                continue
            if re.fullmatch(r"}\s*else\s*{", stripped) is not None:
                if not region_stack:
                    raise TTIRParseError(
                        f"line {line_number}: scf.if else appears outside its region"
                    )
                owner_id, owner_indent = region_stack[-1]
                owner = operations[owner_id]
                if owner.name != "scf.if" or indent != owner_indent:
                    raise TTIRParseError(
                        f"line {line_number}: scf.if else must directly switch "
                        "the active single-block positive arm"
                    )
                if owner_id in structured_if_seen_else:
                    raise TTIRParseError("scf.if has more than one negative arm")
                if structured_if_arms.get(owner_id) != "positive":
                    raise TTIRParseError(
                        "scf.if else can only follow its positive arm"
                    )
                structured_if_seen_else.add(owner_id)
                structured_if_arms[owner_id] = "negative"
                scope_bindings.pop()
                scope_bindings.append({})
                continue
            cond_br = _TERMINAL_COND_BR_RE.fullmatch(stripped)
            if cond_br is not None:
                if not has_cf_operation or terminal_fork is not None:
                    raise TTIRParseError(
                        f"line {line_number}: terminal binary fork must have "
                        "exactly one cf.cond_br"
                    )
                if region_stack or active_fork_arm is not None:
                    raise TTIRParseError(
                        f"line {line_number}: cf.cond_br must terminate the "
                        "function entry block"
                    )
                true_label = cond_br.group("true")
                false_label = cond_br.group("false")
                if true_label == false_label:
                    raise TTIRParseError(
                        "terminal binary fork requires two distinct successor arms"
                    )
                predicate = resolve(cond_br.group("predicate"))
                if value_types.get(predicate) != "i1":
                    raise TTIRParseError(
                        "terminal binary fork predicate must be a scalar i1 SSA value"
                    )
                terminal_fork = _TerminalFork(
                    0, predicate, (), len(operations)
                )
                fork_targets = {
                    true_label: "true",
                    false_label: "false",
                }
                entry_terminated = True
                continue
            if stripped.startswith(('cf.', '"cf.')):
                raise TTIRParseError(
                    f"line {line_number}: only the exact operand-free "
                    "cf.cond_br terminal binary-fork shell is supported"
                )
            if stripped.startswith(("})", "}")):
                if not region_stack:
                    raise TTIRParseError(
                        f"line {line_number}: unexpected structured-region close"
                    )
                first_closed = True
                closed_any = False
                while region_stack and indent <= region_stack[-1][1]:
                    owner_id = region_stack[-1][0]
                    close_region(
                        owner_id,
                        stripped if first_closed else "}",
                        line_number=line_number,
                    )
                    region_stack.pop()
                    scope_bindings.pop()
                    first_closed = False
                    closed_any = True
                if not closed_any:
                    raise TTIRParseError(
                        f"line {line_number}: region close indentation does not "
                        "match an active owner"
                    )
                continue

            block_header = _parse_block_header(line)
            if block_header is not None:
                if not region_stack:
                    label, block_arguments = block_header
                    if terminal_fork is None or not entry_terminated:
                        raise TTIRParseError(
                            f"line {line_number}: top-level block label "
                            f"^{label} is unsupported; Gate1 TTIR v7 only "
                            "accepts the closed terminal binary-fork profile"
                        )
                    role = fork_targets.get(label)
                    if role is None:
                        raise TTIRParseError(
                            f"line {line_number}: unexpected terminal-fork "
                            f"block ^{label}"
                        )
                    if block_arguments:
                        raise TTIRParseError(
                            "terminal-fork successor blocks cannot take block "
                            "arguments"
                        )
                    if role in seen_fork_arms:
                        raise TTIRParseError(
                            f"terminal-fork {role} arm is declared more than once"
                        )
                    if (
                        active_fork_arm is not None
                        and active_fork_arm not in terminated_fork_arms
                    ):
                        raise TTIRParseError(
                            f"terminal-fork {active_fork_arm} arm does not end "
                            "in void tt.return"
                        )
                    active_fork_arm = role
                    seen_fork_arms.add(role)
                    continue
                block_args = [name for name, _ in block_header[1]]
                block_arg_types = [type_text for _, type_text in block_header[1]]
                parent_id = region_stack[-1][0]
                parent = operations[parent_id]
                if parent.name == "scf.if":
                    raise TTIRParseError(
                        f"line {line_number}: scf.if arms must use the implicit "
                        "single block and cannot declare a block header"
                    )
                if region_block_counts.get(parent_id, 0) >= 1:
                    raise TTIRParseError(
                        f"line {line_number}: structured region "
                        f"{parent.name} has more than one block; Gate1 TTIR "
                        "v7 supports only single-block structured regions"
                    )
                region_block_counts[parent_id] = 1
                qualified_args = [
                    define(name, line_number=line_number)
                    for name in block_args
                ]
                for name, type_text in zip(
                    qualified_args, block_arg_types, strict=True
                ):
                    value_types[name] = type_text
                parent.region_block_args.extend(qualified_args)
                path = "/".join(operations[node].name for node, _ in region_stack)
                for index, name in enumerate(qualified_args):
                    leaf_labels[name] = f"region[{path}].arg[{index}]"
                continue

            if entry_terminated and active_fork_arm is None:
                raise TTIRParseError(
                    f"line {line_number}: only the two cf.cond_br target "
                    "labels may follow the entry terminator"
                )
            if active_fork_arm in terminated_fork_arms:
                raise TTIRParseError(
                    f"line {line_number}: operation after terminal-fork "
                    f"{active_fork_arm} arm tt.return"
                )

            if region_stack and region_stack[-1][0] in region_terminator_types:
                owner = operations[region_stack[-1][0]]
                raise TTIRParseError(
                    f"line {line_number}: operation after {owner.name} terminator"
                )
            if region_stack and operations[region_stack[-1][0]].name == "scf.if":
                owner_id = region_stack[-1][0]
                role = structured_if_arms.get(owner_id)
                if role is not None and (
                    owner_id, role
                ) in structured_if_terminated_arms:
                    raise TTIRParseError(
                        f"line {line_number}: operation after scf.if {role} "
                        "arm scf.yield"
                    )

            head = _operation_head(line)
            if head is None:
                raise TTIRParseError(
                    f"line {line_number}: cannot parse non-empty TTIR body "
                    f"line: {stripped!r}"
                )
            results, name, tail, quoted = head
            enclosing_if_at_line = enclosing_structured_if()
            if (
                enclosing_if_at_line is not None
                and operations[enclosing_if_at_line[0]].result_names
                and name in _REGION_OWNERS
            ):
                raise TTIRParseError(
                    "value-producing scf.if arms cannot contain a nested region"
                )
            if has_cf_operation and name == "tt.call":
                raise TTIRParseError(
                    "a terminal-fork tt.func cannot contain tt.call; CFG "
                    "helper interprocedural expansion is unsupported"
                )
            if has_cf_operation and name == "tt.return":
                if (
                    terminal_fork is None
                    or region_stack
                    or active_fork_arm is None
                ):
                    raise TTIRParseError(
                        "terminal binary fork permits tt.return only as the "
                        "top-level terminator of an arm"
                    )
            if (
                name not in _REGION_OWNERS
                and stripped.endswith("{")
            ):
                raise TTIRParseError(
                    f"line {line_number}: unsupported region-bearing TTIR "
                    f"operation {name!r}; Gate1 refuses to flatten an unknown "
                    "SSA scope"
                )
            if quoted and name in _TERMINATORS:
                raise TTIRParseError(
                    f"{name} must use Triton 3.6 custom terminator syntax"
                )
            generic_contract: tuple[
                tuple[str, ...], tuple[str, ...], tuple[str, ...]
            ] | None = None
            if quoted and name not in _REGION_OWNERS:
                generic_contract = _validate_quoted_generic_shell(
                    name,
                    tail,
                    result_count=len(results),
                )
            raw_operands = (
                list(generic_contract[0])
                if generic_contract is not None
                else _SSA_RE.findall(tail)
            )
            raw_block_args: list[str] = []
            scan_axis: int | None = None
            scan_reverse: bool | None = None
            for_induction_type: str | None = None
            for_result_types: tuple[str, ...] = ()
            if_result_type: str | None = None
            parsed_terminator_types: tuple[str, ...] = ()
            if name == "scf.for":
                (
                    raw_operands,
                    raw_block_args,
                    for_induction_type,
                    for_result_types,
                ) = _parse_for_header(tail)
            elif name == "scf.if":
                if enclosing_structured_if() is not None:
                    raise TTIRParseError(
                        "nested scf.if is unsupported by the Gate1 v10 closed profile"
                    )
                predicate, if_result_type = _parse_closed_if_header(
                    tail, result_count=len(results)
                )
                raw_operands = [predicate]
            elif name == "tt.reduce":
                if not quoted:
                    raise TTIRParseError(
                        "tt.reduce outer operation must use Triton 3.6 "
                        "quoted-generic syntax"
                    )
                raw_operands, scan_axis = _parse_reduce_header(tail)
                if len(results) != len(raw_operands):
                    raise TTIRParseError(
                        "tt.reduce must have the same number of SSA results as inputs"
                    )
            elif name == "tt.scan":
                if not quoted:
                    raise TTIRParseError(
                        "tt.scan outer operation must use Triton 3.6 quoted-generic syntax"
                    )
                raw_operands, scan_axis, scan_reverse = _parse_scan_header(tail)
                if len(results) != len(raw_operands):
                    raise TTIRParseError(
                        "tt.scan must have the same number of SSA results as inputs"
                    )
            elif name == "tt.scan.return":
                raw_operands, parsed_terminator_types = _parse_scan_return(tail)
            elif name == "tt.reduce.return":
                raw_operands, parsed_terminator_types = _parse_typed_terminator(
                    name, tail, allow_empty=False
                )
            elif name == "scf.yield":
                raw_operands, parsed_terminator_types = _parse_typed_terminator(
                    name, tail, allow_empty=True
                )
            elif name == "tt.return":
                raw_operands, parsed_terminator_types = _parse_typed_terminator(
                    name, tail, allow_empty=True
                )
            callee: str | None = None
            call_operand_types: tuple[str, ...] = ()
            call_result_types: tuple[str, ...] = ()
            if name == "tt.call":
                callee, call_operand_types, call_result_types = (
                    _parse_call_contract(tail)
                )
            operands = [resolve(operand) for operand in raw_operands]
            active_if_for_operands = enclosing_structured_if()
            if active_if_for_operands is not None:
                owner = operations[active_if_for_operands[0]]
                if any(operand in owner.result_names for operand in operands):
                    raise TTIRParseError(
                        "scf.if result does not dominate either of its arms"
                    )
            if active_fork_arm is not None:
                for operand in operands:
                    if operand not in value_fork_arms:
                        raise TTIRParseError(
                            f"line {line_number}: SSA value {operand} does "
                            "not dominate its terminal-fork arm use"
                        )
                    defining_arm = value_fork_arms.get(operand)
                    if defining_arm not in {None, active_fork_arm}:
                        raise TTIRParseError(
                            f"line {line_number}: terminal-fork "
                            f"{active_fork_arm} arm uses SSA value from its "
                            f"{defining_arm} sibling"
                        )
            operand_types = tuple(value_types.get(operand) for operand in operands)
            if generic_contract is not None:
                declared_generic_inputs = generic_contract[1]
                if len(operand_types) != len(declared_generic_inputs):
                    raise TTIRParseError(
                        f'quoted generic operation "{name}" operand/type arity mismatch'
                    )
                for index, (actual, declared) in enumerate(
                    zip(operand_types, declared_generic_inputs, strict=True)
                ):
                    if actual is None:
                        raise TTIRParseError(
                            f'quoted generic operation "{name}" operand {index} '
                            "actual SSA type cannot be established"
                        )
                    if _normalize_space(actual) != declared:
                        raise TTIRParseError(
                            f'quoted generic operation "{name}" operand {index} '
                            f"actual SSA type {actual} does not match {declared}"
                        )
            if name == "tt.call":
                if len(operand_types) != len(call_operand_types):
                    raise TTIRParseError("tt.call operand SSA/type arity mismatch")
                for index, (actual, declared) in enumerate(
                    zip(operand_types, call_operand_types, strict=True)
                ):
                    if actual is None:
                        raise TTIRParseError(
                            f"tt.call operand {index} actual SSA type cannot "
                            "be established"
                        )
                    if actual != declared:
                        raise TTIRParseError(
                            f"tt.call operand {index} actual SSA type {actual} "
                            f"does not match its explicit call type {declared}"
                        )
            if name == "scf.for":
                assert for_induction_type is not None
                if any(
                    actual is None or _normalize_space(actual) != for_induction_type
                    for actual in operand_types[:3]
                ):
                    raise TTIRParseError(
                        "scf.for lower/upper/step SSA types must match its "
                        "explicit induction type"
                    )
                init_types = tuple(operand_types[3:])
                if any(type_text is None for type_text in init_types):
                    raise TTIRParseError(
                        "scf.for iter_arg input SSA type cannot be established"
                    )
                if tuple(init_types) != for_result_types:
                    raise TTIRParseError(
                        "scf.for iter_arg input types must match its explicit "
                        "result types"
                    )
                if len(results) != len(for_result_types):
                    raise TTIRParseError(
                        "scf.for SSA result arity must match its iter_args"
                    )
            if name == "scf.if":
                if len(operand_types) != 1 or operand_types[0] != "i1":
                    raise TTIRParseError(
                        "scf.if predicate must be a scalar i1 SSA value"
                    )
            if name == "tt.return":
                declared_returns = parsed_terminator_types
                if declared_returns != function.results:
                    raise TTIRParseError(
                        f"tt.return explicit types {declared_returns} do not "
                        f"match tt.func @{function.name} results "
                        f"{function.results}"
                    )
                for index, (actual, declared) in enumerate(
                    zip(operand_types, declared_returns, strict=True)
                ):
                    if actual is None:
                        raise TTIRParseError(
                            f"tt.return operand {index} actual SSA type cannot "
                            "be established"
                        )
                    if actual != declared:
                        raise TTIRParseError(
                            f"tt.return operand {index} actual SSA type "
                            f"{actual} does not match its explicit return "
                            f"type {declared}"
                        )
            qualified_results = tuple(
                define(result, line_number=line_number) for result in results
            )
            inferred_result_types = _operation_result_types(
                name,
                tail,
                len(qualified_results),
                operand_types,
                call_result_types=call_result_types,
            )
            if name == "scf.if":
                inferred_result_types = (
                    (if_result_type,) if if_result_type is not None else ()
                )
            for result, type_text in zip(
                qualified_results, inferred_result_types, strict=True
            ):
                value_types[result] = type_text
            axis_match = re.search(r"\baxis\s*=\s*(-?[0-9]+)", tail)
            enclosing_if = enclosing_structured_if()
            operation = _Operation(
                node_id=len(operations),
                name=name,
                source_location=function.source_locations[line_number][:3] if line_number in function.source_locations else None,
                source_call_path=function.source_locations[line_number][3] if line_number in function.source_locations else (),
                result_names=qualified_results,
                operands=list(operands),
                local_signature=_local_signature(name, tail),
                context=tuple(node for node, _ in region_stack),
                result_types=tuple(inferred_result_types),
                axis=(
                    scan_axis
                    if name == "tt.scan"
                    else int(axis_match.group(1)) if axis_match else None
                ),
                reverse=scan_reverse,
                cmp_predicate=_cmp_info(name, tail)[0],
                cmp_flipped=_cmp_info(name, tail)[1],
                header_operand_count=len(operands),
                region_block_args=[],
                constant_value=(
                    _scalar_constant(tail)
                    if name == "arith.constant"
                    else None
                ),
                callee=callee,
                call_operand_types=call_operand_types,
                call_result_types=call_result_types,
                terminator_types=parsed_terminator_types,
                fork_id=(
                    terminal_fork.fork_id
                    if active_fork_arm is not None
                    else None
                ),
                fork_arm=active_fork_arm,
                structured_if_id=(
                    enclosing_if[0] if enclosing_if is not None else None
                ),
                structured_if_arm=(
                    enclosing_if[1] if enclosing_if is not None else None
                ),
                structured_if_result_type=(
                    if_result_type if name == "scf.if" else None
                ),
            )
            operations.append(operation)
            for result in qualified_results:
                if result in result_to_node or result in leaf_labels:
                    raise TTIRParseError(
                        f"line {line_number}: duplicate internal SSA key {result}"
                    )
                result_to_node[result] = operation.node_id

            if name in _REGION_TERMINATORS:
                if not region_stack:
                    raise TTIRParseError(
                        f"line {line_number}: {name} appears outside a structured region"
                    )
                parent_id = region_stack[-1][0]
                parent = operations[parent_id]
                expected = _EXPECTED_REGION_TERMINATOR[parent.name]
                if name != expected:
                    raise TTIRParseError(
                        f"line {line_number}: {parent.name} requires {expected}, got {name}"
                    )
                if results:
                    raise TTIRParseError(
                        f"line {line_number}: region terminator {name} cannot define SSA results"
                    )
                if parent.name == "scf.if":
                    role = structured_if_arms.get(parent_id)
                    if role not in {"positive", "negative"}:
                        raise TTIRParseError(
                            "scf.if yield has no active arm polarity"
                        )
                    key = (parent_id, role)
                    if key in structured_if_terminated_arms:
                        raise TTIRParseError(
                            f"scf.if {role} arm has more than one scf.yield"
                        )
                    if parent.result_names:
                        if (
                            len(operands) != 1
                            or len(parsed_terminator_types) != 1
                        ):
                            raise TTIRParseError(
                                "value-producing scf.if requires exactly one "
                                "typed yielded SSA value per arm"
                            )
                        structured_if_terminator_types[key] = parsed_terminator_types
                        parent.region_results.extend(operands)
                    elif operands or parsed_terminator_types:
                        raise TTIRParseError(
                            "effect-only scf.if requires a zero-operand scf.yield"
                        )
                    structured_if_terminated_arms.add(key)
                elif parent_id in region_terminator_types:
                    raise TTIRParseError(
                        f"line {line_number}: {parent.name} has more than one "
                        f"{expected}"
                    )
                else:
                    region_terminator_types[parent_id] = parsed_terminator_types
                    if parent.name == "tt.scan":
                        scan_return_types[parent_id] = parsed_terminator_types
                    parent.region_results.extend(operands)

            if name == "tt.return" and terminal_fork is not None:
                assert active_fork_arm is not None
                if results or operands or parsed_terminator_types:
                    raise TTIRParseError(
                        "terminal-fork arms must end in an empty void tt.return"
                    )
                terminated_fork_arms.add(active_fork_arm)

            if name in _REGION_OWNERS:
                path = "/".join(
                    [*(operations[node].name for node, _ in region_stack), name]
                )
                child_bindings: dict[str, str] = {}
                qualified_block_args = [
                    define(
                        block_name,
                        line_number=line_number,
                        bindings=child_bindings,
                        owner_node=operation.node_id,
                    )
                    for block_name in raw_block_args
                ]
                raw_block_arg_types = (
                    [
                        operand_types[0] if operand_types else None,
                        *operand_types[3:],
                    ]
                    if name == "scf.for"
                    else []
                )
                if len(raw_block_arg_types) != len(qualified_block_args):
                    raise TTIRParseError(
                        "scf.for block argument/type arity mismatch"
                    )
                for block_name, type_text in zip(
                    qualified_block_args, raw_block_arg_types, strict=True
                ):
                    value_types[block_name] = type_text
                operation.region_block_args.extend(qualified_block_args)
                for index, block_name in enumerate(qualified_block_args):
                    leaf_labels[block_name] = f"region[{path}].arg[{index}]"
                region_stack.append((operation.node_id, indent))
                scope_bindings.append(child_bindings)
                if name == "scf.if":
                    structured_if_arms[operation.node_id] = "positive"
                # Custom ``scf.for`` assembly declares its one body block via
                # the induction variable/iter_args in the operation header.
                # ``tt.reduce``/``tt.scan`` declare their block on the
                # following ^bb line.
                region_block_counts[operation.node_id] = (
                    1 if raw_block_args else 0
                )

        if region_stack:
            raise TTIRParseError("unterminated operation region")
        if has_cf_operation:
            if terminal_fork is None:
                raise TTIRParseError(
                    "reachable tt.func uses control flow outside the closed "
                    "terminal binary-fork profile"
                )
            if seen_fork_arms != {"true", "false"}:
                raise TTIRParseError(
                    "terminal binary fork must declare exactly its true and "
                    "false successor blocks"
                )
            if terminated_fork_arms != {"true", "false"}:
                raise TTIRParseError(
                    "each terminal-fork arm must end in an empty void tt.return"
                )
        for operation in operations:
            for operand in [*operation.operands, *operation.region_results]:
                if operand not in result_to_node and operand not in leaf_labels:
                    raise TTIRParseError(
                        f"unresolved SSA value {operand!r} used by {operation.name}"
                    )
        return operations, result_to_node, leaf_labels, terminal_fork

    def _expand_reachable_calls(
        self,
        public: _FunctionParts,
        functions: Mapping[str, _FunctionParts],
    ) -> tuple[
        list[_Operation],
        dict[str, int],
        dict[str, str],
        list[_TerminalFork],
    ]:
        """Inline the reachable acyclic ``tt.call`` graph into one SSA graph.

        This is a fingerprint-layer canonicalization, not a compiler rewrite:
        both Source and Replay pass through it and the public function ABI is
        kept verbatim.  Private function names, private argument attributes,
        and the frontend's choice to outline or inline are therefore erased,
        while every reachable operation and memory effect remains observable.

        The current TTIR observer models structured
        ``scf.for``/``tt.reduce``/``tt.scan`` regions and one closed terminal
        binary-fork shape.  The latter is expanded as a synthetic conditional
        control member whose true/false roles remain distinct; its arms are
        never flattened into an unconditional sequence.  Arbitrary CFGs,
        joins, phi/block arguments, backedges, and calls from a CFG helper
        remain unsupported.  Unreachable helpers are deliberately never
        parsed and cannot perturb the fingerprint.
        """

        parsed: dict[
            str,
            tuple[
                list[_Operation],
                dict[str, int],
                dict[str, str],
                _TerminalFork | None,
            ],
        ] = {}
        expanded: list[_Operation] = []
        expanded_results: dict[str, int] = {}
        expanded_terminal_forks: list[_TerminalFork] = []
        leaf_labels = {
            name: f"arg[{index}]:{type_text}"
            for index, (name, type_text, _) in enumerate(public.arguments)
        }
        value_serial = 0
        fork_serial = 0

        def fresh_value() -> str:
            nonlocal value_serial
            value = f"%__gate1_ip{value_serial}"
            value_serial += 1
            return value

        def parse_reachable(
            function: _FunctionParts,
        ) -> tuple[
            list[_Operation],
            dict[str, int],
            dict[str, str],
            _TerminalFork | None,
        ]:
            cached = parsed.get(function.name)
            if cached is not None:
                return cached
            _validate_function_metadata_boundary(function)
            cached = self._parse_operations(function)
            parsed[function.name] = cached
            return cached

        def mapped_value(value_map: Mapping[str, str], value: str) -> str:
            try:
                return value_map[value]
            except KeyError as error:
                raise TTIRParseError(
                    f"interprocedural expansion lost SSA value {value!r}"
                ) from error

        def validate_call(
            operation: _Operation,
            caller: _FunctionParts,
        ) -> _FunctionParts:
            if operation.callee is None:
                raise TTIRParseError(
                    f"tt.call in @{caller.name} has no statically known callee"
                )
            callee = functions.get(operation.callee)
            if callee is None:
                raise TTIRParseError(
                    f"tt.call in @{caller.name} references unknown callee "
                    f"@{operation.callee}"
                )
            declared_operands = tuple(
                type_text for _, type_text, _ in callee.arguments
            )
            if len(operation.operands) != len(operation.call_operand_types):
                raise TTIRParseError(
                    f"tt.call @{callee.name} operand SSA/type arity mismatch: "
                    f"{len(operation.operands)} versus "
                    f"{len(operation.call_operand_types)}"
                )
            if len(operation.call_operand_types) != len(declared_operands):
                raise TTIRParseError(
                    f"tt.call @{callee.name} argument arity does not match its "
                    "tt.func signature"
                )
            if operation.call_operand_types != declared_operands:
                raise TTIRParseError(
                    f"tt.call @{callee.name} operand types do not match its "
                    "tt.func signature"
                )
            if len(operation.result_names) != len(operation.call_result_types):
                raise TTIRParseError(
                    f"tt.call @{callee.name} result SSA/type arity mismatch: "
                    f"{len(operation.result_names)} versus "
                    f"{len(operation.call_result_types)}"
                )
            if len(operation.call_result_types) != len(callee.results):
                raise TTIRParseError(
                    f"tt.call @{callee.name} result arity does not match its "
                    "tt.func signature"
                )
            if operation.call_result_types != callee.results:
                raise TTIRParseError(
                    f"tt.call @{callee.name} result types do not match its "
                    "tt.func signature"
                )
            return callee

        def emit_function(
            function: _FunctionParts,
            actual_arguments: tuple[str, ...],
            context_prefix: tuple[int, ...],
            stack: tuple[str, ...],
            *,
            preserve_return: bool,
            inherited_structured_if: tuple[int, str] | None = None,
            inherited_call_path: _SourceCallPath | None = (),
        ) -> tuple[str, ...]:
            nonlocal fork_serial
            if function.name in stack:
                cycle = " -> ".join((*stack, function.name))
                raise TTIRParseError(
                    f"recursive tt.call graph is unsupported: {cycle}"
                )
            if len(actual_arguments) != len(function.arguments):
                raise TTIRParseError(
                    f"tt.func @{function.name} argument arity mismatch: "
                    f"expected {len(function.arguments)}, got "
                    f"{len(actual_arguments)}"
                )

            local_operations, _, _, local_terminal_fork = parse_reachable(
                function
            )
            if inherited_structured_if is not None and (
                local_terminal_fork is not None
                or any(
                    operation.name == "scf.if"
                    for operation in local_operations
                )
            ):
                raise TTIRParseError(
                    "nested conditional control introduced by tt.call inside "
                    "scf.if is unsupported by the Gate1 v10 closed profile"
                )
            expanded_fork_id: int | None = None
            if local_terminal_fork is not None:
                expanded_fork_id = fork_serial
                fork_serial += 1
            value_map = {
                name: actual
                for (name, _, _), actual in zip(
                    function.arguments, actual_arguments, strict=True
                )
            }
            context_map: dict[int, int] = {}
            deferred_region_results: list[tuple[_Operation, tuple[str, ...]]] = []
            returned_values: list[tuple[str, ...]] = []
            saw_top_level_return = False
            expanded_fork_sequence_index: int | None = None

            for operation in local_operations:
                if (
                    operation.fork_id is not None
                    and expanded_fork_sequence_index is None
                ):
                    expanded_fork_sequence_index = len(expanded)
                if saw_top_level_return and local_terminal_fork is None:
                    if operation.name == "tt.return":
                        raise TTIRParseError(
                            f"tt.func @{function.name} has more than one "
                            "top-level tt.return"
                        )
                    raise TTIRParseError(
                        f"tt.func @{function.name} has an operation after its "
                        "top-level tt.return"
                    )
                try:
                    mapped_context = context_prefix + tuple(
                        context_map[parent] for parent in operation.context
                    )
                except KeyError as error:
                    raise TTIRParseError(
                        f"tt.func @{function.name} has a region whose owner "
                        "does not dominate it"
                    ) from error

                if operation.structured_if_id is not None:
                    try:
                        mapped_structured_if = (
                            context_map[operation.structured_if_id],
                            operation.structured_if_arm,
                        )
                    except KeyError as error:
                        raise TTIRParseError(
                            f"tt.func @{function.name} has an scf.if arm whose "
                            "owner does not dominate it"
                        ) from error
                else:
                    mapped_structured_if = inherited_structured_if
                if mapped_structured_if is not None and (
                    mapped_structured_if[1]
                    not in {"positive", "negative"}
                ):
                    raise TTIRParseError(
                        "expanded scf.if member has an invalid arm polarity"
                    )

                if operation.name == "tt.call":
                    callee = validate_call(operation, function)
                    actuals = tuple(
                        mapped_value(value_map, operand)
                        for operand in operation.operands
                    )
                    returned = emit_function(
                        callee,
                        actuals,
                        mapped_context,
                        (*stack, function.name),
                        preserve_return=False,
                        inherited_structured_if=mapped_structured_if,
                        inherited_call_path=(
                            (*inherited_call_path, *operation.source_call_path, operation.source_location[1:])
                            if inherited_call_path is not None and operation.source_location is not None
                            else None
                        ),
                    )
                    if len(returned) != len(operation.result_names):
                        raise TTIRParseError(
                            f"tt.call @{callee.name} returned {len(returned)} "
                            f"values for {len(operation.result_names)} results"
                        )
                    value_map.update(
                        zip(operation.result_names, returned, strict=True)
                    )
                    continue

                mapped_operands = [
                    mapped_value(value_map, operand)
                    for operand in operation.operands
                ]
                if operation.name == "tt.return":
                    if operation.context:
                        raise TTIRParseError(
                            f"tt.return in @{function.name} appears inside a "
                            "structured operation region"
                        )
                    if len(mapped_operands) != len(function.results):
                        raise TTIRParseError(
                            f"tt.return in @{function.name} has "
                            f"{len(mapped_operands)} values, expected "
                            f"{len(function.results)}"
                        )
                    returned_values.append(tuple(mapped_operands))
                    if local_terminal_fork is None:
                        saw_top_level_return = True
                    if not preserve_return:
                        continue

                new_results = tuple(fresh_value() for _ in operation.result_names)
                new_block_args = [
                    fresh_value() for _ in operation.region_block_args
                ]
                clone = _Operation(
                    node_id=len(expanded),
                    name=operation.name,
                    source_location=operation.source_location if inherited_call_path is not None else None,
                    source_call_path=(*inherited_call_path, *operation.source_call_path) if inherited_call_path is not None else (),
                    result_names=new_results,
                    operands=mapped_operands,
                    local_signature=operation.local_signature,
                    context=mapped_context,
                    result_types=operation.result_types,
                    axis=operation.axis,
                    reverse=operation.reverse,
                    cmp_predicate=operation.cmp_predicate,
                    cmp_flipped=operation.cmp_flipped,
                    header_operand_count=operation.header_operand_count,
                    region_results=[],
                    region_block_args=new_block_args,
                    constant_value=operation.constant_value,
                    callee=None,
                    terminator_types=operation.terminator_types,
                    fork_id=(
                        expanded_fork_id
                        if operation.fork_id is not None
                        else None
                    ),
                    fork_arm=operation.fork_arm,
                    structured_if_id=(
                        mapped_structured_if[0]
                        if mapped_structured_if is not None
                        else None
                    ),
                    structured_if_arm=(
                        mapped_structured_if[1]
                        if mapped_structured_if is not None
                        else None
                    ),
                    structured_if_result_type=operation.structured_if_result_type,
                )
                expanded.append(clone)
                context_map[operation.node_id] = clone.node_id
                for old, new in zip(
                    operation.result_names, new_results, strict=True
                ):
                    value_map[old] = new
                    expanded_results[new] = clone.node_id
                for old, new in zip(
                    operation.region_block_args, new_block_args, strict=True
                ):
                    value_map[old] = new
                deferred_region_results.append(
                    (clone, tuple(operation.region_results))
                )

            for clone, local_results in deferred_region_results:
                clone.region_results.extend(
                    mapped_value(value_map, value) for value in local_results
                )

            if local_terminal_fork is not None:
                assert expanded_fork_id is not None
                try:
                    mapped_predicate = value_map[
                        local_terminal_fork.predicate
                    ]
                    mapped_fork_context = context_prefix + tuple(
                        context_map[parent]
                        for parent in local_terminal_fork.context
                    )
                except KeyError as error:
                    raise TTIRParseError(
                        f"tt.func @{function.name} terminal fork lost its "
                        "predicate or structured region context"
                    ) from error
                expanded_terminal_forks.append(
                    _TerminalFork(
                        expanded_fork_id,
                        mapped_predicate,
                        mapped_fork_context,
                        (
                            len(expanded)
                            if expanded_fork_sequence_index is None
                            else expanded_fork_sequence_index
                        ),
                    )
                )

            if not returned_values:
                raise TTIRParseError(f"tt.func @{function.name} has no tt.return")
            if function.results:
                first = returned_values[0]
                if any(values != first for values in returned_values[1:]):
                    raise TTIRParseError(
                        f"tt.func @{function.name} has multiple non-identical "
                        "SSA returns; CFG result merging is unsupported"
                    )
                return first
            return ()

        public_arguments = tuple(name for name, _, _ in public.arguments)
        emit_function(
            public,
            public_arguments,
            (),
            (),
            preserve_return=True,
            inherited_structured_if=None,
        )

        # Reconstruct region leaves from the expanded graph so an outlined
        # helper called inside a supported structured region gets exactly the
        # same structural path as the equivalent inline operations.
        for operation in expanded:
            path = "/".join(
                [
                    *(expanded[parent].name for parent in operation.context),
                    operation.name,
                ]
            )
            for index, name in enumerate(operation.region_block_args):
                leaf_labels[name] = f"region[{path}].arg[{index}]"

        for operation in expanded:
            for operand in [*operation.operands, *operation.region_results]:
                if operand not in expanded_results and operand not in leaf_labels:
                    raise TTIRParseError(
                        f"unresolved expanded SSA value {operand!r} used by "
                        f"{operation.name}"
                    )
        return (
            expanded,
            expanded_results,
            leaf_labels,
            expanded_terminal_forks,
        )

    def _fingerprint(
        self,
        function: _FunctionParts,
        operations: list[_Operation],
        result_to_node: dict[str, int],
        leaf_labels: dict[str, str],
        terminal_forks: list[_TerminalFork],
        *,
        python_interface: Mapping[str, Any] | None,
        schedule: Mapping[str, Any] | None,
        representation_anchor_exemptions: frozenset[str],
        runtime_values: Mapping[str, int | bool] | None,
        source_locations: tuple[tuple[str, int, int], ...],
        scalar_owner_graphs: tuple[Mapping[str, Any], ...] = (),
        tensor_owner_graphs: tuple[Mapping[str, Any], ...] = (),
    ) -> TTIRFingerprint:
        raw_argument_aliases = (
            python_interface.get("ttir_argument_aliases", {})
            if isinstance(python_interface, Mapping)
            else {}
        )
        if raw_argument_aliases:
            if not isinstance(raw_argument_aliases, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in raw_argument_aliases.items()
            ):
                raise TTIRParseError("TTIR argument alias map is malformed")
            missing_aliases = [
                name.removeprefix("%")
                for name, _type, _attrs in function.arguments
                if name.removeprefix("%") not in raw_argument_aliases
            ]
            if missing_aliases:
                raise TTIRParseError(
                    "TTIR runtime arguments lack J/Harness ABI aliases: "
                    + ", ".join(missing_aliases)
                )
            # Alpha-normalize only the public ABI leaves.  Reachable-call
            # expansion may already have added region block arguments (for
            # example a tt.reduce combiner); replacing the whole map here made
            # those valid leaves unresolved during operand fingerprinting.
            leaf_labels.update(
                {
                    name: (
                        f"arg[{raw_argument_aliases[name.removeprefix('%')]}]:"
                        f"{type_text}"
                    )
                    for name, type_text, _attrs in function.arguments
                }
        )
        node_by_id = {operation.node_id: operation for operation in operations}

        def source_memory_kind(operation: str) -> str | None:
            if operation == "tt.load":
                return "load"
            if operation == "tt.store":
                return "store"
            if operation.startswith("tt.atomic"):
                return "update"
            return None

        # Provenance travels with each parser-owned operation, including after
        # scoped SSA renaming and reachable-call expansion. Pairing two lists
        # by operation family or order is not an identity proof.
        operation_source_locations = {
            operation.node_id: operation.source_location
            for operation in operations
            if operation.source_location is not None
            and operation.source_location[0] == operation.name
        }
        diagnostic_memory_source_locations = tuple(
            (
                kind,
                operation.header_operand_count >= (2 if kind == "load" else 3),
                location[1],
                location[2],
                operation.source_call_path,
            )
            for operation in operations
            if (kind := source_memory_kind(operation.name)) is not None
            and (location := operation_source_locations.get(operation.node_id)) is not None
        )
        value_if_ids = {
            operation.node_id
            for operation in operations
            if operation.name == "scf.if" and operation.result_names
        }
        for operation in operations:
            if operation.structured_if_id not in value_if_ids:
                continue
            if operation.name in _REGION_OWNERS:
                raise TTIRParseError(
                    "value-producing scf.if arms cannot contain a nested region"
                )
            if (
                operation.name not in _TERMINATORS
                and _operation_has_effect(operation)
            ):
                raise TTIRParseError(
                    "value-producing scf.if arms must be side-effect free; "
                    f"found {operation.name}"
                )
        effect_occurrences = _effect_occurrence_identities(
            operations, terminal_forks
        )
        identity_nodes = _closed_consumer_identity_nodes(
            operations,
            result_to_node,
            representation_anchor_exemptions,
        )
        identity_aliases = {
            operation.result_names[0]: operation.operands[1]
            for operation in operations
            if operation.node_id in identity_nodes
            and len(operation.result_names) == 1
            and len(operation.operands) >= 2
        }
        representation_only = _representation_only_nodes(
            operations,
            result_to_node,
            terminal_forks,
            anchor_exemptions=representation_anchor_exemptions,
            identity_nodes=identity_nodes,
        )
        _zero_bound_nodes, zero_bound_aliases = _zero_lower_bound_mask_profile(
            operations, result_to_node, representation_anchor_exemptions
        )
        if not _zero_bound_nodes.issubset(representation_only):
            zero_bound_aliases = {}
        normalize_access_addresses = (
            "tt.addptr" in representation_anchor_exemptions
        )
        structural_cache: dict[int, str] = {}
        semantic_cache: dict[int, str] = {}
        representation_atom_cache: dict[str, tuple[str, ...]] = {}
        representation_ref_cache: dict[str, str] = {}

        def result_contract(operation: _Operation) -> str:
            """Return a bounded result-type view without retaining its formula."""

            signature = _trailing_type_signature(operation.local_signature)
            if signature is None:
                return f"results={len(operation.result_names)}"
            arrow = _find_top_level_token(signature, "->", last=True)
            if arrow is not None:
                result_text = signature[arrow + 2 :]
            else:
                conversion = _find_top_level_token(signature, " to ", last=True)
                result_text = (
                    signature[conversion + 4 :]
                    if conversion is not None
                    else signature
                )
            return _normalize_space(result_text)

        runtime_constants = {
            f"%{name.removeprefix('%')}": int(value)
            for name, value in (runtime_values or {}).items()
            if isinstance(value, (bool, int))
        }
        polynomial_cache: dict[
            str, dict[tuple[str, ...], int] | None
        ] = {}

        def polynomial_add(
            left: Mapping[tuple[str, ...], int],
            right: Mapping[tuple[str, ...], int],
            *,
            sign: int = 1,
        ) -> dict[tuple[str, ...], int]:
            result = dict(left)
            for monomial, coefficient in right.items():
                result[monomial] = result.get(monomial, 0) + sign * coefficient
                if result[monomial] == 0:
                    del result[monomial]
            return result

        def polynomial_mul(
            left: Mapping[tuple[str, ...], int],
            right: Mapping[tuple[str, ...], int],
        ) -> dict[tuple[str, ...], int] | None:
            if len(left) * len(right) > 128:
                return None
            result: dict[tuple[str, ...], int] = {}
            for left_terms, left_coefficient in left.items():
                for right_terms, right_coefficient in right.items():
                    monomial = tuple(sorted((*left_terms, *right_terms)))
                    result[monomial] = result.get(monomial, 0) + (
                        left_coefficient * right_coefficient
                    )
                    if result[monomial] == 0:
                        del result[monomial]
            return result

        def polynomial_payload(
            value: Mapping[tuple[str, ...], int]
        ) -> tuple[tuple[tuple[str, ...], int], ...]:
            return tuple(sorted(value.items()))

        def integer_polynomial(
            name: str,
            active: frozenset[str] = frozenset(),
        ) -> dict[tuple[str, ...], int] | None:
            if name in polynomial_cache:
                cached = polynomial_cache[name]
                return None if cached is None else dict(cached)
            if name in active:
                raise TTIRParseError("cyclic integer representation expression")
            if name in leaf_labels:
                if name in runtime_constants:
                    result = {(): runtime_constants[name]}
                else:
                    result = {(leaf_labels[name],): 1}
                polynomial_cache[name] = result
                return dict(result)
            node_id = result_to_node.get(name)
            if node_id is None:
                polynomial_cache[name] = None
                return None
            operation = node_by_id[node_id]
            next_active = active | {name}
            result: dict[tuple[str, ...], int] | None
            if operation.name == "arith.constant":
                constant = operation.constant_value
                result = (
                    {(): int(constant)}
                    if constant is not None
                    and float(constant).is_integer()
                    else None
                )
            elif operation.name == "tt.get_program_id":
                match = re.search(
                    r"\btt\.get_program_id\s+([xyz])\b",
                    operation.local_signature,
                )
                result = (
                    {(f"program_axis:{match.group(1)}",): 1}
                    if match is not None
                    else None
                )
            elif operation.name == "tt.make_range":
                result = {(f"lane:{operation.local_signature}",): 1}
            elif operation.name == "tt.load" and operation.operands:
                # A scalar integer loaded by the Source can legitimately
                # participate in a later pointer offset. It is neither an
                # affine constant nor an arbitrary unknown: record its
                # source-proven load address as one opaque typed integer atom.
                # The enclosing integer operation still proves that this value
                # is used as an integer address expression.
                loaded_from = pointer_form(operation.operands[0], next_active)
                if loaded_from is None:
                    result = None
                else:
                    atom = _stable_json(
                        [
                            "tt.load.integer_address_input",
                            loaded_from[0],
                            polynomial_payload(loaded_from[1]),
                        ]
                    )
                    result = {(f"integer_memory_value:{atom}",): 1}
            elif operation.name in {
                "arith.extsi", "arith.extui", "arith.index_cast",
                "arith.index_castui", "arith.trunci", "tt.splat",
                "tt.broadcast", "tt.expand_dims", "tt.reshape", "tt.trans",
            } and len(operation.operands) == 1:
                result = integer_polynomial(operation.operands[0], next_active)
            elif operation.name in {"arith.addi", "arith.subi", "arith.muli"} and len(operation.operands) == 2:
                left = integer_polynomial(operation.operands[0], next_active)
                right = integer_polynomial(operation.operands[1], next_active)
                if left is None or right is None:
                    result = None
                elif operation.name == "arith.addi":
                    result = polynomial_add(left, right)
                elif operation.name == "arith.subi":
                    result = polynomial_add(left, right, sign=-1)
                else:
                    result = polynomial_mul(left, right)
            elif operation.name in {"arith.divsi", "arith.divui", "arith.remsi", "arith.remui"} and len(operation.operands) == 2:
                numerator = integer_polynomial(operation.operands[0], next_active)
                denominator = integer_polynomial(operation.operands[1], next_active)
                denominator_value = (
                    denominator.get(())
                    if denominator is not None and set(denominator) == {()}
                    else None
                )
                if numerator is None or denominator is None or denominator_value == 0:
                    result = None
                else:
                    # A dynamic divisor is not affine, but it is still a
                    # typed, source-proven integer address relation.  Dropping
                    # it here made the address sidecar shorter than the
                    # transfer/origin sidecars, so otherwise valid kernels
                    # failed Source preparation before Lift began.  Preserve
                    # the complete relation as one opaque integer atom.  It
                    # remains a proof obligation (not an affine rewrite and
                    # not a Gate1 equivalence relaxation).
                    atom = _stable_json(
                        [
                            operation.name,
                            polynomial_payload(numerator),
                            polynomial_payload(denominator),
                        ]
                    )
                    result = {(f"integer_relation:{atom}",): 1}
            else:
                result = None
            polynomial_cache[name] = None if result is None else dict(result)
            return None if result is None else dict(result)

        # ``pointer_form`` carries a fourth, host-only typed-offset-layer
        # profile in addition to its legacy base/polynomial/element facts.
        pointer_cache: dict[str, Any] = {}
        value_type_cache: dict[str, str | None] = {}

        def closed_value_type(
            name: str,
            active: frozenset[str] = frozenset(),
        ) -> str | None:
            """Recover only an explicit compiler type for one SSA value."""

            if name in value_type_cache:
                return value_type_cache[name]
            if name in active:
                return None
            label = leaf_labels.get(name)
            if isinstance(label, str) and label.startswith("arg[") and "]:" in label:
                result = _normalize_space(label.split("]:", 1)[1])
                value_type_cache[name] = result or None
                return result or None
            node_id = result_to_node.get(name)
            operation = node_by_id.get(node_id) if node_id is not None else None
            if operation is None or name not in operation.result_names:
                value_type_cache[name] = None
                return None
            operand_types = tuple(
                closed_value_type(operand, active | {name})
                for operand in operation.operands
            )
            try:
                result_types = _operation_result_types(
                    operation.name,
                    operation.local_signature,
                    len(operation.result_names),
                    operand_types,
                    call_result_types=operation.call_result_types,
                )
            except (TTIRParseError, ValueError):
                result_types = (None,) * len(operation.result_names)
            result = result_types[operation.result_names.index(name)]
            result = _normalize_space(result) if isinstance(result, str) else None
            value_type_cache[name] = result
            return result

        def integer_index_type(type_text: str | None) -> str | None:
            """Return the explicit signless integer/index element type."""

            if not isinstance(type_text, str):
                return None
            normalized = _normalize_space(type_text)
            if re.fullmatch(r"(?:i[0-9]+|index)", normalized) is not None:
                return normalized
            matches = re.findall(r"(?:^|x)(i[0-9]+|index)(?=[,>])", normalized)
            return matches[-1] if matches else None

        type_proof_cache: dict[str, tuple[str, ...]] = {}

        def integer_type_proof(
            name: str,
            active: frozenset[str] = frozenset(),
        ) -> tuple[str, ...]:
            """Return a bounded compiler-SSA type propagation witness.

            The witness retains operation families and explicit types, but no
            raw SSA spelling. The exact TTIR fingerprint remains the source of
            truth; an unknown, cyclic or unbounded chain returns no proof.
            """

            cached = type_proof_cache.get(name)
            if cached is not None:
                return cached
            index_type = integer_index_type(closed_value_type(name))
            if index_type is None or name in active:
                type_proof_cache[name] = ()
                return ()
            if name in leaf_labels:
                result = (f"arg:{index_type}",)
                type_proof_cache[name] = result
                return result
            node_id = result_to_node.get(name)
            operation = node_by_id.get(node_id) if node_id is not None else None
            if operation is None or name not in operation.result_names:
                type_proof_cache[name] = ()
                return ()
            pieces: list[str] = []
            for operand in operation.operands:
                if integer_index_type(closed_value_type(operand)) is None:
                    continue
                proof = integer_type_proof(operand, active | {name})
                if not proof:
                    type_proof_cache[name] = ()
                    return ()
                pieces.extend(proof)
            pieces.append(f"{operation.name}:{index_type}")
            if len(pieces) > 64:
                type_proof_cache[name] = ()
                return ()
            result = tuple(pieces)
            type_proof_cache[name] = result
            return result

        loop_carried_initials: dict[str, str] = {}
        if normalize_access_addresses:
            for candidate in operations:
                if candidate.name != "scf.for":
                    continue
                # scf.for operands are lower/upper/step followed by init args;
                # its entry block arguments are induction variable followed by
                # the matching carried values.  Under the closed all-access
                # address quotient, a pointer carry retains its public ABI base
                # even when the Source advances it and Replay recomputes it.
                for slot, block_argument in enumerate(
                    candidate.region_block_args[1:]
                ):
                    operand_index = 3 + slot
                    if operand_index < len(candidate.operands):
                        loop_carried_initials[block_argument] = (
                            candidate.operands[operand_index]
                        )

        def pointer_form(
            name: str,
            active: frozenset[str] = frozenset(),
        ) -> Any:
            if name in pointer_cache:
                cached = pointer_cache[name]
                return None if cached is None else (
                    cached[0], dict(cached[1]), cached[2], cached[3]
                )
            if name in active:
                raise TTIRParseError("cyclic pointer representation expression")
            if name in leaf_labels:
                initial = loop_carried_initials.get(name)
                result = (
                    pointer_form(initial, active | {name})
                    if initial is not None
                    else (
                        leaf_labels[name],
                        {},
                        _pointer_dtype(closed_value_type(name) or ""),
                        (),
                    )
                )
                pointer_cache[name] = result
                return None if result is None else (
                    result[0], dict(result[1]), result[2], result[3]
                )
            node_id = result_to_node.get(name)
            if node_id is None:
                pointer_cache[name] = None
                return None
            operation = node_by_id[node_id]
            next_active = active | {name}
            result: Any = None
            if operation.name == "tt.addptr" and len(operation.operands) >= 2:
                base = pointer_form(operation.operands[0], next_active)
                offset = integer_polynomial(operation.operands[1], next_active)
                pointer_element = _pointer_dtype(
                    closed_value_type(operation.operands[0]) or ""
                )
                index_type = integer_index_type(closed_value_type(operation.operands[1]))
                type_proof = integer_type_proof(operation.operands[1])
                if base is not None and offset is not None:
                    closed_element = (
                        pointer_element
                        if isinstance(pointer_element, str)
                        and pointer_element == base[2]
                        else None
                    )
                    combined_polynomial = polynomial_add(base[1], offset)
                    # Preserve every nonzero addptr layer independently. A
                    # later i32 vector offset cannot overwrite an earlier i64
                    # scalar offset; both compiler proofs reach the Source
                    # transfer ledger. Zero layers carry no address term.
                    layers = base[3]
                    if offset:
                        layers = (
                            *layers,
                            (
                                polynomial_payload(offset),
                                index_type,
                                type_proof,
                            ),
                        )
                    result = (
                        base[0],
                        combined_polynomial,
                        closed_element,
                        layers,
                    )
            elif operation.name in {
                "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape",
            } and len(operation.operands) == 1:
                result = pointer_form(operation.operands[0], next_active)
            elif operation.name == "scf.for" and name in operation.result_names:
                slot = operation.result_names.index(name)
                operand_index = 3 + slot
                if operand_index < len(operation.operands):
                    result = pointer_form(
                        operation.operands[operand_index], next_active
                    )
            pointer_cache[name] = (
                None
                if result is None
                else (result[0], dict(result[1]), result[2], result[3])
            )
            return None if result is None else (
                result[0], dict(result[1]), result[2], result[3]
            )

        def hard_local_signature(operation: _Operation) -> str:
            if operation.name != "tt.addptr":
                return operation.local_signature
            signature = _trailing_type_signature(operation.local_signature)
            pointer_type: str | None = None
            if signature is not None:
                arrow = _find_top_level_token(signature, "->", last=True)
                declared = _declared_types(
                    signature[arrow + 2 :] if arrow is not None else signature
                )
                pointer_type = next(
                    (
                        item
                        for item in declared
                        if _pointer_dtype(item) is not None
                    ),
                    None,
                )
            if pointer_type is None:
                return "tt.addptr pointer=unknown"
            tensor = _ranked_tensor_contract(pointer_type)
            rank = len(tensor[0]) if tensor is not None else 0
            return (
                f"tt.addptr pointer_dtype={_pointer_dtype(pointer_type)} "
                f"pointer_rank={rank}"
            )

        def representation_atoms(
            name: str, active: frozenset[int] = frozenset()
        ) -> tuple[str, ...]:
            cached = representation_atom_cache.get(name)
            if cached is not None:
                return cached
            leaf = leaf_labels.get(name)
            if leaf is not None:
                cached = (leaf,)
                representation_atom_cache[name] = cached
                return cached
            node_id = result_to_node[name]
            if node_id not in representation_only:
                cached = (operand_ref(name, semantic=True),)
                representation_atom_cache[name] = cached
                return cached
            if node_id in active:
                raise TTIRParseError(
                    "cyclic representation-only SSA dependency"
                )
            operation = node_by_id[node_id]
            atoms: set[str] = set()
            if operation.name == "tt.make_range":
                atoms.add(f"lane:{result_contract(operation)}")
            for operand in [*operation.operands, *operation.region_results]:
                atoms.update(
                    representation_atoms(operand, active | {node_id})
                )
            if not atoms:
                # Constants specialized from C and other closed literal-only
                # scaffolding retain their type but not a formula/value already
                # owned by the schedule binding or a surviving predicate.
                atoms.add(f"literal:{result_contract(operation)}")
            cached = tuple(sorted(atoms))
            representation_atom_cache[name] = cached
            return cached

        def representation_ref(name: str) -> str:
            cached = representation_ref_cache.get(name)
            if cached is not None:
                return cached
            node_id = result_to_node[name]
            operation = node_by_id[node_id]
            slot = (
                operation.result_names.index(name)
                if len(operation.result_names) > 1
                else 0
            )
            pointer = pointer_form(name)
            polynomial = integer_polynomial(name)
            if pointer is not None:
                payload = {
                    "contract": result_contract(operation),
                    "pointer_base": pointer[0],
                    "offset": polynomial_payload(pointer[1]),
                    "slot": slot,
                }
            elif polynomial is not None:
                payload = {
                    "contract": result_contract(operation),
                    "integer_polynomial": polynomial_payload(polynomial),
                    "slot": slot,
                }
            else:
                payload = {
                    "contract": result_contract(operation),
                    "origins": representation_atoms(name),
                    "slot": slot,
                }
            cached = f"representation[{_digest(payload)[:20]}]"
            representation_ref_cache[name] = cached
            return cached

        def operand_ref(name: str, *, semantic: bool) -> str:
            alias_seen: set[str] = set()
            while name in identity_aliases:
                if name in alias_seen:
                    raise TTIRParseError(
                        "cyclic closed-consumer identity alias"
                    )
                alias_seen.add(name)
                name = identity_aliases[name]
            while name in zero_bound_aliases:
                if name in alias_seen:
                    raise TTIRParseError(
                        "cyclic default-zero-bound predicate alias"
                    )
                alias_seen.add(name)
                name = zero_bound_aliases[name]
            if name in leaf_labels:
                return leaf_labels[name]
            node_id = result_to_node[name]
            producer = node_by_id[node_id]
            if node_id in representation_only:
                if _is_identity_reshape(producer) and producer.operands:
                    return operand_ref(producer.operands[0], semantic=semantic)
                return representation_ref(name)
            base = node_ref(node_id, semantic=semantic)
            if len(producer.result_names) <= 1:
                return base
            try:
                slot = producer.result_names.index(name)
            except ValueError as error:  # pragma: no cover - map is constructed together
                raise TTIRParseError(
                    f"SSA value {name!r} is absent from its multi-result producer"
                ) from error
            return f"{base}#result[{slot}]"

        def normalized_operand_refs(
            operation: _Operation, *, semantic: bool
        ) -> list[str]:
            names = [*operation.operands, *operation.region_results]
            refs = [operand_ref(name, semantic=semantic) for name in names]
            if operation.name == "tt.load" and operation.header_operand_count >= 2:
                refs[1] = operand_ref(names[1], semantic=semantic)
            elif operation.name == "tt.store" and operation.header_operand_count >= 3:
                refs[2] = operand_ref(names[2], semantic=semantic)
            if (
                normalize_access_addresses
                and names
                and (
                    operation.name in {"tt.load", "tt.store"}
                    or operation.name.startswith("tt.atomic")
                )
            ):
                # A closed all-access ``address_recomputed``/``pointer_advanced``
                # witness authorizes only the address-form quotient.  Preserve
                # the public pointer base (so swapping logical tensors remains
                # hard), while dropping the particular addptr/polynomial tree.
                pointer = pointer_form(names[0])
                if pointer is not None:
                    refs[0] = f"address-base[{pointer[0]}]"
            if operation.cmp_flipped and len(refs) >= 2:
                refs[0], refs[1] = refs[1], refs[0]
            if (
                operation.name in _COMMUTATIVE_OPS
                or operation.cmp_predicate in _SYMMETRIC_CMP
            ):
                header = refs[: operation.header_operand_count]
                header.sort()
                refs = header + refs[operation.header_operand_count :]
            return refs

        def node_ref(node_id: int, *, semantic: bool) -> str:
            cache = semantic_cache if semantic else structural_cache
            if node_id in cache:
                return cache[node_id]
            operation = node_by_id[node_id]
            refs = normalized_operand_refs(operation, semantic=semantic)
            if semantic:
                local = hard_local_signature(operation)
            else:
                # Producer/consumer topology deliberately uses operation kinds,
                # while predicates, fill values, dot attributes and reductions
                # are compared in their dedicated hard components.
                local = _semantic_op_name(operation, node_by_id, result_to_node)
            label = f"{operation.name}[{_digest([local, refs])[:20]}]"
            occurrence = effect_occurrences.get(node_id)
            if occurrence is not None:
                label = f"{label}@{occurrence}"
            cache[node_id] = label
            return label

        op_counts = Counter(
            _semantic_op_name(operation, node_by_id, result_to_node)
            for operation in operations
            if operation.name not in _TERMINATORS
            and operation.node_id not in representation_only
        )
        if terminal_forks:
            op_counts["control.conditional"] += len(terminal_forks)

        edges: list[tuple[str, str, str]] = []
        for operation in operations:
            if (
                operation.name in _TERMINATORS
                or operation.node_id in representation_only
            ):
                continue
            # Hard dataflow labels include normalized operation attributes and
            # constants, not just graph topology.  Otherwise an unchanged op
            # multiset wired through ``0.5`` instead of ``2.0`` could close.
            # Dedicated reduction/predicate fields may report the same mutation
            # more specifically; exactness is intentionally preferred here.
            refs = normalized_operand_refs(operation, semantic=True)
            consumer = node_ref(operation.node_id, semantic=True)
            for index, producer in enumerate(refs):
                if operation.name == "scf.if":
                    role = (
                        "condition"
                        if index == 0
                        else "positive_yield"
                        if index == 1
                        else "negative_yield"
                    )
                elif index < operation.header_operand_count:
                    role = f"operand[{index}]"
                else:
                    role = f"region_result[{index - operation.header_operand_count}]"
                edges.append((producer, consumer, role))

        # The function result mapping is semantic dataflow, not terminator
        # presentation.  Without these positional edges, returning a different
        # already-computed SSA value (or swapping equal-typed multi-results)
        # would leave the operation graph unchanged and falsely hard-pass.
        public_returns = [
            operation
            for operation in operations
            if operation.name == "tt.return" and not operation.context
        ]
        fork_returns = [
            operation
            for operation in public_returns
            if operation.fork_id is not None
        ]
        if fork_returns:
            if (
                len(public_returns) != 2
                or {operation.fork_arm for operation in fork_returns}
                != {"true", "false"}
                or any(operation.operands for operation in fork_returns)
            ):
                raise TTIRParseError(
                    "expanded public terminal fork must have exactly one void "
                    "tt.return in each role-preserving arm"
                )
        else:
            if len(public_returns) != 1:
                raise TTIRParseError(
                    "expanded public tt.func must contain exactly one "
                    "top-level tt.return"
                )
            for index, value in enumerate(public_returns[0].operands):
                edges.append(
                    (
                        operand_ref(value, semantic=True),
                        "region[root]",
                        f"function.return[{index}]",
                    )
                )

        # Region ownership is hard evidence.  A global operation multiset plus
        # descendant counts cannot distinguish non-isomorphic loop trees, and
        # sorting memory effects loses the order of writes to the same address.
        # Build a recursive, content-sensitive region identity without using
        # SSA names, source function names, or numeric node ids.  Direct member
        # edges bind every operation/effect to its parent; the ordered effect
        # trace retains lexical happens-before for reads, writes, atomics,
        # barriers, and assertions.  Outlined calls have already been expanded,
        # so outlined and equivalent inline programs traverse this same graph.
        region_ids = {
            operation.node_id
            for operation in operations
            if operation.name in _REGION_OWNERS
        }
        structured_if_ids = {
            operation.node_id
            for operation in operations
            if operation.name == "scf.if"
        }
        terminal_fork_by_id = {
            item.fork_id: item for item in terminal_forks
        }
        if len(terminal_fork_by_id) != len(terminal_forks):
            raise TTIRParseError("expanded terminal-fork identities are not unique")
        direct_terminal_forks: dict[int | None, list[_TerminalFork]] = {}
        for terminal_fork in terminal_forks:
            parent = (
                terminal_fork.context[-1]
                if terminal_fork.context
                else None
            )
            if parent is not None and parent not in region_ids:
                raise TTIRParseError(
                    "terminal fork has a non-structured caller context"
                )
            direct_terminal_forks.setdefault(parent, []).append(terminal_fork)
        direct_children: dict[int | None, list[int]] = {}
        direct_members: dict[int | None, list[_Operation]] = {}
        direct_sequence: dict[int | None, list[_Operation]] = {}
        direct_if_arms: dict[tuple[int, str], list[_Operation]] = {}
        for operation in operations:
            parent = operation.context[-1] if operation.context else None
            if parent is not None and parent not in region_ids:
                raise TTIRParseError(
                    f"operation {operation.name} has a non-region parent"
                )
            terminal_fork = (
                terminal_fork_by_id.get(operation.fork_id)
                if operation.fork_id is not None
                else None
            )
            if operation.fork_id is not None and terminal_fork is None:
                raise TTIRParseError(
                    "operation refers to an unknown expanded terminal fork"
                )
            if operation.fork_arm not in {None, "true", "false"}:
                raise TTIRParseError("operation has an invalid terminal-fork role")
            if operation.structured_if_id is not None:
                if (
                    operation.structured_if_id not in structured_if_ids
                    or operation.structured_if_id not in operation.context
                    or operation.structured_if_arm
                    not in {"positive", "negative"}
                ):
                    raise TTIRParseError(
                        "operation has an invalid structured scf.if ancestry"
                    )
            elif any(
                owner in structured_if_ids for owner in operation.context
            ):
                raise TTIRParseError(
                    "operation inside scf.if lost its arm polarity"
                )
            is_direct_fork_member = (
                terminal_fork is not None
                and operation.context == terminal_fork.context
            )
            # The two arms are mutually exclusive.  Their direct members are
            # attached to the synthetic control node below, never flattened
            # into the caller's ordinary region sequence.  Nested structured
            # regions still retain their own internal membership graph.
            if not is_direct_fork_member:
                if parent in structured_if_ids:
                    if operation.structured_if_id != parent:
                        raise TTIRParseError(
                            "direct scf.if member is bound to the wrong owner"
                        )
                    assert operation.structured_if_arm is not None
                    direct_if_arms.setdefault(
                        (parent, operation.structured_if_arm), []
                    ).append(operation)
                else:
                    direct_sequence.setdefault(parent, []).append(operation)
                if operation.node_id in region_ids:
                    direct_children.setdefault(parent, []).append(operation.node_id)
                elif (
                    operation.name not in _TERMINATORS
                    and operation.node_id not in representation_only
                ):
                    direct_members.setdefault(parent, []).append(operation)

            restricted_ancestor = next(
                (
                    node_by_id[ancestor].name
                    for ancestor in reversed(operation.context)
                    if node_by_id[ancestor].name in {"tt.reduce", "tt.scan"}
                ),
                None,
            )
            # Classification is total over the pinned observer profile.  Do
            # this even at function root: unknown operations may be effectful,
            # so treating them as pure until a future release understands them
            # would make lexical reordering invisible.
            operation_has_effect = _operation_has_effect(operation)
            if operation_has_effect and restricted_ancestor is not None:
                raise TTIRParseError(
                    f"side-effecting operation {operation.name} inside "
                    f"{restricted_ancestor} is unsupported"
                )

        effectful_cache: dict[int, bool] = {}

        def terminal_fork_has_effect(terminal_fork: _TerminalFork) -> bool:
            return any(
                operation.name not in _TERMINATORS
                and _operation_has_effect(operation)
                for operation in operations
                if operation.fork_id == terminal_fork.fork_id
            )

        def region_has_effect(region_id: int) -> bool:
            cached = effectful_cache.get(region_id)
            if cached is not None:
                return cached
            value = any(
                _operation_has_effect(operation)
                for operation in direct_members.get(region_id, ())
            ) or any(
                region_has_effect(child)
                for child in direct_children.get(region_id, ())
            ) or any(
                terminal_fork_has_effect(terminal_fork)
                for terminal_fork in direct_terminal_forks.get(region_id, ())
            )
            effectful_cache[region_id] = value
            return value

        region_ref_cache: dict[int, str] = {}
        structured_if_fingerprint_cache: dict[
            int, StructuredIfFingerprint
        ] = {}
        terminal_fork_fingerprint_cache: dict[
            int, TerminalForkFingerprint
        ] = {}
        terminal_fork_ref_cache: dict[int, str] = {}

        structured_if_ordinals: dict[int, int] = {}
        structured_if_containers: dict[
            tuple[tuple[int, ...], int | None, str | None], list[int]
        ] = {}
        for owner_id in structured_if_ids:
            owner = node_by_id[owner_id]
            key = (owner.context, owner.fork_id, owner.fork_arm)
            structured_if_containers.setdefault(key, []).append(owner_id)
        for owners in structured_if_containers.values():
            for ordinal, owner_id in enumerate(sorted(owners)):
                structured_if_ordinals[owner_id] = ordinal

        def structured_if_arm(
            owner_id: int, role: str
        ) -> StructuredIfArmFingerprint:
            owner = node_by_id[owner_id]
            direct = direct_if_arms.get((owner_id, role), ())
            members = sorted(
                node_ref(operation.node_id, semantic=True)
                for operation in direct
                if operation.node_id not in region_ids
                and operation.name not in _TERMINATORS
                and operation.node_id not in representation_only
            )
            structured_regions = sorted(
                region_ref(operation.node_id)
                for operation in direct
                if operation.node_id in region_ids
            )
            effect_order: list[str] = []
            for operation in direct:
                if operation.node_id in region_ids:
                    if region_has_effect(operation.node_id):
                        effect_order.append(
                            f"region:{region_ref(operation.node_id)}"
                        )
                elif (
                    operation.name not in _TERMINATORS
                    and _operation_has_effect(operation)
                ):
                    effect_order.append(
                        f"{operation.name}:"
                        f"{node_ref(operation.node_id, semantic=True)}"
                    )
            yielded_value: str | None = None
            yield_type: str | None = None
            terminal = "scf.yield:void"
            if owner.result_names:
                if len(owner.region_results) != 2:
                    raise TTIRParseError(
                        "expanded value-producing scf.if lost one arm yield"
                    )
                yield_index = 0 if role == "positive" else 1
                yielded_value = operand_ref(
                    owner.region_results[yield_index], semantic=True
                )
                yield_type = owner.structured_if_result_type
                if yield_type is None:
                    raise TTIRParseError(
                        "expanded value-producing scf.if lost its result type"
                    )
                terminal = "scf.yield:value"
            return StructuredIfArmFingerprint(
                role=role,
                members=tuple(members),
                effect_order=tuple(effect_order),
                structured_regions=tuple(structured_regions),
                yielded_value=yielded_value,
                yield_type=yield_type,
                terminal=terminal,
            )

        def structured_if_fingerprint(
            owner_id: int,
        ) -> StructuredIfFingerprint:
            cached = structured_if_fingerprint_cache.get(owner_id)
            if cached is not None:
                return cached
            owner = node_by_id[owner_id]
            if len(owner.operands) != 1 or len(owner.result_names) not in {0, 1}:
                raise TTIRParseError(
                    "expanded scf.if is outside the zero-or-one-result profile"
                )
            if bool(owner.result_names) != bool(owner.structured_if_result_type):
                raise TTIRParseError(
                    "expanded scf.if result arity/type contract is inconsistent"
                )
            cached = StructuredIfFingerprint(
                ordinal=structured_if_ordinals[owner_id],
                predicate=operand_ref(owner.operands[0], semantic=True),
                region_context=tuple(
                    node_ref(ancestor, semantic=True)
                    for ancestor in owner.context
                ),
                enclosing_terminal_arm=owner.fork_arm,
                result_type=owner.structured_if_result_type,
                positive_arm=structured_if_arm(owner_id, "positive"),
                negative_arm=structured_if_arm(owner_id, "negative"),
            )
            structured_if_fingerprint_cache[owner_id] = cached
            return cached

        def structured_if_arm_ref(
            arm: StructuredIfArmFingerprint,
        ) -> str:
            return (
                f"control.scf.if.arm[{arm.role}:"
                f"{_digest(arm.to_dict())[:20]}]"
            )

        def terminal_fork_arm(
            terminal_fork: _TerminalFork, role: str
        ) -> TerminalForkArmFingerprint:
            direct = [
                operation
                for operation in operations
                if operation.fork_id == terminal_fork.fork_id
                and operation.fork_arm == role
                and operation.context == terminal_fork.context
            ]
            members = sorted(
                node_ref(operation.node_id, semantic=True)
                for operation in direct
                if operation.node_id not in region_ids
                and operation.name not in _TERMINATORS
                and operation.node_id not in representation_only
            )
            structured_regions = sorted(
                region_ref(operation.node_id)
                for operation in direct
                if operation.node_id in region_ids
            )
            effect_order: list[str] = []
            for operation in direct:
                if operation.node_id in region_ids:
                    if region_has_effect(operation.node_id):
                        effect_order.append(
                            f"region:{region_ref(operation.node_id)}"
                        )
                elif (
                    operation.name not in _TERMINATORS
                    and _operation_has_effect(operation)
                ):
                    effect_order.append(
                        f"{operation.name}:"
                        f"{node_ref(operation.node_id, semantic=True)}"
                    )
            return TerminalForkArmFingerprint(
                role=role,
                members=tuple(members),
                effect_order=tuple(effect_order),
                structured_regions=tuple(structured_regions),
            )

        def terminal_fork_fingerprint(
            terminal_fork: _TerminalFork,
        ) -> TerminalForkFingerprint:
            cached = terminal_fork_fingerprint_cache.get(
                terminal_fork.fork_id
            )
            if cached is not None:
                return cached
            cached = TerminalForkFingerprint(
                predicate=operand_ref(
                    terminal_fork.predicate, semantic=True
                ),
                region_context=tuple(
                    node_ref(owner, semantic=True)
                    for owner in terminal_fork.context
                ),
                true_arm=terminal_fork_arm(terminal_fork, "true"),
                false_arm=terminal_fork_arm(terminal_fork, "false"),
            )
            terminal_fork_fingerprint_cache[terminal_fork.fork_id] = cached
            return cached

        def terminal_fork_ref(terminal_fork: _TerminalFork) -> str:
            cached = terminal_fork_ref_cache.get(terminal_fork.fork_id)
            if cached is not None:
                return cached
            fingerprint = terminal_fork_fingerprint(terminal_fork)
            cached = (
                "control.conditional["
                f"{_digest(fingerprint.to_dict())[:20]}]"
            )
            terminal_fork_ref_cache[terminal_fork.fork_id] = cached
            return cached

        def terminal_arm_ref(
            arm: TerminalForkArmFingerprint,
        ) -> str:
            return (
                f"control.arm[{arm.role}:"
                f"{_digest(arm.to_dict())[:20]}]"
            )

        def direct_effect_events(parent: int | None) -> list[str]:
            if parent in structured_if_ids:
                raise TTIRParseError(
                    "scf.if arm effects must be observed by polarity, not "
                    "flattened into one region sequence"
                )
            events: list[str] = []
            sequence: list[tuple[int, int, _Operation | _TerminalFork]] = [
                (operation.node_id, 1, operation)
                for operation in direct_sequence.get(parent, ())
            ]
            sequence.extend(
                (terminal_fork.sequence_index, 0, terminal_fork)
                for terminal_fork in direct_terminal_forks.get(parent, ())
            )
            for _, _, item in sorted(
                sequence, key=lambda entry: (entry[0], entry[1])
            ):
                if isinstance(item, _TerminalFork):
                    if terminal_fork_has_effect(item):
                        events.append(f"control:{terminal_fork_ref(item)}")
                elif item.node_id in region_ids:
                    if region_has_effect(item.node_id):
                        events.append(f"region:{region_ref(item.node_id)}")
                elif _operation_has_effect(item):
                    events.append(
                        f"{item.name}:"
                        f"{node_ref(item.node_id, semantic=True)}"
                    )
            return events

        def region_ref(region_id: int) -> str:
            cached = region_ref_cache.get(region_id)
            if cached is not None:
                return cached
            operation = node_by_id[region_id]
            if region_id in structured_if_ids:
                fingerprint = structured_if_fingerprint(region_id)
                cached = (
                    "region[scf.if:"
                    f"{_digest(fingerprint.to_dict())[:20]}]"
                )
                region_ref_cache[region_id] = cached
                return cached
            child_refs = sorted(
                region_ref(child)
                for child in direct_children.get(region_id, ())
            )
            member_refs = sorted(
                node_ref(member.node_id, semantic=True)
                for member in direct_members.get(region_id, ())
                if member.node_id not in representation_only
            )
            member_refs.extend(
                sorted(
                    terminal_fork_ref(terminal_fork)
                    for terminal_fork in direct_terminal_forks.get(
                        region_id, ()
                    )
                )
            )
            payload = {
                "owner": node_ref(region_id, semantic=True),
                "children": child_refs,
                "members": member_refs,
                "effect_order": direct_effect_events(region_id),
            }
            cached = f"region[{operation.name}:{_digest(payload)[:20]}]"
            region_ref_cache[region_id] = cached
            return cached

        diagnostic_effect_traces: list[
            tuple[str, str, tuple[str, ...]]
        ] = []
        for parent in [None, *sorted(region_ids)]:
            parent_ref = "region[root]" if parent is None else region_ref(parent)
            if parent in structured_if_ids:
                fingerprint = structured_if_fingerprint(parent)
                control = region_ref(parent)
                edges.extend(
                    [
                        (fingerprint.predicate, control, "condition"),
                        (
                            structured_if_arm_ref(fingerprint.positive_arm),
                            control,
                            "positive",
                        ),
                        (
                            structured_if_arm_ref(fingerprint.negative_arm),
                            control,
                            "negative",
                        ),
                    ]
                )
                if fingerprint.result_type is not None:
                    assert fingerprint.positive_arm.yielded_value is not None
                    assert fingerprint.negative_arm.yielded_value is not None
                    edges.extend(
                        [
                            (
                                fingerprint.positive_arm.yielded_value,
                                control,
                                "positive_yield",
                            ),
                            (
                                fingerprint.negative_arm.yielded_value,
                                control,
                                "negative_yield",
                            ),
                        ]
                    )
                continue
            for child in direct_children.get(parent, ()):
                edges.append((parent_ref, region_ref(child), "region.child"))
            for member in direct_members.get(parent, ()):
                if member.node_id in representation_only:
                    continue
                edges.append(
                    (
                        parent_ref,
                        node_ref(member.node_id, semantic=True),
                        "region.member",
                    )
                )
            for terminal_fork in direct_terminal_forks.get(parent, ()):
                fingerprint = terminal_fork_fingerprint(terminal_fork)
                control = terminal_fork_ref(terminal_fork)
                edges.extend(
                    [
                        (
                            fingerprint.predicate,
                            control,
                            "condition",
                        ),
                        (
                            terminal_arm_ref(fingerprint.true_arm),
                            control,
                            "true",
                        ),
                        (
                            terminal_arm_ref(fingerprint.false_arm),
                            control,
                            "false",
                        ),
                        (
                            parent_ref,
                            control,
                            "region.member",
                        ),
                    ]
                )
            effect_events = direct_effect_events(parent)
            if effect_events:
                trace_ref = f"effect.trace[{_digest(effect_events)[:20]}]"
                edges.append(
                    (
                        parent_ref,
                        trace_ref,
                        "effect.order",
                    )
                )
                diagnostic_effect_traces.append(
                    (trace_ref, parent_ref, tuple(effect_events))
                )

        reductions: list[ReductionFingerprint] = []
        for operation in operations:
            if operation.name != "tt.reduce":
                continue
            if operation.axis is None:
                raise TTIRParseError("tt.reduce is missing its axis attribute")
            combiners: list[str] = []
            for result_name in operation.region_results:
                producer = node_by_id[result_to_node[result_name]]
                combiners.append(producer.name)
            if not combiners:
                raise TTIRParseError("tt.reduce region has no returned combiner")
            reductions.append(
                ReductionFingerprint(
                    axis=operation.axis,
                    combiners=tuple(combiners),
                    input_count=operation.header_operand_count,
                    result_count=max(1, len(operation.result_names)),
                )
            )

        memory_effects: list[MemoryEffect] = []
        diagnostic_memory_transfers: list[MemoryEffect] = []
        diagnostic_memory_predicate_evidence: list[
            tuple[str, bool, int, int, str | None, int | None, int | None, bool, _SourceCallPath | None, int | None, _SourceCallPath]
        ] = []
        diagnostic_memory_owner_evidence: list[
            tuple[str, bool, int, int, str | None, int | None, int | None, bool, _SourceCallPath]
        ] = []
        effect_counts: Counter[str] = Counter()

        direct_users: dict[str, list[_Operation]] = defaultdict(list)
        for candidate in operations:
            for operand in candidate.operands:
                direct_users[operand].append(candidate)

        def value_ref(value: str) -> str:
            """Expose typed constants while keeping other SSA values opaque."""

            node_id = result_to_node.get(value)
            if node_id is not None:
                producer = node_by_id[node_id]
                if producer.name == "arith.constant":
                    return producer.local_signature
            return operand_ref(value, semantic=True)

        def direct_constant_ref(value: str) -> str | None:
            """Return an exact typed constant only for a direct SSA producer."""

            node_id = result_to_node.get(value)
            if node_id is None:
                return None
            producer = node_by_id[node_id]
            return (
                producer.local_signature
                if producer.name == "arith.constant"
                else None
            )

        def predicate_evidence(
            operation: _Operation,
            *,
            kind: str,
            masked: bool,
            predicate_operand: int | None,
        ) -> None:
            """Emit an occurrence-aligned, fail-closed predicate receipt."""

            access_location = operation_source_locations.get(operation.node_id)
            if access_location is None:
                return
            _access_operation, access_line, access_column = access_location
            predicate_operation: str | None = None
            predicate_line: int | None = None
            predicate_column: int | None = None
            predicate_is_i1 = False
            predicate_call_path: _SourceCallPath | None = None
            predicate_producer_id: int | None = None
            if masked and predicate_operand is not None:
                if predicate_operand >= len(operation.operands):
                    return
                producer_id = result_to_node.get(operation.operands[predicate_operand])
                producer = node_by_id.get(producer_id) if producer_id is not None else None
                producer_location = (
                    operation_source_locations.get(producer.node_id)
                    if producer is not None else None
                )
                if producer is not None:
                    predicate_operation = producer.name
                    predicate_producer_id = producer.node_id
                    if producer_location is not None:
                        _producer_operation, predicate_line, predicate_column = producer_location
                        predicate_call_path = producer.source_call_path
                    # The producer's local signature normally describes its
                    # *input* operands (notably for comparisons), so it is not
                    # sound type evidence for the value feeding the memory
                    # predicate.  Use the parser-validated result type for
                    # this exact SSA result instead.  This is only a
                    # type-profile prerequisite; the Source ledger and
                    # boundary validator still prove the target relation and
                    # projection separately.
                    try:
                        result_index = producer.result_names.index(
                            operation.operands[predicate_operand]
                        )
                        predicate_type = producer.result_types[result_index]
                    except (IndexError, ValueError):
                        predicate_type = None
                    predicate_is_i1 = _is_i1_predicate_type(predicate_type)
            diagnostic_memory_predicate_evidence.append(
                (
                    kind,
                    masked,
                    access_line,
                    access_column,
                    predicate_operation,
                    predicate_line,
                    predicate_column,
                    predicate_is_i1,
                    predicate_call_path,
                    predicate_producer_id,
                    operation.source_call_path,
                )
            )

        def owner_evidence(
            operation: _Operation,
            *,
            kind: str,
            masked: bool,
        ) -> None:
            """Emit one exact direct value-edge receipt for a memory access."""

            access_location = operation_source_locations.get(operation.node_id)
            if access_location is None:
                return
            _access_operation, access_line, access_column = access_location
            owner: _Operation | None = None
            if kind == "load" and len(operation.result_names) == 1:
                users = direct_users.get(operation.result_names[0], ())
                if len(users) == 1:
                    owner = users[0]
            elif kind == "store" and len(operation.operands) >= 2:
                producer_id = result_to_node.get(operation.operands[1])
                owner = (
                    node_by_id.get(producer_id)
                    if producer_id is not None
                    else None
                )
            owner_location = (
                operation_source_locations.get(owner.node_id)
                if owner is not None else None
            )
            owner_line = owner_location[1] if owner_location is not None else None
            owner_column = owner_location[2] if owner_location is not None else None
            diagnostic_memory_owner_evidence.append(
                (
                    kind,
                    masked,
                    access_line,
                    access_column,
                    owner.name if owner is not None else None,
                    owner_line,
                    owner_column,
                    owner is not None,
                    operation.source_call_path,
                )
            )

        for operation in operations:
            if operation.name == "tt.load":
                masked = operation.header_operand_count >= 2
                predicate = (
                    operand_ref(operation.operands[1], semantic=True)
                    if masked else None
                )
                other = (
                    value_ref(operation.operands[2])
                    if operation.header_operand_count >= 3
                    else None
                )
                effect = MemoryEffect(
                    "load", masked, predicate, other, operation.local_signature,
                    operation.source_call_path,
                )
                memory_effects.append(effect)
                diagnostic_memory_transfers.append(effect)
                predicate_evidence(
                    operation,
                    kind="load",
                    masked=masked,
                    predicate_operand=1 if masked else None,
                )
                owner_evidence(operation, kind="load", masked=masked)
                effect_counts["read.masked" if masked else "read"] += 1
            elif operation.name == "tt.store":
                masked = operation.header_operand_count >= 3
                predicate = (
                    operand_ref(operation.operands[2], semantic=True)
                    if masked else None
                )
                effect = MemoryEffect(
                    "store",
                    masked,
                    predicate,
                    direct_constant_ref(operation.operands[1]),
                    operation.local_signature,
                    operation.source_call_path,
                )
                memory_effects.append(effect)
                # A store value is not a load ``other=`` fill.  Keep the
                # occurrence-aligned sidecar explicit so consumers cannot
                # accidentally treat a constant store as a masked-load fill.
                diagnostic_memory_transfers.append(
                    MemoryEffect(
                        "store",
                        masked,
                        predicate,
                        None,
                        operation.local_signature,
                        operation.source_call_path,
                    )
                )
                predicate_evidence(
                    operation,
                    kind="store",
                    masked=masked,
                    predicate_operand=2 if masked else None,
                )
                owner_evidence(operation, kind="store", masked=masked)
                effect_counts["write.masked" if masked else "write"] += 1
            elif operation.name.startswith("tt.atomic"):
                effect_counts[operation.name] += 1
            elif operation.name in {"tt.assert", "tt.device_assert"}:
                effect_counts[operation.name] += 1
            elif "barrier" in operation.name or operation.name in {
                "gpu.wait",
                "tt.async_wait",
            }:
                effect_counts[operation.name] += 1
            elif operation.name == "scf.for":
                effect_counts["control.scf.for"] += 1
            elif operation.name == "scf.if":
                effect_counts["control.scf.if"] += 1

        dots = [
            DotFingerprint(
                operation.name,
                operation.header_operand_count,
                operation.local_signature,
            )
            for operation in operations
            if operation.name in {"tt.dot", "tt.dot_scaled"}
        ]

        loops: list[LoopFingerprint] = []
        normalize_address_captures = (
            "tt.addptr" in representation_anchor_exemptions
        )
        for operation in operations:
            if operation.name != "scf.for":
                continue
            header = [operand_ref(name, semantic=True) for name in operation.operands]
            if len(header) < 3:
                raise TTIRParseError("scf.for has fewer than lower/upper/step operands")
            body_effects = Counter()
            body_ops: Counter[str] = Counter()
            local_region_arguments = set(operation.region_block_args)
            for nested_region in operations:
                if operation.node_id in nested_region.context:
                    local_region_arguments.update(
                        nested_region.region_block_args
                    )
            captured_values: set[str] = set()
            region_argument_use_counts: Counter[str] = Counter()
            directly_nested_forks = [
                terminal_fork
                for terminal_fork in terminal_forks
                if terminal_fork.context
                and terminal_fork.context[-1] == operation.node_id
            ]
            directly_nested_fork_ids = {
                terminal_fork.fork_id
                for terminal_fork in directly_nested_forks
            }
            nested_structured_if_ids = {
                nested.node_id
                for nested in operations
                if nested.name == "scf.if"
                and operation.node_id in nested.context
                and nested.fork_id not in directly_nested_fork_ids
            }
            for nested in operations:
                if operation.node_id not in nested.context:
                    continue
                if nested.fork_id in directly_nested_fork_ids:
                    # The mutually exclusive arms are summarized by one
                    # role-preserving synthetic control member below.  Do not
                    # turn them into an unconditional flat loop body.
                    continue
                inside_structured_if = (
                    nested.structured_if_id in nested_structured_if_ids
                )
                if (
                    nested.name not in _TERMINATORS
                    and not inside_structured_if
                    and nested.node_id not in representation_only
                ):
                    body_ops[
                        _semantic_op_name(
                            nested, node_by_id, result_to_node
                        )
                    ] += 1
                nested_values = [*nested.operands, *nested.region_results]
                for operand_index, operand in enumerate(nested_values):
                    if normalize_address_captures:
                        # Once every authored J access has a typed address-
                        # representation binding, the hard memory slice already
                        # retains the exact pointer polynomial, transfer kind,
                        # predicate, fill, value, effect occurrence and order.
                        # Do not count the same pure address spelling again as a
                        # semantic loop capture or induction use: computing a
                        # base pointer outside the loop and recomputing it inside
                        # are the same closed realization when the surviving
                        # memory operation still has the same hard pointer form.
                        #
                        # Representation nodes reaching a stored value or a
                        # predicate are excluded from ``representation_only`` by
                        # construction and therefore remain visible.  For an
                        # effect operation only operand zero is the pointer;
                        # store/update values and all predicates remain hard.
                        if nested.node_id in representation_only:
                            continue
                        if operand_index == 0 and (
                            nested.name in {"tt.load", "tt.store"}
                            or nested.name.startswith("tt.atomic")
                        ):
                            continue
                    if operand in local_region_arguments:
                        region_argument_use_counts[operand] += 1
                        continue
                    producer_id = result_to_node.get(operand)
                    if producer_id is not None:
                        producer = node_by_id[producer_id]
                        if operation.node_id in producer.context:
                            continue
                    captured_values.add(operand_ref(operand, semantic=True))
                if inside_structured_if:
                    # Arm-local operations are represented by the strong,
                    # polarity-preserving StructuredIfFingerprint.  Counting
                    # them here would flatten mutually exclusive execution.
                    continue
                if nested.name == "tt.load":
                    body_effects[
                        "read.masked" if nested.header_operand_count >= 2 else "read"
                    ] += 1
                elif nested.name == "tt.store":
                    body_effects[
                        "write.masked" if nested.header_operand_count >= 3 else "write"
                    ] += 1
                elif nested.name.startswith("tt.atomic"):
                    body_effects[nested.name] += 1
            for terminal_fork in directly_nested_forks:
                body_ops["control.conditional"] += 1
                if terminal_fork_has_effect(terminal_fork):
                    body_effects["control.conditional"] += 1
                predicate = terminal_fork.predicate
                producer_id = result_to_node.get(predicate)
                if producer_id is None or operation.node_id not in node_by_id[
                    producer_id
                ].context:
                    captured_values.add(
                        operand_ref(predicate, semantic=True)
                    )
            for owner_id in nested_structured_if_ids:
                if region_has_effect(owner_id):
                    body_effects["control.scf.if"] += 1
            loops.append(
                LoopFingerprint(
                    lower=header[0],
                    upper=header[1],
                    step=header[2],
                    init_values=tuple(header[3:]),
                    yielded_values=tuple(
                        operand_ref(name, semantic=True)
                        for name in operation.region_results
                    ),
                    body_effects=tuple(sorted(body_effects.items())),
                    block_argument_count=len(operation.region_block_args),
                    induction_use_count=(
                        0
                        if not operation.region_block_args
                        else region_argument_use_counts[
                            operation.region_block_args[0]
                        ]
                    ),
                    carried_argument_use_counts=tuple(
                        region_argument_use_counts[name]
                        for name in operation.region_block_args[1:]
                    ),
                    captured_values=tuple(sorted(captured_values)),
                    body_op_multiset=tuple(sorted(body_ops.items())),
                )
            )

        access, diagnostic_memory_effect_origins = self._infer_access(
            function, operations, result_to_node, leaf_labels, node_by_id
        )
        interface_arguments: list[InterfaceArgument] = []
        for position, (name, type_text, attrs) in enumerate(function.arguments):
            dtype = _pointer_dtype(type_text)
            if dtype is not None:
                kind = "pointer"
                rank = None
                argument_access = access.get(position, "unused")
            else:
                tensor = re.fullmatch(r"tensor<(?P<body>.*)>", type_text)
                if tensor:
                    kind = "tensor"
                    pieces = tensor.group("body").split("x")
                    rank = max(0, len(pieces) - 1)
                    dtype = pieces[-1]
                else:
                    kind = "scalar"
                    rank = 0
                    dtype = type_text
                argument_access = "value"
                # The exact Python ABI is checked before observation.  At TTIR,
                # Triton may specialize or retain scalar carriers differently
                # for equivalent block-pointer and direct-address forms.  Their
                # compiled-function signature presence is therefore not a
                # J/C-owned structural atom; semantic scalar use remains visible
                # through the operation/dataflow fingerprint below.
                continue
            interface_arguments.append(
                InterfaceArgument(
                    position,
                    type_text,
                    kind,
                    dtype,
                    rank,
                    argument_access,
                    attrs,
                )
            )

        memory_effects.sort(key=lambda item: _stable_json(item.to_dict()))
        reductions.sort(key=lambda item: _stable_json(item.to_dict()))
        dots.sort(key=lambda item: _stable_json(item.to_dict()))
        loops.sort(key=lambda item: _stable_json(item.to_dict()))
        terminal_fork_fingerprints = sorted(
            (
                terminal_fork_fingerprint(item)
                for item in terminal_forks
            ),
            key=lambda item: _stable_json(item.to_dict()),
        )
        structured_if_fingerprints = sorted(
            (
                structured_if_fingerprint(owner_id)
                for owner_id in structured_if_ids
            ),
            key=lambda item: (
                item.region_context,
                item.enclosing_terminal_arm or "",
                item.ordinal,
                _stable_json(item.to_dict()),
            ),
        )
        mask_predicates = sorted(
            effect.predicate
            for effect in memory_effects
            if effect.predicate is not None
        )

        def diagnostic_operand_profile(value: str) -> str:
            """Describe one immediate producer without recursive node hashes."""

            leaf = leaf_labels.get(value)
            if leaf is not None:
                return leaf
            producer_id = result_to_node[value]
            producer = node_by_id[producer_id]
            signature = producer.local_signature
            if len(producer.result_names) <= 1:
                return signature
            try:
                slot = producer.result_names.index(value)
            except ValueError as error:  # pragma: no cover - maps share construction
                raise TTIRParseError(
                    f"SSA value {value!r} is absent from its multi-result producer"
                ) from error
            return f"{signature}#result[{slot}]"

        def diagnostic_node_profile(
            operation: _Operation,
        ) -> tuple[
            str,
            str,
            tuple[tuple[str, str], ...],
            tuple[str, ...],
        ]:
            immediate = [
                diagnostic_operand_profile(value)
                for value in (*operation.operands, *operation.region_results)
            ]
            if operation.cmp_flipped and len(immediate) >= 2:
                immediate[0], immediate[1] = immediate[1], immediate[0]
            if (
                operation.name in _COMMUTATIVE_OPS
                or operation.cmp_predicate in _SYMMETRIC_CMP
            ):
                header = immediate[: operation.header_operand_count]
                header.sort()
                immediate = header + immediate[operation.header_operand_count :]

            inputs: list[tuple[str, str]] = []
            for index, producer in enumerate(immediate):
                if operation.name == "scf.if":
                    role = (
                        "condition"
                        if index == 0
                        else "positive_yield"
                        if index == 1
                        else "negative_yield"
                    )
                elif index < operation.header_operand_count:
                    role = f"operand[{index}]"
                else:
                    role = (
                        f"region_result[{index - operation.header_operand_count}]"
                    )
                inputs.append((role, producer))

            context = tuple(
                node_by_id[owner].name for owner in operation.context
            )
            if operation.structured_if_arm is not None:
                context += (f"structured_if_arm={operation.structured_if_arm}",)
            if operation.fork_arm is not None:
                context += (f"terminal_fork_arm={operation.fork_arm}",)
            return (
                node_ref(operation.node_id, semantic=True),
                operation.local_signature,
                tuple(inputs),
                context,
            )

        diagnostic_memory_address_forms: list[
            tuple[
                str,
                str,
                tuple[tuple[tuple[str, ...], int], ...],
                tuple[str, ...],
                str | None,
                tuple[
                    tuple[
                        tuple[tuple[tuple[str, ...], int], ...],
                        str | None,
                        tuple[str, ...],
                    ],
                    ...,
                ],
                _SourceCallPath,
            ]
        ] = []
        for operation in operations:
            if operation.name not in {"tt.load", "tt.store"} and not operation.name.startswith(
                "tt.atomic"
            ):
                continue
            if not operation.operands:
                continue
            pointer = pointer_form(operation.operands[0])
            if pointer is None:
                continue
            context = tuple(
                f"{node_by_id[owner].name}@{owner}"
                for owner in operation.context
            )
            if operation.structured_if_arm is not None:
                context += (f"structured_if_arm={operation.structured_if_arm}",)
            if operation.fork_arm is not None:
                context += (f"terminal_fork_arm={operation.fork_arm}",)
            diagnostic_memory_address_forms.append(
                (
                    _semantic_op_name(operation, node_by_id, result_to_node),
                    pointer[0],
                    polynomial_payload(pointer[1]),
                    context,
                    pointer[2],
                    pointer[3],
                    operation.source_call_path,
                )
            )

        return TTIRFingerprint(
            interface=RuntimeInterface(
                tuple(interface_arguments),
                function.results,
                _canonical_python_interface(python_interface),
            ),
            op_multiset=tuple(sorted(op_counts.items())),
            dataflow_edges=tuple(sorted(edges)),
            reductions=tuple(reductions),
            memory_effects=tuple(memory_effects),
            mask_predicates=tuple(mask_predicates),
            effect_multiset=tuple(sorted(effect_counts.items())),
            dots=tuple(dots),
            loops=tuple(loops),
            structured_ifs=tuple(structured_if_fingerprints),
            terminal_forks=tuple(terminal_fork_fingerprints),
            schedule_view=_canonical_schedule(schedule),
            diagnostic_node_signatures=tuple(
                sorted(
                    (
                        node_ref(operation.node_id, semantic=True),
                        operation.local_signature,
                    )
                    for operation in operations
                    if operation.name not in _TERMINATORS
                )
            ),
            diagnostic_node_profiles=tuple(
                sorted(
                    diagnostic_node_profile(operation)
                    for operation in operations
                    if operation.name not in _TERMINATORS
                )
            ),
            diagnostic_representation_profiles=tuple(
                sorted(
                    {
                        (
                            reference,
                            result_contract(node_by_id[result_to_node[name]]),
                            representation_atoms(name),
                        )
                        for name, reference in representation_ref_cache.items()
                    }
                )
            ),
            diagnostic_iteration_bases=_diagnostic_iteration_bases(
                function,
                operations,
                result_to_node,
                leaf_labels,
            ),
            diagnostic_effect_traces=tuple(
                sorted(diagnostic_effect_traces)
            ),
            diagnostic_memory_transfers=tuple(diagnostic_memory_transfers),
            diagnostic_memory_effect_origins=diagnostic_memory_effect_origins,
            diagnostic_memory_address_forms=tuple(
                diagnostic_memory_address_forms
            ),
            diagnostic_source_locations=source_locations,
            diagnostic_memory_source_locations=diagnostic_memory_source_locations,
            diagnostic_memory_predicate_evidence=tuple(
                diagnostic_memory_predicate_evidence
            ),
            diagnostic_memory_owner_evidence=tuple(
                diagnostic_memory_owner_evidence
            ),
            diagnostic_scalar_owner_graphs=scalar_owner_graphs,
            diagnostic_tensor_owner_graphs=tensor_owner_graphs,
        )

    def _infer_access(
        self,
        function: _FunctionParts,
        operations: list[_Operation],
        result_to_node: dict[str, int],
        leaf_labels: dict[str, str],
        node_by_id: dict[int, _Operation],
    ) -> tuple[
        dict[int, str],
        tuple[tuple[str, tuple[int, ...], tuple[str, ...]], ...],
    ]:
        arg_position = {
            name: index for index, (name, _, _) in enumerate(function.arguments)
        }
        value_sources: dict[str, tuple[str, ...]] = {}

        for operation in operations:
            if operation.name == "tt.scan":
                input_count = len(operation.operands)
                if (
                    input_count < 1
                    or len(operation.result_names) != input_count
                    or len(operation.region_results) != input_count
                    or len(operation.region_block_args) != 2 * input_count
                ):
                    raise TTIRParseError(
                        "tt.scan recurrence arities changed after call expansion"
                    )
                for index, block_arg in enumerate(operation.region_block_args):
                    value_sources[block_arg] = (
                        operation.operands[index % input_count],
                    )
                for result_name, initial, returned in zip(
                    operation.result_names,
                    operation.operands,
                    operation.region_results,
                    strict=True,
                ):
                    value_sources[result_name] = (initial, returned)
                continue
            if operation.name != "scf.for":
                sources = tuple(
                    [*operation.operands, *operation.region_results]
                    if operation.name == "scf.if"
                    else operation.operands
                )
                for result_name in operation.result_names:
                    value_sources[result_name] = sources
                continue

            # scf.for has three control operands followed by one init value per
            # iter_arg.  Its first region block argument is the induction
            # variable; the remaining block arguments, operation results, and
            # scf.yield operands correspond positionally.  A carried value may
            # depend both on its initial value and on values captured by the
            # yielded update.  Keep those equations per slot: joining all
            # iter_args would incorrectly turn unrelated input/output pointers
            # into inout arguments.
            carried_block_args = tuple(operation.region_block_args[1:])
            init_values = tuple(operation.operands[3:])
            yielded_values = tuple(operation.region_results)
            result_names = tuple(operation.result_names)
            arities = {
                len(carried_block_args),
                len(init_values),
                len(yielded_values),
                len(result_names),
            }
            if len(arities) != 1:
                raise TTIRParseError(
                    "scf.for iter_args, results, block arguments, and yield "
                    "values must have identical arity"
                )
            if not operation.region_block_args:
                raise TTIRParseError(
                    "scf.for is missing its induction block argument"
                )

            value_sources[operation.region_block_args[0]] = tuple(
                operation.operands[:3]
            )
            for block_arg, result_name, initial, yielded in zip(
                carried_block_args,
                result_names,
                init_values,
                yielded_values,
                strict=True,
            ):
                sources = (initial, yielded)
                value_sources[block_arg] = sources
                value_sources[result_name] = sources

        # Region-carried SSA contains intentional cycles: a yielded pointer is
        # commonly computed from the corresponding iter_arg.  Recursive DFS
        # either drops that dependency at a region leaf or rejects the valid
        # cycle.  Compute the least fixed point of function-argument provenance
        # instead.  The lattice is finite (sets of argument positions), so this
        # terminates deterministically and also captures a different pointer
        # introduced by a yielded update.
        dependencies: dict[str, set[int]] = {
            name: {position} for name, position in arg_position.items()
        }
        for name in leaf_labels:
            dependencies.setdefault(name, set())
        for name in result_to_node:
            dependencies.setdefault(name, set())

        changed = True
        while changed:
            changed = False
            for name, sources in value_sources.items():
                inferred = (
                    set().union(
                        *(dependencies[source] for source in sources)
                    )
                    if sources
                    else set()
                )
                if not inferred.issubset(dependencies[name]):
                    dependencies[name].update(inferred)
                    changed = True

        # Preserve the pre-existing rejection of an ordinary cyclic SSA
        # producer graph.  Valid loop-carried cycles pass through region block
        # arguments (leaves in this structural walk) and are handled only by
        # the fixed-point equations above.
        acyclic: set[str] = set()

        def validate_acyclic(
            name: str,
            active: frozenset[int] = frozenset(),
        ) -> None:
            if name in acyclic or name in leaf_labels:
                return
            node_id = result_to_node[name]
            if node_id in active:
                raise TTIRParseError(
                    "cyclic SSA dependency while inferring interface"
                )
            for operand in node_by_id[node_id].operands:
                validate_acyclic(operand, active | {node_id})
            acyclic.add(name)

        modes: dict[int, set[str]] = {}
        diagnostic_memory_effect_origins: list[
            tuple[str, tuple[int, ...], tuple[str, ...]]
        ] = []
        for operation in operations:
            if not operation.operands:
                continue
            if operation.name == "tt.load":
                mode = "read"
            elif operation.name == "tt.store":
                mode = "write"
            elif operation.name.startswith("tt.atomic"):
                mode = "readwrite"
            else:
                continue
            validate_acyclic(operation.operands[0])
            pointer_origins = tuple(sorted(dependencies[operation.operands[0]]))
            for position in pointer_origins:
                modes.setdefault(position, set()).add(mode)
            if operation.name in {"tt.load", "tt.store"}:
                # The numeric suffix is used only to distinguish two sibling
                # regions inside this one compiled fingerprint.  It is never
                # serialized or compared across Source and Replay.
                context = tuple(
                    f"{node_by_id[owner].name}@{owner}"
                    for owner in operation.context
                )
                if operation.structured_if_arm is not None:
                    context += (
                        f"structured_if_arm={operation.structured_if_arm}",
                    )
                if operation.fork_arm is not None:
                    context += (f"terminal_fork_arm={operation.fork_arm}",)
                diagnostic_memory_effect_origins.append(
                    (
                        _semantic_op_name(
                            operation, node_by_id, result_to_node
                        ),
                        pointer_origins,
                        context,
                    )
                )
        access: dict[int, str] = {}
        for position, seen in modes.items():
            read = bool(seen & {"read", "readwrite"})
            write = bool(seen & {"write", "readwrite"})
            access[position] = "inout" if read and write else "input" if read else "output"
        return access, tuple(diagnostic_memory_effect_origins)


def _semantic_op_name(
    operation: _Operation,
    node_by_id: Mapping[int, _Operation],
    result_to_node: Mapping[str, int],
) -> str:
    if operation.name == "tt.load":
        return "tt.load.masked" if operation.header_operand_count >= 2 else "tt.load"
    if operation.name == "tt.store":
        return "tt.store.masked" if operation.header_operand_count >= 3 else "tt.store"
    if operation.name == "scf.if":
        return "control.scf.if"
    return operation.name


def parse_ttir(
    text: str,
    *,
    python_interface: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
    representation_anchor_exemptions: frozenset[str] = frozenset(),
    runtime_values: Mapping[str, int | bool] | None = None,
    external_math: Mapping[str, Any] | None = None,
    external_math_source: str | None = None,
) -> TTIRFingerprint:
    """Parse TTIR and return its normalized Gate1 fingerprint."""

    return TTIRParser().parse(
        text,
        python_interface=python_interface,
        schedule=schedule,
        representation_anchor_exemptions=representation_anchor_exemptions,
        runtime_values=runtime_values,
        external_math=external_math,
        external_math_source=external_math_source,
    )


def runtime_argument_names(text: str) -> tuple[str, ...]:
    """Return surviving TTIR runtime argument names in ABI order.

    Triton may specialize ordinary scalar Python parameters and remove them
    from the TTIR function signature.  The remaining SSA names retain their
    Python parameter spelling and let Gate1 validate the surviving subsequence
    without pretending that specialization changed the public Python ABI.
    """

    cleaned = _strip_locations(text)
    function = _parse_function(cleaned)
    return tuple(name.removeprefix("%") for name, _, _ in function.arguments)


def compiled_program_id_axes(ttir: str) -> tuple[int, ...] | None:
    """Return the literal Triton launch-axis multiset from one TTIR observation.

    Both Replay admission and unary Gate1 normalization consume this parser so
    a ``pi.program_axes`` witness and the compiled-obligation check name the
    same physical x/y/z observation.  Unknown syntax stays unavailable; this
    helper never infers an axis from operation order or from the peer TTIR.
    """

    axes: list[int] = []
    axis_names = {"x": 0, "y": 1, "z": 2}
    for line in ttir.splitlines():
        if "tt.get_program_id" not in line:
            continue
        named = re.search(r"\btt\.get_program_id\s+([xyz])(?:\s|:)", line)
        if named is not None:
            axes.append(axis_names[named.group(1)])
            continue
        generic = re.search(r"\baxis\s*=\s*([0-2])(?:\s*:\s*i32)?", line)
        if generic is not None:
            axes.append(int(generic.group(1)))
            continue
        return None
    return tuple(sorted(axes))


def fingerprint_ttir(
    text: str,
    *,
    python_interface: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
    representation_anchor_exemptions: frozenset[str] = frozenset(),
    runtime_values: Mapping[str, int | bool] | None = None,
    external_math: Mapping[str, Any] | None = None,
    external_math_source: str | None = None,
) -> TTIRFingerprint:
    return parse_ttir(
        text,
        python_interface=python_interface,
        schedule=schedule,
        representation_anchor_exemptions=representation_anchor_exemptions,
        runtime_values=runtime_values,
        external_math=external_math,
        external_math_source=external_math_source,
    )


extract_ttir_fingerprint = fingerprint_ttir
