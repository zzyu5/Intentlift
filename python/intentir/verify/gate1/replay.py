"""Source-blind Replay writers, ingress, and public synthetic compile probe."""

from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from intentir.ir.canonical_j import (
    canonicalize_gate1_artifact,
    elaborate_j,
    j_elaboration,
)
from intentir.extraction.providers.client import call_responses
from intentir.extraction.providers.provider import ProviderConfig, get_provider

from .codegen import (
    CodeExtractionError,
    UnsafeReplayCodeError,
    extract_python_module,
)
from .compiler import (
    EXTERNAL_MATH_FAILURE_RULES,
    CompileCase,
    CompileResult,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from .cuda_codegen import extract_cuda_module
from .cute_codegen import extract_cute_module
from .model import (
    Frontend,
    Gate1Artifact,
    SemanticArchive,
    _opaque_reference_data,
    opaque_replay_representation,
)
from .pi import (
    NORMALIZATION_WITNESS_SCHEMA,
    ProgramCoordinateProjectionError,
    canonical_program_coordinate_projection,
    per_program_scalar_keepdims_reductions,
)
from .prompt import (
    assert_source_isolated_payload,
    build_rewriter_prompt,
    build_rewriter_payload,
)
from .replay_semantic_core import (
    materialize_replay_compiler_index,
    materialize_replay_semantics,
)
from .tilelang_codegen import extract_tilelang_module
from .ttir import (
    TTIRFingerprint,
    TTIRParseError,
    compiled_program_id_axes,
    parse_ttir,
)


MINISWE_REPLAY_BACKEND = "minisweagent.intentir-replay.v34"
RESPONSES_REPLAY_BACKEND = "openai.responses.intentir-replay.v1"
STATIC_REPLAY_BACKEND = "intentir.static-replay-test.v1"
REPLAY_CHECK_PROTOCOL = "intentir.scrc.replay-check.v15"
_ANONYMOUS_HANDLE = re.compile(r"^[pc][0-9]+$")


def _canonical_packet_j(
    packet: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Return the accepted J wire and its one typed construction view.

    Replay preflight used to read individual J tables directly.  That made a
    stale scope/type row appear valid when the canonical elaborator would have
    rejected or rewritten the same artifact.  Packet consumers now enter
    through the same J authority as Gate1; a malformed packet fails at this
    boundary instead of receiving a second interpretation.
    """

    raw = packet.get("j")
    if not isinstance(raw, Mapping):
        raise ValueError("Replay packet J must be an object")
    accepted = elaborate_j(SemanticArchive.from_json(copy.deepcopy(dict(raw))))
    return accepted.to_json(), j_elaboration(accepted)


@dataclass(frozen=True)
class ReplayGeneration:
    code: str
    function_name: str
    model: str
    response_id: str | None
    context_receipt: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, int] = field(default_factory=dict)
    response_model: str | None = None
    entry_function_name: str | None = None
    response_text: str = field(default="", repr=False)
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    trajectory: Mapping[str, Any] = field(default_factory=dict, repr=False)
    backend_protocol: str = RESPONSES_REPLAY_BACKEND
    physical_provider_calls: int = 0
    transport_retries: int = 0


@dataclass(frozen=True)
class ReplayAttempt:
    model: str
    diagnostic: str
    response_text: str = field(repr=False)
    response_id: str | None = None
    response_model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    trajectory: Mapping[str, Any] = field(default_factory=dict, repr=False)
    backend_protocol: str = RESPONSES_REPLAY_BACKEND
    physical_provider_calls: int = 0
    transport_retries: int = 0
    ingress_diagnostics: tuple[Mapping[str, Any], ...] = ()


class ReplayGenerationError(RuntimeError):
    def __init__(self, attempt: ReplayAttempt) -> None:
        self.attempt = attempt
        super().__init__(f"Replay response rejected: {attempt.diagnostic}")


class ReplayWriter(Protocol):
    def generate(
        self,
        artifact: Gate1Artifact,
        *,
        masked: bool = False,
        callable_abi_order: Sequence[str] | None = None,
    ) -> ReplayGeneration: ...


class RecordedReplayWriter(ReplayWriter, Protocol):
    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]: ...


@dataclass(frozen=True)
class ReplayIngressResult:
    """Closed result of Replay ingress or its source-blind compile probe."""

    status: str
    code: str | None = field(default=None, repr=False)
    entry_function_name: str | None = None
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    def to_check_response(self) -> dict[str, Any]:
        # The Replay worker receives a deliberately closed diagnostic wire.  A
        # local ingress result may retain richer host-side context (for
        # example expected/observed values), but the isolated environment
        # accepts exactly these four fields so a rejected source remains a
        # repairable Replay failure rather than an environment RuntimeError.
        wire_diagnostics = [
            {
                key: item.get(key)
                for key in ("rule_code", "path", "anonymous_handles", "line")
            }
            for item in self.diagnostics
        ]
        return {
            "protocol": REPLAY_CHECK_PROTOCOL,
            "status": self.status,
            "diagnostics": wire_diagnostics,
        }


def _closed_attempt_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Retain only bounded source-free Replay ingress metadata."""

    result: list[dict[str, Any]] = []
    for item in diagnostics[:20]:
        if not isinstance(item, Mapping):
            continue
        rule_code = item.get("rule_code")
        path = item.get("path")
        line = item.get("line")
        handles = item.get("anonymous_handles")
        if not isinstance(rule_code, str) or not isinstance(path, str):
            continue
        projected: dict[str, Any] = {
            "rule_code": rule_code[:128],
            "path": path[:256],
            "anonymous_handles": (
                [
                    value[:64]
                    for value in handles[:32]
                    if isinstance(value, str)
                    and _ANONYMOUS_HANDLE.fullmatch(value) is not None
                ]
                if isinstance(handles, (list, tuple))
                else []
            ),
            "line": (
                line
                if isinstance(line, int)
                and not isinstance(line, bool)
                and line > 0
                else None
            ),
        }
        exception_type = item.get("exception_type")
        if (
            isinstance(exception_type, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", exception_type)
            is not None
        ):
            projected["exception_type"] = exception_type
        failure_class = item.get("failure_class")
        if (
            isinstance(failure_class, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}", failure_class)
            is not None
        ):
            projected["failure_class"] = failure_class
        result.append(projected)
    return tuple(result)


class _ReplayABIViolation(ValueError):
    def __init__(
        self,
        rule_code: str,
        *,
        path: str,
        handles: tuple[str, ...] = (),
        line: int | None = None,
        expected: Mapping[str, Any] | None = None,
        observed: Mapping[str, Any] | None = None,
    ) -> None:
        self.rule_code = rule_code
        self.path = path
        self.handles = handles
        self.line = line
        self.expected = dict(expected) if expected is not None else None
        self.observed = dict(observed) if observed is not None else None
        super().__init__(rule_code)


def _packet_c_slots(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    c = packet.get("c")
    if not isinstance(c, Mapping):
        return []
    slots = c.get("slots")
    if not isinstance(slots, list):
        return []
    return [item for item in slots if isinstance(item, Mapping)]


def _assigned_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Starred):
        return _assigned_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for item in node.elts:
            result.update(_assigned_names(item))
        return result
    return set()


def _constant_truth(
    node: ast.AST,
    constants: Mapping[str, bool | int | float | str] | None = None,
) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)):
        return bool(node.value)
    if (
        isinstance(node, ast.Name)
        and constants is not None
        and node.id in constants
    ):
        return bool(constants[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _constant_truth(node.operand, constants)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_constant_truth(item, constants) for item in node.values]
        if isinstance(node.op, ast.And):
            if False in values:
                return False
            return True if all(item is True for item in values) else None
        if isinstance(node.op, ast.Or):
            if True in values:
                return True
            return False if all(item is False for item in values) else None
    return None


def _assert_packet_c_matches_artifact(
    packet: Mapping[str, Any], artifact: Gate1Artifact
) -> None:
    """Require Replay C to be the accepted Source/Harness-bound C.

    Gate1 constructs the Replay compile case only after ``_replay_case`` has
    matched every source C constexpr/compiler binding to the accepted artifact.
    Consequently ``artifact.c`` at this boundary is the already source-bound
    C, not a target-specific C synthesized for Replay.
    """
    c = packet.get("c")
    canonical_artifact = canonicalize_gate1_artifact(artifact)
    if not canonical_artifact.c.slots:
        if c is not None:
            raise ValueError("J-only Replay packet must omit C")
        return
    if not isinstance(c, Mapping):
        raise ValueError("Replay packet omitted required complete C")
    canonical_artifact.c.validate_complete_values(path="Replay artifact C")
    expected = canonical_artifact.rewriter_view(masked=False)["c"]
    if dict(c) != expected:
        raise ValueError("Replay packet complete C differs from the artifact")


def _assert_packet_representation_matches_artifact(
    packet: Mapping[str, Any], artifact: Gate1Artifact
) -> None:
    """Require the packet's typed representation view to be artifact-derived."""

    expected = opaque_replay_representation(canonicalize_gate1_artifact(artifact))
    u_f = packet.get("u_f")
    actual = u_f.get("representation_contract") if isinstance(u_f, Mapping) else None
    if expected:
        if not isinstance(actual, Mapping) or dict(actual) != expected:
            raise ValueError(
                "Replay packet representation contract differs from the artifact"
            )
    elif actual is not None:
        raise ValueError(
            "Replay packet contains representation ownership absent from the artifact"
        )


def _assert_packet_j_matches_artifact(
    packet: Mapping[str, Any], artifact: Gate1Artifact
) -> None:
    """Require the source-blind J projection to be exactly artifact-derived."""

    u_f = packet.get("u_f")
    masked = (
        isinstance(u_f, Mapping)
        and u_f.get("semantic_view") == "masked_j"
    )
    expected = canonicalize_gate1_artifact(artifact).rewriter_view(
        masked=masked
    )["j"]
    actual = packet.get("j")
    if not isinstance(actual, Mapping) or dict(actual) != expected:
        raise _ReplayABIViolation(
            "replay_packet_j_projection_mismatch",
            path="<j>",
        )


def _check_triton_address_carrier_uses(
    function: ast.FunctionDef,
    carrier_names: set[str],
    *,
    forbidden_carrier_names: set[str] | frozenset[str] = frozenset(),
) -> tuple[_ReplayABIViolation, ...]:
    """Allow null-logical-ref scalars only in a closed pointer-address slice.

    ``logical_ref=null`` is deliberately not a license to use an ABI scalar as
    a hidden logical extent.  Real Triton call ABIs nevertheless carry physical
    strides and related address operands, so rejecting every body reference
    would make a valid non-contiguous tensor lowering impossible.  This check
    resolves the small, source-independent distinction needed at ingress:

    * transfer pointer operands and pointer-producing base/stride expressions
      may consume a representation carrier;
    * masks, values, predicates, shapes, loop bounds, and all other semantic
      expressions may not consume it;
    * a local derived from a carrier is checked transitively, so it cannot be
      used once for addressing and then reused as a semantic value.

    The routine is intentionally syntactic and conservative.  It does not
    infer tensor layouts, J coordinates, or Source facts.
    """

    all_carrier_names = set(carrier_names) | set(forbidden_carrier_names)
    if not all_carrier_names:
        return ()

    def call_name(node: ast.Call) -> str:
        target = node.func
        if isinstance(target, ast.Attribute):
            return target.attr
        if isinstance(target, ast.Name):
            return target.id
        return ""

    def target_names(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, (ast.Tuple, ast.List)):
            names: list[str] = []
            for item in node.elts:
                names.extend(target_names(item))
            return tuple(names)
        if isinstance(node, ast.Starred):
            return target_names(node.value)
        return ()

    def transfer_pointer(node: ast.Call) -> ast.AST | None:
        if len(node.args) > 0:
            return node.args[0]
        for keyword in node.keywords:
            if keyword.arg in {"pointer", "ptr"}:
                return keyword.value
        return None

    def named_argument(
        node: ast.Call,
        position: int,
        names: set[str],
    ) -> ast.AST | None:
        if len(node.args) > position:
            return node.args[position]
        for keyword in node.keywords:
            if keyword.arg in names:
                return keyword.value
        return None

    transfer_names = {"load", "store"}

    # Keep all local definitions in one conservative lexical map.  If a local
    # is used as a pointer, all of its possible definitions must be address-only
    # as well.  Loop targets are intentionally not reconstructed here: a carrier
    # hidden in a loop iterator remains rejected rather than being guessed as an
    # address definition.
    definitions: dict[str, list[ast.AST]] = {}
    reassigned_carriers: dict[str, ast.Name] = {}
    for statement in ast.walk(function):
        if isinstance(statement, ast.Assign):
            value = statement.value
            for target in statement.targets:
                for name in target_names(target):
                    definitions.setdefault(name, []).append(value)
                    if name in all_carrier_names and isinstance(target, ast.Name):
                        reassigned_carriers.setdefault(name, target)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            value = statement.value
            for name in target_names(statement.target):
                definitions.setdefault(name, []).append(value)
                if name in all_carrier_names and isinstance(statement.target, ast.Name):
                    reassigned_carriers.setdefault(name, statement.target)
        elif isinstance(statement, ast.AugAssign):
            for name in target_names(statement.target):
                definitions.setdefault(name, []).append(statement.value)
                if name in all_carrier_names and isinstance(statement.target, ast.Name):
                    reassigned_carriers.setdefault(name, statement.target)

    address_nodes: set[int] = set()
    visited_address_nodes: set[int] = set()
    visiting_names: set[str] = set()

    def mark_address_expression(expression: ast.AST | None) -> None:
        if expression is None:
            return
        expression_id = id(expression)
        if expression_id in visited_address_nodes:
            return
        visited_address_nodes.add(expression_id)

        if isinstance(expression, ast.Name):
            if isinstance(expression.ctx, ast.Load):
                address_nodes.add(expression_id)
                if expression.id not in carrier_names and expression.id not in visiting_names:
                    visiting_names.add(expression.id)
                    for definition in definitions.get(expression.id, ()):
                        mark_address_expression(definition)
                    visiting_names.remove(expression.id)
            return
        if isinstance(expression, ast.Constant):
            return
        if isinstance(expression, ast.Call):
            called = call_name(expression)
            if called in transfer_names or called.startswith("atomic_"):
                mark_address_expression(transfer_pointer(expression))
                return
            if called == "make_block_ptr":
                mark_address_expression(
                    named_argument(expression, 0, {"base"})
                )
                mark_address_expression(
                    named_argument(expression, 2, {"strides", "stride"})
                )
                return
            if called in {"advance", "multiple_of", "max_contiguous"}:
                for argument in expression.args:
                    mark_address_expression(argument)
                for keyword in expression.keywords:
                    mark_address_expression(keyword.value)
                return
            if called == "cast":
                mark_address_expression(
                    named_argument(expression, 0, {"input", "x", "value"})
                )
                return
            if called == "to" and isinstance(expression.func, ast.Attribute):
                mark_address_expression(expression.func.value)
                return
            # A predicate-producing or value-producing call is not an address
            # wrapper.  Do not recursively bless carrier uses hidden inside it.
            return
        if isinstance(expression, ast.IfExp):
            # The condition is a predicate/control input, never an address
            # carrier.  Only the selected pointer values belong to this slice.
            mark_address_expression(expression.body)
            mark_address_expression(expression.orelse)
            return
        if isinstance(expression, (ast.Compare, ast.BoolOp)):
            return
        if isinstance(expression, ast.Attribute):
            mark_address_expression(expression.value)
            return
        for child in ast.iter_child_nodes(expression):
            mark_address_expression(child)

    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        called = call_name(node)
        if called in transfer_names or called.startswith("atomic_"):
            mark_address_expression(transfer_pointer(node))

    dependency_cache: dict[int, frozenset[str]] = {}
    dependency_stack: set[tuple[str, int]] = set()

    def carrier_dependencies(expression: ast.AST | None) -> frozenset[str]:
        if expression is None:
            return frozenset()
        expression_id = id(expression)
        cached = dependency_cache.get(expression_id)
        if cached is not None:
            return cached
        if isinstance(expression, ast.Name):
            if not isinstance(expression.ctx, ast.Load):
                result = frozenset()
            elif expression.id in all_carrier_names:
                result = frozenset({expression.id})
            elif expression.id in definitions:
                key = (expression.id, expression_id)
                if key in dependency_stack:
                    result = frozenset()
                else:
                    dependency_stack.add(key)
                    result = frozenset(
                        set().union(
                            *(
                                carrier_dependencies(definition)
                                for definition in definitions[expression.id]
                            )
                        )
                    )
                    dependency_stack.remove(key)
            else:
                result = frozenset()
            dependency_cache[expression_id] = result
            return result
        if isinstance(expression, ast.Constant):
            result = frozenset()
        elif isinstance(expression, ast.Call):
            called = call_name(expression)
            if called in transfer_names or called.startswith("atomic_"):
                # The transfer result is tensor data, not an address carrier.
                result = frozenset()
            elif called == "make_block_ptr":
                relevant = (
                    named_argument(expression, 0, {"base"}),
                    named_argument(expression, 2, {"strides", "stride"}),
                )
                result = frozenset(
                    set().union(*(carrier_dependencies(item) for item in relevant))
                )
            elif called in {"advance", "multiple_of", "max_contiguous", "cast"}:
                result = frozenset(
                    set().union(
                        *(
                            carrier_dependencies(item)
                            for item in expression.args
                        ),
                        *(
                            carrier_dependencies(item.value)
                            for item in expression.keywords
                        ),
                    )
                )
            elif called == "to" and isinstance(expression.func, ast.Attribute):
                result = carrier_dependencies(expression.func.value)
            else:
                result = frozenset(
                    set().union(
                        *(carrier_dependencies(child) for child in ast.iter_child_nodes(expression))
                    )
                )
        else:
            result = frozenset(
                set().union(
                    *(carrier_dependencies(child) for child in ast.iter_child_nodes(expression))
                )
            )
        dependency_cache[expression_id] = result
        return result

    first_bad: dict[str, ast.AST] = {}
    used_address_carriers: set[str] = set()
    for name, node in reassigned_carriers.items():
        first_bad.setdefault(name, node)
    for node in ast.walk(function):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if node.id in forbidden_carrier_names:
            first_bad.setdefault(node.id, node)
        dependencies = carrier_dependencies(node)
        if not dependencies:
            continue
        if id(node) in address_nodes:
            used_address_carriers.update(
                name for name in dependencies if name in carrier_names
            )
            if not dependencies & set(forbidden_carrier_names):
                continue
        for name in dependencies:
            if name in forbidden_carrier_names or id(node) not in address_nodes:
                first_bad.setdefault(name, node)

    for name in sorted(set(carrier_names) - used_address_carriers):
        first_bad.setdefault(name, function)

    violations: list[_ReplayABIViolation] = []
    for name, node in sorted(
        first_bad.items(),
        key=lambda item: (getattr(item[1], "lineno", 0), getattr(item[1], "col_offset", 0), item[0]),
    ):
        violations.append(
            _ReplayABIViolation(
                "unowned_value_carrier_use_forbidden",
                path=f"<parameter:{name}>",
                handles=(name,),
                line=getattr(node, "lineno", None),
                expected={
                    "allowed_use": (
                        "pointer/address expression only for a carrier listed "
                        "by representation_contract.address_bindings"
                    ),
                    "forbidden_use": (
                        "any use for an unlisted carrier; shape, extent, grid, "
                        "coordinate, predicate, fill, value, reducer, loop bound, "
                        "or schedule"
                    ),
                },
                observed={
                    "use": (
                        "unowned carrier read"
                        if name in forbidden_carrier_names
                        else "outside pointer/address expression"
                    )
                },
            )
        )
    return tuple(violations[:20])


def _check_triton_exact_abi(
    code: str,
    packet: Mapping[str, Any],
    function_name: str,
    artifact: Gate1Artifact,
) -> tuple[_ReplayABIViolation, ...]:
    """Check the exact packet-derived ABI where Triton syntax is unambiguous."""

    tree = ast.parse(code, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    j_parameters = packet["j"]["signature"]["parameters"]
    c_slots = [
        item
        for item in _packet_c_slots(packet)
        if item.get("kind") == "constexpr"
    ]
    u_f = packet.get("u_f")
    callable_abi_order = (
        u_f.get("callable_abi_order")
        if isinstance(u_f, Mapping)
        else None
    )
    expected_names = (
        list(callable_abi_order)
        if isinstance(callable_abi_order, (list, tuple))
        else [
            *(str(item["name"]) for item in j_parameters),
            *(str(item["name"]) for item in c_slots),
        ]
    )
    actual_names = [item.arg for item in function.args.args]
    if len(actual_names) != len(expected_names):
        raise _ReplayABIViolation(
            "abi_parameter_count",
            path="<entry-signature>",
            handles=tuple(expected_names[:32]),
            line=function.lineno,
        )
    if actual_names != expected_names:
        visible = tuple(
            dict.fromkeys(
                item
                for item in [*expected_names, *actual_names]
                if _ANONYMOUS_HANDLE.fullmatch(item)
            )
        )[:32]
        raise _ReplayABIViolation(
            "abi_parameter_order",
            path="<entry-signature>",
            handles=visible,
            line=function.lineno,
        )
    j_constexpr = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") == "constexpr"
    }
    specialization_carriers = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") == "constexpr"
        and item.get("logical_ref") is None
    }
    unowned_value_carriers = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") == "scalar"
        and item.get("logical_ref") is None
    }
    u_f = packet.get("u_f")
    representation_contract = (
        u_f.get("representation_contract")
        if isinstance(u_f, Mapping)
        else None
    )
    address_bindings = (
        representation_contract.get("address_bindings", [])
        if isinstance(representation_contract, Mapping)
        else []
    )
    owned_address_carriers = {
        str(item.get("carrier"))
        for item in address_bindings
        if isinstance(item, Mapping) and isinstance(item.get("carrier"), str)
    }
    owned_address_carriers &= unowned_value_carriers
    unowned_address_carriers = unowned_value_carriers - owned_address_carriers
    absent_carriers = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") == "opaque"
        and str(item.get("dtype", "")).casefold() == "none"
    }
    forbidden_carriers = {
        **{
            name: "specialization_carrier_use_forbidden"
            for name in specialization_carriers
        },
        **{name: "absent_carrier_use_forbidden" for name in absent_carriers},
    }
    expected_constexpr = j_constexpr | {str(item["name"]) for item in c_slots}
    violations: list[_ReplayABIViolation] = []
    for argument in function.args.args:
        is_constexpr = (
            isinstance(argument.annotation, ast.Attribute)
            and isinstance(argument.annotation.value, ast.Name)
            and argument.annotation.value.id == "tl"
            and argument.annotation.attr == "constexpr"
        )
        if argument.arg in expected_constexpr and not is_constexpr:
            violations.append(
                _ReplayABIViolation(
                    "constexpr_annotation_required",
                    path=f"<parameter:{argument.arg}>",
                    handles=(argument.arg,),
                    line=argument.lineno,
                )
            )
        if argument.arg not in expected_constexpr and is_constexpr:
            violations.append(
                _ReplayABIViolation(
                    "constexpr_annotation_forbidden",
                    path=f"<parameter:{argument.arg}>",
                    handles=(argument.arg,),
                    line=argument.lineno,
                )
            )
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        receiver = node.func.value
        if node.func.attr != "to" or not isinstance(receiver, ast.Name):
            continue
        if receiver.id not in expected_constexpr:
            continue
        violations.append(
            _ReplayABIViolation(
                "constexpr_scalar_method",
                path=f"<parameter:{receiver.id}>.to",
                handles=(receiver.id,),
                line=getattr(node, "lineno", None),
                expected={"allowed": "typed arithmetic on the callable scalar"},
                observed={"method": "to"},
            )
        )
    first_carrier_load: dict[str, ast.Name] = {}
    for statement in function.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            if node.id in forbidden_carriers:
                first_carrier_load.setdefault(node.id, node)
    for name, node in sorted(
        first_carrier_load.items(),
        key=lambda item: (item[1].lineno, item[1].col_offset, item[0]),
    ):
        violations.append(
            _ReplayABIViolation(
                forbidden_carriers[name],
                path=f"<parameter:{name}>",
                handles=(name,),
                line=node.lineno,
            )
        )
    violations.extend(
        _check_triton_address_carrier_uses(
            function,
            owned_address_carriers,
            forbidden_carrier_names=unowned_address_carriers,
        )
    )
    violations.extend(
        _check_triton_rank0_scalar_materialization(
            function,
            packet,
            artifact,
        )
    )
    return tuple(violations[:20])


def _check_triton_rank0_scalar_materialization(
    function: ast.FunctionDef,
    packet: Mapping[str, Any],
    artifact: Gate1Artifact,
) -> tuple[_ReplayABIViolation, ...]:
    """Reject the known-invalid scalar-to-vector materialization pattern.

    A J rank-0 value is a scalar until a real vector consumer requires a
    broadcast.  In Triton, materializing ``tl.broadcast_to(s, (1, n))`` and
    then indexing it with ``[0, :]`` is not equivalent to a proved J
    broadcast: the first index is a constexpr tensor index and the frontend
    rejects it (and it also hides the actual consumer rank).  Replay must
    construct the consumer-shaped broadcast directly.  This check is
    source-isolated and uses no Source exception; it is a deterministic ingress
    contract derived from the canonical J packet.
    """

    def call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
        return ""

    def assigned_names(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, (ast.Tuple, ast.List)):
            names: list[str] = []
            for item in node.elts:
                names.extend(assigned_names(item))
            return tuple(names)
        return ()

    def fixed_integer_index(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, int) and not isinstance(node.value, bool)
        if isinstance(node, ast.Tuple):
            return any(fixed_integer_index(item) for item in node.elts)
        # Python 3.8's ast.Index is absent on current interpreters but keeping
        # this branch makes the guard stable for the supported source range.
        index_type = getattr(ast, "Index", ())
        if index_type and isinstance(node, index_type):
            return fixed_integer_index(node.value)
        return False

    def is_rank0_tensor_type(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and value.get("kind", "tensor") == "tensor"
            and value.get("shape") == []
        )

    # The guard is source-blind, but it is not J-blind.  Only handles whose
    # canonical packet type proves rank zero may seed scalar propagation.  In
    # particular, an arbitrary p* spelling is not evidence that the provider
    # received a scalar; p* remains an ABI carrier until J binds its rank.
    rank0_handles: set[str] = set()
    u_f = packet.get("u_f")
    masked_j = (
        isinstance(u_f, Mapping)
        and u_f.get("semantic_view") == "masked_j"
    )
    raw_j = packet.get("j")
    if masked_j:
        # A masked-J view intentionally replaces operation/type semantics with
        # sentinels and therefore cannot pass normal J elaboration. Its tensor
        # declarations still preserve rank exactly. Consume only that narrow
        # host-authored structure here; all normal packets continue through the
        # canonical elaborator below.
        if not isinstance(raw_j, Mapping):
            raise _ReplayABIViolation(
                "masked_j_rank_projection_unavailable",
                path="<j>",
            )
        raw_intent = raw_j.get("intent")
        raw_function = (
            raw_intent.get("intent_function")
            if isinstance(raw_intent, Mapping)
            else None
        )
        if not isinstance(raw_function, Mapping):
            raise _ReplayABIViolation(
                "masked_j_rank_projection_unavailable",
                path="<j.intent.intent_function>",
            )
        for collection in ("inputs", "outputs"):
            values = raw_function.get(collection)
            if not isinstance(values, list):
                raise _ReplayABIViolation(
                    "masked_j_rank_projection_unavailable",
                    path=f"<j.intent.intent_function.{collection}>",
                )
            for index, value in enumerate(values):
                if (
                    not isinstance(value, Mapping)
                    or not isinstance(value.get("name"), str)
                    or value.get("dtype") != "<masked>"
                    or not isinstance(value.get("shape"), list)
                    or any(extent != "<masked>" for extent in value["shape"])
                ):
                    raise _ReplayABIViolation(
                        "masked_j_rank_projection_unavailable",
                        path=(
                            f"<j.intent.intent_function.{collection}[{index}]>"
                        ),
                    )
                if not value["shape"]:
                    rank0_handles.add(str(value["name"]))
        j = raw_j
        _elaboration = None
        aliases: Mapping[str, str] = {}
    else:
        # The normal packet is an intentionally source-blind projection: it
        # omits host-only authorization receipts and therefore is not a second
        # complete SemanticArchive wire.  Re-parsing it as one caused every
        # accepted current candidate to fail here with SchemaError before ABI
        # inspection.  Packet/artifact equality is checked at ingress; consume
        # the artifact's already-accepted elaboration and translate only its
        # canonical references into this exact opaque packet namespace.
        canonical_artifact = canonicalize_gate1_artifact(artifact)
        aliases, _bindings, _order = _opaque_reference_data(canonical_artifact)
        j = raw_j
        _elaboration = j_elaboration(canonical_artifact.j)
    if isinstance(j, Mapping):
        intent = j.get("intent")
        intent_function = (
            intent.get("intent_function")
            if isinstance(intent, Mapping)
            else None
        )
        if isinstance(intent_function, Mapping):
            for collection in ("inputs", "outputs"):
                values = intent_function.get(collection)
                if not isinstance(values, list):
                    continue
                for value in values:
                    if (
                        isinstance(value, Mapping)
                        and isinstance(value.get("name"), str)
                        and is_rank0_tensor_type(value)
                    ):
                        rank0_handles.add(value["name"])

            # The elaborator's operation records are the only accepted path
            # and region interpretation.  Do not recursively walk the raw
            # body here: Replay must not create a second operation tree whose
            # rank facts can diverge from Gate1's canonical J construction.
            for record in (
                _elaboration.operation_records
                if _elaboration is not None
                else ()
            ):
                operation = record.get("wire")
                if not isinstance(operation, Mapping):
                    continue
                result = operation.get("result")
                result_names = (
                    [result]
                    if isinstance(result, str)
                    else result
                    if isinstance(result, list)
                    else []
                )
                for name in result_names:
                    value_type = _elaboration.value_type_by_name.get(name)
                    if value_type is not None and value_type.is_tensor and not value_type.shape:
                        rank0_handles.add(aliases.get(name, name))
                for state in operation.get("carry", []):
                    if not isinstance(state, Mapping):
                        continue
                    state_names = (
                        state.get("body_argument"),
                        state.get("next_value"),
                        state.get("result_value"),
                    )
                    if any(
                        isinstance(name, str)
                        and (
                            value_type := _elaboration.value_type_by_name.get(name)
                        ) is not None
                        and value_type.is_tensor
                        and not value_type.shape
                        for name in state_names
                    ):
                        rank0_handles.update(
                            aliases.get(name, name)
                            for name in state_names
                            if isinstance(name, str)
                            and (
                                value_type := _elaboration.value_type_by_name.get(name)
                            ) is not None
                            and value_type.is_tensor
                            and not value_type.shape
                        )
                for scan in operation.get("scan", []):
                    if (
                        isinstance(scan, Mapping)
                        and isinstance(scan.get("from"), str)
                        and aliases.get(scan["from"], scan["from"])
                        in rank0_handles
                        and isinstance(scan.get("name"), str)
                    ):
                        rank0_handles.add(
                            aliases.get(scan["name"], scan["name"])
                        )

        signature = j.get("signature")
        parameters = (
            signature.get("parameters")
            if isinstance(signature, Mapping)
            else None
        )
        for parameter in parameters or []:
            if not isinstance(parameter, Mapping):
                continue
            name = parameter.get("name")
            kind = parameter.get("kind")
            if (
                isinstance(name, str)
                and parameter.get("rank") == 0
                and kind in {"scalar", "constexpr"}
            ):
                rank0_handles.add(name)
            logical_ref = parameter.get("logical_ref")
            if (
                isinstance(name, str)
                and isinstance(logical_ref, str)
                and logical_ref in rank0_handles
            ):
                rank0_handles.add(name)

    c = packet.get("c")
    if isinstance(c, Mapping):
        for slot in c.get("slots", []):
            if isinstance(slot, Mapping) and slot.get("kind") == "constexpr":
                if isinstance(slot.get("name"), str):
                    rank0_handles.add(slot["name"])

    broadcast_values: set[str] = set()
    scalar_aliases: set[str] = set(rank0_handles)
    violations: list[_ReplayABIViolation] = []
    for statement in ast.walk(function):
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else (statement.target,)
            )
            names = tuple(
                name
                for target in targets
                for name in assigned_names(target)
            )
            if (
                isinstance(value, ast.Call)
                and call_name(value) == "broadcast_to"
                and value.args
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id in scalar_aliases
            ):
                broadcast_values.update(names)
            if isinstance(value, ast.Name) and value.id in scalar_aliases:
                scalar_aliases.update(names)
            scalar_aliases.update(name for name in names if name in rank0_handles)

    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        is_broadcast_value = (
            isinstance(value, ast.Name) and value.id in broadcast_values
        ) or (
            isinstance(value, ast.Call)
            and call_name(value) == "broadcast_to"
            and value.args
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in scalar_aliases
        )
        if not is_broadcast_value or not fixed_integer_index(node.slice):
            continue
        violations.append(
            _ReplayABIViolation(
                "rank0_scalar_broadcast_index_forbidden",
                path="<replay-rank0-materialization>",
                line=getattr(node, "lineno", None),
                expected={
                    "form": "direct broadcast to the real J consumer shape",
                    "forbidden": "broadcast_to(...)[constant_index, ...]",
                },
                observed={"form": "indexed broadcast tensor"},
            )
        )

    # The same scalar issue appears when the provider first reshapes a scalar
    # to a singleton vector.  We reject only the unambiguous rank-0-to-[1]
    # spelling; larger target shapes remain governed by J broadcast contracts.
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or call_name(node) != "reshape":
            continue
        if len(node.args) < 2:
            continue
        target = node.args[1]
        singleton = (
            isinstance(target, (ast.Tuple, ast.List))
            and len(target.elts) == 1
            and isinstance(target.elts[0], ast.Constant)
            and target.elts[0].value == 1
        )
        source = node.args[0]
        if not singleton or not isinstance(source, ast.Name):
            continue
        if source.id not in scalar_aliases:
            continue
        violations.append(
            _ReplayABIViolation(
                "rank0_scalar_singleton_reshape_forbidden",
                path="<replay-rank0-materialization>",
                line=getattr(node, "lineno", None),
                expected={"form": "retain scalar until a real vector consumer"},
                observed={"form": "reshape(scalar, (1,))"},
            )
        )
    return tuple(violations[:20])


def _check_triton_program_id_axes(code: str, function_name: str) -> None:
    """Reject Replay code that cannot name a real Triton launch-grid axis."""

    tree = ast.parse(code, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        called = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else ""
        )
        if called != "program_id":
            continue
        keyword_axes = [item.value for item in node.keywords if item.arg == "axis"]
        axis = (
            node.args[0]
            if len(node.args) == 1 and not keyword_axes
            else keyword_axes[0]
            if not node.args and len(keyword_axes) == 1
            else None
        )
        value = axis.value if isinstance(axis, ast.Constant) else None
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value not in {0, 1, 2}
        ):
            raise _ReplayABIViolation(
                "program_id_axis_range",
                path="<program_id-axis>",
                line=node.lineno,
            )



def _closed_ingress_diagnostic(exc: Exception) -> dict[str, Any]:
    """Project an exception to bounded source-free rule/path metadata."""

    if isinstance(exc, _ReplayABIViolation):
        return {
            "rule_code": exc.rule_code,
            "path": exc.path,
            "anonymous_handles": list(exc.handles),
            "line": exc.line,
            "exception_type": type(exc).__name__,
            **({"expected": exc.expected} if exc.expected is not None else {}),
            **({"observed": exc.observed} if exc.observed is not None else {}),
        }
    if isinstance(exc, (CodeExtractionError, UnsafeReplayCodeError)):
        return {
            "rule_code": exc.rule_code,
            "path": exc.path,
            "anonymous_handles": [],
            "line": exc.line,
            "exception_type": exc.exception_type,
        }
    return {
        "rule_code": "replay_static_policy_rejected",
        "path": "<replay>",
        "anonymous_handles": [],
        "line": None,
        "exception_type": type(exc).__name__,
    }


def _extract_replay_code(
    response: str,
    *,
    frontend: Frontend,
    function_name: str,
) -> tuple[str, str | None]:
    if frontend is Frontend.TRITON:
        return (
            extract_python_module(
                response,
                required_function=function_name,
            ),
            None,
        )
    if frontend is Frontend.CUTE:
        entry = "scrc_entry"
        return (
            extract_cute_module(
                response,
                required_kernel=function_name,
                required_entry=entry,
            ),
            entry,
        )
    if frontend is Frontend.TILELANG:
        return (
            extract_tilelang_module(
                response,
                required_function=function_name,
            ),
            None,
        )
    if frontend is Frontend.CUDA:
        return (
            extract_cuda_module(
                response,
                required_kernel=function_name,
            ),
            None,
        )
    raise ValueError(
        f"source-blind Replay has no audited validator for {frontend.value!r}"
    )


def replay_source_ingress(
    source: str,
    *,
    artifact: Gate1Artifact,
    frontend: Frontend | str,
    function_name: str = "scrc_k0",
    packet: Mapping[str, Any] | None = None,
) -> ReplayIngressResult:
    """Run the one host-owned static ingress used by check and submission.

    This path deliberately does not invoke a frontend compiler, Gate1 observer,
    execution harness, or any source-side view.
    """

    try:
        selected_frontend = Frontend(frontend)
    except (TypeError, ValueError):
        return ReplayIngressResult(
            "fail",
            diagnostics=(
                {
                    "rule_code": "frontend_not_supported",
                    "path": "<frontend>",
                    "anonymous_handles": [],
                    "line": None,
                },
            ),
        )
    if packet is None:
        packet = build_rewriter_payload(
            artifact,
            frontend=selected_frontend,
            masked=False,
        )
    try:
        assert_source_isolated_payload(packet)
        packet_frontend = packet.get("u_f", {}).get("frontend")
        if packet_frontend != selected_frontend.value:
            return ReplayIngressResult(
                "fail",
                diagnostics=(
                    {
                        "rule_code": "frontend_packet_mismatch",
                        "path": "<u_f.frontend>",
                        "anonymous_handles": [],
                        "line": None,
                        "expected": {"frontend": selected_frontend.value},
                        "observed": {"frontend": packet_frontend},
                    },
                ),
            )
        _assert_packet_c_matches_artifact(packet, artifact)
        _assert_packet_representation_matches_artifact(packet, artifact)
        _assert_packet_j_matches_artifact(packet, artifact)
        code, entry = _extract_replay_code(
            source,
            frontend=selected_frontend,
            function_name=function_name,
        )
        if selected_frontend is Frontend.TRITON:
            abi_violations = _check_triton_exact_abi(
                code,
                packet,
                function_name,
                artifact,
            )
            if abi_violations:
                return ReplayIngressResult(
                    "fail",
                    diagnostics=tuple(
                        _closed_ingress_diagnostic(item)
                        for item in abi_violations
                    ),
                )
            _check_triton_program_id_axes(code, function_name)
        elif selected_frontend in {
            Frontend.CUTE,
            Frontend.TILELANG,
            Frontend.CUDA,
        }:
            # The frontend-specific extractor owns the source language's
            # canonical entry and static safety policy.  ABI binding and native
            # IR/PTX observation remain in the matching Gate1 compiler; this
            # ingress must not invent a Triton TTIR check for native sources.
            if not code.strip():
                raise ValueError("frontend Replay source is empty after extraction")
        else:  # pragma: no cover - Frontend is a closed enum
            raise ValueError(
                f"source-blind Replay has no audited validator for {selected_frontend.value!r}"
            )
        # Replay construction preflight owns syntax and the exact J/C ABI only.
        # Pi is deliberately absent from the Replay packet; realization
        # differences are handled later by Gate1's fixed normalizer.
    except Exception as exc:
        return ReplayIngressResult(
            "fail", diagnostics=(_closed_ingress_diagnostic(exc),)
        )
    return ReplayIngressResult(
        "pass",
        code=code,
        entry_function_name=entry,
    )


_SYNTHETIC_PROBE_DTYPES = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "ui8": "uint8",
    "ui16": "uint16",
    "ui32": "uint32",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
}
_COMPILE_LINE = re.compile(r"(?:CompilationError:\s*)?at\s+(\d+):\d+")


def _synthetic_probe_dtype(dtype: str) -> str:
    normalized = dtype.removeprefix("torch.").casefold()
    return _SYNTHETIC_PROBE_DTYPES.get(normalized, normalized)


def _synthetic_probe_scalar(dtype: str) -> bool | int | float | str:
    normalized = dtype.removeprefix("torch.").casefold()
    if normalized in {"bool", "i1"}:
        return True
    if normalized.startswith("f") or normalized.startswith("float") or normalized in {
        "bf16",
        "bfloat16",
        "half",
        "double",
    }:
        return 1.0
    if normalized in {"str", "string"}:
        return "synthetic"
    return 16


def build_source_blind_replay_compile_case(
    artifact: Gate1Artifact,
) -> CompileCase:
    """Build a source-isolated profile from J and the accepted Source-bound C.

    J ABI values remain public deterministic placeholders. When C exists, its
    exact Source/Harness-bound implementation values are used as callable
    constexpr or external compiler bindings. Replay never synthesizes a
    separate target C. Tensor storage size is irrelevant to Triton's pointer
    type during warmup, so every pointer uses one element.
    """

    if artifact.c.slots:
        artifact.c.validate_complete_values(path="Replay compile C")

    arguments: dict[str, RuntimeArgument] = {}
    schedule: dict[str, bool | int | float | str] = {}
    for parameter in artifact.j.signature.parameters:
        kind = parameter.kind.value
        dtype = parameter.dtype or "i64"
        if kind in {"pointer", "tensor"}:
            arguments[parameter.name] = RuntimeArgument.tensor(
                (1,), _synthetic_probe_dtype(dtype)
            )
        elif kind == "scalar":
            arguments[parameter.name] = RuntimeArgument.scalar(
                _synthetic_probe_scalar(dtype),
                _synthetic_probe_dtype(dtype),
            )
        elif kind in {"none", "opaque"}:
            arguments[parameter.name] = RuntimeArgument.none()
        elif kind == "constexpr":
            schedule[parameter.name] = _synthetic_probe_scalar(dtype)
        else:  # pragma: no cover - closed enum defensive boundary
            raise ValueError(f"unsupported Replay probe parameter kind {kind!r}")
    for slot in artifact.c.slots:
        value = artifact.c.values[slot.name]
        if slot.kind.value == "compiler":
            if slot.role in arguments or slot.role in schedule:
                raise ValueError(
                    f"compiler C role {slot.role!r} collides with a compile binding"
                )
            schedule[slot.role] = value
            continue
        if slot.name in arguments or slot.name in schedule:
            raise ValueError(
                f"C constexpr handle {slot.name!r} collides with a compile binding"
            )
        schedule[slot.name] = value
    return CompileCase(arguments, (1, 1, 1), schedule)


def _source_blind_compile_failure(result: CompileResult) -> ReplayIngressResult:
    diagnostic = result.diagnostic or "Triton frontend rejected Replay"
    line_matches = _COMPILE_LINE.findall(diagnostic)
    return ReplayIngressResult(
        "fail",
        diagnostics=(
            {
                "rule_code": "replay_frontend_preflight_compile_failed",
                "path": "<replay-compile-preflight>",
                "anonymous_handles": [],
                "line": int(line_matches[-1]) if line_matches else None,
                "failure_class": (
                    result.failure_class
                    or "triton_frontend_compile_failed"
                ),
                **(
                    {"exception_type": result.exception_type}
                    if result.exception_type is not None
                    else {}
                ),
            },
        ),
    )














def _packet_required_compiled_program_ids(
    packet: Mapping[str, Any],
) -> tuple[int | None, tuple[str, ...], tuple[int, ...] | None]:
    """Project canonical J domain-to-launch-axis obligations."""

    raw_j = packet.get("j")
    pi = packet.get("pi")
    u_f = packet.get("u_f")
    if not isinstance(raw_j, Mapping) or not isinstance(pi, Mapping):
        return None, (), None
    _j, elaboration = _canonical_packet_j(packet)

    representation_contract = (
        u_f.get("representation_contract")
        if isinstance(u_f, Mapping)
        else None
    )
    if isinstance(representation_contract, Mapping):
        rows = representation_contract.get("program_bindings", [])
        if isinstance(rows, list) and rows:
            axes_by_domain = {
                str(item.get("domain_ref")): int(item["axis"])
                for item in rows
                if isinstance(item, Mapping)
                and isinstance(item.get("domain_ref"), str)
                and isinstance(item.get("axis"), int)
                and not isinstance(item.get("axis"), bool)
            }
            if len(axes_by_domain) == len(rows):
                axes = tuple(sorted(set(axes_by_domain.values())))
                return len(axes), tuple(sorted(axes_by_domain)), axes

    if pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        return None, (), None
    try:
        projection = canonical_program_coordinate_projection(pi)
    except ProgramCoordinateProjectionError:
        return None, (), None
    declared_domains = {str(name) for name in elaboration.boundary_domains}
    if any(
        domain_ref not in declared_domains
        for domain_ref, _axis in projection.domain_to_axis
    ):
        return None, (), None
    domains = tuple(sorted(domain_ref for domain_ref, _axis in projection.domain_to_axis))
    return len(projection.physical_axes), domains, projection.physical_axes


def compiled_packet_obligation_diagnostics(
    packet: Mapping[str, Any],
    ttir: str,
    *,
    fingerprint: TTIRFingerprint | None = None,
    path_prefix: str = "<compiled-obligation>",
) -> tuple[Mapping[str, Any], ...]:
    """Check only canonical J/pi compiler obligations.

    Source transfer occurrence, effect order, fills, access predicates, loop
    structure, and address ownership are certified by the shared construction
    ledger and canonical J elaboration.  The retired materialized pi graph is
    not an alternate source of compiled obligations.
    """

    pi = packet.get("pi")
    if not isinstance(pi, Mapping) or pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        return (
            {
                "rule_code": "canonical_pi_schema_missing",
                "path": "pi.schema",
                "anonymous_handles": [],
                "line": None,
            },
        )

    observed = fingerprint if fingerprint is not None else parse_ttir(ttir)
    actual_program_ids = dict(observed.op_multiset).get("tt.get_program_id", 0)
    expected_program_ids, domain_refs, expected_axes = (
        _packet_required_compiled_program_ids(packet)
    )
    if expected_program_ids is None:
        return ()

    diagnostics: list[Mapping[str, Any]] = []
    if expected_program_ids != actual_program_ids:
        diagnostics.append(
            {
                "rule_code": "compiled_program_id_count",
                "path": (
                    f"{path_prefix} tt.get_program_id: expected={expected_program_ids}, "
                    f"actual={actual_program_ids}"
                ),
                "anonymous_handles": list(domain_refs[:32]),
                "line": None,
            }
        )
    if expected_axes is not None:
        actual_axes = compiled_program_id_axes(ttir)
        if actual_axes is None or actual_axes != expected_axes:
            diagnostics.append(
                {
                    "rule_code": "compiled_program_id_axes",
                    "path": (
                        f"{path_prefix} tt.get_program_id axes: "
                        f"expected={list(expected_axes)}, "
                        f"actual={list(actual_axes) if actual_axes is not None else 'unrecognized'}"
                    ),
                    "anonymous_handles": list(domain_refs[:32]),
                    "line": None,
                }
            )
    return tuple(diagnostics)


def source_blind_replay_compile_check(
    code: str,
    artifact: Gate1Artifact,
    function_name: str,
    *,
    compiler: TritonCompiler | None = None,
    packet: Mapping[str, Any] | None = None,
) -> ReplayIngressResult:
    """Compile Replay against J and the accepted Source-bound C profile.

    This check has no acceptance authority.  Formal Gate1 recompiles the exact
    submitted bytes against the accepted artifact after the agent episode ends.
    """

    selected_packet = (
        packet
        if packet is not None
        else build_rewriter_payload(
            artifact,
            frontend=Frontend.TRITON,
            masked=False,
        )
    )
    assert_source_isolated_payload(selected_packet)
    _assert_packet_c_matches_artifact(selected_packet, artifact)
    _assert_packet_representation_matches_artifact(selected_packet, artifact)
    selected_compiler = compiler or TritonCompiler(timeout=120.0)
    try:
        source = KernelSource.from_replay(code, function_name)
        result = selected_compiler.compile(
            source,
            build_source_blind_replay_compile_case(artifact),
        )
    except Exception as exc:
        result = CompileResult(
            False,
            diagnostic=f"{type(exc).__name__}: {exc}",
            failure_class="triton_frontend_preflight_failed",
            exception_type=type(exc).__name__,
        )
    if not result.ok:
        return _source_blind_compile_failure(result)
    if result.ttir is None:
        return ReplayIngressResult(
            "fail",
            diagnostics=(
                {
                    "rule_code": "synthetic_replay_observer_failed",
                    "path": "<synthetic-frontend-compile>",
                    "anonymous_handles": [],
                    "line": None,
                },
            ),
        )
    try:
        # This is a construction compile probe, not a semantic-membership oracle.
        # Parsing proves that the submitted source reaches the supported raw
        # observer.  The closed J/C obligation projection adds only reject-only
        # cardinality/order checks already defined for this packet; formal
        # Source-vs-Replay closure remains exclusively outside the episode.
        fingerprint = parse_ttir(result.ttir, external_math=result.external_math)
    except TTIRParseError as exc:
        return ReplayIngressResult(
            "fail",
            diagnostics=(
                {
                    "rule_code": str(exc) if str(exc) in EXTERNAL_MATH_FAILURE_RULES else "synthetic_replay_observer_failed",
                    "path": "<synthetic-frontend-compile>",
                    "anonymous_handles": [],
                    "line": None,
                },
            ),
        )
    obligation_diagnostics = compiled_packet_obligation_diagnostics(
        selected_packet,
        result.ttir,
        fingerprint=fingerprint,
        path_prefix="<synthetic-frontend-compile>",
    )
    if obligation_diagnostics:
        return ReplayIngressResult(
            "fail",
            diagnostics=obligation_diagnostics,
        )
    return ReplayIngressResult(
        "pass",
        code=code,
        entry_function_name=function_name,
    )


ReplayCompileChecker = Callable[
    [str, Gate1Artifact, str], ReplayIngressResult
]


def _usage(raw: Mapping[str, Any]) -> dict[str, int]:
    value = raw.get("usage")
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    return result


def _context_receipt(raw: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    trace: list[dict[str, str]] = []
    output = raw.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if isinstance(item_type, str) and "search" in item_type:
                trace.append(
                    {
                        "type": item_type,
                        "id": item.get("id") if isinstance(item.get("id"), str) else "",
                    }
                )
    return {
        "replay_input_protocol": str(context.get("protocol", "")),
        "retrieval_trace": trace,
    }


def _agent_usage_and_identity(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, int], str | None, str | None]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen = False
    response_id: str | None = None
    response_model: str | None = None
    messages = trajectory.get("messages", ())
    if not isinstance(messages, list):
        return {}, None, None
    for message in messages:
        if not isinstance(message, Mapping) or message.get("object") != "response":
            continue
        if isinstance(message.get("id"), str):
            response_id = message["id"]
        if isinstance(message.get("model"), str):
            response_model = message["model"]
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            continue
        seen = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
    return (totals if seen else {}), response_id, response_model


def _sanitize_agent_trajectory(value: Any, secret: str, *, key: str = "") -> Any:
    """Apply the same recursive secret-key and literal redaction as Lift."""

    normalized_key = key.casefold().replace("-", "_")
    if normalized_key in {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "password",
        "secret",
    } or normalized_key.endswith("_api_key"):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_agent_trajectory(
                item, secret, key=str(item_key)
            )
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_agent_trajectory(item, secret) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_agent_trajectory(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "<redacted>")
    return value


def _trajectory_bytes(trajectory: Mapping[str, Any]) -> bytes:
    return json.dumps(
        trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _frontend_replay_guidance(frontend: Frontend) -> str:
    """Append the target-specific ownership boundary after shared guidance."""

    if frontend is Frontend.TRITON:
        return (
            "\nFor this episode the frontend is Triton, so the Triton-specific "
            "contracts above apply.\n"
        )
    if frontend is Frontend.CUDA:
        return """
Frontend-specific override: this episode is CUDA C++, not Triton Python. Return
one standalone CUDA translation unit with exactly one 'extern "C" __global__'
kernel named 'scrc_k0'; use the positional runtime ABI from J and keep C values
as compile/launch choices rather than extra runtime parameters. Do not emit
'tl.*', '@triton.jit', Triton program-id code, Python, host code, inline PTX, or
any Triton-only operation/TTIR assumption. replay_ttir_inspect is unavailable
for this frontend; the frontend-specific compile checker is the construction
checkpoint and formal Gate1 owns the native PTX observation.
"""
    if frontend is Frontend.TILELANG:
        return """
Frontend-specific override: this episode is TileLang, not Triton. Return one
self-contained module using exactly 'import tilelang.language as T' and one
'@T.prim_func' named 'scrc_k0'; express the logical tensor ABI through public
TileLang annotations and use only public TileLang CUDA realization APIs. Do not
emit '@triton.jit', 'tl.*', CUDA host code, or assume a Triton TTIR. C values
are compile-time schedule choices, never extra logical runtime parameters.
replay_ttir_inspect is unavailable for this frontend; the frontend-specific
compile checker is the construction checkpoint and formal Gate1 owns the native
TileLang/PTX observation.
"""
    if frontend is Frontend.CUTE:
        return """
Frontend-specific override: this episode is NVIDIA CuTe, not Triton. Use only
the public CuTe source and entry contract in 'frontend.json', keep the semantic
runtime ABI positional, and construct representation-only objects inside the
entry. Do not emit Triton Python or assume a Triton TTIR. The frontend-specific
compile checker is the construction checkpoint and formal Gate1 owns the native
CuTe MLIR/PTX observation.
"""
    raise ValueError(f"no source-blind frontend guidance for {frontend.value!r}")


def _replay_projection_clause(artifact: Gate1Artifact) -> str:
    """Render the one J/π-certified carrier fact visible to Replay.

    The raw π witness remains outside the source-blind episode.  This derived
    class is intentionally small: it constrains a legal target carrier, not a
    Source recipe or an operation-by-operation implementation plan.
    """

    reductions = per_program_scalar_keepdims_reductions(artifact.j, artifact.pi)
    if not reductions:
        return ""
    return (
        "The J/π-proven rowwise keepdims carrier class applies to "
        + ", ".join(reductions)
        + ": one directly mapped target row holds its logical kept singleton "
        "as a scalar. Do not reshape that scalar to `(1,)`; broadcast only at "
        "an actual vector consumer.\n"
    )


def _replay_input_files(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend,
    masked: bool,
    callable_abi_order: Sequence[str] | None = None,
) -> tuple[dict[str, str], dict[str, Any], str]:
    """Build the complete, closed allowlist mounted for one agent episode."""

    packet = build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=masked,
        callable_abi_order=callable_abi_order,
    )
    assert_source_isolated_payload(packet)
    packet_text = json.dumps(
        packet,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"
    semantic_program = materialize_replay_semantics(packet["j"])
    semantic_program_text = json.dumps(
        semantic_program.to_json(),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"
    compiler_index = materialize_replay_compiler_index(
        semantic_program.to_json(), packet.get("c")
    )
    compiler_index_text = json.dumps(
        compiler_index.to_json(),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"
    c = packet.get("c")
    c_text = (
        json.dumps(
            c,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
        if isinstance(c, Mapping)
        else None
    )
    frontend_text = json.dumps(
        packet["u_f"],
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"
    implementation_contract = (
        "Complete C is present in `c.json` and `packet.json`. Its values are exact "
        "implementation bindings. Keep every kind=constexpr c* as a callable "
        "placeholder and use kind=compiler values only for external invocation.\n"
        if c_text is not None
        else "No C is required for this episode. Compile parameterized J directly "
        "and retain every J.signature ABI placeholder.\n"
    )
    instructions = """# Replay ownership contract

J alone owns operators, operands, constants, predicates, domains, effects,
result relations and boundary indices.
""" + implementation_contract + _replay_projection_clause(artifact) + """
Follow `frontend.json.compile_entry_contract` completely for API, ABI, control
and structural ingress. Compile all of the read-only, parameterized
`semantic-program.intent`/`.json`; these encode J, not target templates.
Source/compiler IR, non-C Harness facts, Gate evidence, previous Replay and
kernel templates are unavailable.
For Triton, `u_f.callable_abi_order` fixes callable order without changing J/C
ownership; compiler-kind C stays outside the callable. Use pointwise-only scalar
parameters directly, never `.to(...)`: specialization may make them Python scalars.
Reserve `do_not_specialize` for access/domain/predicate/loop carriers requiring
runtime identity.

Choose target instructions, addresses, mapping, control, layout, memory,
loops/carriers, synchronization and structure, without altering J semantics.
Physical realization adds no semantic authority. Comparison uses fixed
normalization outside this episode; no Gate observations are available here.
`compiler-index.intent` cross-references J dependencies and optional C bindings;
a syntactic dependency subset does not prove domain containment. Compile the
complete `boundary_ownership`, preserving distinct accesses/effects and keeping
domain guards separate from access-local predicates.

For masked loads, emit `other=` exactly from `fill_value`, or omit it if absent.
Each consumer_extension is a distinct observable consumer edge, not a load fill.
Deliver its typed outside identity to the exact operand along its declared path:
insert predicate/identity immediately after the first fill-changing operation,
before later identity-preserving steps. If the load fill already equals the
identity and the path preserves it, omit redundant selects. Preserve the access
predicate and one independently observable edge per extension row.
Effect/shape rules:
- Fold exact all-zero J add identities, including broadcast-shape identities;
  preserve the other operand, output mapping, access shape, predicate and effect.
- Emit transfers in boundary_ownership.access_effects order. Store at the earliest
  legal authored effect position once its producer exists, before later independent
  consumers; never group stores into an epilogue. One multi-axis access is one
  transfer, optionally split afterward, never one transfer per component.
- Ordinary finite domains must not become serial Python/Triton loops: unrolling
  changes effect multiplicity. A standalone reduce uses one local tile reduction
  unless J authors a loop or a closed case below applies.
- Keep collapsed reduce results collapsed (keep_dims=False); introduce singleton
  axes only at consumers/stores that require them.

Closed additive updates:
- A scatter must have a matching add/combine/unordered update, singleton inout
  target and contribution domain attached to one C reduction_tile_extent.
- Alternatively J must define exactly a rank-0 inout result as
  base + reduce(sum, contributions), with that same contribution attachment.
- For either case use program_id(0)*tile + arange(0,tile), retain J predicates,
  locally sum active contributions and commit via tl.atomic_add. The atomic IS
  the inout read-write effect: no separate base load or ordinary store.
- These cases do not authorize overwrite/one-of, non-singleton scatter,
  output-only scalar, non-additive reducer or unattached-domain atomics.
  Keep program id, lanes, tile arithmetic and runtime bound at the J domain/bound
  carrier width; widen only for an exact J coordinate/type requirement.

Program/domain mapping (J plus optional C only; never Source/grid/pi):
- Factor effect axes into shared, tensor-specific alternatives and C-tiled axes.
  Attached finite domains may use program_id*tile + arange(0,tile), retaining
  every affected access's authored upper-bound predicate; no extra logical axis.
  Untiled shared axes may product-linearize; mutually exclusive tensor-specific
  axes may share a guarded coordinate without changing effects.
- Up to three separately authored parallel/output d* domains each attached to its
  own tile_extent use distinct program_id axes in declaration order, never a
  product coordinate. Within ONE multi-axis domain, a linear program_id(0) may
  decode per-axis tile counts with div/rem.
- reduction_tile_extent or reduction-only/contraction-only axes are not launch
  axes; only the closed singleton additive-scatter case above is excepted.
- Untiled parallel/output coordinates with scalar stores use scalar program_id.
  No arange(0,1) or fabricated singleton expansion. Lane tensors require a real
  anonymous tile attachment or J vector value.

Fully attached block pointers:
For a dense full-rank tensor of rank >=2 with unambiguous block_order attachments
on EVERY axis, use tl.make_block_ptr for every access, never expanded flat
pointers. Use J shapes/dimensions, ABI stride/address carriers and attached order.
Compute num_tiles[d]=tl.cdiv(extent[d],tile[d]); decode program_id(0) trailing-to-
leading with % and //, cast per-axis tile offsets to tl.int32, and use the attached
order tuple for both make_block_ptr(order=...) and boundary_check. No parallel
elementwise mask. This rule requires the complete unambiguous attachment set.

Closed tiled reductions/contractions:
- A J contraction with C reduction_tile_extent uses that tile's lanes and an
  in-kernel loop over the full contraction extent, carrying the J accumulator.
  Coordinates are loop base + lanes; retain predicates, then final reduction/store.
  Keep the loop even if tile equals extent; never make it a launch axis.
  This does not generally authorize loops for standalone reduce.
- Diagonal gather: one scalar reduction depends on an access/gather repeating
  the same logical coordinate across two tensor axes, with one contribution
  reduction_tile_extent. Carry a tile-sized vector accumulator through the full
  diagonal extent, preserve predicate/fill, then locally reduce and store scalar.
  Derive addresses from J's repeated coordinate, never inferred Source strides.
- Rank-2 column reduction: one C-tiled parallel/output axis plus C-tiled axis-0
  standalone reduction uses [parallel_tile,reduction_tile] physical order.
  Coordinates are [program_id(0)*parallel_tile+arange(0,parallel_tile),1] and
  [1,loop_base+arange(0,reduction_tile)]. Carry that matrix through the loop,
  then reduce its last axis. Compute each bound comparison once and broadcast
  those booleans into the joint mask. A later J access after intervening effects
  needs its own physical loop at that effect position, not reused transfers.
- ij,j->i contraction: retain [parallel_tile,1] output coordinates and
  [1,reduction_tile] contraction coordinates throughout addressing/predicates.
  Compute each expanded-rank bound once and share where J shares the predicate.
  Accumulate loaded elementwise products in [parallel_tile,reduction_tile];
  reduce axis 1 only after the loop and keep the singleton through the store.
  Rank-2-by-rank-2 matrix multiplication instead uses per-K-tile tl.dot.

Coordinate/guard rules:
- Decode every authored coordinate when product-linearizing. Lower bounds proved
  by program_id, nonnegative div/rem or arange need no runtime comparison; retain
  all remaining authored upper bounds. A terminal complement of their positive
  conjunction uses the direct disjunction of individual upper-bound violations,
  never not(combined_positive) or a product-extent comparison.
- Coordinate dtype comes from J iota; for boundary-only axes, use the exact
  signature/value-dimension scalar carrier, or i64 only if no carrier exists.
  Widen base and lanes BEFORE summing when required; never narrow from anonymous
  valuations. In guarded fill with complete C (no inputs, one output/runtime
  extent/tile), cast program_id, arange and extent individually only as J requires.
  Do not pre-cast uncast J scalar operands; let public mixed-integer promotion
  handle coordinate*stride.
- For Cartesian multi-domain accesses sharing one conjunction, expand each
  coordinate to transfer rank BEFORE its bound comparison; combine those
  predicates and reuse the coordinates for addresses. Do not expand rank-one
  booleans afterward. True rank-one accesses stay rank-one.
- tile_extent never authorizes guard elimination, a c0==logical_extent branch
  or unmasked access. Only J predicates or separately attached C guard
  specialization can. Unattached specialization_branch slots cannot change masks.
- Invent no early return, structured conditional or extra predicate. Each
  transfer mask contains exactly its J access predicate, sharing computations
  where J does. Domain membership alone adds no mask clause. Keep tensor-specific
  effect guards as sibling masks with unchanged ownership.
- Constexpr-controlled guard elision keeps guarded/unguarded transfers lexically
  inside their respective alternatives. An always-true mask is not unmasked;
  never hoist a shared masked operation out of those alternatives.
"""
    instructions += _frontend_replay_guidance(frontend)
    c_readme = (
        "- `c.json`: exact complete C implementation configuration.\n"
        if c_text is not None
        else "- No C file is present; this is a parameterized J-only episode.\n"
    )
    readme = """# IntentIR source-blind Replay input

This immutable directory is the complete input allowlist for one fresh episode.

- `packet.json`: exact J, the accepted Source/Harness-bound C when present, and public U_f.
- `semantic-program.intent`: compact, typed, parameterized read-only S_J.
- `semantic-program.json`: exact structured form of the same S_J.
- `compiler-index.intent` / `.json`: compact exact S_J plus optional C cross-reference.
""" + c_readme + """
- `frontend.json`: public source-neutral U_f and required callable entry.
- `REPLAY_CONTRACT.md`: ownership and no-leak boundary.
- `TOOL.md`: coding, query, compile, own-IR, checkpoint, and submit commands.

Read files by path and only page sections needed for the current construction. No
Source, CompileCase/grid, Gate result, history, repository, or network is available.
S_J is derived only from this episode's J and contains no pi or C value,
target operator, API, source statement, instruction selection, or repair.
"""
    tool = """# Replay agent tools

The exact target API, ABI, control and structural ingress contract is already
bound in `frontend.json.compile_entry_contract`; follow that single copy completely.
Act like a compiler coding agent. Read `compiler-index.intent`,
`semantic-program.intent`, and `frontend.json` completely; also read `c.json` when
it is present. Query exact J/S_J/C only for a named cross-reference. The provider
tool surface grants two orientation reads, then requires one complete first source
through `submit_replay(source_text)`. After that, mutate the source only through
`apply_replay_patch` or another complete `submit_replay(source_text)`, never with
a shell redirection or helper program. After a submission preflight rejection, the
next source mutation must be an exact `apply_replay_patch` of that source; only the
changed source can reopen a later whole-source submission. For Triton, the submitted module must contain
only the allowed imports and `@triton.jit` function definitions: do not append a host
launch, kernel specialization assignment, or other top-level executable statement
(in particular, do not write `scrc_k0 = scrc_k0[...]`; Gate1 supplies the launch
context).

```text
intentir_semantic_query overview
intentir_semantic_query ownership
intentir_semantic_query node REF
intentir_packet_query abi 0 48
intentir_packet_query graph 0 12
intentir_replay_compile_check /output/replay.source
intentir_replay_inspect /output/replay.source jc
```

The only source-writing tools are `submit_replay(source_text)` and
`apply_replay_patch(path, old_text, new_text, submit?)`; shell commands are
read-only with respect to `/output/replay.source`. You may compile and inspect only
your current Replay IR, then patch and submit repeatedly within this one episode.
When submission rejects current source, use an exact patch before the next complete
submission; do not repeatedly replace rejected source without consuming its latest
preflight diagnostic.
After construction PASS, use the bounded `jc` view; revisit your own IR when a
concrete J obligation remains. Every operand and result dtype printed in S_J is
authoritative: preserve intermediate result dtypes at the corresponding operation
boundary and insert an explicit target cast when implicit promotion would skip it. In
particular, an f16 intermediate later consumed by an f32 operation must remain f16 before
that consumer; matching only final store dtype or operation counts is insufficient.
For Triton/CUDA, preserve the public target realization of authored predicate
and transcendental nodes: lower `map_unary(fn=erf)` with
`tl.math.erf(x.to(tl.float32))`, not `libdevice.erf`,
`map_unary(fn=exp)` with `libdevice.exp`,
`map_unary(fn=rsqrt_ftz)` only for f32 with `libdevice.rsqrt` on
the authored f32 operand; retain that exact extern call and its compiled
FTZ receipt. A plain `tl.rsqrt`, `math.rsqrt`, or reciprocal/sqrt expansion
does not implement the FTZ node,
`map_unary(fn=tanh)` with `libdevice.tanh(x.to(tl.float32))`,
`map_unary(fn=sigmoid)` as `1 / (1 + libdevice.exp2(-x.to(tl.float32) *
1.4426950408889634))`,
importing `libdevice` from `triton.language.extra.cuda`; call
`libdevice.exp2` literally, not `tl.exp2` or `tl.sigmoid`, because those public
target APIs compile to a different compiler-visible operation graph,
`map_unary(fn=isnan)` with `libdevice.isnan(value.to(tl.float32))`, and an f32
`map_unary(fn=isfinite)` with `libdevice.finitef(value.to(tl.float32))` converted
to the authored i1 result. Do not replace an authored `isnan` with `value !=
value`; that is extensionally plausible but has a different compiler-visible
operation graph. Do not replace a J-authored direct reciprocal multiplication
with division followed by `tl.erf`; that changes the target operation graph even
though the scalar formula looks equivalent. If the callable ABI physically
stores a J-authored i1 tensor as bytes, cast each load to `tl.int1` before its
authored logical consumers. Retain an authored comparison with integer one;
neither using the raw byte as a predicate nor comparing it to Python `True`
preserves that typed J boundary.
Lower `map_unary(fn=logical_not)` as boolean conversion followed by boolean
negation (for example `~value.to(tl.int1)`); do not replace that authored node
with the algebraically equivalent direct comparison `value == 0`, which erases
the compiler-visible negation. A predicated J load with no authored
`fill_value` must omit Triton's `other=` argument; never invent zero merely to
silence inactive lanes.
When exact J directly spells `add(primary, neg(value))` or
`add(primary, sub(zero,value))` and that negated producer has no other J consumer,
lower the target expression as the direct subtraction `primary - value`; do not retain
an extra target zero/neg/add chain. This is local instruction selection, not a general
algebraic rewrite. Use targeted raw/find/observer queries rather than
dumping the full fingerprint.
Remove work not authorized by S_J, and keep inactive constexpr alternatives lexically
inside their branch so the specialized compiler never materializes them.
`submit_replay` checks the exact complete
bytes but neither generates nor repairs them. Formal Gate1 runs outside this episode
and is never visible here. Merely writing source that compiles creates at most a
recoverable checkpoint; it does not submit or terminate the episode. Review your own
TTIR and explicitly call `submit_replay`.
"""
    tool += _frontend_replay_guidance(frontend)
    files = {
        "README.md": readme,
        "packet.json": packet_text,
        "semantic-program.json": semantic_program_text,
        "semantic-program.intent": semantic_program.text,
        "compiler-index.json": compiler_index_text,
        "compiler-index.intent": compiler_index.text,
        "frontend.json": frontend_text,
        "REPLAY_CONTRACT.md": instructions + "\n",
        "TOOL.md": tool,
    }
    if c_text is not None:
        files["c.json"] = c_text
    return (
        files,
        packet,
        instructions,
    )


@dataclass
class ResponsesReplayWriter:
    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    compile_checker: ReplayCompileChecker | None = None
    _records: list[ReplayGeneration | ReplayAttempt] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(self._records)

    def generate(
        self,
        artifact: Gate1Artifact,
        *,
        masked: bool = False,
        callable_abi_order: Sequence[str] | None = None,
    ) -> ReplayGeneration:
        if self.frontend is not Frontend.TRITON and self.compile_checker is None:
            raise ValueError(
                "native Responses Replay requires a frontend-specific compile checker"
            )
        instructions, user = build_rewriter_prompt(
            artifact,
            frontend=self.frontend,
            masked=masked,
            callable_abi_order=callable_abi_order,
        )
        packet = json.loads(user)
        u_f = packet["u_f"]
        function_name = u_f["replay_function"]
        frontend = Frontend(u_f["frontend"])
        if frontend not in {
            Frontend.TRITON,
            Frontend.CUTE,
            Frontend.TILELANG,
            Frontend.CUDA,
        }:
            raise ValueError(
                f"source-blind Replay has no audited validator for {frontend.value!r}"
            )
        reply = call_responses(
            instructions,
            user,
            provider=self.provider,
            effort=self.effort,
            # Gate1 owns the only allowed fresh Replay retry and constructs it
            # with a new writer/workspace/owner after a non-provider ingress
            # rejection. A provider stop must surface immediately instead of
            # being hidden by an in-place transport retry.
            max_retries=1,
        )
        ingress_diagnostics: tuple[Mapping[str, Any], ...] = ()
        try:
            code, entry_function_name = _extract_replay_code(
                reply.text,
                frontend=frontend,
                function_name=function_name,
            )
            ingress = replay_source_ingress(
                code,
                artifact=artifact,
                frontend=frontend,
                function_name=function_name,
                packet=packet,
            )
            if ingress.status != "pass" or ingress.code is None:
                ingress_diagnostics = _closed_attempt_diagnostics(
                    ingress.diagnostics
                )
                raise ValueError(
                    "Replay ingress rejected response: "
                    + json.dumps(
                        list(ingress.diagnostics),
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
            code = ingress.code
            entry_function_name = ingress.entry_function_name
            if self.compile_checker is not None:
                compile_result = self.compile_checker(
                    code,
                    artifact,
                    function_name,
                )
                if not isinstance(compile_result, ReplayIngressResult):
                    raise TypeError(
                        "Replay compile checker must return ReplayIngressResult"
                    )
                if compile_result.status != "pass":
                    ingress_diagnostics = _closed_attempt_diagnostics(
                        compile_result.diagnostics
                    )
                    raise ValueError(
                        "Replay compile preflight rejected response: "
                        + json.dumps(
                            list(compile_result.diagnostics),
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                    )
        except Exception as exc:
            # This is a rejection of the submitted source-blind Replay
            # response, not a Source/J/C/pi diagnostic.  Preserve only its
            # classification so Gate1 lifecycle control never needs to parse
            # or re-emit exception text.
            ingress_diagnostics = (
                *ingress_diagnostics,
                {
                    "rule_code": "replay_response_rejected",
                    "path": "<replay-response>",
                    "exception_type": type(exc).__name__,
                },
            )
            attempt = ReplayAttempt(
                model=reply.model,
                diagnostic=f"{type(exc).__name__}: {exc}",
                response_text=reply.text,
                response_id=(
                    reply.raw.get("id")
                    if isinstance(reply.raw.get("id"), str)
                    else None
                ),
                response_model=(
                    reply.raw.get("model")
                    if isinstance(reply.raw.get("model"), str)
                    else None
                ),
                usage=_usage(reply.raw),
                raw_response=reply.raw_response,
                attempts=reply.attempts,
                physical_provider_calls=reply.attempts,
                transport_retries=max(0, reply.attempts - 1),
                elapsed_ms=reply.elapsed_ms,
                ingress_diagnostics=ingress_diagnostics,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        generation = ReplayGeneration(
            code=code,
            function_name=function_name,
            model=reply.model,
            response_id=(
                reply.raw.get("id")
                if isinstance(reply.raw.get("id"), str) else None
            ),
            context_receipt=_context_receipt(
                reply.raw, packet.get("u_f", {})
            ),
            usage=_usage(reply.raw),
            response_model=(
                reply.raw.get("model")
                if isinstance(reply.raw.get("model"), str)
                else None
            ),
            entry_function_name=entry_function_name,
            response_text=reply.text,
            raw_response=reply.raw_response,
            attempts=reply.attempts,
            physical_provider_calls=reply.attempts,
            transport_retries=max(0, reply.attempts - 1),
            elapsed_ms=reply.elapsed_ms,
        )
        self._records.append(generation)
        return generation


@dataclass
class MiniSWEReplayWriter:
    """Fresh bounded single-owner source-blind Replay agent."""

    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    worker_config: Mapping[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None
    compile_checker: ReplayCompileChecker | None = None
    stop_requested: Callable[[], int | None] | None = field(
        default=None, repr=False
    )
    _records: list[ReplayGeneration | ReplayAttempt] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(self._records)

    def generate(
        self,
        artifact: Gate1Artifact,
        *,
        masked: bool = False,
        callable_abi_order: Sequence[str] | None = None,
    ) -> ReplayGeneration:
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        timeout = self.model_timeout_seconds
        files, packet, instructions = _replay_input_files(
            artifact,
            frontend=self.frontend,
            masked=masked,
            callable_abi_order=callable_abi_order,
        )
        function_name = str(packet["u_f"]["replay_function"])
        try:
            from minisweagent.environments.intentir_replay import (
                IntentIRReplayWorkspace,
            )
            from minisweagent.run.intentir_replay import (
                IntentIRReplayRunError,
                create_intentir_replay_model,
                run_intentir_replay,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MiniSWE Replay backend requires the lift-agent dependencies"
            ) from exc

        # This exact callable is used by every virtual ingress check and once
        # again by the writer after submission. It has no compiler or Source
        # dependency.
        def ingress(source_text: str) -> ReplayIngressResult:
            return replay_source_ingress(
                source_text,
                artifact=artifact,
                frontend=self.frontend,
                function_name=function_name,
                packet=packet,
            )

        compile_checker = self.compile_checker
        probe_compiler: TritonCompiler | None = None
        if compile_checker is None:
            if self.frontend is not Frontend.TRITON:
                raise ValueError(
                    "native MiniSWE Replay requires a frontend-specific compile checker"
                )
            probe_compiler = TritonCompiler(timeout=120.0)

        def compile_check(source_text: str) -> dict[str, Any]:
            ingress_result = ingress(source_text)
            if ingress_result.status != "pass" or ingress_result.code is None:
                return ingress_result.to_check_response()
            compile_result = (
                compile_checker(
                    ingress_result.code,
                    artifact,
                    function_name,
                )
                if compile_checker is not None
                else source_blind_replay_compile_check(
                    ingress_result.code,
                    artifact,
                    function_name,
                    compiler=probe_compiler,
                    packet=packet,
                )
            )
            return compile_result.to_check_response()

        workspace = IntentIRReplayWorkspace.create(files)
        result = None
        formal_ingress: ReplayIngressResult | None = None
        try:
            model = create_intentir_replay_model(
                model_name=provider.model,
                api_base=provider.base_url,
                api_key=provider.api_key,
                reasoning_effort=effective_effort,
                timeout_seconds=timeout,
            )
            set_cancel_requested = getattr(model, "set_cancel_requested", None)
            if callable(set_cancel_requested):
                set_cancel_requested(self.stop_requested)
            result = run_intentir_replay(
                model=model,
                workspace=workspace,
                config_overrides=self.worker_config,
                replay_checker=lambda source_text: ingress(
                    source_text
                ).to_check_response(),
                replay_compile_checker=compile_check,
            )
            assert result.source_text is not None
            formal_ingress = ingress(result.source_text)
        except IntentIRReplayRunError as exc:
            result = exc.result
            safe_trajectory = _sanitize_agent_trajectory(
                result.trajectory, provider.api_key
            )
            raw_trajectory = _trajectory_bytes(safe_trajectory)
            usage, response_id, response_model = _agent_usage_and_identity(
                safe_trajectory
            )
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=(
                    "Replay agent terminated without a formal submission: "
                    + result.exit_status
                ),
                response_text=result.source_text or "",
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                physical_provider_calls=result.physical_provider_calls,
                transport_retries=result.transport_retries,
                elapsed_ms=result.elapsed_ms,
                trajectory=safe_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        except Exception as exc:
            infrastructure_trajectory = {
                "info": {
                    "backend": MINISWE_REPLAY_BACKEND,
                    "fresh_session": True,
                    "exit_status": "InfrastructureError",
                    "exception_type": type(exc).__name__,
                }
            }
            raw_trajectory = _trajectory_bytes(infrastructure_trajectory)
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=(
                    "Replay infrastructure failure: " + type(exc).__name__
                ),
                response_text=(
                    result.source_text
                    if result is not None and result.source_text is not None
                    else ""
                ),
                raw_response=raw_trajectory,
                attempts=result.model_calls if result is not None else 0,
                physical_provider_calls=(
                    result.physical_provider_calls if result is not None else 0
                ),
                transport_retries=(
                    result.transport_retries if result is not None else 0
                ),
                elapsed_ms=result.elapsed_ms if result is not None else 0.0,
                trajectory=infrastructure_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
            )
            self._records.append(attempt)
            raise
        finally:
            workspace.cleanup()

        assert result is not None and result.source_text is not None
        assert formal_ingress is not None
        safe_trajectory = _sanitize_agent_trajectory(
            result.trajectory, provider.api_key
        )
        raw_trajectory = _trajectory_bytes(safe_trajectory)
        usage, response_id, response_model = _agent_usage_and_identity(
            safe_trajectory
        )
        if formal_ingress.status != "pass" or formal_ingress.code is None:
            ingress_diagnostics = _closed_attempt_diagnostics(
                formal_ingress.diagnostics
            )
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=json.dumps(
                    formal_ingress.to_check_response(), sort_keys=True
                ),
                response_text=result.source_text,
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                physical_provider_calls=result.physical_provider_calls,
                transport_retries=result.transport_retries,
                elapsed_ms=result.elapsed_ms,
                trajectory=safe_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
                ingress_diagnostics=ingress_diagnostics,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt)
        info = safe_trajectory.get("info", {})
        replay_check = (
            info.get("replay_check", {}) if isinstance(info, Mapping) else {}
        )
        replay_compile_check = (
            info.get("replay_compile_check", {})
            if isinstance(info, Mapping) else {}
        )
        replay_transaction = (
            info.get("replay_transaction", {})
            if isinstance(info, Mapping) else {}
        )
        context_receipt = _context_receipt({}, packet["u_f"])
        context_receipt.update(
            {
                "backend_protocol": MINISWE_REPLAY_BACKEND,
                "fresh_session": True,
                "model_calls": result.model_calls,
                "physical_provider_calls": result.physical_provider_calls,
                "transport_retries": result.transport_retries,
                "termination_status": result.exit_status,
                "replay_check": (
                    dict(replay_check)
                    if isinstance(replay_check, Mapping)
                    else {}
                ),
                "replay_compile_check": (
                    dict(replay_compile_check)
                    if isinstance(replay_compile_check, Mapping)
                    else {}
                ),
                "replay_transaction": (
                    dict(replay_transaction)
                    if isinstance(replay_transaction, Mapping)
                    else {}
                ),
                "formal_ingress_recheck": True,
                "formal_ingress_protocol": REPLAY_CHECK_PROTOCOL,
                "source_blind_synthetic_compile_probe": True,
            }
        )
        generation = ReplayGeneration(
            code=formal_ingress.code,
            function_name=function_name,
            model=provider.model,
            response_id=response_id,
            context_receipt=context_receipt,
            usage=usage,
            response_model=response_model,
            entry_function_name=formal_ingress.entry_function_name,
            response_text=result.source_text,
            raw_response=raw_trajectory,
            attempts=result.model_calls,
            physical_provider_calls=result.physical_provider_calls,
            transport_retries=result.transport_retries,
            elapsed_ms=result.elapsed_ms,
            trajectory=safe_trajectory,
            backend_protocol=MINISWE_REPLAY_BACKEND,
        )
        self._records.append(generation)
        return generation


@dataclass
class StaticReplayWriter:
    code: str
    model: str = "static-test-writer"
    frontend: Frontend = Frontend.TRITON
    calls: list[bool] = field(default_factory=list, init=False)

    def generate(
        self,
        artifact: Gate1Artifact,
        *,
        masked: bool = False,
        callable_abi_order: Sequence[str] | None = None,
    ) -> ReplayGeneration:
        _instructions, user = build_rewriter_prompt(
            artifact,
            frontend=self.frontend,
            masked=masked,
            callable_abi_order=callable_abi_order,
        )
        packet = json.loads(user)
        u_f = packet["u_f"]
        function_name = u_f["replay_function"]
        frontend = Frontend(u_f["frontend"])
        code, entry_function_name = _extract_replay_code(
            self.code,
            frontend=frontend,
            function_name=function_name,
        )
        self.calls.append(masked)
        return ReplayGeneration(
            code=code,
            function_name=function_name,
            model=self.model,
            response_id=None,
            entry_function_name=entry_function_name,
            context_receipt=_context_receipt({}, packet.get("u_f", {})),
            backend_protocol=STATIC_REPLAY_BACKEND,
        )


__all__ = [
    "MINISWE_REPLAY_BACKEND", "RESPONSES_REPLAY_BACKEND",
    "STATIC_REPLAY_BACKEND", "MiniSWEReplayWriter", "RecordedReplayWriter",
    "ReplayAttempt", "ReplayGeneration", "ReplayGenerationError",
    "ReplayIngressResult", "ReplayWriter", "ResponsesReplayWriter",
    "StaticReplayWriter", "build_source_blind_replay_compile_case",
    "compiled_packet_obligation_diagnostics", "replay_source_ingress",
    "source_blind_replay_compile_check",
]
