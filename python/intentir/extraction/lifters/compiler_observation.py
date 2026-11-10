"""Exact compiler observation exposed only to the source-aware Lift owner.

This module deliberately stops at compilation and the existing Gate1 observer.
It never interprets the observation as J/C/pi and never constructs artifact
fields.  The same frozen anonymous Source specialization is compiled once and
stored in the branch-local read-only Lift input tree for focused queries.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import traceback
from typing import Any, Mapping

from intentir.verify.gate1.compiler import (
    EXTERNAL_MATH_FAILURE_RULES,
    CompileCase,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.ttir import TTIRFingerprint, fingerprint_ttir
from intentir.verify.gate1.replay import compiled_packet_obligation_diagnostics
from intentir.verify.gate1.pi import (
    NORMALIZATION_WITNESS_SCHEMA,
    ProgramCoordinateProjectionError,
    canonical_program_coordinate_projection,
)
from intentir.ir.canonical_construction import (
    SOURCE_CONSTEXPR_UNKNOWN,
    source_constexpr_value,
    CanonicalConstruction,
    SourceFacts,
    SourceTransferLedger,
    validate_source_owner_binding,
    TransferOccurrence,
    canonical_output_records,
    coerce_source_transfer_ledger,
    is_canonical_ordinal,
)
from intentir.ir.canonical_j import ContractSort, parse_contract_expr
from intentir.ir.intent_op_contract import canonicalize_dtype


SOURCE_TTIR_OBSERVATION_PROTOCOL = "intentir.lift-source-ttir.v1"
SOURCE_TTIR_RECONCILE_PROTOCOL = "intentir.lift-source-reconcile.v1"

# Worker failures are closed classes from the compiler, not parsed traceback
# text. Prefixing them with the Source observation stage prevents a Replay
# worker label from being mistaken for a Replay attempt.
_SOURCE_COMPILER_FAILURE_RULES = {
    "triton_replay_entry_lookup_failed": "source_observation_entry_lookup_failed",
    "triton_replay_entry_type_failed": "source_observation_entry_type_failed",
    "triton_replay_module_syntax_failed": "source_observation_module_syntax_failed",
    "triton_replay_callable_abi_failed": "source_observation_callable_abi_failed",
    "triton_compile_cuda_unavailable": "source_observation_cuda_unavailable",
    "triton_compile_resources_exceeded": "source_observation_resources_exceeded",
    "triton_compile_worker_timeout": "source_observation_worker_timeout",
    "triton_compile_worker_result_invalid": "source_observation_worker_result_invalid",
    "triton_compile_worker_process_failed": "source_observation_worker_process_failed",
    "triton_compile_value_validation_failed": "source_observation_value_validation_failed",
    "triton_compile_runtime_failed": "source_observation_runtime_failed",
    "triton_compile_worker_failed": "source_observation_worker_failed",
    "triton_compile_attribute_unavailable": "source_observation_attribute_unavailable",
    "triton_compile_type_incompatible": "source_observation_type_incompatible",
    "triton_frontend_broadcast_invalid": "source_observation_frontend_broadcast_invalid",
    "triton_frontend_reshape_invalid": "source_observation_frontend_reshape_invalid",
    "triton_frontend_constexpr_usage_invalid": "source_observation_frontend_constexpr_invalid",
    "triton_frontend_pointer_expression_invalid": "source_observation_frontend_pointer_invalid",
    "triton_frontend_tensor_index_invalid": "source_observation_frontend_index_invalid",
    "triton_frontend_operand_type_invalid": "source_observation_frontend_operand_type_invalid",
    "triton_frontend_helper_jit_missing": "source_observation_frontend_helper_jit_missing",
    "triton_frontend_compilation_failed": "source_observation_frontend_compile_failed",
}

# This closed, source-blind vocabulary is shared by Lift and persisted workload
# summaries. Neither raw compiler diagnostics nor private ledger reasons may
# enter a public failure receipt by merely matching a string pattern.
SAFE_SOURCE_PREPARATION_REASONS = frozenset({
    "source_observation_facts_unavailable",
    "source_observation_source_missing",
    "source_observation_entry_missing",
    "source_observation_compiler_view_missing",
    "source_observation_compiler_view_invalid",
    "source_observation_compile_failed",
    "source_observation_entry_lookup_failed",
    "source_observation_cuda_unavailable",
    "source_observation_worker_timeout",
    "source_observation_ttir_unavailable",
    "source_observation_abi_projection_unavailable",
    "source_observation_fingerprint_failed",
    "source_observation_unavailable",
    "source_transfer_ledger_unavailable",
    "source_memory_occurrence_sidecars_incomplete",
    "source_memory_occurrence_lexical_ledger_unavailable",
    "source_memory_occurrence_observation_unavailable",
    "source_atomic_update_observation_unavailable",
    "source_atomic_update_authorization_unavailable",
    "source_compiler_access_counterpart_ambiguous",
    "source_compiler_access_counterpart_unavailable",
    "source_predicate_access_anchor_unavailable",
    "source_compiler_predicate_access_projection_unavailable",
    "source_semantic_consumer_unavailable",
    "source_authoring_contract_unavailable",
    "source_exact_owner_binding_unavailable",
    "source_computed_guard_authorization_unavailable",
    "source_extern_target_unproved",
    "source_extern_linkage_unproved",
    "source_extern_library_unavailable",
    "source_extern_library_identity_unproved",
    "source_extern_linked_ir_unavailable",
    "source_extern_ftz_unproved",
    "source_extern_compiler_override_forbidden",
    "source_extern_rsqrt_authorization_unavailable",
    "source_extern_rsqrt_ftz_contract_unavailable",
    "iteration_binding_not_source_proven",
    "source_program_domain_projection_unavailable",
} | set(_SOURCE_COMPILER_FAILURE_RULES.values()))


def source_preparation_failure_reason(
    observation: Mapping[str, Any], ledger: SourceTransferLedger,
) -> str | None:
    """Classify the first unclosed Source stage using only issued rule codes."""

    if ledger.status == "pass":
        return None
    rule = observation.get("rule_code")
    if isinstance(rule, str) and rule in SAFE_SOURCE_PREPARATION_REASONS:
        return rule
    if observation.get("status") == "pass":
        reason = ledger.reason
        return (
            reason
            if isinstance(reason, str) and reason in SAFE_SOURCE_PREPARATION_REASONS
            else "source_transfer_ledger_unavailable"
        )
    return (
        "source_observation_compile_failed"
        if observation.get("status") == "fail"
        else "source_observation_unavailable"
    )


def source_preparation_failure_stage(reason: str) -> str | None:
    """Project one whitelisted rule to its first unclosed Source stage."""

    if reason not in SAFE_SOURCE_PREPARATION_REASONS:
        return None
    if reason == "source_observation_abi_projection_unavailable":
        return "ttir_abi_projection"
    if reason == "source_observation_fingerprint_failed":
        return "ttir_fingerprint"
    if reason == "source_observation_ttir_unavailable":
        return "ttir_extraction"
    if reason in {
        "source_observation_facts_unavailable",
        "source_observation_source_missing",
        "source_observation_entry_missing",
        "source_observation_compiler_view_missing",
        "source_observation_compiler_view_invalid",
        "source_observation_unavailable",
    }:
        return "source_observation_setup"
    if reason.startswith("source_observation_"):
        return "source_compilation"
    if reason in {
        "source_transfer_ledger_unavailable",
        "source_memory_occurrence_sidecars_incomplete",
        "source_memory_occurrence_lexical_ledger_unavailable",
        "source_memory_occurrence_observation_unavailable",
        "source_atomic_update_observation_unavailable",
        "source_atomic_update_authorization_unavailable",
        "source_compiler_access_counterpart_ambiguous",
        "source_compiler_access_counterpart_unavailable",
    }:
        return "source_transfer_ledger"
    return "source_authorization"

_DTYPE_ALIASES = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}


def _runtime_argument(value: Mapping[str, Any]) -> RuntimeArgument:
    kind = value.get("kind")
    dtype_value = value.get("dtype")
    if not isinstance(dtype_value, str) or not dtype_value:
        raise ValueError("anonymous compiler argument requires a dtype")
    dtype = _DTYPE_ALIASES.get(dtype_value, dtype_value)
    if kind == "tensor":
        shape = value.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in shape
        ):
            raise ValueError("anonymous tensor argument requires an integer shape")
        strides = value.get("strides")
        if strides is not None and (
            not isinstance(strides, list) or len(strides) != len(shape)
            or any(type(stride) is not int or stride < 0 for stride in strides)
        ):
            raise ValueError("anonymous tensor strides must match its rank")
        return RuntimeArgument.tensor(tuple(shape), dtype, strides=strides)
    if kind == "scalar":
        scalar = value.get("value")
        if isinstance(scalar, bool) or isinstance(scalar, (int, float)):
            return RuntimeArgument.scalar(scalar, dtype)
        raise ValueError("anonymous scalar argument requires a numeric value")
    if kind == "none":
        return RuntimeArgument.none(dtype)
    raise ValueError(
        "Source TTIR observation supports tensor/scalar/none compiler arguments"
    )


def _compile_case(view: Mapping[str, Any]) -> CompileCase:
    raw_arguments = view.get("arguments")
    if not isinstance(raw_arguments, list) or not raw_arguments:
        raise ValueError("anonymous compiler view requires runtime arguments")
    arguments: dict[str, RuntimeArgument] = {}
    for raw in raw_arguments:
        if not isinstance(raw, Mapping):
            raise TypeError("anonymous compiler arguments must be objects")
        name = raw.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("anonymous compiler argument requires an identifier")
        if name in arguments:
            raise ValueError("anonymous compiler argument names must be unique")
        arguments[name] = _runtime_argument(raw)

    raw_grid = view.get("grid")
    if not isinstance(raw_grid, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw_grid
    ):
        raise ValueError("anonymous compiler view requires an integer grid")

    raw_compiler_slots = view.get("compiler_slots", [])
    raw_compiler_roles = view.get("compiler_roles", {})
    if not isinstance(raw_compiler_slots, list) or not all(
        isinstance(item, str) and item.isidentifier()
        for item in raw_compiler_slots
    ):
        raise TypeError("anonymous compiler_slots must be identifier names")
    if not isinstance(raw_compiler_roles, Mapping):
        raise TypeError("anonymous compiler_roles must be an object")
    compiler_slots = set(raw_compiler_slots)
    compiler_roles = {
        str(name): str(role)
        for name, role in raw_compiler_roles.items()
        if str(name) in compiler_slots
    }
    if set(compiler_roles) != compiler_slots or any(
        not role.isidentifier() for role in compiler_roles.values()
    ):
        raise ValueError("anonymous compiler slots require exact identifier roles")

    schedule: dict[str, bool | int | float | str] = {}
    for field in ("bound_constexpr_values", "schedule_values"):
        values = view.get(field, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"anonymous compiler view {field} must be an object")
        for name, value in values.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("anonymous constexpr names must be identifiers")
            compile_name = compiler_roles.get(name, name)
            if compile_name in schedule:
                raise ValueError("anonymous constexpr bindings must be disjoint")
            if not isinstance(value, (bool, int, float, str)):
                raise TypeError("anonymous constexpr values must be JSON scalars")
            schedule[compile_name] = value
    return CompileCase(arguments, tuple(raw_grid), schedule)


def _source_runtime_parameter_uses(
    source: str,
    *,
    function_name: str,
    constexpr_values: Mapping[str, bool | int | float | str] | None = None,
) -> tuple[str, ...]:
    """Return exact callable-parameter reads before compiler specialization.

    This is syntax-only occurrence evidence.  It does not classify the use,
    assign J/C/pi ownership, or infer a route.
    """

    tree = ast.parse(source, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    parameters = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    static_env = dict(constexpr_values or {})
    writes = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))}
    static_env = {name: value for name, value in static_env.items() if name in parameters and name not in writes}
    unknown = SOURCE_CONSTEXPR_UNKNOWN

    def static_value(node: ast.AST) -> object:
        return source_constexpr_value(node, static_env)

    class SelectedVariantUseVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.used: set[str] = set()

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load) and node.id in parameters:
                self.used.add(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_If(self, node: ast.If) -> None:
            self.visit(node.test)
            selected = static_value(node.test)
            if selected is unknown:
                bodies = (node.body, node.orelse)
            else:
                bodies = (node.body if bool(selected) else node.orelse,)
            for body in bodies:
                for statement in body:
                    self.visit(statement)

        def visit_IfExp(self, node: ast.IfExp) -> None:
            self.visit(node.test)
            selected = static_value(node.test)
            if selected is unknown:
                self.visit(node.body)
                self.visit(node.orelse)
            else:
                self.visit(node.body if bool(selected) else node.orelse)

    visitor = SelectedVariantUseVisitor()
    for statement in function.body:
        visitor.visit(statement)
    return tuple(sorted(visitor.used))


def _ttir_argument_categories(
    text: str,
    *,
    function_name: str,
) -> tuple[tuple[int, str, str], ...] | None:
    """Extract the closed category/order view of the observed TTIR ABI.

    Triton may remove specialized scalar carriers from a compiled function.
    The remaining ``argN`` names therefore cannot be treated as positions in
    the pre-specialization Source ABI.  This deliberately exposes no TTIR
    program text: it returns only the argument number and its coarse ABI
    category required to prove an order-preserving projection.
    """

    if not isinstance(function_name, str) or not function_name.isidentifier():
        return None
    match = re.search(
        rf"(?:^|\n)\s*(?:tt\.func|func\.func)\s+(?:public\s+)?@{re.escape(function_name)}\s*\(",
        text,
    )
    if match is None:
        return None
    start = match.end()
    depth = 1
    end = start
    while end < len(text) and depth:
        char = text[end]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        end += 1
    if depth or end <= start:
        return None
    signature = text[start : end - 1]
    markers = list(re.finditer(r"%arg([0-9]+)\s*:\s*", signature))
    if not markers:
        return None
    result: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    for index, marker in enumerate(markers):
        position = int(marker.group(1))
        if position in seen:
            return None
        seen.add(position)
        type_text = signature[
            marker.end() : markers[index + 1].start() if index + 1 < len(markers) else len(signature)
        ].lstrip()
        if type_text.startswith(("!tt.ptr<", "!tt.tensor<")):
            category = "pointer"
            element = re.match(r"!tt\.(?:ptr|tensor)<([^,>]+)", type_text)
            if element is None:
                return None
            dtype = _DTYPE_ALIASES.get(element.group(1), element.group(1)).casefold()
        elif (integer := re.match(r"(?:i|ui)[0-9]+(?:\b|$)", type_text)) is not None:
            category = "integer"
            dtype = _DTYPE_ALIASES.get(integer.group(0), integer.group(0)).casefold()
        elif (floating := re.match(r"(?:f|bf)[0-9]+(?:\b|$)", type_text)) is not None:
            category = "float"
            dtype = _DTYPE_ALIASES.get(floating.group(0), floating.group(0)).casefold()
        else:
            return None
        result.append((position, category, dtype))
    return tuple(result)


def _source_argument_category(
    argument: Mapping[str, Any],
) -> tuple[str, str] | None:
    kind = argument.get("kind")
    dtype = argument.get("dtype")
    if not isinstance(dtype, str):
        return None
    normalized = _DTYPE_ALIASES.get(dtype, dtype).casefold()
    if kind in {"tensor", "pointer"}:
        return "pointer", normalized
    if kind != "scalar":
        return None
    if normalized.startswith(("float", "bfloat")):
        return "float", normalized
    if normalized.startswith(("int", "uint", "bool")):
        return "integer", normalized
    return None


def _ttir_source_abi_aliases(
    ttir: str,
    source_facts: SourceFacts,
    *,
    function_name: str,
    python_interface: Mapping[str, Any],
) -> dict[str, str] | None:
    """Consume the actual compiler specialization's named ABI receipt.

    Types validate the mapping; they never select it. A missing compiler
    receipt cannot fall back to an order-preserving type match.
    """
    if python_interface.get("compiled_ttir_sha256") != hashlib.sha256(ttir.encode("utf-8")).hexdigest():
        return None
    if not isinstance(python_interface.get("compiled_source_file"), str) or not python_interface["compiled_source_file"]:
        return None
    entry = python_interface.get("compiled_entry_name")
    names = python_interface.get("arg_names")
    runtime = python_interface.get("compiled_arg_names")
    types = python_interface.get("compiled_arg_types")
    specialized = python_interface.get("specialized_arg_names")
    constexpr_indices = python_interface.get("constexpr_indices")
    if (
        not isinstance(entry, str) or not isinstance(function_name, str)
        or python_interface.get("compiled_source_function") != function_name
        or not all(isinstance(items, list) for items in (names, runtime, types, specialized, constexpr_indices))
        or not all(isinstance(name, str) for items in (names, runtime, specialized) for name in items)
        or any(len(set(items)) != len(items) for items in (names, runtime, specialized))
        or set(runtime) & set(specialized) or set(runtime) | set(specialized) != set(names)
        or runtime != [name for name in names if name not in specialized]
        or len(runtime) != len(types)
        or any(type(index) is not int or not 0 <= index < len(names) for index in constexpr_indices)
        or python_interface.get("constexprs") != [names[index] for index in constexpr_indices]
        or not {names[index] for index in constexpr_indices}.issubset(specialized)
    ):
        return None
    observed = _ttir_argument_categories(ttir, function_name=entry)
    if observed is None or len(observed) != len(runtime):
        return None
    # ``arg_names`` may come from either of the two host-owned compiler
    # entries: the exact Source spelling or the already-anonymized Gate
    # spelling.  The SourceFacts mapping is directed from the former to the
    # latter, so normalize only through that explicit mapping (or an exact
    # canonical identity), never through argument order or types.
    parameter_order = tuple(source_facts.parameter_order)
    source_to_opaque = dict(source_facts.source_to_opaque)
    if (
        len(source_to_opaque) != len(source_facts.source_to_opaque)
        or len(set(source_to_opaque.values())) != len(source_to_opaque)
        or set(source_to_opaque.values()) != set(parameter_order)
    ):
        return None
    if all(name in source_to_opaque for name in names):
        canonical_names = tuple(source_to_opaque[name] for name in names)
    elif all(name in parameter_order for name in names):
        canonical_names = tuple(names)
    else:
        return None
    if canonical_names != parameter_order:
        return None
    facts = {row["name"]: row for row in source_facts.arguments}
    # Every Source runtime slot must have one named declaration, even if the
    # compiler explicitly specialized that scalar away.  Compiler constexpr
    # slots are deliberately outside ``facts`` but remain covered by the
    # exact parameter-order check above.
    declared = [name for name in canonical_names if name in facts]
    if len(declared) != len(facts) or set(declared) != set(facts):
        return None
    result: dict[str, str] = {}
    jit_aliases = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "fp64": "float64"}
    for name, signature, (position, category, dtype) in zip(runtime, types, observed, strict=True):
        opaque = source_to_opaque.get(name, name)
        argument = facts.get(opaque)
        if argument is None or not isinstance(signature, str):
            return None
        expected = _source_argument_category(argument)
        pointer = signature.startswith("*")
        spelling = signature[1:] if pointer else signature
        compiled_dtype = jit_aliases.get(spelling, _DTYPE_ALIASES.get(spelling, spelling)).casefold()
        if (category == "pointer") != pointer or compiled_dtype != dtype or expected is None or expected[0] != category:
            return None
        if category != "integer" and expected[1] != dtype:
            return None
        if category == "integer" and expected[1] != dtype:
            width = re.fullmatch(r"(u?)int([0-9]+)", dtype)
            value = argument.get("value")
            if width is None or type(value) is not int:
                return None
            bits = int(width[2])
            low, high = (0, 2 ** bits) if width[1] else (-2 ** (bits - 1), 2 ** (bits - 1))
            if not low <= value < high:
                return None
        match = re.fullmatch(r"p([0-9]+)", opaque or "")
        if match is None:
            return None
        result[f"arg{position}"] = match[1]
    for name in specialized:
        argument = facts.get(source_to_opaque.get(name, name))
        if argument is not None and argument.get("kind") != "scalar":
            return None
    return result


def _source_observation_python_interface(
    ttir: str,
    python_interface: Mapping[str, Any],
    source_facts: SourceFacts,
    *,
    function_name: str,
) -> dict[str, Any] | None:
    """Attach the one proven TTIR-to-Source ABI view to an observation."""

    aliases = _ttir_source_abi_aliases(
        ttir,
        source_facts,
        function_name=function_name,
        python_interface=python_interface,
    )
    if aliases is None:
        return None
    result = dict(python_interface)
    result["ttir_argument_aliases"] = aliases
    return result


def observe_source_ttir(
    source: str,
    compiler_view: Mapping[str, Any] | None,
    *,
    source_function: str,
    compiler: TritonCompiler | None = None,
    import_roots: tuple[str, ...] = (),
    source_facts: SourceFacts | None = None,
) -> dict[str, Any]:
    """Compile one host-only exact Source specialization and expose facts only.

    Failure is returned as a closed observation rather than being reclassified as
    a model failure.  The caller may require a passing observation for a formal
    capability run, while still retaining the exact compiler diagnostic.

    ``source`` never crosses the provider boundary.  The Source/compiler
    provenance ledger deliberately uses its exact debug coordinates, while the
    independently anonymized text remains the only Source representation sent
    to Lift.  This avoids treating coordinates from two different renderings as
    if they named the same occurrence.
    """

    base: dict[str, Any] = {
        "protocol": SOURCE_TTIR_OBSERVATION_PROTOCOL,
        "authority": (
            "exact host-only Source compilation and existing Gate1 observer only; "
            "no J/C/pi field, ownership decision, or repair is generated"
        ),
        "function": "source_entry",
    }
    if not isinstance(source_facts, SourceFacts):
        return {
            **base,
            "status": "unavailable",
            "rule_code": "source_observation_facts_unavailable",
            "diagnostic": "canonical SourceFacts are required for Source observation",
        }
    base["source_facts"] = source_facts.to_json()
    if not isinstance(source, str) or not source:
        return {**base, "status": "fail", "rule_code": "source_observation_source_missing", "diagnostic": "Source is absent"}
    if not isinstance(source_function, str) or not source_function.isidentifier():
        return {**base, "status": "unavailable", "rule_code": "source_observation_entry_missing", "diagnostic": "Source entry is absent"}
    if not isinstance(compiler_view, Mapping):
        return {
            **base,
            "status": "unavailable",
            "rule_code": "source_observation_compiler_view_missing",
            "diagnostic": "anonymous compiler view is absent",
        }
    stage = "compiler_view"
    try:
        case = _compile_case(compiler_view)
        stage = "source_compile"
        result = (compiler or TritonCompiler()).compile(
            KernelSource.from_code(
                source,
                source_function,
                trusted=True,
                import_roots=import_roots,
            ),
            case,
        )
        if not result.ok or not isinstance(result.ttir, str) or not result.ttir:
            return {
                **base,
                "status": "fail",
                "rule_code": (
                    _SOURCE_COMPILER_FAILURE_RULES.get(result.failure_class, "source_observation_compile_failed")
                    if not result.ok else "source_observation_ttir_unavailable"
                ),
                "diagnostic": result.diagnostic or "compiler returned no TTIR",
                "elapsed_ms": result.elapsed_ms,
                "compiler": dict(result.compiler),
                "target": dict(result.target),
            }
        stage = "abi_projection"
        python_interface = _source_observation_python_interface(
            result.ttir,
            result.python_interface,
            source_facts,
            function_name=source_function,
        )
        if python_interface is None:
            return {
                **base,
                "status": "unavailable",
                "rule_code": "source_observation_abi_projection_unavailable",
                "diagnostic": "observed TTIR ABI has no unique SourceFacts projection",
                "elapsed_ms": result.elapsed_ms,
                "compiler": dict(result.compiler),
                "target": dict(result.target),
            }
        stage = "fingerprint"
        fingerprint = fingerprint_ttir(
            result.ttir,
            external_math=result.external_math,
            python_interface=python_interface,
            schedule=case.schedule,
        )
        return {
            **base,
            "status": "pass",
            "ttir": result.ttir,
            "fingerprint": fingerprint.to_dict(),
            "fingerprint_digest": fingerprint.digest,
            "source_runtime_parameter_uses": [
                opaque
                for name in _source_runtime_parameter_uses(
                    source,
                    function_name=source_function,
                    constexpr_values=case.schedule,
                )
                if (opaque := dict(source_facts.source_to_opaque).get(name))
                is not None
                and re.fullmatch(r"p[0-9]+", opaque) is not None
            ],
            "python_interface": python_interface,
            "_external_math": dict(result.external_math),
            "compiler": dict(result.compiler),
            "target": dict(result.target),
            "elapsed_ms": result.elapsed_ms,
            "cache_hit": result.cache_hit,
        }
    except Exception as exc:
        stage_rules = {
            "compiler_view": "source_observation_compiler_view_invalid",
            "source_compile": "source_observation_compile_failed",
            "abi_projection": "source_observation_abi_projection_unavailable",
            "fingerprint": "source_observation_fingerprint_failed",
        }
        return {
            **base,
            "status": "fail",
            "diagnostic": _closed_observer_exception(exc),
            "rule_code": (
                str(exc) if str(exc) in EXTERNAL_MATH_FAILURE_RULES
                else stage_rules[stage]
            ),
        }


def observe_anonymous_source_ttir(
    anonymous_source: str,
    compiler_view: Mapping[str, Any] | None,
    *,
    compiler: TritonCompiler | None = None,
    import_roots: tuple[str, ...] = (),
    source_facts: SourceFacts | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for an already-anonymous Source compilation."""

    return observe_source_ttir(
        anonymous_source,
        compiler_view,
        source_function="scrc_k0",
        compiler=compiler,
        import_roots=import_roots,
        source_facts=source_facts,
    )


def _source_memory_kind(operation: str) -> str | None:
    if operation.startswith("tt.load"):
        return "load"
    if operation.startswith("tt.store"):
        return "store"
    if operation.startswith("tt.atomic"):
        return "update"
    return None


def _polynomial_atoms(
    polynomial: object,
) -> tuple[str, ...]:
    atoms: list[str] = []
    if not isinstance(polynomial, (list, tuple)):
        return ()
    for term in polynomial:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            continue
        monomial = term[0]
        if not isinstance(monomial, (list, tuple)):
            continue
        atoms.extend(item for item in monomial if isinstance(item, str))
    return tuple(atoms)


def _source_program_axes(
    polynomial: object,
) -> tuple[int, ...]:
    axes: set[int] = set()
    for atom in _polynomial_atoms(polynomial):
        match = re.fullmatch(r"program_axis:([xyz])", atom)
        if match is not None:
            axes.add("xyz".index(match.group(1)))
    return tuple(sorted(axes))


def _source_argument_positions(
    polynomial: object,
) -> tuple[int, ...]:
    positions: set[int] = set()
    for atom in _polynomial_atoms(polynomial):
        match = re.fullmatch(r"arg\[(\d+)\]:.*", atom)
        if match is not None:
            positions.add(int(match.group(1)))
    return tuple(sorted(positions))


def _source_address_carrier_program_axis_pairs(
    polynomial: object,
) -> tuple[tuple[int, int], ...]:
    """Return only same-monomial carrier/launch-axis relations.

    Counting all argument atoms and all ``program_axis:*`` atoms globally is
    insufficient for a pointer polynomial: ``base + stride * pid`` has both
    kinds of atoms, but the stride ownership is established by their shared
    product monomial, not by their mere coexistence in the sum.  Keep this
    small positive proof separate from the broader atom inventory used for
    grid projection; callers that bind a stride carrier must use these pairs.
    """

    pairs: set[tuple[int, int]] = set()
    if not isinstance(polynomial, (list, tuple)):
        return ()
    for term in polynomial:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            continue
        atoms = term[0]
        if not isinstance(atoms, (list, tuple)):
            continue
        argument_positions = {
            int(match.group(1))
            for atom in atoms
            if isinstance(atom, str)
            and (match := re.fullmatch(r"arg\[(\d+)\]:.*", atom)) is not None
        }
        program_axes = {
            "xyz".index(match.group(1))
            for atom in atoms
            if isinstance(atom, str)
            and (match := re.fullmatch(r"program_axis:([xyz])", atom)) is not None
        }
        for argument_position in argument_positions:
            for physical_axis in program_axes:
                pairs.add((argument_position, physical_axis))
    return tuple(sorted(pairs))


def _source_pointer_base_position(pointer_base: object) -> int | None:
    if not isinstance(pointer_base, str):
        return None
    match = re.fullmatch(r"arg\[(\d+)\]:.*", pointer_base)
    return int(match.group(1)) if match is not None else None


def _candidate_parameter_projections(
    raw_parameters: object,
    *,
    source_facts: SourceFacts | None,
) -> tuple[
    dict[int, Mapping[str, Any]],
    dict[int, Mapping[str, Any]],
    dict[str, int],
] | None:
    """Project candidate ABI rows into Source-runtime and physical views.

    Source transfer occurrences use ``arg[i]`` where ``i`` is the runtime
    argument ordinal.  A candidate's ``pN`` suffix is a physical ABI handle,
    not that ordinal: constexpr parameters may occupy a physical slot without
    appearing in the runtime call.  Keep both views explicit so a consumer
    cannot accidentally use one namespace as the other.

    ``source_facts`` is required for the authoritative projection.  A missing
    or incomplete Source ABI is therefore reject-only rather than an
    invitation to infer runtime order from candidate-authored ``kind`` rows.
    """

    if not isinstance(raw_parameters, list) or not isinstance(source_facts, SourceFacts):
        return None
    physical_by_position: dict[int, Mapping[str, Any]] = {}
    physical_by_name: dict[str, Mapping[str, Any]] = {}
    for parameter in raw_parameters:
        if not isinstance(parameter, Mapping):
            return None
        name = parameter.get("name")
        match = re.fullmatch(r"p([0-9]+)", name or "")
        if match is None:
            continue
        position = int(match.group(1))
        if position in physical_by_position or name in physical_by_name:
            return None
        physical_by_position[position] = parameter
        physical_by_name[name] = parameter

    runtime_names = tuple(
        name
        for name in source_facts.parameter_order
        if name not in source_facts.constexpr_parameters
    )
    # SourceFacts permits the ABI wire to describe source-side declarations,
    # but the active canonical contract reserves c* for compiler-only slots.
    # Do not silently drop an unexpected c* runtime declaration: doing so
    # would renumber every later Source arg[i] and recreate the original
    # runtime/physical namespace collision.
    if any(
        not isinstance(name, str) or not name.startswith("p")
        for name in runtime_names
    ):
        return None
    runtime_by_position: dict[int, Mapping[str, Any]] = {}
    runtime_position_by_name: dict[str, int] = {}
    for runtime_position, name in enumerate(runtime_names):
        parameter = physical_by_name.get(name)
        if not isinstance(parameter, Mapping):
            return None
        runtime_by_position[runtime_position] = parameter
        runtime_position_by_name[name] = runtime_position
    return runtime_by_position, physical_by_position, runtime_position_by_name


def _source_domain_coordinates(
    boundary: Mapping[str, Any],
) -> dict[str, int]:
    domains = boundary.get("domains")
    if not isinstance(domains, Mapping):
        return {}
    result: dict[str, int] = {}
    for domain_ref, expression in domains.items():
        if not isinstance(domain_ref, str) or not isinstance(expression, str):
            continue
        # Canonical J accepts a direct logical coordinate independently of its
        # spelling.  Earlier Source->pi binding recognized only the historical
        # ``k*`` spelling, while the canonical constructor emits ``i*``.  Both
        # name the same closed ordinal coordinate; accepting the two explicit
        # spellings here keeps a Source transfer occurrence bound to its J
        # access instead of silently dropping its unique address carrier.
        coordinates = {
            int(match)
            for match in re.findall(r"\b(?:i|k)([0-9]+)\b", expression)
        }
        if len(coordinates) == 1:
            result[domain_ref] = next(iter(coordinates))
    return result


def _source_domain_extent_symbols(
    boundary: Mapping[str, Any],
) -> dict[str, str]:
    """Return the unique x* upper extent of a zero-based direct domain.

    The semantic-transfer wire spells each direct access axis with its
    canonical tensor-dimension handle (``x*``), while the host-derived domain
    uses a local coordinate (``i*``) bounded by that same handle.  Preserve
    the existing coordinate spelling, but expose this exact bridge to the
    Source->pi binder.  General affine, non-zero-based, inclusive, guarded,
    or otherwise ambiguous domains deliberately produce no bridge.
    """

    domains = boundary.get("domains")
    if not isinstance(domains, Mapping):
        return {}
    result: dict[str, str] = {}
    for domain_ref, expression in domains.items():
        if not isinstance(domain_ref, str) or not isinstance(expression, str):
            continue
        try:
            parsed = parse_contract_expr(
                expression,
                expected=ContractSort.BOOLEAN,
                context=f"J boundary domain {domain_ref}",
            )
        except Exception:
            continue
        atoms = parsed.args if parsed.op == "and" else (parsed,)
        lower_coordinates: set[str] = set()
        upper_pairs: set[tuple[str, str]] = set()
        recognized_atoms = 0
        for atom in atoms:
            if len(atom.args) != 2:
                continue
            left, right = atom.args
            if (
                atom.op == "le"
                and left.op == "int"
                and left.value == 0
                and right.op == "ref"
                and isinstance(right.value, str)
            ):
                lower_coordinates.add(right.value)
                recognized_atoms += 1
            elif (
                atom.op == "ge"
                and left.op == "ref"
                and isinstance(left.value, str)
                and right.op == "int"
                and right.value == 0
            ):
                lower_coordinates.add(left.value)
                recognized_atoms += 1
            if (
                atom.op == "lt"
                and left.op == "ref"
                and isinstance(left.value, str)
                and right.op == "ref"
                and isinstance(right.value, str)
                and re.fullmatch(r"x[0-9]+", right.value) is not None
            ):
                upper_pairs.add((left.value, right.value))
                recognized_atoms += 1
            elif (
                atom.op == "gt"
                and left.op == "ref"
                and isinstance(left.value, str)
                and re.fullmatch(r"x[0-9]+", left.value) is not None
                and right.op == "ref"
                and isinstance(right.value, str)
            ):
                upper_pairs.add((right.value, left.value))
                recognized_atoms += 1
        # A domain used as a Source launch witness must be exactly the
        # zero-based half-open interval.  Previously unrecognized extra
        # clauses were ignored, so a candidate could retain a stricter or
        # differently constrained domain while still receiving the same
        # x*-extent projection.  Do not treat a partial parse as proof.
        if len(atoms) != 2 or recognized_atoms != len(atoms):
            continue
        closed = {
            (coordinate, extent)
            for coordinate, extent in upper_pairs
            if coordinate in lower_coordinates
            and re.fullmatch(r"(?:i|k)[0-9]+", coordinate) is not None
        }
        if (
            len(lower_coordinates) == 1
            and len(upper_pairs) == 1
            and len(closed) == 1
        ):
            _coordinate, extent = next(iter(closed))
            result[domain_ref] = extent
    return result


def _index_axis_for_coordinate(
    index_expression: object,
    coordinate: int,
    *,
    extent_symbol: str | None = None,
) -> int | None:
    if not isinstance(index_expression, str):
        return None
    expression = index_expression.strip()
    if expression.startswith("[") and expression.endswith("]"):
        expression = expression[1:-1]
    terms = [item.strip() for item in expression.split(",")]
    coordinate_matches = [
        index
        for index, term in enumerate(terms)
        if re.search(rf"\b(?:i|k){coordinate}\b", term)
    ]
    extent_matches = [
        index
        for index, term in enumerate(terms)
        if extent_symbol is not None and term == extent_symbol
    ]
    matches = set((*coordinate_matches, *extent_matches))
    return next(iter(matches)) if len(matches) == 1 else None


def _axis_ref_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"x([0-9]+)", value)
        if match is not None:
            return int(match.group(1))
    return None


def _candidate_program_axis_map(
    pi: Mapping[str, Any],
) -> dict[int, set[tuple[str, int | None]]]:
    """Read already-authored program mappings without choosing a mapping."""

    try:
        projection = canonical_program_coordinate_projection(pi)
    except ProgramCoordinateProjectionError:
        return {}
    result: dict[int, set[tuple[str, int | None]]] = {}
    for row in projection.rows:
        for domain_ref, axis_ref in row.logical_axes:
            numeric_axis = _axis_ref_number(axis_ref)
            # ``program_axes`` intentionally has no logical tensor-axis
            # selector and is represented by None.  Relation rows must carry
            # a concrete axis_ref; the shared projection has already checked
            # that distinction, so a non-numeric symbolic x* is simply not a
            # candidate for this numeric Source-observation index.
            if axis_ref is not None and numeric_axis is None:
                continue
            result.setdefault(row.physical_axis, set()).add(
                (domain_ref, numeric_axis)
            )
    return result


def materialize_source_program_domains(
    packet: Mapping[str, Any],
    *,
    source_facts: SourceFacts,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a uniquely Source-bound launch coordinate into J and pi.

    Fresh MCP semantic entry deliberately has no provider write for host-owned
    program mapping. A direct launch coordinate can nevertheless be closed
    without consulting a provider graph when the mounted compiler grid binds a
    physical launch axis to one ``h*`` coordinate, the mounted ABI maps that
    coordinate to exactly one canonical ``x*`` dimension, and the same axis
    occurs in the closed Source transfer ledger. This is a construction
    projection, never an inference from rank, parameter position, a concrete
    extent, or a candidate-authored pi row.

    Any absent, mixed, or ambiguous fact leaves the packet untouched. An
    existing program-axis mapping is first compared with the independently
    derived Source mapping; only an exactly matching mapping is retained. A
    pre-existing tile iteration row alone is preserved while the independently
    proved launch mapping is projected. Exact provider domains may be reused,
    but only when one unique zero-based domain has the required Source
    dimension extent; otherwise projection fails closed without adding or
    changing a domain.
    """

    input_artifact = copy.deepcopy(dict(packet))
    original = copy.deepcopy(input_artifact)
    result: dict[str, Any] = {
        "protocol": "intentir.lift-source-program-domain.v1",
        "status": "pass",
        "artifact": original,
        "applied": [],
        "diagnostics": [],
    }
    unavailable_reasons = {
        "source_facts_missing_or_malformed",
        "transfer_ledger_not_closed",
        "candidate_wire_malformed",
        "candidate_program_mapping_malformed",
        "candidate_program_mapping_not_source_proven",
        "grid_facts_missing_or_malformed",
        "candidate_parameter_projection_ambiguous",
        "candidate_tensor_projection_malformed",
        "source_tensor_shape_projection_malformed",
        "shape_coordinate_binding_malformed",
        "no_used_physical_program_axis",
        "used_physical_program_axis_out_of_range",
        "direct_axis_coordinate_unavailable",
        "compound_grid_expression_unsupported",
        "compound_grid_factor_coordinate_unavailable",
        "shape_coordinate_extent_missing",
        "logical_dimension_binding_unavailable",
        "grid_extent_mismatch",
        "duplicate_logical_dimension",
        "tile_coordinate_binding_malformed",
        "tile_coordinate_logical_dimension_unavailable",
        "tile_slot_value_unavailable",
        "tile_program_dimension_overlap",
        "duplicate_tile_logical_dimension",
        "iteration_binding_not_source_proven",
        "provider_domain_dimension_unavailable",
        "provider_domain_dimension_ambiguous",
        "no_closed_program_domain_projection",
    }

    def unavailable(
        reason: str,
        *observed: str,
        path: str = "j.boundary.domains",
    ) -> dict[str, Any]:
        """Return one bounded, Source-free projection failure receipt."""

        if reason not in unavailable_reasons:
            raise ValueError("unknown Source program-domain failure reason")
        safe_observed = [f"reason={reason}"]
        for value in observed:
            if (
                isinstance(value, str)
                and value
                and len(value) <= 128
                and re.fullmatch(r"[A-Za-z0-9_.:=,+\-]+", value) is not None
            ):
                safe_observed.append(value)
        result["status"] = "unavailable"
        result["reason"] = reason
        # Projection is one atomic transaction.  Some failures are discovered
        # only after candidate-local domains or pi rows have been staged; never
        # expose that partial state to a caller or diagnostic consumer.
        result["artifact"] = copy.deepcopy(input_artifact)
        result["diagnostics"] = [
            {
                "rule_code": f"source_program_domain_projection_{reason}",
                "path": path,
                "observed_refs_or_types": safe_observed,
            }
        ]
        return result

    if not isinstance(source_facts, SourceFacts):
        return unavailable("source_facts_missing_or_malformed")
    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared_ledger.status != "pass":
        return unavailable(
            "transfer_ledger_not_closed",
            f"ledger_status={shared_ledger.status}",
            f"occurrence_count={len(shared_ledger.occurrences)}",
            f"binding_count={len(shared_ledger.bindings)}",
            path="j.boundary.accesses",
        )

    j = original.get("j")
    pi = original.get("pi")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    signature = j.get("signature") if isinstance(j, Mapping) else None
    if not all(isinstance(value, dict) for value in (j, pi, boundary, function, signature)):
        return unavailable("candidate_wire_malformed")
    domains = boundary.get("domains")
    program_axes = pi.get("program_axes")
    program_bindings = pi.get("program_bindings")
    iteration_bindings = pi.get("iteration_bindings")
    parameters = signature.get("parameters")
    inputs = function.get("inputs")
    outputs = function.get("outputs")
    if (
        not isinstance(domains, dict)
        or not isinstance(program_axes, list)
        or not isinstance(program_bindings, list)
        or not isinstance(iteration_bindings, list)
        or not isinstance(parameters, list)
        or not isinstance(inputs, list)
        or not isinstance(outputs, list)
    ):
        return unavailable("candidate_wire_malformed")
    # A candidate-supplied launch-axis mapping is not an authority merely
    # because it is syntactically valid.  The old early return here allowed a
    # provider-authored ``program_axes``/``program_bindings`` row to bypass the
    # Source grid, shape-coordinate and transfer-ledger projection.  That
    # breaks the Source -> symbolic-domain authorization chain: a wrong
    # logical axis could still look like a complete pi mapping to later
    # consumers.
    #
    # Derive the expected mapping on a private copy with those two candidate
    # collections cleared, then compare the canonical semantic projections.
    # The recursive call is bounded to this one cleared pass and does not
    # inspect or rewrite provider semantic values.  It may add only the
    # already Source-proven domain/launch rows.  If the candidate mapping is
    # absent, the ordinary path below performs that same projection.
    if program_axes or program_bindings:
        try:
            authored_projection = canonical_program_coordinate_projection(pi)
        except ProgramCoordinateProjectionError:
            return unavailable(
                "candidate_program_mapping_malformed",
                path="pi.program_axes+pi.program_bindings",
            )
        verification_packet = copy.deepcopy(original)
        verification_pi = verification_packet.get("pi")
        if not isinstance(verification_pi, dict):
            return unavailable("candidate_wire_malformed")
        verification_pi["program_axes"] = []
        verification_pi["program_bindings"] = []
        projected = materialize_source_program_domains(
            verification_packet,
            source_facts=source_facts,
            ledger=shared_ledger,
        )
        if not isinstance(projected, Mapping) or projected.get("status") != "pass":
            return unavailable(
                "candidate_program_mapping_not_source_proven",
                path="pi.program_axes+pi.program_bindings",
            )
        projected_artifact = projected.get("artifact")
        projected_pi = (
            projected_artifact.get("pi")
            if isinstance(projected_artifact, Mapping)
            else None
        )
        if not isinstance(projected_pi, Mapping):
            return unavailable(
                "candidate_program_mapping_not_source_proven",
                path="pi.program_axes+pi.program_bindings",
            )
        try:
            expected_projection = canonical_program_coordinate_projection(
                projected_pi
            )
        except ProgramCoordinateProjectionError:
            return unavailable(
                "candidate_program_mapping_not_source_proven",
                path="pi.program_axes+pi.program_bindings",
            )
        if authored_projection != expected_projection:
            return unavailable(
                "candidate_program_mapping_not_source_proven",
                path="pi.program_axes+pi.program_bindings",
            )
        # Return the independently projected artifact.  When the mappings
        # match, this preserves the candidate's semantic rows while exposing
        # only host-derived domain/iteration additions to the later strict
        # constructor.  No candidate mapping is copied into Source facts.
        return dict(projected)
    provider_domains_present = bool(domains)

    compiler_facts = source_facts.compiler_facts
    grid = compiler_facts.get("grid")
    grid_contract = compiler_facts.get("grid_contract")
    grid_coordinates = compiler_facts.get("grid_shape_coordinates")
    shape_coordinates = dict(source_facts.shape_coordinates)
    schedule_values = compiler_facts.get("schedule_values")
    schedule_roles = compiler_facts.get("compiler_roles")
    tile_coordinates = compiler_facts.get("schedule_tile_coordinates")
    if not all(
        isinstance(value, Mapping)
        for value in (schedule_values, schedule_roles, tile_coordinates)
    ):
        return unavailable("tile_coordinate_binding_malformed")
    if (
        not isinstance(grid, list)
        or not grid
        or len(grid) > 3
        or not isinstance(grid_contract, list)
        or not isinstance(grid_coordinates, list)
        or len(grid) != len(grid_contract)
        or (grid_coordinates and len(grid) != len(grid_coordinates))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in grid
        )
        or any(not isinstance(value, str) or not value for value in grid_contract)
        or any(
            value is not None and not isinstance(value, str)
            for value in grid_coordinates
        )
    ):
        return unavailable(
            "grid_facts_missing_or_malformed",
            f"grid_rank={len(grid) if isinstance(grid, list) else -1}",
            f"grid_contract_rank={len(grid_contract) if isinstance(grid_contract, list) else -1}",
            f"grid_coordinate_rank={len(grid_coordinates) if isinstance(grid_coordinates, list) else -1}",
            path="pi.program_bindings",
        )
    # A compound grid expression intentionally has no single h* coordinate.
    # Normalize the optional direct-coordinate sidecar to one slot per axis;
    # the closed grid contract below remains the authority for compound axes.
    if not grid_coordinates:
        grid_coordinates = [None] * len(grid)

    parameters_by_name: dict[str, Mapping[str, Any]] = {}
    for parameter in parameters:
        name = parameter.get("name") if isinstance(parameter, Mapping) else None
        if not isinstance(name, str) or name in parameters_by_name:
            return unavailable(
                "candidate_parameter_projection_ambiguous",
                f"parameter_count={len(parameters)}",
            )
        parameters_by_name[name] = parameter
    # Inputs and outputs are separate ABI collections.  A legal inout repeats
    # the same t* declaration once in each collection; only repetition within
    # one collection or a cross-collection type disagreement is ambiguous.
    # Keep this identical to the canonical value-type authority instead of
    # rejecting rope-style inouts as duplicate logical tensors.
    tensors: dict[str, Mapping[str, Any]] = {}
    tensor_types: dict[str, tuple[str, tuple[str | int, ...]]] = {}
    for collection_name, collection in (("inputs", inputs), ("outputs", outputs)):
        collection_names: set[str] = set()
        for row_index, tensor in enumerate(collection):
            name = tensor.get("name") if isinstance(tensor, Mapping) else None
            dtype = tensor.get("dtype") if isinstance(tensor, Mapping) else None
            shape = tensor.get("shape") if isinstance(tensor, Mapping) else None
            if (
                not isinstance(name, str)
                or re.fullmatch(r"t[0-9]+", name) is None
                or name in collection_names
                or not isinstance(dtype, str)
                or not dtype
                or not isinstance(shape, list)
                or any(
                    not isinstance(extent, (str, int))
                    or isinstance(extent, bool)
                    for extent in shape
                )
            ):
                return unavailable(
                    "candidate_tensor_projection_malformed",
                    f"collection={collection_name}",
                    f"row_index={row_index}",
                )
            collection_names.add(name)
            value_type = (dtype, tuple(shape))
            previous_type = tensor_types.get(name)
            if previous_type is not None and previous_type != value_type:
                return unavailable(
                    "candidate_tensor_projection_malformed",
                    f"collection={collection_name}",
                    f"row_index={row_index}",
                    "inout_type_mismatch=1",
                )
            tensor_types[name] = value_type
            tensors.setdefault(name, tensor)

    coordinate_dimensions: dict[str, set[str]] = {}
    for argument_index, argument in enumerate(source_facts.arguments):
        if argument.get("kind") != "tensor":
            continue
        name = argument.get("name")
        # A Source tensor can have a wholly static shape.  SourceFacts records
        # that closed case by omitting ``shape_contract`` and retaining the
        # literal ``shape``; canonical_source_wire_seed deliberately preserves
        # the same literals in J instead of inventing x* dimensions.  Treat
        # that shape as the contract here too.  Requiring a list unconditionally
        # made a valid static tensor prevent an otherwise unrelated, explicitly
        # shape-coordinate-bound launch axis from being projected.
        #
        # This is not a concrete-extent fallback for h* coordinates: a dynamic
        # coordinate still has to map positionally to an explicit x* dimension.
        # The literal branch below only accepts the exact Source static extent
        # at the same tensor axis.
        source_shape = argument.get("shape")
        raw_contracts = argument.get("shape_contract")
        contracts = source_shape if raw_contracts is None else raw_contracts
        parameter = parameters_by_name.get(name) if isinstance(name, str) else None
        logical_ref = parameter.get("logical_ref") if isinstance(parameter, Mapping) else None
        tensor = tensors.get(logical_ref) if isinstance(logical_ref, str) else None
        shape = tensor.get("shape") if isinstance(tensor, Mapping) else None
        if (
            not isinstance(contracts, list)
            or not isinstance(source_shape, list)
            or not isinstance(shape, list)
            or len(contracts) != len(shape)
            or len(source_shape) != len(contracts)
        ):
            return unavailable(
                "source_tensor_shape_projection_malformed",
                f"argument_index={argument_index}",
                f"shape_contract_rank={len(contracts) if isinstance(contracts, list) else -1}",
                f"logical_shape_rank={len(shape) if isinstance(shape, list) else -1}",
            )
        for axis, (coordinate, dimension) in enumerate(
            zip(contracts, shape, strict=True)
        ):
            if isinstance(coordinate, int) and not isinstance(coordinate, bool):
                if dimension != coordinate:
                    return unavailable(
                        "source_tensor_shape_projection_malformed",
                        f"argument_index={argument_index}",
                        f"axis={axis}",
                        "static_extent_mismatch=1",
                    )
                continue
            if (
                not isinstance(coordinate, str)
                or re.fullmatch(r"h[0-9]+", coordinate) is None
                or not isinstance(dimension, str)
                or re.fullmatch(r"x[0-9]+", dimension) is None
            ):
                return unavailable(
                    "source_tensor_shape_projection_malformed",
                    f"argument_index={argument_index}",
                    f"axis={axis}",
                    "shape_axis_contract_invalid=1",
                )
            coordinate_dimensions.setdefault(coordinate, set()).add(dimension)

    # Grid-contract identifiers are already canonical ABI/C names.  Resolve
    # only explicit shape-coordinate bindings from the same SourceFacts: a
    # runtime scalar's declared coordinate or a bound constexpr coordinate.
    # Concrete values alone never select a coordinate because distinct h*
    # dimensions may happen to have the same value.
    factor_coordinates: dict[str, str] = {}
    bound_coordinates = compiler_facts.get("bound_constexpr_shape_coordinates")
    if not isinstance(bound_coordinates, Mapping):
        return unavailable("shape_coordinate_binding_malformed")
    for name, coordinate in bound_coordinates.items():
        if isinstance(name, str) and isinstance(coordinate, str):
            factor_coordinates[name] = coordinate
    for argument_index, argument in enumerate(source_facts.arguments):
        name = argument.get("name") if isinstance(argument, Mapping) else None
        coordinate = (
            argument.get("shape_coordinate")
            if isinstance(argument, Mapping)
            else None
        )
        if isinstance(name, str) and isinstance(coordinate, str):
            previous = factor_coordinates.get(name)
            if previous is not None and previous != coordinate:
                return unavailable(
                    "shape_coordinate_binding_malformed",
                    f"argument_index={argument_index}",
                )
            factor_coordinates[name] = coordinate

    used_launch_axes = {
        axis
        for occurrence in shared_ledger.occurrences
        for axis in _source_program_axes(occurrence.address_terms)
    }
    if not used_launch_axes:
        return unavailable(
            "no_used_physical_program_axis",
            f"occurrence_count={len(shared_ledger.occurrences)}",
            f"grid_rank={len(grid)}",
            path="pi.program_axes",
        )
    out_of_range_axes = sorted(
        axis for axis in used_launch_axes if axis >= len(grid)
    )
    if out_of_range_axes:
        return unavailable(
            "used_physical_program_axis_out_of_range",
            f"physical_axis={out_of_range_axes[0]}",
            f"used_axis_count={len(used_launch_axes)}",
            f"grid_rank={len(grid)}",
            path="pi.program_axes",
        )
    projections: list[tuple[int, str, tuple[str, ...]]] = []
    tiled_program_projections: dict[int, tuple[str, str, str]] = {}
    for physical_axis, coordinate in enumerate(grid_coordinates):
        if physical_axis not in used_launch_axes:
            continue
        relation = "direct"
        coordinates: tuple[str, ...]
        if isinstance(coordinate, str):
            coordinates = (coordinate,)
        else:
            expression = grid_contract[physical_axis].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression) is not None:
                return unavailable(
                    "direct_axis_coordinate_unavailable",
                    f"physical_axis={physical_axis}",
                    "relation=direct",
                    path="pi.program_axes",
                )
            # A grid axis may be the ordinal of one Source-proven tile:
            # ceil_div(shape_coordinate, tile_slot).  This is not a direct
            # program-to-J coordinate and must not be represented as one.
            # The existing tiled_domain iteration row owns the logical axis;
            # retain this physical-axis witness only to close the Source
            # projection and to ensure every used launch axis is accounted for.
            tiled_match = re.fullmatch(
                r"(?:ceil_div|cdiv)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
                r"([A-Za-z_][A-Za-z0-9_]*)\s*\)",
                expression,
            )
            if tiled_match is not None:
                factor_name, tile_name = tiled_match.groups()
                factor_coordinate = factor_coordinates.get(factor_name)
                tile_coordinate = tile_coordinates.get(tile_name)
                choices = (
                    coordinate_dimensions.get(factor_coordinate, set())
                    if isinstance(factor_coordinate, str)
                    else set()
                )
                tile_value = schedule_values.get(tile_name)
                if (
                    not isinstance(factor_coordinate, str)
                    or not isinstance(tile_coordinate, str)
                    or tile_coordinate != factor_coordinate
                    or tile_name not in schedule_roles
                    or schedule_roles.get(tile_name)
                    not in {"tile_extent", "reduction_tile_extent"}
                    or len(choices) != 1
                    or type(tile_value) is not int
                    or tile_value <= 0
                    or re.fullmatch(r"h[0-9]+", factor_coordinate) is None
                    or factor_coordinate not in shape_coordinates
                    or grid[physical_axis]
                    != (shape_coordinates[factor_coordinate] + tile_value - 1)
                    // tile_value
                ):
                    return unavailable(
                        "compound_grid_factor_coordinate_unavailable",
                        f"physical_axis={physical_axis}",
                        "relation=ceil_div_tile",
                        path="pi.program_bindings",
                    )
                dimension = next(iter(choices))
                tiled_program_projections[physical_axis] = (
                    dimension,
                    tile_name,
                    factor_coordinate,
                )
                continue
            if re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\*\s*[A-Za-z_][A-Za-z0-9_]*)+",
                expression,
            ) is None:
                return unavailable(
                    "compound_grid_expression_unsupported",
                    f"physical_axis={physical_axis}",
                    "relation=compound",
                    path="pi.program_bindings",
                )
            factors = tuple(part.strip() for part in expression.split("*"))
            resolved = tuple(factor_coordinates.get(factor) for factor in factors)
            if (
                any(not isinstance(item, str) for item in resolved)
                or len(set(resolved)) != len(resolved)
            ):
                return unavailable(
                    "compound_grid_factor_coordinate_unavailable",
                    f"physical_axis={physical_axis}",
                    "relation=product_linearized",
                    f"factor_count={len(factors)}",
                    f"bound_coordinate_count={len({item for item in resolved if isinstance(item, str)})}",
                    path="pi.program_bindings",
                )
            coordinates = tuple(str(item) for item in resolved)
            relation = "product_linearized"

        dimensions: list[str] = []
        extent_product = 1
        for item in coordinates:
            if (
                re.fullmatch(r"h[0-9]+", item) is None
                or item not in shape_coordinates
            ):
                return unavailable(
                    "shape_coordinate_extent_missing",
                    f"physical_axis={physical_axis}",
                    f"relation={relation}",
                    f"coordinate={item if re.fullmatch(r'h[0-9]+', item) is not None else 'invalid'}",
                    path="pi.program_bindings",
                )
            choices = coordinate_dimensions.get(item, set())
            if len(choices) != 1:
                # No concrete-extent fallback is permitted: same-valued h*
                # axes are not interchangeable logical dimensions.
                return unavailable(
                    "logical_dimension_binding_unavailable",
                    f"physical_axis={physical_axis}",
                    f"relation={relation}",
                    f"coordinate={item}",
                    f"logical_dimension_count={len(choices)}",
                    path="pi.program_bindings",
                )
            dimensions.append(next(iter(choices)))
            extent_product *= shape_coordinates[item]
        if extent_product != grid[physical_axis]:
            return unavailable(
                "grid_extent_mismatch",
                f"physical_axis={physical_axis}",
                f"relation={relation}",
                f"coordinate_count={len(coordinates)}",
                f"logical_dimension_count={len(dimensions)}",
                path="pi.program_bindings",
            )
        if len(set(dimensions)) != len(dimensions):
            return unavailable(
                "duplicate_logical_dimension",
                f"physical_axis={physical_axis}",
                f"relation={relation}",
                f"logical_dimension_count={len(dimensions)}",
                f"unique_logical_dimension_count={len(set(dimensions))}",
                path="pi.program_bindings",
            )
        projections.append((physical_axis, relation, tuple(dimensions)))
    projected_physical_axes = {
        physical_axis for physical_axis, _relation, _dimensions in projections
    } | set(tiled_program_projections)
    if projected_physical_axes != used_launch_axes:
        return unavailable(
            "no_closed_program_domain_projection",
            f"used_axis_count={len(used_launch_axes)}",
            f"projected_axis_count={len(projected_physical_axes)}",
            path="pi.program_bindings",
        )
    if not projections and not tiled_program_projections:
        return unavailable(
            "no_closed_program_domain_projection",
            f"used_axis_count={len(used_launch_axes)}",
            f"grid_rank={len(grid)}",
            path="pi.program_bindings",
        )

    use_program_bindings = any(
        relation != "direct" for _axis, relation, _dimensions in projections
    )
    domain_ordinal = len(domains)
    existing_extent_domains: dict[str, list[str]] = {}
    if provider_domains_present:
        for domain_ref, dimension in _source_domain_extent_symbols(boundary).items():
            existing_extent_domains.setdefault(dimension, []).append(domain_ref)
    domain_by_dimension: dict[str, str] = {}

    def domain_for_dimension(dimension: str) -> str | None:
        nonlocal domain_ordinal
        existing = existing_extent_domains.get(dimension, [])
        if len(existing) > 1:
            return None
        if len(existing) == 1:
            domain_by_dimension[dimension] = existing[0]
            return existing[0]
        if provider_domains_present:
            return None
        domain_ref = f"d{domain_ordinal}"
        coordinate_ref = f"i{domain_ordinal}"
        domains[domain_ref] = (
            f"0 <= {coordinate_ref} and {coordinate_ref} < {dimension}"
        )
        domain_by_dimension[dimension] = domain_ref
        domain_ordinal += 1
        return domain_ref

    claimed_program_dimensions: set[str] = set()
    for physical_axis, relation, dimensions in projections:
        repeated_dimensions = claimed_program_dimensions & set(dimensions)
        if repeated_dimensions:
            return unavailable(
                "duplicate_logical_dimension",
                f"physical_axis={physical_axis}",
                f"relation={relation}",
                f"duplicate_count={len(repeated_dimensions)}",
                path=(
                    "pi.program_bindings"
                    if use_program_bindings
                    else "pi.program_axes"
                ),
            )
        logical_axes: list[dict[str, Any]] = []
        for dimension in dimensions:
            domain_ref = domain_for_dimension(dimension)
            if domain_ref is None:
                reason = (
                    "provider_domain_dimension_ambiguous"
                    if len(existing_extent_domains.get(dimension, [])) > 1
                    else "provider_domain_dimension_unavailable"
                )
                return unavailable(
                    reason,
                    f"dimension={dimension}",
                    f"physical_axis={physical_axis}",
                    path="j.boundary.domains",
                )
            logical_axes.append(
                {"domain_ref": domain_ref, "axis_ref": dimension}
            )
        if use_program_bindings:
            program_bindings.append(
                {
                    "axis": physical_axis,
                    "relation": relation,
                    "logical_axes": logical_axes,
                }
            )
        else:
            program_axes.append(
                {
                    "domain_ref": logical_axes[0]["domain_ref"],
                    "axis": physical_axis,
                }
            )
        claimed_program_dimensions.update(dimensions)
    for physical_axis, (dimension, _tile_slot, _coordinate) in tiled_program_projections.items():
        if dimension in claimed_program_dimensions:
            return unavailable(
                "tile_program_dimension_overlap",
                f"physical_axis={physical_axis}",
                f"dimension={dimension}",
                path="pi.iteration_bindings",
            )
    tile_dimensions: set[str] = set()
    for slot_name, coordinate in tile_coordinates.items():
        role = schedule_roles.get(slot_name)
        tile_value = schedule_values.get(slot_name)
        choices = (
            coordinate_dimensions.get(coordinate, set())
            if isinstance(coordinate, str)
            else set()
        )
        if (
            role not in {"tile_extent", "reduction_tile_extent"}
            or not isinstance(coordinate, str)
            or re.fullmatch(r"h[0-9]+", coordinate) is None
        ):
            return unavailable(
                "tile_coordinate_binding_malformed",
                f"slot={slot_name}",
                path="pi.iteration_bindings",
            )
        if len(choices) != 1:
            return unavailable(
                "tile_coordinate_logical_dimension_unavailable",
                f"slot={slot_name}",
                f"logical_dimension_count={len(choices)}",
                path="pi.iteration_bindings",
            )
        if (
            isinstance(tile_value, bool)
            or not isinstance(tile_value, int)
            or tile_value <= 0
        ):
            return unavailable(
                "tile_slot_value_unavailable",
                f"slot={slot_name}",
                path="pi.iteration_bindings",
            )
        dimension = next(iter(choices))
        if dimension in domain_by_dimension:
            return unavailable(
                "tile_program_dimension_overlap",
                f"slot={slot_name}",
                f"dimension={dimension}",
                path="pi.iteration_bindings",
            )
        if dimension in tile_dimensions:
            return unavailable(
                "duplicate_tile_logical_dimension",
                f"slot={slot_name}",
                f"dimension={dimension}",
                path="pi.iteration_bindings",
            )
        domain_ref = domain_for_dimension(dimension)
        if domain_ref is None:
            reason = (
                "provider_domain_dimension_ambiguous"
                if len(existing_extent_domains.get(dimension, [])) > 1
                else "provider_domain_dimension_unavailable"
            )
            return unavailable(
                reason,
                f"slot={slot_name}",
                f"dimension={dimension}",
                path="j.boundary.domains",
            )
        projected_iteration_binding = {
            "relation": "tiled_domain",
            "logical_axes": [
                {"subject_ref": domain_ref, "axis_ref": dimension}
            ],
            "tile_slot": slot_name,
            "mode_slot": None,
        }
        # A preceding host pass may already have emitted this exact
        # Source-proven tile witness.  Preserve it instead of duplicating the
        # same relation while adding the independently proved launch mapping.
        # We deliberately do not merge or reinterpret a non-identical row:
        # that would choose a tile/domain meaning the Source facts have not
        # uniquely established, so strict validation must reject it later.
        if projected_iteration_binding not in iteration_bindings:
            iteration_bindings.append(projected_iteration_binding)
        domain_by_dimension[dimension] = domain_ref
        tile_dimensions.add(dimension)

    # ``iteration_bindings`` is a Source-owned schedule projection, not a
    # free-form provider witness.  The earlier admission path checked only
    # that a row named a C slot whose role looked like a tile role.  That left
    # three independent claims unchecked: the C slot's Source coordinate,
    # the J domain selected for that coordinate, and whether the row was an
    # unsupported recurrence/extra-domain relation.  A later schema/type
    # check cannot repair this without making the provider-authored row a
    # second domain authority.
    #
    # Require the complete exact Source projection here.  The expected rows
    # are derived solely from ``schedule_tile_coordinates`` and the
    # Source-derived coordinate->dimension/domain maps above.  No row is
    # accepted by shape/rank/value coincidence, and ``mode_slot`` remains
    # null because the mounted host contract does not prove a mode carrier.
    expected_iteration_rows: dict[str, dict[str, Any]] = {}
    for slot_name, coordinate in tile_coordinates.items():
        choices = coordinate_dimensions.get(coordinate, set())
        if len(choices) != 1:
            return unavailable(
                "iteration_binding_not_source_proven",
                f"slot={slot_name}",
                f"logical_dimension_count={len(choices)}",
                path="pi.iteration_bindings",
            )
        dimension = next(iter(choices))
        domain_ref = domain_by_dimension.get(dimension)
        if not isinstance(domain_ref, str):
            return unavailable(
                "iteration_binding_not_source_proven",
                f"slot={slot_name}",
                "domain_ref=unavailable",
                path="pi.iteration_bindings",
            )
        expected_iteration_rows[slot_name] = {
            "relation": "tiled_domain",
            "logical_axes": [
                {"subject_ref": domain_ref, "axis_ref": dimension}
            ],
            "tile_slot": slot_name,
            "mode_slot": None,
        }

    seen_iteration_slots: set[str] = set()
    for row_index, row in enumerate(iteration_bindings):
        if not isinstance(row, Mapping) or set(row) != {
            "relation", "logical_axes", "tile_slot", "mode_slot"
        }:
            return unavailable(
                "iteration_binding_not_source_proven",
                f"row_index={row_index}",
                path="pi.iteration_bindings",
            )
        slot_name = row.get("tile_slot")
        expected = expected_iteration_rows.get(slot_name)
        if (
            not isinstance(slot_name, str)
            or expected is None
            or slot_name in seen_iteration_slots
            or dict(row) != expected
        ):
            return unavailable(
                "iteration_binding_not_source_proven",
                f"row_index={row_index}",
                f"slot={slot_name if isinstance(slot_name, str) else 'invalid'}",
                path="pi.iteration_bindings",
            )
        seen_iteration_slots.add(slot_name)
    if seen_iteration_slots != set(expected_iteration_rows):
        return unavailable(
            "iteration_binding_not_source_proven",
            f"expected={len(expected_iteration_rows)}",
            f"observed={len(seen_iteration_slots)}",
            path="pi.iteration_bindings",
        )
    result["applied"].append(
        {
            "rule_code": "source_program_domain_projection",
            "path": (
                "j.boundary.domains+pi.program_bindings+pi.iteration_bindings"
                if use_program_bindings and tile_coordinates
                else "j.boundary.domains+pi.program_axes+pi.iteration_bindings"
                if tile_coordinates
                else "j.boundary.domains+pi.program_bindings"
                if use_program_bindings
                else "j.boundary.domains+pi.program_axes"
            ),
            "action": (
                "materialize uniquely Source-bound grid shape coordinates or "
                "closed product grid factors plus Source-proven tile coordinates "
                "as canonical J domains, launch-axis rows, and iteration rows"
            ),
        }
    )
    return result


def bind_source_pi(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    source_runtime_parameter_uses: object = None,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> dict[str, Any]:
    """Bind uniquely witnessed Source address facts to the compact pi subwire.

    This is the deterministic Source -> pi producer for the active canonical
    path.  It operates only on an already decoded-shaped ``(J,C,pi)`` packet;
    it never repairs the root envelope and never writes J or C.  A row is added
    only when the compiled address forms, an existing J tensor/access, and an
    existing program-axis witness identify exactly one access and one tensor
    axis.  Any conflict is returned as a reject-only diagnostic and the input
    packet is left unchanged.

    ``ledger`` is the shared Source transfer ledger produced for this exact
    candidate.  When supplied, it is authoritative: this binder does not
    independently rematch a transfer by tensor/kind and thereby create a
    second occurrence relation.  The result is intentionally a plain evidence object so Responses Lift,
    Codex Lift, and the runner can carry the same minimal binding semantics
    while choosing their own transport.  ``artifact`` is the only candidate
    payload; the Source observation is never embedded in it.
    """

    original = copy.deepcopy(dict(packet))
    result: dict[str, Any] = {
        "protocol": "intentir.lift-source-pi-binding.v1",
        "status": "pass",
        "artifact": original,
        "applied": [],
        "diagnostics": [],
    }

    def reject(rule_code: str, path: str, *observed: str) -> dict[str, Any]:
        safe_observed = [
            value
            for value in observed
            if isinstance(value, str)
            and value
            and len(value) <= 128
            and re.fullmatch(r"[A-Za-z0-9_.:=,+\-]+", value) is not None
        ]
        result["status"] = "unavailable"
        result["diagnostics"] = [
            {
                "rule_code": rule_code,
                "path": path,
                "observed_refs_or_types": safe_observed or ["closed=0"],
            }
        ]
        return result

    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared_ledger.status != "pass":
        result["status"] = "unavailable"
        result["diagnostics"] = [dict(item) for item in shared_ledger.diagnostics] or [
            {
                "rule_code": "source_transfer_ledger_unavailable",
                "path": "<source-transfer-ledger>",
                "observed_refs_or_types": [shared_ledger.reason],
            }
        ]
        return result
    # A Source occurrence ledger becomes a binding authority only after the
    # effect-order reconciler has closed its one-to-one occurrence -> J access
    # rows.  Do not silently continue with an occurrence-only ledger: that
    # would let this pi binder independently rediscover ownership from tensor
    # and kind and reintroduce a second transfer matcher.
    if shared_ledger.occurrences and len(shared_ledger.bindings) != len(
        shared_ledger.occurrences
    ):
        result["status"] = "unavailable"
        result["diagnostics"] = [
            {
                "rule_code": "source_transfer_ledger_bindings_required",
                "path": "<source-transfer-ledger>.bindings",
                "observed_refs_or_types": [
                    f"occurrences={len(shared_ledger.occurrences)}",
                    f"bindings={len(shared_ledger.bindings)}",
                ],
            }
        ]
        return result
    j = original.get("j")
    pi = original.get("pi")
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        result["status"] = "unavailable"
        return result
    if pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        # The legacy materialized pi path has its own frontend-specific
        # BindSource implementation.  Do not reinterpret it with compact-pi
        # rules merely because the Source observer is available.
        result["status"] = "unavailable"
        return result

    ledger_by_ordinal: dict[int, Mapping[str, Any]] = {
        row["ordinal"]: row
        for row in shared_ledger.bindings
        if isinstance(row, Mapping)
        and is_canonical_ordinal(row.get("ordinal"))
        and isinstance(row.get("access_ref"), str)
    }

    signature = j.get("signature")
    raw_parameters = (
        signature.get("parameters")
        if isinstance(signature, Mapping)
        else None
    )
    boundary = j.get("boundary")
    if not isinstance(boundary, Mapping):
        return ()
    raw_accesses = (
        boundary.get("accesses")
        if isinstance(boundary, Mapping)
        else None
    )
    if not isinstance(raw_parameters, list) or not isinstance(raw_accesses, list):
        result["status"] = "unavailable"
        return result

    parameter_projection = _candidate_parameter_projections(
        raw_parameters,
        source_facts=source_facts,
    )
    if parameter_projection is None:
        return reject(
            "source_runtime_parameter_projection_unavailable",
            "j.signature.parameters",
            "projection=runtime_physical_abi",
        )
    runtime_parameters, parameters_by_position, runtime_position_by_name = (
        parameter_projection
    )

    accesses = [
        item
        for item in raw_accesses
        if isinstance(item, Mapping)
        and isinstance(item.get("access_id"), str)
        and isinstance(item.get("tensor"), str)
        and isinstance(item.get("kind"), str)
    ]
    access_by_id = {
        str(item["access_id"]): item for item in accesses
    }
    domain_coordinates = _source_domain_coordinates(
        boundary if isinstance(boundary, Mapping) else {}
    )
    domain_extent_symbols = _source_domain_extent_symbols(
        boundary if isinstance(boundary, Mapping) else {}
    )
    program_axis_map = _candidate_program_axis_map(pi)

    program_evidence: dict[int, set[tuple[str, int]]] = {}
    access_evidence: dict[tuple[str, str], set[int]] = {}
    source_used_carriers = {
        item
        for item in source_runtime_parameter_uses
        if isinstance(item, str) and re.fullmatch(r"p[0-9]+", item)
    } if isinstance(source_runtime_parameter_uses, list) else set()
    source_occurrences_by_ordinal = {
        occurrence.ordinal: occurrence
        for occurrence in shared_ledger.occurrences
    }
    if len(source_occurrences_by_ordinal) != len(shared_ledger.occurrences):
        return reject(
            "source_transfer_occurrence_identity_unavailable",
            "<source-transfer-ledger>.occurrences",
            "expected=unique_explicit_ordinals",
        )
    for occurrence in shared_ledger.occurrences:
        ordinal = occurrence.ordinal
        operation = occurrence.operation
        pointer_base = occurrence.pointer_base
        polynomial = occurrence.address_terms
        physical_axes = _source_program_axes(polynomial)
        if len(physical_axes) != 1:
            continue
        physical_axis = physical_axes[0]
        base_position = _source_pointer_base_position(pointer_base)
        base_parameter = runtime_parameters.get(base_position)
        if not isinstance(base_parameter, Mapping):
            continue
        tensor_ref = base_parameter.get("logical_ref")
        memory_kind = _source_memory_kind(operation)
        source_tensor_ref = occurrence.logical_ref
        if (
            not isinstance(tensor_ref, str)
            or memory_kind is None
            or (
                source_tensor_ref is not None
                and tensor_ref != source_tensor_ref
            )
        ):
            continue
        ledger_row = ledger_by_ordinal.get(ordinal)
        access_ref = (
            ledger_row.get("access_ref")
            if isinstance(ledger_row, Mapping)
            else None
        )
        access = access_by_id.get(access_ref) if isinstance(access_ref, str) else None
        if (
            not isinstance(ledger_row, Mapping)
            or not isinstance(access, Mapping)
            or access.get("tensor") != tensor_ref
            or access.get("kind") != memory_kind
            or bool(access.get("predicate") is not None)
            != bool(ledger_row.get("masked"))
        ):
            continue
        if not isinstance(access, Mapping) or not isinstance(access_ref, str):
            continue

        independent_coordinate_axes: set[tuple[str, int]] = set()
        # Only domains attached to this Source-proven physical launch axis
        # may establish the J coordinate for this occurrence.  Iterating over
        # every candidate domain allowed an extra provider-authored domain
        # (for example a second i* spelling of the same x* extent) to win the
        # "independent" coordinate branch before the host mapping was
        # consulted.  The materializer has already proved the launch map; use
        # exactly its domain set here.
        expected_domain_refs = {
            domain_ref
            for domain_ref, _axis_ref in program_axis_map.get(physical_axis, ())
        }
        for domain_ref, coordinate in domain_coordinates.items():
            if domain_ref not in expected_domain_refs:
                continue
            axis = _index_axis_for_coordinate(
                access.get("index_expression"),
                coordinate,
                extent_symbol=domain_extent_symbols.get(domain_ref),
            )
            if axis is not None:
                independent_coordinate_axes.add((domain_ref, axis))

        # Prefer Source + J evidence that does not depend on the candidate's
        # own pi row.  An authored row may be wrong; using it as the only
        # source of its expected axis would make the binder circular.
        if len(independent_coordinate_axes) == 1:
            domain_ref, logical_axis = next(iter(independent_coordinate_axes))
            program_evidence.setdefault(physical_axis, set()).add(
                (domain_ref, logical_axis)
            )
        else:
            candidate_axes = program_axis_map.get(physical_axis, set())
            if not candidate_axes:
                # Without an authored program mapping, a rank>1 or mixed-
                # radix access remains ambiguous and stays provider-authored.
                continue
            logical_axes: set[int] = set()
            for domain_ref, authored_axis in candidate_axes:
                if authored_axis is not None:
                    logical_axes.add(authored_axis)
                    program_evidence.setdefault(physical_axis, set()).add(
                        (domain_ref, authored_axis)
                    )
                    continue
                coordinate = domain_coordinates.get(domain_ref)
                axis = _index_axis_for_coordinate(
                    access.get("index_expression"),
                    coordinate,
                    extent_symbol=domain_extent_symbols.get(domain_ref),
                ) if coordinate is not None else None
                if axis is not None:
                    logical_axes.add(axis)
                    program_evidence.setdefault(physical_axis, set()).add(
                        (domain_ref, axis)
                    )
            if len(logical_axes) != 1:
                continue
            logical_axis = next(iter(logical_axes))

        carrier_program_pairs = set(
            _source_address_carrier_program_axis_pairs(polynomial)
        )
        for position in _source_argument_positions(polynomial):
            if (
                position == base_position
                or (position, physical_axis) not in carrier_program_pairs
            ):
                continue
            carrier = runtime_parameters.get(position)
            if not isinstance(carrier, Mapping):
                continue
            carrier_name = carrier.get("name")
            if (
                not isinstance(carrier_name, str)
                or carrier.get("kind") != "scalar"
                or carrier.get("logical_ref") is not None
            ):
                continue
            access_evidence.setdefault(
                (carrier_name, access_ref), set()
            ).add(logical_axis)

    # Specialization can fold a unit stride out of the final address
    # polynomial.  The Source occurrence still has one exact ABI pairing:
    # a Source-used scalar p(N) immediately follows the tensor p(N-1) whose
    # sole transfer uses the same direct program coordinate.  Recover only
    # this complete, one-to-one form.  This is not p* positional inference:
    # every precondition below is required so an unrelated scalar or a tensor
    # with multiple transfer occurrences remains provider-authored.
    if source_used_carriers:
        for carrier_name in sorted(source_used_carriers):
            carrier_match = re.fullmatch(r"p([0-9]+)", carrier_name)
            if carrier_match is None:
                continue
            if any(key[0] == carrier_name for key in access_evidence):
                continue
            carrier_runtime_position = runtime_position_by_name.get(carrier_name)
            if carrier_runtime_position is None:
                continue
            carrier_parameter = runtime_parameters.get(carrier_runtime_position)
            if (
                not isinstance(carrier_parameter, Mapping)
                or carrier_parameter.get("kind") != "scalar"
                or carrier_parameter.get("logical_ref") is not None
            ):
                # The unit-stride fallback is only for an ABI scalar whose
                # physical slot follows its tensor pointer.  Consecutive
                # tensor arguments are not implicit strides, even when both
                # names occur in the Source syntax.
                continue
            base_parameter = runtime_parameters.get(carrier_runtime_position - 1)
            if (
                not isinstance(base_parameter, Mapping)
                or base_parameter.get("kind") not in {"tensor", "pointer"}
                or not isinstance((tensor_ref := base_parameter.get("logical_ref")), str)
            ):
                continue
            # This fallback is valid only when the shared occurrence ledger
            # proves the exact Source ordinal/access pair.  The previous
            # implementation reused ``access``/``access_ref`` from the
            # preceding loop above, so the result depended on the last
            # transfer visited and could bind an unrelated carrier.  Resolve
            # the access exclusively through this occurrence's ledger row;
            # never rematch Source forms by tensor/kind or by stale loop state.
            matching_rows: list[tuple[TransferOccurrence, str, Mapping[str, Any]]] = []
            for occurrence in shared_ledger.occurrences:
                if (
                    occurrence.pointer_position != carrier_runtime_position - 1
                    or occurrence.logical_ref != tensor_ref
                    or len(_source_program_axes(occurrence.address_terms)) != 1
                ):
                    continue
                occurrence_bindings = [
                    row
                    for row in shared_ledger.bindings
                    if row.get("ordinal") == occurrence.ordinal
                ]
                if len(occurrence_bindings) != 1:
                    continue
                occurrence_access_ref = occurrence_bindings[0].get("access_ref")
                occurrence_access = (
                    access_by_id.get(occurrence_access_ref)
                    if isinstance(occurrence_access_ref, str)
                    else None
                )
                if (
                    not isinstance(occurrence_access_ref, str)
                    or not isinstance(occurrence_access, Mapping)
                    or occurrence_access.get("tensor") != tensor_ref
                    or occurrence_access.get("kind") != occurrence.kind
                    or (occurrence_access.get("predicate") is not None)
                    != occurrence.masked
                ):
                    continue
                matching_rows.append(
                    (occurrence, occurrence_access_ref, occurrence_access)
                )
            if len(matching_rows) != 1:
                continue
            occurrence, access_ref, access = matching_rows[0]
            coordinates: set[tuple[str, int]] = set()
            for domain_ref, coordinate in domain_coordinates.items():
                axis = _index_axis_for_coordinate(
                    access.get("index_expression"),
                    coordinate,
                    extent_symbol=domain_extent_symbols.get(domain_ref),
                )
                if axis is not None:
                    coordinates.add((domain_ref, axis))
            if len(coordinates) != 1:
                continue
            _domain_ref, axis = next(iter(coordinates))
            access_evidence[(carrier_name, access_ref)] = {axis}

    # Close direct address-carrier ownership only after the same Source
    # occurrence ledger has established its program-coordinate evidence.  The
    # earlier pass intentionally tries to prove an axis from each individual
    # access, but a rank-two J index such as ``[d0, d1]`` has two independent
    # coordinates even when the compiled pointer polynomial proves that a
    # scalar carrier is multiplied by the one direct program axis.  Treating
    # that shape as an ambiguity left a Source-used ``p*`` unowned and made
    # SourceReconcile reject an otherwise canonical candidate.
    #
    # This is not a second transfer matcher: every row below starts from the
    # already closed ordinal -> access binding, verifies the exact pointer
    # base and scalar occurrence in that ordinal's address polynomial, and
    # reads the program axis from the same ledger occurrence.  It therefore
    # cannot bind a carrier to an access selected merely by tensor/kind/order.
    def ledger_program_axis(
        occurrence: TransferOccurrence,
        access: Mapping[str, Any],
    ) -> int | None:
        physical_axes = _source_program_axes(occurrence.address_terms)
        if len(physical_axes) != 1:
            return None
        physical_axis = physical_axes[0]
        witnessed = program_evidence.get(physical_axis, set())
        if len(witnessed) == 1:
            _domain_ref, logical_axis = next(iter(witnessed))
            return logical_axis
        authored = program_axis_map.get(physical_axis, set())
        axes: set[int] = set()
        for domain_ref, axis_ref in authored:
            if axis_ref is not None:
                axes.add(axis_ref)
                continue
            coordinate = domain_coordinates.get(domain_ref)
            if coordinate is None:
                continue
            resolved = _index_axis_for_coordinate(
                access.get("index_expression"),
                coordinate,
                extent_symbol=domain_extent_symbols.get(domain_ref),
            )
            if resolved is not None:
                axes.add(resolved)
        return next(iter(axes)) if len(axes) == 1 else None

    for occurrence in shared_ledger.occurrences:
        ledger_row = ledger_by_ordinal.get(occurrence.ordinal)
        access_ref = (
            ledger_row.get("access_ref")
            if isinstance(ledger_row, Mapping)
            else None
        )
        access = access_by_id.get(access_ref) if isinstance(access_ref, str) else None
        if not isinstance(access_ref, str) or not isinstance(access, Mapping):
            continue
        base_position = occurrence.pointer_position
        base_parameter = runtime_parameters.get(base_position)
        if (
            not isinstance(base_position, int)
            or not isinstance(base_parameter, Mapping)
            or not isinstance(base_parameter.get("logical_ref"), str)
            or access.get("tensor") != base_parameter.get("logical_ref")
            or access.get("kind") != occurrence.kind
            or bool(access.get("predicate") is not None) != occurrence.masked
        ):
            continue
        logical_axis = ledger_program_axis(occurrence, access)
        if logical_axis is None:
            continue
        physical_axes = _source_program_axes(occurrence.address_terms)
        carrier_program_pairs = set(
            _source_address_carrier_program_axis_pairs(
                occurrence.address_terms
            )
        )
        for position in _source_argument_positions(occurrence.address_terms):
            if (
                position == base_position
                or len(physical_axes) != 1
                or (position, physical_axes[0]) not in carrier_program_pairs
            ):
                continue
            carrier = runtime_parameters.get(position)
            carrier_name = carrier.get("name") if isinstance(carrier, Mapping) else None
            if (
                not isinstance(carrier_name, str)
                or carrier.get("kind") != "scalar"
                or carrier.get("logical_ref") is not None
            ):
                continue
            access_evidence.setdefault((carrier_name, access_ref), set()).add(
                logical_axis
            )

    diagnostics: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    bound = dict(pi) if isinstance(pi, Mapping) else pi
    if not isinstance(bound, dict):
        result["status"] = "unavailable"
        return result

    raw_program_bindings = bound.get("program_bindings")
    raw_program_axes = bound.get("program_axes")
    if not isinstance(raw_program_bindings, list) or not isinstance(
        raw_program_axes, list
    ):
        result["status"] = "unavailable"
        return result
    if raw_program_bindings and raw_program_axes:
        result["status"] = "unavailable"
        return result

    def existing_program_fact(
        physical_axis: int,
    ) -> set[tuple[str, int]]:
        facts: set[tuple[str, int]] = set()
        if raw_program_bindings:
            for row in raw_program_bindings:
                if not isinstance(row, Mapping) or row.get("axis") != physical_axis:
                    continue
                for logical_axis in row.get("logical_axes", []):
                    if not isinstance(logical_axis, Mapping):
                        continue
                    domain_ref = logical_axis.get("domain_ref")
                    axis = _axis_ref_number(logical_axis.get("axis_ref"))
                    if isinstance(domain_ref, str) and axis is not None:
                        facts.add((domain_ref, axis))
        else:
            for row in raw_program_axes:
                if not isinstance(row, Mapping) or row.get("axis") != physical_axis:
                    continue
                domain_ref = row.get("domain_ref")
                coordinate = domain_coordinates.get(domain_ref)
                if isinstance(domain_ref, str) and coordinate is not None:
                    facts.add((domain_ref, coordinate))
        return facts

    for physical_axis in sorted(program_evidence):
        facts = program_evidence[physical_axis]
        if len(facts) != 1:
            continue
        expected = next(iter(facts))
        existing = existing_program_fact(physical_axis)
        if existing and expected not in existing:
            diagnostics.append(
                {
                    "rule_code": "source_pi_program_binding_conflict",
                    "path": "pi.program_bindings" if raw_program_bindings else "pi.program_axes",
                    "anonymous_handles": [expected[0]],
                    "observed_refs_or_types": [
                        f"source physical axis={physical_axis}",
                        f"source-bound J fact={expected[0]}:{expected[1]}",
                        f"authored facts={sorted(existing)}",
                    ],
                }
            )
            continue
        if existing:
            continue
        domain_ref, axis = expected
        if raw_program_bindings:
            raw_program_bindings.append(
                {
                    "axis": physical_axis,
                    "relation": "direct",
                    "logical_axes": [
                        {"domain_ref": domain_ref, "axis_ref": axis}
                    ],
                }
            )
            applied.append(
                {
                    "family": "program_bindings",
                    "physical_axis": physical_axis,
                    "domain_ref": domain_ref,
                    "axis_ref": axis,
                    "relation": "direct",
                }
            )
        elif not raw_program_axes and not raw_program_bindings:
            # Prefer the relation form for newly produced rows.  It carries
            # the same direct fact and is consumed by Replay projection; the
            # later consumer-unification step handles both legacy forms.
            raw_program_bindings.append(
                {
                    "axis": physical_axis,
                    "relation": "direct",
                    "logical_axes": [
                        {"domain_ref": domain_ref, "axis_ref": axis}
                    ],
                }
            )
            applied.append(
                {
                    "family": "program_bindings",
                    "physical_axis": physical_axis,
                    "domain_ref": domain_ref,
                    "axis_ref": axis,
                    "relation": "direct",
                }
            )
        else:
            raw_program_axes.append(
                {"domain_ref": domain_ref, "axis": physical_axis}
            )
            applied.append(
                {
                    "family": "program_axes",
                    "physical_axis": physical_axis,
                    "domain_ref": domain_ref,
                    "axis_ref": axis,
                }
            )

    expected_rows_by_carrier: dict[str, set[tuple[str, int]]] = {}
    for (carrier, access_ref), axes in access_evidence.items():
        if len(axes) == 1:
            expected_rows_by_carrier.setdefault(carrier, set()).add(
                (access_ref, next(iter(axes)))
            )

    raw_access_bindings = bound.get("access_representation_bindings")
    if not isinstance(raw_access_bindings, list):
        result["status"] = "unavailable"
        return result

    # When exact Source syntax says which runtime carriers are used, erase
    # only impossible authored ownership rows before rebuilding the uniquely
    # observed rows below.  This prevents a tensor base p*, an unused scalar,
    # or a conflicting guessed access from competing with the Source ledger.
    # A used carrier with no unique observed occurrence is retained: fixed
    # code must not guess a route in that case.
    if source_used_carriers:
        cleaned_rows: list[Any] = []
        removed_rows = 0
        for row in raw_access_bindings:
            if not isinstance(row, Mapping):
                cleaned_rows.append(row)
                continue
            carrier = row.get("carrier_ref")
            access_ref = row.get("access_ref")
            axis = _axis_ref_number(row.get("axis_ref"))
            parameter_match = (
                next(
                    (
                        parameter
                        for parameter in parameters_by_position.values()
                        if parameter.get("name") == carrier
                    ),
                    None,
                )
                if isinstance(carrier, str)
                else None
            )
            legal_carrier = (
                isinstance(parameter_match, Mapping)
                and parameter_match.get("kind") == "scalar"
                and parameter_match.get("logical_ref") is None
                and carrier in source_used_carriers
            )
            expected_rows = (
                expected_rows_by_carrier.get(carrier, set())
                if isinstance(carrier, str)
                else set()
            )
            exact_row = (
                isinstance(access_ref, str)
                and isinstance(axis, int)
                and (access_ref, axis) in expected_rows
            )
            if not legal_carrier or (expected_rows and not exact_row):
                removed_rows += 1
                continue
            cleaned_rows.append(row)
        if removed_rows:
            raw_access_bindings[:] = cleaned_rows
            applied.append(
                {
                    "family": "access_representation_bindings",
                    "action": (
                        "remove impossible or Source-conflicting authored carrier rows "
                        "before reconstructing unique Source transfer ownership"
                    ),
                }
            )

    existing_access_rows = {
        (
            str(row.get("carrier_ref")),
            str(row.get("access_ref")),
            _axis_ref_number(row.get("axis_ref")),
        )
        for row in bound.get("access_representation_bindings", [])
        if isinstance(row, Mapping)
    }
    existing_access_pair_axes: dict[tuple[str, str], set[int]] = {}
    for carrier, access_ref, axis in existing_access_rows:
        if axis is not None:
            existing_access_pair_axes.setdefault((carrier, access_ref), set()).add(axis)

    for (carrier, access_ref), axes in sorted(access_evidence.items()):
        if len(axes) != 1 or access_ref not in access_by_id:
            continue
        axis = next(iter(axes))
        existing_axes = existing_access_pair_axes.get((carrier, access_ref), set())
        if existing_axes and axis not in existing_axes:
            diagnostics.append(
                {
                    "rule_code": "source_pi_access_binding_conflict",
                    "path": "pi.access_representation_bindings",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        f"source tensor axis={axis}",
                        f"authored axes={sorted(existing_axes)}",
                    ],
                }
            )
            continue
        row_key = (carrier, access_ref, axis)
        if row_key in existing_access_rows:
            continue
        raw_access_bindings.append(
            {
                "relation": "address_recomputed",
                "access_ref": access_ref,
                "carrier_ref": carrier,
                "axis_ref": axis,
            }
        )
        existing_access_rows.add(row_key)
        applied.append(
            {
                "family": "access_representation_bindings",
                "relation": "address_recomputed",
                "access_ref": access_ref,
                "carrier_ref": carrier,
                "axis_ref": axis,
            }
        )

    if diagnostics:
        result["status"] = "fail"
        result["diagnostics"] = diagnostics
        return result
    if applied:
        original["pi"] = bound
        result["artifact"] = original
    result["applied"] = applied
    return result


def reconcile_candidate_source_ttir(
    candidate_text: str,
    observation: Mapping[str, Any] | None,
    *,
    construction: CanonicalConstruction,
) -> dict[str, Any]:
    """Diagnose candidate-authored obligations against Source syntax and TTIR.

    This runs only inside source-aware Lift.  It neither constructs artifact
    fields nor contributes to TypeCheck, Gate1, or certification.
    """

    base: dict[str, Any] = {
        "protocol": SOURCE_TTIR_RECONCILE_PROTOCOL,
        "authority": (
            "source-aware diagnostic comparison plus deterministic Source-to-pi binding "
            "of model-authored closed J/compact-pi obligations with the existing anonymous "
            "Source TTIR observation; J/C are never constructed or changed, and no Gate "
            "or certification authority is granted"
        ),
    }
    if not isinstance(observation, Mapping) or observation.get("status") != "pass":
        return {
            **base,
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    ttir = observation.get("ttir")
    if not isinstance(ttir, str) or not ttir:
        return {
            **base,
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    try:
        packet = json.loads(candidate_text)
        if not isinstance(packet, Mapping):
            raise TypeError("candidate root is not an object")
        python_interface = observation.get("python_interface")
        fingerprint = fingerprint_ttir(
            ttir,
            external_math=observation.get("_external_math"),
            python_interface=(
                python_interface if isinstance(python_interface, Mapping) else None
            ),
        )
        source_facts = None
        raw_source_facts = observation.get("source_facts")
        if isinstance(raw_source_facts, Mapping):
            source_facts = SourceFacts.from_json(raw_source_facts)
        if source_facts is None:
            raise ValueError("Source observation lacks canonical SourceFacts lineage")
        if source_facts.digest != construction.source_facts.digest:
            raise ValueError("Source observation facts differ from canonical construction")
        construction_ledger = construction.with_observation(fingerprint).transfer_ledger
        # Build one occurrence ledger for this candidate.  Fill validation,
        # pi binding, and effect-order diagnostics all consume this exact
        # object; no later helper rematches a Source transfer independently.
        transfer_ledger = reorder_boundary_effects_from_source_ledger(
            packet,
            fingerprint,
            ledger=construction_ledger,
            source_facts=construction.source_facts,
        )
        reorder_diagnostics = tuple(
            item
            for item in transfer_ledger.get("diagnostics", ())
            if isinstance(item, Mapping)
        ) if isinstance(transfer_ledger, Mapping) else ()
        # SourceReconcile is an independent admission consumer.  It must not
        # rely on canonical construction having already validated a candidate
        # supplied program mapping.  Project the host-owned mapping *before*
        # Source->pi binding: bind_source_pi reads pi.program_axes and
        # pi.program_bindings to resolve a carrier axis, so calling it first
        # would let an unproved candidate row influence a later authorization
        # decision.  This order is part of the independence boundary, not a
        # performance detail.
        domain_projection = materialize_source_program_domains(
            packet,
            source_facts=construction.source_facts,
            ledger=transfer_ledger,
        )
        domain_diagnostics: tuple[Mapping[str, Any], ...] = ()
        if (
            not isinstance(domain_projection, Mapping)
            or domain_projection.get("status") != "pass"
        ):
            reason = (
                domain_projection.get("reason")
                if isinstance(domain_projection, Mapping)
                else None
            )
            if not isinstance(reason, str) or not re.fullmatch(
                r"[a-z0-9_]{1,80}", reason
            ):
                reason = "source_program_domain_projection_unavailable"
            domain_diagnostics = (
                {
                    "rule_code": f"source_program_domain_{reason}",
                    "path": "j.boundary.domains + pi.program_axes + pi.program_bindings",
                    "observed_refs_or_types": [
                        "expected:Source-proven symbolic launch-domain mapping"
                    ],
                },
            )
        else:
            projected_packet = domain_projection.get("artifact")
            if not isinstance(projected_packet, Mapping):
                domain_diagnostics = (
                    {
                        "rule_code": "source_program_domain_projection_unavailable",
                        "path": "j.boundary.domains + pi.program_axes + pi.program_bindings",
                        "observed_refs_or_types": [
                            "expected:Source-proven symbolic launch-domain mapping"
                        ],
                    },
                )
                projected_packet = packet

        # Do not call the π binder after an unavailable domain projection: it
        # must never fall back to candidate-authored program rows.  A failed
        # projection remains a reject-only SourceReconcile diagnostic.
        if domain_diagnostics:
            binding = {
                "status": "unavailable",
                "artifact": copy.deepcopy(dict(packet)),
                "diagnostics": (),
            }
            bound_packet = packet
        else:
            binding = bind_source_pi(
                projected_packet,
                fingerprint,
                source_runtime_parameter_uses=observation.get(
                    "source_runtime_parameter_uses"
                ),
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            )
            bound_packet = binding.get("artifact", projected_packet)
            if not isinstance(bound_packet, Mapping):
                bound_packet = projected_packet

        binding_diagnostics = tuple(
            item
            for item in binding.get("diagnostics", [])
            if isinstance(item, Mapping)
        )
        if reorder_diagnostics:
            binding_diagnostics = (*reorder_diagnostics, *binding_diagnostics)
        diagnostics = compiled_packet_obligation_diagnostics(
            bound_packet,
            ttir,
            fingerprint=fingerprint,
            path_prefix="<source-ttir-obligation>",
        )
        diagnostics = (
            *diagnostics,
            *_compact_j_mask_presence_diagnostics(bound_packet, fingerprint),
            *_source_access_fill_diagnostics(
                bound_packet,
                fingerprint,
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            ),
            *_compact_j_recurrence_presence_diagnostics(bound_packet, fingerprint),
            *_compact_j_terminal_control_presence_diagnostics(bound_packet, fingerprint),
            *_source_boundary_effect_order_diagnostics(
                bound_packet,
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            ),
            *validate_source_semantic_authorizations(
                bound_packet,
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            ),
            *validate_source_store_output_bindings(
                bound_packet,
                fingerprint,
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            ),
            *domain_diagnostics,
            *_source_stride_axis_diagnostics(
                bound_packet,
                ledger=transfer_ledger,
                source_facts=construction.source_facts,
            ),
            *_unowned_source_runtime_carrier_diagnostics(
                bound_packet,
                ttir,
                source_runtime_parameter_uses=observation.get(
                    "source_runtime_parameter_uses"
                ),
            ),
            *binding_diagnostics,
        )
    except Exception as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        failing_function = frames[-1].name if frames else "<unknown>"
        return {
            **base,
            "status": "fail",
            "applicable": True,
            "diagnostics": [
                {
                    "rule_code": "source_reconcile_observer_failed",
                    "path": "<source-ttir-obligation>",
                    "anonymous_handles": [],
                    "observed_refs_or_types": [
                        _closed_observer_exception(exc),
                        f"observer_function={failing_function}",
                    ],
                    "line": None,
                }
            ],
        }
    result = {
        **base,
        "status": "fail" if diagnostics else "pass",
        "applicable": True,
        "diagnostics": [dict(item) for item in diagnostics],
    }
    if isinstance(transfer_ledger, Mapping):
        shared_ledger = transfer_ledger.get("source_transfer_ledger")
        if isinstance(shared_ledger, Mapping):
            result["source_transfer_ledger"] = dict(shared_ledger)
    if binding.get("applied"):
        result["source_pi_binding"] = {
            "status": binding.get("status", "pass"),
            "applied": [
                dict(item)
                for item in binding["applied"]
                if isinstance(item, Mapping)
            ],
        }
    elif binding.get("status") == "fail":
        result["source_pi_binding"] = {
            "status": "fail",
            "applied": [],
        }
    return result


def _source_boundary_effect_order_diagnostics(
    packet: Mapping[str, Any],
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Reject a J boundary ledger whose transfer order contradicts Source.

    ``boundary.effects`` is the ordered observable memory ledger.  The old
    source-aware preflight checked transfer presence and several one-bit
    obligations, but it did not compare this ordering when the same source
    tensors had multiple accesses.  That allowed an artifact to pass Lift and
    TypeCheck even though Replay was required to emit stores in an order that
    the selected Source never uses.

    The SourceTransferLedger is the only occurrence-to-access authority.  This
    diagnostic never falls back to TTIR origin scanning: without a complete
    ledger binding it is unavailable and the formal Gate1 observer remains
    authoritative.  It never constructs or repairs J/C/pi and it does not
    compare arithmetic or address spelling.
    """

    j = packet.get("j")
    if not isinstance(j, Mapping):
        return ()
    boundary = j.get("boundary")
    raw_effects = (
        boundary.get("effects") if isinstance(boundary, Mapping) else None
    )
    if not isinstance(raw_effects, list):
        return ()
    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared_ledger.status != "pass" or not shared_ledger.bindings:
        return ()
    source_access_order = [
        row.get("access_ref") for row in shared_ledger.bindings
    ]
    authored_access_order = [
        row.get("access_ref")
        for row in raw_effects
        if isinstance(row, Mapping) and isinstance(row.get("access_ref"), str)
    ]
    if (
        not all(isinstance(item, str) for item in source_access_order)
        or len(authored_access_order) != len(source_access_order)
    ):
        return ()
    first_difference = next(
        (
            index
            for index, (expected_access, observed_access) in enumerate(
                zip(authored_access_order, source_access_order, strict=True)
            )
            if expected_access != observed_access
        ),
        None,
    )
    if first_difference is None:
        return ()
    expected_access = authored_access_order[first_difference]
    observed_access = source_access_order[first_difference]
    return (
        {
            "rule_code": "source_boundary_effect_order",
            "path": (
                "<source-ttir-obligation> j.boundary.effects: "
                f"access {expected_access!r} is at ordinal {first_difference}, "
                "but the shared Source transfer ledger resolves a different "
                "logical occurrence"
            ),
            "anonymous_handles": [expected_access, observed_access],
            "observed_refs_or_types": [
                f"expected_access={expected_access}",
                f"observed_access={observed_access}",
                f"ordinal={first_difference}",
            ],
            "source_transfer_order": [
                {"ordinal": index, "access_ref": access_ref}
                for index, access_ref in enumerate(source_access_order)
            ],
            "line": None,
        },
    )


def _split_candidate_index_expression(value: object) -> tuple[str, ...] | None:
    """Split one closed J index expression without interpreting its values."""

    if not isinstance(value, str):
        return None
    expression = value.strip()
    if expression.startswith("[") and expression.endswith("]"):
        expression = expression[1:-1].strip()
    if not expression:
        return None
    pieces: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"[": "]", "(": ")", "{": "}"}
    quote: str | None = None
    escaped = False
    for index, char in enumerate(expression):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == "," and not stack:
            item = expression[start:index].strip()
            if not item:
                return None
            pieces.append(item)
            start = index + 1
    if quote is not None or stack:
        return None
    item = expression[start:].strip()
    if not item:
        return None
    pieces.append(item)
    return tuple(pieces)


def _candidate_index_profile(
    index_expression: object,
) -> tuple[int, int, tuple[int, ...]] | None:
    """Return ``(rank, dynamic_components, nonzero_literals)`` for one J index.

    The profile is deliberately syntactic.  A non-literal component is only
    classified as dynamic; this helper never decides which J value it names.
    """

    components = _split_candidate_index_expression(index_expression)
    if components is None:
        return None
    literals: list[int] = []
    dynamic = 0
    for component in components:
        if re.fullmatch(r"[+-]?[0-9]+", component) is not None:
            literals.append(int(component))
        else:
            dynamic += 1
    return len(components), dynamic, tuple(sorted(item for item in literals if item != 0))


def _candidate_tensor_ranks(packet: Mapping[str, Any]) -> dict[str, int] | None:
    """Collect ranks from the already-authored J tensor declarations."""

    j = packet.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, Mapping):
        return None
    ranks: dict[str, int] = {}
    for field in ("inputs", "outputs"):
        values = function.get(field)
        if not isinstance(values, list):
            return None
        for item in values:
            if not isinstance(item, Mapping):
                return None
            name = item.get("name")
            shape = item.get("shape")
            if not isinstance(name, str) or not isinstance(shape, list):
                return None
            rank = len(shape)
            previous = ranks.get(name)
            if previous is not None and previous != rank:
                return None
            ranks[name] = rank
    return ranks


def _candidate_node_contexts(packet: Mapping[str, Any]) -> dict[str, tuple[str, ...]] | None:
    """Project nested J operation regions to a conservative context class."""

    j = packet.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, Mapping):
        return None
    contexts: dict[str, tuple[str, ...]] = {}

    def visit(body: object, context: tuple[str, ...]) -> bool:
        if not isinstance(body, list):
            return False
        for operation in body:
            if not isinstance(operation, Mapping):
                return False
            node_id = operation.get("id")
            if not isinstance(node_id, str):
                return False
            contexts[node_id] = context
            op_name = operation.get("op")
            if op_name == "loop":
                if not visit(operation.get("body"), (*context, "loop")):
                    return False
            elif op_name in {"effect_if", "value_if"}:
                for polarity in ("positive", "negative"):
                    arm = operation.get(polarity)
                    if not isinstance(arm, Mapping):
                        return False
                    if not visit(
                        arm.get("body"),
                        (*context, f"branch:{polarity}"),
                    ):
                        return False
        return True

    return contexts if visit(function.get("body"), ()) else None


def _source_context_class(context: object) -> tuple[str, ...] | None:
    if not isinstance(context, (list, tuple)):
        return None
    result: list[str] = []
    for token in context:
        if not isinstance(token, str):
            return None
        if token.startswith("scf.for@"):
            result.append("loop")
            continue
        if token.startswith("scf.if@"):
            result.append("branch")
            continue
        if token.startswith("structured_if_arm="):
            polarity = token.partition("=")[2]
            if polarity not in {"positive", "negative"}:
                return None
            result.append(f"branch:{polarity}")
            continue
        if token.startswith("terminal_fork_arm="):
            polarity = token.partition("=")[2]
            if polarity == "true":
                polarity = "positive"
            elif polarity == "false":
                polarity = "negative"
            else:
                return None
            result.append(f"branch:{polarity}")
            continue
        return None
    return tuple(result)


def _source_j_context_compatible(
    source_context: object,
    j_context: object,
) -> bool:
    source_class = _source_context_class(source_context)
    if not isinstance(j_context, tuple):
        return False
    if source_class is None:
        return False
    # ``scf.if`` and its arm marker are both present in the Source context;
    # collapse the owner marker because J's arm context already implies it.
    collapsed_source: list[str] = []
    for token in source_class:
        if token == "branch" and collapsed_source and collapsed_source[-1].startswith(
            "branch:"
        ):
            continue
        collapsed_source.append(token)
    collapsed_j: list[str] = []
    for token in j_context:
        if token == "branch" and collapsed_j and collapsed_j[-1].startswith("branch:"):
            continue
        collapsed_j.append(token)
    return tuple(collapsed_source) == tuple(collapsed_j)


def reorder_boundary_effects_from_source_ledger(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
    source_order_sealed: bool = False,
) -> dict[str, Any]:
    """Repair a uniquely witnessed J effect order from the Source ledger.

    This is a bounded ingress repair, not a semantic synthesizer.  It only
    moves already-authored access-bound effect objects when every Source
    transfer has one unique J access match using ABI ownership, transfer kind,
    mask state, direct address/index shape, and region context.  Ambiguous or
    incomplete candidates are returned unchanged and remain reject-only
    diagnostics for SourceReconcile/Gate1.
    """

    original = copy.deepcopy(dict(packet))
    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    result: dict[str, Any] = {
        "protocol": "intentir.lift-source-effect-order-repair.v1",
        "status": "unavailable",
        "artifact": original,
        "applied": [],
        "diagnostics": [],
        # This remains an ingress-local observation.  It is never inserted
        # into J/C/pi or a persisted replay packet.
        "source_access_matches": (),
        "source_access_match_reason": "source_memory_effect_ledger_unavailable",
        "source_transfer_ledger": shared_ledger.to_json(),
    }

    def reject(
        rule_code: str,
        path: str,
        *observed: str,
        status: str = "unavailable",
    ) -> dict[str, Any]:
        """Return one bounded structural ledger failure without payload text."""

        safe_observed = [
            value
            for value in observed
            if isinstance(value, str)
            and value
            and len(value) <= 128
            and re.fullmatch(r"[A-Za-z0-9_.:=,+\-]+", value) is not None
        ]
        result["status"] = status
        result["source_access_match_reason"] = rule_code
        result["source_access_matches"] = ()
        result["diagnostics"] = [
            {
                "rule_code": rule_code,
                "path": path,
                "observed_refs_or_types": safe_observed or ["closed=0"],
            }
        ]
        return result

    if shared_ledger.status != "pass":
        result["status"] = shared_ledger.status
        result["source_access_match_reason"] = shared_ledger.reason
        result["diagnostics"] = [dict(item) for item in shared_ledger.diagnostics]
        return result

    j = original.get("j")
    if not isinstance(j, Mapping):
        return reject("source_transfer_candidate_j_missing", "j")
    signature = j.get("signature")
    raw_parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    boundary = j.get("boundary")
    raw_accesses = (
        boundary.get("accesses") if isinstance(boundary, Mapping) else None
    )
    raw_effects = (
        boundary.get("effects") if isinstance(boundary, Mapping) else None
    )
    if not isinstance(raw_parameters, list) or not isinstance(raw_accesses, list):
        return reject(
            "source_transfer_boundary_rows_missing",
            "j.boundary.accesses",
            f"parameter_rows={len(raw_parameters) if isinstance(raw_parameters, list) else -1}",
            f"access_rows={len(raw_accesses) if isinstance(raw_accesses, list) else -1}",
        )
    if not isinstance(raw_effects, list):
        return reject(
            "source_transfer_boundary_rows_missing",
            "j.boundary.effects",
            "effect_rows=-1",
        )

    parameter_projection = _candidate_parameter_projections(
        raw_parameters,
        source_facts=source_facts,
    )
    if parameter_projection is None:
        return reject(
            "source_runtime_parameter_projection_unavailable",
            "j.signature.parameters",
            "projection=runtime_physical_abi",
        )
    runtime_parameters, _physical_parameters, _runtime_position_by_name = (
        parameter_projection
    )
    logical_parameter_positions: dict[str, list[int]] = {}
    for runtime_position, parameter in runtime_parameters.items():
        logical_ref = parameter.get("logical_ref")
        if isinstance(logical_ref, str):
            logical_parameter_positions.setdefault(logical_ref, []).append(
                runtime_position
            )

    accesses: dict[str, Mapping[str, Any]] = {}
    for row_index, access in enumerate(raw_accesses):
        if not isinstance(access, Mapping):
            return reject(
                "source_transfer_access_row_malformed",
                f"j.boundary.accesses[{row_index}]",
                f"row_index={row_index}",
            )
        access_id = access.get("access_id")
        if (
            not isinstance(access_id, str)
            or access_id in accesses
            or access.get("kind") not in {"load", "store", "update"}
            or not isinstance(access.get("tensor"), str)
        ):
            return reject(
                "source_transfer_access_identity_invalid",
                f"j.boundary.accesses[{row_index}]",
                f"row_index={row_index}",
                f"duplicate_access_id={int(isinstance(access_id, str) and access_id in accesses)}",
            )
        accesses[access_id] = access

    effect_by_access: dict[str, Mapping[str, Any]] = {}
    effect_ids: set[str] = set()
    access_effect_positions: list[tuple[int, str]] = []
    non_access_effect_positions: list[int] = []
    for index, effect in enumerate(raw_effects):
        if not isinstance(effect, Mapping):
            return reject(
                "source_transfer_effect_row_malformed",
                f"j.boundary.effects[{index}]",
                f"row_index={index}",
            )
        effect_id = effect.get("effect_id")
        if (
            not isinstance(effect_id, str)
            or re.fullmatch(r"e[0-9]+", effect_id) is None
            or effect_id in effect_ids
        ):
            # Scope membership is expressed in stable effect ids.  Require
            # those ids for every effect, including non-access effects, before
            # moving any access-bound row; otherwise a later scope rewrite
            # could silently drop or duplicate an effect that this helper did
            # not observe as a memory transfer.
            return reject(
                "source_transfer_effect_identity_invalid",
                f"j.boundary.effects[{index}]",
                f"row_index={index}",
                f"duplicate_effect_id={int(isinstance(effect_id, str) and effect_id in effect_ids)}",
            )
        effect_ids.add(effect_id)
        access_ref = effect.get("access_ref")
        if access_ref is None:
            non_access_effect_positions.append(index)
            continue
        if (
            not isinstance(access_ref, str)
            or access_ref in effect_by_access
            or access_ref not in accesses
        ):
            return reject(
                "source_transfer_effect_access_binding_invalid",
                f"j.boundary.effects[{index}].access_ref",
                f"row_index={index}",
                f"duplicate_access_effect={int(isinstance(access_ref, str) and access_ref in effect_by_access)}",
                f"declared_access={int(isinstance(access_ref, str) and access_ref in accesses)}",
            )
        access = accesses[access_ref]
        expected_effect_kind = {
            "load": "read",
            "store": "write",
            "update": "update",
        }[str(access["kind"])]
        if effect.get("kind") != expected_effect_kind:
            return reject(
                "source_transfer_effect_kind_mismatch",
                f"j.boundary.effects[{index}].kind",
                f"row_index={index}",
                f"access_ref={access_ref}",
                f"expected_kind={expected_effect_kind}",
            )
        if effect.get("target") != access.get("tensor"):
            return reject(
                "source_transfer_effect_owner_mismatch",
                f"j.boundary.effects[{index}].target",
                f"row_index={index}",
                f"access_ref={access_ref}",
            )
        if effect.get("node_ref") != access.get("node_ref"):
            return reject(
                "source_transfer_effect_node_mismatch",
                f"j.boundary.effects[{index}].node_ref",
                f"row_index={index}",
                f"access_ref={access_ref}",
            )
        effect_by_access[access_ref] = effect
        access_effect_positions.append((index, access_ref))

    if set(effect_by_access) != set(accesses):
        return reject(
            "source_transfer_access_effect_coverage_mismatch",
            "j.boundary.effects",
            f"access_count={len(accesses)}",
            f"bound_effect_count={len(effect_by_access)}",
        )
    if not access_effect_positions:
        return reject(
            "source_transfer_access_effect_rows_missing",
            "j.boundary.effects",
            f"access_count={len(accesses)}",
        )

    occurrences = shared_ledger.occurrences
    if len(occurrences) != len(access_effect_positions):
        return reject(
            "source_transfer_occurrence_count_mismatch",
            "j.boundary.accesses",
            f"occurrence_count={len(occurrences)}",
            f"access_effect_count={len(access_effect_positions)}",
        )

    tensor_ranks = _candidate_tensor_ranks(original)
    node_contexts = _candidate_node_contexts(original)
    if tensor_ranks is None or node_contexts is None:
        return reject(
            "source_transfer_candidate_topology_unavailable",
            "j.intent.intent_function.body",
            f"tensor_rank_table={int(tensor_ranks is not None)}",
            f"node_context_table={int(node_contexts is not None)}",
        )

    source_rows: list[dict[str, Any]] = []
    for occurrence in occurrences:
        ordinal = occurrence.ordinal
        origin_positions = occurrence.origin_positions
        source_context = occurrence.context
        pointer_base = occurrence.pointer_base
        base_position = occurrence.pointer_position
        if (
            base_position is None
            or base_position not in origin_positions
            or len(logical_parameter_positions.get(
                runtime_parameters.get(base_position, {}).get("logical_ref"), []
            )) != 1
        ):
            return reject(
                "source_effect_physical_abi_ambiguous",
                "j.boundary.effects",
                f"ordinal={ordinal}",
                status="ambiguous",
            )
        parameter = runtime_parameters[base_position]
        tensor = parameter.get("logical_ref")
        if (
            not isinstance(tensor, str)
            or not tensor.startswith("t")
            or parameter.get("kind") not in {"pointer", "tensor"}
            or (
                occurrence.logical_ref is not None
                and tensor != occurrence.logical_ref
            )
        ):
            return reject(
                "source_effect_logical_owner_unavailable",
                "j.boundary.effects",
                f"ordinal={ordinal}",
                status="ambiguous",
            )
        address_profile = occurrence.address_profile
        if address_profile is None:
            return reject(
                "source_effect_address_not_direct",
                "j.boundary.effects",
                f"ordinal={ordinal}",
                status="ambiguous",
            )
        kind = occurrence.kind
        if kind not in {"load", "store", "update"}:
            return reject(
                "source_transfer_occurrence_kind_unsupported",
                "<source-transfer-ledger>.occurrences",
                f"ordinal={ordinal}",
            )
        source_rows.append(
            {
                "ordinal": ordinal,
                "kind": kind,
                "tensor": tensor,
                "masked": occurrence.masked,
                "address_profile": address_profile,
                "address_terms": occurrence.address_terms,
                "pointer_base": pointer_base,
                # The closed Source occurrence owns the physical pointer
                # position used later in the access/pi binding witness.  Keep
                # it alongside its validated base/origin facts; omitting it
                # made the successful unique-match path raise KeyError while
                # constructing ``source_access_matches``.
                "pointer_position": base_position,
                "context": source_context,
            }
        )

    def access_mismatch_reasons(
        source_row: Mapping[str, Any], access_ref: str, access: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Classify why one occurrence cannot bind one authored access."""

        reasons: list[str] = []
        if access_ref not in effect_by_access:
            reasons.append("effect_binding_missing")
        if access.get("kind") != source_row["kind"]:
            reasons.append("kind_mismatch")
        if access.get("tensor") != source_row["tensor"]:
            reasons.append("tensor_owner_mismatch")
        if (access.get("predicate") is not None) != source_row["masked"]:
            reasons.append("predicate_presence_mismatch")
        node_ref = access.get("node_ref")
        j_context = (
            node_contexts.get(node_ref, ()) if isinstance(node_ref, str) else ()
        )
        if isinstance(node_ref, str) and node_ref not in node_contexts:
            reasons.append("node_consumer_missing")
        if not _source_j_context_compatible(source_row["context"], j_context):
            reasons.append("node_context_mismatch")
        rank = tensor_ranks.get(str(access.get("tensor")))
        index_profile = _candidate_index_profile(access.get("index_expression"))
        if rank is None:
            reasons.append("tensor_rank_unavailable")
        if index_profile is None:
            reasons.append("index_expression_unavailable")
        elif rank is not None and index_profile[0] != rank:
            reasons.append("index_rank_mismatch")
        return tuple(reasons)

    def compatible_access(
        source_row: Mapping[str, Any], access_ref: str, access: Mapping[str, Any]
    ) -> bool:
        return not access_mismatch_reasons(source_row, access_ref, access)

    # A sealed MCP semantic entry already establishes the one Source-order
    # occurrence identity: ``declare_accesses`` checks each ordinal/label and
    # the constructor then allocates the dense a0..aN table without changing
    # that row order.  Retain that provenance through this private transaction
    # instead of trying to recover it from address *term* counts after labels
    # have deliberately been erased.  A flattened physical address can have
    # more dynamic terms than the number of logical J index components, so
    # comparing those two counts is not a sound occurrence identity test.
    #
    # This path is deliberately opt-in and is enabled only by the caller that
    # has just run the Source-sized semantic submission validator.  It still
    # requires a unique dense canonical table plus kind, tensor, mask, rank,
    # lexical-context, and access/effect checks for every occurrence.  Other
    # callers continue to use the conservative address-profile matcher below.
    declared_access_order = list(accesses)
    canonical_source_order = [
        f"a{index}" for index in range(len(source_rows))
    ]
    effect_access_order = [access_ref for _index, access_ref in access_effect_positions]
    if (
        source_order_sealed
        and declared_access_order == canonical_source_order
        and effect_access_order == canonical_source_order
    ):
        candidates_by_source = [
            [access_ref]
            if compatible_access(source_row, access_ref, accesses[access_ref])
            else []
            for source_row, access_ref in zip(
                source_rows, canonical_source_order, strict=True
            )
        ]
    else:
        candidates_by_source = []
        for source_row in source_rows:
            source_profile = source_row["address_profile"]
            candidates: list[str] = []
            for access_ref, access in accesses.items():
                if not compatible_access(source_row, access_ref, access):
                    continue
                index_profile = _candidate_index_profile(access.get("index_expression"))
                if index_profile is None:
                    continue
                if index_profile[1] != source_profile[0]:
                    continue
                if index_profile[2] != source_profile[2]:
                    continue
                candidates.append(access_ref)
            candidates_by_source.append(candidates)

    if any(len(candidates) != 1 for candidates in candidates_by_source):
        result["status"] = "ambiguous"
        ambiguous_observations: list[str] = []
        for index, candidates in enumerate(candidates_by_source):
            if len(candidates) == 1:
                continue
            ambiguous_observations.extend(
                (f"ordinal={index}", f"candidate_count={len(candidates)}")
            )
            if not candidates:
                expected_access = (
                    canonical_source_order[index]
                    if source_order_sealed
                    and index < len(canonical_source_order)
                    and canonical_source_order[index] in accesses
                    else None
                )
                if expected_access is not None:
                    mismatch_reasons = access_mismatch_reasons(
                        source_rows[index],
                        expected_access,
                        accesses[expected_access],
                    )
                    ambiguous_observations.extend(
                        f"mismatch={reason}" for reason in mismatch_reasons[:4]
                    )
            if len(ambiguous_observations) >= 12:
                break
        result["diagnostics"] = [
            {
                "rule_code": "source_transfer_occurrence_access_ambiguous",
                "path": "j.boundary.accesses",
                "observed_refs_or_types": ambiguous_observations[:12],
            }
        ]
        result["source_access_match_reason"] = (
            "source_transfer_occurrence_access_ambiguous"
        )
        return result

    desired_access_order = [candidates[0] for candidates in candidates_by_source]
    if len(set(desired_access_order)) != len(desired_access_order):
        result["status"] = "ambiguous"
        result["diagnostics"] = [
            {
                "rule_code": "source_transfer_occurrence_access_not_one_to_one",
                "path": "j.boundary.accesses",
                "observed_refs_or_types": [
                    f"occurrence_count={len(desired_access_order)}",
                    f"unique_access_count={len(set(desired_access_order))}",
                ],
            }
        ]
        result["source_access_match_reason"] = (
            "source_transfer_occurrence_access_not_one_to_one"
        )
        return result

    # Gate1 canonicalizes ``memory_effects`` by sorting it, so its order is
    # deliberately unusable as an occurrence ledger.  This source-only
    # sidecar retains the lexical order shared by the provenance/address
    # ledgers above; atomics and incomplete parser views remain reject-only.
    source_transfers = shared_ledger.occurrences
    # The occurrence ledger issued owner/guard authorization before a candidate
    # existed.  Reorder changes only the candidate-local access/effect rows;
    # it must carry each already-issued receipt through the same ordinal rather
    # than dropping it (or, worse, deriving a replacement from an access row).
    # A pass ledger has already validated one complete, contiguous binding set,
    # but recheck that narrow shape here before its sidecar crosses the private
    # reorder transaction.
    authorizations_by_ordinal: dict[int, Mapping[str, Any]] = {}
    for binding in shared_ledger.bindings:
        ordinal = binding.get("ordinal") if isinstance(binding, Mapping) else None
        authorizations = (
            binding.get("semantic_authorizations")
            if isinstance(binding, Mapping)
            else None
        )
        if (
            is_canonical_ordinal(ordinal)
            and ordinal not in authorizations_by_ordinal
            and isinstance(authorizations, Mapping)
        ):
            authorizations_by_ordinal[ordinal] = copy.deepcopy(dict(authorizations))
        else:
            return reject(
                "source_transfer_authorization_projection_unavailable",
                "<source-transfer-ledger>.bindings",
                "expected=one_authorization_receipt_per_ordinal",
            )
    source_transfer_ordinals = {
        occurrence.ordinal for occurrence in source_transfers
    }
    if set(authorizations_by_ordinal) != source_transfer_ordinals:
        return reject(
            "source_transfer_authorization_projection_unavailable",
            "<source-transfer-ledger>.bindings",
            "expected=closed_contiguous_authorization_receipts",
        )
    if len(source_transfers) == len(source_rows) and all(
        effect.kind == row["kind"] and effect.masked == row["masked"]
        for effect, row in zip(source_transfers, source_rows)
    ):
        result["source_access_matches"] = tuple(
            {
                "ordinal": row["ordinal"],
                "access_ref": access_ref,
                "kind": row["kind"],
                "tensor": row["tensor"],
                "masked": row["masked"],
                "source_other": effect.fill_value,
                # This is Source-side provenance.  It must remain separate
                # from the J predicate, whose canonical spelling may be a
                # typed/indexed expression rather than the Source spelling.
                "source_predicate": effect.predicate,
                "source_pointer_base": row["pointer_base"],
                "source_element_type": effect.element_type,
                "source_address_type_profile": effect.address_type_profile,
                "address_profile": row["address_profile"],
                "index_expression": {
                    "address_terms": [
                        {"refs": list(atoms), "coefficient": coefficient}
                        for atoms, coefficient in row["address_terms"]
                    ],
                    "address_profile": {
                        "dynamic_terms": row["address_profile"][0],
                        "constant_terms": row["address_profile"][1],
                        "constant_offsets": list(row["address_profile"][2]),
                    },
                },
                # This comprehension runs after the Source-row scan above,
                # where ``ordinal`` otherwise retains the final occurrence's
                # value.  Bind each row to its own Source occurrence order.
                "effect_order": row["ordinal"],
                "node_ref": accesses[access_ref].get("node_ref"),
                "predicate": accesses[access_ref].get("predicate"),
                "fill_value": accesses[access_ref].get("fill_value"),
                "effect_id": effect_by_access[access_ref].get("effect_id"),
                "semantic_authorizations": authorizations_by_ordinal[
                    row["ordinal"]
                ],
                "pi_binding": {
                    "status": "source_address_owner_bound",
                    "access_ref": access_ref,
                    "effect_id": effect_by_access[access_ref].get("effect_id"),
                    "tensor": row["tensor"],
                    "pointer_position": row["pointer_position"],
                    "element_type": effect.element_type,
                    "address_type_profile": [
                        {
                            "address_terms": [
                                {"refs": list(atoms), "coefficient": coefficient}
                                for atoms, coefficient in terms
                            ],
                            "index_type": index_type,
                            "ssa_type_chain": list(chain),
                        }
                        for terms, index_type, chain in effect.address_type_profile
                    ],
                    "address_profile": {
                        "dynamic_terms": row["address_profile"][0],
                        "constant_terms": row["address_profile"][1],
                        "constant_offsets": list(row["address_profile"][2]),
                    },
                },
            }
            for row, access_ref, effect in zip(
                source_rows,
                desired_access_order,
                source_transfers,
            )
        )
        result["source_access_match_reason"] = (
            "unique_J_access_typed_address_root_and_transfer_occurrence"
        )
        shared_ledger = shared_ledger.with_bindings(
            result["source_access_matches"],
            status="pass",
            reason=result["source_access_match_reason"],
        )
        result["source_transfer_ledger"] = shared_ledger.to_json()

    current_access_order = [access_ref for _index, access_ref in access_effect_positions]
    # ``pass`` is reserved for a closed occurrence -> access/effect binding.
    # An equal effect order is not sufficient: the source rows may still have
    # disagreed in kind/mask or the ledger may not have been closed at all.
    # Returning pass here used to let a binding-less occurrence ledger reach
    # fill validation and pi normalization.
    matches = result.get("source_access_matches")
    if not isinstance(matches, (list, tuple)) or len(matches) != len(source_transfers):
        result["status"] = "ambiguous"
        result["source_access_match_reason"] = (
            "source_transfer_occurrence_binding_unavailable"
        )
        result["diagnostics"] = [
            {
                "rule_code": "source_transfer_ledger_bindings_required",
                "path": "<source-transfer-ledger>.bindings",
                "observed_refs_or_types": [
                    f"occurrences={len(source_transfers)}",
                    f"bindings={len(matches) if isinstance(matches, (list, tuple)) else 0}",
                ],
            }
        ]
        result["source_access_matches"] = ()
        return result
    current_access_declaration_order = list(accesses)
    if (
        current_access_order == desired_access_order
        and current_access_declaration_order == desired_access_order
    ):
        result["status"] = "pass"
        return result

    # Keep every non-access effect at its existing slot.  If an access would
    # cross such an effect, its relative event position is no longer proven.
    desired_at_access_slot = dict(
        zip((index for index, _ref in access_effect_positions), desired_access_order)
    )
    current_prefix: set[str] = set()
    desired_prefix: set[str] = set()
    for index, effect in enumerate(raw_effects):
        if index in desired_at_access_slot:
            current_prefix.add(str(effect["access_ref"]))
            desired_prefix.add(desired_at_access_slot[index])
        elif index in non_access_effect_positions and current_prefix != desired_prefix:
            result["status"] = "ambiguous"
            result["diagnostics"] = [
                {
                    "rule_code": "source_effect_non_access_order_barrier",
                    "path": "j.boundary.effects",
                    "observed_refs_or_types": [
                        f"effect_row={index}",
                        "barrier=non_access_effect",
                    ],
                }
            ]
            # The provisional bijection was sufficient to explain the
            # ambiguity, but it is not an authority once a non-access effect
            # blocks the proven reorder.  Remove both projections so a future
            # consumer cannot accidentally use a half-closed ledger.
            result["source_access_matches"] = ()
            shared_ledger = shared_ledger.with_bindings(
                (),
                status="ambiguous",
                reason="source_effect_non_access_order_barrier",
                diagnostics=result["diagnostics"],
            )
            result["source_transfer_ledger"] = shared_ledger.to_json()
            return result

    desired_effects = {
        access_ref: copy.deepcopy(effect_by_access[access_ref])
        for access_ref in desired_access_order
    }
    rewritten_effects: list[dict[str, Any]] = []
    desired_index = 0
    for effect in raw_effects:
        access_ref = effect.get("access_ref") if isinstance(effect, Mapping) else None
        if isinstance(access_ref, str):
            rewritten_effects.append(desired_effects[desired_access_order[desired_index]])
            desired_index += 1
        else:
            rewritten_effects.append(copy.deepcopy(dict(effect)))

    rewritten = copy.deepcopy(original)
    rewritten_j = copy.deepcopy(dict(j))
    rewritten_boundary = copy.deepcopy(dict(boundary))
    # The occurrence ledger owns not only the effect sequence but also the
    # access declaration sequence from which the canonical constructor assigns
    # a*/e*.  Leaving accesses in provider order while moving only effects
    # creates two mechanical order authorities and lets the later allocator
    # attach a fresh e* sequence to the wrong a* declaration order.
    access_by_ref = {
        str(access["access_id"]): copy.deepcopy(dict(access))
        for access in raw_accesses
        if isinstance(access, Mapping) and isinstance(access.get("access_id"), str)
    }
    if set(access_by_ref) != set(desired_access_order):
        return reject(
            "source_transfer_reorder_access_set_mismatch",
            "j.boundary.accesses",
            f"declared_access_count={len(access_by_ref)}",
            f"desired_access_count={len(desired_access_order)}",
            status="ambiguous",
        )
    rewritten_boundary["accesses"] = [
        access_by_ref[access_ref]
        for access_ref in desired_access_order
    ]
    rewritten_boundary["effects"] = rewritten_effects
    rewritten_j["boundary"] = rewritten_boundary

    raw_scopes = rewritten_j.get("scopes")
    if isinstance(raw_scopes, list):
        for scope in raw_scopes:
            if not isinstance(scope, Mapping) or not isinstance(scope.get("effects"), list):
                return reject(
                    "source_effect_scope_order_unavailable",
                    "j.scopes",
                    "scope_effect_membership=malformed",
                    status="ambiguous",
                )
            members = list(scope["effects"])
            if any(member not in effect_ids for member in members) or len(members) != len(set(members)):
                return reject(
                    "source_effect_scope_membership_unavailable",
                    "j.scopes.effects",
                    f"member_count={len(members)}",
                    f"unique_member_count={len(set(members))}",
                    status="ambiguous",
                )
            scope["effects"] = [
                str(effect["effect_id"])
                for effect in rewritten_effects
                if effect.get("effect_id") in members
            ]
            if set(scope["effects"]) != set(members):
                return reject(
                    "source_effect_scope_membership_changed",
                    "j.scopes.effects",
                    f"member_count={len(members)}",
                    status="ambiguous",
                )

    rewritten["j"] = rewritten_j
    result["status"] = "pass"
    result["artifact"] = rewritten
    result["applied"] = [
        {
            "rule_code": "deterministic_source_transfer_order_repair",
            "path": "j.boundary.accesses + j.boundary.effects",
            "action": (
                "reorder access declarations and access-bound effects to the "
                "unique Source transfer ledger order"
            ),
            "source_transfer_ordinals": [
                occurrence.ordinal for occurrence in source_transfers
            ],
            "previous_access_order": current_access_order,
            "new_access_order": desired_access_order,
            "authority": (
                "unique physical ABI owner, transfer kind, mask, direct address/index, "
                "and region-context correspondence"
            ),
        }
    ]
    return result


def _direct_source_fill_json_literal(value: object) -> bool | int | float | None:
    """Return one direct, finite JSON scalar from a Source ``other=`` value.

    The TTIR fingerprint intentionally keeps arbitrary SSA fills opaque.  The
    canonical transfer ledger can additionally retain the exact lexical
    Source scalar (for example ``0``) before TTIR lowers it to an SSA value.
    Accept only that same closed scalar vocabulary or one ``arith.constant``
    spelling; vectors, compound dense values, references, and non-finite
    values remain unproved instead of being projected into J.
    """

    if not isinstance(value, str):
        return None
    text = value.strip()
    dense = re.fullmatch(
        r"arith\.constant\s+dense<(?P<literal>[^<>]+)>\s*:\s*.+",
        text,
    )
    scalar = re.fullmatch(
        r"arith\.constant\s+(?P<literal>[^:\s]+)\s*:\s*.+",
        text,
    )
    match = dense or scalar
    # ``TransferOccurrence.fill_value`` is the Source lexical proof, not a
    # TTIR instruction. A direct scalar has no unsafe SSA/reference content to
    # recover, so it is equally closed evidence for authored-fill validation.
    literal = match.group("literal").strip() if match is not None else text
    if len(literal) > 128:
        return None
    if literal == "true":
        return True
    if literal == "false":
        return False
    if re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?",
        literal,
    ) is None:
        return None
    try:
        if "." not in literal and "e" not in literal.lower():
            return int(literal)
        number = float(literal)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _same_json_scalar(left: object, right: object) -> bool:
    """Compare only the closed scalar vocabulary proven by Source facts."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    if isinstance(left, float) and not math.isfinite(left):
        return False
    if isinstance(right, float) and not math.isfinite(right):
        return False
    return left == right


def _direct_source_special_fill(value: object) -> str | None:
    """Recognize one exact non-finite TTIR fill without making it repairable.

    J uses a closed string spelling for non-finite values because JSON has no
    infinity literal.  The compiler observation may nevertheless expose the
    exact IEEE bit pattern as a dense tensor constant.  This helper is only a
    validation witness: callers must not use its result to fill a missing J
    value automatically.
    """

    if not isinstance(value, str):
        return None
    text = value.strip()
    special_patterns = (
        (r"FF800000", "-inf"),
        (r"7F800000", "inf"),
        (r"FFF0000000000000", "-inf"),
        (r"7FF0000000000000", "inf"),
    )
    for bits, spelling in special_patterns:
        if re.fullmatch(
            rf"arith\.constant\s+dense<0x{bits}>\s*:\s*"
            rf"tensor<[^>]*(?:f32|f64)(?:\s*,[^>]*)?>",
            text,
        ):
            return spelling
    return None


def _same_source_fill(left: object, source_value: object) -> bool:
    """Compare authored J fill syntax with one closed Source fill witness."""

    if source_value in {"-inf", "inf"}:
        return isinstance(left, str) and left in {
            source_value,
            "NEG_INF" if source_value == "-inf" else "POS_INF",
        }
    return _same_json_scalar(left, source_value)


def _source_access_fill_validation_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Validate authored fills against exact Source occurrence evidence."""

    j = packet.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    raw_accesses = boundary.get("accesses") if isinstance(boundary, Mapping) else None
    if not isinstance(raw_accesses, list):
        return ()
    access_by_id: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, access in enumerate(raw_accesses):
        access_ref = access.get("access_id") if isinstance(access, Mapping) else None
        if not isinstance(access_ref, str) or access_ref in access_by_id:
            return ()
        access_by_id[access_ref] = (index, access)
    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    source_transfers = shared_ledger.occurrences
    source_transfer_by_ordinal = {
        effect.ordinal: effect for effect in source_transfers
    }
    source_loads = {
        ordinal: effect
        for ordinal, effect in source_transfer_by_ordinal.items()
        if effect.kind == "load"
    }
    source_masked_loads = {
        ordinal: effect
        for ordinal, effect in source_loads.items()
        if effect.masked
    }
    observed_masked_load_count = 0
    if isinstance(observed, TTIRFingerprint):
        # ``TTIRFingerprint`` deliberately stores multiset rows as an
        # immutable tuple rather than a mapping.  This SourceReconcile
        # consumer needs only a read-only count, so materialize the canonical
        # mapping view here instead of treating the tuple as a second wire
        # representation.  The ledger remains the sole occurrence authority.
        observed_masked_load_count = int(
            dict(observed.op_multiset).get("tt.load.masked", 0)
        )
    # ``TransferOccurrence`` is the shared ledger record.  Its canonical
    # Source ``other=`` spelling is ``fill_value``; the TTIR-side observation
    # objects use ``other`` only while the ledger is being built.  Reading
    # ``effect.other`` here bypasses that contract and raises AttributeError
    # precisely on the unavailable-ledger path that must remain diagnostic.
    source_fill_detail_missing = shared_ledger.status != "pass" and any(
        effect.kind == "load" and effect.fill_value is not None
        for effect in shared_ledger.occurrences
    )
    authored_fills = {
        access_ref
        for access_ref, (_index, access) in access_by_id.items()
        if access.get("fill_value") is not None
    }
    def diagnostic(
        code: str,
        *,
        access_ref: str | None = None,
        index: int | None = None,
        facts: tuple[str, ...] = (),
    ) -> Mapping[str, Any]:
        path = "<source-ttir-obligation> j.boundary.accesses"
        if index is not None:
            path += f"[{index}].fill_value"
        return {
            "rule_code": code,
            "path": path,
            "anonymous_handles": [access_ref] if access_ref is not None else [],
            "observed_refs_or_types": list(facts),
            "line": None,
        }

    if len(source_transfer_by_ordinal) != len(source_transfers):
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=("Source transfer occurrence ordinals are not unique",),
            ),
        )

    if source_fill_detail_missing:
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=(
                    "Source masked-load fill has no lexical transfer occurrence ledger",
                ),
            ),
        )
    if observed_masked_load_count > 0 and shared_ledger.status != "pass":
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=(
                    "Source exposes masked loads but the canonical transfer ledger "
                    f"is unavailable: {shared_ledger.reason}",
                ),
            ),
        )
    if observed_masked_load_count > 0 and not source_masked_loads:
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=(
                    "Source masked-load observation has no corresponding masked "
                    "occurrence in the canonical transfer ledger",
                ),
            ),
        )
    # A masked load with ``other=None`` is still an occurrence-level
    # obligation.  Returning early based only on ``source_fills`` used to
    # mistake "the Source has no other=" for "there is no fill fact", allowing
    # a missing J fill to reach Gate1.  Continue through the bijection and the
    # per-occurrence presence check below whenever Source exposes a masked
    # load, even when its fill value is absent.
    if not source_masked_loads and not authored_fills:
        return ()

    # ``SourceTransferLedger`` is the occurrence authority, while the
    # reorder helper returns its bounded access-match projection.  Always
    # derive that projection from the same ledger; never call ``.get`` on the
    # dataclass or silently treat a bare ledger JSON as a complete match
    # receipt.
    if (
        shared_ledger.status == "pass"
        and len(shared_ledger.bindings) == len(shared_ledger.occurrences)
        and (not shared_ledger.occurrences or shared_ledger.bindings)
    ):
        # A closed dataclass ledger is already the occurrence authority.  Do
        # not invoke reorder a second time merely to obtain its projection;
        # that would create a fresh access matcher for fill validation.
        source_ledger = {
            "status": "pass",
            "source_access_matches": tuple(
                dict(row) for row in shared_ledger.bindings
            ),
            "diagnostics": tuple(shared_ledger.diagnostics),
        }
    elif isinstance(ledger, Mapping) and "source_access_matches" in ledger:
        # A bare access-match projection is ingress-local diagnostic data, not
        # an occurrence ledger.  It must not become an alternate authority for
        # fill validation merely because its outer status says ``pass``.
        source_ledger = {
            "status": "unavailable",
            "source_access_matches": (),
            "source_access_match_reason": (
                "bare_source_access_projection_is_not_transfer_ledger"
            ),
            "diagnostics": (
                {
                    "rule_code": "source_transfer_ledger_bindings_required",
                    "path": "<source-transfer-ledger>",
                    "observed_refs_or_types": [
                        "bare source_access_matches projection is not authoritative"
                    ],
                },
            ),
        }
    else:
        source_ledger = reorder_boundary_effects_from_source_ledger(
            packet,
            observed,
            ledger=shared_ledger,
            source_facts=source_facts,
        )
    ledger_diagnostics = source_ledger.get("diagnostics")
    if not isinstance(ledger_diagnostics, (list, tuple)):
        ledger_diagnostics = ()
    reason = source_ledger.get("source_access_match_reason") or next(
        (
            item.get("rule_code")
            for item in ledger_diagnostics
            if isinstance(item, Mapping) and isinstance(item.get("rule_code"), str)
        ),
        "source_access_occurrence_unavailable",
    )
    reason = str(reason)

    raw_matches = source_ledger.get("source_access_matches")
    matches_by_ordinal: dict[int, str] = {}
    if source_ledger.get("status") != "pass" or not isinstance(
        raw_matches, (list, tuple)
    ):
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=("unique Source transfer/access occurrence unavailable", reason),
            ),
        )
    for match in raw_matches:
        ordinal = match.get("ordinal") if isinstance(match, Mapping) else None
        access_ref = match.get("access_ref") if isinstance(match, Mapping) else None
        valid = (
            is_canonical_ordinal(ordinal)
            and ordinal in source_transfer_by_ordinal
            and isinstance(access_ref, str)
            and access_ref in access_by_id
            and ordinal not in matches_by_ordinal
            and access_ref not in matches_by_ordinal.values()
        )
        if not valid:
            return (
                diagnostic(
                    "source_access_fill_occurrence_unproved",
                    facts=("malformed unique Source transfer/access occurrence", reason),
                ),
            )
        effect = source_transfer_by_ordinal.get(ordinal)
        if effect is None:
            return (
                diagnostic(
                    "source_access_fill_occurrence_unproved",
                    access_ref=access_ref,
                    facts=("Source transfer ordinal is not present in the ledger", reason),
                ),
            )
        if (
            match.get("kind") != effect.kind
            or match.get("masked") != effect.masked
            or match.get("source_other") != effect.fill_value
            or match.get("source_predicate") != effect.predicate
            or match.get("source_pointer_base") != effect.pointer_base
            or match.get("source_element_type") != effect.element_type
            or match.get("source_address_type_profile")
            != effect.address_type_profile
        ):
            return (
                diagnostic(
                    "source_access_fill_occurrence_unproved",
                    access_ref=access_ref,
                    facts=("Source transfer occurrence changed after matching", reason),
                ),
            )
        matches_by_ordinal[ordinal] = access_ref

    # A pass ledger is a complete occurrence/access bijection, not merely a
    # list containing the occurrences that happen to carry a fill value.  Keep
    # this check here as well as in SourceTransferLedger.__post_init__ because
    # this helper also accepts the ingress-local mapping projection.
    if (
        set(matches_by_ordinal) != set(source_transfer_by_ordinal)
        or set(matches_by_ordinal.values()) != set(access_by_id)
    ):
        return (
            diagnostic(
                "source_access_fill_occurrence_unproved",
                facts=("Source transfer/access ledger is not a complete bijection", reason),
            ),
        )

    diagnostics: list[Mapping[str, Any]] = [
        diagnostic(
            "source_access_fill_occurrence_unproved",
            facts=(f"Source masked-load ordinal={ordinal} has other=", reason),
        )
        for ordinal in sorted(set(source_masked_loads) - set(matches_by_ordinal))
    ]
    mapped_accesses = set(matches_by_ordinal.values())
    for access_ref in sorted(authored_fills - mapped_accesses):
        diagnostics.append(
            diagnostic(
                "source_access_fill_occurrence_unproved",
                access_ref=access_ref,
                index=access_by_id[access_ref][0],
                facts=("J fill has no unique Source transfer/access occurrence", reason),
            )
        )
    if diagnostics:
        return tuple(diagnostics)

    for ordinal, access_ref in sorted(matches_by_ordinal.items()):
        effect = source_transfer_by_ordinal.get(ordinal)
        if effect is None:
            diagnostics.append(
                diagnostic(
                    "source_access_fill_occurrence_unproved",
                    access_ref=access_ref,
                    facts=("Source transfer ordinal is not present in the ledger",),
                )
            )
            continue
        if effect.kind != "load":
            continue
        access_index, access = access_by_id[access_ref]
        if access.get("kind") != "load" or (
            effect.fill_value is not None and not effect.masked
        ):
            diagnostics.append(
                diagnostic(
                    "source_access_fill_occurrence_unproved",
                    access_ref=access_ref,
                    index=access_index,
                    facts=(f"Source load ordinal={ordinal} has no compatible J load",),
                )
            )
            continue
        authored_fill = access.get("fill_value")
        if effect.masked and effect.fill_value is None:
            # A masked Source load is not closed merely because its predicate
            # was observed.  The occurrence must also expose the other fill; when
            # that fact is absent, neither a missing nor an authored J fill is
            # safe to accept.  Keep this occurrence-level and fail closed so a
            # different masked load cannot satisfy it by count.
            diagnostics.append(
                diagnostic(
                    (
                        "source_access_fill_presence_mismatch"
                        if authored_fill is not None
                        else "source_access_fill_presence_unproved"
                    ),
                    access_ref=access_ref,
                    index=access_index,
                    facts=(
                        f"Source masked-load ordinal={ordinal} has no observed other= fill",
                    ),
                )
            )
            continue
        if effect.fill_value is None:
            if authored_fill is not None:
                diagnostics.append(
                    diagnostic(
                        "source_access_fill_presence_mismatch",
                        access_ref=access_ref,
                        index=access_index,
                        facts=(f"Source load ordinal={ordinal} has no other= operand",),
                    )
                )
            continue
        literal = _direct_source_fill_json_literal(effect.fill_value)
        special_literal = _direct_source_special_fill(effect.fill_value)
        source_literal = literal if literal is not None else special_literal
        if source_literal is None:
            diagnostics.append(
                diagnostic(
                    "source_access_fill_literal_unproved",
                    access_ref=access_ref,
                    index=access_index,
                    facts=(
                        f"Source masked-load ordinal={ordinal} other= is not a direct closed JSON scalar literal",
                    ),
                )
            )
        elif authored_fill is None:
            diagnostics.append(
                diagnostic(
                    "source_masked_load_fill_missing",
                    access_ref=access_ref,
                    index=access_index,
                    facts=(
                        f"Source masked-load ordinal={ordinal} requires one provider-authored J fill",
                    ),
                )
            )
        elif not _same_source_fill(authored_fill, source_literal):
            diagnostics.append(
                diagnostic(
                    "source_access_fill_mismatch",
                    access_ref=access_ref,
                    index=access_index,
                    facts=(
                        f"Source masked-load ordinal={ordinal} and authored J fill differ",
                    ),
                )
            )
    return tuple(diagnostics)


def _matches_source_issued_receipt(
    access: Mapping[str, Any], field: str, receipt: object,
) -> bool:
    """Compare a claim with an already selected private ledger receipt.

    The caller must select the receipt by the current Source access binding.
    This checks issuance status/token equality only, not token syntax, J
    content, producer identity or semantic equivalence. It never issues one.
    """
    return (
        isinstance(receipt, Mapping)
        and receipt.get("status") == "pass"
        and access.get(field) == receipt.get("token")
    )


def _require_source_owner_receipts(
    access: Mapping[str, Any], *, owner: object, access_view: object,
) -> None:
    """Require both private issuance matches before checking candidate DAGs."""
    if not (
        _matches_source_issued_receipt(access, "semantic_owner_authorization", owner)
        and _matches_source_issued_receipt(access, "semantic_access_authorization", access_view)
    ):
        raise ValueError("source_owner_receipt_unissued")


def validate_source_semantic_authorizations(
    packet: Mapping[str, Any],
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Reject owner/guard claims not issued by the exact Source ledger.

    This is a verifier, not an allocator or semantic repair.  It consumes the
    same post-reorder occurrence/access binding used by SourceReconcile and
    verifies access/guard receipts against that occurrence. Owner claims also
    require an independently issued typed producer DAG commitment, not a family.
    It never derives a token from source order, node numbering, names, shape,
    or a provider declaration.

    Owner verification reconsumes the candidate's actual producer slice; it
    does not authenticate the whole candidate or certify fresh-run ownership.
    The guard branch here checks issuance and operation family only, not full
    predicate equivalence. Boundary/compiled obligations and the Gate chain
    remain separate mandatory consumers; this result must not replace them.
    """

    shared = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared.status != "pass":
        return (
            {
                "rule_code": "semantic_authorization_ledger_unavailable",
                "path": "<source-transfer-ledger>",
                "observed_refs_or_types": ["expected:closed Source authorization ledger"],
            },
        )
    j = packet.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    accesses = boundary.get("accesses") if isinstance(boundary, Mapping) else None
    guard_relations = (
        boundary.get("access_guard_relations") if isinstance(boundary, Mapping)
        else None
    )
    body = function.get("body") if isinstance(function, Mapping) else None
    if not isinstance(accesses, list) or not isinstance(guard_relations, list) or not isinstance(body, list):
        return (
            {
                "rule_code": "semantic_authorization_candidate_topology_unavailable",
                "path": "j",
                "observed_refs_or_types": ["expected:boundary access/guard and operation body tables"],
            },
        )

    value_producers: dict[str, str] = {}
    value_producer_nodes: dict[str, str] = {}
    seen_node_ids: set[str] = set()
    malformed_producer_table = False
    duplicate_value_producer = False

    def visit(rows: object) -> None:
        nonlocal malformed_producer_table, duplicate_value_producer
        if not isinstance(rows, list):
            malformed_producer_table = True
            return
        for operation in rows:
            if not isinstance(operation, Mapping):
                malformed_producer_table = True
                continue
            node_id = operation.get("id")
            op = operation.get("op")
            if (
                not isinstance(node_id, str)
                or not node_id
                or node_id in seen_node_ids
                or not isinstance(op, str)
                or not op
            ):
                malformed_producer_table = True
            else:
                seen_node_ids.add(node_id)
            result = operation.get("result")
            if isinstance(result, str):
                results: tuple[object, ...] = (result,)
            elif isinstance(result, (list, tuple)):
                results = tuple(result)
            elif result is None:
                results = ()
            else:
                results = (result,)
                malformed_producer_table = True
            for value in results:
                if not isinstance(value, str) or not value:
                    malformed_producer_table = True
                    continue
                previous_node = value_producer_nodes.get(value)
                if previous_node is not None:
                    # Equal operation names do not make two SSA producers
                    # interchangeable. A guard/output relation must resolve
                    # to one authored node and one authored result.
                    duplicate_value_producer = True
                    continue
                if not isinstance(node_id, str) or not node_id:
                    malformed_producer_table = True
                    continue
                value_producer_nodes[value] = node_id
                value_producers[value] = op
            for child_field in ("body",):
                child_rows = operation.get(child_field)
                if child_rows is not None:
                    visit(child_rows)
            for arm in ("positive", "negative"):
                arm_value = operation.get(arm)
                if isinstance(arm_value, Mapping):
                    arm_rows = arm_value.get("body")
                    if arm_rows is not None:
                        visit(arm_rows)
                elif arm_value is not None:
                    malformed_producer_table = True

    visit(body)
    if malformed_producer_table or duplicate_value_producer:
        return (
            {
                "rule_code": "source_semantic_producer_table_ambiguous",
                "path": "j.intent.intent_function.body",
                "observed_refs_or_types": [
                    "expected:unique well-formed J node/result producer table"
                ],
            },
        )
    # This verifier must not rely on a caller having already proved the
    # occurrence/access cardinality.  Build the join defensively here as
    # well: a malformed or duplicate row must not disappear from a dict
    # comprehension and make a partial candidate look Source-complete.
    binding_by_access: dict[str, Mapping[str, Any]] = {}
    malformed_binding = False
    for binding in shared.bindings:
        access_ref = binding.get("access_ref") if isinstance(binding, Mapping) else None
        if not isinstance(access_ref, str) or access_ref in binding_by_access:
            malformed_binding = True
            continue
        binding_by_access[access_ref] = binding
    access_by_id = {
        row.get("access_id"): row
        for row in accesses
        if isinstance(row, Mapping) and isinstance(row.get("access_id"), str)
    }
    malformed_access = (
        not isinstance(accesses, list)
        or len(access_by_id) != len(accesses)
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("access_id"), str)
            for row in accesses
        )
    )
    expected_access_refs = set(binding_by_access)
    actual_access_refs = set(access_by_id)
    if (
        malformed_binding
        or malformed_access
        or expected_access_refs != actual_access_refs
        or len(binding_by_access) != len(shared.occurrences)
    ):
        return (
            {
                "rule_code": "source_semantic_access_cardinality_unproved",
                "path": "j.boundary.accesses + <source-transfer-ledger>.bindings",
                "observed_refs_or_types": [
                    "expected:one unique candidate access for every Source occurrence"
                ],
            },
        )
    # Guard relations are an admission input, not an optional annotation.
    # Do this structural join before consuming any private guard receipt so a
    # malformed row cannot disappear from this verifier and be interpreted by
    # a later projection as if the guard set were complete.  This deliberately
    # proves only relation identity/shape; it does not infer a predicate,
    # domain, polarity, or producer from any other field.
    guard_relation_structure_diagnostics: list[Mapping[str, Any]] = []
    seen_guard_access_refs: set[str] = set()
    for relation_index, relation in enumerate(guard_relations):
        relation_path = f"j.boundary.access_guard_relations[{relation_index}]"
        if not isinstance(relation, Mapping):
            guard_relation_structure_diagnostics.append(
                {
                    "rule_code": "source_semantic_guard_relation_structure_invalid",
                    "path": relation_path,
                    "observed_refs_or_types": [
                        "expected:object with access_ref/value_ref"
                    ],
                }
            )
            continue
        access_ref = relation.get("access_ref")
        value_ref = relation.get("value_ref")
        if not isinstance(access_ref, str) or not access_ref:
            guard_relation_structure_diagnostics.append(
                {
                    "rule_code": "source_semantic_guard_relation_structure_invalid",
                    "path": f"{relation_path}.access_ref",
                    "observed_refs_or_types": [
                        "expected:non-empty candidate access reference"
                    ],
                }
            )
            continue
        if access_ref in seen_guard_access_refs:
            guard_relation_structure_diagnostics.append(
                {
                    "rule_code": "source_semantic_guard_relation_ambiguous",
                    "path": f"{relation_path}.access_ref",
                    "observed_refs_or_types": [
                        "expected:one unique guard relation per access"
                    ],
                }
            )
            continue
        seen_guard_access_refs.add(access_ref)
        if not isinstance(value_ref, str) or not value_ref:
            guard_relation_structure_diagnostics.append(
                {
                    "rule_code": "source_semantic_guard_relation_structure_invalid",
                    "path": f"{relation_path}.value_ref",
                    "observed_refs_or_types": [
                        "expected:non-empty authored guard value reference"
                    ],
                }
            )
            continue
        if access_ref not in access_by_id:
            guard_relation_structure_diagnostics.append(
                {
                    "rule_code": "source_semantic_guard_relation_access_unmatched",
                    "path": f"{relation_path}.access_ref",
                    "observed_refs_or_types": [
                        "expected:guard relation access_ref in j.boundary.accesses"
                    ],
                }
            )
    if guard_relation_structure_diagnostics:
        return tuple(guard_relation_structure_diagnostics)
    # The access id is only a host allocation label.  The semantic identity
    # still has to agree with the exact ledger ordinal/kind/tensor; neither a
    # tensor-name match nor list position is sufficient to establish the
    # Source occurrence join.
    occurrence_by_ordinal = {
        occurrence.ordinal: occurrence
        for occurrence in shared.occurrences
    }
    if len(occurrence_by_ordinal) != len(shared.occurrences):
        return (
            {
                "rule_code": "source_semantic_access_occurrence_mismatch",
                "path": "<source-transfer-ledger>.occurrences",
                "observed_refs_or_types": [
                    "expected:unique explicit Source occurrence ordinals"
                ],
            },
        )
    for access_ref, binding in binding_by_access.items():
        access = access_by_id[access_ref]
        ordinal = binding.get("ordinal")
        occurrence = (
            occurrence_by_ordinal.get(ordinal)
            if is_canonical_ordinal(ordinal)
            else None
        )
        if (
            occurrence is None
            or (
                "ordinal" in access
                and access.get("ordinal") != ordinal
            )
            or access.get("kind") != occurrence.kind
            or access.get("tensor") != occurrence.logical_ref
        ):
            return (
                {
                    "rule_code": "source_semantic_access_occurrence_mismatch",
                    "path": f"j.boundary.accesses[{access_ref}]",
                    "observed_refs_or_types": [
                        "expected:exact Source ordinal/kind/tensor binding"
                    ],
                },
            )
    diagnostics: list[Mapping[str, Any]] = []
    for index, access in enumerate(accesses):
        if not isinstance(access, Mapping):
            continue
        access_id = access.get("access_id")
        if not isinstance(access_id, str):
            continue
        binding = binding_by_access.get(access_id)
        authorizations = (
            binding.get("semantic_authorizations") if isinstance(binding, Mapping)
            else None
        )
        access_view = authorizations.get("access_view") if isinstance(authorizations, Mapping) else None
        node_ref = access.get("node_ref")
        if node_ref is None:
            if access.get("kind") == "load" and (
                not _matches_source_issued_receipt(
                    access, "semantic_access_authorization", access_view,
                )
            ):
                diagnostics.append(
                    {
                        "rule_code": "semantic_access_authorization_missing",
                        "path": f"j.boundary.accesses[{index}].semantic_access_authorization",
                        "observed_refs_or_types": ["expected:ledger-issued exact Source access receipt"],
                    }
                )
            continue
        owner = authorizations.get("owner") if isinstance(authorizations, Mapping) else None
        try:
            _require_source_owner_receipts(access, owner=owner, access_view=access_view)
            # Content verification is a separate obligation. A successful
            # issuance comparison cannot discharge the exact producer proof.
            validate_source_owner_binding(
                function, j["signature"]["parameters"], access, boundary,
                value_dimension_relations=j.get("value_dimension_relations"),
            )
        except (ValueError, TypeError, KeyError, RecursionError):
            diagnostics.append({
                "rule_code": "source_exact_owner_binding_failed",
                "path": f"j.boundary.accesses[{index}].node_ref",
                "observed_refs_or_types": ["expected:unique exact typed Source producer DAG"],
            })
    for relation_index, relation in enumerate(guard_relations):
        if not isinstance(relation, Mapping):
            # The structural pass above should already have returned. Keep
            # this branch fail-closed as a defensive invariant if the raw
            # packet is mutated between the two passes.
            return (
                {
                    "rule_code": "source_semantic_guard_relation_structure_invalid",
                    "path": f"j.boundary.access_guard_relations[{relation_index}]",
                    "observed_refs_or_types": [
                        "expected:object with access_ref/value_ref"
                    ],
                },
            )
        access_ref = relation.get("access_ref")
        access = access_by_id.get(access_ref) if isinstance(access_ref, str) else None
        binding = binding_by_access.get(access_ref) if isinstance(access_ref, str) else None
        authorizations = (
            binding.get("semantic_authorizations") if isinstance(binding, Mapping)
            else None
        )
        guard = (
            authorizations.get("computed_guard")
            if isinstance(authorizations, Mapping) else None
        )
        if (
            not isinstance(access, Mapping)
            or not _matches_source_issued_receipt(
                access, "semantic_guard_authorization", guard,
            )
        ):
            diagnostics.append(
                {
                    "rule_code": "semantic_guard_authorization_missing",
                    "path": f"j.boundary.access_guard_relations[{relation_index}]",
                    "observed_refs_or_types": ["expected:ledger-issued computed guard token"],
                }
            )
            continue
        value_ref = relation.get("value_ref")
        if (
            not isinstance(value_ref, str)
            or value_producers.get(value_ref) != guard.get("operation_family")
        ):
            diagnostics.append(
                {
                    "rule_code": "semantic_guard_operation_family_mismatch",
                    "path": f"j.boundary.access_guard_relations[{relation_index}].value_ref",
                    "observed_refs_or_types": [
                        "expected:ledger-authorized guard operation family"
                    ],
                }
            )
    return tuple(diagnostics)


def validate_source_store_output_bindings(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Require one closed Source-store -> J-update -> output binding.

    Source owner authorization proves a typed producer slice, but it does not
    by itself prove that the slice is the final logical output.  This verifier
    joins the already-issued occurrence receipt, the private TTIR store/value
    observation, the authored J operation, the optional LogicalUpdateSpec,
    and the output return/mapping rows.  It then consumes the existing
    FunctionalBoundary partition proof for the complete output.  It never
    creates a row or chooses a producer.  Missing or non-unique joins remain
    rejected.

    Compiler node ids are consumed only from the in-memory observation.  They
    are not serialized, exposed to the provider, or used as J handles.
    """

    shared = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared.status != "pass":
        return ({
            "rule_code": "source_store_output_ledger_unavailable",
            "path": "<source-transfer-ledger>",
            "observed_refs_or_types": ["expected:closed Source occurrence ledger"],
        },)
    # Never route a logical update through the ordinary physical-store proof.
    # Even an internally rehydrated pass ledger must carry a dedicated typed
    # atomic/update observation before its pointer, replacement value,
    # compare/observed result, and final output relation can be admitted.
    # The current observation protocol does not provide that complete record.
    if any(occurrence.kind == "update" for occurrence in shared.occurrences):
        return ({
            "rule_code": "source_atomic_update_authorization_unavailable",
            "path": "<source-transfer-ledger>.occurrences",
            "observed_refs_or_types": [
                "expected:typed logical-update and physical-store proof"
            ],
        },)
    if not isinstance(observed, TTIRFingerprint):
        return ({
            "rule_code": "source_ttir_store_producer_unavailable",
            "path": "<source-ttir-observation>",
            "observed_refs_or_types": ["expected:compiler-attested store/value producer"],
        },)
    j = packet.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    signature = j.get("signature") if isinstance(j, Mapping) else None
    if (
        not isinstance(boundary, Mapping)
        or not isinstance(function, Mapping)
        or not isinstance(signature, Mapping)
        or not isinstance(function.get("body"), list)
        or not isinstance(function.get("outputs"), list)
        or not isinstance(function.get("returns"), list)
        or not isinstance(boundary.get("accesses"), list)
        or not isinstance(boundary.get("effects"), list)
        or not isinstance(boundary.get("output_mappings"), list)
        or not isinstance(boundary.get("updates"), list)
    ):
        return ({
            "rule_code": "source_store_output_topology_unavailable",
            "path": "j.boundary",
            "observed_refs_or_types": [
                "expected:access/effect/update/output-mapping/return tables"
            ],
        },)

    # Keep the physical ABI join local to the final store/output admission
    # check.  The Source occurrence owns a runtime argument position; a
    # matching logical tensor name alone is insufficient because constexpr
    # and compiler-only slots can otherwise make two physical p* positions
    # look interchangeable.
    parameter_projection = _candidate_parameter_projections(
        signature.get("parameters"),
        source_facts=source_facts,
    )
    if parameter_projection is None:
        return ({
            "rule_code": "source_store_physical_abi_unproved",
            "path": "j.signature.parameters",
            "observed_refs_or_types": [
                "expected:unique Source-runtime to candidate physical ABI projection"
            ],
        },)
    runtime_parameters, _physical_parameters, _runtime_position_by_name = (
        parameter_projection
    )

    # Reuse the canonical output constructor as the single authority for
    # ABI-output/return/mapping coherence.  The checks below add the
    # Source-occurrence and private-TTIR joins; they do not create a second
    # producer interpretation.
    if canonical_output_records(
        function,
        boundary,
        None,
        signature=signature,
    ) is None:
        return ({
            "rule_code": "source_store_output_topology_unavailable",
            "path": "j.intent.intent_function.outputs + j.returns + j.boundary.output_mappings",
            "observed_refs_or_types": [
                "expected:canonical output ABI/return/mapping record"
            ],
        },)

    operations: dict[str, Mapping[str, Any]] = {}

    def collect(rows: object) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            node_ref = row.get("id")
            if isinstance(node_ref, str):
                if node_ref in operations:
                    raise ValueError("duplicate J operation id")
                operations[node_ref] = row
            collect(row.get("body"))
            for arm in ("positive", "negative"):
                branch = row.get(arm)
                if isinstance(branch, Mapping):
                    collect(branch.get("body"))

    try:
        collect(function["body"])
    except ValueError:
        return ({
            "rule_code": "source_j_operation_identity_ambiguous",
            "path": "j.intent.intent_function.body",
            "observed_refs_or_types": ["expected:unique operation ids"],
        },)

    def unique_rows(
        rows: object,
        key: str,
    ) -> tuple[dict[str, Mapping[str, Any]], bool]:
        """Index one relation without allowing a later row to overwrite it."""

        indexed: dict[str, Mapping[str, Any]] = {}
        duplicate = False
        if not isinstance(rows, list):
            return indexed, True
        for row in rows:
            identity = row.get(key) if isinstance(row, Mapping) else None
            if not isinstance(identity, str) or not identity:
                # Do not silently drop a malformed relation row.  A dropped
                # row could otherwise make a partial store/update partition
                # appear complete to the later FunctionalBoundary proof.
                duplicate = True
                continue
            if identity in indexed:
                duplicate = True
                continue
            indexed[identity] = row
        return indexed, duplicate

    accesses, duplicate_accesses = unique_rows(boundary["accesses"], "access_id")
    # ``Boundary.effects`` also contains control/state effects whose
    # ``access_ref`` is intentionally null.  They must still have unique
    # effect identities, but they are not entries in the memory-effect lookup
    # used by this store/output join.  Treating null as a relation key made
    # every multi-effect control graph look like a duplicate physical store
    # relation and rejected it before FunctionalBoundary could perform its
    # independent complete check.
    effects: dict[str, Mapping[str, Any]] = {}
    duplicate_effects = False
    seen_effect_ids: set[str] = set()
    raw_effect_rows = boundary["effects"]
    if not isinstance(raw_effect_rows, list):
        duplicate_effects = True
        raw_effect_rows = []
    for effect in raw_effect_rows:
        if not isinstance(effect, Mapping):
            duplicate_effects = True
            continue
        effect_id = effect.get("effect_id")
        if (
            not isinstance(effect_id, str)
            or re.fullmatch(r"e[0-9]+", effect_id) is None
            or effect_id in seen_effect_ids
        ):
            duplicate_effects = True
        else:
            seen_effect_ids.add(effect_id)
        access_ref = effect.get("access_ref")
        if access_ref is None:
            continue
        if not isinstance(access_ref, str) or access_ref in effects:
            duplicate_effects = True
            continue
        effects[access_ref] = effect
    outputs, duplicate_outputs = unique_rows(function["outputs"], "name")
    returns, duplicate_returns = unique_rows(function["returns"], "output")
    mappings, duplicate_mappings = unique_rows(
        boundary["output_mappings"], "output"
    )
    updates, duplicate_updates = unique_rows(boundary["updates"], "node_ref")
    raw_index_bounds = boundary.get("index_bounds")
    index_bounds, duplicate_index_bounds = (
        unique_rows(raw_index_bounds, "node_ref")
        if isinstance(raw_index_bounds, list)
        else ({}, False)
    )
    raw_result_relations = boundary.get("result_relations")
    result_relations, duplicate_result_relations = (
        unique_rows(raw_result_relations, "output")
        if isinstance(raw_result_relations, list)
        else ({}, False)
    )
    if any(
        (
            duplicate_accesses,
            duplicate_effects,
            duplicate_outputs,
            duplicate_returns,
            duplicate_mappings,
            duplicate_updates,
            duplicate_index_bounds,
            duplicate_result_relations,
        )
    ):
        return ({
            "rule_code": "source_store_output_relation_ambiguous",
            "path": "j.boundary",
            "observed_refs_or_types": [
                "expected:unique access/effect/output/update relation identities"
            ],
        },)

    # The Source occurrence and its candidate access are selected only by the
    # already-closed ledger binding, never by tensor name or list position.
    occurrence_by_ordinal = {
        occurrence.ordinal: occurrence for occurrence in shared.occurrences
    }
    binding_by_ordinal: dict[int, Mapping[str, Any]] = {}
    duplicate_binding_ordinals = False
    for row in shared.bindings:
        ordinal = row.get("ordinal") if isinstance(row, Mapping) else None
        if not is_canonical_ordinal(ordinal):
            duplicate_binding_ordinals = True
            continue
        if ordinal in binding_by_ordinal:
            duplicate_binding_ordinals = True
            continue
        binding_by_ordinal[ordinal] = row
    if duplicate_binding_ordinals or set(binding_by_ordinal) != set(occurrence_by_ordinal):
        return ({
            "rule_code": "source_store_output_relation_ambiguous",
            "path": "<source-transfer-ledger>.bindings.ordinal",
            "observed_refs_or_types": [
                "expected:exactly one binding for every Source occurrence"
            ],
        },)
    binding_by_access: dict[str, Mapping[str, Any]] = {}
    for row in shared.bindings:
        access_ref = row.get("access_ref") if isinstance(row, Mapping) else None
        if not isinstance(access_ref, str) or access_ref in binding_by_access:
            return ({
                "rule_code": "source_store_output_relation_ambiguous",
                "path": "<source-transfer-ledger>.bindings",
                "observed_refs_or_types": [
                    "expected:one unique ledger binding per access_ref"
                ],
            },)
        binding_by_access[access_ref] = row
    # Every physical candidate write must be one of the exact Source-bound
    # access occurrences.  An unbound output anchor is not a harmless second
    # spelling of the same store: accepting it would let a candidate create a
    # new producer/owner outside the Source ledger.
    for access_ref, access in accesses.items():
        if access.get("kind") not in {"store", "update"}:
            continue
        binding = binding_by_access.get(access_ref)
        ordinal = binding.get("ordinal") if isinstance(binding, Mapping) else None
        occurrence = occurrence_by_ordinal.get(ordinal)
        if (
            not isinstance(binding, Mapping)
            or not is_canonical_ordinal(ordinal)
            or occurrence is None
            or occurrence.kind != access.get("kind")
            or occurrence.logical_ref != access.get("tensor")
        ):
            return ({
                "rule_code": "source_store_access_unmatched",
                "path": f"j.boundary.accesses[{access_ref}]",
                "observed_refs_or_types": [
                    "expected:physical store/update bound to one Source occurrence"
                ],
            },)
    diagnostics: list[Mapping[str, Any]] = []

    def reject(rule: str, ordinal: int, suffix: str) -> None:
        diagnostics.append({
            "rule_code": rule,
            "path": f"<source-store-binding>[{ordinal}].{suffix}",
            "observed_refs_or_types": ["expected:unique closed Source/J/output relation"],
        })

    # Use the private compiler sidecar to require the direct TTIR store/value
    # edge.  Matching by the exact Source debug coordinate is the only allowed
    # join; no ordinal or operation-order fallback is permitted.
    ttir_rows = tuple(observed.diagnostic_memory_source_locations)
    owner_rows = tuple(observed.diagnostic_memory_owner_evidence)
    tensor_proofs = tuple(observed.diagnostic_tensor_owner_graphs)
    scalar_proofs = tuple(observed.diagnostic_scalar_owner_graphs)
    for occurrence in shared.occurrences:
        if occurrence.kind not in {"store", "update"}:
            continue
        ordinal = occurrence.ordinal
        binding = binding_by_ordinal.get(ordinal)
        access_ref = binding.get("access_ref") if isinstance(binding, Mapping) else None
        access = accesses.get(access_ref)
        if not isinstance(binding, Mapping) or not isinstance(access_ref, str) or not isinstance(access, Mapping):
            reject("source_store_access_binding_missing", ordinal, "access_ref")
            continue
        source_argument = (
            source_facts.argument_by_position(occurrence.pointer_position)
            if isinstance(source_facts, SourceFacts)
            else None
        )
        runtime_parameter = runtime_parameters.get(occurrence.pointer_position)
        if (
            not isinstance(source_argument, Mapping)
            or not isinstance(runtime_parameter, Mapping)
            or runtime_parameter.get("logical_ref") != occurrence.logical_ref
            or runtime_parameter.get("kind") not in {"tensor", "pointer"}
            or runtime_parameter.get("rank") != occurrence.tensor_rank
            or (
                isinstance(source_argument.get("mode"), str)
                and runtime_parameter.get("mode") != source_argument.get("mode")
            )
        ):
            reject("source_store_physical_abi_unproved", ordinal, "pointer")
            continue
        node_ref = access.get("node_ref")
        if not isinstance(node_ref, str) or node_ref not in operations:
            reject("source_store_j_operation_missing", ordinal, "node_ref")
            continue
        if access.get("kind") != occurrence.kind or access.get("tensor") != occurrence.logical_ref:
            reject("source_store_access_source_mismatch", ordinal, "access")
            continue
        if (
            (access.get("predicate") is not None) != occurrence.masked
            or (
                occurrence.kind != "load"
                and access.get("fill_value") is not None
            )
            or (
                occurrence.kind == "load"
                and occurrence.masked
                and access.get("fill_value") is None
            )
        ):
            reject("source_store_access_mask_or_fill_mismatch", ordinal, "access")
            continue
        semantic_access = occurrence.semantic_access
        if (
            semantic_access is None
            or access.get("semantic_access_authorization")
            != semantic_access.token
        ):
            reject("source_store_source_occurrence_missing", ordinal, "source_occurrence")
            continue
        binding_authorizations = binding.get("semantic_authorizations")
        owner_receipt = (
            binding_authorizations.get("owner")
            if isinstance(binding_authorizations, Mapping)
            else None
        )
        if (
            not isinstance(owner_receipt, Mapping)
            or owner_receipt.get("status") != "pass"
            or access.get("semantic_owner_authorization") != owner_receipt.get("token")
        ):
            reject("source_store_owner_authorization_missing", ordinal, "owner")
            continue
        # Reconsume the exact J owner commitment here instead of relying on a
        # previous diagnostic pass.  This verifies the declared node's
        # producer graph, typed value, coordinate/domain/predicate contract,
        # and any Source-authorized outside-domain relation at the same
        # admission boundary that binds the final output.  It never issues a
        # token and never reads a candidate body as repair material.
        try:
            validate_source_owner_binding(
                function,
                signature["parameters"],
                access,
                boundary,
                value_dimension_relations=j.get("value_dimension_relations"),
            )
        except (ValueError, TypeError, KeyError, RecursionError):
            reject("source_store_j_owner_producer_unproved", ordinal, "operation")
            continue
        effect = effects.get(access_ref)
        expected_effect_id = binding.get("effect_id")
        if (
            not isinstance(effect, Mapping)
            or effect.get("target") != occurrence.logical_ref
            or effect.get("access_ref") != access_ref
            or effect.get("node_ref") != node_ref
            or effect.get("kind") != ("write" if occurrence.kind == "store" else "update")
            or effect.get("effect_id") != expected_effect_id
        ):
            reject("source_store_effect_binding_mismatch", ordinal, "effect")
            continue
        if binding.get("effect_order", ordinal) != ordinal:
            reject("source_store_effect_binding_mismatch", ordinal, "effect_order")
            continue

        # A logical ``update`` is lowered to a physical TTIR store.  The
        # compiler sidecar therefore remains a store row; the Source ledger
        # kind must not be compared directly to that physical spelling.
        compiler_kind = "store" if occurrence.kind == "update" else occurrence.kind
        compiler_indices = [
            index
            for index, row in enumerate(ttir_rows)
            if isinstance(row, tuple)
            and len(row) == 5
            and row[0] == compiler_kind
            and row[1] is occurrence.masked
            and row[2] == semantic_access.source_line
            and row[3] == semantic_access.source_column
            and row[4] == semantic_access.source_call_path
        ]
        if len(compiler_indices) != 1 or len(owner_rows) != len(ttir_rows):
            reject("source_store_ttir_producer_unavailable", ordinal, "ttir_producer")
            continue
        compiler_index = compiler_indices[0]
        owner = owner_rows[compiler_index]
        if (
            not isinstance(owner, tuple)
            or len(owner) != 9
            or owner[0] != compiler_kind
            or owner[1] is not occurrence.masked
            or owner[2] != semantic_access.source_line
            or owner[3] != semantic_access.source_column
            or not isinstance(owner[4], str)
            or not owner[4]
            or owner[7] is not True
            or owner[8] != semantic_access.source_call_path
        ):
            reject("source_store_ttir_producer_unproved", ordinal, "ttir_producer")
            continue
        tensor_proof_matches = [
            proof for proof in tensor_proofs
            if isinstance(proof, Mapping)
            and proof.get("kind") == compiler_kind
            and proof.get("line") == semantic_access.source_line
            and proof.get("column") == semantic_access.source_column
            and proof.get("call_path") == semantic_access.source_call_path
            and proof.get("pointer_position") == occurrence.pointer_position
        ]
        # Rank-0 stores are authorized by the separate scalar owner DAG.  It
        # must expose the same occurrence/producer/type contract as the
        # tensor proof; choosing a proof by operation order, family, or shape
        # coincidence is not an acceptable fallback.  Keep both namespaces in
        # one ambiguity check: a duplicate tensor/scalar observation for the
        # same Source location must not be silently resolved by precedence.
        scalar_proof_matches = [
            proof for proof in scalar_proofs
            if isinstance(proof, Mapping)
            and proof.get("kind") == compiler_kind
            and proof.get("line") == semantic_access.source_line
            and proof.get("column") == semantic_access.source_column
            and proof.get("call_path") == semantic_access.source_call_path
            and proof.get("pointer_position") == occurrence.pointer_position
        ]
        proof_matches = [*tensor_proof_matches, *scalar_proof_matches]
        if (
            len(proof_matches) != 1
            or not isinstance(proof_matches[0].get("stored_value_producer_id"), int)
            or not isinstance(proof_matches[0].get("stored_value_type"), str)
            or not isinstance(owner[4], str)
            or proof_matches[0].get("stored_value_producer_operation") != owner[4]
        ):
            reject("source_store_ttir_value_producer_unproved", ordinal, "ttir_producer")
            continue
        compiler_proof = proof_matches[0]
        if (
            "coordinate" not in compiler_proof
            or "physical_shape" not in compiler_proof
            or not isinstance(compiler_proof.get("physical_shape"), (tuple, list))
            or (
                occurrence.masked
                and compiler_proof.get("predicate") is None
            )
            or (
                not occurrence.masked
                and compiler_proof.get("predicate") is not None
            )
        ):
            reject("source_store_ttir_coordinate_unproved", ordinal, "ttir_producer")
            continue

        # The private TTIR type is a compiler spelling, while SourceFacts/J
        # use the closed logical dtype spelling.  Only an exact element-type
        # projection is accepted; an unknown or aggregate type is not
        # silently treated as compatible.
        ttir_type = compiler_proof["stored_value_type"]
        type_match = re.search(
            r"(?:^|x)(bf16|f(?:16|32|64)|i(?:1|8|16|32|64)|ui(?:8|16|32))(?:>|$)",
            ttir_type,
        )
        if type_match is None:
            reject("source_store_ttir_value_type_unproved", ordinal, "ttir_producer")
            continue

        output = outputs.get(occurrence.logical_ref)
        mapping = mappings.get(occurrence.logical_ref)
        returned = returns.get(occurrence.logical_ref)
        source_output = (
            source_facts.argument_by_position(occurrence.pointer_position)
            if isinstance(source_facts, SourceFacts)
            else None
        )
        if not isinstance(source_output, Mapping):
            reject("source_store_final_output_binding_missing", ordinal, "output")
            continue
        try:
            source_dtype = canonicalize_dtype(
                source_output.get("dtype"), context="Source output dtype"
            )
        except (TypeError, ValueError):
            source_dtype = None
        raw_shape_contract = source_output.get("shape_contract")
        raw_shape = source_output.get("shape")
        source_shape_contract: list[str | int] | None = None
        if isinstance(raw_shape, list):
            contract = raw_shape if raw_shape_contract is None else raw_shape_contract
            if isinstance(contract, list) and len(contract) == len(raw_shape):
                dimension_aliases = {
                    coordinate: f"x{index}"
                    for index, (coordinate, _value)
                    in enumerate(source_facts.shape_coordinates)
                } if isinstance(source_facts, SourceFacts) else {}
                projected_shape: list[str | int] = []
                shape_valid = True
                for contract_axis, static_extent in zip(
                    contract, raw_shape, strict=True
                ):
                    if type(contract_axis) is int:
                        if contract_axis != static_extent:
                            shape_valid = False
                            break
                        projected_shape.append(contract_axis)
                    elif (
                        isinstance(contract_axis, str)
                        and contract_axis in dimension_aliases
                    ):
                        projected_shape.append(dimension_aliases[contract_axis])
                    else:
                        shape_valid = False
                        break
                if shape_valid:
                    source_shape_contract = projected_shape
        if (
            not isinstance(output, Mapping)
            or not isinstance(mapping, Mapping)
            or not isinstance(returned, Mapping)
            or not isinstance(source_output, Mapping)
            or source_output.get("mode") not in {"output", "inout"}
            or source_dtype is None
            or output.get("dtype") != source_dtype
            or type_match.group(1) != source_dtype
            or compiler_proof.get("dtype") != source_dtype
            or not isinstance(output.get("shape"), list)
            or source_shape_contract is None
            or output.get("shape") != source_shape_contract
        ):
            reject("source_store_final_output_binding_missing", ordinal, "output")
            continue
        value = mapping.get("value")
        mapping_node_ref = mapping.get("node_ref")
        if (
            not isinstance(value, str)
            or returned.get("value") != value
            or not isinstance(mapping_node_ref, str)
        ):
            reject("source_store_final_output_binding_mismatch", ordinal, "output_mapping")
            continue
        mapped_operation = operations.get(mapping_node_ref)
        mapped_update = updates.get(mapping_node_ref)
        mapped_results = (
            [mapped_operation.get("result")]
            if isinstance(mapped_operation, Mapping)
            and isinstance(mapped_operation.get("result"), str)
            else mapped_operation.get("result")
            if isinstance(mapped_operation, Mapping)
            else None
        )
        # A physical store may be one exact slice of a logical scatter result.
        # In that profile its own J producer intentionally produces a typed
        # values slice, not the final scatter result named by output_mapping.
        # Defer only this explicitly identified relation to the existing
        # complete partition proof; do not treat a tensor-name match or a
        # single store as sufficient.
        deferred_logical_update = (
            isinstance(mapped_operation, Mapping)
            and mapped_operation.get("op") == "scatter"
            and isinstance(mapped_update, Mapping)
            and mapped_update.get("node_ref") == mapping_node_ref
            and isinstance(mapped_results, list)
            and len(mapped_results) == 1
            and mapped_results[0] == value
        )
        # Source owner authorization binds ``access.node_ref`` to the logical
        # producer.  canonical_output_records additionally permits a distinct
        # output-mapping node_ref when it is one unique real store/scatter
        # anchor consuming that producer's returned value.  Keep these two
        # identities separate; neither may be selected by name or position.
        producer_operation = operations[node_ref]
        producer_result = producer_operation.get("result")
        producer_results = (
            [producer_result]
            if isinstance(producer_result, str)
            else producer_result
        )
        assembly_store_relation = (
            isinstance(mapped_operation, Mapping)
            and mapped_operation.get("op") == "concat"
            and mapped_results == [value]
            and isinstance(producer_results, list)
            and len(producer_results) == 1
            and isinstance(mapped_operation.get("operands"), list)
            and mapped_operation["operands"].count(producer_results[0]) == 1
            and mapping_node_ref != node_ref
        )
        if (
            (not isinstance(producer_results, list) or value not in producer_results)
            and not deferred_logical_update
            and not assembly_store_relation
        ):
            reject("source_store_j_output_producer_mismatch", ordinal, "operation")
            continue
        if mapping_node_ref == node_ref:
            pass
        elif deferred_logical_update:
            # The scatter/update itself is not a physical store anchor.  Its
            # output relation is admitted only if the final
            # project_functional_boundary() pass below proves every physical
            # store, coordinate, typed values slice, target, and output
            # binding jointly.
            pass
        elif assembly_store_relation:
            # The physical store is authorized by its exact slice producer;
            # the output mapping names the separately authored logical
            # concat.  Complete coverage, order, coordinates and types are
            # proved jointly by FunctionalBoundary below.
            pass
        else:
            anchor = operations.get(mapping_node_ref)
            anchor_accesses = [
                row for row in boundary["accesses"]
                if (
                    isinstance(row, Mapping)
                    and row.get("kind") in {"store", "update"}
                    and row.get("tensor") == occurrence.logical_ref
                    and row.get("node_ref") == mapping_node_ref
                )
            ]
            anchor_effects = [
                effects.get(row.get("access_id"))
                for row in anchor_accesses
                if isinstance(row.get("access_id"), str)
            ]
            if (
                not isinstance(anchor, Mapping)
                or anchor.get("op") not in {"store", "scatter", "masked_scatter"}
                or len(anchor_accesses) != 1
                or len(anchor_effects) != 1
                or not isinstance(anchor_effects[0], Mapping)
                or anchor_effects[0].get("kind") != "write"
                or anchor_effects[0].get("target") != occurrence.logical_ref
                or anchor_effects[0].get("node_ref") != mapping_node_ref
                or value not in (
                    anchor.get("operands")
                    if isinstance(anchor.get("operands"), list)
                    else ()
                )
            ):
                reject("source_store_final_output_binding_mismatch", ordinal, "output_mapping")
                continue

        if occurrence.kind == "update":
            operation = producer_operation
            update = updates.get(node_ref)
            operands = operation.get("operands")
            attrs = operation.get("attrs")
            atomic = attrs.get("atomic") if isinstance(attrs, Mapping) else None
            if atomic == "cas":
                expected_operands = 4
                result_values = operation.get("result")
                final_update_value = (
                    result_values[0]
                    if isinstance(result_values, list) and len(result_values) == 2
                    else None
                )
                compare_value = (
                    result_values[1]
                    if isinstance(result_values, list) and len(result_values) == 2
                    else None
                )
            else:
                expected_operands = 3
                final_update_value = (
                    operation.get("result")
                    if isinstance(operation.get("result"), str)
                    else None
                )
                compare_value = None
            reduce_name = attrs.get("reduce") if isinstance(attrs, Mapping) else None
            expected_combiner = {
                "none": "overwrite",
                "add": "add",
            }.get(reduce_name)
            index_bound = index_bounds.get(node_ref)
            result_relation = result_relations.get(occurrence.logical_ref)
            if (
                not isinstance(update, Mapping)
                or not isinstance(operands, list)
                or len(operands) != expected_operands
                or not isinstance(attrs, Mapping)
                or update.get("base") != operands[0]
                or update.get("indices") != operands[1]
                or update.get("values") != operands[3 if atomic == "cas" else 2]
                or (update.get("compare") != operands[2] if atomic == "cas" else update.get("compare") is not None)
                or (update.get("observed") != compare_value if atomic == "cas" else update.get("observed") is not None)
                or update.get("axis") != attrs.get("axis")
                or update.get("combiner") != expected_combiner
                or update.get("duplicate_policy") not in {"forbid", "combine", "one_of"}
                or update.get("order") not in {"input_order", "unordered"}
                or (
                    update.get("combiner") == "overwrite"
                    and update.get("duplicate_policy") == "combine"
                )
                or (
                    update.get("combiner") == "add"
                    and update.get("duplicate_policy") == "one_of"
                )
                or (
                    update.get("duplicate_policy") == "one_of"
                    and update.get("order") != "unordered"
                )
                or not isinstance(index_bound, Mapping)
                or index_bound.get("data") != update.get("base")
                or index_bound.get("indices") != update.get("indices")
                or index_bound.get("axis") != update.get("axis")
                or not isinstance(result_relation, Mapping)
                or final_update_value != value
            ):
                reject("source_store_logical_update_mismatch", ordinal, "updates")

    if diagnostics:
        return tuple(diagnostics)

    # FunctionalBoundary already owns the narrow, deterministic partition
    # proofs for multi-store outputs and logical updates.  Consume that
    # existing proof here only after the Source occurrence/TTIR/J joins above
    # are closed.  The projection is ephemeral: it is not serialized, sent to
    # the provider, or substituted for Gate1/Gate2.  Any unsupported or
    # incomplete partition remains a fixed fail-closed admission result.
    try:
        from intentir.ir.functional_boundary import project_functional_boundary
        from intentir.verify.gate1.model import SemanticArchive

        typed_j = SemanticArchive.from_json(dict(j))
        project_functional_boundary(typed_j)
    except Exception:
        return ({
            "rule_code": "source_store_output_partition_unproved",
            "path": "j.boundary.accesses + j.boundary.updates + j.boundary.output_mappings",
            "observed_refs_or_types": [
                "expected:complete Source-authorized physical-store partition"
            ],
        },)

    return tuple(diagnostics)


def validate_source_access_fills_from_source_ledger(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> dict[str, Any]:
    """Validate provider-authored fills without modifying the candidate."""

    original = copy.deepcopy(dict(packet))
    result: dict[str, Any] = {
        "protocol": "intentir.lift-source-access-fill-repair.v1",
        "status": "unavailable",
        "artifact": original,
        "applied": [],
        "diagnostics": [],
    }
    j = original.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    raw_accesses = boundary.get("accesses") if isinstance(boundary, Mapping) else None
    if not isinstance(raw_accesses, list):
        return result

    diagnostics = _source_access_fill_validation_diagnostics(
        original,
        observed,
        ledger=ledger,
        source_facts=source_facts,
    )
    if diagnostics:
        result["status"] = "unproved"
        result["diagnostics"] = list(diagnostics)
        return result
    result["status"] = "pass"
    return result


def _source_access_fill_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint | None,
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Reject missing, extra, or non-provable fills per unique access occurrence."""

    diagnostics = _source_access_fill_validation_diagnostics(
        packet,
        observed,
        ledger=ledger,
        source_facts=source_facts,
    )
    if diagnostics:
        return diagnostics
    return ()


def _closed_observer_exception(exc: Exception) -> str:
    """Return bounded observer diagnostics without exporting host-local paths."""

    detail = str(exc).strip()
    detail = re.sub(r"https?://[^\s,;]+", "<url>", detail)
    detail = re.sub(r"(?:[A-Za-z]:)?/[^\s,;]+", "<path>", detail)
    detail = detail[:512]
    return (
        f"{type(exc).__name__}: {detail}"
        if detail
        else type(exc).__name__
    )


def _compact_j_mask_presence_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint,
) -> tuple[Mapping[str, Any], ...]:
    """Reject total omission of Source-visible masks from an active J draft.

    This is deliberately a one-bit necessary condition, not an occurrence
    projection: frontend optimization may merge or eliminate transfers, so exact
    TTIR counts are not compared.  The Lift owner still chooses every predicate,
    fill, guard, consumer and formula; formal Gate1 checks the complete result.
    """

    pi = packet.get("pi")
    if not isinstance(pi, Mapping) or pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        return ()
    j = packet.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    accesses = boundary.get("accesses") if isinstance(boundary, Mapping) else None
    guards = (
        boundary.get("access_guard_relations")
        if isinstance(boundary, Mapping)
        else None
    )
    if not isinstance(accesses, list) or not isinstance(guards, list):
        return ()

    masked_loads = [
        item
        for item in accesses
        if isinstance(item, Mapping)
        and item.get("kind") == "load"
        and item.get("predicate") is not None
    ]
    masked_store_ids = {
        str(item.get("access_id"))
        for item in accesses
        if isinstance(item, Mapping)
        and item.get("kind") == "store"
        and item.get("predicate") is not None
        and isinstance(item.get("access_id"), str)
    }
    masked_store_ids.update(
        str(item.get("access_ref"))
        for item in guards
        if isinstance(item, Mapping) and isinstance(item.get("access_ref"), str)
    )
    op_counts = dict(observed.op_multiset)
    diagnostics: list[Mapping[str, Any]] = []
    if op_counts.get("tt.load.masked", 0) > 0 and not masked_loads:
        diagnostics.append(
            {
                "rule_code": "j_masked_load_presence",
                "path": (
                    "<source-ttir-obligation> Source has compiler-visible masked "
                    "loads but J.boundary has no predicate-bearing load"
                ),
                "anonymous_handles": [],
                "line": None,
            }
        )
    if op_counts.get("tt.store.masked", 0) > 0 and not masked_store_ids:
        diagnostics.append(
            {
                "rule_code": "j_masked_store_presence",
                "path": (
                    "<source-ttir-obligation> Source has compiler-visible masked "
                    "stores but J.boundary has no predicate/guard-bearing store"
                ),
                "anonymous_handles": [],
                "line": None,
            }
        )
    return tuple(diagnostics)


def _j_operation_names(packet: Mapping[str, Any]) -> frozenset[str]:
    """Return explicitly authored J operation names without interpreting them."""

    names: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            operation = value.get("op")
            if isinstance(operation, str):
                names.add(operation)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    j = packet.get("j")
    if isinstance(j, Mapping):
        visit(j)
    return frozenset(names)


_NONLINEAR_RECURRENCE_OPS = frozenset(
    {
        "math.exp",
        "math.exp2",
        "math.log",
        "math.log2",
        "math.tanh",
    }
)


def _compact_j_recurrence_presence_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint,
) -> tuple[Mapping[str, Any], ...]:
    """Reject omission of an unmistakable nonlinear multi-state recurrence.

    The condition is intentionally weaker than recurrence reconstruction.  It says
    only that a Source loop has at least two carried values, at least two yielded
    values that are not direct pointer increments, and a nonlinear operation in its
    body.  Fixed code neither classifies the carries nor supplies trip/update/yield
    formulas; the Lift owner must author the complete structured loop and formal
    Gate1 remains authoritative.
    """

    pi = packet.get("pi")
    if not isinstance(pi, Mapping) or pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        return ()
    if "loop" in _j_operation_names(packet):
        return ()

    witnesses: list[tuple[int, tuple[str, ...]]] = []
    for ordinal, loop in enumerate(observed.loops):
        body_ops = dict(loop.body_op_multiset)
        non_pointer_yields = tuple(
            value
            for value in loop.yielded_values
            if not value.startswith("tt.addptr[")
        )
        nonlinear = tuple(
            sorted(
                operation
                for operation in _NONLINEAR_RECURRENCE_OPS
                if body_ops.get(operation, 0) > 0
            )
        )
        if (
            len(loop.carried_argument_use_counts) >= 2
            and len(non_pointer_yields) >= 2
            and nonlinear
        ):
            witnesses.append((ordinal, nonlinear))
    if not witnesses:
        return ()
    return (
        {
            "rule_code": "j_nonlinear_multistate_recurrence_presence",
            "path": (
                "<source-ttir-obligation> Source has a compiler-visible nonlinear "
                "multi-state loop recurrence but J contains no structured loop"
            ),
            "anonymous_handles": [],
            "observed_refs_or_types": [
                f"loop[{ordinal}]:{','.join(operations)}"
                for ordinal, operations in witnesses[:4]
            ],
            "line": None,
        },
    )


def _source_stride_axis_diagnostics(
    packet: Mapping[str, Any],
    *,
    ledger: SourceTransferLedger | Mapping[str, Any] | None = None,
    source_facts: SourceFacts | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Reject a stride binding whose axis contradicts Source address evidence.

    ``axis_ref`` is the logical tensor axis whose coordinate is multiplied by
    a representation-only stride carrier.  The existing schema checks only
    that the axis is in range, which is insufficient: a row-stride carrier can
    be validly attached to either axis of a rank-2 tensor while only one choice
    matches the compiled Source address polynomial.  This diagnostic is
    intentionally source-aware and necessary-only.  It uses the closed
    SourceTransferLedger occurrence/access binding plus the candidate's
    already-authored program binding; it never constructs or repairs an
    artifact.

    The closed check currently applies when a carrier is visibly multiplied by
    one direct physical ``program_id`` axis.  More complex address forms are
    left to the formal Gate1/Gate2 path rather than guessed here.
    """

    pi = packet.get("pi")
    j = packet.get("j")
    if not isinstance(pi, Mapping) or not isinstance(j, Mapping):
        return ()
    raw_bindings = pi.get("access_representation_bindings")
    if not isinstance(raw_bindings, list):
        return (
            {
                "rule_code": "source_stride_axis_binding_unproved",
                "path": "pi.access_representation_bindings",
                "anonymous_handles": [],
                "observed_refs_or_types": [
                    "expected:closed access representation binding array"
                ],
                "line": None,
            },
        )

    boundary = j.get("boundary")
    raw_accesses = (
        boundary.get("accesses") if isinstance(boundary, Mapping) else None
    )
    if not isinstance(boundary, Mapping) or not isinstance(raw_accesses, list):
        return (
            {
                "rule_code": "source_stride_axis_binding_unproved",
                "path": "j.boundary.accesses",
                "anonymous_handles": [],
                "observed_refs_or_types": [
                    "expected:closed J access table for address binding"
                ],
                "line": None,
            },
        )

    # Mirror the closed PI relation identity here before doing any Source
    # projection. The formal PI verifier also performs this check, but this
    # source-aware path must not silently filter a malformed/duplicate row and
    # then prove only the surviving subset.
    access_ids: set[str] = set()
    malformed_access_table = False
    for access in raw_accesses:
        access_id = access.get("access_id") if isinstance(access, Mapping) else None
        if (
            not isinstance(access_id, str)
            or re.fullmatch(r"a[0-9]+", access_id) is None
            or access_id in access_ids
        ):
            malformed_access_table = True
        else:
            access_ids.add(access_id)

    relation_rows: list[tuple[int, Mapping[str, Any]]] = []
    seen_relation_rows: set[tuple[str, str, str | None, int | None]] = set()
    malformed_binding_rows: list[Mapping[str, Any]] = []
    address_relations = {"address_recomputed", "pointer_advanced"}
    legal_relations = {
        *address_relations,
        "boolean_storage_byte",
        "guard_lower_bound_elided",
    }
    for binding_index, row in enumerate(raw_bindings):
        if not isinstance(row, Mapping):
            malformed_binding_rows.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}]",
                    "anonymous_handles": [],
                    "observed_refs_or_types": [
                        "expected:object access representation relation"
                    ],
                    "line": None,
                }
            )
            continue
        relation = row.get("relation")
        access_ref = row.get("access_ref")
        carrier = row.get("carrier_ref")
        axis = row.get("axis_ref")
        valid_identity = (
            isinstance(relation, str)
            and relation in legal_relations
            and isinstance(access_ref, str)
            and re.fullmatch(r"a[0-9]+", access_ref) is not None
            and access_ref in access_ids
        )
        address_relation = relation in address_relations
        if address_relation:
            valid_identity = valid_identity and (
                isinstance(carrier, str)
                and re.fullmatch(r"p[0-9]+", carrier) is not None
                and type(axis) is int
                and axis >= 0
            )
        else:
            valid_identity = valid_identity and carrier is None and axis is None
        key = (
            relation if isinstance(relation, str) else "<invalid>",
            access_ref if isinstance(access_ref, str) else "<invalid>",
            carrier if isinstance(carrier, str) else None,
            axis if type(axis) is int else None,
        )
        if not valid_identity or key in seen_relation_rows:
            malformed_binding_rows.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}]",
                    "anonymous_handles": [
                        item for item in (access_ref, carrier)
                        if isinstance(item, str)
                    ],
                    "observed_refs_or_types": [
                        "expected:unique closed access/carrier/axis relation"
                    ],
                    "line": None,
                }
            )
            continue
        seen_relation_rows.add(key)
        relation_rows.append((binding_index, row))

    if malformed_access_table or malformed_binding_rows:
        if malformed_access_table:
            malformed_binding_rows.insert(
                0,
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": "j.boundary.accesses",
                    "anonymous_handles": [],
                    "observed_refs_or_types": [
                        "expected:unique canonical access identities"
                    ],
                    "line": None,
                },
            )
        return tuple(malformed_binding_rows[:64])

    shared_ledger = coerce_source_transfer_ledger(ledger, source_facts=source_facts)
    if shared_ledger.status != "pass" or not shared_ledger.bindings:
        return ()

    try:
        program_projection = canonical_program_coordinate_projection(pi)
    except ProgramCoordinateProjectionError:
        return ()

    # Map a physical launch axis to the explicitly authored logical J axes.
    # The shared projection preserves the distinction between direct
    # program_axes rows (no tensor-axis selector) and relation rows (explicit
    # axis_ref), so this diagnostic never falls back to p* order or layout
    # convention.
    program_axis_to_logical_axes: dict[int, set[int]] = {}
    program_axis_to_domains: dict[int, set[str]] = {}
    for row in program_projection.rows:
        for domain_ref, axis_ref in row.logical_axes:
            if axis_ref is None:
                program_axis_to_domains.setdefault(
                    row.physical_axis, set()
                ).add(domain_ref)
                continue
            if isinstance(axis_ref, int):
                logical_axis = axis_ref
            else:
                match = re.fullmatch(r"x([0-9]+)", axis_ref)
                logical_axis = int(match.group(1)) if match is not None else None
            if logical_axis is not None:
                program_axis_to_logical_axes.setdefault(
                    row.physical_axis, set()
                ).add(logical_axis)
    signature = j.get("signature")
    raw_parameters = (
        signature.get("parameters")
        if isinstance(signature, Mapping)
        else None
    )
    parameter_projection = _candidate_parameter_projections(
        raw_parameters,
        source_facts=source_facts,
    )
    if parameter_projection is None:
        return ()
    runtime_parameters, _physical_parameters, runtime_position_by_name = (
        parameter_projection
    )
    tensor_to_parameter_positions: dict[str, set[int]] = {}
    for runtime_position, parameter in runtime_parameters.items():
        logical_ref = parameter.get("logical_ref")
        if parameter.get("kind") == "tensor" and isinstance(logical_ref, str):
            tensor_to_parameter_positions.setdefault(logical_ref, set()).add(
                runtime_position
            )

    accesses = {
        str(item.get("access_id")): item
        for item in raw_accesses
        if isinstance(item, Mapping) and isinstance(item.get("access_id"), str)
    }

    occurrences_by_ordinal = {
        occurrence.ordinal: occurrence
        for occurrence in shared_ledger.occurrences
    }
    occurrence_by_access: dict[str, TransferOccurrence] = {}
    for row in shared_ledger.bindings:
        ordinal = row.get("ordinal")
        access_ref = row.get("access_ref")
        occurrence = occurrences_by_ordinal.get(ordinal)
        if (
            not isinstance(access_ref, str)
            or occurrence is None
            or access_ref in occurrence_by_access
        ):
            return ()
        occurrence_by_access[access_ref] = occurrence
    diagnostics: list[Mapping[str, Any]] = []

    # Every scalar carrier/program-axis pair proven by one Source address
    # monomial is a mandatory Source->J ownership edge.  A missing candidate
    # row must not be treated as "no diagnostic": that would leave the
    # physical stride outside the authorization chain.  Require one unique
    # address binding and an independently Source-proven logical axis.
    candidate_address_rows = [
        row
        for _binding_index, row in relation_rows
        if row.get("relation") in {"address_recomputed", "pointer_advanced"}
    ]
    source_domain_coordinates = _source_domain_coordinates(boundary)
    domain_extent_symbols = _source_domain_extent_symbols(boundary)
    source_expected_binding_keys: set[tuple[str, str, int]] = set()
    matched_candidate_binding_keys: set[tuple[str, str, int]] = set()
    for access_ref, occurrence in occurrence_by_access.items():
        if occurrence.pointer_position is None:
            continue
        access = accesses.get(access_ref)
        if not isinstance(access, Mapping):
            continue
        for carrier_position, physical_axis in _source_address_carrier_program_axis_pairs(
            occurrence.address_terms
        ):
            if carrier_position == occurrence.pointer_position:
                continue
            carrier_parameter = runtime_parameters.get(carrier_position)
            carrier_name = (
                carrier_parameter.get("name")
                if isinstance(carrier_parameter, Mapping)
                else None
            )
            if (
                not isinstance(carrier_name, str)
                or carrier_parameter.get("kind") != "scalar"
                or carrier_parameter.get("logical_ref") is not None
            ):
                diagnostics.append(
                    {
                        "rule_code": "source_stride_axis_binding_unproved",
                        "path": "pi.access_representation_bindings",
                        "anonymous_handles": [access_ref],
                        "observed_refs_or_types": [
                            "expected:Source-proven scalar stride carrier"
                        ],
                        "line": None,
                    }
                )
                continue

            expected_axes: set[int] = set(
                program_axis_to_logical_axes.get(physical_axis, ())
            )
            for domain_ref in program_axis_to_domains.get(physical_axis, ()):
                coordinate = source_domain_coordinates.get(domain_ref)
                axis = (
                    _index_axis_for_coordinate(
                        access.get("index_expression"),
                        coordinate,
                        extent_symbol=domain_extent_symbols.get(domain_ref),
                    )
                    if coordinate is not None
                    else None
                )
                if axis is not None:
                    expected_axes.add(axis)

            matching_rows = [
                row
                for row in candidate_address_rows
                if row.get("carrier_ref") == carrier_name
                and row.get("access_ref") == access_ref
            ]
            if len(expected_axes) != 1 or len(matching_rows) != 1:
                diagnostics.append(
                    {
                        "rule_code": "source_stride_axis_binding_unproved",
                        "path": "pi.access_representation_bindings",
                        "anonymous_handles": [carrier_name, access_ref],
                        "observed_refs_or_types": [
                            "expected:one Source-proven carrier/access axis binding"
                        ],
                        "line": None,
                    }
                )
                continue
            expected_axis = next(iter(expected_axes))
            source_expected_binding_keys.add(
                (carrier_name, access_ref, expected_axis)
            )
            authored_axis = _axis_ref_number(matching_rows[0].get("axis_ref"))
            if authored_axis != expected_axis:
                diagnostics.append(
                    {
                        "rule_code": "source_stride_axis_mismatch",
                        "path": "pi.access_representation_bindings",
                        "anonymous_handles": [carrier_name, access_ref],
                        "observed_refs_or_types": [
                            "candidate carrier axis differs from Source-proven axis"
                        ],
                        "line": None,
                    }
                )
            elif isinstance(authored_axis, int):
                matched_candidate_binding_keys.add(
                    (carrier_name, access_ref, authored_axis)
                )

    for binding_index, binding in relation_rows:
        if binding.get("relation") not in {"address_recomputed", "pointer_advanced"}:
            continue
        carrier = binding.get("carrier_ref")
        access_ref = binding.get("access_ref")
        authored_axis = _axis_ref_number(binding.get("axis_ref"))
        if (
            not isinstance(carrier, str)
            or not isinstance(access_ref, str)
            or not isinstance(authored_axis, int)
        ):
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}]",
                    "anonymous_handles": [
                        item for item in (carrier, access_ref)
                        if isinstance(item, str)
                    ],
                    "observed_refs_or_types": [
                        "expected:typed Source-bound address relation"
                    ],
                    "line": None,
                }
            )
            continue
        if not re.fullmatch(r"p[0-9]+", carrier):
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].carrier_ref",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        "expected:p* Source ABI carrier"
                    ],
                    "line": None,
                }
            )
            continue
        access = accesses.get(access_ref)
        if not isinstance(access, Mapping):
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].access_ref",
                    "anonymous_handles": [access_ref],
                    "observed_refs_or_types": [
                        "expected:address relation bound to a J access"
                    ],
                    "line": None,
                }
            )
            continue
        occurrence = occurrence_by_access.get(access_ref)
        if occurrence is None:
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].access_ref",
                    "anonymous_handles": [access_ref],
                    "observed_refs_or_types": [
                        "expected:address relation bound to one Source occurrence"
                    ],
                    "line": None,
                }
            )
            continue
        tensor = access.get("tensor")
        tensor_positions = tensor_to_parameter_positions.get(tensor, set())
        if len(tensor_positions) != 1:
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].carrier_ref",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        "expected:unique Source physical ABI tensor position"
                    ],
                    "line": None,
                }
            )
            continue
        base_position = next(iter(tensor_positions))
        carrier_runtime_position = runtime_position_by_name.get(carrier)
        if carrier_runtime_position is None:
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].carrier_ref",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        "expected:declared Source-runtime scalar carrier"
                    ],
                    "line": None,
                }
            )
            continue
        access_kind = access.get("kind")
        if occurrence.kind != access_kind:
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].access_ref",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        "expected:Source access kind matches address relation"
                    ],
                    "line": None,
                }
            )
            continue
        if occurrence.logical_ref is not None and occurrence.logical_ref != tensor:
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].access_ref",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        "expected:Source tensor owner matches address relation"
                    ],
                    "line": None,
                }
            )
            continue
        if occurrence.pointer_position != base_position:
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].access_ref",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        "expected:Source pointer base matches address relation"
                    ],
                    "line": None,
                }
            )
            continue
        expected_axes: set[int] = set()
        polynomial = occurrence.address_terms
        carrier_program_axes = {
            physical_axis
            for position, physical_axis in
            _source_address_carrier_program_axis_pairs(polynomial)
            if position == carrier_runtime_position
        }
        if not carrier_program_axes:
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].carrier_ref",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        "expected:carrier/program-axis product in Source address polynomial"
                    ],
                    "line": None,
                }
            )
            continue
        for physical_axis in carrier_program_axes:
            expected_axes.update(
                program_axis_to_logical_axes.get(physical_axis, ())
            )
            for domain_ref in program_axis_to_domains.get(physical_axis, ()):
                coordinate = _source_domain_coordinates(boundary).get(domain_ref)
                axis = (
                    _index_axis_for_coordinate(
                        access.get("index_expression"), coordinate
                    )
                    if coordinate is not None
                    else None
                )
                if axis is not None:
                    expected_axes.add(axis)
        if len(expected_axes) != 1:
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unproved",
                    "path": f"pi.access_representation_bindings[{binding_index}].axis_ref",
                    "anonymous_handles": [carrier, access_ref],
                    "observed_refs_or_types": [
                        "expected:one Source-proven logical tensor axis"
                    ],
                    "line": None,
                }
            )
            continue
        expected_axis = next(iter(expected_axes))
        source_expected_binding_keys.add(
            (carrier, access_ref, expected_axis)
        )
        if authored_axis == expected_axis:
            matched_candidate_binding_keys.add(
                (carrier, access_ref, authored_axis)
            )
            continue
        diagnostics.append(
            {
                "rule_code": "source_stride_axis_mismatch",
                "path": (
                    "<source-ttir-obligation> "
                    f"pi.access_representation_bindings[{binding_index}].axis_ref"
                ),
                "anonymous_handles": [carrier, access_ref],
                "observed_refs_or_types": [
                    f"source-bound stride carrier {carrier} uses the direct launch axis",
                    f"candidate program binding resolves that launch axis to tensor axis {expected_axis}",
                    f"authored axis_ref={authored_axis}",
                ],
                "line": None,
            }
        )
    for binding_index, binding in relation_rows:
        if binding.get("relation") not in {"address_recomputed", "pointer_advanced"}:
            continue
        carrier = binding.get("carrier_ref")
        access_ref = binding.get("access_ref")
        axis = binding.get("axis_ref")
        key = (
            carrier if isinstance(carrier, str) else "<invalid>",
            access_ref if isinstance(access_ref, str) else "<invalid>",
            axis if type(axis) is int else -1,
        )
        if key not in source_expected_binding_keys:
            diagnostics.append(
                {
                    "rule_code": "source_stride_axis_binding_unowned",
                    "path": f"pi.access_representation_bindings[{binding_index}]",
                    "anonymous_handles": [
                        item for item in (carrier, access_ref)
                        if isinstance(item, str)
                    ],
                    "observed_refs_or_types": [
                        "expected:address relation exactly covered by Source carrier/access/axis proof"
                    ],
                    "line": None,
                }
            )
    if source_expected_binding_keys != matched_candidate_binding_keys:
        diagnostics.append(
            {
                "rule_code": "source_stride_axis_binding_unproved",
                "path": "pi.access_representation_bindings",
                "anonymous_handles": [],
                "observed_refs_or_types": [
                    "expected:complete one-to-one Source carrier/access/axis partition"
                ],
                "line": None,
            }
        )
    return tuple(diagnostics[:64])


def _compact_j_terminal_control_presence_diagnostics(
    packet: Mapping[str, Any],
    observed: TTIRFingerprint,
) -> tuple[Mapping[str, Any], ...]:
    """Reject total omission of Source-visible terminal effect control."""

    pi = packet.get("pi")
    if not isinstance(pi, Mapping) or pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        return ()
    effectful_forks = tuple(
        fork
        for fork in observed.terminal_forks
        if fork.true_arm.effect_order and fork.false_arm.effect_order
    )
    if not effectful_forks or "effect_if" in _j_operation_names(packet):
        return ()
    return (
        {
            "rule_code": "j_terminal_effect_control_presence",
            "path": (
                "<source-ttir-obligation> Source has compiler-visible terminal "
                "effect control but J contains no effect_if"
            ),
            "anonymous_handles": [],
            "observed_refs_or_types": [
                f"two_sided_effectful_terminal_fork_count:{len(effectful_forks)}"
            ],
            "line": None,
        },
    )


def _unowned_source_runtime_carrier_diagnostics(
    packet: Mapping[str, Any],
    ttir: str,
    *,
    source_runtime_parameter_uses: object = None,
) -> tuple[Mapping[str, Any], ...]:
    """Check canonical address-carrier ownership against exact Source use."""

    pi = packet.get("pi")
    j = packet.get("j")
    if (
        not isinstance(j, Mapping)
        or not isinstance(pi, Mapping)
        or pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA
    ):
        return ()
    signature = j.get("signature")
    parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    if not isinstance(parameters, list):
        return ()

    access_carriers = {
        item.get("carrier_ref")
        for item in pi.get("access_representation_bindings", [])
        if isinstance(item, Mapping) and isinstance(item.get("carrier_ref"), str)
    }
    source_syntax_uses = {
        item for item in source_runtime_parameter_uses if isinstance(item, str)
    } if isinstance(source_runtime_parameter_uses, list) else set()
    diagnostics: list[Mapping[str, Any]] = []
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            continue
        name = parameter.get("name")
        if (
            not isinstance(name, str)
            or parameter.get("kind") not in {"pointer", "scalar"}
            or parameter.get("logical_ref") is not None
        ):
            continue
        token = re.compile(rf"%{re.escape(name)}(?![A-Za-z0-9_.$])")
        source_used_in_ttir = len(token.findall(ttir)) > 1
        source_used_in_syntax = name in source_syntax_uses
        source_used = source_used_in_ttir or source_used_in_syntax
        observed_use = (
            "source-ttir-runtime-use"
            if source_used_in_ttir
            else "source-syntax-runtime-use"
        )
        owned = name in access_carriers
        if source_used and not owned:
            diagnostics.append(
                {
                    "rule_code": "source_runtime_carrier_without_access_owner",
                    "path": (
                        "<source-ttir-obligation> "
                        "pi.access_representation_bindings: "
                        f"Source-used runtime carrier {name} has logical_ref=null "
                        "but no authored address ownership row"
                    ),
                    "anonymous_handles": [name],
                    "observed_refs_or_types": [observed_use],
                    "line": None,
                }
            )
        elif owned and not source_used:
            diagnostics.append(
                {
                    "rule_code": "source_unused_runtime_carrier_has_access_owner",
                    "path": (
                        "<source-ttir-obligation> "
                        "pi.access_representation_bindings: "
                        f"runtime carrier {name} has an address ownership row, "
                        "but the selected Source has no body use"
                    ),
                    "anonymous_handles": [name],
                    "observed_refs_or_types": ["source-ttir-runtime-unused"],
                    "line": None,
                }
            )
    return tuple(diagnostics[:64])


__all__ = [
    "SAFE_SOURCE_PREPARATION_REASONS",
    "source_preparation_failure_reason",
    "source_preparation_failure_stage",
    "SOURCE_TTIR_RECONCILE_PROTOCOL",
    "SOURCE_TTIR_OBSERVATION_PROTOCOL",
    "bind_source_pi",
    "materialize_source_program_domains",
    "observe_source_ttir",
    "observe_anonymous_source_ttir",
    "validate_source_semantic_authorizations",
    "validate_source_store_output_bindings",
    "validate_source_access_fills_from_source_ledger",
    "reconcile_candidate_source_ttir",
    "reorder_boundary_effects_from_source_ledger",
]
