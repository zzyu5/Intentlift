"""Dependent, fail-closed TypeCheck for canonical ``(J, C, pi)`` artifacts.

The checker enforces explicit ownership, reference and representation
authorization rules.  It intentionally does not prove Source/J equivalence,
run an SMT solver, or decide whether a generated Replay behaves correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Iterable, Mapping

from intentir.ir.canonical_j import (
    CanonicalValueType,
    canonicalize_semantic_archive,
    direct_unit_dimension_refs,
    j_elaboration,
)
from intentir.ir.canonical_construction import (
    CanonicalHandleAllocator,
    canonical_operation_records,
    canonical_value_type_records,
)
from intentir.ir.intent_op_contract import (
    map_binary_result_dtype,
    map_unary_result_dtype,
    CANONICAL_J_OPS,
    RAGGED_ARITY,
    _broadcast_projection,
    canonicalize_dtype,
    is_closed_boolean_dtype,
    is_closed_dtype,
    is_closed_floating_dtype,
    is_closed_integer_dtype,
    is_consumer_reduction_identity,
    is_non_boolean_integer_dtype,
    is_ragged_carrier_dtype,
    IntentOpContractError,
    operation_attr_contract,
    operation_attr_enum,
    RAGGED_OPS,
    STRUCTURED_INTENT_OPS,
    validate_const_literal,
)
from intentir.ir.model import intent_op_producers, walk_intent_op_scopes
from intentir.ir.contract_ir import ContractIRProjectionError, project_contract_ir
from intentir.ir.dim_expr import (
    DimExprError,
    DimensionConstraintSet,
    dimension_expression_references,
    equivalent_dimension_expr,
    parse_dimension_expr,
)
from intentir.ir.dimension_relation import (
    DimensionRelationError,
    declared_dimension_handles,
    validate_dimension_relations,
)
from intentir.ir.functional_boundary import (
    _numpy_broadcast_positions,
    complete_conditional_effect_outputs,
    functional_boundary_closure_issues,
    functional_consumer_extension_requirements,
    functional_reduction_padding_issues,
    resolve_consumer_local_tile_axis,
)
from intentir.ir.value_dimension import (
    ValueDimensionRelationError,
    validate_value_dimension_relations,
)
from intentir.verify.gate1.model import (
    AccessGuardPolarity,
    AccessKind,
    ImplementationConfig,
    ParameterKind,
    ParameterMode,
    RefinementKind,
    ResultRelationKind,
    ScheduleSlotKind,
    SemanticArchive,
    implementation_value_matches,
)
from intentir.verify.gate1.pi import (
    NORMALIZATION_CLASS_SCOPES,
    NORMALIZATION_WITNESS_SCHEMA,
    PER_PROGRAM_SCALAR_KEEPDIMS_CLASS,
    ProgramCoordinateProjectionError,
    RealizationWitness,
    RealizationWitnessError,
    canonical_program_coordinate_projection,
    closed_consumer_identity_extensions,
    per_program_scalar_keepdims_reductions,
    zero_lower_bound_coordinates_from_j,
    validate_normalization_bindings_against_j,
)


TYPECHECK_PROTOCOL = "intentir.scrc.typecheck.v32"
_MAP_UNARY_FNS = operation_attr_enum("map_unary", "fn")
_MAP_BINARY_FNS = operation_attr_enum("map_binary", "fn")
_MAP_ATTRS = frozenset(operation_attr_contract("map_unary")["allowed"])
_GATHER_MODES = operation_attr_enum("gather", "mode")
_SCATTER_MODES = operation_attr_enum("scatter", "mode")
_SCATTER_REDUCES = operation_attr_enum("scatter", "reduce")
_SCATTER_ATOMIC_KINDS = operation_attr_enum("scatter", "atomic")
_SCATTER_ORDERINGS = operation_attr_enum("scatter", "ordering")
_SCATTER_MEMORY_SCOPES = operation_attr_enum("scatter", "memory_scope")
_SCATTER_CAS_RETRIES = operation_attr_enum("scatter", "cas_retry")
_REDUCE_CONTRACT = operation_attr_contract("reduce")
_REDUCE_FNS = frozenset(
    _REDUCE_CONTRACT["properties"]["fn"].get("enum", ())
)
_REDUCE_ASSOCIATIVITY = frozenset(
    _REDUCE_CONTRACT["properties"]["assoc"].get("enum", ())
)
_REDUCE_ATTRS = frozenset(_REDUCE_CONTRACT["allowed"])
_PAD_CONTRACT = operation_attr_contract("pad")
_PAD_ATTRS = frozenset(_PAD_CONTRACT["allowed"])
_PAD_REQUIRED_ATTRS = frozenset(_PAD_CONTRACT["required"])
_CONTRACTION_ATTRS = frozenset(
    operation_attr_contract("contraction")["allowed"]
)
_TILE_ATTRS = frozenset(operation_attr_contract("tile")["allowed"])
_SCAN_FNS = operation_attr_enum("scan", "fn")
_ARG_REDUCE_FNS = operation_attr_enum("arg_reduce", "fn")
_SORT_RETURN_MODES = operation_attr_enum("sort", "return")
_POOL_FNS = operation_attr_enum("pool", "fn")
_UPSAMPLE_MODES = operation_attr_enum("upsample", "mode")
_EINSUM_TERM = re.compile(r"^[A-Za-z]*$")
_KNOWN_OPERATION_CONTRACTS = CANONICAL_J_OPS - STRUCTURED_INTENT_OPS


class TypeCheckStatus(str, Enum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"


class TypeCheckErrorCode(str, Enum):
    J_ELABORATION_FAILED = "j_elaboration_failed"
    J_VALUE_TYPE_INCOMPLETE = "j_value_type_incomplete"
    OUTPUT_BINDING_INCOMPLETE = "output_binding_incomplete"
    OUTPUT_BINDING_TYPE_MISMATCH = "output_binding_type_mismatch"
    KNOWN_OP_CONTRACT_INVALID = "known_op_contract_invalid"
    DUPLICATE_J_REF = "duplicate_j_ref"
    INVALID_C_SLOT = "invalid_c_slot"
    C_REQUIRED_VALUE_MISSING = "c_required_value_missing"
    C_VALUE_TYPE_MISMATCH = "c_value_type_mismatch"
    C_SUBJECT_UNKNOWN = "c_subject_unknown"
    C_AXIS_INVALID = "c_axis_invalid"
    INVALID_ACCESS_FILL = "invalid_access_fill"
    REDUCTION_BOUNDARY_INVALID = "reduction_boundary_invalid"
    FUNCTIONAL_BOUNDARY_INCOMPLETE = "functional_boundary_incomplete"
    PADDED_MEAN_COUNT_UNPROVEN = "padded_mean_count_unproven"
    LOGICAL_INPUT_UNUSED = "logical_input_unused"
    BOUNDARY_VALUE_NOT_DOMINATING = "boundary_value_not_dominating"
    ACCESS_GUARD_RELATION_INVALID = "access_guard_relation_invalid"
    EFFECT_CONDITIONAL_INVALID = "effect_conditional_invalid"
    VALUE_CONDITIONAL_INVALID = "value_conditional_invalid"
    TERMINAL_CONTROL_INVALID = "terminal_control_invalid"
    REPRESENTATION_CARRIER_J_USE_FORBIDDEN = (
        "representation_carrier_j_use_forbidden"
    )
    RESULT_RELATION_INCOMPATIBLE = "result_relation_incompatible"
    UNRESOLVED_ANCHOR = "unresolved_anchor"
    ROUTE_CYCLE = "route_cycle"
    ROUTE_SLOT_OUT_OF_RANGE = "route_slot_out_of_range"
    ROUTE_TYPE_MISMATCH = "route_type_mismatch"
    ROUTE_EXTENT_TRANSFORM_INVALID = "route_extent_transform_invalid"
    PROGRAM_AXIS_BINDING_INVALID = "program_axis_binding_invalid"
    PI_NORMALIZATION_BINDING_INVALID = "pi_normalization_binding_invalid"
    PROGRAM_LINEARIZATION_INVALID = "program_linearization_invalid"
    TRANSFER_ADDRESS_ROUTE_MISSING = "transfer_address_route_missing"
    INVALID_SHARING = "invalid_sharing"
    ITERATION_WITNESS_INVALID = "iteration_witness_invalid"
    ITERATION_WITNESS_MISSING = "iteration_witness_missing"
    CONSUMER_EXTENSION_INVALID = "consumer_extension_invalid"
    CONSUMER_EXTENSION_MISSING = "consumer_extension_missing"
    ACCESS_REPRESENTATION_INVALID = "access_representation_invalid"
    DIMENSION_CONSTRAINT_INVALID = "dimension_constraint_invalid"
    VALUE_DIMENSION_RELATION_INVALID = "value_dimension_relation_invalid"
    DIMENSION_RELATION_INVALID = "dimension_relation_invalid"
    PAD_SHAPE_INVALID = "pad_shape_invalid"
    PAD_SHAPE_CONSTRAINT_MISSING = "pad_shape_constraint_missing"
    TILE_SHAPE_INVALID = "tile_shape_invalid"
    LOOP_CARRIER_INVALID = "loop_carrier_invalid"
    PHYSICAL_ITERATION_INVALID = "physical_iteration_invalid"
    PHYSICAL_LOOP_CARRIER_INVALID = "physical_loop_carrier_invalid"
    OBSERVABLE_REQUIRES_J_ANCHOR = "observable_requires_j_anchor"
    AUTHORIZATION_SUBJECT_UNKNOWN = "authorization_subject_unknown"
    ABSENT_CARRIER_SEMANTIC_USE_FORBIDDEN = (
        "absent_carrier_semantic_use_forbidden"
    )
    REFINEMENT_AUTH_MISSING = "refinement_auth_missing"
    REFINEMENT_AUTH_UNKNOWN = "refinement_auth_unknown"
    REFINEMENT_AUTH_KIND_MISMATCH = "refinement_auth_kind_mismatch"
    REFINEMENT_AUTH_SUBJECT_MISMATCH = "refinement_auth_subject_mismatch"
    REFINEMENT_AUTH_TYPE_MISMATCH = "refinement_auth_type_mismatch"
    REFINEMENT_AUTH_INVALID_WIDEN = "refinement_auth_invalid_widen"
    REFINEMENT_TOPOLOGY_INVALID = "refinement_topology_invalid"
    GLOBAL_SAFE_AUTH_FORBIDDEN = "global_safe_auth_forbidden"


@dataclass(frozen=True)
class TypeCheckDiagnostic:
    code: TypeCheckErrorCode
    path: str
    detail: str
    ref: str | None = None

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code.value,
            "path": self.path,
            "detail": self.detail,
        }
        if self.ref is not None:
            result["ref"] = self.ref
        return result


@dataclass(frozen=True)
class TypeCheckReport:
    status: TypeCheckStatus
    diagnostics: tuple[TypeCheckDiagnostic, ...]
    protocol: str = TYPECHECK_PROTOCOL

    @property
    def passed(self) -> bool:
        return self.status is TypeCheckStatus.PASS

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "status": self.status.value,
            "diagnostics": [item.to_json() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class _ValueType:
    dtype: str | None
    rank: int
    kind: str = "tensor"


@dataclass(frozen=True)
class _Anchor:
    owner: str
    kind: str
    value_type: _ValueType | None = None
    operand_types: tuple[_ValueType | None, ...] = ()
    carrier_class: str | None = None


def _operation_results(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _explicit_type(operation: Mapping[str, Any]) -> _ValueType | None:
    value = operation.get("type")
    if not isinstance(value, Mapping):
        return None
    if value.get("kind") in {"record", "ragged"}:
        try:
            parsed = CanonicalValueType.from_json(
                dict(value), context="IntentOp.type"
            )
        except (TypeError, ValueError):
            return None
        return _ValueType(parsed.dtype, len(parsed.shape), parsed.kind)
    dtype = value.get("dtype")
    shape = value.get("shape")
    if not isinstance(dtype, str) or not isinstance(shape, list):
        return None
    return _ValueType(dtype, len(shape))


def _surface_value_types(j: SemanticArchive) -> dict[str, CanonicalValueType]:
    """Read explicit value types through the shared operation record.

    A malformed in-memory archive may not have a recordable operation table.
    In that case only the already-declared tensor types remain available; the
    checker must not reconstruct a second operation grammar from raw body
    fields.
    """

    result = {
        tensor.name: CanonicalValueType(tensor.dtype, tuple(tensor.shape))
        for tensor in [*j.intent.inputs, *j.intent.outputs]
    }
    intent = _surface_intent_function(j)
    records = canonical_operation_records(intent)
    if records is not None:
        explicit_types = canonical_value_type_records(intent, records)
        if explicit_types is None:
            return result
        for name, declared in explicit_types.items():
            if declared.get("kind") in {"record", "ragged"}:
                try:
                    result[name] = CanonicalValueType.from_json(
                        dict(declared), context=f"IntentOp {name}.type"
                    )
                except (TypeError, ValueError):
                    continue
            else:
                dtype = declared.get("dtype")
                shape = declared.get("shape")
                if isinstance(dtype, str) and isinstance(shape, list):
                    result[name] = CanonicalValueType(dtype, tuple(shape))
        return result
    return result


def _surface_intent_function(j: SemanticArchive) -> dict[str, Any]:
    """Project typed J without invoking operation semantic serialization.

    TypeCheck must be able to diagnose a malformed in-memory operation after a
    caller mutates a typed archive.  ``IntentFunction.to_json`` intentionally
    rejects such a mutation, so the diagnostic pass uses the raw structural
    payload here and leaves the normal serialization boundary fail-closed.
    """

    fn = j.intent
    return {
        "name": fn.name,
        "inputs": [item.to_json() for item in fn.inputs],
        "outputs": [item.to_json() for item in fn.outputs],
        "returns": [item.to_json() for item in fn.returns],
        "body": [item._structural_payload() for item in fn.body],
    }


def _canonical_operation_items(
    j: SemanticArchive,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Return the one operation wire view used by valid TypeCheck paths.

    A valid SemanticArchive always uses the shared construction record.  If a
    caller mutates an archive so that the construction record is unavailable,
    return no operation rows and let the surrounding fail-closed checks report
    the malformed archive; do not build a second path/result/type grammar.
    """

    try:
        records = j_elaboration(j).operation_records
    except (IndexError, KeyError, TypeError, ValueError):
        records = None
    if records is not None:
        return tuple(
            (record["path"], record["wire"])
            for record in records
            if isinstance(record.get("wire"), Mapping)
        )
    return ()


def _canonical_operation_body(j: SemanticArchive) -> tuple[Any, ...]:
    """Return the canonical typed top-level body for scope-sensitive checks."""

    try:
        rows = j_elaboration(j).typed_operation_rows
    except (IndexError, KeyError, TypeError, ValueError):
        return ()
    if rows is None:
        return ()
    return tuple(
        operation
        for path, operation in rows
        if "." not in path and path.startswith("body[")
    )


def _declared_dimensions(j: SemanticArchive) -> frozenset[str]:
    return declared_dimension_handles(j)


def _check_dimension_and_pad_contracts(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Prove model-authored affine pad shapes without Harness valuations."""

    dimensions = _declared_dimensions(j)
    try:
        constraints = DimensionConstraintSet.parse(
            j.dimension_constraints,
            dimensions=dimensions,
        )
    except DimExprError as exc:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.DIMENSION_CONSTRAINT_INVALID,
                "j.dimension_constraints",
                str(exc),
            )
        )
        return

    elaboration = j_elaboration(j)
    value_types = elaboration.value_type_by_name
    for operation_path, operation in _canonical_operation_items(j):
        if operation.get("op") != "pad":
            continue
        base = f"j.intent.intent_function.{operation_path}"
        ref_value = operation.get("id")
        ref = ref_value if isinstance(ref_value, str) else operation_path
        attrs_value = operation.get("attrs")
        attrs = attrs_value if isinstance(attrs_value, Mapping) else {}
        operands = operation.get("operands")
        result = operation.get("result")
        if (
            not _PAD_REQUIRED_ATTRS.issubset(attrs)
            or set(attrs) - _PAD_ATTRS
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.attrs",
                    "pad requires exactly the closed attrs lo, hi, and value",
                    ref,
                )
            )
            continue
        if (
            not isinstance(operands, list)
            or len(operands) != 1
            or not isinstance(operands[0], str)
            or not isinstance(result, str)
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.operands",
                    "pad requires exactly one typed operand and one result",
                    ref,
                )
            )
            continue
        operand_type = value_types.get(operands[0])
        result_type = value_types.get(result)
        if operand_type is None or result_type is None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.type",
                    "pad operand and result require closed J value types",
                    ref,
                )
            )
            continue
        if operand_type.dtype != result_type.dtype:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.type.dtype",
                    "pad preserves the operand dtype",
                    ref,
                )
            )
        lo = attrs.get("lo")
        hi = attrs.get("hi")
        rank = len(operand_type.shape)
        if (
            not isinstance(lo, list)
            or not isinstance(hi, list)
            or len(result_type.shape) != rank
            or len(lo) != rank
            or len(hi) != rank
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.PAD_SHAPE_INVALID,
                    f"{base}.attrs",
                    "pad lo/hi lengths and result rank must equal the operand rank",
                    ref,
                )
            )
            continue

        for axis, (input_dim, low, high, output_dim) in enumerate(
            zip(operand_type.shape, lo, hi, result_type.shape, strict=True)
        ):
            axis_path = f"{base}.attrs"
            try:
                input_form = parse_dimension_expr(
                    input_dim,
                    dimensions=dimensions,
                    context=f"{base}.operand_shape[{axis}]",
                )
                low_form = parse_dimension_expr(
                    low,
                    dimensions=dimensions,
                    context=f"{base}.attrs.lo[{axis}]",
                )
                high_form = parse_dimension_expr(
                    high,
                    dimensions=dimensions,
                    context=f"{base}.attrs.hi[{axis}]",
                )
                output_form = parse_dimension_expr(
                    output_dim,
                    dimensions=dimensions,
                    context=f"{base}.result_shape[{axis}]",
                )
            except DimExprError as exc:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.PAD_SHAPE_INVALID,
                        axis_path,
                        str(exc),
                        ref,
                    )
                )
                continue

            expected = input_form.plus(
                low_form, context=f"{base}.shape[{axis}]"
            ).plus(high_form, context=f"{base}.shape[{axis}]")
            if not equivalent_dimension_expr(expected, output_form):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.PAD_SHAPE_INVALID,
                        axis_path,
                        "pad shape equation is not a closed identity: "
                        f"input + lo + hi normalizes to {expected.to_text()!r}, "
                        f"but result is {output_form.to_text()!r}",
                        ref,
                    )
                )
            for field, form in (("lo", low_form), ("hi", high_form)):
                if constraints.proves_dimension_nonnegative(form):
                    continue
                code = (
                    TypeCheckErrorCode.PAD_SHAPE_INVALID
                    if form.is_constant
                    else TypeCheckErrorCode.PAD_SHAPE_CONSTRAINT_MISSING
                )
                diagnostics.append(
                    TypeCheckDiagnostic(
                        code,
                        f"{base}.attrs.{field}[{axis}]",
                        "pad extent is not mechanically proven non-negative "
                        "by J.dimension_constraints",
                        ref,
                    )
                )


def _check_value_dimension_relations(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    try:
        validate_value_dimension_relations(j)
    except ValueDimensionRelationError as exc:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.VALUE_DIMENSION_RELATION_INVALID,
                "j.value_dimension_relations",
                str(exc),
            )
        )


def _check_dimension_relations(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    try:
        validate_dimension_relations(j)
    except DimensionRelationError as exc:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.DIMENSION_RELATION_INVALID,
                "j.dimension_relations",
                str(exc),
            )
        )


def _known_contract_diagnostic(
    diagnostics: list[TypeCheckDiagnostic],
    path: str,
    detail: str,
    ref: str,
) -> None:
    diagnostics.append(
        TypeCheckDiagnostic(
            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
            path,
            detail,
            ref,
        )
    )


def _known_tensor_type(
    value_types: Mapping[str, CanonicalValueType],
    name: Any,
    *,
    path: str,
    diagnostics: list[TypeCheckDiagnostic],
    ref: str,
) -> CanonicalValueType | None:
    value_type = value_types.get(name) if isinstance(name, str) else None
    if value_type is None:
        _known_contract_diagnostic(
            diagnostics,
            path,
            "operation operand/result requires a closed J value type",
            ref,
        )
        return None
    if not value_type.is_tensor:
        _known_contract_diagnostic(
            diagnostics,
            path,
            "operation requires a tensor value rather than a record or ragged descriptor",
            ref,
        )
        return None
    if not is_closed_dtype(value_type.dtype):
        _known_contract_diagnostic(
            diagnostics,
            path,
            f"operation value has a dtype outside the closed J grammar: {value_type.dtype!r}",
            ref,
        )
        return None
    return value_type


def _known_result_names(value: Any) -> tuple[str, ...]:
    return _operation_results(value)


def _known_exact_attrs(
    attrs: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    base: str,
    diagnostics: list[TypeCheckDiagnostic],
    ref: str,
    label: str,
) -> bool:
    # ``allowed``/``required`` remain in the call shape for the existing
    # operation branches, but they are no longer an authority.  Every ordinary
    # operation attr set is read from the same discriminated schema contract
    # used by MCP/schema_lookup and the provider-facing wire profile.  This
    # prevents a schema change such as gather.mode or reduce.keepdims from
    # requiring a second repository-wide hand-maintained field update.
    del allowed, required
    try:
        contract = operation_attr_contract(label)
    except (KeyError, TypeError, ValueError) as exc:
        _known_contract_diagnostic(
            diagnostics,
            f"{base}.attrs",
            f"{label} has no canonical schema attr contract: {exc}",
            ref,
        )
        return False
    schema_allowed = frozenset(contract["allowed"])
    schema_required = frozenset(contract["required"])
    valid = True
    missing = sorted(schema_required - set(attrs))
    unknown = sorted(set(attrs) - schema_allowed)
    if missing:
        _known_contract_diagnostic(
            diagnostics,
            f"{base}.attrs",
            f"{label} is missing required attr(s): {', '.join(missing)}",
            ref,
        )
        valid = False
    if unknown:
        for key in unknown:
            _known_contract_diagnostic(
                diagnostics,
                f"{base}.attrs.{key}",
                f"{label} contains unsupported attr {key!r}",
                ref,
            )
        valid = False
    return valid


def _known_arity(
    operation: Mapping[str, Any],
    *,
    operands: int | tuple[int, ...],
    results: int | tuple[int, ...],
    base: str,
    diagnostics: list[TypeCheckDiagnostic],
    ref: str,
    label: str,
) -> tuple[list[str], tuple[str, ...], bool]:
    raw_operands = operation.get("operands")
    operand_refs = (
        list(raw_operands)
        if isinstance(raw_operands, list)
        else []
    )
    raw_results = _known_result_names(operation.get("result"))
    operand_ok = (
        len(operand_refs) in operands
        if isinstance(operands, tuple)
        else len(operand_refs) == operands
    ) and all(isinstance(item, str) and item for item in operand_refs)
    result_ok = (
        len(raw_results) in results
        if isinstance(results, tuple)
        else len(raw_results) == results
    ) and len(raw_results) == len(set(raw_results))
    if not operand_ok:
        _known_contract_diagnostic(
            diagnostics,
            f"{base}.operands",
            f"{label} has an invalid operand count or operand reference list",
            ref,
        )
    if not result_ok:
        _known_contract_diagnostic(
            diagnostics,
            f"{base}.result",
            f"{label} has an invalid result count or result reference list",
            ref,
        )
    return operand_refs, raw_results, operand_ok and result_ok


def _known_axis(
    value: Any,
    *,
    rank: int,
    base: str,
    diagnostics: list[TypeCheckDiagnostic],
    ref: str,
    attribute: str = "axis",
    allow_negative: bool = True,
) -> int | None:
    if type(value) is not int:
        _known_contract_diagnostic(
            diagnostics,
            f"{base}.attrs.{attribute}",
            f"{attribute} must be an integer axis",
            ref,
        )
        return None
    normalized = value + rank if allow_negative and value < 0 else value
    if not 0 <= normalized < rank:
        _known_contract_diagnostic(
            diagnostics,
            f"{base}.attrs.{attribute}",
            f"{attribute} {value} is outside rank-{rank}",
            ref,
        )
        return None
    return normalized


def _known_integer_sequence(
    value: Any,
    *,
    base: str,
    attribute: str,
    diagnostics: list[TypeCheckDiagnostic],
    ref: str,
    minimum: int | None = None,
    nonempty: bool = False,
) -> list[int] | None:
    if not isinstance(value, list) or (nonempty and not value):
        _known_contract_diagnostic(
            diagnostics,
            f"{base}.attrs.{attribute}",
            f"{attribute} must be a {'non-empty ' if nonempty else ''}integer array",
            ref,
        )
        return None
    result: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or (minimum is not None and item < minimum):
            suffix = (
                f" >= {minimum}"
                if minimum is not None
                else ""
            )
            _known_contract_diagnostic(
                diagnostics,
                f"{base}.attrs.{attribute}[{index}]",
                f"{attribute} entries must be integers{suffix}",
                ref,
            )
            continue
        result.append(item)
    return result if len(result) == len(value) else None


def _known_shape_entries(
    value: Any,
    *,
    base: str,
    attribute: str,
    dimensions: frozenset[str],
    diagnostics: list[TypeCheckDiagnostic],
    ref: str,
) -> list[int | str] | None:
    if not isinstance(value, list):
        _known_contract_diagnostic(
            diagnostics,
            f"{base}.attrs.{attribute}",
            f"{attribute} must be an array of non-negative integers or symbolic dimensions",
            ref,
        )
        return None
    result: list[int | str] = []
    for index, item in enumerate(value):
        if type(item) is int and item >= 0:
            result.append(item)
            continue
        if isinstance(item, str) and item:
            try:
                parse_dimension_expr(
                    item,
                    dimensions=dimensions,
                    context=f"{base}.attrs.{attribute}[{index}]",
                )
            except DimExprError as exc:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.{attribute}[{index}]",
                    str(exc),
                    ref,
                )
            else:
                result.append(item)
            continue
        else:
            _known_contract_diagnostic(
                diagnostics,
                f"{base}.attrs.{attribute}[{index}]",
                f"{attribute} entries must be non-negative integers or non-empty symbols",
                ref,
            )
    return result if len(result) == len(value) else None


def _known_dimension_entry(
    value: Any,
    *,
    base: str,
    attribute: str,
    dimensions: frozenset[str],
    diagnostics: list[TypeCheckDiagnostic],
    ref: str,
) -> int | str | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value:
        try:
            parse_dimension_expr(
                value,
                dimensions=dimensions,
                context=f"{base}.attrs.{attribute}",
            )
        except DimExprError as exc:
            _known_contract_diagnostic(
                diagnostics,
                f"{base}.attrs.{attribute}",
                str(exc),
                ref,
            )
            return None
        return value
    _known_contract_diagnostic(
        diagnostics,
        f"{base}.attrs.{attribute}",
        f"{attribute} must be a non-negative integer or a closed dimension expression",
        ref,
    )
    return None


def _known_integer_attr(
    value: Any,
    *,
    base: str,
    attribute: str,
    diagnostics: list[TypeCheckDiagnostic],
    ref: str,
    minimum: int | None = None,
) -> int | None:
    if type(value) is int and (minimum is None or value >= minimum):
        return value
    suffix = f" >= {minimum}" if minimum is not None else ""
    _known_contract_diagnostic(
        diagnostics,
        f"{base}.attrs.{attribute}",
        f"{attribute} must be an integer{suffix}",
        ref,
    )
    return None


def _closed_integer_dtype_info(dtype: object) -> tuple[str, int] | None:
    """Return signedness/width only for the closed logical J integer set."""

    if not is_closed_integer_dtype(dtype):
        return None
    try:
        canonical = canonicalize_dtype(dtype, context="integer dtype")
    except ValueError:
        return None
    return canonical[0], int(canonical[1:])


def _known_broadcast_contract(
    j: SemanticArchive,
    *,
    operation_path: str,
    operand_refs: list[str],
    result_type: CanonicalValueType,
    value_types: Mapping[str, CanonicalValueType],
    base: str,
    diagnostics: list[TypeCheckDiagnostic],
    ref: str,
) -> None:
    """Validate select/map broadcast projections at their consuming node."""

    path_node = next(
        (
            node_ref
            for node_ref, path in j.node_ids.items()
            if path == operation_path
        ),
        None,
    )
    node_refs = {operation_path, ref}
    if path_node is not None:
        node_refs.add(path_node)
    explicit: dict[str, tuple[int, ...]] = {}
    direct_unit_dimensions = direct_unit_dimension_refs(
        j.dimension_constraints
    )
    for projection in j.boundary.broadcasts:
        if projection.node_ref not in node_refs:
            continue
        operand = projection.operand
        if operand not in operand_refs:
            _known_contract_diagnostic(
                diagnostics,
                f"{base}.boundary.broadcasts",
                f"broadcast projection names a non-consumed operand {operand!r}",
                ref,
            )
            continue
        if any(type(axis) is not int for axis in projection.axes):
            _known_contract_diagnostic(
                diagnostics,
                f"{base}.boundary.broadcasts",
                "broadcast axes must be integer positions in the result rank",
                ref,
            )
            continue
        axes = tuple(projection.axes)
        previous = explicit.get(operand)
        if previous is not None and previous != axes:
            _known_contract_diagnostic(
                diagnostics,
                f"{base}.boundary.broadcasts",
                f"conflicting broadcast projections for operand {operand!r}",
                ref,
            )
            continue
        explicit[operand] = axes

    for operand in operand_refs:
        operand_type = value_types.get(operand)
        if operand_type is None or not operand_type.is_tensor:
            continue
        axes = explicit.get(operand)
        if axes is None:
            positions = _numpy_broadcast_positions(
                operand_type.shape,
                result_type.shape,
                unit_dimensions=direct_unit_dimensions,
            )
        else:
            try:
                projected, inserted = _broadcast_projection(
                    list(operand_type.shape),
                    axes,
                    list(result_type.shape),
                    unit_dimensions=direct_unit_dimensions,
                )
            except (NotImplementedError, ValueError):
                positions = None
            else:
                positions = (
                    tuple(
                        index
                        for index in range(len(result_type.shape))
                        if index not in set(inserted)
                    )
                    if projected is not None
                    else None
                )
                if positions is not None and len(positions) != len(operand_type.shape):
                    positions = None
        if positions is None:
            _known_contract_diagnostic(
                diagnostics,
                f"{base}.boundary.broadcasts",
                f"operand {operand!r} cannot broadcast to the result shape "
                f"{result_type.shape!r} under its authored axes",
                ref,
            )


def _check_known_operation_contracts(
    j: SemanticArchive,
) -> tuple[TypeCheckDiagnostic, ...]:
    """Reject malformed attrs for the small set of closed known-op contracts.

    This checker validates the model-authored spelling; it never translates
    aliases or repairs dependent result types into the canonical fields
    consumed by RefEval/Emit.
    """

    diagnostics: list[TypeCheckDiagnostic] = []
    try:
        value_types = j_elaboration(j).value_type_by_name
    except (IndexError, KeyError, TypeError, ValueError):
        # Canonical elaboration below remains authoritative for graph/type
        # errors.  Explicitly authored types are still sufficient for local
        # dependent-contract diagnostics; they are never completed here.
        value_types = _surface_value_types(j)

    dimensions = _declared_dimensions(j)
    for operation_path, operation in _canonical_operation_items(j):
        op_name = operation.get("op")
        if op_name not in CANONICAL_J_OPS:
            base = f"j.intent.intent_function.{operation_path}"
            ref_value = operation.get("id")
            ref = ref_value if isinstance(ref_value, str) else operation_path
            _known_contract_diagnostic(
                diagnostics,
                f"{base}.op",
                f"operation opcode {op_name!r} is outside the closed IntentIR set",
                ref,
            )
            continue
        if op_name not in _KNOWN_OPERATION_CONTRACTS:
            continue
        base = f"j.intent.intent_function.{operation_path}"
        ref_value = operation.get("id")
        ref = ref_value if isinstance(ref_value, str) else operation_path
        attrs_value = operation.get("attrs")
        attrs = attrs_value if isinstance(attrs_value, Mapping) else {}

        if op_name == "cast":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="cast",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"to_dtype"}),
                required=frozenset({"to_dtype"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="cast",
            )
            target = attrs.get("to_dtype")
            if not is_closed_dtype(target):
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.to_dtype",
                    f"cast target dtype {target!r} is outside the closed J dtype set",
                    ref,
                )
                attrs_ok = False
            if (
                arity_ok
                and isinstance(operation.get("result"), str)
                and attrs_ok
            ):
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None and result_type is not None:
                    expected_dtype = canonicalize_dtype(
                        target,
                        context=f"{base}.attrs.to_dtype",
                    )
                    if source_type.shape != result_type.shape:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.shape",
                            "cast preserves the operand shape",
                            ref,
                        )
                    if result_type.dtype != expected_dtype:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.dtype",
                            "cast result dtype must equal attrs.to_dtype",
                            ref,
                        )
            continue

        if op_name == "select":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=3,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="select",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset(),
                required=frozenset(),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="select",
            )
            if arity_ok and isinstance(operation.get("result"), str):
                condition = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                positive = _known_tensor_type(
                    value_types,
                    operands[1],
                    path=f"{base}.operands[1]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                negative = _known_tensor_type(
                    value_types,
                    operands[2],
                    path=f"{base}.operands[2]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if (
                    condition is not None
                    and positive is not None
                    and negative is not None
                    and result_type is not None
                ):
                    if not is_closed_boolean_dtype(condition.dtype):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.operands[0].type.dtype",
                            "select condition must have boolean/i1 dtype",
                            ref,
                        )
                    if not (
                        positive.dtype
                        == negative.dtype
                        == result_type.dtype
                    ):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.dtype",
                            "select value operands and result must have one common dtype",
                            ref,
                        )
                    if attrs_ok:
                        _known_broadcast_contract(
                            j,
                            operation_path=operation_path,
                            operand_refs=operands,
                            result_type=result_type,
                            value_types=value_types,
                            base=base,
                            diagnostics=diagnostics,
                            ref=ref,
                        )
            continue

        if op_name == "gather":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=2,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="gather",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"axis", "mode"}),
                required=frozenset({"axis", "mode"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="gather",
            )
            mode = attrs.get("mode")
            if mode not in _GATHER_MODES:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.mode",
                    "gather mode must be 'along_axis' or 'take'",
                    ref,
                )
                attrs_ok = False
            if arity_ok and isinstance(operation.get("result"), str):
                data_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                index_type = _known_tensor_type(
                    value_types,
                    operands[1],
                    path=f"{base}.operands[1]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if data_type is not None and index_type is not None and result_type is not None:
                    if not is_non_boolean_integer_dtype(index_type.dtype):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.operands[1].type.dtype",
                            "gather index must have a closed non-boolean integer dtype",
                            ref,
                        )
                    axis = _known_axis(
                        attrs.get("axis"),
                        rank=len(data_type.shape),
                        base=base,
                        diagnostics=diagnostics,
                        ref=ref,
                    )
                    if attrs_ok and axis is not None:
                        if mode == "along_axis":
                            data_rank = len(data_type.shape)
                            index_rank = len(index_type.shape)
                            equal_rank = index_rank == data_rank
                            compressed_rank = (
                                index_rank == data_rank - 1 and index_rank > 0
                            )
                            if not (equal_rank or compressed_rank):
                                _known_contract_diagnostic(
                                    diagnostics,
                                    f"{base}.operands[1].type.shape",
                                    "along_axis gather index rank must equal "
                                    "data rank or data rank minus one",
                                    ref,
                                )
                            else:
                                expected_index_shape = (
                                    data_type.shape
                                    if equal_rank
                                    else (
                                        data_type.shape[:axis]
                                        + data_type.shape[axis + 1 :]
                                    )
                                )
                                if index_type.shape != expected_index_shape:
                                    _known_contract_diagnostic(
                                        diagnostics,
                                        f"{base}.operands[1].type.shape",
                                        "along_axis gather index must match "
                                        "data on every non-gather axis",
                                        ref,
                                    )
                            expected_shape = index_type.shape
                        else:
                            expected_shape = (
                                *data_type.shape[:axis],
                                *index_type.shape,
                                *data_type.shape[axis + 1 :],
                            )
                        if result_type.shape != expected_shape:
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.type.shape",
                                f"gather result shape must be {expected_shape!r} for mode={mode!r}",
                                ref,
                            )
                    if result_type.dtype != data_type.dtype:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.dtype",
                            "gather result dtype must equal data dtype",
                            ref,
                        )
            continue

        if op_name == "scatter":
            operands = operation.get("operands")
            operand_refs = operands if isinstance(operands, list) else []
            result = operation.get("result")
            attrs_value = operation.get("attrs")
            attrs = attrs_value if isinstance(attrs_value, Mapping) else {}
            atomic = attrs.get("atomic")
            is_cas = atomic == "cas"
            valid_arity = (
                len(operand_refs) == (4 if is_cas else 3)
                and all(isinstance(item, str) and item for item in operand_refs)
                and (
                    isinstance(result, str)
                    and bool(result)
                    if not is_cas
                    else isinstance(result, list)
                    and len(result) == 2
                    and all(isinstance(item, str) and item for item in result)
                    and len(set(result)) == 2
                )
            )
            if not valid_arity:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.operands",
                        (
                            "atomic='cas' scatter requires data, index, compare, "
                            "replacement, and [updated, observed] results"
                            if is_cas
                            else "scatter requires base, index, source, and one result"
                        ),
                        ref,
                    )
                )
                continue

            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset(),
                required=frozenset(),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="scatter",
            )
            axis = attrs.get("axis")
            if (
                not attrs_ok
                or type(axis) is not int
                or attrs.get("mode") not in _SCATTER_MODES
                or attrs.get("reduce") not in _SCATTER_REDUCES
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs",
                        "scatter requires closed axis, mode='along_axis', reduce, "
                        "and closed atomic ordering/scope/retry attrs",
                        ref,
                    )
                )
            if atomic is not None:
                if atomic not in _SCATTER_ATOMIC_KINDS:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.attrs.atomic",
                            f"atomic must be one of {sorted(_SCATTER_ATOMIC_KINDS)}",
                            ref,
                        )
                    )
                if attrs.get("ordering") not in _SCATTER_ORDERINGS:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.attrs.ordering",
                            "atomic scatter requires a closed ordering",
                            ref,
                        )
                    )
                if attrs.get("memory_scope") not in _SCATTER_MEMORY_SCOPES:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.attrs.memory_scope",
                            "atomic scatter requires a closed memory_scope",
                            ref,
                        )
                    )
                if atomic == "add" and attrs.get("reduce") != "add":
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.attrs.atomic",
                            "atomic='add' requires reduce='add'",
                            ref,
                        )
                    )
                if atomic == "add" and "cas_retry" in attrs:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.attrs.cas_retry",
                            "cas_retry is only valid for atomic='cas'",
                            ref,
                        )
                    )
                if atomic == "cas":
                    if attrs.get("reduce") != "none":
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                f"{base}.attrs.atomic",
                                "atomic='cas' requires reduce='none'",
                                ref,
                            )
                        )
                    if attrs.get("cas_retry", "single_attempt") not in _SCATTER_CAS_RETRIES:
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                f"{base}.attrs.cas_retry",
                                "cas_retry must be single_attempt or until_success",
                                ref,
                            )
                        )
            elif (
                "ordering" in attrs
                or "memory_scope" in attrs
                or "cas_retry" in attrs
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs",
                        "ordering and memory_scope require atomic",
                        ref,
                    )
                )

            data_type = value_types.get(operand_refs[0])
            index_type = value_types.get(operand_refs[1])
            compare_type = value_types.get(operand_refs[2]) if is_cas else None
            source_index = 3 if is_cas else 2
            source_type = value_types.get(operand_refs[source_index])
            result_names = list(result) if isinstance(result, list) else [result]
            result_type = value_types.get(result_names[0])
            observed_type = (
                value_types.get(result_names[1]) if is_cas else None
            )
            required_types = [
                data_type,
                index_type,
                source_type,
                result_type,
            ]
            if is_cas:
                required_types.extend([compare_type, observed_type])
            if any(item is None for item in required_types):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.type",
                        "scatter operands and result require closed J value types",
                        ref,
                    )
                )
                continue
            assert data_type is not None
            assert index_type is not None
            if is_cas:
                assert compare_type is not None
            assert source_type is not None
            assert result_type is not None
            if not is_non_boolean_integer_dtype(index_type.dtype):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.operands[1]",
                        "scatter index must have an integer dtype",
                        ref,
                    )
                )
            if (
                index_type.shape != source_type.shape
                or (is_cas and index_type.shape != compare_type.shape)
                or len(index_type.shape) != len(data_type.shape)
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.operands[1]",
                        "scatter index and source must have one common shape and "
                        "the data rank",
                        ref,
                    )
                )
            if source_type.dtype != data_type.dtype:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.operands[{source_index}]",
                        "scatter source dtype must equal the data dtype",
                        ref,
                    )
                )
            if is_cas and compare_type.dtype != data_type.dtype:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.operands[2]",
                        "compare dtype must equal the data dtype",
                        ref,
                    )
                )
            if result_type != data_type or (
                is_cas and observed_type is not None and observed_type != data_type
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.type",
                        "scatter updated and observed result types must equal "
                        "the data type",
                        ref,
                    )
                )
            if type(axis) is int and not 0 <= axis < len(data_type.shape):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs.axis",
                        "scatter axis is outside the data rank",
                        ref,
                    )
                )
            continue

        if op_name == "reshape":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="reshape",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"shape"}),
                required=frozenset({"shape"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="reshape",
            )
            shape = _known_shape_entries(
                attrs.get("shape"),
                base=base,
                attribute="shape",
                dimensions=dimensions,
                diagnostics=diagnostics,
                ref=ref,
            )
            if shape is not None and "?" in shape:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.shape",
                    "reshape does not accept the dynamic '?' extent marker",
                    ref,
                )
                shape = None
            if (
                arity_ok
                and attrs_ok
                and shape is not None
                and isinstance(operation.get("result"), str)
            ):
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None and result_type is not None:
                    if source_type.dtype != result_type.dtype:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.dtype",
                            "reshape preserves the operand dtype",
                            ref,
                        )
                    if tuple(shape) != result_type.shape:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.shape",
                            "reshape result shape must equal attrs.shape",
                            ref,
                        )
                    if all(type(item) is int for item in source_type.shape) and all(
                        type(item) is int for item in shape
                    ):
                        source_elements = math.prod(source_type.shape) if source_type.shape else 1
                        result_elements = math.prod(shape) if shape else 1
                        if source_elements != result_elements:
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.attrs.shape",
                                "reshape must preserve the total element count",
                                ref,
                            )
            continue

        if op_name == "transpose":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="transpose",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"perm"}),
                required=frozenset({"perm"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="transpose",
            )
            if arity_ok and isinstance(operation.get("result"), str):
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                perm = _known_integer_sequence(
                    attrs.get("perm"),
                    base=base,
                    attribute="perm",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None and result_type is not None and perm is not None and attrs_ok:
                    rank = len(source_type.shape)
                    if len(perm) != rank or sorted(perm) != list(range(rank)):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.attrs.perm",
                            "transpose perm must be a complete non-repeating input-axis permutation",
                            ref,
                        )
                    else:
                        expected_shape = tuple(source_type.shape[axis] for axis in perm)
                        if result_type.shape != expected_shape:
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.type.shape",
                                "transpose result shape must follow attrs.perm",
                                ref,
                            )
                    if result_type.dtype != source_type.dtype:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.dtype",
                            "transpose preserves the operand dtype",
                            ref,
                        )
            continue

        if op_name == "concat":
            raw_operands = operation.get("operands")
            operands = list(raw_operands) if isinstance(raw_operands, list) else []
            results = _known_result_names(operation.get("result"))
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"axis"}),
                required=frozenset({"axis"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="concat",
            )
            if (
                not operands
                or any(not isinstance(item, str) or not item for item in operands)
            ):
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.operands",
                    "concat requires at least one tensor operand",
                    ref,
                )
                operands = []
            if len(results) != 1 or not isinstance(operation.get("result"), str):
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.result",
                    "concat requires one string result",
                    ref,
                )
                results = ()
            if operands and results and attrs_ok:
                operand_types = [
                    _known_tensor_type(
                        value_types,
                        operand,
                        path=f"{base}.operands[{index}]",
                        diagnostics=diagnostics,
                        ref=ref,
                    )
                    for index, operand in enumerate(operands)
                ]
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if all(item is not None for item in operand_types) and result_type is not None:
                    typed_operands = [item for item in operand_types if item is not None]
                    rank = len(typed_operands[0].shape)
                    axis = _known_axis(
                        attrs.get("axis"),
                        rank=rank,
                        base=base,
                        diagnostics=diagnostics,
                        ref=ref,
                    )
                    if any(len(item.shape) != rank for item in typed_operands[1:]):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.operands",
                            "concat operands must have one common rank",
                            ref,
                        )
                    elif axis is not None:
                        first = typed_operands[0]
                        for dimension in range(rank):
                            if dimension == axis:
                                continue
                            if any(
                                item.shape[dimension] != first.shape[dimension]
                                for item in typed_operands[1:]
                            ):
                                _known_contract_diagnostic(
                                    diagnostics,
                                    f"{base}.operands",
                                    "concat operands must agree on every non-concat dimension",
                                    ref,
                                )
                        if any(item.dtype != first.dtype for item in typed_operands) or result_type.dtype != first.dtype:
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.type.dtype",
                                "concat operands and result must share one dtype",
                                ref,
                            )
                        axis_extents = [item.shape[axis] for item in typed_operands]
                        if all(type(item) is int for item in axis_extents):
                            expected_shape = list(first.shape)
                            expected_shape[axis] = sum(axis_extents)
                            if result_type.shape != tuple(expected_shape):
                                _known_contract_diagnostic(
                                    diagnostics,
                                    f"{base}.type.shape",
                                    "concat result axis extent must equal the sum of input extents",
                                    ref,
                                )
            continue

        if op_name == "pad":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="pad",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"lo", "hi", "value"}),
                required=frozenset({"lo", "hi", "value"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="pad",
            )
            lo = _known_shape_entries(
                attrs.get("lo"),
                base=base,
                attribute="lo",
                dimensions=dimensions,
                diagnostics=diagnostics,
                ref=ref,
            )
            hi = _known_shape_entries(
                attrs.get("hi"),
                base=base,
                attribute="hi",
                dimensions=dimensions,
                diagnostics=diagnostics,
                ref=ref,
            )
            for field, values in (("lo", lo), ("hi", hi)):
                if values is None:
                    continue
                for index, item in enumerate(values):
                    if type(item) is int and item >= 0:
                        continue
                    if not isinstance(item, str):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.attrs.{field}[{index}]",
                            f"pad {field} entry must be a non-negative integer "
                            "or a closed dimension expression",
                            ref,
                        )
                        continue
                    try:
                        parse_dimension_expr(
                            item,
                            dimensions=dimensions,
                            context=f"{base}.attrs.{field}[{index}]",
                        )
                    except DimExprError as exc:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.attrs.{field}[{index}]",
                            str(exc),
                            ref,
                        )
            value = attrs.get("value")
            if not (
                value is None
                or type(value) in {bool, int, float}
                or isinstance(value, str)
            ):
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.value",
                    "pad value must be a scalar literal",
                    ref,
                )
            if arity_ok and attrs_ok and isinstance(operation.get("result"), str):
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None and result_type is not None:
                    if source_type.dtype != result_type.dtype:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.dtype",
                            "pad preserves the operand dtype",
                            ref,
                        )
                    if (
                        lo is not None
                        and hi is not None
                        and (len(lo) != len(source_type.shape) or len(hi) != len(source_type.shape))
                    ):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.attrs",
                            "pad lo/hi lengths must equal the operand rank",
                            ref,
                        )
            continue

        if op_name == "flip":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="flip",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"axes"}),
                required=frozenset({"axes"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="flip",
            )
            if arity_ok and isinstance(operation.get("result"), str):
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                axes = _known_integer_sequence(
                    attrs.get("axes"),
                    base=base,
                    attribute="axes",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None and result_type is not None and axes is not None and attrs_ok:
                    normalized = [
                        axis + len(source_type.shape) if axis < 0 else axis
                        for axis in axes
                    ]
                    if len(set(normalized)) != len(normalized) or any(
                        axis < 0 or axis >= len(source_type.shape) for axis in normalized
                    ):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.attrs.axes",
                            "flip axes must be unique and within the operand rank",
                            ref,
                        )
                    if result_type.shape != source_type.shape or result_type.dtype != source_type.dtype:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type",
                            "flip preserves the complete operand type",
                            ref,
                        )
            continue

        if op_name == "triu":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="triu",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"k"}),
                required=frozenset({"k"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="triu",
            )
            if type(attrs.get("k")) is not int:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.k",
                    "triu k must be an integer diagonal offset",
                    ref,
                )
            if arity_ok and attrs_ok and isinstance(operation.get("result"), str):
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None and result_type is not None:
                    if len(source_type.shape) != 2:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.operands[0].type.shape",
                            "triu requires a rank-2 operand",
                            ref,
                        )
                    if result_type.shape != source_type.shape or result_type.dtype != source_type.dtype:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type",
                            "triu preserves the complete operand type",
                            ref,
                        )
            continue

        if op_name == "record":
            operands = operation.get("operands")
            result = operation.get("result")
            fields = attrs.get("fields")
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"fields"}),
                required=frozenset({"fields"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="record",
            )
            operands_ok = (
                isinstance(operands, list)
                and all(isinstance(item, str) and item for item in operands)
            )
            if not operands_ok:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.operands",
                    "record operands must be a list of value references",
                    ref,
                )
            result_ok = isinstance(result, str) and bool(result)
            if not result_ok:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.result",
                    "record requires exactly one result value reference",
                    ref,
                )
            fields_ok = (
                isinstance(fields, list)
                and bool(fields)
                and all(isinstance(name, str) and bool(name) for name in fields)
                and len(fields) == len(set(fields))
            )
            if not fields_ok:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.fields",
                    "record fields must be a non-empty list of unique names",
                    ref,
                )
            result_type = value_types.get(result) if result_ok else None
            if result_type is None or not result_type.is_record:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.type",
                    "record result must have a closed record value type",
                    ref,
                )
            elif fields_ok:
                declared_fields = tuple(name for name, _ in result_type.fields)
                if tuple(fields) != declared_fields:
                    _known_contract_diagnostic(
                        diagnostics,
                        f"{base}.attrs.fields",
                        "record attrs.fields must exactly match record result "
                        "field names and order",
                        ref,
                    )
            if (
                operands_ok
                and fields_ok
                and result_type is not None
                and result_type.is_record
                and len(operands) == len(result_type.fields)
            ):
                for index, ((field_name, field_type), operand) in enumerate(
                    zip(result_type.fields, operands, strict=True)
                ):
                    operand_type = value_types.get(operand)
                    if operand_type is None:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.operands[{index}]",
                            "record field operand requires a closed J value type",
                            ref,
                        )
                    elif operand_type != field_type:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.operands[{index}]",
                            f"record field {field_name!r} operand type must "
                            "exactly match its declared field type",
                            ref,
                        )
            continue

        if op_name == "record_get":
            operands = operation.get("operands")
            result = operation.get("result")
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"field"}),
                required=frozenset({"field"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="record_get",
            )
            operands_ok = (
                isinstance(operands, list)
                and len(operands) == 1
                and isinstance(operands[0], str)
                and bool(operands[0])
            )
            if not operands_ok:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.operands",
                    "record_get requires exactly one record operand reference",
                    ref,
                )
            result_ok = isinstance(result, str) and bool(result)
            if not result_ok:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.result",
                    "record_get requires exactly one result value reference",
                    ref,
                )
            field = attrs.get("field")
            field_ok = isinstance(field, str) and bool(field)
            if not field_ok:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.field",
                    "record_get field must be a non-empty string",
                    ref,
                )
            record_type = value_types.get(operands[0]) if operands_ok else None
            if record_type is None or not record_type.is_record:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.operands[0]",
                    "record_get operand must have a closed record value type",
                    ref,
                )
            field_type = None
            if record_type is not None and record_type.is_record and field_ok:
                try:
                    field_type = record_type.field(field)
                except KeyError:
                    _known_contract_diagnostic(
                        diagnostics,
                        f"{base}.attrs.field",
                        f"record_get field {field!r} is not present in the "
                        "record operand type",
                        ref,
                    )
            result_type = value_types.get(result) if result_ok else None
            if result_type is None:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.type",
                    "record_get result requires a closed J value type",
                    ref,
                )
            elif field_type is not None and result_type != field_type:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.type",
                    "record_get result type must exactly match the selected "
                    "record field type",
                    ref,
                )
            continue

        if op_name in RAGGED_OPS:
            operands = operation.get("operands")
            result = operation.get("result")
            if (
                not isinstance(operands, list)
                or len(operands) != RAGGED_ARITY[op_name]
                or not isinstance(result, str)
                or not isinstance(operation.get("type"), Mapping)
                or set(attrs)
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}",
                        f"{op_name} requires explicit closed operand/result types "
                        "and no attrs",
                        ref,
                    )
                )
                continue
            if op_name == "ragged":
                offsets = value_types.get(operands[0])
                indices = value_types.get(operands[1])
                result_type = value_types.get(result)
                if (
                    result_type is None
                    or not result_type.is_ragged
                    or offsets is None
                    or indices is None
                    or not offsets.is_tensor
                    or not indices.is_tensor
                    or len(offsets.shape) != 1
                    or len(indices.shape) != 1
                    or not is_ragged_carrier_dtype(offsets.dtype)
                    or not is_ragged_carrier_dtype(indices.dtype)
                    or result_type.field("offsets") != offsets
                    or result_type.field("indices") != indices
                ):
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.type",
                            "ragged result fields must exactly match offsets and indices operands",
                            ref,
                        )
                    )
                continue
            descriptor = value_types.get(operands[0])
            if descriptor is None or not descriptor.is_ragged:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.operands[0]",
                        f"{op_name} requires a ragged descriptor operand",
                        ref,
                    )
                )
                continue
            result_type = value_types.get(result)
            if result_type is None or not result_type.is_tensor:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.type",
                        f"{op_name} requires a closed tensor result type",
                        ref,
                    )
                )
                continue
            expected_rank = 0 if op_name == "ragged_outer" else 1
            if len(result_type.shape) != expected_rank or not is_ragged_carrier_dtype(result_type.dtype):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.type",
                        f"{op_name} result must be an integer rank-{expected_rank} tensor",
                        ref,
                    )
                )
            if op_name == "ragged_member" and result_type.shape != ("?",):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.type.shape",
                        "ragged_member result must use the dynamic shape ['?']",
                        ref,
                    )
                )
            if op_name == "ragged_member":
                index_type = value_types.get(operands[1])
                if index_type is None or not index_type.is_tensor or len(index_type.shape) != 0 or not is_ragged_carrier_dtype(index_type.dtype):
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.operands[1]",
                            "ragged_member outer index must be a rank-0 integer tensor",
                            ref,
                        )
                    )
                elif result_type.dtype != descriptor.field("indices").dtype:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.type.dtype",
                            "ragged_member result dtype must equal descriptor indices dtype",
                            ref,
                        )
                    )
            else:
                expected_dtype = descriptor.field("offsets").dtype
                if result_type.dtype != expected_dtype:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.type.dtype",
                            f"{op_name} result dtype must equal offsets dtype",
                            ref,
                        )
                    )
            continue

        if op_name == "const":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=0,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="const",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"value", "dtype", "shape"}),
                required=frozenset({"value"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="const",
            )
            if arity_ok and attrs_ok:
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if result_type is not None:
                    try:
                        validate_const_literal(
                            attrs,
                            result_type=result_type.to_json(),
                            context=base,
                        )
                    except IntentOpContractError as exc:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.attrs",
                            str(exc),
                            ref,
                        )
            continue

        if op_name in {"map_unary", "map_binary"}:
            expected_fns = _MAP_UNARY_FNS if op_name == "map_unary" else _MAP_BINARY_FNS
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=_MAP_ATTRS,
                required=_MAP_ATTRS,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label=op_name,
            )
            fn = attrs.get("fn")
            if not isinstance(fn, str) or fn not in expected_fns:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.fn",
                    f"{op_name} requires exactly one closed fn; got {fn!r}",
                    ref,
                )
                attrs_ok = False
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1 if op_name == "map_unary" else 2,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label=op_name,
            )
            if not attrs_ok or not arity_ok:
                continue

            operand_types = [
                _known_tensor_type(
                    value_types,
                    operand,
                    path=f"{base}.operands[{index}]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                for index, operand in enumerate(operands)
            ]
            result_type = _known_tensor_type(
                value_types,
                results[0],
                path=f"{base}.result",
                diagnostics=diagnostics,
                ref=ref,
            )
            if any(item is None for item in operand_types) or result_type is None:
                continue
            if op_name == "map_unary":
                operand_type = operand_types[0]
                assert operand_type is not None
                expected_dtype = map_unary_result_dtype(fn, operand_type.dtype)
                if operand_type.shape != result_type.shape:
                    _known_contract_diagnostic(
                        diagnostics,
                        f"{base}.type.shape",
                        "map_unary preserves the complete operand shape",
                        ref,
                    )
            else:
                lhs_type, rhs_type = operand_types
                assert lhs_type is not None and rhs_type is not None
                expected_dtype = map_binary_result_dtype(
                    fn, lhs_type.dtype, rhs_type.dtype
                )
                _known_broadcast_contract(
                    j,
                    operation_path=operation_path,
                    operand_refs=operands,
                    result_type=result_type,
                    value_types=value_types,
                    base=base,
                    diagnostics=diagnostics,
                    ref=ref,
                )
            if expected_dtype is None:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.type.dtype",
                    f"{op_name} fn {fn!r} has no valid closed dtype contract",
                    ref,
                )
            elif result_type.dtype != expected_dtype:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.type.dtype",
                    f"{op_name} fn {fn!r} result dtype must be {expected_dtype!r}",
                    ref,
                )
            continue

        if op_name == "tile":
            for key in sorted(set(attrs) - _TILE_ATTRS):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.attrs.{key}",
                        f"tile attrs contain unsupported key {key!r}",
                        ref,
                    )
                )
            if set(attrs) != _TILE_ATTRS:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.attrs",
                        "tile requires exactly the closed attr reps",
                        ref,
                    )
                )

            operands = operation.get("operands")
            operand_refs = operands if isinstance(operands, list) else []
            result = operation.get("result")
            valid_arity = (
                len(operand_refs) == 1
                and isinstance(operand_refs[0], str)
                and isinstance(result, str)
                and bool(result)
            )
            if not valid_arity:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.operands",
                        "tile requires exactly one value operand and one result",
                        ref,
                    )
                )

            reps = attrs.get("reps")
            valid_reps = isinstance(reps, list) and all(
                type(repetition) is int and repetition >= 0
                for repetition in reps
            )
            if not valid_reps:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.attrs.reps",
                        "tile reps must be a closed array of non-negative "
                        "integers",
                        ref,
                    )
                )

            if not valid_arity or not valid_reps:
                continue
            assert isinstance(result, str)
            operand_type = value_types.get(operand_refs[0])
            result_type = value_types.get(result)
            if operand_type is None or result_type is None:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.type",
                        "tile operand and result require closed J value types",
                        ref,
                    )
                )
                continue
            if operand_type.dtype != result_type.dtype:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.type.dtype",
                        "tile preserves the operand dtype",
                        ref,
                    )
                )
            rank = len(operand_type.shape)
            if len(result_type.shape) != rank or len(reps) != rank:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.TILE_SHAPE_INVALID,
                        f"{base}.attrs.reps",
                        "tile reps length and result rank must equal the "
                        "operand rank",
                        ref,
                    )
                )
                continue

            dimensions = _declared_dimensions(j)
            try:
                constraints = DimensionConstraintSet.parse(
                    j.dimension_constraints,
                    dimensions=dimensions,
                )
            except DimExprError:
                # The dedicated dimension pass owns malformed constraints.
                constraints = None
            for axis, (input_dim, repetition, output_dim) in enumerate(
                zip(
                    operand_type.shape,
                    reps,
                    result_type.shape,
                    strict=True,
                )
            ):
                try:
                    input_form = parse_dimension_expr(
                        input_dim,
                        dimensions=dimensions,
                        context=f"{base}.operand_shape[{axis}]",
                    )
                    output_form = parse_dimension_expr(
                        output_dim,
                        dimensions=dimensions,
                        context=f"{base}.result_shape[{axis}]",
                    )
                    expected = input_form.scaled(
                        repetition,
                        context=f"{base}.shape[{axis}]",
                    )
                except DimExprError as exc:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.TILE_SHAPE_INVALID,
                            f"{base}.type.shape[{axis}]",
                            str(exc),
                            ref,
                        )
                    )
                    continue
                if equivalent_dimension_expr(expected, output_form):
                    continue
                difference = output_form.plus(
                    expected,
                    factor=-1,
                    context=f"{base}.shape[{axis}]",
                )
                proven_equal = (
                    constraints is not None
                    and constraints.proves_dimension_nonnegative(difference)
                    and constraints.proves_dimension_nonnegative(
                        difference.scaled(
                            -1,
                            context=f"{base}.shape[{axis}]",
                        )
                    )
                )
                if not proven_equal:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.TILE_SHAPE_INVALID,
                            f"{base}.type.shape[{axis}]",
                            "tile shape equation is not proven: input * reps "
                            f"normalizes to {expected.to_text()!r}, but result "
                            f"is {output_form.to_text()!r}",
                            ref,
                        )
                    )
            continue

        if op_name == "contraction":
            for key in sorted(set(attrs) - _CONTRACTION_ATTRS):
                detail = (
                    "contraction uses only canonical attr 'subscripts'; "
                    f"alias {key!r} is forbidden and is never translated by "
                    "fixed code"
                    if key in {"lhs_contract_axes", "rhs_contract_axes"}
                    else f"contraction attrs contain unsupported key {key!r}"
                )
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs.{key}",
                        detail,
                        ref,
                    )
                )

            operands = operation.get("operands")
            operand_refs = operands if isinstance(operands, list) else []
            if (
                len(operand_refs) != 2
                or not all(isinstance(item, str) for item in operand_refs)
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.operands",
                        "contraction requires exactly two value operands",
                        ref,
                    )
                )

            result = operation.get("result")
            if not isinstance(result, str) or not result:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.result",
                        "contraction requires exactly one value result",
                        ref,
                    )
                )

            subscripts = attrs.get("subscripts")
            lhs_labels: str | None = None
            rhs_labels: str | None = None
            output_labels: str | None = None
            if not isinstance(subscripts, str):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                        f"{base}.attrs.subscripts",
                        "contraction subscripts must be an explicit binary "
                        "einsum string such as 'mk,kn->mn'",
                        ref,
                    )
                )
            else:
                compact = "".join(subscripts.split())
                inputs_and_output = compact.split("->")
                input_terms = (
                    inputs_and_output[0].split(",")
                    if len(inputs_and_output) == 2
                    else []
                )
                if (
                    len(inputs_and_output) != 2
                    or len(input_terms) != 2
                    or not all(_EINSUM_TERM.fullmatch(term) for term in input_terms)
                    or _EINSUM_TERM.fullmatch(inputs_and_output[-1]) is None
                ):
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            f"{base}.attrs.subscripts",
                            "contraction subscripts must use the closed binary "
                            "einsum grammar '<labels>,<labels>-><labels>' "
                            "without ellipsis",
                            ref,
                        )
                    )
                else:
                    lhs_labels, rhs_labels = input_terms
                    output_labels = inputs_and_output[1]
                    if len(set(output_labels)) != len(output_labels):
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                f"{base}.attrs.subscripts",
                                "contraction output labels must be unique",
                                ref,
                            )
                        )
                    unknown_output = sorted(
                        set(output_labels) - set(lhs_labels) - set(rhs_labels)
                    )
                    if unknown_output:
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                f"{base}.attrs.subscripts",
                                "contraction output labels must occur in an "
                                f"input term; unknown labels {unknown_output!r}",
                                ref,
                            )
                        )

            if (
                lhs_labels is not None
                and rhs_labels is not None
                and output_labels is not None
                and len(operand_refs) == 2
                and all(isinstance(item, str) for item in operand_refs)
            ):
                operand_types = [value_types.get(item) for item in operand_refs]
                for index, (labels, value_type) in enumerate(
                    zip((lhs_labels, rhs_labels), operand_types, strict=True)
                ):
                    if value_type is not None and len(labels) != len(value_type.shape):
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                f"{base}.attrs.subscripts",
                                f"contraction operand {index} labels have rank "
                                f"{len(labels)}, but its typed J value has rank "
                                f"{len(value_type.shape)}",
                                ref,
                            )
                        )

                if all(
                    value_type is not None
                    and len(labels) == len(value_type.shape)
                    for labels, value_type in zip(
                        (lhs_labels, rhs_labels), operand_types, strict=True
                    )
                ):
                    label_extents: dict[str, Any] = {}
                    for labels, value_type in zip(
                        (lhs_labels, rhs_labels), operand_types, strict=True
                    ):
                        assert value_type is not None
                        for label, extent in zip(
                            labels, value_type.shape, strict=True
                        ):
                            previous = label_extents.get(label, extent)
                            if previous != extent:
                                diagnostics.append(
                                    TypeCheckDiagnostic(
                                        TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                        f"{base}.attrs.subscripts",
                                        f"contraction label {label!r} has "
                                        f"inconsistent typed extents {previous!r} "
                                        f"and {extent!r}",
                                        ref,
                                    )
                                )
                            else:
                                label_extents[label] = extent

                    result_type = (
                        value_types.get(result)
                        if isinstance(result, str)
                        else None
                    )
                    if (
                        result_type is not None
                        and set(output_labels).issubset(label_extents)
                    ):
                        expected_shape = tuple(
                            label_extents[label] for label in output_labels
                        )
                        if result_type.shape != expected_shape:
                            diagnostics.append(
                                TypeCheckDiagnostic(
                                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                                    f"{base}.type.shape",
                                    "contraction typed result shape "
                                    f"{result_type.shape!r} does not match "
                                    f"subscripts result {expected_shape!r}",
                                    ref,
                                )
                            )
            continue

        if op_name == "rand":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=0,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="rand",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"seed", "shape", "dtype"}),
                required=frozenset({"seed", "shape"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="rand",
            )
            seed = _known_integer_attr(
                attrs.get("seed"),
                base=base,
                attribute="seed",
                diagnostics=diagnostics,
                ref=ref,
            )
            shape = _known_shape_entries(
                attrs.get("shape"),
                base=base,
                attribute="shape",
                dimensions=dimensions,
                diagnostics=diagnostics,
                ref=ref,
            )
            dtype = attrs.get("dtype")
            if dtype is not None and not is_closed_dtype(dtype):
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.dtype",
                    "rand dtype must be in the closed J dtype set",
                    ref,
                )
                dtype = None
            if (
                arity_ok
                and attrs_ok
                and seed is not None
                and shape is not None
                and isinstance(operation.get("result"), str)
            ):
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if result_type is not None:
                    if tuple(shape) != result_type.shape:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.shape",
                            "rand result shape must equal attrs.shape",
                            ref,
                        )
                    if dtype is not None and result_type.dtype != canonicalize_dtype(
                        dtype, context=f"{base}.attrs.dtype"
                    ):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.dtype",
                            "rand result dtype must equal attrs.dtype",
                            ref,
                        )
            continue

        if op_name == "iota":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=0,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="iota",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"n", "dtype"}),
                required=frozenset({"n"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="iota",
            )
            n = _known_dimension_entry(
                attrs.get("n"),
                base=base,
                attribute="n",
                dimensions=dimensions,
                diagnostics=diagnostics,
                ref=ref,
            )
            # The wire attribute is optional.  If it is omitted, use the
            # authored result type when available; this preserves an explicit
            # Source/compiler index width such as ``index``/i64 and keeps
            # TypeCheck aligned with emit_mlir and the structured reference
            # evaluator.  A result-less malformed operation still receives the
            # historical i32 fallback so the diagnostic remains structural.
            result_type = None
            if arity_ok and attrs_ok and n is not None:
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
            dtype = attrs.get(
                "dtype",
                result_type.dtype if result_type is not None else "i32",
            )
            dtype_ok = is_non_boolean_integer_dtype(dtype)
            if not dtype_ok:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.dtype",
                    "iota dtype must be a closed non-boolean integer dtype",
                    ref,
                )
            if arity_ok and attrs_ok and n is not None and dtype_ok:
                if result_type is not None:
                    if len(result_type.shape) != 1 or result_type.shape[0] != n:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.shape",
                            "iota result must be rank one with extent attrs.n",
                            ref,
                        )
                    if result_type.dtype != canonicalize_dtype(
                        dtype, context=f"{base}.attrs.dtype"
                    ):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.dtype",
                            "iota result dtype must equal attrs.dtype",
                            ref,
                        )
            continue

        if op_name == "scan":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="scan",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"fn", "axis"}),
                required=frozenset({"fn", "axis"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="scan",
            )
            fn = attrs.get("fn")
            if fn not in _SCAN_FNS:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.fn",
                    "scan fn must be one of ['max', 'min', 'sum']",
                    ref,
                )
                attrs_ok = False
            if arity_ok and attrs_ok:
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None and result_type is not None:
                    _known_axis(
                        attrs.get("axis"),
                        rank=len(source_type.shape),
                        base=base,
                        diagnostics=diagnostics,
                        ref=ref,
                    )
                    if result_type != source_type:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type",
                            "scan preserves the complete operand type",
                            ref,
                        )
            continue

        if op_name == "arg_reduce":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="arg_reduce",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"fn", "axis"}),
                required=frozenset({"fn", "axis"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="arg_reduce",
            )
            if attrs.get("fn") not in _ARG_REDUCE_FNS:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.fn",
                    "arg_reduce fn must be 'argmax' or 'argmin'",
                    ref,
                )
                attrs_ok = False
            if arity_ok and attrs_ok:
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None and result_type is not None:
                    axis = _known_axis(
                        attrs.get("axis"),
                        rank=len(source_type.shape),
                        base=base,
                        diagnostics=diagnostics,
                        ref=ref,
                    )
                    if axis is not None:
                        expected_shape = tuple(
                            dim
                            for index, dim in enumerate(source_type.shape)
                            if index != axis
                        )
                        if result_type.shape != expected_shape:
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.type.shape",
                                "arg_reduce result must remove attrs.axis",
                                ref,
                            )
                    if result_type.dtype != "i32":
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.dtype",
                            "arg_reduce result dtype must be i32",
                            ref,
                        )
            continue

        if op_name == "sort":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=(1, 2),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="sort",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"descending", "axis", "k", "return"}),
                required=frozenset({"descending", "axis"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="sort",
            )
            if type(attrs.get("descending")) is not bool:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.descending",
                    "sort descending must be boolean",
                    ref,
                )
                attrs_ok = False
            return_mode = attrs.get("return")
            if return_mode is not None and return_mode not in _SORT_RETURN_MODES:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.return",
                    "sort return, when present, must be 'indices'",
                    ref,
                )
                attrs_ok = False
            if arity_ok and attrs_ok:
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None:
                    axis = _known_axis(
                        attrs.get("axis"),
                        rank=len(source_type.shape),
                        base=base,
                        diagnostics=diagnostics,
                        ref=ref,
                    )
                    if axis is None or not source_type.shape or axis != len(source_type.shape) - 1:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.attrs.axis",
                            "sort axis must be the last operand axis",
                            ref,
                        )
                    width = attrs.get("k")
                    if width is not None and (
                        type(width) is not int or width <= 0
                    ):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.attrs.k",
                            "sort k must be a positive integer",
                            ref,
                        )
                        width = None
                    if (
                        width is not None
                        and source_type.shape
                        and type(source_type.shape[-1]) is int
                        and width > source_type.shape[-1]
                    ):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.attrs.k",
                            "sort k cannot exceed the last operand extent",
                            ref,
                        )
                    expected_shape = list(source_type.shape)
                    if width is not None:
                        expected_shape[-1] = width
                    raw_result = operation.get("result")
                    if return_mode == "indices" and not (
                        isinstance(raw_result, str) and len(results) == 1
                    ):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.result",
                            "sort return='indices' requires one string index result",
                            ref,
                        )
                    elif return_mode is None and isinstance(raw_result, list) and len(results) != 2:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.result",
                            "sort result list must contain values and indices",
                            ref,
                        )
                    if return_mode is None and isinstance(raw_result, list) and len(results) == 2:
                        value_result = _known_tensor_type(
                            value_types,
                            results[0],
                            path=f"{base}.result[0]",
                            diagnostics=diagnostics,
                            ref=ref,
                        )
                        index_result = _known_tensor_type(
                            value_types,
                            results[1],
                            path=f"{base}.result[1]",
                            diagnostics=diagnostics,
                            ref=ref,
                        )
                        if value_result is not None and (
                            value_result.shape != tuple(expected_shape)
                            or value_result.dtype != source_type.dtype
                        ):
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.result[0].type",
                                "sort values result must preserve the source dtype and sorted shape",
                                ref,
                            )
                        if index_result is not None and (
                            index_result.shape != tuple(expected_shape)
                            or index_result.dtype != "i32"
                        ):
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.result[1].type",
                                "sort index result must have sorted shape and i32 dtype",
                                ref,
                            )
                    elif isinstance(raw_result, str):
                        result_type = _known_tensor_type(
                            value_types,
                            raw_result,
                            path=f"{base}.result",
                            diagnostics=diagnostics,
                            ref=ref,
                        )
                        expected_dtype = "i32" if return_mode == "indices" else source_type.dtype
                        if result_type is not None and (
                            result_type.shape != tuple(expected_shape)
                            or result_type.dtype != expected_dtype
                        ):
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.type",
                                "sort result has an incompatible shape or dtype",
                                ref,
                            )
            continue

        if op_name == "pool":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=(1, 2),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="pool",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"fn", "kH", "kW", "sH", "sW", "padH", "padW"}),
                required=frozenset({"fn", "kH", "kW", "sH", "sW", "padH", "padW"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="pool",
            )
            fn = attrs.get("fn")
            if fn not in _POOL_FNS:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.fn",
                    "pool fn must be 'avg' or 'max'",
                    ref,
                )
                attrs_ok = False
            scalar_attrs = {
                name: _known_integer_attr(
                    attrs.get(name),
                    base=base,
                    attribute=name,
                    diagnostics=diagnostics,
                    ref=ref,
                    minimum=0 if name.startswith("pad") else 1,
                )
                for name in ("kH", "kW", "sH", "sW", "padH", "padW")
            }
            if arity_ok and attrs_ok:
                if fn == "avg" and len(results) != 1:
                    _known_contract_diagnostic(
                        diagnostics,
                        f"{base}.result",
                        "avg pool has one result; only max pool may return indices",
                        ref,
                    )
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None:
                    if len(source_type.shape) != 4:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.operands[0].type.shape",
                            "pool requires an NCHW rank-4 operand",
                            ref,
                        )
                    result_type = _known_tensor_type(
                        value_types,
                        results[0],
                        path=f"{base}.result[0]" if len(results) == 2 else f"{base}.result",
                        diagnostics=diagnostics,
                        ref=ref,
                    )
                    if result_type is not None and len(source_type.shape) == 4:
                        if result_type.dtype != source_type.dtype or result_type.shape[:2] != source_type.shape[:2]:
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.result[0].type",
                                "pool output must preserve N/C and operand dtype",
                                ref,
                            )
                        if all(type(source_type.shape[index]) is int for index in (2, 3)) and all(
                            value is not None for value in scalar_attrs.values()
                        ):
                            oh = (source_type.shape[2] + 2 * scalar_attrs["padH"] - scalar_attrs["kH"]) // scalar_attrs["sH"] + 1
                            ow = (source_type.shape[3] + 2 * scalar_attrs["padW"] - scalar_attrs["kW"]) // scalar_attrs["sW"] + 1
                            expected_shape = (source_type.shape[0], source_type.shape[1], oh, ow)
                            if oh < 0 or ow < 0 or result_type.shape != expected_shape:
                                _known_contract_diagnostic(
                                    diagnostics,
                                    f"{base}.type.shape",
                                    "pool result shape does not match the NCHW window formula",
                                    ref,
                                )
                    if len(results) == 2:
                        if fn != "max":
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.result",
                                "only max pool may produce an indices result",
                                ref,
                            )
                        index_type = _known_tensor_type(
                            value_types,
                            results[1],
                            path=f"{base}.result[1]",
                            diagnostics=diagnostics,
                            ref=ref,
                        )
                        if index_type is not None and result_type is not None and (
                            index_type.shape != result_type.shape or index_type.dtype != "i64"
                        ):
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.result[1].type",
                                "pool indices must match output shape and use i64",
                                ref,
                            )
            continue

        if op_name == "upsample":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="upsample",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"mode", "scale"}),
                required=frozenset({"mode", "scale"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="upsample",
            )
            if attrs.get("mode") not in _UPSAMPLE_MODES:
                _known_contract_diagnostic(
                    diagnostics,
                    f"{base}.attrs.mode",
                    "upsample mode must be 'nearest' or 'bicubic'",
                    ref,
                )
                attrs_ok = False
            scale = _known_integer_sequence(
                attrs.get("scale"),
                base=base,
                attribute="scale",
                diagnostics=diagnostics,
                ref=ref,
                minimum=1,
                nonempty=True,
            )
            if arity_ok and attrs_ok and scale is not None:
                source_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if source_type is not None and result_type is not None:
                    if len(source_type.shape) != 4 or len(scale) != 2:
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.shape",
                            "upsample requires rank-4 NCHW input and a two-element scale",
                            ref,
                        )
                    else:
                        expected_shape = list(source_type.shape)
                        if type(expected_shape[2]) is int:
                            expected_shape[2] *= scale[0]
                        if type(expected_shape[3]) is int:
                            expected_shape[3] *= scale[1]
                        if result_type.shape != tuple(expected_shape) or result_type.dtype != source_type.dtype:
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.type",
                                "upsample result must preserve dtype/N/C and scale H/W",
                                ref,
                            )
            continue

        if op_name == "conv":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=(2, 3),
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="conv",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset({"stride", "padding", "dilation", "groups"}),
                required=frozenset({"stride", "padding", "dilation", "groups"}),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="conv",
            )
            groups = _known_integer_attr(
                attrs.get("groups"),
                base=base,
                attribute="groups",
                diagnostics=diagnostics,
                ref=ref,
                minimum=1,
            )
            if arity_ok and attrs_ok:
                input_type = _known_tensor_type(
                    value_types,
                    operands[0],
                    path=f"{base}.operands[0]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                weight_type = _known_tensor_type(
                    value_types,
                    operands[1],
                    path=f"{base}.operands[1]",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                result_type = _known_tensor_type(
                    value_types,
                    results[0],
                    path=f"{base}.result",
                    diagnostics=diagnostics,
                    ref=ref,
                )
                if input_type is not None and weight_type is not None and result_type is not None:
                    spatial_rank = len(input_type.shape) - 2
                    if spatial_rank not in {1, 2, 3} or len(weight_type.shape) != len(input_type.shape):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type.shape",
                            "conv input and weight must be rank 3, 4, or 5 with matching ranks",
                            ref,
                        )
                    else:
                        sequences = {
                            "stride": _known_integer_sequence(
                                attrs.get("stride"), base=base, attribute="stride",
                                diagnostics=diagnostics, ref=ref, minimum=1,
                            ),
                            "padding": _known_integer_sequence(
                                attrs.get("padding"), base=base, attribute="padding",
                                diagnostics=diagnostics, ref=ref, minimum=0,
                            ),
                            "dilation": _known_integer_sequence(
                                attrs.get("dilation"), base=base, attribute="dilation",
                                diagnostics=diagnostics, ref=ref, minimum=1,
                            ),
                        }
                        if any(value is None or len(value) != spatial_rank for value in sequences.values()):
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.attrs",
                                "conv stride, padding, and dilation lengths must equal the spatial rank",
                                ref,
                            )
                        if groups is not None:
                            cin, cout, weight_cin = input_type.shape[1], weight_type.shape[0], weight_type.shape[1]
                            if all(type(value) is int for value in (cin, cout, weight_cin)):
                                if cin % groups != 0 or cout % groups != 0 or weight_cin != cin // groups:
                                    _known_contract_diagnostic(
                                        diagnostics,
                                        f"{base}.type.shape",
                                        "conv channel dimensions are incompatible with groups",
                                        ref,
                                    )
                        if len(operands) == 3:
                            bias_type = _known_tensor_type(
                                value_types,
                                operands[2],
                                path=f"{base}.operands[2]",
                                diagnostics=diagnostics,
                                ref=ref,
                            )
                            if bias_type is not None and (
                                len(bias_type.shape) != 1
                                or bias_type.shape[0] != weight_type.shape[0]
                                or bias_type.dtype != input_type.dtype
                            ):
                                _known_contract_diagnostic(
                                    diagnostics,
                                    f"{base}.operands[2].type",
                                    "conv bias must be a [Cout] tensor with the input dtype",
                                    ref,
                                )
                        if (
                            weight_type.dtype != input_type.dtype
                            or result_type.dtype != input_type.dtype
                            or result_type.shape[:2] != (input_type.shape[0], weight_type.shape[0])
                        ):
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.type",
                                "conv input, weight, bias, and result must use one dtype and result N/C must be [input N, weight Cout]",
                                ref,
                            )
                        if all(
                            type(input_type.shape[2 + index]) is int
                            and type(weight_type.shape[2 + index]) is int
                            and sequences[name] is not None
                            for index in range(spatial_rank)
                            for name in ("stride", "padding", "dilation")
                        ):
                            expected_spatial = tuple(
                                (
                                    input_type.shape[2 + index]
                                    + 2 * sequences["padding"][index]
                                    - sequences["dilation"][index]
                                    * (weight_type.shape[2 + index] - 1)
                                    - 1
                                )
                                // sequences["stride"][index]
                                + 1
                                for index in range(spatial_rank)
                            )
                            if any(value < 0 for value in expected_spatial) or result_type.shape[2:] != expected_spatial:
                                _known_contract_diagnostic(
                                    diagnostics,
                                    f"{base}.type.shape",
                                    "conv result spatial shape does not match the convolution formula",
                                    ref,
                                )
            continue

        if op_name == "nonzero":
            operands, results, arity_ok = _known_arity(
                operation,
                operands=1,
                results=1,
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="nonzero",
            )
            attrs_ok = _known_exact_attrs(
                attrs,
                allowed=frozenset(),
                required=frozenset(),
                base=base,
                diagnostics=diagnostics,
                ref=ref,
                label="nonzero",
            )
            if arity_ok and attrs_ok:
                source_type = _known_tensor_type(value_types, operands[0], path=f"{base}.operands[0]", diagnostics=diagnostics, ref=ref)
                result_type = _known_tensor_type(value_types, results[0], path=f"{base}.result", diagnostics=diagnostics, ref=ref)
                if source_type is not None and result_type is not None:
                    if len(result_type.shape) != 2 or result_type.dtype != "i64" or result_type.shape[1] != len(source_type.shape):
                        _known_contract_diagnostic(
                            diagnostics,
                            f"{base}.type",
                            "nonzero result must be dynamic rank-2 [nnz, input_rank] i64",
                            ref,
                        )
                    if all(type(dim) is int for dim in source_type.shape) and result_type.shape[0] != "?":
                        expected_capacity = math.prod(source_type.shape) if source_type.shape else 1
                        if result_type.shape[0] != expected_capacity:
                            _known_contract_diagnostic(
                                diagnostics,
                                f"{base}.type.shape[0]",
                                "nonzero dynamic leading extent must use '?'",
                                ref,
                            )
            continue

        if op_name == "masked_select":
            operands, results, arity_ok = _known_arity(operation, operands=2, results=1, base=base, diagnostics=diagnostics, ref=ref, label="masked_select")
            attrs_ok = _known_exact_attrs(attrs, allowed=frozenset(), required=frozenset(), base=base, diagnostics=diagnostics, ref=ref, label="masked_select")
            if arity_ok and attrs_ok:
                data_type = _known_tensor_type(value_types, operands[0], path=f"{base}.operands[0]", diagnostics=diagnostics, ref=ref)
                mask_type = _known_tensor_type(value_types, operands[1], path=f"{base}.operands[1]", diagnostics=diagnostics, ref=ref)
                result_type = _known_tensor_type(value_types, results[0], path=f"{base}.result", diagnostics=diagnostics, ref=ref)
                if data_type is not None and mask_type is not None and result_type is not None:
                    if mask_type.dtype != "i1" or mask_type.shape != data_type.shape:
                        _known_contract_diagnostic(diagnostics, f"{base}.operands[1].type", "masked_select mask must be i1 with the data shape", ref)
                    if result_type.shape != ("?",) or result_type.dtype != data_type.dtype:
                        _known_contract_diagnostic(diagnostics, f"{base}.type", "masked_select result must be a dynamic rank-1 tensor with the data dtype", ref)
            continue

        if op_name == "masked_scatter":
            operands, results, arity_ok = _known_arity(operation, operands=3, results=1, base=base, diagnostics=diagnostics, ref=ref, label="masked_scatter")
            attrs_ok = _known_exact_attrs(attrs, allowed=frozenset(), required=frozenset(), base=base, diagnostics=diagnostics, ref=ref, label="masked_scatter")
            if arity_ok and attrs_ok:
                data_type = _known_tensor_type(value_types, operands[0], path=f"{base}.operands[0]", diagnostics=diagnostics, ref=ref)
                mask_type = _known_tensor_type(value_types, operands[1], path=f"{base}.operands[1]", diagnostics=diagnostics, ref=ref)
                source_type = _known_tensor_type(value_types, operands[2], path=f"{base}.operands[2]", diagnostics=diagnostics, ref=ref)
                result_type = _known_tensor_type(value_types, results[0], path=f"{base}.result", diagnostics=diagnostics, ref=ref)
                if data_type is not None and mask_type is not None and source_type is not None and result_type is not None:
                    if mask_type.dtype != "i1" or mask_type.shape != data_type.shape:
                        _known_contract_diagnostic(diagnostics, f"{base}.operands[1].type", "masked_scatter mask must be i1 with the data shape", ref)
                    if len(source_type.shape) != 1 or source_type.dtype != data_type.dtype:
                        _known_contract_diagnostic(diagnostics, f"{base}.operands[2].type", "masked_scatter source must be a rank-1 vector with the data dtype", ref)
                    if result_type != data_type:
                        _known_contract_diagnostic(diagnostics, f"{base}.type", "masked_scatter result must preserve the data type", ref)
            continue

        if op_name == "polar":
            operands, results, arity_ok = _known_arity(operation, operands=2, results=2, base=base, diagnostics=diagnostics, ref=ref, label="polar")
            attrs_ok = _known_exact_attrs(attrs, allowed=frozenset(), required=frozenset(), base=base, diagnostics=diagnostics, ref=ref, label="polar")
            if arity_ok and attrs_ok:
                absolute = _known_tensor_type(value_types, operands[0], path=f"{base}.operands[0]", diagnostics=diagnostics, ref=ref)
                angle = _known_tensor_type(value_types, operands[1], path=f"{base}.operands[1]", diagnostics=diagnostics, ref=ref)
                real = _known_tensor_type(value_types, results[0], path=f"{base}.result[0]", diagnostics=diagnostics, ref=ref)
                imaginary = _known_tensor_type(value_types, results[1], path=f"{base}.result[1]", diagnostics=diagnostics, ref=ref)
                if all(item is not None for item in (absolute, angle, real, imaginary)):
                    assert absolute is not None and angle is not None and real is not None and imaginary is not None
                    if not is_closed_floating_dtype(absolute.dtype) or angle != absolute or real != absolute or imaginary != absolute:
                        _known_contract_diagnostic(diagnostics, f"{base}.type", "polar inputs and both results must share one closed floating tensor type", ref)
            continue

        if op_name != "reduce":
            continue

        for key in sorted(set(attrs) - _REDUCE_ATTRS):
            detail = (
                "reduce uses canonical attr 'fn'; alias 'kind' is forbidden "
                "and is never translated by fixed code"
                if key == "kind"
                else f"reduce attrs contain unsupported key {key!r}"
            )
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.{key}",
                    detail,
                    ref,
                )
            )

        fn = attrs.get("fn")
        if not isinstance(fn, str) or fn not in _REDUCE_FNS:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.fn",
                    "reduce fn must be one of "
                    f"{sorted(_REDUCE_FNS)!r}; aliases are not accepted",
                    ref,
                )
            )

        assoc = attrs.get("assoc")
        if not isinstance(assoc, str) or assoc not in _REDUCE_ASSOCIATIVITY:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.assoc",
                    "reduce assoc must be 'associative' or "
                    "'float_nondeterministic'",
                    ref,
                )
            )

        operands = operation.get("operands")
        operand_refs = operands if isinstance(operands, list) else []
        if len(operand_refs) != 1 or not isinstance(operand_refs[0], str):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.operands",
                    "reduce requires exactly one value operand",
                    ref,
                )
            )
            operand_rank = None
        else:
            operand_type = value_types.get(operand_refs[0])
            operand_rank = (
                len(operand_type.shape) if operand_type is not None else None
            )

        axes = attrs.get("axes")
        if not isinstance(axes, list) or not axes:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.axes",
                    "reduce axes must be a non-empty array of integer axes",
                    ref,
                )
            )
        else:
            normalized_axes: set[int] = set()
            for index, axis in enumerate(axes):
                axis_path = f"{base}.attrs.axes[{index}]"
                if type(axis) is not int:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            axis_path,
                            "reduce axis must be an integer, not bool or text",
                            ref,
                        )
                    )
                    continue
                normalized = (
                    axis + operand_rank
                    if operand_rank is not None and axis < 0
                    else axis
                )
                if operand_rank is not None and not 0 <= normalized < operand_rank:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            axis_path,
                            f"reduce axis {axis} is out of range for "
                            f"rank-{operand_rank} operand",
                            ref,
                        )
                    )
                    continue
                if normalized in normalized_axes:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                            axis_path,
                            "reduce axes must be unique after negative-axis "
                            "normalization",
                            ref,
                        )
                    )
                    continue
                normalized_axes.add(normalized)

        if "keepdims" in attrs and type(attrs["keepdims"]) is not bool:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID,
                    f"{base}.attrs.keepdims",
                    "optional reduce keepdims must be a boolean",
                    ref,
                )
            )

    return tuple(
        sorted(
            diagnostics,
            key=lambda item: (item.path, item.code.value, item.detail),
        )
    )


def _ordered_expression_refs(expression: Any) -> tuple[str, ...]:
    result: list[str] = []

    def visit(node: Any) -> None:
        if getattr(node, "op", None) == "ref":
            value = getattr(node, "value", None)
            if isinstance(value, str) and value not in result:
                result.append(value)
        for argument in getattr(node, "args", ()):
            visit(argument)

    visit(expression)
    return tuple(result)


def _check_boundary_value_dominance(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Require Layer-B access expressions to use visible Layer-A values.

    ``SemanticArchive`` already checks ordinary Layer-A operands by walking
    each operation sequence with an incrementally extended visible-value set.
    Boundary index/predicate expressions are stored separately from that
    sequence, so their Layer-A value references need the same source-free
    use-order check at the access event selected by ``access.node_ref``.  Axis
    and dimension references are excluded simply because they are not members
    of the Layer-A value set; the checker does not infer an alternate meaning
    from their position in an index expression.
    """

    inputs = {item.name for item in j.intent.inputs}
    canonical_body = _canonical_operation_body(j)
    all_values = set(inputs)
    visible_before: dict[str, frozenset[str]] = {}
    for path, _operation, visible in walk_intent_op_scopes(
        canonical_body,
        visible=inputs,
    ):
        visible_before[path] = frozenset(visible)
    for _path, operation in _canonical_operation_items(j):
        all_values.update(_operation_results(operation.get("result")))
        all_values.update(_operation_dependencies(operation))
    visible_at_function_exit = set(
        intent_op_producers(canonical_body, visible=inputs)
    )
    elaboration = j_elaboration(j)
    visible_by_access: dict[str, tuple[frozenset[str], str]] = {}
    for index, access in enumerate(j.boundary.accesses):
        parsed = elaboration.access_by_id.get(access.access_id)
        if parsed is None:
            continue
        if access.node_ref is None:
            visible = visible_at_function_exit
            event_ref = "function exit"
        else:
            path = j.node_ids.get(access.node_ref, access.node_ref)
            visible = visible_before.get(path)
            if visible is None:
                # Ordinary reference checking owns unknown node handles.
                continue
            event_ref = f"access node {access.node_ref!r}"
        visible_by_access[access.access_id] = (visible, event_ref)

        expressions: list[tuple[str, Any]] = [
            (
                f"j.boundary.accesses[{index}].index_expression",
                component,
            )
            for component in parsed.index.components
        ]
        if parsed.predicate is not None:
            expressions.append(
                (
                    f"j.boundary.accesses[{index}].predicate",
                    parsed.predicate,
                )
            )
        for expression_path, expression in expressions:
            for value_ref in _ordered_expression_refs(expression):
                if value_ref not in all_values or value_ref in visible:
                    continue
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.BOUNDARY_VALUE_NOT_DOMINATING,
                        expression_path,
                        f"Layer-A value {value_ref!r} does not dominate "
                        f"the {event_ref}",
                        access.access_id,
                    )
                )

    for index, relation in enumerate(j.boundary.access_guard_relations):
        visibility = visible_by_access.get(relation.access_ref)
        if visibility is None:
            # Boundary/model reference closure owns an unknown access.
            continue
        visible, event_ref = visibility
        if relation.value_ref not in all_values or relation.value_ref in visible:
            continue
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.BOUNDARY_VALUE_NOT_DOMINATING,
                f"j.boundary.access_guard_relations[{index}].value_ref",
                f"Layer-A value {relation.value_ref!r} does not dominate "
                f"the {event_ref}",
                relation.access_ref,
            )
        )


def _check_access_guard_relations(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check the closed v5 access-local computed-guard contract.

    The relation is entirely J-owned.  This routine validates authored facts;
    it never copies a predicate, invents broadcast axes, inserts a ``select``,
    changes an output parameter into ``inout``, or consults pi.
    """

    if not j.boundary.access_guard_relations:
        return

    from intentir.ir.intent_op_contract import _broadcast_projection

    elaboration = j_elaboration(j)
    operation_records = elaboration.operation_records
    by_path: dict[str, Mapping[str, Any]] = {}
    loop_stack_by_path: dict[str, tuple[tuple[str, Mapping[str, Any]], ...]] = {}
    declared_values = {item.name for item in j.intent.inputs}
    scope_by_id = elaboration.scope_by_id
    loop_by_scope = {
        record.get("wire", {}).get("scope_id"): record
        for record in operation_records
        if record.get("op") == "loop"
        and isinstance(record.get("wire"), Mapping)
        and isinstance(record["wire"].get("scope_id"), str)
    }
    for record in operation_records:
        operation = record.get("wire")
        if not isinstance(operation, Mapping):
            continue
        path = str(record["path"])
        by_path[path] = operation
        stack: list[tuple[str, Mapping[str, Any]]] = []
        scope_id = record.get("region_scope_id")
        while isinstance(scope_id, str):
            scope = scope_by_id.get(scope_id)
            if scope is None:
                break
            if scope.get("kind") == "loop_region":
                loop_record = loop_by_scope.get(scope_id)
                if isinstance(loop_record, Mapping) and isinstance(
                    loop_record.get("wire"), Mapping
                ):
                    stack.append((str(loop_record["path"]), loop_record["wire"]))
            scope_id = scope.get("parent_scope")
        loop_stack_by_path[path] = tuple(reversed(stack))
        declared_values.update(
            result for result in record.get("results", ()) if isinstance(result, str)
        )
        for scan in operation.get("scan", ()):
            if isinstance(scan, Mapping) and isinstance(scan.get("name"), str):
                declared_values.add(str(scan["name"]))
        for carry in operation.get("carry", ()):
            if isinstance(carry, Mapping) and isinstance(
                carry.get("body_argument"), str
            ):
                declared_values.add(str(carry["body_argument"]))
    nodes = dict(by_path)
    for node_ref, path in j.node_ids.items():
        operation = by_path.get(path)
        if operation is not None:
            nodes[node_ref] = operation

    accesses = {item.access_id: item for item in j.boundary.accesses}
    value_types = elaboration.value_type_by_name
    input_names = {
        item.name for item in j.intent.inputs
    }
    output_names = {
        binding.output for binding in elaboration.output_bindings
    }
    output_values = {
        item.output: item.value for item in elaboration.output_bindings
    }
    parameters_by_logical: dict[str, list[Any]] = {}
    for parameter in j.signature.parameters:
        if parameter.logical_ref is not None:
            parameters_by_logical.setdefault(
                parameter.logical_ref, []
            ).append(parameter)

    seen_pairs: set[tuple[str, str]] = set()
    relations_by_access: dict[str, list[int]] = {}
    carry_relations: dict[tuple[str, int], list[int]] = {}

    def reject(
        index: int,
        field: str,
        detail: str,
        ref: str | None = None,
    ) -> None:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.ACCESS_GUARD_RELATION_INVALID,
                f"j.boundary.access_guard_relations[{index}].{field}",
                detail,
                ref,
            )
        )

    for index, relation in enumerate(j.boundary.access_guard_relations):
        pair = (relation.access_ref, relation.value_ref)
        if pair in seen_pairs:
            reject(
                index,
                "access_ref",
                "one (access_ref, value_ref) pair may be declared only once",
                relation.access_ref,
            )
        seen_pairs.add(pair)
        relations_by_access.setdefault(relation.access_ref, []).append(index)

        access = accesses.get(relation.access_ref)
        if access is None:
            # The strict Boundary model owns unknown access references.
            continue
        guard_type = value_types.get(relation.value_ref)
        if relation.value_ref not in declared_values or guard_type is None:
            reject(
                index,
                "value_ref",
                "value_ref must name a real typed Layer-A declaration",
                relation.value_ref,
            )
            continue
        if (
            not guard_type.is_tensor
            or not is_closed_boolean_dtype(guard_type.dtype)
        ):
            reject(
                index,
                "value_ref",
                "access guard value_ref must have canonical dtype i1",
                relation.value_ref,
            )

        axes = tuple(relation.axes)
        if axes != tuple(sorted(set(axes))):
            reject(
                index,
                "axes",
                "access guard axes must be strictly increasing and unique",
                relation.access_ref,
            )

        target_type = value_types.get(access.tensor)
        if target_type is None:
            reject(
                index,
                "access_ref",
                "access target must have one closed Layer-A tensor type",
                access.tensor,
            )
        else:
            try:
                _broadcast_projection(
                    list(guard_type.shape),
                    axes,
                    list(target_type.shape),
                    unit_dimensions=direct_unit_dimension_refs(
                        j.dimension_constraints
                    ),
                )
            except (NotImplementedError, ValueError) as exc:
                reject(
                    index,
                    "axes",
                    "guard shape cannot project to the access target using "
                    f"the authored broadcast axes: {exc}",
                    relation.access_ref,
                )

        parsed_access = elaboration.access_by_id.get(access.access_id)
        if (
            parsed_access is not None
            and parsed_access.predicate is not None
            and relation.value_ref
            in _ordered_expression_refs(parsed_access.predicate)
        ):
            reject(
                index,
                "value_ref",
                "the relation-bound computed guard must not also appear in "
                "the access predicate",
                relation.access_ref,
            )

        if access.kind is AccessKind.LOAD:
            if access.fill_value is None:
                reject(
                    index,
                    "access_ref",
                    "a relation-guarded load requires an explicit fill_value",
                    relation.access_ref,
                )
            continue
        if access.kind is AccessKind.UPDATE:
            reject(
                index,
                "access_ref",
                "access-guard relation v1 has no closed update merge contract",
                relation.access_ref,
            )
            continue

        assert access.kind is AccessKind.STORE
        # A physical tail-write guard has a second closed representation that
        # deliberately does *not* manufacture an inout pre-state or a select
        # merge.  It is valid only when the provider-authored store owner is
        # itself the one complete, output-only logical result.  The neutral
        # boundary predicate means the local i1 is a realization guard rather
        # than a J logical-domain restriction.
        target_parameters = parameters_by_logical.get(access.tensor, [])
        parsed_access = elaboration.access_by_id.get(access.access_id)
        parsed_predicate = (
            parsed_access.predicate if parsed_access is not None else None
        )
        is_literal_true = (
            access.predicate == "true"
            and parsed_predicate is not None
            and parsed_predicate.op == "bool"
            and parsed_predicate.value is True
            and not parsed_predicate.args
        )
        has_output_only_target = (
            access.tensor in output_names
            and access.tensor not in input_names
            and len(target_parameters) == 1
            and target_parameters[0].mode is ParameterMode.OUTPUT
        )
        if has_output_only_target:
            if not is_literal_true:
                reject(
                    index,
                    "access_ref",
                    "an output-only realization-guarded store requires the "
                    "exact literal logical predicate true",
                    relation.access_ref,
                )
            operation = nodes.get(access.node_ref or "")
            result = operation.get("result") if operation is not None else None
            if operation is None or not isinstance(result, str):
                reject(
                    index,
                    "access_ref",
                    "an output-only realization-guarded store requires one "
                    "explicit single-result Layer-A owner node",
                    relation.access_ref,
                )
                continue
            if relation.value_ref in operation.get("operands", ()):
                reject(
                    index,
                    "value_ref",
                    "the physical realization guard must not also be a logical "
                    "operand of the output producer",
                    relation.access_ref,
                )
            if output_values.get(access.tensor) != result:
                reject(
                    index,
                    "access_ref",
                    "the output-only realization-guarded store owner result must "
                    "be the exact observable output binding",
                    relation.access_ref,
                )
            result_type = value_types.get(result)
            target_type = value_types.get(access.tensor)
            if result_type is None or target_type is None or result_type != target_type:
                reject(
                    index,
                    "access_ref",
                    "the output-only realization-guarded store owner result type "
                    "must equal the output tensor type exactly",
                    relation.access_ref,
                )
            duplicate_guard_broadcasts = [
                item
                for item in j.boundary.broadcasts
                if item.operand == relation.value_ref
                and nodes.get(item.node_ref) is operation
            ]
            if duplicate_guard_broadcasts:
                reject(
                    index,
                    "axes",
                    "access_guard_relations is the unique physical-guard axes "
                    "authority; boundary.broadcasts must not duplicate it",
                    relation.access_ref,
                )
            continue

        access_path = (
            j.node_ids.get(access.node_ref, access.node_ref)
            if access.node_ref is not None
            else None
        )
        loop_stack = (
            loop_stack_by_path.get(access_path, ())
            if access_path is not None
            else ()
        )
        observable_value = output_values[access.tensor]
        matches: list[tuple[str, int, Mapping[str, Any]]] = []
        for loop_path, loop in loop_stack:
            raw_carries = loop.get("carry")
            carries = raw_carries if isinstance(raw_carries, list) else []
            for ordinal, carry in enumerate(carries):
                if (
                    isinstance(carry, Mapping)
                    and carry.get("result_value") == observable_value
                ):
                    matches.append((loop_path, ordinal, carry))

        has_real_inout = (
            access.tensor in input_names
            and access.tensor in output_names
            and len(target_parameters) == 1
            and target_parameters[0].mode is ParameterMode.INOUT
        )
        if not has_real_inout:
            reject(
                index,
                "access_ref",
                "a relation-guarded store must target one tensor declared as "
                "both J input and output and bound by exactly one mode=inout "
                "parameter; TypeCheck never upgrades output to inout",
                relation.access_ref,
            )

        operation = nodes.get(access.node_ref or "")
        operands = operation.get("operands") if operation is not None else None
        result = operation.get("result") if operation is not None else None
        if (
            operation is None
            or operation.get("op") != "select"
            or not isinstance(operands, list)
            or len(operands) != 3
            or not all(isinstance(item, str) for item in operands)
            or not isinstance(result, str)
        ):
            reject(
                index,
                "access_ref",
                "a relation-guarded store must anchor one explicit "
                "three-operand Layer-A select merge",
                relation.access_ref,
            )
            continue
        if operands[0] != relation.value_ref:
            reject(
                index,
                "value_ref",
                "the explicit select condition must equal relation.value_ref",
                relation.access_ref,
            )

        if loop_stack:
            if len(matches) != 1:
                reject(
                    index,
                    "access_ref",
                    "a relation-guarded store inside a structured loop must "
                    "map to exactly one enclosing carry result",
                    relation.access_ref,
                )
                continue
            loop_path, carry_ordinal, carry = matches[0]
            carry_relations.setdefault(
                (loop_path, carry_ordinal), []
            ).append(index)
            if carry.get("init_value") != access.tensor:
                reject(
                    index,
                    "access_ref",
                    "the guarded carry init_value must equal the exact inout "
                    "access tensor pre-state",
                    relation.access_ref,
                )
            if carry.get("next_value") != result:
                reject(
                    index,
                    "access_ref",
                    "carry.next_value must be the explicit select merge result",
                    relation.access_ref,
                )
            current = carry.get("body_argument")
        else:
            current = access.tensor
            matching_mappings = [
                item
                for item in j.boundary.output_mappings
                if item.output == access.tensor
                and item.value == result
                and item.node_ref is not None
                and nodes.get(item.node_ref) is operation
            ]
            if (
                output_values[access.tensor] != result
                or len(matching_mappings) > 1
            ):
                reject(
                    index,
                    "access_ref",
                    "a non-loop guarded store canonical output binding must "
                    "name the explicit select merge result exactly once",
                    relation.access_ref,
                )

        inactive_index = (
            2
            if relation.polarity is AccessGuardPolarity.POSITIVE
            else 1
        )
        active_index = 1 if inactive_index == 2 else 2
        if operands[inactive_index] != current:
            reject(
                index,
                "polarity",
                "the select branch inactive under the authored polarity must "
                "be the current carry body_argument",
                relation.access_ref,
            )
        if operands[active_index] == current:
            reject(
                index,
                "polarity",
                "the select branch active under the authored polarity must "
                "be the candidate update, not the current carry",
                relation.access_ref,
            )

        duplicate_guard_broadcasts = [
            item
            for item in j.boundary.broadcasts
            if item.operand == relation.value_ref
            and nodes.get(item.node_ref) is operation
        ]
        if duplicate_guard_broadcasts:
            reject(
                index,
                "axes",
                "access_guard_relations is the unique select-guard axes "
                "authority; boundary.broadcasts must not duplicate it",
                relation.access_ref,
            )

    for access_ref, indexes in sorted(relations_by_access.items()):
        if len(indexes) <= 1:
            continue
        for index in indexes:
            reject(
                index,
                "access_ref",
                "access-guard relation v1 permits exactly one guard per access",
                access_ref,
            )
    for (_loop_path, _carry_ordinal), indexes in sorted(carry_relations.items()):
        if len(indexes) <= 1:
            continue
        for index in indexes:
            reject(
                index,
                "access_ref",
                "access-guard relation v1 fails closed on multiple guarded "
                "writes to one loop carry",
                j.boundary.access_guard_relations[index].access_ref,
            )


def _axis_refs_v1(
    j: SemanticArchive,
    subject_ref: str,
    node_operations: Mapping[str, Mapping[str, Any]],
) -> frozenset[int | str]:
    """Return the closed dependent axis domain for one target E_C attachment."""

    dimensions = {
        reference
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dim in tensor.shape
        for reference in dimension_expression_references(dim)
        if re.fullmatch(r"x[0-9]+", reference)
    }
    if re.fullmatch(r"n[0-9]+", subject_ref) is not None:
        operation = node_operations.get(subject_ref)
        if operation is None:
            return frozenset()
        if operation.get("op") == "reduce":
            attrs = operation.get("attrs")
            attrs = attrs if isinstance(attrs, Mapping) else {}
            axes = attrs.get("axes")
            if not isinstance(axes, list):
                return frozenset()
            return frozenset(
                axis
                for axis in axes
                if isinstance(axis, int)
                and not isinstance(axis, bool)
                and axis >= 0
            )
        if operation.get("op") == "loop":
            trip = operation.get("trip")
            axis = trip.get("axis") if isinstance(trip, Mapping) else None
            result: set[int | str] = set()
            if isinstance(axis, str) and axis in dimensions:
                result.add(0)
                result.add(axis)
            return frozenset(result)
        if operation.get("op") == "contraction":
            attrs = operation.get("attrs")
            subscripts = attrs.get("subscripts") if isinstance(attrs, Mapping) else None
            operands = operation.get("operands")
            if not isinstance(subscripts, str) or not isinstance(operands, list):
                return frozenset()
            try:
                input_text, output_text = subscripts.split("->", 1)
                input_terms = input_text.split(",")
            except ValueError:
                return frozenset()
            if len(input_terms) != len(operands):
                return frozenset()
            contracted: list[str] = []
            for term in input_terms:
                for label in term:
                    if label not in output_text and label not in contracted:
                        contracted.append(label)
            result: set[int | str] = set(range(len(contracted)))
            value_types = j_elaboration(j).value_type_by_name
            for term, operand in zip(input_terms, operands, strict=True):
                value_type = value_types.get(operand) if isinstance(operand, str) else None
                if value_type is None or len(term) != len(value_type.shape):
                    continue
                for label, extent in zip(term, value_type.shape, strict=True):
                    if label in contracted:
                        result.update(
                            reference
                            for reference in dimension_expression_references(extent)
                            if reference in dimensions
                        )
            return frozenset(result)
        return frozenset()

    elaboration = j_elaboration(j)
    if re.fullmatch(r"t[0-9]+", subject_ref) is not None:
        tensor = next(
            (
                item
                for item in [*j.intent.inputs, *j.intent.outputs]
                if item.name == subject_ref
            ),
            None,
        )
        if tensor is None:
            return frozenset()
        return frozenset(
            [
                *range(len(tensor.shape)),
                *(
                    reference
                    for dim in tensor.shape
                    for reference in dimension_expression_references(dim)
                    if reference in dimensions
                ),
            ]
        )
    if re.fullmatch(r"d[0-9]+", subject_ref) is not None:
        expression = elaboration.domain_by_id.get(subject_ref)
        if expression is None:
            return frozenset()
        refs = _ordered_expression_refs(expression)
        logical_axes = tuple(ref for ref in refs if ref not in dimensions)
        return frozenset(
            [*range(len(logical_axes)), *(ref for ref in refs if ref in dimensions)]
        )
    if re.fullmatch(r"a[0-9]+", subject_ref) is not None:
        access = elaboration.access_by_id.get(subject_ref)
        if access is None:
            return frozenset()
        access_spec = next(
            (
                item
                for item in j.boundary.accesses
                if item.access_id == subject_ref
            ),
            None,
        )
        tensors = {
            item.name: item for item in [*j.intent.inputs, *j.intent.outputs]
        }
        tensor = (
            tensors.get(access_spec.tensor) if access_spec is not None else None
        )
        symbolic = {
            reference
            for index, dim in enumerate(tensor.shape if tensor is not None else [])
            for reference in dimension_expression_references(dim)
            if index < len(access.index.components) and reference in dimensions
        }
        return frozenset([*range(len(access.index.components)), *symbolic])
    # Effect/scope/authorization/storage attachments are intentionally axis-free.
    return frozenset()


_TOLERANCE_UNARY_FNS = frozenset(
    {
        "exp", "log", "log2", "sqrt", "rsqrt", "tanh", "sigmoid", "cos",
        "sin", "tan", "exp2", "atan", "erf", "acos", "asin", "cbrt",
        "ceil", "floor", "round", "roundeven", "trunc", "log10", "log1p",
        "expm1", "erfc",
    }
)


def _floating_dtype(dtype: str | None) -> bool:
    return is_closed_floating_dtype(dtype)


def _operation_dependencies(operation: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value not in result:
            result.append(value)

    for value in operation.get("operands", []):
        add(value)
    if operation.get("op") in {"effect_if", "value_if"}:
        add(operation.get("predicate"))
    if operation.get("op") == "value_if":
        for polarity in ("positive", "negative"):
            arm = operation.get(polarity)
            if isinstance(arm, Mapping):
                add(arm.get("yield_value"))
    if operation.get("op") == "loop":
        trip = operation.get("trip")
        if isinstance(trip, Mapping) and "upper_bound" in trip:
            add(trip.get("upper_bound"))
        for item in operation.get("scan", []):
            if isinstance(item, Mapping):
                add(item.get("from"))
        for item in operation.get("carry", []):
            if isinstance(item, Mapping):
                add(item.get("init_value"))
                add(item.get("next_value"))
        for value in operation.get("yield", []):
            add(value)
    return tuple(result)


def _output_dependency(
    j: SemanticArchive, output: str
) -> tuple[set[str], list[Mapping[str, Any]]]:
    operations = dict(_canonical_operation_items(j))
    path_to_node = j_elaboration(j).authored_node_by_path(j.node_ids)
    producers: dict[str, tuple[str, str, Mapping[str, Any]]] = {}
    for path, operation in operations.items():
        node_ref = path_to_node.get(path, path)
        for result in _operation_results(operation.get("result")):
            producers[result] = (node_ref, path, operation)

    bindings = {
        item.output: item for item in j_elaboration(j).output_bindings
    }
    binding = bindings.get(output)
    if binding is None:
        return set(), []

    refs: set[str] = set()
    dependency_ops: list[Mapping[str, Any]] = []
    pending = [binding.value]
    visited_values: set[str] = set()
    while pending:
        value = pending.pop()
        if value in visited_values:
            continue
        visited_values.add(value)
        producer = producers.get(value)
        if producer is None:
            continue
        node_ref, path, operation = producer
        refs.update({node_ref, path})
        dependency_ops.append(operation)
        pending.extend(_operation_dependencies(operation))
    return refs, dependency_ops


def _semantic_value_dependency_closure(
    j: SemanticArchive,
    seeds: Iterable[str],
) -> set[str]:
    """Return values contributing to explicit J semantic roots.

    This is ordinary finite graph reachability over the already typed J.  It
    does not inspect Source, C, pi, parameter names, or runtime values.
    """

    producers: dict[str, Mapping[str, Any]] = {}
    for _path, operation in _canonical_operation_items(j):
        for result in _operation_results(operation.get("result")):
            producers[result] = operation

    live: set[str] = set()
    pending = list(seeds)
    while pending:
        value = pending.pop()
        if value in live:
            continue
        live.add(value)
        producer = producers.get(value)
        if producer is not None:
            pending.extend(_operation_dependencies(producer))
    return live


def _rank_zero_semantic_live_values(j: SemanticArchive) -> set[str]:
    """Collect J-owned roots that can make a rank-0 logical input meaningful."""

    elaboration = j_elaboration(j)
    semantic_refs: set[str] = {
        binding.value for binding in elaboration.output_bindings
    }
    node_seed_refs: set[str] = set()

    operations = dict(_canonical_operation_items(j))
    path_to_node = elaboration.authored_node_by_path(j.node_ids)
    nodes: dict[str, Mapping[str, Any]] = dict(operations)
    for path, operation in operations.items():
        node = path_to_node.get(path)
        if node is not None:
            nodes[node] = operation

    def add_expression_refs(expression: Any) -> None:
        semantic_refs.update(_ordered_expression_refs(expression))

    for _domain_id, expression in elaboration.domains:
        add_expression_refs(expression)
    accesses = elaboration.access_by_id
    for access in j.boundary.accesses:
        semantic_refs.add(access.tensor)
        parsed = accesses.get(access.access_id)
        if parsed is not None:
            for component in parsed.index.components:
                add_expression_refs(component)
            if parsed.predicate is not None:
                add_expression_refs(parsed.predicate)
        if access.node_ref is not None:
            node_seed_refs.add(access.node_ref)
    for effect in j.boundary.effects:
        semantic_refs.add(effect.target)
        if effect.node_ref is not None:
            node_seed_refs.add(effect.node_ref)
    for item in j.boundary.broadcasts:
        semantic_refs.add(item.operand)
        node_seed_refs.add(item.node_ref)
    for item in j.boundary.aliases:
        semantic_refs.update({item.left, item.right})
    for item in j.boundary.output_mappings:
        semantic_refs.add(item.value)
        if item.node_ref is not None:
            node_seed_refs.add(item.node_ref)
    for item in j.boundary.reductions:
        if isinstance(item.tile_axis, str):
            semantic_refs.add(item.tile_axis)
        node_seed_refs.add(item.node_ref)
    for item in j.boundary.index_bounds:
        semantic_refs.update({item.data, item.indices})
        node_seed_refs.add(item.node_ref)
    for item in j.boundary.updates:
        semantic_refs.update({item.base, item.indices, item.values})
        if item.compare is not None:
            semantic_refs.add(item.compare)
        if item.observed is not None:
            semantic_refs.add(item.observed)
        node_seed_refs.add(item.node_ref)
    for tensor in [*j.intent.inputs, *j.intent.outputs]:
        semantic_refs.update(
            reference
            for dimension in tensor.shape
            for reference in dimension_expression_references(dimension)
        )
    for relation in j.value_dimension_relations:
        semantic_refs.add(relation.value_ref)
        semantic_refs.add(relation.dimension_ref)
    for relation in j.dimension_relations:
        semantic_refs.add(relation.result)
        semantic_refs.add(relation.expr.arg)
        if relation.expr.rhs is not None:
            semantic_refs.add(relation.expr.rhs)

    for node_ref in node_seed_refs:
        operation = nodes.get(node_ref)
        if operation is not None:
            semantic_refs.update(_operation_results(operation.get("result")))
    for operation in operations.values():
        if operation.get("op") in {"effect_if", "value_if"} and isinstance(
            operation.get("predicate"), str
        ):
            # A typed conditional predicate controls J-owned effects, so it is
            # semantic liveness rather than a representation-only pi use.
            semantic_refs.add(str(operation["predicate"]))
    return _semantic_value_dependency_closure(j, semantic_refs)


def _check_rank_zero_logical_input_liveness(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Reject only mechanically dead rank-0 signature logical inputs.

    General tensor liveness is deliberately out of scope for v7.  A rank-0 J
    input is accepted when it reaches an observable output dependency or is
    explicitly used by a J-owned boundary/shape relation.  Merely routing its
    ABI parameter through pi is representation use and does not make it a
    logical input.
    """

    rank_zero_inputs = {
        tensor.name
        for tensor in j.intent.inputs
        if not tensor.shape
    }
    has_representation_scalar = any(
        parameter.kind is ParameterKind.SCALAR
        and parameter.logical_ref is None
        for parameter in j.signature.parameters
    )
    if not rank_zero_inputs and not has_representation_scalar:
        return
    live = _rank_zero_semantic_live_values(j)
    representation_j_uses = set(live)
    for _path, operation in _canonical_operation_items(j):
        representation_j_uses.update(_operation_dependencies(operation))
    for index, parameter in enumerate(j.signature.parameters):
        logical_ref = parameter.logical_ref
        if (
            parameter.kind is ParameterKind.SCALAR
            and logical_ref is None
            and parameter.name in representation_j_uses
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REPRESENTATION_CARRIER_J_USE_FORBIDDEN,
                    f"j.signature.parameters[{index}].logical_ref",
                    "representation-only scalar without logical_ref cannot "
                    "participate in J output, boundary, or shape semantics; "
                    "it may be consumed only by physical address/representation "
                    "realization outside J, while pi records topology without "
                    "inventing logical ownership",
                    parameter.name,
                )
            )
            continue
        if logical_ref not in rank_zero_inputs or logical_ref in live:
            continue
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.LOGICAL_INPUT_UNUSED,
                f"j.signature.parameters[{index}].logical_ref",
                "rank-0 logical input does not contribute to any J output "
                "dependency or explicit boundary/shape semantic relation; "
                "a representation-only ABI parameter must use "
                "logical_ref=null and may be realized only by physical address "
                "or representation handling outside J (with pi carrying topology)",
                logical_ref,
            )
        )


def _check_result_relations(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    output_types = {
        binding.output: binding.output_type.dtype
        for binding in j_elaboration(j).output_bindings
    }
    relations = {
        item.output: (index, item)
        for index, item in enumerate(j.boundary.result_relations)
    }
    reassociation_subjects = {
        item.subject_ref
        for item in j.boundary.realization_authorizations
        if item.kind is RefinementKind.REASSOCIATION
    }
    for output, dtype in output_types.items():
        if not _floating_dtype(dtype):
            continue
        dependency_refs, dependency_ops = _output_dependency(j, output)
        requires_tolerance = bool(dependency_refs & reassociation_subjects)
        for operation in dependency_ops:
            result_type = _explicit_type(operation)
            if (
                operation.get("op") == "reduce"
                and result_type is not None
                and _floating_dtype(result_type.dtype)
            ):
                requires_tolerance = True
            attrs = operation.get("attrs")
            attrs = attrs if isinstance(attrs, Mapping) else {}
            if (
                operation.get("op") == "map_unary"
                and attrs.get("fn") in _TOLERANCE_UNARY_FNS
            ):
                requires_tolerance = True
        if not requires_tolerance:
            continue
        indexed = relations.get(output)
        relation = indexed[1] if indexed is not None else None
        if relation is not None and relation.kind is ResultRelationKind.TOLERANCE:
            continue
        path = (
            f"j.boundary.result_relations[{indexed[0]}].kind"
            if indexed is not None else "j.boundary.result_relations"
        )
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.RESULT_RELATION_INCOMPATIBLE,
                path,
                "floating transcendental/reduction/reassociation output "
                "requires explicit tolerance relation",
                output,
            )
        )




def _build_anchors(
    j: SemanticArchive,
    c: ImplementationConfig,
    diagnostics: list[TypeCheckDiagnostic],
) -> tuple[dict[str, _Anchor], dict[str, Mapping[str, Any]]]:
    anchors: dict[str, _Anchor] = {}

    def add(ref: str, anchor: _Anchor, path: str) -> None:
        previous = anchors.get(ref)
        if previous is not None and previous != anchor:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.DUPLICATE_J_REF,
                    path,
                    f"handle {ref!r} has more than one J declaration",
                    ref,
                )
            )
            return
        anchors[ref] = anchor

    elaboration = j_elaboration(j)
    value_types: dict[str, _ValueType] = {
        name: _ValueType(value.dtype, len(value.shape), value.kind)
        for name, value in elaboration.value_type_by_name.items()
    }
    for tensor in [*j.intent.inputs, *j.intent.outputs]:
        value_type = value_types.get(tensor.name)
        if value_type is None:
            # A mutated in-memory archive may be unable to elaborate.  Keep
            # the diagnostic anchor available, but valid accepted J always
            # reaches this branch with the canonical value-type record.
            value_type = _ValueType(tensor.dtype, len(tensor.shape))
        add(
            tensor.name,
            _Anchor("j", "tensor", value_type, carrier_class="value"),
            "j.signature",
        )
        for dim in tensor.shape:
            if isinstance(dim, str):
                add(
                    dim,
                    _Anchor("j", "dimension", _ValueType("index", 0)),
                    "j.signature",
                )

    for parameter in j.signature.parameters:
        add(
            parameter.name,
            _Anchor(
                "j",
                "signature",
                (
                    _ValueType(parameter.dtype, parameter.rank or 0)
                    if parameter.dtype is not None
                    else None
                ),
                carrier_class=(
                    "pointer"
                    if parameter.kind is ParameterKind.POINTER
                    else "value"
                    if parameter.kind is not ParameterKind.OPAQUE
                    else None
                ),
            ),
            f"j.signature.parameters[{parameter.name}]",
        )

    operations = dict(_canonical_operation_items(j))

    node_operations: dict[str, Mapping[str, Any]] = {}
    for node_id, path in j.node_ids.items():
        operation = operations.get(path)
        if operation is None:
            continue
        node_operations[node_id] = operation
        operands = operation.get("operands")
        operand_refs = operands if isinstance(operands, list) else []
        operand_types = tuple(
            value_types.get(item) if isinstance(item, str) else None
            for item in operand_refs
        )
        result_type = next(
            (
                value_types[result]
                for result in _operation_results(operation.get("result"))
                if result in value_types
            ),
            None,
        )
        anchor = _Anchor(
            "j", "node", result_type, operand_types, carrier_class="value"
        )
        add(node_id, anchor, f"j.node_ids.{node_id}")
        add(path, anchor, f"j.node_ids.{node_id}")
        node_operations[path] = operation
    for value, value_type in value_types.items():
        if value not in anchors:
            add(
                value,
                _Anchor("j", "value", value_type, carrier_class="value"),
                "j.intent",
            )

    for node_ref, operation in node_operations.items():
        if not re.fullmatch(r"n[0-9]+", node_ref) or operation.get("op") != "loop":
            continue
        carries = operation.get("carry")
        if not isinstance(carries, list):
            continue
        for ordinal, carry in enumerate(carries):
            if not isinstance(carry, Mapping):
                continue
            state_ref = carry.get("state_id")
            type_value = carry.get("type")
            if not isinstance(state_ref, str) or not isinstance(type_value, Mapping):
                continue
            if type_value.get("kind") == "record":
                try:
                    parsed_type = CanonicalValueType.from_json(
                        dict(type_value),
                        context=f"j.node_ids.{node_ref}.carry[{ordinal}].type",
                    )
                except (TypeError, ValueError):
                    continue
                carry_type = _ValueType(
                    parsed_type.dtype, len(parsed_type.shape), parsed_type.kind
                )
            else:
                dtype = type_value.get("dtype")
                shape = type_value.get("shape")
                if not isinstance(dtype, str) or not isinstance(shape, list):
                    continue
                carry_type = _ValueType(dtype, len(shape))
            add(
                state_ref,
                _Anchor(
                    "j",
                    "loop_state",
                    carry_type,
                    carrier_class="value",
                ),
                f"j.node_ids.{node_ref}.carry[{ordinal}]",
            )
            for role_name in ("body_argument", "next_value", "result_value"):
                role_ref = carry.get(role_name)
                if not isinstance(role_ref, str) or role_ref in anchors:
                    continue
                add(
                    role_ref,
                    _Anchor(
                        "j",
                        "value",
                        carry_type,
                        carrier_class="value",
                    ),
                    f"j.node_ids.{node_ref}.carry[{ordinal}].{role_name}",
                )

    for item in j.boundary.domains:
        add(item, _Anchor("j", "domain"), f"j.boundary.domains.{item}")
    tensor_types = {
        item.name: value_types.get(item.name)
        or _ValueType(item.dtype, len(item.shape))
        for item in [*j.intent.inputs, *j.intent.outputs]
    }
    for item in j.boundary.accesses:
        value_type = tensor_types.get(item.tensor)
        arity = (value_type,) if item.kind.value in {"store", "update"} else ()
        add(
            item.access_id,
            _Anchor(
                "j", "access", value_type, arity, carrier_class="value"
            ),
            f"j.boundary.accesses.{item.access_id}",
        )
    for item in j.boundary.effects:
        assert item.effect_id is not None
        add(
            item.effect_id,
            _Anchor("j", "effect"),
            f"j.boundary.effects.{item.effect_id}",
        )
    for scope_id in elaboration.scope_by_id:
        add(scope_id, _Anchor("j", "scope"), f"j.scopes.{scope_id}")
    for item in j.storage:
        add(
            item.storage_id,
            _Anchor("j", "storage"),
            f"j.storage.{item.storage_id}",
        )
    for item in j.boundary.realization_authorizations:
        add(
            item.auth_id,
            _Anchor("j", "authorization"),
            f"j.boundary.realization_authorizations.{item.auth_id}",
        )

    for index, slot in enumerate(c.slots):
        add(slot.name, _Anchor("c", "slot"), f"c.slots[{index}]")
    return anchors, node_operations


def _check_scatter_atomic_authorizations(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Bind each J scatter carrying atomic metadata to one authorization.

    Canonical J uses ``scatter`` with atomic attributes; the separate
    ``atomic_scatter`` Intent/MLIR op is an internal lowering form and never
    reaches this checker.  Atomic ordering and scope are Layer-A semantics,
    while the authorization is the explicit J/Boundary acknowledgement that a
    realization may use native atomic update machinery.  The stable
    ``node_ids`` handle is used as the subject so a positional body path cannot
    silently retarget the permission.
    """

    path_to_node = j_elaboration(j).authored_node_by_path(j.node_ids)
    native_atomic_subjects = [
        item.subject_ref
        for item in j.boundary.realization_authorizations
        if item.kind is RefinementKind.NATIVE_ATOMIC
    ]
    for operation_path, operation in _canonical_operation_items(j):
        if operation.get("op") != "scatter":
            continue
        attrs_value = operation.get("attrs")
        attrs = attrs_value if isinstance(attrs_value, Mapping) else {}
        if attrs.get("atomic") is None:
            continue
        node_ref = path_to_node.get(operation_path)
        base = f"j.intent.intent_function.{operation_path}"
        if node_ref is None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REFINEMENT_AUTH_MISSING,
                    f"{base}.id",
                    "atomic scatter requires a stable node_ids handle so its "
                    "native_atomic authorization has an unambiguous subject",
                    operation_path,
                )
            )
            continue
        matches = sum(subject == node_ref for subject in native_atomic_subjects)
        if matches != 1:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REFINEMENT_AUTH_MISSING,
                    "j.boundary.realization_authorizations",
                    "atomic scatter requires exactly one native_atomic "
                    f"authorization for subject_ref {node_ref!r}",
                    node_ref,
                )
            )


def _check_total_j_value_typing(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Require every Layer-A result to have one closed canonical value type."""

    typed_values = j_elaboration(j).value_type_by_name
    for record in j_elaboration(j).operation_records:
        path = record["path"]
        for result in record["results"]:
            if result in typed_values:
                continue
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.J_VALUE_TYPE_INCOMPLETE,
                    f"j.intent.intent_function.{path}.result",
                    "Layer-A result has no uniquely determined closed value type",
                    result,
                )
            )


def _check_output_binding_types(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Require every canonical output destination to preserve full J type."""

    elaboration = j_elaboration(j)
    value_types = elaboration.value_type_by_name
    mapping_positions = {
        item.output: item.mapping_index
        for item in elaboration.output_bindings
        if item.mapping_index is not None
    }
    return_positions = {
        item.output: item.return_index for item in elaboration.output_bindings
    }
    output_positions = {
        item.output: item.abi_index for item in elaboration.output_bindings
    }
    outputs = {
        item.output: item.output_type for item in elaboration.output_bindings
    }
    producible_values = {item.name for item in j.intent.inputs}
    for record in elaboration.operation_records:
        producible_values.update(record["results"])
    try:
        effect_defined_outputs = complete_conditional_effect_outputs(j)
    except (NotImplementedError, ValueError):
        # Full functional projection owns this narrow proof.  If another boundary
        # capability is unsupported, retain the ordinary fail-closed output rule.
        effect_defined_outputs = frozenset()
    for binding in elaboration.output_bindings:
        if binding.value not in producible_values:
            if binding.output in effect_defined_outputs:
                continue
            if binding.output in mapping_positions:
                path = (
                    "j.boundary.output_mappings"
                    f"[{mapping_positions[binding.output]}].value"
                )
            elif binding.output in return_positions:
                path = (
                    "j.intent.intent_function.returns"
                    f"[{return_positions[binding.output]}].value"
                )
            else:
                path = (
                    "j.intent.intent_function.outputs"
                    f"[{output_positions[binding.output]}]"
                )
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.OUTPUT_BINDING_INCOMPLETE,
                    path,
                    "canonical output binding value is neither a J input nor "
                    "an authored operation result",
                    binding.output,
                )
            )
            continue
        expected = outputs[binding.output]
        actual = value_types.get(binding.value)
        if actual is None or actual == expected:
            continue
        if binding.output in mapping_positions:
            path = (
                "j.boundary.output_mappings"
                f"[{mapping_positions[binding.output]}].value"
            )
        elif binding.output in return_positions:
            path = (
                "j.intent.intent_function.returns"
                f"[{return_positions[binding.output]}].value"
            )
        else:
            path = (
                "j.intent.intent_function.outputs"
                f"[{output_positions[binding.output]}]"
            )
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.OUTPUT_BINDING_TYPE_MISMATCH,
                path,
                "canonical output binding type differs from its declared "
                f"output: expected={expected.to_json()}, actual={actual.to_json()}",
                binding.output,
            )
        )


_ITERATION_TILE_ROLES = frozenset(
    {"tile_extent", "reduction_tile_extent"}
)








def _check_effect_conditionals(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check J-owned effect_if predicate, scope, and effect partition."""

    elaboration = j_elaboration(j)
    scopes = elaboration.scope_by_id
    value_types = elaboration.value_type_by_name
    effects = {
        effect.effect_id: effect
        for effect in j.boundary.effects
        if effect.effect_id is not None
    }
    children: dict[str, set[str]] = {scope_id: set() for scope_id in scopes}
    for scope_id, scope in scopes.items():
        parent = scope.get("parent_scope")
        if isinstance(parent, str):
            children.setdefault(parent, set()).add(scope_id)

    def descendants(scope_id: str) -> set[str]:
        seen: set[str] = set()
        pending = [scope_id]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(children.get(current, ()))
        return seen

    guarded_accesses = {
        relation.access_ref for relation in j.boundary.access_guard_relations
    }
    for record in elaboration.operation_records:
        operation = record.get("wire")
        if record.get("op") != "effect_if" or not isinstance(operation, Mapping):
            continue
        base = f"j.intent.intent_function.{record['path']}"
        predicate = operation.get("predicate")
        predicate_type = value_types.get(predicate) if isinstance(predicate, str) else None
        if (
            predicate_type is None
            or not predicate_type.is_tensor
            or predicate_type.dtype is None
            or predicate_type.dtype.casefold() != "i1"
            or predicate_type.shape
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    f"{base}.predicate",
                    "effect_if predicate must be one dominating scalar i1 value",
                    predicate if isinstance(predicate, str) else None,
                )
            )

        owners: dict[str, str] = {}
        for polarity in ("positive", "negative"):
            arm = operation.get(polarity)
            scope_id = arm.get("scope_id") if isinstance(arm, Mapping) else None
            if not isinstance(scope_id, str) or scope_id not in scopes:
                continue
            for descendant in descendants(scope_id):
                for effect_ref in scopes[descendant].get("effects", ()):
                    previous = owners.get(str(effect_ref))
                    if previous is not None and previous != polarity:
                        diagnostics.append(
                            TypeCheckDiagnostic(
                                TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                                f"{base}.{polarity}.scope_id",
                                "one conditional effect cannot belong to both arms",
                                str(effect_ref),
                            )
                        )
                    owners[str(effect_ref)] = polarity
        if not owners:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                    base,
                    "effect_if requires at least one J-owned effect across its arms",
                )
            )
        for effect_ref in owners:
            effect = effects.get(effect_ref)
            if effect is not None and effect.access_ref in guarded_accesses:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.EFFECT_CONDITIONAL_INVALID,
                        "j.boundary.access_guard_relations",
                        "effect_if branch control and access_guard_relation cannot "
                        "both own one access predicate",
                        effect.access_ref,
                    )
                )


_VALUE_IF_PURE_OPS = frozenset(
    {
        "map_binary", "map_unary", "reduce", "iota", "const",
        "contraction", "select", "cast", "transpose", "gather", "scatter",
        "reshape", "record", "record_get",
        *RAGGED_OPS,
        "concat", "pad", "tile", "flip", "triu", "rand", "scan",
        "arg_reduce", "sort", "pool", "upsample", "polar", "conv",
        "nonzero", "masked_select", "masked_scatter",
    }
)

def _check_value_conditionals(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check J-owned value_if purity, arm-local effects, and yield types."""

    elaboration = j_elaboration(j)
    operations = dict(_canonical_operation_items(j))
    value_types = elaboration.value_type_by_name
    scopes = elaboration.scope_by_id
    children: dict[str, set[str]] = {scope_id: set() for scope_id in scopes}
    for scope_id, scope in scopes.items():
        parent = scope.get("parent_scope")
        if isinstance(parent, str):
            children.setdefault(parent, set()).add(scope_id)

    def descendants(scope_id: str) -> set[str]:
        seen: set[str] = set()
        pending = [scope_id]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(children.get(current, ()))
        return seen

    path_to_node = elaboration.authored_node_by_path(j.node_ids)
    node_to_path = {node: path for path, node in path_to_node.items()}
    for path, operation in operations.items():
        if operation.get("op") != "value_if":
            continue
        base = f"j.intent.intent_function.{path}"
        predicate = operation.get("predicate")
        predicate_type = value_types.get(predicate) if isinstance(predicate, str) else None
        if (
            predicate_type is None
            or not predicate_type.is_tensor
            or predicate_type.dtype is None
            or predicate_type.dtype.casefold() != "i1"
            or predicate_type.shape
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                    f"{base}.predicate",
                    "value_if predicate must be one dominating scalar i1 value",
                    predicate if isinstance(predicate, str) else None,
                )
            )

        result = operation.get("result")
        result_type = value_types.get(result) if isinstance(result, str) else None
        for polarity in ("positive", "negative"):
            arm = operation.get(polarity)
            if not isinstance(arm, Mapping):
                continue
            arm_base = f"{base}.{polarity}"
            scope_id = arm.get("scope_id")
            if not isinstance(scope_id, str) or scope_id not in scopes:
                continue
            for descendant in descendants(scope_id):
                if scopes[descendant].get("effects", ()):
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                            f"{arm_base}.scope_id",
                            "value_if arms and descendant scopes must be pure",
                            descendant,
                        )
                    )
            prefix = f"{path}.{polarity}.body"
            arm_paths = {
                candidate for candidate in operations if candidate.startswith(f"{prefix}[")
            }
            for arm_path in arm_paths:
                op_name = operations[arm_path].get("op")
                if op_name not in _VALUE_IF_PURE_OPS:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                            f"j.intent.intent_function.{arm_path}.op",
                            "value_if arms permit only known pure ordinary IntentIR operations",
                            str(op_name) if op_name is not None else None,
                        )
                    )
            for access in j.boundary.accesses:
                access_path = (
                    node_to_path.get(access.node_ref, access.node_ref)
                    if access.node_ref is not None else None
                )
                if access_path in arm_paths:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                            "j.boundary.accesses",
                            "value_if arms must not own logical accesses",
                            access.access_id,
                        )
                    )
            for effect in j.boundary.effects:
                effect_path = (
                    node_to_path.get(effect.node_ref, effect.node_ref)
                    if effect.node_ref is not None else None
                )
                if effect_path in arm_paths:
                    diagnostics.append(
                        TypeCheckDiagnostic(
                            TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                            "j.boundary.effects",
                            "value_if arms must not own effects",
                            effect.effect_id,
                        )
                    )
            yield_value = arm.get("yield_value")
            yield_type = value_types.get(yield_value) if isinstance(yield_value, str) else None
            if result_type is None or yield_type is None or yield_type != result_type:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.VALUE_CONDITIONAL_INVALID,
                        f"{arm_base}.yield_value",
                        "each value_if arm yield must match the authored result dtype and shape",
                        yield_value if isinstance(yield_value, str) else None,
                    )
                )










_CONSUMER_EXTENSION_OPS = frozenset(
    {
        "map_unary",
        "map_binary",
        "select",
        "cast",
        "transpose",
        "pad",
        "gather",
    }
)
_CONSUMER_EXTENSION_FNS = frozenset(
    {"sum", "prod", "max", "min", "any", "all"}
)


def _same_closed_scalar(left: Any, right: Any) -> bool:
    aliases = {
        "ZERO": "zero",
        "NEG_INF": "neg_inf",
        "-inf": "neg_inf",
        "POS_INF": "pos_inf",
        "+inf": "pos_inf",
        "inf": "pos_inf",
    }
    if isinstance(left, str) or isinstance(right, str):
        return aliases.get(str(left), str(left)) == aliases.get(str(right), str(right))
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and float(left) == float(right)
    )


def _consumer_dependency_path(
    j: SemanticArchive,
    *,
    operand: str,
    access_node_ref: str | None,
    access_tensor: str,
) -> tuple[bool, tuple[Mapping[str, Any], ...]]:
    """Find only the J dependency path from one access to one operand.

    A node-bearing access is tied to that exact node.  A node-less ABI load is
    tied to its tensor leaf.  No tensor-name fallback is used for a node-bearing
    access, so a same-tensor sibling access cannot silently satisfy it.
    """

    path_to_ref = {path: ref for ref, path in j.node_ids.items()}
    producers: dict[str, tuple[str | None, Mapping[str, Any]]] = {}
    for path, operation in _canonical_operation_items(j):
        for result in _operation_results(operation.get("result")):
            producers[result] = (path_to_ref.get(path), operation)

    memo: dict[str, tuple[bool, tuple[Mapping[str, Any], ...]]] = {}
    active: set[str] = set()

    def path_score(
        path: tuple[Mapping[str, Any], ...],
    ) -> tuple[int, int, tuple[str, ...]]:
        names = tuple(str(operation.get("op")) for operation in path)
        return (
            sum(name not in _CONSUMER_EXTENSION_OPS for name in names),
            len(path),
            names,
        )

    def visit(value: str) -> tuple[bool, tuple[Mapping[str, Any], ...]]:
        if access_node_ref is None and value == access_tensor:
            return True, ()
        if value in memo:
            return memo[value]
        if value in active:
            return False, ()
        active.add(value)
        producer = producers.get(value)
        if producer is None:
            result = (False, ())
        else:
            node_ref, operation = producer
            if access_node_ref is not None and node_ref == access_node_ref:
                result = (True, (operation,))
            else:
                candidates: list[tuple[Mapping[str, Any], ...]] = []
                for dependency in _operation_dependencies(operation):
                    child_found, child_path = visit(dependency)
                    if child_found:
                        candidates.append((*child_path, operation))
                result = (
                    (True, min(candidates, key=path_score))
                    if candidates
                    else (False, ())
                )
        active.remove(value)
        memo[value] = result
        return result

    return visit(operand)


def _validate_consumer_path(
    path_operations: tuple[Mapping[str, Any], ...],
    *,
    outside_value: Any,
    outside_dtype: str,
) -> str | None:
    for operation in path_operations:
        op_name = operation.get("op")
        if op_name not in _CONSUMER_EXTENSION_OPS:
            return (
                "consumer-domain provenance crosses unsupported J op "
                f"{op_name!r}; a node-bound access is consumer-local: "
                "use node_ref=null for one shared physical load, or author "
                "distinct access/effect rows for distinct Source loads"
            )
        if op_name != "pad":
            continue
        attrs = operation.get("attrs")
        operands = operation.get("operands")
        result_type = _explicit_type(operation)
        if (
            not isinstance(attrs, Mapping)
            or not _PAD_REQUIRED_ATTRS.issubset(attrs)
            or set(attrs) - _PAD_ATTRS
            or not isinstance(operands, list)
            or len(operands) != 1
            or result_type is None
        ):
            return "pad provenance requires closed lo/hi/value and one typed operand"
        lo = attrs.get("lo")
        hi = attrs.get("hi")
        if (
            not isinstance(lo, list)
            or not isinstance(hi, list)
            or len(lo) != result_type.rank
            or len(hi) != result_type.rank
            or any(
                isinstance(item, bool) or not isinstance(item, (int, str))
                for item in [*lo, *hi]
            )
        ):
            return (
                "pad provenance requires closed affine lo/hi for every axis; "
                "their shape and non-negativity are checked by Layer-A TypeCheck"
            )
        if result_type.dtype != outside_dtype:
            return "pad result dtype differs from consumer outside dtype"
        if not _same_closed_scalar(attrs.get("value"), outside_value):
            return "pad value differs from the declared consumer outside value"
    return None


def _check_reduction_boundaries(
    j: SemanticArchive,
    node_operations: Mapping[str, Mapping[str, Any]],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Keep Layer-B algebra contracts attached to semantic consumers.

    ``ReductionBoundary`` describes padding/algebra for an existing Layer-A
    reduce or inclusive scan.  It cannot manufacture a reduction for a loop or
    any other node.  Enforcing that ownership here prevents a physically
    matching Replay from hiding a missing semantic contraction in J.
    """

    seen: set[str] = set()
    for index, reduction in enumerate(j.boundary.reductions):
        path = f"j.boundary.reductions[{index}]"
        operation = node_operations.get(reduction.node_ref)
        if operation is None or operation.get("op") not in {"reduce", "scan"}:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REDUCTION_BOUNDARY_INVALID,
                    f"{path}.node_ref",
                    "reduction boundary must reference an existing semantic "
                    "reduce or inclusive scan node",
                    reduction.node_ref,
                )
            )
        if reduction.node_ref in seen:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REDUCTION_BOUNDARY_INVALID,
                    path,
                    "one semantic reduce/scan may have at most one reduction "
                    "boundary contract",
                    reduction.node_ref,
                )
            )
        seen.add(reduction.node_ref)


def _check_consumer_extensions(
    j: SemanticArchive,
    node_operations: Mapping[str, Mapping[str, Any]],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Check the unique per-consumer outside-domain authority."""

    value_types = j_elaboration(j).value_type_by_name
    accesses = {item.access_id: item for item in j.boundary.accesses}
    declared = {
        (item.consumer_ref, item.operand_index, item.access_ref): item
        for item in j.boundary.consumer_extensions
    }
    required = {
        (item.consumer_ref, item.operand_index, item.access_ref)
        for item in functional_consumer_extension_requirements(j)
    }

    for index, extension in enumerate(j.boundary.consumer_extensions):
        path = f"j.boundary.consumer_extensions[{index}]"
        operation = node_operations.get(extension.consumer_ref)
        attrs = operation.get("attrs") if operation is not None else None
        fn = attrs.get("fn") if isinstance(attrs, Mapping) else None
        operands = operation.get("operands") if operation is not None else None
        if (
            operation is None
            or operation.get("op") not in {"reduce", "scan"}
            or fn not in _CONSUMER_EXTENSION_FNS
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.consumer_ref",
                    "consumer extension requires a supported reduce or scan with "
                    "a closed two-sided identity",
                    extension.consumer_ref,
                )
            )
            continue
        if (
            not isinstance(operands, list)
            or extension.operand_index >= len(operands)
            or not isinstance(operands[extension.operand_index], str)
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.operand_index",
                    "consumer operand ordinal is outside the closed operand list",
                    str(extension.operand_index),
                )
            )
            continue
        operand = str(operands[extension.operand_index])
        axis_issue: str | None = None
        try:
            local_axis = resolve_consumer_local_tile_axis(
                j, extension.consumer_ref
            )
        except (NotImplementedError, ValueError) as exc:
            local_axis = None
            axis_issue = str(exc)
        if axis_issue is not None or extension.tile_axis != local_axis:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.tile_axis",
                    axis_issue
                    or "consumer extension tile axis differs from the consumer's "
                    "closed local axis",
                    str(extension.tile_axis),
                )
            )
        operand_type = value_types.get(operand)
        if (
            operand_type is None
            or not operand_type.is_tensor
            or operand_type.dtype is None
            or extension.outside.dtype.casefold() != operand_type.dtype.casefold()
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.outside.dtype",
                    "outside dtype must equal the selected consumer operand dtype",
                    extension.outside.dtype,
                )
            )
        if not is_consumer_reduction_identity(
            str(fn), extension.outside.dtype, extension.outside.value
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.outside.value",
                    "outside value is not the declared consumer's two-sided identity",
                    extension.consumer_ref,
                )
            )
        access = accesses.get(extension.access_ref)
        if access is None or access.kind.value != "load" or access.predicate is None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.access_ref",
                    "consumer extension requires one predicate-bearing load access",
                    extension.access_ref,
                )
            )
            continue
        found, selected_path = _consumer_dependency_path(
            j,
            operand=operand,
            access_node_ref=access.node_ref,
            access_tensor=access.tensor,
        )
        if not found:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.access_ref",
                    "selected access provenance does not reach this consumer "
                    "operand",
                    extension.access_ref,
                )
            )
            continue
        path_issue = _validate_consumer_path(
            selected_path,
            outside_value=extension.outside.value,
            outside_dtype=extension.outside.dtype,
        )
        if path_issue is not None:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.CONSUMER_EXTENSION_INVALID,
                    f"{path}.access_ref",
                    path_issue,
                    extension.access_ref,
                )
            )

    for consumer_ref, operand_index, access_ref in sorted(required - set(declared)):
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.CONSUMER_EXTENSION_MISSING,
                "j.boundary.consumer_extensions",
                "predicate-bearing reduce/scan operand requires an explicit "
                "outside-domain identity relation",
                f"{consumer_ref}:{operand_index}:{access_ref}",
            )
        )




def _check_signature_mode_partition(
    j: SemanticArchive,
) -> tuple[TypeCheckDiagnostic, ...]:
    """Require the authored ABI mode to match J's logical boundary partition."""

    input_refs = {tensor.name for tensor in j.intent.inputs}
    try:
        output_refs = {
            binding.output
            for binding in j_elaboration(
                canonicalize_semantic_archive(j)
            ).output_bindings
        }
    except (TypeError, ValueError, IndexError, KeyError):
        # This precheck runs before the main canonicalization pass.  A
        # malformed surface archive may therefore only provide a diagnostic
        # declaration view; strict admission remains owned by the later pass.
        output_refs = {tensor.name for tensor in j.intent.outputs}
    diagnostics: list[TypeCheckDiagnostic] = []
    for index, parameter in enumerate(j.signature.parameters):
        ref = parameter.logical_ref
        if ref is None:
            continue
        membership = (ref in input_refs, ref in output_refs)
        expected = {
            ParameterMode.INPUT: (True, False),
            ParameterMode.OUTPUT: (False, True),
            ParameterMode.INOUT: (True, True),
        }.get(parameter.mode)
        if expected is None or membership == expected:
            continue
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.J_ELABORATION_FAILED,
                f"j.signature.parameters[{index}].mode",
                "signature mode must exactly match the logical tensor boundary "
                "partition: input binds input-only, output binds output-only, "
                "and an input/output overlap must be authored as inout",
                ref,
            )
        )
    return tuple(diagnostics)


def _check_fresh_inout_boundary_closure(
    j: SemanticArchive,
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Require every fresh logical inout to author its complete state boundary.

    ``mode=inout`` is not permission for fixed code to infer memory effects from
    the ABI.  The active compact-wire contract therefore requires the Lift owner
    to name a pre-state load, a post-state store, their matching read/write
    effects, and one canonical output binding.  This is a source-free dependent
    check: it rejects an incomplete J but never creates an access, effect, or
    binding.  Pure output parameters remain governed by ordinary return/output
    binding and are intentionally not forced to duplicate functional semantics as
    boundary effects.
    """

    try:
        returns = {
            binding.output
            for binding in j_elaboration(
                canonicalize_semantic_archive(j)
            ).output_bindings
        }
    except (TypeError, ValueError, IndexError, KeyError):
        # This check intentionally runs before canonicalization so it can
        # report an authored inout omission.  The raw return names are only a
        # diagnostic fallback for a surface J that cannot yet elaborate.
        returns = {item.output for item in j.intent.returns}
    for index, parameter in enumerate(j.signature.parameters):
        logical_ref = parameter.logical_ref
        if parameter.mode is not ParameterMode.INOUT or logical_ref is None:
            continue

        loads = {
            access.access_id
            for access in j.boundary.accesses
            if access.tensor == logical_ref and access.kind is AccessKind.LOAD
        }
        stores = {
            access.access_id
            for access in j.boundary.accesses
            if access.tensor == logical_ref and access.kind is AccessKind.STORE
        }
        read_effects = {
            effect.access_ref
            for effect in j.boundary.effects
            if effect.target == logical_ref
            and effect.kind.value == "read"
            and effect.access_ref in loads
        }
        write_effects = {
            effect.access_ref
            for effect in j.boundary.effects
            if effect.target == logical_ref
            and effect.kind.value == "write"
            and effect.access_ref in stores
        }

        missing: list[str] = []
        if not loads:
            missing.append("pre-state load")
        elif not read_effects:
            missing.append("load-bound read effect")
        if not stores:
            missing.append("post-state store")
        elif not write_effects:
            missing.append("store-bound write effect")
        if logical_ref not in returns:
            missing.append("canonical return binding")
        if not missing:
            continue
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.FUNCTIONAL_BOUNDARY_INCOMPLETE,
                f"j.signature.parameters[{index}].logical_ref",
                "fresh logical inout must explicitly close its state boundary; "
                "missing " + ", ".join(missing),
                logical_ref,
            )
        )




def _check_program_axes(
    j: SemanticArchive,
    pi: RealizationWitness,
    diagnostics: list[TypeCheckDiagnostic],
    *,
    required: bool,
) -> None:
    """Check the canonical pi domain-to-launch-axis mapping."""

    try:
        projection = canonical_program_coordinate_projection(pi)
    except ProgramCoordinateProjectionError as exc:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.PROGRAM_AXIS_BINDING_INVALID,
                "pi.program_axes" if pi.program_axes else "pi.program_bindings",
                f"canonical program mapping is invalid: {exc}",
            )
        )
        return

    declared_domains = set(j.boundary.domains)
    bound_domains = {domain_ref for domain_ref, _axis in projection.domain_to_axis}
    for domain_ref in sorted(bound_domains - declared_domains):
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.PROGRAM_AXIS_BINDING_INVALID,
                "pi.program_axes"
                if projection.family == "program_axes"
                else "pi.program_bindings",
                "program mapping must name a declared J boundary domain",
                domain_ref,
            )
        )
    if required and not bound_domains:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.PROGRAM_AXIS_BINDING_INVALID,
                "pi.program_axes",
                "the requested launch mapping has no bound J boundary domain",
            )
        )


def _check_normalization_bindings(
    j: SemanticArchive,
    c: ImplementationConfig,
    pi: RealizationWitness,
    anchors: Mapping[str, _Anchor],
    node_operations: Mapping[str, Mapping[str, Any]],
    diagnostics: list[TypeCheckDiagnostic],
) -> None:
    """Dependently check the active direct J/C-bound pi relations.

    This checker never discovers or completes a relation.  It only rejects rows
    whose authored J/C handles cannot denote the closed relation selected by
    Lift.
    """

    if pi.wire_schema != NORMALIZATION_WITNESS_SCHEMA:
        return

    elaboration = j_elaboration(j)
    boundary_accesses = {
        access.access_id: access for access in j.boundary.accesses
    }
    signature_parameters = {
        parameter.name: parameter for parameter in j.signature.parameters
    }
    tensor_ranks = {
        tensor.name: len(tensor.shape)
        for tensor in [*j.intent.inputs, *j.intent.outputs]
    }
    logical_value_types = elaboration.value_type_by_name

    def zero_lower_bound_coordinates(access_ref: str) -> frozenset[str]:
        # π and Gate1 share this exact J-only dense-domain certificate.  The
        # access predicate may carry only the upper mask; the zero lower bound
        # is owned by its explicit canonical J domain.
        return zero_lower_bound_coordinates_from_j(j, access_ref)

    def reject(
        path: str,
        detail: str,
        ref: str | None = None,
        *,
        code: TypeCheckErrorCode = TypeCheckErrorCode.PI_NORMALIZATION_BINDING_INVALID,
    ) -> None:
        diagnostics.append(
            TypeCheckDiagnostic(
                code,
                path,
                detail,
                ref,
            )
        )

    seen_normalization_classes: set[str] = set()
    for class_index, normalization_class in enumerate(pi.normalization_classes):
        class_path = f"pi.normalization_classes[{class_index}]"
        class_name = normalization_class.get("name")
        class_scope = normalization_class.get("scope")
        expected_scope = NORMALIZATION_CLASS_SCOPES.get(str(class_name))
        if expected_scope is None:
            reject(
                f"{class_path}.name",
                "normalization class is not in the closed Gate1 class set",
                str(class_name),
            )
            continue
        if class_name in seen_normalization_classes:
            reject(
                f"{class_path}.name",
                "one normalization class may be declared only once",
                str(class_name),
            )
        seen_normalization_classes.add(str(class_name))
        if class_scope != expected_scope:
            reject(
                f"{class_path}.scope",
                f"normalization class {class_name!r} requires scope "
                f"{expected_scope!r}",
                str(class_scope),
            )
            continue
        if class_name == "default_zero_lower_bound":
            predicated_accesses = {
                access.access_id
                for access in j.boundary.accesses
                if access.predicate is not None
            }
            if not predicated_accesses:
                reject(
                    f"{class_path}.scope",
                    "default_zero_lower_bound has no predicated J access to cover",
                )
                continue
            uncovered = sorted(
                access_ref
                for access_ref in predicated_accesses
                if not zero_lower_bound_coordinates(access_ref)
            )
            if uncovered:
                reject(
                    f"{class_path}.name",
                    "default_zero_lower_bound must cover every predicated J "
                    "access with a J-proven zero lower bound and finite upper "
                    f"bound; uncovered accesses: {', '.join(uncovered)}",
                    str(class_name),
                )
        elif class_name == "closed_consumer_identity":
            extensions = closed_consumer_identity_extensions(j, pi)
            expected = len(j.boundary.consumer_extensions)
            if not expected:
                reject(
                    f"{class_path}.scope",
                    "closed_consumer_identity has no consumer extension to cover",
                    str(class_name),
                )
            elif len(extensions) != expected:
                reject(
                    f"{class_path}.name",
                    "closed_consumer_identity must cover every consumer extension "
                    "with a fixed J-proven producer mapping",
                    str(class_name),
                )
        elif class_name == PER_PROGRAM_SCALAR_KEEPDIMS_CLASS:
            reductions = per_program_scalar_keepdims_reductions(j, pi)
            if not reductions:
                reject(
                    f"{class_path}.name",
                    "per_program_scalar_keepdims requires every keepdims reduce "
                    "to be the J-proven rank-2 rowwise form with one directly "
                    "mapped retained program domain",
                    str(class_name),
                )

    def check_axis(subject_ref: str, axis_ref: int | str, path: str) -> bool:
        anchor = anchors.get(subject_ref)
        if anchor is None or anchor.owner != "j" or anchor.kind not in {"domain", "node"}:
            reject(path, "pi logical axis must name a declared J domain or node", subject_ref)
            return False
        allowed = _axis_refs_v1(j, subject_ref, node_operations)
        if axis_ref not in allowed:
            reject(
                path,
                "pi axis_ref is outside AxisRefs_v1(J,subject_ref); "
                f"allowed axis_ref values are {sorted(allowed, key=lambda item: (str(type(item)), str(item)))}",
                str(axis_ref),
            )
            return False
        return True

    claimed_program_axes: set[tuple[str, int | str]] = set()
    claimed_physical_axes: set[int] = set()
    physical_axis_by_domain: dict[str, int] = {}
    for binding_index, binding in enumerate(pi.program_bindings):
        relation = str(binding["relation"])
        axes = binding["logical_axes"]
        physical_axis = binding.get("axis")
        binding_path = f"pi.program_bindings[{binding_index}]"
        if (
            isinstance(physical_axis, bool)
            or not isinstance(physical_axis, int)
            or not 0 <= physical_axis <= 2
        ):
            reject(
                f"{binding_path}.axis",
                "program binding requires one physical launch axis in [0, 2]",
                str(physical_axis),
            )
            continue
        if physical_axis in claimed_physical_axes:
            reject(
                f"{binding_path}.axis",
                "one program binding is permitted for each physical launch axis",
                str(physical_axis),
            )
        claimed_physical_axes.add(physical_axis)
        if relation == "direct" and len(axes) != 1:
            reject(
                f"pi.program_bindings[{binding_index}].logical_axes",
                "direct program binding requires exactly one logical axis",
            )
        if relation in {"product_linearized", "guarded_union"} and len(axes) < 2:
            reject(
                f"pi.program_bindings[{binding_index}].logical_axes",
                f"{relation} requires at least two logical axes",
            )
        for axis_index, axis in enumerate(axes):
            domain_ref = str(axis["domain_ref"])
            axis_ref = axis["axis_ref"]
            path = (
                f"pi.program_bindings[{binding_index}].logical_axes"
                f"[{axis_index}]"
            )
            anchor = anchors.get(domain_ref)
            if anchor is None or anchor.owner != "j" or anchor.kind != "domain":
                reject(path, "program binding must name a declared J domain", domain_ref)
                continue
            check_axis(domain_ref, axis_ref, f"{path}.axis_ref")
            key = (domain_ref, axis_ref)
            if key in claimed_program_axes:
                reject(path, "one logical J axis cannot have two program bindings", domain_ref)
            claimed_program_axes.add(key)
            previous_physical_axis = physical_axis_by_domain.setdefault(
                domain_ref, physical_axis
            )
            if previous_physical_axis != physical_axis:
                reject(
                    path,
                    "one J program domain cannot map to multiple physical launch axes",
                    domain_ref,
                )

    slots = {slot.name: slot for slot in c.slots}
    iteration_tile_slot_refs: set[str] = set()
    for binding_index, binding in enumerate(pi.iteration_bindings):
        path = f"pi.iteration_bindings[{binding_index}]"
        relation = str(binding["relation"])
        axes = binding["logical_axes"]
        valid_axis_keys: set[tuple[str, int | str]] = set()
        for axis_index, axis in enumerate(axes):
            subject_ref = str(axis["subject_ref"])
            axis_ref = axis["axis_ref"]
            axis_path = f"{path}.logical_axes[{axis_index}]"
            if check_axis(subject_ref, axis_ref, f"{axis_path}.axis_ref"):
                valid_axis_keys.add((subject_ref, axis_ref))
                logical_axis_key = (subject_ref, axis_ref)
                if logical_axis_key in claimed_program_axes:
                    reject(
                        axis_path,
                        "one logical J axis cannot be both a direct program "
                        "binding and a tiled iteration binding",
                        f"{subject_ref}:{axis_ref}",
                    )
            anchor = anchors.get(subject_ref)
            if relation == "tiled_domain" and (
                anchor is None or anchor.kind != "domain"
            ):
                reject(
                    f"{axis_path}.subject_ref",
                    "tiled_domain may bind only J boundary domains",
                    subject_ref,
                )
            if relation == "tiled_recurrence":
                operation = node_operations.get(subject_ref)
                if anchor is None or anchor.kind != "node" or operation is None or operation.get("op") != "loop":
                    reject(
                        f"{axis_path}.subject_ref",
                        "tiled_recurrence may bind only an authored J loop node",
                        subject_ref,
                    )

        tile_slot_ref = str(binding["tile_slot"])
        tile_slot = slots.get(tile_slot_ref)
        if tile_slot is None:
            reject(f"{path}.tile_slot", "iteration tile_slot is not declared by C", tile_slot_ref)
        else:
            if tile_slot.role not in _ITERATION_TILE_ROLES:
                reject(
                    f"{path}.tile_slot",
                    "iteration tile_slot must have a closed tile_extent role",
                    tile_slot_ref,
                )
            if tile_slot.kind is not ScheduleSlotKind.CONSTEXPR:
                reject(
                    f"{path}.tile_slot",
                    "iteration tile_slot must reference a constexpr C slot",
                    tile_slot_ref,
                )
            if tile_slot_ref in iteration_tile_slot_refs:
                reject(
                    f"{path}.tile_slot",
                    "one C tile slot cannot be reused by multiple iteration bindings",
                    tile_slot_ref,
                )
            iteration_tile_slot_refs.add(tile_slot_ref)
        mode_slot_ref = binding["mode_slot"]
        if mode_slot_ref is not None and str(mode_slot_ref) not in slots:
            reject(
                f"{path}.mode_slot",
                "iteration mode_slot is not declared by C",
                str(mode_slot_ref),
            )

    for index, slot in enumerate(c.slots):
        if slot.role not in _ITERATION_TILE_ROLES:
            continue
        path = f"c.slots[{index}]"
        if slot.kind is not ScheduleSlotKind.CONSTEXPR:
            reject(
                f"{path}.kind",
                "a C tile role must be declared on a constexpr slot",
                slot.name,
            )
        if slot.name not in iteration_tile_slot_refs:
            reject(
                path,
                "a C tile role must be consumed by pi.iteration_bindings",
                slot.name,
            )

    for index, binding in enumerate(pi.control_bindings):
        subject_ref = str(binding["subject_ref"])
        relation = str(binding["relation"])
        anchor = anchors.get(subject_ref)
        allowed_kinds = (
            {"domain"}
            if relation == "terminal_domain_guard"
            else {"node", "effect"}
        )
        if anchor is None or anchor.owner != "j" or anchor.kind not in allowed_kinds:
            reject(
                f"pi.control_bindings[{index}].subject_ref",
                f"{relation} requires a J-owned {sorted(allowed_kinds)} anchor",
                subject_ref,
            )

    for index, binding in enumerate(pi.materialization_bindings):
        value_ref = str(binding["value_ref"])
        anchor = anchors.get(value_ref)
        if anchor is None or anchor.owner != "j" or anchor.kind not in {
            "tensor", "value", "loop_state", "node"
        }:
            reject(
                f"pi.materialization_bindings[{index}].value_ref",
                "materialization binding must name a J value-producing anchor",
                value_ref,
            )

    seen_access_axes: dict[tuple[str, int], str] = {}
    seen_access_carriers: set[tuple[str, str, int]] = set()
    for index, binding in enumerate(pi.access_representation_bindings):
        access_ref = str(binding["access_ref"])
        relation = str(binding["relation"])
        anchor = anchors.get(access_ref)
        if anchor is None or anchor.owner != "j" or anchor.kind != "access":
            reject(
                f"pi.access_representation_bindings[{index}].access_ref",
                "access representation binding must name a declared J access",
                access_ref,
            )
            continue
        access = boundary_accesses.get(access_ref)
        if access is None:
            # The anchor table and Boundary must agree, but retain an explicit
            # fail-closed diagnostic if a malformed caller bypasses construction.
            reject(
                f"pi.access_representation_bindings[{index}].access_ref",
                "access representation binding lacks its J boundary declaration",
                access_ref,
            )
            continue
        if relation == "guard_lower_bound_elided" and not zero_lower_bound_coordinates(
            access_ref
        ):
            reject(
                f"pi.access_representation_bindings[{index}].access_ref",
                "guard_lower_bound_elided requires an exact access-local J "
                "domain-coordinate lower bound of zero",
                access_ref,
            )
        if relation == "boolean_storage_byte":
            value_type = logical_value_types.get(access.tensor)
            if (
                value_type is None
                or not value_type.is_tensor
                or value_type.dtype is None
                or value_type.dtype.casefold() not in {"bool", "i1"}
            ):
                reject(
                    f"pi.access_representation_bindings[{index}].access_ref",
                    "boolean_storage_byte requires a logical boolean J tensor",
                    access_ref,
                )
        carrier_ref = binding.get("carrier_ref")
        axis_ref = binding.get("axis_ref")
        if relation in {"address_recomputed", "pointer_advanced"}:
            if not isinstance(carrier_ref, str) or not re.fullmatch(
                r"p[0-9]+", carrier_ref
            ):
                reject(
                    f"pi.access_representation_bindings[{index}].carrier_ref",
                    "address representation requires one p* ABI carrier",
                    str(carrier_ref),
                )
            else:
                parameter = signature_parameters.get(carrier_ref)
                if (
                    parameter is None
                    or parameter.kind is not ParameterKind.SCALAR
                    or parameter.mode is not ParameterMode.VALUE
                    or parameter.logical_ref is not None
                ):
                    observed_binding = (
                        "missing"
                        if parameter is None
                        else (
                            f"kind={parameter.kind.value!r}, "
                            f"mode={parameter.mode.value!r}, "
                            f"logical_ref={parameter.logical_ref!r}"
                        )
                    )
                    reject(
                        f"pi.access_representation_bindings[{index}].carrier_ref",
                        "address representation carrier must be a null-logical-ref "
                        f"scalar parameter (actual {observed_binding})",
                        carrier_ref,
                    )
            if (
                isinstance(axis_ref, bool)
                or not isinstance(axis_ref, int)
                or axis_ref < 0
            ):
                reject(
                    f"pi.access_representation_bindings[{index}].axis_ref",
                    "address representation axis_ref must be a non-negative integer",
                    str(axis_ref),
                )
            elif axis_ref >= tensor_ranks.get(access.tensor, 0):
                reject(
                    f"pi.access_representation_bindings[{index}].axis_ref",
                    "address representation axis_ref exceeds the access tensor rank",
                    str(axis_ref),
                )
            if isinstance(carrier_ref, str) and isinstance(axis_ref, int) and not isinstance(axis_ref, bool):
                access_axis = (access_ref, axis_ref)
                previous_carrier = seen_access_axes.get(access_axis)
                if (
                    previous_carrier is not None
                    and previous_carrier != carrier_ref
                ):
                    reject(
                        f"pi.access_representation_bindings[{index}]",
                        "one J access axis cannot have conflicting representation "
                        "carriers; preserve one Source-authored carrier or split "
                        "the access occurrence before binding it",
                        access_ref,
                        code=TypeCheckErrorCode.ACCESS_REPRESENTATION_INVALID,
                    )
                else:
                    seen_access_axes[access_axis] = carrier_ref
                key = (access_ref, carrier_ref, axis_ref)
                if key in seen_access_carriers:
                    reject(
                        f"pi.access_representation_bindings[{index}]",
                        "one address carrier/axis occurrence may be bound only once",
                        access_ref,
                    )
                seen_access_carriers.add(key)
        elif carrier_ref is not None or axis_ref is not None:
            reject(
                f"pi.access_representation_bindings[{index}]",
                "carrier_ref/axis_ref are only valid for address relations",
                access_ref,
            )



def type_check(
    j: SemanticArchive,
    c: ImplementationConfig,
    pi: RealizationWitness,
    *,
    require_program_axes: bool = False,
) -> TypeCheckReport:
    """Mechanically check ``Gamma_(J,C) |- pi : RealizationWitness(J,C)``."""

    if not isinstance(j, SemanticArchive):
        raise TypeError("type_check requires SemanticArchive J")
    if not isinstance(c, ImplementationConfig):
        raise TypeError("type_check requires canonical ImplementationConfig C")
    if not isinstance(pi, RealizationWitness):
        raise TypeError("type_check requires parsed RealizationWitness pi")

    # This contract is about explicit authorship on the submitted wire.  The
    # canonicalizer intentionally removes redundant identity output mappings,
    # so inspect the raw authored J before canonicalization rather than
    # accidentally rejecting the same mapping after it has been normalized.
    authored_boundary_diagnostics: list[TypeCheckDiagnostic] = []
    if pi.wire_schema == NORMALIZATION_WITNESS_SCHEMA:
        _check_fresh_inout_boundary_closure(j, authored_boundary_diagnostics)

    signature_diagnostics = _check_signature_mode_partition(j)
    if signature_diagnostics:
        return TypeCheckReport(TypeCheckStatus.FAIL, signature_diagnostics)

    known_op_diagnostics = _check_known_operation_contracts(j)
    if known_op_diagnostics:
        return TypeCheckReport(TypeCheckStatus.FAIL, known_op_diagnostics)

    try:
        j = canonicalize_semantic_archive(j)
    except ValueError as exc:
        return TypeCheckReport(
            TypeCheckStatus.FAIL,
            (
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.J_ELABORATION_FAILED,
                    "j",
                    str(exc),
                ),
            ),
        )

    # The first pass above rejects malformed surface attrs without asking the
    # elaborator to interpret them.  Re-run on accepted canonical J so axis
    # bounds can use its closed operand ranks.
    known_op_diagnostics = _check_known_operation_contracts(j)
    if known_op_diagnostics:
        return TypeCheckReport(TypeCheckStatus.FAIL, known_op_diagnostics)

    # ContractIR is the target-neutral, one-to-one typed projection of every
    # authored Layer-B domain, index, and predicate.  Failure here is a J type
    # error, not a RefEval/backend capability decision.  In particular, an x*
    # dimension already has integer sort and cannot also stand alone as a boolean
    # domain predicate.  No Source, C value, pi topology, or target is consulted.
    try:
        project_contract_ir(j)
    except ContractIRProjectionError as exc:
        return TypeCheckReport(
            TypeCheckStatus.FAIL,
            (
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.J_ELABORATION_FAILED,
                    "j.boundary",
                    str(exc),
                ),
            ),
        )

    diagnostics: list[TypeCheckDiagnostic] = list(authored_boundary_diagnostics)
    _check_dimension_and_pad_contracts(j, diagnostics)
    _check_value_dimension_relations(j, diagnostics)
    _check_dimension_relations(j, diagnostics)
    _check_total_j_value_typing(j, diagnostics)
    _check_output_binding_types(j, diagnostics)
    _check_rank_zero_logical_input_liveness(j, diagnostics)
    _check_boundary_value_dominance(j, diagnostics)
    _check_access_guard_relations(j, diagnostics)
    _check_effect_conditionals(j, diagnostics)
    _check_value_conditionals(j, diagnostics)
    diagnostics.extend(
        TypeCheckDiagnostic(
            TypeCheckErrorCode.FUNCTIONAL_BOUNDARY_INCOMPLETE,
            issue.path,
            issue.detail,
            issue.ref,
        )
        for issue in functional_boundary_closure_issues(j)
    )
    diagnostics.extend(
        TypeCheckDiagnostic(
            TypeCheckErrorCode.PADDED_MEAN_COUNT_UNPROVEN,
            issue.path,
            issue.detail,
            issue.ref,
        )
        for issue in functional_reduction_padding_issues(j)
    )
    relation_guarded_accesses = {
        item.access_ref for item in j.boundary.access_guard_relations
    }
    for index, access in enumerate(j.boundary.accesses):
        if access.fill_value is None:
            continue
        path = f"j.boundary.accesses[{index}].fill_value"
        if access.kind.value != "load":
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.INVALID_ACCESS_FILL,
                    path,
                    "fill_value is legal only on a load access",
                    access.access_id,
                )
            )
        if (
            access.predicate is None
            and access.access_id not in relation_guarded_accesses
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.INVALID_ACCESS_FILL,
                    path,
                    "fill_value requires an authored access predicate or "
                    "access_guard_relation",
                    access.access_id,
                )
            )

    expected_c = list(
        CanonicalHandleAllocator.canonical_names("c", len(c.slots))
    )
    if [item.name for item in c.slots] != expected_c:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.INVALID_C_SLOT,
                "c.slots",
                "C slot ids must be declaration-order handles c0, c1, ...",
            )
        )
    callable_names = {parameter.name for parameter in j.signature.parameters}
    callable_names.update(
        slot.name for slot in c.slots if slot.kind is ScheduleSlotKind.CONSTEXPR
    )
    compiler_roles: set[str] = set()
    for index, slot in enumerate(c.slots):
        has_value = slot.name in c.values
        value = c.values[slot.name] if has_value else None
        if slot.required and not has_value:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.C_REQUIRED_VALUE_MISSING,
                    f"c.values.{slot.name}",
                    "required C slot has no Harness valuation",
                    slot.name,
                )
            )
        if has_value and not implementation_value_matches(slot.value_type, value):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.C_VALUE_TYPE_MISMATCH,
                    f"c.values.{slot.name}",
                    f"value does not match closed C type {slot.value_type!r}",
                    slot.name,
                )
            )
        if slot.kind not in {ScheduleSlotKind.CONSTEXPR, ScheduleSlotKind.COMPILER}:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.INVALID_C_SLOT,
                    f"c.slots[{index}]",
                    "unsupported C binding kind",
                    slot.name,
                )
            )
        if slot.kind is ScheduleSlotKind.COMPILER:
            if slot.role in callable_names:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.INVALID_C_SLOT,
                        f"c.slots[{index}].role",
                        "compiler C role collides with a callable or constexpr handle",
                        slot.role,
                    )
                )
            if slot.role in compiler_roles:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.INVALID_C_SLOT,
                        f"c.slots[{index}].role",
                        "compiler C roles must be unique",
                        slot.role,
                    )
                )
            compiler_roles.add(slot.role)

    anchors, node_operations = _build_anchors(j, c, diagnostics)
    _check_scatter_atomic_authorizations(j, diagnostics)
    _check_reduction_boundaries(j, node_operations, diagnostics)
    _check_consumer_extensions(j, node_operations, diagnostics)
    absent_signature_refs = {
        parameter.name
        for parameter in j.signature.parameters
        if parameter.kind is ParameterKind.OPAQUE and parameter.dtype == "none"
    }
    _check_result_relations(j, diagnostics)
    try:
        validate_normalization_bindings_against_j(j, c, pi)
    except RealizationWitnessError as exc:
        diagnostics.append(
            TypeCheckDiagnostic(
                TypeCheckErrorCode.PI_NORMALIZATION_BINDING_INVALID,
                "pi",
                str(exc),
            )
        )
    _check_normalization_bindings(
        j,
        c,
        pi,
        anchors,
        node_operations,
        diagnostics,
    )
    _check_program_axes(
        j,
        pi,
        diagnostics,
        required=require_program_axes,
    )
    valid_subjects = {
        ref
        for ref, anchor in anchors.items()
        if anchor.owner == "j" and anchor.kind != "authorization"
    }
    for index, authorization in enumerate(
        j.boundary.realization_authorizations
    ):
        if authorization.subject_ref in absent_signature_refs:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.ABSENT_CARRIER_SEMANTIC_USE_FORBIDDEN,
                    f"j.boundary.realization_authorizations[{index}].subject_ref",
                    "opaque(dtype=none) is an absent ABI carrier and cannot "
                    "authorize a semantic realization refinement",
                    authorization.subject_ref,
                )
            )
            continue
        subject = anchors.get(authorization.subject_ref)
        if authorization.subject_ref not in valid_subjects:
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.AUTHORIZATION_SUBJECT_UNKNOWN,
                    f"j.boundary.realization_authorizations[{index}].subject_ref",
                    "authorization subject is not a J carrier",
                    authorization.subject_ref,
                )
            )
            continue
        if authorization.kind is RefinementKind.NUMERIC_CAST:
            operation = node_operations.get(authorization.subject_ref)
            valid_cast = (
                subject is not None
                and subject.kind == "node"
                and operation is not None
                and operation.get("op") == "cast"
                and len(subject.operand_types) == 1
                and subject.operand_types[0] is not None
                and subject.value_type is not None
                and authorization.source_dtype is not None
                and authorization.target_dtype is not None
            )
            if not valid_cast:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.REFINEMENT_AUTH_TYPE_MISMATCH,
                        f"j.boundary.realization_authorizations[{index}]",
                        "numeric_cast authorization requires one typed cast "
                        "operand/result and explicit source/target dtype",
                        authorization.auth_id,
                    )
                )
                continue
            assert subject is not None
            assert subject.operand_types[0] is not None
            assert subject.value_type is not None
            if (
                authorization.source_dtype != subject.operand_types[0].dtype
                or authorization.target_dtype != subject.value_type.dtype
            ):
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.REFINEMENT_AUTH_TYPE_MISMATCH,
                        f"j.boundary.realization_authorizations[{index}]",
                        "numeric_cast dtype must match cast operand/result types",
                        authorization.auth_id,
                    )
                )
            continue
        if authorization.kind in {
            RefinementKind.SIGNED_WIDEN,
            RefinementKind.UNSIGNED_WIDEN,
        }:
            source_info = _closed_integer_dtype_info(authorization.source_dtype)
            target_info = _closed_integer_dtype_info(authorization.target_dtype)
            expected_signedness = (
                "i"
                if authorization.kind is RefinementKind.SIGNED_WIDEN
                else "u"
            )
            valid_widen = (
                source_info is not None
                and target_info is not None
                and source_info[0] == expected_signedness
                and target_info[0] == expected_signedness
                and target_info[1] > source_info[1]
            )
            if not valid_widen:
                diagnostics.append(
                    TypeCheckDiagnostic(
                        TypeCheckErrorCode.REFINEMENT_AUTH_INVALID_WIDEN,
                        f"j.boundary.realization_authorizations[{index}]",
                        f"{authorization.kind.value} requires explicit "
                        "same-signedness integer source/target dtype with "
                        "strictly increasing width",
                        authorization.auth_id,
                    )
                )
        if (
            authorization.source_dtype is not None
            and subject is not None
            and subject.value_type is not None
            and authorization.source_dtype != subject.value_type.dtype
        ):
            diagnostics.append(
                TypeCheckDiagnostic(
                    TypeCheckErrorCode.REFINEMENT_AUTH_TYPE_MISMATCH,
                    f"j.boundary.realization_authorizations[{index}].source_dtype",
                    (
                        "authorization source dtype differs from its J carrier: "
                        f"subject_ref={authorization.subject_ref!r}, "
                        f"expected={subject.value_type.dtype!r}, "
                        f"actual={authorization.source_dtype!r}"
                    ),
                    authorization.auth_id,
                )
            )

    ordered = tuple(
        sorted(diagnostics, key=lambda item: (item.path, item.code.value, item.detail))
    )
    return TypeCheckReport(
        TypeCheckStatus.PASS if not ordered else TypeCheckStatus.FAIL,
        ordered,
    )


__all__ = [
    "TYPECHECK_PROTOCOL",
    "TypeCheckDiagnostic",
    "TypeCheckErrorCode",
    "TypeCheckReport",
    "TypeCheckStatus",
    "type_check",
]
