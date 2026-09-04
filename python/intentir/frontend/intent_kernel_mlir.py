"""Fail-closed projection from canonical J to Intent Kernel MLIR.

This module is deliberately separate from :mod:`intentir.frontend.emit_mlir`.
The latter materializes the historical tensor/linalg-oriented IntentIR graph;
this exporter emits a TianchenIR-owned, target-neutral Kernel IR exchange
artifact.  Independent lowering consumers may read that artifact through a
file or command-line boundary, but they are not implementation dependencies of
this module.

The initial representability profiles are intentionally small and both require
an empty ``E_C`` interface.  The dense profile accepts a single, elementwise
``map_binary(add|mul)`` over equal-rank views with a direct load/load/store
boundary.  The second profile accepts one local ``ragged`` descriptor followed
by one or two direct ``ragged_outer``/``ragged_member`` query results.  Static
axes form the outer logical ``intent.parallel`` region in the dense profile;
runtime-bounded axes use an ordinary scalar inner loop as a deterministic
J-only placeholder.  Ragged descriptors exist only as tuple SSA values inside
the function and never cross the function or target ABI boundary.

Reordering axes is sound only for the validated independent pointwise profile.
A later target binding may replace that placeholder through a validated
Layer-C plugin, but this projection never guesses a tile or Layer-C value.  No
target plan, layout, launch geometry, source spelling, pi witness, or Layer-C
value is emitted.  The module carries a machine-readable J/E_C digest and the
J-authored runtime shape relation needed by an independent lower consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

from intentir.ir.canonical_j import (
    ContractExpr,
    ContractSort,
    elaborate_j,
    j_elaboration,
    parse_contract_expr,
)
from intentir.ir.contract_ir import (
    CONTRACT_IR_PROTOCOL,
    CanonicalContractIR,
    project_contract_ir,
)
from intentir.ir.layer_c import (
    LayerCInterface,
    binding_interface_digest,
    layer_c_content_digest,
    validate_layer_c_attachments,
)
from intentir.ir.dimension_relation import declared_dimension_handles
from intentir.ir.intent_op_contract import RAGGED_OPS, is_ragged_carrier_dtype
from intentir.ir.model import walk_intent_ops
from intentir.verify.gate1.model import (
    AccessKind,
    EffectKind as JEffectKind,
    ParameterKind as JParameterKind,
    ParameterMode,
    ResultRelationKind,
    SemanticArchive,
)
from intentir.ir.value_dimension import validate_value_dimension_relations


class IntentKernelCapabilityError(NotImplementedError):
    """Canonical J is valid, but this Kernel-IR profile cannot preserve it."""


@dataclass(frozen=True)
class KernelABIShapeRelation:
    """One explicit J-owned scalar/dimension/view ABI relation."""

    parameter: str
    dimension: str
    view_parameter: str
    axis: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "scalar_equals_view_dimension",
            "parameter": self.parameter,
            "dimension": self.dimension,
            "view_parameter": self.view_parameter,
            "axis": self.axis,
        }


def _handoff_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _handoff_constraint_keys(
    values: Any,
    *,
    context: str,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{context} must be an array")
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise ValueError(f"{context}[{index}] must be a string")
        try:
            expression = parse_contract_expr(
                value,
                expected=ContractSort.BOOLEAN,
                context=f"{context}[{index}]",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        result.append(_handoff_json(expression.to_json()))
    return tuple(result)


def _validate_kernel_ir_handoff(handoff: Any) -> None:
    """Ensure all serialized handoff copies describe one contract.

    The handoff has convenient top-level relation attributes as well as the
    complete ``CanonicalContractIR`` payload.  Comparing them at construction
    time prevents a producer from updating one copy while leaving another
    stale.  This validator never adds a field or infers a missing relation.
    """

    contract_ir = handoff.contract_ir
    if not isinstance(contract_ir, Mapping):
        raise ValueError("Kernel IR handoff requires a complete contract IR object")
    if contract_ir.get("protocol") != CONTRACT_IR_PROTOCOL:
        raise ValueError("Kernel IR handoff contract IR protocol is unsupported")
    contract = contract_ir.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Kernel IR handoff contract IR lacks its contract object")

    expected_digest = hashlib.sha256(
        _handoff_json(dict(contract_ir)).encode("utf-8")
    ).hexdigest()
    if handoff.contract_ir_digest != expected_digest:
        raise ValueError(
            "Kernel IR handoff contract_ir_digest does not match contract IR"
        )

    def compare_json_sequences(
        name: str,
        actual: Any,
        expected: Any,
    ) -> None:
        if not isinstance(actual, (list, tuple)) or not isinstance(expected, list):
            raise ValueError(f"Kernel IR handoff {name} relation copy is not an array")
        try:
            actual_json = _handoff_json([dict(item) for item in actual])
            expected_json = _handoff_json(expected)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Kernel IR handoff {name} relation copy is malformed"
            ) from exc
        if actual_json != expected_json:
            raise ValueError(
                f"Kernel IR handoff {name} differs from contract IR relation copy"
            )

    compare_json_sequences(
        "dimension_relations",
        handoff.dimension_relations,
        contract.get("dimension_relations", []),
    )
    compare_json_sequences(
        "value_dimension_relations",
        handoff.value_dimension_relations,
        contract.get("value_dimension_relations", []),
    )
    boundary_relations = contract.get("boundary_relations")
    if not isinstance(boundary_relations, Mapping):
        raise ValueError("Kernel IR handoff contract IR lacks boundary relations")
    compare_json_sequences(
        "result_relations",
        handoff.result_relations,
        boundary_relations.get("result_relations", []),
    )

    nested_constraints = contract.get("dimension_constraints", [])
    top_constraints = _handoff_constraint_keys(
        handoff.dimension_constraints,
        context="Kernel IR handoff dimension_constraints",
    )
    contract_constraints = _handoff_constraint_keys(
        nested_constraints,
        context="Kernel IR contract dimension_constraints",
    )
    if top_constraints != contract_constraints:
        raise ValueError(
            "Kernel IR handoff dimension_constraints differ from contract IR"
        )
    programs = handoff.dimension_constraint_programs
    if not isinstance(programs, (list, tuple)) or len(programs) != len(top_constraints):
        raise ValueError(
            "Kernel IR handoff dimension_constraint_programs must cover every constraint"
        )
    for index, (raw, program) in enumerate(
        zip(handoff.dimension_constraints, programs, strict=True)
    ):
        if not isinstance(program, Mapping):
            raise ValueError(
                f"Kernel IR handoff dimension_constraint_programs[{index}] must be an object"
            )
        expression = parse_contract_expr(
            raw,
            expected=ContractSort.BOOLEAN,
            context=f"Kernel IR handoff dimension_constraints[{index}]",
        )
        if program.get("expression") != expression.to_json():
            raise ValueError(
                "Kernel IR handoff typed dimension constraint expression differs "
                "from its text relation"
            )
        if program.get("result_sort") != ContractSort.BOOLEAN.value:
            raise ValueError(
                "Kernel IR handoff dimension constraint result sort is not boolean"
            )
        reference_sorts = program.get("reference_sorts")
        if not isinstance(reference_sorts, Mapping):
            raise ValueError(
                "Kernel IR handoff dimension constraint reference_sorts must be an object"
            )
        if set(reference_sorts) != set(expression.references):
            raise ValueError(
                "Kernel IR handoff dimension constraint reference sorts are incomplete"
            )
        if any(
            value not in {"integer", "boolean"}
            for value in reference_sorts.values()
        ):
            raise ValueError(
                "Kernel IR handoff dimension constraint reference sort is unsupported"
            )

    signature = contract.get("signature")
    if not isinstance(signature, Mapping) or not isinstance(
        signature.get("parameters"), list
    ):
        raise ValueError("Kernel IR handoff contract IR lacks J signature parameters")
    signature_parameters = signature["parameters"]
    runtime_parameters = handoff.runtime_abi_parameters
    if not isinstance(runtime_parameters, (list, tuple)):
        raise ValueError("Kernel IR handoff runtime_abi_parameters must be an array")
    if len(runtime_parameters) != len(signature_parameters):
        raise ValueError(
            "Kernel IR handoff runtime ABI does not cover the J signature"
        )
    for index, (actual, expected) in enumerate(
        zip(runtime_parameters, signature_parameters, strict=True)
    ):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise ValueError(
                f"Kernel IR handoff runtime ABI parameter {index} is malformed"
            )
        for field in ("name", "kind", "mode", "dtype", "rank", "logical_ref"):
            if actual.get(field) != expected.get(field):
                raise ValueError(
                    f"Kernel IR handoff runtime ABI parameter {index}.{field} "
                    "differs from J.signature"
                )

    ownership = handoff.parameter_ownership
    if not isinstance(ownership, (list, tuple)):
        raise ValueError("Kernel IR handoff parameter_ownership must be an array")
    expected_parameter_names = [item.get("name") for item in signature_parameters]
    actual_ownership_names = [
        item.get("parameter")
        for item in ownership
        if isinstance(item, Mapping)
    ]
    if (
        len(ownership) != len(signature_parameters)
        or actual_ownership_names != expected_parameter_names
    ):
        raise ValueError(
            "Kernel IR handoff parameter_ownership does not cover J.signature in order"
        )

    signature_by_name = {
        item.get("name"): item
        for item in signature_parameters
        if isinstance(item, Mapping)
    }
    seen_shape_relations: set[tuple[str, str, str, int]] = set()
    for index, relation in enumerate(handoff.shape_relations):
        if not isinstance(relation, KernelABIShapeRelation):
            raise ValueError(
                f"Kernel IR handoff shape_relations[{index}] is not typed"
            )
        key = (
            relation.parameter,
            relation.dimension,
            relation.view_parameter,
            relation.axis,
        )
        if key in seen_shape_relations:
            raise ValueError("Kernel IR handoff shape_relations must be unique")
        seen_shape_relations.add(key)
        scalar = signature_by_name.get(relation.parameter)
        view = signature_by_name.get(relation.view_parameter)
        if scalar is None or view is None:
            raise ValueError(
                f"Kernel IR handoff shape relation {index} references an unknown parameter"
            )
        if not (
            scalar.get("kind") == JParameterKind.SCALAR.value
            and scalar.get("mode") == ParameterMode.VALUE.value
            and scalar.get("rank") == 0
            and scalar.get("logical_ref") == relation.dimension
        ):
            raise ValueError(
                f"Kernel IR handoff shape relation {index} does not bind a J dimension scalar"
            )
        if view.get("kind") not in {
            JParameterKind.POINTER.value,
            JParameterKind.TENSOR.value,
        }:
            raise ValueError(
                f"Kernel IR handoff shape relation {index} view parameter is not a tensor"
            )
        view_rank = view.get("rank")
        if (
            isinstance(view_rank, bool)
            or not isinstance(view_rank, int)
            or relation.axis < 0
            or relation.axis >= view_rank
        ):
            raise ValueError(
                f"Kernel IR handoff shape relation {index} axis is outside the view rank"
            )


@dataclass(frozen=True)
class KernelIRHandoff:
    """Machine-readable ownership handoff embedded in the neutral module."""

    entry_symbol: str
    j_digest: str
    layer_c_interface_digest: str
    shape_relations: tuple[KernelABIShapeRelation, ...]
    dimension_relations: tuple[dict[str, Any], ...]
    value_dimension_relations: tuple[dict[str, Any], ...]
    runtime_abi_parameters: tuple[dict[str, Any], ...]
    parameter_ownership: tuple[dict[str, Any], ...]
    result_relations: tuple[dict[str, Any], ...]
    e_c_declarations: tuple[dict[str, Any], ...]
    contract_ir_digest: str
    dimension_constraints: tuple[str, ...] = ()
    dimension_constraint_programs: tuple[dict[str, Any], ...] = ()
    contract_ir: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_kernel_ir_handoff(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_symbol": self.entry_symbol,
            "j_digest": self.j_digest,
            "layer_c_interface_digest": self.layer_c_interface_digest,
            "shape_relations": [
                item.to_dict() for item in self.shape_relations
            ],
            "dimension_relations": [dict(item) for item in self.dimension_relations],
            "value_dimension_relations": [
                dict(item) for item in self.value_dimension_relations
            ],
            "runtime_abi_parameters": [
                dict(item) for item in self.runtime_abi_parameters
            ],
            "parameter_ownership": [dict(item) for item in self.parameter_ownership],
            "result_relations": [dict(item) for item in self.result_relations],
            "e_c_declarations": [dict(item) for item in self.e_c_declarations],
            "contract_ir_digest": self.contract_ir_digest,
            "dimension_constraints": list(self.dimension_constraints),
            "dimension_constraint_programs": [
                dict(item) for item in self.dimension_constraint_programs
            ],
            "contract_ir": (
                dict(self.contract_ir) if self.contract_ir is not None else None
            ),
        }


@dataclass(frozen=True)
class IntentKernelModule:
    """One canonical Kernel module plus its explicit exchange handoff."""

    mlir: str
    parameter_value_ids: tuple[tuple[str, int], ...]
    j_node_operation_ids: tuple[tuple[str, int], ...]
    access_operation_ids: tuple[tuple[str, int], ...]
    consumed_relations: tuple[str, ...]
    handoff: KernelIRHandoff

    def handoff_record(self) -> dict[str, Any]:
        """Return the public handoff without exposing internal emitter state."""

        return self.handoff.to_dict()


@dataclass(frozen=True)
class _PointwiseProjection:
    j: SemanticArchive
    contract: CanonicalContractIR
    operation: Any
    operator: str
    shape: tuple[int | str, ...]
    coordinates: tuple[str, ...]
    domains: tuple[str, ...]
    view_inputs: tuple[Any, ...]
    tensor_parameters: tuple[Any, ...]
    scalar_inputs: tuple[Any, ...]
    scalar_parameters: tuple[Any, ...]
    dimension_parameters: tuple[Any, ...]
    dimension_references: tuple[str, ...]
    dimension_constraints: tuple[str, ...]
    dimension_constraint_programs: tuple[dict[str, Any], ...]
    interface_digest: str
    shape_relations: tuple[KernelABIShapeRelation, ...]
    e_c_declarations: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _RaggedProjection:
    """Closed neutral projection for one local Ragged query graph.

    Ragged descriptors are Layer-A values and therefore reuse the registered
    ``intent.ragged*`` operations.  The profile deliberately uses ordinary
    tensor arguments/results at this exchange boundary: the CUDA Ragged
    consumer turns those logical tensors into its explicit destination-passing
    memref ABI, while the common CPU/RISC-V consumer lowers the same graph to
    tensor/scf IR.  The tuple descriptor is never an argument or a function
    result.
    """

    j: SemanticArchive
    contract: CanonicalContractIR
    producer: Any
    queries: tuple[Any, ...]
    input_tensors: tuple[Any, ...]
    output_tensors: tuple[Any, ...]
    input_parameters: tuple[Any, ...]
    output_parameters: tuple[Any, ...]
    projected_parameters: tuple[Any, ...]
    output_bindings: tuple[tuple[str, str], ...]
    interface_digest: str
    e_c_declarations: tuple[dict[str, Any], ...]


_COORDINATE = re.compile(r"k[0-9]+")
_SYMBOLIC_DIMENSION = re.compile(r"x[0-9]+")
KERNEL_ENTRY_SYMBOL = "scrc_j0"
_MLIR_LOCATION = 'loc("<canonical-j>":1:1)'
_LOGICAL_INDEX_TYPE = '!intent.logical_index<"index<domain_axis_0>">'
_I64_MAX = (1 << 63) - 1


def _capability(detail: str) -> None:
    raise IntentKernelCapabilityError(
        "canonical J is outside the current Intent Kernel IR profile: " + detail
    )


def _require_empty_layer_c_interface(
    interface: LayerCInterface,
    *,
    profile: str,
) -> None:
    """Keep target realization declarations outside the neutral exporter."""

    if interface.points:
        _capability(
            f"the {profile} profile requires an empty E_C interface; "
            "bind J-addressed extension points only in the independent "
            "target bridge"
        )


def _ref(expression: ContractExpr, value: str) -> bool:
    return expression.op == "ref" and expression.value == value


def _integer(expression: ContractExpr, value: int) -> bool:
    return expression.op == "int" and expression.value == value


def _flatten_and(expression: ContractExpr) -> tuple[ContractExpr, ...]:
    if expression.op != "and":
        return (expression,)
    return tuple(
        term
        for argument in expression.args
        for term in _flatten_and(argument)
    )


def _match_dense_domain(
    expression: ContractExpr,
    *,
    coordinate: str,
    extent: int | str,
) -> bool:
    terms = _flatten_and(expression)
    if len(terms) != 2:
        return False
    lower = any(
        (
            term.op == "le"
            and len(term.args) == 2
            and _integer(term.args[0], 0)
            and _ref(term.args[1], coordinate)
        )
        or (
            term.op == "ge"
            and len(term.args) == 2
            and _ref(term.args[0], coordinate)
            and _integer(term.args[1], 0)
        )
        for term in terms
    )
    upper = any(
        term.op == "lt"
        and len(term.args) == 2
        and _ref(term.args[0], coordinate)
        and (
            _integer(term.args[1], extent)
            if isinstance(extent, int)
            else _ref(term.args[1], extent)
        )
        for term in terms
    )
    return lower and upper


def _direct_index(access: Any, coordinates: tuple[str, ...]) -> bool:
    if len(access.index) != len(coordinates):
        return False
    return all(
        component.expression.op == "ref"
        and component.expression.value == coordinate
        for component, coordinate in zip(access.index, coordinates, strict=True)
    )


def _validate_closed_pointwise_profile(
    j: SemanticArchive,
    interface: LayerCInterface,
    bound_extension_values: Mapping[str, object] | None,
) -> _PointwiseProjection:
    canonical = elaborate_j(j)
    contract = project_contract_ir(canonical)
    validate_layer_c_attachments(canonical, interface)
    _require_empty_layer_c_interface(interface, profile="pointwise")

    if bound_extension_values:
        _capability("Layer-C actual values cannot enter target-neutral Kernel IR")

    function = canonical.intent
    if len(function.outputs) != 1:
        _capability("exactly one output view is required")
    view_inputs = tuple(item for item in function.inputs if item.shape)
    scalar_inputs = tuple(item for item in function.inputs if not item.shape)
    if len(view_inputs) != 2:
        _capability("exactly two non-scalar input views are required")
    if any(
        item.dtype
        not in {
            "i8", "i16", "i32", "i64", "u8", "u16", "u32",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "index",
        }
        for item in scalar_inputs
    ):
        _capability("relation-only scalar inputs must use an integral dtype")
    if len(function.body) != 1:
        _capability("exactly one flat Layer-A operation is required")
    operation = function.body[0]
    if operation.op != "map_binary" or operation.body:
        _capability("only one flat map_binary operation is supported")
    if operation.positive is not None or operation.negative is not None:
        _capability("conditional regions are unsupported")
    if operation.node_id is None:
        _capability("the authored operation requires a stable J node id")
    if canonical.node_ids != {operation.node_id: "body[0]"}:
        _capability("node_ids must identify only the flat authored operation")
    if set(operation.attrs) != {"fn"}:
        _capability("map_binary may carry only its semantic fn attribute")
    operator = operation.attrs["fn"]
    # Canonical J owns the Layer-A vocabulary, whose multiplication spelling is
    # `mul`.  The neutral Kernel-IR serializer may use its readable `multiply`
    # spelling, but that alias must not be admitted as a second canonical-J
    # operation spelling before elaboration/validation.
    if operator not in {"add", "mul"}:
        _capability("map_binary supports only add or mul")
    if operation.operands != [item.name for item in view_inputs]:
        _capability("map_binary operands must be the two input views in ABI order")
    output = function.outputs[0]
    if operation.result != output.name:
        _capability("map_binary result must directly define the sole output")

    tensors = [*function.inputs, output]
    dtype = view_inputs[0].dtype
    shape = tuple(view_inputs[0].shape)
    if dtype != "f32" or any(item.dtype != dtype for item in [*view_inputs, output]):
        _capability("the initial pointwise profile requires matching f32 tensors")
    if not shape or any(tuple(item.shape) != shape for item in [*view_inputs, output]):
        _capability("all pointwise views require one equal, non-scalar shape")
    if any(
        isinstance(extent, bool)
        or not isinstance(extent, (int, str))
        or (isinstance(extent, int) and not 0 < extent <= _I64_MAX)
        or (isinstance(extent, str) and _SYMBOLIC_DIMENSION.fullmatch(extent) is None)
        for extent in shape
    ):
        _capability(
            "shape extents must be positive int64 values or canonical x* dimensions"
        )
    if operation.type != {"dtype": dtype, "shape": list(shape)}:
        _capability("map_binary result type must exactly match the output view")

    elaboration = j_elaboration(canonical)
    bindings = tuple((item.output, item.value) for item in elaboration.output_bindings)
    if bindings != ((output.name, output.name),):
        _capability("the sole output requires one exact direct result binding")

    unsupported_boundary = {
        "aliases": canonical.boundary.aliases,
        "broadcasts": canonical.boundary.broadcasts,
        "consumer_extensions": canonical.boundary.consumer_extensions,
        "access_guard_relations": canonical.boundary.access_guard_relations,
        "reductions": canonical.boundary.reductions,
        "index_bounds": canonical.boundary.index_bounds,
        "output_mappings": canonical.boundary.output_mappings,
        "updates": canonical.boundary.updates,
    }
    present = sorted(name for name, values in unsupported_boundary.items() if values)
    if present:
        _capability("unsupported Layer-B relation(s): " + ", ".join(present))
    try:
        validate_value_dimension_relations(canonical)
        constraint_programs = contract.typed_dimension_constraints
    except (TypeError, ValueError) as exc:
        _capability(f"J dimension relation grammar is not consumable: {exc}")

    declared_dimensions = set(declared_dimension_handles(canonical))
    relation_dimensions: list[str] = []
    for relation in canonical.dimension_relations:
        relation_dimensions.extend(
            [
                relation.result,
                relation.expr.arg,
                *((relation.expr.rhs,) if relation.expr.rhs is not None else ()),
            ]
        )
    relation_dimensions.extend(
        item.dimension_ref for item in canonical.value_dimension_relations
    )
    constraint_references = sorted(
        {
            reference
            for program in constraint_programs
            for reference in program.expression.references
        }
    )
    all_dimension_references = tuple(
        dict.fromkeys(
            [
                *(
                    extent
                    for extent in shape
                    if isinstance(extent, str)
                ),
                *relation_dimensions,
                *constraint_references,
            ]
        )
    )
    invalid_dimension_references = sorted(
        reference
        for reference in all_dimension_references
        if not isinstance(reference, str)
        or _SYMBOLIC_DIMENSION.fullmatch(reference) is None
        or reference not in declared_dimensions
    )
    if invalid_dimension_references:
        _capability(
            "dimension relations reference undeclared/non-symbolic dimension(s): "
            + ", ".join(invalid_dimension_references)
        )
    value_relation_inputs = {
        relation.value_ref for relation in canonical.value_dimension_relations
    }
    scalar_input_names = {item.name for item in scalar_inputs}
    if value_relation_inputs != scalar_input_names:
        missing = sorted(scalar_input_names - value_relation_inputs)
        extra = sorted(value_relation_inputs - scalar_input_names)
        detail = []
        if missing:
            detail.append("unrelated scalar input(s) " + ", ".join(missing))
        if extra:
            detail.append("non-input value relation(s) " + ", ".join(extra))
        _capability(
            "relation-only scalar inputs must be covered exactly by "
            "value_dimension_relations"
            + (": " + "; ".join(detail) if detail else "")
        )
    if canonical.storage:
        _capability("semantic storage is unsupported")
    if len(canonical.scopes) != 1 or canonical.scopes[0].kind.value != "function":
        _capability("only one flat function scope is supported")
    scope = canonical.scopes[0]
    if scope.parent_scope is not None or scope.captures:
        _capability("the flat function scope cannot capture or nest")

    result_relations = canonical.boundary.result_relations
    if (
        len(result_relations) != 1
        or result_relations[0].output != output.name
        or result_relations[0].kind is not ResultRelationKind.EXACT
    ):
        _capability("the output requires one exact result relation")

    dimension_reference_set = {
        reference for reference in all_dimension_references if isinstance(reference, str)
    }
    symbolic_dimensions = tuple(
        parameter.logical_ref
        for parameter in canonical.signature.parameters
        if parameter.logical_ref in dimension_reference_set
    )
    if (
        len(symbolic_dimensions) != len(dimension_reference_set)
        or set(symbolic_dimensions) != dimension_reference_set
    ):
        _capability(
            "every J dimension relation reference requires one ordered signature "
            "parameter"
        )
    tensor_parameters: list[Any] = []
    scalar_parameters: list[Any] = []
    dimension_parameters: list[Any] = []
    expected_tensor_modes = {
        view_inputs[0].name: ParameterMode.INPUT,
        view_inputs[1].name: ParameterMode.INPUT,
        output.name: ParameterMode.OUTPUT,
    }
    view_by_name = {item.name: item for item in [*view_inputs, output]}
    scalar_by_name = {item.name: item for item in scalar_inputs}
    tensor_by_name = {item.name: item for item in tensors}
    for parameter in canonical.signature.parameters:
        if parameter.alias_of is not None:
            _capability("signature aliases are unsupported")
        if parameter.logical_ref in view_by_name:
            tensor = view_by_name[parameter.logical_ref]
            if parameter.kind not in {JParameterKind.POINTER, JParameterKind.TENSOR}:
                _capability(f"{parameter.name} must be a tensor/pointer view")
            if (
                parameter.mode is not expected_tensor_modes[tensor.name]
                or parameter.dtype != tensor.dtype
                or parameter.rank != len(tensor.shape)
            ):
                _capability(f"{parameter.name} does not exactly match {tensor.name}")
            tensor_parameters.append(parameter)
            continue
        if parameter.logical_ref in scalar_by_name:
            scalar = scalar_by_name[parameter.logical_ref]
            if (
                parameter.kind is not JParameterKind.SCALAR
                or parameter.mode is not ParameterMode.VALUE
                or parameter.rank != 0
                or parameter.dtype != scalar.dtype
            ):
                _capability(
                    f"{parameter.name} does not exactly match relation-only scalar "
                    f"input {scalar.name}"
                )
            scalar_parameters.append(parameter)
            continue
        if parameter.logical_ref in symbolic_dimensions:
            if (
                parameter.kind is not JParameterKind.SCALAR
                or parameter.mode is not ParameterMode.VALUE
                or parameter.rank != 0
                or parameter.dtype not in {"i32", "i64"}
            ):
                _capability(f"{parameter.name} is not a canonical runtime dimension scalar")
            dimension_parameters.append(parameter)
            continue
        _capability(f"signature parameter {parameter.name} is not used by this J projection")
    if [item.logical_ref for item in tensor_parameters] != [
        *[item.name for item in view_inputs], output.name
    ]:
        _capability("view parameters must cover input/input/output in signature order")
    if [item.logical_ref for item in scalar_parameters] != [
        item.name for item in scalar_inputs
    ]:
        _capability("relation-only scalar parameters must cover scalar inputs in order")
    if [item.logical_ref for item in dimension_parameters] != list(symbolic_dimensions):
        _capability(
            "runtime dimension scalars must cover every J dimension relation "
            "reference once in canonical order"
        )

    parameter_by_dimension = {
        parameter.logical_ref: parameter
        for parameter in dimension_parameters
    }
    parameter_by_tensor = {
        parameter.logical_ref: parameter
        for parameter in tensor_parameters
    }
    first_view_parameter = parameter_by_tensor[view_inputs[0].name]
    shape_relations = tuple(
        KernelABIShapeRelation(
            parameter=parameter_by_dimension[extent].name,
            dimension=extent,
            view_parameter=first_view_parameter.name,
            axis=axis,
        )
        for axis, extent in enumerate(shape)
        if isinstance(extent, str)
    )

    coordinates = tuple(f"k{axis}" for axis in range(len(shape)))
    domains = tuple(f"d{axis}" for axis in range(len(shape)))
    if len(contract.domains) != len(shape):
        _capability("one dense domain per tensor axis is required")
    for axis, (domain, extent) in enumerate(zip(contract.domains, shape, strict=True)):
        if domain.domain_id != domains[axis] or not _match_dense_domain(
            domain.predicate.expression,
            coordinate=coordinates[axis],
            extent=extent,
        ):
            _capability(f"{domain.domain_id} is not the direct dense domain for axis {axis}")

    expected_accesses = (
        (AccessKind.LOAD.value, view_inputs[0].name),
        (AccessKind.LOAD.value, view_inputs[1].name),
        (AccessKind.STORE.value, output.name),
    )
    if len(contract.accesses) != len(expected_accesses):
        _capability("exactly load/load/store accesses are required")
    for access, expected in zip(contract.accesses, expected_accesses, strict=True):
        if (access.kind, access.tensor) != expected:
            _capability("access order must be load(input0), load(input1), store(output)")
        if access.node_ref != operation.node_id:
            _capability(f"access {access.access_id} is not anchored to the authored node")
        if access.predicate is not None or access.fill_value is not None:
            _capability(f"access {access.access_id} must be unpredicated and unfilled")
        if not _direct_index(access, coordinates):
            _capability(f"access {access.access_id} must use the complete direct index vector")

    effects = canonical.boundary.effects
    effects_by_access = {item.access_ref: item for item in effects}
    if (
        len(effects) != len(contract.accesses)
        or len(effects_by_access) != len(effects)
        or len({item.effect_id for item in effects}) != len(effects)
        or set(effects_by_access) != {item.access_id for item in contract.accesses}
    ):
        _capability("each access requires exactly one matching effect")
    for access in contract.accesses:
        effect = effects_by_access[access.access_id]
        expected_kind = (
            JEffectKind.READ if access.kind == AccessKind.LOAD.value else JEffectKind.WRITE
        )
        if (
            effect.kind is not expected_kind
            or effect.target != access.tensor
            or effect.node_ref != operation.node_id
        ):
            _capability(f"effect for {access.access_id} does not match its access")
    if set(scope.effects) != {item.effect_id for item in canonical.boundary.effects}:
        _capability("the function scope must own all and only boundary effects")
    if set(scope.definitions) != {item.name for item in tensors} or len(
        scope.definitions
    ) != len(tensors):
        _capability("the function scope must define all and only pointwise tensors")

    domain_ids = set(domains)
    authorizations = canonical.boundary.realization_authorizations
    if len(authorizations) != len(domains) or {
        item.subject_ref for item in authorizations
    } != domain_ids:
        _capability("each dense domain requires one signed-widen authorization")
    for authorization in authorizations:
        if (
            authorization.kind.value != "signed_widen"
            or authorization.subject_ref not in domain_ids
            or authorization.source_dtype not in {"i32", "i64"}
            or authorization.target_dtype != "i64"
        ):
            _capability(
                f"realization authorization {authorization.auth_id} is unsupported"
            )

    return _PointwiseProjection(
        j=canonical,
        contract=contract,
        operation=operation,
        operator="multiply" if operator == "mul" else "add",
        shape=shape,
        coordinates=coordinates,
        domains=domains,
        view_inputs=view_inputs,
        tensor_parameters=tuple(tensor_parameters),
        scalar_inputs=scalar_inputs,
        scalar_parameters=tuple(scalar_parameters),
        dimension_parameters=tuple(dimension_parameters),
        dimension_references=symbolic_dimensions,
        dimension_constraints=tuple(
            program.expression.to_text() for program in constraint_programs
        ),
        dimension_constraint_programs=tuple(
            {
                "expression": program.expression.to_json(),
                "reference_sorts": {
                    name: sort.value
                    for name, sort in program.reference_sorts
                },
                "result_sort": program.result_sort.value,
            }
            for program in constraint_programs
        ),
        interface_digest=binding_interface_digest(interface),
        shape_relations=shape_relations,
        e_c_declarations=tuple(point.to_dict() for point in interface.points),
    )


def _validate_closed_ragged_profile(
    j: SemanticArchive,
    interface: LayerCInterface,
    bound_extension_values: Mapping[str, object] | None,
) -> _RaggedProjection:
    """Validate the direct, backend-shared Ragged query profile.

    This profile is intentionally small but total over the three Ragged ops:
    one local descriptor producer followed by one ``ragged_outer`` and/or one
    ``ragged_member`` query.  All queries must be direct function results.  It
    is the same structural envelope consumed by the real CUDA Ragged pass and
    by the common tensor lowering; an unsupported graph fails here instead of
    producing a partial neutral module.
    """

    canonical = elaborate_j(j)
    contract = project_contract_ir(canonical)
    validate_layer_c_attachments(canonical, interface)
    _require_empty_layer_c_interface(interface, profile="Ragged")
    if bound_extension_values:
        _capability("Layer-C actual values cannot enter target-neutral Kernel IR")

    function = canonical.intent
    body = tuple(function.body)
    if len(body) not in {2, 3} or body[0].op != "ragged":
        _capability(
            "Ragged Kernel IR requires one leading ragged producer and one or "
            "two direct outer/member queries"
        )
    if any(
        operation.op not in RAGGED_OPS
        or operation.body
        or operation.positive is not None
        or operation.negative is not None
        for operation in body
    ):
        _capability("Ragged Kernel IR does not admit nested or conditional operations")
    if any(operation.node_id is None for operation in body):
        _capability("every Ragged operation requires a stable J node id")
    expected_paths = {f"body[{index}]" for index in range(len(body))}
    if set(canonical.node_ids.values()) != expected_paths:
        _capability("node_ids must identify exactly the local Ragged operation graph")

    producer = body[0]
    if producer.attrs or not isinstance(producer.result, str) or len(producer.operands) != 2:
        _capability("ragged requires two carrier operands, one result, and no attrs")
    offsets_name, indices_name = producer.operands
    input_by_name = {item.name: item for item in function.inputs}
    output_by_name = {item.name: item for item in function.outputs}
    if offsets_name not in input_by_name or indices_name not in input_by_name:
        _capability("ragged carrier operands must be direct function inputs")

    def closed_extent(value: object, *, context: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > _I64_MAX
        ):
            _capability(f"{context} must be a positive concrete int64 extent")
        return value

    def validate_carrier(name: str, *, context: str, rank: int) -> Any:
        tensor = input_by_name.get(name)
        if tensor is None or len(tensor.shape) != rank:
            _capability(f"{context} must be a rank-{rank} function-input tensor")
        if not is_ragged_carrier_dtype(tensor.dtype):
            _capability(f"{context} must use a closed Ragged carrier dtype")
        for axis, extent in enumerate(tensor.shape):
            closed_extent(extent, context=f"{context}.shape[{axis}]")
        return tensor

    offsets = validate_carrier(offsets_name, context="ragged offsets", rank=1)
    indices = validate_carrier(indices_name, context="ragged indices", rank=1)
    if offsets_name == indices_name:
        _capability("ragged offsets and indices must be distinct inputs")

    elaboration = j_elaboration(canonical)
    value_types = elaboration.value_type_by_name
    descriptor = value_types.get(producer.result)
    if descriptor is None or not descriptor.is_ragged:
        _capability("ragged must define a closed descriptor value")
    if tuple(name for name, _ in descriptor.fields) != ("offsets", "indices"):
        _capability("ragged descriptor fields must be offsets followed by indices")
    if descriptor.field("offsets").to_json() != {
        "dtype": offsets.dtype,
        "shape": list(offsets.shape),
    }:
        _capability("ragged descriptor offsets field differs from its operand")
    if descriptor.field("indices").to_json() != {
        "dtype": indices.dtype,
        "shape": list(indices.shape),
    }:
        _capability("ragged descriptor indices field differs from its operand")

    queries: list[Any] = []
    selector_names: set[str] = set()
    result_to_query: dict[str, Any] = {}
    seen_kinds: set[str] = set()
    for operation in body[1:]:
        if operation.op not in {"ragged_outer", "ragged_member"}:
            _capability(
                "only ragged_outer and ragged_member may follow the ragged "
                "descriptor producer"
            )
        if operation.op in seen_kinds:
            _capability(f"Ragged Kernel IR admits at most one {operation.op} query")
        seen_kinds.add(operation.op)
        if operation.attrs or not isinstance(operation.result, str):
            _capability(f"{operation.op} requires one result and no attrs")
        expected_arity = 1 if operation.op == "ragged_outer" else 2
        if len(operation.operands) != expected_arity:
            _capability(f"{operation.op} has an invalid operand count")
        if operation.operands[0] != producer.result:
            _capability(f"{operation.op} must consume the local ragged descriptor")
        if operation.result in result_to_query:
            _capability("Ragged query result names must be unique")
        result_type = value_types.get(operation.result)
        if result_type is None or not result_type.is_tensor:
            _capability(f"{operation.op} result type is not a tensor")
        if operation.op == "ragged_outer":
            if result_type.shape != () or result_type.dtype != offsets.dtype:
                _capability(
                    "ragged_outer result must be a rank-0 offsets-carrier tensor"
                )
        else:
            selector_name = operation.operands[1]
            selector = validate_carrier(
                selector_name,
                context="ragged_member selector",
                rank=0,
            )
            if result_type.shape != ("?",) or result_type.dtype != indices.dtype:
                _capability(
                    "ragged_member result must be a dynamic rank-1 indices-carrier tensor"
                )
            selector_names.add(selector.name)
        queries.append(operation)
        result_to_query[operation.result] = operation

    expected_inputs = {offsets_name, indices_name, *selector_names}
    if set(input_by_name) != expected_inputs:
        _capability(
            "Ragged Kernel IR requires exactly the carrier and selector inputs "
            "used by its local query graph"
        )
    bindings = tuple(
        (item.output, item.value) for item in elaboration.output_bindings
    )
    output_names = tuple(output.name for output in function.outputs)
    if tuple(output for output, _ in bindings) != output_names:
        _capability(
            "Ragged output bindings must cover function outputs once and in ABI order"
        )
    if len({output for output, _ in bindings}) != len(bindings):
        _capability("Ragged output bindings must not repeat an output tensor")
    query_results = tuple(operation.result for operation in queries)
    if len({value for _, value in bindings}) != len(bindings):
        _capability("Ragged output bindings must not repeat a query result")
    if set(output for output, _ in bindings) != set(output_by_name):
        _capability(
            "function outputs must be exactly the outputs named by Ragged bindings"
        )
    if set(value for _, value in bindings) != set(query_results):
        _capability(
            "each Ragged output must bind exactly one direct outer/member query result"
        )
    if any(value not in result_to_query for _, value in bindings):
        _capability(
            "Ragged output bindings may reference only direct query result values"
        )

    unsupported_boundary = {
        "domains": canonical.boundary.domains,
        "broadcasts": canonical.boundary.broadcasts,
        "aliases": canonical.boundary.aliases,
        "consumer_extensions": canonical.boundary.consumer_extensions,
        "access_guard_relations": canonical.boundary.access_guard_relations,
        "reductions": canonical.boundary.reductions,
        "index_bounds": canonical.boundary.index_bounds,
        "updates": canonical.boundary.updates,
    }
    present = sorted(name for name, values in unsupported_boundary.items() if values)
    if present:
        _capability("unsupported Ragged Layer-B relation(s): " + ", ".join(present))
    if canonical.storage:
        _capability("Ragged descriptors cannot carry semantic storage")
    if canonical.dimension_constraints or canonical.dimension_relations:
        _capability("Ragged direct query profile does not have runtime dimensions")
    if canonical.value_dimension_relations:
        _capability("Ragged direct query profile does not have value/dimension inputs")
    if canonical.scopes:
        if len(canonical.scopes) != 1:
            _capability("Ragged direct query profile admits only one function scope")
        scope = canonical.scopes[0]
        if scope.kind.value != "function" or scope.parent_scope is not None or scope.captures:
            _capability("Ragged direct query scope cannot capture or nest")

    tensor_names = set(input_by_name) | set(output_by_name)
    signature_parameters = tuple(canonical.signature.parameters)
    if len(signature_parameters) != len(tensor_names):
        _capability("Ragged J.signature must bind every input and output tensor once")
    parameter_by_ref: dict[str, Any] = {}
    input_parameters: list[Any] = []
    output_parameters: list[Any] = []
    for parameter in signature_parameters:
        if (
            parameter.logical_ref is None
            or parameter.logical_ref not in tensor_names
            or parameter.alias_of is not None
            or parameter.logical_ref in parameter_by_ref
        ):
            _capability("Ragged signature parameters must be direct non-aliased tensors")
        tensor = input_by_name.get(parameter.logical_ref) or output_by_name.get(
            parameter.logical_ref
        )
        expected_mode = (
            ParameterMode.INPUT
            if parameter.logical_ref in input_by_name
            else ParameterMode.OUTPUT
        )
        if (
            parameter.kind not in {JParameterKind.POINTER, JParameterKind.TENSOR}
            or parameter.mode is not expected_mode
            or parameter.dtype != tensor.dtype
            or parameter.rank != len(tensor.shape)
        ):
            _capability(
                f"signature parameter {parameter.name} does not exactly match "
                f"Ragged tensor {parameter.logical_ref}"
            )
        parameter_by_ref[parameter.logical_ref] = parameter
        if expected_mode is ParameterMode.INPUT:
            input_parameters.append(parameter)
        else:
            output_parameters.append(parameter)
    if set(parameter_by_ref) != tensor_names:
        _capability("Ragged signature does not cover every logical tensor")

    return _RaggedProjection(
        j=canonical,
        contract=contract,
        producer=producer,
        queries=tuple(queries),
        input_tensors=tuple(function.inputs),
        output_tensors=tuple(function.outputs),
        input_parameters=tuple(input_parameters),
        output_parameters=tuple(output_parameters),
        projected_parameters=signature_parameters,
        output_bindings=bindings,
        interface_digest=binding_interface_digest(interface),
        e_c_declarations=tuple(point.to_dict() for point in interface.points),
    )


@dataclass(frozen=True)
class _KernelValue:
    """One private SSA value in TianchenIR's closed Kernel MLIR profile."""

    value_id: int
    mlir_type: str
    metadata_type: str

    @property
    def ref(self) -> str:
        return f"%v{self.value_id}"


def _mlir_string(value: str, *, context: str) -> str:
    """Quote one already-validated metadata value without an injection surface.

    The current profile never needs arbitrary byte strings.  Reject characters
    that would require MLIR escaping rather than maintaining a second general
    string codec beside the canonical JSON decoder.
    """

    if not value or any(
        character in {'"', "\\"}
        or ord(character) < 0x20
        or ord(character) > 0x7E
        for character in value
    ):
        _capability(f"{context} cannot be represented as closed MLIR metadata")
    return f'"{value}"'


def _mlir_json_string(value: Any, *, context: str) -> str:
    """Encode one closed JSON record as an MLIR string attribute."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _capability(f"{context} is not a closed JSON record: {exc}")
    escaped = encoded.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _kernel_ragged_dtype(dtype: str) -> str:
    """Render one logical Ragged carrier dtype in MLIR syntax."""

    if dtype == "index":
        # `index` is a logical ABI spelling; the current native/device carrier
        # is the signed 64-bit integer used by the common and CUDA lowerings.
        return "i64"
    return {"u8": "ui8", "u16": "ui16", "u32": "ui32"}.get(dtype, dtype)


def _kernel_ragged_tensor_type(shape: tuple[Any, ...], dtype: str) -> str:
    rendered_dtype = _kernel_ragged_dtype(dtype)
    if not shape:
        return f"tensor<{rendered_dtype}>"
    return "tensor<" + "x".join([*(str(item) for item in shape), rendered_dtype]) + ">"


def _kernel_ragged_descriptor_type(offsets: Any, indices: Any) -> str:
    return (
        "tuple<"
        + _kernel_ragged_tensor_type(tuple(offsets.shape), offsets.dtype)
        + ", "
        + _kernel_ragged_tensor_type(tuple(indices.shape), indices.dtype)
        + ">"
    )


class _PointwiseKernelEmitter:
    """TianchenIR-owned serializer for the validated pointwise projection.

    This is intentionally not a public, general-purpose MLIR builder.  Its
    operation vocabulary, types, attributes, region shapes, and ID allocation
    are closed over :class:`_PointwiseProjection`, which has already passed the
    fail-closed representability checks above.
    """

    _OPERATIONS = frozenset(
        {
            "constant",
            "dim",
            "domain",
            "parallel",
            "for",
            "view_load",
            "binary",
            "view_store",
            "yield",
            "return",
        }
    )

    def __init__(self, projection: _PointwiseProjection) -> None:
        self.projection = projection
        self._next_value_id = 0
        self._next_operation_id = 0
        self._access_operation_ids: list[tuple[str, int]] = []
        self._binary_operation_id: int | None = None
        self._tensor_by_ref = {
            item.name: item
            for item in [
                *projection.j.intent.inputs,
                *projection.j.intent.outputs,
            ]
        }
        self._view_by_ref = {
            item.name: item
            for item in [*projection.view_inputs, *projection.j.intent.outputs]
        }
        self._scalar_input_by_ref = {
            item.name: item for item in projection.scalar_inputs
        }
        signature_position = {
            item.name: index
            for index, item in enumerate(projection.j.signature.parameters)
        }
        projected_parameters = [
            *projection.tensor_parameters,
            *projection.scalar_parameters,
            *projection.dimension_parameters,
        ]
        projected_parameters.sort(key=lambda item: signature_position[item.name])
        self.projected_parameters = tuple(projected_parameters)
        self.parameter_values: dict[str, _KernelValue] = {}
        for parameter in self.projected_parameters:
            logical_ref = parameter.logical_ref
            tensor = self._view_by_ref.get(logical_ref)
            if tensor is None:
                mlir_type = str(parameter.dtype)
            else:
                view_kind = {
                    ParameterMode.INPUT: "in",
                    ParameterMode.OUTPUT: "out",
                }[parameter.mode]
                mlir_type = (
                    f"!intent.view<{self._runtime_tensor_type(tuple(tensor.shape))}, "
                    f'"{view_kind}">'
                )
            self.parameter_values[logical_ref] = self._new_value(
                mlir_type,
                mlir_type,
            )

    @staticmethod
    def _runtime_tensor_type(shape: tuple[int | str, ...]) -> str:
        dimensions = ["?" if isinstance(item, str) else str(item) for item in shape]
        return "tensor<" + "x".join([*dimensions, "f32"]) + ">"

    @staticmethod
    def _logical_tensor_type(shape: tuple[int | str, ...]) -> str:
        return "tensor<" + "x".join([*(str(item) for item in shape), "f32"]) + ">"

    def _new_value(self, mlir_type: str, metadata_type: str) -> _KernelValue:
        value = _KernelValue(self._next_value_id, mlir_type, metadata_type)
        self._next_value_id += 1
        return value

    def _new_operation_id(self) -> int:
        operation_id = self._next_operation_id
        self._next_operation_id += 1
        return operation_id

    def _common_attributes(
        self,
        operation_id: int,
        results: tuple[_KernelValue, ...],
        result_names: tuple[str, ...],
    ) -> list[str]:
        if len(results) != len(result_names):
            raise AssertionError("Kernel result metadata is not positional")
        names = ", ".join(
            _mlir_string(name, context="Kernel result name")
            for name in result_names
        )
        nodes = ", ".join(f"{value.value_id} : i64" for value in results)
        types = ", ".join(
            _mlir_string(value.metadata_type, context="Kernel result type")
            for value in results
        )
        return [
            f"intent.node = {operation_id} : i64",
            f"intent.result_names = [{names}]",
            f"intent.result_nodes = [{nodes}]",
            f"intent.result_types = [{types}]",
        ]

    def _operation_line(
        self,
        *,
        name: str,
        operation_id: int,
        operands: tuple[_KernelValue, ...] = (),
        results: tuple[_KernelValue, ...] = (),
        result_names: tuple[str, ...] = (),
        leading_attributes: tuple[str, ...] = (),
        node_attributes: tuple[str, ...] = (),
        trailing_attributes: tuple[str, ...] = (),
        level: int,
    ) -> str:
        if name not in self._OPERATIONS:
            raise AssertionError(f"operation {name!r} is outside the closed profile")
        if len(results) > 1:
            raise AssertionError("the closed pointwise profile has at most one result")
        common = self._common_attributes(operation_id, results, result_names)
        rendered_attributes = ", ".join(
            [
                *leading_attributes,
                common[0],
                *node_attributes,
                *common[1:],
                *trailing_attributes,
            ]
        )
        operand_refs = ", ".join(value.ref for value in operands)
        operand_types = ", ".join(value.mlir_type for value in operands)
        result_type = results[0].mlir_type if results else "()"
        assignment = f"{results[0].ref} = " if results else ""
        return (
            f'{"  " * level}{assignment}"intent.{name}"({operand_refs}) '
            f"{{{rendered_attributes}}} : ({operand_types}) -> {result_type} "
            f"{_MLIR_LOCATION}"
        )

    @staticmethod
    def _index_attribute(*, operand_offset: int, rank: int) -> str:
        terms = ", ".join(
            "{kind = \"value_index\", operands = "
            f"[{axis + operand_offset} : i64], static = []}}"
            for axis in range(rank)
        )
        return f"intent.index = [{terms}]"

    @staticmethod
    def _effect_attribute(kind: str) -> str:
        if kind not in {"read", "write"}:
            raise AssertionError("the closed pointwise profile has read/write effects")
        return (
            "intent.effects = [{kind = "
            f'"{kind}", resource = "external_view", target = 0 : i64}}]'
        )

    @staticmethod
    def _shape_relation_attribute(
        relation: KernelABIShapeRelation,
    ) -> str:
        return (
            f"{{axis = {relation.axis} : i64, "
            f"dimension = {_mlir_string(relation.dimension, context='ABI dimension')}, "
            'kind = "scalar_equals_view_dimension", '
            f"parameter = {_mlir_string(relation.parameter, context='ABI parameter')}, "
            f"view_parameter = {_mlir_string(relation.view_parameter, context='ABI view parameter')}}}"
        )

    def _emit_result_operation(
        self,
        *,
        name: str,
        operands: tuple[_KernelValue, ...],
        result_type: str,
        metadata_type: str,
        result_name: str,
        leading_attributes: tuple[str, ...] = (),
        node_attributes: tuple[str, ...] = (),
        trailing_attributes: tuple[str, ...] = (),
        level: int,
    ) -> tuple[int, _KernelValue, str]:
        operation_id = self._new_operation_id()
        result = self._new_value(result_type, metadata_type)
        line = self._operation_line(
            name=name,
            operation_id=operation_id,
            operands=operands,
            results=(result,),
            result_names=(result_name,),
            leading_attributes=leading_attributes,
            node_attributes=node_attributes,
            trailing_attributes=trailing_attributes,
            level=level,
        )
        return operation_id, result, line

    def _emit_effect_operation(
        self,
        *,
        name: str,
        operands: tuple[_KernelValue, ...],
        leading_attributes: tuple[str, ...] = (),
        node_attributes: tuple[str, ...] = (),
        trailing_attributes: tuple[str, ...] = (),
        level: int,
    ) -> tuple[int, str]:
        operation_id = self._new_operation_id()
        return operation_id, self._operation_line(
            name=name,
            operation_id=operation_id,
            operands=operands,
            leading_attributes=leading_attributes,
            node_attributes=node_attributes,
            trailing_attributes=trailing_attributes,
            level=level,
        )

    def _emit_domains(self, *, level: int) -> tuple[list[_KernelValue], list[str]]:
        domain_values: list[_KernelValue] = []
        lines: list[str] = []
        first_view = self.parameter_values[
            self.projection.tensor_parameters[0].logical_ref
        ]
        for axis, (domain_ref, extent) in enumerate(
            zip(self.projection.domains, self.projection.shape, strict=True)
        ):
            _, lower, lower_line = self._emit_result_operation(
                name="constant",
                operands=(),
                result_type="index",
                metadata_type="index",
                result_name=f"{domain_ref}_lower",
                trailing_attributes=("intent.value = 0 : i64",),
                level=level,
            )
            lines.append(lower_line)
            if isinstance(extent, int):
                _, upper, upper_line = self._emit_result_operation(
                    name="constant",
                    operands=(),
                    result_type="index",
                    metadata_type="index",
                    result_name=f"{domain_ref}_upper",
                    trailing_attributes=(f"intent.value = {extent} : i64",),
                    level=level,
                )
                flavor = "dense"
            else:
                # J binds the runtime scalar parameter to this symbolic
                # dimension.  The handoff records that relation explicitly;
                # the view dimension is the safe representative used for the
                # generated loop, so an invalid external scalar valuation
                # cannot make the loop exceed the view.
                _, upper, upper_line = self._emit_result_operation(
                    name="dim",
                    operands=(first_view,),
                    result_type="index",
                    metadata_type="index",
                    result_name=f"{extent}_view_dimension",
                    leading_attributes=(f"intent.axis = {axis} : i64",),
                    level=level,
                )
                flavor = "runtime"
            lines.append(upper_line)
            metadata_type = f"domain<{flavor},1>"
            mlir_type = f'!intent.domain<"{metadata_type}">'
            _, domain, domain_line = self._emit_result_operation(
                name="domain",
                operands=(lower, upper),
                result_type=mlir_type,
                metadata_type=metadata_type,
                result_name=domain_ref,
                level=level,
            )
            domain_values.append(domain)
            lines.append(domain_line)
        return domain_values, lines

    def _emit_element(
        self,
        *,
        coordinates: tuple[_KernelValue, ...],
        level: int,
    ) -> list[str]:
        if len(coordinates) != len(self.projection.shape):
            raise AssertionError("pointwise coordinate rank is incomplete")
        lines: list[str] = []
        input_values: list[_KernelValue] = []
        load_index = self._index_attribute(operand_offset=1, rank=len(coordinates))
        for access, tensor in zip(
            self.projection.contract.accesses[:2],
            self.projection.view_inputs,
            strict=True,
        ):
            view = self.parameter_values[tensor.name]
            operation_id, value, line = self._emit_result_operation(
                name="view_load",
                operands=(view, *coordinates),
                result_type="f32",
                metadata_type="f32",
                result_name=access.access_id,
                leading_attributes=(self._effect_attribute("read"), load_index),
                level=level,
            )
            self._access_operation_ids.append((access.access_id, operation_id))
            input_values.append(value)
            lines.append(line)

        output = self.projection.j.intent.outputs[0]
        operation_id, binary, binary_line = self._emit_result_operation(
            name="binary",
            operands=tuple(input_values),
            result_type="f32",
            metadata_type="f32",
            result_name=output.name,
            node_attributes=(f'intent.operator = "{self.projection.operator}"',),
            level=level,
        )
        self._binary_operation_id = operation_id
        lines.append(binary_line)

        output_view = self.parameter_values[output.name]
        store_id, store_line = self._emit_effect_operation(
            name="view_store",
            operands=(output_view, binary, *coordinates),
            leading_attributes=(
                self._effect_attribute("write"),
                self._index_attribute(operand_offset=2, rank=len(coordinates)),
            ),
            trailing_attributes=(
                "intent.value_operand_index = 1 : i64",
            ),
            level=level,
        )
        self._access_operation_ids.append(
            (self.projection.contract.accesses[2].access_id, store_id)
        )
        lines.append(store_line)
        return lines

    def _emit_axis(
        self,
        *,
        depth: int,
        loop_axes: tuple[int, ...],
        domain_values: tuple[_KernelValue, ...],
        coordinates: Mapping[int, _KernelValue],
        level: int,
    ) -> list[str]:
        axis = loop_axes[depth]
        argument = self._new_value(_LOGICAL_INDEX_TYPE, "logical_index")
        nested_coordinates = dict(coordinates)
        nested_coordinates[axis] = argument
        body_level = level + 3
        if depth + 1 == len(loop_axes):
            body_lines = self._emit_element(
                coordinates=tuple(
                    nested_coordinates[item]
                    for item in range(len(self.projection.shape))
                ),
                level=body_level,
            )
        else:
            body_lines = self._emit_axis(
                depth=depth + 1,
                loop_axes=loop_axes,
                domain_values=domain_values,
                coordinates=nested_coordinates,
                level=body_level,
            )
        _, yield_line = self._emit_effect_operation(
            name="yield",
            operands=(),
            level=body_level,
        )
        body_lines.append(yield_line)

        opcode = (
            "for"
            if isinstance(self.projection.shape[axis], str)
            else "parallel"
        )
        operation_id = self._new_operation_id()
        common = self._common_attributes(operation_id, (), ())
        region_attributes = [
            f"intent.region_argument_names = "
            f"[[[{_mlir_string(self.projection.coordinates[axis], context='coordinate name')}]]]",
            f"intent.region_argument_nodes = [[[{argument.value_id} : i64]]]",
        ]
        attributes = ", ".join([common[0], *region_attributes, *common[1:]])
        domain = domain_values[axis]
        indent = "  " * level
        return [
            f'{indent}"intent.{opcode}"({domain.ref}) (',
            f"{indent}  {{",
            f"{indent}    ^bb0({argument.ref}: {_LOGICAL_INDEX_TYPE}):",
            *body_lines,
            f"{indent}  }}",
            f"{indent}) {{{attributes}}} : ({domain.mlir_type}) -> () {_MLIR_LOCATION}",
        ]

    def _parameter_metadata(self) -> list[str]:
        metadata: list[str] = []
        for parameter in self.projected_parameters:
            tensor = self._view_by_ref.get(parameter.logical_ref)
            name = _mlir_string(parameter.name, context="Kernel parameter name")
            logical_ref = (
                f', logical_ref = {_mlir_string(parameter.logical_ref, context="Kernel logical reference")}'
                if parameter.logical_ref is not None
                else ""
            )
            if tensor is None:
                dtype = _mlir_string(str(parameter.dtype), context="scalar type")
                kind = (
                    "logical_scalar"
                    if parameter.logical_ref in self._scalar_input_by_ref
                    else "runtime_scalar"
                )
                metadata.append(
                    f'{{kind = "{kind}", name = {name}, '
                    f'type = {dtype}{logical_ref}}}'
                )
                continue
            view_kind = {
                ParameterMode.INPUT: "in",
                ParameterMode.OUTPUT: "out",
            }[parameter.mode]
            shape = ", ".join(
                _mlir_string(str(item), context="shape extent")
                for item in tensor.shape
            )
            tensor_type = _mlir_string(
                self._logical_tensor_type(tuple(tensor.shape)),
                context="logical tensor type",
            )
            metadata.append(
                "{constraints = {alias = unit, alignment = unit, layout = unit, "
                "noalias = false, strides = unit}, kind = \"view\", "
                f"logical_ref = {_mlir_string(parameter.logical_ref, context='Kernel logical reference')}, "
                f"name = {name}, shape = [{shape}], type = {tensor_type}, "
                f'view_kind = "{view_kind}"}}'
            )
        return metadata

    def emit(self) -> IntentKernelModule:
        static_axes = tuple(
            axis
            for axis, extent in enumerate(self.projection.shape)
            if isinstance(extent, int)
        )
        runtime_axes = tuple(
            axis
            for axis, extent in enumerate(self.projection.shape)
            if isinstance(extent, str)
        )
        if not static_axes:
            _capability(
                "an all-runtime pointwise space is outside the current "
                "target-neutral export profile; a later target binding requires "
                "an explicit Layer-C partition plugin"
            )
        loop_axes = (*static_axes, *runtime_axes)
        domain_values, body_lines = self._emit_domains(level=2)
        body_lines.extend(
            self._emit_axis(
                depth=0,
                loop_axes=loop_axes,
                domain_values=tuple(domain_values),
                coordinates={},
                level=2,
            )
        )
        _, return_line = self._emit_effect_operation(
            name="return",
            operands=(),
            level=2,
        )
        body_lines.append(return_line)
        if self._binary_operation_id is None:
            raise AssertionError("pointwise binary operation was not emitted")

        arguments = ", ".join(
            f"{self.parameter_values[parameter.logical_ref].ref}: "
            f"{self.parameter_values[parameter.logical_ref].mlir_type}"
            for parameter in self.projected_parameters
        )
        parameter_nodes = ", ".join(
            f"{self.parameter_values[parameter.logical_ref].value_id} : i64"
            for parameter in self.projected_parameters
        )
        parameters = ", ".join(self._parameter_metadata())
        runtime_abi_parameters = tuple(
            {
                "name": parameter.name,
                "kind": parameter.kind.value,
                "mode": parameter.mode.value,
                "dtype": parameter.dtype,
                "rank": parameter.rank,
                "logical_ref": parameter.logical_ref,
            }
            for parameter in self.projected_parameters
        )
        view_refs = set(self._view_by_ref)
        scalar_refs = set(self._scalar_input_by_ref)
        dimension_refs = set(self.projection.dimension_references)
        parameter_ownership = tuple(
            {
                "parameter": parameter.name,
                "logical_ref": parameter.logical_ref,
                "owner": (
                    "j_tensor"
                    if parameter.logical_ref in view_refs
                    else "j_value_dimension_relation"
                    if parameter.logical_ref in scalar_refs
                    else "j_derived_dimension"
                    if parameter.logical_ref in dimension_refs
                    else "j"
                ),
            }
            for parameter in self.projected_parameters
        )
        handoff = KernelIRHandoff(
            entry_symbol=KERNEL_ENTRY_SYMBOL,
            j_digest=layer_c_content_digest(self.projection.j.to_json()),
            layer_c_interface_digest=self.projection.interface_digest,
            shape_relations=self.projection.shape_relations,
            dimension_relations=tuple(
                item.to_json() for item in self.projection.j.dimension_relations
            ),
            value_dimension_relations=tuple(
                item.to_json()
                for item in self.projection.j.value_dimension_relations
            ),
            runtime_abi_parameters=runtime_abi_parameters,
            parameter_ownership=parameter_ownership,
            result_relations=tuple(
                item.to_json()
                for item in self.projection.j.boundary.result_relations
            ),
            e_c_declarations=self.projection.e_c_declarations,
            contract_ir_digest=self.projection.contract.digest,
            dimension_constraints=self.projection.dimension_constraints,
            dimension_constraint_programs=self.projection.dimension_constraint_programs,
            contract_ir=self.projection.contract.to_json(),
        )
        shape_relation_attributes = ", ".join(
            self._shape_relation_attribute(item)
            for item in handoff.shape_relations
        )
        function_attributes = ", ".join(
            [
                'intent.kind = "kernel"',
                f"intent.parameter_nodes = [{parameter_nodes}]",
                f"intent.parameters = [{parameters}]",
                "intent.results = []",
                'intent.source = "<canonical-j>:1:1"',
            ]
        )
        lines = [
            "module attributes {"
            f'intent.source_module = "{KERNEL_ENTRY_SYMBOL}", '
            f'intent.entry_symbol = "{handoff.entry_symbol}", '
            f'intent.j_digest = "{handoff.j_digest}", '
            f'intent.layer_c_interface_digest = "{handoff.layer_c_interface_digest}", '
            f"intent.abi_shape_relations = [{shape_relation_attributes}], "
            f'intent.contract_ir_digest = "{handoff.contract_ir_digest}", '
            "intent.j_dimension_constraints = "
            + _mlir_json_string(
                list(handoff.dimension_constraints),
                context="J dimension constraints",
            )
            + ", intent.j_dimension_constraint_programs = "
            + _mlir_json_string(
                list(handoff.dimension_constraint_programs),
                context="typed J dimension constraints",
            )
            + ", intent.j_value_dimension_relations = "
            + _mlir_json_string(
                list(handoff.value_dimension_relations),
                context="J value dimension relations",
            )
            + ", intent.j_dimension_relations = "
            + _mlir_json_string(
                list(handoff.dimension_relations),
                context="J dimension relations",
            )
            + ", intent.e_c_declarations = "
            + _mlir_json_string(
                list(handoff.e_c_declarations),
                context="E_C declarations",
            )
            + ", intent.kernel_ir_contract = "
            + _mlir_json_string(
                handoff.contract_ir,
                context="Kernel contract IR",
            )
            + ", intent.kernel_ir_handoff = "
            + _mlir_json_string(handoff.to_dict(), context="Kernel IR handoff"),
            "} {",
            f"  func.func @{KERNEL_ENTRY_SYMBOL}({arguments}) attributes {{{function_attributes}}} {{",
            *body_lines,
            "  }",
            "}",
            "",
        ]

        consumed = [
            "j.intent.intent_function",
            "j.node_ids",
            "j.signature",
            "j.boundary.domains",
            "j.boundary.accesses",
            "j.boundary.effects",
            "j.boundary.result_relations",
            "j.scopes",
            "j.signature.dimension_parameter_bindings",
            "j.dimension_constraints",
            "j.dimension_relations",
            "j.value_dimension_relations",
            "j.contract_ir",
            "e_c.declarations",
        ]
        if self.projection.j.boundary.realization_authorizations:
            consumed.append("j.boundary.realization_authorizations")
        return IntentKernelModule(
            mlir="\n".join(lines),
            parameter_value_ids=tuple(
                (
                    parameter.name,
                    self.parameter_values[parameter.logical_ref].value_id,
                )
                for parameter in self.projected_parameters
            ),
            j_node_operation_ids=(
                (self.projection.operation.node_id, self._binary_operation_id),
            ),
            access_operation_ids=tuple(self._access_operation_ids),
            consumed_relations=tuple(consumed),
            handoff=handoff,
        )


class _RaggedKernelEmitter:
    """Serialize the validated direct Ragged query profile.

    The descriptor is an SSA tuple only inside the function.  Carrier tensors
    and query results remain ordinary logical tensor values at this boundary;
    the real C++ CUDA pass converts the exact same graph to its explicit
    destination-passing ABI, while the common pass consumes it as tensor IR.
    """

    _OPERATIONS = frozenset({"ragged", "ragged_outer", "ragged_member", "return"})

    def __init__(self, projection: _RaggedProjection) -> None:
        self.projection = projection
        self._next_value_id = 0
        self._next_operation_id = 0
        self._node_operation_ids: list[tuple[str, int]] = []
        self._access_operation_ids: list[tuple[str, int]] = []
        self._input_by_ref = {item.name: item for item in projection.input_tensors}
        self._output_by_ref = {item.name: item for item in projection.output_tensors}
        self._output_value_by_ref = dict(projection.output_bindings)
        self._output_ref_by_value = {
            value: output for output, value in projection.output_bindings
        }
        self._tensor_by_ref = {**self._input_by_ref, **self._output_by_ref}
        self._input_values: dict[str, _KernelValue] = {}
        self._result_values: dict[str, _KernelValue] = {}
        for parameter in projection.input_parameters:
            tensor = self._input_by_ref[parameter.logical_ref]
            mlir_type = _kernel_ragged_tensor_type(
                tuple(tensor.shape), tensor.dtype
            )
            self._input_values[parameter.logical_ref] = self._new_value(
                mlir_type, mlir_type
            )

        self._parameter_by_ref = {
            parameter.logical_ref: parameter
            for parameter in projection.projected_parameters
        }

    def _value_for_logical_ref(self, logical_ref: str) -> _KernelValue:
        value = self._input_values.get(logical_ref)
        if value is None:
            result_ref = self._output_value_by_ref.get(logical_ref)
            if result_ref is not None:
                value = self._result_values.get(result_ref)
        if value is None:
            raise AssertionError(
                f"no emitted Kernel value is bound to logical reference {logical_ref!r}"
            )
        return value

    def _new_value(self, mlir_type: str, metadata_type: str) -> _KernelValue:
        value = _KernelValue(self._next_value_id, mlir_type, metadata_type)
        self._next_value_id += 1
        return value

    def _new_operation_id(self) -> int:
        operation_id = self._next_operation_id
        self._next_operation_id += 1
        return operation_id

    def _common_attributes(
        self,
        operation_id: int,
        results: tuple[_KernelValue, ...],
        result_names: tuple[str, ...],
    ) -> list[str]:
        if len(results) != len(result_names):
            raise AssertionError("Ragged result metadata is not positional")
        names = ", ".join(
            _mlir_string(name, context="Ragged result name")
            for name in result_names
        )
        nodes = ", ".join(f"{value.value_id} : i64" for value in results)
        types = ", ".join(
            _mlir_string(value.metadata_type, context="Ragged result type")
            for value in results
        )
        return [
            f"intent.node = {operation_id} : i64",
            f"intent.result_names = [{names}]",
            f"intent.result_nodes = [{nodes}]",
            f"intent.result_types = [{types}]",
        ]

    def _operation_line(
        self,
        *,
        name: str,
        operation_id: int,
        operands: tuple[_KernelValue, ...] = (),
        results: tuple[_KernelValue, ...] = (),
        result_names: tuple[str, ...] = (),
        trailing_attributes: tuple[str, ...] = (),
        level: int,
    ) -> str:
        if name not in self._OPERATIONS:
            raise AssertionError(f"operation {name!r} is outside the Ragged profile")
        if len(results) > 1:
            raise AssertionError("a Ragged query has at most one result")
        attributes = ", ".join(
            [
                *self._common_attributes(operation_id, results, result_names),
                *trailing_attributes,
            ]
        )
        operand_refs = ", ".join(value.ref for value in operands)
        operand_types = ", ".join(value.mlir_type for value in operands)
        result_type = results[0].mlir_type if results else "()"
        assignment = f"{results[0].ref} = " if results else ""
        return (
            f'{"  " * level}{assignment}"intent.{name}"({operand_refs}) '
            f"{{{attributes}}} : ({operand_types}) -> {result_type} "
            f"{_MLIR_LOCATION}"
        )

    def _emit_result(
        self,
        *,
        name: str,
        operands: tuple[_KernelValue, ...],
        result_type: str,
        result_name: str,
        level: int,
    ) -> tuple[int, _KernelValue, str]:
        operation_id = self._new_operation_id()
        result = self._new_value(result_type, result_type)
        line = self._operation_line(
            name=name,
            operation_id=operation_id,
            operands=operands,
            results=(result,),
            result_names=(result_name,),
            level=level,
        )
        return operation_id, result, line

    def _emit_return(
        self, operands: tuple[_KernelValue, ...], *, level: int
    ) -> tuple[int, str]:
        operation_id = self._new_operation_id()
        return operation_id, self._operation_line(
            name="return",
            operation_id=operation_id,
            operands=operands,
            level=level,
        )

    def _parameter_metadata(self) -> list[str]:
        metadata: list[str] = []
        for parameter in self.projection.projected_parameters:
            tensor = self._tensor_by_ref[parameter.logical_ref]
            kind = "input_tensor" if parameter in self.projection.input_parameters else "output_tensor"
            shape = ", ".join(
                _mlir_string(str(item), context="Ragged shape extent")
                for item in tensor.shape
            )
            metadata.append(
                "{kind = "
                + _mlir_string(kind, context="Ragged parameter kind")
                + ", logical_ref = "
                + _mlir_string(parameter.logical_ref, context="Ragged logical reference")
                + ", mode = "
                + _mlir_string(parameter.mode.value, context="Ragged parameter mode")
                + ", name = "
                + _mlir_string(parameter.name, context="Ragged parameter name")
                + f", rank = {parameter.rank} : i64, shape = [{shape}], type = "
                + _mlir_string(
                    _kernel_ragged_tensor_type(tuple(tensor.shape), tensor.dtype),
                    context="Ragged parameter type",
                )
                + "}"
            )
        return metadata

    def emit(self) -> IntentKernelModule:
        producer = self.projection.producer
        offsets = self._input_values[producer.operands[0]]
        indices = self._input_values[producer.operands[1]]
        offsets_tensor = self._input_by_ref[producer.operands[0]]
        indices_tensor = self._input_by_ref[producer.operands[1]]
        descriptor_type = _kernel_ragged_descriptor_type(offsets_tensor, indices_tensor)
        producer_id, descriptor, producer_line = self._emit_result(
            name="ragged",
            operands=(offsets, indices),
            result_type=descriptor_type,
            result_name=producer.result,
            level=2,
        )
        self._result_values[producer.result] = descriptor
        self._node_operation_ids.append((producer.node_id, producer_id))
        lines = [producer_line]

        for query in self.projection.queries:
            operands = [descriptor]
            if query.op == "ragged_member":
                operands.append(self._input_values[query.operands[1]])
            output_ref = self._output_ref_by_value[query.result]
            output_tensor = self._output_by_ref[output_ref]
            result_type = _kernel_ragged_tensor_type(
                tuple(output_tensor.shape), output_tensor.dtype
            )
            operation_id, result, line = self._emit_result(
                name=query.op,
                operands=tuple(operands),
                result_type=result_type,
                result_name=query.result,
                level=2,
            )
            self._result_values[query.result] = result
            self._node_operation_ids.append((query.node_id, operation_id))
            lines.append(line)

        return_values = tuple(
            self._result_values[self._output_value_by_ref[output.name]]
            for output in self.projection.output_tensors
        )
        return_id, return_line = self._emit_return(return_values, level=2)
        lines.append(return_line)

        node_to_operation = dict(self._node_operation_ids)
        self._access_operation_ids.extend(
            (access.access_id, node_to_operation[access.node_ref])
            for access in self.projection.contract.accesses
            if access.node_ref in node_to_operation
        )

        result_types = tuple(value.mlir_type for value in return_values)
        function_result_clause = " -> (" + ", ".join(result_types) + ")"
        arguments = ", ".join(
            f"{self._input_values[parameter.logical_ref].ref}: "
            f"{self._input_values[parameter.logical_ref].mlir_type}"
            for parameter in self.projection.input_parameters
        )
        parameter_nodes = ", ".join(
            f"{self._value_for_logical_ref(parameter.logical_ref).value_id} : i64"
            for parameter in self.projection.projected_parameters
        )
        result_metadata = ", ".join(
            "{name = "
            + _mlir_string(output.name, context="Ragged output name")
            + ", type = "
            + _mlir_string(
                _kernel_ragged_tensor_type(tuple(output.shape), output.dtype),
                context="Ragged output type",
            )
            + "}"
            for output in self.projection.output_tensors
        )
        handoff = KernelIRHandoff(
            entry_symbol=KERNEL_ENTRY_SYMBOL,
            j_digest=layer_c_content_digest(self.projection.j.to_json()),
            layer_c_interface_digest=self.projection.interface_digest,
            shape_relations=(),
            dimension_relations=tuple(
                item.to_json() for item in self.projection.j.dimension_relations
            ),
            value_dimension_relations=tuple(
                item.to_json()
                for item in self.projection.j.value_dimension_relations
            ),
            runtime_abi_parameters=tuple(
                {
                    "name": parameter.name,
                    "kind": parameter.kind.value,
                    "mode": parameter.mode.value,
                    "dtype": parameter.dtype,
                    "rank": parameter.rank,
                    "logical_ref": parameter.logical_ref,
                    "binding": (
                        "argument"
                        if parameter.logical_ref in self._input_by_ref
                        else "result"
                    ),
                }
                for parameter in self.projection.projected_parameters
            ),
            parameter_ownership=tuple(
                {
                    "parameter": parameter.name,
                    "logical_ref": parameter.logical_ref,
                    "owner": (
                        "j_tensor_input"
                        if parameter.logical_ref in self._input_by_ref
                        else "j_tensor_output"
                    ),
                }
                for parameter in self.projection.projected_parameters
            ),
            result_relations=tuple(
                item.to_json()
                for item in self.projection.j.boundary.result_relations
            ),
            e_c_declarations=self.projection.e_c_declarations,
            contract_ir_digest=self.projection.contract.digest,
            dimension_constraints=(),
            dimension_constraint_programs=(),
            contract_ir=self.projection.contract.to_json(),
        )
        function_attributes = ", ".join(
            [
                'intent.kind = "kernel"',
                f"intent.parameter_nodes = [{parameter_nodes}]",
                f"intent.parameters = [{', '.join(self._parameter_metadata())}]",
                f"intent.results = [{result_metadata}]",
                'intent.source = "<canonical-j>:1:1"',
            ]
        )
        lines = [
            "module attributes {"
            f'intent.source_module = "{KERNEL_ENTRY_SYMBOL}", '
            f'intent.entry_symbol = "{handoff.entry_symbol}", '
            'intent.kernel_ir_profile = "ragged_query", '
            f'intent.j_digest = "{handoff.j_digest}", '
            f'intent.layer_c_interface_digest = "{handoff.layer_c_interface_digest}", '
            "intent.j_dimension_constraints = "
            + _mlir_json_string([], context="Ragged J dimension constraints")
            + ", intent.j_dimension_constraint_programs = "
            + _mlir_json_string([], context="Ragged typed J dimension constraints")
            + ", intent.j_value_dimension_relations = "
            + _mlir_json_string(
                list(handoff.value_dimension_relations),
                context="Ragged J value dimension relations",
            )
            + ", intent.j_dimension_relations = "
            + _mlir_json_string(
                list(handoff.dimension_relations),
                context="Ragged J dimension relations",
            )
            + ", intent.e_c_declarations = "
            + _mlir_json_string(
                list(handoff.e_c_declarations), context="Ragged E_C declarations"
            )
            + ", intent.kernel_ir_contract = "
            + _mlir_json_string(handoff.contract_ir, context="Ragged Kernel contract IR")
            + ", intent.kernel_ir_handoff = "
            + _mlir_json_string(handoff.to_dict(), context="Ragged Kernel IR handoff")
            + "} {",
            f"  func.func @{KERNEL_ENTRY_SYMBOL}({arguments}){function_result_clause} "
            f"attributes {{{function_attributes}}} {{",
            *lines,
            "  }",
            "}",
            "",
        ]
        consumed = [
            "j.intent.intent_function",
            "j.node_ids",
            "j.signature",
            "j.boundary.accesses",
            "j.boundary.effects",
            "j.boundary.result_relations",
            "j.scopes",
            "j.ragged.value_graph",
            "j.contract_ir",
            "e_c.declarations",
        ]
        if self.projection.j.boundary.realization_authorizations:
            consumed.append("j.boundary.realization_authorizations")
        return IntentKernelModule(
            mlir="\n".join(lines),
            parameter_value_ids=tuple(
                (
                    parameter.name,
                    self._value_for_logical_ref(parameter.logical_ref).value_id,
                )
                for parameter in self.projection.projected_parameters
            ),
            j_node_operation_ids=tuple(self._node_operation_ids),
            access_operation_ids=tuple(self._access_operation_ids),
            consumed_relations=tuple(consumed),
            handoff=handoff,
        )


def _emit_pointwise_kernel(projection: _PointwiseProjection) -> IntentKernelModule:
    """Serialize one validated projection without any external implementation."""

    return _PointwiseKernelEmitter(projection).emit()


def _emit_ragged_kernel(projection: _RaggedProjection) -> IntentKernelModule:
    """Serialize one validated Ragged projection without target realization."""

    return _RaggedKernelEmitter(projection).emit()


def export_intent_kernel_mlir(
    j: SemanticArchive | Mapping[str, Any],
    extension_interface: LayerCInterface | Mapping[str, Any] | None = None,
    *,
    bound_extension_values: Mapping[str, object] | None = None,
) -> IntentKernelModule:
    """Project one canonical J into one target-neutral Intent Kernel module.

    The serializer is owned entirely by TianchenIR and has no external builder,
    package, source-tree, or sibling-path dependency.  It dispatches between
    the closed dense pointwise and direct Ragged query profiles; both require
    an empty Layer-C interface.  Unsupported semantic relations raise
    :class:`IntentKernelCapabilityError`; the exporter never guesses a target
    plan or Layer-C actual value.
    """

    archive = j if isinstance(j, SemanticArchive) else SemanticArchive.from_json(dict(j))
    if extension_interface is None:
        interface = LayerCInterface()
    elif isinstance(extension_interface, LayerCInterface):
        interface = extension_interface
    else:
        interface = LayerCInterface.from_dict(extension_interface)
    body = tuple(archive.intent.body)
    if any(operation.op in RAGGED_OPS for operation in walk_intent_ops(body)):
        projection = _validate_closed_ragged_profile(
            archive,
            interface,
            bound_extension_values,
        )
        return _emit_ragged_kernel(projection)

    projection = _validate_closed_pointwise_profile(
        archive,
        interface,
        bound_extension_values,
    )
    return _emit_pointwise_kernel(projection)


__all__ = [
    "IntentKernelCapabilityError",
    "IntentKernelModule",
    "KERNEL_ENTRY_SYMBOL",
    "KernelABIShapeRelation",
    "KernelIRHandoff",
    "export_intent_kernel_mlir",
]
