"""Target-neutral deterministic J materialization and erasable C attachment."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from intentir.backend.protocol import BackendTarget
from intentir.frontend.canonical_j import (
    DYNAMIC_VALUE_LOOP_MATERIALIZATION,
    LOGICAL_FULL_AXIS_STEP_MATERIALIZATION,
    STRUCTURED_SCAN_LOOP_MATERIALIZATION,
    materialize_j_function,
)
from intentir.frontend.emit_mlir import emit_module
from intentir.ir.canonical_j import elaborate_j, j_elaboration
from intentir.ir.dense_predicate_materialization import (
    BoundaryMaterializationProjection,
    materialize_dense_load_predicates,
)
from intentir.ir.typed_boundary_materialization import (
    TypedBoundaryMaterializationProjection,
    materialize_typed_boundary,
)
from intentir.ir.contract_materialization import (
    ContractMaterializationProjection,
    project_contract_materialization,
)
from intentir.ir.layer_c import (
    canonical_layer_c_json,
    layer_c_content_digest,
)
from intentir.ir.model import IntentFunction, IntentOp
from intentir.ir.paired_rotation import derive_paired_rotation_view
from intentir.ir.value_dimension import resolve_value_dimension_equalities
from intentir.verify.gate1.model import (
    ImplementationConfig,
    ParameterKind,
    ParameterMode,
    SemanticArchive,
)


MATERIALIZER_VERSION = "intentir.production.materializer.v2"
ATTACH_C_VERSION = "intentir.production.attach-c.v1"


class MaterializationError(ValueError):
    pass


@dataclass(frozen=True)
class MaterializedJ:
    """Immutable M0 plus the target-neutral Layer-B materialization."""

    j: SemanticArchive
    intent_mlir: str
    concrete_extents: tuple[tuple[str, int], ...]
    boundary_projection: (
        ContractMaterializationProjection
        | BoundaryMaterializationProjection
        | TypedBoundaryMaterializationProjection
    )
    form: str
    consumed_structures: tuple[str, ...]
    constexpr_bindings: tuple[tuple[str, bool | int | float], ...] = ()
    materializer_version: str = MATERIALIZER_VERSION

    def __post_init__(self) -> None:
        if not self.intent_mlir.strip():
            raise MaterializationError("materialization produced empty Intent MLIR")
        if self.boundary_projection.runtime_predicate_proofs:
            raise MaterializationError(
                "production materialization cannot consume runtime proof records"
            )

    @property
    def extents(self) -> dict[str, int]:
        return dict(self.concrete_extents)

    @property
    def concrete_extents_map(self) -> dict[str, int]:
        """Return the concrete J dimensions for backend lowering callers.

        ``MaterializedJ`` is the producer-side counterpart of the lowering
        result objects, which expose the same mapping under
        ``concrete_extents_map``.  Keep both names as explicit public views of
        the immutable tuple so CUDA/RISC-V consumers cannot accidentally
        depend on a non-existent attribute or on mutable internal state.
        """

        return dict(self.concrete_extents)

    @property
    def runtime_input_names(self) -> tuple[str, ...]:
        specialized = {name for name, _value in self.constexpr_bindings}
        return tuple(
            item.name for item in self.j.intent.inputs if item.name not in specialized
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "materializer_version": self.materializer_version,
            "j_digest": layer_c_content_digest(self.j.to_json()),
            "materialization_form": self.form,
            "concrete_extents": self.extents,
            "constexpr_bindings": dict(self.constexpr_bindings),
            "runtime_input_names": list(self.runtime_input_names),
            "boundary_projection": self.boundary_projection.to_json(),
            "consumed_structures": list(self.consumed_structures),
        }


@dataclass(frozen=True)
class CPrior:
    slot: str
    role: str
    value: bool | int | float | str

    def to_record(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "role": self.role,
            "value": self.value,
        }


@dataclass(frozen=True)
class BackendSearchState:
    """S_C with M0 unchanged and C represented only as discardable priors."""

    base: MaterializedJ
    target: BackendTarget
    priors: tuple[CPrior, ...]
    ignored_slots: tuple[tuple[str, str], ...]
    attach_version: str = ATTACH_C_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.base, MaterializedJ):
            raise MaterializationError(
                "BackendSearchState requires a MaterializedJ base"
            )
        if self.base.form == STRUCTURED_SCAN_LOOP_MATERIALIZATION or any(
            ":structured_scan_tile:" in record
            for record in self.base.consumed_structures
        ):
            raise MaterializationError(
                "BackendSearchState.base must be target-neutral M0; "
                "use materialize_target_lowering for a temporary target IR"
            )
        if not isinstance(self.target, BackendTarget):
            raise MaterializationError(
                "BackendSearchState requires a BackendTarget"
            )

    @property
    def base_ir(self) -> str:
        return self.base.intent_mlir

    def erase_c(self) -> str:
        return self.base.intent_mlir

    def to_record(self) -> dict[str, Any]:
        return {
            "attach_version": self.attach_version,
            "j_digest": layer_c_content_digest(self.base.j.to_json()),
            "target": self.target.to_dict(),
            "priors": [item.to_record() for item in self.priors],
            "ignored_slots": [
                {"slot": slot, "reason": reason}
                for slot, reason in self.ignored_slots
            ],
            "erase_c_matches_base": self.erase_c() == self.base_ir,
        }


def _specialize_constexpr_inputs(
    j: SemanticArchive,
    function: IntentFunction,
    bindings: Mapping[str, bool | int | float],
) -> tuple[IntentFunction, tuple[str, ...]]:
    if not bindings:
        return function, ()
    value_types = j_elaboration(j).value_type_by_name
    tensors = {item.name: item for item in j.intent.inputs}
    allowed = {
        parameter.logical_ref
        for parameter in j.signature.parameters
        if parameter.kind is ParameterKind.CONSTEXPR
        and parameter.logical_ref in tensors
    }
    unknown = sorted(set(bindings) - allowed)
    if unknown:
        raise MaterializationError(
            "constexpr specialization is not authorized by J.signature: "
            + ", ".join(unknown)
        )
    constants: list[IntentOp] = []
    for name, value in sorted(bindings.items()):
        tensor = tensors[name]
        if tensor.shape:
            raise MaterializationError(
                f"constexpr logical input {name!r} must have rank zero"
            )
        dtype = tensor.dtype.removeprefix("torch.").lower()
        if dtype in {"bool", "i1"}:
            valid = isinstance(value, bool)
        elif dtype in {
            "i8", "i16", "i32", "i64", "u8", "u16", "u32",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32",
        }:
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid:
            raise MaterializationError(
                f"constexpr logical input {name!r} value is incompatible with {tensor.dtype}"
            )
        constants.append(
            IntentOp(
                op="const",
                result=name,
                attrs={"value": value, "dtype": tensor.dtype},
                type={"dtype": tensor.dtype, "shape": []},
            )
        )

    consumed_control: list[str] = []

    def specialize_body(body: list[IntentOp]) -> list[IntentOp]:
        specialized: list[IntentOp] = []
        for original in body:
            operation = copy.deepcopy(original)
            operation.body = specialize_body(operation.body)
            for arm in (operation.positive, operation.negative):
                if arm is not None:
                    arm.body = specialize_body(arm.body)
            if (
                operation.op in {"effect_if", "value_if"}
                and operation.predicate in bindings
            ):
                condition = bindings[operation.predicate]
                if not isinstance(condition, bool):
                    raise MaterializationError(
                        "effect_if constexpr predicate must be boolean"
                    )
                arm = operation.positive if condition else operation.negative
                if arm is None:
                    raise MaterializationError(
                        "effect_if constexpr specialization requires both arms"
                    )
                consumed_control.append(
                    f"{operation.op}:"
                    f"{operation.node_id or 'anonymous'}:"
                    f"constexpr_{str(condition).lower()}"
                )
                specialized.extend(arm.body)
                if operation.op == "value_if":
                    result_type = (
                        value_types.get(operation.result)
                        if isinstance(operation.result, str)
                        else None
                    )
                    if (
                        not isinstance(operation.result, str)
                        or not isinstance(arm.yield_value, str)
                        or result_type is None
                        or not result_type.is_tensor
                        or result_type.dtype is None
                    ):
                        raise MaterializationError(
                            "value_if specialization requires one typed yield"
                        )
                    if arm.yield_value != operation.result:
                        specialized.append(
                            IntentOp(
                                op="cast",
                                result=operation.result,
                                operands=[arm.yield_value],
                                attrs={
                                    "to_dtype": result_type.dtype,
                                },
                                type=result_type.to_json(),
                            )
                        )
                continue
            specialized.append(operation)
        return specialized

    return IntentFunction(
        name=function.name,
        inputs=[item for item in function.inputs if item.name not in bindings],
        outputs=list(function.outputs),
        body=[*constants, *specialize_body(function.body)],
        returns=list(function.returns),
    ), tuple(consumed_control)


def _project_value_dimension_equalities(
    j: SemanticArchive,
    extents: Mapping[str, int],
    function: IntentFunction,
    bindings: Mapping[str, bool | int | float],
) -> tuple[tuple[int, int], ...]:
    """Validate folded equalities or remap them to the specialized ABI."""

    resolved = resolve_value_dimension_equalities(j, extents)
    original_inputs = list(j.intent.inputs)
    runtime_indices = {
        item.name: index for index, item in enumerate(function.inputs)
    }
    projected: list[tuple[int, int]] = []
    for original_index, expected in resolved:
        name = original_inputs[original_index].name
        if name in bindings:
            actual = bindings[name]
            if (
                isinstance(actual, bool)
                or not isinstance(actual, int)
                or actual != expected
            ):
                raise MaterializationError(
                    "constexpr specialization violates J value/dimension "
                    f"equality: {name}={actual!r}, expected {expected}"
                )
            continue
        if name not in runtime_indices:
            raise MaterializationError(
                f"value/dimension equality input {name!r} vanished from ABI"
            )
        projected.append((runtime_indices[name], expected))
    return tuple(projected)


def _project_inplace_outputs(
    j: SemanticArchive, function: IntentFunction
) -> tuple[int, ...]:
    """Map J-owned inout results back to their unchanged runtime inputs."""

    inout_refs = {
        item.logical_ref
        for item in j.signature.parameters
        if item.mode is ParameterMode.INOUT and item.logical_ref is not None
    }
    positions = {item.name: index for index, item in enumerate(function.inputs)}
    output_by_name = {item.name: item for item in function.outputs}
    output_bindings = j_elaboration(j).output_bindings
    if tuple(item.output for item in output_bindings) != tuple(
        item.name for item in function.outputs
    ) or set(output_by_name) != {item.output for item in output_bindings}:
        raise MaterializationError(
            "canonical output binding order differs from materialized function"
        )
    projected = tuple(
        positions.get(binding.output, -1)
        if binding.output in inout_refs
        else -1
        for binding in output_bindings
    )
    return projected if any(index >= 0 for index in projected) else ()


def _materialize_j_impl(
    j: SemanticArchive,
    concrete_extents: Mapping[str, int],
    *,
    constexpr_bindings: Mapping[str, bool | int | float] | None = None,
    loop_schedule_tile: int | None = None,
) -> MaterializedJ:
    if not isinstance(j, SemanticArchive):
        raise MaterializationError("materialize_j requires canonical J")
    extents: dict[str, int] = {}
    for name, value in concrete_extents.items():
        if not isinstance(name, str) or not name:
            raise MaterializationError("concrete extent names must be non-empty")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MaterializationError(
                f"concrete extent {name!r} must be a non-negative integer"
            )
        extents[name] = value
    canonical = elaborate_j(j)
    scalar_bindings = dict(constexpr_bindings or {})
    try:
        logical = materialize_j_function(
            canonical,
            extents,
            loop_schedule_tile=loop_schedule_tile,
        )
        typed_boundary = materialize_typed_boundary(
            canonical,
            logical.function,
            extents,
            consume_loop_storage=(
                logical.form
                in {
                    DYNAMIC_VALUE_LOOP_MATERIALIZATION,
                    LOGICAL_FULL_AXIS_STEP_MATERIALIZATION,
                    STRUCTURED_SCAN_LOOP_MATERIALIZATION,
                }
            ),
        )
        predicated = None
        if typed_boundary is not None:
            projection = typed_boundary.projection
            function = typed_boundary.function
        else:
            predicated = materialize_dense_load_predicates(
                canonical,
                logical.function,
            )
        if typed_boundary is None and predicated is not None:
            projection = predicated.projection
            function = predicated.function
        elif typed_boundary is None:
            projection = project_contract_materialization(
                canonical,
                consume_loop_storage=(
                    logical.form
                    in {
                        DYNAMIC_VALUE_LOOP_MATERIALIZATION,
                        LOGICAL_FULL_AXIS_STEP_MATERIALIZATION,
                        STRUCTURED_SCAN_LOOP_MATERIALIZATION,
                    }
                ),
            )
            function = logical.function
        function, consumed_control = _specialize_constexpr_inputs(
            canonical,
            function,
            scalar_bindings,
        )
        value_dimension_equalities = _project_value_dimension_equalities(
            canonical,
            extents,
            function,
            scalar_bindings,
        )
        inplace_outputs = _project_inplace_outputs(canonical, function)
        paired_rotation_view = derive_paired_rotation_view(
            canonical,
            extents,
            scalar_bindings,
        )
        paired_rotation = (
            None
            if paired_rotation_view is None
            else (
                paired_rotation_view.layout,
                paired_rotation_view.input_indices(function),
                paired_rotation_view.sections,
            )
        )
        output_bindings = j_elaboration(canonical).output_bindings
        mlir = emit_module(
            function,
            extents,
            broadcast_axes=projection.emitter_broadcast_axes(),
            proven_index_bounds=projection.proven_index_bounds,
            value_dimension_equalities=value_dimension_equalities,
            output_values={
                binding.output: binding.value
                for binding in output_bindings
            },
            canonical_output_bindings=output_bindings,
            inplace_outputs=inplace_outputs,
            paired_rotation=paired_rotation,
            loop_schedule_tile=logical.loop_schedule_tile,
        )
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise MaterializationError(
            f"deterministic J materialization failed: {type(exc).__name__}: {exc}"
        ) from exc
    return MaterializedJ(
        j=canonical,
        intent_mlir=mlir,
        concrete_extents=tuple(sorted(extents.items())),
        boundary_projection=projection,
        form=logical.form,
        consumed_structures=(
            *logical.consumed_structures,
            *consumed_control,
        ),
        constexpr_bindings=tuple(sorted(scalar_bindings.items())),
    )


def materialize_j(
    j: SemanticArchive,
    concrete_extents: Mapping[str, int],
    *,
    constexpr_bindings: Mapping[str, bool | int | float] | None = None,
) -> MaterializedJ:
    """Materialize canonical J into target-neutral M0.

    This is the only public J-to-materialization entry point.  Target schedule
    values are deliberately not accepted here: a CUDA/RISC-V plugin is bound
    only after this immutable neutral artifact has become the search base.
    Use :func:`materialize_target_lowering` for a temporary target-specific
    lowering view, never as the ``BackendSearchState.base``.
    """

    return _materialize_j_impl(
        j,
        concrete_extents,
        constexpr_bindings=constexpr_bindings,
    )


def materialize_target_lowering(
    base: MaterializedJ,
    *,
    loop_schedule_tile: int,
) -> MaterializedJ:
    """Create an ephemeral target lowering view from neutral M0.

    The target-specific tile is applied only after a target plugin has been
    validated.  The returned object is intentionally not accepted as a search
    base; it is a compile-time source view whose canonical J, concrete extents,
    and semantic constexpr bindings must remain identical to ``base``.
    """

    if not isinstance(base, MaterializedJ):
        raise MaterializationError(
            "target lowering materialization requires one MaterializedJ base"
        )
    if (
        isinstance(loop_schedule_tile, bool)
        or not isinstance(loop_schedule_tile, int)
        or loop_schedule_tile <= 0
    ):
        raise MaterializationError(
            "target lowering loop_schedule_tile must be a positive integer"
        )
    if base.form == STRUCTURED_SCAN_LOOP_MATERIALIZATION or any(
        ":structured_scan_tile:" in record
        for record in base.consumed_structures
    ):
        raise MaterializationError(
            "target lowering materialization requires neutral M0, not an "
            "already specialized structured-scan view"
        )
    try:
        lowered = _materialize_j_impl(
            base.j,
            base.extents,
            constexpr_bindings=dict(base.constexpr_bindings),
            loop_schedule_tile=loop_schedule_tile,
        )
    except MaterializationError:
        raise
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise MaterializationError(
            "target-specific materialization failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    base_j_json = canonical_layer_c_json(base.j.to_json())
    lowered_j_json = canonical_layer_c_json(lowered.j.to_json())
    base_j_digest = layer_c_content_digest(base.j.to_json())
    lowered_j_digest = layer_c_content_digest(lowered.j.to_json())
    if base_j_json != lowered_j_json or base_j_digest != lowered_j_digest:
        raise MaterializationError(
            "target-specific materialization changed canonical J bytes/digest"
        )
    if lowered.concrete_extents != base.concrete_extents:
        raise MaterializationError(
            "target-specific materialization changed concrete extents"
        )
    if lowered.constexpr_bindings != base.constexpr_bindings:
        raise MaterializationError(
            "target-specific materialization changed semantic constexpr bindings"
        )
    if lowered.form != STRUCTURED_SCAN_LOOP_MATERIALIZATION:
        raise MaterializationError(
            "target loop tile did not produce the registered structured-scan "
            "lowering form"
        )
    return lowered


def attach_c(
    base: MaterializedJ,
    c: ImplementationConfig | None,
    target: BackendTarget,
) -> BackendSearchState:
    """Attach legacy Source C as erasable search priors after M0.

    This compatibility path does not create a target Layer-C plugin and must
    never be used as the production lowering binding boundary.  Active target
    plugins are validated separately and are not reduced to ``CPrior``.
    """

    if not isinstance(base, MaterializedJ):
        raise MaterializationError("Attach_C requires one MaterializedJ base")
    if not isinstance(target, BackendTarget):
        raise MaterializationError("Attach_C requires BackendTarget")
    if c is None:
        return BackendSearchState(base, target, (), ())
    if not isinstance(c, ImplementationConfig):
        raise MaterializationError("Attach_C requires canonical Source C")
    try:
        c.validate_complete_values(path="Attach_C.SourceC")
    except (TypeError, ValueError) as exc:
        raise MaterializationError(str(exc)) from exc
    priors: list[CPrior] = []
    ignored: list[tuple[str, str]] = []
    for slot in c.slots:
        # ``validate_complete_values`` above makes absence/type errors fatal.
        # The remaining ignored list is retained only for record compatibility.
        assert slot.name in c.values
        value = c.values[slot.name]
        if not isinstance(value, (bool, int, float, str)):
            raise MaterializationError(
                f"Source C slot {slot.name!r} is not a scalar search prior"
            )
        priors.append(
            CPrior(
                slot=slot.name,
                role=slot.role,
                value=value,
            )
        )
    state = BackendSearchState(
        base=base,
        target=target,
        priors=tuple(priors),
        ignored_slots=tuple(ignored),
    )
    if state.erase_c() != base.intent_mlir:
        raise AssertionError("Erase_C invariant violated")
    return state


__all__ = [
    "ATTACH_C_VERSION",
    "MATERIALIZER_VERSION",
    "BackendSearchState",
    "CPrior",
    "MaterializationError",
    "MaterializedJ",
    "attach_c",
    "materialize_j",
    "materialize_target_lowering",
]
