"""Fail-closed canonical-J lowering boundary for the CUDA backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from intentir.backend.protocol import (
    BackendConformanceRecord,
    BackendTarget,
    ComparisonEvidence,
    LoweringRequest,
    TargetBindingContext,
    TargetPluginArtifact,
    deserialize_target_artifact,
    serialize_target_artifact,
)
from intentir.backend.contract_updates import (
    consume_exact_internal_overwrite_update,
)
from intentir.backend.materialize import (
    BackendSearchState,
    MaterializedJ,
    materialize_j,
    materialize_target_lowering,
)
from intentir.frontend.canonical_j import (
    DYNAMIC_VALUE_LOOP_MATERIALIZATION,
    LOGICAL_FULL_AXIS_STEP_MATERIALIZATION,
    STRUCTURED_SCAN_LOOP_MATERIALIZATION,
)
from intentir.ir.dense_predicate_materialization import (
    BoundaryMaterializationProjection,
)
from intentir.ir.canonical_j import elaborate_j, j_elaboration
from intentir.ir.layer_c import (
    LayerCInterface,
    binding_interface_digest,
    canonical_layer_c_json,
    layer_c_content_digest,
)
from intentir.ir.contract_materialization import (
    ContractMaterializationProjection,
    project_contract_materialization,
)
from intentir.ir.typed_boundary_materialization import (
    TypedBoundaryMaterializationProjection,
)
from intentir.ir.model import IntentOp, intent_op_regions
from intentir.ir.value_dimension import (
    validate_ragged_runtime_invocation,
    validate_value_dimension_instantiation,
)
from intentir.verify.gate1.model import (
    DuplicatePolicy,
    ResultRelationKind,
    SemanticArchive,
    UpdateCombiner,
    UpdateOrder,
)

from .pipeline import LoweredProgram, execute_cuda_artifact, lower_to_cuda
from .selection import LoweringTarget


class CudaJLoweringError(ValueError):
    pass


class CudaJExecutionError(ValueError):
    pass


def _cuda_cubin_bundle(program: LoweredProgram) -> bytes:
    """Frame the actual compiler-produced CUDA binaries for evidence.

    This is an evidence container, not a compiler input.  It includes the
    names, lowered binary MLIR, and cubin bytes returned by the real CUDA
    compiler so a target-record application cannot be represented by an
    arbitrary caller-authored digest.
    """

    if not isinstance(program, LoweredProgram):
        raise CudaJLoweringError(
            "CUDA target realization evidence requires a LoweredProgram"
        )
    payload = bytearray(b"intentir-cuda-cubin-bundle\0")
    payload.extend(len(program.kernels).to_bytes(4, "big", signed=False))
    for kernel in program.kernels:
        for value in (
            kernel.binary_name.encode("utf-8"),
            kernel.kernel_name.encode("utf-8"),
            kernel.binary_mlir.encode("utf-8"),
            bytes(kernel.cubin),
        ):
            payload.extend(len(value).to_bytes(8, "big", signed=False))
            payload.extend(value)
    if len(payload) <= len(b"intentir-cuda-cubin-bundle\0") + 4:
        raise CudaJLoweringError("CUDA compiler produced an empty cubin bundle")
    return bytes(payload)


def _deserialize_cuda_target_artifact(
    request: LoweringRequest,
    context: TargetBindingContext,
) -> TargetPluginArtifact:
    """Reparse the complete ``E_C``/``C_target`` handoff at CUDA ingress."""

    if request.c is None:
        raise CudaJLoweringError(
            "CUDA target-artifact ingress requires a target Layer-C plugin"
        )
    # A plugin-only request is compatible only when the shared E_C is empty.
    # For a non-empty interface the producer must have crossed the explicit
    # TargetPluginArtifact boundary; reconstructing E_C here would make the
    # consumer the owner of producer state and hide a lineage break.
    if request.e_c is None:
        if context.e_c.points:
            raise CudaJLoweringError(
                "CUDA lowering requires an explicit TargetPluginArtifact for "
                "non-empty shared E_C"
            )
        e_c = context.e_c
        e_c_mapping = ()
    else:
        e_c = request.e_c
        e_c_mapping = request.e_c_mapping
    try:
        envelope = TargetPluginArtifact(
            e_c=e_c,
            plugin=request.c,
            e_c_mapping=e_c_mapping,
        )
        return deserialize_target_artifact(
            context,
            serialize_target_artifact(envelope),
            expected_e_c=e_c,
            expected_e_c_mapping=e_c_mapping,
        )
    except (TypeError, ValueError) as exc:
        raise CudaJLoweringError(
            "CUDA target artifact failed public ingress validation: " + str(exc)
        ) from exc


def _child_regions(op: IntentOp) -> tuple[Sequence[IntentOp], ...]:
    return tuple(region for _suffix, region in intent_op_regions(op))


def _nodes(j: SemanticArchive) -> dict[str, IntentOp]:
    elaboration = j_elaboration(j)
    result = elaboration.construction_records.typed_operation_by_ref(j.node_ids)
    if result is None:
        raise CudaJLoweringError(
            "CUDA cannot materialize canonical operation references"
        )
    return result


def _contract_projection(
    j: SemanticArchive,
) -> ContractMaterializationProjection:
    """Derive CUDA's ephemeral Layer-B view from the one accepted J."""

    try:
        return project_contract_materialization(j)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise CudaJLoweringError(
            "typed contract projection failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def analyze_j_for_cuda(
    j: SemanticArchive,
    *,
    materialization: MaterializedJ | None = None,
) -> dict[str, Any]:
    """Check or consume every active Layer-B collection before emission."""

    if not isinstance(j, SemanticArchive):
        raise CudaJLoweringError("CUDA analysis requires canonical J")
    j = elaborate_j(j)
    unsupported: list[str] = []
    consumed: list[str] = []
    projection: (
        ContractMaterializationProjection
        | BoundaryMaterializationProjection
        | TypedBoundaryMaterializationProjection
        | None
    ) = (
        materialization.boundary_projection
        if materialization is not None
        else None
    )
    if projection is None:
        try:
            projection = _contract_projection(j)
        except CudaJLoweringError as exc:
            unsupported.append(str(exc))
    if projection is not None:
        consumed.extend(projection.consumed)
        unsupported.extend(projection.unsupported_lowering)
    projected_consumed = set(projection.consumed) if projection is not None else set()
    if j.boundary.access_guard_relations:
        guards_consumed = all(
            any(
                ref.startswith(
                    f"access_guard:{item.access_ref}:{item.value_ref}:"
                )
                and ref.endswith(f":{item.polarity.value}")
                for ref in projected_consumed
            )
            for item in j.boundary.access_guard_relations
        )
        if not guards_consumed:
            # Fail closed unless the shared J-only functional projection has
            # consumed every authored access-local guard.  The projection is
            # the existing generic consumer; CUDA does not invent a second
            # access recipe or infer permission from the surrounding select.
            unsupported.append("access_guard_materialization_not_implemented")
    if j.boundary.aliases:
        unsupported.append("alias_relation")
    consumed.extend(
        f"effect:{item.effect_id or index}:{item.kind.value}"
        for index, item in enumerate(j.boundary.effects)
    )
    for item in j.storage:
        storage_ref = f"storage:{item.storage_id}"
        if item.mutability.value != "mutable":
            consumed.append(storage_ref)
        elif (
            materialization is not None
            and storage_ref in projected_consumed
        ):
            # The registered full-axis-step materialization consumes the exact
            # structured carry/storage relation before the flat CUDA path is
            # considered.  Arbitrary mutable storage remains unsupported.
            consumed.append(storage_ref)
        else:
            unsupported.append(f"general_mutable_storage:{item.storage_id}")

    typed_rows = j_elaboration(j).typed_operation_rows
    if typed_rows is None:
        raise CudaJLoweringError(
            "CUDA operation view is unavailable from canonical records"
        )
    operations = tuple(operation for _path, operation in typed_rows)
    op_names = {op.op for op in operations}
    if "masked_scatter" in op_names:
        unsupported.append("masked_scatter_logical_index_relation")

    nodes = _nodes(j)
    elaboration = j_elaboration(j)
    # Boundary update type/rank checks consume the construction-owned value
    # type table.  Rebuilding this map from input/output declarations would
    # give CUDA a second output type authority beside the output record.
    tensor_by_name = {
        name: value_type
        for name, value_type in elaboration.value_type_by_name.items()
        if name.startswith("t")
    }
    observable_values = {
        item.output: item.value
        for item in elaboration.output_bindings
    }
    relation_by_output = {
        item.output: item for item in j.boundary.result_relations
    }
    for item in j.boundary.index_bounds:
        consumed.append(f"index_bounds:{item.node_ref}")
    for item in j.boundary.result_relations:
        consumed.append(f"result_relation:{item.output}:{item.kind.value}")

    for item in j.boundary.updates:
        op = nodes[item.node_ref]
        base = tensor_by_name.get(item.base)
        indices = tensor_by_name.get(item.indices)
        values = tensor_by_name.get(item.values)
        if base is None or indices is None or values is None:
            # A functional scatter may update an internal SSA tensor rather than
            # a boundary tensor.  Its complete update semantics are already
            # represented by the Intent op when the authored Layer-B relation
            # agrees exactly with the op: overwrite, no duplicate destination,
            # and input order.  This is the form used by structured attention
            # scratch/output materialization.  It needs no atomic permission and
            # is independent of tensor rank; target lowering still has to consume
            # the resulting scatter or replace the enclosing graph as a whole.
            update_record = consume_exact_internal_overwrite_update(
                j,
                item,
                nodes=nodes,
            )
            if update_record is not None:
                consumed.append(update_record)
                continue
            unsupported.append(f"update_internal_operand:{item.node_ref}")
            continue
        if not all(len(tensor.shape) == 2 for tensor in (base, indices, values)):
            unsupported.append(f"update_rank_not_2:{item.node_ref}")
            continue
        outputs = [
            output
            for output, value in observable_values.items()
            if value == op.result
        ]
        if len(outputs) != 1:
            unsupported.append(f"update_not_direct_output:{item.node_ref}")
            continue
        relation = relation_by_output[outputs[0]]
        if item.duplicate_policy is DuplicatePolicy.ONE_OF:
            unsupported.append(
                f"racing_overwrite_not_cuda_defined:{item.node_ref}"
            )
            continue
        if item.combiner is UpdateCombiner.ADD:
            if base.dtype.removeprefix("torch.").lower() not in {"f32", "float32"}:
                unsupported.append(f"atomic_add_dtype:{item.node_ref}")
                continue
            if (
                item.duplicate_policy is DuplicatePolicy.COMBINE
                and item.order is not UpdateOrder.UNORDERED
            ):
                unsupported.append(f"ordered_duplicate_add:{item.node_ref}")
                continue
            if (
                item.duplicate_policy is DuplicatePolicy.COMBINE
                and relation.kind is not ResultRelationKind.TOLERANCE
            ):
                unsupported.append(
                    f"floating_duplicate_add_requires_tolerance:{item.node_ref}"
                )
                continue
        consumed.append(
            "update:"
            f"{item.node_ref}:{item.combiner.value}:"
            f"{item.duplicate_policy.value}:{item.order.value}"
        )
    if "loop" in op_names:
        loop_nodes = {
            node_ref for node_ref, operation in nodes.items()
            if node_ref in j.node_ids and operation.op == "loop"
        }
        typed_loops_consumed = all(
            f"loop:{node_ref}" in projected_consumed for node_ref in loop_nodes
        )
        logical_loop_consumed = (
            materialization is not None
            and materialization.form
            in {
                DYNAMIC_VALUE_LOOP_MATERIALIZATION,
                LOGICAL_FULL_AXIS_STEP_MATERIALIZATION,
                STRUCTURED_SCAN_LOOP_MATERIALIZATION,
            }
        )
        if not (typed_loops_consumed or logical_loop_consumed):
            unsupported.append("semantic_recurrence_not_lowered_by_flat_cuda_path")
        elif materialization is not None:
            consumed.extend(materialization.consumed_structures)
            consumed.extend(f"loop:{node_ref}" for node_ref in loop_nodes)

    for item in j.boundary.reductions:
        op = nodes.get(item.node_ref)
        if op is None or op.op not in {"reduce", "scan"}:
            unsupported.append(f"invalid_reduction:{item.node_ref}")
            continue
        reduction_ref = f"reduction:{item.node_ref}"
        if reduction_ref in projected_consumed:
            consumed.append(reduction_ref)
            continue
        fn = op.attrs.get("fn")
        declared = item.identity
        neutral = {
            "sum": {0, "ZERO"},
            "mean": {0, "ZERO"},
            "prod": {1, "ONE"},
            "any": {False, "false"},
            "all": {True, "true"},
            "max": {"NEG_INF", "-inf", float("-inf")},
            "min": {"POS_INF", "inf", "+inf", float("inf")},
        }.get(fn)
        if neutral is None or declared not in neutral:
            unsupported.append(f"non_neutral_reduction_boundary:{item.node_ref}")
        else:
            consumed.append(f"reduction:{item.node_ref}")

    canonical_output_order = j_elaboration(j).output_names
    if tuple(observable_values) != canonical_output_order:
        unsupported.append("output_destination_relation")
    else:
        consumed.extend(f"output:{name}" for name in observable_values)
    consumed.extend(
        f"scope:{item['scope_id']}"
        for item in j_elaboration(j).scope_records
    )
    return {
        "supported": not unsupported,
        "consumed": sorted(set(consumed)),
        "unsupported": sorted(set(unsupported)),
        "boundary_projection": (
            projection.to_json() if projection is not None else None
        ),
    }


@dataclass(frozen=True)
class CudaJLowering:
    request: LoweringRequest
    boundary_consumption: Mapping[str, Any]
    intent_mlir: str
    artifact: LoweredProgram
    concrete_extents: tuple[tuple[str, int], ...]
    materializer_version: str
    materialization_form: str
    neutral_materialization_form: str | None = None

    def __post_init__(self) -> None:
        if not self.boundary_consumption.get("supported", False):
            raise CudaJLoweringError("CUDA lowering left Layer-B relations unsupported")
        if self.request.c is not None:
            contract = self.request.c.contract
            evidence = self.boundary_consumption.get("target_realization")
            if contract is None or not isinstance(evidence, Mapping):
                raise CudaJLoweringError(
                    "CUDA lowering lacks exact target Layer-C consumption evidence"
                )
            try:
                contract.validate_consumption_evidence(evidence)
            except (TypeError, ValueError) as exc:
                raise CudaJLoweringError(str(exc)) from exc

    @property
    def concrete_extents_map(self) -> dict[str, int]:
        return dict(self.concrete_extents)

    def conformance_record(
        self,
        *,
        test_domain: Mapping[str, Any],
        comparisons: Sequence[ComparisonEvidence],
    ) -> BackendConformanceRecord:
        return BackendConformanceRecord(
            target=self.request.target,
            variant=self.request.variant,
            boundary_consumption=dict(self.boundary_consumption),
            test_domain=dict(test_domain),
            comparisons=tuple(comparisons),
        )

    def to_record(self) -> dict[str, Any]:
        target_artifact = None
        if self.request.c is not None:
            if self.request.e_c is None:
                raise CudaJLoweringError(
                    "CUDA target plugin record is missing its shared E_C"
                )
            target_artifact = TargetPluginArtifact(
                e_c=self.request.e_c,
                plugin=self.request.c,
                e_c_mapping=self.request.e_c_mapping,
            )
        return {
            "target": self.request.target.to_dict(),
            "variant": self.request.variant,
            "j_digest": layer_c_content_digest(self.request.j.to_json()),
            "layer_c_interface_digest": self.request.c.interface_digest
            if self.request.c is not None
            else None,
            "constexpr_bindings": dict(self.request.constexpr_bindings),
            "concrete_extents": self.concrete_extents_map,
            "materializer_version": self.materializer_version,
            "materialization_form": self.materialization_form,
            "neutral_materialization_form": self.neutral_materialization_form,
            "boundary_consumption": dict(self.boundary_consumption),
            "e_c_interface_digest": (
                target_artifact.e_c_digest if target_artifact is not None else None
            ),
            "e_c_mapping_digest": (
                target_artifact.e_c_mapping_digest
                if target_artifact is not None
                else None
            ),
            "target_plugin_digest": (
                self.request.c.content_digest if self.request.c is not None else None
            ),
            "target_artifact_digest": (
                target_artifact.content_digest if target_artifact is not None else None
            ),
            "target_realization": self.boundary_consumption.get(
                "target_realization"
            ),
            "cuda": self.artifact.to_record(),
        }


def lower_j_to_cuda(
    request: LoweringRequest,
    concrete_extents: Mapping[str, int],
) -> CudaJLowering:
    if not isinstance(request, LoweringRequest):
        raise CudaJLoweringError("CUDA lowering requires LoweringRequest")
    if request.target.family != "cuda":
        raise CudaJLoweringError("CUDA backend requires a CUDA target")
    if request.variant not in {"generic", "auto", "specialized"}:
        raise CudaJLoweringError(
            "CUDA lowering accepts generic, auto, or specialized variant"
        )
    if request.c is None and request.variant != "generic":
        raise CudaJLoweringError(
            "CUDA auto/specialized lowering requires a target Layer-C plugin"
        )
    from .search import cuda_layer_c_interface

    extension_interface = cuda_layer_c_interface(request.j)
    extension_interface_digest = binding_interface_digest(extension_interface)
    required_extension_slots = tuple(
        item.slot for item in extension_interface.points if item.required
    )
    if request.c is None and required_extension_slots:
        raise CudaJLoweringError(
            "CUDA generic lowering requires a target Layer-C plugin for "
            "required extension slot(s): "
            + ", ".join(required_extension_slots)
        )
    if request.c is not None and request.c.interface_digest != extension_interface_digest:
        raise CudaJLoweringError(
            "CUDA target Layer-C interface is stale or detached from J"
        )
    j_before = canonical_layer_c_json(request.j.to_json())
    j_digest_before = layer_c_content_digest(request.j.to_json())
    try:
        materialized = materialize_j(
            request.j,
            concrete_extents,
            constexpr_bindings=dict(request.constexpr_bindings),
        )
    except (TypeError, ValueError) as exc:
        raise CudaJLoweringError(str(exc)) from exc
    consumption = analyze_j_for_cuda(
        request.j,
        materialization=materialized,
    )
    consumption["layer_c_interface_digest"] = extension_interface_digest
    consumption["layer_c_extension_count"] = len(extension_interface.points)
    if not consumption["supported"]:
        raise CudaJLoweringError(
            "CUDA cannot consume complete Layer B: "
            + ", ".join(consumption["unsupported"])
        )
    mlir = materialized.intent_mlir
    target_materialization: MaterializedJ | None = None
    target = LoweringTarget(
        backend="cuda",
        chip=request.target.architecture,
        features=frozenset(request.target.features),
    )
    try:
        if request.c is None:
            # The only C-free production route is deterministic J-only generic
            # lowering.  No auto selector may introduce a hardware realization.
            artifact = lower_to_cuda(
                mlir,
                target=target,
                allow_specialized=False,
            )
        else:
            from .search import (
                compile_cuda_candidate,
                resolve_cuda_target_plugin,
            )

            state = BackendSearchState(
                base=materialized,
                target=request.target,
                priors=(),
                ignored_slots=(),
            )
            from .search import (
                cuda_allowed_authorizations,
                derive_cuda_contract_view,
            )

            # Cross the backend-neutral file/CLI boundary before candidate
            # resolution.  The lowering path must consume the parsed complete
            # E_C/C_target artifact, not the pre-serialization Python object
            # supplied by its caller.
            contract_view = derive_cuda_contract_view(state)
            shared_e_c = (
                request.e_c
                if request.e_c is not None
                else LayerCInterface()
            )
            target_artifact = _deserialize_cuda_target_artifact(
                request,
                TargetBindingContext(
                    j=state.base.j,
                    target=state.target,
                    interface=extension_interface,
                    capabilities=contract_view.capabilities,
                    allowed_authorizations=cuda_allowed_authorizations(state),
                    e_c=shared_e_c,
                ),
            )
            request = replace(
                request,
                c=target_artifact,
                e_c=None,
                e_c_mapping=(),
            )
            candidate = resolve_cuda_target_plugin(state, request.c)
            # ``resolve_cuda_target_plugin`` validates the complete plugin and
            # candidate against neutral M0 first.  Only after that boundary is
            # crossed may an optional target tile create a temporary lowering
            # view.  The neutral state remains the sole search/contract base.
            from .search import cuda_plugin_loop_schedule_tile

            loop_schedule_tile = cuda_plugin_loop_schedule_tile(
                request.j,
                request.target,
                request.c,
            )
            if loop_schedule_tile is not None:
                try:
                    target_materialization = materialize_target_lowering(
                        materialized,
                        loop_schedule_tile=loop_schedule_tile,
                    )
                except (TypeError, ValueError) as exc:
                    raise CudaJLoweringError(str(exc)) from exc
                mlir = target_materialization.intent_mlir
            if request.variant == "generic" and not candidate.generic:
                raise CudaJLoweringError(
                    "CUDA generic variant cannot select a specialized Layer-C candidate"
                )
            if request.variant == "specialized" and candidate.generic:
                raise CudaJLoweringError(
                    "CUDA specialized variant cannot select a generic Layer-C candidate"
                )
            artifact = compile_cuda_candidate(
                state,
                candidate,
                intent_module=mlir,
            )
            # ``observation_changing`` is a Q/D classification, not a generic
            # authorization token.  Candidate-specific J/B authorizations are
            # checked by ``resolve_cuda_target_plugin``; target-only numeric
            # changes are admitted here and must pass SearchAndClose before a
            # benchmark winner is accepted.
    except CudaJLoweringError:
        raise
    except (TypeError, ValueError) as exc:
        raise CudaJLoweringError(str(exc)) from exc
    j_after = canonical_layer_c_json(request.j.to_json())
    j_digest_after = layer_c_content_digest(request.j.to_json())
    if (
        j_after != j_before
        or j_digest_after != j_digest_before
        or (
            request.c is not None
            and request.c.j_digest != j_digest_before
        )
    ):
        raise CudaJLoweringError(
            "CUDA lowering mutated J or detached target Layer C from canonical J"
        )
    if request.c is not None:
        contract = request.c.contract
        if contract is None:
            raise CudaJLoweringError(
                "CUDA target plugin is missing its realization contract"
            )
        cubin_bundle = _cuda_cubin_bundle(artifact)
        applications = []
        for record_id in contract.declared_record_ids():
            if record_id == "lowering":
                applications.append(
                    contract.record_consumption_from_artifact(
                        record_id,
                        consumer="cuda.native",
                        phase="cuda_lowering",
                        artifact_kind="cubin_bundle",
                        artifact=cubin_bundle,
                    )
                )
                continue
            field_name = record_id.split(":", 1)[0]
            if field_name == "pipeline":
                applications.append(
                    contract.record_consumption_from_artifact(
                        record_id,
                        consumer="cuda.native",
                        phase="cuda_pipeline",
                        artifact_kind="cuda_realization_mlir",
                        artifact=artifact.realization_mlir,
                    )
                )
                continue
            if field_name == "split" and target_materialization is not None:
                applications.append(
                    contract.record_consumption_from_artifact(
                        record_id,
                        consumer="cuda.native",
                        phase="target_tile_materialization",
                        artifact_kind="target_materialization_mlir",
                        artifact=target_materialization.intent_mlir,
                    )
                )
                continue
            raise CudaJLoweringError(
                "CUDA has no independent real compiler artifact for target "
                f"realization record {record_id!r}"
            )
        try:
            consumption["target_realization"] = contract.consumption_evidence(applications)
        except (TypeError, ValueError) as exc:
            raise CudaJLoweringError(
                "CUDA target Layer-C consumption is not exact: " + str(exc)
            ) from exc
    return CudaJLowering(
        request=request,
        boundary_consumption=consumption,
        intent_mlir=mlir,
        artifact=artifact,
        concrete_extents=materialized.concrete_extents,
        materializer_version=materialized.materializer_version,
        materialization_form=(
            target_materialization.form
            if target_materialization is not None
            else materialized.form
        ),
        neutral_materialization_form=materialized.form,
    )


def execute_j_cuda(
    lowering: CudaJLowering,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one compiled J lowering using only J-owned logical handles.

    The MLIR-level orchestrator intentionally supports positional ``argN``
    fallbacks for legacy callers.  This J-facing boundary does not: the exact
    logical input set comes from the same accepted J that produced the cubin,
    and ordered ``outN`` runtime buffers are mapped back to J output handles.
    """

    if not isinstance(lowering, CudaJLowering):
        raise CudaJExecutionError("CUDA J execution requires CudaJLowering")
    if not isinstance(inputs, Mapping):
        raise CudaJExecutionError(
            "CUDA J execution inputs must map J logical handles to values"
        )
    if any(not isinstance(name, str) for name in inputs):
        raise CudaJExecutionError(
            "CUDA J execution input keys must be J logical handle strings"
        )
    constexpr_bindings = dict(lowering.request.constexpr_bindings)
    expected_inputs = tuple(
        item.name
        for item in lowering.request.j.intent.inputs
        if item.name not in constexpr_bindings
    )
    provided = set(inputs)
    expected = set(expected_inputs)
    if provided != expected:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise CudaJExecutionError(
            "CUDA J execution inputs differ from J logical inputs: "
            + "; ".join(details)
        )
    try:
        logical_inputs = {
            **constexpr_bindings,
            **{name: inputs[name] for name in expected_inputs},
        }
        validate_value_dimension_instantiation(
            lowering.request.j,
            lowering.concrete_extents_map,
            logical_inputs,
        )
        validate_ragged_runtime_invocation(
            lowering.request.j.intent,
            logical_inputs,
            dims=lowering.concrete_extents_map,
        )
    except (TypeError, ValueError) as exc:
        raise CudaJExecutionError(str(exc)) from exc

    runtime_outputs = execute_cuda_artifact(
        lowering.artifact,
        lowering.intent_mlir,
        {name: inputs[name] for name in expected_inputs},
    )
    if not isinstance(runtime_outputs, Mapping):
        raise CudaJExecutionError("CUDA runtime did not return an output mapping")
    logical_outputs = j_elaboration(lowering.request.j).output_names
    runtime_keys = tuple(f"out{index}" for index in range(len(logical_outputs)))
    if set(runtime_outputs) != set(runtime_keys):
        raise CudaJExecutionError(
            "CUDA runtime outputs differ from J logical outputs: expected "
            f"{list(runtime_keys)}, got {sorted(runtime_outputs)}"
        )
    return {
        logical_name: runtime_outputs[runtime_name]
        for logical_name, runtime_name in zip(
            logical_outputs, runtime_keys, strict=True
        )
    }


def cuda_target(architecture: str, *, features: Sequence[str] = ()) -> BackendTarget:
    return BackendTarget("cuda", architecture, tuple(features))


__all__ = [
    "CudaJExecutionError", "CudaJLowering", "CudaJLoweringError",
    "analyze_j_for_cuda", "cuda_target", "execute_j_cuda",
    "lower_j_to_cuda",
]
