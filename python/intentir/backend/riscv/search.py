"""Candidate-universe and discardable-C planning for the native RVV backend.

The candidate universe is constructed without consulting C.  ``Attach_C`` may
only reorder that fixed universe; erasing C therefore changes neither
admission nor the executable states available to the search.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from intentir.backend.materialize import BackendSearchState, CPrior
from intentir.backend.riscv.compile import (
    RiscvLoweringCandidate,
    RiscvSchedule,
    RiscvTransformationState,
    ragged_schedule_supported,
)
from intentir.backend.riscv.contract import (
    RiscvContractError,
    derive_riscv_contract_view,
    riscv_allowed_authorizations,
    validate_riscv_target_contract,
)
from intentir.ir.canonical_j import j_elaboration
from intentir.ir.layer_c import (
    ImplementationChoice,
    LayerCBinding,
    LayerCExtensionPoint,
    LayerCInterface,
    TargetLowering,
    TargetParallelMapping,
    TargetPipeline,
    TargetRealizationContract,
    TargetSplit,
    TargetLayerCPlugin,
    layer_c_content_digest,
    validate_layer_c_attachments,
)
from intentir.backend.protocol import (
    TargetBindingContext,
    TargetExtensionMapping,
    TargetPluginArtifact,
    bind_target,
    deserialize_target_plugin,
    serialize_target_plugin,
    validate_target_plugin,
)
from intentir.verify.gate1.model import SemanticArchive


RVV_SEARCH_PLAN_PROTOCOL = "intentir.backend.riscv.search-plan.v11"
RVV_C_PROJECTION_PROTOCOL = "intentir.backend.riscv.c-projection.v1"
RVV_LAYER_C_BACKEND_KIND = "riscv"

_PARALLEL_TILES = (1, 2, 4, 8, 16, 32, 64, 128)
_REDUCTION_TILES = (1, 2, 4, 8, 16, 32, 64, 128)
_JOINT_TILES = ((1, 4), (2, 8), (4, 16), (8, 32), (16, 64), (32, 128))


def _canonical_pattern_records(
    j: SemanticArchive,
) -> tuple[Any, tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Return the one elaborated operation/boundary view for RVV matching.

    Search features are admissibility hints, but they still must not rebuild a
    second interpretation by walking raw ``intent_function.body``.  The
    elaboration's construction bundle owns lexical paths, typed values, output
    order and boundary rows, so every structural matcher below receives only
    projections from that bundle.
    """

    elaboration = j_elaboration(j)
    all_records = tuple(elaboration.operation_records)
    if any(
        not isinstance(record, Mapping)
        or not isinstance(record.get("wire"), Mapping)
        for record in all_records
    ):
        raise RiscvContractError(
            "RISC-V search requires complete canonical operation records"
        )
    top_level = tuple(
        record
        for record in all_records
        if re.fullmatch(r"body\[\d+\]", str(record.get("path")))
    )
    return elaboration, top_level, all_records


def _record_type(
    elaboration: Any,
    value: Any,
) -> Any | None:
    return (
        elaboration.value_type_by_name.get(value)
        if isinstance(value, str)
        else None
    )


def _record_result(record: Mapping[str, Any]) -> str | None:
    results = record.get("results")
    return (
        results[0]
        if isinstance(results, (tuple, list))
        and len(results) == 1
        and isinstance(results[0], str)
        else None
    )


def _schedule_universe() -> tuple[RiscvSchedule, ...]:
    schedules = [
        RiscvSchedule(opt, vectorize, 128, 0, 0, thread_count, False)
        for thread_count in (1,)
        for opt in (2, 3)
        for vectorize in (False, True)
    ]
    # RVV's scalable-vector policy is a real LLVM schedule state, not a
    # spelling of an arbitrary VLEN.  Keep the untiled anchors in the complete
    # space so a target with a different implementation VLEN can choose
    # strip-mined loops without changing J or the MLIR graph.
    schedules.append(RiscvSchedule(3, True, 0, 0, 0, 1, True))
    schedules.extend(
        RiscvSchedule(3, True, 128, parallel, reduction, thread_count, False)
        for thread_count in (1,)
        for parallel, reduction in (
            *((value, 0) for value in _PARALLEL_TILES),
            *((0, value) for value in _REDUCTION_TILES),
            *_JOINT_TILES,
        )
    )
    # A small set of tiled scalable states is enough to expose the interaction
    # between LLVM strip-mining and the MLIR retile.  The exhaustive fixed-VLEN
    # tile domain remains intact; scalable candidates are not represented by a
    # fake SEW/LMUL knob.
    schedules.extend(
        RiscvSchedule(3, True, 0, parallel, reduction, 1, True)
        for parallel, reduction in (
            (4, 0),
            (16, 0),
            (64, 0),
            (0, 4),
            (0, 16),
            (0, 64),
            *_JOINT_TILES,
        )
    )
    unique: dict[
        tuple[int, bool, int, int, int, int, bool, int], RiscvSchedule
    ] = {}
    for item in schedules:
        key = (
            item.opt_level,
            item.vectorize,
            item.vector_bits,
            item.parallel_tile,
            item.reduction_tile,
            item.thread_count,
            item.scalable_vectorization,
            item.vector_lanes,
        )
        unique.setdefault(key, item)
    return tuple(unique[key] for key in sorted(unique))


def enumerate_riscv_candidates(
    *,
    enable_b_transformations: bool = True,
    enable_reassociation: bool = True,
    enable_rowmask_zero_fill: bool = False,
    enable_composed_gather: bool = False,
    enable_sibling_output_fusion: bool = False,
    enable_paired_rotation: bool = False,
    enable_streaming_attention: bool = False,
) -> tuple[RiscvLoweringCandidate, ...]:
    """Build the target schedule/structure universe without reading C."""

    transformations = [RiscvTransformationState(False, False, False)]
    if enable_b_transformations:
        transformations.append(RiscvTransformationState(True, False, False))
    if enable_rowmask_zero_fill:
        transformations.append(
            RiscvTransformationState(False, rowmask_zero_fill=True)
        )
    if enable_composed_gather:
        transformations.append(
            RiscvTransformationState(False, composed_indexed_gather=True)
        )
    if enable_sibling_output_fusion and enable_b_transformations:
        transformations.append(
            RiscvTransformationState(True, fuse_sibling_outputs=True)
        )
    if enable_paired_rotation:
        # The contiguous direct loop is the canonical complete consumer of the
        # closed paired-rotation view, not a Full-only optimization.  Full may
        # additionally change loop topology by hoisting angles across heads.
        transformations.append(
            RiscvTransformationState(False, paired_rotation=True)
        )
        if enable_b_transformations:
            transformations.append(
                RiscvTransformationState(
                    False,
                    paired_rotation=True,
                    paired_rotation_head_angle_reuse=True,
                )
            )
    if enable_reassociation:
        transformations.append(RiscvTransformationState(False, True, False))
        if enable_streaming_attention and enable_b_transformations:
            transformations.append(
                RiscvTransformationState(
                    False,
                    True,
                    streaming_attention=True,
                )
            )
        if enable_b_transformations:
            transformations.append(
                RiscvTransformationState(
                    True,
                    True,
                    mixed_precision_dot_reduction=True,
                )
            )
        transformations.append(RiscvTransformationState(False, True, True))
        if enable_b_transformations:
            # The two Softmax states are placed before the generic fused
            # reduction states so a practical round-robin budget observes both
            # the execution-domain checked vector-exp path and the exact-exp
            # fallback.  This is ordering only; C never admits either state.
            transformations.append(
                RiscvTransformationState(
                    False,
                    True,
                    fused_softmax=True,
                    approximate_exp=True,
                )
            )
            transformations.append(
                RiscvTransformationState(False, True, fused_softmax=True)
            )
            transformations.append(RiscvTransformationState(True, True, False))
            transformations.append(RiscvTransformationState(True, True, True))
            transformations.append(RiscvTransformationState(True, True, True, True))
    schedules = _schedule_universe()
    candidates: list[RiscvLoweringCandidate] = []
    for transformation in transformations:
        for base_schedule in schedules:
            if transformation.paired_rotation and (
                base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
                or (
                    transformation.paired_rotation_head_angle_reuse
                    and base_schedule.vectorize
                )
            ):
                continue
            if transformation.rowmask_zero_fill and (
                base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
                or base_schedule.scalable_vectorization
            ):
                continue
            if transformation.composed_indexed_gather and (
                not base_schedule.vectorize
                or base_schedule.scalable_vectorization
                or base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
            ):
                continue
            if (
                transformation.explicit_vector_reduction
                and not base_schedule.scalable_vectorization
            ):
                continue
            if transformation.mixed_precision_dot_reduction and (
                not base_schedule.vectorize
                or base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
            ):
                continue
            if transformation.fused_softmax and (
                not base_schedule.vectorize
                or base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
                or (
                    transformation.approximate_exp
                    and base_schedule.scalable_vectorization
                )
            ):
                continue
            if transformation.streaming_attention and (
                not base_schedule.vectorize
                or base_schedule.scalable_vectorization
                or base_schedule.parallel_tile != 0
                or base_schedule.reduction_tile != 0
            ):
                continue
            if transformation.streaming_attention:
                lane_values = (2, 4, 8, 16)
            elif transformation.mixed_precision_dot_reduction:
                lane_values = (
                    (4,) if base_schedule.scalable_vectorization else (16,)
                )
            else:
                lane_values = (
                    (4,)
                    if not transformation.fused_softmax
                    or base_schedule.scalable_vectorization
                    else (2, 4, 8, 16)
                )
            candidates.extend(
                RiscvLoweringCandidate(
                    transformation,
                    replace(base_schedule, vector_lanes=lanes),
                )
                for lanes in lane_values
            )
    return tuple(candidates)


@dataclass(frozen=True)
class _ProjectedCPrior:
    prior: CPrior
    schedule_dimension: str
    target_value: int
    rule: str

    def to_record(self) -> dict[str, Any]:
        return {
            "source_prior": self.prior.to_record(),
            "schedule_dimension": self.schedule_dimension,
            "target_value": self.target_value,
            "rule": self.rule,
        }


def _dtype_bits(dtype: str) -> int | None:
    match = re.fullmatch(r"(?:bf|f)(16|32|64)", dtype.lower())
    return int(match.group(1)) if match is not None else None


def _operation_facts(
    j: SemanticArchive,
) -> tuple[set[tuple[int, ...]], bool, list[int], bool, bool]:
    reduction_axes: set[tuple[int, ...]] = set()
    has_scan = False
    float_bits: list[int] = []
    has_reassociable_float_reduction = False
    has_contraction = False

    elaboration, _top_level, records = _canonical_pattern_records(j)
    for value_type in elaboration.value_type_by_name.values():
        dtype = getattr(value_type, "dtype", None)
        if isinstance(dtype, str):
            bits = _dtype_bits(dtype)
            if bits is not None:
                float_bits.append(bits)
    for record in records:
        op = record.get("op")
        has_contraction = has_contraction or op == "contraction"
        if op not in {"reduce", "scan"}:
            continue
        has_scan = has_scan or op == "scan"
        attrs = record.get("attrs")
        if not isinstance(attrs, Mapping):
            continue
        if op == "reduce" and attrs.get("assoc") == "float_nondeterministic":
            has_reassociable_float_reduction = True
        raw_axes = attrs.get("axes", [attrs.get("axis")])
        if isinstance(raw_axes, list) and all(
            isinstance(axis, int) for axis in raw_axes
        ):
            reduction_axes.add(tuple(int(axis) for axis in raw_axes))
    return (
        reduction_axes,
        has_scan,
        float_bits,
        has_reassociable_float_reduction,
        has_contraction,
    )


def _has_typed_rowmask_graph(j: SemanticArchive) -> bool:
    elaboration, body, _records = _canonical_pattern_records(j)
    if len(body) != 2 or len(elaboration.output_bindings) != 1:
        return False
    cast, product = body
    cast_operands = cast.get("operands")
    product_operands = product.get("operands")
    cast_result = _record_result(cast)
    product_result = _record_result(product)
    cast_type = _record_type(elaboration, cast_result)
    product_type = _record_type(elaboration, product_result)
    attrs = product.get("attrs")
    if (
        cast.get("op") != "cast"
        or not isinstance(cast_operands, list)
        or len(cast_operands) != 1
        or getattr(cast_type, "dtype", None) != "f32"
        or cast_result is None
        or product.get("op") != "map_binary"
        or not isinstance(attrs, Mapping)
        or attrs.get("fn", "mul") != "mul"
        or not isinstance(product_operands, list)
        or len(product_operands) != 2
        or cast_result not in product_operands
        or getattr(product_type, "dtype", None) != "f32"
    ):
        return False
    mask = _record_type(elaboration, cast_operands[0])
    data_name = next(
        (operand for operand in product_operands if operand != cast_result),
        None,
    )
    data = _record_type(elaboration, data_name)
    mask_shape = getattr(mask, "shape", None)
    data_shape = getattr(data, "shape", None)
    result_shape = getattr(product_type, "shape", None)
    output = elaboration.output_bindings[0]
    return bool(
        getattr(mask, "dtype", None) == "i1"
        and isinstance(mask_shape, tuple)
        and len(mask_shape) == 1
        and getattr(data, "dtype", None) == "f32"
        and isinstance(data_shape, tuple)
        and len(data_shape) == 2
        and data_shape[0] == mask_shape[0]
        and result_shape == data_shape
        and output.value == product_result
        and output.output_type.dtype == "f32"
        and output.output_type.shape == data_shape
    )


def _has_typed_gather_chain(j: SemanticArchive) -> bool:
    elaboration, body, _records = _canonical_pattern_records(j)
    if len(body) != 2 or len(elaboration.output_bindings) != 1:
        return False
    first, second = body
    first_operands = first.get("operands")
    second_operands = second.get("operands")
    first_attrs = first.get("attrs")
    second_attrs = second.get("attrs")
    first_result = _record_result(first)
    second_result = _record_result(second)
    first_type = _record_type(elaboration, first_result)
    second_type = _record_type(elaboration, second_result)
    output = elaboration.output_bindings[0]
    if (
        first.get("op") != "gather"
        or second.get("op") != "gather"
        or not isinstance(first_operands, list)
        or len(first_operands) != 2
        or not isinstance(second_operands, list)
        or len(second_operands) != 2
        or second_operands[0] != first_result
        or not isinstance(first_attrs, Mapping)
        or first_attrs.get("mode") != "take"
        or first_attrs.get("axis") != 0
        or not isinstance(second_attrs, Mapping)
        or second_attrs.get("mode") != "along_axis"
        or second_attrs.get("axis") != 1
        or getattr(first_type, "dtype", None) != "f32"
        or getattr(second_type, "dtype", None) != "f32"
        or output.value != second_result
    ):
        return False
    data = _record_type(elaboration, first_operands[0])
    rows = _record_type(elaboration, first_operands[1])
    columns = _record_type(elaboration, second_operands[1])
    return bool(
        getattr(data, "dtype", None) == "f32"
        and isinstance(getattr(data, "shape", None), tuple)
        and len(data.shape) == 2
        and getattr(rows, "dtype", None) == "i32"
        and isinstance(getattr(rows, "shape", None), tuple)
        and len(rows.shape) == 1
        and getattr(columns, "dtype", None) == "i32"
        and columns.shape == rows.shape
        and second_type.shape == rows.shape
    )


def _has_two_public_outputs(j: SemanticArchive) -> bool:
    return len(j_elaboration(j).output_bindings) == 2


def _has_typed_flash_attention_graph(j: SemanticArchive) -> bool:
    """Match the closed structured stable-attention J without kernel names."""

    elaboration, top_level_operations, all_records = _canonical_pattern_records(j)
    boundary = elaboration.boundary_record
    constraints = tuple(j.dimension_constraints)
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("aliases") not in (None, [])
    ):
        return False
    # Physical public outputs are owned by the construction output rows; do
    # not re-read ``intent.outputs`` as a second ABI declaration.
    if len(elaboration.output_bindings) != 3 or not top_level_operations:
        return False
    operations = list(all_records)
    contractions = [item for item in operations if item.get("op") == "contraction"]
    loops = [item for item in operations if item.get("op") == "loop"]
    scatters = [item for item in operations if item.get("op") == "scatter"]
    reductions = [item for item in operations if item.get("op") == "reduce"]
    if (
        len(contractions) != 2
        or len(loops) != 1
        or len(scatters) != 3
        or len(reductions) != 2
    ):
        return False
    qk = next(
        (
            item
            for item in contractions
            if isinstance(item.get("attrs"), Mapping)
            and item["attrs"].get("subscripts") == "bqhd,bkhd->bhqk"
        ),
        None,
    )
    pv = next(
        (
            item
            for item in contractions
            if isinstance(item.get("attrs"), Mapping)
            and item["attrs"].get("subscripts") == "bhqk,bkhd->bqhd"
        ),
        None,
    )
    if qk is None or pv is None:
        return False
    qk_operands = qk.get("operands")
    pv_operands = pv.get("operands")
    if (
        not isinstance(qk_operands, list)
        or len(qk_operands) != 2
        or not isinstance(pv_operands, list)
        or len(pv_operands) != 2
    ):
        return False
    loop_wire = loops[0].get("wire")
    scans = loop_wire.get("scan") if isinstance(loop_wire, Mapping) else None
    scan_sources = {
        item.get("name"): item.get("from")
        for item in scans
        if isinstance(item, Mapping)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("from"), str)
    } if isinstance(scans, list) else {}

    def boundary_input(name: Any) -> Any:
        return _record_type(elaboration, scan_sources.get(name, name))

    query = boundary_input(qk_operands[0])
    key = boundary_input(qk_operands[1])
    value_input = boundary_input(pv_operands[1])
    if any(item is None for item in (query, key, value_input)):
        return False
    query_shape = query.shape
    key_shape = key.shape
    value_shape = value_input.shape
    if (
        query.dtype != "f16"
        or key.dtype != "f16"
        or value_input.dtype != "f16"
        or not isinstance(query_shape, tuple)
        or not isinstance(key_shape, tuple)
        or len(query_shape) != 4
        or len(key_shape) != 4
        or value_shape != key_shape
        or query_shape[0] != key_shape[0]
        or query_shape[2:] != key_shape[2:]
    ):
        return False
    output_types = [item.output_type for item in elaboration.output_bindings]
    rank_four_outputs = [
        item for item in output_types
        if item.dtype == "f16" and item.shape == query_shape
    ]
    rank_three_outputs = [
        item for item in output_types
        if item.dtype == "f32"
        and len(item.shape) == 3
        and item.shape[:2] == (query_shape[0], query_shape[2])
    ]
    output_only = {
        item.output
        for item in elaboration.output_bindings
        if item.abi_parameter_mode == "output"
    }
    input_types = [
        value_type
        for name, value_type in elaboration.value_type_by_name.items()
        if name.startswith("t") and name not in output_only
    ]
    scalar_f32_inputs = [
        item for item in input_types
        if item.dtype == "f32" and item.shape == ()
    ]
    scratch_inputs = [
        item for item in input_types
        if item.dtype == "f32"
        and len(item.shape) == 3
        and item.shape[:2] == (query_shape[0], query_shape[2])
    ]
    if (
        len(rank_four_outputs) != 1
        or len(rank_three_outputs) != 2
        or len(scalar_f32_inputs) != 1
        or len(scratch_inputs) != 1
    ):
        return False
    loop_trip = loop_wire.get("trip") if isinstance(loop_wire, Mapping) else None
    if not isinstance(loop_trip, Mapping) or loop_trip.get("axis") != key_shape[1]:
        return False
    reduction_facts = {
        (
            item.get("attrs", {}).get("fn"),
            tuple(item.get("attrs", {}).get("axes", [])),
            item.get("attrs", {}).get("assoc"),
        )
        for item in reductions
        if isinstance(item.get("attrs"), Mapping)
    }
    if reduction_facts != {
        ("max", (3,), "float_nondeterministic"),
        ("sum", (3,), "float_nondeterministic"),
    }:
        return False
    if any(
        not isinstance(item.get("attrs"), Mapping)
        or item["attrs"].get("axis") != 2
        or item["attrs"].get("mode") != "along_axis"
        or item["attrs"].get("reduce") != "none"
        for item in scatters
    ):
        return False
    updates = boundary.get("updates")
    relations = boundary.get("result_relations")
    if (
        not isinstance(updates, list)
        or len(updates) != 3
        or any(
            not isinstance(item, Mapping)
            or item.get("axis") != 2
            or item.get("combiner") != "overwrite"
            or item.get("duplicate_policy") != "forbid"
            or item.get("order") != "input_order"
            for item in updates
        )
        or not isinstance(relations, list)
        or len(relations) != 3
        or any(
            not isinstance(item, Mapping) or item.get("kind") != "tolerance"
            for item in relations
        )
    ):
        return False
    padded = scratch_inputs[0].shape[2]
    return any(
        isinstance(item, str)
        and item.replace(" ", "")
        in {
            f"{query_shape[1]}<={padded}",
            f"{padded}>={query_shape[1]}",
        }
        for item in constraints
    )


def _target_vlen_bits(state: BackendSearchState) -> int:
    return validate_riscv_target_contract(state)


def _power_of_two_floor(value: float) -> int:
    return 1 << max(0, math.floor(math.log2(max(1.0, value))))


def _project_c_priors(
    state: BackendSearchState,
) -> tuple[
    tuple[_ProjectedCPrior, ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    reduction_axes, has_scan, float_bits, _has_reassociation, _has_contraction = (
        _operation_facts(state.base.j)
    )
    vlen_bits = _target_vlen_bits(state)
    element_bits = max(float_bits, default=32)
    vector_lanes = max(1, vlen_bits // element_bits)
    projected: list[_ProjectedCPrior] = []
    ignored: list[dict[str, Any]] = []
    for prior in state.priors:
        if prior.role not in {
            "tile_extent",
            "reduction_tile_extent",
            "constexpr",
        }:
            ignored.append(
                {
                    **prior.to_record(),
                    "reason": "role_has_no_registered_single_core_rvv_projection",
                }
            )
            continue
        if (
            isinstance(prior.value, bool)
            or not isinstance(prior.value, (int, float))
            or float(prior.value) <= 0
        ):
            ignored.append(
                {**prior.to_record(), "reason": "non_positive_numeric_value"}
            )
            continue
        # CPrior is deliberately target-neutral and carries no Layer-C axis
        # attachment.  Keep the projection explicit instead of referring to
        # a non-existent per-prior axis field.
        axis_ref = None
        if prior.role == "reduction_tile_extent" and len(reduction_axes) == 1:
            maps_to_reduction = True
        elif prior.role == "reduction_tile_extent" and not reduction_axes:
            ignored.append(
                {
                    **prior.to_record(),
                    "reason": "reduction_tile_prior_has_no_reduction_axis",
                }
            )
            continue
        elif prior.role == "reduction_tile_extent":
            ignored.append(
                {
                    **prior.to_record(),
                    "reason": "unanchored_reduction_tile_has_multiple_reduction_axes",
                }
            )
            continue
        elif axis_ref is not None:
            maps_to_reduction = any(axis_ref in axes for axes in reduction_axes)
        elif len(reduction_axes) == 1:
            maps_to_reduction = True
        elif not reduction_axes:
            maps_to_reduction = False
        else:
            ignored.append(
                {
                    **prior.to_record(),
                    "reason": "unanchored_tile_prior_has_multiple_reduction_axes",
                }
            )
            continue
        dimension = "reduction_tile" if maps_to_reduction else "parallel_tile"
        target_cap = 2 * vector_lanes if maps_to_reduction else vector_lanes
        target_value = _power_of_two_floor(
            min(float(prior.value), float(target_cap))
        )
        projected.append(
            _ProjectedCPrior(
                prior,
                dimension,
                target_value,
                (
                    "unique_reduction_axis_to_two_rvv_vectors"
                    if maps_to_reduction
                    else "parallel_axis_to_one_rvv_vector"
                ),
            )
        )
    projection = {
        "protocol": RVV_C_PROJECTION_PROTOCOL,
        "target_vlen_bits": vlen_bits,
        "target_identity": state.target.to_dict(),
        "element_bits": element_bits,
        "vector_lanes": vector_lanes,
        "reduction_axes": [list(item) for item in sorted(reduction_axes)],
        "has_scan": has_scan,
        "projected_priors": [item.to_record() for item in projected],
        "ignored_priors": list(ignored),
        "raw_c_value_used_as_target_tile": False,
    }
    return tuple(projected), tuple(ignored), projection


def _c_score(
    schedule: RiscvSchedule, priors: tuple[_ProjectedCPrior, ...]
) -> float:
    if not priors:
        return 0.0
    distances: list[float] = []
    for prior in priors:
        candidate = int(getattr(schedule, prior.schedule_dimension))
        if candidate <= 0:
            return float("inf")
        distances.append(
            abs(math.log2(float(candidate) / float(prior.target_value)))
        )
    return statistics.fmean(distances)


def _erase_schedule_key(schedule: RiscvSchedule) -> tuple[Any, ...]:
    # C-free deterministic order starts at the strong untiled LLVM baseline,
    # then explores small-to-large parallel, reduction and joint tilings.
    return (
        schedule.parallel_tile != 0 or schedule.reduction_tile != 0,
        schedule.parallel_tile == 0,
        schedule.reduction_tile == 0,
        max(schedule.parallel_tile, schedule.reduction_tile),
        schedule.parallel_tile,
        schedule.reduction_tile,
        not schedule.vectorize,
        schedule.scalable_vectorization,
        -schedule.opt_level,
        schedule.thread_count,
        abs(schedule.vector_lanes - 4),
    )


def _c_ranked_with_robust_anchors(
    candidates: tuple[RiscvLoweringCandidate, ...],
    priors: tuple[_ProjectedCPrior, ...],
) -> tuple[RiscvLoweringCandidate, ...]:
    """Rank by C while retaining early target-independent anchor schedules.

    C is a discardable search prior rather than an oracle for a different
    target.  The first state therefore remains the strongest projected-C
    candidate (the C-Fixed arm), but a finite search budget must also see the
    two robust untiled O3 anchors before spending all visits around that
    projection.  This preserves the candidate universe and lets C accelerate
    discovery without making a stale cross-target tile actively harmful.
    """

    if not priors:
        return tuple(sorted(candidates, key=lambda item: _erase_schedule_key(item.schedule)))

    ranked_groups: list[tuple[RiscvLoweringCandidate, ...]] = []
    projected_dimensions = {item.schedule_dimension for item in priors}

    def unprojected_tile_count(candidate: RiscvLoweringCandidate) -> int:
        # C-Fixed is the mechanical projection of the authored C, not an
        # opportunity to silently activate another schedule axis.  A tied
        # candidate therefore keeps every unmentioned tile dimension at its
        # canonical zero value.  Schedule-Only still contains the complete
        # target-independent universe and may subsequently explore it.
        schedule = candidate.schedule
        return sum(
            int(getattr(schedule, dimension) != 0)
            for dimension in ("parallel_tile", "reduction_tile")
            if dimension not in projected_dimensions
        )

    for thread_count in sorted({item.schedule.thread_count for item in candidates}):
        group = tuple(
            item for item in candidates if item.schedule.thread_count == thread_count
        )
        ranked = tuple(
            sorted(
                group,
                key=lambda item: (
                    _c_score(item.schedule, priors),
                    unprojected_tile_count(item),
                    _erase_schedule_key(item.schedule),
                ),
            )
        )
        first = ranked[0]
        anchors: list[RiscvLoweringCandidate] = []
        for vectorize, scalable in (
            (True, False),
            (True, True),
            (False, False),
        ):
            anchor = next(
                (
                    item
                    for item in group
                    if item.schedule.opt_level == 3
                    and item.schedule.vectorize is vectorize
                    and item.schedule.scalable_vectorization is scalable
                    and item.schedule.parallel_tile == 0
                    and item.schedule.reduction_tile == 0
                ),
                None,
            )
            if anchor is not None and anchor != first:
                anchors.append(anchor)
        promoted = {first.candidate_id, *(item.candidate_id for item in anchors)}
        ranked_groups.append(
            (
                first,
                *anchors,
                *(item for item in ranked if item.candidate_id not in promoted),
            )
        )
    return tuple(
        group[index]
        for index in range(max(len(group) for group in ranked_groups))
        for group in ranked_groups
        if index < len(group)
    )


def _interleave_lowering_paths(
    canonical: tuple[RiscvLoweringCandidate, ...],
    expanded_paths: tuple[tuple[RiscvLoweringCandidate, ...], ...],
) -> tuple[RiscvLoweringCandidate, ...]:
    """Give each ranked schedule one visit for every admitted MLIR state.

    A state+schedule budget must actually reach both concrete MLIR paths.  The
    previous concatenation spent every practical budget on the canonical path,
    making E3 observationally identical to E2 even when a faster fused state
    existed. Interleaving preserves each state's own valid schedule order while
    preventing fusion or explicit reduction reassociation from being starved;
    different MLIR states need not expose identical schedule domains.
    """

    if not expanded_paths:
        return canonical
    result: list[RiscvLoweringCandidate] = []
    all_paths = (canonical, *expanded_paths)
    for index in range(max(len(path) for path in all_paths)):
        result.extend(path[index] for path in all_paths if index < len(path))
    return tuple(result)


@dataclass(frozen=True)
class RiscvSearchPlan:
    universe: tuple[RiscvLoweringCandidate, ...]
    schedule_only_erase_c_order: tuple[str, ...]
    schedule_only_c_prior_order: tuple[str, ...]
    erase_c_order: tuple[str, ...]
    c_prior_order: tuple[str, ...]
    c_fixed_candidate_id: str
    c_scores: tuple[tuple[str, float | None], ...]
    consumed_c_priors: tuple[CPrior, ...]
    ignored_c_priors: tuple[dict[str, Any], ...]
    c_projection: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "protocol": RVV_SEARCH_PLAN_PROTOCOL,
            "candidate_universe_count": len(self.universe),
            "schedule_only_erase_c_order": list(
                self.schedule_only_erase_c_order
            ),
            "schedule_only_c_prior_order": list(
                self.schedule_only_c_prior_order
            ),
            "erase_c_order": list(self.erase_c_order),
            "c_prior_order": list(self.c_prior_order),
            "c_fixed_candidate_id": self.c_fixed_candidate_id,
            "c_scores": {
                candidate_id: score for candidate_id, score in self.c_scores
            },
            "consumed_c_priors": [
                item.to_record() for item in self.consumed_c_priors
            ],
            "ignored_c_priors": list(self.ignored_c_priors),
            "c_projection": dict(self.c_projection),
            "same_universe_for_erase_c_and_c_prior": True,
            "c_controls_admission": False,
            "c_controls_structure": False,
        }


def _riscv_loop_attachment(
    j: SemanticArchive,
) -> tuple[str, int | str | None] | None:
    if not isinstance(j, SemanticArchive):
        raise TypeError("RISC-V Layer-C interface requires SemanticArchive")
    typed_rows = j_elaboration(j).typed_operation_rows
    if typed_rows is None:
        raise RiscvContractError(
            "RISC-V structured loop requires canonical typed operation records"
        )
    records = {
        record["path"]: record
        for record in j_elaboration(j).operation_records
    }
    authored_by_path = j_elaboration(j).authored_node_by_path(j.node_ids)
    loops: list[tuple[str, Any]] = []
    for path, operation in typed_rows:
        if "." in path or not path.startswith("body["):
            continue
        if operation.op != "loop":
            continue
        record = records.get(path)
        node_id = record.get("id") if isinstance(record, Mapping) else None
        if not isinstance(node_id, str):
            raise RiscvContractError(
                "RISC-V structured loop lacks a stable J node attachment"
            )
        if authored_by_path.get(path, node_id) != node_id:
            raise RiscvContractError(
                "RISC-V structured loop node id disagrees with canonical J"
            )
        if operation.node_id is not None and operation.node_id != node_id:
            raise RiscvContractError(
                "RISC-V structured loop typed id disagrees with canonical J"
            )
        loops.append((node_id, operation))
    if not loops:
        return None
    if len(loops) != 1:
        raise RiscvContractError(
            "RISC-V Layer-C currently requires at most one top-level loop"
        )
    node_id, loop = loops[0]
    if loop.trip is None:
        raise RiscvContractError("RISC-V structured loop lacks a trip axis")
    # Dynamic value trips have no semantic dimension axis.  Keep the loop
    # subject available for target pipeline provenance, but do not expose a
    # fake tile attachment to RVV search or the target plugin grammar.
    return node_id, loop.trip.axis


def riscv_layer_c_interface(j: SemanticArchive) -> LayerCInterface:
    """Declare only J-addressed values consumed before RVV search/lowering."""

    if not isinstance(j, SemanticArchive):
        raise TypeError("RISC-V Layer-C interface requires SemanticArchive")
    attachment = _riscv_loop_attachment(j)
    if attachment is None:
        return LayerCInterface()
    subject_ref, axis_ref = attachment
    if axis_ref is None:
        return LayerCInterface()
    return LayerCInterface(
        (
            LayerCExtensionPoint(
                slot="loop_schedule_tile",
                role="loop_schedule_tile",
                value_type="i64",
                required=False,
                subject_ref=subject_ref,
                axis_ref=axis_ref,
            ),
        )
    )


def _riscv_plugin_bindings(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
) -> tuple[LayerCBinding, ...]:
    del state, candidate
    # No currently registered RiscvLoweringCandidate requests pre-M0 scan
    # strip-mining.  The optional point remains declared for a future closed
    # candidate kind, while today's plugins deliberately leave it unbound and
    # consume the deterministic full-axis J-only materialization.
    return ()


def _riscv_target_realization_contract(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
    choice: ImplementationChoice,
) -> TargetRealizationContract:
    """Project one admitted RVV schedule into the shared typed grammar."""

    attachment = _riscv_loop_attachment(state.base.j)
    subject_ref, axis_ref = attachment if attachment is not None else (None, None)
    schedule = candidate.schedule
    transformation = candidate.transformation
    if (attachment is None or axis_ref is None) and (
        schedule.parallel_tile or schedule.reduction_tile
    ):
        raise RiscvContractError(
            "RISC-V tiled schedule requires a static-axis J-attached top-level loop"
        )
    parallel_mapping: tuple[TargetParallelMapping, ...] = ()
    split: tuple[TargetSplit, ...] = ()
    if subject_ref is not None and axis_ref is not None:
        if schedule.parallel_tile:
            parallel_mapping = (
                TargetParallelMapping(
                    slot=None,
                    subject_ref=subject_ref,
                    axis_ref=axis_ref,
                    mapping="parallel",
                    factor=schedule.parallel_tile,
                ),
            )
        if schedule.reduction_tile:
            split = (
                TargetSplit(
                    slot=None,
                    subject_ref=subject_ref,
                    axis_ref=axis_ref,
                    factor=schedule.reduction_tile,
                ),
            )
    return TargetRealizationContract(
        parallel_mapping=parallel_mapping,
        pipeline=(
            TargetPipeline(
                stage=transformation.lowering_path_kind,
                depth=1,
                subject_ref=subject_ref,
                axis_ref=axis_ref,
            ),
        ),
        split=split,
        lowering=TargetLowering(
            backend_kind=choice.backend_kind,
            candidate_key=choice.candidate_key,
            candidate_digest=choice.candidate_digest,
        ),
    )


def _riscv_target_binding_context(
    state: BackendSearchState,
    contract_view: Any,
    *,
    e_c: LayerCInterface | Mapping[str, Any] | None = None,
) -> TargetBindingContext:
    target_interface = riscv_layer_c_interface(state.base.j)
    return TargetBindingContext(
        j=state.base.j,
        target=state.target,
        interface=target_interface,
        capabilities=contract_view.capabilities,
        allowed_authorizations=riscv_allowed_authorizations(state),
        # C_target is not Source-C.  Shared E_C is populated only when the
        # backend caller supplies it explicitly; omission means this context
        # binds a target-only plugin.
        e_c=e_c,
    )


def riscv_candidate_digest(candidate: RiscvLoweringCandidate) -> str:
    """Digest the complete internal candidate, never caller-authored payload."""

    if not isinstance(candidate, RiscvLoweringCandidate):
        raise TypeError("RISC-V candidate digest requires RiscvLoweringCandidate")
    return layer_c_content_digest(candidate.to_dict())


def riscv_candidate_key(candidate: RiscvLoweringCandidate) -> str:
    """Return a bounded opaque registry coordinate for one internal candidate."""

    digest = riscv_candidate_digest(candidate)
    return "riscv.candidate." + hashlib.sha256(
        (candidate.candidate_id + ":" + digest).encode("utf-8")
    ).hexdigest()


def _required_authorizations(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
) -> tuple[str, ...]:
    # The current registered RVV candidates do not consume a standalone
    # realization-authorization payload.  Reassociation admission comes from
    # the typed reduce ``assoc`` contract inspected by plan_riscv_search; other
    # transforms are exact closed compiler states.  Keeping this empty is
    # deliberate: a plugin that injects even a J-declared but irrelevant u*
    # reference is rejected by resolve_riscv_target_plugin below.
    del state, candidate
    return ()


def build_riscv_target_plugin(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
    *,
    e_c: LayerCInterface | Mapping[str, Any] | None = None,
) -> TargetLayerCPlugin:
    """Freeze one plan-admitted RVV candidate as a closed target plugin."""

    if not isinstance(state, BackendSearchState):
        raise TypeError("RISC-V plugin construction requires BackendSearchState")
    if not isinstance(candidate, RiscvLoweringCandidate):
        raise TypeError("RISC-V plugin construction requires RiscvLoweringCandidate")
    contract = derive_riscv_contract_view(state)
    plan = plan_riscv_search(state, enable_b_transformations=True)
    matches = [item for item in plan.universe if item == candidate]
    if len(matches) != 1:
        raise RiscvContractError(
            "RISC-V candidate is absent or ambiguous in the rebuilt search universe"
        )
    required_capabilities = tuple(
        sorted(candidate.transformation.required_b_capabilities)
    )
    missing = set(required_capabilities) - set(contract.capabilities)
    if missing:
        raise RiscvContractError(
            "RISC-V candidate lacks internally derived capabilities: "
            + ", ".join(sorted(missing))
        )
    choice = ImplementationChoice(
        backend_kind=RVV_LAYER_C_BACKEND_KIND,
        candidate_key=riscv_candidate_key(candidate),
        candidate_digest=riscv_candidate_digest(candidate),
        authorization_refs=_required_authorizations(state, candidate),
    )
    target_contract = _riscv_target_realization_contract(
        state,
        candidate,
        choice,
    )
    return bind_target(
        _riscv_target_binding_context(state, contract, e_c=e_c),
        bindings=_riscv_plugin_bindings(state, candidate),
        choice=choice,
        required_capabilities=required_capabilities,
        contract=target_contract,
    )


def build_riscv_target_artifact(
    state: BackendSearchState,
    candidate: RiscvLoweringCandidate,
    *,
    e_c: LayerCInterface | Mapping[str, Any] | None = None,
    e_c_mapping: Sequence[TargetExtensionMapping | Mapping[str, Any]] | None = None,
) -> TargetPluginArtifact:
    """Build the complete explicitly typed RVV target handoff."""

    target_interface = riscv_layer_c_interface(state.base.j)
    shared_e_c = (
        LayerCInterface()
        if e_c is None
        else (
            e_c
            if isinstance(e_c, LayerCInterface)
            else LayerCInterface.from_dict(e_c)
        )
    )
    validate_layer_c_attachments(state.base.j, shared_e_c)
    plugin = build_riscv_target_plugin(state, candidate, e_c=shared_e_c)
    if e_c_mapping is None:
        if shared_e_c == plugin.interface:
            e_c_mapping = tuple(
                TargetExtensionMapping(item.slot, item.slot)
                for item in shared_e_c.points
            )
        elif not shared_e_c.points:
            e_c_mapping = ()
        else:
            raise RiscvContractError(
                "RISC-V shared E_C and target interface differ; an explicit "
                "e_c_mapping is required"
            )
    return TargetPluginArtifact(
        e_c=shared_e_c,
        plugin=plugin,
        e_c_mapping=tuple(e_c_mapping),
    )


def resolve_riscv_target_plugin(
    state: BackendSearchState,
    plugin: TargetLayerCPlugin,
) -> RiscvLoweringCandidate:
    """Rebuild plan/universe and resolve exactly one key+digest; never fallback."""

    if not isinstance(state, BackendSearchState):
        raise TypeError("RISC-V plugin resolution requires BackendSearchState")
    if not isinstance(plugin, TargetLayerCPlugin):
        raise TypeError("RISC-V lowering requires TargetLayerCPlugin")
    contract_view = derive_riscv_contract_view(state)
    context = _riscv_target_binding_context(state, contract_view)
    plugin = deserialize_target_plugin(
        context,
        serialize_target_plugin(plugin),
    )
    if plugin.choice.backend_kind != RVV_LAYER_C_BACKEND_KIND:
        raise RiscvContractError("RISC-V plugin has the wrong backend kind")
    plan = plan_riscv_search(state, enable_b_transformations=True)
    matches = [
        item
        for item in plan.universe
        if riscv_candidate_key(item) == plugin.choice.candidate_key
        and riscv_candidate_digest(item) == plugin.choice.candidate_digest
    ]
    if len(matches) != 1:
        raise RiscvContractError(
            "RISC-V plugin candidate key+digest is absent or ambiguous for this J/target"
        )
    candidate = matches[0]
    expected_bindings = _riscv_plugin_bindings(state, candidate)
    if plugin.bindings != expected_bindings:
        raise RiscvContractError(
            "RISC-V plugin bindings do not match the rebuilt candidate requirements"
        )
    required_capabilities = tuple(
        sorted(candidate.transformation.required_b_capabilities)
    )
    if plugin.required_capabilities != required_capabilities:
        raise RiscvContractError(
            "RISC-V plugin capability requirements do not match the rebuilt candidate"
        )
    if plugin.choice.authorization_refs != _required_authorizations(state, candidate):
        raise RiscvContractError(
            "RISC-V plugin authorizations do not match the rebuilt candidate"
        )
    expected_choice = ImplementationChoice(
        backend_kind=RVV_LAYER_C_BACKEND_KIND,
        candidate_key=riscv_candidate_key(candidate),
        candidate_digest=riscv_candidate_digest(candidate),
        authorization_refs=_required_authorizations(state, candidate),
    )
    expected_contract = _riscv_target_realization_contract(
        state,
        candidate,
        expected_choice,
    )
    try:
        validate_target_plugin(
            context,
            plugin,
            expected_choice=expected_choice,
            expected_bindings=expected_bindings,
            expected_contract=expected_contract,
        )
    except (TypeError, ValueError) as exc:
        raise RiscvContractError(str(exc)) from exc
    return candidate


def plan_riscv_search(
    state: BackendSearchState,
    *,
    enable_b_transformations: bool = True,
) -> RiscvSearchPlan:
    if not isinstance(state, BackendSearchState):
        raise TypeError("RVV search planning requires BackendSearchState")
    contract = derive_riscv_contract_view(state)
    reduction_axes, _has_scan, _float_bits, has_reassociation, has_contraction = (
        _operation_facts(state.base.j)
    )
    has_typed_rowmask = _has_typed_rowmask_graph(state.base.j)
    has_typed_gather_chain = _has_typed_gather_chain(state.base.j)
    has_paired_rotation = (
        'intent.paired_rotation = "interleaved"' in state.base.intent_mlir
        and "intentir.paired-last-axis-rotation.v1" in state.base.intent_mlir
    )
    has_streaming_attention = _has_typed_flash_attention_graph(state.base.j)
    has_sibling_outputs = (
        _has_two_public_outputs(state.base.j)
        and not reduction_axes
        and not has_paired_rotation
    )
    projected_priors, ignored_priors, projection = _project_c_priors(state)
    raw_universe = enumerate_riscv_candidates(
        enable_b_transformations=enable_b_transformations,
        enable_reassociation=has_reassociation,
        enable_rowmask_zero_fill=has_typed_rowmask,
        enable_composed_gather=has_typed_gather_chain,
        enable_sibling_output_fusion=has_sibling_outputs,
        enable_paired_rotation=has_paired_rotation,
        enable_streaming_attention=has_streaming_attention,
    )
    has_ragged = bool(contract.ragged_ops)
    has_dynamic_ragged_member = contract.dynamic_ragged_member_output
    ragged_schedule_rejections = 0
    if has_dynamic_ragged_member:
        # Ragged values have a data-dependent result extent.  Keep the
        # materialized carrier-tensor realization as the sole C++/RVV path;
        # vector/fusion/tiled candidates are flat-tensor transformations and
        # must not be selected around a dynamic output ABI.  The predicate is
        # shared with the compile boundary so search and execution cannot
        # disagree about the admitted schedule.
        before = len(raw_universe)
        raw_universe = tuple(
            item
            for item in raw_universe
            if (
                item.transformation == RiscvTransformationState(False)
                and ragged_schedule_supported(item.schedule)
            )
        )
        ragged_schedule_rejections = before - len(raw_universe)
    target_vlen_bits = int(projection["target_vlen_bits"])
    incompatible_vlen_schedules_rejected = sum(
        item.schedule.vectorize
        and not item.schedule.scalable_vectorization
        and item.schedule.vector_bits != target_vlen_bits
        for item in raw_universe
    )
    raw_universe = tuple(
        item
        for item in raw_universe
        if not item.schedule.vectorize
        or item.schedule.scalable_vectorization
        or item.schedule.vector_bits == target_vlen_bits
    )
    minimum_reduction_tile = int(projection["vector_lanes"])
    subvector_reduction_schedules_rejected = sum(
        0 < item.schedule.reduction_tile < minimum_reduction_tile
        for item in raw_universe
    )
    contraction_reduction_schedules_rejected = sum(
        has_contraction and item.schedule.reduction_tile > 0
        for item in raw_universe
    )
    universe = tuple(
        item
        for item in raw_universe
        if item.schedule.reduction_tile == 0
        or (
            not has_contraction
            and item.schedule.reduction_tile >= minimum_reduction_tile
        )
    )
    schedule_attachment = _riscv_loop_attachment(state.base.j)
    schedule_attachment_rejections = 0
    # A dynamic value loop is a valid J control construct, but it has no
    # semantic axis for RVV strip-mining or reduction tiling.  Remove those
    # candidates before scoring/admission so search and target-contract
    # validation see the same closed universe.
    if schedule_attachment is None or schedule_attachment[1] is None:
        before = len(universe)
        universe = tuple(
            item
            for item in universe
            if item.schedule.parallel_tile == 0
            and item.schedule.reduction_tile == 0
        )
        schedule_attachment_rejections = before - len(universe)
    projection = {
        **projection,
        "minimum_reduction_tile": minimum_reduction_tile,
        "target_schedule_rejections": len(raw_universe) - len(universe),
        "incompatible_vlen_schedules_rejected": (
            incompatible_vlen_schedules_rejected
        ),
        "fixed_schedule_vlen_rule": (
            "fixed-vector candidates must match target vlenN exactly; "
            "other targets use scalable candidates"
        ),
        "subvector_reduction_schedules_rejected": (
            subvector_reduction_schedules_rejected
        ),
        "subvector_reduction_rule": (
            "a vectorized reduction tile must cover at least one target vector"
        ),
        "generic_contraction_reduction_tiling_admitted": not has_contraction,
        "generic_contraction_reduction_schedules_rejected": (
            contraction_reduction_schedules_rejected
        ),
        "schedule_attachment_rejections": schedule_attachment_rejections,
        "ragged_schedule_rejections": ragged_schedule_rejections,
        "ragged_schedule_rule": (
            "only scalar, untiled, single-thread schedules may carry the "
            "dynamic ragged_member output ABI"
        ) if has_dynamic_ragged_member else (
            "ragged_outer keeps the ordinary RVV schedule universe; no "
            "dynamic member output ABI is present"
            if has_ragged else "not_applicable"
        ),
    }
    scores = {
        item.candidate_id: _c_score(item.schedule, projected_priors)
        for item in universe
    }
    if has_paired_rotation:
        # Once the J has a closed interleaved paired view, carrying the exploded
        # 215-op materialized graph as C-Fixed would recreate the invalid
        # generic-fallback-vs-complete-consumer ablation.  Transport only the
        # complete contiguous spine and its genuine head-reuse alternative.
        universe = tuple(
            item for item in universe if item.transformation.paired_rotation
        )
        canonical = tuple(
            item
            for item in universe
            if not item.transformation.paired_rotation_head_angle_reuse
        )
    else:
        canonical = tuple(
            item
            for item in universe
            if not item.transformation.fuse_producers
            and not item.transformation.reassociate_reductions
            and not item.transformation.rowmask_zero_fill
            and not item.transformation.composed_indexed_gather
            and not item.transformation.fuse_sibling_outputs
            and not item.transformation.paired_rotation
        )
    alternative_states = tuple(
        dict.fromkeys(
            (
                item.transformation.fuse_producers,
                item.transformation.reassociate_reductions,
                item.transformation.explicit_vector_reduction,
                item.transformation.joint_vector_reduction,
                item.transformation.fused_softmax,
                item.transformation.rowmask_zero_fill,
                item.transformation.composed_indexed_gather,
                item.transformation.fuse_sibling_outputs,
                item.transformation.approximate_exp,
                item.transformation.mixed_precision_dot_reduction,
                item.transformation.paired_rotation,
                item.transformation.paired_rotation_head_angle_reuse,
                item.transformation.streaming_attention,
            )
            for item in universe
            if item not in canonical
        )
    )
    expanded = tuple(
        tuple(
            item
            for item in universe
            if (
                item.transformation.fuse_producers,
                item.transformation.reassociate_reductions,
                item.transformation.explicit_vector_reduction,
                item.transformation.joint_vector_reduction,
                item.transformation.fused_softmax,
                item.transformation.rowmask_zero_fill,
                item.transformation.composed_indexed_gather,
                item.transformation.fuse_sibling_outputs,
                item.transformation.approximate_exp,
                item.transformation.mixed_precision_dot_reduction,
                item.transformation.paired_rotation,
                item.transformation.paired_rotation_head_angle_reuse,
                item.transformation.streaming_attention,
            )
            == state_key
        )
        for state_key in alternative_states
    )
    schedule_erase = tuple(
        sorted(canonical, key=lambda item: _erase_schedule_key(item.schedule))
    )
    schedule_c = _c_ranked_with_robust_anchors(canonical, projected_priors)
    expanded_erase = tuple(
        tuple(sorted(path, key=lambda item: _erase_schedule_key(item.schedule)))
        for path in expanded
    )
    expanded_c = tuple(
        _c_ranked_with_robust_anchors(path, projected_priors)
        for path in expanded
    )
    # E2 retains the complete canonical schedule order. E3 has the same total
    # visit budget but round-robin interleaves canonical and structurally
    # expanded MLIR. Each state owns its valid schedule domain; in particular,
    # the dynamic-VL explicit vector state only admits scalable schedules.
    # C-Fixed remains exactly the first target-projected canonical candidate.
    erase_order = _interleave_lowering_paths(schedule_erase, expanded_erase)
    c_order = _interleave_lowering_paths(schedule_c, expanded_c)
    c_fixed = schedule_c[0]
    return RiscvSearchPlan(
        universe=universe,
        schedule_only_erase_c_order=tuple(
            item.candidate_id for item in schedule_erase
        ),
        schedule_only_c_prior_order=tuple(
            item.candidate_id for item in schedule_c
        ),
        erase_c_order=tuple(item.candidate_id for item in erase_order),
        c_prior_order=tuple(item.candidate_id for item in c_order),
        c_fixed_candidate_id=c_fixed.candidate_id,
        c_scores=tuple(
            (
                item.candidate_id,
                None if math.isinf(scores[item.candidate_id]) else scores[item.candidate_id],
            )
            for item in universe
        ),
        consumed_c_priors=tuple(item.prior for item in projected_priors),
        ignored_c_priors=ignored_priors,
        c_projection=projection,
    )


__all__ = [
    "RVV_SEARCH_PLAN_PROTOCOL",
    "RVV_C_PROJECTION_PROTOCOL",
    "RVV_LAYER_C_BACKEND_KIND",
    "RiscvSearchPlan",
    "build_riscv_target_plugin",
    "build_riscv_target_artifact",
    "enumerate_riscv_candidates",
    "plan_riscv_search",
    "resolve_riscv_target_plugin",
    "riscv_candidate_digest",
    "riscv_candidate_key",
    "riscv_layer_c_interface",
]
